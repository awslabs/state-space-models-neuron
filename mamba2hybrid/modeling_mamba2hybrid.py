# coding=utf-8
# Copyright 2024 state-spaces/mamba2 org and HuggingFace Inc. team.
# Modifications Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Adaptation of transformers.models.mamba2.modeling_mamba2 to be compatible with neuronx-distributed and implement a
Mamba2Hybrid model.
"""


import math
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from neuronx_distributed.parallel_layers import ParallelEmbedding, RowParallelLinear, ColumnParallelLinear
from neuronx_distributed.parallel_layers.loss_functions import parallel_cross_entropy
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.types import Device
import torch.distributed as distrib
import torch_xla.core.xla_model as xm


from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)

from .configuration_mamba2hybrid import Mamba2HybridConfig
from .mamba2hybrid_mlp import Mamba2HybridMLP
from .mamba2hybrid_attn import Mamba2HybridAttention
from .mamba2hybrid_layer_allocation import allocate_layers
from .mamba2hybrid_layer_allocation import Symbols as LayerSymbols

from mamba2.mamba2_mixer import Mamba2Mixer

from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
from transformers.models.mamba2.modeling_mamba2 import Mamba2RMSNorm as Mamba2RMSNormHF
from neuronx_distributed.parallel_layers import mappings

logger = logging.get_logger(__name__)

selective_state_update = None

causal_conv1d_update, causal_conv1d_fn = None, None


_CHECKPOINT_FOR_DOC = "mistralai/mamba-codestral-7B-v0.1"
_CONFIG_FOR_DOC = "Mamba2HybridConfig"


def _init_normal(std, w):
    return nn.init.normal_(w, mean=0.0, std=std)


class Mamba2RMSNorm(Mamba2RMSNormHF):
    def __init__(self, hidden_size, eps=1e-6, sequence_parallel_enabled=False):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(hidden_size, eps=eps)
        setattr(self.weight, "sequence_parallel_enabled", sequence_parallel_enabled)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        # if os.environ.get("XLA_DOWNCAST_BF16", None) == "1":
        #     hidden_states = hidden_states.to(torch.double)
        # else:
        hidden_states = hidden_states.to(torch.float32)
        self.variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(self.variance + self.variance_epsilon)
        output = self.weight * hidden_states
        return output.to(input_dtype)


class PartitionBreak(torch.autograd.Function):
    """Workaround to help the compiler detect layers.

    The compiler looks for all_gather/batch_norm/clamp to separate layers. Since those are not present in our model,
    we insert a fake torch.clamp operation with inf boundary (so it acts like the identity) to ensure it will try to
    break the layer at that point.
    """

    @staticmethod
    def forward(ctx, input):
        return torch.clamp(input, min=None, max=torch.inf)

    @staticmethod
    def backward(ctx, d_output):
        return torch.clamp(d_output, min=None, max=torch.inf)


def partition_break(x):
    return PartitionBreak.apply(x)


class Mamba2HybridBlock(nn.Module):
    """Simple block wrapping Mixer block with normalization and residual connection."""
    def __init__(self, config, layer_type, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon, sequence_parallel_enabled=config.sequence_parallel_enabled)

        if layer_type == LayerSymbols.MAMBA:
            self.layer = Mamba2Mixer(config, layer_idx=layer_idx)
        elif layer_type == LayerSymbols.ATTENTION:
            self.layer = Mamba2HybridAttention(config, layer_idx=layer_idx)
        elif layer_type == LayerSymbols.MLP:
            self.layer = Mamba2HybridMLP(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Unrecognized layer symbol: {layer_type}")

    def forward(
            self,
            hidden_states,
            cache_params: Optional[Mamba2Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        if self.layer_type == LayerSymbols.MAMBA:
            hidden_states = self.layer(
                hidden_states, cache_params=cache_params, cache_position=cache_position,
                attention_mask=attention_mask
            )
        elif self.layer_type == LayerSymbols.ATTENTION:
            hidden_states = self.layer(hidden_states, attention_mask=attention_mask)[0]
        elif self.layer_type == LayerSymbols.MLP:
            hidden_states = self.layer(hidden_states)
        else:
            raise ValueError(f"Unrecognized layer symbol: {self.layer_type}")

        hidden_states = residual + hidden_states

        return hidden_states


class Mamba2HybridPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Mamba2HybridConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["Mamba2HybridBlock"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, Mamba2Mixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True
        elif isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # fixme: this won't give consistent initialization with tp
            nn.init.normal_(module.weight, std=self.config.initializer_range)
        elif isinstance(module, (ParallelEmbedding, RowParallelLinear, ColumnParallelLinear)):
            # module.init_weight_cpu()
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_hidden_layers)


@dataclass
# Copied from transformers.models.mamba.modeling_mamba.MambaOutput with MAMBA->MAMBA2ATTN,Mamba->Mamba2Attn
class Mamba2HybridOutput(ModelOutput):
    """
    Class for the MAMBA2Hybrid model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`Mamba2Cache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[Mamba2Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# Copied from transformers.models.mamba.modeling_mamba.MambaCausalLMOutput with MAMBA->MAMBA2ATTN,Mamba->Mamba2Attn
class Mamba2HybridCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`Mamba2Cache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[Mamba2Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


MAMBA2HYBRID_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Mamba2HybridConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MAMBA2HYBRID_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            Indices of input sequence tokens in the vocabulary.

            If `cache_params.seqlen_offset>0`, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        cache_params (`Mamba2Cache`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        use_cache (`bool`, *optional*):
            If set to `True`, the `cache_params` is returned and can be used to quickly generate the next logits.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            The position of the current input in the cache. This is used to ensure that the cache is correctly updated.
            If `cache_params` is passed, `cache_position` should also be passed.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
"""


@add_start_docstrings(
    "The bare MAMBA2ATTN Model transformer outputting raw hidden-states without any specific head on top.",
    MAMBA2HYBRID_START_DOCSTRING,
)
class Mamba2HybridModel(Mamba2HybridPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        init_method = partial(_init_normal, config.initializer_range)
        self.embeddings = ParallelEmbedding(config.vocab_size, config.hidden_size, init_method=init_method)
        self.sequence_parallel_enabled = config.sequence_parallel_enabled

        self.layer_allocation = allocate_layers(
            total_layers_count=config.num_hidden_layers,
            target_attention_ratio=config.hybrid_attention_ratio,
            target_mlp_ratio=config.hybrid_mlp_ratio,
            override_pattern=config.hybrid_override_pattern,
        )

        print('### Layer allocation: ', self.layer_allocation)

        self.layers = nn.ModuleList([
            Mamba2HybridBlock(config, layer_type=layer_type, layer_idx=layer_idx)
                    for layer_idx, layer_type in enumerate(self.layer_allocation)
        ])

        self.gradient_checkpointing = False
        self.norm_f = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon, sequence_parallel_enabled=config.sequence_parallel_enabled)
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(MAMBA2HYBRID_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Mamba2HybridOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            cache_params: Optional[Mamba2Cache] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Union[Tuple, Mamba2HybridOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if cache_params is None:
                cache_params = Mamba2Cache(
                    self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                cache_position = torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)
            elif cache_position is None:
                # cases when we do manual forward instead of using `model.generate` which will initiate
                # `cache_position` and makes sure it is not None, throw error here instead of doing some
                # hack to conjecture the current cache position
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically"
                )
        else:
            cache_params = None

        hidden_states = inputs_embeds
        if self.sequence_parallel_enabled:
            hidden_states = mappings.scatter_to_sequence_parallel_region(hidden_states, 1)
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                raise NotImplementedError("gradient checkpointing is not implemented yet")
                hidden_states = self._gradient_checkpointing_func(
                    mixer_block.__call__, hidden_states, cache_params, cache_position, attention_mask
                )
            else:
                hidden_states = partition_break(mixer_block(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                ))

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return Mamba2HybridOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


@add_start_docstrings(
    """
    The MAMBA2HYBRID Model transformer with a language modeling head on top (linear layer with weights not tied to the 
    input embeddings).
    """,
    MAMBA2HYBRID_START_DOCSTRING,
)
class Mamba2HybridForCausalLM(Mamba2HybridPreTrainedModel, GenerationMixin):
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.backbone = Mamba2HybridModel(config)
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        init_method = partial(_init_normal, config.initializer_range)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=False,
            init_method=init_method,
            sequence_parallel_enabled=config.sequence_parallel_enabled,
            sequence_dimension=1,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def prepare_inputs_for_generation(
            self,
            input_ids,
            inputs_embeds=None,
            use_cache=None,
            cache_params: Optional[Mamba2Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        # Overwitten -- uses `cache_params` as opposed to `past_key_values`

        if inputs_embeds is not None:
            past_len = inputs_embeds.shape[1] + input_ids.shape[1]
        else:
            past_len = input_ids.shape[1]
        if use_cache:
            # `cache_position` should have been initialized in `generate`
            if cache_position is None:
                raise ValueError(
                    "`cache_position` should not be None as it should have been initialized in "
                    "`model.generate`, you are responsible for passing in a valid `cache_position` if "
                    "you are calling `prepare_inputs_for_generation` directly with `use_cache=True`"
                )
            # how do we detect that we are in decoding without cache?
            if cache_position[0] > 0:
                input_ids = input_ids[:, -1][..., None]
                attention_mask = attention_mask[:, -1][..., None]
            else:
                # we initialize the `cache_position` to full size of `conv_states` at prefill stage
                # considering padding will be applied when input length is shorter, and truncation
                # will be applied when it is longer, so it will be equivalent to always have it match
                # the length of `cache_params.conv_states`, which is `config.conv_kernel`
                cache_position = torch.arange(0, past_len, device=input_ids.device)
                # if the cache is not used, we also do have to extend the attention mask here
                # TODO there is likely a cleverer way to do this
                extended_mask = torch.ones(
                    attention_mask.size(0), past_len - attention_mask.shape[1], device=attention_mask.device
                )
                attention_mask = torch.cat([attention_mask, extended_mask], dim=1)
                cache_params = None

        if attention_mask.shape[1] < past_len:
            # we have to update manually the attention mask if
            # we are in decoding without cache
            # and we don't have position_ids here
            # TODO but we should be able to use cache_position though at a later time
            extended_mask = torch.ones(
                attention_mask.size(0), past_len - attention_mask.shape[1], device=attention_mask.device
            )
            attention_mask = torch.cat([attention_mask, extended_mask], dim=1)
        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "cache_params": cache_params,
                "use_cache": use_cache,
                "cache_position": cache_position,
            }
        )
        return model_inputs

    @add_start_docstrings_to_model_forward(MAMBA2HYBRID_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Mamba2HybridCausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            cache_params: Optional[Mamba2Cache] = None,
            labels: Optional[torch.LongTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, Mamba2HybridCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba2hybrid_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = mamba2hybrid_outputs[0]

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = parallel_cross_entropy
            # Flatten the tokens
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.mean()

        if not return_dict:
            output = (logits,) + mamba2hybrid_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Mamba2HybridCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=mamba2hybrid_outputs.cache_params,
            hidden_states=mamba2hybrid_outputs.hidden_states,
        )