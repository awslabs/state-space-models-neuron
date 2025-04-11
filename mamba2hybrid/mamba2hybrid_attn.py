# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch LLaMA model."""
import math
import os
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch_xla.core.xla_model as xm
from packaging import version
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LLAMA_INPUTS_DOCSTRING,
    LLAMA_START_DOCSTRING,
)
from transformers.models.llama.modeling_llama import LlamaAttention as LlamaAttentionHF
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer as LlamaDecoderLayerHF,
)
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM as LlamaForCausalLMHF,
)
from transformers.models.llama.modeling_llama import (
    LlamaForSequenceClassification,
    LlamaLinearScalingRotaryEmbedding,
)
from transformers.models.llama.modeling_llama import LlamaMLP as LlamaMLPHF
from transformers.models.llama.modeling_llama import LlamaModel as LlamaModelHF
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm as LlamaRMSNormHF
from transformers.models.llama.modeling_llama import (
    ROPE_INIT_FUNCTIONS,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

import neuronx_distributed.parallel_layers.utils as neuronx_dist_utils
from neuronx_distributed.kernels.flash_attn import nki_flash_attn_func
from neuronx_distributed.modules.qkv_linear import GQAQKVColumnParallelLinear
from neuronx_distributed.parallel_layers import mappings
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.loss_functions import parallel_cross_entropy
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_size,
)
from neuronx_distributed.utils.model_utils import move_model_to_device
from .configuration_mamba2hybrid import Mamba2HybridConfig
from dataclasses import dataclass
from mamba2.conv1d_grouped import ConvNKI


def _init_normal(std, w):
    return nn.init.normal_(w, mean=0.0, std=std)


if version.parse(torch.__version__) >= version.parse("2.1"):
    from torch_xla.utils.checkpoint import checkpoint

    checkpoint_method = checkpoint
else:
    checkpoint_method = torch.utils.checkpoint.checkpoint


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(x, cos, sin, ro_dim):
    """As in FlashAttention's implementation: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py#L29C12-L32C6"""
    return torch.cat(
        [x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim]) * sin, x[..., ro_dim:]],
        dim=-1,
    )

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)

    q_embed = _apply_rotary_pos_emb(q, cos, sin, cos.shape[-1])
    k_embed = _apply_rotary_pos_emb(k, cos, sin, cos.shape[-1])

    return q_embed, k_embed


def _compute_default_rope_parameters(config, device, seq_len=None):
    '''From: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py'''
    attention_factor = 1.0  # Unused in this type of RoPE

    base = config.rope_theta
    dim = config.rotary_emb_dim
    # dim = config.head_dim

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).double().to(device) / dim))
    return inv_freq, attention_factor


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()

        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        # self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        self.rope_init_fn = _compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):

        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        # device_type = x.device.type
        # device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        # with torch.autocast(device_type=device_type, enabled=False):
        #     freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        #     emb = torch.cat((freqs, freqs), dim=-1)
        #     cos = emb.cos()
        #     sin = emb.sin()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class CoreAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_states, key_states, value_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        kv_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        causal_mask = torch.triu(torch.ones((1, 1, q_len, kv_seq_len), device="xla"), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill_(causal_mask, -10000.0)

        dtype = torch.double if os.environ.get("XLA_DOWNCAST_BF16", None) == "1" else torch.float32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=dtype).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output


class LlamaAttention(LlamaAttentionHF):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.total_num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attn_out_proj_bias = config.attn_out_proj_bias
        self.attn_qkv_bias = config.attn_qkv_bias

        if not hasattr(config, "kv_shared_group_size"):
            config.kv_shared_group_size = 1

        if not hasattr(config, "qkv_linear"):
            config.qkv_linear = False

        if not hasattr(config, "fuse_qkv"):
            config.fuse_qkv = False

        if not hasattr(config, "use_flash_attention"):
            self.use_flash_attention = False
        else:
            self.use_flash_attention = config.use_flash_attention

        # self._init_rope()
        if config.rotary_emb_dim is not None:
            self.rotary_emb = LlamaRotaryEmbedding(config)

        init_method = partial(_init_normal, config.initializer_range)
        if self.config.qkv_linear:
            self.qkv_proj = GQAQKVColumnParallelLinear(
                self.hidden_size,
                [self.num_heads * self.head_dim, self.num_key_value_heads * self.head_dim],
                bias=self.attn_qkv_bias,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=self.config.sequence_parallel_enabled,
                kv_size_multiplier=self.config.kv_shared_group_size,
                fuse_qkv=self.config.fuse_qkv,
                sequence_dimension=1
                # dtype=self.config.torch_dtype,
            )
        elif self.config.fuse_qkv and self.num_heads == self.num_key_value_heads:
            self.qkv_proj = ColumnParallelLinear(
                self.hidden_size,
                3 * self.num_heads * self.head_dim,
                stride=3,
                bias=self.attn_qkv_bias,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=self.config.sequence_parallel_enabled,
                sequence_dimension=1
                # dtype=self.config.torch_dtype,
            )
            self.split_size = self.num_heads * self.head_dim // get_tensor_model_parallel_size()
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=self.attn_qkv_bias,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=self.config.sequence_parallel_enabled,
                sequence_dimension=1
                # dtype=self.config.torch_dtype,
            )
            self.k_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=self.attn_qkv_bias,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=self.config.sequence_parallel_enabled,
                sequence_dimension=1
                # dtype=self.config.torch_dtype,
            )
            self.v_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=self.attn_qkv_bias,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=self.config.sequence_parallel_enabled,
                sequence_dimension=1
                # dtype=self.config.torch_dtype,
            )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=self.attn_out_proj_bias,
            input_is_parallel=True,
            init_method=init_method,
            sequence_parallel_enabled=self.config.sequence_parallel_enabled,
            sequence_dimension=1
            # dtype=self.config.torch_dtype,
        )
        self.num_heads = neuronx_dist_utils.divide(config.num_attention_heads, get_tensor_model_parallel_size())
        self.num_key_value_heads = neuronx_dist_utils.divide(
            config.num_key_value_heads * self.config.kv_shared_group_size, get_tensor_model_parallel_size()
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.core_attn = CoreAttention()

        if config.move_model_to_device:
            move_model_to_device(self, xm.xla_device())


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        assert use_cache is False, "KV-Cache flow is not fully supported"
        bsz, q_len, _ = hidden_states.size()

        if self.config.sequence_parallel_enabled:
            q_len = q_len * get_tensor_model_parallel_size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            if (
                self.config.fuse_qkv
                and self.num_heads == self.num_key_value_heads
                and self.config.kv_shared_group_size == 1
            ):
                qkv_states = self.qkv_proj(hidden_states)
                query_states, key_states, value_states = qkv_states.split(self.split_size, dim=2)
            elif self.config.qkv_linear:
                query_states, key_states, value_states = self.qkv_proj(hidden_states)
            else:
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if position_ids is None:
            position_ids = torch.arange(
                hidden_states.shape[1], dtype=torch.long, device=hidden_states.device
            )[None, :].repeat(hidden_states.shape[0], 1)

        if self.config.rotary_emb_dim is not None:
            cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = (
            nki_flash_attn_func(query_states, key_states, value_states)
            if self.use_flash_attention
            else self.core_attn(query_states, key_states, value_states)
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, (self.total_num_heads * self.head_dim) // get_tensor_model_parallel_size())

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


@dataclass
class Mamba2HybridAttentionConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    initializer_range: float
    max_position_embeddings: int
    rope_theta: float
    rotary_emb_dim: int
    rope_scaling: dict
    attention_dropout: bool
    attn_out_proj_bias: bool
    attn_qkv_bias: bool
    use_flash_attention: bool
    pretraining_tp: int
    sequence_parallel_enabled: bool
    move_model_to_device: bool


class Mamba2HybridAttention(LlamaAttention):
    def __init__(self, config: Mamba2HybridConfig, layer_idx: int):

        attn_config = Mamba2HybridAttentionConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.attn_num_heads,
            num_key_value_heads=config.attn_key_value_heads,
            head_dim=config.attn_head_dim,
            initializer_range=config.initializer_range,
            max_position_embeddings=8192,
            rope_theta=-1,
            rotary_emb_dim=None,
            rope_scaling=None,
            attention_dropout=False,
            attn_out_proj_bias=False,
            attn_qkv_bias=False,
            use_flash_attention=True,
            pretraining_tp=0,
            sequence_parallel_enabled=config.sequence_parallel_enabled,
            move_model_to_device=False,
        )

        super().__init__(attn_config)
        self.layer_idx = layer_idx




