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
""" PyTorch LLaMA's MLP from
https://github.com/aws-neuron/neuronx-distributed/blob/main/examples/training/llama/modeling_llama_nxd.py"""


# import math
# import os
from functools import partial
# from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch_xla.core.xla_model as xm
from packaging import version
from torch import nn
from transformers.activations import ACT2FN

from transformers.models.llama.modeling_llama import LlamaMLP as LlamaMLPHF

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_size,
)
from neuronx_distributed.utils.model_utils import move_model_to_device

from .configuration_mamba2hybrid import Mamba2HybridConfig
from dataclasses import dataclass


def _init_normal(std, w):
    return nn.init.normal_(w, mean=0.0, std=std)


if version.parse(torch.__version__) >= version.parse("2.1"):
    from torch_xla.utils.checkpoint import checkpoint

    checkpoint_method = checkpoint
else:
    checkpoint_method = torch.utils.checkpoint.checkpoint


class LlamaMLP(LlamaMLPHF):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.pretraining_tp = config.pretraining_tp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        # Set to 2 when Gated Linear Unit is used
        self.ffn_stride = 1
        if self.config.gated_linear_unit:
            self.ffn_stride = 2
        init_method = partial(_init_normal, config.initializer_range)
        self.gate_up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.ffn_stride * self.intermediate_size,
            stride=self.ffn_stride,
            bias=False,
            gather_output=False,
            init_method=init_method,
            sequence_parallel_enabled=self.config.sequence_parallel_enabled,
            sequence_dimension=1
            # dtype=self.config.torch_dtype,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=init_method,
            sequence_parallel_enabled=self.config.sequence_parallel_enabled,
            sequence_dimension=1
            # dtype=self.config.torch_dtype,
        )
        self.split_size = self.intermediate_size // get_tensor_model_parallel_size()
        if config.move_model_to_device:
            move_model_to_device(self, xm.xla_device())

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)]
            down_proj = sum(down_proj)
        else:
            if self.config.gated_linear_unit:
                gate_proj, up_proj = self.gate_up_proj(x).split(self.split_size, dim=2)
            else:
                gate_proj = self.gate_up_proj(x)
                up_proj = 1.

            def activation_mlp(gate_proj, up_proj):
                activation_output = self.act_fn(gate_proj)
                return activation_output * up_proj

            # We checkpoint the MLP compute too, since we see extra data movement which is more
            # expensive than the recompute in this case.
            if self.config.selective_checkpoint_enabled:
                intermediate_states = checkpoint_method(activation_mlp, gate_proj, up_proj)
            else:
                intermediate_states = self.act_fn(gate_proj) * up_proj
            down_proj = self.down_proj(intermediate_states)

        return down_proj


@dataclass
class Mamba2HybridMLPConfig:
    hidden_size: int
    hidden_act: str
    intermediate_size: int
    initializer_range: float
    pretraining_tp: int
    sequence_parallel_enabled: bool
    move_model_to_device: bool
    selective_checkpoint_enabled: bool
    gated_linear_unit: bool


class Mamba2HybridMLP(LlamaMLP):
    def __init__(self, config: Mamba2HybridConfig, layer_idx: int):

        mlp_config = Mamba2HybridMLPConfig(
            hidden_size=config.hidden_size,
            hidden_act=config.mlp_hidden_act,
            intermediate_size=config.mlp_hidden_size,
            initializer_range=config.initializer_range,
            pretraining_tp=0,
            sequence_parallel_enabled=config.sequence_parallel_enabled,
            move_model_to_device=False,
            selective_checkpoint_enabled=False,
            gated_linear_unit=False,
        )

        super().__init__(mlp_config)
        self.layer_idx = layer_idx
