# coding=utf-8
# Copyright 2025 state-spaces/mamba2 org and HuggingFace Inc. team.
# Modifications Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import math
from typing import Optional

import torch
from torch import nn
from torch.types import Device
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
from transformers.activations import ACT2FN
from einops import rearrange
import neuronx_distributed.parallel_layers.parallel_state as ps
from neuronx_distributed.parallel_layers import RowParallelLinear, ColumnParallelLinear

from .configuration_mamba2 import Mamba2Config
from .fused_kernel import mamba2_fused_kernel

from .conv1d_grouped import ConvNKI
from mamba2.mamba2_kernel_inference import mamba2_kernel_inference


def softplus(x, threshold=10):
    # FIXME: an (inefficient) workaround since F.softplus was computing the identity function in Trainium
    return torch.where(x < threshold, torch.log(1 + torch.exp(x)), x)


# Note: this is different from the same module in `transformers` when n_groups>1
#       this module does the normalization independently for each group,
#       while the original does not care about groups and would give different
#       results for different tp degrees
class MambaRMSNormGated(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None, n_groups: int = 1, rmsnorm_within_groups=True,
                 name=None, hidden_size=None):
        """Gated Root Mean Square Layer Normalization with support for groups

        Paper: https://arxiv.org/abs/1910.07467

        Mamba Official: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layernorm_gated.py#L18
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))
        self.n_groups = n_groups
        self.rmsnorm_within_groups = rmsnorm_within_groups
        self.name = name
        self.hidden_size = hidden_size
        self.parallel_split()

    def parallel_split(self):
        tp_rank = ps.get_tensor_model_parallel_rank()
        tp_size = ps.get_tensor_model_parallel_size()
        dim = self.weight.shape[0]

        # Special case for TP=1 with merged checkpoint
        if not self.training and tp_size == 1:
            if 'mixer.norm' in self.name:
                # For mixer norm - keep the full concatenated weights (8192)
                assert dim == self.hidden_size * 2, f"Expected mixer norm weight size {self.hidden_size * 2}, got {dim}"
            else:
                # For layer norm - use just one copy (4096)
                if dim != self.hidden_size:
                    assert dim == self.hidden_size * 2, f"Expected layer norm weight size {self.hidden_size} or {self.hidden_size * 2}, got {dim}"
                    self.weight.data = self.weight.data[:self.hidden_size].detach().clone()
            return self

        # Original TP logic for training
        assert dim % tp_size == 0, f"Weight dimension {dim} must be divisible by tp_size {tp_size}"
        assert self.n_groups % tp_size == 0, f"Number of groups {self.n_groups} must be divisible by tp_size {tp_size}"
        self.n_groups = self.n_groups // tp_size
        chunk = slice(dim // tp_size * tp_rank, dim // tp_size * (tp_rank + 1))
        self.weight.data = self.weight.data[chunk].detach().clone()
        return self

    def forward(self, hidden_states, gate=None):
        hidden_states = hidden_states.to(torch.float32)

        if self.rmsnorm_within_groups:
            hidden_states = rearrange(hidden_states, "... (g d) -> ... g d", g=self.n_groups)
            if gate is not None:
                gate = rearrange(gate, "... (g d) -> ... g d", g=self.n_groups)

        if gate is not None:
            hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        
        if self.rmsnorm_within_groups:
            hidden_states = rearrange(hidden_states, "... g d -> ... (g d)", g=self.n_groups)
        res = self.weight * hidden_states
        return res


class Mamba2Mixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: Mamba2Config, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = int(config.expand * self.hidden_size)
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.rms_norm = config.rms_norm
        self.rmsnorm_within_groups = config.rmsnorm_within_groups

        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max
        self.sequence_parallel_enabled = config.sequence_parallel_enabled

        assert self.intermediate_size % self.head_dim == 0
        assert self.intermediate_size // self.head_dim == self.num_heads

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = ConvNKI(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.conv_dim,
            padding=config.conv_kernel - 1,
        )

        # projection of the input hidden states
        self.in_proj = ColumnParallelLinear(self.hidden_size, self.intermediate_size + self.conv_dim + self.num_heads, 
                                            bias=config.use_bias, gather_output=False, dtype=config.dtype,
                                            sequence_parallel_enabled=config.sequence_parallel_enabled, sequence_dimension=1)
        # time step projection (discretization)
        dt = torch.exp(
            torch.rand(config.num_heads)
            * (math.log(config.time_step_max) - math.log(config.time_step_min))
            + math.log(config.time_step_min)
        ).clamp(min=config.time_step_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_reinit = True

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.norm = MambaRMSNormGated(
            self.intermediate_size,
            eps=self.layer_norm_epsilon,
            n_groups=self.n_groups,
            rmsnorm_within_groups=self.rmsnorm_within_groups,
            name=f"backbone.layers.{layer_idx}.mixer.norm",
            hidden_size=config.hidden_size  # Pass hidden_size from config
        )
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.out_proj = RowParallelLinear(self.intermediate_size, self.hidden_size, 
                                          bias=config.use_bias, input_is_parallel=True, dtype=config.dtype,
                                          sequence_parallel_enabled=config.sequence_parallel_enabled, sequence_dimension=1)
        self.use_bias = config.use_bias
        self.parallel_split()

    def parallel_split(self):
        tp_rank = ps.get_tensor_model_parallel_rank()
        tp_size = ps.get_tensor_model_parallel_size()
        assert self.intermediate_size % tp_size == 0
        assert self.n_groups % tp_size == 0
        self.intermediate_size_tp = self.intermediate_size // tp_size
        self.n_groups_tp = self.n_groups // tp_size
        self.num_heads_tp = self.num_heads // tp_size
        self.conv_dim_tp = self.conv_dim // tp_size
        head_chunk = slice(self.num_heads_tp * tp_rank, self.num_heads_tp * (tp_rank + 1))
        # note the .clone(), otherwise it would a view and would keep the original in memory
        self.D.data = self.D.data[head_chunk].detach().clone()
        self.A_log.data = self.A_log.data[head_chunk].detach().clone()
        self.dt_bias.data = self.dt_bias.data[head_chunk].detach().clone()
        return self

    def nki_kernels_forward(
            self,
            hidden_states: torch.Tensor,
            cache_params: Optional[Mamba2Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        # set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size_tp = self.n_groups_tp * self.ssm_state_size
        d_to_remove = 2 * self.intermediate_size + 2 * self.n_groups * self.ssm_state_size + self.num_heads

        assert cache_params is None, "cache not supported yet"
        assert self.training, "only training supported right now"
        assert attention_mask is None, "attention mask not supported yet"
        assert self.time_step_limit[0] == 0.0 and self.time_step_limit[1] == float("inf"), "dt limit not supported yet"
        assert self.activation in ["silu", "swish"]

        A = -torch.exp(self.A_log.float())  # (num_heads) or (intermediate_size, state_size)

        zxBCdt = self.in_proj(hidden_states)
        scan_output = mamba2_fused_kernel(zxBCdt, A, self.D,
                                       self.conv1d.weight, self.conv1d.bias,
                                       self.dt_bias, self.norm.weight,
                                       self.n_groups_tp, self.ssm_state_size, self.intermediate_size_tp, self.head_dim)
        out = self.out_proj(scan_output)

        return out

    def forward(
            self,
            hidden_states,
            cache_params: Optional[Mamba2Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        assert "xla" in self.in_proj.weight.device.type, "This model only supports forward on an XLA device"
        return self.nki_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)


    def step(self, hidden_states, cache_params: Optional[Mamba2Cache] = None, ):
        """
        hidden_states: (B, 1, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        assert (seqlen == 1)

        zxBCdt = self.in_proj(hidden_states)

        batch_size, seq_len, zxBCdt_size = zxBCdt.shape

        n_groups, d_state, d_inner, d_head = self.n_groups_tp, self.ssm_state_size, self.intermediate_size_tp, self.head_dim
        n_heads = d_inner // d_head
        d_conv = d_inner + 2 * n_groups * d_state
        assert zxBCdt_size == d_inner + d_conv + n_heads

        z, xBC, dt = torch.split(zxBCdt, [d_inner, d_conv, n_heads], dim=-1)

        dt = softplus(dt + self.dt_bias)

        A = -torch.exp(self.A_log.float())  # (num_heads) or (intermediate_size, state_size)
        y, new_conv_state, new_ssm_state = mamba2_kernel_inference(xBC, cache_params.conv_states[self.layer_idx],
                                                                  self.conv1d.weight.squeeze(1), self.conv1d.bias,
                                                                  cache_params.ssm_states[self.layer_idx], dt,
                                                                  A, self.D, n_groups=self.n_groups_tp)

        cache_params.conv_states[self.layer_idx] = new_conv_state
        cache_params.ssm_states[self.layer_idx] = new_ssm_state

        y = y.view(batch, seqlen, -1)
        y = self.norm(y, z)
        out = self.out_proj(y)

        return out