import torch
import pytest
import numpy as np
import torch_xla.core.xla_model as xm
from mamba2.mamba2_kernel_inference import mamba_inference_kernel_, mamba2_kernel_inference
from mamba2.fused_kernel import softplus_with_bias_test_kernel_
from mamba2.mamba2_kernel import mamba_kernel_test_
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np
from mamba2.mamba2_mixer import Mamba2Mixer
from predefined_configs import get_config
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
from typing import Optional
from mamba2.conv1d_grouped import conv1d_with_transpose


def test_mamba2_kernel_inference_basic():
    device = 'xla:0'
    # Test parameters
    batch_size = 2
    seq_len = 1
    n_heads = 4
    d_head = 8
    d_state = 16
    kernel_size = 4
    n_groups = 1
    eps = 1e-6

    # Calculate dimensions correctly
    d_inner = n_heads * d_head  # Should be 32
    d_xBC = d_inner + 2 * n_groups * d_state  # Should be 64 (32 + 2 * 1 * 16)

    # Fixed syntax errors and shape/dimension issues
    xBC_tensor = np.random.randn(batch_size, seq_len, d_xBC).astype(np.float32)
    conv_state_tensor = np.zeros((batch_size, d_xBC, kernel_size), dtype=np.float32)
    conv_weight_tensor = np.random.randn(d_xBC, kernel_size).astype(np.float32)
    initial_state = np.zeros((batch_size, n_heads, d_head, d_state), dtype=np.float32)
    dt = np.random.randn(batch_size, seq_len, n_heads).astype(np.float32)  # Changed 1 to seq_len to match xBC
    A = np.ones(n_heads, dtype=np.float32) * (-eps)  # Small negative value
    D = np.ones(n_heads, dtype=np.float32) * eps

    out_Y = np.empty((batch_size, seq_len, d_inner), dtype=np.float32)
    out_conv_state = np.empty_like(conv_state_tensor, dtype=np.float32)
    conv_bias = np.ones(d_xBC, dtype=np.float32)
    out_S = np.empty_like(initial_state, dtype=np.float32)

    nki.simulate_kernel(mamba_inference_kernel_[batch_size],
                        xBC_tensor, conv_state_tensor, conv_weight_tensor, conv_bias,
                        initial_state, dt, A,
                        D, out_S,
                        out_Y, out_conv_state, n_groups=n_groups)

    # Basic assertions with correct dimensions
    assert out_Y.shape == (batch_size, seq_len, d_inner)
    assert out_conv_state.shape == (batch_size, d_xBC, kernel_size)
    assert out_S.shape == (batch_size, n_heads, d_head, d_state)
    assert not np.isnan(out_Y).any()
    assert not np.isinf(out_Y).any()


def test_mamba2_kernel_inference_zero_input():
    device = 'xla:0'
    batch_size = 1
    seq_len = 1
    n_heads = 2
    d_head = 4
    d_state = 16
    kernel_size = 4
    n_groups = 1
    d_inner = n_heads * d_head
    d_xBC = d_inner + 2 * n_groups * d_state
    eps = 1e-6

    xBC_tensor = np.zeros((batch_size, seq_len, d_xBC), dtype=np.float32)
    conv_state_tensor = np.zeros((batch_size, d_xBC, kernel_size), dtype=np.float32)
    conv_weight_tensor = np.zeros((d_xBC, kernel_size), dtype=np.float32)
    initial_state = np.zeros((batch_size, n_heads, d_head, d_state), dtype=np.float32)
    dt = np.zeros((batch_size, 1, n_heads), dtype=np.float32)
    A = np.ones((n_heads), dtype=np.float32) * (-eps)  # Small negative value
    D = np.ones(n_heads, dtype=np.float32) * eps

    out_Y = np.empty((batch_size, seq_len, d_inner), dtype=np.float32)
    out_conv_state = np.empty_like(conv_state_tensor, dtype=np.float32)
    conv_bias = np.zeros(d_xBC, dtype=np.float32)
    out_S = np.empty_like(initial_state, dtype=np.float32)

    nki.simulate_kernel(mamba_inference_kernel_[batch_size],
                        xBC_tensor, conv_state_tensor, conv_weight_tensor, conv_bias,
                        initial_state, dt, A,
                        D, out_S,
                        out_Y, out_conv_state, n_groups=n_groups)

    print(out_Y)
    # Check if outputs are all zeros
    assert np.allclose(out_Y, np.zeros_like(out_Y))
    assert np.allclose(out_conv_state, np.zeros_like(out_conv_state))
    assert np.allclose(out_S, np.zeros_like(out_S))

    # Basic assertions with correct dimensions
    assert out_Y.shape == (batch_size, seq_len, d_inner)
    assert out_conv_state.shape == (batch_size, d_xBC, kernel_size)
    assert out_S.shape == (batch_size, n_heads, d_head, d_state)
    assert not np.isnan(out_Y).any()
    assert not np.isinf(out_Y).any()


def test_mamba2_kernel_inference_batch_consistency():
    device = 'xla:0'
    batch_size = 3
    seq_len = 1
    n_heads = 3
    d_head = 6
    d_state = 16
    kernel_size = 4
    n_groups = 1
    d_inner = n_heads * d_head
    d_xBC = d_inner + 2 * n_groups * d_state
    eps = 1e-6

    # Create identical inputs for all batches using numpy
    xBC_tensor = np.tile(np.random.randn(1, seq_len, d_xBC).astype(np.float32), (batch_size, 1, 1))
    conv_state_tensor = np.tile(np.random.randn(1, d_xBC, kernel_size).astype(np.float32), (batch_size, 1, 1))
    conv_weight_tensor = np.random.randn(d_xBC, kernel_size).astype(np.float32)
    initial_state = np.zeros((batch_size, n_heads, d_head, d_state), dtype=np.float32)
    dt = np.tile(np.random.randn(1, seq_len, n_heads).astype(np.float32), (batch_size, 1, 1))
    A = np.ones(n_heads, dtype=np.float32) * (-eps)
    D = np.ones(n_heads, dtype=np.float32) * eps

    # Create output tensors
    out_Y = np.empty((batch_size, seq_len, d_inner), dtype=np.float32)
    out_conv_state = np.empty_like(conv_state_tensor, dtype=np.float32)
    conv_bias = np.ones(d_xBC, dtype=np.float32)
    out_S = np.empty_like(initial_state, dtype=np.float32)

    nki.simulate_kernel(mamba_inference_kernel_[batch_size],
                        xBC_tensor, conv_state_tensor, conv_weight_tensor, conv_bias,
                        initial_state, dt, A,
                        D, out_S,
                        out_Y, out_conv_state, n_groups=n_groups)

    print(f"Y: {0}", out_Y[0])
    # Check if all batches produce the same output
    for i in range(1, batch_size):
        print(f"Y: {i}", out_Y[i])
        assert np.allclose(out_Y[0], out_Y[i], rtol=1e-5, atol=1e-5)
        assert np.allclose(out_conv_state[0], out_conv_state[i], rtol=1e-5, atol=1e-5)
        assert np.allclose(out_S[0], out_S[i], rtol=1e-5, atol=1e-5)


def test_mamba2_kernel_inference_different_shapes():
    device = 'xla:0'
    batch_sizes = [1, 2]
    seq_lens = [1]
    n_heads_list = [2, 4]
    d_heads = [8, 16]
    d_state = 16
    kernel_size = 4
    n_groups = 1
    eps = 1e-6

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            for n_heads in n_heads_list:
                for d_head in d_heads:
                    d_inner = n_heads * d_head
                    d_xBC = d_inner + 2 * n_groups * d_state

                    # Create tensors using numpy
                    xBC_tensor = np.random.randn(batch_size, seq_len, d_xBC).astype(np.float32) * 0.1
                    conv_state_tensor = np.random.randn(batch_size, d_xBC, kernel_size).astype(np.float32) * 0.1
                    conv_weight_tensor = np.random.randn(d_xBC, kernel_size).astype(np.float32) * 0.1
                    initial_state = np.zeros((batch_size, n_heads, d_head, d_state), dtype=np.float32)
                    dt = np.random.rand(batch_size, seq_len, n_heads).astype(np.float32) * 0.1
                    A = np.ones(n_heads, dtype=np.float32) * (-eps)
                    D = np.ones(n_heads, dtype=np.float32) * eps

                    # Create output tensors
                    out_Y = np.empty((batch_size, seq_len, d_inner), dtype=np.float32)
                    out_conv_state = np.empty_like(conv_state_tensor, dtype=np.float32)
                    conv_bias = np.ones(d_xBC, dtype=np.float32)
                    out_S = np.empty_like(initial_state, dtype=np.float32)

                    nki.simulate_kernel(mamba_inference_kernel_[batch_size],
                                        xBC_tensor, conv_state_tensor, conv_weight_tensor, conv_bias,
                                        initial_state, dt, A,
                                        D, out_S,
                                        out_Y, out_conv_state, n_groups=n_groups)

                    # Basic assertions with correct dimensions
                    assert out_Y.shape == (batch_size, seq_len, d_inner)
                    assert out_conv_state.shape == (batch_size, d_xBC, kernel_size)
                    assert out_S.shape == (batch_size, n_heads, d_head, d_state)
                    assert not np.isnan(out_Y).any()
                    assert not np.isinf(out_Y).any()


def softplus(x, threshold=10):
    # np.where has same signature as torch.where
    return np.where(x < threshold, np.log1p(np.exp(x)), x)


def test_against_batch_forward():
    # Test parameters matching Mamba2Mixer configuration
    batch_size = 3
    seq_len = 2048
    n_groups = 8
    eps = 1e-6

    config = get_config("Mamba370M", vocab_size=65536, n_groups=n_groups)
    hidden_size = config.hidden_size
    d_state = config.state_size
    d_head = config.head_dim
    n_heads = config.num_heads
    intermediate_size = int(config.expand * hidden_size)
    d_xBC = intermediate_size + 2 * n_groups * d_state
    print("\n\nd_xBC", d_xBC, "\n\n ")
    d_inner = n_heads * d_head
    kernel_size = config.conv_kernel

    xBC_tensor = np.random.randn(batch_size, seq_len, d_xBC).clip(-1, 1).astype(np.float32)  / np.sqrt(d_xBC)
    conv_state_tensor = np.zeros((batch_size, d_xBC, kernel_size), dtype=np.float32)
    conv_weight_tensor = np.random.randn(d_xBC, kernel_size).astype(np.float32).clip(-1, 1) / (d_xBC + kernel_size)
    initial_state = np.zeros((batch_size, n_heads, d_head, d_state), dtype=np.float32)
    # dt = np.random.randn(batch_size, seq_len, n_heads).astype(np.float32)
    dt = np.exp(np.random.uniform(low=np.log(0.001), high=np.log(0.1),
                                  size=(batch_size, seq_len, n_heads)).astype(np.float32))
    dt_bias = dt[0, 0, :]
    A = np.ones(n_heads, dtype=np.float32) * (-eps)  # Small negative value
    D = np.ones(n_heads, dtype=np.float32) * eps

    A_tensor = -np.exp(A)  # (num_heads) or (intermediate_size, state_size)

    # Create output tensors
    out_Y_step = np.empty((batch_size, 1, d_inner), dtype=np.float32)
    out_conv_state = np.empty_like(conv_state_tensor, dtype=np.float32)
    conv_bias = np.ones(d_xBC, dtype=np.float32) / d_xBC
    out_S = np.empty_like(initial_state, dtype=np.float32)
    dt_softplus = softplus(dt + dt_bias)

    output_list = []
    for i in range(seq_len):
        nki.simulate_kernel(mamba_inference_kernel_[batch_size],
                            xBC_tensor[:, i:i+1, :], conv_state_tensor, conv_weight_tensor, conv_bias,
                            initial_state, dt_softplus[:, i:i+1, :], A_tensor,
                            D, out_S,
                            out_Y_step, out_conv_state, n_groups=n_groups, activation='silu')
        conv_state_tensor = out_conv_state
        initial_state = out_S
        output_list.append(out_Y_step.copy())

    conv_out = nki.simulate_kernel(conv1d_with_transpose, xBC_tensor, conv_weight_tensor[:, None, :], conv_bias,
                                   activation='silu')
    # dt_softplus = np.empty_like(dt)
    # nki.simulate_kernel(softplus_with_bias_test_kernel_, dt, dt_bias, dt_softplus)
    out_Y_batch = np.empty((batch_size, seq_len, d_inner)).astype(np.float32)
    nki.simulate_kernel(mamba_kernel_test_, dt_softplus, A_tensor, conv_out, D, n_groups, d_state, d_inner, d_head,
                        out_Y_batch)

    out_Y_step_seq = np.concatenate(output_list, axis=1)
    assert out_Y_batch.shape == out_Y_step_seq.shape
    assert not np.isnan(out_Y_batch).any()
    assert not np.isinf(out_Y_batch).any()
    assert not np.isnan(out_Y_step_seq).any()
    assert not np.isinf(out_Y_step_seq).any()
    print(f"out_Y_batch: {out_Y_batch}")
    print(f"out_Y_step: {out_Y_step_seq}")
    assert np.allclose(out_Y_batch, out_Y_step, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])