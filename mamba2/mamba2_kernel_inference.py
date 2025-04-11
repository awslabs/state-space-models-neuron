import torch
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from torch_neuronx import nki_jit
from mamba2.conv1d_grouped import apply_activation


def vector_mul(tile, column):
    return nisa.tensor_scalar(tile, op0=nl.multiply, operand0=1, op1=nl.multiply, operand1=column)


def conv(input_tensor, state_tensor, weight_tensor, conv_bias, start, size, batch_id, out_conv_state, k_conv=4,
         activation='silu'):
    tile_slice = slice(start, start + size)
    conv_state = nl.ndarray((size, k_conv), dtype=input_tensor.dtype)
    conv_bias = nl.load(conv_bias[tile_slice])
    conv_state[:, :k_conv - 1] = nl.load(state_tensor[batch_id, tile_slice, 1:])  # (128)
    conv_state[:, k_conv - 1] = nl.load(input_tensor[batch_id, 0, tile_slice])  # (128)
    conv_weights = nl.load(weight_tensor[tile_slice, :])  # (128, 4)
    nl.store(out_conv_state[batch_id, tile_slice, :], conv_state)
    res = nl.sum(conv_state * conv_weights, axis=1)
    return apply_activation(res, conv_bias, activation=activation)


@nki_jit
def mamba_inference_kernel_(xBC_tensor, conv_state_tensor, conv_weight_tensor, conv_bias_tensor, S_tensor, dt_tensor,
                            logA_tensor, D_tensor,
                            out_S_tensor,
                            out_Y_tensor,
                            out_conv_tensor,
                            n_groups,
                            activation='silu'
                            ):
    """
    xBC: (batch_size, 1, d_inner + 2 * ngroups * d_state)
    conv_tensor: (batch_size, d_inner + 2 * ngroups * d_state, 4)
    conv_weight_tensor: (d_inner + 2 * ngroups * d_state, 4)
    S_tensor: (batch_size, n_head, d_head, d_state)
    dt_tensor: (batch_size, 1, n_heads)
    logA_tensor: (n_heads)
    D_tensor: (n_heads)
    """

    dtype = xBC_tensor.dtype
    batch_size, n_heads, d_head, d_state = S_tensor.shape
    seq_len = dt_tensor.shape[1]
    d_inner = n_heads * d_head
    k_conv = conv_weight_tensor.shape[-1]
    d_xBC = xBC_tensor.shape[-1]

    assert d_xBC == d_inner + 2 * n_groups * d_state, f"Expected d_xBC {d_inner + 2 * n_groups * d_state}, got {d_xBC}"
    assert seq_len == 1
    assert d_inner == out_Y_tensor.shape[-1]
    assert n_heads % n_groups == 0, "Number of heads must be divisible by number of groups"

    heads_per_group = n_heads // n_groups
    batch_id = nl.program_id(0)

    # (nhead,)
    logA_cache = nl.load(logA_tensor.reshape((1, n_heads))).broadcast_to((d_head, n_heads))

    if D_tensor is not None:
        D = nl.load(D_tensor.reshape((1, n_heads))).broadcast_to((d_head, n_heads))
    else:
        D = None

    for group_id in nl.affine_range(n_groups):
        B = conv(xBC_tensor, conv_state_tensor, conv_weight_tensor, conv_bias_tensor,
                 start=d_inner + group_id * d_state,
                 size=d_state,
                 batch_id=batch_id, out_conv_state=out_conv_tensor, k_conv=k_conv)
        B = nl.transpose(B).broadcast_to((d_head, d_state))

        C = conv(xBC_tensor, conv_state_tensor, conv_weight_tensor, conv_bias_tensor,
                 start=d_inner + n_groups * d_state + group_id * d_state,
                 size=d_state,
                 batch_id=batch_id, out_conv_state=out_conv_tensor, k_conv=k_conv)

        dt = nl.load(dt_tensor[batch_id, :, :]).broadcast_to((d_head, n_heads))
        A = nl.exp(logA_cache * dt)

        base_head_idx = group_id * heads_per_group
        for head_offset in nl.affine_range(heads_per_group):
            head_id = base_head_idx + head_offset

            x = conv(xBC_tensor, conv_state_tensor, conv_weight_tensor, conv_bias_tensor,
                     start=head_id * d_head,
                     size=d_head,
                     batch_id=batch_id, out_conv_state=out_conv_tensor, k_conv=k_conv, activation=activation)

            # (dhead, dstate)
            S = nl.load(S_tensor[batch_id, head_id, :, :])

            dx = dt[:, head_id] * x
            Bx = dx * B
            S_new = S * A[:, head_id] + Bx

            nl.store(out_S_tensor[batch_id, head_id, :, :], S_new)

            y = nl.matmul(S_new, C)
            y = y + D[:, head_id] * x

            nl.store(out_Y_tensor[batch_id, 0, d_head * head_id:(head_id + 1) * d_head], y)


def mamba2_kernel_inference(xBC_tensor, conv_state_tensor, conv_weight_tensor, conv_bias_tensor, S_tensor, dt_tensor,
                            logA_tensor,
                            D_tensor=None, n_groups=1):
    batch_size, n_heads, d_head, d_state = S_tensor.shape
    seq_len = xBC_tensor.shape[1]
    assert seq_len == 1
    d_inner = n_heads * d_head
    out_Y = torch.empty((batch_size, seq_len, d_inner), device=xBC_tensor.device, dtype=xBC_tensor.dtype)
    out_conv_state = torch.empty_like(conv_state_tensor)
    out_S = torch.empty_like(S_tensor)
    mamba_inference_kernel_[batch_size](xBC_tensor, conv_state_tensor, conv_weight_tensor, conv_bias_tensor, S_tensor,
                                        dt_tensor,
                                        logA_tensor,
                                        D_tensor, out_S,
                                        out_Y, out_conv_state, n_groups=n_groups)
    return out_Y, out_conv_state, out_S

