import math
import os

import torch
import torch_xla.core.xla_model as xm

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np

from .utils import chunk
from .mamba2_kernel import mamba_kernel_, mamba_kernel_bwd_
from .conv1d_grouped import conv1d_with_transpose, conv1d_with_transpose_bwd
import copy


# for unit testing, remove creation of output tensor within kernel to avoid complaints when testing on CPU
def softplus_with_bias_test_kernel_(input_tensor, bias_tensor, output_tensor):
    batch_size, seq_len, d_inner = input_tensor.shape
    bias_tensor = bias_tensor.reshape((1, d_inner))

    bias = nl.load(bias_tensor[:1, :]).broadcast_to((128, d_inner))

    for batch_id in range(batch_size):
        for seq_id in range(seq_len // 128):
            seq_slice = chunk(seq_id, 128)
            input = nl.load(input_tensor[batch_id, seq_slice, :])
            output = nl.softplus(input + bias)
            nl.store(output_tensor[batch_id, seq_slice, :], output)


def softplus_with_bias(input_tensor, bias_tensor, output_tensor=None):
    batch_size, seq_len, d_inner = input_tensor.shape
    bias_tensor = bias_tensor.reshape((1, d_inner))

    if output_tensor is None:
        output_tensor = nl.ndarray(input_tensor.shape, input_tensor.dtype, buffer=nl.private_hbm)
    bias = nl.load(bias_tensor[:1, :]).broadcast_to((128, d_inner))

    for batch_id in range(batch_size):
        for seq_id in range(seq_len // 128):
            seq_slice = chunk(seq_id, 128)
            input = nl.load(input_tensor[batch_id, seq_slice, :])
            output = nl.softplus(input + bias)
            nl.store(output_tensor[batch_id, seq_slice, :], output)
    return output_tensor


def softplus_with_bias_bwd(input_tensor, bias_tensor, d_output_tensor, d_bias_tensor):
    # todo: check of some part needs to be done in fp32
    dtype = input_tensor.dtype
    batch_size, seq_len, channels = input_tensor.shape
    bias_tensor = bias_tensor.reshape((1, channels))

    d_input_tensor = nl.ndarray(input_tensor.shape, input_tensor.dtype, buffer=nl.private_hbm)
    # d_bias_tensor = nl.ndarray((d_inner,), bias_tensor.dtype, buffer=nl.private_hbm)

    bias = nl.load(bias_tensor[:1, :]).broadcast_to((128, channels))
    d_bias = nl.zeros_like(bias)
    ones = nl.ones((128, 1), dtype=dtype)

    for batch_id in range(batch_size):
        for seq_id in range(seq_len // 128):
            seq_slice = chunk(seq_id, 128)
            input = nl.load(input_tensor[batch_id, seq_slice, :])
            d_output = nl.load(d_output_tensor[batch_id, seq_slice, :])
            d_input = d_output * nl.sigmoid(input + bias)
            nl.store(d_input_tensor[batch_id, seq_slice, :], d_input)
            d_bias += d_input
    d_bias_reduced = nl.matmul(d_bias, ones, transpose_x=True)
    nl.store(d_bias_tensor[:], d_bias_reduced)
    return d_input_tensor, d_bias_tensor


def rmsnorm(input_tensor, z_tensor, weight_tensor, output_tensor, n_groups, eps=1e-5):
    # note: pass n_groups = 1 if you don't want to normalize within groups

    dtype = input_tensor.dtype
    batch_size, seq_len, d_inner = input_tensor.shape
    assert d_inner % n_groups == 0
    channels_per_group = d_inner // n_groups

    weight_tensor = weight_tensor.reshape((1, weight_tensor.shape[0]))

    # output_tensor = nl.ndarray((batch_size, seq_len, d_inner), dtype=dtype, buffer=nl.private_hbm)

    for group_id in range(n_groups):
        group_slice = chunk(group_id, channels_per_group)
        w = nl.load(weight_tensor[:1, group_slice])
        w_bcast = w.broadcast_to((128, channels_per_group))
        for batch_id in range(batch_size):
            for tile_id in range(seq_len // 128):
                tile_slice = chunk(tile_id, 128)
                x = nl.load(input_tensor[batch_id, tile_slice, group_slice], dtype=nl.float32)
                if z_tensor is not None:
                    z = nl.load(z_tensor[batch_id, tile_slice, group_slice], dtype=nl.float32)
                    x = x * nl.silu(z)
                variance = nl.mean(nl.square(x), axis=[1])
                normalized_x = x * nl.rsqrt(variance + eps)
                out = normalized_x * w_bcast
                nl.store(output_tensor[batch_id, tile_slice, group_slice], out)
    return output_tensor


def sum_partition(x_tile: nl.ndarray, ones: nl.ndarray) -> nl.ndarray:
    """Return the reduce sum of x_tile across the partition dimension.
    Computed using the tensor engine by multiplying with a vector of all ones.

    :param x_tile: (p, seq_len) tile to sum across p
    :param ones: (p, 1) Should be nl.ones((p,1)), we pass it as argument to avoid creating a constant every function call
    :return: sum(x_tile, dim=0) of shape (1, seq_len)
    """
    p, seq_len = x_tile.shape
    assert seq_len % 512 == 0, "seq_len must be multiple of 512"

    i_p = nl.arange(p)[:, None]
    res = nl.ndarray(shape=(1, seq_len), dtype=x_tile.dtype)
    i_f_tile0 = nl.arange(512)[None, :]
    for i in nl.static_range(seq_len // 512):
        i_f_tile = i * 512 + i_f_tile0
        psum_tile = nl.matmul(ones, x_tile[i_p, i_f_tile], transpose_x=True)
        res[nl.arange(1)[:, None], i_f_tile] = nl.copy(psum_tile)  # copy the result from psum to sbuf
    return res


def rmsnorm_bwd(input_tensor, z_tensor, weight_tensor, d_out_tensor, n_groups, eps=1e-5):
    # note: pass n_groups = 1 if you don't want to normalize within groups

    dtype = input_tensor.dtype
    batch_size, seq_len, d_inner = input_tensor.shape
    assert d_inner % n_groups == 0
    channels_per_group = d_inner // n_groups

    weight_tensor = weight_tensor.reshape((1, d_inner))

    d_input_tensor = nl.ndarray((batch_size, seq_len, d_inner), dtype=dtype, buffer=nl.private_hbm)
    d_z_tensor = nl.ndarray((batch_size, seq_len, d_inner), dtype=dtype, buffer=nl.private_hbm)
    d_weight_tensor = nl.ndarray((1, d_inner), dtype=dtype, buffer=nl.private_hbm)

    ones = nl.ones((128, 1), dtype=dtype)

    for group_id in range(n_groups):
        group_slice = chunk(group_id, channels_per_group)
        w = nl.load(weight_tensor[:1, group_slice])
        w_bcast = w.broadcast_to((128, channels_per_group))
        # eventually we need to sum over first axis, but it is faster to do it only once at the end
        d_w = nl.zeros((128, channels_per_group), dtype=dtype)
        for batch_id in range(batch_size):
            for tile_id in range(seq_len // 128):
                tile_slice = chunk(tile_id, 128)
                x = nl.load(input_tensor[batch_id, tile_slice, group_slice], dtype=nl.float32)
                g = nl.load(d_out_tensor[batch_id, tile_slice, group_slice], dtype=nl.float32)
                if z_tensor is not None:
                    z = nl.load(z_tensor[batch_id, tile_slice, group_slice], dtype=nl.float32)
                    gate = nl.silu(z)
                else:
                    gate = 1
                h = x * gate

                variance = nl.mean(nl.square(h), axis=[1]) + eps
                rms_inv = nl.rsqrt(variance)
                gw = g * w_bcast
                common_factor = np.mean(h * gw, axis=[1]) / variance
                d_h = rms_inv * (gw - common_factor * h)

                nl.store(d_input_tensor[batch_id, tile_slice, group_slice], d_h * gate)
                if z_tensor is not None:
                    nl.store(d_z_tensor[batch_id, tile_slice, group_slice], d_h * x * nl.silu_dx(z))

                d_w += g * h * rms_inv

        d_w_reduced = sum_partition(d_w, ones)
        nl.store(d_weight_tensor[:, group_slice], d_w_reduced)
    return d_input_tensor, d_z_tensor, d_weight_tensor


def split_zxBCdt_shared_hbm(input_tensor, z, xBC, dt):
    batch_size, seq_len, d_inner = input_tensor.shape
    sizes = [z.shape[2], xBC.shape[2], dt.shape[2]]
    assert d_inner == sum(sizes)

    for batch_id in range(batch_size):
        for seq_id in range(seq_len // 128):
            seq_slice = chunk(seq_id, 128)
            input = nl.load(input_tensor[batch_id, seq_slice, :])
            offset = 0

            size = sizes[0]
            nl.store(z[batch_id, seq_slice, :], input[:, offset:offset + size])
            offset += size
            
            size = sizes[1]
            nl.store(xBC[batch_id, seq_slice, :], input[:, offset:offset + size])
            offset += size
            
            size = sizes[2]
            nl.store(dt[batch_id, seq_slice, :], input[:, offset:offset + size])
            offset += size
    return z, xBC, dt


def split(input_tensor, size0, size1, size2):
    batch_size, seq_len, d_inner = input_tensor.shape
    sizes = [size0, size1, size2]
    assert d_inner == sum(sizes)

    output_tensors = tuple(
        nl.ndarray((batch_size, seq_len, s), dtype=input_tensor.dtype, buffer=nl.private_hbm) for s in sizes)

    for batch_id in range(batch_size):
        for seq_id in range(seq_len // 128):
            seq_slice = chunk(seq_id, 128)
            input = nl.load(input_tensor[batch_id, seq_slice, :])
            offset = 0
            for i in nl.static_range(len(sizes)):
                size = sizes[i]
                nl.store(output_tensors[i][batch_id, seq_slice, :], input[:, offset:offset + size])
                offset += size

    return output_tensors


def merge(input0_tensor, input1_tensor, input2_tensor, out=None):
    input_tensors = (input0_tensor, input1_tensor, input2_tensor)
    # for v in input_tensors:
    #     print(v.shape)

    batch_size, seq_len, _ = input0_tensor.shape
    sizes = [x.shape[2] for x in input_tensors]
    d_inner = sum(sizes)

    output_tensor = nl.ndarray((batch_size, seq_len, d_inner), dtype=input0_tensor.dtype) if out is None else out
    assert output_tensor.shape[2] == d_inner

    for batch_id in range(batch_size):
        for seq_id in range(seq_len // 128):
            seq_slice = chunk(seq_id, 128)
            # output buffer to ensure all DMAs are on contiguous memory (not sure if it actually improves performance)
            output = nl.ndarray((128, d_inner), dtype=input0_tensor.dtype)
            offset = 0
            for i in nl.static_range(len(sizes)):
                size = sizes[i]
                output[:, offset:offset + size] = nl.load(input_tensors[i][batch_id, seq_slice, :])
                offset += size
            nl.store(output_tensor[batch_id, seq_slice, :], output)

    return output_tensor


@nki.jit
def fused_mamba2_kernel_fwd(zxBCdt_tensor, logA_tensor, D_tensor, conv_weight, conv_bias, dt_bias,
                        norm_weight_tensor, n_groups, d_state, d_inner, d_head):
    norm_eps = 1e-5
    normalize_within_group = True
    dtype = zxBCdt_tensor.dtype
    batch_size, seq_len, zxBCdt_size = zxBCdt_tensor.shape
    n_heads = d_inner // d_head
    d_conv = d_inner + 2 * n_groups * d_state
    assert zxBCdt_size == d_inner + d_conv + n_heads

    z = nl.ndarray((batch_size, seq_len, d_inner), dtype=dtype, buffer=nl.shared_hbm)
    xBC = nl.ndarray((batch_size, seq_len, d_conv), dtype=dtype, buffer=nl.shared_hbm)
    dt = nl.ndarray((batch_size, seq_len, n_heads), dtype=dtype, buffer=nl.shared_hbm)
    # split_sizes = [d_inner, d_conv, n_heads]
    # z, xBC, dt = split(zxBCdt_tensor, d_inner, d_conv, n_heads)
    z, xBC, dt = split_zxBCdt_shared_hbm(zxBCdt_tensor, z, xBC, dt)
    
    conv_out = nl.ndarray(xBC.shape, dtype=dtype, buffer=nl.shared_hbm)
    conv_out = conv1d_with_transpose(xBC, conv_weight, conv_bias, conv_out, activation='silu')
    
    dt_softplus = nl.ndarray(dt.shape, dtype=dtype, buffer=nl.shared_hbm)
    dt_softplus = softplus_with_bias(dt, dt_bias, dt_softplus)
    
    scan_out = nl.ndarray((batch_size, seq_len, d_inner), dtype=dtype, buffer=nl.shared_hbm)
    scan_out = mamba_kernel_(dt_softplus, logA_tensor, conv_out, D_tensor, n_groups, d_state, d_inner, d_head, scan_out)

    # fixme: output tensor must be in shared_hbm but cannot create shared_hbm inside called functions
    norm_out = nl.ndarray((batch_size, seq_len, d_inner), dtype=dtype, buffer=nl.shared_hbm)
    norm_out = rmsnorm(scan_out, z, norm_weight_tensor, norm_out, n_groups, norm_eps)

    return norm_out, z, xBC, dt, conv_out, dt_softplus, scan_out


@nki.jit
def fused_mamba2_kernel_bwd_no_recompute(zxBCdt_tensor, logA_tensor, D_tensor, conv_weight, conv_bias, dt_bias,
                                        norm_weight_tensor, d_output_tensor, z, xBC, dt, conv_out, dt_softplus, scan_out, n_groups, d_state, d_inner, d_head):
    norm_eps = 1e-5
    normalize_within_group = True
    dtype = zxBCdt_tensor.dtype
    batch_size, seq_len, zxBCdt_size = zxBCdt_tensor.shape
    assert batch_size == 1, "Need to figure out how to call the mamba_kernel_bwd with batch_size > 1"
    n_heads = d_inner // d_head
    d_conv = d_inner + 2 * n_groups * d_state
    assert zxBCdt_size == d_inner + d_conv + n_heads

    # backward
    d_scan_out, d_z, d_norm_weight = rmsnorm_bwd(scan_out, z, norm_weight_tensor, d_output_tensor, n_groups, norm_eps)
    d_scan_out = d_scan_out.reshape((batch_size, seq_len, n_heads, d_head))

    d_dt_softplus = nl.ndarray(dt_softplus.shape, dtype=dt_softplus.dtype, buffer=nl.shared_hbm)
    d_logA = nl.ndarray((batch_size, n_heads), dtype=logA_tensor.dtype, buffer=nl.shared_hbm)
    d_D = nl.ndarray((batch_size, n_heads), dtype=D_tensor.dtype,
                     buffer=nl.shared_hbm) if D_tensor is not None else None

    x, B, C = split(conv_out, d_inner, n_groups * d_state, n_groups * d_state)
    x = x.reshape((batch_size, seq_len, n_heads, d_head))
    B = B.reshape((batch_size, seq_len, n_groups, d_state))
    C = C.reshape((batch_size, seq_len, n_groups, d_state))

    d_x = nl.ndarray((batch_size, seq_len, n_heads, d_head), dtype=nl.float32, buffer=nl.shared_hbm)
    d_B = nl.ndarray((batch_size, seq_len, n_groups, d_state), dtype=nl.float32, buffer=nl.shared_hbm)
    d_C = nl.ndarray((batch_size, seq_len, n_groups, d_state), dtype=nl.float32, buffer=nl.shared_hbm)

    mamba_kernel_bwd_(dt_tensor=dt_softplus.reshape((batch_size, seq_len, n_heads)),
                      logA_tensor=logA_tensor.reshape((n_heads,)),
                      X_tensor=x,
                      B_tensor=B,
                      C_tensor=C,
                      d_out_tensor=d_scan_out.reshape((batch_size, seq_len, n_heads, d_head)),
                      ddt_tensor=d_dt_softplus.reshape((batch_size, seq_len, n_heads)),
                      dlogA_tensor=d_logA,
                      dX_tensor=d_x,
                      dB_tensor=d_B,
                      dC_tensor=d_C,
                      D_tensor=D_tensor.reshape((n_heads,)),
                      dD_tensor=d_D,
                      )

    d_conv_out = nl.ndarray(xBC.shape, dtype=dtype, buffer=nl.private_hbm)
    d_conv_out = merge(d_x.reshape([batch_size, seq_len, d_inner]),
                       d_B.reshape([batch_size, seq_len, d_state * n_groups]),
                       d_C.reshape([batch_size, seq_len, d_state * n_groups]),
                       out=d_conv_out)

    d_dt_bias = nl.ndarray(dt_bias.shape, dtype=dtype, buffer=nl.shared_hbm)

    d_dt, d_dt_bias = softplus_with_bias_bwd(dt, dt_bias, d_dt_softplus, d_dt_bias)
    d_xBC, d_conv_weight, d_conv_bias = conv1d_with_transpose_bwd(xBC, conv_weight, conv_bias, d_conv_out,
                                                                  activation='silu')

    d_zxBCdt = nl.ndarray(zxBCdt_tensor.shape, dtype=dtype, buffer=nl.shared_hbm)
    d_zxBCdt = merge(d_z.reshape((batch_size, seq_len, d_inner)),
                     d_xBC.reshape((batch_size, seq_len, d_conv)),
                     d_dt.reshape((batch_size, seq_len, n_heads)),
                     out=d_zxBCdt)

    return d_zxBCdt, d_logA, d_D, d_conv_weight, d_conv_bias, d_dt_bias, d_norm_weight


@nki.jit
def fused_mamba2_kernel_bwd(zxBCdt_tensor, logA_tensor, D_tensor, conv_weight, conv_bias, dt_bias,
                            norm_weight_tensor, d_output_tensor, n_groups, d_state, d_inner, d_head):
    norm_eps = 1e-5
    normalize_within_group = True
    dtype = zxBCdt_tensor.dtype
    batch_size, seq_len, zxBCdt_size = zxBCdt_tensor.shape
    assert batch_size == 1, "Need to figure out how to call the mamba_kernel_bwd with batch_size > 1"
    n_heads = d_inner // d_head
    d_conv = d_inner + 2 * n_groups * d_state
    assert zxBCdt_size == d_inner + d_conv + n_heads

    z, xBC, dt = split(zxBCdt_tensor, d_inner, d_conv, n_heads)
    conv_out = conv1d_with_transpose(xBC, conv_weight, conv_bias, activation='silu')

    dt_softplus = softplus_with_bias(dt, dt_bias)
    scan_out = mamba_kernel_(dt_softplus, logA_tensor, conv_out, D_tensor, n_groups, d_state, d_inner, d_head)

    # backward
    d_scan_out, d_z, d_norm_weight = rmsnorm_bwd(scan_out, z, norm_weight_tensor, d_output_tensor, n_groups, norm_eps)
    d_scan_out = d_scan_out.reshape((batch_size, seq_len, n_heads, d_head))

    d_dt_softplus = nl.ndarray(dt_softplus.shape, dtype=dt_softplus.dtype, buffer=nl.shared_hbm)
    d_logA = nl.ndarray((batch_size, n_heads), dtype=logA_tensor.dtype, buffer=nl.shared_hbm)
    d_D = nl.ndarray((batch_size, n_heads), dtype=D_tensor.dtype,
                     buffer=nl.shared_hbm) if D_tensor is not None else None

    x, B, C = split(conv_out, d_inner, n_groups * d_state, n_groups * d_state)
    x = x.reshape((batch_size, seq_len, n_heads, d_head))
    B = B.reshape((batch_size, seq_len, n_groups, d_state))
    C = C.reshape((batch_size, seq_len, n_groups, d_state))

    d_x = nl.ndarray((batch_size, seq_len, n_heads, d_head), dtype=nl.float32, buffer=nl.shared_hbm)
    d_B = nl.ndarray((batch_size, seq_len, n_groups, d_state), dtype=nl.float32, buffer=nl.shared_hbm)
    d_C = nl.ndarray((batch_size, seq_len, n_groups, d_state), dtype=nl.float32, buffer=nl.shared_hbm)

    mamba_kernel_bwd_(dt_tensor=dt_softplus.reshape((batch_size, seq_len, n_heads)),
                      logA_tensor=logA_tensor.reshape((n_heads,)),
                      X_tensor=x,
                      B_tensor=B,
                      C_tensor=C,
                      d_out_tensor=d_scan_out.reshape((batch_size, seq_len, n_heads, d_head)),
                      ddt_tensor=d_dt_softplus.reshape((batch_size, seq_len, n_heads)),
                      dlogA_tensor=d_logA,
                      dX_tensor=d_x,
                      dB_tensor=d_B,
                      dC_tensor=d_C,
                      D_tensor=D_tensor.reshape((n_heads,)),
                      dD_tensor=d_D,
                      )

    d_conv_out = nl.ndarray(xBC.shape, dtype=dtype, buffer=nl.private_hbm)
    d_conv_out = merge(d_x.reshape([batch_size, seq_len, d_inner]),
                       d_B.reshape([batch_size, seq_len, d_state * n_groups]),
                       d_C.reshape([batch_size, seq_len, d_state * n_groups]),
                       out=d_conv_out)

    d_dt_bias = nl.ndarray(dt_bias.shape, dtype=dtype, buffer=nl.shared_hbm)

    d_dt, d_dt_bias = softplus_with_bias_bwd(dt, dt_bias, d_dt_softplus, d_dt_bias)
    d_xBC, d_conv_weight, d_conv_bias = conv1d_with_transpose_bwd(xBC, conv_weight, conv_bias, d_conv_out,
                                                                  activation='silu')

    d_zxBCdt = nl.ndarray(zxBCdt_tensor.shape, dtype=dtype, buffer=nl.shared_hbm)
    d_zxBCdt = merge(d_z.reshape((batch_size, seq_len, d_inner)),
                     d_xBC.reshape((batch_size, seq_len, d_conv)),
                     d_dt.reshape((batch_size, seq_len, n_heads)),
                     out=d_zxBCdt)

    return d_zxBCdt, d_logA, d_D, d_conv_weight, d_conv_bias, d_dt_bias, d_norm_weight


class Mamba2FusedKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, zxBCdt, logA, D, conv1d_weight, conv1d_bias, dt_bias, 
                norm_weight, n_groups, ssm_state_size, intermediate_size, 
                head_dim):
        norm_out, *_ = fused_mamba2_kernel_fwd(zxBCdt, logA, D, conv1d_weight, 
                                           conv1d_bias,dt_bias, norm_weight,
                                           n_groups, ssm_state_size, 
                                           intermediate_size, head_dim)
        ctx.n_groups = n_groups
        ctx.ssm_state_size = ssm_state_size
        ctx.intermediate_size = intermediate_size
        ctx.head_dim = head_dim
        ctx.save_for_backward(zxBCdt, logA, D, conv1d_weight, conv1d_bias, dt_bias, norm_weight)
        
        return norm_out

    @staticmethod
    def backward(ctx, d_output):
        zxBCdt, logA, D, conv1d_weight, conv1d_bias, dt_bias, norm_weight = ctx.saved_tensors 
        d_zxBCdt, d_logA, d_D, d_conv_weight, d_conv_bias, d_dt_bias, d_norm_weight = fused_mamba2_kernel_bwd(
            zxBCdt, logA, D, conv1d_weight, conv1d_bias, dt_bias,norm_weight, d_output, ctx.n_groups, 
            ctx.ssm_state_size, ctx.intermediate_size, ctx.head_dim)
        return d_zxBCdt, d_logA, d_D, d_conv_weight, d_conv_bias, d_dt_bias, d_norm_weight, None, None, None, None


class Mamba2FusedKernelWithCaching(torch.autograd.Function):
    @staticmethod
    def forward(ctx, zxBCdt, logA, D, conv1d_weight, conv1d_bias, dt_bias, 
                norm_weight, n_groups, ssm_state_size, intermediate_size, 
                head_dim):
        norm_out, z, xBC, dt, conv_out, dt_softplus, scan_out = fused_mamba2_kernel_fwd(zxBCdt, logA, D, conv1d_weight, 
                                           conv1d_bias,dt_bias, norm_weight,
                                           n_groups, ssm_state_size, 
                                           intermediate_size, head_dim)
        ctx.n_groups = n_groups
        ctx.ssm_state_size = ssm_state_size
        ctx.intermediate_size = intermediate_size
        ctx.head_dim = head_dim
        ctx.save_for_backward(zxBCdt, logA, D, conv1d_weight, conv1d_bias, dt_bias, norm_weight, z, xBC, dt, conv_out, dt_softplus, scan_out)
        
        return norm_out

    @staticmethod
    def backward(ctx, d_output):
        zxBCdt, logA, D, conv1d_weight, conv1d_bias, dt_bias, norm_weight, z, xBC, dt, conv_out, dt_softplus, scan_out = ctx.saved_tensors 
        d_zxBCdt, d_logA, d_D, d_conv_weight, d_conv_bias, d_dt_bias, d_norm_weight = fused_mamba2_kernel_bwd_no_recompute(
            zxBCdt, logA, D, conv1d_weight, conv1d_bias, dt_bias, norm_weight, d_output, z, xBC, dt, conv_out, dt_softplus, scan_out, ctx.n_groups, ctx.ssm_state_size, ctx.intermediate_size, ctx.head_dim)
        return d_zxBCdt, d_logA, d_D, d_conv_weight, d_conv_bias, d_dt_bias, d_norm_weight, None, None, None, None


def mamba2_fused_kernel(zxBCdt, logA, D, conv1d_weight, conv1d_bias, dt_bias, 
                        norm_weight, n_groups, ssm_state_size, intermediate_size, 
                        head_dim):
    return Mamba2FusedKernel.apply(zxBCdt, logA, D, conv1d_weight, conv1d_bias, dt_bias, 
                                   norm_weight, n_groups, ssm_state_size, intermediate_size, 
                                   head_dim)


def mamba2_fused_kernel_with_caching(zxBCdt, logA, D, conv1d_weight, conv1d_bias, dt_bias, 
                        norm_weight, n_groups, ssm_state_size, intermediate_size, 
                        head_dim):
    return Mamba2FusedKernelWithCaching.apply(zxBCdt, logA, D, conv1d_weight, conv1d_bias, dt_bias, 
                                              norm_weight, n_groups, ssm_state_size, intermediate_size, 
                                              head_dim)


def how_close_rtol(out, ref, rtol=1.e-2):
    assert out.shape == ref.shape, f"Shapes don't match! {out.shape} != {ref.shape}"
    return (~torch.isclose(out, ref, rtol=rtol)).sum() / out.numel()


def how_close_atol(out, ref, atol=1.e-6):
    assert out.shape == ref.shape, f"Shapes don't match! {out.shape} != {ref.shape}"
    return (~torch.isclose(out, ref, atol=atol)).sum() / out.numel()


def print_stats(tensor, ref_tensor, tensor_type):
        print(f'###################### {tensor_type} ######################')
        print(tensor.dtype, ref_tensor.dtype)

        tensor = tensor.cpu().to(torch.float32)
        ref_tensor = ref_tensor.cpu().to(torch.float32)
        if torch.allclose(tensor, ref_tensor, atol=1e-3, rtol=3e-2):
            print("NKI and Torch MATCH")
        else:
            print("NKI and Torch DIFFER")

        print(f"Max absolute error:  {torch.abs(tensor - ref_tensor).max():.5f}")
        print(f"Mean absolute error: {torch.abs(tensor - ref_tensor).mean():.5f}")
        print(f"norm: {tensor.abs().mean()} ref norm: {ref_tensor.abs().mean()}")
        # print(f"Max rel. error:  {(torch.abs(tensor - ref_tensor) / ref_tensor.abs().clamp(min=0.001)).max() * 100:.5f}%")
        print(f"Mean rel. error: {(torch.abs(tensor - ref_tensor) / ref_tensor.abs().clamp(min=0.001)).mean() * 100:.5f}%")

        print("Largest relative discrepancy:")
        idx = (torch.abs(tensor - ref_tensor) / ref_tensor.abs().clamp(min=0.001)).argmax()
        
        print("rows ")
        
        print(f"index {np.unravel_index(idx, tensor.shape)} out={tensor.flatten()[idx].item():.5f} ref={ref_tensor.flatten()[idx].item():.5f}")


if __name__ == "__main__":
    device = xm.xla_device()
    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["XLA_HLO_DEBUG"] = "1"
    os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
    os.environ["NEURON_CC_FLAGS"] = "--auto-cast=none"

    import numpy as np
    import transformers
    torch.manual_seed(42)
    np.random.seed(42)
    
    batch, seq_len, d_head, d_state, n_groups = 1, 1024, 128, 128, 1
    d_model = 256
    d_inner = d_model * 2
    n_heads = d_inner // d_head
    d_conv = d_inner + 2 * n_groups * d_state
    zxBCdt_size = d_inner + d_conv + n_heads
    print(f"d_conv {d_conv} zxBCdt {zxBCdt_size}")

    dtype = torch.float32

    cfg = transformers.models.mamba2.Mamba2Config(num_heads=n_heads, head_dim=d_head, hidden_size=d_model,
                                                  n_groups=1)
    
    # Validate fwd pass
    print("Validating fwd...")
    layer = transformers.models.mamba2.modeling_mamba2.Mamba2Mixer(cfg, 1)
    u = torch.randn(batch, seq_len, d_model, dtype=dtype)
    # NOTE: to get projected_states (zxBCdt), modify the reference transformers implemenation to also return projected_states at this line https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py#L653  
    layer_out_ref, projected_states = layer(u)
    projected_states.retain_grad()
    d_output = torch.randn_like(layer_out_ref)
    layer_out_ref.backward(d_output)
    xm.mark_step()

    # Make copies to pass to the fused kernel to avoid in-place update of the original layer.*.grad    
    layer = layer.to(device=device)
    in_proj, out_proj = copy.deepcopy(layer.in_proj), copy.deepcopy(layer.out_proj)
    D = layer.D.detach().clone().requires_grad_(True)
    conv1d_weight = layer.conv1d.weight.detach().clone().requires_grad_(True)
    conv1d_bias = layer.conv1d.bias.detach().clone().requires_grad_(True)
    dt_bias = layer.dt_bias.detach().clone().requires_grad_(True)
    norm_weight = layer.norm.weight.detach().clone().requires_grad_(True)
    logA = -torch.exp(layer.A_log.detach().clone())#.requires_grad_(True)  # (num_heads) or (intermediate_size, state_size)
    logA.requires_grad_(True)
    xm.mark_step()
    zxBCdt = in_proj(u.to(device=device))
    xm.mark_step()
    zxBCdt.retain_grad()
    
    # norm_out = mamba2_fused_kernel(zxBCdt, logA, D,
    #                                conv1d_weight, conv1d_bias,
    #                                dt_bias, norm_weight,
    #                                layer.n_groups, layer.ssm_state_size, layer.intermediate_size, layer.head_dim)
    norm_out = mamba2_fused_kernel_with_caching(zxBCdt, logA, D,
                                                conv1d_weight, conv1d_bias,
                                                dt_bias, norm_weight,
                                                layer.n_groups, layer.ssm_state_size, layer.intermediate_size, layer.head_dim)
    xm.mark_step()
    layer_out = out_proj(norm_out)
    xm.mark_step()

    print(layer_out_ref)
    print(layer_out)

    layer_out_ref = layer_out_ref.to(device=device)
    n = layer_out.numel()
    print("Fraction not close rtol 1e-2:", (~torch.isclose(layer_out_ref, layer_out, rtol=1e-2)).sum() / n)
    print("Fraction not close atol 1e-6:", (~torch.isclose(layer_out_ref, layer_out, atol=1e-6)).sum() / n)
    print("Fraction not close atol 1e-7:", (~torch.isclose(layer_out_ref, layer_out, atol=1e-7)).sum() / n)

    # Validate bwd pass
    print("Validating bwd...")
    layer_out.backward(d_output.to(device=device))
    xm.mark_step()
    
    print("out_proj.weight.grad rtol=1e-2: ", how_close_rtol(out_proj.weight.grad, layer.out_proj.weight.grad, rtol=1e-2))
    print("projected_states.grad rtol=1e-2: ", how_close_rtol(zxBCdt.grad.to("cpu"), projected_states.grad, rtol=1e-2))
    
    # NOTE: in_proj have 50% mismatches at 1% rtol.
    print("in_proj.weight.grad rtol=1e-2: ", how_close_rtol(in_proj.weight.grad, layer.in_proj.weight.grad, rtol=1e-2))
    
    print("conv1d_weight.grad rtol=1e-2: ", how_close_rtol(conv1d_weight.grad, layer.conv1d.weight.grad, rtol=1e-2))
    print("conv1d_bias.grad rtol=1e-2: ", how_close_rtol(conv1d_bias.grad, layer.conv1d.bias.grad, rtol=1e-2))
    print("D.grad rtol=1e-2: ", how_close_rtol(D.grad, layer.D.grad, rtol=1e-2))
    print("dt_bias.grad rtol=1e-2: ", how_close_rtol(dt_bias.grad, layer.dt_bias.grad, rtol=1e-2))
    print("norm_weight.grad rtol=1e-2: ", how_close_rtol(norm_weight.grad, layer.norm.weight.grad, rtol=1e-2))
    print("logA.grad rtol=1e-2: ", how_close_rtol(logA.grad, layer.A_log.grad, rtol=1e-2))