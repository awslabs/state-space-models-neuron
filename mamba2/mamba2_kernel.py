import os

import torch
import torch_xla.core.xla_model as xm

import neuronxcc.nki.language as nl
from torch_neuronx import nki_jit
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import numpy as np

from .utils import chunk


def compute_a_factors(logA, ones_triu, chunk_size, d_state, d_head=None):
    dtype = logA.dtype
    i_p = nl.arange(chunk_size)[:, None]
    if d_head is None:
        d_head = chunk_size
    i_p_head = nl.arange(d_head)[:, None]
    i_f = nl.arange(chunk_size)[None, :]

    # we reuse this bcast later in the code, ensure it is large enough for all uses
    bcast_size = max(chunk_size, d_state)

    logA_bcast = logA.broadcast_to((chunk_size, bcast_size))
    l_bcast = nl.matmul(ones_triu, logA_bcast, transpose_x=True)
    l = l_bcast[:chunk_size, 0]
    l_t = nl.transpose(l_bcast, dtype=dtype)

    # === compute the _transpose_ of the 128x128 lower triangular matrix L ===
    partial_sums = l_t[:chunk_size, :chunk_size] - l
    L_full_t = nl.exp(partial_sums)
    L_t = nisa.affine_select(i_f >= i_p, L_full_t, 0)

    a_right = L_t[i_p, chunk_size - 1]
    a_left = nl.exp(l_t[i_p_head, i_f])

    if d_head != chunk_size:
        a_center_t = a_left[:, chunk_size - 1].broadcast_to((d_head, bcast_size))
        a_center = nl.transpose(a_center_t, dtype=dtype)
    else:
        a_center = a_left[:, chunk_size - 1]

    # a_left = nl.copy(a_left)
    a_center = nl.copy(a_center)

    return L_t, a_left, a_center, a_right

def compute_chunk_output(BC_t, L_t, X, C_t, a_left, S,
                         transpose_gate=False,
                         transpose_broadcast=False):
    # Diagonal term computation
    M_diag_t = L_t * BC_t

    Y_diag = nl.matmul(M_diag_t, X, transpose_x=not transpose_gate)

    # Compute the off-diagonal contribution using the state
    barC_t = C_t * a_left

    Y_off = nl.matmul(barC_t, S, transpose_x=not transpose_broadcast)
    return Y_diag + Y_off



# @nki.jit
def mamba_kernel_(dt_tensor, logA_tensor, xBC_tensor, D_tensor, n_groups, d_state, d_inner, d_head, out_Y_tensor=None):
    """
    dt_tensor: (batch_size, seq_len, n_heads)
    logA_tensor: (n_heads)
    xBC_tensor: (batch_size, seq_len, d_inner + 2 * n_groups * d_state)
    D_tensor: (n_heads)
    """
    # For now, we force the kernel to run in fp32 since it is much more precise and the speedup in bf16 is only 15%
    # todo: figure out the least number of tensors to keep in fp32 while maintaining high precision
    # dtype = X_tensor.dtype
    dtype = nl.float32
    block_size = 128
    # todo: why is the kernel slower with using this layout instead of [batch_size, n_heads, seq_len, d_head]?
    batch_size, seq_len, xBC_size = xBC_tensor.shape
    assert xBC_size == d_inner + 2 * n_groups * d_state
    assert seq_len % block_size == 0
    assert d_inner % d_head == 0

    n_heads = d_inner // d_head
    n_chunks = seq_len // block_size
    n_heads_per_group = n_heads // n_groups


    i_p = nl.arange(block_size)[:, None]
    i_f = nl.arange(block_size)[None, :]
    i_f_state = nl.arange(d_state)[None, :]
    i_f_head = nl.arange(d_head)[None, :]

    if out_Y_tensor is None:
        out_Y_tensor = nl.ndarray((batch_size, seq_len, d_inner), xBC_tensor.dtype, buffer=nl.private_hbm)

    # upper triangular matrix of ones
    ones_triu = nisa.affine_select(i_p <= i_f, nl.ones((block_size, block_size), dtype=dtype), 0)

    for batch_id in range(batch_size):
        for group_id in nl.affine_range(n_groups):
            # === Preload/compute logA, B, C_t and B @ C_t ====
            # (they are shared between multiple heads in the same group)
            B_cache = nl.ndarray((block_size, n_chunks, d_state), dtype=dtype)
            # todo: storing in this format may be a bad idea if d_state != 128?
            C_t_cache = nl.ndarray((d_state, n_chunks, block_size), dtype=dtype)
            BC_t_cache = nl.ndarray((block_size, n_chunks, block_size), dtype=dtype)
            B_offset = d_inner
            C_offset = d_inner + n_groups * d_state
            for chunk_id in nl.affine_range(n_chunks):
                i_p_in = i_p + chunk_id * block_size
                # todo: change this to load the current group when n_groups > 1
                B = nl.load(xBC_tensor[batch_id, i_p_in, B_offset + d_state * group_id + i_f_state], dtype=dtype)
                C = nl.load(xBC_tensor[batch_id, i_p_in, C_offset + d_state * group_id + i_f_state], dtype=dtype)
                # note: caching the transpose/matmul here doesn't seem to save a lot of time because the rest of the code is
                #       bottle-necked by the VectorE/ScalarE anyway
                C_t = nisa.nc_transpose(C)
                B_cache[:, chunk_id, :] = B
                C_t_cache[:, chunk_id, :] = C_t
                BC_t_cache[:, chunk_id, :] = nl.copy(nl.matmul(B, C_t), dtype=dtype)

            logA_cache = nl.load(logA_tensor.reshape((1, n_heads)), dtype=dtype).broadcast_to((block_size, n_heads))
            if D_tensor is not None:
                D = nl.load(D_tensor.reshape((1, n_heads)), dtype=dtype).broadcast_to((block_size, n_heads))
            else:
                D = None


            # == Actual code ===
            for head_id_in_group in nl.affine_range(n_heads_per_group):  # the n_heads are completely independent
                # get the global head_id given current group and current head in group
                head_id = group_id * n_heads_per_group + head_id_in_group
                # We iterate over the diagonal blocks and compute each Y_diag
                # At the same time, we update our running sum S and use it to compute Y_off.
                # We store Y = Y_diag + Y_off, and we move to the next block
                S = nl.zeros((d_state, d_head), dtype=dtype)
                for chunk_id in nl.sequential_range(n_chunks):
                    i_p_in = i_p + chunk_id * block_size

                    # broadcast dt and logA together
                    dt = nl.load(dt_tensor[batch_id, i_p_in, head_id], dtype=dtype)
                    logA = logA_cache[:, head_id] * dt

                    # load from cache the relevant blocks
                    B = B_cache[:, chunk_id, :]
                    C_t = C_t_cache[:, chunk_id, :]
                    BC_t = BC_t_cache[:, chunk_id, :]

                    # broadcast X and dt
                    X0 = nl.load(xBC_tensor[batch_id, i_p_in, d_head * head_id + i_f_head], dtype=dtype)
                    X = dt * X0

                    # Compute all logA related factors for this chunk
                    L_t, a_left, a_center, a_right = compute_a_factors(logA, ones_triu, block_size, d_state)

                    Y = compute_chunk_output(BC_t, L_t, X, C_t, a_left, S)

                    # Update running sum S (will be used in the next iteration)
                    barB = B * a_right
                    barBX = nl.matmul(barB, X, transpose_x=True)
                    # fixme: a_center removed because it leads to a compiler error
                    S[...] = a_center * S + barBX
                    # S[...] = S + barBX

                    if D is not None:
                        Y = Y + D[:, head_id] * X0

                    nl.store(out_Y_tensor[batch_id, i_p_in, d_head * head_id + i_f_head], Y)
    return out_Y_tensor


# for unit testing, remove creation of output tensor within kernel to avoid complaints when testing on CPU
def mamba_kernel_test_(dt_tensor, logA_tensor, xBC_tensor, D_tensor, n_groups, d_state, d_inner, d_head, out_Y_tensor):
    """
    dt_tensor: (batch_size, seq_len, n_heads)
    logA_tensor: (n_heads)
    xBC_tensor: (batch_size, seq_len, d_inner + 2 * n_groups * d_state)
    D_tensor: (n_heads)
    """
    # For now, we force the kernel to run in fp32 since it is much more precise and the speedup in bf16 is only 15%
    # todo: figure out the least number of tensors to keep in fp32 while maintaining high precision
    # dtype = X_tensor.dtype
    dtype = nl.float32
    block_size = 128
    # todo: why is the kernel slower with using this layout instead of [batch_size, n_heads, seq_len, d_head]?
    batch_size, seq_len, xBC_size = xBC_tensor.shape
    assert xBC_size == d_inner + 2 * n_groups * d_state
    assert seq_len % block_size == 0
    assert d_inner % d_head == 0

    n_heads = d_inner // d_head
    n_chunks = seq_len // block_size
    n_heads_per_group = n_heads // n_groups

    i_p = nl.arange(block_size)[:, None]
    i_f = nl.arange(block_size)[None, :]
    i_f_state = nl.arange(d_state)[None, :]
    i_f_head = nl.arange(d_head)[None, :]

    # out_Y_tensor = nl.ndarray((batch_size, seq_len, d_inner), xBC_tensor.dtype, buffer=nl.private_hbm)

    # upper triangular matrix of ones
    ones_triu = nisa.affine_select(i_p <= i_f, nl.ones((block_size, block_size), dtype=dtype), 0)

    for batch_id in range(batch_size):
        for group_id in nl.affine_range(n_groups):
            # === Preload/compute logA, B, C_t and B @ C_t ====
            # (they are shared between multiple heads in the same group)
            B_cache = nl.ndarray((block_size, n_chunks, d_state), dtype=dtype)
            # todo: storing in this format may be a bad idea if d_state != 128?
            C_t_cache = nl.ndarray((d_state, n_chunks, block_size), dtype=dtype)
            BC_t_cache = nl.ndarray((block_size, n_chunks, block_size), dtype=dtype)
            B_offset = d_inner
            C_offset = d_inner + n_groups * d_state
            for chunk_id in nl.affine_range(n_chunks):
                i_p_in = i_p + chunk_id * block_size
                # todo: change this to load the current group when n_groups > 1
                B = nl.load(xBC_tensor[batch_id, i_p_in, B_offset + d_state * group_id + i_f_state], dtype=dtype)
                C = nl.load(xBC_tensor[batch_id, i_p_in, C_offset + d_state * group_id + i_f_state], dtype=dtype)
                # note: caching the transpose/matmul here doesn't seem to save a lot of time because the rest of the code is
                #       bottle-necked by the VectorE/ScalarE anyway
                C_t = nisa.nc_transpose(C)
                B_cache[:, chunk_id, :] = B
                C_t_cache[:, chunk_id, :] = C_t
                BC_t_cache[:, chunk_id, :] = nl.copy(nl.matmul(B, C_t), dtype=dtype)

            logA_cache = nl.load(logA_tensor.reshape((1, n_heads)), dtype=dtype).broadcast_to((block_size, n_heads))
            if D_tensor is not None:
                D = nl.load(D_tensor.reshape((1, n_heads)), dtype=dtype).broadcast_to((block_size, n_heads))
            else:
                D = None

            # == Actual code ===
            for head_id_in_group in nl.affine_range(n_heads_per_group):  # the n_heads are completely independent
                # get the global head_id given current group and current head in group
                head_id = group_id * n_heads_per_group + head_id_in_group
                # We iterate over the diagonal blocks and compute each Y_diag
                # At the same time, we update our running sum S and use it to compute Y_off.
                # We store Y = Y_diag + Y_off, and we move to the next block
                S = nl.zeros((d_state, d_head), dtype=dtype)
                for chunk_id in nl.sequential_range(n_chunks):
                    i_p_in = i_p + chunk_id * block_size

                    # broadcast dt and logA together
                    dt = nl.load(dt_tensor[batch_id, i_p_in, head_id], dtype=dtype)
                    logA = logA_cache[:, head_id] * dt

                    # load from cache the relevant blocks
                    B = B_cache[:, chunk_id, :]
                    C_t = C_t_cache[:, chunk_id, :]
                    BC_t = BC_t_cache[:, chunk_id, :]

                    # broadcast X and dt
                    X0 = nl.load(xBC_tensor[batch_id, i_p_in, d_head * head_id + i_f_head], dtype=dtype)
                    X = dt * X0

                    # Compute all logA related factors for this chunk
                    L_t, a_left, a_center, a_right = compute_a_factors(logA, ones_triu, block_size, d_state)

                    Y = compute_chunk_output(BC_t, L_t, X, C_t, a_left, S)

                    # Update running sum S (will be used in the next iteration)
                    barB = B * a_right
                    barBX = nl.matmul(barB, X, transpose_x=True)
                    # fixme: a_center removed because it leads to a compiler error
                    S[...] = a_center * S + barBX
                    # S[...] = S + barBX

                    if D is not None:
                        Y = Y + D[:, head_id] * X0

                    nl.store(out_Y_tensor[batch_id, i_p_in, d_head * head_id + i_f_head], Y)
        # return out_Y_tensor

# @nki_jit
def mamba_kernel_bwd_(dt_tensor, logA_tensor, X_tensor, B_tensor, C_tensor,
                      d_out_tensor,
                      ddt_tensor,
                      dlogA_tensor,
                      dX_tensor,
                      dB_tensor,
                      dC_tensor,
                      D_tensor,
                      dD_tensor,
                      ):
    """
    dt_tensor: (batch_size, seq_len, n_heads)
    logA_tensor: (n_heads)
    X_tensor: (batch_size, seq_len, n_heads, d_head)
    B_tensor: (batch_size, seq_len, n_groups, d_state)
    C_tensor: (batch_size, seq_len, n_groups, d_state)
    D_tensor: (n_heads)
    d_out_tensor: (batch_size, seq_len, n_heads, d_head)
    All other derivative tensors (d_*) have the same shape as their corresponding input counterparts.
    """

    # Note: since saving the intermediate results of the forward pass would use too much memory, this kernel also
    # recomputes the forward pass while computing the gradients.

    # Since this kernel requires high-precision, we run all internal computations in fp32.
    # Note: the speedup by using bf16 everywhere would be ~15%
    dtype = nl.float32
    block_size = 128  # we will split seq_len in chunks of size `block_size`
    batch_size, seq_len, n_heads, d_head = X_tensor.shape
    _, _, n_groups, d_state = B_tensor.shape
    assert seq_len % block_size == 0
    n_chunks = seq_len // block_size
    n_heads_per_group = n_heads // n_groups

    assert d_state == 128
    assert d_head <= 128
    assert block_size <= 128

    # fixme: implement batch dimension
    # batch_id = nl.program_id(0)
    batch_id = 0

    i_p = nl.arange(block_size)[:, None]
    i_f = nl.arange(block_size)[None, :]
    i_f_state = nl.arange(d_state)[None, :]
    i_f_head = nl.arange(d_head)[None, :]

    # upper triangular matrix of ones
    ones_triu = nisa.affine_select(i_p <= i_f, nl.ones((block_size, block_size), dtype=dtype), 0)
    ones_tril = nl.copy(nl.transpose(ones_triu), dtype=dtype)
    ones_sum_right = nl.ones([d_state, 1], dtype=dtype)
    ones_sum_left = nl.ones([1, d_state], dtype=dtype)
    ones_sum_right_head = nl.ones([d_head, 1], dtype=dtype)

    for group_id in nl.affine_range(n_groups):  # iterate in parallel over all channel groups (they are independent)
        # Preload/compute logA, B, C_t and B @ C_t (which are shared between multiple heads in the same group)
        B_cache = nl.ndarray((block_size, n_chunks, d_state), dtype=dtype)
        C_t_cache = nl.ndarray((d_state, n_chunks, block_size), dtype=dtype)
        BC_t_cache = nl.ndarray((block_size, n_chunks, block_size), dtype=dtype)
        for chunk_id in nl.affine_range(n_chunks):
            # i_p_in = i_p + chunk_id * block_size
            seq_slice = chunk(chunk_id, block_size)
            B = nl.load(B_tensor[batch_id, seq_slice, group_id, :], dtype=dtype)
            C = nl.load(C_tensor[batch_id, seq_slice, group_id, :], dtype=dtype)
            C_t = nisa.nc_transpose(C)
            B_cache[:, chunk_id, :] = B
            C_t_cache[:, chunk_id, :] = C_t
            BC_t_cache[:, chunk_id, :] = nl.copy(nl.matmul(B, C_t), dtype=dtype)

        logA_cache = nl.load(logA_tensor.reshape((1, n_heads)), dtype=dtype).broadcast_to((block_size, n_heads))
        if D_tensor is not None:
            D = nl.load(D_tensor.reshape((1, n_heads)), dtype=dtype).broadcast_to((block_size, n_heads))
        else:
            D = None

        dC_accumulation = nl.zeros((block_size, n_chunks, d_state), dtype=dtype)
        dB_accumulation = nl.zeros((block_size, n_chunks, d_state), dtype=dtype)
        dA_final = nl.zeros((1, n_heads), dtype=dtype)
        if D is not None:
            dD_final = nl.zeros((1, n_heads), dtype=dtype)

        for head_id_in_group in nl.affine_range(n_heads_per_group):  # the n_heads are completely independent
            # get the global head_id given current group and current head in group
            head_id = group_id * n_heads_per_group + head_id_in_group
            dA_accumulation = nl.zeros((block_size, n_chunks, d_state), dtype=dtype)
            S = nl.zeros((d_state, d_head), dtype=dtype)
            for chunk_id in nl.sequential_range(n_chunks):
                # <forward pass>
                i_p_in = i_p + chunk_id * block_size
                # broadcast dt and logA together
                dt = nl.load(dt_tensor[batch_id, i_p_in, head_id])
                logA = logA_cache[:, head_id] * dt
                # Compute all logA related factors for this chunk
                L_t, a_left, a_center, a_right = compute_a_factors(logA, ones_triu, block_size, d_state, d_head=d_head)
                # load from cache the relevant blocks
                B = B_cache[:, chunk_id, :]
                C = nl.load(C_tensor[batch_id, i_p_in, group_id, i_f_state])
                # broadcast X and dt
                X0 = nl.load(X_tensor[batch_id, i_p_in, head_id, i_f_head])
                X = dt * X0
                # </forward pass>

                # compute dC gradient
                dO = nl.load(d_out_tensor[batch_id, i_p_in, head_id, i_f_head])
                dO_t = nisa.nc_transpose(dO)
                UdO_t = nl.matmul(X, dO_t)  # (B, L, nheads, hdim)
                S_t = nisa.nc_transpose(S)
                # fixme
                # dC = compute_chunk_output(UdO_t, L_t, B, dO_t, a_left, S_t)
                # <inlined_function>
                M_diag_t = L_t * UdO_t
                Y_diag = nl.matmul(M_diag_t, B, transpose_x=True)
                barC_t = dO_t * a_left
                Y_off = nl.matmul(barC_t, S_t, transpose_x=True)
                dC = Y_diag + Y_off
                # </inlined_function>

                # <forward pass>
                # Update the state: running sum S (will be used in the next iteration)
                barB = B * a_right
                barBX = nl.matmul(barB, X, transpose_x=True)
                S[...] = a_center * S + barBX
                # </forward pass>
                dC_accumulation[:, chunk_id, :] += dC
                dA_accumulation[:, chunk_id, :] = dA_accumulation[:, chunk_id, :] + C * dC

            dS = nl.zeros((d_state, d_head), dtype=dtype)
            cumsum_dA = nl.zeros((1, d_state), dtype=dtype)
            for chunk_id in nl.sequential_range(n_chunks):
                chunk_id = n_chunks - 1 - chunk_id  # To reverse time
                i_p_in = i_p + chunk_id * block_size

                # === Recompute forward pass ===
                # broadcast dt and logA together
                dt = nl.load(dt_tensor[batch_id, i_p_in, head_id])
                logA = logA_cache[:, head_id] * dt
                # Compute all logA related factors for this chunk
                L_t, a_left, a_center, a_right = compute_a_factors(logA, ones_triu, block_size, d_state, d_head=d_head)
                # load from cache the relevant blocks
                B = B_cache[:, chunk_id, :]
                C = nl.load(C_tensor[batch_id, i_p_in, group_id, i_f_state])
                C_t = nisa.nc_transpose(C)
                BC_t = BC_t_cache[:, chunk_id, :]
                # broadcast X and dt
                X0 = nl.load(X_tensor[batch_id, i_p_in, head_id, i_f_head])
                X = dt * X0

                # === Compute dX gradient ===
                dO = nl.load(d_out_tensor[batch_id, i_p_in, head_id, i_f_head])
                dU = compute_chunk_output(BC_t, L_t, dO, B, a_right, dS, transpose_gate=True, transpose_broadcast=True)

                # === Compute dB gradient ===
                X_t = nisa.nc_transpose(X)
                dO_Xt = nl.matmul(dO, X_t)
                L_t_ = nisa.nc_transpose(L_t)
                dS_t = nisa.nc_transpose(dS)

                C = nl.load(C_tensor[batch_id, i_p_in, group_id, i_f_state], dtype=dtype)

                # === Compute dB gradient ===
                # dB = nl.zeros_like(B)
                # dB = compute_chunk_output(dO_Xt, L_t_, C, X, a_right, dS_t, transpose_broadcast=True)
                # <inlined_function>
                M_diag_t = L_t_ * dO_Xt
                Y_diag = nl.matmul(M_diag_t, C, transpose_x=not False)
                barC_t = X * a_right
                Y_off = nl.matmul(barC_t, dS_t, transpose_x=not True)
                dB = Y_diag + Y_off
                # </inlined_function>


                # === Update reverse time state dState ===
                # Update the state: running sum dS (will be used in the next iteration)
                barC = C_t * a_left[:1, :].broadcast_to((block_size, block_size))

                barC_tX = nl.matmul(barC, dO, transpose_x=False)
                dS[...] = a_center * dS + barC_tX

                dB_accumulation[:, chunk_id, :] += dB
                dA_accumulation[:, chunk_id, :] -= B * dB

                # === Reverse cumulative sum for dA ===
                cumsum_chunk = nl.matmul(ones_tril, dA_accumulation[:, chunk_id, :], transpose_x=True)
                cumsum_chunk[...] = cumsum_chunk + nl.copy(cumsum_dA, dtype=dtype).broadcast_to((block_size, d_state))
                cumsum_dA[0, i_f_state] = cumsum_chunk[0, i_f_state]

                ddt = nl.matmul(cumsum_chunk * logA_cache[:, head_id], ones_sum_right) + nl.matmul(dU * X0, ones_sum_right_head)

                dA_chunk = nl.matmul(cumsum_chunk * dt, ones_sum_right)
                dA_final[:, head_id] += nl.matmul(ones_sum_left, dA_chunk)

                dX = dU * dt

                if D is not None:
                    dD_chunk = nl.matmul(dO * X0, ones_sum_right_head)
                    dD_final[:, head_id] += nl.copy(nl.matmul(ones_sum_left, dD_chunk), dtype=dtype)
                    dX[...] = dX + dO * D[:, head_id]

                nl.store(dX_tensor[batch_id, i_p_in, head_id, i_f_head], dX)
                nl.store(ddt_tensor[batch_id, i_p_in, head_id], ddt)

            nl.store(dlogA_tensor[batch_id, head_id], dA_final[0, head_id])
            if D is not None:
                nl.store(dD_tensor[batch_id, head_id], dD_final[0, head_id])

        for chunk_id in nl.sequential_range(n_chunks):
            i_p_in = i_p + chunk_id * block_size
            nl.store(dC_tensor[batch_id, i_p_in, group_id, i_f_state], dC_accumulation[:, chunk_id, :])
            nl.store(dB_tensor[batch_id, i_p_in, group_id, i_f_state], dB_accumulation[:, chunk_id, :])


class Mamba2Kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dt, A, X, B, C, D):
        batch_size, seq_len,  n_heads, d_head = X.shape
        ctx.save_for_backward(dt, A, X, B, C, D)
        out_Y = torch.empty_like(X)
        mamba_kernel_[batch_size](dt, A, X, B, C, out_Y, D)
        return out_Y

    @staticmethod
    def backward(ctx, d_output):
        dt, A, X, B, C, D = ctx.saved_tensors
        batch_size, seq_len, n_heads, d_head = X.shape

        # out_Y = torch.empty_like(X)
        ddt = torch.empty_like(dt)
        dA = torch.empty_like(A.unsqueeze(0).repeat(batch_size, 1))
        dX = torch.empty_like(X)
        dB = torch.empty_like(B)
        dC = torch.empty_like(C)
        dD = torch.empty_like(D.unsqueeze(0).repeat(batch_size, 1))

        ### For debugging will keep outputting the correct fwd kernel output
        mamba_kernel_bwd_[batch_size](dt, A, X, B, C, d_output,
                                      # out_Y,
                                      ddt, dA, dX, dB, dC, D, dD)
        dA, dD = dA.sum(0), dD.sum(0)
        return ddt, dA, dX, dB, dC, dD


def mamba2_kernel(dt, A, X, B, C, D):
    return Mamba2Kernel.apply(dt, A, X, B, C, D)


if __name__ == "__main__":
    import argparse

    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["XLA_HLO_DEBUG"] = "1"
    os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
    # os.environ["NEURON_CC_FLAGS"] = " --disable-internal-io-dge --auto-cast=none "
    os.environ["NEURON_CC_FLAGS"] = " --disable-internal-io-dge "
    os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"

    parser = argparse.ArgumentParser(description="Mamba2 Kernel Configuration")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"], help="Data type for computation")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--block_len", type=int, default=128, help="Block length")
    parser.add_argument("--d_head", type=int, default=128, help="Head dimension")
    parser.add_argument("--d_state", type=int, default=128, help="State dimension")
    parser.add_argument("--dim", type=int, default=2048, help="Model dimension")
    parser.add_argument("--n_groups", type=int, default=1, help="Number of groups")
    parser.add_argument("--seed", type=int, default=0, help="Random seed to use")
    parser.add_argument("--compute_bwd", "-b", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    mixed_precision = (dtype == torch.bfloat16)
    print(f'Is mixed precision: {mixed_precision}', dtype)

    batch, seq_len, d_head, d_state, dim, n_groups = args.batch, args.seq_len, args.d_head, args.d_state, args.dim, args.n_groups
    block_len = args.block_len
    n_heads = dim // d_head
    # assume n_groups is 1 for now (if more then we parallelize cross devices)

    torch.manual_seed(args.seed)

    # dtype = torch.float32
    # dt = torch.rand(batch, seq_len, n_heads, dtype=dtype)
    dt = torch.exp(torch.empty(batch, seq_len, n_heads, dtype=dtype).uniform_(np.log(0.001), np.log(0.1)))
    A = -torch.empty(n_heads, dtype=dtype).uniform_(1, 16)
    X = torch.randn(batch, seq_len, n_heads, d_head, dtype=dtype)
    B = torch.randn(batch, seq_len, n_groups, d_state, dtype=dtype)
    C = torch.randn(batch, seq_len, n_groups, d_state, dtype=dtype)
    D = torch.randn(n_heads, dtype=dtype)
    # dtype = getattr(torch, args.dtype)

    def prepare_data(tensor, device, dtype, grad=False):
        data = tensor.data.to(device=device, dtype=dtype)
        return torch.nn.Parameter(data, requires_grad=grad)

    [dt_ref, A_ref, X_ref, B_ref, C_ref, D_ref] = [prepare_data(x, 'cpu', torch.float32, True) for x in (dt, A, X, B, C, D)]

    from mamba2_kernel_reference import ssd_minimal_discrete_fused

    # for the reference implementation keep it in float32
    ref_out = ssd_minimal_discrete_fused(dt_ref, X_ref, A_ref, B_ref, C_ref, D_ref, block_len=block_len)
    assert ref_out.dtype == torch.float32

    d_output_ref = torch.randn_like(ref_out)
    ref_out.backward(d_output_ref)

    device = xm.xla_device()

    [dt_ours, A_ours, X_ours, B_ours, C_ours, D_ours] = [prepare_data(x, device, dtype, True) for x in (dt, A, X, B, C, D)]
    d_output = prepare_data(d_output_ref, device, dtype)

    xm.mark_step()

    if not args.compute_bwd:
        with torch.no_grad():
            out = mamba2_kernel(dt_ours, A_ours, X_ours, B_ours, C_ours, D_ours)
    else:
        out = mamba2_kernel(dt_ours, A_ours, X_ours, B_ours, C_ours, D_ours)
        out.backward(d_output)

    xm.mark_step()

    if args.profile:
        out.cpu()
        exit()

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
        print(f"index {np.unravel_index(idx, tensor.shape)} out={tensor.flatten()[idx].item():.5f} ref={ref_tensor.flatten()[idx].item():.5f}")

    print_stats(out, ref_out, 'Reference Output')
    if args.compute_bwd:
        print_stats(C_ours.grad, C_ref.grad, 'Grad C')
        print_stats(X_ours.grad, X_ref.grad, 'Grad X')
        print_stats(B_ours.grad, B_ref.grad, 'Grad B')
        print_stats(A_ours.grad, A_ref.grad, 'Grad A')
        print_stats(dt_ours.grad, dt_ref.grad, 'Grad dt')
        print_stats(D_ours.grad, D_ref.grad, 'Grad dD')