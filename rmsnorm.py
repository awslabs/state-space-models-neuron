import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import torch, math
import numpy as np
from einops import rearrange
from torch import nn


def MambaRMSNormGated_forward(
    hidden_states, gate, weight, rmsnorm_within_groups, n_groups, eps
):
    hidden_states = hidden_states.to(torch.float32)
    print(hidden_states.shape)

    if rmsnorm_within_groups:
        hidden_states = rearrange(hidden_states, "... (g d) -> ... g d", g=n_groups)
        if gate is not None:
            gate = rearrange(gate, "... (g d) -> ... g d", g=n_groups)
    print(hidden_states.shape)

    if gate is not None:
        hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))

    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)

    if rmsnorm_within_groups:
        hidden_states = rearrange(hidden_states, "... g d -> ... (g d)", g=n_groups)
    res = weight * hidden_states
    return res


@nki.jit
def nki_rmsnorm_kernel(a_tensor, g_tensor):
    # Calculate out_tensor = a_tensor/RMS(a_tensor) * g_tensor
    # Where RMS(a_tensor) = sqrt((1/N) * sum(a_tensor * a_tensor))
    # and N = a_tensor.shape[1]
    # Reduction (mean) is performed in the free (2nd) dimension
    out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    # Make sure shapes match
    assert a_tensor.shape[1] == g_tensor.shape[0], f"{a_tensor.shape} {g_tensor.shape}"

    # Generate tensor indices to index input tensor
    pmax = nl.tile_size.pmax
    ix = nl.arange(pmax)[:, None]
    iw = nl.arange(1)[:, None]
    iy = nl.arange(a_tensor.shape[1])[None, :]

    num_rows = a_tensor.shape[0]

    # Load RMSNorm weight once, reused by rows/tiles of a_tensor
    g_tile = nl.load(g_tensor.reshape((1, g_tensor.shape[0]))[iw, iy])

    # Process pmax (128) rows at a time due to 128-partition tile size limitation
    # Since we're not reducing across the first dimension
    # Tiles can be processed independently
    for i in nl.affine_range(math.ceil(num_rows / pmax)):

        # Load input data from external memory to on-chip memory
        a_tile = nl.load(a_tensor[i * pmax + ix, iy], mask=(i * pmax + ix < num_rows))
        print(f"a_tile shape = {a_tile.shape}")

        # Compute element-wise square of a_tensor
        in_square = nl.square(a_tile)

        # Calculate sum of squared elements, along last dimension
        square_sum = nl.sum(in_square, axis=[1])

        # Scale and get a reciprocal
        mean = square_sum / a_tensor.shape[1]

        # Take square root of mean and then reciprocal with
        # rsqrt API (one ISA instruction)
        rms_reciprocal = nl.rsqrt(mean)

        # Scale the input tensor
        out_tile = nl.multiply(a_tile, rms_reciprocal)

        # Broadcast weight along first axis to match tensor shape
        # num_rows_active = min(num_rows - i * 128, 128)
        g_bcast = g_tile.broadcast_to((pmax, g_tensor.shape[0]))

        # Multiply with the RMSNorm weight
        out_tile[...] = nl.multiply(out_tile, g_bcast, mask=(i * pmax + ix < num_rows))

        # store the addition results back to external memory (out_tensor)
        nl.store(
            out_tensor[i * pmax + ix, iy],
            value=out_tile,
            mask=(i * pmax + ix < num_rows),
        )

    return out_tensor


if __name__ == "__main__":
    batch = 1
    seqlen = 1024
    d_model = 768
    tp_size = 8
    hidden_states = torch.randn(batch, seqlen, d_model // tp_size)
    weight = torch.ones(hidden_states.shape[-1])
    mamba_out = MambaRMSNormGated_forward(hidden_states, None, weight, True, 1, 1e-5)
    mamba_out = mamba_out.numpy()[0]
    print(mamba_out, mamba_out.shape)

    hidden_states = hidden_states.numpy()[0]
    weight = weight.numpy()
    output_nki = nki_rmsnorm_kernel(hidden_states, weight)
    print(f"output_nki={output_nki} {output_nki.shape}")

    allclose = np.allclose(mamba_out, output_nki, atol=1e-4, rtol=1e-2)
    if allclose:
        print("NKI and Mamba match")

    assert allclose
