import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronx_distributed.parallel_layers.parallel_state as ps
from torch_neuronx import nki_jit
import torch_xla.core.xla_model as xm

from .utils import chunk


def apply_activation(x, bias, activation: str):
    if activation is None:
        return x + bias if bias is not None else x
    elif activation == 'relu':
        return nisa.activation(nl.relu, x, bias=bias)
    elif activation == 'd_relu':
        assert bias is None
        return x >= 0
    elif activation == 'silu':
        z = x + bias if bias is not None else x
        return nl.silu(z)
    elif activation == 'd_silu':
        z = x + bias if bias is not None else x
        return nl.silu_dx(z)
        # return nl.sigmoid(z) * (1 + z * (1 - nl.sigmoid(z)))
    elif activation == 'd_identity':
        return 1
    else:
        raise ValueError(f'Invalid activation {activation}')


def _conv_tile(data_tile, weight_tile, b_tile, dtype, activation=None):
    """Computes and returns the convolution of a data tile given weights and bias.

    Isolated from the rest of the code since we use it both for forward and backward.
    """
    p_size, n = data_tile.shape
    kernel_size = weight_tile.shape[1]

    conv = nl.ndarray(shape=(p_size, n), dtype=dtype)

    chunk_size = n // kernel_size

    i_p = nl.arange(p_size)[:, None]

    for j in nl.affine_range(kernel_size):
        i_f_plus_j = j + nl.arange(chunk_size)[None, :] * kernel_size
        res = nki.isa.tensor_scalar(data_tile[i_p, i_f_plus_j], op0=np.multiply,
                                    operand0=weight_tile[i_p, 0], dtype=dtype)
        for i in nl.static_range(1, kernel_size):
            res = res + nki.isa.tensor_scalar(data_tile[i_p, i_f_plus_j + i], op0=np.multiply,
                                              operand0=weight_tile[i_p, i],
                                              dtype=dtype)
        conv[i_p, i_f_plus_j] = apply_activation(res, bias=b_tile, activation=activation)
    return conv


@nki.jit
def conv1d_with_transpose(input_tensor, w_tensor, bias_tensor, output_tensor=None, activation=None):
    """NKI kernel to compute grouped 1d causal convolution for sequences of all lengths by processing them in sub-sequences

    equivalent to:

        D, L = x.shape

        conv = nn.Conv1d(
            in_channels=D,
            out_channels=D,
            bias=True,
            kernel_size=kernel_size,
            groups=D,
            padding=kernel_size - 1,
        )
        y = conv(x.transpose(1, 2))[:, :L].transpose(1, 2)

    Args:
      input_tensor: input tensor of shape [L, D]
      w_tensor: conv weights of shape [D, kernel_size]
      bias_tensor: conv bias of shape [D]
      output: output tensor of shape [L, D]
    """

    batch_size, seq_len, channels = input_tensor.shape
    ch, _, ks = w_tensor.shape
    dtype = input_tensor.dtype

    # fixme: make the code work for any size
    assert channels % 128 == 0 and ch == channels
    assert seq_len % ks == 0 and seq_len > ks  # check seq_len is a multiple of kernel size
    assert seq_len % 128 == 0
    assert ks == 4  # fixme: don't think this constrain is needed

    pad = ks - 1

    if output_tensor is None:
        output_tensor = nl.ndarray((batch_size, seq_len, channels), dtype=dtype, buffer=nl.private_hbm)

    # Iterate over channel dimension then over batch dimension (so we load the weights only once for all samples)
    for channel_tile_id in nl.affine_range(channels // 128):
        channel_slice = chunk(channel_tile_id, 128)
        # weights and biases for current tile
        weight = nl.load(w_tensor.reshape((ch, ks))[channel_slice, :])
        bias = nl.load(bias_tensor[channel_slice])

        # todo: run the convolution while loading the tiles, so the transpose can happen in parallel
        for batch_id in nl.affine_range(batch_size):
            padded_input = nl.zeros(shape=(128, pad + 128), dtype=dtype)
            for seq_id in nl.sequential_range(seq_len // 128):
                seq_slice = chunk(seq_id, 128)
                input_tile = nl.load(input_tensor[batch_id, seq_slice, channel_slice])  # [seq=128, ch=128]
                padded_input[:, pad:] = nl.copy(nl.transpose(input_tile))  # [ch=128, seq=128+pad]
                # run the convolution
                conv = _conv_tile(padded_input, weight, bias, dtype, activation=activation)
                output_tile = nl.transpose(conv[:, :128])  # [seq=128, ch=128]
                nl.store(output_tensor[batch_id, seq_slice, channel_slice], output_tile)
                # move final elements at beginning to serve as padding for next sequence
                padded_input[:, :pad] = padded_input[:, 128:]
    return output_tensor


@nki.jit
def conv1d_with_transpose_bwd(input_tensor, w_tensor, bias_tensor, d_output_tensor, activation=None):
    batch_size, seq_len, channels = input_tensor.shape
    ch, _, ks = w_tensor.shape
    dtype = input_tensor.dtype

    # fixme: make the code work for any size
    assert activation == 'silu'
    assert channels % 128 == 0 and ch == channels
    assert seq_len % ks == 0 and seq_len > ks  # check seq_len is a multiple of kernel size
    assert seq_len % 128 == 0
    assert ks == 4  # fixme: don't think this constrain is needed

    pad = ks - 1

    d_input_tensor = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.private_hbm)
    d_w_tensor = nl.ndarray(w_tensor.shape, dtype=w_tensor.dtype, buffer=nl.private_hbm)
    d_bias_tensor = nl.ndarray(bias_tensor.shape, dtype=bias_tensor.dtype, buffer=nl.private_hbm)

    # Iterate over channel dimension then over batch dimension (so we load the weights only once for all samples)
    for channel_tile_id in nl.affine_range(channels // 128):
        channel_slice = chunk(channel_tile_id, 128)
        # weights and biases for current tile
        weight = nl.load(w_tensor.reshape((ch, ks))[channel_slice, :])
        bias = nl.load(bias_tensor[channel_slice])

        # flip the weights (needed to compute the gradients)
        i_p = nl.arange(128)[:, None]
        i_flip = 128 - 1 - nl.arange(128)[None, :]

        d_weight = nl.zeros((128, 4), dtype=dtype)
        d_bias = nl.zeros((128, 1), dtype=dtype)

        for batch_id in nl.affine_range(batch_size):
            padded_input = nl.zeros(shape=(128, pad + 128), dtype=dtype)
            for seq_id in nl.sequential_range(seq_len // 128):
                seq_slice = chunk(seq_id, 128)
                input_tile = nl.load(input_tensor[batch_id, seq_slice, channel_slice])  # [seq=128, ch=128]
                d_out = nl.load(d_output_tensor[batch_id, seq_slice, channel_slice])

                padded_input[:, pad:] = nl.copy(nl.transpose(input_tile))  # [ch=128, seq=128+pad]

                # run the convolution
                conv = _conv_tile(padded_input, weight, bias, dtype)
                # remember that out = nl.silu(conv)
                # todo: check if this is the correct way to pad
                d_conv_temp = nl.transpose(d_out) * nl.silu_dx(conv[:, :128])
                d_conv_flipped = nl.zeros_like(padded_input)
                d_conv_flipped[:, pad:] = d_conv_temp[i_p, i_flip]

                d_input = _conv_tile(d_conv_flipped, weight, None, dtype)
                d_input_no_pad = nl.zeros(shape=(128, 128), dtype=dtype)
                d_input_no_pad[:, :] = d_input[i_p, i_flip]
                nl.store(d_input_tensor[batch_id, seq_slice, channel_slice], nl.transpose(d_input_no_pad))

                # Compute d_weight and d_bias
                d_bias += nl.sum(d_conv_temp, axis=[1])
                d_weight_batch = nl.zeros_like(d_weight)
                for i in nl.static_range(ks):
                    d_weight_batch[:, i] = nl.sum(padded_input[:, i:i + 128] * d_conv_temp, axis=[1])
                d_weight += d_weight_batch

                # move final elements at beginning to serve as padding for next sequence
                padded_input[:, :pad] = padded_input[:, 128:]
        nl.store(d_w_tensor[channel_slice, 0, :], d_weight)
        nl.store(d_bias_tensor[channel_slice], d_bias)
    return d_input_tensor, d_w_tensor, d_bias_tensor


# @nki_jit
def conv1d_grouped_kernel_grad(input_data, w, d_output, d_input, d_w, d_b, activation=None):
    batch_size, p, n = input_data.shape
    ch, _, ks = w.shape
    dtype = input_data.dtype

    assert p % 128 == 0 and ch == p and p == ch
    assert n % ks == 0 and n > ks  # check n is a multiple of kernel size
    assert ks == 4

    i_p = nl.arange(128)[:, None]
    i_f_n = nl.arange(n)[None, :]
    i_f_w = nl.arange(ks)[None, :]
    seq_len = n + ks - 1
    i_f_seq_len = nl.arange(seq_len)[None, :]

    if activation is not None:
        d_activation = 'd_' + activation
    else:
        d_activation = 'd_identity'

    for chunk_id in nl.affine_range(input_data.shape[1] // 128):
        i_p_input = chunk_id * 128 + nl.arange(128)[:, None]
        w_tile = nl.load(w[i_p_input, 0, i_f_w])
        # we don't need the bias to compute gradients
        b_tile = None

        db_accumulation = nl.zeros([128, 1], dtype=dtype)
        dw_accumulation = nl.zeros([128, ks], dtype=dtype)

        for batch_id in nl.affine_range(input_data.shape[0]):
            # fixme: probably don't need to pad this
            x = nl.zeros(shape=(128, n + ks - 1), dtype=dtype)
            x[i_p, ks - 1 + i_f_n] = nl.load(input_data[batch_id, i_p_input, i_f_n])

            if activation is not None:
                preact_grad = _conv_tile(x, w_tile, b_tile, dtype, activation=d_activation)[i_p, i_f_n]
            else:
                preact_grad = 1
            dout_tile = nl.zeros(shape=(128, n + ks - 1), dtype=dtype)
            dout_tile[i_p, i_f_n] = preact_grad * nl.load(d_output[batch_id, i_p_input, i_f_n])

            # Compute db
            db_accumulation += nl.sum(dout_tile[i_p, i_f_n], axis=[1])

            # Compute d_input
            dout_reverse = nl.ndarray((128, seq_len), dtype=dtype)
            # fixme: we should simply index the tile with flipped indexes, no need for the copy
            #   but it will break down later as double indexing tile[i_p, i_f][i_p1, i_f1] is not supported
            dout_reverse[i_p, i_f_seq_len] = dout_tile[i_p, seq_len - 1 - i_f_seq_len]
            # dout_reverse = dout_tile[i_p, seq_len - 1 - i_f_seq_len]

            conv = _conv_tile(dout_reverse, w_tile, b_tile=None, dtype=dtype, activation=None)

            # We flip the result while storing
            nl.store(d_input[batch_id, i_p_input, i_f_n], conv[i_p, seq_len - ks - i_f_n])

            dw_batch = nl.ndarray((128, 4), dtype=dtype)
            # Compute dw
            for i in nl.static_range(ks):
                # todo: the vector engine should be able to execute both element-wise product and sum in one instruction
                dw_batch[i_p, i] = nl.sum(x[i_p, i + i_f_n] * dout_tile[i_p, i_f_n], axis=[1])
            dw_accumulation += dw_batch

        nl.store(d_b[i_p_input], db_accumulation[i_p, 0])
        nl.store(d_w[i_p_input, 0, i_f_w], dw_accumulation[i_p, i_f_w])


class GroupedConv1dNKI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, activation=None):
        # fixme: if output is too large we might avoid to store it and recomputed it during backprop
        output = torch.empty_like(input)
        # if input.shape[2] <= 2048:
        #     output = conv1d_grouped_kernel(input, weight, bias, output, activation=activation)
        # else:
        output = conv1d_grouped_kernel_longseq(input, weight, bias, output, activation=activation)
        ctx.save_for_backward(input, weight, bias)
        ctx.activation = activation
        return output

    @staticmethod
    def backward(ctx, d_output):
        input, weight, bias = ctx.saved_tensors
        dinput = torch.empty_like(input)
        dweight = torch.empty_like(weight)
        dbias = torch.empty_like(bias)
        if input.shape[2] > 2048:
            raise NotImplementedError('Gradient not implemented for conv1d with seq_len>2048')
        # dinput, dweight, dbias = conv1d_grouped_kernel_bwd(input, weight, bias, d_output)
        conv1d_grouped_kernel_grad(input, weight, d_output, dinput, dweight, dbias, activation=ctx.activation)
        return dinput, dweight, dbias, None


def nki_conv1d(input, weight, bias=None, activation=None):
    return GroupedConv1dNKI.apply(input, weight, bias, activation)


class ConvNKI(nn.Conv1d):
    """
    Custom layer implemented in NKI to compute efficiently a grouped convolution,
    equivalent to nn.Conv1d with groups == in_channels == out_channels.

    Parameters:
        input: (B_tensor, C_tensor, L)
        weight: (C_tensor, 1, kernel_size)
        bias: (C_tensor)
    Return:
        output: (B_tensor, C_tensor, L) Each input channel sequence input[b, c, :] is convolved with its own conv weight[c, 0, :].
                          The results are then stacked together.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: Union[str, int] = 0, dilation: int = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros', device=None, dtype=None, activation=None) -> None:
        # We only support a very specific use case, check we are in it
        assert groups == in_channels, "NKI grouped conv kernel only supports groups == in_channels"
        assert padding == kernel_size - 1
        assert padding_mode == 'zeros'
        assert dilation == 1
        assert stride == 1
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode,
                         device, dtype)
        self.activation = activation
        self.parallel_split()

    def parallel_split(self):
        tp_rank = ps.get_tensor_model_parallel_rank()
        tp_size = ps.get_tensor_model_parallel_size()

        chunk = slice(self.out_channels // tp_size * tp_rank, self.out_channels // tp_size * (tp_rank + 1))
        self.weight.data = self.weight.data[chunk].detach().clone()
        self.bias.data = self.bias.data[chunk].detach().clone()
        self.in_channels = self.out_channels // tp_size
        self.out_channels = self.out_channels // tp_size

    def forward(self, input):
        return GroupedConv1dNKI.apply(input, self.weight, self.bias, self.activation)


if __name__ == "__main__":
    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["XLA_HLO_DEBUG"] = "1"
    os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
    os.environ["NEURON_CC_FLAGS"] = " --disable-internal-io-dge "

    torch.manual_seed(0)
    B, D, L = 1, 128 * 16, 1024
    kernel_size = 4
    x = torch.normal(0, 1, (B, D, L), requires_grad=True)

    # reference implementation
    conv = nn.Conv1d(
        in_channels=D,
        out_channels=D,
        bias=True,
        kernel_size=kernel_size,
        groups=D,
        padding=kernel_size - 1,
    )
    conv.bias.data[:] = 0

    # w = torch.ones_like(conv.weight)
    w = torch.rand_like(conv.weight)
    conv.weight = torch.nn.Parameter(w)
    b = conv.bias
    mask = torch.rand_like(x)

    # forward and backward for reference
    y_ref = conv(x)[:, :, :L]
    y_ref = F.silu(y_ref)
    y_ref.sum().backward()

    device = xm.xla_device()
    # dtype = torch.bfloat16
    dtype = torch.float32

    # NKI implementation
    conv_nki = ConvNKI(
        in_channels=D,
        out_channels=D,
        bias=True,
        kernel_size=kernel_size,
        groups=D,
        padding=kernel_size - 1,
        activation='silu'
    )
    conv_nki.weight = torch.nn.Parameter(w)
    conv_nki.bias = torch.nn.Parameter(b)
    a = torch.nn.Parameter(x.data.to(device=device, dtype=dtype))
    conv_nki.to(device=device, dtype=dtype)
    mask = mask.to(device)
    xm.mark_step()

    # forward and backward for NKI
    out = conv_nki(a)
    xm.mark_step()

    out = out.cpu().to(torch.float32)

    dist = lambda a, b: (torch.abs(a - b) / (torch.abs(a) + 1e-12)).mean()

    print('nn.conv1d: ', y_ref[0])
    print('NKI: ', out[0])

    print('NKI kernel is ', torch.allclose(out, y_ref, atol=1e-5))
    print('Avg distance: ', dist(y_ref, out).item())
