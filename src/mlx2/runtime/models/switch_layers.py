# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
import math
import mlx.core as mx
import mlx.nn as nn
from .activations import swiglu

_SORTED_GATHER_TAIL_BUG = True
_GATHER_SORT_MIN_ASSIGNMENTS = 20
_QMM_TILE = 32
_SORTED_QMM_K_TILE = 64


def _quantized_gather_tail_policy(mode: str, input_dims: int, sorted_indices: bool):
    """Return ``dense``, ``unsorted``, or ``native`` for a gather_qmm call."""
    if mode == "nvfp4" and input_dims % _QMM_TILE:
        return "dense"
    if sorted_indices and input_dims % _SORTED_QMM_K_TILE:
        return "unsorted"
    return "native"


def _gather_sort(x, indices):
    (*_, M) = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    x = x.flatten(0, -3)[order // M]
    indices = indices[order]
    n = indices.size
    if _SORTED_GATHER_TAIL_BUG and n > 32768 and (n % 64 != 0):
        pad = 64 - n % 64
        x = mx.concatenate([x, mx.broadcast_to(x[-1:], (pad,) + x.shape[1:])], axis=0)
        indices = mx.concatenate(
            [indices, mx.broadcast_to(indices[-1:], (pad,))], axis=0
        )
    return (x, indices, inv_order)


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        (self.weight, self.scales, *biases) = mx.quantize(
            mx.random.uniform(
                low=-scale, high=scale, shape=(num_experts, output_dims, input_dims)
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        tail_policy = _quantized_gather_tail_policy(
            self.mode, int(x.shape[-1]), sorted_indices
        )
        if tail_policy == "dense":
            quantized = [self["weight"], self["scales"]]
            if self.get("biases") is not None:
                quantized.append(self["biases"])
            weight = mx.dequantize(
                *quantized, group_size=self.group_size, bits=self.bits, mode=self.mode
            )
            x = mx.gather_mm(
                x,
                weight.swapaxes(-1, -2),
                rhs_indices=indices,
                sorted_indices=sorted_indices,
            )
        else:
            x = mx.gather_qmm(
                x,
                self["weight"],
                self["scales"],
                self.get("biases"),
                rhs_indices=indices,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
                sorted_indices=False if tail_policy == "unsorted" else sorted_indices,
            )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale, high=scale, shape=(num_experts, output_dims, input_dims)
        )
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        (num_experts, output_dims, input_dims) = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims, output_dims, num_experts, False, group_size, bits, mode=mode
        )
        (ql.weight, ql.scales, *biases) = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None
        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()
        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= _GATHER_SORT_MIN_ASSIGNMENTS
        idx = indices
        inv_order = None
        if do_sort:
            (x, idx, inv_order) = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


class SwitchMLP(nn.Module):
    """Two-projection expert MLP with locality-aware gather ordering."""

    def __init__(
        self, input_dims: int, hidden_dims: int, num_experts: int,
        activation=None, bias: bool = False,
    ):
        super().__init__()
        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation if activation is not None else nn.GELU(approx="precise")

    def __call__(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= _GATHER_SORT_MIN_ASSIGNMENTS
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.fc2(self.activation(x), idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)
