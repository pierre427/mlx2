# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
import math
import os

import mlx.core as mx
import mlx.nn as nn
from .activations import swiglu
from .import_env import snapshot as _import_env_snapshot

_import_env_snapshot(__name__)

_SORTED_GATHER_TAIL_BUG = True
_GATHER_SORT_MIN_ASSIGNMENTS = 20
_QMM_TILE = 32
_SORTED_QMM_K_TILE = 64

# MLX's GatherQMM streams each expert once (``gather_qmm_rhs``) only when a
# sorted gather has M == 1, B >= 16 and B // E >= 4 (mlx 39400a0d4,
# quantized.cpp GatherQMM::eval_gpu); below that every row runs a
# ``gather_qmv`` that re-reads its expert's weights.  When a sorted quantized
# gather has at least ``_RHS_PAD_MIN_ROWS_PER_EXPERT`` rows per expert, the
# sorted rows are padded up to 4 per expert so MLX picks the streaming kernel
# (ddalcu/mlx-serve#671).  The pad rows repeat the last sorted row, so they
# stay sorted, touch no new expert, and ``inv_order`` never reads them.
# 0 = off (default): the kernel switch is not bit-exact against gather_qmv.
_RHS_ROWS_PER_EXPERT = 4
_RHS_MIN_ROWS = 16
_RHS_PAD_MIN_ROWS_PER_EXPERT = int(os.environ.get("MLX2_MOE_RHS_PAD_MIN_ROWS", "0") or 0)
# Observed-use counters: sorted gathers padded, and pad rows added.
rhs_pad_calls = 0
rhs_pad_rows = 0


def _quantized_gather_tail_policy(mode: str, input_dims: int, sorted_indices: bool):
    """Return ``dense``, ``unsorted``, or ``native`` for a gather_qmm call."""
    if mode == "nvfp4" and input_dims % _QMM_TILE:
        return "dense"
    if sorted_indices and input_dims % _SORTED_QMM_K_TILE:
        return "unsorted"
    return "native"


def _rhs_stream_pad(n: int, num_experts) -> int:
    """Pad rows that lift ``n`` sorted rows to MLX's streaming-kernel floor."""
    floor = _RHS_PAD_MIN_ROWS_PER_EXPERT
    if floor <= 0 or not num_experts or n < floor * num_experts:
        return 0
    target = max(_RHS_ROWS_PER_EXPERT * num_experts, _RHS_MIN_ROWS)
    return max(0, target - n)


def _rhs_pad_experts(*projections):
    """Expert count for the streaming pad, or None when it cannot apply.

    Only a natively sorted quantized gather has the rows-per-expert cliff;
    ``gather_mm`` streams every sorted gather already.
    """
    experts = None
    for proj in projections:
        if not isinstance(proj, QuantizedSwitchLinear):
            return None
        policy = _quantized_gather_tail_policy(proj.mode, int(proj.input_dims), True)
        if policy != "native":
            return None
        experts = max(experts or 0, int(proj.num_experts))
    return experts


def _pad_sorted_tail(x, indices, pad):
    x = mx.concatenate([x, mx.broadcast_to(x[-1:], (pad,) + x.shape[1:])], axis=0)
    indices = mx.concatenate([indices, mx.broadcast_to(indices[-1:], (pad,))], axis=0)
    return x, indices


def _gather_sort(x, indices, num_experts=None):
    global rhs_pad_calls, rhs_pad_rows
    (*_, M) = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    x = x.flatten(0, -3)[order // M]
    indices = indices[order]
    n = indices.size
    pad = _rhs_stream_pad(n, num_experts)
    if pad:
        rhs_pad_calls += 1
        rhs_pad_rows += pad
    elif _SORTED_GATHER_TAIL_BUG and n > 32768 and (n % 64 != 0):
        pad = 64 - n % 64
    if pad:
        x, indices = _pad_sorted_tail(x, indices, pad)
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
        # The constructor initializes and quantizes a random full expert bank,
        # which conversion immediately discards. Build only the real bank.
        ql = QuantizedSwitchLinear.__new__(QuantizedSwitchLinear)
        nn.Module.__init__(ql)
        ql.group_size, ql.bits, ql.mode = group_size, bits, mode
        (ql.weight, ql.scales, *biases) = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None
        if "bias" in self:
            ql.bias = self.bias
        ql.freeze()
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
            (x, idx, inv_order) = _gather_sort(
                x, indices,
                _rhs_pad_experts(self.gate_proj, self.up_proj, self.down_proj),
            )
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
            x, idx, inv_order = _gather_sort(
                x, indices, _rhs_pad_experts(self.fc1, self.fc2)
            )
        if self.training:
            idx = mx.stop_gradient(idx)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.fc2(self.activation(x), idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)
