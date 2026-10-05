# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
import math
import os

import mlx.core as mx
import mlx.nn as nn

from . import invariant_prefill as _invariant
from . import moe_nax_gather as _nax
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
# ``gather_qmv`` that re-reads its expert's weights.  With at least
# ``_RHS_PAD_MIN_ROWS_PER_EXPERT`` sorted rows per expert, a natively sorted
# quantized gather is padded up to 4 rows per expert so MLX picks the
# streaming kernel (ddalcu/mlx-serve#671).  The pad rows repeat the last
# sorted row, so they stay sorted and touch no new expert; their outputs are
# sliced off.  Default 3 (Pierre, 2026-10-01; 0 = off).  Not bit-exact: the
# streaming kernel reduces in another order -- the same change a prompt gets
# today when it crosses MLX's own floor.  On an M5, 3 cut short-prompt TTFT by
# up to 23% (Qwen3.6-35B) and 13% (Nemotron-3.5) without a measured loss; 2
# lost 11% on Nemotron at 2.25 rows/expert
# (qualification/runs/recon-20261001/l5-moe-pad).  Unqualified until the next
# qualification campaign.
_RHS_ROWS_PER_EXPERT = 4
_RHS_MIN_ROWS = 16


def _rhs_pad_floor(environ=os.environ) -> int:
    return int(environ.get("MLX2_MOE_RHS_PAD_MIN_ROWS", "3") or 0)


_RHS_PAD_MIN_ROWS_PER_EXPERT = _rhs_pad_floor()
# Observed-use counters: padded gather_qmm calls, and pad rows added.
rhs_pad_calls = 0
rhs_pad_rows = 0

# Adaptive kernel choice (module default off: MLX2_MOE_RHS_PAD_POLICY=adaptive;
# Flash-Next selects it through its policy, see flash_next_policy).
# Instead of one rows-per-expert floor for every table, each sorted gather
# picks per-row ``gather_qmv`` or the padded ``gather_qmm_rhs`` from a cost
# model calibrated per expert-table class (format, bits, group, experts,
# output x input dims) on this hardware:
#     qmv_ms = q0 + q1 * assignments
#     rhs_ms = r0 + r1 * E * (1 - exp(-assignments / E))   (expected experts touched)
# The two projections of a SwiGLU expert differ (the 640-out gate/up tables
# never gain from streaming below MLX's own floor; the 640-in down table
# does from ~2 rows/expert).  A table with no calibration keeps the fixed
# floor (reason ``uncalibrated``).  ``always`` (an A/B arm, MTPLX #549's
# rule) pads every sorted gather to the streaming kernel so the expert
# kernel never depends on row count.  ``MLX2_MOE_RHS_PAD_MIN_ROWS=0`` still
# turns all padding off.  Not bit-exact against the floor: qmv and rhs
# reduce in different orders (qualification/runs/moe-adaptive-pad-20261001).
def _rhs_pad_policy(environ=os.environ) -> str:
    value = (environ.get("MLX2_MOE_RHS_PAD_POLICY", "floor") or "floor").strip().lower()
    if value not in {"floor", "adaptive", "always"}:
        raise ValueError(
            "MLX2_MOE_RHS_PAD_POLICY must be 'floor', 'adaptive' or 'always', "
            f"got {value!r}"
        )
    return value


_RHS_PAD_POLICY = _rhs_pad_policy()


def set_pad_policy(policy: str) -> str:
    """Select the pad policy at run time; returns the previous one.

    An adapter that owns the choice (Flash-Next ``moe_rhs_pad_policy``) pins
    ``MLX2_MOE_RHS_PAD_POLICY`` and calls this after load, so an import of
    this module under another value cannot change the route behind the
    receipt (the same contract as ``moe_nax_gather.set_mode``).
    """
    global _RHS_PAD_POLICY
    value = _rhs_pad_policy({"MLX2_MOE_RHS_PAD_POLICY": policy})
    old, _RHS_PAD_POLICY = _RHS_PAD_POLICY, value
    return old
# (mode, bits, group_size, experts, output_dims, input_dims) -> (q0, q1, r0, r1),
# ms per call.  Fitted in-model on M5 Max (mlx 39400a0d4) from real weights
# and real routing, every sorted gather of one forward replayed both ways at
# 16..204 rows (scripts/calibrate_moe_pad_inmodel.py, Flash-Next 4-bit
# experts; qualification/runs/moe-adaptive-pad-20261001).  The touched term
# assumes uniform routing; real routing touches fewer experts, which the
# fitted r0/r1 absorb for this table.
_PAD_COST_MODELS = {
    # Flash-Next gate/up: 512 experts, 2560 -> 640.  rhs wins from ~120 rows.
    ("affine", 4, 64, 512, 640, 2560): (-0.091401, 0.00096, 0.163268, 0.001819),
    # Flash-Next down: 512 experts, 640 -> 2560.  rhs wins from ~30 rows.
    ("affine", 4, 64, 512, 2560, 640): (-0.26367, 0.002081, -0.009772, 0.001673),
}
pad_choice_counts = {}


def _record_pad_choice(reason: str) -> None:
    pad_choice_counts[reason] = pad_choice_counts.get(reason, 0) + 1


def moe_pad_status(*, reset: bool = False) -> dict:
    """Which sorted-gather kernel was chosen, and why, since the last reset."""
    out = {
        "policy": _RHS_PAD_POLICY,
        "floor_rows_per_expert": _RHS_PAD_MIN_ROWS_PER_EXPERT,
        "padded_calls": rhs_pad_calls,
        "pad_rows": rhs_pad_rows,
        "choices": dict(pad_choice_counts),
    }
    if reset:
        pad_choice_counts.clear()
    return out


def _adaptive_pad(n: int, layer) -> tuple:
    """``(pad_rows, reason)`` for ``n`` sorted rows into ``layer``'s table."""
    num_experts = layer.num_experts
    stream_floor = max(_RHS_ROWS_PER_EXPERT * num_experts, _RHS_MIN_ROWS)
    if n >= stream_floor:
        return 0, "mlx_streams"
    if _RHS_PAD_POLICY == "always":
        return stream_floor - n, "always_rhs"
    key = (layer.mode, layer.bits, layer.group_size, num_experts,
           layer.output_dims, layer.input_dims)
    model = _PAD_COST_MODELS.get(key)
    if model is None:
        pad = _rhs_stream_pad(n, num_experts)
        return pad, "uncalibrated_rhs" if pad else "uncalibrated_qmv"
    q0, q1, r0, r1 = model
    touched = num_experts * (1.0 - math.exp(-n / num_experts))
    if r0 + r1 * touched < q0 + q1 * n:
        return stream_floor - n, "cost_rhs"
    return 0, "cost_qmv"


def _quantized_gather_tail_policy(mode: str, input_dims: int, sorted_indices: bool):
    """Return ``dense``, ``unsorted``, or ``native`` for a gather_qmm call."""
    if mode == "nvfp4" and input_dims % _QMM_TILE:
        return "dense"
    if sorted_indices and input_dims % _SORTED_QMM_K_TILE:
        return "unsorted"
    return "native"


def _rhs_stream_pad(n: int, num_experts: int) -> int:
    """Pad rows that lift ``n`` sorted rows to MLX's streaming-kernel floor."""
    floor = _RHS_PAD_MIN_ROWS_PER_EXPERT
    if floor <= 0 or n < floor * num_experts:
        return 0
    return max(0, max(_RHS_ROWS_PER_EXPERT * num_experts, _RHS_MIN_ROWS) - n)


def _pad_sorted_tail(x, indices, pad):
    x = mx.concatenate([x, mx.broadcast_to(x[-1:], (pad,) + x.shape[1:])], axis=0)
    indices = mx.concatenate([indices, mx.broadcast_to(indices[-1:], (pad,))], axis=0)
    return x, indices


def _gather_sort(x, indices):
    (*_, M) = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    x = x.flatten(0, -3)[order // M]
    indices = indices[order]
    n = indices.size
    if _SORTED_GATHER_TAIL_BUG and n > 32768 and (n % 64 != 0):
        (x, indices) = _pad_sorted_tail(x, indices, 64 - n % 64)
    return (x, indices, inv_order)


def _sort_routes(x, indices):
    """``_gather_sort`` with the replicated rows left lazy.

    Returns ``(x_sorted, idx, inv_order, x_tok, row_map)``: the same ops as
    ``_gather_sort`` (including its tail pad, applied to the index and the
    row map), so ``x_sorted == x_tok[row_map]`` is exactly ``_gather_sort``'s
    sorted ``x``.  MLX is lazy: ``x_sorted`` is only computed if a consumer
    that is evaluated reads it (the NAX row-mapped gate/up kernel reads
    ``x_tok`` through ``row_map`` instead; omlx #4029 ``sort_routes``).
    """
    (*_, M) = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    x_tok = x.flatten(0, -3)
    row_map = order // M
    indices = indices[order]
    n = indices.size
    if _SORTED_GATHER_TAIL_BUG and n > 32768 and (n % 64 != 0):
        pad = 64 - n % 64
        row_map = mx.concatenate([row_map, mx.broadcast_to(row_map[-1:], (pad,))])
        indices = mx.concatenate([indices, mx.broadcast_to(indices[-1:], (pad,))])
    return (x_tok[row_map], indices, inv_order, x_tok, row_map)


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
        global rhs_pad_calls, rhs_pad_rows
        tail_policy = _quantized_gather_tail_policy(
            self.mode, int(x.shape[-1]), sorted_indices
        )
        if _invariant.active() and (tail_policy != "native" or not sorted_indices):
            _invariant.not_invariant(f"gather_{tail_policy}_sorted{int(bool(sorted_indices))}")
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
            rows, pad = indices.size, 0
            lane = (
                _invariant.active()
                and sorted_indices
                and tail_policy == "native"
                and indices.ndim == 1
                and x.ndim == 3
                and x.shape[0] == rows
            )
            if lane:
                # Invariant prefill lane: always the streaming kernel, whatever
                # the floor policy (its kill switch governs the default route).
                pad = _invariant.sorted_gather_pad(rows, self.num_experts)
            elif sorted_indices and tail_policy == "native" and _RHS_PAD_MIN_ROWS_PER_EXPERT:
                if indices.ndim == 1 and x.ndim == 3 and x.shape[0] == rows:
                    if _RHS_PAD_POLICY != "floor":
                        pad, reason = _adaptive_pad(rows, self)
                        _record_pad_choice(reason)
                    else:
                        pad = _rhs_stream_pad(rows, self.num_experts)
            rhs = indices
            if pad:
                rhs_pad_calls += 1
                rhs_pad_rows += pad
                (x, rhs) = _pad_sorted_tail(x, indices, pad)
            nax = None
            if (
                _nax.MODE != "off"
                # The invariant prefill lane pins one kernel configuration at
                # every width; the NAX gather is bit-identical to the stock
                # sorted kernel but its admission depends on the row count, so
                # it stands aside rather than enter the lane unproven.
                and not lane
                and sorted_indices
                and tail_policy == "native"
                and "bias" not in self
                # Prefill forwards only: a padded decode or verify call can
                # reach the NAX row floor (Codex port review item 7).
                and _nax.admit_phase()
            ):
                # The segmented NAX kernel, bit-identical to the stock
                # sorted rhs kernel it replaces (omlx #3995); None keeps it.
                nax = _nax.try_gather(x, self, rhs)
            if nax is not None:
                x = nax
            else:
                x = mx.gather_qmm(
                    x,
                    self["weight"],
                    self["scales"],
                    self.get("biases"),
                    rhs_indices=rhs,
                    transpose=True,
                    group_size=self.group_size,
                    bits=self.bits,
                    mode=self.mode,
                    sorted_indices=False if tail_policy == "unsorted" else sorted_indices,
                )
            if pad:
                x = x[:rows]
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
        do_sort = indices.size >= _GATHER_SORT_MIN_ASSIGNMENTS or _invariant.active()
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
