# SPDX-License-Identifier: Apache-2.0
# Launch structure adapted from jundot/omlx PR #4038 (Apache-2.0, head
# 2608f678); see docs/PROVENANCE.md and provenance/omlx-4038-qwen4-hc-decode.json.
# The arithmetic is re-derived from the MLX fork this venv runs (39400a0d4:
# rms_norm.metal rms_single_row, quantized.h qmv_fast/qmv, unary/binary ops,
# reduce_col.h col_reduce_small), NOT from omlx's kernels: omlx's two-launch
# kernels are bit-equal to omlx's older fused kernels, whose FP32 epilogues and
# K-slice sums round differently from the composed MLX ops mlx2 serves.
"""Two-launch hyper-connection (HC) decode for Flash-Next (qwen4_exp).

A one-token ``GatedResidual`` call runs, as composed MLX ops, about nineteen
dependent launches: the grouped RMS norm (astype, ``fast.rms_norm``, weight
astype, multiply, astype), the down projection, ``/ hc`` and SiLU, the up
projection and sigmoid, the stream product and the mean (sum, scale), and the
inject projection with ``/ hc``, sigmoid and ``* 2``.  This module computes the
same three outputs in two launches:

``norm_down``  (one threadgroup of ``hidden / 4`` threads per row group)
    Every threadgroup normalises all four streams with ``rms_single_row``'s
    lane mapping, add order and ``precise::rsqrt``, then applies the weight
    multiply and the bf16 round as the eager ops do, keeping the normed row in
    threadgroup memory.  Each simdgroup then runs whole ``qmv_fast`` rows of the
    down projection (the eager path's kernel at one row), and lane 0 applies
    ``/ hc`` and the compiled SiLU in bf16.  One extra threadgroup runs the four
    inject rows (``qmv_fast_rows`` arithmetic) and ``2 * sigmoid(raw / hc)``.
``up_mix``
    One simdgroup per hidden column ``h`` runs the four ``qmv`` up rows
    ``s * hidden + h`` (the eager path's kernel for a 320-wide input), then
    lane 0 applies the sigmoid, the stream product and the stream sum in
    ``col_reduce_small``'s bf16 order, then the mean's ``* 0.25``.

Both launches are intended to be bit-identical to the composed path; that is a
Metal-checked claim (``scripts/check_qwen4_hc_decode.py``), not an assumption.
The pending residual write ``residual + branch * inject`` (``_apply_inject``)
is NOT folded in: it stays a separable stage owned by the composed path (see
``PENDING_WRITE_SEAM``).

Rows: MLX folds every leading dimension into the matvec's M, and picks the
kernel by M.  One row runs ``qmv_fast``/``qmv`` (law 0 above).  For 2..8 rows
on Apple GPU family 15+ it runs ``qmv_wide`` (law 1: an 8-lane group per
output row, per-group 8-value dequantised sub-chunks, a shuffle-down ladder),
which both launches also transcribe, so verify windows and multi-lane decode
stay bit-identical to their own composed call (not to one-row decode, exactly
as the composed path).  Other row counts are declined and counted.  Every
row is its own grid row, so 2..8 rows re-read the weights per row where MLX
reads them once: a win for all-4-bit layouts, a loss with 8-bit projections
or a dense inject.  ``MLX_QWEN4_HC_MULTI_ROW`` (policy ``hc_decode_multi_row``)
is ``auto`` by default (multi-row only for all-4-bit layouts), or ``on`` /
``off``; declined calls are counted.

Row-exact verify windows (``qwen4_row_exact``, omlx #4041/#4105): inside a
``row_exact_verify.window`` every row must carry its one-token step's bits, so
both launches run law 0 with one grid row per verify row (up to
``ROW_EXACT_MAX_ROWS``), never qmv_wide.  The route's class-swapped
projections (``_row_exact_base`` = ``nn.QuantizedLinear``) are admitted.  Opt-in
(``MLX_QWEN4_HC_ROW_EXACT``, policy ``row_exact_window_kernels``); off, a window
keeps the composed path, which is row-exact too.

Admission is structural and exact-shape: rows as above, bf16 activations,
four streams, affine group-64 projections with bf16 scales/biases that are
exactly ``nn.QuantizedLinear``, and the eager norm profile (fast RMS norm on,
fused group norm and compiled glue off).  Anything else keeps the composed
body.

Weight formats: down, up and a quantized inject may each be 4-bit or 8-bit
(mixed precision artifacts quantize the HC projections at 8 bits); every
launch transcribes the format's own MLX arithmetic (``quantized.h``
``load_vector``/``qdot``/``dequantize`` ``bits == 8`` branches: unscaled
inputs, one weight byte per value, a 256-value ``qmv_fast`` block and a
128-value ``qmv`` block).  A dense (bf16 ``nn.Linear``) inject runs in
norm_down's inject threadgroup with MLX's dense matvec arithmetic
(``gemv.h``): ``GEMVKernel`` at one row (eight simdgroups over K, four
products per lane per row, shuffle ladder, simdgroup sum in order) and
``GemvWide`` at 2..8 rows (one row per simdgroup, eight float4 ``dot`` per
lane per step).

Default off.  ``MLX_QWEN4_HC_DECODE=1`` (the Flash-Next policy field
``hc_decode_kernels``) enables it at import; ``set_hc_decode_enabled`` switches
live; ``MLX_QWEN4_HC_DECODE=0`` or the setter is the kill switch.
"""

from __future__ import annotations

import os
import re
import threading
from collections import Counter
from itertools import chain
from operator import is_

import mlx.core as mx
from mlx import nn

from .served_exp import ServedExpGate, metal_helper

HC_DECODE_ENV = "MLX_QWEN4_HC_DECODE"
HC_COUNT = 4
# Weight formats the launches transcribe (affine, group 64), per projection:
# MLX's 4-bit and 8-bit qmv_fast/qmv/qmv_wide arithmetic.
BITS = 4
FORMAT_BITS = (4, 8)
GROUP_SIZE = 64
# Down rows per simdgroup in norm_down; one row keeps the most threadgroups.
DOWN_ROWS = 1
# Simdgroups (hidden columns) per up_mix threadgroup.
UP_SIMDGROUPS = 8
# Widest folded row count served with qmv_wide's arithmetic (verify windows).
MAX_WIDE_ROWS = 8
# Widest row-exact verify window served with the one-row law (copy drafts
# reach 17 rows; each row is its own grid row, so the bound is only a guard).
ROW_EXACT_MAX_ROWS = 32

# Where #4024's deferred write and the separate raw-gate+inject fusion meet:
# norm_down's first load reads ``hyper_input``, which is exactly the output of
# the previous block's ``_apply_inject(residual, branch, inject)``.  A pending
# write variant would load (residual, branch, inject) there and compute
# ``residual + branch * inject`` in bf16 (Multiply then Add, the eager order)
# before the sum of squares, storing the written residual as the passthrough.
PENDING_WRITE_SEAM = "norm_down:load(hyper_input) == _apply_inject output"


def enabled_from_env() -> bool:
    raw = os.environ.get(HC_DECODE_ENV, "0").strip().lower()
    if raw in {"", "0", "false", "off", "no"}:
        return False
    if raw in {"1", "true", "on", "yes"}:
        return True
    raise ValueError(f"{HC_DECODE_ENV}={raw!r}: expected 0/off or 1/on")


_ENABLED = enabled_from_env()
# Row-exact verify windows (qwen4_row_exact) served with the one-row law.
# Default off (the Flash-Next policy field ``row_exact_window_kernels`` sets
# MLX_QWEN4_HC_ROW_EXACT=1); off, windows keep the composed path (still
# row-exact).  Needs the kernels themselves on (MLX_QWEN4_HC_DECODE).
HC_ROW_EXACT_ENV = "MLX_QWEN4_HC_ROW_EXACT"


def row_exact_from_env() -> bool:
    raw = os.environ.get(HC_ROW_EXACT_ENV, "0").strip().lower()
    if raw in {"", "0", "false", "off", "no"}:
        return False
    if raw in {"1", "true", "on", "yes"}:
        return True
    raise ValueError(f"{HC_ROW_EXACT_ENV}={raw!r}: expected 0/off or 1/on")


_ROW_EXACT = row_exact_from_env()
# Multi-row (2..8 folded rows: verify windows, multi-lane decode) outside a
# row-exact window, served with the qmv_wide law.  Each row is its own grid
# row, so the launches read every weight once per row where MLX's qmv_wide /
# gemv_wide read it once for all rows.  Full-model A/Bs (2026-10-01,
# qualification/runs/omlx-w3-8bit-20261001): on the all-4-bit artifact the
# multi-row path wins (off: 4 lanes -5.8%, MTP on -6.8%); with 8-bit
# projections or a dense inject it loses (on: 4 lanes -5.2%, MTP on flat).
# Modes (MLX_QWEN4_HC_MULTI_ROW, policy ``hc_decode_multi_row``):
#   auto  (default) serve 2..8 rows only for all-4-bit layouts
#   on    serve 2..8 rows for every admitted layout
#   off   one-row calls only
# Row-exact windows (one-row law per row) are not affected.
HC_MULTI_ROW_ENV = "MLX_QWEN4_HC_MULTI_ROW"
MULTI_ROW_MODES = ("auto", "on", "off")


def multi_row_from_env() -> str:
    raw = os.environ.get(HC_MULTI_ROW_ENV, "auto").strip().lower()
    raw = {"": "auto", "1": "on", "true": "on", "yes": "on",
           "0": "off", "false": "off", "no": "off"}.get(raw, raw)
    if raw not in MULTI_ROW_MODES:
        raise ValueError(f"{HC_MULTI_ROW_ENV}={raw!r}: expected one of {MULTI_ROW_MODES}")
    return raw


_MULTI_ROW = multi_row_from_env()
MULTI_ROW_DECLINE = "rows>1 (multi-row path off)"
MULTI_ROW_FORMAT_DECLINE = "rows>1 (multi-row path auto: not an all-4-bit layout)"
_LOCK = threading.Lock()
_STATS: Counter = Counter()
_DECLINES: Counter = Counter()
_FORMATS: Counter = Counter()
_LAST_DECLINE: str | None = None
_LAST_ERROR: str | None = None
_BROKEN = False
_VALIDATED: set = set()


def hc_decode_enabled() -> bool:
    return _ENABLED


def set_hc_row_exact_enabled(enabled: bool) -> bool:
    """Live switch for the row-exact window mode; returns the previous value."""
    global _ROW_EXACT
    previous = _ROW_EXACT
    _ROW_EXACT = bool(enabled)
    return previous


def set_hc_multi_row_mode(mode: str) -> str:
    """Live switch for the multi-row (qmv_wide law) path: ``auto``, ``on``
    or ``off``; returns the previous mode."""
    global _MULTI_ROW
    if mode not in MULTI_ROW_MODES:
        raise ValueError(f"multi-row mode {mode!r}: expected one of {MULTI_ROW_MODES}")
    previous = _MULTI_ROW
    _MULTI_ROW = mode
    return previous


def set_hc_decode_enabled(enabled: bool) -> bool:
    """Live switch (also the kill switch); returns the previous value."""
    global _ENABLED
    previous = _ENABLED
    _ENABLED = bool(enabled)
    return previous


def hc_decode_status(*, reset: bool = False) -> dict:
    """Engagement counters.  ``calls`` counts GatedResidual calls served by the
    two launches (two per call); ``declines`` counts calls that stayed composed,
    by reason; ``errors`` counts first-use failures that demoted the path."""
    global _LAST_DECLINE, _LAST_ERROR
    with _LOCK:
        report = {
            "enabled": bool(_ENABLED),
            "broken": bool(_BROKEN),
            "calls": int(_STATS["calls"]),
            "inject_calls": int(_STATS["inject_calls"]),
            "row_exact_calls": int(_STATS["row_exact_calls"]),
            "row_exact_mode": bool(_ROW_EXACT),
            "multi_row_mode": _MULTI_ROW,
            "launches": int(2 * _STATS["calls"]),
            "declines": dict(_DECLINES),
            "calls_by_format": dict(_FORMATS),
            "errors": int(_STATS["errors"]),
            "last_decline": _LAST_DECLINE,
            "last_error": _LAST_ERROR,
        }
        if reset:
            _STATS.clear()
            _DECLINES.clear()
            _FORMATS.clear()
            _LAST_DECLINE = None
            _LAST_ERROR = None
    return report


def decline(reason: str) -> None:
    """Count a decline decided by the caller."""
    _decline(reason)


def _decline(reason: str) -> None:
    global _LAST_DECLINE
    with _LOCK:
        _DECLINES[reason] += 1
        _LAST_DECLINE = reason


def _plain_class(layer) -> type:
    """The module class the launches transcribe.

    The row-exact verify route (``qwen4_row_exact``) class-swaps every trunk
    ``QuantizedLinear`` to a subclass that records its original class as
    ``_row_exact_base``; outside a verify window that subclass is the plain
    layer (one-row calls pass straight through), and inside one it gives every
    row the one-row arithmetic these launches transcribe (law 0)."""
    cls = type(layer)
    return getattr(cls, "_row_exact_base", cls)


def _quantized_reason(layer) -> str | None:
    if _plain_class(layer) is not nn.QuantizedLinear:
        return "projection is not a plain QuantizedLinear"
    if "_lane_prepared" in layer.__dict__:
        return "projection owned by the lane matmul"
    if getattr(layer, "bits", None) not in FORMAT_BITS:
        return "bits"
    if getattr(layer, "group_size", None) != GROUP_SIZE:
        return "group size"
    if getattr(layer, "mode", "affine") != "affine":
        return "mode"
    if "bias" in layer or "biases" not in layer:
        return "needs affine biases and no linear bias"
    if layer["weight"].dtype != mx.uint32:
        return "weight dtype"
    if layer["scales"].dtype != mx.bfloat16 or layer["biases"].dtype != mx.bfloat16:
        return "scales/biases dtype"
    return None


def qmv_fast_alignment(bits: int) -> int:
    """MLX ``qmv_fast_k_alignment``: the K step of ``qmv_fast_impl``
    (``pack_factor * 2 * 32``): 512 at 4 bits, 256 at 8 bits."""
    return (32 // bits) * 2 * 32


def _dense_inject_reason(layer) -> str | None:
    """A dense (unquantized) inject projection: exactly ``nn.Linear`` without
    bias, bf16 weight (MLX ``gemv`` at one row, ``gemv_wide`` at 2..8)."""
    if _plain_class(layer) is not nn.Linear:
        return "inject: not a QuantizedLinear or a plain Linear"
    if "bias" in layer:
        return "inject: dense with bias"
    if layer["weight"].dtype != mx.bfloat16:
        return "inject: dense weight dtype"
    return None


def static_admission(module) -> str | None:
    """Model-layout check (cached per module by identity of its tensors).

    Mirrors the conditions under which the composed path runs the kernels this
    module transcribes: ``qmv_fast`` for the down and inject rows (input width
    a multiple of the format's qmv_fast block: 512 at 4 bits, 256 at 8),
    plain ``qmv`` for the up rows (input width not a multiple of that block,
    not 64/128), ``rms_single_row`` with whole four-element reads (hidden a
    multiple of 128, at most 4096).  Down, up and a quantized inject may each
    be 4- or 8-bit; a dense bf16 inject is admitted where MLX runs the
    ``gemv``/``gemv_wide`` tiling the launch transcribes.
    """
    hc = getattr(module, "hc_count", None)
    hidden = getattr(module, "hidden_size", None)
    if hc != HC_COUNT:
        return "hc_count"
    if not isinstance(hidden, int) or hidden % 128 or hidden > 4096:
        return "hidden size"
    width = hc * hidden
    norm = getattr(module, "hc_norm", None)
    weight = getattr(norm, "weight", None)
    if (
        norm is None
        or getattr(norm, "group_size", None) != hidden
        or not isinstance(weight, mx.array)
        or weight.shape != (width,)
        or weight.dtype not in (mx.bfloat16, mx.float32)
    ):
        return "hc_norm layout"
    down = getattr(module, "input_mix_weight_down", None)
    up = getattr(module, "input_mix_weight_up", None)
    for name, layer in (("down", down), ("up", up)):
        reason = _quantized_reason(layer)
        if reason is not None:
            return f"{name}: {reason}"
    lowrank = down["weight"].shape[0]
    down_bits, up_bits = down.bits, up.bits
    if tuple(down["weight"].shape) != (lowrank, width * down_bits // 32):
        return "down shape"
    if tuple(up["weight"].shape) != (width, lowrank * up_bits // 32):
        return "up shape"
    if width % qmv_fast_alignment(down_bits) or lowrank % 8:
        return "width (MLX would not run qmv_fast for the down rows)"
    if (
        lowrank % qmv_fast_alignment(up_bits) == 0
        or lowrank in (64, 128)
        or lowrank % GROUP_SIZE
    ):
        return "lowrank (MLX would not run plain qmv for the up rows)"
    simdgroups = hidden // 4 // 32
    if lowrank % (simdgroups * DOWN_ROWS) or lowrank % (simdgroups * 4):
        return "lowrank row tiling"
    if hidden % UP_SIMDGROUPS:
        return "hidden column tiling"
    if "block_inject_weight" in module:
        inject = module["block_inject_weight"]
        if _plain_class(inject) is nn.QuantizedLinear:
            reason = _quantized_reason(inject)
            if reason is not None:
                return f"inject: {reason}"
            if tuple(inject["weight"].shape) != (hc, width * inject.bits // 32):
                return "inject shape"
            if width % qmv_fast_alignment(inject.bits):
                return "inject width (MLX would not run qmv_fast)"
        else:
            reason = _dense_inject_reason(inject)
            if reason is not None:
                return reason
            if tuple(inject["weight"].shape) != (hc, width):
                return "inject shape"
            # MLX gemv at one row: K >= 16 * hc picks (BM 1, BN 8, SN 32,
            # TM 4, TN 4); whole 1024-column blocks (no leftover tail).
            if width % 1024 or width < 16 * hc:
                return "inject width (MLX gemv tiling not transcribed)"
    return None


def _inject_mode(module) -> str:
    """'none', 'quantized' or 'dense' (both computed in norm_down)."""
    if "block_inject_weight" not in module:
        return "none"
    inject = module["block_inject_weight"]
    return "quantized" if _plain_class(inject) is nn.QuantizedLinear else "dense"


def _identity(module) -> tuple:
    """Every child module, its class, and every array the launches read.

    Compared with ``is`` so a replaced tensor, projection or class swap (the
    TensorFold installer changes ``__class__``) re-runs the layout check; the
    cache holds the objects, so a freed id cannot alias a new one."""
    children = tuple(dict.values(module))
    return (
        *children,
        *map(type, children),
        *chain.from_iterable(map(dict.values, children)),
    )


def _cached_plan(module):
    """(reason, plan): the layout verdict and, when admitted, the launch plan."""
    refs = _identity(module)
    cached = module.__dict__.get("_hc_decode_plan")
    if cached is not None:
        old_refs, reason, plan = cached
        if len(old_refs) == len(refs) and all(map(is_, old_refs, refs)):
            return reason, plan
    try:
        reason = static_admission(module)
    except Exception as exc:  # noqa: BLE001 - odd layouts stay composed
        reason = f"layout check raised {type(exc).__name__}"
    plan = None if reason is not None else {}
    module.__dict__["_hc_decode_plan"] = (refs, reason, plan)
    return reason, plan


def _law_plan(module, plans: dict, law: int) -> dict:
    plan = plans.get(law)
    if plan is None:
        plan = plans[law] = _build_plan(module, law)
    return plan


def _cached_static_admission(module) -> str | None:
    return _cached_plan(module)[0]


def runtime_supported() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


# --------------------------------------------------------------------------
# Metal.  4- and 8-bit, group-64 transcriptions of the MLX fork's quantized.h helpers
# (load_vector / load_vector_safe accumulate the input sum in float since the
# fork's 2d6705923), qdot's add order, and the unary/binary/reduce operators the
# eager path dispatches, evaluated in T = bfloat16_t exactly as those kernels do.
# --------------------------------------------------------------------------
HEADER = r"""
using namespace metal;

template <typename P, int N>
inline float hcd_load(P x, thread float* x_thread) {
  float sum = 0;
  for (int i = 0; i < N; i += 4) {
    sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  return sum;
}

template <typename P, int VPT>
inline float hcd_load_safe(P x, thread float* x_thread, int N) {
  float sum = 0;
  for (int i = 0; i < N; i += 4) {
    sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  for (int i = N; i < VPT; i++) {
    x_thread[i] = 0;
  }
  return sum;
}

inline float hcd_qdot(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum,
    int N) {
  float accum = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < (N / 4); i++) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}

// 8-bit affine (MLX quantized.h load_vector / load_vector_safe / qdot /
// qdot_safe, ``bits == 8`` branches): the input sum adds the raw T values to
// the float sum, x_thread holds them unscaled, and each weight byte multiplies
// its value; then ``scale * accum + sum * bias``.
template <typename P, int N>
inline float hcd_load8(P x, thread float* x_thread) {
  float sum = 0;
  for (int i = 0; i < N; i++) {
    sum += x[i];
    x_thread[i] = x[i];
  }
  return sum;
}

template <typename P, int VPT>
inline float hcd_load8_safe(P x, thread float* x_thread, int N) {
  float sum = 0;
  for (int i = 0; i < N; i++) {
    sum += x[i];
    x_thread[i] = x[i];
  }
  for (int i = N; i < VPT; i++) {
    x_thread[i] = 0;
  }
  return sum;
}

inline float hcd_qdot8(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum,
    int N) {
  float accum = 0;
  for (int i = 0; i < N; i++) {
    accum += x_thread[i] * w[i];
  }
  return scale * accum + sum * bias;
}

// Format dispatch (BITS is a template constant: the other branch is dead).
template <int BITS, typename P, int N>
inline float hcd_loadq(P x, thread float* x_thread) {
  return BITS == 8 ? hcd_load8<P, N>(x, x_thread) : hcd_load<P, N>(x, x_thread);
}

template <int BITS, typename P, int VPT>
inline float hcd_loadq_safe(P x, thread float* x_thread, int N) {
  return BITS == 8 ? hcd_load8_safe<P, VPT>(x, x_thread, N)
                   : hcd_load_safe<P, VPT>(x, x_thread, N);
}

template <int BITS>
inline float hcd_qdotq(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum,
    int N) {
  return BITS == 8 ? hcd_qdot8(w, x_thread, scale, bias, sum, N)
                   : hcd_qdot(w, x_thread, scale, bias, sum, N);
}

// qmv_wide (affine, k_lanes = 8) for 4- and 8-bit group-64 rows: the 8-lane
// group of one output row strides over its groups, decodes each 8-value
// sub-chunk as dequantize() does, dots it from 0, adds the sub-chunk sums in
// order, then the shuffle-down ladder 4, 2, 1.  All 32 lanes must call it.
template <typename T, int BITS, typename P>
inline float hcd_wide_row(
    P x,
    const device uint8_t* wrow,
    const device T* srow,
    const device T* brow,
    int groups,
    int k_lane) {
  float result = 0;
  for (int g = k_lane; g < groups; g += 8) {
    float scale = srow[g];
    float bias = brow[g];
    for (int sc = 0; sc < 8; sc++) {
      const int k0 = g * 64 + sc * 8;
      const device uint8_t* wc = wrow + k0 * BITS / 8;
      float w_dq[8];
      const float s = float(scale);
      const float b = float(bias);
      if (BITS == 8) {
        for (int i = 0; i < 8; i++) {
          w_dq[i] = static_cast<float>(s * wc[i] + b);
        }
      } else {
        float scv[2] = {s, s / 16.0f};
        for (int i = 0; i < 4; i++) {
          w_dq[2 * i] = static_cast<float>(scv[0] * (wc[i] & 0x0f) + b);
          w_dq[2 * i + 1] = static_cast<float>(scv[1] * (wc[i] & 0xf0) + b);
        }
      }
      float acc = 0;
      for (int i = 0; i < 8; i++) {
        acc += static_cast<float>(x[k0 + i]) * w_dq[i];
      }
      result += acc;
    }
  }
  result += simd_shuffle_down(result, 4);
  result += simd_shuffle_down(result, 2);
  result += simd_shuffle_down(result, 1);
  return result;
}

// MLX Sigmoid, evaluated in T.  Two variants because the eager path runs it
// from two differently compiled libraries: the compiled ``nn.silu`` is JIT
// source (``metal::exp`` resolves as in this kernel), while ``mx.sigmoid`` is
// the prebuilt unary kernel (built with -fno-fast-math, so its ``metal::exp``
// is the precise one).  Over all 65,536 bf16 inputs each variant equals its
// eager counterpart and differs from the other one at one input (GPU,
// 2026-09-30; scripts/check_qwen4_hc_decode.py re-checks the table).
template <typename U>
inline U hcd_sigmoid_jit(U x) {
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

template <typename U>
inline U hcd_sigmoid_unary(U x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
"""

# One threadgroup of NT = H / 4 threads per (row group gy, row z).  NSG = NT/32.
NORM_DOWN_SOURCE = r"""
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint gy = threadgroup_position_in_grid.y;
    const uint row = threadgroup_position_in_grid.z;
    constexpr int K = HC * H;
    constexpr int NSG = H / 4 / 32;

    threadgroup T xs[K];
    threadgroup float local_sums[HC][32];
    threadgroup float local_inv[HC];
    threadgroup float dense_part[INJ == 2 ? 8 * HC : 1];

    // rms_single_row on each stream (the eager path's float32 rows of H).
    const device T* xr = x + size_t(row) * K;
    float v[HC][4];
    float part[HC];
    for (int s = 0; s < HC; ++s) {
      float acc = 0;
      const device T* xp = xr + s * H + t * 4;
      for (int i = 0; i < 4; i++) {
        v[s][i] = xp[i];
        acc += v[s][i] * v[s][i];
      }
      part[s] = simd_sum(acc);
    }
    if (sg == 0) {
      for (int s = 0; s < HC; ++s) {
        local_sums[s][lane] = 0;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
      for (int s = 0; s < HC; ++s) {
        local_sums[s][sg] = part[s];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      for (int s = 0; s < HC; ++s) {
        float acc = simd_sum(local_sums[s][lane]);
        if (lane == 0) {
          local_inv[s] = metal::precise::rsqrt(acc / axis[0] + eps[0]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // rms output (weight 1), times the float32 norm weight, rounded to T.
    for (int s = 0; s < HC; ++s) {
      const float inv = local_inv[s];
      for (int i = 0; i < 4; i++) {
        const int k = s * H + int(t) * 4 + i;
        const float n = v[s][i] * inv;
        const float m = n * float(nw[k]);
        const T y = static_cast<T>(m);
        xs[k] = y;
        if (gy == 0) {
          xn[size_t(row) * K + k] = y;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down rows use format DB, inject rows IB (4 or 8 bits, group 64).
    constexpr int IN_W = K * DB / 8;   // bytes per down row
    constexpr int INJ_W = K * IB / 8;  // bytes per inject row
    constexpr int IN_G = K / 64;       // groups per row
    // qmv_fast lane layout: VPT values per lane per 32 * VPT block, 8 weight
    // bytes per lane either way.
    constexpr int D_VPT = DB == 8 ? 8 : 16;
    constexpr int I_VPT = IB == 8 ? 8 : 16;
    float x_thread[16];
    if (LAW == 1) {
      // Rows 2..8: MLX runs qmv_wide; one 8-lane group per output row.
      const int k_lane = int(lane) % 8;
      const int sg_row = int(lane) / 8;
      if (int(gy) < ND) {
        const int out_row = int(gy) * (NSG * 4) + int(sg) * 4 + sg_row;
        const float r = hcd_wide_row<T, DB>(
            (const threadgroup T*)xs,
            (const device uint8_t*)down_w + out_row * IN_W,
            down_s + out_row * IN_G, down_b + out_row * IN_G, IN_G, k_lane);
        if (k_lane == 0) {
          const T d = static_cast<T>(r);
          const T a = d / T(HC);
          const T g = hcd_sigmoid_jit<T>(a);
          act[size_t(row) * R + out_row] = a * g;
        }
      } else if (INJ == 2) {
        // Dense bf16 inject, MLX gemv_wide (k_lanes 32: one row per
        // simdgroup, 4 rows per threadgroup; unroll 8 float4 blocks per lane
        // per step, a strided float4 tail, the halving shuffle ladder).
        if (int(sg) < HC) {
          const int out_row = int(sg);
          const device vec<T, 4>* w4 =
              (const device vec<T, 4>*)((const device T*)inj_w + out_row * K);
          constexpr int N_V4 = K / 4;
          constexpr int N_MAIN = N_V4 - N_V4 % (32 * 8);
          float result = 0;
          for (int base = 0; base < N_MAIN; base += 32 * 8) {
            float4 wf[8];
            for (int i = 0; i < 8; i++) {
              wf[i] = float4(w4[base + i * 32 + int(lane)]);
            }
            float acc = 0;
            for (int i = 0; i < 8; i++) {
              const int j = 4 * (base + i * 32 + int(lane));
              acc += dot(wf[i], float4(xs[j], xs[j + 1], xs[j + 2], xs[j + 3]));
            }
            result += acc;
          }
          for (int idx = N_MAIN + int(lane); idx < N_V4; idx += 32) {
            const int j = 4 * idx;
            result += dot(float4(w4[idx]), float4(xs[j], xs[j + 1], xs[j + 2], xs[j + 3]));
          }
          for (ushort off = 16; off >= 1; off >>= 1) {
            result += simd_shuffle_down(result, off);
          }
          if (lane == 0) {
            const T raw = static_cast<T>(result);
            const T q = raw / T(HC);
            const T g = hcd_sigmoid_unary<T>(q);
            inj[size_t(row) * HC + out_row] = T(2) * g;
          }
        }
      } else if (INJ != 0 && sg == 0) {
        const int out_row = sg_row;
        const float r = hcd_wide_row<T, IB>(
            (const threadgroup T*)xs,
            (const device uint8_t*)inj_w + out_row * INJ_W,
            inj_s + out_row * IN_G, inj_b + out_row * IN_G, IN_G, k_lane);
        if (k_lane == 0) {
          const T raw = static_cast<T>(r);
          const T q = raw / T(HC);
          const T g = hcd_sigmoid_unary<T>(q);
          inj[size_t(row) * HC + out_row] = T(2) * g;
        }
      }
    } else if (int(gy) < ND) {
      const int out_row = int(gy) * (NSG * RPS) + int(sg) * RPS;
      const device uint8_t* ws =
          (const device uint8_t*)down_w + out_row * IN_W + int(lane) * 8;
      const device T* sc = down_s + out_row * IN_G + int(lane) / (64 / D_VPT);
      const device T* bs = down_b + out_row * IN_G + int(lane) / (64 / D_VPT);
      const threadgroup T* xp = xs + int(lane) * D_VPT;
      float result[RPS] = {0};
      for (int k = 0; k < K; k += 32 * D_VPT) {
        float sum = hcd_loadq<DB, const threadgroup T*, D_VPT>(xp, x_thread);
        for (int r = 0; r < RPS; r++) {
          float s = sc[r * IN_G];
          float b = bs[r * IN_G];
          result[r] += hcd_qdotq<DB>(ws + r * IN_W, x_thread, s, b, sum, D_VPT);
        }
        ws += 256;
        sc += 32 * D_VPT / 64;
        bs += 32 * D_VPT / 64;
        xp += 32 * D_VPT;
      }
      for (int r = 0; r < RPS; r++) {
        result[r] = simd_sum(result[r]);
      }
      if (lane == 0) {
        const T hc = T(HC);
        for (int r = 0; r < RPS; r++) {
          const T d = static_cast<T>(result[r]);
          const T a = d / hc;
          const T g = hcd_sigmoid_jit<T>(a);
          act[size_t(row) * R + out_row + r] = a * g;
        }
      }
    } else if (INJ == 2) {
      // Dense bf16 inject, MLX gemv (BM 1, BN 8, SM 1, SN 32, TM HC, TN 4):
      // simdgroup g, lane l owns columns (32 g + l) * 4 + 1024 i; per block
      // each of the HC rows adds its four products in order, then the
      // shuffle-down ladder 16..1 and the threadgroup sum of simdgroups 1..7
      // onto simdgroup 0 in order.
      float result[HC];
      for (int tm = 0; tm < HC; tm++) {
        result[tm] = 0;
      }
      if (int(sg) < 8) {
        const device T* wd = (const device T*)inj_w;
        int bn = (int(sg) * 32 + int(lane)) * 4;
        for (int i = 0; i < K / 1024; ++i) {
          float v[4];
          for (int tn = 0; tn < 4; tn++) {
            v[tn] = static_cast<float>(xs[bn + tn]);
          }
          int mat_offset = 0;
          for (int tm = 0; tm < HC; tm++) {
            T inter[4];
            for (int tn = 0; tn < 4; tn++) {
              inter[tn] = wd[mat_offset + bn + tn];
            }
            for (int tn = 0; tn < 4; tn++) {
              result[tm] += inter[tn] * v[tn];
            }
            mat_offset += K;
          }
          bn += 1024;
        }
        for (int tm = 0; tm < HC; tm++) {
          for (ushort sn = 16; sn >= 1; sn >>= 1) {
            result[tm] += simd_shuffle_down(result[tm], sn);
          }
        }
        if (lane == 0) {
          for (int tm = 0; tm < HC; tm++) {
            dense_part[int(sg) * HC + tm] = result[tm];
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0 && lane == 0) {
        for (int tm = 0; tm < HC; tm++) {
          float r = result[tm];
          for (int sgn = 1; sgn < 8; sgn++) {
            r += dense_part[sgn * HC + tm];
          }
          const T raw = static_cast<T>(r);
          const T q = raw / T(HC);
          const T g = hcd_sigmoid_unary<T>(q);
          inj[size_t(row) * HC + tm] = T(2) * g;
        }
      }
    } else if (INJ != 0 && int(sg) < HC) {
      const int out_row = int(sg);
      const device uint8_t* ws =
          (const device uint8_t*)inj_w + out_row * INJ_W + int(lane) * 8;
      const device T* sc = inj_s + out_row * IN_G + int(lane) / (64 / I_VPT);
      const device T* bs = inj_b + out_row * IN_G + int(lane) / (64 / I_VPT);
      const threadgroup T* xp = xs + int(lane) * I_VPT;
      float result = 0;
      for (int k = 0; k < K; k += 32 * I_VPT) {
        float sum = hcd_loadq<IB, const threadgroup T*, I_VPT>(xp, x_thread);
        float s = sc[0];
        float b = bs[0];
        result += hcd_qdotq<IB>(ws, x_thread, s, b, sum, I_VPT);
        ws += 256;
        sc += 32 * I_VPT / 64;
        bs += 32 * I_VPT / 64;
        xp += 32 * I_VPT;
      }
      result = simd_sum(result);
      if (lane == 0) {
        const T raw = static_cast<T>(result);
        const T q = raw / T(HC);
        const T g = hcd_sigmoid_unary<T>(q);
        inj[size_t(row) * HC + out_row] = T(2) * g;
      }
    }
"""

UP_MIX_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int h = int(threadgroup_position_in_grid.y) * NSGB + int(sg);
    const uint row = threadgroup_position_in_grid.z;
    constexpr int K = HC * H;
    // Up rows use format UB (4 or 8 bits, group 64); MLX runs plain qmv
    // (R is not a multiple of the qmv_fast block): VPT values per lane per
    // 32 * VPT block, 4 weight bytes per lane either way, guarded tail.
    constexpr int IN_W = R * UB / 8;
    constexpr int IN_G = R / 64;
    constexpr int U_VPT = UB == 8 ? 4 : 8;
    constexpr int U_BLK = 32 * U_VPT;
    const device T* ar = act + size_t(row) * R;
    float x_thread[8];
    float u[HC];
    if (LAW == 1) {
      // qmv_wide: lane group s (8 lanes) runs up row s * H + h.
      const int s = int(lane) / 8;
      const int n = s * H + h;
      const float r = hcd_wide_row<T, UB>(
          ar, (const device uint8_t*)up_w + n * IN_W, up_s + n * IN_G,
          up_b + n * IN_G, IN_G, int(lane) % 8);
      for (int j = 0; j < HC; ++j) {
        u[j] = simd_shuffle(r, ushort(j * 8));
      }
    } else for (int s = 0; s < HC; ++s) {
      const int n = s * H + h;
      const device uint8_t* ws = (const device uint8_t*)up_w + n * IN_W + int(lane) * 4;
      const device T* sc = up_s + n * IN_G + int(lane) / (64 / U_VPT);
      const device T* bs = up_b + n * IN_G + int(lane) / (64 / U_VPT);
      const device T* xp = ar + int(lane) * U_VPT;
      float result = 0;
      int k = 0;
      for (; k < R - U_BLK; k += U_BLK) {
        float sum = hcd_loadq<UB, const device T*, U_VPT>(xp, x_thread);
        float sv = sc[0];
        float bv = bs[0];
        result += hcd_qdotq<UB>(ws, x_thread, sv, bv, sum, U_VPT);
        ws += 128;
        sc += U_BLK / 64;
        bs += U_BLK / 64;
        xp += U_BLK;
      }
      const int remaining = clamp(int(R - k - int(lane) * U_VPT), 0, U_VPT);
      if (remaining > 0) {
        float sum = hcd_loadq_safe<UB, const device T*, U_VPT>(xp, x_thread, remaining);
        float sv = sc[0];
        float bv = bs[0];
        result += hcd_qdotq<UB>(ws, x_thread, sv, bv, sum, remaining);
      }
      u[s] = simd_sum(result);
    }
    if (lane == 0) {
      const device T* xr = xn + size_t(row) * K;
      T total = T(0);
      for (int s = 0; s < HC; ++s) {
        const T g = hcd_sigmoid_unary<T>(static_cast<T>(u[s]));
        const T p = g * xr[s * H + h];
        const T ps = p + T(0);
        total = (s == 0) ? ps : T(ps + total);
      }
      mixed[size_t(row) * H + h] = total * T(0.25f);
    }
"""

# Served-graph gates (omlx #4122 follow-up; served_exp): each helper above
# runs only while its spelling still reproduces its eager counterpart on this
# build (bf16, the only admitted dtype).
SILU_GATE = ServedExpGate(
    "hc_silu",
    served_name="compiled nn.silu",
    served=nn.silu,
    header=metal_helper(HEADER, "hcd_sigmoid_jit"),
    body="    y = x * hcd_sigmoid_jit<T>(x);",
)
SIGMOID_GATE = ServedExpGate(
    "hc_sigmoid",
    served_name="eager sigmoid",
    served=mx.sigmoid,
    header=metal_helper(HEADER, "hcd_sigmoid_unary"),
    body="    y = hcd_sigmoid_unary<T>(x);",
    kernel_exp="metal::precise::exp",
)


def served_exp_refusal() -> str | None:
    """Why the HC kernels' SiLU or sigmoid may not run on this build, or None."""
    return SILU_GATE.refusal(mx.bfloat16) or SIGMOID_GATE.refusal(mx.bfloat16)


_KERNELS: dict = {}


def _kernel(name: str, inputs, outputs, source: str):
    kernel = _KERNELS.get(name)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=inputs,
            output_names=outputs,
            header=HEADER,
            source=source,
            ensure_row_contiguous=True,
        )
        _KERNELS[name] = kernel
    return kernel


_SCALARS: dict = {}


def _scalars(hidden: int, eps: float):
    key = (hidden, float(eps))
    pair = _SCALARS.get(key)
    if pair is None:
        pair = (mx.array([hidden], dtype=mx.uint32), mx.array([eps], dtype=mx.float32))
        mx.eval(*pair)
        _SCALARS[key] = pair
    return pair


def _build_plan(module, law: int = 0) -> dict:
    """Everything a launch needs except the input rows, built once per layout
    and projection law (0: one row, MLX qmv/qmv_fast; 1: 2..8 rows, qmv_wide)."""
    hidden = module.hidden_size
    width = HC_COUNT * hidden
    down = module["input_mix_weight_down"]
    up = module["input_mix_weight_up"]
    lowrank = down["weight"].shape[0]
    inject_mode = _inject_mode(module)
    has_inject = inject_mode != "none"
    dense = inject_mode == "dense"
    inject = module["block_inject_weight"] if has_inject else down
    # A dense inject binds its bf16 weight; the scales/biases slots take the
    # down arrays as unused placeholders.
    inject_arrays = (
        (inject["weight"], down["scales"], down["biases"])
        if dense
        else (inject["weight"], inject["scales"], inject["biases"])
    )
    threads = hidden // 4
    down_groups = lowrank // (threads // 32 * (4 if law else DOWN_ROWS))
    axis, eps = _scalars(hidden, module.hc_norm.eps)
    dtype = mx.bfloat16
    plan = {
        "hidden": hidden,
        "has_inject": has_inject,
        "inject_mode": inject_mode,
        "all_4bit": (
            down.bits == 4 and up.bits == 4
            and (inject_mode == "none" or (inject_mode == "quantized" and inject.bits == 4))
        ),
        "format": (
            f"down b{down.bits} up b{up.bits} inject "
            + (f"b{inject.bits}" if inject_mode == "quantized" else inject_mode)
        ),
        "a_kernel": _kernel(
            "mlx2_qwen4_hc_decode_norm_down",
            ["x", "nw", "axis", "eps", "down_w", "down_s", "down_b", "inj_w", "inj_s", "inj_b"],
            ["xn", "act", "inj"],
            NORM_DOWN_SOURCE,
        ),
        "a_inputs": [
            module.hc_norm.weight, axis, eps,
            down["weight"], down["scales"], down["biases"],
            *inject_arrays,
        ],
        "a_template": [
            ("T", dtype),
            ("H", hidden),
            ("HC", HC_COUNT),
            ("R", lowrank),
            ("RPS", DOWN_ROWS),
            ("ND", down_groups),
            # 0: none, 1: quantized (qmv_fast / qmv_wide), 2: dense bf16
            # (gemv / gemv_wide).
            ("INJ", 2 if dense else int(has_inject)),
            ("LAW", law),
            ("DB", int(down.bits)),
            ("IB", 4 if dense else int(inject.bits)),
        ],
        "a_groups": (threads, down_groups + int(has_inject)),
        "b_kernel": _kernel(
            "mlx2_qwen4_hc_decode_up_mix",
            ["xn", "act", "up_w", "up_s", "up_b"],
            ["mixed"],
            UP_MIX_SOURCE,
        ),
        "b_weights": [up["weight"], up["scales"], up["biases"]],
        "b_template": [
            ("T", dtype),
            ("H", hidden),
            ("HC", HC_COUNT),
            ("R", lowrank),
            ("NSGB", UP_SIMDGROUPS),
            ("LAW", law),
            ("UB", int(up.bits)),
        ],
        "width": width,
        "lowrank": lowrank,
        "by_rows": {},
    }
    return plan


def _launches(plan, rows: int):
    """The two launch descriptions for ``rows`` input rows."""
    threads, groups = plan["a_groups"]
    dtype = mx.bfloat16
    a_launch = dict(
        template=plan["a_template"],
        grid=(threads, groups, rows),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, plan["width"]), (rows, plan["lowrank"]), (rows, HC_COUNT)],
        output_dtypes=[dtype, dtype, dtype],
    )
    b_launch = dict(
        template=plan["b_template"],
        grid=(32 * UP_SIMDGROUPS, plan["hidden"] // UP_SIMDGROUPS, rows),
        threadgroup=(32 * UP_SIMDGROUPS, 1, 1),
        output_shapes=[(rows, plan["hidden"])],
        output_dtypes=[dtype],
    )
    return a_launch, b_launch


_COMPILED: dict = {}


def _compiled_pair(plan, rows: int):
    """Both launches as one ``mx.compile`` function per launch geometry.

    The weights are inputs, so every layer with this geometry shares it; the
    replay costs ~2.3 us of host time against ~5.7 us for two direct
    ``metal_kernel`` calls (M5 Max), and records the same two dispatches."""
    fn = plan["by_rows"].get(rows)
    if fn is not None:
        return fn
    key = (tuple(plan["a_template"]), tuple(plan["b_template"]), plan["a_groups"], rows)
    fn = _COMPILED.get(key)
    if fn is None:
        a_kernel, b_kernel = plan["a_kernel"], plan["b_kernel"]
        a_launch, b_launch = _launches(plan, rows)

        def pair(flat, nw, axis, eps, dw, ds, db, iw, is_, ib, uw, us, ub):
            xn, act, inj = a_kernel(
                inputs=[flat, nw, axis, eps, dw, ds, db, iw, is_, ib], **a_launch
            )
            mixed = b_kernel(inputs=[xn, act, uw, us, ub], **b_launch)[0]
            return mixed, inj

        fn = mx.compile(pair)
        _COMPILED[key] = fn
    plan["by_rows"][rows] = fn
    return fn


def projection_law(rows: int) -> int | None:
    """Which MLX matvec the composed path runs for ``rows`` folded rows.

    0: one row, ``qmv_fast``/``qmv``.  1: 2..8 rows on an Apple GPU of family
    15 or newer, where ``dispatch_qmv`` takes ``qmv_wide`` (below the
    ``qmv_nax`` and ``qmm`` crossovers for these shapes and below the fast
    RMS norm's width cap).  None: not transcribed."""
    if rows == 1:
        return 0
    if 2 <= rows <= MAX_WIDE_ROWS and _gpu_family() >= 15 and "MLX_QMV_LIMIT" not in os.environ:
        return 1
    return None


_GPU_FAMILY: int | None = None


def _gpu_family() -> int:
    global _GPU_FAMILY
    if _GPU_FAMILY is None:
        try:
            info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
            arch = str(info.get("architecture", ""))
            match = re.search(r"g(\d+)", arch)
            _GPU_FAMILY = int(match.group(1)) if match else 0
        except Exception:  # noqa: BLE001 - unknown device: one-row law only
            _GPU_FAMILY = 0
    return _GPU_FAMILY


def hc_decode_launch(module, flat, *, debug: bool = False, plan=None):
    """The two launches for ``flat`` [rows, 4 * hidden]; returns (mixed, inject|None).

    ``debug`` also returns the normed rows and the SiLU activations, for the
    Metal check that localises a mismatch."""
    rows = flat.shape[0]
    if plan is None:
        law = projection_law(rows)
        reason, plans = _cached_plan(module)
        if plans is None or law is None:
            raise ValueError(f"not admitted: {reason or 'rows'}")
        plan = _law_plan(module, plans, law)
    if not debug:
        mixed, inj = _compiled_pair(plan, rows)(flat, *plan["a_inputs"], *plan["b_weights"])
        return mixed, (inj if plan["has_inject"] else None)
    a_launch, b_launch = _launches(plan, rows)
    xn, act, inj = plan["a_kernel"](inputs=[flat, *plan["a_inputs"]], **a_launch)
    mixed = plan["b_kernel"](inputs=[xn, act, *plan["b_weights"]], **b_launch)[0]
    return mixed, (inj if plan["has_inject"] else None), xn, act


def try_hc_decode(module, hyper_input, *, eager_norm: bool, compile_glue: bool):
    """Serve one GatedResidual call in two launches, or return None (counted).

    ``eager_norm`` says the caller's composed norm is the eager fast-RMS path
    this module transcribes; ``compile_glue`` says the caller would run its
    compiled glue spans (not transcribed, so declined).
    """
    global _BROKEN, _LAST_ERROR
    if not _ENABLED:
        return None
    if _BROKEN:
        _decline("demoted after an error")
        return None
    if not isinstance(hyper_input, mx.array) or hyper_input.ndim != 3:
        _decline("input rank")
        return None
    if hyper_input.dtype != mx.bfloat16:
        _decline("input dtype")
        return None
    rows = hyper_input.shape[0] * hyper_input.shape[1]
    window = _row_exact_window() if rows > 1 else None
    if window is not None:
        # A row-exact verify window: every row must carry the bits of its
        # one-token decode step, which the composed path (row-exact
        # projections) gives it; law 0 is that arithmetic per grid row.
        # (Never law 1 here: qmv_wide's multi-row arithmetic is not the
        # one-token step's.)
        if not _ROW_EXACT:
            _decline("row-exact window mode off")
            window.note("hc", "composed", 1)
            return None
        if rows > ROW_EXACT_MAX_ROWS:
            _decline("row-exact window wider than ROW_EXACT_MAX_ROWS")
            window.note("hc", "composed", 1)
            return None
        law = 0
    elif rows > 1 and _MULTI_ROW == "off":
        _decline(MULTI_ROW_DECLINE)
        return None
    else:
        law = projection_law(rows)
    if law is None:
        _decline("rows (MLX runs a matvec this module does not transcribe)")
        return None
    fused = _try_launch(module, hyper_input, rows, law, eager_norm, compile_glue)
    if window is not None:
        window.note("hc", "composed" if fused is None else "kernel_one_row_law", 1)
        if fused is not None:
            _STATS["row_exact_calls"] += 1
    return fused


def _row_exact_window():
    from mlx2.runtime import row_exact_verify

    return row_exact_verify.current()


def _try_launch(module, hyper_input, rows: int, law: int, eager_norm: bool, compile_glue: bool):
    global _BROKEN, _LAST_ERROR
    if getattr(module, "training", False):
        _decline("training")
        return None
    if not eager_norm:
        _decline("norm is not the eager fast-RMS path")
        return None
    if compile_glue:
        _decline("compiled glue enabled")
        return None
    reason, plans = _cached_plan(module)
    if reason is not None:
        _decline(reason)
        return None
    if hyper_input.shape[2] != HC_COUNT * module.hidden_size:
        _decline("input width")
        return None
    if not runtime_supported():
        _decline("Metal runtime unavailable")
        return None
    refusal = served_exp_refusal()
    if refusal is not None:
        _decline(refusal)
        return None
    try:
        plan = _law_plan(module, plans, law)
        if law == 1 and _MULTI_ROW == "auto" and not plan["all_4bit"]:
            _decline(MULTI_ROW_FORMAT_DECLINE)
            return None
        flat = hyper_input.reshape(rows, -1)
        mixed, inject = hc_decode_launch(module, flat, plan=plan)
        signature = (
            plan["hidden"],
            plan["lowrank"],
            module.hc_norm.weight.dtype,
            inject is not None,
            plan["format"],
            law,
            rows,
        )
        if signature not in _VALIDATED:
            # Metal compilation is lazy: evaluate the first call of each
            # specialization here so a compile failure demotes the path
            # instead of failing a later model eval.
            mx.eval(mixed) if inject is None else mx.eval(mixed, inject)
            _VALIDATED.add(signature)
    except Exception as exc:  # noqa: BLE001 - optional native path
        with _LOCK:
            _BROKEN = True
            _STATS["errors"] += 1
            _LAST_ERROR = f"{type(exc).__name__}: {exc}"[:400]
        _decline("demoted after an error")
        return None
    lead = hyper_input.shape[:-1]
    mixed = mixed.reshape(*lead, plan["hidden"])
    # Hot path: the model runs on one thread; no lock for the two increments.
    _STATS["calls"] += 1
    _FORMATS[plan["format"]] += 1
    if inject is not None:
        _STATS["inject_calls"] += 1
    if inject is None:
        return mixed
    return (mixed, hyper_input, inject.reshape(*lead, HC_COUNT))


def reset_for_tests() -> None:
    """Clear the demotion flag, the validated set and the counters."""
    global _BROKEN
    _BROKEN = False
    _VALIDATED.clear()
    hc_decode_status(reset=True)


__all__ = [
    "HC_DECODE_ENV",
    "PENDING_WRITE_SEAM",
    "hc_decode_enabled",
    "hc_decode_launch",
    "hc_decode_status",
    "set_hc_decode_enabled",
    "static_admission",
    "try_hc_decode",
]
