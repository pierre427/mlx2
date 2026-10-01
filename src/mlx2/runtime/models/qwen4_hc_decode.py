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

Admission is structural and exact-shape: one row (verify rows are refused and
counted, because MLX runs ``qmv_wide`` there, a different arithmetic), bf16
activations, four streams, affine 4-bit group-64 projections with bf16
scales/biases that are exactly ``nn.QuantizedLinear``, and the eager norm
profile (fast RMS norm on, fused group norm and compiled glue off).  Anything
else keeps the composed body.

Default off.  ``MLX_QWEN4_HC_DECODE=1`` (the Flash-Next policy field
``hc_decode_kernels``) enables it at import; ``set_hc_decode_enabled`` switches
live; ``MLX_QWEN4_HC_DECODE=0`` or the setter is the kill switch.
"""

from __future__ import annotations

import os
import threading
from collections import Counter
from itertools import chain
from operator import is_

import mlx.core as mx
from mlx import nn

HC_DECODE_ENV = "MLX_QWEN4_HC_DECODE"
HC_COUNT = 4
BITS = 4
GROUP_SIZE = 64
# Down rows per simdgroup in norm_down; one row keeps the most threadgroups.
DOWN_ROWS = 1
# Simdgroups (hidden columns) per up_mix threadgroup.
UP_SIMDGROUPS = 8

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
_LOCK = threading.Lock()
_STATS: Counter = Counter()
_DECLINES: Counter = Counter()
_LAST_DECLINE: str | None = None
_LAST_ERROR: str | None = None
_BROKEN = False
_VALIDATED: set = set()


def hc_decode_enabled() -> bool:
    return _ENABLED


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
            "launches": int(2 * _STATS["calls"]),
            "declines": dict(_DECLINES),
            "errors": int(_STATS["errors"]),
            "last_decline": _LAST_DECLINE,
            "last_error": _LAST_ERROR,
        }
        if reset:
            _STATS.clear()
            _DECLINES.clear()
            _LAST_DECLINE = None
            _LAST_ERROR = None
    return report


def _decline(reason: str) -> None:
    global _LAST_DECLINE
    with _LOCK:
        _DECLINES[reason] += 1
        _LAST_DECLINE = reason


def _quantized_reason(layer) -> str | None:
    if type(layer) is not nn.QuantizedLinear:
        return "projection is not a plain QuantizedLinear"
    if "_lane_prepared" in layer.__dict__:
        return "projection owned by the lane matmul"
    if getattr(layer, "bits", None) != BITS:
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


def static_admission(module) -> str | None:
    """Model-layout check (cached per module by identity of its tensors).

    Mirrors the conditions under which the composed path runs the kernels this
    module transcribes: ``qmv_fast`` for the down and inject rows (input width
    a multiple of 512), plain ``qmv`` for the up rows (input width not a
    multiple of 512, not 64/128), ``rms_single_row`` with whole four-element
    reads (hidden a multiple of 128, at most 4096).
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
    if tuple(down["weight"].shape) != (lowrank, width * BITS // 32):
        return "down shape"
    if tuple(up["weight"].shape) != (width, lowrank * BITS // 32):
        return "up shape"
    if width % 512:
        return "width % 512 (MLX would not run qmv_fast)"
    if lowrank % 512 == 0 or lowrank in (64, 128) or lowrank % GROUP_SIZE:
        return "lowrank (MLX would not run plain qmv for the up rows)"
    simdgroups = hidden // 4 // 32
    if lowrank % (simdgroups * DOWN_ROWS):
        return "lowrank row tiling"
    if hidden % UP_SIMDGROUPS:
        return "hidden column tiling"
    if "block_inject_weight" in module:
        inject = module["block_inject_weight"]
        reason = _quantized_reason(inject)
        if reason is not None:
            return f"inject: {reason}"
        if tuple(inject["weight"].shape) != (hc, width * BITS // 32):
            return "inject shape"
    return None


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
    plan = None if reason is not None else _build_plan(module)
    module.__dict__["_hc_decode_plan"] = (refs, reason, plan)
    return reason, plan


def _cached_static_admission(module) -> str | None:
    return _cached_plan(module)[0]


def runtime_supported() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


# --------------------------------------------------------------------------
# Metal.  4-bit, group-64 transcriptions of the MLX fork's quantized.h helpers
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

    constexpr int IN_W = K / 2;   // bytes per 4-bit row
    constexpr int IN_G = K / 64;  // groups per row
    float x_thread[16];
    if (int(gy) < ND) {
      const int out_row = int(gy) * (NSG * RPS) + int(sg) * RPS;
      const device uint8_t* ws =
          (const device uint8_t*)down_w + out_row * IN_W + int(lane) * 8;
      const device T* sc = down_s + out_row * IN_G + int(lane) / 4;
      const device T* bs = down_b + out_row * IN_G + int(lane) / 4;
      const threadgroup T* xp = xs + int(lane) * 16;
      float result[RPS] = {0};
      for (int k = 0; k < K; k += 512) {
        float sum = hcd_load<const threadgroup T*, 16>(xp, x_thread);
        for (int r = 0; r < RPS; r++) {
          float s = sc[r * IN_G];
          float b = bs[r * IN_G];
          result[r] += hcd_qdot(ws + r * IN_W, x_thread, s, b, sum, 16);
        }
        ws += 256;
        sc += 8;
        bs += 8;
        xp += 512;
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
    } else if (INJ != 0 && int(sg) < HC) {
      const int out_row = int(sg);
      const device uint8_t* ws =
          (const device uint8_t*)inj_w + out_row * IN_W + int(lane) * 8;
      const device T* sc = inj_s + out_row * IN_G + int(lane) / 4;
      const device T* bs = inj_b + out_row * IN_G + int(lane) / 4;
      const threadgroup T* xp = xs + int(lane) * 16;
      float result = 0;
      for (int k = 0; k < K; k += 512) {
        float sum = hcd_load<const threadgroup T*, 16>(xp, x_thread);
        float s = sc[0];
        float b = bs[0];
        result += hcd_qdot(ws, x_thread, s, b, sum, 16);
        ws += 256;
        sc += 8;
        bs += 8;
        xp += 512;
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
    constexpr int IN_W = R / 2;
    constexpr int IN_G = R / 64;
    const device T* ar = act + size_t(row) * R;
    float x_thread[8];
    float u[HC];
    for (int s = 0; s < HC; ++s) {
      const int n = s * H + h;
      const device uint8_t* ws = (const device uint8_t*)up_w + n * IN_W + int(lane) * 4;
      const device T* sc = up_s + n * IN_G + int(lane) / 8;
      const device T* bs = up_b + n * IN_G + int(lane) / 8;
      const device T* xp = ar + int(lane) * 8;
      float result = 0;
      int k = 0;
      for (; k < R - 256; k += 256) {
        float sum = hcd_load<const device T*, 8>(xp, x_thread);
        float sv = sc[0];
        float bv = bs[0];
        result += hcd_qdot(ws, x_thread, sv, bv, sum, 8);
        ws += 128;
        sc += 4;
        bs += 4;
        xp += 256;
      }
      const int remaining = clamp(int(R - k - int(lane) * 8), 0, 8);
      if (remaining > 0) {
        float sum = hcd_load_safe<const device T*, 8>(xp, x_thread, remaining);
        float sv = sc[0];
        float bv = bs[0];
        result += hcd_qdot(ws, x_thread, sv, bv, sum, remaining);
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


def _build_plan(module) -> dict:
    """Everything a launch needs except the input row, built once per layout."""
    hidden = module.hidden_size
    width = HC_COUNT * hidden
    down = module["input_mix_weight_down"]
    up = module["input_mix_weight_up"]
    lowrank = down["weight"].shape[0]
    has_inject = "block_inject_weight" in module
    inject = module["block_inject_weight"] if has_inject else down
    threads = hidden // 4
    down_groups = lowrank // (threads // 32 * DOWN_ROWS)
    axis, eps = _scalars(hidden, module.hc_norm.eps)
    dtype = mx.bfloat16
    plan = {
        "hidden": hidden,
        "has_inject": has_inject,
        "a_kernel": _kernel(
            "mlx2_qwen4_hc_decode_norm_down",
            ["x", "nw", "axis", "eps", "down_w", "down_s", "down_b", "inj_w", "inj_s", "inj_b"],
            ["xn", "act", "inj"],
            NORM_DOWN_SOURCE,
        ),
        "a_inputs": [
            module.hc_norm.weight, axis, eps,
            down["weight"], down["scales"], down["biases"],
            inject["weight"], inject["scales"], inject["biases"],
        ],
        "a_template": [
            ("T", dtype),
            ("H", hidden),
            ("HC", HC_COUNT),
            ("R", lowrank),
            ("RPS", DOWN_ROWS),
            ("ND", down_groups),
            ("INJ", int(has_inject)),
        ],
        "a_grid": (threads, down_groups + int(has_inject), 1),
        "a_threadgroup": (threads, 1, 1),
        "a_shapes": [(1, width), (1, lowrank), (1, HC_COUNT)],
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
        ],
        "b_grid": (32 * UP_SIMDGROUPS, hidden // UP_SIMDGROUPS, 1),
        "b_threadgroup": (32 * UP_SIMDGROUPS, 1, 1),
        "b_shapes": [(1, hidden)],
        "dtypes3": [dtype, dtype, dtype],
        "dtypes1": [dtype],
    }
    plan["compiled"] = _compiled_pair(plan)
    return plan


_COMPILED: dict = {}


def _compiled_pair(plan):
    """Both launches as one ``mx.compile`` function per launch geometry.

    The weights are inputs, so every layer with this geometry shares it; the
    replay costs ~2.3 us of host time against ~5.7 us for two direct
    ``metal_kernel`` calls (M5 Max), and records the same two dispatches."""
    key = (tuple(plan["a_template"]), tuple(plan["b_template"]), plan["a_grid"], plan["b_grid"])
    fn = _COMPILED.get(key)
    if fn is None:
        a_kernel, b_kernel = plan["a_kernel"], plan["b_kernel"]
        a_launch = dict(
            template=plan["a_template"], grid=plan["a_grid"],
            threadgroup=plan["a_threadgroup"], output_shapes=plan["a_shapes"],
            output_dtypes=plan["dtypes3"],
        )
        b_launch = dict(
            template=plan["b_template"], grid=plan["b_grid"],
            threadgroup=plan["b_threadgroup"], output_shapes=plan["b_shapes"],
            output_dtypes=plan["dtypes1"],
        )

        def pair(flat, nw, axis, eps, dw, ds, db, iw, is_, ib, uw, us, ub):
            xn, act, inj = a_kernel(
                inputs=[flat, nw, axis, eps, dw, ds, db, iw, is_, ib], **a_launch
            )
            mixed = b_kernel(inputs=[xn, act, uw, us, ub], **b_launch)[0]
            return mixed, inj

        fn = mx.compile(pair)
        _COMPILED[key] = fn
    return fn


def hc_decode_launch(module, flat, *, debug: bool = False, plan=None):
    """The two launches for one row ``flat`` [1, 4 * hidden]; returns (mixed, inject|None).

    ``debug`` also returns the normed row and the SiLU activations, for the
    Metal check that localises a mismatch."""
    if plan is None:
        reason, plan = _cached_plan(module)
        if plan is None:
            raise ValueError(f"layout not admitted: {reason}")
    if flat.shape[0] != 1:
        raise ValueError("the launches transcribe the one-row kernels only")
    if not debug:
        mixed, inj = plan["compiled"](flat, *plan["a_inputs"], *plan["b_weights"])
        return mixed, (inj if plan["has_inject"] else None)
    xn, act, inj = plan["a_kernel"](
        inputs=[flat, *plan["a_inputs"]],
        template=plan["a_template"],
        grid=plan["a_grid"],
        threadgroup=plan["a_threadgroup"],
        output_shapes=plan["a_shapes"],
        output_dtypes=plan["dtypes3"],
    )
    mixed = plan["b_kernel"](
        inputs=[xn, act, *plan["b_weights"]],
        template=plan["b_template"],
        grid=plan["b_grid"],
        threadgroup=plan["b_threadgroup"],
        output_shapes=plan["b_shapes"],
        output_dtypes=plan["dtypes1"],
    )[0]
    inject = inj if plan["has_inject"] else None
    if debug:
        return mixed, inject, xn, act
    return mixed, inject


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
    if rows != 1:
        _decline("rows != 1 (MLX runs qmv_wide/qmm there)")
        return None
    if getattr(module, "training", False):
        _decline("training")
        return None
    if not eager_norm:
        _decline("norm is not the eager fast-RMS path")
        return None
    if compile_glue:
        _decline("compiled glue enabled")
        return None
    reason, plan = _cached_plan(module)
    if reason is not None:
        _decline(reason)
        return None
    if hyper_input.shape[2] != HC_COUNT * plan["hidden"]:
        _decline("input width")
        return None
    if not runtime_supported():
        _decline("Metal runtime unavailable")
        return None
    try:
        flat = hyper_input.reshape(rows, -1)
        mixed, inject = hc_decode_launch(module, flat, plan=plan)
        signature = (
            plan["hidden"],
            plan["a_template"][3][1],
            module.hc_norm.weight.dtype,
            inject is not None,
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
