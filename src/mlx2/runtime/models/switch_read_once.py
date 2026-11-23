# SPDX-License-Identifier: MIT
"""Read-once small-row kernels for 4-bit affine SwiGLU expert banks.

Original mlx2 code; design references (no code copied) in
provenance/hy4-expert-read-once.json.  See docs/hy4-flash/expert-read-once.md.

Why: MLX's ``gather_qmm`` streams each expert once (``gather_qmm_rhs``)
only when a sorted gather has ``rows // num_experts >= 4`` -- for a
512-expert table that is 2048 assignments.  Below it every assignment runs
``gather_qmv`` and re-reads its expert, so a 7-8 row verify (top-16,
~1.8-2.1 assignments per touched expert) reads most experts twice.

What: the flat ``(row, slot)`` assignments are sorted by expert (the same
``argsort`` the stock path runs).  One threadgroup per (output row block,
chunk of up to ``MCH`` assignments of one expert) loads each of that
block's 4-bit words once into registers and multiplies it against every
assignment of the chunk (activations prescaled by 1/16^i so a nibble is
masked in place; fp32 ``scale * sum(q * x) + bias * sum(x)`` per 64-wide
group, then a simdgroup sum).  Kernel 1 does gate and up together and
writes the clamped SwiGLU in the activation dtype; kernel 2 does the down
projection and writes each assignment's output straight to its unsorted
slot, so the caller's router-weight combine is unchanged.

Numerics: fp32 accumulation in another order than ``gather_qmv``, with
gate/up rounded to the activation dtype before the activation like the
stock path.  Not bit-exact; measured against the stock path in
docs/hy4-flash/expert-read-once.md.  Inference only, Metal only, default
off; callers opt in (Hy4-flash ``moe_expert_read_once``).
"""

from __future__ import annotations

import math
import threading
from functools import lru_cache

import mlx.core as mx

from . import invariant_prefill as _invariant

GROUP = 64
BITS = 4
# Rows per simdgroup, simdgroups per threadgroup, assignments per chunk,
# packed words per lane per step.  Chosen on the M5 Max chain-marginal sweep
# (qualification/runs/hy4-readonce-config-sweep-20261121-139130069):
# 4/2/4/1 and 4/2/2/1 tie; VPL 2 is 1.3-1.6x slower, SG 1 and RPS 2 lose
# from 12 rows, MCH 8 loses everywhere (registers).
RPS = 4
SG = 2
MCH = 4
VPL = 1
# Admission window.  One row (B=1 decode) keeps the stock unsorted
# gather_qmv, which the read-once kernel only ties; above MAX_ASSIGNMENTS
# (rows * top_k) the stock route is kept (docs/hy4-flash/expert-read-once.md).
MIN_ROWS = 2
MAX_ASSIGNMENTS = 1024

_STATS_LOCK = threading.Lock()
_stats = {"calls": 0, "assignments": 0, "rows": {}, "fallbacks": 0, "reasons": {}}


def status(*, reset: bool = False) -> dict:
    """Observed use: calls served, assignments, rows histogram, refusals."""
    with _STATS_LOCK:
        out = {"implemented": True, "qualified": False, "calls": _stats["calls"],
               "assignments": _stats["assignments"], "rows": dict(_stats["rows"]),
               "fallbacks": _stats["fallbacks"], "reasons": dict(_stats["reasons"]),
               "observed_used": _stats["calls"] > 0,
               "config": {"rows_per_simdgroup": RPS, "simdgroups": SG, "chunk": MCH, "words_per_lane": VPL,
                          "min_rows": MIN_ROWS, "max_assignments": MAX_ASSIGNMENTS}}
        if reset:
            _stats.update(calls=0, assignments=0, fallbacks=0)
            _stats["rows"].clear()
            _stats["reasons"].clear()
    return out


def _refuse(reason: str):
    with _STATS_LOCK:
        _stats["fallbacks"] += 1
        _stats["reasons"][reason] = _stats["reasons"].get(reason, 0) + 1
    return None


def _served(rows: int, assignments: int) -> None:
    with _STATS_LOCK:
        _stats["calls"] += 1
        _stats["assignments"] += assignments
        key = str(rows)
        _stats["rows"][key] = _stats["rows"].get(key, 0) + 1


HEADER = """
// 8*VPL activations of one lane, prescaled by 1/16^(i%8) so a nibble masked
// in place (w & (0xF << 4i), no shift) multiplies to q * x; returns sum(x).
template <typename T, int VPL>
inline float ro_load_x(const device T* p, thread float* v) {
    float s = 0.f;
    _Pragma("clang loop unroll(full)")
    for (int c = 0; c < 2 * VPL; ++c) {
        vec<T, 4> a = ((const device vec<T, 4>*)p)[c];
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < 4; ++i) {
            float f = float(a[i]);
            s += f;
            int k = c * 4 + i;
            v[k] = f * ((k % 8) == 0 ? 1.f : (k % 8) == 1 ? 0.0625f : (k % 8) == 2 ? 0.00390625f
                      : (k % 8) == 3 ? 0.000244140625f : (k % 8) == 4 ? 1.52587890625e-05f
                      : (k % 8) == 5 ? 9.5367431640625e-07f : (k % 8) == 6 ? 5.9604644775390625e-08f
                      : 3.725290298461914e-09f);
        }
    }
    return s;
}
template <int VPL>
inline float ro_dot(thread const float* v, thread const uint* w) {
    float d = 0.f;
    _Pragma("clang loop unroll(full)")
    for (int c = 0; c < VPL; ++c) {
        uint word = w[c];
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < 8; ++i) d = fma(v[c * 8 + i], float(word & (0xFu << (4 * i))), d);
    }
    return d;
}
"""

# One source for both kernels.  DUAL = 1: gate (gq) and up (uq) rows of one
# block, clamped SwiGLU epilogue, x rows read through order / KTOP, output in
# sorted order.  DUAL = 0: one bank, input rows in sorted order, output
# written to order[p] (the unsorted slot).  Every loop over rows (RPS),
# assignments (MCH) and words (VPL) has a compile-time trip count and is
# fully unrolled, so the accumulators stay in registers.
SOURCE = """
    constexpr int NW = K / 8;           // packed words per row
    constexpr int NG = K / 64;          // scale groups per row
    constexpr int LW = 32 * VPL;        // words per simdgroup step
    constexpr int STEPS = NW / LW;
    uint lane = thread_index_in_simdgroup;
    uint sgi = simdgroup_index_in_threadgroup;
    uint p = threadgroup_position_in_grid.y;
    uint row0 = (threadgroup_position_in_grid.x * SG + sgi) * RPS;
    uint e = se[p];
    uint rank = 0;
    while (rank < p && se[p - rank - 1] == e) ++rank;
    if (rank % MCH != 0 || row0 >= N) return;
    uint m = 1;
    while (m < MCH && p + m < NS && se[p + m] == e) ++m;
    const device T* xp[MCH];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < MCH; ++j) {
        uint q = p + min(uint(j), m - 1);
        xp[j] = x + (size_t)(DUAL ? order[q] / KTOP : q) * K + lane * VPL * 8;
    }
    float ag[MCH * RPS], au[MCH * RPS];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < MCH * RPS; ++i) { ag[i] = 0.f; au[i] = 0.f; }
    const device uint* gw = gq + ((size_t)e * GSTRIDE + row0) * NW + lane * VPL;
    const device uint* uw = uq + ((size_t)e * GSTRIDE + UOFF + row0) * NW + lane * VPL;
    size_t gsr = ((size_t)e * GSTRIDE + row0) * NG;
    size_t usr = ((size_t)e * GSTRIDE + UOFF + row0) * NG;
    for (int s = 0; s < STEPS; ++s) {
        int g = (s * LW + int(lane) * VPL) / 8;
        uint wg[RPS * VPL], wu[RPS * VPL];
        float sg_[RPS], bg_[RPS], su_[RPS], bu_[RPS];
        _Pragma("clang loop unroll(full)")
        for (int r = 0; r < RPS; ++r) {
            _Pragma("clang loop unroll(full)")
            for (int c = 0; c < VPL; ++c) wg[r * VPL + c] = gw[r * NW + s * LW + c];
            sg_[r] = float(gs[gsr + r * NG + g]);
            bg_[r] = float(gb[gsr + r * NG + g]);
            if (DUAL) {
                _Pragma("clang loop unroll(full)")
                for (int c = 0; c < VPL; ++c) wu[r * VPL + c] = uw[r * NW + s * LW + c];
                su_[r] = float(us[usr + r * NG + g]);
                bu_[r] = float(ub[usr + r * NG + g]);
            }
        }
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < MCH; ++j) {
            if (uint(j) < m) {
                float xv[8 * VPL];
                float xs = ro_load_x<T, VPL>(xp[j] + s * LW * 8, xv);
                _Pragma("clang loop unroll(full)")
                for (int r = 0; r < RPS; ++r) {
                    ag[j * RPS + r] += sg_[r] * ro_dot<VPL>(xv, wg + r * VPL) + bg_[r] * xs;
                    if (DUAL) au[j * RPS + r] += su_[r] * ro_dot<VPL>(xv, wu + r * VPL) + bu_[r] * xs;
                }
            }
        }
    }
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < MCH; ++j) {
        if (uint(j) < m) {
            size_t orow = (size_t)(DUAL ? (p + j) : order[p + j]) * N + row0;
            _Pragma("clang loop unroll(full)")
            for (int r = 0; r < RPS; ++r) {
                float a = simd_sum(ag[j * RPS + r]);
                if (DUAL) {
                    float u = simd_sum(au[j * RPS + r]);
                    if (lane == 0) {
                        // stock: the gate/up gathers write T, then silu(min(gate, l)) * clip(up, -l, l) in T
                        float lim = limit[0];
                        float gm = metal::min(float(T(a)), lim);
                        float sig = float(T(1.f / (1.f + metal::exp(-gm))));
                        float si = float(T(gm * sig));
                        float uc = float(T(metal::clamp(float(T(u)), -lim, lim)));
                        y[orow + r] = T(si * uc);
                    }
                } else if (lane == 0) {
                    y[orow + r] = T(a);
                }
            }
        }
    }
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        if not mx.metal.is_available():
            return None
        _kernel = mx.fast.metal_kernel(
            name="mlx2_read_once_q4_experts",
            input_names=["x", "gq", "gs", "gb", "uq", "us", "ub", "se", "order", "limit"],
            output_names=["y"], source=SOURCE, header=HEADER)
    return _kernel


def _banks(glu):
    """``(gate, up, down, gstride, uoff)`` quantized planes, or a refusal string."""
    from .switch_layers import GateUpFusedSwitchGLU, QuantizedSwitchLinear, SwitchGLU

    if type(glu) is SwitchGLU:
        gate, up, down = glu.gate_proj, glu.up_proj, glu.down_proj
        banks = (gate, up, down)
    elif type(glu) is GateUpFusedSwitchGLU:
        gate = up = glu.gate_up_proj
        down = glu.down_proj
        banks = (gate, down)
    else:
        return f"module_{type(glu).__name__}"
    for b in banks:
        if type(b) is not QuantizedSwitchLinear:
            return f"bank_{type(b).__name__}"
        if (b.mode, b.bits, b.group_size) != ("affine", BITS, GROUP):
            return "table_not_affine_q4_g64"
        if "bias" in b or b.get("biases") is None:
            return "bias_present_or_biases_missing"
        w, sc, bias = b.weight, b.scales, b.biases
        if (w.dtype != mx.uint32 or w.ndim != 3 or sc.ndim != 3
                or sc.shape != bias.shape or w.shape[:2] != sc.shape[:2]
                or w.shape[-1] * 8 != sc.shape[-1] * GROUP):
            return "quantized_plane_layout"
    if type(glu) is SwitchGLU:
        F = gate.output_dims
        return gate, up, down, F, 0, F
    F = glu.hidden_dims
    return gate, up, down, 2 * F, F, F


def _limit(activation) -> "float | None":
    """Clamp limit of a (clamped) SwiGLU, ``inf`` for the plain one, None otherwise."""
    from .switch_layers import SwiGLU

    if type(activation) is SwiGLU:
        return math.inf
    from .hy_v4_flash import ClampedSwiGLU

    # Exactly Hy4's silu(min(g, l)) * clip(u, -l, l); another clamped variant
    # (an alpha, a gate offset) would need its own epilogue.
    if type(activation) is ClampedSwiGLU and isinstance(activation.limit, float):
        return activation.limit
    return None


def refusal(glu, x, indices) -> "str | None":
    """Why the read-once kernels may not serve this call, or None.

    Structural checks first (so a CPU run still names the structural
    reason), then the row window, then the device."""
    if _invariant.active():
        return "invariant_lane"
    if x.dtype not in (mx.bfloat16, mx.float16):
        return "activation_dtype"
    banks = _banks(glu)
    if isinstance(banks, str):
        return banks
    if _limit(glu.activation) is None:
        return "activation_not_swiglu"
    if (x.ndim < 1 or indices.ndim < 1 or indices.shape[-1] == 0
            or x.shape[:-1] != indices.shape[:-1]):
        return "routing_shape"
    gate, up, down, stride, _, F = banks
    D = int(x.shape[-1])
    if (gate.input_dims != D or gate.output_dims != stride or down.output_dims != D
            or down.input_dims != F or down.num_experts != gate.num_experts):
        return "geometry"
    if up is not gate and (up.input_dims, up.output_dims, up.num_experts) != (
            gate.input_dims, gate.output_dims, gate.num_experts):
        return "geometry"
    if any(b[p].dtype != x.dtype for b in {id(b): b for b in (gate, up, down)}.values()
           for p in ("scales", "biases")):
        # gather_qmm promotes mixed activation/scale dtypes; the kernels round
        # every stage to x.dtype, so they would not follow it.
        return "scale_dtype_mismatch"
    if D % (256 * VPL) or F % (256 * VPL) or F % (RPS * SG) or D % (RPS * SG):
        return "dims_not_multiple_of_lane_step"
    n = int(indices.size)
    k = int(indices.shape[-1]) if indices.ndim else 1
    if n // max(k, 1) < MIN_ROWS:
        return "below_min_rows"
    if n > MAX_ASSIGNMENTS:
        return "above_max_assignments"
    if mx.default_device() != mx.gpu:
        return "cpu_device"
    if _get_kernel() is None:
        return "metal_unavailable"
    return None


@lru_cache(maxsize=8)
def _limit_array(limit: float):
    a = mx.array([limit if math.isfinite(limit) else 3.0e38], dtype=mx.float32)
    mx.eval(a)
    return a


def forward(x, indices, gq, gs, gb, uq, us, ub, dq, ds, db, *, F, gstride, uoff, limit):
    """x [T, D], indices [T, k] -> per-assignment outputs [T, k, D] (x.dtype)."""
    kern = _get_kernel()
    T_, k = indices.shape
    D = x.shape[-1]
    n = T_ * k
    flat = indices.reshape(n).astype(mx.uint32)
    order = mx.argsort(flat)
    se = flat[order]
    lim = _limit_array(limit)
    tg = (32 * SG, 1, 1)
    base = [("T", x.dtype), ("NS", n), ("KTOP", k), ("RPS", RPS), ("SG", SG), ("MCH", MCH), ("VPL", VPL)]
    h = kern(inputs=[x, gq, gs, gb, uq, us, ub, se, order, lim],
             template=base + [("N", F), ("K", D), ("DUAL", 1), ("GSTRIDE", gstride), ("UOFF", uoff)],
             grid=(32 * SG * (F // (RPS * SG)), n, 1), threadgroup=tg,
             output_shapes=[(n, F)], output_dtypes=[x.dtype])[0]
    y = kern(inputs=[h, dq, ds, db, dq, ds, db, se, order, lim],
             template=base + [("N", D), ("K", F), ("DUAL", 0), ("GSTRIDE", D), ("UOFF", 0)],
             grid=(32 * SG * (D // (RPS * SG)), n, 1), threadgroup=tg,
             output_shapes=[(n, D)], output_dtypes=[x.dtype])[0]
    return y.reshape(T_, k, D)


def switch_glu(glu, x, indices, *, count: bool = True):
    """Per-assignment output of ``glu`` (``indices.shape + [D]``) or None (refused, counted)."""
    reason = refusal(glu, x, indices)
    if reason is not None:
        return _refuse(reason) if count else None
    gate, up, down, gstride, uoff, F = _banks(glu)
    k = int(indices.shape[-1])
    T_ = int(indices.size) // k
    D = int(x.shape[-1])
    y = forward(x.reshape(T_, D), indices.reshape(T_, k),
                gate["weight"], gate["scales"], gate["biases"],
                up["weight"], up["scales"], up["biases"],
                down["weight"], down["scales"], down["biases"],
                F=F, gstride=gstride, uoff=uoff, limit=_limit(glu.activation))
    if count:
        _served(T_, T_ * k)
    return y.reshape(*indices.shape, D)
