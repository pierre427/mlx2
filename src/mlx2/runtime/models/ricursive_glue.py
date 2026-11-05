# SPDX-License-Identifier: Apache-2.0
"""One-launch Metal kernels for Ripple's per-layer glue (opt-in ``metal_glue``).

The glue ablation (qualification/runs/ripple-perf-ab-20261104/window7) bounds
RoPE at ~3.4 ms, SwiGLU at ~3.2 ms and the value blend at ~2.3 ms per decode
token, each several eager kernels. These kernels keep every rounding point
of the eager source law in ``ricursive``:

* ``qk_norm_rope``: per (row, head) RMSNorm (FP32 sum of squares, precise
  rsqrt, round, weight multiply with one rounding), then (local layers) the
  split-half rotation ``T(y1*c - y2*s)``, ``T(y2*c + y1*s)`` with the
  BF16-rounded FP32 tables, written straight into head-major ``[B, H, L, D]``.
* ``swiglu``: ``T(T(float(g)*sigmoid(float(g))) * u)`` after the clamps, with
  the precise-sigmoid formula (``precise_ops``) that reproduces mx.sigmoid.
* ``blend``: ``T(seed + b * (float(x) - seed))``.

Only the per-row FP32 sum order (norm) and possible FMA contraction differ
from eager, so results are validated like a reordered reduction (gate +
near-tie classified A/B), not claimed bit-exact. ``math_mode='safe'``. Any
other device, dtype or layout falls back to eager (callers get ``None``).
"""

from functools import lru_cache

import mlx.core as mx

STATS = {"qk_norm_rope": 0, "qk_norm": 0, "swiglu": 0, "blend": 0, "fallbacks": 0}
_SAFE = {"math_mode": "safe"}

_QK = r"""
    // grid: (32, rows*heads), one simdgroup per (row, head); D = 128.
    const uint lane = thread_position_in_threadgroup.x;
    const uint item = threadgroup_position_in_grid.y;       // row * H + head
    const uint head = item % H;
    const uint row = item / H;                                // b * L + l
    const uint b = row / L;
    const uint l = row % L;
    const size_t src = size_t(item) * D;                      // [B, L, H, D] contiguous
    float acc = 0.0f;
    for (uint i = lane; i < D; i += 32) { const float v = float(x[src + i]); acc += v * v; }
    acc = simd_sum(acc);
    const float r = metal::precise::rsqrt(acc / float(D) + eps[0]);
    const size_t dst = ((size_t(b) * H + head) * L + l) * D;  // [B, H, L, D]
    const uint half_d = D / 2;
    for (uint i = lane; i < half_d; i += 32) {
        const T n1 = T(float(x[src + i]) * r);
        const T n2 = T(float(x[src + i + half_d]) * r);
        const T m1 = T(float(n1) * float(w[i]));
        const T m2 = T(float(n2) * float(w[i + half_d]));
        ROTATE
    }
"""
_ROTATE = {
    True: r"""
        const size_t t = (size_t(TABLE_BATCH > 1 ? b : 0) * L + l) * half_d + i;
        const float c = cosv[t];
        const float s = sinv[t];
        const float y1 = float(m1);
        const float y2 = float(m2);
        const float a1 = y1 * c;
        const float b1 = y2 * s;
        const float a2 = y2 * c;
        const float b2 = y1 * s;
        out[dst + i] = T(a1 - b1);
        out[dst + i + half_d] = T(a2 + b2);""",
    False: r"""
        out[dst + i] = m1;
        out[dst + i + half_d] = m2;""",
}

_SWIGLU = r"""
    const uint i = thread_position_in_grid.x;
    if (i >= N) { return; }
    // min/clamp in FP32 then round: exact, both operands are already T
    // values (bfloat16_t is a struct, so metal::min/clamp on it is ambiguous).
    const float c = float(T(clampv[0]));
    const T g = T(metal::min(float(gate[i]), c));
    const T u = T(metal::clamp(float(up[i]), -c, c));
    const float g32 = float(g);
    const float e = 1.0f / (1.0f + metal::precise::exp(metal::abs(g32)));
    const float sig = g32 < 0.0f ? e : 1.0f - e;
    const float prod = g32 * sig;
    const T silu = T(prod);
    out[i] = T(float(silu) * float(u));
"""

_BLEND = r"""
    const uint i = thread_position_in_grid.x;
    if (i >= N) { return; }
    const float s = seed[i];
    const float d = float(x[i]) - s;
    const float p = blend[0] * d;
    out[i] = T(s + p);
"""


@lru_cache(maxsize=None)
def _qk_kernel(rope):
    names = ["x", "w", "eps", "cosv", "sinv"] if rope else ["x", "w", "eps"]
    return mx.fast.metal_kernel(name="mlx2_ripple_qk_norm" + ("_rope" if rope else ""), input_names=names,
                                output_names=["out"], source=_QK.replace("ROTATE", _ROTATE[rope]),
                                ensure_row_contiguous=True, compile_options=_SAFE)


@lru_cache(maxsize=None)
def _swiglu_kernel():
    return mx.fast.metal_kernel(name="mlx2_ripple_swiglu", input_names=["gate", "up", "clampv"], output_names=["out"],
                                source=_SWIGLU, ensure_row_contiguous=True, compile_options=_SAFE)


@lru_cache(maxsize=None)
def _blend_kernel():
    return mx.fast.metal_kernel(name="mlx2_ripple_blend", input_names=["seed", "x", "blend"],
                                output_names=["out"], source=_BLEND, ensure_row_contiguous=True,
                                compile_options=_SAFE)


def _metal():
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def _fallback():
    STATS["fallbacks"] += 1
    return None


def qk_norm_rope(raw, w, eps, cos=None, sin=None):
    """``[B, L, H, D]`` projection -> normed (and rotated) ``[B, H, L, D]``."""
    if (not _metal() or raw.ndim != 4 or raw.shape[-1] != 128 or w.shape != (128,)
            or raw.dtype not in (mx.bfloat16, mx.float16) or w.dtype != raw.dtype):
        return _fallback()
    b, length, heads, d = raw.shape
    rope = cos is not None
    inputs = [raw, w, mx.array([eps], dtype=mx.float32)]
    table_batch = 1
    if rope:
        # rotary_tables: [L, half] (shared offset) or [B, 1, L, half] (per lane).
        if cos.ndim == 2 and cos.shape == (length, d // 2):
            tables = (cos, sin)
        elif cos.ndim == 4 and cos.shape[1] == 1 and cos.shape[2:] == (length, d // 2) and cos.shape[0] in (1, b):
            table_batch = cos.shape[0]
            tables = (cos.reshape(table_batch, length, d // 2), sin.reshape(table_batch, length, d // 2))
        else:
            return _fallback()
        if tables[0].dtype != mx.float32:
            return _fallback()
        inputs += list(tables)
    out = _qk_kernel(rope)(
        inputs=inputs, template=[("T", raw.dtype), ("D", d), ("H", heads), ("L", length),
                                 ("TABLE_BATCH", table_batch)],
        grid=(32, b * length * heads, 1), threadgroup=(32, 1, 1),
        output_shapes=[(b, heads, length, d)], output_dtypes=[raw.dtype])[0]
    STATS["qk_norm_rope" if rope else "qk_norm"] += 1
    return out


def swiglu(gate, up, clamp):
    if (not _metal() or not clamp or gate.shape != up.shape or gate.dtype != up.dtype
            or gate.dtype not in (mx.bfloat16, mx.float16)):
        return _fallback()
    n = gate.size
    out = _swiglu_kernel()(inputs=[gate, up, mx.array([clamp], dtype=mx.float32)],
                           template=[("T", gate.dtype), ("N", n)],
                           grid=(n, 1, 1), threadgroup=(256, 1, 1),
                           output_shapes=[gate.shape], output_dtypes=[gate.dtype])[0]
    STATS["swiglu"] += 1
    return out


def blend(seed32, x, value):
    if (not _metal() or seed32.dtype != mx.float32 or seed32.shape != x.shape
            or x.dtype not in (mx.bfloat16, mx.float16)):
        return _fallback()
    n = x.size
    out = _blend_kernel()(inputs=[seed32, x, value.reshape(1).astype(mx.float32)],
                          template=[("T", x.dtype), ("N", n)], grid=(n, 1, 1), threadgroup=(256, 1, 1),
                          output_shapes=[x.shape], output_dtypes=[x.dtype])[0]
    STATS["blend"] += 1
    return out
