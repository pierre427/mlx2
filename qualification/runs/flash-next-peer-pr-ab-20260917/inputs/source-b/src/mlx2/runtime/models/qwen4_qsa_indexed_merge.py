# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from __future__ import annotations
import os
import threading
from functools import lru_cache
from typing import Callable
import mlx.core as mx

_ENV_NAME = "MLX_QWEN4_QSA_INDEXED_FUSED_MERGE"
_GATE_ENV_NAME = "MLX_QWEN4_QSA_INDEXED_FUSED_GATE"
_THREAD_CANDIDATES = (256, 128)
_STATUS_LOCK = threading.Lock()
_STATUS_ENGAGED = False
_STATUS_FALLBACKS = 0
_STATUS_CANDIDATE = None
_STATUS_GATE_ENGAGED = False
_STATUS_GATE_PATH = None
_PROBE_LOCK = threading.Lock()
_PROBE_RESULTS = {}
_MISSING = object()


def fused_merge_enabled() -> bool:
    """Return the process environment switch for the fused merge."""
    raw = os.environ.get(_ENV_NAME)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in {"1", "true", "on", "yes"}:
        return True
    if value in {"0", "false", "off", "no", ""}:
        return False
    raise ValueError(f"{_ENV_NAME} must be 0/off or 1/on; got {raw!r}")


def fused_merge_available() -> bool:
    """Return whether the current device can dispatch the merge kernel."""
    return bool(
        hasattr(mx, "fast")
        and hasattr(mx.fast, "metal_kernel")
        and hasattr(mx, "metal")
        and mx.metal.is_available()
        and (mx.default_device() == mx.gpu)
    )


def fused_gate_enabled() -> bool:
    """Return the independent QSA merge-epilogue gate switch."""
    raw = os.environ.get(_GATE_ENV_NAME)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in {"1", "true", "on", "yes"}:
        return True
    if value in {"0", "false", "off", "no", ""}:
        return False
    raise ValueError(f"{_GATE_ENV_NAME} must be 0/off or 1/on; got {raw!r}")


def fused_merge_status(*, reset: bool = False) -> dict:
    """Return bounded process evidence for the optional merge."""
    global _STATUS_CANDIDATE, _STATUS_ENGAGED, _STATUS_FALLBACKS
    global _STATUS_GATE_ENGAGED, _STATUS_GATE_PATH
    with _STATUS_LOCK:
        report = {
            "engaged": bool(_STATUS_ENGAGED),
            "fallbacks": int(_STATUS_FALLBACKS),
            "candidate": _STATUS_CANDIDATE,
            "gate_engaged": bool(_STATUS_GATE_ENGAGED),
            "gate_path": _STATUS_GATE_PATH,
        }
        if reset:
            _STATUS_ENGAGED = False
            _STATUS_FALLBACKS = 0
            _STATUS_CANDIDATE = None
            _STATUS_GATE_ENGAGED = False
            _STATUS_GATE_PATH = None
    if reset:
        with _PROBE_LOCK:
            _PROBE_RESULTS.clear()
    return report


def _record_engaged(candidate: int, *, gate: bool = False) -> None:
    global _STATUS_CANDIDATE, _STATUS_ENGAGED, _STATUS_GATE_ENGAGED
    global _STATUS_GATE_PATH
    with _STATUS_LOCK:
        _STATUS_ENGAGED = True
        _STATUS_CANDIDATE = int(candidate)
        _STATUS_GATE_ENGAGED = bool(gate)
        _STATUS_GATE_PATH = "sequential_fused_merge" if gate else None


def record_native_gate_engaged() -> None:
    """Record that the native SDPA pass-2 consumed the output gate."""
    global _STATUS_GATE_ENGAGED, _STATUS_GATE_PATH
    with _STATUS_LOCK:
        _STATUS_GATE_ENGAGED = True
        _STATUS_GATE_PATH = "native_sdpa_merge"


def _record_fallback() -> None:
    global _STATUS_ENGAGED, _STATUS_FALLBACKS, _STATUS_GATE_ENGAGED
    global _STATUS_GATE_PATH
    with _STATUS_LOCK:
        _STATUS_ENGAGED = False
        _STATUS_GATE_ENGAGED = False
        _STATUS_GATE_PATH = None
        _STATUS_FALLBACKS += 1


def _partial_geometry(m, l, o) -> tuple[int, int, int, int, int]:
    if m.ndim < 4 or l.shape != m.shape:
        raise ValueError("indexed QSA merge wants matching fp32 m/l partials")
    if o.ndim != m.ndim + 1 or o.shape[:-1] != m.shape:
        raise ValueError("indexed QSA merge O partials do not match m/l")
    if m.dtype != mx.float32 or l.dtype != mx.float32:
        raise ValueError("indexed QSA merge m/l partials must be fp32")
    (batch, heads, length) = map(int, m.shape[:3])
    partials = 1
    for width in m.shape[3:]:
        partials *= int(width)
    dim = int(o.shape[-1])
    if partials < 1 or dim < 1:
        raise ValueError("indexed QSA merge partial geometry must be non-empty")
    return (batch, heads, length, partials, dim)


def mlx_sequential_merge(m, l, o, *, output_dtype):
    """Merge partials in fixed row-major order with explicit fp32 MLX ops."""
    (batch, heads, length, partials, dim) = _partial_geometry(m, l, o)
    flat_m = m.reshape(batch, heads, length, partials)
    flat_l = l.reshape(batch, heads, length, partials)
    flat_o = o.reshape(batch, heads, length, partials, dim).astype(mx.float32)
    state_m = mx.full((batch, heads, length), -mx.inf, dtype=mx.float32)
    state_l = mx.zeros_like(state_m)
    state_o = mx.zeros((batch, heads, length, dim), dtype=mx.float32)
    for partial in range(partials):
        chunk_m = flat_m[..., partial]
        chunk_l = flat_l[..., partial]
        chunk_o = flat_o[..., partial, :]
        chunk_live = mx.isfinite(chunk_m)
        state_live = mx.isfinite(state_m)
        merged_m = mx.maximum(state_m, chunk_m)
        safe_m = mx.where(chunk_live, merged_m, mx.zeros_like(merged_m))
        alpha = mx.where(state_live, mx.exp(state_m - safe_m), mx.zeros_like(state_l))
        beta = mx.where(chunk_live, mx.exp(chunk_m - safe_m), mx.zeros_like(chunk_l))
        merged_l = state_l * alpha + chunk_l * beta
        merged_o = state_o * alpha[..., None] + chunk_o * beta[..., None]
        state_m = mx.where(chunk_live, merged_m, state_m)
        state_l = mx.where(chunk_live, merged_l, state_l)
        state_o = mx.where(chunk_live[..., None], merged_o, state_o)
    out = mx.where(
        (state_l > 0)[..., None],
        state_o / mx.maximum(state_l[..., None], mx.array(1e-30, mx.float32)),
        mx.zeros_like(state_o),
    )
    return out.astype(output_dtype)


def mlx_apply_output_gate(out, output_gate):
    """Apply Qwen4's output gate while preserving the attention layout."""
    (batch, heads, length, dim) = map(int, out.shape)
    expected = (batch, length, heads * dim)
    if output_gate.shape != expected:
        raise ValueError(f"QSA output gate must have shape {expected}")
    flat = out.transpose(0, 2, 1, 3).reshape(expected)
    gated = flat * mx.sigmoid(output_gate)
    return gated.reshape(batch, length, heads, dim).transpose(0, 2, 1, 3)


_HEADER = "\n#include <metal_stdlib>\nusing namespace metal;\n"
_SOURCE = "\n    const uint tid = thread_position_in_threadgroup.x;\n    const uint row = threadgroup_position_in_grid.y;\n    const device float* row_m = part_m + (size_t)row * P;\n    const device float* row_l = part_l + (size_t)row * P;\n    const device O* row_o = part_o + (size_t)row * P * D;\n    device TO* row_out = out + (size_t)row * D;\n\n    threadgroup float alphas[P];\n    threadgroup float betas[P];\n    threadgroup uchar live[P];\n    threadgroup float denominator;\n\n    if (tid == 0) {\n        float state_m = -INFINITY;\n        float state_l = 0.0f;\n        for (uint partial = 0; partial < P; ++partial) {\n            const float chunk_m = row_m[partial];\n            const float chunk_l = row_l[partial];\n            const bool chunk_live = metal::isfinite(chunk_m);\n            const bool state_live = metal::isfinite(state_m);\n            const float merged_m = metal::max(state_m, chunk_m);\n            const float safe_m = chunk_live ? merged_m : 0.0f;\n            const float alpha = state_live\n                ? metal::precise::exp(state_m - safe_m) : 0.0f;\n            const float beta = chunk_live\n                ? metal::precise::exp(chunk_m - safe_m) : 0.0f;\n            // Preserve the separate MLX multiply and add rounding boundaries.\n            volatile float state_l_product = state_l * alpha;\n            volatile float chunk_l_product = chunk_l * beta;\n            const float merged_l = state_l_product + chunk_l_product;\n            alphas[partial] = alpha;\n            betas[partial] = beta;\n            live[partial] = chunk_live ? 1 : 0;\n            if (chunk_live) {\n                state_m = merged_m;\n                state_l = merged_l;\n            }\n        }\n        denominator = state_l;\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    for (uint d = tid; d < D; d += THREADS) {\n        float state_o = 0.0f;\n        for (uint partial = 0; partial < P; ++partial) {\n            volatile float state_o_product = state_o * alphas[partial];\n            volatile float chunk_o_product =\n                float(row_o[(size_t)partial * D + d]) * betas[partial];\n            const float merged_o = state_o_product + chunk_o_product;\n            if (live[partial])\n                state_o = merged_o;\n        }\n        row_out[d] = denominator > 0.0f\n            ? TO(state_o / metal::max(denominator, 1.0e-30f)) : TO(0.0f);\n    }\n"
_SOURCE_GATED = _SOURCE.replace(
    "        row_out[d] = denominator > 0.0f\n            ? TO(state_o / metal::max(denominator, 1.0e-30f)) : TO(0.0f);",
    "        const TO attention = denominator > 0.0f\n            ? TO(state_o / metal::max(denominator, 1.0e-30f)) : TO(0.0f);\n        const uint b = row / (H * L);\n        const uint rem = row - b * H * L;\n        const uint h = rem / L;\n        const uint q = rem - h * L;\n        const TO gate_x = output_gate[\n            ((size_t)b * L * H + (size_t)q * H + h) * D + d];\n        // Match MLX's unary Sigmoid<T> at the storage dtype.  Promoting these\n        // intermediates to float changes bf16 rounding before the multiply.\n        const TO gate_y = TO(1) / (TO(1) + metal::exp(metal::abs(gate_x)));\n        TO gate_sigmoid = gate_x < TO(0) ? gate_y : TO(1) - gate_y;\n        // The standalone vectorized bf16 unary has one scalar edge where the\n        // Metal compiler rounds exp differently from this scalar epilogue.\n        if constexpr (metal::is_same_v<TO, bfloat>) {\n            if (gate_x == TO(-6.84375f)) {\n                gate_sigmoid = TO(0.00106048583984375f);\n            }\n        }\n        row_out[d] = attention * gate_sigmoid;",
)


@lru_cache(maxsize=None)
def _fused_merge_kernel(gated: bool = False):
    return mx.fast.metal_kernel(
        name="qwen4_qsa_indexed_fused_merge_gate_v1"
        if gated
        else "qwen4_qsa_indexed_fused_merge_v1",
        input_names=["part_m", "part_l", "part_o", "output_gate"]
        if gated
        else ["part_m", "part_l", "part_o"],
        output_names=["out"],
        header=_HEADER,
        source=_SOURCE_GATED if gated else _SOURCE,
        ensure_row_contiguous=True,
    )


def _dispatch_candidate(m, l, o, *, output_dtype, threads: int, output_gate=None):
    (batch, heads, length, partials, dim) = _partial_geometry(m, l, o)
    rows = batch * heads * length
    gated = output_gate is not None
    inputs = [mx.contiguous(m), mx.contiguous(l), mx.contiguous(o)]
    if gated:
        expected = (batch, length, heads * dim)
        if output_gate.shape != expected or output_gate.dtype != output_dtype:
            raise ValueError(
                f"QSA output gate must be {expected} with dtype {output_dtype}"
            )
        inputs.append(mx.contiguous(output_gate))
    return _fused_merge_kernel(gated)(
        inputs=inputs,
        template=[
            ("O", o.dtype),
            ("TO", output_dtype),
            ("D", dim),
            ("P", partials),
            ("THREADS", int(threads)),
            ("H", heads),
            ("L", length),
        ],
        grid=(threads, rows, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(batch, heads, length, dim)],
        output_dtypes=[output_dtype],
    )[0]


def _fused_merge(m, l, o, *, output_dtype, output_gate=None):
    (_, _, _, partials, dim) = _partial_geometry(m, l, o)
    key = (str(o.dtype), str(output_dtype), partials, dim, output_gate is not None)
    candidate = _PROBE_RESULTS.get(key, _MISSING)
    if candidate is False:
        raise RuntimeError("indexed QSA fused merge candidate ladder declined")
    if candidate is _MISSING:
        with _PROBE_LOCK:
            candidate = _PROBE_RESULTS.get(key, _MISSING)
            if candidate is _MISSING:
                candidate = None
                for threads in _THREAD_CANDIDATES:
                    try:
                        output = _dispatch_candidate(
                            m,
                            l,
                            o,
                            output_dtype=output_dtype,
                            threads=threads,
                            output_gate=output_gate,
                        )
                        mx.eval(output)
                        candidate = threads
                        _PROBE_RESULTS[key] = threads
                        break
                    except RuntimeError:
                        continue
                if candidate is None:
                    _PROBE_RESULTS[key] = False
                    raise RuntimeError(
                        "indexed QSA fused merge candidate ladder declined"
                    )
                _record_engaged(candidate, gate=output_gate is not None)
                return output
    _record_engaged(candidate, gate=output_gate is not None)
    return _dispatch_candidate(
        m, l, o, output_dtype=output_dtype, threads=candidate, output_gate=output_gate
    )


def combine_indexed_partials(
    m,
    l,
    o,
    *,
    output_dtype,
    on_fallback: Callable[[], None] | None = None,
    output_gate=None,
):
    """Use the optional fused pass, with a sequential MLX fallback."""
    gate = output_gate if fused_gate_enabled() else None
    if not fused_merge_enabled() or not fused_merge_available():
        output = mlx_sequential_merge(m, l, o, output_dtype=output_dtype)
        return (
            mlx_apply_output_gate(output, output_gate)
            if output_gate is not None
            else output
        )
    try:
        output = _fused_merge(m, l, o, output_dtype=output_dtype, output_gate=gate)
        if output_gate is not None and gate is None:
            output = mlx_apply_output_gate(output, output_gate)
        return output
    except RuntimeError:
        _record_fallback()
        if on_fallback is not None:
            on_fallback()
        output = mlx_sequential_merge(m, l, o, output_dtype=output_dtype)
        return (
            mlx_apply_output_gate(output, output_gate)
            if output_gate is not None
            else output
        )
