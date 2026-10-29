# SPDX-License-Identifier: Apache-2.0
# Fusion boundaries mined from analytics-zoo/vllm-xpu-kernels 265bbe2.
# Original MLX/Metal implementation; provenance/pla-xpu-kernels-20261024.json.
"""Opt-in HC prefill glue for four streams, H2560/H5120, rows 9..8192.

No projection weights are copied or cached. Combine rounds the product and
residual addition in the activation dtype before the FP32 grouped norm.
Outputs are private arrays; dispatch errors propagate without replaying a
partially submitted operation against caller-owned state.
"""
from __future__ import annotations

import os
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, partial

import mlx.core as mx

ENV = "MLX_QWEN4_HC_PREFILL"
MAX_ROWS = 8192
_stats = Counter()
_scope = ContextVar("mlx2_hc_prefill_scope", default=False)


def active():
    return bool(_scope.get())


@contextmanager
def scope(allowed):
    token = _scope.set(bool(allowed))
    try:
        yield
    finally:
        _scope.reset(token)


def enabled():
    return os.environ.get(ENV, "0") == "1"


def status():
    return {"enabled": enabled(), "qualified": False, "max_rows": MAX_ROWS,
            "counters": dict(_stats)}


def refusal(x, weight, hidden, eps):
    if x.ndim != 3 or hidden not in (2560, 5120) or x.shape[-1] != 4 * hidden:
        return "geometry"
    rows = x.shape[0] * x.shape[1]
    if not 9 <= rows <= MAX_ROWS or x.shape[1] < 9:
        return "prefill_rows"
    if x.dtype not in (mx.float16, mx.bfloat16):
        return "activation_dtype"
    if weight.shape != (4 * hidden,) or weight.dtype not in (x.dtype, mx.float32):
        return "weight"
    if eps != 1e-6:
        return "epsilon"
    return None


def eager_norm(x, weight, hidden, eps):
    xf = x.astype(mx.float32).reshape(*x.shape[:-1], 4, hidden)
    norm = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (norm.reshape(x.shape) * weight.astype(mx.float32)).astype(x.dtype)


def eager_combine(x, branch, inject):
    return x + (branch[..., None, :] * inject[..., None]).reshape(x.shape)


_NORM = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint group = threadgroup_position_in_grid.x;
    uint stream = group % 4u;
    uint row = group / 4u;
    size_t base = (size_t)group * H;
    threadgroup float partial[8];
    float sum = 0.0f;
    for (uint d = lane; d < H; d += 256u) {
        T value = x[base + d];
        if constexpr (COMBINE) {
            T product = T(branch[(size_t)row * H + d] * inject[row * 4u + stream]);
            value = T(value + product);
            combined[base + d] = value;
        }
        float f = float(value);
        sum += f * f;
    }
    sum = simd_sum(sum);
    if ((lane % 32u) == 0u) partial[lane / 32u] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 32u) {
        sum = lane < 8u ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0u) partial[0] = metal::precise::rsqrt(sum / float(H) + 1.0e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = lane; d < H; d += 256u) {
        T value = COMBINE ? combined[base + d] : x[base + d];
        normed[base + d] = T((float(value) * partial[0]) * float(weight[stream * H + d]));
    }
"""


@lru_cache(maxsize=2)
def _kernel(combine):
    return mx.fast.metal_kernel(name="mlx2_hc_prefill_combine_norm" if combine else "mlx2_hc_prefill_norm",
        input_names=["x", "weight", "branch", "inject"],
        output_names=["combined", "normed"], source=_NORM)


def _admit(x, weight, hidden, eps):
    if not enabled():
        return False
    reason = refusal(x, weight, hidden, eps)
    if reason is None and (mx.default_device() != mx.gpu or not mx.metal.is_available()):
        reason = "Metal_unavailable"
    if reason:
        _stats["decline_" + reason] += 1
        return False
    return True


def try_norm(x, weight, hidden, eps):
    if not _admit(x, weight, hidden, eps):
        return None
    dummy = mx.zeros((1,), dtype=x.dtype)
    _, normed = _kernel(False)(inputs=[mx.contiguous(x), mx.contiguous(weight), dummy, dummy],
        template=[("T", x.dtype), ("H", hidden), ("COMBINE", False)],
        grid=(256 * x.shape[0] * x.shape[1] * 4, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(1,), x.shape], output_dtypes=[x.dtype, x.dtype])
    _stats["norm"] += 1
    return normed


def try_combine_norm(x, branch, inject, weight, hidden, eps):
    if not _admit(x, weight, hidden, eps):
        return None
    if (branch.shape != (*x.shape[:-1], hidden) or inject.shape != (*x.shape[:-1], 4)
            or branch.dtype != x.dtype or inject.dtype != x.dtype):
        _stats["decline_combine_operands"] += 1
        return None
    result = _kernel(True)(inputs=[mx.contiguous(a) for a in (x, weight, branch, inject)],
        template=[("T", x.dtype), ("H", hidden), ("COMBINE", True)],
        grid=(256 * x.shape[0] * x.shape[1] * 4, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[x.shape, x.shape], output_dtypes=[x.dtype, x.dtype])
    _stats["combine_norm"] += 1
    return tuple(result)


@partial(mx.compile, shapeless=True)
def _gate_mix(normed, raw_gate):
    # Explicit storage-dtype publication before multiply and stream mean.
    weights = mx.sigmoid(raw_gate).astype(normed.dtype)
    product = (weights * normed).astype(normed.dtype)
    return mx.mean(product, axis=-2)


def try_gate_mix(normed, raw_gate, hidden):
    if not enabled() or mx.default_device() != mx.gpu:
        return None
    if (normed.shape != raw_gate.shape or normed.ndim != 3 or
            hidden not in (2560, 5120) or normed.shape[-1] != 4 * hidden or
            normed.dtype not in (mx.float16, mx.bfloat16) or
            raw_gate.dtype != normed.dtype or
            not 9 <= normed.shape[0] * normed.shape[1] <= MAX_ROWS or normed.shape[1] < 9):
        _stats["decline_gate_mix"] += 1
        return None
    out = _gate_mix(normed.reshape(*normed.shape[:-1], 4, hidden),
                    raw_gate.reshape(*raw_gate.shape[:-1], 4, hidden))
    _stats["gate_mix"] += 1
    return out
