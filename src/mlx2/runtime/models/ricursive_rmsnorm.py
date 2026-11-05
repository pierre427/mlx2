# SPDX-License-Identifier: Apache-2.0
"""One-launch Ripple RMSNorm (and post-norm + residual) for Metal.

The eager source law (``ricursive._rms``) is about five kernels per norm:
FP32 cast and square, mean, ``+ eps``, ``rsqrt``, then the scale, BF16
rounding and weight multiply. Ripple runs 480 of them per decode token (four
block norms plus per-head Q/K norms in each of 80 layers). This kernel does
the same arithmetic per row in one launch:

    y = float(x);  r = precise::rsqrt(sum(y*y) / D + eps)
    n = T(y * r);  out = T(float(n) * float(w))        [norm]
    out = T(float(residual) + float(T(float(n) * float(w))))  [residual norm]

Every rounding point matches the eager law. Only the FP32 summation order
differs (a threadgroup tree instead of MLX's reduction), so results are not
guaranteed bit-identical: the option is validated like any reordered
reduction (component mismatch rate, near-tie classified serving A/B).
``math_mode='safe'`` keeps ``rsqrt`` and the casts IEEE-faithful.
Selected per model with ``Model.set_metal_rmsnorm``; any other device, dtype
or shape falls back to the eager law.
"""

from functools import lru_cache

import mlx.core as mx

DTYPES = (mx.bfloat16, mx.float16, mx.float32)

_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint group = simdgroup_index_in_threadgroup;
    const size_t base = size_t(row) * D;
    float acc = 0.0f;
    for (uint i = tid; i < D; i += TG) {
        const float v = float(x[base + i]);
        acc += v * v;
    }
    acc = simd_sum(acc);
    threadgroup float partial[32];
    if (lane == 0) {
        partial[group] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (group == 0) {
        float total = lane < (TG + 31) / 32 ? partial[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) {
            partial[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float r = metal::precise::rsqrt(partial[0] / float(D) + eps[0]);
    for (uint i = tid; i < D; i += TG) {
        const T n = T(float(x[base + i]) * r);
        const T m = T(float(n) * float(w[i]));
        STORE
    }
"""
_STORE = {False: "out[base + i] = m;",
          True: "out[base + i] = T(float(residual[base + i]) + float(m));"}


@lru_cache(maxsize=None)
def _kernel(residual):
    inputs = ["x", "w", "eps", "residual"] if residual else ["x", "w", "eps"]
    return mx.fast.metal_kernel(
        name="mlx2_ripple_rmsnorm" + ("_residual" if residual else ""),
        input_names=inputs, output_names=["out"], source=_SOURCE.replace("STORE", _STORE[residual]),
        ensure_row_contiguous=True, compile_options={"math_mode": "safe"},
    )


def supported(x, w, residual=None):
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return False
    if x.dtype not in DTYPES or w.dtype != x.dtype or w.ndim != 1 or x.shape[-1] != w.shape[0]:
        return False
    if residual is not None and (residual.shape != x.shape or residual.dtype != x.dtype):
        return False
    return 1 <= x.shape[-1] <= 65536 and x.size > 0


def threads(dims):
    return 32 if dims <= 256 else (256 if dims <= 8192 else 1024)


STATS = {"calls": 0, "residual_calls": 0, "fallbacks": 0}


def rms_norm(x, w, eps, residual=None):
    """``[residual +] _rms(x, w, eps)`` in one launch, else ``None``."""
    if not supported(x, w, residual):
        STATS["fallbacks"] += 1
        return None
    dims = x.shape[-1]
    rows = x.size // dims
    tg = threads(dims)
    inputs = [x, w, mx.array([eps], dtype=mx.float32)]
    if residual is not None:
        inputs.append(residual)
    out = _kernel(residual is not None)(
        inputs=inputs, template=[("T", x.dtype), ("D", dims), ("TG", tg)],
        grid=(rows * tg, 1, 1), threadgroup=(tg, 1, 1),
        output_shapes=[x.shape], output_dtypes=[x.dtype],
    )[0]
    STATS["residual_calls" if residual is not None else "calls"] += 1
    return out
