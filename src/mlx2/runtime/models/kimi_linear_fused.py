# SPDX-License-Identifier: MIT
"""Kimi FLA output norm/gate using the existing row/head Metal template.

Qwen's early BF16 norm cast is removed deliberately: FLA multiplies the
FP32 normalized value, FP32 norm weight and sigmoid before the final cast.
See provenance/kimi-linear-reap.json and the existing omlx template provenance.
"""
from functools import lru_cache

import mlx.core as mx


def norm_source():
    from .qwen4_fused_gdn_prefill import prefill_norm_gate_source
    source = prefill_norm_gate_source("sigmoid")
    old = "const T normed = norm_w[lane * 4 + i] * T(xs[i] * inv);"
    if source.count(old) != 1:
        raise RuntimeError("Kimi fused norm donor template changed")
    return source.replace(old, "const float normed = float(norm_w[lane * 4 + i]) * (xs[i] * inv);")


@lru_cache(maxsize=1)
def _norm_kernel():
    return mx.fast.metal_kernel(name="mlx2_kimi_fla_norm_gate_v1",
        input_names=["y","z","norm_w","eps"], output_names=["out"],
        source=norm_source(), ensure_row_contiguous=True)


def norm_gate(out, gate, weight, eps):
    b,t,h,d = out.shape
    if d != 128 or gate.shape != out.shape or weight.shape != (128,):
        raise ValueError("Kimi fused norm requires matching 128-channel heads")
    return _norm_kernel()(inputs=[out,gate,weight,mx.array(eps,mx.float32)],
        template=[("T",out.dtype),("HV",h),("DV",d)],
        grid=(32,b*t,h), threadgroup=(32,1,1),
        output_shapes=[(b,t,h*d)], output_dtypes=[out.dtype])[0]
