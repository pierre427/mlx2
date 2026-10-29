# SPDX-License-Identifier: Apache-2.0
"""Fixed-row HC-only FP32 projection; explicit unqualified arithmetic profile.

Original kernel, provenance/deepseek-v4-hc-fixed-projection.json. A row always
uses the same32-lane accumulation order, independently of the number of rows.
This does not claim parity with the native shape-dependent GEMV/GEMM route.
"""

import math
from functools import lru_cache

SOURCE = r"""
const uint lane=thread_index_in_simdgroup;
const uint output=threadgroup_position_in_grid.y;
const uint row=threadgroup_position_in_grid.z;
float sum=0.0f;
for(uint k=lane;k<K;k+=32) {
    sum+=x[ulong(row)*K+k]*weight[ulong(output)*K+k];
}
sum=simd_sum(sum);
if(lane==0) out[ulong(row)*OUTPUTS+output]=sum;
"""
_SCHEDULED_CALLS = 0


def validate_geometry(value, weight):
    if len(value.shape) < 1 or len(weight.shape) != 2 or weight.shape[0] not in (4, 24):
        raise ValueError("fixed HC projection requires [...,K] and [4|24,K]")
    k = weight.shape[1]
    if k < 32 or k % 32 or value.shape[-1] != k:
        raise ValueError("fixed HC projection input width must match aligned K>=32")
    rows = math.prod(value.shape[:-1])
    if rows < 1 or rows > 2**31 - 1 or k > 2**31 - 32:
        raise ValueError("fixed HC projection requires bounded nonempty rows/K")
    return rows, k, weight.shape[0]


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="mlx2_deepseek_v4_hc_fixed_projection_fp32_v1",
        input_names=["x", "weight"],
        output_names=["out"],
        source=SOURCE,
        ensure_row_contiguous=True,
    )


def project(normalized, fn_fp32):
    global _SCHEDULED_CALLS
    import mlx.core as mx

    rows, k, outputs = validate_geometry(normalized, fn_fp32)
    if normalized.dtype != mx.float32 or fn_fp32.dtype != mx.float32:
        raise ValueError("fixed HC projection requires exact FP32 operands")
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("fixed HC projection requires an owned Metal GPU window")
    result = _kernel()(
        inputs=[normalized, fn_fp32],
        template=[("K", k), ("OUTPUTS", outputs)],
        grid=(32, outputs, rows),
        threadgroup=(32, 1, 1),
        output_shapes=[(*normalized.shape[:-1], outputs)],
        output_dtypes=[mx.float32],
    )[0]
    _SCHEDULED_CALLS += 1
    return result


def route_status():
    return {
        "policy": "fixed_fp32",
        "scheduled_calls": _SCHEDULED_CALLS,
        "reduction": "fixed32lane_strided_dot_simd_sum_per_row_output",
        "native_gemv_bitwise_parity_claim": False,
        "qualified": False,
        "counter_scope": "scheduled only; no completion/qualification assertion",
    }


def reference_project(normalized, fn_fp32):
    import numpy as np

    validate_geometry(normalized, fn_fp32)
    if normalized.dtype != np.float32 or fn_fp32.dtype != np.float32:
        raise ValueError("fixed HC reference requires FP32 operands")
    return normalized.astype(np.float64) @ fn_fp32.astype(np.float64).T


def metal_compile_source(*, k=16384, outputs=24, constant_inputs=()):
    if (
        k < 32
        or k % 32
        or outputs not in (4, 24)
        or not set(constant_inputs) <= {"x", "weight"}
    ):
        raise ValueError("invalid fixed HC compiler fixture")
    args = []
    for i, name in enumerate(("x", "weight")):
        space = "constant" if name in constant_inputs else "device"
        args.append(f"{space} const float*{name} [[buffer({i})]]")
    args += [
        "device float*out [[buffer(2)]]",
        "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
        "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]",
    ]
    return (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        + f"constant uint K={k},OUTPUTS={outputs};\nkernel void check("
        + ",".join(args)
        + "){\n"
        + SOURCE
        + "\n}\n"
    )
