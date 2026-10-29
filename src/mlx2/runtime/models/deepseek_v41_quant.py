# SPDX-License-Identifier: MIT
"""V4.1 QAT graph; no Hadamard and no rotary-tail exemption.

Derived from the official kernels and Rapid-MLX fakequant.py; see provenance.
Each activation store rounds to BF16 separately from its physical FP32 storage.
"""

import mlx.core as mx


def fp8_round(x):
    return mx.from_fp8(mx.to_fp8(mx.clip(x, -448, 448)), mx.float32)


def pow2_scale(x):
    bits = x.astype(mx.float32).view(mx.uint32)
    exponent = ((bits >> 23) & 255).astype(mx.int32) + ((bits & 0x7FFFFF) != 0).astype(
        mx.int32
    )
    return (exponent.astype(mx.uint32) << 23).view(mx.float32)


def fp4_round(x):
    a = mx.abs(x)
    index = mx.zeros(a.shape, mx.int32)
    for threshold in (0.25, 1.25, 2.5, 5.0):
        index = index + (a > threshold).astype(mx.int32)
    for threshold in (0.75, 1.75, 3.5):
        index = index + (a >= threshold).astype(mx.int32)
    return (
        mx.sign(x)
        * mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], mx.float32)[index]
    )


def quantize(x, kind):
    if kind not in ("fp8", "index_fp4", "compressed_fp4"):
        raise ValueError("unknown V4.1 QAT format")
    block = 16 if kind == "compressed_fp4" else 32
    if x.shape[-1] % block:
        raise ValueError("V4.1 QAT block alignment mismatch")
    z = x.astype(mx.float32).reshape(*x.shape[:-1], -1, block)
    maximum = mx.max(mx.abs(z), axis=-1, keepdims=True)
    if kind == "fp8":
        scale = pow2_scale(mx.maximum(maximum, 1e-4) * mx.array(1 / 448, mx.float32))
        result = fp8_round(z / scale) * scale
    else:
        scale = (
            fp8_round(mx.maximum(maximum, 6 * 2.0**-9) / 6)
            if kind == "compressed_fp4"
            else pow2_scale(
                mx.maximum(maximum, 6 * 2.0**-126) * mx.array(1 / 6, mx.float32)
            )
        )
        result = fp4_round(mx.clip(z / scale, -6, 6)) * scale
    return result.reshape(x.shape).astype(x.dtype)


# Sister-session QAT fusion mechanism, adapted to V4.1's no-Hadamard graph
# and IEEE exponent scale law. Never call V4's index_fp4() on V4.1 tensors.
FUSED_HEADER = r"""
#pragma clang fp contract(off)
inline float v41_e4m3_round(float value) {
    const float a=min(abs(value),448.0f);
    const int exponent=max(int((as_type<uint>(a)>>23)&255u)-127-3,-9);
    const float step=as_type<float>(uint(exponent+127)<<23);
    return copysign(min(rint(a/step)*step,448.0f),value);
}
inline float v41_e2m1_round(float value) {
    const float a=min(abs(value),6.0f);
    float q;
    if(a<=0.25f) q=0.0f;
    else if(a<0.75f) q=0.5f;
    else if(a<=1.25f) q=1.0f;
    else if(a<1.75f) q=1.5f;
    else if(a<=2.5f) q=2.0f;
    else if(a<3.5f) q=3.0f;
    else if(a<=5.0f) q=4.0f;
    else q=6.0f;
    return value==0.0f ? 0.0f : copysign(q,value);
}
"""
FUSED_SOURCE = r"""
const uint i=thread_position_in_grid.x;
if(i>=ELEMENTS) return;
const float v=value[i];
float maximum=abs(v);
for(uint step=BLOCK/2;step>0;step/=2) maximum=max(maximum,simd_shuffle_xor(maximum,step));
float scale;
if(KIND==2) {
    scale=v41_e4m3_round(max(maximum,6.0f/512.0f)/6.0f);
} else {
    maximum=max(maximum,KIND==0 ? 1e-4f : 6.0f*1.1754943508222875e-38f);
    const uint bits=as_type<uint>(maximum*inverse[0]);
    const uint exponent=((bits>>23)&255u)+uint((bits&0x7fffffu)!=0u);
    scale=as_type<float>(exponent<<23);
}
const float normalized=v/scale;
const float quantized=KIND==0 ? v41_e4m3_round(clamp(normalized,-448.0f,448.0f)) : v41_e2m1_round(clamp(normalized,-6.0f,6.0f));
out[i]=quantized*scale;
"""
_FUSED_KINDS = {"fp8": 0, "index_fp4": 1, "compressed_fp4": 2}
_FUSED_READY = False
_FUSED_SCHEDULED = 0


def _fused(value, kind):
    # Function cache belongs to this module, not to the per-call wrapper.
    global _FUSED_KERNEL, _FUSED_SCHEDULED
    if "_FUSED_KERNEL" not in globals():
        _FUSED_KERNEL = mx.fast.metal_kernel(
            name="mlx2_v41_qat_epilogue_v1",
            input_names=["value", "inverse"],
            output_names=["out"],
            header=FUSED_HEADER,
            source=FUSED_SOURCE,
            ensure_row_contiguous=True,
        )
    code = _FUSED_KINDS[kind]
    block = 16 if code == 2 else 32
    result = _FUSED_KERNEL(
        inputs=[value, mx.array([1 / (448 if code == 0 else 6)], mx.float32)],
        template=[("ELEMENTS", value.size), ("BLOCK", block), ("KIND", code)],
        grid=(value.size, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[value.shape],
        output_dtypes=[mx.float32],
    )[0]
    _FUSED_SCHEDULED += 1
    return result


def fused_quantize(value, kind):
    global _FUSED_READY
    if (
        kind not in _FUSED_KINDS
        or value.dtype != mx.float32
        or not value.size
        or value.shape[-1] % 32
    ):
        raise ValueError(
            "V4.1 fused QAT requires finite FP32 values in whole SIMD groups"
        )
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise ValueError("V4.1 fused QAT requires an owned Metal GPU window")
    if not _FUSED_READY:
        # First-use check happens only when the selected route owns the GPU.
        import numpy as np

        levels = np.array(
            [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, -0.0],
            np.float32,
        )
        values = np.tile(np.concatenate((levels, -levels)), (2, 4))
        for fmt in _FUSED_KINDS:
            for scale in (2.0**-40, 1.0, 2.0**40):
                sample = mx.array(values * scale)
                expected = quantize(sample, fmt)
                actual = _fused(sample, fmt)
                mx.eval(expected, actual)
                if not np.array_equal(
                    np.array(actual).view(np.uint32), np.array(expected).view(np.uint32)
                ):
                    raise ValueError(
                        "V4.1 fused QAT first-use reference gate failed: " + fmt
                    )
        _FUSED_READY = True
    return _fused(value, kind)


def fused_status():
    return {
        "policy": "fused_fp32",
        "first_use_checked": _FUSED_READY,
        "scheduled_calls": _FUSED_SCHEDULED,
        "observation": "scheduling and synthetic gate only; no checkpoint qualification",
    }


def compilation_source(kind):
    code = _FUSED_KINDS[kind]
    constants = (
        f"constant uint ELEMENTS=128,BLOCK={16 if code == 2 else 32},KIND={code};\n"
    )
    return (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        + FUSED_HEADER
        + constants
        + "kernel void v41_qat(device const float* value [[buffer(0)]],device const float* inverse [[buffer(1)]],device float* out [[buffer(2)]],uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"
        + FUSED_SOURCE
        + "\n}\n"
    )
