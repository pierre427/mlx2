# SPDX-License-Identifier: MIT
"""Metal implementation of the exact V4.1 native packed-QAT codec.

Original mlx2 code; see provenance/deepseek-v41-packed-fused-consumers.json.
``pack`` emits the same original FP8/FP4 codes and E8M0/E4M3 scale bytes as
``deepseek_v41_packed_qat.produce`` in one kernel; it is a producer applied to
the pre-QAT FP32 value, never a requantizer of decoded cache values.
``decode``/``gather_decode`` rebuild the same BF16-rounded FP32 values as the
reference decoder with integer bit construction. A first-use gate requires
bitwise agreement with the MLX-operation reference on the owned GPU before any
serving call. Unsupported devices or geometries fail closed.
"""

import hashlib
from functools import lru_cache

import mlx.core as mx

from .deepseek_v41_quant import FUSED_HEADER

CODEC = "fused_metal_v1"
_KINDS = {"fp8": 0, "index_fp4": 1, "compressed_fp4": 2}

HEADER = FUSED_HEADER + r"""
inline uint v41_e4m3_bits(float q) {
    const uint sign=(as_type<uint>(q)>>31)<<7;
    const float a=abs(q);
    if(a==0.0f) return sign;
    const uint bits=as_type<uint>(a);
    const int e=int((bits>>23)&255u)-127;
    if(e>=-6) return sign|(uint(e+7)<<3)|((bits>>20)&7u);
    return sign|uint(a*512.0f);
}
inline float v41_e4m3_value(uint b) {
    const uint e=(b>>3)&15u, m=b&7u;
    const float a=e==0u ? float(m)*0.001953125f : as_type<float>(((e+120u)<<23)|(m<<20));
    return (b&128u)!=0u ? -a : a;
}
inline float v41_bf16(float x) {
    uint bits=as_type<uint>(x);
    return as_type<float>((bits+0x7fffu+((bits>>16)&1u))&0xffff0000u);
}
inline float v41_decode(uint code, uint scale_byte, uint kind) {
    if(kind==0u) {
        const float magnitude=abs(v41_e4m3_value(code));
        const float value=magnitude*as_type<float>(scale_byte<<23);
        return v41_bf16(as_type<float>(as_type<uint>(value)|((code&128u)<<24)));
    }
    const uint level=code&7u, sign=(code&8u)<<28;
    if(level==0u) return as_type<float>(sign);
    const int exponent=int(level>>1)-1;
    const uint half_bit=(level>1u)&&((level&1u)!=0u) ? 1u : 0u;
    if(kind==2u) {
        const float base=as_type<float>((uint(exponent+127)<<23)|(half_bit<<22));
        const float value=base*v41_e4m3_value(scale_byte);
        return v41_bf16(as_type<float>(as_type<uint>(value)|sign));
    }
    const int biased=int(scale_byte)+exponent;
    uint bits;
    if(biased>=1) bits=(uint(biased)<<23)|(half_bit<<22);
    else {
        const int shift=1-biased;
        bits=shift>=25 ? 0u : (((1u<<23)|(half_bit<<22))>>uint(shift));
    }
    return v41_bf16(as_type<float>(bits|sign));
}
"""

PACK_SOURCE = r"""
const uint i=thread_position_in_grid.x;
if(i>=elements[0]) return;
const float v=value[i];
float maximum=abs(v);
for(uint step=BLOCK/2;step>0;step/=2) maximum=max(maximum,simd_shuffle_xor(maximum,step));
float scale;
uint scale_byte;
if(KIND==2) {
    scale=v41_e4m3_round(max(maximum,6.0f/512.0f)/6.0f);
    scale_byte=v41_e4m3_bits(scale);
} else {
    maximum=max(maximum,KIND==0 ? 1e-4f : 6.0f*1.1754943508222875e-38f);
    const uint bits=as_type<uint>(maximum*inverse[0]);
    scale_byte=((bits>>23)&255u)+uint((bits&0x7fffffu)!=0u);
    scale=as_type<float>(scale_byte<<23);
}
const float normalized=v/scale;
if(KIND==0) {
    codes[i]=uint8_t(v41_e4m3_bits(v41_e4m3_round(clamp(normalized,-448.0f,448.0f))));
} else {
    const float c=clamp(normalized,-6.0f,6.0f);
    const float a=abs(c);
    uint code=uint(a>0.25f)+uint(a>1.25f)+uint(a>2.5f)+uint(a>5.0f)
             +uint(a>=0.75f)+uint(a>=1.75f)+uint(a>=3.5f);
    code|=uint(c<0.0f)<<3;
    const uint next=simd_shuffle_down(code,1u);
    if((i&1u)==0u) codes[i/2]=uint8_t(code|(next<<4));
}
if(i%BLOCK==0u) scales[i/BLOCK]=uint8_t(scale_byte);
"""

DECODE_SOURCE = r"""
const uint i=thread_position_in_grid.x;
if(i>=elements[0]) return;
const uint d=i%DIM;
const uint r=i/DIM;
const uint row=ROW_EXPR;
const uint block=KIND==2 ? 16u : 32u;
uint code;
if(KIND==0) code=uint(codes[row*DIM+d]);
else {
    const uint byte=uint(codes[row*(DIM/2)+d/2]);
    code=(d&1u)!=0u ? (byte>>4) : (byte&15u);
}
out[i]=v41_decode(code,uint(scales[row*(DIM/block)+d/block]),uint(KIND));
"""


def implementation_identity():
    payload = "\n".join((CODEC, HEADER, PACK_SOURCE, DECODE_SOURCE))
    return hashlib.sha256(payload.encode()).hexdigest()


@lru_cache(maxsize=1)
def _pack_kernel():
    return mx.fast.metal_kernel(
        name="mlx2_v41_packed_qat_pack_v1", input_names=["value", "inverse", "elements"],
        output_names=["codes", "scales"], header=HEADER, source=PACK_SOURCE,
        ensure_row_contiguous=True)


@lru_cache(maxsize=2)
def _decode_kernel(gather):
    return mx.fast.metal_kernel(
        name=f"mlx2_v41_packed_qat_decode_{int(gather)}_v1",
        input_names=["codes", "scales", "elements", *(["ids"] if gather else [])],
        output_names=["out"], header=HEADER,
        source=DECODE_SOURCE.replace("ROW_EXPR", "uint(ids[r])" if gather else "r"),
        ensure_row_contiguous=True)


def _owned_gpu():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise ValueError("V4.1 fused packed codec requires an owned Metal GPU window")


def _pack(x, kind):
    from .deepseek_v41_packed_qat import PackedQAT

    code = _KINDS[kind]
    block = 16 if code == 2 else 32
    codes_shape = (*x.shape[:-1], x.shape[-1] // (1 if code == 0 else 2))
    scales_shape = (*x.shape[:-1], x.shape[-1] // block)
    codes, scales = _pack_kernel()(
        # Sizes are runtime operands: decode geometry grows every step and
        # must not instantiate a new pipeline per context length.
        inputs=[x, mx.array([1 / (448 if code == 0 else 6)], mx.float32), mx.array([x.size], mx.uint32)],
        template=[("BLOCK", block), ("KIND", code)],
        grid=(x.size, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[codes_shape, scales_shape], output_dtypes=[mx.uint8, mx.uint8])
    return PackedQAT(codes, scales, kind, CODEC)


def _decode(packed, ids=None):
    code = _KINDS[packed.kind]
    dim = packed.shape[-1]
    rows = packed.codes.size // packed.codes.shape[-1]
    shape = packed.shape if ids is None else (*packed.shape[:-2], *ids.shape[1:], dim)
    count = (rows if ids is None else ids.size) * dim
    inputs = [packed.codes, packed.scales, mx.array([count], mx.uint32)] + ([] if ids is None else [ids[0]])
    return _decode_kernel(ids is not None)(
        inputs=inputs,
        template=[("DIM", dim), ("KIND", code)],
        grid=(max(1, count), 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[shape], output_dtypes=[mx.float32])[0]


def _check_codec(codec):
    if codec != CODEC:
        raise ValueError("unknown V4.1 packed codec")
    _owned_gpu()
    first_use_check()


def produce(x, kind, *, codec=CODEC):
    _check_codec(codec)
    if (kind not in _KINDS or x.dtype != mx.float32 or not x.size
            or x.shape[-1] % 32 or x.size % 32):
        raise ValueError("V4.1 fused packed producer requires FP32 whole SIMD groups")
    return _pack(x, kind)


def decode(packed):
    _check_codec(packed.codec)
    if not packed.codes.size:
        return mx.zeros(packed.shape, mx.float32)
    return _decode(packed)


def gather_decode(packed, indices):
    _check_codec(packed.codec)
    if (len(packed.shape) != 3 or packed.shape[0] != 1 or indices.ndim != 3
            or indices.shape[0] != 1 or indices.dtype not in (mx.int32, mx.uint32)):
        raise ValueError("native QAT gather requires B1 query/selection IDs")
    if not indices.size:
        return mx.zeros((1, *indices.shape[1:], packed.shape[-1]), mx.float32)
    return _decode(packed, indices)


_READY = False


def first_use_check():
    """Bitwise gate against the MLX-operation reference codec on this device."""
    global _READY
    if _READY:
        return
    import numpy as np

    from . import deepseek_v41_packed_qat as reference

    rng = np.random.default_rng(41890)
    levels = np.array([0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, -0.0], np.float32)
    edge = np.tile(np.concatenate((levels, -levels)), (4, 16))
    samples = [edge * scale for scale in (2.0**-124, 2.0**-40, 2.0**-12, 1.0, 2.0**40)]
    samples += [rng.normal(size=(7, 512)).astype(np.float32) * scale for scale in (2.0**-30, 1e-3, 1.0, 37.0)]
    samples.append(np.zeros((2, 512), np.float32))
    for values in samples:
        x = mx.array(values)
        for kind in _KINDS:
            want = reference.produce(x, kind)
            got = _pack(x, kind)
            ids = mx.array(rng.integers(0, values.shape[0], size=(1, 3, 5)).astype(np.uint32))
            pool = reference.PackedQAT(want.codes[None], want.scales[None], kind)
            pairs = ((got.codes, want.codes), (got.scales, want.scales),
                     (_decode(want), reference.reference_decode(want)),
                     (_decode(reference.PackedQAT(pool.codes, pool.scales, kind), ids),
                      reference.gather_decode(pool, ids)))
            for actual, expected in pairs:
                mx.eval(actual, expected)
                if np.array(actual).tobytes() != np.array(expected).tobytes():
                    raise ValueError("V4.1 fused packed codec first-use gate failed: " + kind)
    _READY = True


def status():
    return {"codec": CODEC, "first_use_checked": _READY,
            "implementation": implementation_identity(),
            "observation": "synthetic bitwise gate only; no checkpoint qualification"}
