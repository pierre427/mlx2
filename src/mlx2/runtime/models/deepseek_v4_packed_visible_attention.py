# SPDX-License-Identifier: Apache-2.0
"""Direct packed-pool visible attention, without whole-history unpack workspace.

See provenance/deepseek-v4-packed-visible-attention.json. No model/cache wiring
or GPU ownership is created here. Arithmetic/traversal follows the independent
ordinary visible FP32 route, with only the pool's load operation replaced.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from types import SimpleNamespace

from .deepseek_v4_packed_kv import LAYOUT_REVISION, unpack_fp8_kv
from .deepseek_v4_visible_attention import (
    HEADER,
    reference_visible_attention,
    validate_geometry,
)
from .deepseek_v4_visible_attention import SOURCE as VISIBLE_SOURCE

_LOCK = threading.RLock()
_READY = False
_FAILURE = None
_CHECK_CONTEXT = threading.local()
_SCHEDULED_CALLS = 0

DECODE = r"""
inline float ds4_e4m3_value(uchar code) {
    const uint exponent=(uint(code)>>3)&15, fraction=uint(code)&7;
    const float magnitude=exponent==0 ? float(fraction)/512.0f
        : (1.0f+float(fraction)/8.0f)*exp2(float(int(exponent)-7));
    return (code&128) ? -magnitude : magnitude;
}
inline float4 ds4_float4(device const float *values) {
    return *((device const float4*)values);
}
inline float4 ds4_float4(constant const float *values) {
    return *((constant const float4*)values);
}
inline uchar4 ds4_uchar4(device const uchar *values) {
    return *((device const uchar4*)values);
}
inline uchar4 ds4_uchar4(constant const uchar *values) {
    return *((constant const uchar4*)values);
}
template<typename Codes, typename Scales, typename Rope>
inline float4 ds4_packed_row4(Codes codes, Scales scales, Rope rope,
                            uint row, uint vector) {
    const uint component=vector*4;
    if(component>=448) {
        return ds4_float4(rope+ulong(row)*64+component-448);
    }
    const uchar4 bits=ds4_uchar4(codes+ulong(row)*448+component);
    const float scale=scales[ulong(row)*7+component/64];
    if(!isfinite(scale) || scale<=0.0f) return float4(NAN);
    return float4(ds4_e4m3_value(bits.x),ds4_e4m3_value(bits.y),
                  ds4_e4m3_value(bits.z),ds4_e4m3_value(bits.w))*scale;
}
"""

_OLD_LOAD = """        device const float4 *values=(device const float4 *)(compressed ? pooled : kv)+ulong(index)*128;
        const float4 v0=values[lane],v1=values[lane+32],v2=values[lane+64],v3=values[lane+96];"""
_NEW_LOAD = """        float4 v0,v1,v2,v3;
        if(compressed) {
            v0=ds4_packed_row4(codes,scales,rope,index,lane);
            v1=ds4_packed_row4(codes,scales,rope,index,lane+32);
            v2=ds4_packed_row4(codes,scales,rope,index,lane+64);
            v3=ds4_packed_row4(codes,scales,rope,index,lane+96);
        } else {
            device const float4 *values=(device const float4 *)kv+ulong(index)*128;
            v0=values[lane];v1=values[lane+32];v2=values[lane+64];v3=values[lane+96];
        }"""
if VISIBLE_SOURCE.count(_OLD_LOAD) != 1:
    raise RuntimeError(
        "ordinary visible-attention load source changed; explicit review required"
    )
SOURCE = VISIBLE_SOURCE.replace(_OLD_LOAD, _NEW_LOAD)


def validate_packed_geometry(q, local, packed, indices, sinks, offset, window, ratio):
    if packed is None:
        pooled = None
    else:
        if (
            packed.layout != LAYOUT_REVISION
            or len(packed.codes.shape) != 3
            or packed.codes.shape[0] != 1
            or packed.codes.shape[-1] != 448
            or packed.scales.shape != (*packed.codes.shape[:-1], 7)
            or packed.rope.shape != (*packed.codes.shape[:-1], 64)
        ):
            raise ValueError(
                "packed visible attention requires exact codes/scales/RoPE pool layout"
            )
        pooled = SimpleNamespace(ndim=3, shape=(1, packed.codes.shape[1], 512))
    validate_geometry(q, local, pooled, indices, sinks, offset, window, ratio)


def reference_packed_visible_attention(
    q, local, packed, indices, sinks, *, offset, window, ratio
):
    import numpy as np

    validate_packed_geometry(q, local, packed, indices, sinks, offset, window, ratio)
    pooled = None if packed is None else unpack_fp8_kv(packed, xp=np)
    return reference_visible_attention(
        q, local, pooled, indices, sinks, offset=offset, window=window, ratio=ratio
    )


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="mlx2_deepseek_v4_packed_pool_visible_fp32_v1",
        input_names=["q", "kv", "codes", "scales", "rope", "topk", "sinks", "params"],
        output_names=["out"],
        header=HEADER + DECODE,
        source=SOURCE,
        ensure_row_contiguous=True,
    )


def packed_visible_attention(
    q, local, packed, indices, sinks, *, offset, window, ratio
):
    global _SCHEDULED_CALLS
    import mlx.core as mx

    validate_packed_geometry(q, local, packed, indices, sinks, offset, window, ratio)
    if any(x.dtype != mx.float32 for x in (q, local, sinks)):
        raise ValueError("packed visible attention requires FP32 queries/local/sinks")
    if packed is not None and (
        packed.codes.dtype != mx.uint8
        or packed.scales.dtype != mx.float32
        or packed.rope.dtype != mx.float32
    ):
        raise ValueError("packed pool physical dtypes differ from exact layout")
    if indices is not None and indices.dtype not in (mx.int32, mx.uint32):
        raise ValueError("packed visible attention requires int32 pooled indices")
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError(
            "packed visible attention requires an owned Metal GPU window"
        )
    if not getattr(_CHECK_CONTEXT, "active", False):
        self_check()
    length = q.shape[2]
    heads = q.shape[1]
    count = 0 if packed is None else packed.codes.shape[1]
    codes = mx.zeros((1, 448), dtype=mx.uint8) if packed is None else packed.codes[0]
    scales = mx.ones((1, 7), dtype=mx.float32) if packed is None else packed.scales[0]
    rope = mx.zeros((1, 64), dtype=mx.float32) if packed is None else packed.rope[0]
    width = 0 if indices is None else indices.shape[-1]
    chosen = (
        mx.zeros((1, 1), dtype=mx.uint32)
        if indices is None
        else mx.sort(indices[0].astype(mx.uint32), axis=-1)
    )
    params = mx.array(
        [offset, window, ratio, count, local.shape[2], length, width], dtype=mx.int32
    )
    out = _kernel()(
        inputs=[q[0], local[0, 0], codes, scales, rope, chosen, sinks, params],
        template=[("HEADS", heads)],
        grid=(((heads + 3) // 4) * 128, length, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(heads, length, 512)],
        output_dtypes=[mx.float32],
    )[0]
    if not getattr(_CHECK_CONTEXT, "active", False):
        _SCHEDULED_CALLS += 1
    return out[None]


def workspace_bytes(*, heads, length, topk=0):
    """Output, sorted index rows and fixed params; independent of total pool."""
    if (
        any(type(x) is not int or x < 0 for x in (heads, length, topk))
        or not heads
        or not length
        or topk > 512
    ):
        raise ValueError("invalid packed attention workspace geometry")
    return heads * length * 512 * 4 + length * topk * 4 + 7 * 4


def metal_compile_source(*, constant_inputs=()):
    allowed = {"codes", "scales", "rope"}
    if not set(constant_inputs) <= allowed:
        raise ValueError("unsupported compile fixture constant input")
    source = (
        HEADER
        + DECODE
        + "\nconstant uint HEADS=64;\nkernel void compile_probe("
        + "device const float*q [[buffer(0)]],device const float*kv [[buffer(1)]],"
        + "device const uchar*codes [[buffer(2)]],device const float*scales [[buffer(3)]],"
        + "device const float*rope [[buffer(4)]],device const uint*topk [[buffer(5)]],"
        + "device const float*sinks [[buffer(6)]],device const int*params [[buffer(7)]],"
        + "device float*out [[buffer(8)]],uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],"
        + "uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]) {\n"
        + SOURCE
        + "\n}\n"
    )
    for name in constant_inputs:
        dtype = "uchar" if name == "codes" else "float"
        source = source.replace(
            f"device const {dtype}*{name} ", f"constant const {dtype}*{name} "
        )
    return source


def self_check():
    """Bounded first-use bitwise gate under caller-owned GPU, not qualification."""
    global _READY, _FAILURE
    import mlx.core as mx
    import numpy as np

    from .deepseek_v4_packed_kv import PackedKV, pack_fp8_kv
    from .deepseek_v4_visible_attention import visible_attention

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError(
            "packed visible attention requires an owned Metal GPU window"
        )
    with _LOCK:
        if _FAILURE is not None:
            raise RuntimeError(
                "packed visible first-use numerical gate previously failed"
            ) from _FAILURE
        if _READY:
            return
        try:
            _CHECK_CONTEXT.active = True
            rng = np.random.default_rng(817928)
            q = mx.array(rng.normal(0, 0.2, (1, 4, 3, 512)).astype(np.float32))
            local = mx.array(rng.normal(0, 0.3, (1, 1, 6, 512)).astype(np.float32))
            raw = pack_fp8_kv(rng.normal(0, 0.3, (1, 4, 512)).astype(np.float32), xp=np)
            packed = PackedKV(*(mx.array(x) for x in (raw.codes, raw.scales, raw.rope)))
            sinks = mx.array(rng.normal(0, 0.3, (4,)).astype(np.float32))
            top = mx.array([[[3, 0, 2], [1, 3, 0], [2, 1, 3]]], dtype=mx.uint32)
            for ratio, offset, indices in ((4, 13, top), (128, 509, None)):
                actual = packed_visible_attention(
                    q,
                    local,
                    packed,
                    indices,
                    sinks,
                    offset=offset,
                    window=4,
                    ratio=ratio,
                )
                ordinary = visible_attention(
                    q,
                    local,
                    unpack_fp8_kv(packed, xp=mx),
                    indices,
                    sinks,
                    offset=offset,
                    window=4,
                    ratio=ratio,
                )
                mx.eval(actual, ordinary)
                if not np.array_equal(
                    np.asarray(actual).view(np.uint32),
                    np.asarray(ordinary).view(np.uint32),
                ):
                    raise RuntimeError(
                        "direct packed decode/attention differs from unpackall visibleFP32 bits"
                    )
            _READY = True
        except Exception as error:
            _FAILURE = error
            raise RuntimeError(
                "packed visible first-use numerical gate failed"
            ) from error
        finally:
            _CHECK_CONTEXT.active = False


def route_status():
    return {
        "policy": "packed_visible_rows_fp32",
        "qualified": False,
        "synthetic_bitwise_self_check_passed": _READY,
        "failed": _FAILURE is not None,
        "scheduled_attention_calls": _SCHEDULED_CALLS,
        "pool_fp32_materialization": False,
        "arithmetic": "same chronological FP32 visible attention; direct E4M3FN/scales/RoPE pool loads",
    }
