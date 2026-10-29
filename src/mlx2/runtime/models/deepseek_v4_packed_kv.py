"""Opt-in physical mixed FP8/FP32 KV storage primitives; not a serving route.

Original implementation of DS4 E4M3FN QAT and oMLX's code/scale separation.
See provenance/deepseek-v4-packed-kv.json. No backend import until called.
"""
from dataclasses import dataclass

LAYOUT_REVISION = "deepseek-v4-e4m3fn-g64-fp32-scale-rope64-fp32.v1"


def _owned(xp, value):
    return (xp.asarray(value, copy=True) if xp.__name__ == "mlx.core"
            else xp.array(value, copy=True))


@dataclass(frozen=True)
class PackedKV:
    codes: object
    scales: object
    rope: object
    layout: str = LAYOUT_REVISION

    @property
    def nbytes(self):
        return self.codes.nbytes + self.scales.nbytes + self.rope.nbytes


def pack_fp8_kv(value, *, xp):
    """QAT then physically pack non-RoPE448 into uint8; preserve FP32 tail64.

    Inputs must be finite FP32, trailing width512. This reference primitive
    uses portable tensor operations (MLX CPU or Metal), with group64 scales.
    Caller validates finiteness at its bounded probe/load boundary.
    """
    if value.shape[-1] != 512 or value.dtype != xp.float32:
        raise ValueError("packed DeepSeek V4 KV requires FP32 width512")
    groups = value[..., :448].reshape(*value.shape[:-1], 7, 64)
    amax = xp.maximum(xp.max(xp.abs(groups), axis=-1, keepdims=True), 1e-4)
    scales = xp.power(2., xp.ceil(xp.log2(amax / 448.))).astype(xp.float32)
    scaled = xp.clip(groups / scales, -448., 448.)
    magnitude = xp.abs(scaled)
    exponent = xp.floor(xp.log2(xp.maximum(magnitude, 2. ** -6)))
    step = xp.power(2., xp.maximum(exponent - 3, -9))
    rounded = xp.round(magnitude / step) * step
    exponent = xp.floor(xp.log2(xp.maximum(rounded, 2. ** -6)))
    normal = (exponent + 7) * 8 + xp.round((rounded / xp.power(2., exponent) - 1) * 8)
    code = xp.where(rounded < 2. ** -6, xp.round(rounded * 512), normal)
    # Preserve negative zero via IEEE sign bit, including values rounded to zero.
    sign = (scaled.view(xp.uint32) >> 31) << 7
    codes = (code.astype(xp.uint32) | sign).astype(xp.uint8)
    return PackedKV(_owned(xp, codes.reshape(*value.shape[:-1], 448)),
                    _owned(xp, scales[..., 0]), _owned(xp, value[..., 448:]))


def unpack_fp8_kv(packed, *, xp):
    if packed.layout != LAYOUT_REVISION:
        raise ValueError("packed KV layout mismatch")
    codes, scales, rope = packed.codes, packed.scales, packed.rope
    if (codes.dtype != xp.uint8 or codes.shape[-1] != 448
            or scales.shape != (*codes.shape[:-1], 7)
            or rope.shape != (*codes.shape[:-1], 64)
            or scales.dtype != xp.float32 or rope.dtype != xp.float32):
        raise ValueError("invalid packed KV geometry")
    groups = codes.reshape(*codes.shape[:-1], 7, 64).astype(xp.uint32)
    exponent, fraction = (groups >> 3) & 15, groups & 7
    normal = (1 + fraction.astype(xp.float32) / 8) * xp.power(2., exponent.astype(xp.float32) - 7)
    magnitude = xp.where(exponent == 0, fraction.astype(xp.float32) / 512, normal)
    signed = xp.where((groups & 128) != 0, -magnitude, magnitude)
    values = (signed * scales[..., None]).reshape(*codes.shape[:-1], 448)
    return xp.concatenate((values, rope), axis=-1)
