# SPDX-License-Identifier: MIT
"""Original QAT codes/scales, captured before dequantization or BF16 stores.

This is a producer, never a packer for previously decoded QAT values. See
provenance/deepseek-v41-csa2-ced.json. Decode scratch is not persistent state.
"""

from dataclasses import dataclass

import mlx.core as mx

from .deepseek_v41_quant import fp8_round, pow2_scale


@dataclass(frozen=True)
class PackedQAT:
    codes: object
    scales: object
    kind: str

    @property
    def shape(self):
        return (*self.codes.shape[:-1], self.codes.shape[-1] * (1 if self.kind == "fp8" else 2))

    @property
    def nbytes(self):
        return self.codes.nbytes + self.scales.nbytes

    def sliced(self, index):
        return PackedQAT(self.codes[index], self.scales[index], self.kind)


def produce(x, kind):
    """Apply the reference QAT law once and retain its original quantizer output."""
    if kind not in ("fp8", "index_fp4", "compressed_fp4"):
        raise ValueError("unknown V4.1 packed QAT format")
    block = 16 if kind == "compressed_fp4" else 32
    if x.dtype != mx.float32 or x.shape[-1] % block:
        raise ValueError("packed QAT requires FP32 aligned producer values")
    z = x.reshape(*x.shape[:-1], x.shape[-1] // block, block)
    maximum = mx.max(mx.abs(z), axis=-1, keepdims=True)
    if kind == "fp8":
        scale = pow2_scale(mx.maximum(maximum, 1e-4) * mx.array(1 / 448, mx.float32))
        codes = mx.to_fp8(mx.clip(z / scale, -448, 448)).reshape(x.shape)
    else:
        scale = (fp8_round(mx.maximum(maximum, 6 * 2.0**-9) / 6)
                 if kind == "compressed_fp4" else
                 pow2_scale(mx.maximum(maximum, 6 * 2.0**-126) * mx.array(1 / 6, mx.float32)))
        normalized = mx.clip(z / scale, -6, 6)
        a = mx.abs(normalized)
        code = mx.zeros(a.shape, mx.uint8)
        for threshold in (0.25, 1.25, 2.5, 5.0):
            code = code + (a > threshold).astype(mx.uint8)
        for threshold in (0.75, 1.75, 3.5):
            code = code + (a >= threshold).astype(mx.uint8)
        code = (code | ((normalized < 0).astype(mx.uint8) << 3)).reshape(x.shape)
        codes = code[..., ::2] | (code[..., 1::2] << 4)
    scales = (mx.to_fp8(scale) if kind == "compressed_fp4" else
              ((scale.view(mx.uint32) >> 23) & 255).astype(mx.uint8))
    return PackedQAT(codes, scales.squeeze(-1), kind)


def decode(packed):
    """Reconstruct the model's BF16-rounded FP32 values, including zero signs."""
    if packed.kind == "fp8":
        values = mx.from_fp8(packed.codes, mx.float32)
    else:
        codes = mx.stack((packed.codes & 15, packed.codes >> 4), axis=-1).reshape(packed.shape)
        levels = mx.array([0., .5, 1., 1.5, 2., 3., 4., 6.], mx.float32)
        values = levels[codes & 7] * mx.where((codes & 8) != 0, -1., 1.)
    scales = (mx.from_fp8(packed.scales, mx.float32) if packed.kind == "compressed_fp4" else
              (packed.scales.astype(mx.uint32) << 23).view(mx.float32))
    block = 16 if packed.kind == "compressed_fp4" else 32
    values = (values.reshape(*values.shape[:-1], values.shape[-1] // block, block) * scales[..., None]).reshape(packed.shape)
    return values.astype(mx.bfloat16).astype(mx.float32)


def gather_decode(packed, indices):
    """Gather original rows before decoding; no full-history float staging."""
    if (len(packed.shape) != 3 or packed.shape[0] != 1 or indices.ndim != 3
            or indices.shape[0] != 1 or indices.dtype not in (mx.int32, mx.uint32)):
        raise ValueError("native QAT gather requires B1 query/selection IDs")
    return decode(PackedQAT(packed.codes[0, indices[0]][None],
                            packed.scales[0, indices[0]][None], packed.kind))


def validate(packed, *, shape, kind, xp):
    block = 16 if kind == "compressed_fp4" else 32
    if (packed.kind != kind or packed.shape != shape
            or packed.codes.dtype != xp.uint8 or packed.scales.dtype != xp.uint8
            or packed.scales.shape != (*shape[:-1], shape[-1] // block)):
        raise ValueError("invalid V4.1 packed QAT plane geometry")


def concatenate(a, b, axis):
    if a.kind != b.kind:
        raise ValueError("mixed packed QAT formats")
    return PackedQAT(mx.concatenate((a.codes, b.codes), axis),
                     mx.concatenate((a.scales, b.scales), axis), a.kind)
