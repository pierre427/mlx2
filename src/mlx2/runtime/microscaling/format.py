# SPDX-License-Identifier: MIT
# SC-MXFP4 codec derived from Shinyoung Kang's MIT mxfp4-int8-m5 implementation.
# See provenance/integer-exact-microscaling.json for pinned source and license.
"""Immutable CPU-only SC-MXFP4 packing; never selects a serving route.

An existing MXFP4 tensor is folded losslessly. Quantizing floating weights is
an explicit offline artifact conversion, not a runtime eligibility fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np


class SCFormatError(ValueError):
    """An array cannot satisfy the exact SC-MXFP4 operand contract."""


@dataclass(frozen=True)
class SCFormat:
    """One supported format; no process-global mutable code tables."""

    block: int = 32
    code_mag: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8, 12)
    mult: tuple[int, ...] = (1, 2, 4, 8)
    bc_bits: int = 2

    def __post_init__(self):
        if (self.block, self.code_mag, self.mult, self.bc_bits) != (
            32, (0, 1, 2, 3, 4, 6, 8, 12), (1, 2, 4, 8), 2
        ):
            raise SCFormatError("only standard-compatible SC-MXFP4 is supported")

    @property
    def per_byte(self) -> int:
        return 4

    @property
    def max_abs_operand(self) -> int:
        return 96

    @property
    def max_reduction(self) -> int:
        return (2**31 - 1) // (127 * self.max_abs_operand)

    @property
    def revision(self) -> str:
        return "sc-mxfp4-block32-shift2-e2m1-v1"

    def bits_per_weight(self, k: int) -> float:
        _validate_k(k)
        return 4 + self.bc_bits / self.block + 32 / k

    def metal_header(self) -> str:
        return (
            "#define SC_BLOCK 32\n#define SC_BC_BITS 2\n#define SC_PER_BYTE 4\n"
            "inline int sc_mag(uint i) { return int((0xC8643210u >> (4 * i)) & 0xFu); }\n"
            "inline int sc_mult(uint i) { return 1 << i; }\n"
        )


SC_FORMAT = SCFormat()


def _validate_k(k: int) -> None:
    if type(k) is not int or k <= 0 or k % 128:
        raise SCFormatError("K must be a positive multiple of 128")
    if k > SC_FORMAT.max_reduction:
        raise SCFormatError("K can overflow conservative INT32 accumulation")


def _validate_shape(shape: tuple[int, ...]) -> None:
    if len(shape) not in (2, 3) or any(type(d) is not int or d <= 0 for d in shape):
        raise SCFormatError("weights must be a nonempty dense [N,K] or grouped [E,N,K] array")
    _validate_k(shape[-1])


def _owned_readonly(value, dtype, shape, name):
    a = np.asarray(value)
    if a.dtype != np.dtype(dtype) or a.shape != tuple(shape):
        raise SCFormatError(f"{name} must have dtype {np.dtype(dtype)} and shape {tuple(shape)}")
    # Copy before freezing, otherwise a writable owner could mutate an alias.
    out = np.array(a, dtype=dtype, order="C", copy=True)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class PackedSCWeight:
    """Owned read-only numpy payload for one dense or grouped weight tensor."""

    codes: np.ndarray
    bcodes: np.ndarray
    row_scales: np.ndarray
    shape: tuple[int, ...]
    stats: Mapping[str, int | float | str] = field(default_factory=dict)
    format: SCFormat = SC_FORMAT

    def __post_init__(self):
        shape = tuple(self.shape)
        _validate_shape(shape)
        if self.format != SC_FORMAT:
            raise SCFormatError("unsupported format descriptor")
        codes = _owned_readonly(self.codes, np.uint32, (*shape[:-1], shape[-1] // 8), "codes")
        bc = _owned_readonly(self.bcodes, np.uint8, (*shape[:-1], shape[-1] // 128), "bcodes")
        ws = _owned_readonly(self.row_scales, np.float32, shape[:-1], "row_scales")
        if not np.all(np.isfinite(ws)) or np.any(ws <= 0):
            raise SCFormatError("row scales must be positive finite float32 values")
        exponent = np.log2(ws.astype(np.float64))
        if not np.array_equal(exponent, np.rint(exponent)):
            raise SCFormatError("row scales must be powers of two")
        exponents = exponent[..., None] + 1 + _unpack_shifts(bc, shape)
        if np.any(exponents < -127) or np.any(exponents > 127):
            raise SCFormatError("row scale and shifts exceed finite E8M0 range")
        # A finite E8M0 byte does not imply its FP4 value fits the F32
        # scaling epilogue (for example, 4 * 2**126 overflows). Check the
        # represented values without materializing an unpacked float tensor.
        word_max = np.zeros(codes.shape, np.uint8)
        magnitudes = np.asarray(SC_FORMAT.code_mag, np.uint8)
        for lane in range(8):
            np.maximum(word_max, magnitudes[(codes >> (4 * lane)) & 7], out=word_max)
        block_max = word_max.reshape(*shape[:-1], shape[-1] // 32, 4).max(axis=-1)
        max_operand = np.max(block_max.astype(np.int16) << _unpack_shifts(bc, shape), axis=-1)
        if np.any(max_operand.astype(np.float64) * ws.astype(np.float64) > np.finfo(np.float32).max):
            raise SCFormatError("decoded SC-MXFP4 weights exceed finite float32 range")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "codes", codes)
        object.__setattr__(self, "bcodes", bc)
        object.__setattr__(self, "row_scales", ws)
        object.__setattr__(self, "stats", MappingProxyType(dict(self.stats)))

    @property
    def n(self) -> int:
        return self.shape[-2]

    @property
    def k(self) -> int:
        return self.shape[-1]

    @property
    def wscale(self):
        return self.row_scales

    @property
    def nbytes(self) -> int:
        return self.codes.nbytes + self.bcodes.nbytes + self.row_scales.nbytes


def _unpack_shifts(bcodes, shape):
    return np.stack([(bcodes >> (2 * i)) & 3 for i in range(4)], axis=-1).reshape(
        *shape[:-1], shape[-1] // 32
    )


def _pack_shifts(shifts):
    b = shifts.reshape(*shifts.shape[:-1], shifts.shape[-1] // 4, 4).astype(np.uint8)
    return np.ascontiguousarray(b[..., 0] | (b[..., 1] << 2) | (b[..., 2] << 4) | (b[..., 3] << 6))


def _code_nibbles(codes, shape):
    return np.stack([(codes >> np.uint32(4 * i)) & 15 for i in range(8)], axis=-1).reshape(shape)


def metadata_from_mxfp4_scales_numpy(scales) -> tuple[np.ndarray, np.ndarray]:
    """Strict metadata-only loader, without copying or inspecting weight words.

    Unlike :func:`from_mxfp4_numpy`, this cannot identify zero blocks and thus
    requires every block's exponent to fit the same four-value row window.
    Offline SC conversion produces this stricter, canonical layout. Returns
    owned read-only ``(bcodes, row_scales)`` arrays for instance-local binding.
    """
    s = np.asarray(scales)
    if s.dtype != np.uint8 or s.ndim not in (2, 3):
        raise SCFormatError("MXFP4 metadata requires dense/grouped uint8 E8M0 scales")
    shape = (*s.shape[:-1], s.shape[-1] * 32)
    _validate_shape(shape)
    if np.any(s == 255):
        raise SCFormatError("MXFP4 metadata contains nonfinite E8M0 scales")
    e = s.astype(np.int16) - 127
    lo, hi = e.min(axis=-1), e.max(axis=-1)
    if np.any(hi - lo > 3):
        raise SCFormatError("MXFP4 metadata exponent spread exceeds three; offline conversion required")
    # Metadata-only installation cannot inspect the code magnitudes. Above
    # exponent 125 a legal FP4 magnitude can exceed F32, so refuse this
    # exceptional range conservatively rather than install an unchecked row.
    if np.any(hi > 125):
        raise SCFormatError("MXFP4 metadata cannot prove finite float32 weight values")
    bc = _pack_shifts((e - lo[..., None]).astype(np.uint8))
    ws = np.ascontiguousarray(np.exp2(lo.astype(np.float64) - 1).astype(np.float32))
    bc.flags.writeable = False
    ws.flags.writeable = False
    return bc, ws


def from_mxfp4_numpy(codes, scales) -> PackedSCWeight:
    """Losslessly fold standard E2M1 words/E8M0 scales into SC metadata.

    Entirely zero blocks may have normalized exponents because their values
    do not depend on scale. Nonzero exponent spread greater than three refuses;
    this loader never coarsens a block or changes a code. E8M0 255 is NaN and
    refuses even in zero blocks.
    """
    c, s = np.asarray(codes), np.asarray(scales)
    if c.dtype != np.uint32 or s.dtype != np.uint8 or c.ndim not in (2, 3):
        raise SCFormatError("MXFP4 requires dense/grouped uint32 words and uint8 E8M0 scales")
    shape = (*c.shape[:-1], c.shape[-1] * 8)
    _validate_shape(shape)
    if s.shape != (*shape[:-1], shape[-1] // 32):
        raise SCFormatError("MXFP4 E8M0 shape does not match the code words")
    if np.any(s == 255):
        raise SCFormatError("MXFP4 contains nonfinite E8M0 scales")
    nib = _code_nibbles(c, shape).reshape(*shape[:-1], shape[-1] // 32, 32)
    nonzero = np.any((nib & 7) != 0, axis=-1)
    e = s.astype(np.int16) - 127
    lo = np.min(np.where(nonzero, e, 128), axis=-1)
    hi = np.max(np.where(nonzero, e, -128), axis=-1)
    all_zero = ~np.any(nonzero, axis=-1)
    lo, hi = np.where(all_zero, 0, lo), np.where(all_zero, 0, hi)
    if np.any(hi - lo > 3):
        raise SCFormatError("nonzero MXFP4 exponent spread exceeds three; offline conversion required")
    normalized = np.where(nonzero, e, lo[..., None])
    shifts = (normalized - lo[..., None]).astype(np.uint8)
    ws = np.exp2(lo.astype(np.float64) - 1).astype(np.float32)
    return PackedSCWeight(c, _pack_shifts(shifts), ws, shape, {
        "conversion": "lossless_mxfp4_relayout", "rows": int(np.prod(shape[:-1])),
        "zero_blocks_normalized": int(np.count_nonzero(normalized != e)),
    })


def to_mxfp4_scales_numpy(packed: PackedSCWeight) -> np.ndarray:
    """Restore standard contiguous E8M0 scale bytes for ordinary decode."""
    exponents = np.log2(packed.row_scales.astype(np.float64))[..., None] + 1
    encoded = exponents + _unpack_shifts(packed.bcodes, packed.shape) + 127
    return np.ascontiguousarray(encoded.astype(np.uint8))


def decode_numpy(packed: PackedSCWeight) -> np.ndarray:
    """Decode exact INT8 operands; applying row scales is separate."""
    code = _code_nibbles(packed.codes, packed.shape)
    shifts = np.repeat(_unpack_shifts(packed.bcodes, packed.shape), 32, axis=-1)
    mag = np.asarray(SC_FORMAT.code_mag, dtype=np.int16)[code & 7] << shifts
    return np.where(code & 8, -mag, mag).astype(np.int8)


def pack_numpy(weights, *, allow_requantization: bool = False) -> PackedSCWeight:
    """Offline exponent-search SC-MXFP4 conversion requiring explicit opt-in.

    Tie handling and exponent search follow the pinned MIT source. The result
    retains a standard MXFP4 interpretation for ordinary decode.
    """
    if allow_requantization is not True:
        raise SCFormatError("offline quantization requires allow_requantization=True")
    original = np.asarray(weights)
    if original.dtype.kind != "f":
        raise SCFormatError("offline source weights must be floating point")
    shape = tuple(original.shape)
    _validate_shape(shape)
    w = np.asarray(original, dtype=np.float64)
    if not np.all(np.isfinite(w)):
        raise SCFormatError("offline source contains nonfinite weights")
    b = w.reshape(*shape[:-1], shape[-1] // 32, 32)
    amax = np.max(np.abs(b), axis=-1)
    active = amax > 0
    e = np.floor(np.log2(np.where(active, amax, 1.0))) - 2
    values = np.asarray(SC_FORMAT.code_mag, dtype=np.float64) / 2
    mids = (values[:-1] + values[1:]) / 2

    def quantized_codes(exponents):
        scaled = np.abs(b) / np.exp2(exponents[..., None])
        return np.searchsorted(mids, np.minimum(scaled, 6.0)).astype(np.uint8)

    def error(exponents):
        idx = quantized_codes(exponents)
        dq = np.copysign(values[idx], b) * np.exp2(exponents[..., None])
        return np.sum((dq - b) ** 2, axis=-1)

    best, best_error = e.copy(), error(e)
    for delta in (-1, 1):
        candidate = e + delta
        candidate_error = error(candidate)
        better = candidate_error < best_error
        best = np.where(better, candidate, best)
        best_error = np.where(better, candidate_error, best_error)
    hi = np.max(np.where(active, best, -np.inf), axis=-1, keepdims=True)
    hi = np.where(np.isfinite(hi), hi, 0)
    ec = np.maximum(np.where(active, best, -np.inf), hi - 3)
    if np.any(ec < -127) or np.any(ec > 127):
        raise SCFormatError("quantized exponents exceed finite E8M0 range")
    idx = quantized_codes(ec)
    code = (idx | (((b < 0) & (idx != 0)).astype(np.uint8) << 3)).reshape(shape)
    lanes = code.reshape(*shape[:-1], shape[-1] // 8, 8).astype(np.uint32)
    words = np.zeros((*shape[:-1], shape[-1] // 8), dtype=np.uint32)
    for i in range(8):
        words |= lanes[..., i] << np.uint32(4 * i)
    lo = ec.min(axis=-1)
    shifts = (ec - lo[..., None]).astype(np.uint8)
    ws = np.exp2(lo - 1).astype(np.float32)
    unconstrained = np.where(active, best, ec)
    return PackedSCWeight(words, _pack_shifts(shifts), ws, shape, {
        "conversion": "offline_requantization", "rows": int(np.prod(shape[:-1])),
        "constrained_blocks": int(np.count_nonzero(active & (ec != best))),
        "constrained_rows": int(np.count_nonzero(np.any(active & (ec != best), axis=-1))),
        "max_unconstrained_spread": int(np.max(np.max(unconstrained, axis=-1) - np.min(unconstrained, axis=-1))),
    })
