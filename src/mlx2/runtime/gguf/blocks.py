"""Bounded CPU oracle for packed GGML blocks; never a whole-model fallback.

Semantics mined from pinned antirez/ds4; see provenance/deepseek-v4-gguf-packed.json.
Returned values are float32. A caller must explicitly bound how many packed
blocks may be expanded; the default limit is 1 MiB of float32 output.
"""

from __future__ import annotations

import struct

import numpy as np

from .iq1_codebook import IQ1_GRID
from .iq2_codebook import IQ2_GRID
from .k_blocks import SIZES, decode_k_blocks
from .reader import FORMATS, GGUFError


def decode_blocks(
    data: bytes | memoryview, ggml_type: int, *, max_elements: int = 262144
) -> np.ndarray:
    if ggml_type not in (0, 1, 8, 10, 16, 19, 30) and ggml_type not in SIZES:
        raise GGUFError(f"no reference decoder for GGML type {ggml_type}")
    _, count, size = FORMATS[ggml_type]
    if len(data) % size:
        raise GGUFError("truncated packed quantization block")
    elements = len(data) // size * count
    if elements > max_elements:
        raise GGUFError("reference decoding exceeds explicit element budget")
    if ggml_type in SIZES:
        return decode_k_blocks(data, ggml_type)
    if ggml_type in (0, 1):
        return np.frombuffer(data, dtype="<f4" if ggml_type == 0 else "<f2").astype(
            np.float32
        )
    if ggml_type == 30:
        return (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view(
            np.float32
        )
    out = np.empty(elements, dtype=np.float32)
    for b in range(len(data) // size):
        start = b * size
        block = memoryview(data)[start : start + size]
        dest = out[b * count : (b + 1) * count]
        if ggml_type == 8:
            d = np.float32(struct.unpack_from("<e", block, 0)[0])
            dest[:] = d * np.frombuffer(block[2:], dtype=np.int8).astype(np.float32)
        elif ggml_type == 10:
            d, dmin = (np.float32(v) for v in struct.unpack_from("<ee", block, 80))
            for group in range(16):
                scale = block[group]
                base = 16 + 32 * (group // 8) + 16 * (group & 1)
                shift = ((group // 2) & 3) * 2
                q = np.frombuffer(block[base : base + 16], dtype=np.uint8)
                dest[group * 16 : (group + 1) * 16] = d * np.float32(scale & 15) * (
                    (q >> shift) & 3
                ).astype(np.float32) - dmin * np.float32(scale >> 4)
        elif ggml_type == 19:
            d = np.float32(struct.unpack_from("<e", block, 0)[0])
            for group in range(8):
                qh = struct.unpack_from("<H", block, 34 + group * 2)[0]
                scale = d * np.float32(2 * ((qh >> 12) & 7) + 1)
                delta = np.float32(-0.125 if qh & 0x8000 else 0.125)
                for j in range(4):
                    grid = IQ1_GRID[block[2 + group * 4 + j] | (((qh >> (3 * j)) & 7) << 8)]
                    for lane in range(8):
                        value = (grid >> (8 * lane)) & 255
                        value = value - 256 if value > 127 else value
                        dest[group * 32 + j * 8 + lane] = scale * (np.float32(value) + delta)
        else:
            d = np.float32(struct.unpack_from("<e", block, 0)[0])
            for group in range(8):
                indexes, aux = struct.unpack_from("<II", block, 2 + group * 8)
                dl = d * np.float32(0.5 + (aux >> 28)) * np.float32(0.25)
                for j in range(4):
                    grid = IQ2_GRID[(indexes >> (8 * j)) & 255]
                    seven = (aux >> (7 * j)) & 127
                    # The eighth sign bit gives even parity across the byte.
                    signs = seven | ((seven.bit_count() & 1) << 7)
                    for k in range(8):
                        magnitude = (grid >> (8 * k)) & 255
                        dest[group * 32 + j * 8 + k] = (
                            dl * magnitude * (-1 if signs & (1 << k) else 1)
                        )
    return out
