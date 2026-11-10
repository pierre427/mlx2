# SPDX-License-Identifier: Apache-2.0
"""CPU-only exact-value GGUF packing for a research MLX affine layer probe.

This changes storage, not the represented finite weights. It does not establish
equivalence of native affine matmul reductions with the original IQ1_S kernels.
IEEE signed zero can change under affine multiply/add and is reported separately.

Local provenance: mlx2@89e6938ea58a0060026815c94e0ed4d33729d775 plus reviewed
snapshot 936c714a49355056ee67fcbb72856d16b22e2596308dae10e897ddec009159a7,
src/mlx2/runtime/gguf/{blocks.py,iq1_codebook.py,reader.py}. The existing IQ1
codebook retains its MIT notice and pinned source in
provenance/deepseek-v41-adapter.json; block decoding provenance is retained in
provenance/deepseek-v4-gguf-packed.json. No new external implementation is mined.
Original vectorized packing and bounded chunk interfaces below are Apache-2.0.

``shape`` always means GGUFTensor.numpy_shape: contiguous input K is LAST.
All leading axes (including experts) retain their original order. Packing is
little-lane-first: sixteen 2-bit or four 8-bit codes per uint32 word. Scales and
biases are float32, one per 32 original elements, with no requantization.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from mlx2.runtime.gguf.blocks import decode_blocks
from mlx2.runtime.gguf.iq1_codebook import IQ1_GRID


MAX_CHUNK_ELEMENTS = 1 << 20
_FORMATS = {19: (2, 256, 50), 8: (8, 32, 34)}


def affine_spec(ggml_type, shape):
    """Return exact source/output geometry without allocating tensor payload."""
    if type(ggml_type) is not int or ggml_type not in _FORMATS:
        raise ValueError("affine conversion supports only IQ1_S(19) and Q8_0(8)")
    if (not isinstance(shape, (tuple, list)) or not shape
            or any(type(size) is not int or size <= 0 for size in shape)):
        raise ValueError("shape must be positive integer GGUF numpy_shape dimensions")
    shape = tuple(shape)
    bits, elements, block_bytes = _FORMATS[ggml_type]
    if shape[-1] % elements:
        raise ValueError("contiguous input width must contain complete GGUF blocks")
    count = math.prod(shape)
    packed_shape = (*shape[:-1], shape[-1] // (32 // bits))
    scale_shape = (*shape[:-1], shape[-1] // 32)
    return dict(ggml_type=ggml_type, bits=bits, group_size=32, shape=shape,
                packed_shape=packed_shape, scales_shape=scale_shape, biases_shape=scale_shape,
                elements=count, source_bytes=count // elements * block_bytes,
                output_bytes=4 * (math.prod(packed_shape) + 2 * math.prod(scale_shape)))


def _limit(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _raw_view(raw, size):
    if isinstance(raw, np.ndarray):
        if raw.dtype != np.uint8 or not raw.flags.c_contiguous:
            raise ValueError("source must be contiguous uint8 packed bytes")
        result = raw.reshape(-1)
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        view = memoryview(raw)
        if view.format != "B" or not view.c_contiguous:
            raise ValueError("source must be contiguous uint8 packed bytes")
        result = np.frombuffer(view, np.uint8)
    else:
        raise ValueError("source must be a uint8 array or byte buffer")
    if result.size != size:
        raise ValueError("source byte count does not match GGUF tensor shape")
    return result


@lru_cache(maxsize=1)
def _iq1_affine_grid():
    """Each existing eight-lane ternary grid becomes one packed 16-bit word."""
    grid = np.asarray(IQ1_GRID, np.uint64)[:, None]
    lanes = np.arange(8, dtype=np.uint64)
    signed = ((grid >> (8 * lanes)) & 255).astype(np.uint8).view(np.int8)
    if np.any((signed < -1) | (signed > 1)):
        raise ValueError("IQ1 codebook contains a non-ternary value")
    codes = (signed + np.int8(1)).astype(np.uint32)
    packed = np.bitwise_or.reduce(codes << (2 * lanes.astype(np.uint32)), axis=1)
    packed.flags.writeable = False
    return packed


def convert_affine(raw, ggml_type, shape, *, max_elements=MAX_CHUNK_ELEMENTS):
    """Convert one explicitly bounded chunk, returning packed/scales/biases.

    Input buffers are viewed without a full-source copy. Output owns its arrays;
    modifying or closing the input after return cannot change converted values.
    No scalar-per-weight loop, floating full-weight expansion or MLX call occurs.
    Use iter_affine_chunks or repeated row-bounded calls for a complete layer.
    """
    spec = affine_spec(ggml_type, shape)
    if spec["elements"] > _limit(max_elements, "max_elements"):
        raise ValueError("affine conversion exceeds explicit element budget")
    source = _raw_view(raw, spec["source_bytes"])
    _, _, block_bytes = _FORMATS[ggml_type]
    blocks = source.reshape(-1, block_bytes)
    scales = blocks[:, :2].view("<f2").reshape(-1).astype(np.float32)
    if not np.isfinite(scales).all():
        raise ValueError("affine conversion requires finite source scales")
    if ggml_type == 19:
        high = blocks[:, 34:50].view("<u2")
        shifts = np.arange(4, dtype=np.uint16) * 3
        indices = (blocks[:, 2:34].reshape(-1, 8, 4).astype(np.uint16)
                   | (((high[..., None] >> shifts) & 7) << 8))
        half_words = _iq1_affine_grid()[indices]
        packed = half_words[..., ::2] | (half_words[..., 1::2] << np.uint32(16))
        scales = scales[:, None] * (2 * ((high >> 12) & 7) + 1).astype(np.float32)
        biases = scales * np.where(high & 0x8000, np.float32(-1.125), np.float32(-.875))
    else:
        # int8+128 is exactly the source byte with its sign bit toggled. The
        # xor allocates owned contiguous bytes; viewing groups of four packs
        # low-address/low-lane-first without a per-weight integer expansion.
        packed = (blocks[:, 2:] ^ np.uint8(128)).view("<u4")
        biases = scales * np.float32(-128)
    return (packed.reshape(spec["packed_shape"]), scales.reshape(spec["scales_shape"]),
            biases.reshape(spec["biases_shape"]))


@dataclass(frozen=True)
class AffineChunk:
    row_start: int
    row_stop: int
    packed: np.ndarray
    scales: np.ndarray
    biases: np.ndarray


def iter_affine_chunks(raw, ggml_type, shape, *, rows_per_chunk=64,
                       max_chunk_elements=MAX_CHUNK_ELEMENTS):
    """Yield bounded flattened row chunks over a zero-copy source buffer view.

    Reshape destination memmaps to (-1, packed_width) and (-1, K//32), then
    assign each [row_start:row_stop] slice. This preserves arbitrary original
    leading axes without retaining previous chunks. The caller owns/retires
    source mappings and disk output; this iterator never opens model files.
    """
    spec = affine_spec(ggml_type, shape)
    rows_per_chunk = _limit(rows_per_chunk, "rows_per_chunk")
    limit = _limit(max_chunk_elements, "max_chunk_elements")
    width = spec["shape"][-1]
    if width > limit:
        raise ValueError("one tensor row exceeds explicit chunk element budget")
    source = _raw_view(raw, spec["source_bytes"])
    rows = spec["elements"] // width
    row_bytes = spec["source_bytes"] // rows
    step = min(rows_per_chunk, limit // width)
    for first in range(0, rows, step):
        last = min(rows, first + step)
        packed, scales, biases = convert_affine(source[first * row_bytes:last * row_bytes],
                                                ggml_type, (last - first, width), max_elements=limit)
        yield AffineChunk(first, last, packed, scales, biases)


def dequantize_affine(packed, scales, biases, ggml_type, shape, *, max_elements=MAX_CHUNK_ELEMENTS):
    """Bounded CPU affine oracle; not a runtime dense-weight fallback."""
    spec = affine_spec(ggml_type, shape)
    if spec["elements"] > _limit(max_elements, "max_elements"):
        raise ValueError("affine oracle exceeds explicit element budget")
    if (not isinstance(packed, np.ndarray) or packed.dtype != np.uint32
            or packed.shape != spec["packed_shape"]
            or any(not isinstance(x, np.ndarray) or x.dtype != np.float32
                   or x.shape != spec["scales_shape"] for x in (scales, biases))):
        raise ValueError("affine arrays differ from declared shape/dtype")
    if not np.isfinite(scales).all() or not np.isfinite(biases).all():
        raise ValueError("affine arrays require finite scales and biases")
    bits = spec["bits"]
    shifts = np.arange(32 // bits, dtype=np.uint32) * bits
    codes = ((packed[..., None] >> shifts) & ((1 << bits) - 1)).reshape(*spec["shape"][:-1], -1, 32)
    return (codes.astype(np.float32) * scales[..., None] + biases[..., None]).reshape(spec["shape"])


def validate_affine(raw, ggml_type, shape, converted=None, *, max_elements=MAX_CHUNK_ELEMENTS,
                    require_bitwise=False):
    """Compare against existing decode_blocks; report signed-zero changes.

    All finite numerical values must match exactly; no tolerance is accepted.
    require_bitwise additionally refuses changed IEEE zero signs. This validates
    weights only, not a different matmul's summation or activation boundaries.
    """
    spec = affine_spec(ggml_type, shape)
    if spec["elements"] > _limit(max_elements, "max_elements"):
        raise ValueError("affine validation exceeds explicit element budget")
    if type(require_bitwise) is not bool:
        raise ValueError("require_bitwise must be boolean")
    source = _raw_view(raw, spec["source_bytes"])
    arrays = convert_affine(source, ggml_type, shape, max_elements=max_elements) if converted is None else converted
    actual = dequantize_affine(*arrays, ggml_type, shape, max_elements=max_elements)
    expected = decode_blocks(memoryview(source), ggml_type, max_elements=max_elements).reshape(spec["shape"])
    if not np.isfinite(expected).all() or not np.array_equal(actual, expected):
        raise ValueError("affine reconstruction changed finite source weight values")
    signs = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
    if signs and require_bitwise:
        raise ValueError(f"affine reconstruction changed {signs} signed-zero bit patterns")
    return dict(elements=spec["elements"], value_exact=True, bitwise_exact=not signs,
                signed_zero_differences=signs, bits=spec["bits"], group_size=32,
                scale_dtype="float32", bias_dtype="float32")
