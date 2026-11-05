"""glm53_prompt_decoded=decoded_f32_v1: the K-block f32 decode kernel and the
prompt-phase GEMM routes (Metal, MLX2_RUN_GPU_TESTS=1).

Element values must equal the CPU K-block oracle bit for bit on every format
the policy admits; the grouped and dense routes must equal a float64 matmul of
those values to fp32 GEMM tolerance and engage only at the scoped prompt width.
"""

import os
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.gguf.k_blocks import SIZES, decode_k_blocks
from mlx2.runtime.gguf.kblock_prefill_metal import DECODED_F32_FORMATS, kblock_decode_f32, kblock_decode_fp16
from mlx2.runtime.gguf.packed_metal import row_bytes

GPU = os.environ.get("MLX2_RUN_GPU_TESTS") == "1"
metal = pytest.mark.skipif(not (GPU and mx.metal.is_available()),
                           reason="Metal kernel test; set MLX2_RUN_GPU_TESTS=1")


@dataclass
class Packed:
    data: mx.array
    shape: tuple
    ggml_type: int
    row_bytes: int


# Byte offsets of the half-precision block scales (d, and dmin for Q4_K/Q5_K).
SCALE_BYTES = {2: (0,), 11: (108,), 12: (0, 2), 13: (0, 2), 14: (208,)}


def packed(rng, shape, kind):
    """Random packed rows whose block scales are finite halves, so every
    decoded value is finite and the comparison can be bit for bit."""
    count, size = SIZES[kind]
    k = shape[-1]
    rows = int(np.prod(shape[:-1]))
    raw = rng.integers(0, 256, (rows, k // count * size), dtype=np.uint8)
    blocks = raw.reshape(rows, k // count, size)
    for offset in SCALE_BYTES[kind]:
        halves = rng.integers(0x0400, 0x7BFF, (rows, k // count), dtype=np.uint16)  # normal, finite
        blocks[:, :, offset] = halves & 0xFF
        blocks[:, :, offset + 1] = halves >> 8
    return Packed(mx.array(raw.reshape(-1)), tuple(shape), kind, row_bytes(kind, k)), raw


def oracle(raw, shape, kind):
    return np.stack([decode_k_blocks(row.tobytes(), kind) for row in raw]).reshape(shape)


@pytest.fixture
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


@metal
@pytest.mark.parametrize("kind", DECODED_F32_FORMATS)
def test_decode_f32_matches_the_cpu_oracle_bitwise(gpu, kind):
    rng = np.random.default_rng(kind)
    for shape in (((16, 128), (3, 32, 256)) if kind == 2 else ((16, 256), (3, 32, 512))):
        weight, raw = packed(rng, shape, kind)
        got = np.array(kblock_decode_f32(weight))
        want = oracle(raw, shape, kind)
        assert got.shape == shape and got.dtype == np.float32
        assert np.all(np.isfinite(want)), kind
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), kind


@metal
def test_decode_f32_refuses_what_it_cannot_decode(gpu):
    rng = np.random.default_rng(7)
    from mlx2.runtime.gguf.reader import GGUFError

    with pytest.raises(GGUFError):
        kblock_decode_f32(Packed(mx.zeros(16, dtype=mx.uint8), (4, 40), 2, 18))
    with pytest.raises(GGUFError):  # a 256-block format needs whole 256-blocks
        kblock_decode_f32(Packed(mx.zeros(16, dtype=mx.uint8), (4, 128), 14, 210))
    weight, _ = packed(rng, (4, 32), 2)
    with pytest.raises(GGUFError):
        kblock_decode_f32(Packed(weight.data, (4, 32), 8, 34))


@metal
@pytest.mark.parametrize("kind", DECODED_F32_FORMATS)
def test_decode_fp16_is_the_f32_values_rounded_to_half_in_every_shape(gpu, kind):
    rng = np.random.default_rng(100 + kind)
    for shape in (((16, 128), (3, 32, 256)) if kind == 2 else ((16, 256), (3, 32, 512))):
        weight, raw = packed(rng, shape, kind)
        half = np.array(kblock_decode_fp16(weight))
        want = oracle(raw, shape, kind).astype(np.float16)
        assert half.shape == shape and half.dtype == np.float16
        assert np.array_equal(half.view(np.uint16), want.view(np.uint16)), kind
