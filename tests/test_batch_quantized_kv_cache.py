"""Batched and rotating quantized KV caches against their B1 and dense references (CPU)."""

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.models.cache import (
    BatchQuantizedKVCache,
    BatchRotatingQuantizedKVCache,
    QuantizedKVCache,
    RotatingQuantizedKVCache,
    _empty_quantized,
)

BITS = (3, 4, 5, 6, 8)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("head_dim", (64, 128, 256))
def test_empty_quantized_buffer_matches_the_mx_quantize_layout(bits, head_dim):
    packed, scales, biases = mx.quantize(
        mx.zeros((1, 1, 1, head_dim)), group_size=64, bits=bits
    )
    empty = _empty_quantized(1, 1, 1, head_dim, 64, bits, mx.float32)
    assert [a.shape for a in empty] == [packed.shape, scales.shape, biases.shape]


@pytest.mark.parametrize("bits", BITS)
def test_merged_quantized_rows_decode_at_every_supported_width(bits):
    mx.random.seed(0)
    rows = []
    for n in (5, 3):
        cache = QuantizedKVCache(group_size=64, bits=bits)
        cache.update_and_fetch(
            mx.random.normal((1, 2, n, 128)), mx.random.normal((1, 2, n, 128))
        )
        rows.append(cache)
    batch = BatchQuantizedKVCache.merge(rows)
    new_k = mx.random.normal((2, 2, 1, 128))
    keys, values = batch.update_and_fetch(new_k, mx.random.normal((2, 2, 1, 128)))
    mx.eval(keys, values)
    assert keys[0].shape[-2] == 6
    stored = mx.dequantize(
        *[x[..., 5:6, :] for x in keys], group_size=64, bits=bits
    )
    expected = mx.dequantize(
        *mx.quantize(new_k, group_size=64, bits=bits), group_size=64, bits=bits
    )
    assert mx.allclose(stored, expected).item()


@pytest.mark.parametrize("bits", BITS)
def test_rotating_quantized_caches_grow_at_every_supported_width(bits):
    mx.random.seed(0)
    single = RotatingQuantizedKVCache(max_size=16, group_size=64, bits=bits)
    batch = BatchRotatingQuantizedKVCache(16, [0, 0], group_size=64, bits=bits)
    for cache, lanes in ((single, 1), (batch, 2)):
        for _ in range(3):
            keys, values = cache.update_and_fetch(
                mx.random.normal((lanes, 2, 1, 128)),
                mx.random.normal((lanes, 2, 1, 128)),
            )
            mx.eval(keys, values)
        assert keys[0].shape[-1] == 128 * bits // 32
