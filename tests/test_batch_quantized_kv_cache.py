"""Batched and rotating quantized KV caches against their B1 and dense references (CPU)."""

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.models.base import rotate_last, scaled_dot_product_attention
from mlx2.runtime.models.cache import (
    BatchQuantizedKVCache,
    BatchRotatingQuantizedKVCache,
    KVCache,
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


def _rotated_lane(keys, values, *, rotate):
    cache = KVCache()
    cache.update_and_fetch(keys, values)
    return cache.to_quantized(
        group_size=64, bits=8, key_bits=8, value_bits=8, rotate=rotate
    )


def test_rotated_batch_cache_rotates_keys_only_like_the_b1_cache():
    # Hadamard rotation is compensated on the query side only, so only keys may
    # be stored rotated. The batch cache used to rotate every appended value
    # too, and nothing ever un-rotated the attention output.
    mx.random.seed(0)
    heads, dim, history = 2, 64, 12
    k0, v0 = mx.random.normal((1, heads, history, dim)), mx.random.normal(
        (1, heads, history, dim)
    )
    k1, v1 = mx.random.normal((1, heads, 1, dim)), mx.random.normal((1, heads, 1, dim))
    query = mx.random.normal((1, heads, 1, dim))
    scale = dim**-0.5
    dense = mx.fast.scaled_dot_product_attention(
        query,
        mx.concatenate([k0, k1], axis=2),
        mx.concatenate([v0, v1], axis=2),
        scale=scale,
    )

    single = _rotated_lane(k0, v0, rotate=True)
    keys, values = single.update_and_fetch(k1, v1)
    b1 = scaled_dot_product_attention(
        query, keys, values, cache=single, scale=scale, mask=None
    )
    batch = BatchQuantizedKVCache.merge([_rotated_lane(k0, v0, rotate=True)])
    keys, values = batch.update_and_fetch(k1, v1)
    batched = scaled_dot_product_attention(
        query, keys, values, cache=batch, scale=scale, mask=None
    )
    stored_value = mx.dequantize(
        *[x[..., history : history + 1, :] for x in batch.values],
        group_size=64,
        bits=8,
    )

    assert mx.abs(stored_value - v1).max().item() < 0.05
    assert mx.abs(batched - dense).max().item() < 0.05
    assert mx.allclose(batched, b1, atol=1e-5).item()
    extracted = batch.extract(0)
    for got, want in zip(extracted.values, single.values):
        assert mx.array_equal(got, want[..., : history + 1, :]).item()


def test_rotated_batch_qsa_conversion_rotates_keys_only():
    from mlx2.runtime.models.qwen4_exp import BatchQSAKVCache, QSAKVCache

    mx.random.seed(1)
    keys, values = mx.random.normal((2, 2, 7, 64)), mx.random.normal((2, 2, 7, 64))
    source = BatchQSAKVCache([0, 0])
    source.update_and_fetch(keys, values)
    packed = source.to_quantized(group_size=64, bits=8, rotate=True)
    stored_values = mx.dequantize(*packed.values, group_size=64, bits=8)
    stored_keys = mx.dequantize(*packed.keys, group_size=64, bits=8)

    assert mx.abs(stored_values[..., :7, :] - values).max().item() < 0.05
    assert mx.abs(stored_keys[..., :7, :] - rotate_last(keys)).max().item() < 0.05
    single = QSAKVCache()
    single.update_and_fetch(keys[:1], values[:1])
    reference = single.to_quantized(group_size=64, bits=8, rotate=True)
    for got, want in zip(packed.values, reference.values):
        assert mx.array_equal(got[:1, :, :7], want).item()
