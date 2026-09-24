"""BatchKVCache hands SDPA "causal" for unpadded rows (2026-09-24).

An all-true-below-diagonal array mask kept head_dim-256 prefill on MLX's
array-mask kernel, which cannot skip fully masked blocks. The string is only
returned while the host knows the *current* ``left_padding`` array is all
<= 0; every padding change (in place or rebinding, inside the class or not)
must fall back to the exact array.
"""
import mlx.core as mx
import pytest

from mlx2.runtime.models.cache import BatchKVCache


def _fill(cache, n, *, heads=2, dim=8, seed=0):
    mx.random.seed(seed)
    B = int(cache.left_padding.shape[0])
    k = mx.random.normal((B, heads, n, dim))
    v = mx.random.normal((B, heads, n, dim))
    return cache.update_and_fetch(k, v)


def test_unpadded_prefill_gets_causal_string():
    cache = BatchKVCache([0])
    assert cache.make_mask(16) == "causal"
    _fill(cache, 16)
    assert cache.make_mask(8) == "causal"


@pytest.mark.parametrize("kwargs", [{"return_array": True}, {"window_size": 4}])
def test_explicit_array_or_window_keeps_the_array(kwargs):
    cache = BatchKVCache([0, 0])
    assert isinstance(cache.make_mask(8, **kwargs), mx.array)


def test_decode_step_is_unchanged():
    cache = BatchKVCache([0])
    _fill(cache, 4)
    assert isinstance(cache.make_mask(1), mx.array)


def test_padded_rows_keep_the_array():
    assert isinstance(BatchKVCache([0, 3]).make_mask(8), mx.array)


def test_prepare_padding_in_place_drops_the_proof():
    cache = BatchKVCache([0, 0])
    identity = cache.left_padding
    cache.prepare(left_padding=[2, 0])
    assert cache.left_padding is identity  # in-place: identity alone can't tell
    assert isinstance(cache.make_mask(8), mx.array)


def test_outside_rebinding_drops_the_proof():
    cache = BatchKVCache([0, 0])
    cache.left_padding = mx.array([0, 5])
    assert isinstance(cache.make_mask(8), mx.array)


def test_finalize_right_padding_drops_the_proof():
    cache = BatchKVCache([0, 0])
    _fill(cache, 6)
    cache.prepare(right_padding=[0, 2])
    _fill(cache, 3, seed=1)
    cache.finalize(reclaim=False)
    assert isinstance(cache.make_mask(8), mx.array)


def test_filter_recomputes_exactly():
    cache = BatchKVCache([0, 3])
    _fill(cache, 6)
    cache.filter(mx.array([0]))
    assert cache.make_mask(4) == "causal"
    padded = BatchKVCache([0, 3])
    _fill(padded, 6)
    padded.filter(mx.array([1]))
    # Row 1's padding is inside the shared cursor, so filter reclaims it.
    assert padded.make_mask(4) == "causal"
    assert int(padded.left_padding[0].item()) == 0


def test_merge_of_equal_lengths_is_unpadded():
    a, b = BatchKVCache([0]), BatchKVCache([0])
    _fill(a, 5)
    _fill(b, 5, seed=2)
    ex = [a.extract(0), b.extract(0)]
    assert BatchKVCache.merge(ex).make_mask(3) == "causal"
    c = BatchKVCache([0])
    _fill(c, 2, seed=3)
    assert isinstance(BatchKVCache.merge([ex[0], c.extract(0)]).make_mask(3), mx.array)


@pytest.mark.parametrize("prefix,n", [(0, 7), (9, 5), (32, 16)])
def test_causal_string_attends_like_the_array_mask(prefix, n):
    cache = BatchKVCache([0])
    if prefix:
        _fill(cache, prefix, seed=4)
    array_mask = cache.make_mask(n, return_array=True)
    string_mask = cache.make_mask(n)
    assert string_mask == "causal"
    keys, values = _fill(cache, n, seed=5)
    mx.random.seed(6)
    q = mx.random.normal((1, 2, n, 8))
    got = mx.fast.scaled_dot_product_attention(q, keys, values, scale=0.35, mask=string_mask)
    want = mx.fast.scaled_dot_product_attention(q, keys, values, scale=0.35, mask=array_mask)
    assert mx.allclose(got, want, atol=1e-5).item()
