"""Per-row decode attention backend for BatchKVCache (jundot/omlx#5090), CPU.

Opt-in research option ``MLX_LM_BATCH_ATTENTION_BACKEND=rows|rows_masked``:
each row's batch-1 SDPA reads only its own ``keys[i, :, pad_i:_idx]`` view.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx2.runtime.models.base import scaled_dot_product_attention  # noqa: E402
from mlx2.runtime.models.cache import BatchKVCache, KVCache  # noqa: E402

HQ, HKV, D = 8, 2, 32


def _caches(lengths, backend, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for length in lengths:
        cache = KVCache()
        k = mx.array(rng.standard_normal((1, HKV, length, D)).astype(np.float32))
        v = mx.array(rng.standard_normal((1, HKV, length, D)).astype(np.float32))
        cache.update_and_fetch(k, v)
        rows.append(cache)
    merged = BatchKVCache.merge(rows)
    merged._configure_attention_backend(backend)
    return merged


def _decode(cache, seed=1):
    rng = np.random.default_rng(seed)
    b = cache.offset.shape[0]
    q = mx.array(rng.standard_normal((b, HQ, 1, D)).astype(np.float32))
    k = mx.array(rng.standard_normal((b, HKV, 1, D)).astype(np.float32))
    v = mx.array(rng.standard_normal((b, HKV, 1, D)).astype(np.float32))
    mask = cache.make_mask(1)
    keys, values = cache.update_and_fetch(k, v)
    return scaled_dot_product_attention(q, keys, values, cache, scale=D ** -0.5, mask=mask)


@pytest.mark.parametrize("backend", ["rows", "rows_masked"])
@pytest.mark.parametrize("lengths", [[24, 100], [100, 24, 57], [64, 64]])
def test_rows_backend_matches_padded_dense(backend, lengths):
    with mx.stream(mx.cpu):
        dense = _caches(lengths, "sdpa")
        rows = _caches(lengths, backend)
        for step in range(3):
            want = _decode(dense, seed=10 + step)
            got = _decode(rows, seed=10 + step)
            np.testing.assert_allclose(np.array(got), np.array(want), rtol=2e-5, atol=2e-6)
        stats = rows._attention_backend_stats
        assert stats["rows_calls"] == 3
        assert stats["rows_padding_reads"] == 1  # padding read once, reused
        assert stats["dense_fallbacks"] == 0


def test_rows_backend_declines_verify_rows_and_batch_one():
    with mx.stream(mx.cpu):
        cache = _caches([10, 30], "rows")
        q = mx.zeros((2, HQ, 3, D))
        assert cache.bucketed_attention(q, 1.0, None) is None
        assert cache._attention_backend_stats["fallback_reasons"] == {"not_decode": 1}
        single = _caches([10], "rows")
        assert single.bucketed_attention(mx.zeros((1, HQ, 1, D)), 1.0, None) is None


def test_padding_change_invalidates_the_row_padding():
    with mx.stream(mx.cpu):
        cache = _caches([10, 30, 20], "rows")
        _decode(cache)
        cache.filter([0, 2])
        dense = _caches([10, 30, 20], "sdpa")
        _decode(dense)
        dense.filter([0, 2])
        np.testing.assert_allclose(np.array(_decode(cache, seed=5)), np.array(_decode(dense, seed=5)),
                                   rtol=2e-5, atol=2e-6)
        assert cache._attention_backend_stats["rows_padding_reads"] == 2


def test_backend_name_is_validated():
    with pytest.raises(ValueError):
        BatchKVCache([0, 0], attention_backend="row")


def test_rows_backend_slices_a_mask_it_did_not_build():
    """A caller's own boolean decode mask is never dropped by ``rows``."""
    from mlx2.runtime.models import cache as cache_module

    with mx.stream(mx.cpu):
        dense = _caches([20, 40], "sdpa")
        rows = _caches([20, 40], "rows")
        rng = np.random.default_rng(4)
        q = mx.array(rng.standard_normal((2, HQ, 1, D)).astype(np.float32))
        k = mx.array(rng.standard_normal((2, HKV, 1, D)).astype(np.float32))
        v = mx.array(rng.standard_normal((2, HKV, 1, D)).astype(np.float32))
        want_mask = dense.make_mask(1)
        dense_keys, dense_values = dense.update_and_fetch(k, v)
        rows.make_mask(1)
        keys, values = rows.update_and_fetch(k, v)
        # Hide the newest key column of row 1, a mask no cache built.
        foreign = np.array(want_mask)
        foreign[1, ..., -1] = False
        before = cache_module.ROWS_ATTENTION_STATS["rows_foreign_mask"]
        got = scaled_dot_product_attention(q, keys, values, rows, scale=D ** -0.5, mask=mx.array(foreign))
        want = scaled_dot_product_attention(q, dense_keys, dense_values, dense, scale=D ** -0.5,
                                            mask=mx.array(foreign))
        np.testing.assert_allclose(np.array(got), np.array(want), rtol=2e-5, atol=2e-6)
        assert cache_module.ROWS_ATTENTION_STATS["rows_foreign_mask"] == before + 1
