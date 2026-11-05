"""Synthetic CPU KDA transaction oracles; no model checkpoint is loaded."""
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.bailing_hybrid import KimiDeltaAttention
from mlx2.runtime.models.cache import ArraysCache
from test_kimi_linear import args


def _layer():
    mx.random.seed(11303)
    layer = KimiDeltaAttention(replace(args(), head_dim=16, no_kda_lora=True))
    layer.eval()
    return layer


def _close(actual, expected):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-5, atol=2e-6)


def _cache_close(actual, expected):
    for got, want in zip(actual, expected):
        assert got.shape == want.shape and got.dtype == want.dtype
        _close(got, want)


@pytest.mark.parametrize("bounded", [True, False])
def test_short_verify_after_wide_verify_restores_all_kda_planes(bounded):
    layer = _layer()
    if not bounded:
        layer.lower_bound = None
    prefix = mx.random.normal((2, 4, 32))
    wide = mx.random.normal((2, 5, 32))
    short = mx.random.normal((2, 2, 32))
    following = mx.random.normal((2, 1, 32))
    cache = ArraysCache(4)
    layer(prefix, cache=cache)
    cache.start_speculation()
    layer(wide, cache=cache)
    assert cache._rollbacks[-1].span == 5
    cache.trim_ragged([3, 1])
    layer(short, cache=cache)
    assert cache._rollbacks[-1].span == 2
    cache.trim_ragged([1, 2])
    for row, (wide_keep, short_keep) in enumerate(((2, 1), (4, 0))):
        cold = ArraysCache(4)
        replay = mx.concatenate([prefix[row:row + 1], wide[row:row + 1, :wide_keep],
                                 short[row:row + 1, :short_keep]], axis=1)
        layer(replay, cache=cold)
        actual = cache.extract(row)
        _cache_close(actual, cold)
        _close(layer(following[row:row + 1], cache=actual),
               layer(following[row:row + 1], cache=cold))


def test_right_padded_kda_rollback_credits_each_rows_valid_span():
    layer = _layer()
    prefix = mx.random.normal((2, 3, 32))
    slab = mx.random.normal((2, 5, 32))
    cache = ArraysCache(4)
    layer(prefix, cache=cache)
    cache.start_speculation()
    cache.prepare(lengths=[2, 5])
    layer(slab, mask=cache.make_mask(5), cache=cache)
    assert cache._rollbacks[-1].depth_list(2) == [2, 5]
    cache.trim_ragged([1, 3])
    cache.finalize()
    for row, keep in enumerate((1, 2)):
        cold = ArraysCache(4)
        layer(mx.concatenate([prefix[row:row + 1], slab[row:row + 1, :keep]], axis=1),
              cache=cold)
        _cache_close(cache.extract(row), cold)


def test_full_rollback_restores_entry_state_and_drops_record():
    layer = _layer()
    cache = ArraysCache(4)
    layer(mx.random.normal((1, 3, 32)), cache=cache)
    before = list(cache)
    cache.start_speculation()
    layer(mx.random.normal((1, 4, 32)), cache=cache)
    assert cache.trim(4) == 4
    assert not cache._rollbacks
    for actual, expected in zip(cache, before):
        assert bool(mx.array_equal(actual, expected))


def test_unexplained_mask_is_not_recorded_as_replayable():
    layer = _layer()
    cache = ArraysCache(4)
    layer(mx.random.normal((1, 3, 32)), cache=cache)
    cache.start_speculation()
    layer(mx.random.normal((1, 4, 32)), mask=mx.array([[True, False, True, True]]), cache=cache)
    assert not cache._rollbacks
    with pytest.raises(RuntimeError, match="exact rollback"):
        cache.trim(1)
