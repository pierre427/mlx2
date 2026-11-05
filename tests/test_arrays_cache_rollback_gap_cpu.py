"""An unrecordable recurrent forward breaks the exact rollback chain.

Synthetic GDN weights on CPU, no checkpoint or serving qualification.
"""

import mlx.core as mx
import pytest

from mlx2.runtime.models.cache import ArraysCache
from test_short_gdn_rollback import _assert_cache_equal, _layer


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize("geometry", ["unexplained_mask", "leading_padding"])
def test_unrecordable_forward_invalidates_older_records_and_epoch(geometry):
    mx.random.seed(1103)
    layer = _layer()
    cache = ArraysCache(2)
    layer(mx.random.normal((1, 3, 32)), cache=cache)
    cache.start_speculation()
    marker = cache.rollback_marker()
    layer(mx.random.normal((1, 2, 32)), cache=cache)
    assert cache._rollbacks
    if geometry == "leading_padding":
        cache.left_padding = mx.array([1])
        mask = cache.make_mask(3)
    else:
        mask = mx.array([[True, False, True]])
    layer(mx.random.normal((1, 3, 32)), mask=mask, cache=cache)
    assert not cache._rollbacks
    with pytest.raises(RuntimeError, match="exact rollback"):
        cache.trim(1)
    with pytest.raises(RuntimeError, match="exact rollback"):
        cache.trim_ragged([1])
    with pytest.raises(RuntimeError, match="stale epoch"):
        cache.rewind_to_rollback_marker(marker)
    with pytest.raises(RuntimeError, match="no live exact rollback"):
        cache.latest_exact_rollback_boundary()

    # A later describable forward can establish a new exact suffix, without
    # making the discarded history replayable again.
    cache.finalize()
    boundary = ArraysCache(2)
    boundary.cache = list(cache.cache)
    new_marker = cache.rollback_marker()
    layer(mx.random.normal((1, 2, 32)), cache=cache)
    assert cache.rewind_to_rollback_marker(new_marker) == 2
    _assert_cache_equal(cache, boundary)
    with pytest.raises(RuntimeError, match="exact rollback"):
        cache.trim(1)


def test_unrecordable_geometry_probe_preserves_live_tensors():
    cache = ArraysCache(1)
    cache.cache = [mx.ones((1, 2))]
    cache.start_speculation()
    cache.record_rollback(1, lambda _: [mx.zeros((1, 2))], [mx.zeros((1, 2))])
    current = cache.cache[0]
    assert cache.rollback_spans(1, mx.array([[False]])) is None
    assert cache.cache[0] is current
    assert not cache._rollbacks
