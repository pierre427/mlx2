"""The Agnes PLD full-rewind cache regression from the retired unified fork."""

import mlx.core as mx

from mlx2.runtime.models.cache import ArraysCache


def test_full_rewind_restores_lazy_batch_positions():
    cache = ArraysCache(size=1)
    cache.start_speculation(rollback_window=4)
    cache.record_rollback(1, lambda marker: [mx.full((2, 1), marker)], [None])
    cache.cache = [mx.ones((2, 1))]
    cache.record_rollback(
        1, lambda marker: [mx.full((2, 1), 1 + marker)], list(cache.cache)
    )
    cache.cache = [mx.full((2, 1), 2)]
    assert cache._rollback_positions == [2, 2]

    cache.trim(2)
    assert cache.cache[0] is None
    assert cache._rollback_positions is None
    assert cache._rollback_position == 0

    cache.record_rollback(1, lambda marker: [mx.full((2, 1), marker)], [None])
    cache.cache = [mx.ones((2, 1))]
    cache.record_rollback(
        1, lambda marker: [mx.full((2, 1), 1 + marker)], list(cache.cache)
    )
    assert cache._rollback_positions == [2, 2]
