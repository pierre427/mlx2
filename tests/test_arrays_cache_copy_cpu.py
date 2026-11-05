"""CPU clone/recovery contracts for live recurrent rollback records."""

import copy

import mlx.core as mx
import pytest

from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.models.cache import ArraysCache
from test_short_gdn_rollback import _assert_cache_equal, _layer


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize("per_row", [False, True])
def test_reference_clone_preserves_live_ragged_replay_and_independent_ownership(monkeypatch, per_row):
    from mlx2.runtime.models import qwen3_5

    monkeypatch.setattr(qwen3_5, "_GDN_ARRAY_ACCEPT", per_row)
    mx.random.seed(110304)
    layer = _layer()
    prefix = mx.random.normal((2, 3, 32))
    proposed = mx.random.normal((2, 5, 32))
    cache = ArraysCache(2)
    layer(prefix, cache=cache)
    cache.start_speculation()
    layer(proposed, cache=cache)
    cache.trim_ragged([1, 3])
    original = cache._rollbacks[-1]
    assert original.num_tokens == 5 and original.depths == [4, 2]

    cloned = ExternalDraftBatchGenerator._copy_reference_cache([cache])[0]
    record = cloned._rollbacks[-1]
    assert record is not original
    assert record.num_tokens == 5 and record.depths == [4, 2]
    assert record.depths is not original.depths
    assert record.snapshot is not original.snapshot
    assert record.fn is original.fn and record.per_row_fn is original.per_row_fn

    cloned.trim_ragged([2, 1])
    assert original.depths == [4, 2]
    assert cache._rollback_positions == [4, 2]
    assert cloned._rollback_positions == [2, 1]
    for row, keep in enumerate((2, 1)):
        reference = ArraysCache(2)
        layer(mx.concatenate([prefix[row:row + 1], proposed[row:row + 1, :keep]], axis=1),
              cache=reference)
        _assert_cache_equal(cloned.extract(row), reference)


def test_shallow_record_copy_keeps_original_forward_width_after_ragged_trim():
    cache = ArraysCache(1)
    cache.cache = [mx.ones((2, 1))]
    cache.start_speculation()
    def replay(n):
        return [mx.full((2, 1), n)]
    cache.record_rollback(5, replay, [mx.zeros((2, 1))], depths=[4, 2])
    original = cache._rollbacks[-1]
    cloned = copy.copy(original)
    assert cloned.num_tokens == 5 and cloned[0] == 2
    assert cloned.fn is replay and cloned.depths == [4, 2]
