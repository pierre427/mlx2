"""CPU ownership and state-lifetime oracles for the tree-spec intake."""

import copy

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.cache import KVCache
from mlx2.runtime.models.qwen38_shared_draft import SharedDraftKV
from mlx2.runtime.models.qwen38_tree_gdn import live_parent_slots


def test_shared_history_attention_matches_repeated_prefix_and_is_private():
    mx.set_default_device(mx.cpu)
    keys = mx.random.normal((1, 2, 13, 64))
    values = mx.random.normal(keys.shape)
    committed = KVCache()
    committed.update_and_fetch(keys, values)
    before = copy.deepcopy(committed)
    candidate = SharedDraftKV(committed, 3)
    history = [mx.repeat(keys, 3, axis=0), mx.repeat(values, 3, axis=0)]
    for step in range(4):
        q = mx.random.normal((3, 6, 1, 64))
        k, v = (mx.random.normal((3, 2, 1, 64)) for _ in range(2))
        history = [mx.concatenate((old, new), axis=2) for old, new in zip(history, (k, v))]
        expected = mx.fast.scaled_dot_product_attention(q, *history, scale=0.125)
        actual = candidate.append_and_attend(q, k, v, 0.125)
        np.testing.assert_allclose(np.array(actual), np.array(expected), atol=2e-6, rtol=2e-5)
        assert candidate.offset == 14 + step
        assert candidate.kt.shape[2] == step + 1
        assert candidate.keys.shape[0] == 1
    assert committed.offset == before.offset
    assert mx.array_equal(committed.keys, before.keys).item()
    assert mx.array_equal(committed.values, before.values).item()
    with pytest.raises(RuntimeError, match="private attention"):
        candidate.update_and_fetch(keys, values)


def test_shared_history_rejects_unsupported_geometry_before_append():
    cache = KVCache()
    cache.update_and_fetch(mx.zeros((1, 1, 4, 64)), mx.zeros((1, 1, 4, 64)))
    private = SharedDraftKV(cache, 2)
    with pytest.raises(ValueError, match="geometry"):
        private.append_and_attend(mx.zeros((2, 2, 2, 64)),
            mx.zeros((2, 1, 2, 64)), mx.zeros((2, 1, 2, 64)), .125)
    assert private.offset == 4 and private.kt is None


def test_live_parent_slots_matches_independent_per_node_states():
    rng = np.random.default_rng(27)
    topologies = [[-1], list(range(-1, 31)), [-1] + [0] * 31]
    topologies += [[-1] + [int(rng.integers(0, i)) for i in range(1, 32)] for _ in range(100)]
    for parents in topologies:
        reads, writes, peak = live_parent_slots(parents)
        retained, reference = {}, []
        for row, parent in enumerate(parents):
            expected = (1 if parent < 0 else reference[parent]) * 3 + row
            actual = (1 if parent < 0 else retained[reads[row]]) * 3 + row
            assert actual == expected
            reference.append(expected)
            if writes[row] >= 0:
                retained[writes[row]] = actual
                assert writes[row] < peak
            if row not in parents:
                assert writes[row] == -1
        assert peak <= max(1, len(parents) - 1)
    assert live_parent_slots([-1] + [0] * 31)[2] == 1
    assert live_parent_slots(list(range(-1, 31)))[2] == 1


@pytest.mark.parametrize("name", ["models.qwen38_shared_draft", "lane.tiled_prefill"])
def test_candidates_require_explicit_boolean_selection(monkeypatch, name):
    from importlib import import_module
    module = import_module("mlx2.runtime." + name)
    monkeypatch.delenv(module.ENV, raising=False)
    assert not module.enabled()
    monkeypatch.setenv(module.ENV, "1")
    assert module.enabled() and not module.status()["qualified"]
    monkeypatch.setenv(module.ENV, "yes")
    with pytest.raises(ValueError):
        module.enabled()
