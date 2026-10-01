"""CPU cohort parity, isolation, rejection and concurrent caller contracts."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

mx = pytest.importorskip("mlx.core")
from mlx2.experimental.hysparse2.batching import ResearchBatcher
from mlx2.experimental.hysparse2.config import Config
from mlx2.experimental.hysparse2.model import Model


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(42)
    yield
    mx.set_default_device(previous)


def test_ragged_cohorts_match_independent_decode():
    model = Model(replace(Config.smoke(), local_window=4))
    model.eval()
    batcher = ResearchBatcher(model, max_lanes=2)
    prompts = [list(range(1, 10)), [3, 5, 7], list(range(2, 11)), [2, 4, 6], [7, 8, 9]]
    reference = [model.prefill(mx.array([p])) for p in prompts]
    logits, caches, receipt = batcher.prefill(prompts)
    assert receipt["model_calls"] == 3 and receipt["padding_tokens"] == 0
    for got, (expected, _) in zip(logits, reference):
        assert float(mx.max(mx.abs(got - expected)).item()) < 1e-3
    lengths = [c.length for c in caches]
    for step in range(3):
        rows = [[12 + step + i] for i in range(len(prompts))]
        got, updated, receipt = batcher.decode(rows, caches)
        for i, row in enumerate(rows):
            expected = model.decode(mx.array([row]), reference[i][1])
            assert float(mx.max(mx.abs(got[i] - expected)).item()) < 1e-3
        assert [c.length for c in caches] == lengths
        caches = updated
        lengths = [c.length for c in caches]
    assert len({id(c.self_kv) for c in caches}) == len(caches)


def test_preflight_failure_does_not_mutate_request_state():
    model = Model(Config.smoke())
    model.eval()
    batcher = ResearchBatcher(model)
    _, caches, _ = batcher.prefill([[1, 2, 3], [4, 5]])
    other = Model(Config.smoke())
    other.eval()
    _, foreign = other.prefill(mx.array([[1, 2]]))
    with pytest.raises(ValueError, match="owner"):
        batcher.decode([[6], [7]], [caches[0], foreign])
    assert caches[0].length == 3
    with pytest.raises(ValueError, match="unique"):
        batcher.decode([[6], [7]], [caches[0], caches[0]])
    assert caches[0].length == 3


def test_serialized_concurrent_callers():
    model = Model(Config.smoke())
    model.eval()
    mx.eval(model.parameters())
    batcher = ResearchBatcher(model)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda row: batcher.prefill([row]), [[1, 2, 3], [4, 5, 6]])
        )
    for row, result in zip([[1, 2, 3], [4, 5, 6]], results):
        expected, _ = model.prefill(mx.array([row]))
        assert float(mx.max(mx.abs(result[0][0] - expected)).item()) < 1e-3


@pytest.mark.parametrize("damage", ["missing_layer", "offset", "boundary", "history"])
@pytest.mark.parametrize("batched", [False, True])
def test_incomplete_state_rejected_before_decode_mutation(damage, batched):
    model = Model(Config.smoke())
    model.eval()
    batcher = ResearchBatcher(model)
    _, cache = model.prefill(mx.array([[1, 2, 3, 4]]))
    if damage == "missing_layer":
        del cache.self_kv[0]
    elif damage == "offset":
        k, v, start = cache.cross_kv[0][0]
        cache.cross_kv[0][0] = k, v, start + 1
    elif damage == "boundary":
        cache.boundary = cache.boundary[:, :, :1]
    else:
        cache.ple_history = None
    before = (cache.length, cache.self_layer_calls, cache.cross_layer_calls, cache.ple_history)
    with pytest.raises(ValueError, match="endpoint state"):
        if batched:
            batcher.decode([[5]], [cache])
        else:
            model.decode(mx.array([[5]]), cache)
    after = (cache.length, cache.self_layer_calls, cache.cross_layer_calls, cache.ple_history)
    assert before[:3] == after[:3] and before[3] is after[3]
