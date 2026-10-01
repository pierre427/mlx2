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


@pytest.mark.parametrize("value", [-1, 48])
@pytest.mark.parametrize("operation", ["prefill", "decode"])
def test_invalid_tokens_reject_before_existing_cache_mutation(value, operation):
    model = Model(replace(Config.smoke(), prefill_chunk=2))
    model.eval()
    _, cache = model.prefill(mx.array([[1, 2]]))
    before = (cache.length, cache.self_layer_calls, cache.cross_layer_calls, cache.boundary, cache.ple_history)
    with pytest.raises(ValueError, match="vocabulary"):
        if operation == "prefill":
            model.prefill(mx.array([[3, 4, value]]), cache)
        else:
            model.decode(mx.array([[value]]), cache)
    after = (cache.length, cache.self_layer_calls, cache.cross_layer_calls, cache.boundary, cache.ple_history)
    assert before[:3] == after[:3] and before[3] is after[3] and before[4] is after[4]


@pytest.mark.parametrize("phase", ["merge", "decode_split", "prefill_split"])
def test_model_revision_change_cannot_relabel_batched_state(monkeypatch, phase):
    model = Model(Config.smoke())
    model.eval()
    batcher = ResearchBatcher(model, max_lanes=2)
    _, caches, _ = batcher.prefill([[1, 2, 3], [4, 5, 6]])
    before = [list(c.arrays()) for c in caches]
    old_owner = model._cache_owner
    method = "_merge" if phase == "merge" else "_split"
    original = getattr(batcher, method)
    def changed(*args):
        model.update({"embedding": {"weight": model.embedding.weight}})
        return original(*args)
    monkeypatch.setattr(batcher, method, changed)
    with pytest.raises(ValueError, match="revision|owner"):
        if phase == "prefill_split":
            batcher.prefill([[1, 2, 3], [4, 5, 6]])
        else:
            batcher.decode([[7], [8]], caches)
    assert all(c.owner is old_owner and c.length == 3 for c in caches)
    assert all(all(a is b for a, b in zip(previous, c.arrays()))
               for previous, c in zip(before, caches))
def test_prefill_cannot_return_cohorts_from_different_revisions(monkeypatch):
    model = Model(Config.smoke())
    model.eval()
    batcher = ResearchBatcher(model, max_lanes=2)
    original = model.prefill
    calls = 0

    def changed(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            model.update({"embedding": {"weight": model.embedding.weight}})
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "prefill", changed)
    with pytest.raises(ValueError, match="revision|owner"):
        batcher.prefill([[1, 2, 3], [4, 5]])


@pytest.mark.parametrize("operation", ["prefill", "decode", "prefill_cache_only"])
@pytest.mark.parametrize("change", ["parameters", "capsules"])
def test_ordinary_inference_rejects_revision_changed_before_return(monkeypatch, operation, change):
    model = Model(Config.smoke())
    model.eval()
    tokens = mx.array([[1, 2, 3, 4]])
    _, cache = model.prefill(tokens)
    hook = "_append" if operation == "prefill_cache_only" else "_cross"
    original = getattr(model, hook)

    def changed(*args, **kwargs):
        value = original(*args, **kwargs)
        if change == "parameters":
            model.update({"embedding": {"weight": model.embedding.weight}})
        else:
            model.attach_semantic_capsules(None)
        return value

    monkeypatch.setattr(model, hook, changed)
    with pytest.raises(ValueError, match="revision|owner|another model"):
        if operation == "decode":
            model.decode(mx.array([[5]]), cache)
        else:
            model.prefill(tokens, return_logits=operation != "prefill_cache_only")
    monkeypatch.setattr(model, hook, original)
    _, fresh = model.prefill(tokens)
    assert bool(mx.all(mx.isfinite(model.decode(mx.array([[5]]), fresh))).item())
