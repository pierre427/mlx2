"""Inkling 2 multi-lane verify over its sliding-window rings (CPU, tiny model).

Each lane holds rings (local visits), KVCache (global visits) and the
four-shortconv recurrent state, across three loop passes. A batched round
through ``HybridVerifyRows(batched_rings=True)`` must give every lane the
logits a B1 round gives, and leave every lane where B1 decode of its accepted
prefix leaves it: wrapped and unwrapped rings, ragged widths, a one-token
lane, abort, a plain round with a released lane, deferred ring commits.
Batched projections and MoE may differ from B1 in the last bits on CPU,
so logits compare at the B1 suite's 1e-4.
"""

import mlx.core as mx
import pytest

from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
from mlx2.runtime.segmented_batch_ring import SegmentedBatchRotatingKVCache

TOL = 1e-4


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def _tiny(**kernels):
    import test_inkling2_cpu as t

    model = t._tiny_model()
    if kernels:
        model.configure_kernels(**kernels)
    return model


def _tokens(lane, n):
    # Distinct, non-repeating streams per lane.
    return [(37 * lane + 11 * i + (i * i) % 13) % 500 + 3 for i in range(n)]


def _prefilled(model, toks, prompt):
    cache = model.make_cache()
    model(mx.array([toks[:prompt]]), cache=cache)
    return cache


def _padded(blocks):
    width = max(len(b) for b in blocks)
    return mx.array([b + [0] * (width - len(b)) for b in blocks])


def _b1_round_logits(model, toks, prompt, count):
    cache = _prefilled(model, toks, prompt)
    txn = HybridVerifyRows([cache]).begin([count])
    logits = model(mx.array([toks[prompt : prompt + count]]), cache=txn.caches)
    txn.abort()
    return logits


def _next_after(model, toks, prompt, accepted):
    """Ordinary decode reference: logits for the token after the accepted prefix."""
    cache = _prefilled(model, toks, prompt)
    for t in toks[prompt : prompt + accepted]:
        model(mx.array([[t]]), cache=cache)
    return model(mx.array([[toks[prompt + accepted]]]), cache=cache)


def _close(a, b):
    return mx.abs(a - b).max().item() < TOL


@pytest.mark.parametrize("prompts,lengths,accepted", [
    ((30, 5), (4, 4), (2, 4)),      # lane 0 wrapped (window 8), lane 1 not
    ((30, 41), (5, 1), (3, 1)),     # both wrapped, a one-token lane
    ((6, 30), (2, 5), (1, 5)),      # ragged, full accept on the wide lane
])
@pytest.mark.parametrize("deferred", [False, True])
def test_two_lanes_match_b1_rounds_and_ordinary_decode(prompts, lengths, accepted, deferred):
    model = _tiny(verify_deferred_commit=True) if deferred else _tiny()
    assert model.supports_batched_ring_verify
    toks = [_tokens(i, 60) for i in range(2)]
    rows = [_prefilled(model, t, p) for t, p in zip(toks, prompts)]
    owner = HybridVerifyRows(rows, deferred_commit=deferred, batched_rings=True)
    txn = owner.begin(list(lengths))
    assert any(isinstance(c, SegmentedBatchRotatingKVCache) for c in txn.caches)
    blocks = [t[p : p + n] for t, p, n in zip(toks, prompts, lengths)]
    logits = model(_padded(blocks), cache=txn.caches)
    for lane, (t, p, n) in enumerate(zip(toks, prompts, lengths)):
        assert _close(logits[lane : lane + 1, :n], _b1_round_logits(model, t, p, n))
    committed = txn.commit(list(accepted))
    for lane, (t, p, a) in enumerate(zip(toks, prompts, accepted)):
        assert not any(getattr(c, "speculating", False) for c in committed[lane])
        got = model(mx.array([[t[p + a]]]), cache=committed[lane])
        assert _close(got, _next_after(model, t, p, a))
    counts = sum(
        layer.self_attn.__dict__.get("segmented_lane_counts", {}).get("visits", 0)
        for layer in model.model.layers
    )
    assert counts > 0
    if deferred:  # the ring visits really took the deferred path, lane by lane
        assert sum(layer.self_attn.__dict__.get("deferred_commit_counts", {}).get("calls", 0)
                   for layer in model.model.layers) > 0


def test_back_to_back_rounds_and_abort_keep_lanes_exact():
    model = _tiny()
    prompts = (30, 9)
    toks = [_tokens(i, 80) for i in range(2)]
    rows = [_prefilled(model, t, p) for t, p in zip(toks, prompts)]
    owner = HybridVerifyRows(rows, batched_rings=True)
    pos = list(prompts)
    for lengths, accepted in [((4, 4), (2, 1)), ((3, 5), (3, 5)), ((5, 2), (1, 2))]:
        txn = owner.begin(list(lengths))
        model(_padded([t[p : p + n] for t, p, n in zip(toks, pos, lengths)]), cache=txn.caches)
        txn.commit(list(accepted))
        pos = [p + a for p, a in zip(pos, accepted)]
    txn = owner.begin([4, 3])
    model(_padded([t[p : p + n] for t, p, n in zip(toks, pos, (4, 3))]), cache=txn.caches)
    txn.abort()
    for lane, (t, p0, p) in enumerate(zip(toks, prompts, pos)):
        got = model(mx.array([[t[p]]]), cache=rows[lane])
        assert _close(got, _next_after(model, t, p0, p - p0))


def test_plain_round_with_released_lane():
    model = _tiny()
    prompts = (30, 12)
    toks = [_tokens(i, 60) for i in range(2)]
    rows = [_prefilled(model, t, p) for t, p in zip(toks, prompts)]
    owner = HybridVerifyRows(rows, batched_rings=True)
    txn = owner.begin([1, 1], plain_single_token=True)
    model(_padded([[toks[0][30]], [toks[1][12]]]), cache=txn.caches)
    txn.release_row(0)
    txn.commit([1, 1])
    assert _close(model(mx.array([[toks[0][30]]]), cache=rows[0]),
                  _next_after(model, toks[0], 30, 0))
    assert _close(model(mx.array([[toks[1][13]]]), cache=rows[1]),
                  _next_after(model, toks[1], 12, 1))


def test_b1_rounds_do_not_take_the_lane_path():
    model = _tiny()
    toks = _tokens(0, 40)
    cache = _prefilled(model, toks, 30)
    txn = HybridVerifyRows([cache]).begin([3])
    model(mx.array([toks[30:33]]), cache=txn.caches)
    txn.commit([2])
    assert not any(layer.self_attn.__dict__.get("segmented_lane_counts")
                   for layer in model.model.layers)


@pytest.mark.parametrize("prompts,lengths,accepted", [
    ((30, 12), (5, 3), (2, 1)),          # both lanes partial, padded lane 1
    ((30, 5, 41), (4, 2, 5), (1, 2, 3)),  # three lanes
])
def test_partial_acceptance_on_every_lane(prompts, lengths, accepted):
    model = _tiny()
    toks = [_tokens(i, 60) for i in range(len(prompts))]
    rows = [_prefilled(model, t, p) for t, p in zip(toks, prompts)]
    txn = HybridVerifyRows(rows, batched_rings=True).begin(list(lengths))
    logits = model(_padded([t[p : p + n] for t, p, n in zip(toks, prompts, lengths)]),
                   cache=txn.caches)
    for lane, (t, p, n) in enumerate(zip(toks, prompts, lengths)):
        assert _close(logits[lane : lane + 1, :n], _b1_round_logits(model, t, p, n))
    txn.commit(list(accepted))
    for lane, (t, p, a) in enumerate(zip(toks, prompts, accepted)):
        assert _close(model(mx.array([[t[p + a]]]), cache=rows[lane]), _next_after(model, t, p, a))
    receipt = model.kernel_receipt()["batched_ring_verify"]
    assert receipt["lane_visits"] == receipt["visits"] * len(prompts)
    assert receipt["lane_rows"] == receipt["visits"] * sum(lengths)


@pytest.mark.parametrize("layer_index", [0, 5, 10, 20])
def test_failure_inside_a_lane_visit_aborts_every_lane_exactly(layer_index):
    model = _tiny()
    prompts = (30, 12)
    toks = [_tokens(i, 60) for i in range(2)]
    rows = [_prefilled(model, t, p) for t, p in zip(toks, prompts)]
    txn = HybridVerifyRows(rows, batched_rings=True).begin([4, 3])
    attn = model.model.layers[layer_index].self_attn
    original = type(attn)._attend

    def boom(self, x, kv, *args):
        if self is attn and any(kv is c for c in rows[1]):  # lane 1's own cache
            raise RuntimeError("boom")
        return original(self, x, kv, *args)

    type(attn)._attend = boom
    try:
        with pytest.raises(RuntimeError, match="boom"):
            model(_padded([t[p : p + n] for t, p, n in zip(toks, prompts, (4, 3))]),
                  cache=txn.caches)
    finally:
        type(attn)._attend = original
    txn.abort()
    for lane, (t, p) in enumerate(zip(toks, prompts)):
        assert not any(getattr(c, "speculating", False) for c in rows[lane])
        assert _close(model(mx.array([[t[p]]]), cache=rows[lane]), _next_after(model, t, p, 0))
