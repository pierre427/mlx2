"""HybridVerifyRows over sliding-window rings beside recurrent state (CPU).

Inkling lanes hold RotatingKVCache (local layers) + recurrent conv state +
KVCache. A verify round must commit any accepted prefix and abort exactly,
including after the ring has wrapped; batched rows over rings fail closed;
rows without recurrent state keep the SegmentedKVRows owner.
"""

import mlx.core as mx
import pytest

from mlx2.runtime.hybrid_verify_rows import (
    HybridVerifyRows,
    HybridVerifyUnsupported,
    is_hybrid_rows,
)
from mlx2.runtime.models.cache import ArraysCache, KVCache, RotatingKVCache


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def _tiny():
    import test_inkling2_cpu as t

    return t._tiny_model()


@pytest.mark.parametrize("prompt,accepted", [(5, 2), (30, 1), (30, 3), (30, 4)])
def test_inkling2_b1_verify_commit_matches_ordinary(prompt, accepted):
    """Window 8: prompt 30 has wrapped every ring before the round."""
    model = _tiny()
    tokens = mx.random.randint(0, 500, (1, prompt + 6)).tolist()[0]
    ref = model.make_cache()
    model(mx.array([tokens[:prompt]]), cache=ref)
    for t in tokens[prompt : prompt + accepted]:
        model(mx.array([[t]]), cache=ref)
    expect = model(mx.array([[tokens[prompt + accepted]]]), cache=ref)

    cache = model.make_cache()
    model(mx.array([tokens[:prompt]]), cache=cache)
    assert is_hybrid_rows([cache], allow_rings=True)
    assert not is_hybrid_rows([cache])  # prompt lookup keeps its per-lane rounds
    owner = HybridVerifyRows([cache])
    txn = owner.begin([4])
    model(mx.array([tokens[prompt : prompt + 4]]), cache=txn.caches)
    (row,) = txn.commit([accepted])
    assert not any(getattr(c, "speculating", False) for c in row)
    got = model(mx.array([[tokens[prompt + accepted]]]), cache=row)
    assert mx.abs(got - expect).max().item() < 1e-4


@pytest.mark.parametrize("plain", [False, True])
def test_inkling2_b1_abort_restores_wrapped_rings(plain):
    model = _tiny()
    tokens = mx.random.randint(0, 500, (1, 34)).tolist()[0]
    ref = model.make_cache()
    model(mx.array([tokens[:30]]), cache=ref)
    expect = model(mx.array([[tokens[30]]]), cache=ref)

    cache = model.make_cache()
    model(mx.array([tokens[:30]]), cache=cache)
    owner = HybridVerifyRows([cache])
    width = 1 if plain else 3
    txn = owner.begin([width], plain_single_token=plain)
    model(mx.array([tokens[31 : 31 + width]]), cache=txn.caches)
    txn.abort()
    assert not any(getattr(c, "speculating", False) for c in cache)
    got = model(mx.array([[tokens[30]]]), cache=cache)
    assert mx.abs(got - expect).max().item() < 1e-4


def test_batched_rows_over_rings_fail_closed_and_kv_only_rows_unchanged():
    model = _tiny()
    rows = [model.make_cache(), model.make_cache()]
    for row in rows:
        model(mx.array([[1, 2, 3]]), cache=row)
    with pytest.raises(HybridVerifyUnsupported, match="rotating KV"):
        HybridVerifyRows(rows)
    # No recurrent plane: rings and plain KV stay on the segmented owner.
    assert not is_hybrid_rows([[RotatingKVCache(8), KVCache()]], allow_rings=True)
    # Recurrent + plain KV (gated-delta families) is still hybrid.
    assert is_hybrid_rows([[ArraysCache(2), KVCache()]])


PLD_PROMPTS = [[3, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 7] * 3,
               [5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7, 8] * 3]


def _greedy(model, prompt, n):
    cache, toks, out = model.make_cache(), list(prompt), []
    for step in range(n):
        logits = model(mx.array([toks if step == 0 else [toks[-1]]]), cache=cache)
        toks.append(int(mx.argmax(logits[0, -1]).item()))
        out.append(toks[-1])
    return out


@pytest.mark.parametrize("lanes", [1, 2])
def test_prompt_lookup_on_inkling_rings_unchanged(lanes):
    """Fresh-review D1/D2: prompt lookup must not route Inkling's
    ring+recurrent lanes to the hybrid owner (two lanes crashed)."""
    from mlx2.runtime.pld import PromptLookupBatchGenerator

    model = _tiny()
    gen = PromptLookupBatchGenerator(model, completion_batch_size=lanes, prefill_step_size=5,
                                     prompt_lookup={"num_draft": 4})
    prompts = PLD_PROMPTS[:lanes]
    uids = gen.insert(prompts, max_tokens=[16] * lanes)
    out, done = {u: [] for u in uids}, set()
    for _ in range(2000):
        _, responses = gen.next()
        for r in responses:
            out[r.uid].append(r.token)
            if r.finish_reason:
                done.add(r.uid)
        if len(done) == lanes:
            break
    assert not gen.scheduler_stats.get("pld_batched_rounds")
    for uid, prompt in zip(uids, prompts):
        assert out[uid] == _greedy(model, prompt, 16)


def test_owner_reuse_plain_rounds_release_and_midforward_raise():
    model = _tiny()
    toks = mx.random.randint(0, 400, (1, 60)).tolist()[0]
    ref = model.make_cache()
    model(mx.array([toks[:30]]), cache=ref)
    cache = model.make_cache()
    model(mx.array([toks[:30]]), cache=cache)
    owner, pos = HybridVerifyRows([cache]), 30
    for accepted, plain in [(2, False), (1, True), (3, False), (1, True)]:
        width = 1 if plain else 4
        txn = owner.begin([width], plain_single_token=plain)
        model(mx.array([toks[pos:pos + width]]), cache=txn.caches)
        txn.commit([accepted])
        for t in toks[pos:pos + accepted]:
            model(mx.array([[t]]), cache=ref)
        pos += accepted
    rings = [c for c in cache if isinstance(c, RotatingKVCache)]
    original = RotatingKVCache.update_and_fetch
    txn = owner.begin([4])

    def boom(self, k, v):
        if self is rings[1]:
            raise RuntimeError("boom")
        return original(self, k, v)

    RotatingKVCache.update_and_fetch = boom
    try:
        with pytest.raises(RuntimeError, match="boom"):
            model(mx.array([toks[pos:pos + 4]]), cache=txn.caches)
    finally:
        RotatingKVCache.update_and_fetch = original
    txn.abort()
    assert not any(getattr(c, "speculating", False) for c in cache)
    txn = owner.begin([1], plain_single_token=True)
    model(mx.array([[toks[pos]]]), cache=txn.caches)
    txn.release_row(0)
    txn.commit([1])
    got = model(mx.array([[toks[pos]]]), cache=cache)
    want = model(mx.array([[toks[pos]]]), cache=ref)
    assert mx.abs(got - want).max().item() < 1e-4
    # A ring speculating under another owner is refused at begin.
    rings[0].start_speculation()
    with pytest.raises(RuntimeError, match="speculation owner"):
        owner.begin([2])
    rings[0].stop_speculation()


def test_inkling2_dspark_refuses_more_than_one_lane():
    from mlx2.adapters.inkling2 import Inkling2Adapter

    adapter = object.__new__(Inkling2Adapter)
    adapter.draft_model = object()
    with pytest.raises(ValueError, match="B1 only"):
        adapter.execution_config(max_lanes=2, prefill_step=512)
