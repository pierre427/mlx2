"""Concurrent cost-latched prompt-lookup lanes (the default PLD policy).

GPU defect (Qwen3.8-27B-MLX-8bit@08ac8d9e, B4, four identical 26-31-token
greedy prompts, 2026-10-21): one lane completed at ~5.7 tok/s, the others
emitted 0-3 tokens and were retired by the serving stall watchdog, and chat
followers restoring the leader's published APCv2 boundary emitted garbage or
looped.  Every lane held a ``CostAwarePLDLatch`` (the server's default PLD
policy), so ``_decode_phase`` ran them one at a time through
``_round_batched([lane])``.  For a parked hybrid lane that call dispatched a
*pipelined* plain round, and the next lane's call overwrote
``self._inflight``: every round but the last was never read, committed or
released, although its forward had already advanced that lane's caches.  The
orphaned lane re-fed the same anchor next poll, decoding from a corrupted
state, and only the lane last in poll order progressed reliably.

Now at most one plain round is ever in flight, and cost-latched lanes in a
wider cohort (where the latch is suspended for width) decode plain in the
batched cohort instead of B sequential B1 forwards.
"""

import copy
import queue
import threading
import time
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime.pld import PromptLookupBatchGenerator

# server.DEFAULT_PROMPT_LOOKUP_POLICY
DEFAULT_POLICY = {
    "cost_aware_admission": True,
    "cost_shadow_span": 4,
    "cost_probe_stride": 4,
    "cost_shadow_window": 2,
    "cost_shadow_gate": 0.75,
    "cost_plain_rounds": 8,
    "cost_explore_rounds": 2,
    "cost_margin": 0.05,
    "cost_reprobe_interval": 32,
    "cost_park_rounds": 4,
}

PROMPT = [(7 * i + 3) % 120 + 1 for i in range(31)]


def _hybrid_model():
    from test_segmented_mtp import _tiny_hybrid_mtp_model

    return _tiny_hybrid_mtp_model()


def _greedy(model, prompt, count):
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)
    tokens = []
    for _ in range(count):
        token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(token)
        logits = model(mx.array([[token]]), cache=cache)
    return tokens


def _boundary(model, prefix):
    """A published prompt boundary: the prefix's caches, evaluated."""
    cache = model.make_cache()
    mx.eval(model(mx.array([prefix]), cache=cache))
    mx.eval([c.state for c in cache])
    return cache


def test_cost_latched_followers_restoring_a_boundary_match_greedy_and_batch():
    model = _hybrid_model()
    leader_tokens, follower_tokens = 48, 24
    reference = _greedy(model, PROMPT, leader_tokens)
    generator = PromptLookupBatchGenerator(
        model, completion_batch_size=4, prefill_step_size=64,
        prompt_lookup=dict(DEFAULT_POLICY),
    )
    tokens, finals = {}, {}
    try:
        (leader,) = generator.insert(
            [PROMPT], max_tokens=[leader_tokens], caches=[model.make_cache()]
        )
        tokens[leader] = []
        polls = 0

        def poll():
            nonlocal polls
            polls += 1
            _prompts, responses = generator.next()
            for response in responses:
                tokens[response.uid].append(response.token)
                if response.finish_reason:
                    finals[response.uid] = response

        # The leader decodes alone: its parked latch pipelines plain rounds.
        while len(tokens[leader]) < 4:
            poll()
        assert generator._inflight is not None
        # Followers restore the leader's published 30-token boundary (each a
        # private copy, as an APCv2 restore gives) and prefill the last token.
        boundary = _boundary(model, PROMPT[:-1])
        followers = generator.insert(
            [PROMPT[-1:]] * 3,
            max_tokens=[follower_tokens] * 3,
            caches=[copy.deepcopy(boundary) for _ in range(3)],
            all_tokens=[PROMPT[:-1]] * 3,
        )
        tokens.update({uid: [] for uid in followers})
        assert all(generator.lanes[uid].cost_latch is not None for uid in followers)
        while generator.lanes and polls < 400:
            poll()
        stats = dict(generator.scheduler_stats)
    finally:
        generator.close()
    # No lane starves: every lane finished within a bounded number of polls.
    assert set(finals) == {leader, *followers}
    assert tokens[leader] == reference
    for uid in followers:
        assert tokens[uid] == reference[:follower_tokens], uid
    # The latched lanes ran as one batched cohort, not width-one splits.
    assert stats["pld_batched_max_width"] > 1
    assert stats.get("pld_cost_width1_cohort_splits", 0) == 0
    for uid in followers:
        latch = finals[uid].speculative_receipt["cost_latch"]
        assert latch["activations"] == 0
        assert latch["state"] == "parked"


# ----------------------------------------------- serving, warm APCv2 restore
def _serving_engine(adapter, lanes):
    from mlx2.serving import ServingEngine

    engine = ServingEngine(
        "tiny", adapter_factory=adapter, qualification_mode=True,
        max_lanes=lanes, max_inflight=8, prefill_step=16, mtp=False,
        prompt_lookup=True,
        execution_policy={"prompt_lookup": dict(DEFAULT_POLICY)},
    )
    assert engine.ready.wait(60), engine.error
    return engine


def _collect(job, stall=30.0):
    tokens = []
    while True:
        try:
            event = job.events.get(timeout=stall)
        except queue.Empty:
            return tokens, f"stall after {len(tokens)} tokens"
        if "error" in event:
            return tokens, event["error"]
        delta = (event.get("delta") or {}).get("content", "")
        if delta:
            tokens.extend(int(t) for t in delta.split())
        if "finish_reason" in event:
            return tokens, None


@pytest.fixture
def tiny_serving(monkeypatch):
    from test_apc_hits_hybrid_gdn_self_mtp import make_adapter, tiny_qwen38_mtp

    from mlx2 import memory, serving
    from mlx2.contracts import Capability
    from mlx2.runtime import os_memory

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    model, vocab = tiny_qwen38_mtp()
    # Rollback-capable planes: the lanes take the batched transaction, as
    # the 27B's do.
    model.supports_speculative_rollback = True
    adapter = make_adapter(model, vocab)
    adapter.descriptor = SimpleNamespace(
        capabilities=frozenset({Capability.PROMPT_LOOKUP})
    )
    return adapter


def test_served_b4_followers_on_a_warm_boundary_match_b1(tiny_serving):
    prompt = [(7 * i + 3) % 126 + 1 for i in range(31)]
    b1 = _serving_engine(tiny_serving, 1)
    try:
        reference, error = _collect(
            b1.submit({"tokens": prompt, "max_tokens": 96, "temperature": 0})
        )
        assert error is None
    finally:
        b1.close()
    engine = _serving_engine(tiny_serving, 4)
    try:
        # Publish the 30-token boundary the followers restore.
        warm, error = _collect(
            engine.submit({"tokens": prompt, "max_tokens": 8, "temperature": 0})
        )
        assert error is None and warm == reference[:8]
        results = {}
        leader = engine.submit({"tokens": prompt, "max_tokens": 96, "temperature": 0})
        threads = [threading.Thread(
            target=lambda: results.__setitem__("leader", _collect(leader))
        )]
        threads[0].start()
        deadline = time.monotonic() + 30
        while engine.status()["scheduler"].get("pld_pipelined_rounds", 0) < 2:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        jobs = [
            engine.submit({"tokens": prompt, "max_tokens": 32, "temperature": 0})
            for _ in range(3)
        ]
        for index, job in enumerate(jobs):
            thread = threading.Thread(
                target=lambda i=index, j=job: results.__setitem__(f"f{i}", _collect(j))
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join(120)
        cached = [job.cached_tokens for job in jobs]
        scheduler = engine.status()["scheduler"]
    finally:
        engine.close()
    assert results["leader"] == (reference, None)
    for index in range(3):
        assert results[f"f{index}"] == (reference[:32], None), index
    assert cached == [30, 30, 30]
    assert scheduler.get("pld_cost_width1_cohort_splits", 0) == 0
