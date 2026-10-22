"""Cost-latched prompt-lookup lanes in one cohort: every lane completes, bitwise the solo run.

apc-warm-hit-hang-20261021: Qwen3.8-27B-MLX-8bit ``--prompt-lookup`` with four identical
short greedy prompts (the h2h B4 cell) completed one lane at ~5.7 tok/s (19 at B1), starved
the others until the serving stall watchdog retired them after 60 s as "memory admission did
not permit progress", and the followers that did run emitted a few tokens of garbage and
stopped.  Mechanism: with the cost-aware latch engaged (``cost_aware_admission``, the server's
default prompt-lookup policy) every lane is ``ordinary`` and parked, so ``_decode_phase`` runs
each one as a lone ``_round_batched([lane])``; ``_pipeline_eligible`` accepts a lone parked
cost-latched hybrid lane, so each dispatch assigned ``self._inflight`` and overwrote the
previous lane's dispatched, unread plain round.  Only the last lane in the poll was ever read
(one token per poll after N-1 wasted forwards); the others got no response, and their caches
kept the appended rounds nobody committed or aborted, so a lane read later continued from a
misaligned state.

This drives the real ServingEngine with a tiny Qwen3.8-shaped hybrid GDN model whose lanes
are batched (``supports_speculative_rollback``, as the 27B ``Model`` declares) under the
server's default cost-aware prompt-lookup policy: a pre-warmed engine, four identical warm-hit
requests at once, and every lane's greedy tokens must equal the solo run's.
"""

import queue
import threading
import time

import pytest
from test_apc_hits_hybrid_gdn_self_mtp import host, make_adapter, tiny_qwen38_mtp  # noqa: F401
from types import SimpleNamespace

from mlx2.contracts import Capability
from mlx2.runtime import pld as pld_module
from mlx2.server import DEFAULT_PROMPT_LOOKUP_POLICY

LANES = 4
MAX_TOKENS = 48
EVENT_TIMEOUT = 20.0


def _engine(adapter, lanes):
    from mlx2.serving import ServingEngine

    engine = ServingEngine(
        "tiny", adapter_factory=adapter, qualification_mode=True, max_lanes=lanes,
        max_inflight=8, prefill_step=16, mtp=False, prompt_lookup=True,
        execution_policy={"prompt_lookup": dict(DEFAULT_PROMPT_LOOKUP_POLICY)},
    )
    assert engine.ready.wait(60), engine.error
    return engine


def _collect(job):
    tokens, started = [], time.monotonic()
    while True:
        try:
            event = job.events.get(timeout=EVENT_TIMEOUT)
        except queue.Empty:
            return {"tokens": tokens, "error": f"no event for {EVENT_TIMEOUT}s after {len(tokens)} tokens"}
        if "error" in event:
            return {"tokens": tokens, "error": event["error"]}
        delta = (event.get("delta") or {}).get("content", "")
        tokens.extend(int(t) for t in delta.split())
        if "finish_reason" in event:
            return {"tokens": tokens, "error": None, "finish": event["finish_reason"],
                    "wall": round(time.monotonic() - started, 3), "cached": int(job.cached_tokens or 0)}


def _batched_hybrid_adapter():
    model, vocab = tiny_qwen38_mtp()
    # The 27B ``Model`` declares this; the tiny text model is the same family.
    model.supports_speculative_rollback = True
    adapter = make_adapter(model, vocab)
    adapter.descriptor = SimpleNamespace(capabilities=frozenset({Capability.PROMPT_LOOKUP}))
    prompt = [(7 * i + 3) % (vocab - 2) + 1 for i in range(31)]
    return adapter, prompt


@pytest.fixture
def pipeline_spy(monkeypatch):
    """How often a pipelined plain round was dispatched while one was already in flight."""
    clobbered = []
    original = pld_module.PromptLookupBatchGenerator._dispatch_plain

    def dispatch_plain(self, lanes, steps, anchors):
        if self._inflight is not None:
            clobbered.append([lane.uid for lane in lanes])
        return original(self, lanes, steps, anchors)

    monkeypatch.setattr(pld_module.PromptLookupBatchGenerator, "_dispatch_plain", dispatch_plain)
    return clobbered


def test_cost_latched_cohort_completes_bitwise_the_solo_run(host, pipeline_spy):  # noqa: F811
    adapter, prompt = _batched_hybrid_adapter()
    request = {"tokens": prompt, "max_tokens": MAX_TOKENS, "temperature": 0}

    solo = _engine(adapter, 1)
    try:
        reference = _collect(solo.submit(dict(request)))
    finally:
        solo.close()
    assert reference["error"] is None and len(reference["tokens"]) == MAX_TOKENS, reference

    engine = _engine(adapter, LANES)
    try:
        warm = _collect(engine.submit({**request, "max_tokens": 4}))
        assert warm["error"] is None, warm
        jobs = [engine.submit(dict(request)) for _ in range(LANES)]
        results = [None] * LANES
        threads = [
            threading.Thread(target=lambda i=i: results.__setitem__(i, _collect(jobs[i])))
            for i in range(LANES)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(EVENT_TIMEOUT * 3)
        assert all(not thread.is_alive() for thread in threads), "a lane never returned"
        status = engine.status()
    finally:
        engine.close()

    for lane, result in enumerate(results):
        assert result is not None and result["error"] is None, (lane, result)
        assert result["cached"] == len(prompt) - 1, (lane, result)
        assert result["finish"] == "length", (lane, result)
        assert result["tokens"] == reference["tokens"], (
            f"lane {lane} diverged from the solo greedy run: {result['tokens'][:12]} vs "
            f"{reference['tokens'][:12]}"
        )
    assert status["healthy"]
    # The mechanism itself: one in-flight slot, never overwritten.
    assert not pipeline_spy, f"pipelined plain rounds dispatched over an in-flight one: {pipeline_spy}"
