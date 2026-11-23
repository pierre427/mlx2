"""A warm hit that memory admission never lets decode is recomputed, not refused.

Hy4-flash 128K DFly2 with the APCv2 disk tier (2026-11-21, ladder
hy4-ladder128-20261121-b106e81a8-disk, Hy4-flash-MLX-4.6bit-mlx2@bdf83d7e...):
the cold request's pressure eviction spilled its boundary, the warm request
restored it, hit, and was then refused with 429 "memory admission did not
permit progress"; without the disk tier the same warm request missed,
recomputed and was served.  Scaled down on the tiny Hy4 fixture: decode
admission refuses every lane that attached on a prefix hit (the state the hit
carries is what cannot fit), so before the fix the warm request got 429.
"""

import pytest

from mlx2 import memory, serving
from mlx2.runtime import os_memory
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator as EDBG

import test_hy_v4_flash_apcv2_cpu as T
from test_hy_v4_flash_apcv2_cpu import artifact  # noqa: F401 - fixture

POLICY = {"execution_policy": {"draft_model": "embedded_dfly", "num_draft": 3}}


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(serving.ServingEngine, "MEMORY_STALL_SECONDS", 1.0)


def _refuse_decode_after_a_hit(monkeypatch):
    """Decode admission refuses a lane whose prefill resumed past position 0."""
    original_admit, original_prefill = EDBG._admit, EDBG._prefill
    hit_lanes = set()

    def prefill(self, lane, **kwargs):
        if lane.history and lane.uid not in getattr(self, "_seen_lanes", set()):
            hit_lanes.add(lane.uid)
        self._seen_lanes = getattr(self, "_seen_lanes", set()) | {lane.uid}
        return original_prefill(self, lane, **kwargs)

    def admit(self, lanes, append, *, prefill=False):
        if not prefill and any(lane.uid in hit_lanes for lane in lanes):
            return False
        return original_admit(self, lanes, append, prefill=prefill)

    monkeypatch.setattr(EDBG, "_prefill", prefill)
    monkeypatch.setattr(EDBG, "_admit", admit)
    return hit_lanes


def _submit(engine, tokens, progress=None):
    request = {"tokens": list(tokens), "max_tokens": 4, "temperature": 0,
               "logprobs": True, "top_logprobs": 5}
    if progress is not None:
        request["return_progress"] = True
    job = engine.submit(request)
    ids = []
    while True:
        event = job.events.get(timeout=180)
        if progress is not None and "prompt_progress" in event:
            progress.append(event["prompt_progress"]["processed"])
        if "error" in event:
            return job, event, ids
        if "logprob" in event:
            ids.append(event["logprob"]["id"])
        if "finish_reason" in event:
            return job, event, ids


def test_memory_stalled_warm_hit_recomputes_cold_and_is_served(host, artifact, monkeypatch):  # noqa: F811
    prompt = T.SEQ[: 6 * T.PREFILL_STEP + 1]
    engine = T._engine(artifact, **POLICY)
    try:
        _, cold_event, cold_ids = _submit(engine, prompt)
        assert "error" not in cold_event
        hit_lanes = _refuse_decode_after_a_hit(monkeypatch)
        replays = []
        emit = engine._emit_prompt_progress

        def record(job, value, *, replay=False):
            if replay:
                replays.append(value)
            return emit(job, value, replay=replay)

        engine._emit_prompt_progress = record
        progress = []
        job, warm_event, warm_ids = _submit(engine, prompt, progress)
        counts = dict(engine.counts)
    finally:
        engine.close()
    assert hit_lanes, "the warm request did not attach on a hit"
    assert "error" not in warm_event, warm_event
    assert warm_ids == cold_ids
    assert counts.get("apcv2_memory_stall_cold_fallbacks") == 1
    assert job.cached_tokens == 0
    assert job.apcv2_miss_reason == "memory_stall_cold_fallback"
    assert job.attached_cached_tokens == 0
    # return_progress: the client is told the prompt restarts cold.
    assert progress and replays == [(0, len(prompt))]


def _lane(**overrides):
    from types import SimpleNamespace

    fields = dict(attached_cached_tokens=96, attached_hit_kind="exact",
                  prefill_handoff_receipt=None, replaying=False, preemptions=0,
                  completion_tokens=0, apc_cold_fallback=None)
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_cold_fallback_gate():
    eligible = serving.cold_fallback_eligible
    assert eligible(_lane(), 1, 1, {1})
    assert eligible(_lane(), 1, 1, None)  # executor without the hook
    # Prefill handoff: re-admission would restore the same handoff state.
    assert not eligible(_lane(attached_hit_kind="prefill_handoff"), 1, 1, {1})
    assert not eligible(_lane(prefill_handoff_receipt={"observed_used": True}), 1, 1, {1})
    # The current attachment reused nothing (an already-cold replay).
    assert not eligible(_lane(attached_cached_tokens=0), 1, 1, {1})
    assert not eligible(_lane(replaying=True), 1, 1, {1})
    assert not eligible(_lane(preemptions=1), 1, 1, {1})
    # Delivered tokens, a second fallback, other active lanes, not memory-queued.
    assert not eligible(_lane(completion_tokens=1), 1, 1, {1})
    assert not eligible(_lane(apc_cold_fallback="memory_stall_cold_fallback"), 1, 1, {1})
    assert not eligible(_lane(), 1, 2, {1})
    assert not eligible(_lane(), 1, 1, set())
    assert not eligible(None, 1, 1, {1})
