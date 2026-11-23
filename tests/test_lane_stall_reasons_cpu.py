"""Lane stalls end with a reason: numbers on a memory stall, a deadline on holds.

Generic serving helpers (any model, any route) plus the Hy4 DFly2 harness:

* the stall watchdog's 429 names the bytes the lane needed, the bytes that
  were available, the blocking headroom term and the lane state;
* a lane the executor keeps reporting as scheduler-held (cohort boundary
  hold, partial decode fit, prefill waiting its turn) no longer escapes the
  watchdog forever: after ``SCHEDULER_HOLD_TIMEOUT`` without real progress it
  fails with a scheduler-starvation reason.
"""

from types import SimpleNamespace

import pytest

from mlx2 import memory, serving
from mlx2.runtime import os_memory
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator as EDBG

import test_hy_v4_flash_apcv2_cpu as T
from test_hy_v4_flash_apcv2_cpu import artifact  # noqa: F401 - fixture

POLICY = {"execution_policy": {"draft_model": "embedded_dfly", "num_draft": 3}}


# ------------------------------------------------------------ generic units --

def test_memory_stall_event_names_needed_available_and_term():
    job = SimpleNamespace(completion_tokens=3, prompt_tokens=131519, cached_tokens=0)
    record = {"phase": "decode", "append_rows": 8, "lanes": 1, "depth": 131522,
              "generated": 3, "required_bytes": 10 << 30, "available_bytes": 8 << 30,
              "short_bytes": 2 << 30, "blocking_term": "advisory_term",
              "terms": {"advisory_term": 1, "host_term": 2}}
    event = serving.memory_stall_event(record, job, 60.0)
    assert event["status"] == 429
    assert event["error"].startswith("memory admission did not permit progress: decode")
    assert "needs 10.00 GiB, 8.00 GiB available (blocking term advisory_term)" in event["error"]
    wait = event["memory_wait"]
    assert wait["required_bytes"] == 10 << 30 and wait["completion_tokens"] == 3
    assert wait["terms"] == {"advisory_term": 1, "host_term": 2}
    # An executor that reports nothing keeps the historical message.
    plain = serving.memory_stall_event(None, job, 60.0)
    assert plain["error"] == "memory admission did not permit progress"


def test_scheduler_hold_refresh_expires_only_without_real_progress():
    job = SimpleNamespace(last_progress=0.0, scheduler_hold_since=None, scheduler_hold_mark=None)
    refresh = serving.scheduler_hold_refresh
    assert refresh(job, 10.0, 100.0)
    job.last_progress = 10.0  # the hold's own refresh
    assert refresh(job, 60.0, 100.0)
    job.last_progress = 60.0
    job.last_progress = 75.0  # a real response restarts the hold
    assert refresh(job, 80.0, 100.0) and job.scheduler_hold_since == 80.0
    job.last_progress = 80.0
    assert refresh(job, 179.0, 100.0)
    job.last_progress = 179.0
    assert not refresh(job, 180.0, 100.0)  # 100 s held with no real progress


def test_external_lane_refusal_is_recorded_per_lane():
    lanes = [SimpleNamespace(uid=4, history=[0] * 10, generated=2)]
    owner = SimpleNamespace()
    from mlx2.runtime.external_speculative import _note_lane_refusal

    _note_lane_refusal(owner, lanes, 8, 300, 100, prefill=False)
    record = EDBG.memory_wait_receipt(owner, 4)
    assert record["phase"] == "decode" and record["short_bytes"] == 200
    assert record["depth"] == 10 and record["generated"] == 2
    assert record["blocking_term"] in {"advisory_term", "host_term"} or "terms_error" in record


# --------------------------------------------------------- Hy4 DFly2 harness --

@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(serving.ServingEngine, "MEMORY_STALL_SECONDS", 1.0)


def _final(job):
    while True:
        event = job.events.get(timeout=180)
        if "error" in event or "finish_reason" in event:
            return event


def test_memory_stall_429_carries_the_numbers(host, artifact, monkeypatch):  # noqa: F811
    original = EDBG._admit

    def admit(self, lanes, append, *, prefill=False):
        if not prefill:
            self.memory_headroom = lambda: 0  # decode can never fit
        return original(self, lanes, append, prefill=prefill)

    monkeypatch.setattr(EDBG, "_admit", admit)
    engine = T._engine(artifact, **POLICY)
    try:
        event = _final(engine.submit({"tokens": T.SEQ[:40], "max_tokens": 4,
                                      "temperature": 0}))
    finally:
        engine.close()
    assert event["status"] == 429
    assert "needs" in event["error"] and "available" in event["error"]
    wait = event["memory_wait"]
    assert wait["phase"] == "decode" and wait["available_bytes"] == 0
    assert wait["required_bytes"] > 0 and wait["blocking_term"]


def test_scheduler_held_lane_fails_with_a_reason_instead_of_waiting_forever(
        host, artifact, monkeypatch):  # noqa: F811
    # A prefill the scheduler never runs: the lane is reported scheduler-held
    # (not memory-queued) every poll, which used to spare it the watchdog
    # forever.
    monkeypatch.setattr(EDBG, "_prefill_phase", lambda self, candidates: [])
    monkeypatch.setattr(serving.ServingEngine, "SCHEDULER_HOLD_TIMEOUT", 1.0)
    engine = T._engine(artifact, **POLICY)
    try:
        event = _final(engine.submit({"tokens": T.SEQ[:40], "max_tokens": 4,
                                      "temperature": 0}))
        counts = dict(engine.counts)
    finally:
        engine.close()
    assert event.get("code") == "scheduler_starvation", event
    assert "held by the scheduler" in event["error"]
    assert counts.get("scheduler_hold_deadline_failures") == 1


def test_refusal_records_each_lanes_own_share_and_clear_on_admission():
    import mlx.core as mx

    from mlx2.runtime.models.cache import KVCache

    def kv(tokens):
        cache = KVCache()
        cache.update_and_fetch(mx.zeros((1, 2, tokens, 8)), mx.zeros((1, 2, tokens, 8)))
        return cache

    target = SimpleNamespace(num_key_value_heads=2, head_dim=8, num_hidden_layers=2,
                             hidden_size=16, intermediate_size=32, vocab_size=64)
    draft = SimpleNamespace(num_hidden_layers=0, intermediate_size=0, hidden_size=0,
                            num_key_value_heads=0, head_dim=0, feature_width=0)
    headroom = {"bytes": 0}
    batch = SimpleNamespace(memory_headroom=lambda: headroom["bytes"], _target_args=target,
                            draft=SimpleNamespace(config=draft), layers=(), scheduler_stats={},
                            continuation_policy=None, continuation_pool_ast_bridge=False)
    small, large = (SimpleNamespace(uid=uid, cache=[kv(n)], draft_cache=[], remaining=[0],
                                    tail=mx.zeros((1, 0)), history=[0] * n)
                    for uid, n in ((1, 64), (2, 1024)))
    assert not EDBG._admit(batch, [small, large], 1)
    one, two = EDBG.memory_wait_receipt(batch, 1), EDBG.memory_wait_receipt(batch, 2)
    assert one["required_bytes"] < two["required_bytes"]
    assert one["group_required_bytes"] == two["group_required_bytes"] == (
        batch.scheduler_stats["reservation_bytes"])
    headroom["bytes"] = 1 << 40
    assert EDBG._admit(batch, [small], 1)
    assert EDBG.memory_wait_receipt(batch, 1) is None
    assert EDBG.memory_wait_receipt(batch, 2) is not None


def test_hold_deadline_counts_engine_progress_and_real_progress_detection():
    refresh = serving.scheduler_hold_refresh
    job = SimpleNamespace(last_progress=0.0, scheduler_hold_since=None, scheduler_hold_mark=None)
    assert refresh(job, 10.0, 100.0, engine_progress=0.0)
    job.last_progress = 10.0
    # Queued behind another lane's long prefill: that lane progressed at 150.
    assert refresh(job, 200.0, 100.0, engine_progress=150.0)
    job.last_progress = 200.0
    assert not refresh(job, 260.0, 100.0, engine_progress=150.0)  # engine idle 110 s
    a = SimpleNamespace(last_progress=5.0, scheduler_hold_mark=5.0)  # hold's own write
    b = SimpleNamespace(last_progress=7.0, scheduler_hold_mark=3.0)  # a real response
    assert serving.real_progress_at([a, b]) == 7.0
    assert serving.real_progress_at([a]) is None


def _engine2(path, **kw):
    from mlx2.adapters import hy_v4_flash as A

    class Adapter(A.HYV4FlashAdapter):
        def prompt_tokens(self, request):
            if "tokens" in request:
                return list(request["tokens"])
            return super().prompt_tokens(request)

    engine = serving.ServingEngine(str(path), adapter_factory=Adapter, qualification_mode=True,
                                   mtp=False, max_lanes=2, prefill_step=T.PREFILL_STEP, **kw)
    assert engine.ready.wait(120), engine.error
    return engine


def test_queued_behind_a_long_prefill_is_not_failed(host, artifact, monkeypatch):  # noqa: F811
    """Lanes take prefill slices in turn: each waits (scheduler-held) through the
    other lane's slice.

    A slice outlasts the hold deadline; the waiting lane must not fail, because
    the engine is making progress.  Hold clocks are cleared when holds end.
    """
    import time

    original = EDBG._prefill

    def slow(self, lane, **kwargs):
        time.sleep(0.5)
        return original(self, lane, **kwargs)

    monkeypatch.setattr(EDBG, "_prefill", slow)
    monkeypatch.setattr(serving.ServingEngine, "SCHEDULER_HOLD_TIMEOUT", 0.4)
    rescued = []
    refresh = serving.scheduler_hold_refresh

    def watch(job, now, timeout, engine_progress=None):
        alive = refresh(job, now, timeout, engine_progress)
        if alive and now - job.scheduler_hold_since >= timeout:
            rescued.append(job.id)  # only the engine's progress kept it alive
        return alive

    monkeypatch.setattr(serving, "scheduler_hold_refresh", watch)
    engine = _engine2(artifact, **POLICY)
    try:
        jobs = [engine.submit({"tokens": T.SEQ[i: i + 6 * T.PREFILL_STEP + 1],
                               "max_tokens": 3, "temperature": 0}) for i in (0, 50)]
        events = [_final(job) for job in jobs]
        counts = dict(engine.counts)
    finally:
        engine.close()
    assert all("finish_reason" in event for event in events), events
    assert rescued, "no lane waited past the deadline behind a progressing prefill"
    assert counts.get("scheduler_hold_deadline_failures", 0) == 0
    assert all(job.scheduler_hold_since is None for job in jobs)


def test_expired_hold_never_triggers_memory_preemption(host, artifact, monkeypatch):  # noqa: F811
    """Two held lanes, no engine progress, preemption on: both fail as starvation."""
    monkeypatch.setattr(EDBG, "_prefill_phase", lambda self, candidates: [])
    monkeypatch.setattr(serving.ServingEngine, "SCHEDULER_HOLD_TIMEOUT", 1.0)
    policy = {"execution_policy": dict(POLICY["execution_policy"],
                                       memory_preemption={"enabled": True, "stall_seconds": 1})}
    engine = _engine2(artifact, **policy)
    try:
        jobs = [engine.submit({"tokens": T.SEQ[i: i + 40], "max_tokens": 3,
                               "temperature": 0}) for i in (0, 60)]
        events = [_final(job) for job in jobs]
        counts = dict(engine.counts)
    finally:
        engine.close()
    assert [event.get("code") for event in events] == ["scheduler_starvation"] * 2, events
    assert all("engine made no progress" in event["error"] for event in events)
    assert counts.get("memory_preemptions", 0) == 0
    assert counts.get("scheduler_hold_deadline_failures") == 2
