from collections import Counter, deque
from queue import Queue
from types import SimpleNamespace as NS
import threading
import time

import pytest

from mlx2.serving import AdmissionClosed, Job, ServingEngine, SuspendUnavailable
from test_structured_deferral import _collect, scripted_engine  # noqa: F401 - shared fixture


def _engine(*, disk=True):
    engine = ServingEngine.__new__(ServingEngine)
    engine.lock = threading.Lock()
    engine.submission_lock = threading.Lock()
    engine.counts = Counter()
    engine._quiesce_complete = threading.Event()
    engine._quiesce_complete.set()
    engine._service_state = "serving"
    engine._service_state_since = time.time()
    engine._service_timestamps = {"serving": engine._service_state_since}
    engine._last_service_transition = {
        "from": None,
        "to": "serving",
        "at": engine._service_state_since,
        "result": {"status": "initialized"},
    }
    engine._drain_deadline = None
    engine._drain_suspend = False
    engine._drain_started_monotonic = None
    engine._quiesce_worker_owned = False
    engine._admin_prefetch_queue = deque()
    engine._admin_prefetch_limit = 32
    engine._admission_leases = set()
    engine.apc_persist_dir = "/tmp/apc" if disk else None
    engine.cache_dir = None
    engine.api_resources = {}
    engine.jobs = {}
    engine.apc = None
    return engine


def test_quiesce_requires_disk_only_for_new_suspend_transition():
    engine = _engine(disk=False)
    with pytest.raises(SuspendUnavailable):
        engine.quiesce(suspend=True)
    assert engine.service_state()["state"] == "serving"
    first = engine.quiesce(suspend=False, drain_timeout_seconds=2)
    assert first["state"] == "draining"
    # Idempotence returns the active transition even though this second body's
    # default suspend request cannot be started without a disk tier.
    assert engine.quiesce()["state"] == "draining"
    assert engine.counts["quiesce_requests"] == 3


def test_accepted_lifecycle_drains_and_new_admission_is_rejected():
    engine = _engine()
    lease = engine.acquire_admission("generation")
    engine.quiesce(suspend=False, drain_timeout_seconds=2)
    with pytest.raises(AdmissionClosed, match="draining"):
        engine._ensure_admission("generation")
    # A continuation from the accepted lifecycle is still allowed while the
    # worker has not claimed the final idle boundary.
    engine._ensure_admission("generation", admitted=True)
    assert engine._worker_quiesce_action() is None
    engine.release_admission(lease)
    action = engine._worker_quiesce_action()
    assert action["target"] == "quiesced" and not action["timed_out"]
    engine._complete_worker_quiesce(action)
    assert engine.service_state()["state"] == "quiesced"
    assert engine.counts["drains_completed"] == 1


def test_resume_during_draining_cancels_before_worker_claim():
    engine = _engine()
    engine.quiesce(suspend=True, drain_timeout_seconds=2)
    resumed = engine.resume()
    assert resumed["state"] == "serving"
    assert resumed["last_transition"]["result"]["status"] == "drain_cancelled"
    assert engine._worker_quiesce_action() is None
    assert engine.counts["resumes"] == 1


def test_drain_timeout_claim_invalidates_internal_bypass():
    engine = _engine()
    engine.jobs["stuck"] = object()
    engine.quiesce(suspend=False, drain_timeout_seconds=0.1)
    engine._drain_deadline = time.monotonic() - 1
    action = engine._worker_quiesce_action()
    assert action["timed_out"]
    assert engine.counts["drain_timeouts"] == 1
    with pytest.raises(AdmissionClosed):
        engine._ensure_admission("batch", admitted=True)


def test_serving_worker_quiesce_check_does_not_touch_resource_locks():
    class UntouchedBatches:
        def status(self):
            raise AssertionError("serving hot path touched BatchManager")

    class UntouchedAPC:
        @property
        def has_pending_prefetch(self):
            raise AssertionError("serving hot path touched APCv2")

        def cancel_pending_prefetch(self):
            raise AssertionError("serving hot path touched APCv2")

    engine = _engine()
    engine.api_resources = {"batches": UntouchedBatches()}
    engine.apc = UntouchedAPC()
    assert engine._worker_quiesce_action() is None


def test_pending_prefetch_completes_during_normal_drain_then_stops_blocking():
    class APC:
        def __init__(self):
            self.pending = True
            self.cancelled = 0
            self.restored = 0

        @property
        def has_pending_prefetch(self):
            return self.pending

        def cancel_pending_prefetch(self):
            if not self.pending:
                return False
            self.pending = False
            self.cancelled += 1
            return True

        def service_pending_prefetch(self):
            assert self.pending
            self.pending = False
            self.restored += 1
            return True

    engine = _engine()
    engine.apc = APC()
    engine.quiesce(suspend=True, drain_timeout_seconds=2)
    assert engine._worker_quiesce_action() is None
    assert engine._service_pending_prefetch(engine.apc)
    assert engine.apc.restored == 1
    action = engine._worker_quiesce_action()
    assert action and not action["timed_out"]
    assert engine.apc.cancelled == 0
    assert not engine.apc.has_pending_prefetch
    engine._complete_worker_quiesce(action, suspend_report={"entries": 0})
    assert engine.service_state()["state"] == "suspended"
    engine.apc.pending = True
    assert engine._service_pending_prefetch(engine.apc)
    assert engine.apc.cancelled == 1
    assert engine.apc.restored == 1


def test_drain_timeout_cancels_pending_prefetch_before_suspend():
    class APC:
        def __init__(self):
            self.pending = True
            self.cancelled = 0

        @property
        def has_pending_prefetch(self):
            return self.pending

        def cancel_pending_prefetch(self):
            if not self.pending:
                return False
            self.pending = False
            self.cancelled += 1
            return True

        def service_pending_prefetch(self):
            raise AssertionError("timed-out prefetch was restored")

    engine = _engine()
    engine.apc = APC()
    engine.jobs["stuck"] = object()
    engine.quiesce(suspend=True, drain_timeout_seconds=0.1)
    engine._drain_deadline = time.monotonic() - 1
    action = engine._worker_quiesce_action()
    assert action["timed_out"]
    assert engine.apc.cancelled == 1
    assert engine.counts["prefetches_cancelled"] == 1
    engine.jobs.clear()
    engine._complete_worker_quiesce(action, suspend_report={"entries": 0})
    assert engine._service_pending_prefetch(engine.apc) is False


def test_admin_prefetch_queue_is_sequenced_on_worker():
    class APC:
        def __init__(self):
            self.calls = []

        def resume_session(self, tenant, session_id, *, ttl_seconds=None):
            self.calls.append((tenant, session_id, ttl_seconds))

    engine = _engine()
    engine.apc = APC()
    state = engine.resume(prefetch_sessions=(("tenant", "one"), ("tenant", "two")))
    assert state["state"] == "serving"
    assert engine.counts["prefetches_queued"] == 2
    assert engine._service_admin_prefetch(engine.apc)
    assert engine._service_admin_prefetch(engine.apc)
    assert engine.apc.calls == [
        ("tenant", "one", None),
        ("tenant", "two", None),
    ]


def test_queued_and_memory_deferred_jobs_finish_before_drain_completes():
    engine = _engine()
    engine.slots = threading.BoundedSemaphore(2)
    engine.slots.acquire()
    engine.slots.acquire()
    engine.batch_metrics = NS(terminal=lambda *_: None)
    engine._emit = lambda job, event: job.events.put(event)
    queued = Job({})
    deferred = Job({})
    engine.jobs = {queued.id: queued, deferred.id: deferred}
    engine.quiesce(suspend=False, drain_timeout_seconds=2)
    assert engine._worker_quiesce_action() is None
    engine._finish(queued, {"finish_reason": "stop"})
    assert engine._worker_quiesce_action() is None
    engine._finish(deferred, {"finish_reason": "stop"})
    action = engine._worker_quiesce_action()
    assert action and not action["timed_out"]
    assert engine.counts["jobs_drained"] == 2


def test_drain_timeout_fails_active_queued_and_deferred_jobs_with_503():
    engine = _engine()
    jobs = [Job({}) for _ in range(3)]
    engine.jobs = {job.id: job for job in jobs}
    engine.slots = threading.BoundedSemaphore(3)
    for _ in jobs:
        engine.slots.acquire()
    engine.batch_metrics = NS(terminal=lambda *_: None)
    engine._emit = lambda job, event: job.events.put(event)
    engine.incoming = Queue()
    engine.incoming.put(jobs[1])
    engine.pending_cohorts = {("t", "c"): {"jobs": [jobs[1]]}}
    engine.fanout_capsules = {}
    engine.fanout_waiting = {}
    engine.queued_jobs = 2
    active = {1: jobs[0]}
    deferred = deque([jobs[2]])
    published = deque([jobs[1]])
    removed = []
    batch = NS(remove=lambda uids: removed.extend(uids))

    engine.quiesce(suspend=False, drain_timeout_seconds=0.1)
    engine._drain_deadline = time.monotonic() - 1
    action = engine._worker_quiesce_action()
    assert action["timed_out"]
    engine._fail_drain_timeout(
        batch, active, deferred, published, None, None
    )
    assert removed == [1]
    assert not engine.jobs and engine.queued_jobs == 0
    for job in jobs:
        assert job.events.get_nowait() == {"error": "drain timeout", "status": 503}


def _slow_scripted_decode(monkeypatch):
    """Make each scripted decode step take a few milliseconds of wall time."""
    from mlx2.runtime import generate

    original_next = generate.BatchGenerator.next

    def slow_next(self):
        time.sleep(0.005)
        return original_next(self)

    monkeypatch.setattr(generate.BatchGenerator, "next", slow_next)


def _wait_for_tokens(job, count=1, timeout=5.0):
    deadline = time.monotonic() + timeout
    while job.completion_tokens < count:
        assert time.monotonic() < deadline, "generation never started"
        time.sleep(0.01)


def test_exclusive_operation_lets_inflight_generation_drain(scripted_engine, monkeypatch):
    build, state = scripted_engine
    engine = build(declare_marker=True)
    _slow_scripted_decode(monkeypatch)
    state["script"] = []  # never EOS: the lane runs to max_tokens
    request = {"messages": [{"role": "user", "content": "x"}], "temperature": 0}
    job = engine.submit({**request, "max_tokens": 120})
    _wait_for_tokens(job)
    order, result = [], {}

    def callback(adapter):
        with engine.lock:
            order.append(("callback", len(engine.jobs)))
        return "ran"

    def operation():
        try:
            result["outcome"] = engine._exclusive_adapter_operation(
                "probe", callback, timeout=5.0
            )
        except Exception as exc:  # noqa: BLE001 - reported through the assertion
            result["outcome"] = f"{type(exc).__name__}: {exc}"

    def late_submit():
        late = engine.submit({**request, "max_tokens": 2})
        order.append(("late_submitted", None))
        result["late"] = _collect(late)[2]

    operation_thread = threading.Thread(target=operation)
    operation_thread.start()
    time.sleep(0.05)
    late_thread = threading.Thread(target=late_submit)
    late_thread.start()
    samples = []
    for _ in range(5):
        time.sleep(0.05)
        samples.append(job.completion_tokens)
    operation_thread.join(10)
    late_thread.join(10)
    # The lane kept decoding while the operation waited for it to drain.
    assert samples == sorted(samples) and samples[-1] > samples[0], samples
    assert result["outcome"] == "ran", result
    assert _collect(job)[2]["finish_reason"] == "length"
    assert job.completion_tokens == 120
    # The operation ran on an idle engine, and a submission that arrived while
    # it was draining waited for it instead of being rejected or overtaking it.
    assert order == [("callback", 0), ("late_submitted", None)]
    assert result["late"]["finish_reason"] == "length"
    assert engine.counts["probe_completed"] == 1
    assert engine.counts["probe_drain_timeouts"] == 0


def test_slow_media_preparation_does_not_stall_an_unrelated_lane(scripted_engine, monkeypatch):
    build, state = scripted_engine
    engine = build(declare_marker=True)
    _slow_scripted_decode(monkeypatch)

    def slow_prepare(request, file_loader=None):
        time.sleep(1.0)  # stands in for image/video/audio decode and file loads
        raise ValueError("media rejected after preprocessing")

    engine.adapter.prepare_multimodal_request = slow_prepare
    state["script"] = []
    job = engine.submit(
        {"messages": [{"role": "user", "content": "x"}], "temperature": 0, "max_tokens": 400}
    )
    _wait_for_tokens(job)
    failures = []

    def media():
        try:
            engine.submit(
                {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
            )
        except ValueError as exc:
            failures.append(str(exc))

    thread = threading.Thread(target=media)
    thread.start()
    samples = []
    for _ in range(8):
        time.sleep(0.1)
        samples.append(job.completion_tokens)
    thread.join(5)
    assert len(set(samples)) >= 6, samples
    assert failures == ["media rejected after preprocessing"]
    job.cancelled.set()
    _collect(job)
    # The failed preparation returned its inflight slot.
    deadline = time.monotonic() + 5
    while True:
        with engine.lock:
            if not engine.jobs:
                break
        assert time.monotonic() < deadline
        time.sleep(0.01)
    for _ in range(engine.max_inflight):
        assert engine.slots.acquire(blocking=False)
    for _ in range(engine.max_inflight):
        engine.slots.release()


def _wait_for_terminal(job, timeout):
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        assert remaining > 0, "no terminal event in time"
        event = job.events.get(timeout=remaining)
        if "finish_reason" in event or "error" in event:
            return event


def test_cancelled_queued_job_releases_its_slot_without_a_free_lane(scripted_engine, monkeypatch):
    build, state = scripted_engine
    engine = build(declare_marker=True, max_lanes=1, max_inflight=2)
    _slow_scripted_decode(monkeypatch)
    state["script"] = []
    request = {"messages": [{"role": "user", "content": "a"}], "temperature": 0}
    running = engine.submit({**request, "max_tokens": 400})
    _wait_for_tokens(running)
    queued = engine.submit({**request, "max_tokens": 5})
    time.sleep(0.05)
    queued.cancelled.set()  # the client disconnected while waiting for a lane
    assert _wait_for_terminal(queued, 1.0) == {"error": "cancelled"}
    # The slot came back while the only lane was still busy.
    replacement = engine.submit({**request, "max_tokens": 5})
    assert running.completion_tokens < 400
    status = engine.status()
    assert status["inflight"] == 2 and status["queue_depth"] == 1
    running.cancelled.set()
    assert _wait_for_terminal(running, 5.0) == {"error": "cancelled"}
    assert _wait_for_terminal(replacement, 5.0)["finish_reason"] == "length"


def test_cancelled_member_fails_a_held_cohort_without_a_free_lane(scripted_engine, monkeypatch):
    build, state = scripted_engine
    engine = build(declare_marker=True, max_lanes=2, max_inflight=3)
    _slow_scripted_decode(monkeypatch)
    state["script"] = []
    request = {"messages": [{"role": "user", "content": "a"}], "temperature": 0}
    running = engine.submit({**request, "max_tokens": 400})
    _wait_for_tokens(running)
    # A declared cohort never joins a live batch: it is held until the
    # running lane finishes.
    members = [
        engine.submit({**request, "max_tokens": 5, "batch_cohort": {"id": "c", "size": 2}})
        for _ in range(2)
    ]
    time.sleep(0.05)
    members[0].cancelled.set()
    for member in members:
        event = _wait_for_terminal(member, 1.0)
        assert event["status"] == 429 and "cancelled" in event["error"], event
    assert running.completion_tokens < 400
    status = engine.status()
    assert status["inflight"] == 1 and status["queue_depth"] == 0
    running.cancelled.set()
    assert _wait_for_terminal(running, 5.0) == {"error": "cancelled"}


def test_submission_racing_worker_exit_gets_a_terminal_event(scripted_engine):
    from types import SimpleNamespace

    build, state = scripted_engine
    engine = build(declare_marker=True)
    worker = engine.thread
    engine.stop_event.set()
    worker.join(10)
    assert not worker.is_alive()
    # A submission that passed its liveness check just before the worker's
    # final sweep: pretend the check still sees a live thread.
    engine.thread = SimpleNamespace(is_alive=lambda: True, join=lambda timeout=None: None)
    job = engine.submit({"messages": [{"role": "user", "content": "x"}], "max_tokens": 2})
    assert _wait_for_terminal(job, 1.0) == {"error": "server stopped", "status": 503}
    with engine.lock:
        assert not engine.jobs


def test_short_self_mtp_checkpoints_suspend_park_and_restore(monkeypatch, tmp_path):
    # A two-token prompt commits a one-token boundary whose draft plane has
    # nothing written.  Saving it raised std::bad_cast: suspend dropped the
    # entry as a failure and a session park never completed.
    from route_harness import collect, make_engine, patch_host, tiny_qwen38_mtp

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, max_lanes=2, cache_dir=str(tmp_path))
    try:
        apc = engine.apc
        request = {"tokens": [5, 6], "max_tokens": 3, "temperature": 0,
                   "session_id": "s1"}
        first = collect(engine.submit(dict(request)), timeout=60)
        assert first.get("error") is None
        engine.apc_session_park("default", "s1", ttl_seconds=600)
        deadline = time.monotonic() + 10
        while (
            engine.apc_session_state("default", "s1")["park_pending"]
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        state = engine.apc_session_state("default", "s1")
        assert state["park_pending"] == 0 and state["resident_entries"] == 0

        collect(engine.submit({"tokens": [9, 8], "max_tokens": 3, "temperature": 0}),
                timeout=60)
        engine.quiesce(suspend=True, drain_timeout_seconds=5)
        assert engine.wait_for_quiesce(10)
        report = engine.service_state()["last_transition"]["result"]["suspend"]
        assert report["failures"] == 0
        assert report["resident_entries_after"] == 0
        engine.resume()

        second = collect(engine.submit(dict(request)), timeout=60)
        assert second["tokens"] == first["tokens"]
        assert apc.apc_stats["idle_disk"]["restores"] >= 1
        assert apc.apc_stats["idle_disk"]["spill_failures"] == 0
        alive, error = engine.thread.is_alive(), engine.error
    finally:
        engine.close()
    assert alive and error is None
