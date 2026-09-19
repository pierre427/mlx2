from collections import Counter, deque
from queue import Queue
from types import SimpleNamespace as NS
import threading
import time

import pytest

from mlx2.serving import AdmissionClosed, Job, ServingEngine, SuspendUnavailable


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
