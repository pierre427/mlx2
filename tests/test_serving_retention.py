"""Real worker-loop ownership with metadata-only fake execution/cache objects."""
from collections import Counter
import gc
import queue
import threading
import time
import weakref
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize("lanes", [1, 2, 4])
def test_small_batch_coalescing_keeps_five_millisecond_window(lanes):
    from mlx2.serving import IdleAdmissionCoalescer, coalescing_window

    initial = coalescing_window()
    assert initial == pytest.approx(0.005)
    coalescer = IdleAdmissionCoalescer(initial)
    coalescer.note_attachment(now=1.0)
    assert coalescer.deadline == pytest.approx(1.005)


@pytest.mark.parametrize("initial", [-1, "bad", float("nan"), True])
def test_coalescing_window_validation(initial):
    from mlx2.serving import coalescing_window

    with pytest.raises(ValueError):
        coalescing_window(initial)


def test_declared_batch_cohort_is_published_atomically():
    from mlx2 import serving

    admitted = []
    engine = serving.ServingEngine.__new__(serving.ServingEngine)
    engine.max_lanes = 20
    engine.pending_cohorts = {}
    engine.jobs = {}
    engine.queued_jobs = 0
    engine.incoming = queue.Queue(maxsize=20)
    engine.lock = threading.Lock()
    engine.counts = Counter()
    engine.batch_metrics = NS(
        admitted=lambda job_id, tenant_id, depth:
        admitted.append((job_id, tenant_id, depth))
    )
    cohort = {"id": "atomic-b20", "size": 20}
    jobs = [NS(id=f"job-{i}", request={"batch_cohort": cohort}, tenant_id="t")
            for i in range(20)]

    for job in jobs[:9]:
        engine._publish_job(job)
    assert engine.incoming.empty()
    assert len(engine.jobs) == 9
    assert not admitted

    for job in jobs[9:]:
        engine._publish_job(job)
    published = engine.incoming.get_nowait()
    assert isinstance(published, serving.PublishedCohort)
    assert list(published.jobs) == jobs
    assert engine.queued_jobs == 20
    assert [depth for _, _, depth in admitted] == list(range(1, 21))
    assert engine.counts["batch_cohort_releases"] == 1


def test_batch_cohort_ids_are_isolated_by_tenant():
    from mlx2 import serving

    engine = serving.ServingEngine.__new__(serving.ServingEngine)
    engine.max_lanes = 2
    engine.pending_cohorts = {}
    engine.jobs = {}
    engine.queued_jobs = 0
    engine.incoming = queue.Queue(maxsize=4)
    engine.lock = threading.Lock()
    engine.counts = Counter()
    engine.batch_metrics = NS(admitted=lambda *args: None)
    request = {"batch_cohort": {"id": "shared-name", "size": 2}}
    a1 = NS(id="a1", request=request, tenant_id="tenant-a")
    b1 = NS(id="b1", request=request, tenant_id="tenant-b")
    a2 = NS(id="a2", request=request, tenant_id="tenant-a")

    engine._publish_job(a1)
    engine._publish_job(b1)
    assert engine.incoming.empty()
    engine._publish_job(a2)

    published = engine.incoming.get_nowait()
    assert published.jobs == (a1, a2)
    assert ("tenant-b", "shared-name") in engine.pending_cohorts


def test_incomplete_batch_cohort_fails_closed_at_deadline():
    from mlx2 import serving

    events = []
    released = []
    job = NS(id="job", request={}, tenant_id="t", cache_branch=None,
             admission_hit=None, admission_tokens=None)
    engine = serving.ServingEngine.__new__(serving.ServingEngine)
    engine.pending_cohorts = {
        ("t", "partial"): {"size": 20, "created": 0.0, "jobs": [job]}
    }
    engine.batch_cohort_timeout_seconds = 0.001
    engine.submission_lock = threading.Lock()
    engine.lock = threading.Lock()
    engine.jobs = {job.id: job}
    engine.slots = NS(release=lambda: released.append(job.id))
    engine.batch_metrics = NS(terminal=lambda *args: None)
    engine.counts = Counter()
    engine._emit = lambda observed, event: events.append((observed, event))

    engine._expire_pending_cohorts()

    assert not engine.pending_cohorts
    assert events[0][0] is job
    assert events[0][1]["status"] == 429
    assert "1/20 requests" in events[0][1]["error"]
    assert engine.counts["batch_cohort_timeouts"] == 1
    assert released == [job.id]
    assert not engine.jobs


def test_unpublished_cohort_member_failure_reaches_requests_total():
    from mlx2 import serving
    from mlx2.batch_metrics import BatchRuntimeMetrics

    engine = serving.ServingEngine.__new__(serving.ServingEngine)
    engine.lock = threading.Lock()
    engine.submission_lock = threading.Lock()
    engine.jobs = {}
    engine.slots = threading.Semaphore(8)
    engine.counts = Counter()
    engine.batch_metrics = BatchRuntimeMetrics()
    engine.pending_cohorts = {}
    engine.batch_cohort_timeout_seconds = 0.0
    engine.max_lanes = 4
    engine.fanout_waiting = {}
    engine.incoming = queue.Queue()
    engine.queued_jobs = 0

    def job(name, request):
        return NS(id=name, tenant_id="t", request=request, fault=None,
                  cache_branch=None, admission_hit=None, admission_tokens=None,
                  lora_slot=None, events=queue.Queue(), uid=None,
                  completion_tokens=0)

    def failed_requests():
        counters = engine.batch_metrics.prometheus_snapshot()["counters"]
        return sum(
            value
            for (name, labels), value in counters.items()
            if name == "mlx2_requests_total" and dict(labels)["outcome"] == "failed"
        )

    published = job("published", {"messages": []})
    engine.slots.acquire()
    engine._publish_job(published)
    engine._finish(published, {"error": "boom", "status": 500})
    assert failed_requests() == 1

    staged = job("staged", {"messages": [], "batch_cohort": {"id": "c1", "size": 2}})
    engine.slots.acquire()
    engine._publish_job(staged)
    engine._expire_pending_cohorts()
    assert staged.events.get_nowait()["status"] == 429
    # The cohort member never reached publication but its client received a
    # 429; it must be counted exactly once, and it is no longer running.
    assert failed_requests() == 2
    gauges = engine.batch_metrics.prometheus_snapshot()["gauges"]
    assert gauges["mlx2_num_requests_running"] == 0


def test_publication_records_admission_before_the_worker_can_finish_it():
    from mlx2 import serving
    from mlx2.batch_metrics import BatchRuntimeMetrics

    engine = serving.ServingEngine.__new__(serving.ServingEngine)
    engine.lock = threading.Lock()
    engine.submission_lock = threading.Lock()
    engine.jobs = {}
    engine.slots = threading.Semaphore(8)
    engine.counts = Counter()
    engine.batch_metrics = BatchRuntimeMetrics()
    engine.pending_cohorts = {}
    engine.max_lanes = 4
    engine.fanout_waiting = {}
    engine.queued_jobs = 0

    class WorkerWinsTheRace(queue.Queue):
        """The worker dequeues and fails the job before put_nowait returns."""

        def put_nowait(self, item):
            super().put_nowait(item)
            engine._finish(self.get_nowait(), {"error": "bad sampling", "status": 400})

    engine.incoming = WorkerWinsTheRace()
    job = NS(id="fast", tenant_id="t", request={"messages": []}, fault=None,
             cache_branch=None, admission_hit=None, admission_tokens=None,
             lora_slot=None, events=queue.Queue(), uid=None, completion_tokens=0)
    engine.slots.acquire()
    engine._publish_job(job)

    assert job.events.get_nowait()["status"] == 400
    snapshot = engine.batch_metrics.prometheus_snapshot()
    # It used to stay "running" forever: admitted() ran after terminal().
    assert snapshot["gauges"]["mlx2_num_requests_running"] == 0
    failed = sum(
        value
        for (name, labels), value in snapshot["counters"].items()
        if name == "mlx2_requests_total" and dict(labels)["outcome"] == "failed"
    )
    assert failed == 1

    class Full(queue.Queue):
        def put_nowait(self, item):
            raise queue.Full

    engine.incoming = Full()
    refused = NS(**{**vars(job), "id": "refused", "events": queue.Queue()})
    with pytest.raises(queue.Full):
        engine._publish_job(refused)
    gauges = engine.batch_metrics.prometheus_snapshot()["gauges"]
    assert gauges["mlx2_num_requests_running"] == 0


def test_terminal_event_is_published_after_branch_and_inflight_release():
    from mlx2 import serving

    events = []
    job = NS(id="job", cache_branch=NS(close=lambda: events.append("branch_closed")),
             admission_hit=object(), admission_tokens=[1])
    engine = serving.ServingEngine.__new__(serving.ServingEngine)
    engine.lock = threading.Lock()
    engine.jobs = {job.id: job}
    engine.slots = NS(release=lambda: events.append("slot_released"))
    engine.batch_metrics = NS(terminal=lambda job_id, status:
                              events.append(("terminal", job_id, status)))

    def emit(observed_job, event):
        assert observed_job is job
        assert job.cache_branch is None
        assert job.id not in engine.jobs
        events.append(("emit", event))

    engine._emit = emit
    engine._finish(job, {"finish_reason": "length"})
    assert events == [
        "branch_closed",
        "slot_released",
        ("terminal", "job", "completed"),
        ("emit", {"finish_reason": "length"}),
    ]


def test_idle_worker_does_not_retain_evicted_cache_transfers(monkeypatch):
    from mlx2 import serving, memory
    from mlx2.runtime import apc_v2, generate, os_memory
    references = []
    apc_configuration = {}
    class Payload:
        def __init__(self): references.append(weakref.ref(self))
        def close(self): pass
    class APC:
        def __init__(self, **kw):
            apc_configuration.update(kw)
            self.apc_stats = {}
        def key(self, *a, **kw): return 'key'
        def lookup(self, *a, **kw):
            return NS(cache=[Payload()], cached_tokens=1, remaining_tokens=[2], miss_reason=None,
                      sidecar=NS(state=Payload()))
        def store(self, *a, **kw): pass  # Model immediate pressure eviction.
        def spill_idle_entries(self): pass
        def evict_oldest_unleased(self): return False
        def clear(self): pass
    class Batch:
        scheduler_stats = {}
        def __init__(self, *a, **kw): self.pending = False
        def insert(self, *a, **kw): self.pending = True; return [0]
        def next(self):
            assert self.pending
            self.pending = False
            return ([NS(uid=0, end_of_prompt=True)], [NS(uid=0,
                execution_width=1, finish_reason='length', token=3,
                mtp_state=None, all_tokens=[1, 2, 3], prompt_cache=Payload(),
                mtp_receipt=None)])
        def pop_prompt_boundary(self, uid):
            return dict(committed_only=True, tokens=[1], target_cache=Payload(),
                        mtp_state=Payload(), covered_tokens=1)
        def close(self): pass
    class Detokenizer:
        last_segment = 'ok'
        def reset(self): pass
        def add_token(self, t): pass
        def finalize(self): pass
    class Adapter:
        max_context = 1000
        identity = {'fingerprint': 'fake'}
        environment = {}
        layout = 'fake'
        model = None
        tokenizer = NS(vocab_size=100, eos_token_ids=[], detokenizer=Detokenizer())
        def __init__(self, path): pass
        def profile_name(self, mtp): return 'fake'
        def execution_config(self, **kw): return {'num_draft': 0}
        def prompt_tokens(self, request): return [1, 2]
        def output_parser(self, request):
            return NS(push=lambda *a, **kw: [], stopped=False, tool_count=0)
        def diagnostics(self): return {}
        def close(self): pass
    monkeypatch.setattr(serving, 'runtime_identity', lambda: {'source_sha256': 'fake'})
    monkeypatch.setattr(memory, 'execution_headroom', lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, 'physical_footprint_bytes', lambda: 0)
    monkeypatch.setattr(apc_v2, 'APCv2', APC)
    monkeypatch.setattr(apc_v2, 'MTPAPCSidecar', lambda state, covered: NS(state=state))
    monkeypatch.setattr(generate, 'BatchGenerator', Batch)
    engine = serving.ServingEngine('fake', adapter_factory=Adapter,
                                   qualification_mode=True, mtp=False,
                                   max_inflight=20, max_lanes=20)
    try:
        assert engine.ready.wait(5)
        assert apc_configuration["max_size"] == 20
        assert apc_configuration["max_bytes"] == engine.cache_bytes
        assert apc_configuration["max_tokens"] == engine.max_context == 1000
        # The ready event is the public status boundary.  Qualification takes
        # its counter baseline immediately, before the periodic refresh has a
        # chance to run, so the APCv2 section must already be present.
        initial_status = engine.status()
        assert "apcv2" in initial_status
        assert "execution" in initial_status
        assert "scheduler" in initial_status
        job = engine.submit({'max_tokens': 1})
        assert job.events.get(timeout=5)['finish_reason'] == 'length'
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            gc.collect()
            if references and all(ref() is None for ref in references):
                break
            time.sleep(.01)
        assert len(references) == 5
        assert all(ref() is None for ref in references), 'idle worker retained evicted cache state'
        assert not engine.error
    finally:
        engine.close()


def test_idle_admission_window_starts_after_first_lane_is_attached(monkeypatch):
    """Late HTTP handlers can still join the first fixed-width MTP cohort."""
    from mlx2 import serving, memory
    from mlx2.runtime import apc_v2, generate, os_memory

    first_attached = threading.Event()
    observed_widths = []

    class APC:
        def __init__(self, **kw): self.apc_stats = {}
        def key(self, *a, **kw): return "key"
        def lookup(self, *a, **kw):
            return NS(cache=[], cached_tokens=1, remaining_tokens=[2],
                      miss_reason=None, sidecar=None)
        def store(self, *a, **kw): pass
        def spill_idle_entries(self): pass
        def evict_oldest_unleased(self): return False
        def clear(self): pass

    class Batch:
        scheduler_stats = {}
        def __init__(self, *a, **kw): self.pending = {}; self.uid = 0
        def insert(self, *a, **kw):
            uid = self.uid
            self.uid += 1
            self.pending[uid] = True
            if uid == 0:
                first_attached.set()
            return [uid]
        def next(self):
            width = len(self.pending)
            observed_widths.append(width)
            result = [NS(uid=uid, execution_width=width,
                         finish_reason="length", token=3, mtp_state=None,
                         all_tokens=[1, 2, 3], prompt_cache=[], mtp_receipt=None)
                      for uid in self.pending]
            self.pending.clear()
            return [], result
        def remove(self, uids):
            for uid in uids: self.pending.pop(uid, None)
        def close(self): pass

    class Detokenizer:
        last_segment = "ok"
        def reset(self): pass
        def add_token(self, token): pass
        def finalize(self): pass

    class Adapter:
        max_context = 1000
        identity = {"fingerprint": "fake"}
        environment = {}; layout = "fake"; model = None
        tokenizer = NS(vocab_size=100, eos_token_ids=[], detokenizer=Detokenizer())
        def __init__(self, path): pass
        def profile_name(self, mtp): return "fake"
        def execution_config(self, **kw): return {"num_draft": 0}
        def prompt_tokens(self, request):
            # Exhaust the old dequeue-based 5 ms window before attachment.
            time.sleep(.02)
            return [1, 2]
        def output_parser(self, request):
            return NS(push=lambda *a, **kw: [], stopped=False, tool_count=0)
        def diagnostics(self): return {}
        def close(self): pass

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(apc_v2, "APCv2", APC)
    monkeypatch.setattr(generate, "BatchGenerator", Batch)
    engine = serving.ServingEngine(
        "fake", adapter_factory=Adapter, qualification_mode=True,
        mtp=False, max_lanes=4,
    )
    try:
        assert engine.ready.wait(5)
        jobs = [engine.submit({"max_tokens": 1})]
        assert first_attached.wait(5)
        jobs.extend(engine.submit({"max_tokens": 1}) for _ in range(3))
        results = [job.events.get(timeout=5) for job in jobs]
        assert all(result["finish_reason"] == "length" for result in results)
        assert observed_widths == [4]
        assert all(result["receipt"]["ordinary_compute_width"] == 4
                   for result in results)
        assert not engine.error
    finally:
        engine.close()


def test_atomic_publication_forms_one_segmented_mtp_b20_cohort(monkeypatch):
    """Declared HTTP peers publish atomically despite a long handler gap."""
    from mlx2 import memory, serving
    from mlx2.runtime import apc_v2, generate, os_memory

    first_attached = threading.Event()
    five_attached = threading.Event()
    observed_widths = []

    class APC:
        def __init__(self, **kw): self.apc_stats = {}
        def key(self, *a, **kw): return "key"
        def lookup(self, *a, **kw):
            return NS(cache=[], cached_tokens=1, remaining_tokens=[2],
                      miss_reason=None, sidecar=None)
        def store(self, *a, **kw): pass
        def spill_idle_entries(self): pass
        def evict_oldest_unleased(self): return False
        def clear(self): pass

    class Batch:
        scheduler_stats = {}
        def __init__(self, *a, **kw): self.pending = {}; self.uid = 0
        def insert(self, *a, **kw):
            uid = self.uid
            self.uid += 1
            self.pending[uid] = True
            if len(self.pending) == 1:
                first_attached.set()
            if len(self.pending) == 5:
                five_attached.set()
            return [uid]
        def next(self):
            width = len(self.pending)
            observed_widths.append(width)
            receipt = {
                "route": "segmented_self_mtp",
                "observed_compute_widths": [width],
                "num_draft": 2,
            }
            result = [
                NS(
                    uid=uid,
                    execution_width=width,
                    finish_reason="length",
                    token=3,
                    mtp_state=None,
                    all_tokens=[1, 2, 3],
                    prompt_cache=[],
                    mtp_receipt=receipt,
                )
                for uid in self.pending
            ]
            self.pending.clear()
            return [], result
        def remove(self, uids):
            for uid in uids: self.pending.pop(uid, None)
        def close(self): pass

    class Detokenizer:
        last_segment = "ok"
        def reset(self): pass
        def add_token(self, token): pass
        def finalize(self): pass

    class Adapter:
        max_context = 1000
        identity = {"fingerprint": "fake"}
        environment = {}; layout = "fake"; model = None
        tokenizer = NS(vocab_size=100, eos_token_ids=[], detokenizer=Detokenizer())
        def __init__(self, path): pass
        def profile_name(self, mtp): return "fake"
        def execution_config(self, **kw):
            return {"num_draft": 2, "segment_aware_live_tip": True,
                    "segment_aware_cohort_size": 20}
        def prompt_tokens(self, request):
            # Serial request preparation takes far longer than the ordinary
            # 5 ms coalescing window.  A PublishedCohort still owns the full
            # attachment boundary and must reach one physical B20 cycle.
            time.sleep(.010)
            return [1, 2]
        def output_parser(self, request):
            return NS(push=lambda *a, **kw: [], stopped=False, tool_count=0)
        def diagnostics(self): return {}
        def close(self): pass

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(apc_v2, "APCv2", APC)
    monkeypatch.setattr(generate, "BatchGenerator", Batch)
    engine = serving.ServingEngine(
        "fake",
        adapter_factory=Adapter,
        qualification_mode=True,
        mtp=True,
        max_inflight=20,
        max_lanes=20,
    )
    try:
        assert engine.ready.wait(5)
        request = {
            "max_tokens": 1,
            "batch_cohort": {"id": "live-b20", "size": 20},
        }
        jobs = [engine.submit(request) for _ in range(9)]
        # Exceed both the ordinary 5 ms window and the disproved 25 ms wide
        # timer. No cohort member is visible to the generation queue yet.
        time.sleep(0.040)
        assert not first_attached.is_set()
        assert engine.status()["queue_depth"] == 0
        jobs.extend(engine.submit(request) for _ in range(11))
        assert first_attached.wait(5)
        assert five_attached.wait(5)
        results = [job.events.get(timeout=5) for job in jobs]
        assert observed_widths == [20]
        assert all(
            result["receipt"]["mtp"]["observed_compute_widths"] == [20]
            for result in results
        )
        assert all(
            result["receipt"]["request_controls"]["batch_cohort"]
            == request["batch_cohort"]
            for result in results
        )
        counts = engine.status()["counts"]
        assert counts["batch_cohort_releases"] == 1
        assert counts["batch_cohort_jobs_released"] == 20
        assert not engine.error
    finally:
        engine.close()


@pytest.mark.parametrize("route", ["native_mtp", "segmented_mtp"])
def test_cancelled_prefilling_cohort_member_spares_queued_ungrouped_work(
    monkeypatch, route
):
    """Cancelling one member of a prefilling cohort failed unrelated work.

    The cancel sweep removed only that member, so the next admission pass
    saw the survivor at the queue head and failed every uid in the first
    ``size`` queue slots, including an ungrouped request queued behind it
    (429 "could not fit one scheduler admission boundary").  The survivor
    now fails as soon as its sibling is removed and the ungrouped request
    completes.
    """
    from route_harness import collect, make_engine, patch_host, tiny_qwen38_mtp

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    call = type(model).__call__

    def slow_call(self, *args, **kwargs):
        # Keep the 15-chunk cohort prefill in flight across the cancel sweep.
        time.sleep(0.01)
        return call(self, *args, **kwargs)

    monkeypatch.setattr(type(model), "__call__", slow_call)
    extra = (
        {"segment_aware_live_tip": True, "segment_aware_cohort_size": 3}
        if route == "segmented_mtp"
        else None
    )
    engine = make_engine(model, vocab, max_lanes=3, extra=extra)
    try:
        cohort = {"id": "c1", "size": 2}

        def prompt(seed):
            return [(seed * 7 + 3 * i) % 120 + 1 for i in range(240)]

        first = engine.submit({
            "tokens": prompt(1), "max_tokens": 4, "temperature": 0,
            "batch_cohort": dict(cohort), "return_progress": True,
        })
        second = engine.submit({
            "tokens": prompt(2), "max_tokens": 4, "temperature": 0,
            "batch_cohort": dict(cohort),
        })
        ungrouped = engine.submit(
            {"tokens": [9, 8, 7, 6, 5, 4], "max_tokens": 4, "temperature": 0}
        )
        assert "prompt_progress" in first.events.get(timeout=60)
        first.cancelled.set()
        results = [collect(job, timeout=60) for job in (first, second, ungrouped)]
        alive, error = engine.thread.is_alive(), engine.error
    finally:
        engine.close()
    assert results[0]["error"] == "cancelled"
    assert results[1]["status"] == 429
    assert "lost a member" in results[1]["error"]
    assert results[2].get("error") is None
    assert results[2]["finish"] == "length" and len(results[2]["tokens"]) == 4
    assert alive and error is None


@pytest.mark.parametrize(
    "max_tokens, logprobs",
    [(256, False), (300, True)],
    ids=["overflow_on_finish", "overflow_mid_decode"],
)
def test_output_overflow_is_one_429_terminal_counted_failed(
    monkeypatch, max_tokens, logprobs
):
    """A slow reader overflowing its 256-event queue gets exactly one 429.

    Every later emission used to queue another 429 terminal, and the request
    was counted as a client cancellation; an overflow landing on the finish
    event itself was counted completed with a logged receipt while the
    client got the 429.
    """
    from route_harness import make_engine, patch_host, tiny_qwen38_mtp

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=False, max_lanes=2)

    def outcomes():
        counters = engine.batch_metrics.prometheus_snapshot()["counters"]
        return {
            dict(labels)["outcome"]: value
            for (name, labels), value in counters.items()
            if name == "mlx2_requests_total"
        }

    try:
        job = engine.submit({
            "tokens": list(range(1, 20)), "max_tokens": max_tokens,
            "temperature": 0, "logprobs": logprobs,
        })
        deadline = time.monotonic() + 60
        while engine.jobs and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.2)
        events = []
        while not job.events.empty():
            events.append(job.events.get_nowait())
        terminals = [e for e in events if "error" in e or "finish_reason" in e]
        assert terminals == [
            {"error": "client did not consume output fast enough", "status": 429}
        ]
        assert events[-1] is terminals[0]
        assert outcomes().get("failed") == 1
        assert not outcomes().get("completed") and not outcomes().get("cancelled")
        assert engine.recent_receipts() == []
        assert engine.counts["completed"] == 0
        assert engine.slots._value == engine.max_inflight
        alive, error = engine.thread.is_alive(), engine.error
    finally:
        engine.close()
    assert alive and error is None
