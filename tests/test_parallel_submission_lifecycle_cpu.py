"""Parallel preparation owns the same admission boundary as single requests."""

import threading
import time
from collections import deque
from queue import Queue
from types import SimpleNamespace

import pytest

from mlx2.serving import AdmissionClosed, Job
from test_serving_quiesce import _engine


def engine_for_submission():
    engine = _engine()
    engine.ready = threading.Event()
    engine.ready.set()
    engine.error = None
    engine.thread = SimpleNamespace(is_alive=lambda: True)
    engine.slots = threading.BoundedSemaphore(2)
    engine.incoming = Queue(2)
    engine.queued_jobs = 0
    engine.fanout_waiting = {}
    engine.operation_receipts = deque()
    engine.model_revision = 0
    engine.adapter = object()
    engine._exclusive_operation_done = threading.Condition(engine.submission_lock)
    engine.batch_metrics = SimpleNamespace(
        admitted=lambda *_: None, rejected=lambda *_: None, terminal=lambda *_: None
    )
    return engine


def test_parallel_prepare_prevents_quiesce_claim_and_publishes_accepted_jobs():
    engine = engine_for_submission()
    observations = []

    def prepare(request, *, tenant_id):
        if not observations:
            engine.quiesce(suspend=False, drain_timeout_seconds=10)
        observations.append(engine._worker_quiesce_action())
        return Job(request, tenant_id=tenant_id)

    engine._prepare_job = prepare
    jobs = engine.submit_many([{"prompt": "same"}, {"prompt": "same"}])
    assert observations == [None, None]
    assert engine._admissions_preparing == 0
    assert len(jobs) == len(engine.jobs) == 2
    assert engine._worker_quiesce_action() is None


def test_closed_admission_rejects_parallel_before_adapter_preparation():
    engine = engine_for_submission()
    engine.quiesce(suspend=False, drain_timeout_seconds=10)
    prepared = []
    engine._prepare_job = lambda request, **_: prepared.append(request) or Job(request)
    with pytest.raises(AdmissionClosed):
        engine.submit_many([{"prompt": "same"}, {"prompt": "same"}])
    assert prepared == []
    assert engine._admissions_preparing == 0


def test_parallel_prepare_failure_releases_exclusive_operation_waiter():
    engine = engine_for_submission()
    callback = threading.Event()
    operation = None
    observed = []

    def prepare(request, *, tenant_id):
        nonlocal operation
        operation = threading.Thread(target=lambda: engine._exclusive_adapter_operation(
            "test", lambda _: callback.set(), timeout=2
        ))
        operation.start()
        # Give the exclusive operation time to observe this preparation.
        observed.append(callback.wait(0.05))
        raise ValueError("preparation failed")

    engine._prepare_job = prepare
    try:
        with pytest.raises(ValueError, match="preparation failed"):
            engine.submit_many([{"prompt": "same"}, {"prompt": "same"}])
    finally:
        if operation is not None:
            operation.join(3)
    assert observed == [False]
    assert callback.is_set()
    assert not operation.is_alive()
    assert engine._admissions_preparing == 0
    assert engine.slots.acquire(blocking=False)
    assert engine.slots.acquire(blocking=False)


def test_parallel_publication_can_finish_while_exclusive_operation_waits():
    engine = engine_for_submission()
    callback = threading.Event()
    operation = None
    errors = []

    def run_operation():
        try:
            engine._exclusive_adapter_operation("test", lambda _: callback.set(), timeout=2)
        except Exception as error:
            errors.append(error)

    def prepare(request, *, tenant_id):
        nonlocal operation
        if operation is None:
            operation = threading.Thread(target=run_operation)
            operation.start()
            deadline = time.monotonic() + 1
            while not engine._exclusive_operation_active and time.monotonic() < deadline:
                time.sleep(0.001)
            assert engine._exclusive_operation_active
        return Job(request, tenant_id=tenant_id)

    engine._prepare_job = prepare
    try:
        jobs = engine.submit_many([{"prompt": "same"}, {"prompt": "same"}])
        assert callback.is_set() is False
        for job in reversed(jobs):
            engine._finish(job, {"error": "cancelled"})
    finally:
        if operation is not None:
            operation.join(3)
    assert errors == []
    assert callback.is_set()
    assert not operation.is_alive()
