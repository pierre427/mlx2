"""Resident reload lifecycle: admission, cache ownership and failure containment.

These are model-free/CPU contract tests, not checkpoint qualification.
"""

import threading
import time
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
from test_serving_quiesce import _engine as quiesce_engine

from mlx2.serving import AdmissionClosed, HostPromptCache

_ENGINES = []


@pytest.fixture(autouse=True)
def release_fixture_owners(monkeypatch):
    import mlx2.kernel_reload as reload_module

    # Production intentionally retains contamination after teardown. Each CPU
    # fixture represents its own interpreter, so isolate both registries.
    monkeypatch.setattr(reload_module, "_OWNERS", {})
    monkeypatch.setattr(reload_module, "_CHANGED_OWNERS", set())
    yield
    from mlx2.kernel_reload import unregister_owner

    for engine in _ENGINES:
        unregister_owner(engine)
    _ENGINES.clear()


class FakeMLX:
    def __init__(self):
        self.calls = []

    def synchronize(self):
        self.calls.append(("synchronize", threading.get_ident()))

    def clear_compile_cache(self):
        self.calls.append(("clear_compile_cache", threading.get_ident()))

    def clear_cache(self):
        self.calls.append(("clear_cache", threading.get_ident()))

    def get_cache_memory(self):
        return 0


class FakeReloader:
    def __init__(self, *, prepare_error=None, apply_error=None):
        self.prepare_error = prepare_error
        self.apply_error = apply_error
        self.prepared = []
        self.applied = []

    def prepare(self, *, modules=(), settings=(), calls=(), adapter=None):
        if self.prepare_error:
            raise self.prepare_error
        plan = NS(modules=tuple(modules), settings=tuple(settings), calls=tuple(calls))
        self.prepared.append(plan)
        return plan

    def apply(self, plan, adapter=None):
        self.applied.append((plan, adapter, threading.get_ident()))
        if self.apply_error:
            raise self.apply_error
        return {
            "status": "applied", "generation": len(self.applied),
            "modules": list(plan.modules), "source_sha256": "fixture-source",
            "settings": list(plan.settings), "calls": list(plan.calls),
            "functions_patched": 1, "classes_preserved": 1,
            "caches_invalidated": 1,
        }

    def status(self):
        return {"enabled": True, "generation": len(self.applied)}


class Cache:
    def __init__(self, *, error=None):
        self.entries = 3
        self.calls = []
        self.error = error
        self.has_pending_prefetch = False

    def clear(self, **_kwargs):
        self.calls.append("clear")
        if self.error:
            raise self.error
        entries, self.entries = self.entries, 0
        return {"entries": entries, "pinned_parks_lost": 1, "lost_sessions": ["old"]}

    def cancel_pending_prefetch(self):
        old, self.has_pending_prefetch = self.has_pending_prefetch, False
        return old


def engine_fixture(*, reloader=None, apc=None):
    engine = quiesce_engine()
    engine._kernel_reloader = reloader or FakeReloader()
    engine._kernel_reload_request = None
    engine._kernel_reload_mutex = threading.Lock()
    engine._kernel_reload_epoch = 0
    engine._kernel_reload_identity = None
    engine._kernel_reload_last = None
    engine._kernel_reload_rebind = None
    engine.prompt_lock = threading.Lock()
    engine._exclusive_operation_done = threading.Condition(engine.submission_lock)
    engine._admissions_preparing = 0
    engine._terminal_cleanup_pending = deque()
    engine._terminal_cleanup_in_progress = 0
    engine._worker_stopped = False
    engine.stop_event = threading.Event()
    engine.ready = threading.Event()
    engine.ready.set()
    engine.error = None
    engine.thread = NS(is_alive=lambda: True)
    engine.model_revision = 4
    engine.operation_receipts = deque(maxlen=32)
    engine.host_prompt_cache = HostPromptCache()
    engine.host_prompt_cache.put({"prompt": "old"}, [1, 2, 3])
    engine.incremental_tokenizer_cache = NS(clear=lambda: None)
    engine.adapter = NS(
        model=NS(weight=object()), tokenizer=object(),
        identity={"fingerprint": "fixture-artifact"},
    )
    engine.snapshot = {"qualification": "qualified", "route_receipt": {"qualified": True}}
    engine.qualification_state = "qualified"
    engine.qualification = None
    engine.fanout_capsules = {}
    engine._prefill_handoff_execution_digest = "old-handoff-digest"
    engine.apc = apc or Cache()
    from mlx2.kernel_reload import register_owner

    register_owner(engine)
    _ENGINES.append(engine)
    return engine


def start_reload(engine, **kwargs):
    result = {}

    def invoke():
        try:
            result["receipt"] = engine.reload_kernels(**kwargs)
        except BaseException as error:  # noqa: BLE001 - capture worker/caller failures as test evidence
            result["error"] = error

    caller = threading.Thread(target=invoke, daemon=True)
    caller.start()
    deadline = time.monotonic() + 2
    while engine._kernel_reload_request is None and caller.is_alive():
        assert time.monotonic() < deadline, "reload never queued a worker request"
        time.sleep(0.001)
    return caller, result


def finish_reload(engine, caller, result, *, mx=None, timed_out=False):
    if timed_out:
        engine._drain_deadline = time.monotonic() - 1
    action = engine._worker_quiesce_action()
    assert action is not None, "idle reload did not acquire the worker boundary"
    assert engine._worker_reload_kernels(action, mx or FakeMLX())
    caller.join(timeout=2)
    assert not caller.is_alive(), "reload caller remained blocked after worker receipt"
    return result


def test_reload_runs_on_worker_keeps_model_storage_and_purges_state():
    engine = engine_fixture()
    model, weight, tokenizer = engine.adapter.model, engine.adapter.model.weight, engine.adapter.tokenizer
    mx = FakeMLX()
    caller, result = start_reload(engine, modules=["fixture.kernel"])
    assert engine._kernel_reloader.applied == []
    with pytest.raises(AdmissionClosed):
        engine._ensure_admission("generation")
    finish_reload(engine, caller, result, mx=mx)
    assert "error" not in result, result.get("error")
    assert engine._kernel_reloader.applied[0][2] == threading.get_ident()
    assert all(tid == threading.get_ident() for _, tid in mx.calls)
    assert engine.adapter.model is model and engine.adapter.model.weight is weight
    assert engine.adapter.tokenizer is tokenizer
    assert engine.apc.entries == 0 and engine.apc.calls == ["clear"]
    assert engine.host_prompt_cache.status()["entries"] == 0
    purge = result["receipt"]["apcv2"]
    assert engine.counts["kernel_reload_host_prompt_entries_purged"] == purge["host_prompt_cache_entries"] == 1
    assert engine.counts["kernel_reload_mlx_cache_bytes_released"] == purge["mlx_cache_bytes_released"]
    assert engine._kernel_reload_epoch == 1
    assert engine.model_revision > 4
    assert engine.qualification_state == "unqualified"
    assert engine._prefill_handoff_execution_digest != "old-handoff-digest"
    assert engine.service_state()["state"] == "serving"


@pytest.mark.parametrize("kind", ["job", "preparing", "lease", "cleanup", "batch", "prefetch"])
def test_reload_waits_for_every_owner_of_old_execution_state(kind):
    engine = engine_fixture()
    if kind == "job":
        engine.jobs["old"] = object()
    elif kind == "preparing":
        engine._admissions_preparing = 1
    elif kind == "lease":
        engine._admission_leases.add("old")
    elif kind == "cleanup":
        engine._terminal_cleanup_in_progress = 1
    elif kind == "batch":
        engine._active_batch_jobs = lambda: 1
    else:
        engine.apc.has_pending_prefetch = True
    caller, result = start_reload(engine, modules=["fixture.kernel"])
    try:
        assert engine._worker_quiesce_action() is None
        assert engine._kernel_reloader.applied == [] and engine.apc.calls == []
        engine.jobs.clear()
        engine._admissions_preparing = 0
        engine._admission_leases.clear()
        engine._terminal_cleanup_in_progress = 0
        engine._active_batch_jobs = lambda: 0
        engine.apc.has_pending_prefetch = False
        finish_reload(engine, caller, result)
        assert "error" not in result, result.get("error")
    finally:
        engine.stop_event.set()
        caller.join(timeout=2)


def test_drain_timeout_keeps_old_code_and_active_request_alive():
    engine = engine_fixture()
    old = engine.jobs["old"] = object()
    caller, result = start_reload(engine, modules=["fixture.kernel"], drain_timeout_seconds=0.1)
    finish_reload(engine, caller, result, timed_out=True)
    assert isinstance(result.get("error"), TimeoutError)
    assert engine.jobs["old"] is old
    assert engine._kernel_reloader.applied == [] and engine.apc.calls == []
    assert engine.model_revision == 4 and engine._kernel_reload_epoch == 0
    assert engine.service_state()["state"] == "serving"


def test_drain_timeout_can_abandon_reload_while_terminal_cleanup_is_pending():
    engine = engine_fixture()
    engine._terminal_cleanup_in_progress = 1
    caller, result = start_reload(engine, modules=["fixture.kernel"], drain_timeout_seconds=0.1)
    try:
        finish_reload(engine, caller, result, timed_out=True)
        assert isinstance(result.get("error"), TimeoutError)
        assert engine._terminal_cleanup_in_progress == 1
        assert engine._kernel_reloader.applied == [] and engine.apc.calls == []
    finally:
        engine.stop_event.set()
        caller.join(timeout=2)


def test_syntax_error_is_rejected_before_admission_or_cache_changes():
    engine = engine_fixture(reloader=FakeReloader(prepare_error=SyntaxError("broken kernel")))
    with pytest.raises(SyntaxError, match="broken kernel"):
        engine.reload_kernels(modules=["fixture.kernel"])
    assert engine._kernel_reload_request is None
    assert engine.service_state()["state"] == "serving"
    assert engine.apc.calls == []


def test_second_reload_cannot_replace_a_pending_request():
    engine = engine_fixture()
    engine.jobs["old"] = object()
    caller, result = start_reload(engine, modules=["fixture.first"])
    pending = engine._kernel_reload_request
    try:
        with pytest.raises(RuntimeError, match="already pending"):
            engine.reload_kernels(modules=["fixture.second"])
        assert engine._kernel_reload_request is pending
        assert len(engine._kernel_reloader.prepared) == 1
        engine.jobs.clear()
        finish_reload(engine, caller, result)
        assert "error" not in result, result.get("error")
    finally:
        engine.stop_event.set()
        caller.join(timeout=2)


def test_stopped_worker_releases_the_waiting_admin_caller():
    engine = engine_fixture()
    engine.jobs["old"] = object()
    caller, result = start_reload(engine, modules=["fixture.kernel"])
    engine.stop_event.set()
    caller.join(timeout=2)
    assert not caller.is_alive()
    assert isinstance(result.get("error"), RuntimeError)
    assert "stopped" in str(result["error"])
    assert engine._kernel_reloader.applied == [] and engine.apc.calls == []


@pytest.mark.parametrize("failed", [False, True])
def test_shutdown_waits_for_claimed_apply_and_records_only_its_outcome(failed):
    entered, release, attempted_cancel = (threading.Event() for _ in range(3))

    class BlockingReloader(FakeReloader):
        def apply(self, plan, adapter=None):
            entered.set()
            assert release.wait(2), "test did not release the worker-owned apply"
            return super().apply(plan, adapter=adapter)

    error = RuntimeError("worker apply failed") if failed else None
    engine = engine_fixture(reloader=BlockingReloader(apply_error=error))
    cancel = engine._cancel_pending_kernel_reload

    def observe_cancel(*args, **kwargs):
        cancel(*args, **kwargs)
        if kwargs.get("only_unclaimed"):
            attempted_cancel.set()

    engine._cancel_pending_kernel_reload = observe_cancel
    caller, result = start_reload(engine, modules=["fixture.kernel"])
    pending = engine._kernel_reload_request
    action = engine._worker_quiesce_action()
    assert action is not None and engine._quiesce_worker_owned
    worker = threading.Thread(
        target=engine._worker_reload_kernels, args=(action, FakeMLX()), daemon=True,
    )
    worker.start()
    try:
        assert entered.wait(2), "worker never entered apply"
        engine.stop_event.set()
        assert attempted_cancel.wait(2), "caller never observed shutdown"
        assert caller.is_alive() and result == {}
        assert engine._kernel_reload_request is pending
        assert not pending["event"].is_set()
        assert engine.counts["kernel_reload_rejected"] == 0
    finally:
        release.set()
        worker.join(timeout=2)
        caller.join(timeout=2)
    assert not worker.is_alive() and not caller.is_alive()
    assert len(engine._kernel_reloader.applied) == 1
    if failed:
        assert result == {"error": error}
    else:
        assert set(result) == {"receipt"}
        assert result["receipt"]["status"] == "completed"
    # Worker teardown and a late caller observation must not add a second
    # settlement after the worker has already published its outcome.
    cancel(reason="generation worker stopped")
    cancel(pending, "late shutdown observation")
    assert len(engine.operation_receipts) == 1
    assert engine.counts["kernel_reload_completed"] == int(not failed)
    assert engine.counts["kernel_reload_failed"] == int(failed)
    assert engine.counts["kernel_reload_rejected"] == 0
    assert engine.counts["kernel_reload_requests"] == 1


@pytest.mark.parametrize("shutdown", ["unclaimed_stop", "dead_after_claim", "worker_finally"])
def test_unexecuted_shutdown_cancels_once_without_late_worker_publication(shutdown):
    engine = engine_fixture()
    caller, result = start_reload(engine, modules=["fixture.kernel"])
    pending = engine._kernel_reload_request
    if shutdown == "dead_after_claim":
        assert engine._worker_quiesce_action() is not None
        assert engine._quiesce_worker_owned
        engine.thread = NS(is_alive=lambda: False)
    elif shutdown == "worker_finally":
        # This is the exact settlement hook in ServingEngine._run's finally.
        engine._cancel_pending_kernel_reload(reason="generation worker stopped")
    else:
        engine.stop_event.set()
    caller.join(timeout=2)
    assert not caller.is_alive()
    assert set(result) == {"error"}
    assert isinstance(result["error"], RuntimeError)
    assert result["error"].kernel_reload_recorded is True
    assert pending["event"].is_set()
    assert engine._kernel_reload_request is None
    assert engine._quiesce_worker_owned is False
    # Only a worker-claimed drain may have started purging/applying; an
    # unclaimed one executed nothing and is a worker_stopped rejection.
    claimed = shutdown == "dead_after_claim"
    # Nothing ran unclaimed, so no quarantine; but the worker cannot serve.
    assert engine.service_state()["state"] == ("kernel_reload_failed" if claimed else "stopped")
    assert engine.kernel_reload_status()["phase"] == ("failed" if claimed else "idle")
    if not claimed:
        assert result["error"].status == 503
        assert engine.operation_receipts[0]["rejection"] == "worker_stopped"
        assert engine.operation_receipts[0]["code_executed"] is False
        assert engine.snapshot["qualification"] == "qualified"
    engine._cancel_pending_kernel_reload(pending, "late shutdown observation")
    engine._cancel_pending_kernel_reload(reason="generation worker stopped")
    assert engine._worker_quiesce_action() is None
    assert engine._worker_reload_kernels({"timed_out": False}, FakeMLX()) is False
    assert engine._kernel_reloader.applied == [] and engine.apc.calls == []
    assert len(engine.operation_receipts) == 1
    assert engine.operation_receipts[0]["status"] == ("failed" if claimed else "rejected")
    assert engine.counts["kernel_reload_failed"] == int(claimed)
    assert engine.counts["kernel_reload_failed_other"] == int(claimed)
    assert engine.counts["kernel_reload_completed"] == 0
    assert engine.counts["kernel_reload_rejected"] == int(not claimed)
    assert engine.counts["kernel_reload_rejected_worker_stopped"] == int(not claimed)
    assert engine.counts["kernel_reload_requests"] == 1


def _stopped_state_report(engine):
    from mlx2.prometheus import render_engine_metrics

    engine.expert_stream = None
    rendered = render_engine_metrics(engine)
    gauges = sorted(line for line in rendered.splitlines() if line.startswith("mlx2_server_state{"))
    return engine.service_state()["state"], gauges


@pytest.mark.parametrize("shutdown", ["close", "crash"])
def test_worker_stopped_reload_reports_the_same_state_as_an_ordinary_stopped_engine(shutdown):
    def stop(engine):
        if shutdown == "close":
            engine.stop_event.set()
        else:
            # ServingEngine._run's finally after an exception: no stop
            # request, settlement hook first, then the published flag.
            engine.error = "RuntimeError: device lost"
            engine._cancel_pending_kernel_reload(reason=engine.error)
        with engine.lock:
            engine._worker_stopped = True

    engine = engine_fixture()
    caller, result = start_reload(engine, modules=["fixture.kernel"])
    if shutdown == "close":
        engine.stop_event.set()
        caller.join(timeout=2)
        stop(engine)
    else:
        stop(engine)
        caller.join(timeout=2)
    assert not caller.is_alive() and set(result) == {"error"}
    # Built after the reload: one interpreter has one kernel-reload owner.
    ordinary = engine_fixture()
    stop(ordinary)
    expected = _stopped_state_report(ordinary)
    assert expected[0] == "stopped"
    assert 'mlx2_server_state{state="stopped"} 1' in expected[1]
    assert 'mlx2_server_state{state="serving"} 0' in expected[1]
    assert _stopped_state_report(engine) == expected
    receipt = engine.kernel_reload_status()["last_reload"]
    assert receipt["status"] == "rejected" and receipt["rejection"] == "worker_stopped"
    assert receipt["code_executed"] is False and receipt["qualified"] is False
    assert engine.kernel_reload_status()["phase"] == "idle"
    assert engine.snapshot["qualification"] == "qualified"
    assert engine._kernel_reloader.applied == [] and engine.apc.calls == []
    # A later request against either stopped engine is refused the same way.
    for stopped in (ordinary, engine):
        with pytest.raises(RuntimeError) as refused:
            stopped.reload_kernels(modules=["fixture.kernel"])
        assert refused.value.status == 503
        assert refused.value.kernel_reload_rejection == "not_ready"


def test_refused_reload_on_quarantined_engine_keeps_failed_phase():
    engine = engine_fixture(reloader=FakeReloader(apply_error=RuntimeError("bad kernel")))
    caller, result = start_reload(engine, modules=["fixture.kernel"])
    finish_reload(engine, caller, result)
    assert engine.service_state()["state"] == "kernel_reload_failed"
    with pytest.raises(RuntimeError, match="serving, healthy"):
        engine.reload_kernels(modules=["fixture.kernel"])
    assert engine.kernel_reload_status()["phase"] == "failed"
    assert engine.counts["kernel_reload_rejected_not_ready"] == 1
    assert len(engine._kernel_reloader.prepared) == 1


def test_source_saved_during_drain_is_refused_before_purge_and_keeps_serving(tmp_path, monkeypatch):
    import importlib.util
    import sys

    from mlx2.kernel_reload import KernelReloader, KernelReloadError

    name = "mlx2.runtime.kernel_reload_drain_edit_fixture"
    path = tmp_path / "fixture.py"
    path.write_text("def kernel():\n    return 1\n")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    engine = engine_fixture(reloader=KernelReloader(tmp_path))
    path.write_text("def kernel():\n    return 2\n")
    caller, result = start_reload(engine, modules=[name])
    path.write_text("def kernel():\n    return 3\n")  # Editor save during the drain.
    finish_reload(engine, caller, result)
    assert isinstance(result.get("error"), KernelReloadError)
    assert "Source changed after preparation" in str(result["error"])
    assert module.kernel() == 1
    assert engine.apc.calls == [] and engine.apc.entries == 3
    assert engine.service_state()["state"] == "serving"
    assert engine.kernel_reload_status()["phase"] == "idle"
    assert engine._kernel_reload_epoch == 0 and engine.model_revision == 4
    assert engine.operation_receipts[-1]["status"] == "rejected"
    assert engine.counts["kernel_reload_rejected"] == engine.counts["kernel_reload_rejected_prepare"] == 1
    assert engine.counts["kernel_reload_failed"] == 0
    engine._ensure_admission("generation")  # Admission reopened.


def test_source_saved_during_purge_is_refused_before_exec_and_keeps_serving(tmp_path, monkeypatch):
    import importlib.util
    import sys

    from mlx2.kernel_reload import KernelReloader, KernelReloadError

    name = "mlx2.runtime.kernel_reload_purge_edit_fixture"
    path = tmp_path / "fixture.py"
    path.write_text("def kernel():\n    return 1\n")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)

    class SavingCache(Cache):
        def clear(self, **kwargs):
            path.write_text("def kernel():\n    return 3\n")  # Saved after the first recheck.
            return super().clear(**kwargs)

    engine = engine_fixture(reloader=KernelReloader(tmp_path), apc=SavingCache())
    engine.qualification_state = "qualified"
    path.write_text("def kernel():\n    return 2\n")
    caller, result = start_reload(engine, modules=[name])
    finish_reload(engine, caller, result)
    assert isinstance(result.get("error"), KernelReloadError)
    assert "Source changed after preparation" in str(result["error"])
    assert module.kernel() == 1
    assert engine.apc.calls == ["clear"]  # The purge had already happened.
    assert engine.service_state()["state"] == "serving"
    assert engine.kernel_reload_status()["phase"] == "idle"
    assert engine.qualification_state == "qualified"
    assert engine._kernel_reload_epoch == 0 and engine.model_revision == 4
    receipt = engine.operation_receipts[-1]
    assert receipt["status"] == "rejected" and receipt["code_executed"] is False
    assert receipt["apcv2_purged"] is True and receipt["apcv2"] is not None
    assert engine.counts["kernel_reload_rejected_prepare"] == 1
    assert engine.counts["kernel_reload_failed"] == 0
    engine._ensure_admission("generation")


def test_model_park_guard_rechecks_reload_state_under_the_admission_lock():
    from mlx2.serving import SuspendUnavailable

    engine = engine_fixture()
    engine.apc_persist_dir = "/nonexistent-but-configured"
    engine.adapter.model_residency = object()
    admission = engine._admission_section

    @contextmanager
    def reload_completes_before_lock():
        engine._kernel_reload_epoch = 1  # A reload finished after the unlocked check.
        with admission():
            yield

    engine._admission_section = reload_completes_before_lock
    with pytest.raises(SuspendUnavailable, match="kernel-reloaded"):
        engine.quiesce(drain_timeout_seconds=1, suspend=True, park_model=True)
    assert getattr(engine, "_drain_park_model", False) is False
    assert engine.service_state()["state"] == "serving"


@pytest.mark.parametrize("pending", [False, True])
def test_model_park_is_refused_before_suspension_after_or_during_reload(pending):
    from mlx2.serving import SuspendUnavailable

    engine = engine_fixture()
    engine.apc_persist_dir = "/nonexistent-but-configured"
    engine.adapter.model_residency = object()
    if pending:
        engine.jobs["old"] = object()
        caller, result = start_reload(engine, modules=["fixture.kernel"])
    else:
        caller, result = start_reload(engine, modules=["fixture.kernel"])
        finish_reload(engine, caller, result)
        assert "error" not in result and engine.service_state()["state"] == "serving"
    try:
        with pytest.raises(SuspendUnavailable, match="kernel-reloaded"):
            engine.quiesce(drain_timeout_seconds=1, suspend=True, park_model=True)
        assert engine._drain_park_model is False
        assert engine.service_state()["state"] == ("draining" if pending else "serving")
    finally:
        engine.stop_event.set()
        caller.join(timeout=2)


def test_reload_identity_is_not_only_a_restartable_integer_epoch():
    engine = engine_fixture()
    caller, result = start_reload(engine, modules=["fixture.kernel"])
    finish_reload(engine, caller, result)
    assert "error" not in result, result.get("error")
    first = engine._kernel_cache_revision("base-runtime")
    caller, result = start_reload(engine, modules=["fixture.kernel"])
    finish_reload(engine, caller, result)
    assert "error" not in result, result.get("error")
    second = engine._kernel_cache_revision("base-runtime")
    assert second != first and "kernel-dev:" in second
    engine._kernel_reload_epoch = 0
    assert engine._kernel_cache_revision("base-runtime") == second


@pytest.mark.parametrize("failure", ["apply", "purge"])
def test_failed_mutation_or_purge_does_not_reopen_admission(failure):
    reloader = FakeReloader(apply_error=RuntimeError("apply failed") if failure == "apply" else None)
    cache = Cache(error=RuntimeError("purge failed") if failure == "purge" else None)
    engine = engine_fixture(reloader=reloader, apc=cache)
    caller, result = start_reload(engine, modules=["fixture.kernel"])
    finish_reload(engine, caller, result)
    assert isinstance(result.get("error"), RuntimeError)
    assert engine.service_state()["state"] == "kernel_reload_failed"
    with pytest.raises(AdmissionClosed):
        engine._ensure_admission("generation")
    with pytest.raises(RuntimeError):
        engine.resume()


def test_source_system_exit_is_reported_as_failure_not_successful_null():
    engine = engine_fixture(reloader=FakeReloader(apply_error=SystemExit("bad source import")))
    caller, result = start_reload(engine, modules=["fixture.kernel"])
    finish_reload(engine, caller, result)
    assert isinstance(result.get("error"), RuntimeError)
    assert "SystemExit" in str(result["error"])
    assert "receipt" not in result
    assert engine.service_state()["state"] == "kernel_reload_failed"


def test_representation_contract_cannot_restore_old_qualification_after_reload():
    engine = engine_fixture()
    engine._kernel_reload_identity = "new-implementation"
    engine._kernel_reload_epoch = 1
    engine.snapshot["route_receipt"] = {"route": "embeddings", "qualified": True}
    engine.adapter.representation_contract = lambda: {
        "implemented": True, "qualified": True, "serving_qualified": True,
        "selected": True, "observed_used": True,
    }
    state = engine._representation_route_state(engine.adapter)
    assert state["representation"]["qualified"] is False
    assert state["route_receipt"]["qualified"] is False
    assert state["route_receipt"]["representation"]["serving_qualified"] is False
    assert state["route_receipt"]["kernel_reload"]["execution_identity"] == "new-implementation"


def test_settings_only_switch_has_the_same_cache_invalidation_boundary():
    engine = engine_fixture()
    caller, result = start_reload(engine, settings=[{"module": "fixture.kernel", "name": "MODE", "value": "candidate"}])
    finish_reload(engine, caller, result)
    assert "error" not in result, result.get("error")
    assert engine.apc.calls == ["clear"] and engine.apc.entries == 0
    assert engine._kernel_reload_epoch == 1


def test_resume_cannot_cancel_reload_that_queues_between_admission_sections():
    engine = engine_fixture()
    original = engine._admission_section
    sections = 0

    @contextmanager
    def admit_with_reload_race():
        nonlocal sections
        with original():
            sections += 1
            yield
        if sections == 1:
            # reload_kernels acquired admission after resume's thaw check,
            # before resume reacquired it to change the service state.
            engine._kernel_reload_request = {"pending": True}
            engine._service_state = "draining"
            engine._quiesce_complete.clear()

    engine._admission_section = admit_with_reload_race
    with pytest.raises(RuntimeError, match="reload"):
        engine.resume()
    assert engine.service_state()["state"] == "draining"


def test_persistent_parked_prefix_cannot_return_after_reload_or_restart(tmp_path):
    from test_apcv2_sessions_persistence import _identity, _persistent, _state

    from mlx2.runtime.apc_v2 import APCSessionNotFound

    key = _identity()
    tokens = list(range(8))
    tag = ("tenant", "pre-reload")
    apc = _persistent(tmp_path)
    second = None
    try:
        apc.store(key, tokens, [_state(8)], session_tag=tag)
        apc.park_session(*tag, ttl_seconds=600)
        disk_bytes = apc._disk_bytes
        assert disk_bytes > 0
        assert list(tmp_path.glob("apc-idle-*"))
        engine = engine_fixture(apc=apc)
        caller, result = start_reload(engine, modules=["fixture.kernel"])
        finish_reload(engine, caller, result)
        assert "error" not in result, result.get("error")
        assert result["receipt"]["apcv2"]["apc"]["disk_bytes"] == disk_bytes
        assert engine.counts["kernel_reload_apc_disk_bytes_purged"] == disk_bytes
        with pytest.raises(APCSessionNotFound):
            apc.session_state(*tag)
        assert not list(tmp_path.glob("apc-idle-*"))
        apc._release_persist_lock()
        second = _persistent(tmp_path)
        assert second.lookup(key, tokens + [9]).cached_tokens == 0
        with pytest.raises(APCSessionNotFound):
            second.session_state(*tag)
    finally:
        if second is not None:
            second.close()
        apc.close()
