"""Bounded parallel scheduling and lossless handoff to the frozen finalizer."""
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import threading

import pytest

from test_repack_deepseek_v41_iq1_cpu import affine_artifact, repacker  # noqa: F401


@pytest.fixture(scope="module")
def parallel(repacker):
    return importlib.import_module("repack_deepseek_v41_iq1_parallel")


def test_bounded_submission_never_exceeds_worker_count(parallel, monkeypatch):
    lock = threading.Lock()
    active = peak = 0
    barrier = threading.Barrier(4)
    def work(task):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=5)
        with lock:
            active -= 1
        return task
    monkeypatch.setattr(parallel, "write_one", work)
    seen = []
    assert parallel.drain_bounded(range(12), 4, completed=seen.append,
                                 executor_factory=lambda n: ThreadPoolExecutor(n)) == 12
    assert peak == 4 and sorted(seen) == list(range(12))


def test_failure_joins_running_worker_before_artifact_unlock(parallel, tmp_path, monkeypatch):
    entered, failing, release = threading.Event(), threading.Event(), threading.Event()
    errors = []
    def work(task):
        if task == 0:
            entered.set()
            assert release.wait(5)
            return task
        assert entered.wait(5)
        failing.set()
        raise RuntimeError("injected worker failure")
    monkeypatch.setattr(parallel, "write_one", work)
    def orchestrate():
        try:
            with parallel.base.conversion_lock(tmp_path):
                parallel.drain_bounded(range(2), 2, executor_factory=lambda n: ThreadPoolExecutor(n))
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=orchestrate)
    thread.start()
    assert failing.wait(5)
    with pytest.raises(RuntimeError, match="another converter"):
        with parallel.base.conversion_lock(tmp_path):
            pass
    release.set()
    thread.join(5)
    assert not thread.is_alive() and str(errors[0]) == "injected worker failure"
    with parallel.base.conversion_lock(tmp_path):
        pass


@pytest.mark.parametrize("workers", [0, 5, True])
def test_worker_bound_rejected(parallel, workers):
    with pytest.raises(ValueError, match="bounded CPU"):
        parallel.drain_bounded([], workers)


def test_real_spawn_workers_resume_serial_receipt_without_changing_recipe(parallel, repacker, affine_artifact):
    source, target, _ = affine_artifact
    partial = repacker.repack(source, target, reserve_bytes=0, max_files=1, phase=lambda _: None)
    assert partial["completed_experts"] == 1
    first = next(target.rglob("*.safetensors"))
    seal = repacker.stamp(first)
    old = json.loads((target / repacker.MANIFEST).read_text())
    result = parallel.parallel_repack(source, target, workers=2, reserve_bytes=0, finalize=False, phase=lambda _: None)
    assert result["parallel_passed"] and result["resumed_experts"] == result["newly_written"] == 1
    assert repacker.stamp(first) == seal
    new = json.loads((target / repacker.MANIFEST).read_text())
    assert new["recipe_sha256"] == old["recipe_sha256"]
    assert new["parallel_repack_sessions"][0]["finished"]
    assert not new["finished"]
    receipt_rows = [json.loads(p.read_text()) for p in (target / "conversion-state").glob("*.json")]
    executed = [r["record"]["execution"] for r in receipt_rows if "execution" in r["record"]]
    assert len(executed) == 1 and executed[0]["orchestrator_sha256"] == result["orchestrator_sha256"]
    for row in receipt_rows:
        assert repacker.verify_file(target, row["path"], row["record"]) == row["record"]["stat"]


def test_existing_receipt_corruption_refuses_parallel_handoff(parallel, repacker, affine_artifact):
    source, target, _ = affine_artifact
    repacker.repack(source, target, reserve_bytes=0, max_files=1, phase=lambda _: None)
    first = next(target.rglob("*.safetensors"))
    first.write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash/extent"):
        parallel.parallel_repack(source, target, workers=2, reserve_bytes=0, finalize=False, phase=lambda _: None)
