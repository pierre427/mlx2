"""CPU-only contracts for the APC persistence qualification scripts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gpu_check_refuses_execution_without_explicit_ownership():
    script = load_script("gpu_check_apc_prefetch")
    with pytest.raises(SystemExit) as raised:
        script.main([])
    assert raised.value.code == 2


def test_gpu_check_dry_run_is_cpu_only_and_prints_exact_plan(tmp_path, capsys, monkeypatch):
    script = load_script("gpu_check_apc_prefetch")
    called = []
    monkeypatch.setattr(script, "run_engine_check", lambda _args: called.append(True))
    before = mx.default_device()
    result = script.main(
        [
            "--dry-run",
            "--dir",
            str(tmp_path / "not-created"),
            "--max-tokens",
            "7",
            "--concurrent-tokens",
            "11",
        ]
    )
    plan = json.loads(capsys.readouterr().out)
    assert result == 0
    assert called == []
    assert mx.default_device() == before == mx.cpu
    assert plan["will_execute"] is False
    assert plan["device"] == "metal"
    assert plan["model"]["kind"] == "built_in_tiny_qwen4_hybrid_fixture"
    assert plan["requests"]["concurrent_max_tokens"] == 11
    assert not (tmp_path / "not-created").exists()


def test_gpu_check_dry_run_describes_real_artifact_without_loading_it(tmp_path, capsys):
    script = load_script("gpu_check_apc_prefetch")
    artifact = tmp_path / "not-loaded-model"
    assert script.main(["--dry-run", "--model-path", str(artifact)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["model"] == {
        "kind": "local_artifact",
        "path": str(artifact.resolve()),
    }
    assert not artifact.exists()


def test_gpu_check_tiny_fixture_completes_deferred_park_on_cpu(
    tmp_path, capsys, monkeypatch
):
    script = load_script("gpu_check_apc_prefetch")
    queue_scaffolding_calls = []
    original_queue_empty_type = script._queue_empty_type

    def checked_queue_empty_type():
        queue_scaffolding_calls.append(True)
        return original_queue_empty_type()

    monkeypatch.setattr(script, "_queue_empty_type", checked_queue_empty_type)
    output = tmp_path / "result.json"
    assert script.main(
        [
            "--cpu",
            "--dir",
            str(tmp_path / "scratch"),
            "--output",
            str(output),
            "--max-tokens",
            "2",
            "--concurrent-tokens",
            "4",
            "--timeout-seconds",
            "30",
        ]
    ) == 0
    capsys.readouterr()
    report = json.loads(output.read_text())
    assert report["passed"] is True
    assert report["device"] == "cpu"
    assert report["apcv2"]["park_final_pending"] == 0
    assert report["apcv2"]["park_disk_entries"] > 0
    assert report["apcv2"]["prefetch_restores_ok"] > 0
    assert report["tokens"]["exact_match"] is True
    assert report["tokens"]["admin_suspend_resume_exact_match"] is True
    assert report["apcv2"]["admin_suspended_state"] == "suspended"
    assert report["apcv2"]["admin_suspended_session_state"] == "disk"
    assert report["apcv2"]["admin_resident_session_state"] == "resident"
    assert report["apcv2"]["admin_entries_after_resume"] == report["apcv2"][
        "admin_entries_before_suspend"
    ]
    assert all(value > 0 for value in report["receipts"]["admin_warm_cached_tokens"])
    assert report["timings"]["park_wait_seconds"] >= 0
    assert queue_scaffolding_calls


def test_gpu_check_fails_immediately_when_engine_worker_dies():
    script = load_script("gpu_check_apc_prefetch")
    engine = SimpleNamespace(
        error="RuntimeError: adapter load failed",
        ready=threading.Event(),
        thread=SimpleNamespace(is_alive=lambda: False),
    )
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="adapter load failed"):
        script._wait_for_engine_ready(engine, timeout=30)
    assert time.monotonic() - started < 0.5

    job = SimpleNamespace(id="job", events=queue.Queue())
    engine.ready.set()
    with pytest.raises(RuntimeError, match="adapter load failed"):
        script._collect(job, timeout=30, engine=engine)


def test_benchmark_guards_total_size_and_repository_directory(tmp_path):
    script = load_script("benchmark_apc_persistence")
    with pytest.raises(ValueError, match="outside the repository"):
        script.validate_scratch_root(ROOT / "qualification")
    args = argparse.Namespace(
        dir=tmp_path,
        entry_gib=4.0,
        entries=3,
        max_total_gib=8.0,
    )
    with pytest.raises(ValueError, match="exceeds --max-total-gib"):
        script.validate_args(args)


def test_benchmark_vm_stat_parser_and_small_rescan_cleanup(tmp_path, monkeypatch):
    script = load_script("benchmark_apc_persistence")
    parsed = script.parse_vm_stat(
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free: 2000000.\nPages speculative: 100.\n"
    )
    assert parsed["page_size"] == 16384
    assert parsed["Pages free"] == 2_000_000
    monkeypatch.setattr(script, "free_memory_bytes", lambda: 128 * script.GIB)
    monkeypatch.setattr(
        script,
        "machine_info",
        lambda root: {"test": True, "scratch_filesystem": {"path": str(root)}},
    )
    args = argparse.Namespace(
        dir=tmp_path,
        entry_gib=1 / 1024,
        entries=1,
        max_total_gib=8.0,
        output=None,
        rescan_only=True,
    )
    report = script.run(args)
    assert report["device"] == "cpu"
    assert report["startup_rescan"]["entries"] == 1
    assert report["startup_rescan"]["metadata_only_confirmed"] is True
    assert report["spill"]["digest_computed_once_per_payload"] is True
    assert list(tmp_path.iterdir()) == []


def test_provenance_degrades_outside_a_git_checkout(tmp_path, monkeypatch):
    """A qualified deployment is a git-archive export with no .git.

    The qualifier runs the snapshot's own unit tests from that directory, so a
    provenance probe that requires git fails the whole deployment rather than
    losing one optional field.
    """
    script = load_script("benchmark_apc_persistence")
    monkeypatch.setattr(script, "repository_root", lambda: tmp_path)
    assert script._git_head() is None
