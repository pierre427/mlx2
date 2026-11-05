"""CPU contract tests: never load a model or mutate a LaunchAgent."""

import importlib.util
import json
import plistlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "scripts/research/kimi_moe"
sys.path.insert(0, str(HERE))
import service_restore_protocol as protocol


@pytest.fixture
def original(monkeypatch, tmp_path):
    plist = tmp_path / "original.plist"
    plist.write_bytes(plistlib.dumps({"ProgramArguments": ["python", "original_server.py"]}))
    monkeypatch.setattr(protocol.budget, "PLIST", plist)
    monkeypatch.setattr(protocol, "definition", lambda: "pid = 123\nprivate_environment = SECRET_TEST\n")
    monkeypatch.setattr(protocol.psutil, "Process", lambda pid=None: SimpleNamespace(
        cmdline=lambda: ["python", "original_server.py"], create_time=lambda: 1234.5))
    status = {"state": "ready", "model": protocol.budget.MODEL.name, "inflight": 0, "queue_depth": 0}
    monkeypatch.setattr(protocol.budget, "http", lambda *a, **kw: dict(status))
    monkeypatch.setattr(protocol, "cpg_free", lambda: True)
    monkeypatch.setattr(protocol.paired, "LOCKS", [])
    monkeypatch.setattr(protocol.paired, "WAITERS", tmp_path / "waiters")
    monkeypatch.setattr(protocol.paired, "foreign_model_processes", lambda: ["123 python original_server.py"])
    return status


def test_capture_is_private_and_revision_bound(original, tmp_path):
    out = tmp_path / "private"
    snapshot = protocol.capture(out, checkpoint="exact@fingerprint", source_revision="current-source")
    assert snapshot["source_revision"] == "current-source"
    assert snapshot["checkpoint"] == "exact@fingerprint"
    assert out.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in out.iterdir())
    assert b"SECRET_TEST" in (out / "launchctl.txt").read_bytes()
    assert "SECRET_TEST" not in json.dumps(snapshot)
    with pytest.raises(FileExistsError):
        protocol.capture(out, checkpoint="exact@fingerprint", source_revision="current-source")


@pytest.mark.parametrize("field,value", [("inflight", 1), ("queue_depth", 1), ("state", "loading")])
def test_busy_original_is_not_suspended(original, field, value):
    original[field] = value
    with pytest.raises(RuntimeError, match="ready and idle"):
        protocol.admit()


def test_foreign_model_and_waiters_are_not_preempted(original, monkeypatch, tmp_path):
    monkeypatch.setattr(protocol.paired, "foreign_model_processes", lambda: ["999 python foreign_model.py"])
    with pytest.raises(RuntimeError, match="Foreign model"):
        protocol.admit()
    monkeypatch.setattr(protocol.paired, "foreign_model_processes", list)
    waiters = tmp_path / "waiters"
    waiters.mkdir()
    (waiters / "first").touch()
    with pytest.raises(RuntimeError, match="owner/waiter"):
        protocol.admit()


def test_pid_reuse_refuses_bootout(original, monkeypatch, tmp_path):
    snapshot = protocol.capture(tmp_path / "private", checkpoint="x@y", source_revision="rev")
    snapshot["created"] = 0
    monkeypatch.setattr(protocol.subprocess, "run", lambda *a, **k: pytest.fail("must not bootout"))
    with pytest.raises(RuntimeError, match="PID identity"):
        protocol.suspend(snapshot)


def test_definition_change_never_overwrites(original, monkeypatch, tmp_path):
    snapshot = protocol.capture(tmp_path / "private", checkpoint="x@y", source_revision="rev")
    protocol.budget.PLIST.write_bytes(b"changed definition")
    monkeypatch.setattr(protocol.subprocess, "run", lambda *a, **k: pytest.fail("must not bootstrap"))
    with pytest.raises(RuntimeError, match="definition changed"):
        protocol.restore(tmp_path / "restore", snapshot=snapshot)


def test_existing_ready_requires_no_gpu_claim(original, monkeypatch, tmp_path):
    snapshot = protocol.capture(tmp_path / "private", checkpoint="x@y", source_revision="rev")
    monkeypatch.setattr(protocol, "DesktopCPGLease", lambda *a, **k: pytest.fail("must not claim GPU"))
    assert protocol.restore(tmp_path / "restore", snapshot=snapshot) == original
    assert json.loads((tmp_path / "restore/state.json").read_text())["gpu_held_while_waiting"] is False


def test_restore_waits_without_gpu_for_foreign_work(original, monkeypatch, tmp_path):
    snapshot = protocol.capture(tmp_path / "private", checkpoint="x@y", source_revision="rev")
    calls = iter([None, original])
    monkeypatch.setattr(protocol, "original_ready", lambda *a, **k: next(calls))
    monkeypatch.setattr(protocol.paired, "foreign_model_processes", lambda: ["999 foreign_model"])
    monkeypatch.setattr(protocol.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(protocol, "DesktopCPGLease", lambda *a, **k: pytest.fail("must not claim GPU"))
    assert protocol.restore(tmp_path / "restore", snapshot=snapshot) == original


def test_unprotected_startup_fails_before_bootstrap(original, monkeypatch, tmp_path):
    out = tmp_path / "startup"
    out.mkdir()
    (out / "spec.json").write_text(json.dumps({"service_snapshot": {"plist_sha256": protocol.sha(protocol.budget.PLIST)}}))
    monkeypatch.setattr(protocol, "check_ownership", lambda spec: (_ for _ in ()).throw(RuntimeError("no ownership")))
    monkeypatch.setattr(protocol.subprocess, "run", lambda *a, **k: pytest.fail("must not bootstrap"))
    with pytest.raises(RuntimeError, match="no ownership"):
        protocol.restore_child(out)


def test_retired_parent_has_no_side_effects():
    for path in [ROOT / "scripts/research/kimi_expert_budget_probe.py", ROOT / "scripts/research/kimi_domain_adaptive_sweep.py",
                 HERE / "loop_probe_v2.py", HERE / "longqa.py"]:
        spec = importlib.util.spec_from_file_location("retired_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with pytest.raises(RuntimeError, match="Retired unsafe"):
            module.run(None)


def test_signal_during_suspension_already_has_cleanup_guard():
    # Verify every published parent arms its mandatory finally before bootout.
    for path in HERE.glob("resume_*py"):
        source = path.read_text()
        assert source.index("suspended = True") < source.index("service_restore_protocol.suspend(snapshot)")
        assert "snapshot=snapshot" in source
        assert 'deadline = None' in source
        assert '== "running"' in source
