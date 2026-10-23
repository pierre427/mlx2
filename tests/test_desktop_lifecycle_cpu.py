"""Lifecycle tests use fake engines/processes only; never stop a real model/UI."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2 import desktop_lifecycle as lifecycle


class Engine:
    def __init__(self, path, identity):
        self.child = SimpleNamespace(poll=lambda: None)
        self.worker = {"path": str(path), "url": "http://127.0.0.1:9999", "checkpoint": "synthetic@fingerprint", "state": "Ready"}
        self.settings = {"port": 9999}
        self.calls = []
        self.library = SimpleNamespace(checked_path=lambda raw, given: Path(raw) if given == identity else (_ for _ in ()).throw(ValueError("stale")))
    def stop(self):
        self.calls.append("stop")
        self.child = None
    def begin_start(self, path, identity):
        self.calls.append(("start", path, identity))
    def preparing(self):
        return False


def setup(tmp_path, monkeypatch):
    path = tmp_path / "synthetic"
    path.mkdir()
    monkeypatch.setattr(lifecycle, "source_identity", lambda: {"source_revision": "synthetic-source"})
    monkeypatch.setattr(lifecycle, "artifact_row", lambda p: {"identity": "exact-identity"})
    monkeypatch.setattr(lifecycle, "checkpoint", lambda p: "synthetic@fingerprint")
    engine = Engine(path, "exact-identity")
    events = []
    control = SimpleNamespace(home=tmp_path, primary=engine, engines={"default": engine}, profile=lambda name: None,
                              event=lambda *args, **kw: events.append((args, kw)))
    return lifecycle.Lifecycle(control), engine


def test_unload_returns_receipt_before_fake_engine_stops(tmp_path, monkeypatch):
    manager, engine = setup(tmp_path, monkeypatch)
    result = manager.dispatch({"command": "engine_unload_self", "_target": {"url": engine.worker["url"]}})
    assert result["terminal_action"] and result["state"] == "accepted"
    assert engine.calls == []
    for timer in list(manager.pending.values()):
        timer.join(2)
    assert engine.calls == ["stop"]
    receipt = json.loads((manager.home / (result["operation_id"] + ".json")).read_text())
    assert receipt["state"] == "completed"
    manager.close()


def test_restart_checks_identity_and_restarts_only_exact_owned_engine(tmp_path, monkeypatch):
    manager, engine = setup(tmp_path, monkeypatch)
    result = manager.dispatch({"command": "engine_restart_self", "_target": {"url": engine.worker["url"]}})
    for timer in list(manager.pending.values()):
        timer.join(2)
    assert engine.calls == ["stop", ("start", engine.worker["path"], "exact-identity")]
    assert result["terminal_action"]
    manager.close()


def test_foreign_and_changed_artifact_refused_before_any_stop(tmp_path, monkeypatch):
    manager, engine = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="not owned"):
        manager.dispatch({"command": "engine_unload_self", "_target": {"url": "http://127.0.0.1:1234"}})
    monkeypatch.setattr(lifecycle, "checkpoint", lambda path: "changed@fingerprint")
    with pytest.raises(ValueError, match="changed"):
        manager.dispatch({"command": "engine_restart_self", "_target": {"url": engine.worker["url"]}})
    assert engine.calls == []


def test_one_time_resume_plan_and_stale_checkpoint_refusal(tmp_path, monkeypatch):
    manager, engine = setup(tmp_path, monkeypatch)
    spec = manager._spec("default", engine)
    manager.resume_path.write_text(json.dumps({"id": "synthetic-resume", "created": time.time(), "engines": [spec]}))
    assert manager.startup_resume()
    assert not manager.resume_path.exists()
    assert manager.startup_resume() is False
    assert len(engine.calls) == 1
    engine.calls.clear()
    monkeypatch.setattr(lifecycle, "checkpoint", lambda path: "changed@fingerprint")
    manager.resume_path.write_text(json.dumps({"id": "synthetic-resume", "created": time.time(), "engines": [spec]}))
    assert manager.startup_resume()
    assert engine.calls == []


def test_ui_restart_prepare_requires_trusted_native_parent(tmp_path, monkeypatch):
    manager, engine = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="native UI parent"):
        manager.prepare_restart(str(tmp_path), -123)
    assert not manager.resume_path.exists()
    result = manager.dispatch({"command": "ui_close"})
    assert result["native_action"] == "ui_close" and result["terminal_action"]
    result = manager.dispatch({"command": "ui_restart"})
    assert result["requires_native_restart_prepare"] and result["terminal_action"]
    assert engine.calls == []


def test_restart_preparation_uses_fixed_helper_and_exact_resume_plan(tmp_path, monkeypatch):
    import psutil
    manager, engine = setup(tmp_path, monkeypatch)
    app = tmp_path / "MLX2.app"
    binary = app / "Contents/MacOS/MLX2"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr(lifecycle.os, "getppid", lambda: 98765)
    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(exe=lambda: str(binary), create_time=lambda: 123.5))
    calls = []
    monkeypatch.setattr(lifecycle.subprocess, "Popen", lambda command, **options: calls.append((command, options)))
    result = manager.prepare_restart(str(app), 98765)
    assert result["restart_prepared"] and result["terminal_action"]
    assert engine.calls == []
    plan = json.loads(manager.resume_path.read_text())
    assert plan["engines"][0]["checkpoint"] == "synthetic@fingerprint"
    assert plan["engines"][0]["identity"] == "exact-identity"
    command, options = calls[0]
    assert command[1:3] == ["-m", "mlx2.desktop_lifecycle"]
    assert command[command.index("--app-path") + 1] == str(app)
    assert options["start_new_session"] and "shell" not in options


def test_replaced_owned_child_is_never_stopped(tmp_path, monkeypatch):
    manager, engine = setup(tmp_path, monkeypatch)
    result = manager.dispatch({"command": "engine_unload_self", "_target": {"url": engine.worker["url"]}})
    engine.child = SimpleNamespace(poll=lambda: None)
    for timer in list(manager.pending.values()):
        timer.join(2)
    assert engine.calls == []
    receipt = json.loads((manager.home / (result["operation_id"] + ".json")).read_text())
    assert receipt["state"] == "failed"
