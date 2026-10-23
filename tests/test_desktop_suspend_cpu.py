"""Fake admin receipts and processes only: no server or model is started."""
from __future__ import annotations

import copy
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mlx2.desktop import artifact_row
from mlx2.desktop_runner import digest, source_manifest
from mlx2.desktop_suspend import Suspension


@pytest.fixture
def lifecycle(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"synthetic"}')
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    (source / "src/fixture.py").write_text("original")
    runner = {"python": sys.executable, "source": str(source), "binding": {"source_sha256": digest(source_manifest(source))}}
    worker = {"path": str(model), "state": "Ready", "checkpoint": "model@cpu-fake", "execution_kind": "autoregressive",
        "url": "http://127.0.0.1:12345", "api_key_file": str(tmp_path / "key"), "apc_persist_dir": str(tmp_path / "cache"),
        "effective_context": 8192, "resume_lineage_id": "stable-lineage", "runner": runner}
    owner = SimpleNamespace(home=tmp_path, lifecycle_lock=threading.RLock(), reap=Mock(), preparing=lambda: False,
        child=SimpleNamespace(pid=12345), worker=worker, worker_error=None, settings={"context": 8192}, context_resolver=None,
        preparation_cancel=threading.Event())
    def checked(path, identity):
        if artifact_row(Path(path))["identity"] != identity:
            raise ValueError("artifact changed")
        return Path(path)
    owner.library = SimpleNamespace(checked_path=checked)
    helper = Suspension(owner)
    def stop():
        assert json.loads((tmp_path / "desktop-suspension.json").read_text())["state"] == "Saved"
        owner.child = None
    owner.stop_worker = Mock(side_effect=stop)
    def request(method, route, body=None):
        if route == "/v1/status":
            return {"state": "ready", "artifact": "cpu-fake", "runtime": {"mlx": "cpu"}, "apcv2": {"persistence": {"rescan": {"registered": 0, "inventory_sha256": "f" * 64}}}}
        if route.endswith("/state"):
            return {"state": "suspended", "process_id": owner.child.pid, "last_transition": {"result": {"suspend": {"failures": 0, "resident_entries_after": 0}}}}
        if route.endswith("suspend-export"):
            return {"suspension_id": helper.record["id"], "frozen_at": 1.0, "persistence_dir": worker["apc_persist_dir"],
                "artifact": "cpu-fake", "runtime": {"mlx": "cpu"}, "process_id": owner.child.pid, "entries": 0, "inventory_sha256": "f" * 64, "branches": {"branches": []}}
        return {"state": "draining"}
    helper.request = Mock(side_effect=request)
    return helper, owner


def join(helper):
    helper.thread.join(2)
    assert not helper.busy()


def test_suspend_requires_confirmed_disk_save_before_retiring_and_recovers_cpu_only(lifecycle):
    helper, owner = lifecycle
    helper.begin_suspend()
    join(helper)
    assert owner.child is None
    owner.stop_worker.assert_called_once()
    assert owner.worker["state"] == "Suspended"
    assert owner.worker["resume_previous_pid"] == 12345
    assert owner.worker["resume_previous_checkpoint"] == "model@cpu-fake"
    assert owner.worker["resume_lineage_id"] == "stable-lineage"
    restored = SimpleNamespace(home=owner.home, worker=None)
    Suspension(restored)
    assert restored.worker["state"] == "Suspended"
    assert restored.worker["apc_branches"] == {"branches": []}


@pytest.mark.parametrize("failure", ["disk_export", "resident_entries", "cancel"])
def test_failed_or_cancelled_suspend_never_unloads_process(lifecycle, failure):
    helper, owner = lifecycle
    original = helper.request.side_effect
    def request(method, route, body=None):
        if failure == "disk_export" and route.endswith("suspend-export"):
            return {"frozen": False}
        if failure == "resident_entries" and route.endswith("/state"):
            return {"state": "suspended", "process_id": owner.child.pid, "last_transition": {"result": {"suspend": {"failures": 1, "resident_entries_after": 1}}}}
        if failure == "cancel" and route.endswith("/state"):
            helper.cancel.set()
            return {"state": "draining"}
        return original(method, route, body)
    helper.request.side_effect = request
    helper.begin_suspend()
    join(helper)
    owner.stop_worker.assert_not_called()
    assert owner.child is not None
    assert owner.worker["state"] == "SuspendError"
    assert helper.record["state"] == ("Cancelled" if failure == "cancel" else "Error")
    assert any(call.args[1] == "/v1/admin/resume" for call in helper.request.call_args_list)


@pytest.mark.parametrize("change", ["settings", "artifact", "runner", "context"])
def test_resume_refuses_identity_or_configuration_drift_before_launch(lifecycle, monkeypatch, change):
    helper, owner = lifecycle
    helper.begin_suspend()
    join(helper)
    owner.launch = Mock()
    monkeypatch.setattr("mlx2.adapters.registry.inspect_model", lambda path: SimpleNamespace(artifact_fingerprint="cpu-fake"))
    if change == "settings":
        owner.settings["context"] = 1234
    elif change == "artifact":
        (Path(owner.worker["path"]) / "changed").write_text("new")
    elif change == "runner":
        (Path(owner.worker["runner"]["source"]) / "src/fixture.py").write_text("changed")
    else:
        owner.context_resolver = lambda path, context: 16384
    with pytest.raises(ValueError, match="changed"):
        helper.begin_resume()
    owner.launch.assert_not_called()


def test_resume_reuses_exact_saved_runner_cache_and_chat_lineage(lifecycle, monkeypatch):
    helper, owner = lifecycle
    helper.begin_suspend()
    join(helper)
    saved = copy.deepcopy(owner.worker)
    monkeypatch.setattr("mlx2.adapters.registry.inspect_model", lambda path: SimpleNamespace(artifact_fingerprint="cpu-fake"))
    def launch(path, resolution, execution, runner, *, resume_worker):
        assert runner == saved["runner"]
        assert resume_worker["apc_persist_dir"] == saved["apc_persist_dir"]
        assert resume_worker["resume_previous_pid"] == 12345
        owner.child = SimpleNamespace(pid=23456)
        owner.worker.update(state="Loading")
    owner.launch = Mock(side_effect=launch)
    helper.begin_resume(helper.record["id"])
    join(helper)
    owner.launch.assert_called_once()
    assert helper.record["state"] == "Resumed"
    assert owner.worker["resume_lineage_id"] == "stable-lineage"


def test_partial_restore_never_restores_trusted_chat_lineage(lifecycle, monkeypatch):
    helper, owner = lifecycle
    helper.begin_suspend()
    join(helper)
    helper.record["apc_receipt"]["entries"] = 2
    monkeypatch.setattr("mlx2.adapters.registry.inspect_model", lambda path: SimpleNamespace(artifact_fingerprint="cpu-fake"))
    def launch(path, resolution, execution, runner, *, resume_worker):
        owner.child = SimpleNamespace(pid=23456)
        owner.worker.update(state="Loading")
    owner.launch = Mock(side_effect=launch)
    helper.begin_resume()
    join(helper)
    assert owner.worker["state"] == "ResumeError"
    assert owner.worker["cache_restore"]["confirmed"] is False
    assert owner.worker["resume_lineage_id"].startswith("pending:")
    assert "resume_previous_pid" not in owner.worker
    assert helper.record["state"] == "ResumeError"


@pytest.mark.parametrize("live", [False, True])
def test_crash_after_saved_export_never_adopts_a_live_previous_process(lifecycle, monkeypatch, live):
    import psutil
    helper, owner = lifecycle
    helper.begin_suspend()
    join(helper)
    helper.record["state"] = "Saved"
    helper.save()
    def process(pid):
        if live:
            return SimpleNamespace(status=lambda: psutil.STATUS_RUNNING)
        raise psutil.NoSuchProcess(pid)
    monkeypatch.setattr(psutil, "Process", process)
    recovered = SimpleNamespace(home=owner.home, worker=None)
    restored = Suspension(recovered)
    assert recovered.worker["state"] == ("SuspensionRecoveryRequired" if live else "Suspended")
    assert restored.record["state"] == ("Saved" if live else "Suspended")


@pytest.mark.parametrize("change", ["extra_entries", "different_inventory", "foreign_pid", "consumed_epoch"])
def test_resume_rejects_extra_entries_wrong_inventory_or_foreign_metadata_pid(lifecycle, monkeypatch, change):
    helper, owner = lifecycle
    helper.begin_suspend()
    join(helper)
    monkeypatch.setattr("mlx2.adapters.registry.inspect_model", lambda path: SimpleNamespace(artifact_fingerprint="cpu-fake"))
    original = helper.request.side_effect
    def request(method, route, body=None):
        result = original(method, route, body)
        if route == "/v1/status":
            if change == "extra_entries":
                result["apcv2"]["persistence"]["rescan"]["registered"] = 1
            elif change == "consumed_epoch":
                result["apcv2"]["persistence"]["rescan"]["suspension_epochs"] = {helper.record["id"]: 1}
            elif change == "different_inventory":
                result["apcv2"]["persistence"]["rescan"]["inventory_sha256"] = "a" * 64
        if change == "foreign_pid" and route.endswith("/state"):
            result["process_id"] = 99999
        return result
    helper.request.side_effect = request
    def launch(*args, **kwargs):
        owner.child = SimpleNamespace(pid=23456)
        owner.worker.update(state="Loading")
    owner.launch = Mock(side_effect=launch)
    helper.begin_resume()
    join(helper)
    assert owner.worker["state"] == "ResumeError"
    assert owner.worker["resume_lineage_id"].startswith("pending:")
    assert "resume_previous_pid" not in owner.worker


def test_suspend_refuses_replaced_worker_without_unloading_or_resuming_foreign_child(lifecycle):
    helper, owner = lifecycle
    original = helper.request.side_effect
    def request(method, route, body=None):
        result = original(method, route, body)
        if route.endswith("suspend-export"):
            owner.child = SimpleNamespace(pid=99999)
            owner.worker = {**owner.worker, "state": "Ready", "checkpoint": "replacement@other"}
        return result
    helper.request.side_effect = request
    helper.begin_suspend()
    join(helper)
    owner.stop_worker.assert_not_called()
    assert owner.worker["state"] == "Ready"
    assert owner.worker["checkpoint"] == "replacement@other"
    assert not any(call.args[1] == "/v1/admin/resume" for call in helper.request.call_args_list)


def _registered_lora(lifecycle):
    from mlx2.runtime.apc_suspension import lora_file_receipt
    helper, owner = lifecycle
    directory = owner.home / "registered-adapter"
    directory.mkdir()
    (directory / "adapter_config.json").write_text(json.dumps({"lora_parameters": {"rank": 2, "scale": 1.0, "keys": ["layer.query"]}}))
    (directory / "adapters.safetensors").write_bytes(b"synthetic descriptor hashing only")
    descriptor = {"name": "adapted", **lora_file_receipt(directory)}
    owner.settings.update(lora_root=str(directory), max_loras=1, max_lora_rank=2)
    original = helper.request.side_effect
    loaded = []
    def request(method, route, body=None):
        if route.endswith("load_lora_adapter"):
            loaded.append(body)
            return {"id": body["lora_name"], "fingerprint": descriptor["fingerprint"]}
        result = original(method, route, body)
        if route.endswith("suspend-export"):
            result["loras"] = [descriptor]
        if route == "/v1/status":
            result["multi_lora"] = {"enabled": True, "registered": ["adapted"] if owner.child.pid == 12345 or loaded else []}
        return result
    helper.request.side_effect = request
    return descriptor, loaded


def test_resume_reloads_exact_registered_lora_before_trusting_old_chat_lineage(lifecycle, monkeypatch):
    helper, owner = lifecycle
    descriptor, loaded = _registered_lora(lifecycle)
    helper.begin_suspend()
    join(helper)
    monkeypatch.setattr("mlx2.adapters.registry.inspect_model", lambda path: SimpleNamespace(artifact_fingerprint="cpu-fake"))
    def launch(*args, **kwargs):
        owner.child = SimpleNamespace(pid=23456)
        owner.worker.update(state="Loading")
    owner.launch = Mock(side_effect=launch)
    helper.begin_resume()
    join(helper)
    assert loaded == [{"lora_name": "adapted", "lora_path": descriptor["path"]}]
    assert owner.worker["state"] == "Ready"
    assert owner.worker["resume_lineage_id"] == "stable-lineage"
    assert helper.record["state"] == "Resumed"


def test_changed_lora_source_refuses_resume_before_any_model_launch(lifecycle):
    helper, owner = lifecycle
    descriptor, _loaded = _registered_lora(lifecycle)
    helper.begin_suspend()
    join(helper)
    (Path(descriptor["path"]) / "adapters.safetensors").write_bytes(b"different quantized adapter source")
    owner.launch = Mock()
    with pytest.raises(ValueError, match="changed since suspension"):
        helper.begin_resume()
    owner.launch.assert_not_called()


def test_wrong_reloaded_lora_fingerprint_never_restores_trusted_lineage(lifecycle, monkeypatch):
    helper, owner = lifecycle
    _registered_lora(lifecycle)
    helper.begin_suspend()
    join(helper)
    original = helper.request.side_effect
    def request(method, route, body=None):
        result = original(method, route, body)
        if route.endswith("load_lora_adapter"):
            result["fingerprint"] = "different-loaded-weights"
        return result
    helper.request.side_effect = request
    monkeypatch.setattr("mlx2.adapters.registry.inspect_model", lambda path: SimpleNamespace(artifact_fingerprint="cpu-fake"))
    def launch(*args, **kwargs):
        owner.child = SimpleNamespace(pid=23456)
        owner.worker.update(state="Loading")
    owner.launch = Mock(side_effect=launch)
    helper.begin_resume()
    join(helper)
    assert owner.worker["state"] == "ResumeError"
    assert owner.worker["resume_lineage_id"].startswith("pending:")
