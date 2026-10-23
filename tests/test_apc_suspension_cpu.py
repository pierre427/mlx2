"""CPU-only cache metadata suspension; no model loads or inference."""
import json
from types import SimpleNamespace

import pytest
from test_apcv2_sessions_persistence import _identity, _persistent, _state
from test_serving_quiesce import _engine

from mlx2.runtime.apc_suspension import SuspensionRestoreError, consume_epoch


def test_export_freezes_pin_idle_and_retention_age_then_consumes_epoch_once(tmp_path):
    wall = [1000.0]
    apc = _persistent(tmp_path, wall=wall)
    tag = ("tenant-a", "paused-session")
    apc.store(_identity(), [1, 2, 3], [_state(3)], session_tag=tag)
    apc.park_session(*tag, ttl_seconds=60)
    before = json.loads(next(tmp_path.glob("*.manifest.json")).read_text())
    receipt = apc.freeze_suspension("epoch-one")
    assert receipt["clock_frozen"] and receipt["entries"] == 1
    branches = apc.branch_diagnostics()
    assert branches["clock_frozen"]
    assert branches["entries"][0]["remaining_ttl_seconds"] == 60
    wall[0] += 3600
    assert apc.session_state(*tag)["disk_pin_expires_at"] == 1060
    assert apc.branch_diagnostics()["entries"][0]["remaining_ttl_seconds"] == 60
    apc.close()
    resumed = _persistent(tmp_path, wall=wall)
    state = resumed.session_state(*tag)
    assert resumed.apc_stats["persistence"]["rescan"]["inventory_sha256"] == receipt["inventory_sha256"]
    assert resumed.apc_stats["persistence"]["rescan"]["suspension_epochs"] == {"epoch-one": 1}
    assert state["disk_pin_expires_at"] == 4660
    after = json.loads(next(tmp_path.glob("*.manifest.json")).read_text())
    assert after.get("suspension") is None
    assert after["created_at"] - before["created_at"] == 3600
    assert after["last_access_at"] - before["last_access_at"] == 3600
    assert resumed.branch_diagnostics()["entries"][0]["remaining_ttl_seconds"] == 60
    resumed.close()
    wall[0] += 10
    restarted = _persistent(tmp_path, wall=wall)
    assert restarted.session_state(*tag)["disk_pin_expires_at"] == 4660
    assert restarted.branch_diagnostics()["entries"][0]["remaining_ttl_seconds"] == 50
    restarted.close()


def test_cancel_export_thaws_in_process_and_fresh_restarts_do_not_repause(tmp_path):
    wall = [1000.0]
    apc = _persistent(tmp_path, wall=wall)
    tag = ("tenant-a", "paused-session")
    apc.store(_identity(), [1, 2, 3], [_state(3)], session_tag=tag)
    apc.park_session(*tag, ttl_seconds=30)
    apc.freeze_suspension("epoch-two")
    assert apc.freeze_suspension("epoch-two")["entries"] == 1
    with pytest.raises(ValueError, match="different"):
        apc.freeze_suspension("different")
    wall[0] += 100
    assert apc.thaw_suspension()["entries"] == 1
    assert apc.session_state(*tag)["disk_pin_expires_at"] == 1130
    assert not apc.branch_diagnostics()["clock_frozen"]
    assert apc.thaw_suspension()["entries"] == 0
    apc.close()


def test_partial_export_fails_closed_and_can_be_explicitly_thawed(tmp_path, monkeypatch):
    wall = [1000.0]
    apc = _persistent(tmp_path, wall=wall)
    apc.store(_identity(), [1, 2], [_state(2)], session_tag=("tenant-a", "s"))
    apc.park_session("tenant-a", "s", ttl_seconds=60)
    write = apc._write_manifest_locked
    monkeypatch.setattr(apc, "_write_manifest_locked", lambda *_args: False)
    with pytest.raises(RuntimeError, match="commit failed"):
        apc.freeze_suspension("partial")
    with pytest.raises(RuntimeError, match="Incomplete"):
        apc.freeze_suspension("partial")
    monkeypatch.setattr(apc, "_write_manifest_locked", write)
    wall[0] += 100
    apc.thaw_suspension()
    assert apc.session_state("tenant-a", "s")["disk_pin_expires_at"] == 1160
    apc.close()


def test_invalid_clock_or_epoch_never_silently_resets_expiry():
    value = {"suspension": {"id": "epoch", "frozen_at": 1000}, "created_at": 900,
             "last_access_at": 950, "disk_pins": [{"expires_at": 1060}]}
    with pytest.raises(SuspensionRestoreError):
        consume_epoch(value, 999)
    assert value["disk_pins"][0]["expires_at"] == 1060
    thawed, delta = consume_epoch(value, 2000)
    assert delta == 1000
    assert thawed["disk_pins"][0]["expires_at"] == 2060
    assert consume_epoch(thawed, 3000)[1] == 0


def test_engine_export_refuses_active_failed_or_incomplete_drain():
    engine = _engine()
    calls = []
    engine.apc = SimpleNamespace(freeze_suspension=lambda identity: calls.append(identity) or {"suspension_id": identity},
                                branch_diagnostics=lambda limit: {"entries": [], "clock_frozen": True})
    engine.snapshot = {"artifact": "exact-artifact", "runtime": {"source_sha256": "exact-source"}}
    for state, failures, residents, complete in (("serving", 0, 0, True), ("suspended", 1, 0, True),
                                               ("suspended", 0, 1, True), ("suspended", 0, 0, False)):
        engine._service_state = state
        engine._last_service_transition = {"result": {"suspend": {"failures": failures, "resident_entries_after": residents}}}
        engine._quiesce_complete.set() if complete else engine._quiesce_complete.clear()
        with pytest.raises(Exception, match="fully drained"):
            engine.export_suspension("epoch")
    assert calls == []
    engine._quiesce_complete.set()
    result = engine.export_suspension("epoch")
    assert calls == ["epoch"]
    assert result["artifact"] == "exact-artifact"
    assert result["runtime"] == {"source_sha256": "exact-source"}
    assert result["branches"]["clock_frozen"]


def test_branch_diagnostics_exposes_inferred_prefix_without_tokens_or_prompts(tmp_path):
    apc = _persistent(tmp_path)
    key = _identity()
    for tokens in ([10, 11], [10, 11, 12], [10, 11, 13]):
        apc.store(key, tokens, [_state(len(tokens))])
    report = apc.branch_diagnostics(3)
    # APCv2 supersedes the shorter terminal entry; both alternate leaves still
    # prove their shared prefix without inventing a retained parent.
    assert len(report["entries"]) == 2
    assert all(row["parent_id"] is None for row in report["entries"])
    assert {row["shared_prefix_tokens"] for row in report["entries"]} == {2}
    assert {row["relation"] for row in report["entries"]} == {"inferred_shared_prefix"}
    assert {row["shared_prefix_entry_id"] for row in report["entries"]} == {row["id"] for row in report["entries"]}
    assert all("tokens" not in row and "prompt" not in row for row in report["entries"])
    assert apc.branch_diagnostics(1)["truncated"]
    for limit in (0, 501, True, "200"):
        with pytest.raises(ValueError):
            apc.branch_diagnostics(limit)
    apc.close()


def test_indeterminate_export_cannot_later_return_a_successful_receipt(tmp_path, monkeypatch):
    from mlx2.runtime.apc_v2 import APCMetadataIndeterminate
    apc = _persistent(tmp_path)
    apc.store(_identity(), [1, 2], [_state(2)], session_tag=("tenant-a", "s"))
    apc.park_session("tenant-a", "s", ttl_seconds=60)
    write = apc._write_manifest_locked
    def fail(*_args):
        raise APCMetadataIndeterminate("synthetic indeterminate write")
    monkeypatch.setattr(apc, "_write_manifest_locked", fail)
    with pytest.raises(APCMetadataIndeterminate):
        apc.freeze_suspension("uncertain")
    monkeypatch.setattr(apc, "_write_manifest_locked", write)
    with pytest.raises(RuntimeError, match="Incomplete"):
        apc.freeze_suspension("uncertain")
    apc.thaw_suspension()
    apc.close()


def test_resume_epoch_write_failure_preserves_original_marker_and_fails_closed(tmp_path, monkeypatch):
    import mlx2.runtime.apc_v2 as module
    wall = [1000.0]
    apc = _persistent(tmp_path, wall=wall)
    apc.store(_identity(), [1, 2], [_state(2)], session_tag=("tenant-a", "s"))
    apc.park_session("tenant-a", "s", ttl_seconds=60)
    apc.freeze_suspension("atomic-epoch")
    apc.close()
    manifest = next(tmp_path.glob("*.manifest.json"))
    original = manifest.read_bytes()
    wall[0] += 100
    def fail(*_args):
        raise OSError("synthetic epoch commit failure")
    with monkeypatch.context() as patch:
        patch.setattr(module, "_write_manifest_payload", fail)
        with pytest.raises(SuspensionRestoreError, match="partial resume"):
            _persistent(tmp_path, wall=wall)
    assert manifest.read_bytes() == original
    resumed = _persistent(tmp_path, wall=wall)
    assert resumed.session_state("tenant-a", "s")["disk_pin_expires_at"] == 1160
    resumed.close()


def test_export_binds_registered_lora_files_and_rejects_global_mutation_mode(tmp_path):
    import threading

    from mlx2.runtime.apc_suspension import lora_file_receipt
    from mlx2.serving import AdmissionClosed, SuspendUnavailable
    engine = _engine()
    engine.snapshot = {"artifact": "exact-artifact", "runtime": {"source_sha256": "exact-source"}}
    engine._service_state = "suspended"
    engine._last_service_transition = {"result": {"suspend": {"failures": 0, "resident_entries_after": 0}}}
    engine._quiesce_complete.set()
    freezes = []
    engine.apc = SimpleNamespace(freeze_suspension=lambda identity: freezes.append(identity) or {"suspension_id": identity},
                                branch_diagnostics=lambda limit: {"entries": [], "clock_frozen": True})
    directory = tmp_path / "adapter"
    directory.mkdir()
    (directory / "adapter_config.json").write_text(json.dumps({"lora_parameters": {"rank": 2, "scale": 1.0, "keys": ["layer.query"]}}))
    (directory / "adapters.safetensors").write_bytes(b"synthetic source only")
    descriptor = lora_file_receipt(directory)
    engine.lora_root = directory
    engine.multi_lora = SimpleNamespace(lock=threading.RLock(), registry={"adapted": SimpleNamespace(path=str(directory), fingerprint=descriptor["fingerprint"])})
    result = engine.export_suspension("one")
    assert result["loras"] == [{"name": "adapted", **descriptor}]
    assert freezes == ["one"]
    for operation in (lambda: engine.load_lora_adapter("other", str(directory)), lambda: engine.unload_lora_adapter("adapted")):
        with pytest.raises(AdmissionClosed):
            operation()
    (directory / "adapters.safetensors").write_bytes(b"source changed after runtime registration")
    with pytest.raises(SuspendUnavailable, match="source changed"):
        engine.export_suspension("two")
    assert freezes == ["one"]
    engine.lora_session = SimpleNamespace(name="legacy-global")
    with pytest.raises(SuspendUnavailable, match="Global-mutation"):
        engine.export_suspension("three")


def test_suspended_payload_corruption_refuses_resume_before_epoch_consumption(tmp_path):
    wall = [1000.0]
    apc = _persistent(tmp_path, wall=wall)
    apc.store(_identity(), [1, 2], [_state(2)], session_tag=("tenant-a", "s"))
    apc.park_session("tenant-a", "s", ttl_seconds=60)
    apc.freeze_suspension("payload-integrity")
    apc.close()
    manifest = next(tmp_path.glob("*.manifest.json"))
    before = manifest.read_bytes()
    record = json.loads(before)["files"]["target"]
    payload = tmp_path / record["name"]
    raw = payload.read_bytes()
    payload.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    wall[0] += 100
    with pytest.raises(SuspensionRestoreError, match="payload digest mismatch"):
        _persistent(tmp_path, wall=wall)
    assert manifest.read_bytes() == before
    payload.write_bytes(raw)
    resumed = _persistent(tmp_path, wall=wall)
    assert resumed.session_state("tenant-a", "s")["disk_pin_expires_at"] == 1160
    resumed.close()


def test_partial_thaw_rolls_back_every_entry_before_retry(tmp_path, monkeypatch):
    wall = [1000.0]
    apc = _persistent(tmp_path, wall=wall)
    for tokens, name in (([1, 2], 'one'), ([9, 10], 'two')):
        apc.store(_identity(), tokens, [_state(2)], session_tag=('tenant-a', name))
        apc.park_session('tenant-a', name, ttl_seconds=60)
    receipt = apc.freeze_suspension('atomic-thaw')
    assert receipt['entries'] == 2
    writer = apc._write_manifest_locked
    calls = []
    def fail_second(*args):
        calls.append(args)
        return writer(*args) if len(calls) == 1 else False
    monkeypatch.setattr(apc, '_write_manifest_locked', fail_second)
    wall[0] = 1100
    with pytest.raises(RuntimeError):
        apc.thaw_suspension()
    assert apc.branch_diagnostics()['clock_frozen']
    for name in ('one', 'two'):
        assert apc.session_state('tenant-a', name)['disk_pin_expires_at'] == 1060
    monkeypatch.setattr(apc, '_write_manifest_locked', writer)
    wall[0] = 1200
    assert apc.thaw_suspension()['entries'] == 2
    for name in ('one', 'two'):
        assert apc.session_state('tenant-a', name)['disk_pin_expires_at'] == 1260
    apc.close()


def test_indeterminate_thaw_remains_gated_after_failed_commit(tmp_path, monkeypatch):
    from mlx2.runtime.apc_v2 import APCMetadataIndeterminate
    wall = [1000.0]
    apc = _persistent(tmp_path, wall=wall)
    apc.store(_identity(), [1, 2], [_state(2)], session_tag=('tenant-a', 'one'))
    apc.park_session('tenant-a', 'one', ttl_seconds=60)
    apc.freeze_suspension('poisoned-thaw')
    writer = apc._write_manifest_locked
    def poison(*args):
        apc._metadata_failure = 'synthetic indeterminate write'
        raise APCMetadataIndeterminate('synthetic')
    monkeypatch.setattr(apc, '_write_manifest_locked', poison)
    wall[0] = 1100
    with pytest.raises(APCMetadataIndeterminate):
        apc.thaw_suspension()
    monkeypatch.setattr(apc, '_write_manifest_locked', writer)
    with pytest.raises(APCMetadataIndeterminate):
        apc.thaw_suspension()
    assert apc._suspension_clock is not None
    apc._metadata_failure = None  # test cleanup only; production requires reload
    apc.thaw_suspension()
    apc.close()
