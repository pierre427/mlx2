"""CPU-only desktop media, source-preserving conversion and refusal contracts."""
from __future__ import annotations

import json
import struct
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from mlx2.desktop_conversion import Conversions, inspect_source
from mlx2.desktop_conversion_uniform import convert, quantize
from mlx2.desktop_media import (
    MediaController,
    known_services,
    unwrap,
    validate_services,
)
from mlx2.desktop_records import RecordStore


def safetensors(path, arrays):
    header, chunks, offset = {}, [], 0
    for key, array in arrays.items():
        raw = array.tobytes()
        dtype = {np.dtype("float32"): "F32", np.dtype("float16"): "F16", np.dtype("uint16"): "BF16"}[array.dtype]
        header[key] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


def source_model(root):
    path = root / "bf16-model"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": "test", "hidden_size": 64}))
    (path / "tokenizer.json").write_text('{"version":"1.0"}')
    (path / "private-chat.log").write_text("private")
    safetensors(path / "model.safetensors", {
        "layers.0.mlp.up_proj.weight": np.linspace(-1, 1, 128, dtype=np.float32).reshape(2, 64),
        "layers.0.mlp.gate.weight": np.ones((2, 64), dtype=np.float16),
        "layers.0.input_layernorm.weight": np.ones(64, dtype=np.float32),
    })
    return path


def controller(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    return SimpleNamespace(home=state, preferences={"media_services": [], "conversion_reserve_bytes": 1024},
        store=RecordStore(state), primary=SimpleNamespace(library=SimpleNamespace(root=root, refresh=Mock())),
        event=Mock(), configure_preferences=Mock())


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_uniform_layout_and_reconstruction(bits):
    values = np.linspace(-2, 2, 256, dtype=np.float32).reshape(4, 64)
    packed, scales, biases = quantize(values, bits)
    assert packed.shape == (4, 64 * bits // 32)
    codes = ((packed[..., None] >> (np.arange(32 // bits) * bits)) & ((1 << bits) - 1)).reshape(values.shape)
    sf = (scales.astype(np.uint32) << 16).view(np.float32)
    bf = (biases.astype(np.uint32) << 16).view(np.float32)
    restored = codes.reshape(4, 1, 64) * sf[..., None] + bf[..., None]
    assert np.max(np.abs(restored.reshape(values.shape) - values)) <= .51 / ((1 << bits) - 1) + .025


def test_conversion_preserves_source_protected_weights_and_privacy(tmp_path):
    source = source_model(tmp_path)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / "output"
    progress = []
    convert(source, output, 4, lambda *args: progress.append(args))
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    assert not (output / "private-chat.log").exists()
    assert (output / "tokenizer.json").read_bytes() == before["tokenizer.json"]
    with (output / "model.safetensors").open("rb") as stream:
        header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
    assert header["layers.0.mlp.up_proj.weight"]["dtype"] == "U32"
    assert header["layers.0.mlp.up_proj.scales"]["dtype"] == "BF16"
    assert header["layers.0.mlp.gate.weight"]["dtype"] == "F16"
    assert "layers.0.mlp.gate.scales" not in header
    assert len(progress) > 5
    assert progress[-1][1] == 100
    with pytest.raises(FileExistsError):
        convert(source, output, 4, lambda *args: None)


def test_disk_refusal_precedes_queue(tmp_path, monkeypatch):
    host = controller(tmp_path)
    source = source_model(host.primary.library.root)
    conversion = Conversions(host)
    monkeypatch.setattr("mlx2.desktop_conversion.shutil.disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="Insufficient free space"):
        conversion.start({"path": str(source), "recipe": "q4"})
    assert host.store.jobs() == []
    assert not (host.home / "conversion-jobs").exists()
    host.store.close()


def test_reject_quantized_and_outside_library(tmp_path):
    host = controller(tmp_path)
    source = source_model(host.primary.library.root)
    (source / "config.json").write_text('{"quantization":{"bits":4}}')
    with pytest.raises(ValueError, match="original BF16"):
        inspect_source(source)
    elsewhere = source_model(tmp_path)
    with pytest.raises(ValueError, match="library"):
        Conversions(host).plan({"path": str(elsewhere), "recipe": "q4"})
    host.store.close()


def test_service_validation_no_arbitrary_process_commands():
    validate_services(known_services())
    profile = known_services()[0]
    with pytest.raises(ValueError, match="Unknown"):
        validate_services([{**profile, "command": "arbitrary shell"}])
    with pytest.raises(ValueError, match="LaunchAgents"):
        validate_services([{**profile, "launch_agents": ["../../bad"]}])
    with pytest.raises(ValueError, match="endpoint"):
        validate_services([{**profile, "endpoint": "http://evil.example/mcp"}])


def test_unwrap_refuses_unstructured_errors():
    assert unwrap({"content": [{"type": "text", "text": '{"job_id":"one"}'}]}) == {"job_id": "one"}
    with pytest.raises(ValueError):
        unwrap({"isError": True, "content": [{"type": "text", "text": "private error"}]})


def test_media_durable_submission_polls_same_id_and_does_not_stop_remote(tmp_path):
    host = controller(tmp_path)
    host.preferences["media_services"] = [known_services()[0]]
    media = MediaController(host)
    media.call = Mock(return_value={"job_id": "remote-one", "status": "queued"})
    result = media.generate({"provider_id": "mlx2-image", "parameters": {"prompt": "a tree"}})
    job = host.store.jobs()[0]
    assert job["result"]["remote_job_id"] == "remote-one"
    assert "prompt" not in json.dumps(job["spec"])
    with pytest.raises(ValueError, match="active"):
        media._require_idle(known_services()[0])
    media.close()
    assert media.call.call_count == 1
    assert result["remote_job_id"] == "remote-one"
    host.store.close()


def test_unknown_submission_is_retained_without_retry(tmp_path):
    host = controller(tmp_path)
    host.preferences["media_services"] = [known_services()[0]]
    media = MediaController(host)
    media.call = Mock(side_effect=TimeoutError)
    with pytest.raises(TimeoutError):
        media.generate({"provider_id": "mlx2-image", "parameters": {"prompt": "a tree"}})
    assert host.store.jobs()[0]["state"] == "submission_unknown"
    media.close()
    assert media.call.call_count == 1
    host.store.close()


def test_shared_yue_active_jobs_refuse_stop(tmp_path):
    host = controller(tmp_path)
    media = MediaController(host)
    state = tmp_path / "yue.json"
    state.write_text('{"jobs":{"job-one":{"status":"running"}}}')
    profile = {**next(p for p in known_services() if p["id"] == "yue2"), "state_file": str(state)}
    with pytest.raises(ValueError, match="another client"):
        media._require_idle(profile)
    media.close()
    host.store.close()


def test_uniform_worker_never_claims_gpu_and_writes_complete_candidate(tmp_path, monkeypatch):
    import mlx2.desktop_campaign_worker as worker
    from mlx2.desktop_conversion import run_conversion
    source = source_model(tmp_path)
    request = {**inspect_source(source), "recipe": "q2", "staging": str(tmp_path / "converted"),
               "progress_path": str(tmp_path / "progress.json"), "reserve_bytes": 0}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    ownership = Mock(side_effect=AssertionError("CPU conversion must not claim GPU"))
    monkeypatch.setattr(worker, "check_ownership", ownership)
    run_conversion(path)
    ownership.assert_not_called()
    assert json.loads((tmp_path / "progress.json").read_text())["percent"] == 100
    assert (tmp_path / "converted/model.safetensors").is_file()


def test_moe_three_dimensional_affine_layout(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"test"}')
    safetensors(source / "model.safetensors", {
        "layers.0.switch_mlp.up_proj.weight": np.linspace(-1, 1, 256, dtype=np.float32).reshape(2, 2, 64)})
    output = tmp_path / "output"
    convert(source, output, 4, lambda *args: None)
    with (output / "model.safetensors").open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
    assert header["layers.0.switch_mlp.up_proj.weight"]["shape"] == [2, 2, 8]
    assert header["layers.0.switch_mlp.up_proj.scales"]["shape"] == [2, 2, 1]


def test_shared_service_without_atomic_drain_refuses_stop_even_when_idle(tmp_path, monkeypatch):
    host = controller(tmp_path)
    host.preferences["media_services"] = [known_services()[0]]
    media = MediaController(host)
    monkeypatch.setattr(media, "_require_idle", lambda _: None)
    monkeypatch.setattr("mlx2.desktop_media.Path.home", lambda: tmp_path)
    import plistlib
    agents = tmp_path / "Library/LaunchAgents"
    agents.mkdir(parents=True)
    label = known_services()[0]["launch_agents"][0]
    (agents / (label + ".plist")).write_bytes(plistlib.dumps({"Label": label}))
    run = Mock(side_effect=AssertionError("No process operation before atomic drain"))
    monkeypatch.setattr("mlx2.desktop_media.subprocess.run", run)
    with pytest.raises(ValueError, match="atomic admission-drain"):
        media.lifecycle("mlx2-image", False)
    run.assert_not_called()
    media.close()
    host.store.close()


def test_empty_library_snapshot_and_conversion_refusal_create_no_work(tmp_path):
    host = controller(tmp_path)
    media = MediaController(host)
    try:
        snapshot = media.snapshot()
        assert snapshot["jobs"] == [] and snapshot["conversions"]["jobs"] == []
        assert snapshot["services"]
        for command in ("conversion_plan", "conversion_start"):
            with pytest.raises(ValueError, match="Choose a source model"):
                media.dispatch({"command": command, "path": "", "recipe": "q4"})
        assert host.store.jobs() == []
        assert list(host.primary.library.root.iterdir()) == []
    finally:
        media.close()
        host.store.close()


def test_conversion_rejects_non_object_config_with_actionable_error(tmp_path):
    source = source_model(tmp_path)
    (source / "config.json").write_text("[]")
    with pytest.raises(ValueError, match="configuration object"):
        inspect_source(source)


def test_closed_conversion_controller_rejects_new_work_before_queue(tmp_path):
    host = controller(tmp_path)
    conversion = Conversions(host)
    conversion.close()
    for action in (lambda: conversion.start({"path": "", "recipe": "q4"}), conversion.prepare):
        with pytest.raises(ValueError, match="closing"):
            action()
    assert host.store.jobs() == []
    host.store.close()


def test_converter_preparation_cancellation_retires_child_group(tmp_path, monkeypatch):
    host = controller(tmp_path)
    conversion = Conversions(host)
    child = Mock()
    child.poll.return_value = None
    monkeypatch.setattr("mlx2.desktop_conversion.subprocess.Popen", Mock(return_value=child))
    stop = Mock()
    monkeypatch.setattr("mlx2.desktop.stop_group", stop)
    monkeypatch.setattr(conversion.cancel, "wait", lambda seconds: True)
    with pytest.raises(InterruptedError, match="cancelled"):
        conversion._prepare_command(["git", "fixed"], "synthetic")
    stop.assert_called_once_with(child)
    assert conversion.process is None
    host.store.close()


@pytest.mark.parametrize("scenario", ["cancel", "destination_race", "refresh_failure"])
def test_conversion_publication_commit_boundary_is_safe(tmp_path, monkeypatch, scenario):
    from pathlib import Path

    from mlx2.desktop_downloads import Downloads
    host = controller(tmp_path)
    source = source_model(host.primary.library.root)
    conversion = Conversions(host)
    plan = conversion.plan({"path": str(source), "recipe": "q4"})
    plan.update(python="/fake/python", source_identity={"source_revision": "cpu-test"})
    identity = host.store.enqueue("conversion", plan)
    def spawn(args, **kwargs):
        request = json.loads(Path(args[-1]).read_text())
        staging = Path(request["staging"])
        staging.mkdir()
        (staging / "config.json").write_text("{}")
        (staging / "model.safetensors").write_bytes(b"synthetic-completed-output")
        if scenario == "cancel":
            conversion.cancel.set()
        return SimpleNamespace(poll=lambda: 0, returncode=0)
    monkeypatch.setattr("mlx2.desktop_conversion.subprocess.Popen", spawn)
    if scenario == "destination_race":
        publish = Downloads._publish
        def collide(staging, destination):
            destination.mkdir()
            publish(staging, destination)
        monkeypatch.setattr(Downloads, "_publish", staticmethod(collide))
    if scenario == "refresh_failure":
        host.primary.library.refresh.side_effect = RuntimeError("refresh failed")
    conversion._run(identity, plan)
    job = host.store.jobs()[0]
    destination = Path(plan["destination"])
    if scenario == "cancel":
        assert job["state"] == "cancelled"
        assert not destination.exists()
    elif scenario == "destination_race":
        assert job["state"] == "failed"
        assert list(destination.iterdir()) == []
    else:
        assert job["state"] == "completed"
        assert (destination / "mlx2-conversion.json").is_file()
    assert conversion.process is None
    host.store.close()


def test_media_shutdown_never_starts_another_refresh_or_submission(tmp_path):
    host = controller(tmp_path)
    media = MediaController(host)
    media.close()
    media.call = Mock()
    with pytest.raises(ValueError, match="nothing was submitted"):
        media.generate({"provider_id": "unused", "parameters": {"prompt": "unused"}})
    assert media.refresh() == {}
    media.call.assert_not_called()
    assert host.store.jobs() == []
    host.store.close()
