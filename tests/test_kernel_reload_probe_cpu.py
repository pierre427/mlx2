"""Probe admission and resident wrapping only; no checkpoint or GPU is loaded."""

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/research/probe_kernel_reload.py"
SPEC = importlib.util.spec_from_file_location("kernel_reload_probe_test", PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_probe_refuses_before_ownership_or_gpu_without_explicit_worker_flag(tmp_path):
    with pytest.raises(ValueError, match="i-own-the-gpu"):
        probe.run(SimpleNamespace(i_own_the_gpu=False, out=tmp_path))


def test_probe_refuses_nonempty_receipt_directory_before_gpu(tmp_path):
    (tmp_path / "receipt.json").write_text("keep")
    with pytest.raises(FileExistsError, match="nonempty"):
        probe.run(SimpleNamespace(i_own_the_gpu=True, out=tmp_path))
    assert (tmp_path / "receipt.json").read_text() == "keep"


def test_probe_output_hook_keeps_model_and_weights_and_captured_alias():
    class Model:
        def __init__(self):
            self.weight = object()

        def logits(self, value, *, row_exact=False):
            return value + (1 if row_exact else 0)

    model = Model()
    original_weight = model.weight
    calls = []

    def alias(value):
        calls.append(value)
        return value

    target, original, receipt = probe.install_probe(SimpleNamespace(model=model), alias)
    assert target is model and model.weight is original_weight
    assert model.logits(3, row_exact=True) == 4 and calls == [4]
    assert receipt["seam"] == "model.logits"
    target.__class__ = original
    assert model.logits(3) == 3 and calls == [4]


def test_probe_sources_are_valid_fresh_python_with_distinct_versions():
    first, second = probe.source_text(1), probe.source_text(2)
    assert first != second
    compile(first, "version1.py", "exec")
    compile(second, "version2.py", "exec")
    assert "metal_kernel" in first and "lru_cache" in first


def test_probe_names_hf_snapshot_checkpoint_and_explicit_override(tmp_path):
    cached = tmp_path / "models--mlx-community--SmolLM-135M-Instruct-4bit" / "snapshots" / ("a" * 40)
    assert probe.artifact_name(cached) == "SmolLM-135M-Instruct-4bit"
    assert probe.artifact_name(tmp_path / "Qwen3-0.6B-4bit") == "Qwen3-0.6B-4bit"
    args = probe.build_parser().parse_args([
        "--model", str(cached), "--out", str(tmp_path / "out"),
        "--artifact-name", "Qwen3-0.6B-4bit",
    ])
    assert probe.artifact_name(args.model, args.artifact_name) == "Qwen3-0.6B-4bit"
    for invalid in ("", " ", "model@hash", "model\nother"):
        with pytest.raises(ValueError, match="artifact name"):
            probe.artifact_name(cached, invalid)


def music3_snapshot(monkeypatch, *, changes=None, rss=13216, executable=None, health_error=None):
    health = {"status": "ok", "model": "MiniMaxAI/MiniMax-Music3", "loaded": False,
              "busy": False, "lifecycle": "on_demand", **(changes or {})}

    def open_health(endpoint, timeout):
        assert endpoint == "http://127.0.0.1:8600/health" and timeout == 2
        if health_error:
            raise health_error
        response = io.BytesIO(json.dumps(health).encode())
        response.status = 200
        response.geturl = lambda: endpoint
        return response

    def process(command, *, text, timeout):
        assert command[0] == "/bin/ps" and command[-1] == "pid=,rss=,comm="
        assert text and timeout == 2
        path = executable or str(Path.home() / "Library/Application Support/mlxuag-music3/runtime/.venv/bin/python")
        return f" {command[2]} {rss} {path}\n"

    monkeypatch.setattr(probe, "urlopen", open_health)
    monkeypatch.setattr(probe.subprocess, "check_output", process)
    snapshot = {
        "foreign_live_services": [
            {"pid": 11, "label": "com.example.music3-api"},
            {"pid": 12, "label": "com.example.music3-mcp"},
            {"pid": 13, "label": "other.music3-api"},
        ],
        "foreign_model_processes": [{"pid": 14, "surface": "foreign-model"}],
    }
    return probe.host_snapshot(SimpleNamespace(host_snapshot=lambda: snapshot))


def test_probe_recognizes_only_evidenced_idle_music3_services(monkeypatch):
    snapshot = music3_snapshot(monkeypatch)
    assert snapshot["foreign_live_services"] == [{"pid": 13, "label": "other.music3-api"}]
    assert snapshot["foreign_model_processes"] == [{"pid": 14, "surface": "foreign-model"}]
    assert {row["pid"] for row in snapshot["observed_idle_music3_services"]} == {11, 12}
    assert all(row["rss_kib"] == 13216 for row in snapshot["observed_idle_music3_services"])
    assert snapshot["music3_idle_check"]["health"]["loaded"] is False
    assert snapshot["music3_idle_check"]["errors"] == []


@pytest.mark.parametrize("options", [
    {"changes": {"loaded": True}}, {"changes": {"busy": True}},
    {"changes": {"loaded": 0}}, {"changes": {"lifecycle": "resident"}},
    {"changes": {"model": "other"}}, {"changes": {"status": "error"}},
    {"rss": 256 * 1024}, {"rss": -1}, {"executable": "/foreign/python"},
    {"health_error": OSError("health unavailable")},
])
def test_probe_music3_unknown_or_active_remains_contamination(monkeypatch, options):
    snapshot = music3_snapshot(monkeypatch, **options)
    assert len(snapshot["foreign_live_services"]) == 3
    assert not snapshot.get("observed_idle_music3_services")
