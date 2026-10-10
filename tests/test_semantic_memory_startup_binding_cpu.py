"""--semantic-memory must start for every adapter artifact shape (CPU, no weights).

``AdapterResolution.artifact`` comes in two shapes: the standard decoder,
Qwen3.8-27B, North and similar resolvers nest ``{"identity": {...}}``, while
Flash-Next, Gemma 3n/4, MiniCPM-o, LFM2.5-VL, Muse Glimmer and the vision
candidates carry a top-level ``fingerprint``.  server.main bound semantic
memory with ``artifact["identity"]["fingerprint"]``, so the flat shape raised
KeyError after the model had fully loaded (sweep 2026-10-08).
"""

import json
import sys

import pytest

from mlx2 import server
from mlx2.adapters.registry import AdapterResolution, inspect_model


def _flash_next_artifact(root):
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    (root / "ple_rows.bin").write_bytes(b"\0" * 8)
    (root / "model-00001.safetensors").write_bytes(b"\0" * 8)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "model-00001.safetensors"}})
    )
    return root


def _gemma3n_artifact(root):
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "gemma3n"}))
    (root / "model.safetensors").write_bytes(b"\0" * 8)
    return root


class _FakeEngine:
    reasoning_signer = None
    # main() validates the engine arguments before binding; keep the contract.
    validate_arguments = staticmethod(server.ServingEngine.validate_arguments)

    def __init__(self, model, **kwargs):
        self.closed = False

    def close(self):
        self.closed = True


class _ReachedHandler(Exception):
    pass


@pytest.mark.parametrize("build", [_flash_next_artifact, _gemma3n_artifact])
def test_semantic_memory_binds_top_level_fingerprint_artifacts(
    build, tmp_path, monkeypatch
):
    model = build(tmp_path / "model")
    resolution = inspect_model(model)
    assert "identity" not in resolution.artifact  # the flat artifact shape
    expected = resolution.artifact["fingerprint"]
    captured = {}

    def handler_for(*args, semantic_middleware=None, **kwargs):
        captured["middleware"] = semantic_middleware
        raise _ReachedHandler

    monkeypatch.setattr(server, "ServingEngine", _FakeEngine)
    monkeypatch.setattr(server, "handler_for", handler_for)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx2.server", "--model", str(model), "--port", "0",
            "--semantic-memory", "--api-state-dir", str(tmp_path / "state"),
        ],
    )
    with pytest.raises(_ReachedHandler):
        server.main()  # before the fix: KeyError: 'identity' at the binding line
    bindings = captured["middleware"].memory.bindings
    assert bindings["model_binding"] == expected
    assert bindings["tokenizer_binding"] == expected


def test_artifact_fingerprint_reads_both_shapes_and_fails_closed():
    def resolution(artifact):
        return AdapterResolution(object, None, artifact)

    assert resolution({"identity": {"fingerprint": "nested"}}).artifact_fingerprint == "nested"
    assert resolution({"fingerprint": "flat", "path": "/m"}).artifact_fingerprint == "flat"
    for artifact in ({}, {"identity": {}}, {"fingerprint": ""}, {"identity": "x"}):
        with pytest.raises(ValueError, match="fingerprint"):
            resolution(artifact).artifact_fingerprint
