"""Host-only unit tests for the payload-hashing benchmark."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_artifact_payload_hashing.py"
SPEC = importlib.util.spec_from_file_location("benchmark_artifact_payload_hashing", SCRIPT)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_weight_paths_are_deduplicated_and_safe(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"a")
    (tmp_path / "b.safetensors").write_bytes(b"bb")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"one": "b.safetensors", "two": "a.safetensors", "three": "b.safetensors"}})
    )

    assert [path.name for path in benchmark._weight_paths(tmp_path)] == [
        "a.safetensors",
        "b.safetensors",
    ]

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"escape": "../outside.safetensors"}})
    )
    with pytest.raises(ValueError, match="unsafe weight path"):
        benchmark._weight_paths(tmp_path)


def test_measure_rejects_identity_drift():
    identities = iter(("first", "second"))
    with pytest.raises(ValueError, match="identity changed"):
        benchmark._measure("fixture", 1, lambda: next(identities), 2)
