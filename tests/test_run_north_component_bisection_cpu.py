from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_north_component_bisection.py"


def load_driver():
    spec = importlib.util.spec_from_file_location(
        "north_native_bisection_driver", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


D = load_driver()


def quantization():
    value = {"bits": 4, "group_size": 64, "mode": "affine"}
    value.update(
        {
            f"model.layers.{index}.mlp.gate": {
                "bits": 8,
                "group_size": 64,
                "mode": "affine",
            }
            for index in range(1, 49)
        }
    )
    return value


def test_describe_is_host_only_and_names_all_contract_stages():
    command = [sys.executable, str(SCRIPT), "--describe"]
    code = (
        "import runpy,sys;"
        + f"sys.argv={command[1:]!r};"
        + "\ntry: runpy.run_path("
        + repr(str(SCRIPT))
        + ",run_name='__main__')\n"
        + "except SystemExit as e:\n"
        + " print('MLX_IMPORTED='+str('mlx' in sys.modules),file=sys.stderr);raise\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["required_stages"] == list(D.STAGES)
    assert report["production_route_changed"] is False
    assert report["qualification"] is False
    assert report["pinned_artifact_fingerprint"] == D.PINNED_ARTIFACT_FINGERPRINT
    assert "MLX_IMPORTED=False" in result.stderr


def test_native_refuses_before_artifact_or_mlx_without_explicit_ownership(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-native",
            "--model",
            str(tmp_path / "missing"),
            "--out",
            str(tmp_path / "receipt.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--i-own-the-gpu" in result.stderr
    assert not (tmp_path / "receipt.json").exists()


def test_paired_ownership_requires_matching_environment_and_receipts(tmp_path):
    paths = (tmp_path / "shared.json", tmp_path / "temporary.json")
    owner = {
        "lease_id": "lease-a",
        "session": "session-a",
        "label": "north-bisection",
        "pid": 123,
        "since": "2026-10-06T00:00:00Z",
        "cpg_used": True,
    }
    for path in paths:
        path.write_text(json.dumps(owner))
    proof = D.prove_gpu_ownership(
        paths, {"GPUQ_LEASE": "lease-a", "GPUQ_SESSION": "session-a"}
    )
    assert proof["lease_id"] == "lease-a"
    assert len(proof["receipt_sha256"]) == 2
    changed = dict(owner, lease_id="lease-b")
    paths[1].write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="byte-identical"):
        D.prove_gpu_ownership(
            paths, {"GPUQ_LEASE": "lease-a", "GPUQ_SESSION": "session-a"}
        )
    with pytest.raises(RuntimeError, match="GPUQ_LEASE"):
        D.prove_gpu_ownership(paths, {})


def test_geometry_is_exact_and_fails_closed_on_missing_or_extra_router():
    config = {"quantization": quantization(), "tie_word_embeddings": None}
    assert D.validate_quantization(config) == {
        "global": "affine-q4-group64",
        "router": "affine-q8-group64",
        "router_overrides": 48,
        "tied_head": True,
    }
    del config["quantization"]["model.layers.17.mlp.gate"]
    with pytest.raises(ValueError, match="override set"):
        D.validate_quantization(config)
    config = {"quantization": quantization()}
    config["quantization"]["model.layers.49.mlp.gate"] = {
        "bits": 8,
        "group_size": 64,
        "mode": "affine",
    }
    with pytest.raises(ValueError, match="override set"):
        D.validate_quantization(config)


def test_captured_inputs_are_b4_and_independent_between_stages():
    payloads = D._captured_inputs(hidden=32, context=7, seed=3)
    assert payloads["q4_projection"]["x"].shape == (4, 1, 32)
    assert payloads["sdpa"]["queries"].shape == (4, 32, 1, 128)
    assert payloads["sdpa"]["cache_keys"].shape == (4, 4, 7, 128)
    payloads["q4_projection"]["x"][0, 0, 0] = 99
    assert payloads["q8_router"]["x"][0, 0, 0] != 99


def test_atomic_json_replaces_complete_document_and_cleans_temp(tmp_path):
    destination = tmp_path / "receipt.json"
    D.atomic_json(destination, {"status": "loading", "value": 1})
    D.atomic_json(destination, {"status": "complete", "value": 2})
    assert json.loads(destination.read_text()) == {"status": "complete", "value": 2}
    assert not list(tmp_path.glob(".receipt.json.*.tmp"))


def test_native_loader_closes_adapter_in_finally_and_records_loaded_geometry():
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_native"
    )
    assert any(
        isinstance(node, ast.Try)
        and any(
            isinstance(item, ast.Expr)
            and isinstance(item.value, ast.Call)
            and isinstance(item.value.func, ast.Attribute)
            and item.value.func.attr == "close"
            for item in node.finalbody
        )
        for node in ast.walk(function)
    )
    assert '"observed_module_geometry"' in source
    assert '"tied_head": _observed_module_geometry(head)' in source
