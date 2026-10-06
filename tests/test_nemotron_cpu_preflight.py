"""Host-only Nemotron artifact and qualification-boundary preflight."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_nemotron_cpu_preflight.py"
DIARIZATION = Path("/Volumes/T7/models/Nemotron-3-Diarization")
HISTORICAL_RECEIPT = Path(
    "/Volumes/T7/qualification/nemotron3-diarization/runs/"
    "v3-calibrated/final-offline-2/qualification.json"
)
LIGHTNING = Path.home() / (
    "mlx-models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-mlx-8Bit"
)
SUPER = Path.home() / "mlx-models/Nemotron-3-Super-120B-A12B-5bit-MTP"


def _module():
    spec = importlib.util.spec_from_file_location("nemotron_cpu_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_historical_diarization_receipt_explains_current_refusal():
    if not DIARIZATION.is_dir() or not HISTORICAL_RECEIPT.is_file():
        pytest.skip("resident diarization artifact or historical receipt absent")
    module = _module()
    artifact = module.inspect_diarization(DIARIZATION, verify_hash=True)
    record = json.loads(HISTORICAL_RECEIPT.read_text())
    differences = module.qualification_differences(record, artifact, profile="offline")
    assert "adapter_source_sha256" in differences
    if record["runtime"]["mlx"] == importlib.metadata.version("mlx"):
        assert "runtime.mlx" not in differences
    else:
        assert "runtime.mlx" in differences
    assert not {"artifact_sha256", "model_revision", "profiles.offline"} & set(
        differences
    )


def test_combined_real_artifact_preflight_imports_no_mlx_and_stays_pending(tmp_path):
    if not DIARIZATION.is_dir() or not LIGHTNING.is_dir() or not SUPER.is_dir():
        pytest.skip("resident Nemotron artifacts absent")
    output = tmp_path / "preflight.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--diarization-model",
        str(DIARIZATION),
        "--lightning-model",
        str(LIGHTNING),
        "--super-model",
        str(SUPER),
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "MLX_DEVICE": "cpu"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "passed"
    assert receipt["mlx_imported"] is False
    assert receipt["execution_numerics"] == {
        "accepted": True,
        "observed": {"MLX_ENABLE_TF32": "0"},
        "required": {"MLX_ENABLE_TF32": "0"},
    }
    assert receipt["artifacts"]["diarization"]["static_artifact_gate"] == "passed"
    lightning = receipt["artifacts"]["lightning"]
    assert lightning["static_artifact_gate"] == "passed"
    assert lightning["target_shards"] == 7
    assert lightning["mtp_tensor_count"] == 270
    assert lightning["default_route"] == "ordinary"
    assert lightning["qualification"] == "pending"
    super_artifact = receipt["artifacts"]["super"]
    assert super_artifact["static_artifact_gate"] == "passed"
    assert super_artifact["target_shards"] == 17
    assert super_artifact["mtp_tensor_count"] == 40
    assert super_artifact["has_embedded_mtp"] is True
    assert super_artifact["default_route"] == "ordinary"
    assert super_artifact["qualification"] == "pending"


def test_source_identity_binds_shared_nemotron_dependencies():
    identity = _module()._source_identity()
    source = identity["source_sha256"]
    assert {
        "scripts/check_nemotron_cpu_preflight.py",
        "src/mlx2/adapters/nemotron3_diarization.py",
        "src/mlx2/adapters/nemotron35_lightning.py",
        "src/mlx2/adapters/nemotron3_super.py",
        "src/mlx2/diarize_cli.py",
        "src/mlx2/diarization_qualification.py",
        "src/mlx2/process_env.py",
        "src/mlx2/runtime/nemotron_prefix_reuse.py",
        "src/mlx2/runtime/models/nemotron_h.py",
        "src/mlx2/runtime/models/ssm.py",
    } <= source.keys()
    source_tree = __import__("hashlib").sha256()
    for path in sorted((ROOT / "src/mlx2").rglob("*.py")):
        source_tree.update(str(path.relative_to(ROOT)).encode())
        source_tree.update(b"\0")
        source_tree.update(path.read_bytes())
        source_tree.update(b"\0")
    assert identity["source_tree_sha256"] == source_tree.hexdigest()


def test_preflight_refuses_tf32_enabling_process(tmp_path):
    if not LIGHTNING.is_dir():
        pytest.skip("resident Lightning artifact absent")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--lightning-model",
            str(LIGHTNING),
            "--output",
            str(tmp_path / "must-not-exist.json"),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "MLX_ENABLE_TF32": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    receipt = json.loads((tmp_path / "must-not-exist.json").read_text())
    assert receipt["status"] == "refused"
    numerics = receipt["execution_numerics"]
    assert numerics["accepted"] is False
    assert numerics["required"] == {"MLX_ENABLE_TF32": "0"}
    assert numerics["observed"] == {"MLX_ENABLE_TF32": "1"}
    assert "MLX_ENABLE_TF32='1'" in numerics["reason"]
    assert receipt["artifacts"]["lightning"]["static_artifact_gate"] == "passed"
    assert receipt["mlx_imported"] is False


def test_require_current_diarization_qualification_refuses_historical_receipt(tmp_path):
    if not DIARIZATION.is_dir() or not HISTORICAL_RECEIPT.is_file():
        pytest.skip("resident diarization artifact or historical receipt absent")
    output = tmp_path / "refused.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--diarization-model",
            str(DIARIZATION),
            "--diarization-receipt",
            str(HISTORICAL_RECEIPT),
            "--require-diarization-qualified",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "MLX_DEVICE": "cpu"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "refused"
    qualification = receipt["artifacts"]["diarization"]["qualification"]
    assert qualification["state"] == "refused_stale_or_incomplete"
    assert qualification["accepted"] is False
    assert "adapter_source_sha256" in qualification["differences"]
