"""End-to-end synthetic numerical demo, with non-CPU frameworks blocked."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REVISION = "e448f070b84642bfc5d11dc04719e40336326a68"


def demo(tmp_path, *, strength="2"):
    command = """
import builtins, runpy
original_import = builtins.__import__
def cpu_only(name, *args, **kwargs):
    if name.split('.')[0] in {'mlx', 'torch', 'tensorflow', 'jax', 'cupy'}:
        raise AssertionError('Non-CPU framework import attempted: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = cpu_only
runpy.run_module('mlx2.experimental.weight_space.cpu_demo', run_name='__main__')
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["MLX2_PREFLIGHT_CPU_ONLY"] = "1"
    return subprocess.run(
        [sys.executable, "-c", command, "--output", str(tmp_path), "--source-revision", REVISION, "--strength", strength],
        capture_output=True, text=True, env=env, timeout=30, check=False,
    )


def test_real_numpy_repair_promotes_and_rolls_back(tmp_path):
    output = tmp_path / "run"
    process = demo(output)
    assert process.returncode == 0, process.stderr
    report = json.loads((output / "receipt.json").read_text())
    assert report["model_checkpoint_run"] is None and report["gpu"] is False
    assert report["transition_sequence"] == ["reported", "candidate", "trial", "validated", "promoting", "promoted", "rolled_back"]
    assert report["before_repair"]["arguments"] == {"filename": "demo.txt"}
    assert report["after_trial"]["arguments"] == {"path": "demo.txt"}
    assert report["after_rollback"]["arguments"] == {"filename": "demo.txt"}
    assert report["rollback_completed"]
    assert report["final_deployment"]["default_adapter"] is None
    assert report["final_deployment"]["generation"] == 2
    assert report["session_after"]["active_adapter"] is None
    assert report["validation"]["validation_sampling"] == {"temperature": 0.0, "seed": 0}
    assert report["validation"]["live_base"]["passed"] is False
    assert report["validation"]["live"]["passed"] is True
    assert report["validation"]["failure_source"]["kind"] == "operator_seed"
    candidate = [r for r in report["executions"] if r["phase"] == "controller_candidate_fresh_replay"]
    heldout = next(r for r in candidate if r["case_id"] == "heldout")
    regression = next(r for r in candidate if r["case_id"] == "regression")
    assert heldout["base_logits"] == pytest.approx([1, 0.4])
    assert heldout["logits"] == pytest.approx([0, 1.4], abs=1e-6)
    assert heldout["adapter_delta_applications"] == 1
    assert heldout["delta_l2_norm"] > 1
    assert regression["delta_l2_norm"] == 0
    assert regression["logits"] == regression["base_logits"]
    for file, digest in report["source_snapshot_sha256"].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest
    assert json.loads(process.stdout)["scope"] == "synthetic_numpy_network_only"


def test_wrong_direction_is_rejected_by_real_numerical_output(tmp_path):
    output = tmp_path / "negative"
    process = demo(output, strength="-2")
    assert process.returncode == 0, process.stderr
    report = json.loads((output / "receipt.json").read_text())
    assert report["trial_state"] == "rejected"
    assert report["transition_sequence"] == ["reported", "candidate", "trial", "rejected"]
    assert report["validation"]["repair_passed"] is False
    assert report["validation"]["regressions_passed"] is True
    assert report["after_trial"]["arguments"] == {"filename": "demo.txt"}
    assert report["final_deployment"]["generation"] == 0


def test_demo_refuses_existing_directory_without_overwriting_receipt(tmp_path):
    output = tmp_path / "run"
    assert demo(output).returncode == 0
    original = (output / "receipt.json").read_bytes()
    process = demo(output)
    assert process.returncode == 2
    assert "refuses to overwrite" in process.stderr
    assert (output / "receipt.json").read_bytes() == original
