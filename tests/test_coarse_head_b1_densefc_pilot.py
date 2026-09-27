"""CPU checks for the dense-FC/coarse-only isolation arm."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


RUNNER = Path(__file__).resolve().parents[1] / "scripts/research/coarse_head_b1_densefc_pilot.py"
ORIGINAL = RUNNER.with_name("coarse_head_b1_pilot.py")
ANALYZER = RUNNER.with_name("analyze_coarse_head_densefc_pilot.py")


def test_tiny_cpu_dense_coarse_matches_full_head_on_same_checkpoint(tmp_path):
    out = tmp_path / "dense.npz"
    subprocess.run([sys.executable, str(RUNNER), "--tiny-cpu", "--out", str(out)],
                   check=True, capture_output=True, text=True)
    receipt = json.loads(out.with_suffix(".json").read_text())
    assert receipt["capture_kind"] == "synthetic_test"
    assert receipt["route"] == "research_dense_fc_coarse_head"
    assert receipt["capture_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert set(receipt["mechanism_counts"].values()) == {3}
    with np.load(out, allow_pickle=False) as data:
        assert data["dense_fc_full_logits"].shape == (3, 128)
        assert data["target_logits"].shape == (3, 128)
        for j in range(3):
            ids = data["dense_fc_coarse_candidate_ids"][j]
            assert np.unique(ids).size == 64
            np.testing.assert_allclose(data["dense_fc_candidate_exact_logits"][j],
                                       data["dense_fc_full_logits"][j, ids],
                                       rtol=0, atol=0.125)
            assert data["dense_fc_full_row_sha256"][j] == data["dense_fc_coarse_row_sha256"][j]
            assert data["target_row_sha256"][j] == data["dense_fc_full_row_sha256"][j]


def test_metal_path_requires_source_bound_baseline(tmp_path):
    result = subprocess.run([sys.executable, str(RUNNER), "--i-own-the-gpu",
                             "--out", str(tmp_path / "rows.npz")],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires baseline capture and manifest" in result.stderr


def test_cpu_analyzer_checks_replay_and_fails_on_rescore_tamper(tmp_path):
    original = tmp_path / "original.npz"
    dense = tmp_path / "dense.npz"
    for runner, out in ((ORIGINAL, original), (RUNNER, dense)):
        subprocess.run([sys.executable, str(runner), "--tiny-cpu", "--out", str(out)],
                       check=True, capture_output=True, text=True)
    spec = importlib.util.spec_from_file_location("analyze_coarse_head_densefc_pilot", ANALYZER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.analyze(dense, dense.with_suffix(".json"),
                            original, original.with_suffix(".json"))
    assert report["pilot_gate_pass"]
    assert not report["gpu_ab_admissible"]
    assert report["summary"]["max_dense_replay_abs"] == 0
    with np.load(dense, allow_pickle=False) as saved:
        changed = {name: saved[name] for name in saved.files}
    changed["dense_fc_candidate_exact_logits"][0, 0] += 1
    np.savez_compressed(dense, **changed)
    manifest = json.loads(dense.with_suffix(".json").read_text())
    manifest["capture_sha256"] = hashlib.sha256(dense.read_bytes()).hexdigest()
    dense.with_suffix(".json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="rescore disagrees"):
        module.analyze(dense, dense.with_suffix(".json"),
                       original, original.with_suffix(".json"))
