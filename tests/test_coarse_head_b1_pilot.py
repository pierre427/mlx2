"""CPU mechanics check for the source-bound B1 coarse-head pilot."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/research/coarse_head_b1_pilot.py"
ANALYZER = ROOT / "scripts/research/offline_coarse_head_preflight.py"


def test_tiny_cpu_produces_matched_real_arm_rows(tmp_path):
    out = tmp_path / "rows.npz"
    subprocess.run([sys.executable, str(RUNNER), "--tiny-cpu", "--out", str(out)],
                   check=True, capture_output=True, text=True)
    manifest = json.loads(out.with_suffix(".json").read_text())
    assert manifest["schema"] == "mlx2-coarse-head-pilot-v1"
    assert manifest["capture_kind"] == "synthetic_test"
    assert manifest["capture_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert manifest["capture_code_sha256"] == hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    assert set(manifest["mechanism_counts"].values()) == {3}

    spec = importlib.util.spec_from_file_location("offline_coarse_head_preflight", ANALYZER)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    with np.load(out, allow_pickle=False) as rows:
        assert rows["baseline_dense_fc_logits"].shape == (3, 128)
        assert rows["draft_exact_logits"].shape == (3, 128)
        assert rows["target_logits"].shape == (3, 128)
        assert np.max(np.abs(rows["baseline_dense_fc_logits"] - rows["draft_exact_logits"])) > 0
        for j in range(3):
            ids = rows["coarse_candidate_ids"][j]
            assert np.unique(ids).size == 64
            np.testing.assert_allclose(rows["candidate_exact_logits"][j],
                                       rows["draft_exact_logits"][j, ids],
                                       rtol=0, atol=0.125)
            a, b = rows["row_token_offsets"][j:j + 2]
            history = rows["row_token_ids"][a:b]
            assert history.size == 17 + j
            metadata = {key: int(rows[key][j]) for key in (
                "context_length", "lane_id", "seed", "draft_position",
                "target_offset", "draft_offset")}
            digest = analyzer.teacher_forced_row_sha256(history, **metadata).encode()
            for arm in ("dense_fc_row_sha256", "q4_fc_full_row_sha256",
                        "q4_fc_coarse_row_sha256", "target_row_sha256"):
                assert rows[arm][j] == digest


def test_gpu_path_is_explicit_and_output_is_immutable(tmp_path):
    out = tmp_path / "rows.npz"
    missing_mode = subprocess.run([sys.executable, str(RUNNER), "--out", str(out)],
                                  capture_output=True, text=True)
    assert missing_mode.returncode != 0
    assert "choose exactly one" in missing_mode.stderr
    out.touch()
    overwrite = subprocess.run([sys.executable, str(RUNNER), "--tiny-cpu", "--out", str(out)],
                               capture_output=True, text=True)
    assert overwrite.returncode != 0
    assert "output already exists" in overwrite.stderr
