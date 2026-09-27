"""CPU boundary checks for the source-bound coarse-head candidate screen."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/research/offline_coarse_head_preflight.py"
SPEC = importlib.util.spec_from_file_location("offline_coarse_head_preflight", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_full_topk_inside_candidates_is_equivalent_without_top_p():
    logits = np.linspace(5.0, -7.0, 128, dtype=np.float32)
    ids = np.arange(64, dtype=np.int32)
    exact = MODULE.mlx2_q(logits, temp=0.8, top_p=0, top_k=20)
    coarse = MODULE.coarse_q(logits, ids, temp=0.8, top_p=0, top_k=20)
    np.testing.assert_allclose(coarse, exact, atol=1e-7)
    row = MODULE.evaluate_row(logits, ids, logits[::-1], temp=0.8, top_p=0, top_k=20)
    assert row["top20_coverage"] == 1.0
    assert row["full_q_support_misses"] == 0
    assert abs(row["expected_accept_delta"]) < 1e-7
    with_baseline = MODULE.evaluate_row(
        logits, ids, logits[::-1], temp=0.8, top_p=0, top_k=20,
        baseline_dense_fc_logits=logits)
    assert abs(with_baseline["expected_accept_combined_delta_vs_original"]) < 1e-7


def test_candidate_normalized_top_p_can_change_q_with_full_top20_coverage():
    logits = np.linspace(0.1, 0.0, 128, dtype=np.float32)
    ids = np.arange(64, dtype=np.int32)
    row = MODULE.evaluate_row(logits, ids, logits, temp=1.0, top_p=0.05, top_k=20)
    assert row["top20_coverage"] == 1.0
    assert row["q_total_variation"] > 0
    assert row["q_support_symmetric_difference"] > 0


def test_duplicate_candidate_ids_fail_closed():
    logits = np.arange(128, dtype=np.float32)
    ids = np.zeros(64, dtype=np.int32)
    with pytest.raises(ValueError, match="64 unique"):
        MODULE.coarse_q(logits, ids, temp=1.0, top_p=0.95, top_k=20)


def test_manifest_requires_matched_provenance(tmp_path):
    capture = tmp_path / "rows.npz"
    row_tokens = np.arange(17, dtype=np.uint32)
    row_digest = np.array([MODULE.teacher_forced_row_sha256(
        row_tokens, context_length=16, lane_id=0, seed=1, draft_position=0,
        target_offset=16, draft_offset=15)],
                          dtype="S64")
    np.savez(capture, baseline_dense_fc_logits=np.zeros((1, 128), dtype=np.float32),
             draft_exact_logits=np.zeros((1, 128), dtype=np.float32),
             coarse_candidate_ids=np.arange(64, dtype=np.int32)[None],
             candidate_exact_logits=np.zeros((1, 64), dtype=np.float32),
             target_logits=np.zeros((1, 128), dtype=np.float32),
             context_length=np.array([16], dtype=np.int32),
             lane_id=np.array([0], dtype=np.int32),
             seed=np.array([1], dtype=np.int32),
             draft_position=np.array([0], dtype=np.int32),
             target_offset=np.array([16], dtype=np.int32),
             draft_offset=np.array([15], dtype=np.int32),
             dense_fc_row_sha256=row_digest, q4_fc_full_row_sha256=row_digest,
             q4_fc_coarse_row_sha256=row_digest, target_row_sha256=row_digest,
             row_token_offsets=np.array([0, 17], dtype=np.int32), row_token_ids=row_tokens)
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    (artifact / "model.safetensors.index.json").write_text("{}")
    capture_code = tmp_path / "capture.py"
    capture_code.write_text("# synthetic fixture\n")
    source_root = PATH.parents[2]
    source_revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    manifest = {"schema": MODULE.SCHEMA, "mlx2_source_revision": source_revision,
                "omlx_source_revision": MODULE.OMLX_REV,
                "mlx2_source_root": str(source_root),
                "mlx2_contract_sha256": {relative: MODULE.sha256(source_root / relative)
                                         for relative in MODULE.MLX2_CONTRACT_PATHS},
                "capture_code_path": str(capture_code),
                "capture_code_sha256": MODULE.sha256(capture_code),
                "coarse_head_parameters_sha256": "b" * 64,
                "capture_sha256": MODULE.sha256(capture),
                "artifact_config_sha256": MODULE.sha256(artifact / "config.json"),
                "artifact_index_sha256": MODULE.sha256(artifact / "model.safetensors.index.json"),
                "route": "self_mtp_draft",
                "alignment": "teacher_forced_same_prefix_and_draft_tokens",
                "capture_kind": "synthetic_test",
                "mechanism_counts": {name: 1 for name in MODULE.REQUIRED_MECHANISMS},
                "sampling": {"temperature": 1, "top_p": 0.95, "top_k": 20,
                             "min_p": 0, "processors": [], "accept_rule": "residual"}}
    assert MODULE.validate_manifest(manifest, capture, artifact)["top_k"] == 20
    manifest_path = tmp_path / "capture-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    report = MODULE.analyze(capture, manifest_path, artifact)
    assert report["rows"] == 1
    assert report["strata"]["context_length"] == [16]
    assert report["qualification"] == "synthetic_only"
    assert not report["gpu_ab_admissible"]
    assert "synthetic capture" in report["gate_reasons"]
    out = tmp_path / "report.json"
    args = ["--capture", str(capture), "--manifest", str(manifest_path),
            "--artifact", str(artifact), "--out", str(out)]
    assert MODULE.main(args) == 0
    assert json.loads(out.read_text())["manifest_sha256"] == MODULE.sha256(manifest_path)
    with pytest.raises(SystemExit, match="2"):
        MODULE.main(args)
    manifest["mechanism_counts"]["q4_fc_coarse_head_steps"] = 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="engage once"):
        MODULE.analyze(capture, manifest_path, artifact)
    manifest["mechanism_counts"]["q4_fc_coarse_head_steps"] = 1
    manifest_path.write_text(json.dumps(manifest))
    manifest["sampling"]["processors"] = ["grammar"]
    with pytest.raises(ValueError, match="processors"):
        MODULE.validate_manifest(manifest, capture, artifact)
    manifest["sampling"]["processors"] = []
    manifest["mlx2_contract_sha256"][MODULE.MLX2_CONTRACT_PATHS[0]] = "0" * 64
    with pytest.raises(ValueError, match="mlx2 contract SHA-256 mismatch"):
        MODULE.validate_manifest(manifest, capture, artifact)
    manifest["mlx2_contract_sha256"][MODULE.MLX2_CONTRACT_PATHS[0]] = (
        MODULE.sha256(source_root / MODULE.MLX2_CONTRACT_PATHS[0]))
    capture_code.write_text("# altered source\n")
    with pytest.raises(ValueError, match="capture code SHA-256"):
        MODULE.validate_manifest(manifest, capture, artifact)
    capture_code.write_text("# synthetic fixture\n")
    manifest["capture_kind"] = "matched_model"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="bounded B1 strata"):
        MODULE.analyze(capture, manifest_path, artifact)
    with np.load(capture, allow_pickle=False) as saved:
        changed = {key: saved[key] for key in saved.files}
    changed["target_row_sha256"] = np.array(["c" * 64], dtype="S64")
    np.savez(capture, **changed)
    manifest["capture_sha256"] = MODULE.sha256(capture)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identical, unique teacher-forced row digests"):
        MODULE.analyze(capture, manifest_path, artifact)
    changed["target_row_sha256"] = row_digest
    changed["row_token_ids"] = np.arange(17, dtype=np.uint32)[::-1]
    np.savez(capture, **changed)
    manifest["capture_sha256"] = MODULE.sha256(capture)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest does not match token history"):
        MODULE.analyze(capture, manifest_path, artifact)


def test_arm_alignment_and_gather_parity_are_required(tmp_path):
    digest = np.array(["a" * 64], dtype="S64")
    mismatch = np.array(["b" * 64], dtype="S64")
    assert MODULE._digest_rows(digest, "arm", 1) == ["a" * 64]
    with pytest.raises(ValueError, match="invalid SHA-256"):
        MODULE._digest_rows(np.array(["z" * 64], dtype="S64"), "arm", 1)
    assert digest.tolist() != mismatch.tolist()
    logits = np.linspace(5, -5, 128, dtype=np.float32)
    ids = np.arange(64, dtype=np.int32)
    with pytest.raises(ValueError, match="rescore disagrees"):
        MODULE.evaluate_row(logits, ids, logits, temp=0.8, top_p=0.95, top_k=20,
                            baseline_dense_fc_logits=logits,
                            candidate_exact_logits=logits[ids] + 1.0)


def test_coarse_support_loss_is_visible_against_both_full_head_arms():
    logits = np.full(128, -10.0, dtype=np.float32)
    logits[100] = 10.0
    ids = np.arange(64, dtype=np.int32)
    row = MODULE.evaluate_row(logits, ids, logits, temp=0.8, top_p=0.95, top_k=20,
                              baseline_dense_fc_logits=logits,
                              candidate_exact_logits=logits[ids])
    assert row["full_q_support_misses"] == 1
    assert row["expected_accept_q4_fc_full"] > 0.99
    assert row["expected_accept_coarse_q"] < 0.01
    assert row["expected_accept_dense_fc_full"] > 0.99


def test_dense_fc_and_q4_fc_effects_are_separate_from_coarse_selection():
    dense = np.full(128, -12.0, dtype=np.float32)
    dense[7] = 12.0
    q4 = np.full(128, -12.0, dtype=np.float32)
    q4[72] = 12.0
    ids = np.arange(32, 96, dtype=np.int32)
    row = MODULE.evaluate_row(q4, ids, dense, temp=0.8, top_p=0.95, top_k=20,
                              baseline_dense_fc_logits=dense,
                              candidate_exact_logits=q4[ids])
    assert row["expected_accept_dense_fc_full"] > 0.99
    assert row["expected_accept_q4_fc_full"] < 0.01
    assert row["expected_accept_coarse_q"] < 0.01
    assert row["expected_accept_q4_fc_delta_vs_original"] < -0.99
    assert row["expected_accept_combined_delta_vs_original"] < -0.99


def test_analyzer_imports_without_real_mlx():
    code = """
import importlib.abc
import importlib.util
import sys
class BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            raise AssertionError('real MLX import attempted')
sys.meta_path.insert(0, BlockMLX())
spec = importlib.util.spec_from_file_location('coarse_cpu_only', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'mlx' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code, str(PATH)], check=True)
