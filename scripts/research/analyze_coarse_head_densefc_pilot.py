#!/usr/bin/env python3
"""CPU-only source-bound screen for the dense-FC/coarse-head B1 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import coarse_head_b1_pilot as pilot  # noqa: E402
from offline_coarse_head_preflight import (  # noqa: E402
    OMLX_REV, evaluate_row, sha256, teacher_forced_row_sha256,
)

ARM_DIGESTS = ("dense_fc_full_row_sha256", "dense_fc_coarse_row_sha256",
               "target_row_sha256")


def verify_provenance(capture: Path, manifest_path: Path,
                      baseline_capture: Path, baseline_manifest_path: Path,
                      artifact: Path | None) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text())
    baseline = json.loads(baseline_manifest_path.read_text())
    if (manifest.get("schema") != "mlx2-coarse-head-densefc-pilot-v1" or
            baseline.get("schema") != "mlx2-coarse-head-pilot-v1" or
            manifest.get("capture_kind") != baseline.get("capture_kind") or
            manifest.get("omlx_source_revision") != OMLX_REV or
            baseline.get("omlx_source_revision") != OMLX_REV):
        raise ValueError("pilot schemas, capture kinds or pinned upstream revision differ")
    for name, value, path in (
        ("dense capture", manifest.get("capture_sha256"), capture),
        ("dense runner", manifest.get("capture_code_sha256"),
         Path(manifest["capture_code_path"])),
        ("baseline capture", baseline.get("capture_sha256"), baseline_capture),
        ("baseline runner", baseline.get("capture_code_sha256"),
         Path(baseline["capture_code_path"]))):
        if value != sha256(path):
            raise ValueError(f"{name} SHA-256 mismatch")
    if (manifest.get("baseline_capture_sha256") != (sha256(baseline_capture) if artifact else None)
            or manifest.get("baseline_manifest_sha256") != (sha256(baseline_manifest_path) if artifact else None)
            or manifest.get("coarse_head_parameters_sha256") != baseline.get("coarse_head_parameters_sha256")):
        raise ValueError("baseline binding or coarse parameters mismatch")
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    contracts = {name: sha256(ROOT / name) for name in pilot.MLX2_CONTRACT_PATHS}
    for label, item in (("dense", manifest), ("baseline", baseline)):
        if (item.get("mlx2_source_revision") != revision or
                item.get("mlx2_source_root") != str(ROOT) or
                item.get("mlx2_contract_sha256") != contracts or
                item.get("sampling") != pilot.SAMPLING):
            raise ValueError(f"{label} source or sampling contract mismatch")
    if artifact is not None:
        for label, item in (("dense", manifest), ("baseline", baseline)):
            if (item.get("artifact") != str(artifact) or
                    item.get("artifact_config_sha256") != sha256(artifact / "config.json") or
                    item.get("artifact_index_sha256") != sha256(artifact / "model.safetensors.index.json")):
                raise ValueError(f"{label} model artifact mismatch")
    return manifest, baseline


def analyze(capture: Path, manifest_path: Path,
            baseline_capture: Path, baseline_manifest_path: Path,
            artifact: Path | None = None) -> dict:
    manifest, baseline = verify_provenance(
        capture, manifest_path, baseline_capture, baseline_manifest_path, artifact)
    if manifest["capture_kind"] == "matched_model_pilot" and artifact is None:
        raise ValueError("real model pilot requires artifact path")
    if manifest["capture_kind"] not in ("matched_model_pilot", "synthetic_test"):
        raise ValueError("unknown capture kind")
    with np.load(capture, allow_pickle=False) as current, np.load(
            baseline_capture, allow_pickle=False) as original:
        full = current["dense_fc_full_logits"]
        target = current["target_logits"]
        ids = current["dense_fc_coarse_candidate_ids"]
        scores = current["dense_fc_candidate_exact_logits"]
        base_full = original["baseline_dense_fc_logits"]
        base_target = original["target_logits"]
        if (full.ndim != 2 or full.shape[0] != 3 or full.shape[1] < 128 or
                target.shape != full.shape or base_full.shape != full.shape or
                base_target.shape != full.shape or ids.shape != (3, 64) or
                scores.shape != (3, 64) or any(x.dtype.kind != "f" for x in
                (full, target, base_full, base_target, scores))):
            raise ValueError("expected three aligned B1 rows with full-vocab logits")
        if (not np.all(np.isfinite(full)) or not np.all(np.isfinite(target)) or
                not np.all(np.isfinite(scores))):
            raise ValueError("nonfinite model logits")
        if any(manifest["mechanism_counts"].get(name) != 3 for name in (
                "dense_fc_full_head_steps", "dense_fc_coarse_head_steps",
                "target_verify_steps", "committed_checkpoint_restore_checks")):
            raise ValueError("model arm or checkpoint count did not engage three times")
        for key in ("context_length", "lane_id", "seed", "draft_position",
                    "target_offset", "draft_offset", "row_token_offsets", "row_token_ids"):
            if not np.array_equal(current[key], original[key]):
                raise ValueError(f"baseline row metadata/history differs: {key}")
        for j in range(3):
            a, b = current["row_token_offsets"][j:j+2]
            history = current["row_token_ids"][a:b]
            meta = {key: int(current[key][j]) for key in (
                "context_length", "lane_id", "seed", "draft_position",
                "target_offset", "draft_offset")}
            if (meta["target_offset"] != meta["context_length"] or
                    meta["draft_offset"] != meta["target_offset"] - 1 or
                    len(history) != meta["context_length"] + 1 + j or
                    meta["draft_position"] != j or meta["lane_id"] != 0):
                raise ValueError("committed B1 row invariant failed")
            digest = teacher_forced_row_sha256(history, **meta).encode()
            if any(current[key][j] != digest for key in ARM_DIGESTS):
                raise ValueError("dense/coarse/target arm history digest mismatch")
        full_replay = float(np.max(np.abs(full - base_full)))
        target_replay = float(np.max(np.abs(target - base_target)))
        if full_replay > 0.125 or target_replay > 0.125:
            raise ValueError("dense full-head or target logits failed baseline replay")
        rows = [evaluate_row(full[j], ids[j], target[j], temp=0.8, top_p=0.95,
                             top_k=20, baseline_dense_fc_logits=base_full[j],
                             candidate_exact_logits=scores[j])
                for j in range(3)]
    summary = {
        "min_top20_coverage": min(row["top20_coverage"] for row in rows),
        "max_full_q_support_misses": max(row["full_q_support_misses"] for row in rows),
        "max_q_total_variation": max(row["q_total_variation"] for row in rows),
        "min_coarse_vs_dense_accept_delta": min(row["expected_accept_delta"] for row in rows),
        "min_coarse_vs_original_dense_accept_delta": min(
            row["expected_accept_combined_delta_vs_original"] for row in rows),
        "max_gather_vs_full_head_abs": max(row["gather_vs_full_head_max_abs"] for row in rows),
        "max_dense_replay_abs": full_replay,
        "max_target_replay_abs": target_replay,
    }
    pilot_gate_pass = (summary["min_top20_coverage"] == 1.0 and
                       summary["max_full_q_support_misses"] == 0 and
                       summary["max_q_total_variation"] <= 0.02 and
                       summary["min_coarse_vs_dense_accept_delta"] >= -0.01 and
                       summary["min_coarse_vs_original_dense_accept_delta"] >= -0.01)
    return {
        "schema": "mlx2-coarse-head-densefc-analysis-v1",
        "capture_kind": manifest["capture_kind"],
        "qualification": "pilot_only" if artifact is not None else "synthetic_only",
        "capture_sha256": sha256(capture),
        "manifest_sha256": sha256(manifest_path),
        "baseline_capture_sha256": sha256(baseline_capture),
        "baseline_manifest_sha256": sha256(baseline_manifest_path),
        "analyzer_sha256": sha256(Path(__file__)),
        "rows": rows,
        "summary": summary,
        "pilot_gate_pass": pilot_gate_pass,
        "gpu_ab_admissible": False,
        "limits": ["Three B1 rows are insufficient for the 36-row admission gate",
                   "Dense-FC/coarse is a new candidate distinct from pinned oMLX #3958"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--baseline-capture", type=Path, required=True)
    ap.add_argument("--baseline-manifest", type=Path, required=True)
    ap.add_argument("--artifact", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        ap.error("refusing to overwrite existing report")
    report = analyze(args.capture, args.manifest, args.baseline_capture,
                     args.baseline_manifest, args.artifact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
