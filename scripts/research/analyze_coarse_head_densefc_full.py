#!/usr/bin/env python3
"""CPU-only, source-bound admission screen for the 36-row dense-FC coarse head."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import coarse_head_b1_pilot as pilot  # noqa: E402
import coarse_head_densefc_full_capture as capture_runner  # noqa: E402
from offline_coarse_head_preflight import (  # noqa: E402
    OMLX_REV, _digest_rows, evaluate_row, sha256, teacher_forced_row_sha256,
)

ARM_DIGESTS = ("dense_fc_full_row_sha256", "dense_fc_coarse_row_sha256",
               "target_row_sha256")
META = ("context_length", "lane_id", "seed", "draft_position",
        "target_offset", "draft_offset")
TOP20_MIN = 1.0
TV_MAX = 0.02
ACCEPT_DELTA_MIN = -0.01
GATHER_ATOL = 0.125


def validate_provenance(capture: Path, manifest_path: Path, artifact: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "mlx2-coarse-head-densefc-full36-v1" or
            manifest.get("capture_kind") != "matched_model" or
            manifest.get("qualification") != "capture_only" or
            manifest.get("omlx_source_revision") != OMLX_REV or
            manifest.get("sampling") != pilot.SAMPLING or
            manifest.get("contexts") != list(capture_runner.CONTEXTS) or
            manifest.get("seeds") != list(capture_runner.SEEDS) or
            manifest.get("rows") != 36):
        raise ValueError("capture schema, sampling, or predeclared grid mismatch")
    if (manifest.get("capture_sha256") != sha256(capture) or
            Path(manifest.get("capture_code_path", "")).resolve() != Path(capture_runner.__file__).resolve() or
            manifest.get("capture_code_sha256") != sha256(Path(capture_runner.__file__)) or
            manifest.get("mlx2_source_root") != str(ROOT) or
            manifest.get("mlx2_source_revision") != subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip() or
            manifest.get("mlx2_contract_sha256") != {
                name: sha256(ROOT / name) for name in pilot.MLX2_CONTRACT_PATHS}):
        raise ValueError("capture or source provenance mismatch")
    if (manifest.get("artifact") != str(artifact) or
            manifest.get("artifact_config_sha256") != sha256(artifact / "config.json") or
            manifest.get("artifact_index_sha256") != sha256(
                artifact / "model.safetensors.index.json") or
            manifest.get("corpus_files_sha256") != {
                name: sha256(ROOT / name) for name in capture_runner.CORPUS_FILES}):
        raise ValueError("artifact or corpus provenance mismatch")
    digest = manifest.get("coarse_head_parameters_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("coarse-head parameter digest malformed")
    if (manifest.get("output") is None or
            (ROOT / manifest["output"]).resolve() != capture.resolve() or
            manifest.get("mechanism_counts") != {name: 36 for name in (
                "dense_fc_full_head_steps", "dense_fc_coarse_head_steps",
                "target_verify_steps", "committed_checkpoint_restore_checks")} or
            len(manifest.get("checkpoint_modes", [])) != 12):
        raise ValueError("capture output, mechanism or checkpoint counts mismatch")
    config = json.loads((artifact / "config.json").read_text())
    text_config = config.get("text_config", config)
    quant = config.get("quantization", {})
    if (text_config.get("vocab_size") != 248320 or
            config.get("tie_word_embeddings", text_config.get("tie_word_embeddings")) is not False or
            quant.get("mode") != "affine" or quant.get("bits") != 4 or
            quant.get("group_size") != 64):
        raise ValueError("model geometry differs from predeclared 27B gate")
    return manifest


def analyze(capture: Path, manifest_path: Path, artifact: Path) -> dict:
    manifest = validate_provenance(capture, manifest_path, artifact)
    corpus = capture_runner.corpus_tokens(artifact)
    if (len(corpus) != manifest.get("corpus_token_count") or
            __import__("hashlib").sha256(corpus.astype("<u4").tobytes()).hexdigest()
            != manifest.get("corpus_tokens_sha256")):
        raise ValueError("tokenized corpus digest mismatch")
    with np.load(capture, allow_pickle=False) as data:
        required = ("dense_fc_full_logits", "dense_fc_coarse_candidate_ids",
                    "dense_fc_candidate_exact_logits", "target_logits",
                    "row_token_offsets", "row_token_ids", *ARM_DIGESTS, *META)
        if any(key not in data for key in required):
            raise ValueError("capture array missing")
        full = data["dense_fc_full_logits"]
        target = data["target_logits"]
        ids = data["dense_fc_coarse_candidate_ids"]
        scores = data["dense_fc_candidate_exact_logits"]
        if (full.shape != (36, 248320) or target.shape != full.shape or
                ids.shape != (36, 64) or scores.shape != ids.shape or
                any(a.dtype.kind != "f" for a in (full, target, scores)) or
                ids.dtype.kind not in "iu" or
                any(not np.all(np.isfinite(a)) for a in (full, target, scores))):
            raise ValueError("full capture shapes, dtypes or finite logits invalid")
        meta = {key: data[key] for key in META}
        if any(a.shape != (36,) or a.dtype.kind not in "iu" for a in meta.values()):
            raise ValueError("row metadata must be integer [36] arrays")
        observed = [(int(meta["context_length"][j]), int(meta["seed"][j]),
                     int(meta["draft_position"][j])) for j in range(36)]
        expected = list(product(capture_runner.CONTEXTS, capture_runner.SEEDS, range(3)))
        if (observed != expected or np.any(meta["lane_id"] != 0) or
                np.any(meta["target_offset"] != meta["context_length"]) or
                np.any(meta["draft_offset"] != meta["target_offset"] - 1)):
            raise ValueError("capture does not match the ordered 4 x 3 x 3 B1 grid")
        offsets = data["row_token_offsets"]
        tokens = data["row_token_ids"]
        if (offsets.shape != (37,) or offsets.dtype.kind not in "iu" or
                tokens.ndim != 1 or tokens.dtype.kind not in "iu" or
                offsets[0] != 0 or offsets[-1] != len(tokens) or
                np.any(np.diff(offsets.astype(np.int64)) <= 0) or
                np.any(tokens >= 248320)):
            raise ValueError("row token histories malformed")
        digests = [_digest_rows(data[key], key, 36) for key in ARM_DIGESTS]
        if digests[0] != digests[1] or digests[0] != digests[2] or len(set(digests[0])) != 36:
            raise ValueError("arm histories differ or duplicate rows")
        for i, (context, seed, position) in enumerate(observed):
            history = tokens[int(offsets[i]):int(offsets[i + 1])]
            if (len(history) != context + position + 1 or
                    not np.array_equal(history[:context], corpus[:context])):
                raise ValueError("history length or corpus prefix mismatch")
            if position:
                previous = tokens[int(offsets[i-1]):int(offsets[i])]
                if observed[i-1][:2] != (context, seed) or not np.array_equal(
                        history[:len(previous)], previous):
                    raise ValueError("teacher-forced positions do not share history")
            digest = teacher_forced_row_sha256(history, **{
                key: int(values[i]) for key, values in meta.items()})
            if digest != digests[0][i]:
                raise ValueError("history SHA-256 mismatch")
        rows = []
        for i, (context, seed, position) in enumerate(observed):
            result = evaluate_row(full[i], ids[i], target[i], temp=0.8,
                                  top_p=0.95, top_k=20,
                                  candidate_exact_logits=scores[i])
            rows.append({"context_length": context, "seed": seed,
                         "draft_position": position, **result})
    summary = {
        "rows": len(rows),
        "min_top20_coverage": min(r["top20_coverage"] for r in rows),
        "max_full_q_support_misses": max(r["full_q_support_misses"] for r in rows),
        "max_q_total_variation": max(r["q_total_variation"] for r in rows),
        "min_expected_accept_delta": min(r["expected_accept_delta"] for r in rows),
        "max_gather_vs_full_head_abs": max(r["gather_vs_full_head_max_abs"] for r in rows),
    }
    passed = (summary["min_top20_coverage"] >= TOP20_MIN and
              summary["max_full_q_support_misses"] == 0 and
              summary["max_q_total_variation"] <= TV_MAX and
              summary["min_expected_accept_delta"] >= ACCEPT_DELTA_MIN and
              summary["max_gather_vs_full_head_abs"] <= GATHER_ATOL)
    return {
        "schema": "mlx2-coarse-head-densefc-full36-analysis-v1",
        "qualification": "direct_model_capture_screen_only",
        "capture_sha256": sha256(capture),
        "manifest_sha256": sha256(manifest_path),
        "analyzer_sha256": sha256(Path(__file__)),
        "thresholds": {"top20_coverage_min": TOP20_MIN,
                       "q_total_variation_max": TV_MAX,
                       "expected_accept_delta_min": ACCEPT_DELTA_MIN,
                       "gather_vs_full_head_abs_max": GATHER_ATOL},
        "rows": rows,
        "summary": summary,
        "gpu_ab_admissible": passed,
        "limits": ["Direct-model B1 proposal screen only; no latency or serving qualification"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        ap.error("refusing to overwrite existing report")
    result = analyze(args.capture, args.manifest, args.artifact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result["summary"], "gpu_ab_admissible": result["gpu_ab_admissible"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
