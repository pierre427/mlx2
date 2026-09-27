#!/usr/bin/env python3
"""CPU-only matched-capture screen for oMLX #3958's 3-bit MTP draft head.

Input is an NPZ with baseline_dense_fc_logits [R,V], draft_exact_logits [R,V],
coarse_candidate_ids [R,64], candidate_exact_logits [R,64], and target_logits
[R,V]. Candidate IDs and
draft_exact_logits must come from the same hidden rows after upstream's
4-bit MTP-FC conversion; baseline_dense_fc_logits is the current mlx2
proposal on the same teacher-forced prefixes and draft tokens. A JSON
manifest pins the capture code, mlx2 source, model artifact and sampling. Each
arm also records a digest of its actual teacher-forced token history. See
provenance/coarse-head-pr3958-preflight.json. No MLX/GPU import occurs.

This compares proposal distributions and expected *one-step* acceptance.
Exact residual verification can retain the target distribution even when q
changes, provided the sparse q and transformed target p are used consistently.
No standalone microbench or model-quality claim follows from this script.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

OMLX_REV = "3e9703518984c294c2ed989cf968727d0e0ffa56"
SCHEMA = "mlx2-coarse-head-capture-v2"
GATHER_ATOL = 0.125  # first-pass numerical gate; revise only with captured evidence
MLX2_CONTRACT_PATHS = (
    "src/mlx2/runtime/sample_utils.py",
    "src/mlx2/runtime/hybrid_speculative.py",
    "src/mlx2/runtime/models/qwen38_27b.py",
)
REQUIRED_MECHANISMS = (
    "dense_fc_full_head_steps",
    "q4_fc_full_head_steps",
    "q4_fc_coarse_head_steps",
    "target_verify_steps",
    "committed_checkpoint_restore_checks",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _log_softmax(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    peak = np.max(v)
    if not np.isfinite(peak):
        raise ValueError("logit row has no finite maximum")
    return v - (peak + np.log(np.exp(v - peak, dtype=np.float32).sum(dtype=np.float32)))


def mlx2_q(logits: np.ndarray, *, temp: float, top_p: float, top_k: int) -> np.ndarray:
    """NumPy reading of sample_utils: full-vocab top-p, then top-k, then temp."""
    lp = _log_softmax(logits)
    v = lp.size
    if not (temp > 0 and 0 < top_k < v and (top_p == 0 or 0 < top_p < 1)):
        raise ValueError("unsupported mlx2 sampling values")
    if top_p:
        asc = np.argsort(lp, kind="stable")
        mass = np.cumsum(np.exp(lp[asc], dtype=np.float32), dtype=np.float32)
        keep = np.zeros(v, dtype=bool)
        keep[asc] = mass > np.float32(1 - top_p) * mass[-1]
        keep[asc[-1]] = True
        lp = np.where(keep, lp, -np.inf)
    top = np.argpartition(-lp, top_k - 1)[:top_k]
    keep = np.zeros(v, dtype=bool)
    keep[top] = True
    lp = np.where(keep, lp, -np.inf)
    scaled = lp / np.float32(temp)
    norm = _log_softmax(scaled)
    return np.exp(norm, dtype=np.float32)


def coarse_q(exact_logits: np.ndarray, candidate_ids: np.ndarray, *,
             temp: float, top_p: float, top_k: int,
             candidate_exact_logits: np.ndarray | None = None) -> np.ndarray:
    """PR #3958: normalize exact rescored 64 first, then top-k/top-p/temp."""
    ids = np.asarray(candidate_ids)
    v = exact_logits.size
    if ids.shape != (64,) or ids.dtype.kind not in "iu" or np.any(ids < 0) \
            or np.any(ids >= v) or np.unique(ids).size != 64:
        raise ValueError("coarse_candidate_ids must be 64 unique in-vocab integers")
    if not (0 < top_k <= 32 and temp > 0):
        raise ValueError("coarse route requires 1..32 top-k and temp > 0")
    scores = exact_logits[ids] if candidate_exact_logits is None else candidate_exact_logits
    if np.asarray(scores).shape != (64,):
        raise ValueError("candidate_exact_logits must have 64 scores")
    if not np.all(np.isfinite(scores)):
        raise ValueError("candidate_exact_logits must be finite")
    cand_lp = _log_softmax(scores)
    order = np.lexsort((np.arange(64), -cand_lp))
    rank = order[:top_k]
    vals = cand_lp[rank]
    if top_p:
        prob = np.exp(vals, dtype=np.float32)
        before = np.cumsum(prob, dtype=np.float32) - prob
        vals = np.where(before < top_p, vals, -np.inf)
    scaled = vals / np.float32(temp)
    q = np.zeros(v, dtype=np.float32)
    q[ids[rank]] = np.exp(_log_softmax(scaled), dtype=np.float32)
    return q


def evaluate_row(draft_logits: np.ndarray, ids: np.ndarray, target_logits: np.ndarray,
                 *, temp: float, top_p: float, top_k: int,
                 baseline_dense_fc_logits: np.ndarray | None = None,
                 candidate_exact_logits: np.ndarray | None = None) -> dict:
    for name, values in (("q4FC/full-head", draft_logits), ("target", target_logits),
                         ("denseFC/full-head", baseline_dense_fc_logits)):
        if values is not None and not np.all(np.isfinite(values)):
            raise ValueError(f"{name} logits must be finite")
    exact_lp = _log_softmax(draft_logits)
    full_mass = np.exp(exact_lp, dtype=np.float32)
    q_full = mlx2_q(draft_logits, temp=temp, top_p=top_p, top_k=top_k)
    q_coarse = coarse_q(draft_logits, ids, temp=temp, top_p=top_p, top_k=top_k,
                        candidate_exact_logits=candidate_exact_logits)
    if candidate_exact_logits is not None and not np.allclose(
            candidate_exact_logits, draft_logits[np.asarray(ids, dtype=np.int64)],
            rtol=0, atol=GATHER_ATOL):
        raise ValueError("exact candidate rescore disagrees with q4FC full head")
    p = mlx2_q(target_logits, temp=temp, top_p=top_p, top_k=top_k)
    cand = np.zeros(draft_logits.size, dtype=bool)
    cand[ids] = True
    top20 = np.argsort(-exact_lp, kind="stable")[:20]
    base_accept = float(np.minimum(p, q_full).sum(dtype=np.float64))
    coarse_accept = float(np.minimum(p, q_coarse).sum(dtype=np.float64))
    result = {
        "top20_coverage": float(cand[top20].mean()),
        "full_q_support_misses": int(np.count_nonzero((q_full > 0) & ~cand)),
        "candidate_omitted_full_mass": float(full_mass[~cand].sum(dtype=np.float64)),
        "q_support_symmetric_difference": int(np.count_nonzero((q_full > 0) != (q_coarse > 0))),
        "q_total_variation": float(0.5 * np.abs(q_full - q_coarse).sum(dtype=np.float64)),
        "expected_accept_q4_fc_full": base_accept,
        "expected_accept_coarse_q": coarse_accept,
        "expected_accept_delta": coarse_accept - base_accept,
        "target_top1_in_candidates": bool(cand[int(np.argmax(target_logits))]),
    }
    if candidate_exact_logits is not None:
        result["gather_vs_full_head_max_abs"] = float(np.max(
            np.abs(np.asarray(candidate_exact_logits, dtype=np.float32)
                   - np.asarray(draft_logits[ids], dtype=np.float32))))
    if baseline_dense_fc_logits is not None:
        q_original = mlx2_q(baseline_dense_fc_logits, temp=temp, top_p=top_p,
                            top_k=top_k)
        original_accept = float(np.minimum(p, q_original).sum(dtype=np.float64))
        result.update({
            "q4_fc_vs_original_tv": float(0.5 * np.abs(q_full - q_original).sum(dtype=np.float64)),
            "coarse_vs_original_tv": float(0.5 * np.abs(q_coarse - q_original).sum(dtype=np.float64)),
            "expected_accept_dense_fc_full": original_accept,
            "expected_accept_q4_fc_delta_vs_original": base_accept - original_accept,
            "expected_accept_combined_delta_vs_original": coarse_accept - original_accept,
        })
    return result


def _digest_rows(values: np.ndarray, name: str, rows: int) -> list[str]:
    if values.shape != (rows,) or values.dtype.kind not in "SU":
        raise ValueError(f"{name} must be a [R] string array of SHA-256 digests")
    decoded = [v.decode("ascii") if isinstance(v, bytes) else str(v) for v in values]
    if any(len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
           for v in decoded):
        raise ValueError(f"{name} contains an invalid SHA-256 digest")
    return decoded


def teacher_forced_row_sha256(tokens: np.ndarray, *, context_length: int,
                              lane_id: int, seed: int, draft_position: int,
                              target_offset: int, draft_offset: int) -> str:
    """Canonical digest of the prefix, sampled anchor and forced draft history."""
    h = hashlib.sha256(b"mlx2-coarse-head-row-v2\0")
    h.update(np.asarray((context_length, lane_id, seed, draft_position,
                         target_offset, draft_offset), dtype="<i8").tobytes())
    h.update(np.asarray(tokens, dtype="<u4").tobytes())
    return h.hexdigest()


def validate_manifest(manifest: dict, capture: Path, artifact: Path) -> dict:
    required = ("schema", "mlx2_source_revision", "omlx_source_revision",
                "capture_code_path", "capture_code_sha256", "coarse_head_parameters_sha256",
                "capture_sha256", "artifact_config_sha256", "artifact_index_sha256",
                "route", "alignment", "sampling", "capture_kind",
                "mlx2_source_root", "mlx2_contract_sha256", "mechanism_counts")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"manifest missing {missing}")
    if manifest["schema"] != SCHEMA or manifest["omlx_source_revision"] != OMLX_REV:
        raise ValueError("capture schema or pinned oMLX revision mismatch")
    if manifest["route"] != "self_mtp_draft":
        raise ValueError("capture is not a self-MTP draft route")
    if manifest["alignment"] != "teacher_forced_same_prefix_and_draft_tokens":
        raise ValueError("capture lacks aligned teacher-forced proposal rows")
    if manifest["capture_kind"] not in ("synthetic_test", "matched_model"):
        raise ValueError("unknown capture kind")
    counts = manifest["mechanism_counts"]
    if (not isinstance(counts, dict) or set(counts) != set(REQUIRED_MECHANISMS)
            or any(type(value) is not int or value < 1 for value in counts.values())):
        raise ValueError("capture requires positive per-arm mechanism and restore counts")
    for key in ("mlx2_source_revision", "capture_code_sha256",
                "coarse_head_parameters_sha256"):
        value = manifest[key]
        if not isinstance(value, str) or len(value) != (40 if key.endswith("revision") else 64) \
                or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{key} must be a full lowercase hex digest")
    if sha256(Path(manifest["capture_code_path"])) != manifest["capture_code_sha256"]:
        raise ValueError("capture code SHA-256 mismatch")
    source_root = Path(manifest["mlx2_source_root"])
    source_revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if source_revision != manifest["mlx2_source_revision"]:
        raise ValueError("mlx2 source revision mismatch")
    contracts = manifest["mlx2_contract_sha256"]
    if not isinstance(contracts, dict) or set(contracts) != set(MLX2_CONTRACT_PATHS):
        raise ValueError("mlx2 contract hashes must name exactly the MTP source files")
    for relative in MLX2_CONTRACT_PATHS:
        if sha256(source_root / relative) != contracts[relative]:
            raise ValueError(f"mlx2 contract SHA-256 mismatch: {relative}")
    for name, expected in (("capture", manifest["capture_sha256"]),
                           ("config", manifest["artifact_config_sha256"]),
                           ("index", manifest["artifact_index_sha256"])):
        path = {"capture": capture, "config": artifact / "config.json",
                "index": artifact / "model.safetensors.index.json"}[name]
        if sha256(path) != expected:
            raise ValueError(f"{name} SHA-256 mismatch")
    sampling = manifest["sampling"]
    if (sampling.get("accept_rule") != "residual" or sampling.get("min_p", 0) != 0 or sampling.get("processors")
            or sampling.get("xtc_probability", 0) != 0):
        raise ValueError("coarse route requires residual acceptance and no min-p, processors, or XTC")
    temp = float(sampling["temperature"])
    top_p = float(sampling["top_p"])
    top_k = int(sampling["top_k"])
    if not (temp > 0 and 0 < top_k <= 32 and (top_p == 0 or 0 < top_p < 1)):
        raise ValueError("coarse route requires temp > 0 and 1..32 top-k")
    return {"temp": temp, "top_p": top_p, "top_k": top_k}


def analyze(capture: Path, manifest_path: Path, artifact: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    params = validate_manifest(manifest, capture, artifact)
    with np.load(capture, allow_pickle=False) as arrays:
        for key in ("baseline_dense_fc_logits", "draft_exact_logits",
                    "coarse_candidate_ids", "candidate_exact_logits", "target_logits",
                    "context_length", "lane_id", "seed", "draft_position",
                    "target_offset", "draft_offset", "dense_fc_row_sha256",
                    "q4_fc_full_row_sha256", "q4_fc_coarse_row_sha256",
                    "target_row_sha256", "row_token_offsets", "row_token_ids"):
            if key not in arrays:
                raise ValueError(f"capture missing {key}")
        draft = arrays["draft_exact_logits"]
        baseline = arrays["baseline_dense_fc_logits"]
        ids = arrays["coarse_candidate_ids"]
        exact_candidates = arrays["candidate_exact_logits"]
        target = arrays["target_logits"]
        if (draft.ndim != 2 or target.shape != draft.shape or baseline.shape != draft.shape or
                ids.shape != (draft.shape[0], 64) or
                exact_candidates.shape != ids.shape or draft.shape[0] < 1 or
                draft.shape[1] <= params["top_k"] or
                any(a.dtype.kind != "f" for a in (draft, baseline, target, exact_candidates))):
            raise ValueError("capture arrays must be three [R,V] and two [R,64]")
        if any(count != draft.shape[0] for count in manifest["mechanism_counts"].values()):
            raise ValueError("every arm and checkpoint restore must engage once per captured row")
        arm_keys = [_digest_rows(arrays[key], key, draft.shape[0]) for key in
                    ("dense_fc_row_sha256", "q4_fc_full_row_sha256",
                     "q4_fc_coarse_row_sha256", "target_row_sha256")]
        if any(keys != arm_keys[0] for keys in arm_keys[1:]) or len(set(arm_keys[0])) != len(arm_keys[0]):
            raise ValueError("matched arms need identical, unique teacher-forced row digests")
        meta = {key: arrays[key] for key in ("context_length", "lane_id", "seed",
                                              "draft_position", "target_offset", "draft_offset")}
        if any(a.shape != (draft.shape[0],) or a.dtype.kind not in "iu"
               for a in meta.values()):
            raise ValueError("row metadata must be integer [R] arrays")
        if (np.any(meta["context_length"] < 1) or np.any(meta["lane_id"] < 0)
                or np.any(meta["seed"] < 0) or np.any(meta["draft_position"] < 0)
                or np.any(meta["target_offset"] != meta["context_length"])
                or np.any(meta["draft_offset"] != meta["target_offset"] - 1)):
            raise ValueError("row metadata violates committed MTP boundary invariant")
        offsets = arrays["row_token_offsets"]
        token_ids = arrays["row_token_ids"]
        if (offsets.shape != (draft.shape[0] + 1,) or offsets.dtype.kind not in "iu"
                or token_ids.ndim != 1 or token_ids.dtype.kind not in "iu"
                or offsets[0] != 0 or offsets[-1] != token_ids.size
                or np.any(np.diff(offsets.astype(np.int64)) < 1)
                or np.any(token_ids < 0) or np.any(token_ids >= draft.shape[1])):
            raise ValueError("row token history is malformed or out of vocabulary")
        for i in range(draft.shape[0]):
            history = token_ids[int(offsets[i]):int(offsets[i + 1])]
            if history.size != int(meta["context_length"][i]) + 1 + int(meta["draft_position"][i]):
                raise ValueError("row token history length does not match MTP position")
            digest = teacher_forced_row_sha256(history, **{
                key: int(values[i]) for key, values in meta.items()})
            if digest != arm_keys[0][i]:
                raise ValueError("teacher-forced row digest does not match token history")
        if manifest["capture_kind"] == "matched_model":
            contexts = set(meta["context_length"].tolist())
            seeds = set(meta["seed"].tolist())
            positions = set(meta["draft_position"].tolist())
            observed = list(zip(meta["context_length"].tolist(), meta["seed"].tolist(),
                                meta["draft_position"].tolist()))
            expected = set(product(contexts, seeds, positions))
            if (draft.shape[0] != 36 or draft.shape[1] < 65536 or len(contexts) != 4
                    or len(seeds) != 3 or positions != {0, 1, 2}
                    or np.any(meta["lane_id"] != 0) or set(observed) != expected
                    or len(set(observed)) != 36):
                raise ValueError("matched model capture lacks the bounded B1 strata")
            config = json.loads((artifact / "config.json").read_text())
            index = json.loads((artifact / "model.safetensors.index.json").read_text())
            text_config = config.get("text_config", config)
            quant = config.get("quantization", {})
            weights = index.get("weight_map", {})
            if (text_config.get("vocab_size") != draft.shape[1] or
                    config.get("tie_word_embeddings", text_config.get("tie_word_embeddings")) is not False or
                    quant.get("mode") != "affine" or quant.get("bits") != 4 or
                    quant.get("group_size") != 64 or
                    "language_model.lm_head.weight" not in weights or
                    "language_model.mtp.fc.weight" not in weights):
                raise ValueError("artifact head geometry does not match captured full vocabulary")
        strata = {key: sorted({int(v) for v in arr})
                  for key, arr in meta.items() if key in ("context_length", "lane_id",
                                                        "seed", "draft_position")}
        rows = [evaluate_row(draft[i], ids[i], target[i],
                             baseline_dense_fc_logits=baseline[i],
                             candidate_exact_logits=exact_candidates[i], **params)
                for i in range(draft.shape[0])]
    fields = rows[0]
    lower_is_worse = {"top20_coverage", "expected_accept_delta",
                      "expected_accept_q4_fc_delta_vs_original",
                      "expected_accept_combined_delta_vs_original",
                      "target_top1_in_candidates"}
    summary = {key: {"mean": float(np.mean([r[key] for r in rows])),
                     "worst": float(min(r[key] for r in rows) if key in lower_is_worse
                                    else max(r[key] for r in rows))}
               for key in fields}
    reasons = []
    if manifest["capture_kind"] != "matched_model":
        reasons.append("synthetic capture")
    if summary["top20_coverage"]["worst"] < 1.0 or summary["full_q_support_misses"]["worst"]:
        reasons.append("coarse candidates omit exact full-head top-20 or proposal support")
    if summary["q_total_variation"]["worst"] > 0.02:
        reasons.append("coarse proposal TV exceeds 0.02")
    if summary["expected_accept_delta"]["worst"] < -0.01:
        reasons.append("coarse acceptance loses more than 0.01 against q4FC/full-head")
    if summary["expected_accept_combined_delta_vs_original"]["worst"] < -0.01:
        reasons.append("combined acceptance loses more than 0.01 against denseFC/full-head")
    return {"kind": "coarse_head_offline_preflight",
            "qualification": "candidate_only" if manifest["capture_kind"] == "matched_model" else "synthetic_only",
            "capture_kind": manifest["capture_kind"],
            "gpu_ab_admissible": not reasons, "gate_reasons": reasons,
            "capture": str(capture.resolve()), "capture_sha256": sha256(capture),
            "manifest_sha256": sha256(manifest_path), "artifact": str(artifact.resolve()),
            "mlx2_source_revision": manifest["mlx2_source_revision"],
            "omlx_source_revision": OMLX_REV,
            "coarse_head_parameters_sha256": manifest["coarse_head_parameters_sha256"],
            "mechanism_counts": manifest["mechanism_counts"],
            "analyzer_sha256": sha256(Path(__file__)), "sampling": manifest["sampling"],
            "rows": len(rows), "strata": strata, "summary": summary}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        ap.error("refusing to overwrite an existing receipt")
    try:
        report = analyze(args.capture, args.manifest, args.artifact)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError,
            json.JSONDecodeError) as exc:
        ap.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
