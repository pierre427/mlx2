#!/usr/bin/env python3
"""Research-only B1 dense-FC/coarse-head isolation for oMLX #3958.

This is a new candidate relative to the pinned PR: its MTP FC remains dense.
Metal mode replays the first pilot's exact prompt, anchor and forced drafts.
It requires an externally held GPU lease and host lock.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import coarse_head_b1_pilot as pilot  # noqa: E402
from offline_coarse_head_preflight import (  # noqa: E402
    OMLX_REV, teacher_forced_row_sha256,
)


def read_baseline(capture: Path, manifest_path: Path, model_path: Path):
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "mlx2-coarse-head-pilot-v1" or
            manifest.get("capture_kind") != "matched_model_pilot" or
            manifest.get("capture_sha256") != pilot.sha(capture) or
            manifest.get("capture_code_sha256") != pilot.sha(Path(manifest["capture_code_path"])) or
            manifest.get("omlx_source_revision") != OMLX_REV or
            manifest.get("artifact") != str(model_path) or
            manifest.get("artifact_config_sha256") != pilot.sha(model_path / "config.json") or
            manifest.get("artifact_index_sha256") != pilot.sha(model_path / "model.safetensors.index.json")):
        raise ValueError("baseline pilot source or artifact provenance mismatch")
    current_revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if (manifest.get("mlx2_source_revision") != current_revision or
            manifest.get("mlx2_contract_sha256") != {
                name: pilot.sha(ROOT / name) for name in pilot.MLX2_CONTRACT_PATHS}):
        raise ValueError("baseline mlx2 source contracts changed")
    with np.load(capture, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if (arrays["baseline_dense_fc_logits"].shape[0] != 3 or
            arrays["row_token_offsets"].shape != (4,) or
            len(set(arrays["context_length"].tolist())) != 1 or
            len(set(arrays["seed"].tolist())) != 1):
        raise ValueError("baseline is not one three-position B1 pilot")
    context = int(arrays["context_length"][0])
    prefix = arrays["row_token_ids"][:context]
    anchor = int(manifest["anchor_token"])
    forced = [int(value) for value in manifest["forced_draft_tokens"]]
    if len(forced) != 2:
        raise ValueError("baseline forced draft count mismatch")
    for j in range(3):
        a, b = arrays["row_token_offsets"][j:j + 2]
        expected = np.concatenate((prefix, np.asarray([anchor, *forced[:j]], dtype=np.uint32)))
        if not np.array_equal(arrays["row_token_ids"][a:b], expected):
            raise ValueError("baseline recorded token history does not match forced inputs")
        metadata = {key: int(arrays[key][j]) for key in (
            "context_length", "lane_id", "seed", "draft_position",
            "target_offset", "draft_offset")}
        digest = teacher_forced_row_sha256(expected, **metadata).encode()
        if any(arrays[name][j] != digest for name in (
                "dense_fc_row_sha256", "q4_fc_full_row_sha256",
                "q4_fc_coarse_row_sha256", "target_row_sha256")):
            raise ValueError("baseline arm digest mismatch")
    return manifest, arrays, prefix, anchor, forced


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=pilot.MODEL)
    ap.add_argument("--baseline-capture", type=Path)
    ap.add_argument("--baseline-manifest", type=Path)
    ap.add_argument("--tiny-cpu", action="store_true")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.tiny_cpu == args.i_own_the_gpu:
        ap.error("choose exactly one of --tiny-cpu or --i-own-the-gpu")
    if args.out.exists() or args.out.with_suffix(".json").exists():
        ap.error("output already exists; choose a new path")
    if args.i_own_the_gpu and (args.baseline_capture is None or args.baseline_manifest is None):
        ap.error("Metal mode requires baseline capture and manifest")
    import mlx.core as mx
    if args.tiny_cpu:
        mx.set_default_device(mx.cpu)
        model, prefix, vocab = pilot.tiny_model()
        seed = 11
        anchor, checkpoint, checkpoint_seed = pilot.prepare(model, prefix, seed)
        forced, target = pilot.target_rows(model, checkpoint, anchor, 3)
        baseline_manifest = baseline_arrays = None
    else:
        from mlx2.adapters.registry import resolve_adapter
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            raise RuntimeError("Metal GPU unavailable")
        baseline_manifest, baseline_arrays, prefix_ids, expected_anchor, forced = (
            read_baseline(args.baseline_capture, args.baseline_manifest, args.model))
        seed = int(baseline_arrays["seed"][0])
        prefix = mx.array(prefix_ids, mx.uint32)
        adapter = resolve_adapter(args.model, mtp=True)(str(args.model))
        model = adapter.model
        vocab = int(adapter.tokenizer.vocab_size)
        anchor, checkpoint, checkpoint_seed = pilot.prepare(model, prefix, seed)
        if anchor != expected_anchor:
            raise RuntimeError("sampled anchor did not replay baseline")
        checked_forced, target = pilot.target_rows(model, checkpoint, anchor, 3)
        if checked_forced != forced:
            raise RuntimeError("target greedy continuation did not replay baseline")
        if not np.allclose(np.asarray(target), baseline_arrays["target_logits"],
                           rtol=0, atol=0.125):
            raise RuntimeError("target logits did not replay baseline")
    lang, head, fc = pilot.head_and_fc(model)
    coarse, coarse_digest = pilot.make_coarse_head(head)
    if baseline_manifest is not None and coarse_digest != baseline_manifest["coarse_head_parameters_sha256"]:
        raise RuntimeError("coarse head parameters differ from prior pilot")
    dense, _, _ = pilot.draft_rows(model, checkpoint, anchor, forced)
    pilot.assert_checkpoint(checkpoint, int(prefix.size), checkpoint_seed)
    dense_again, ids, scores = pilot.draft_rows(
        model, checkpoint, anchor, forced, coarse=coarse, head=head)
    pilot.assert_checkpoint(checkpoint, int(prefix.size), checkpoint_seed)
    if (not np.array_equal(np.asarray(dense), np.asarray(dense_again)) or
            len(dense) != len(ids) or len(ids) != 3):
        raise RuntimeError("dense FC full-head and coarse branches diverged")
    if baseline_arrays is not None and not np.allclose(
            np.asarray(dense), baseline_arrays["baseline_dense_fc_logits"],
            rtol=0, atol=0.125):
        raise RuntimeError("dense FC logits did not replay baseline")
    if any(not np.allclose(np.asarray(dense[j])[ids[j]], scores[j],
                           rtol=0, atol=0.125) for j in range(3)):
        raise RuntimeError("dense FC exact candidate rescore mismatch")
    if not isinstance(fc, __import__("mlx.nn", fromlist=["Linear"]).Linear):
        raise RuntimeError("MTP FC was changed during dense-only coarse capture")
    prefix_ids = np.asarray(prefix).astype(np.uint32)
    histories = [np.concatenate((prefix_ids, np.asarray([anchor, *forced[:j]], np.uint32)))
                 for j in range(3)]
    metadata = {
        "context_length": np.full(3, len(prefix_ids), np.int64),
        "lane_id": np.zeros(3, np.int64),
        "seed": np.full(3, seed, np.int64),
        "draft_position": np.arange(3, dtype=np.int64),
        "target_offset": np.full(3, len(prefix_ids), np.int64),
        "draft_offset": np.full(3, len(prefix_ids)-1, np.int64),
    }
    digests = np.array([
        teacher_forced_row_sha256(history, **{key: int(val[j]) for key, val in metadata.items()})
        for j, history in enumerate(histories)], dtype="S64")
    arrays = {
        "dense_fc_full_logits": np.asarray(dense, np.float32),
        "dense_fc_coarse_candidate_ids": np.asarray(ids, np.int32),
        "dense_fc_candidate_exact_logits": np.asarray(scores, np.float32),
        "target_logits": np.asarray(target, np.float32),
        "row_token_ids": np.concatenate(histories),
        "row_token_offsets": np.cumsum([0, *map(len, histories)], dtype=np.int64),
        "dense_fc_full_row_sha256": digests,
        "dense_fc_coarse_row_sha256": digests,
        "target_row_sha256": digests,
        **metadata,
    }
    if any(not np.all(np.isfinite(arrays[key])) for key in (
            "dense_fc_full_logits", "dense_fc_candidate_exact_logits", "target_logits")):
        raise RuntimeError("nonfinite dense-FC capture")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    receipt = {
        "schema": "mlx2-coarse-head-densefc-pilot-v1",
        "capture_kind": "matched_model_pilot" if args.i_own_the_gpu else "synthetic_test",
        "route": "research_dense_fc_coarse_head",
        "alignment": "teacher_forced_same_committed_checkpoint_and_tokens",
        "sampling": pilot.SAMPLING,
        "mlx2_source_root": str(ROOT),
        "mlx2_source_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "mlx2_contract_sha256": {name: pilot.sha(ROOT / name) for name in pilot.MLX2_CONTRACT_PATHS},
        "omlx_source_revision": OMLX_REV,
        "capture_code_path": str(Path(__file__).resolve()),
        "capture_code_sha256": pilot.sha(Path(__file__)),
        "capture_sha256": pilot.sha(args.out),
        "coarse_head_parameters_sha256": coarse_digest,
        "artifact": str(args.model) if args.i_own_the_gpu else None,
        "artifact_config_sha256": pilot.sha(args.model / "config.json") if args.i_own_the_gpu else None,
        "artifact_index_sha256": pilot.sha(args.model / "model.safetensors.index.json") if args.i_own_the_gpu else None,
        "baseline_capture_sha256": pilot.sha(args.baseline_capture) if args.i_own_the_gpu else None,
        "baseline_manifest_sha256": pilot.sha(args.baseline_manifest) if args.i_own_the_gpu else None,
        "anchor_token": anchor,
        "forced_draft_tokens": forced,
        "context_length": len(prefix_ids),
        "seed": seed,
        "vocab_size": vocab,
        "mechanism_counts": {name: 3 for name in (
            "dense_fc_full_head_steps", "dense_fc_coarse_head_steps",
            "target_verify_steps", "committed_checkpoint_restore_checks")},
        "limits": ["Three B1 rows only; new dense-FC/coarse candidate outside pinned oMLX route",
                   "No throughput, serving or quality qualification"],
    }
    args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"capture": str(args.out), "manifest": str(args.out.with_suffix('.json')),
                      "rows": 3, "coarse_head_parameters_sha256": coarse_digest}))


if __name__ == "__main__":
    main()
