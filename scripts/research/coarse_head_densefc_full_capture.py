#!/usr/bin/env python3
"""Source-bound 36-row B1 dense-FC/coarse MTP capture; research only."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import coarse_head_b1_pilot as pilot  # noqa: E402
from offline_coarse_head_preflight import OMLX_REV, teacher_forced_row_sha256  # noqa: E402

CONTEXTS = (128, 8192, 16384, 65536)
SEEDS = (11, 23, 37)
CORPUS_FILES = ("docs/SERVING.md", "docs/PROVENANCE.md")
MIN_FREE_BYTES = 2 * 1024**3


def corpus_tokens(model_path: Path) -> np.ndarray:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False)
    text = "\n\n".join((ROOT / name).read_text() for name in CORPUS_FILES)
    ids = np.asarray(tokenizer.encode(text, add_special_tokens=False), dtype=np.uint32)
    if ids.size < max(CONTEXTS):
        raise ValueError(f"corpus yielded {ids.size} tokens, need at least {max(CONTEXTS)}")
    return ids


def preflight(model_path: Path, out: Path) -> dict:
    if out.exists() or out.with_suffix(".json").exists():
        raise ValueError("capture output already exists")
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    config = json.loads(config_path.read_text())
    index = json.loads(index_path.read_text())
    text_config = config.get("text_config", config)
    quant = config.get("quantization", {})
    weights = index.get("weight_map", {})
    if (text_config.get("vocab_size") != 248320 or
            config.get("tie_word_embeddings", text_config.get("tie_word_embeddings")) is not False or
            quant.get("mode") != "affine" or quant.get("bits") != 4 or
            quant.get("group_size") != 64 or
            "language_model.lm_head.weight" not in weights or
            "language_model.mtp.fc.weight" not in weights):
        raise ValueError("artifact geometry differs from source-bound 27B coarse gate")
    tokens = corpus_tokens(model_path)
    corpus_hash = __import__("hashlib").sha256(tokens.astype("<u4").tobytes()).hexdigest()
    parent = out.parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < MIN_FREE_BYTES:
        raise ValueError(f"capture needs at least 2 GiB free, found {free}")
    vocab = int(text_config["vocab_size"])
    rows = len(CONTEXTS) * len(SEEDS) * 3
    raw_logits = rows * vocab * 4 * 2
    raw_histories = 3 * len(SEEDS) * sum(CONTEXTS) * 4 + rows * 4 * 2
    prior = ROOT / "qualification/runs/coarse-head-3958-pilot/densefc-coarse-b1-c128-s11.npz"
    pilot_bytes = prior.stat().st_size if prior.exists() else None
    return {
        "schema": "mlx2-coarse-head-densefc-full36-preflight-v1",
        "capture_code_path": str(Path(__file__).resolve()),
        "capture_code_sha256": pilot.sha(Path(__file__)),
        "mlx2_source_root": str(ROOT),
        "mlx2_source_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "mlx2_contract_sha256": {name: pilot.sha(ROOT / name)
                                 for name in pilot.MLX2_CONTRACT_PATHS},
        "omlx_source_revision": OMLX_REV,
        "artifact": str(model_path),
        "artifact_config_sha256": pilot.sha(config_path),
        "artifact_index_sha256": pilot.sha(index_path),
        "corpus_files_sha256": {name: pilot.sha(ROOT / name) for name in CORPUS_FILES},
        "corpus_tokens_sha256": corpus_hash,
        "corpus_token_count": int(tokens.size),
        "contexts": list(CONTEXTS), "seeds": list(SEEDS),
        "rows": rows,
        "sampling": pilot.SAMPLING,
        "raw_logits_bytes": raw_logits,
        "raw_history_bytes_upper_bound": raw_histories,
        "pilot_compressed_bytes": pilot_bytes,
        "estimated_capture_bytes": None if pilot_bytes is None else pilot_bytes * 12 + raw_histories,
        "disk_free_bytes": free,
        "output": str(out),
        "qualification": "capture_preflight_only",
    }


def run_capture(model_path: Path, out: Path, admitted: dict) -> dict:
    import mlx.core as mx
    from mlx2.adapters.registry import resolve_adapter

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("Metal GPU unavailable")
    adapter = resolve_adapter(model_path, mtp=True)(str(model_path))
    model = adapter.model
    _, head, _ = pilot.head_and_fc(model)
    coarse, coarse_hash = pilot.make_coarse_head(head)
    tokens = corpus_tokens(model_path)
    if __import__("hashlib").sha256(tokens.astype("<u4").tobytes()).hexdigest() != admitted["corpus_tokens_sha256"]:
        raise RuntimeError("corpus tokens changed after CPU admission")
    storage = {name: [] for name in (
        "dense_fc_full_logits", "dense_fc_coarse_candidate_ids",
        "dense_fc_candidate_exact_logits", "target_logits",
        "context_length", "lane_id", "seed", "draft_position",
        "target_offset", "draft_offset", "row_token_ids")}
    offsets = [0]
    digests = []
    checkpoint_modes = []
    for context in CONTEXTS:
        prefix_ids = tokens[:context]
        for seed in SEEDS:
            prefix = mx.array(prefix_ids, mx.uint32)
            anchor, checkpoint, checkpoint_seed = pilot.prepare(model, prefix, seed)
            checkpoint_modes.append(checkpoint["snapshot_mode"])
            forced, target = pilot.target_rows(model, checkpoint, anchor, 3)
            full, _, _ = pilot.draft_rows(model, checkpoint, anchor, forced)
            full_again, ids, scores = pilot.draft_rows(
                model, checkpoint, anchor, forced, coarse=coarse, head=head)
            pilot.assert_checkpoint(checkpoint, context, checkpoint_seed)
            if not np.array_equal(np.asarray(full), np.asarray(full_again)):
                raise RuntimeError(f"dense/full and dense/coarse logits diverged at {context}/{seed}")
            for j in range(3):
                if not np.allclose(full[j][ids[j]], scores[j], rtol=0, atol=0.125):
                    raise RuntimeError(f"candidate rescore mismatch at {context}/{seed}/{j}")
                history = np.concatenate((prefix_ids, np.asarray([anchor, *forced[:j]], np.uint32)))
                meta = dict(context_length=context, lane_id=0, seed=seed,
                            draft_position=j, target_offset=context, draft_offset=context-1)
                digests.append(teacher_forced_row_sha256(history, **meta))
                for name, value in meta.items():
                    storage[name].append(value)
                storage["row_token_ids"].append(history)
                offsets.append(offsets[-1] + history.size)
                storage["dense_fc_full_logits"].append(full[j])
                storage["dense_fc_coarse_candidate_ids"].append(ids[j])
                storage["dense_fc_candidate_exact_logits"].append(scores[j])
                storage["target_logits"].append(target[j])
            del checkpoint, prefix, full, full_again, target, ids, scores
            gc.collect()
            mx.clear_cache()
    if len(digests) != 36:
        raise RuntimeError("full capture did not produce all 36 rows")
    arrays = {
        "dense_fc_full_logits": np.asarray(storage["dense_fc_full_logits"], np.float32),
        "dense_fc_coarse_candidate_ids": np.asarray(storage["dense_fc_coarse_candidate_ids"], np.int32),
        "dense_fc_candidate_exact_logits": np.asarray(storage["dense_fc_candidate_exact_logits"], np.float32),
        "target_logits": np.asarray(storage["target_logits"], np.float32),
        "row_token_ids": np.concatenate(storage["row_token_ids"]),
        "row_token_offsets": np.asarray(offsets, np.int64),
        **{name: np.asarray(storage[name], np.int64) for name in (
            "context_length", "lane_id", "seed", "draft_position",
            "target_offset", "draft_offset")},
        **{name: np.asarray(digests, dtype="S64") for name in (
            "dense_fc_full_row_sha256", "dense_fc_coarse_row_sha256",
            "target_row_sha256")},
    }
    if any(not np.all(np.isfinite(arrays[name])) for name in (
            "dense_fc_full_logits", "dense_fc_candidate_exact_logits", "target_logits")):
        raise RuntimeError("nonfinite full capture logits")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".partial.npz")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, out)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {**admitted,
               "schema": "mlx2-coarse-head-densefc-full36-v1",
               "capture_kind": "matched_model",
               "capture_sha256": pilot.sha(out),
               "coarse_head_parameters_sha256": coarse_hash,
               "checkpoint_modes": checkpoint_modes,
               "mechanism_counts": {name: 36 for name in (
                   "dense_fc_full_head_steps", "dense_fc_coarse_head_steps",
                   "target_verify_steps", "committed_checkpoint_restore_checks")},
               "qualification": "capture_only",
               "limits": ["Direct model B1 rows, no route implementation or throughput qualification"]}
    out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=pilot.MODEL)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preflight-receipt", type=Path, required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--i-own-the-gpu", action="store_true")
    args = ap.parse_args(argv)
    current = preflight(args.model, args.out)
    if args.preflight_only:
        if args.preflight_receipt.exists():
            ap.error("refusing to overwrite preflight receipt")
        args.preflight_receipt.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_receipt.write_text(json.dumps(current, indent=2) + "\n")
        print(json.dumps({"admitted": True, "rows": current["rows"],
                          "estimated_capture_bytes": current["estimated_capture_bytes"]}))
        return
    saved = json.loads(args.preflight_receipt.read_text())
    stable = set(current) - {"disk_free_bytes"}
    if any(current[key] != saved.get(key) for key in stable):
        raise RuntimeError("CPU preflight no longer matches source, artifact, corpus or output")
    result = run_capture(args.model, args.out, current)
    print(json.dumps({"capture": str(args.out), "capture_sha256": result["capture_sha256"],
                      "rows": result["rows"]}))


if __name__ == "__main__":
    main()
