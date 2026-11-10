#!/usr/bin/env python3
"""Convert verified Ripple BF16 to calibrated oQ4e under paired GPU ownership.

Requires an unmodified checkout of the pinned oMLX quantizer. It is used only
offline; model math uses mlx2's Ripple adapter. Validate the resulting
artifact separately with scripts/probe_ripple_serving.py before source cleanup.
This command never deletes source weights. Verify output hashes separately.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def checkout_revision():
    """The commit of THIS checkout, whatever the caller's working directory."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--oq-source", type=Path, required=True)
    parser.add_argument("--dependency-path", type=Path)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--cpg-task", required=True)
    parser.add_argument("--cpg-generation", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    args = parser.parse_args()
    from qualification_gpu_ownership import require_qualification_lease

    ownership = require_qualification_lease(task_id=args.cpg_task,
        generation=args.cpg_generation, cpg_owner_lock=args.cpg_owner_lock)
    # All lease checks and source version checks precede Metal allocation.
    from mlx2.adapters.ripple import BF16_REPO, BF16_REVISION, inspect_artifact

    oq_revision = subprocess.check_output(["git", "-C", str(args.oq_source), "rev-parse", "HEAD"], text=True).strip()
    if oq_revision != "4213c3284abc7b3746a91464a7af7ccc3732b759":
        raise ValueError("oQ source revision differs from reviewed provenance")
    subprocess.run(["git", "-C", str(args.oq_source), "diff", "--exit-code", "HEAD", "--"], check=True,
                   stdout=subprocess.DEVNULL)
    complete = json.loads((args.model / ".hf-download-complete.json").read_text())
    if complete.get("repo") != BF16_REPO or complete.get("revision") != BF16_REVISION:
        raise ValueError("BF16 download must finish and verify before conversion")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.evidence.mkdir(parents=True, exist_ok=False)
    source_revision = checkout_revision()
    source_artifact = inspect_artifact(args.model)
    if source_artifact["storage_format"] != "bf16":
        raise ValueError("only the original BF16 source may be calibrated")
    receipt = {"schema": "mlx2.ripple-oq4e-conversion.v1", "source_revision": source_revision,
               "source_checkpoint": args.model.name + "@sha256:" + source_artifact["identity"]["fingerprint"],
               "hub_repo": BF16_REPO, "hub_revision": BF16_REVISION,
               "oq_revision": oq_revision, "ownership": ownership,
               "source_deleted": False, "qualified": False,
               "download_receipt": complete, "state": "calibrating"}
    atomic_json(args.evidence / "result.json", receipt)
    try:
        sys.path.insert(0, str(args.oq_source.resolve()))
        if args.dependency_path:
            # Only the offline helper needs these installed local mlx-lm/vlm
            # dependencies; mlx2 production imports remain independent of them.
            sys.path.append(str(args.dependency_path.resolve()))
        import mlx.core as mx
        from omlx import oq
        from transformers import PreTrainedTokenizerFast

        from mlx2.conversion.ripple_oq import LAYER_WALK, BF16Reader, stream_calibrate
        from mlx2.runtime.tokenizer_integrity import repair_loaded_tokenizer

        if not oq.HAS_MLX:
            raise RuntimeError("pinned offline oQ dependencies are incomplete")
        receipt["software"] = {name: importlib.metadata.version(name)
                               for name in ("mlx", "mlx-lm", "mlx-vlm", "numpy", "transformers")}
        receipt["device"] = mx.device_info()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model, local_files_only=True)
        receipt["tokenizer"] = repair_loaded_tokenizer(tokenizer, args.model)
        reader = BF16Reader(source_artifact)
        started = time.monotonic()

        def progress(row):
            atomic_json(args.evidence / "progress.json", {"state": "calibrating", **row})
            print(json.dumps(row), flush=True)

        entries, collection = stream_calibrate(reader, source_artifact["config"], tokenizer, oq,
            evidence=args.evidence, micro_batch_size=args.micro_batch_size, progress=progress)
        if len(entries) != 641 or len(collection["sensitivity_map"]) != 80:
            raise ValueError("incomplete calibrated Ripple projection coverage")
        signature = oq._source_imatrix_signature(args.model, source_artifact["config"],
            num_samples=128, seq_length=512, calib_dataset=oq._OQE_CALIB_DATASET)
        metadata = {**signature, "layer_walk": LAYER_WALK, "collection": collection,
                    "load_kind": "streaming", "entry_count": len(entries), "processed_samples": 128,
                    "source_checkpoint": receipt["source_checkpoint"], "source_revision": source_revision,
                    "oq_revision": oq_revision}
        cache_path = args.evidence / "imatrix.npz"
        oq._save_oqe_imatrix(cache_path, entries, metadata)
        data = oq.OQImatrixData(entries, metadata, str(cache_path), reused=False)
        del reader
        gc.collect()
        mx.clear_cache()
        receipt.update(state="quantizing", calibration=metadata)
        atomic_json(args.evidence / "result.json", receipt)

        def calibrated(*_args, **_kwargs):
            return data

        def identity_sanitizer(*_args, **_kwargs):
            return lambda weights: weights

        # The original quantizer has no Ricursive layer walker. Supply the
        # measured result above and its identity name mapping; leave budget,
        # clipping, imatrix weighting, LSQ and shard writing unmodified.
        with (patch.object(oq, "_load_or_collect_imatrix", calibrated),
              patch.object(oq, "_build_model_sanitizer", identity_sanitizer)):
            oq.quantize_oq_streaming(str(args.model), str(args.output), 4,
                group_size=64, dtype="bfloat16", enhanced=True, imatrix_strict=True,
                imatrix_cache_path=str(cache_path), imatrix_num_samples=128, imatrix_seq_length=512,
                sensitivity_map_override=collection["sensitivity_map"],
                auto_proxy_sensitivity=False, stream_calibration=False,
                progress_callback=lambda phase, pct, detail="", meta=None: atomic_json(args.evidence / "progress.json",
                    {"state": "quantizing", "phase": phase, "percent": pct, "detail": detail, "metadata": meta}))
        report = json.loads((args.output / "oq_imatrix_report.json").read_text())
        if len(report["applied"]) != 641 or report["missing"] or report["mismatched"]:
            raise ValueError("oQ4e did not apply complete measured imatrix coverage")
        shutil.copy2(args.model / "LICENSE", args.output / "LICENSE")
        calibration_dir = args.output / "conversion"
        calibration_dir.mkdir()
        for name in ("imatrix.npz", "imatrix.npz.json", "bf16-reference.json", "bf16-reference-logits.npz"):
            shutil.copy2(args.evidence / name, calibration_dir / name)
        output_artifact = inspect_artifact(args.output)
        receipt.update(state="converted_pending_serving_validation",
            output_checkpoint=args.output.name + "@sha256:" + output_artifact["identity"]["fingerprint"],
            imatrix_report=report, elapsed_seconds=time.monotonic() - started,
            limits=["Output strict load and serving validation require a separate artifact ownership window",
                    "No performance qualification", "Lossy affine conversion; no BF16 bit or token parity claim"])
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        atomic_json(args.evidence / "result.json", receipt)
        atomic_json(args.output / "mlx2-conversion.json", {k: v for k, v in receipt.items() if k != "ownership"})
        print(json.dumps({"state": receipt["state"], "output": str(args.output),
                          "output_checkpoint": receipt["output_checkpoint"]}), flush=True)
    except BaseException as exc:
        # Leave no "calibrating"/"quantizing" record that reads as a run in
        # progress. A partial --output is kept for inspection, never deleted.
        receipt.update(state="failed", error=repr(exc), output_exists=args.output.exists())
        atomic_json(args.evidence / "result.json", receipt)
        raise


if __name__ == "__main__":
    main()
