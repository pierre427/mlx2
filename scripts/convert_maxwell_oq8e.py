#!/usr/bin/env python3
"""Convert Maxwell-V1-Flash BF16 to a calibrated oQ8e artifact (pinned offline oQ quantizer).

  convert_maxwell_oq8e.py --model SRC --output DST --evidence DIR --oq-source OQ [--dependency-path SITE]

Run under the GPU lock (wrap in /Users/Shared/mlxuag/gpuq-reference.sh).  The
oQ quantizer is used only offline, from an unmodified checkout at the
reviewed revision; serving uses mlx2's Maxwell adapter.  The calibration is
mlx2's own layer walk (src/mlx2/conversion/maxwell_oq.py): the quantizer has
no Maxwell walker.  Its budget, clipping, imatrix weighting, LSQ refinement
and shard writing run unmodified at level 8 (``enhanced``), with three
Maxwell-specific hooks:

* identity sanitizer (checkpoint names stay as stored; mlx2 packs experts at
  load), carrying a cast predicate that keeps the 43 F32 ``expert_bias``
  tensors F32 (their spread is below one bf16 step),
* a quantize predicate that keeps the router, mHC mappings/alphas, hc_head
  and the MTP fusion projections full precision (the same intent as oQ's own
  ``_is_mtp_protected_tensor``, which does not match Maxwell's names),
* the measured imatrix in place of the quantizer's collection pass.

Refuses an existing output or evidence directory; never deletes the source.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OQ_REVISION = "4213c3284abc7b3746a91464a7af7ccc3732b759"
LEVEL = 8


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--oq-source", type=Path, required=True)
    parser.add_argument("--dependency-path", type=Path)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    args = parser.parse_args()

    oq_revision = subprocess.check_output(["git", "-C", str(args.oq_source), "rev-parse", "HEAD"], text=True).strip()
    if oq_revision != OQ_REVISION:
        raise ValueError(f"oQ source revision {oq_revision} differs from the reviewed {OQ_REVISION}")
    subprocess.run(["git", "-C", str(args.oq_source), "diff", "--exit-code", "HEAD", "--"], check=True,
                   stdout=subprocess.DEVNULL)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.evidence.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.adapters.maxwell_moe import inspect_artifact

    source = inspect_artifact(args.model)
    if source["storage_format"] != "bf16" or not source["has_mtp"]:
        raise ValueError("oQ8e calibrates the original BF16 checkpoint with its MTP layer")
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    receipt = {"schema": "mlx2.maxwell-oq8e-conversion.v1", "mlx2_commit": rev, "dirty": dirty,
               "source_checkpoint": f"{args.model.name}@sha256:{source['identity']['fingerprint']}",
               "oq_revision": oq_revision, "level": LEVEL, "group_size": 64, "enhanced": True,
               "source_deleted": False, "qualified": False, "state": "calibrating"}
    atomic_json(args.evidence / "result.json", receipt)
    try:
        sys.path.insert(0, str(args.oq_source.resolve()))
        if args.dependency_path:
            sys.path.append(str(args.dependency_path.resolve()))
        import mlx.core as mx
        from omlx import oq
        from transformers import PreTrainedTokenizerFast

        from mlx2.conversion.maxwell_oq import (
            LAYER_WALK, Reader, expected_entries, keep_full_precision, stream_calibrate)

        if not oq.HAS_MLX:
            raise RuntimeError("pinned offline oQ dependencies are incomplete")
        receipt["software"] = {name: importlib.metadata.version(name)
                               for name in ("mlx", "mlx-lm", "numpy", "transformers")}
        receipt["device"] = mx.device_info()
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model, local_files_only=True)
        reader = Reader(source)
        started = time.monotonic()

        def progress(row):
            atomic_json(args.evidence / "progress.json", {"state": "calibrating", **row})
            print(json.dumps(row), flush=True)

        config = source["config"]
        entries, collection = stream_calibrate(reader, config, tokenizer, oq,
                                               micro_batch_size=args.micro_batch_size, progress=progress)
        want = expected_entries(config)
        if sorted(collection["sensitivity_map"]) != list(range(int(config["num_hidden_layers"]))):
            raise ValueError("measured sensitivity does not cover every trunk layer")
        if set(entries) != want:
            raise ValueError(f"calibration coverage differs: missing {sorted(want - set(entries))[:3]}, "
                             f"extra {sorted(set(entries) - want)[:3]}")
        zero = sum(int(e.counts[0]) == 0 for n, e in entries.items() if ".experts." in n)
        signature = oq._source_imatrix_signature(args.model, config, num_samples=128, seq_length=512,
                                                 calib_dataset=oq._OQE_CALIB_DATASET)
        metadata = {**signature, "layer_walk": LAYER_WALK, "collection": collection, "load_kind": "streaming",
                    "entry_count": len(entries), "processed_samples": 128, "zero_count_expert_entries": zero,
                    "source_checkpoint": receipt["source_checkpoint"], "mlx2_commit": rev,
                    "oq_revision": oq_revision}
        cache_path = args.evidence / "imatrix.npz"
        oq._save_oqe_imatrix(cache_path, entries, metadata)
        data = oq.OQImatrixData(entries, metadata, str(cache_path), reused=False)
        del reader
        gc.collect()
        mx.clear_cache()
        receipt.update(state="quantizing", calibration={k: v for k, v in metadata.items() if k != "collection"},
                       calibration_seconds=round(time.monotonic() - started, 1))
        atomic_json(args.evidence / "result.json", receipt)

        def calibrated(*_a, **_k):
            return data

        def identity_sanitizer(*_a, **_k):
            def sanitize(weights):
                return weights
            sanitize._omlx_cast_predicate = lambda name: not name.endswith("expert_bias")
            return sanitize

        original = oq.universal_quant_predicate

        def maxwell_predicate(path, module, config, oq_level=4):
            if keep_full_precision(path):
                return False
            return original(path, module, config, oq_level)

        with (patch.object(oq, "_load_or_collect_imatrix", calibrated),
              patch.object(oq, "_build_model_sanitizer", identity_sanitizer),
              patch.object(oq, "universal_quant_predicate", maxwell_predicate)):
            oq.quantize_oq_streaming(
                str(args.model), str(args.output), LEVEL, group_size=64, dtype="bfloat16", enhanced=True,
                imatrix_strict=True, imatrix_cache_path=str(cache_path), imatrix_num_samples=128,
                imatrix_seq_length=512, auto_proxy_sensitivity=False, stream_calibration=False,
                sensitivity_map_override=collection["sensitivity_map"],
                preserve_mtp=True,
                progress_callback=lambda phase, pct, detail="", meta=None: atomic_json(
                    args.evidence / "progress.json",
                    {"state": "quantizing", "phase": phase, "percent": pct, "detail": detail}))
        report = json.loads((args.output / "oq_imatrix_report.json").read_text())
        applied = set(report["applied"])
        if applied != want or report["missing"] or report["mismatched"]:
            raise ValueError(f"oQ8e imatrix coverage: applied {len(applied)}/{len(want)}, "
                             f"missing {report['missing'][:3]}, mismatched {report['mismatched'][:3]}")
        for name in ("LICENSE", "README.md", "chat_template.jinja", "generation_config.json"):
            if (args.model / name).is_file() and not (args.output / name).exists():
                shutil.copy2(args.model / name, args.output / name)
        (args.output / "conversion").mkdir()
        shutil.copy2(cache_path, args.output / "conversion" / "imatrix.npz")
        out = inspect_artifact(args.output)
        if out["storage_format"] != "affine-quantized" or not out["has_mtp"]:
            raise ValueError("converted artifact does not pass the Maxwell quantized gate")
        quant = out["quantization"]
        bits = {}
        for name in out["quantized_names"]:
            spec = quant.get(name)
            b = spec.get("bits", quant["bits"]) if isinstance(spec, dict) else quant["bits"]
            bits[b] = bits.get(b, 0) + 1
        size = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
        receipt.update(state="converted_pending_serving_validation",
                       output_checkpoint=f"{args.output.name}@sha256:{out['identity']['fingerprint']}",
                       quantized_modules=len(out["quantized_names"]), bits_histogram=bits, output_bytes=size,
                       imatrix_applied=len(applied), elapsed_seconds=round(time.monotonic() - started, 1),
                       limits=["serving validation and performance A/B are separate runs",
                               "lossy affine conversion; no BF16 token parity claim"])
        atomic_json(args.evidence / "result.json", receipt)
        atomic_json(args.output / "mlx2-conversion.json", receipt)
        print(json.dumps({k: receipt[k] for k in ("state", "output_checkpoint", "bits_histogram",
                                                  "output_bytes", "elapsed_seconds")}), flush=True)
    except BaseException as exc:
        receipt.update(state="failed", error=repr(exc), output_exists=args.output.exists())
        atomic_json(args.evidence / "result.json", receipt)
        raise


if __name__ == "__main__":
    main()
