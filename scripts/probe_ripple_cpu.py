#!/usr/bin/env python3
"""Bounded CPU-only source-weight/metadata probe; never loads the full model."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def probe(model_path, output):
    # Select CPU before importing any tensor model implementation.
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from transformers import PreTrainedTokenizerFast

    from mlx2.adapters.registry import inspect_model
    from mlx2.adapters.ripple import REPO, REVISION
    from mlx2.conversion.nvfp4 import decode_rows, read_header
    from mlx2.runtime.models.modelopt_nvfp4 import ModelOptNVFP4Linear
    from mlx2.runtime.tokenizer_integrity import repair_loaded_tokenizer

    model_path = model_path.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()
    complete = json.loads((model_path / ".hf-download-complete.json").read_text())
    if complete.get("repo") != REPO or complete.get("revision") != REVISION:
        raise ValueError("the pinned, hash-verified download must complete before probing")
    resolved = inspect_model(model_path)
    artifact = resolved.artifact
    results = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        prefix = f"model.layers.0.mlp.{projection}"
        shard = model_path / artifact["weight_map"][prefix + ".weight"]
        header, _payload = read_header(shard)
        shape = header[prefix + ".weight"]["shape"]
        rows = sorted({0, 1, shape[0] // 3, shape[0] // 2, shape[0] - 2, shape[0] - 1})

        def read_rows(suffix, dtype, rows=None, prefix=prefix):
            name = prefix + suffix
            path = model_path / artifact["weight_map"][name]
            entries, start = read_header(path)
            entry = entries[name]
            with path.open("rb") as f:
                if rows is None:
                    a, b = entry["data_offsets"]
                    f.seek(start + a)
                    return np.frombuffer(f.read(b - a), dtype).reshape(entry["shape"])
                columns = entry["shape"][1]
                pieces = []
                for row in rows:
                    f.seek(start + entry["data_offsets"][0] + row * columns)
                    pieces.append(np.frombuffer(f.read(columns), dtype))
                return np.stack(pieces)

        packed = read_rows(".weight", np.uint8, rows)
        scales = read_rows(".weight_scale", np.uint8, rows)
        scalar = read_rows(".weight_scale_2", "<f4")
        input_scale = read_rows(".input_scale", "<f4")
        if not np.isfinite(input_scale).all() or input_scale.item() <= 0:
            raise ValueError("source input scale must be finite and positive")
        linear = ModelOptNVFP4Linear(shape[1] * 2, len(rows))
        linear.load_weights([
            ("weight", mx.array(packed.copy().view("<u4"))),
            ("scales", mx.array(scales)), ("global_scale", mx.array(scalar)),
        ], strict=True)
        x = np.random.default_rng(71).normal(size=(2, shape[1] * 2)).astype(np.float32)
        expected = x @ decode_rows(packed, scales, scalar).T
        actual = np.asarray(linear(mx.array(x)))
        error = float(np.max(np.abs(actual - expected)))
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
        results.append({"projection": prefix, "rows": rows, "input_width": shape[1] * 2,
                        "max_absolute_error": error, "tensor_scale": float(scalar),
                        "source_input_scale_validated_not_applied": float(input_scale),
                        "sample_sha256": hashlib.sha256(packed.tobytes() + scales.tobytes()
                                                         + scalar.tobytes()).hexdigest(),
                        "dispatch": linear.dispatch_status()})
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path, local_files_only=True)
    tokenizer_receipt = repair_loaded_tokenizer(tokenizer, model_path)
    result = {
        "schema": "mlx2.ripple-source-cpu-probe.v1", "source_revision": source_revision,
        "checkpoint": model_path.name + "@sha256:" + resolved.artifact_fingerprint,
        "hub_repo": REPO, "hub_revision": REVISION,
        "download_receipt_sha256": hashlib.sha256((model_path / ".hf-download-complete.json").read_bytes()).hexdigest(),
        "device": "cpu", "full_model_loaded": False, "gpu_executed": False,
        "qualified": False, "serving_observed_used": False,
        "tensor_headers_checked": len(artifact["weight_map"]),
        "tokenizer": tokenizer_receipt, "samples": results,
        "numerical_profile": "source NVFP4 weights with FP32 synthetic inputs; W4A16 product law",
        "limitations": ["18 sampled MLP output rows only; no model generation",
                        "Source W4A4 input quantization not implemented or compared",
                        "No Metal, APCv2 serving, long-context or performance claim"],
    }
    atomic_json(output / "result.json", result)
    print(json.dumps({"checkpoint": result["checkpoint"], "headers": len(artifact["weight_map"]),
                      "projections_passed": len(results), "qualified": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe(args.model, args.output)
