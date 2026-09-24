#!/usr/bin/env python3
"""Repack the CPU-converted GGUF transformer as MLX affine 4-bit weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantize(source: Path, output: Path) -> dict:
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    proof = json.loads((source / "mlx2-conversion.json").read_text())
    if (
        proof.get("tensor_count") != 297
        or proof.get("output_dtype") != "bfloat16"
        or proof.get("source_revision") != "40319fb15542f0ad22921e0124a191a8a935a60a"
        or proof.get("source_sha256") != "e79c8a009f2ecbdb6c70fd663d9aea9ee304a0d91f347e4169a756b8ad141b41"
    ):
        raise ValueError("source is not the verified BF16 GGUF conversion")
    output.mkdir(parents=True, exist_ok=True)
    transformer = output / "transformer"
    transformer.mkdir(exist_ok=True)
    config = json.loads((source / "transformer" / "config.json").read_text())
    config["quantization"] = {"group_size": 64, "bits": 4, "mode": "affine"}
    (transformer / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    # Base encoder/VAE linking later changes the receipt; bind to immutable
    # transformer source data so an already-complete 4-bit conversion remains
    # valid after that step.
    input_fingerprint = hashlib.sha256(json.dumps({
        key: proof[key] for key in ("source_sha256", "layout_sha256", "output_files")
    }, sort_keys=True).encode()).hexdigest()
    completed = output / "mlx2-conversion.json"
    if completed.is_file():
        result = json.loads(completed.read_text())
        if result.get("bf16_source_fingerprint") != input_fingerprint:
            raise ValueError("existing 4-bit conversion belongs to another BF16 source")
        for name, record in result["output_files"].items():
            item = transformer / name
            if item.stat().st_size != record["size"] or _sha256(item) != record["sha256"]:
                raise ValueError(f"existing 4-bit shard changed: {name}")
        return result
    progress_path = output / ".mlx2-quantization-progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "bf16_source_fingerprint": input_fingerprint, "output_files": {},
    }
    if progress["bf16_source_fingerprint"] != input_fingerprint:
        raise ValueError("existing 4-bit progress belongs to another BF16 source")
    records = {}
    for name, record in proof["output_files"].items():
        original = source / "transformer" / name
        if original.stat().st_size != record["size"] or _sha256(original) != record["sha256"]:
            raise ValueError(f"BF16 source shard changed: {name}")
        target = transformer / name
        if target.exists():
            prior = progress["output_files"].get(name)
            if not prior or target.stat().st_size != prior["size"] or _sha256(target) != prior["sha256"]:
                raise ValueError(f"unverified existing 4-bit shard: {target}")
            records[name] = prior
            continue
        tensors = mx.load(str(original))
        packed = {}
        for key, value in tensors.items():
            if value.ndim == 2:
                if not key.endswith(".weight") or value.shape[-1] % 64:
                    raise ValueError(f"unsupported MLX affine layer: {key}")
                weight, scales, biases = mx.quantize(value, group_size=64, bits=4, mode="affine")
                packed[key] = weight
                packed[key.removesuffix(".weight") + ".scales"] = scales
                packed[key.removesuffix(".weight") + ".biases"] = biases
            else:
                packed[key] = value
        mx.eval(*packed.values())
        temporary = target.with_name(target.stem + ".part.safetensors")
        temporary.unlink(missing_ok=True)
        mx.save_safetensors(str(temporary), packed)
        os.replace(temporary, target)
        records[name] = {"size": target.stat().st_size, "sha256": _sha256(target)}
        progress["output_files"][name] = records[name]
        progress_path.write_text(json.dumps(progress, indent=2) + "\n")
        print(f"CPU quantized {name}: {len(tensors)} source tensors", flush=True)
        del tensors, packed
    result = dict(proof)
    result.update({
        "base_repo": None,
        "base_revision": None,
        "output_dtype": "mlx-affine-4bit",
        "quantization": config["quantization"],
        "bf16_source_fingerprint": input_fingerprint,
        "output_files": records,
        "execution_qualification": "pending",
    })
    (output / "mlx2-conversion.json").write_text(json.dumps(result, indent=2) + "\n")
    progress_path.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = quantize(args.source, args.output)
    print(f"quantized {len(result['output_files'])} shards on CPU")


if __name__ == "__main__":
    main()
