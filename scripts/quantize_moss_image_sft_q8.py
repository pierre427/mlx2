#!/usr/bin/env python3
"""Create a resumable native MLX affine-Q8 MOSS-Image artifact on CPU.

This conversion does not qualify Q8. It preserves the fp32 VAE and quantizes
the DiT and Qwen3.5 text linear weights at group size 64 for a later A/B run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _quantize_component(source, output, index_name, *, exclude=(), include_prefix=None):
    import mlx.core as mx

    index = json.loads((source / index_name).read_text())
    mapping = index["weight_map"]
    output.mkdir(parents=True, exist_ok=True)
    output_mapping = {}
    for shard in sorted(set(mapping.values())):
        destination = output / shard
        if destination.exists():
            raise ValueError(f"unverified existing Q8 shard: {destination}")
        tensors = mx.load(str(source / shard))
        packed = {}
        for name, value in tensors.items():
            if include_prefix is not None and not name.startswith(include_prefix):
                continue
            should_quantize = (
                value.ndim == 2
                and name.endswith(".weight")
                and value.shape[-1] % 64 == 0
                and not any(name.endswith(suffix) for suffix in exclude)
            )
            if should_quantize:
                weight, scales, biases = mx.quantize(
                    value, group_size=64, bits=8, mode="affine"
                )
                base = name.removesuffix(".weight")
                packed[name] = weight
                packed[base + ".scales"] = scales
                packed[base + ".biases"] = biases
                output_mapping[name] = shard
                output_mapping[base + ".scales"] = shard
                output_mapping[base + ".biases"] = shard
            else:
                packed[name] = value
                output_mapping[name] = shard
        mx.eval(*packed.values())
        temporary = destination.with_name(destination.stem + ".part.safetensors")
        mx.save_safetensors(str(temporary), packed)
        os.replace(temporary, destination)
        print(f"Q8 packed {destination.name}", flush=True)
    _atomic_json(
        output / index_name,
        {"metadata": index.get("metadata", {}), "weight_map": output_mapping},
    )


def convert(source: Path, output: Path):
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    from mlx2.adapters.moss_image import (
        MOSS_IMAGE_REPO,
        MOSS_IMAGE_REVISION,
        Q8,
        inspect_moss_image,
    )

    artifact = inspect_moss_image(source)
    if artifact.kind != "moss-image-sft-bf16":
        raise ValueError("Q8 source must be the verified BF16 MOSS-Image snapshot")
    if output.exists():
        raise ValueError("Q8 output must not already exist")
    staging = output.with_name(output.name + ".partial")
    if staging.exists():
        raise ValueError(f"unfinished conversion exists at {staging}")
    staging.mkdir(parents=True)
    try:
        for name in ("model_index.json", "LICENSE", "README.md"):
            shutil.copy2(source / name, staging / name)
        shutil.copytree(source / "tokenizer", staging / "tokenizer")
        shutil.copytree(source / "scheduler", staging / "scheduler")
        shutil.copytree(source / "vae", staging / "vae")
        shutil.copy2(
            source / "pipeline_moss_image.py", staging / "pipeline_moss_image.py"
        )

        _quantize_component(
            source / "transformer",
            staging / "transformer",
            "diffusion_pytorch_model.safetensors.index.json",
        )
        _quantize_component(
            source / "text_encoder",
            staging / "text_encoder",
            "model.safetensors.index.json",
            exclude=("embed_tokens.weight",),
            include_prefix="language_model.",
        )
        for component in ("transformer", "text_encoder"):
            config = json.loads((source / component / "config.json").read_text())
            config.update(quantization=Q8, mlx_format=True)
            _atomic_json(staging / component / "config.json", config)

        files = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                files[path.relative_to(staging).as_posix()] = {
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
        proof = {
            "source_repo": MOSS_IMAGE_REPO,
            "source_revision": MOSS_IMAGE_REVISION,
            "source_fingerprint": artifact.fingerprint,
            "quantization": Q8,
            "device": "cpu",
            "output_files": files,
            "execution_qualification": "pending",
            "ab_comparison": "pending",
        }
        _atomic_json(staging / "mlx2-conversion.json", proof)
        os.replace(staging, output)
        return proof
    except Exception:  # noqa: TRY203 - preserve the partial tree for diagnosis
        # Keep the distinguishable partial tree for diagnosis/resume work.
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = convert(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "files": len(result["output_files"]),
                "quantization": result["quantization"],
            },
            indent=2,
        )
    )
