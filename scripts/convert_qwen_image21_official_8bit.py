#!/usr/bin/env python3
"""Convert the pinned official Qwen-Image-2.1 snapshot to native MLX 8-bit.

Uses the audited local mlx-vlm image converter for transformer, text encoder,
and VAE layout. The output is staged until its files have been hashed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(source: Path, output: Path, *, gpu_lock: Path) -> dict:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mlx2.adapters.generative_media import (
        QWEN_BACKEND_REVISION,
        QWEN_REVISION,
        _verify_qwen_backend_revision,
        inspect_qwen_image21,
    )

    artifact = inspect_qwen_image21(source)
    if artifact.kind != "qwen-image-2.1" or artifact.source_revision != QWEN_REVISION:
        raise ValueError("source is not the pinned official Qwen-Image-2.1 snapshot")
    _verify_qwen_backend_revision()
    if output.exists():
        existing = inspect_qwen_image21(output)
        if existing.kind != "qwen-image-2.1-official-mlx-8bit":
            raise ValueError("output belongs to another conversion")
        return json.loads((output / "mlx2-official-conversion.json").read_text())
    staging = output.with_name(output.name + ".partial")
    if staging.exists():
        raise ValueError(f"unfinished conversion at {staging}; inspect it before retrying")
    staging.parent.mkdir(parents=True, exist_ok=True)
    gpu_lock.parent.mkdir(parents=True, exist_ok=True)
    with gpu_lock.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        import mlx.core as mx
        from mlx_vlm.models.qwen_image.convert import convert as backend_convert

        mx.set_default_device(mx.gpu)
        print(f"Converting Qwen-Image-2.1 to MLX 8-bit on {mx.default_device()}", flush=True)
        backend_convert(str(source), staging, bits=8, group_size=64, mode="affine")
        quant = {"group_size": 64, "bits": 8, "mode": "affine"}
        for component in ("transformer", "text_encoder"):
            config = json.loads((staging / component / "config.json").read_text())
            if config.get("quantization") != quant or config.get("mlx_format") is not True:
                raise ValueError(f"unexpected {component} conversion config")
        files = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                files[path.relative_to(staging).as_posix()] = {
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
        if not any(name.startswith("transformer/") and name.endswith(".safetensors") for name in files):
            raise ValueError("converted transformer weights are missing")
        if not any(name.startswith("text_encoder/") and name.endswith(".safetensors") for name in files):
            raise ValueError("converted text encoder weights are missing")
        proof = {
            "source_repo": "Qwen/Qwen-Image-2.1",
            "source_revision": QWEN_REVISION,
            "source_manifest_fingerprint": artifact.fingerprint,
            "backend_revision": QWEN_BACKEND_REVISION,
            "quantization": quant,
            "device": "gpu",
            "output_files": files,
            "execution_qualification": "pending",
        }
        (staging / "mlx2-official-conversion.json").write_text(json.dumps(proof, indent=2) + "\n")
        os.replace(staging, output)
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-lock", type=Path, default=Path("/tmp/gpu.lock"))
    args = parser.parse_args()
    result = convert(args.source, args.output, gpu_lock=args.gpu_lock)
    print(f"8-bit MLX conversion complete: {len(result['output_files'])} files", flush=True)


if __name__ == "__main__":
    main()
