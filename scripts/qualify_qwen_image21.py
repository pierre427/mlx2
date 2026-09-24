#!/usr/bin/env python3
"""Run one source-bound Qwen-Image-2.1 GPU generation or edit smoke."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import sys
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mlx2.adapters.generative_media import QWEN_BACKEND_REVISION, QwenImage21Adapter

    target = args.output.expanduser().resolve()
    if target.suffix.lower() != ".png" or target.exists():
        raise ValueError("output must be a new PNG path")
    target.parent.mkdir(parents=True, exist_ok=True)
    args.gpu_lock.parent.mkdir(parents=True, exist_ok=True)
    with args.gpu_lock.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        import mlx.core as mx
        import numpy as np
        from PIL import Image

        mx.set_default_device(mx.gpu)
        adapter = QwenImage21Adapter(args.model)
        edit = args.edit_image is not None
        started = time.perf_counter()
        backend = adapter._model(edit=edit)
        load_seconds = time.perf_counter() - started
        if edit:
            adapter._editor = backend
            started = time.perf_counter()
            result = adapter.edit_image(
                args.prompt, [args.edit_image], width=args.width, height=args.height,
                steps=args.steps, seed=args.seed,
            )
        else:
            adapter._generator = backend
            started = time.perf_counter()
            result = adapter.generate_image(
                args.prompt, width=args.width, height=args.height,
                steps=args.steps, seed=args.seed,
            )
        generate_seconds = time.perf_counter() - started
        target.write_bytes(result.data)
        with Image.open(target) as image:
            image.verify()
        with Image.open(target) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        receipt = {
            "model_path": str(adapter.artifact.path),
            "model_kind": adapter.artifact.kind,
            "model_revision": adapter.artifact.source_revision,
            "model_fingerprint": adapter.artifact.fingerprint,
            "backend_revision": QWEN_BACKEND_REVISION,
            "adapter_source_sha256": _sha256(Path(__file__).resolve().parents[1] / "src/mlx2/adapters/generative_media.py"),
            "operation": "edit" if edit else "generate",
            "prompt": args.prompt,
            "reference_image": str(args.edit_image) if edit else None,
            "width": result.width,
            "height": result.height,
            "steps": args.steps,
            "seed": args.seed,
            "device": str(mx.default_device()),
            "load_seconds": round(load_seconds, 3),
            "generate_seconds": round(generate_seconds, 3),
            "mlx_peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
            "pixel_std": round(float(pixels.std()), 3),
            "output_path": str(target),
            "output_bytes": target.stat().st_size,
            "output_sha256": _sha256(target),
        }
        target.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--edit-image", type=Path)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu-lock", type=Path, default=Path("/tmp/gpu.lock"))
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
