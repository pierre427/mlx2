#!/usr/bin/env python3
"""Exact-artifact Gemma 4 mlx2 candidate smoke under the shared GPU lease."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = Path("/Users/Shared/mlxuag/gpu.lock/owner.json")
sys.path.insert(0, str(ROOT / "src"))


def _owner():
    owner = json.loads(LOCK.read_text())
    if owner.get("pid") != os.getppid() or not owner.get("cpg_generation"):
        raise RuntimeError("Gemma 4 smoke requires cpg_job.py GPU lease and lock")
    return owner


def _image_url():
    from PIL import Image

    image = Image.new("RGB", (32, 32), (230, 20, 20))
    stream = BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def _video_url():
    import cv2
    import numpy as np

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "two-frames.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (32, 32))
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create candidate video")
        try:
            for value in (20, 220):
                writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
        finally:
            writer.release()
        return "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode()


def _run_prompt(adapter, request, *, expect_media=False):
    import mlx.core as mx

    prepared = adapter.prepare_multimodal_request(request) if expect_media else request
    ids = adapter.prompt_tokens(prepared)
    if not ids:
        raise RuntimeError("Prepared Gemma 4 prompt has no tokens")
    inputs = prepared.get("_mlx2_prefill_inputs") or {}
    cache = adapter.model.make_cache()
    logits = adapter.model(mx.array([ids], dtype=mx.int32), cache=cache, **inputs)
    mx.eval(logits)
    if tuple(logits.shape[:2]) != (1, len(ids)):
        raise RuntimeError(f"Unexpected Gemma 4 logits shape: {logits.shape}")
    token = int(mx.argmax(logits[0, -1]).item())
    next_logits = adapter.model(mx.array([[token]], dtype=mx.int32), cache=cache)
    mx.eval(next_logits)
    return {
        "prompt_tokens": len(ids),
        "next_token": token,
        "cache_layers": len(cache),
        "media_token_end": prepared.get("_mlx2_media_token_end"),
        "media_fingerprint": prepared.get("_mlx2_media_fingerprint"),
        "decode_shape": list(next_logits.shape),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    owner = _owner()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MLX_ENABLE_TF32"] = "0"
    import mlx.core as mx

    from mlx2.adapters.registry import inspect_model

    mx.set_memory_limit(110 * 1024**3)
    mx.set_cache_limit(2 * 1024**3)
    resolution = inspect_model(args.model_path)
    adapter = resolution.adapter_type(str(args.model_path))
    source_hash = hashlib.sha256((ROOT / "src/mlx2/adapters/gemma4.py").read_bytes()).hexdigest()
    try:
        text = _run_prompt(adapter, {"prompt": "The capital of France is"})
        if args.text_only:
            receipt = {
                "timestamp": datetime.now(UTC).isoformat(),
                "model_path": str(args.model_path.resolve()),
                "adapter": resolution.adapter_type.__name__,
                "artifact": {k: resolution.artifact[k] for k in
                             ("fingerprint", "variant", "precision", "max_context")},
                "descriptor": resolution.descriptor.key,
                "adapter_source_sha256": source_hash,
                "gpu_owner": owner,
                "text": text,
                "scope": "candidate BF16 direct-adapter text prefill/decode only",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(receipt, indent=2), flush=True)
            return
        image_request = {"messages": [{"role": "user", "content": [
            {"type": "input_image", "image_url": _image_url()},
            {"type": "text", "text": "Describe the color."},
        ]}]}
        image = _run_prompt(adapter, image_request, expect_media=True)
        image_repeat = _run_prompt(adapter, image_request, expect_media=True)
        video_request = {"messages": [{"role": "user", "content": [
            {"type": "input_video", "video_url": _video_url()},
            {"type": "text", "text": "Describe the clip."},
        ]}]}
        video = _run_prompt(adapter, video_request, expect_media=True)
        if not (image["media_token_end"] and video["media_token_end"]):
            raise RuntimeError("Gemma 4 media placeholder boundary was not found")
        cache_stats = adapter.media_feature_cache.snapshot()
        if cache_stats.get("hits", 0) < 1:
            raise RuntimeError("Repeated image request did not reuse projected features")
        receipt = {
            "timestamp": datetime.now(UTC).isoformat(),
            "model_path": str(args.model_path.resolve()),
            "adapter": resolution.adapter_type.__name__,
            "artifact": {k: resolution.artifact[k] for k in (
                "fingerprint", "variant", "precision", "full_attention_layers",
                "sliding_attention_layers", "max_context",
            )},
            "descriptor": resolution.descriptor.key,
            "adapter_source_sha256": source_hash,
            "gpu_owner": owner,
            "text": text,
            "image": image,
            "image_repeat": image_repeat,
            "video": video,
            "feature_cache": cache_stats,
            "scope": "candidate direct-adapter prefill/decode only",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2), flush=True)
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
