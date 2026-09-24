#!/usr/bin/env python3
"""Run one source-bound direct LTX-2.5 GPU video smoke."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
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
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from mlx2.adapters.generative_media import (
        LTX_CONVERTER_SHA256,
        LTX_RUNTIME_REVISION,
        LTX25Adapter,
    )

    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".mp4" or output.exists():
        raise ValueError("output must be a new MP4 path")
    output.parent.mkdir(parents=True, exist_ok=True)
    args.gpu_lock.parent.mkdir(parents=True, exist_ok=True)
    with args.gpu_lock.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.perf_counter()
        adapter = LTX25Adapter(source=args.source, mlx_model=args.model, runtime_root=args.runtime)
        verify_seconds = time.perf_counter() - started
        conversion = json.loads((adapter.mlx_model / ".mlx2-cpu-conversion.json").read_text())
        conversion_fingerprint = hashlib.sha256(json.dumps(conversion, sort_keys=True).encode()).hexdigest()
        started = time.perf_counter()
        result = adapter.generate_video(
            args.prompt, output=output, width=args.width, height=args.height,
            frames=args.frames, frame_rate=args.frame_rate, seed=args.seed,
            timeout_seconds=args.timeout_seconds,
        )
        generate_seconds = time.perf_counter() - started
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,width,height,nb_read_frames", "-of", "json", str(result.path)],
            capture_output=True, text=True, check=True, timeout=30,
        )
        streams = json.loads(probe.stdout).get("streams", [])
        if len(streams) != 1:
            raise ValueError("generated video has no video stream")
        video = streams[0]
        if (video.get("width"), video.get("height")) != (args.width, args.height):
            raise ValueError("generated video dimensions differ from request")
        if int(video.get("nb_read_frames", -1)) != args.frames:
            raise ValueError("generated video frame count differs from request")
        receipt = {
            "source_path": str(adapter.artifact.path),
            "source_revision": adapter.artifact.source_revision,
            "source_fingerprint": adapter.artifact.fingerprint,
            "model_path": str(adapter.mlx_model),
            "conversion_fingerprint": conversion_fingerprint,
            "runtime_revision": LTX_RUNTIME_REVISION,
            "converter_sha256": LTX_CONVERTER_SHA256,
            "adapter_source_sha256": _sha256(root / "src/mlx2/adapters/generative_media.py"),
            "prompt": args.prompt,
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "frame_rate": args.frame_rate,
            "seed": args.seed,
            "artifact_verify_seconds": round(verify_seconds, 3),
            "generate_seconds": round(generate_seconds, 3),
            "video_codec": video["codec_name"],
            "output_path": str(output),
            "output_bytes": output.stat().st_size,
            "output_sha256": _sha256(output),
        }
        output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--frame-rate", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--gpu-lock", type=Path, default=Path("/tmp/gpu.lock"))
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
