#!/usr/bin/env python3
"""Run one bounded, image-conditioned direct adapter candidate smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROMPT = "What object is shown in this image? Answer briefly."


def _write(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(row, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("qwen35", "muse", "phi4mm"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--expected-substring", default="teapot")
    parser.add_argument("--cpg-lease", required=True)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 32:
        parser.error("max-tokens must be 1..32")
    model = args.model.expanduser().resolve()
    image = args.image.expanduser().resolve()
    row = {
        "schema": "mlx2.direct-vision-smoke.v1", "status": "failed",
        "family": args.family, "model_path": str(model), "image_path": str(image),
        "prompt": PROMPT, "max_tokens": args.max_tokens,
        "expected_substring": args.expected_substring,
        "cpg_lease": args.cpg_lease,
        "source_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_at": time.time(),
    }
    try:
        if not image.is_file():
            raise ValueError("image input is missing")
        row["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
        row["mlx_version"] = importlib.metadata.version("mlx")
        if args.family == "qwen35":
            from mlx2.adapters.qwen35_vlm_candidate import Qwen35VLMAdapter as Adapter
            from mlx2.adapters.qwen35_vlm_candidate import SOURCE_REVISION
        elif args.family == "muse":
            from mlx2.adapters.muse_glimmer_vision_candidate import MuseGlimmerVisionCandidate as Adapter
            from mlx2.adapters.muse_glimmer_vision_candidate import SOURCE_REVISION
        else:
            from mlx2.adapters.phi4mm_candidate import Phi4MMCandidate as Adapter
            from mlx2.adapters.phi4mm_candidate import SOURCE_REVISION
        row["backend_revision"] = SOURCE_REVISION
        started = time.perf_counter()
        adapter = Adapter(model)
        try:
            if args.family == "phi4mm":
                result = adapter.generate(PROMPT, images=[image], max_tokens=args.max_tokens)
            else:
                result = adapter.generate_response(PROMPT, images=[image], max_tokens=args.max_tokens)
            row["artifact_fingerprint"] = adapter.artifact.get("fingerprint")
            row["result"] = result
            text = result.get("text", "")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("direct image output is empty")
            if args.expected_substring.lower() not in text.lower():
                raise ValueError("direct image output did not identify the known image")
            row["status"] = "smoke_passed"
            return 0
        finally:
            close = getattr(adapter, "close", None)
            if callable(close):
                close()
            row["model_elapsed_seconds"] = time.perf_counter() - started
    except Exception as error:
        row["error_type"] = type(error).__name__
        row["error"] = str(error)
        row["traceback"] = traceback.format_exc(limit=8)
        return 1
    finally:
        row["finished_at"] = time.time()
        _write(args.out, row)


if __name__ == "__main__":
    raise SystemExit(run())
