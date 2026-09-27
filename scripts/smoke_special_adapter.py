#!/usr/bin/env python3
"""Bounded direct smoke for non-ordinary denoising and draft candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
IMAGE_PROMPT = "What object is shown in this image? Answer briefly."


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("llada", "diffusiongemma", "dspark"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-substring")
    parser.add_argument("--cpg-lease", required=True)
    args = parser.parse_args()
    if args.family == "dspark" and args.draft is None:
        parser.error("--draft is required for dspark")
    if args.family != "llada" and args.image is None:
        parser.error("--image is required for image-conditioned candidates")
    model = args.model.expanduser().resolve()
    row = {
        "schema": "mlx2.special-adapter-smoke.v1", "status": "failed",
        "family": args.family, "model_path": str(model),
        "source_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_at": time.time(), "cpg_lease": args.cpg_lease,
    }
    try:
        image = args.image.expanduser().resolve() if args.image else None
        if image is not None:
            row["image_path"] = str(image)
            row["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
        if args.family == "llada":
            from mlx2.adapters.llada import LLaDADenoisingAdapter

            adapter = LLaDADenoisingAdapter(str(model))
            try:
                result = adapter.generate(
                    messages=[{"role": "user", "content": "Say hello in one sentence."}],
                    gen_length=32, block_length=32, steps=32,
                )
            finally:
                adapter.close()
            row["prompt"] = "Say hello in one sentence."
            row["generation_geometry"] = {"gen_length": 32, "block_length": 32, "steps": 32}
        elif args.family == "diffusiongemma":
            from mlx2.adapters.diffusion_gemma import DiffusionGemmaAdapter

            adapter = DiffusionGemmaAdapter(model)
            try:
                result = asdict(adapter.generate_text(
                    IMAGE_PROMPT, image=image, max_tokens=32, max_denoising_steps=16,
                ))
            finally:
                adapter.close()
            row["prompt"] = IMAGE_PROMPT
            row["generation_geometry"] = {"max_tokens": 32, "max_denoising_steps": 16}
        else:
            from mlx2.adapters.lfm25_vl import generate_candidate_dspark

            draft = args.draft.expanduser().resolve()
            row["draft_path"] = str(draft)
            result = generate_candidate_dspark(
                model, draft, IMAGE_PROMPT, image=image, max_tokens=16,
            )
            row["prompt"] = IMAGE_PROMPT
            row["generation_geometry"] = {"max_tokens": 16, "draft_block_size": 10}
        if is_dataclass(result):
            result = asdict(result)
        row["result"] = result
        output = result.get("text", "")
        if not isinstance(output, str) or not output.strip():
            raise ValueError("candidate produced no visible output")
        if args.expected_substring and args.expected_substring.lower() not in output.lower():
            raise ValueError("candidate output missed expected image object")
        row["status"] = "smoke_passed_output_only" if args.family == "dspark" else "smoke_passed"
        return 0
    except Exception as error:
        row["error_type"] = type(error).__name__
        row["error"] = str(error)
        row["traceback"] = traceback.format_exc(limit=8)
        return 1
    finally:
        row["finished_at"] = time.time()
        row["elapsed_seconds"] = row["finished_at"] - row["started_at"]
        _write(args.out, row)


if __name__ == "__main__":
    raise SystemExit(run())
