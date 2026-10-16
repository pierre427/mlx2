#!/usr/bin/env python3
"""Compare completed BF16/Q8 MOSS-Image probe outputs without model execution.

Run each arm in its own GPU ownership window with run_moss_image.py. This tool
then verifies matched settings, writes a side-by-side PNG, and records pixel
distance as descriptive evidence only; visual prompt adherence is the quality
oracle for deciding whether Q8 is sufficient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _load_arm(path, expected_kind):
    receipt = json.loads((path / "receipt.json").read_text())
    image_path = path / "image.png"
    route = receipt.get("route", {})
    if (
        receipt.get("success") is not True
        or route.get("artifact_kind") != expected_kind
        or route.get("observed_used") is not True
        or not image_path.is_file()
    ):
        raise ValueError(f"invalid completed {expected_kind} arm: {path}")
    raw = image_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != route.get("output_sha256"):
        raise ValueError(f"image bytes differ from the {expected_kind} receipt")
    image = Image.open(image_path).convert("RGB")
    return receipt, image


def compare(bf16, q8, output, quality_verdict="pending_visual_review"):
    bf16_receipt, bf16_image = _load_arm(bf16, "moss-image-sft-bf16")
    q8_receipt, q8_image = _load_arm(q8, "moss-image-sft-mlx-q8")
    if bf16_receipt["parameters"] != q8_receipt["parameters"]:
        raise ValueError("BF16 and Q8 parameters differ")
    if bf16_receipt.get("source_sha256") != q8_receipt.get("source_sha256"):
        raise ValueError("BF16 and Q8 adapter source identities differ")
    if bf16_image.size != q8_image.size:
        raise ValueError("BF16 and Q8 image sizes differ")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    a = np.asarray(bf16_image, dtype=np.float32)
    b = np.asarray(q8_image, dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    comparison = {
        "kind": "moss-image-q8-quality-ab",
        "parameters": bf16_receipt["parameters"],
        "bf16": {
            "artifact_fingerprint": bf16_receipt["route"]["artifact_fingerprint"],
            "image_sha256": bf16_receipt["route"]["output_sha256"],
            "generation_seconds": bf16_receipt.get("generation_seconds"),
            "peak_memory_bytes": bf16_receipt.get("mlx_peak_memory_bytes"),
        },
        "q8": {
            "artifact_fingerprint": q8_receipt["route"]["artifact_fingerprint"],
            "image_sha256": q8_receipt["route"]["output_sha256"],
            "generation_seconds": q8_receipt.get("generation_seconds"),
            "peak_memory_bytes": q8_receipt.get("mlx_peak_memory_bytes"),
        },
        "pixel_metrics_descriptive_only": {
            "mae_8bit": float(np.mean(np.abs(a - b))),
            "mse_8bit": mse,
            "psnr_db": None if mse == 0 else 20 * math.log10(255 / math.sqrt(mse)),
        },
        "quality_verdict": quality_verdict,
        "quality_scope": "one matched prompt, seed, resolution, and step count",
        "performance_verdict": "not_a_controlled_performance_run",
        "qualified": False,
        "selected": False,
    }
    label_height = 36
    canvas = Image.new(
        "RGB", (bf16_image.width * 2, bf16_image.height + label_height), "white"
    )
    canvas.paste(bf16_image, (0, label_height))
    canvas.paste(q8_image, (bf16_image.width, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "BF16", fill="black")
    draw.text((bf16_image.width + 12, 10), "MLX affine Q8", fill="black")
    temporary_image = output / "comparison.png.tmp"
    canvas.save(temporary_image, format="PNG")
    os.replace(temporary_image, output / "comparison.png")
    temporary_receipt = output / "comparison.json.tmp"
    temporary_receipt.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary_receipt, output / "comparison.json")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--q8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quality-verdict",
        choices=(
            "pending_visual_review",
            "visually_sufficient_for_this_prompt",
            "visually_insufficient_for_this_prompt",
        ),
        default="pending_visual_review",
    )
    print(json.dumps(compare(**vars(parser.parse_args())), indent=2, sort_keys=True))
