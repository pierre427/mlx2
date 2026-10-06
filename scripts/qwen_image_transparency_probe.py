#!/usr/bin/env python3
"""CPU-only contract probe for transparent Qwen Image instruction edits."""

from __future__ import annotations

import argparse
import inspect
import io
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image


def probe() -> dict:
    from mlx2.adapters.generative_media import QwenImage21Adapter

    # Exercise mlx2's real PNG encoder without inspecting or loading an artifact.
    adapter = object.__new__(QwenImage21Adapter)
    adapter.artifact = SimpleNamespace(fingerprint="cpu-transparency-probe")
    adapter._lora = None
    adapter._lora_epoch = 0
    adapter._verify_input_identity = lambda: None
    adapter._mark_lora_used = lambda: None
    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[..., :3] = (17, 29, 43)
    rgba[..., 3] = np.array([[0, 64, 128], [192, 255, 31]], dtype=np.uint8)
    encoded = adapter._encode(rgba)
    decoded = np.asarray(Image.open(io.BytesIO(encoded.data)).convert("RGBA"))
    public = inspect.signature(QwenImage21Adapter.edit_image).parameters
    return {
        "schema": "mlx2.qwen-image-transparency-probe.v1",
        "rgba_png_preserved": bool(np.array_equal(decoded, rgba)),
        "edit_api_accepts_transparent": "transparent" in public,
        "current_verdict": (
            "backend-output-ready"
            if "transparent" in public
            else "request-contract-gap: encoder preserves alpha but edit_image cannot request it"
        ),
        "source_candidate": "ddalcu/mlx-serve#647",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-request-support", action="store_true")
    args = parser.parse_args()
    report = probe()
    print(json.dumps(report, indent=2))
    if not report["rgba_png_preserved"]:
        return 2
    return int(
        args.require_request_support and not report["edit_api_accepts_transparent"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
