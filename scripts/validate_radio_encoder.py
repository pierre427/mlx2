#!/usr/bin/env python3
"""Compare real C-RADIOv2-B encoder outputs with NVIDIA's pinned CPU reference.

Reference extras (outside the production runtime): torch, timm==1.0.19,
open_clip_torch==2.32.0, einops, ftfy, wcwidth, transformers, huggingface_hub.
The reference uses reviewed remote code at an immutable revision.
"""

import argparse
import hashlib
import json
from pathlib import Path

REPO = "nvidia/C-RADIOv2-B"
REVISION = "cb109caa670ee8cc38ccf23b8ab0331fa487db0f"


def comparison_metrics(actual, reference, *, label):
    """Require finite evidence; equal zero vectors have cosine one by convention."""
    import numpy as np

    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if (
        a.shape != b.shape
        or a.ndim < 1
        or not a.size
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise AssertionError(f"{label}: shape or finite failure")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        a_norm, b_norm = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
        denominator = a_norm * b_norm
        cosine = np.divide(
            np.sum(a * b, axis=-1),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        both_zero = np.all(a == 0, axis=-1) & np.all(b == 0, axis=-1)
        cosine = np.where(both_zero, 1.0, cosine)
        reference_norm = np.linalg.norm(b)
        relative_l2 = (
            np.linalg.norm(a - b) / reference_norm
            if reference_norm > 0
            else 0.0
            if np.array_equal(a, b)
            else float("inf")
        )
        metrics = {
            "shape": list(a.shape),
            "cosine_min": float(cosine.min()),
            "cosine_mean": float(cosine.mean()),
            "max_abs_error": float(np.max(np.abs(a - b))),
            "relative_l2": float(relative_l2),
        }
    if not all(np.isfinite(value) for key, value in metrics.items() if key != "shape"):
        raise AssertionError(f"{label}: parity metrics are nonfinite or undefined")
    if metrics["cosine_min"] < 0.99999 or metrics["relative_l2"] > 0.0001:
        raise AssertionError(f"{label}: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import mlx.core as mx
    import numpy as np
    import torch
    from transformers import AutoModel

    from mlx2.adapters.radio import RadioImageAdapter

    mx.set_default_device(mx.cpu)
    torch.set_num_threads(4)
    adapter = RadioImageAdapter(args.model)
    reference = (
        AutoModel.from_pretrained(
            REPO,
            revision=REVISION,
            code_revision=REVISION,
            trust_remote_code=True,
        )
        .cpu()
        .eval()
    )
    # Bind both arms to exactly the same checkpoint, not just the same repo name.
    from safetensors.torch import load_file

    weights = load_file(str(Path(args.model) / "model.safetensors"))
    reference.load_state_dict(weights, strict=True)
    del weights
    cases = [
        ("black", np.zeros((1, 3, 32, 48), np.float32)),
        ("white", np.ones((1, 3, 48, 32), np.float32)),
    ]
    rng = np.random.default_rng(42)
    for name, shape in [
        ("rectangle", (1, 3, 64, 96)),
        ("batch", (2, 3, 32, 48)),
        ("processor_resolution", (1, 3, 432, 432)),
        ("default_resolution", (1, 3, 768, 768)),
    ]:
        cases.append((name, rng.random(shape, dtype=np.float32)))
    rows = []
    for name, pixels in cases:
        actual = adapter.encode(mx.array(pixels))
        with torch.inference_mode():
            expected = reference(torch.from_numpy(pixels))
        row = {"name": name, "input_shape": list(pixels.shape), "metrics": {}}
        for kind in ("summary", "features"):
            a = np.asarray(getattr(actual, kind)).astype(np.float64)
            b = getattr(expected, kind).float().numpy().astype(np.float64)
            row["metrics"][kind] = comparison_metrics(a, b, label=f"{name}/{kind}")
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "reference_repository": REPO,
        "reference_revision": REVISION,
        "device": "cpu",
        "passed": True,
        "zero_vector_convention": "equal zero vectors: cosine=1; all-zero exact pair: relative_l2=0",
        "cases": rows,
        "adapter_receipt": adapter.receipt,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path("src/mlx2/runtime/models/radio.py"),
                Path("src/mlx2/adapters/radio.py"),
                Path("src/mlx2/adapters/radio_config.py"),
            ]
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
