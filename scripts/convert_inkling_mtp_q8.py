#!/usr/bin/env python3
"""Convert the Inkling-Small native MTP head to an affine q8 sidecar.

Source: thinkingmachines/Inkling-Small ``mtp.safetensors`` (BF16, 8 layers).
The eight dense projections per layer become affine 8-bit, group 64 (MLX
``QuantizedLinear`` layout: ``.weight`` packed uint32, ``.scales``, ``.biases``);
norms, short-convolution weights, the relative-logit projection and the MLP
global scale stay BF16.  CPU only; refuses a nonempty output directory and a
source whose SHA-256 differs from the pinned Hub LFS object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

SOURCE_REPO = "thinkingmachines/Inkling-Small"
SOURCE_REVISION = "b2d4f225a02032c5d154bff748ab5a00c5ca26e4"
SOURCE_SHA256 = "d286dd21cb982a0052d24ee0077ec6fc38f5a766dc6953e6c4b85c6473bfa7b3"
QUANTIZED = re.compile(
    r"\.(input_proj|transformer_block\.attn\.(wq_du|wk_dv|wv_dv|wr_du|wo_ud)"
    r"|transformer_block\.mlp\.(w13_dn|w2_md))\.weight$"
)
BITS, GROUP_SIZE = 8, 64


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True, help="directory holding mtp.safetensors")
    ap.add_argument("--target", type=Path, required=True, help="the Inkling-Small-MLX-Mixed-2bit artifact")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to overwrite a nonempty output directory")
    source = args.source / "mtp.safetensors"
    digest = sha256(source)
    if digest != SOURCE_SHA256:
        raise SystemExit(f"mtp.safetensors SHA-256 {digest} differs from the pinned {SOURCE_SHA256}")
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    weights = mx.load(str(source))
    out, quantized = {}, []
    for name in sorted(weights):
        value = weights[name]
        if QUANTIZED.search(name):
            if value.ndim != 2 or value.shape[-1] % GROUP_SIZE:
                raise SystemExit(f"{name}: shape {value.shape} is not group-{GROUP_SIZE} quantizable")
            q, scales, biases = mx.quantize(value, group_size=GROUP_SIZE, bits=BITS)
            prefix = name[: -len(".weight")]
            out[prefix + ".weight"], out[prefix + ".scales"], out[prefix + ".biases"] = q, scales, biases
            quantized.append(prefix)
        else:
            out[name] = value
    expected_quantized = 8 * 8
    if len(quantized) != expected_quantized:
        raise SystemExit(f"quantized {len(quantized)} projections, expected {expected_quantized}")
    args.out.mkdir(parents=True, exist_ok=True)
    tmp = args.out / ".mtp.safetensors.tmp.safetensors"
    mx.save_safetensors(str(tmp), out, metadata={"format": "mlx"})
    os.replace(tmp, args.out / "mtp.safetensors")
    target_config = json.loads((args.target / "config.json").read_text())
    config = {
        "mtp_config": target_config["mtp_config"],
        "quantization": {"bits": BITS, "group_size": GROUP_SIZE, "mode": "affine",
                         "modules": sorted(quantized)},
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    provenance = {
        "schema": "mlx2.inkling-mtp-q8-conversion.v1",
        "source": {"repository": SOURCE_REPO, "revision": SOURCE_REVISION,
                   "file": "mtp.safetensors", "sha256": digest, "license": "see source model card"},
        "target_artifact": args.target.name,
        "conversion": {"bits": BITS, "group_size": GROUP_SIZE, "mode": "affine",
                       "quantized_modules": len(quantized), "kept_bf16": sorted(
                           n for n in out if not any(n.startswith(p + ".") for p in quantized))},
        "output_sha256": sha256(args.out / "mtp.safetensors"),
        "converter_sha256": sha256(Path(__file__)),
        "source_revision": revision,
        "mlx_version": mx.__version__,
        "converted_at": time.time(),
        "qualified": False,
    }
    (args.out / "conversion.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print(json.dumps({k: provenance[k] for k in ("output_sha256", "source_revision")}))


if __name__ == "__main__":
    main()
