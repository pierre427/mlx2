#!/usr/bin/env python3
"""Prepare the pinned LTX-2.5 distilled MLX layout on CPU.

This orchestrates the already audited local ltx-2-mlx converter. Every child
sets MLX's default device to CPU before loading the conversion module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

RUNTIME_REVISION = "fbc4b0524dd1e01da2d07a44e14dd9dfe0a74d5e"
STEPS = (
    "config", "transformer-distilled", "connector", "text-encoder",
    "vae", "audio-vae", "duration-head", "upscalers",
)
EXPECTED = {
    "config": ("config.json", "embedded_config.json"),
    "transformer-distilled": ("transformer-distilled.safetensors",),
    "connector": ("connector.safetensors",),
    "text-encoder": ("text_encoder/config.json", "text_encoder/model.safetensors", "text_encoder/tokenizer.json"),
    "vae": ("vae_encoder.safetensors", "vae_decoder.safetensors"),
    "audio-vae": ("audio_vae.safetensors", "vocoder.safetensors"),
    "duration-head": ("duration_head.safetensors",),
    "upscalers": ("spatial_upscaler_x2.safetensors", "temporal_upscaler_x2.safetensors"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matches(output: Path, records: dict, expected: tuple[str, ...]) -> bool:
    if not isinstance(records, dict) or set(records) != set(expected):
        return False
    for name in expected:
        path = output / name
        record = records[name]
        if not isinstance(record, dict) or not path.is_file() or path.stat().st_size != record.get("size"):
            return False
        if _sha256(path) != record.get("sha256"):
            return False
    return True


def prepare(source: Path, output: Path, runtime: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from mlx2.adapters.generative_media import inspect_ltx25_source

    artifact = inspect_ltx25_source(source)
    revision = subprocess.run(
        ["git", "-C", str(runtime), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True, timeout=5,
    ).stdout.strip()
    if revision != RUNTIME_REVISION:
        raise ValueError("LTX converter runtime revision differs from pinned source")
    script = runtime / "scripts" / "convert_ltx25_to_mlx.py"
    python = runtime / ".venv" / "bin" / "python"
    if not script.is_file() or not python.is_file():
        raise FileNotFoundError("pinned LTX converter or interpreter is missing")
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / ".mlx2-cpu-conversion.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
        "source_fingerprint": artifact.fingerprint,
        "source_revision": artifact.source_revision,
        "runtime_revision": revision,
        "steps": {},
        "execution_qualification": "pending",
    }
    if receipt["source_fingerprint"] != artifact.fingerprint or receipt["runtime_revision"] != revision:
        raise ValueError("existing LTX conversion is bound to another source/runtime")
    flags = {
        "--distilled": source / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "--gemma4": source / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "--vae-conv": source / "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "--audio-vae": source / "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "--duration-head": source / "model_patches/ltx-2.5-duration-head-bf16.safetensors",
        "--spatial-upscaler": source / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "--temporal-upscaler": source / "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
    }
    for flag, path in flags.items():
        if not path.is_file():
            raise FileNotFoundError(f"{flag}: {path}")
    launch = (
        "import mlx.core as mx, runpy, sys; "
        "mx.set_default_device(mx.cpu); "
        "sys.argv = [sys.argv[1], *sys.argv[2:]]; "
        "runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    for step in STEPS:
        expected = EXPECTED[step]
        if _matches(output, receipt["steps"].get(step), expected):
            continue
        args = [str(python), "-c", launch, str(script), "--out", str(output), "--step", step]
        for flag, path in flags.items():
            args.extend((flag, str(path)))
        if step == "transformer-distilled":
            args.append("--q8")
        if step == "text-encoder":
            args.extend(("--gemma4-bits", "4"))
        environment = os.environ.copy()
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        print(f"CPU converting LTX-2.5 {step}", flush=True)
        subprocess.run(args, cwd=runtime, env=environment, check=True)
        records = {}
        for name in expected:
            path = output / name
            if not path.is_file():
                raise RuntimeError(f"LTX converter did not produce {name}")
            records[name] = {"size": path.stat().st_size, "sha256": _sha256(path)}
        receipt["steps"][step] = records
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.runtime), indent=2))


if __name__ == "__main__":
    main()
