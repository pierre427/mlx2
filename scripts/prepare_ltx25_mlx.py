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
NEEDED = {
    "config": ("--distilled",),
    "transformer-distilled": ("--distilled",),
    "connector": ("--distilled", "--gemma4"),
    "text-encoder": ("--gemma4",),
    "vae": ("--vae-conv",),
    "audio-vae": ("--audio-vae",),
    "duration-head": ("--duration-head",),
    "upscalers": ("--spatial-upscaler", "--temporal-upscaler"),
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


def _partial_fingerprint(source: Path, paths: tuple[Path, ...]) -> str:
    manifest = json.loads((source / ".hf-download-manifest.json").read_text())
    if (
        manifest.get("repo") != "Lightricks/LTX-2.5"
        or manifest.get("revision") != "5e6e71018ee1756ed329b697a7b4aedc934dfce9"
        or len(manifest.get("files", [])) != 8
    ):
        raise ValueError("LTX partial manifest is not the pinned selected snapshot")
    entries = {entry["path"]: entry for entry in manifest["files"]}
    for path in paths:
        relative = path.relative_to(source).as_posix()
        entry = entries.get(relative)
        if entry is None or not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError(f"LTX source component is incomplete: {relative}")
        if entry.get("sha256") and _sha256(path) != entry["sha256"]:
            raise ValueError(f"LTX source component hash changed: {relative}")
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def prepare(source: Path, output: Path, runtime: Path, *, only_step: str | None = None) -> dict:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    runtime = runtime.expanduser().resolve()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from mlx2.adapters.generative_media import inspect_ltx25_source

    steps = (only_step,) if only_step else STEPS
    if only_step is not None and only_step not in STEPS:
        raise ValueError(f"unsupported LTX conversion step: {only_step}")
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
    flags = {
        "--distilled": source / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "--gemma4": source / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "--vae-conv": source / "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "--audio-vae": source / "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "--duration-head": source / "model_patches/ltx-2.5-duration-head-bf16.safetensors",
        "--spatial-upscaler": source / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "--temporal-upscaler": source / "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
    }
    required = {flag for step in steps for flag in NEEDED[step]}
    for flag in required:
        path = flags[flag]
        if not path.is_file():
            raise FileNotFoundError(f"{flag}: {path}")
    if only_step:
        fingerprint = _partial_fingerprint(source, tuple(flags[flag] for flag in sorted(required)))
        source_revision = "5e6e71018ee1756ed329b697a7b4aedc934dfce9"
    else:
        artifact = inspect_ltx25_source(source)
        fingerprint = artifact.fingerprint
        source_revision = artifact.source_revision
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
        "source_fingerprint": fingerprint,
        "source_revision": source_revision,
        "runtime_revision": revision,
        "steps": {},
        "execution_qualification": "pending",
    }
    if receipt["source_fingerprint"] != fingerprint or receipt["runtime_revision"] != revision:
        raise ValueError("existing LTX conversion is bound to another source/runtime")
    launch = (
        "import mlx.core as mx, runpy, sys; "
        "mx.set_default_device(mx.cpu); "
        "sys.argv = [sys.argv[1], *sys.argv[2:]]; "
        "runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    for step in steps:
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
        pending = receipt_path.with_suffix(".json.tmp")
        pending.write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(pending, receipt_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--step", choices=STEPS, help="Convert one verified component before the full snapshot completes")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.runtime, only_step=args.step), indent=2))


if __name__ == "__main__":
    main()
