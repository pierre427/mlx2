#!/usr/bin/env python3
"""Fail-closed six-family VLM qualification dispatcher and report contract."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

SCHEMA = "mlx2.vlm-route-qualification.v1"
REVISION_67599 = "67599f2e8ec31bf35cbb7b02794114f20844f0bb"
REVISION_8A5E = "8a5e704e0fe43cd8654c144c4ecbd4c8aececeb5"
FAMILIES = {
    "gemma3n": {
        "revision": REVISION_67599,
        "modalities": ("image", "video", "audio"),
        "producer": "qualify_native_vlm_media.py",
    },
    "gemma4": {
        "revision": REVISION_67599,
        "modalities": ("image", "video"),
        "producer": "qualify_native_vlm_media.py",
    },
    "minicpmo": {
        "revision": REVISION_67599,
        "modalities": ("image", "audio"),
        "producer": "qualify_native_vlm_media.py",
    },
    "smolvlm": {
        "revision": REVISION_8A5E,
        "modalities": ("image", "video"),
        "producer": "qualify_media_serving.py",
    },
    "lfm2_vl": {
        "revision": REVISION_8A5E,
        "modalities": ("image", "video"),
        "producer": "qualify_lfm25_media_serving.py",
    },
    "qwen2_5_vl": {
        "revision": REVISION_8A5E,
        "modalities": ("image", "video"),
        "producer": "qualify_qwen25_media_serving.py",
    },
}


def _hex(s):
    return (
        isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)
    )


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def artifact_sha256(path):
    """Stable bounded metadata fingerprint; never read multi-GB weight payloads."""
    path = Path(path)
    h = hashlib.sha256()
    h.update((path / "config.json").read_bytes())
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ValueError("model artifact has no top-level safetensors weights")
    for item in files:
        stat = item.stat()
        h.update(item.name.encode())
        h.update(str(stat.st_size).encode())
    return h.hexdigest()


def campaign_spec(
    family,
    *,
    artifact,
    reference_revision,
    reference_source_sha256,
    runtime,
    settings,
    producer_sha256,
):
    if family not in FAMILIES:
        raise ValueError(f"unsupported VLM family: {family}")
    artifact = Path(artifact).expanduser().resolve()
    if not artifact.is_dir() or not (artifact / "config.json").is_file():
        raise FileNotFoundError(f"model artifact unavailable: {artifact}")
    if reference_revision != FAMILIES[family]["revision"]:
        raise ValueError("reference revision does not match reviewed family contract")
    if not _hex(reference_source_sha256):
        raise ValueError("reference source SHA-256 is required")
    if not _hex(producer_sha256):
        raise ValueError("trusted producer SHA-256 is required")
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError("runtime identity is required")
    if not isinstance(settings, dict) or not settings:
        raise ValueError("serving settings identity is required")
    config = json.loads((artifact / "config.json").read_text())
    if config.get("model_type") != family:
        raise ValueError("artifact model_type does not match requested family")
    return {
        "schema": SCHEMA,
        "family": family,
        "reference_revision": reference_revision,
        "reference_source_sha256": reference_source_sha256,
        "producer_sha256": producer_sha256,
        "artifact": str(artifact),
        "artifact_sha256": artifact_sha256(artifact),
        "runtime": runtime,
        "settings": settings,
        "modalities": list(FAMILIES[family]["modalities"]),
        "required_features": [
            "ordinary_reference",
            "continuous_batch",
            *(["encoder_batching"] if family in {"gemma3n", "minicpmo"} else []),
            *FAMILIES[family]["modalities"],
            "prefix_reuse",
            "replay",
            "receipts",
        ],
    }


LEGACY_SOURCE_ROOT = Path(
    os.environ.get(
        "MLX2_VLM_8A5E_SOURCE_ROOT",
        Path.home() / "mlx-vlm-contract-sources" / "candidate-8a5e704e",
    )
)


def run_legacy_sourcebound_producer(root, producer, artifact, *, source_root=None):
    """Run a 8a5e family in a fresh interpreter with its exact package source."""
    source_root = Path(source_root or LEGACY_SOURCE_ROOT).expanduser().resolve()
    if not (source_root / "mlx_vlm").is_dir():
        raise RuntimeError(
            f"reviewed mlx-vlm source checkout unavailable: {source_root}"
        )
    runner = """
import contextlib, importlib.util, json, sys
path, artifact = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("_vlm_media_producer", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with contextlib.redirect_stdout(sys.stderr):
    report = module.run(artifact)
if not isinstance(report, dict):
    raise TypeError("producer report must be a JSON object")
print(json.dumps(report, default=str))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), str(Path(root).resolve() / "src")]
    )
    result = subprocess.run(
        [sys.executable, "-c", runner, str(producer), str(Path(artifact).resolve())],
        cwd=Path(root),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"source-bound producer failed ({result.returncode}): {result.stderr[-4000:]}"
        )
    try:
        return json.loads(result.stdout), result.returncode
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "source-bound producer did not emit one JSON object"
        ) from exc


def load_producer(root, family):
    name = FAMILIES[family]["producer"]
    if name is None:
        raise RuntimeError(f"no reviewed executable producer for {family}")
    path = Path(root) / "scripts" / name
    if not path.is_file():
        raise RuntimeError(f"reviewed producer unavailable: {path}")
    spec = importlib.util.spec_from_file_location(f"_vlm_producer_{family}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load producer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def dispatch(
    root,
    family,
    artifact,
    *,
    generation,
    cpg_owner_lock,
    cpg_task,
    legacy_source_root=None,
):
    """Run a reviewed family producer under an externally owned paired lease."""
    if family not in FAMILIES:
        raise ValueError(f"unsupported VLM family: {family}")
    if not Path(artifact).is_dir():
        return {"family": family, "passed": False, "status": "artifact_unavailable"}
    from qualification_gpu_ownership import require_qualification_lease

    owner = require_qualification_lease(
        task_id=cpg_task, cpg_owner_lock=cpg_owner_lock, generation=generation
    )
    producer = Path(root) / "scripts" / FAMILIES[family]["producer"]
    if not producer.is_file():
        raise RuntimeError(f"reviewed producer unavailable: {producer}")
    if family in {"gemma3n", "gemma4", "minicpmo"}:
        module, producer = load_producer(root, family)
        output = io.StringIO()
        args = [
            family,
            str(Path(artifact).resolve()),
            "--generation",
            str(generation),
            "--cpg-owner-lock",
            str(cpg_owner_lock),
            "--cpg-task",
            cpg_task,
        ]
        with contextlib.redirect_stdout(output):
            code = module.main(args)
        try:
            report = json.loads(output.getvalue())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "native family producer did not emit one JSON report"
            ) from exc
    else:
        # A fresh process ensures the approved 8a5e source package is loaded,
        # rather than reusing the native family process's 67599 package.
        report, code = run_legacy_sourcebound_producer(
            root, producer, artifact, source_root=legacy_source_root
        )
    if not isinstance(report, dict):
        raise TypeError("family producer report must be an object")
    report["dispatcher"] = {
        "name": "scripts/qualify_vlm_routes.py",
        "sha256": sha256_file(Path(__file__).resolve()),
        "producer": producer.name,
        "producer_sha256": sha256_file(producer),
        "producer_exit_code": code,
        "gpu_lease_owner": owner,
    }
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true")
    p.add_argument("--family", choices=tuple(FAMILIES))
    p.add_argument("--artifact", type=Path)
    p.add_argument("--generation", type=int)
    p.add_argument("--cpg-owner-lock", type=Path)
    p.add_argument("--cpg-task")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--legacy-source-root", type=Path, default=LEGACY_SOURCE_ROOT)
    a = p.parse_args(argv)
    if a.list:
        print(json.dumps(FAMILIES, indent=2))
        return 0
    if (
        not a.family
        or a.artifact is None
        or not a.generation
        or not a.cpg_owner_lock
        or not a.cpg_task
    ):
        p.error(
            "--family, --artifact, --generation, --cpg-owner-lock, and --cpg-task are required"
        )
    try:
        report = dispatch(
            a.root,
            a.family,
            a.artifact,
            generation=a.generation,
            cpg_owner_lock=a.cpg_owner_lock,
            cpg_task=a.cpg_task,
            legacy_source_root=a.legacy_source_root,
        )
    except Exception as exc:  # noqa: BLE001 - CLI must emit a failure report
        report = {
            "family": a.family,
            "passed": False,
            "status": "producer_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
