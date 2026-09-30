#!/usr/bin/env python3
"""Reconcile the pinned HiLS source and mlx2's local q6 conversion."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx2.adapters.olmo_hils import inspect_artifact


REPO_ID = "tencent/HiLS-Attention-7B"
REVISION = "b2cf70681a88ce34c84da927f3631818254ce855"
API = f"https://huggingface.co/api/models/{REPO_ID}/revision/{REVISION}?blobs=true"
TRANSIENT_SUFFIXES = (".incomplete", ".part", ".tmp", ".lock")
Q6_MANIFEST_NAME = ".mlx2-artifact-manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.name != Q6_MANIFEST_NAME
    )


def partials(root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name.endswith(TRANSIENT_SUFFIXES)
    )


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--q6", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-model-manifest", action="store_true")
    args = parser.parse_args(argv)
    source, q6 = args.source.resolve(), args.q6.resolve()
    started = time.time()
    checks: dict[str, bool] = {}
    errors: list[str] = []

    with urllib.request.urlopen(API, timeout=60) as response:
        upstream = json.load(response)
    checks["upstream_repo_id"] = upstream.get("id") == REPO_ID
    checks["upstream_revision"] = upstream.get("sha") == REVISION

    expected = {entry["rfilename"]: entry for entry in upstream["siblings"]}
    source_records = []
    for name, entry in sorted(expected.items()):
        path = source / name
        record = {"name": name, "expected_size": entry.get("size")}
        if path.is_file():
            record["size"] = path.stat().st_size
            record["size_match"] = record["size"] == record["expected_size"]
            lfs = entry.get("lfs")
            if lfs:
                record["expected_sha256"] = lfs["sha256"]
                record["sha256"] = sha256(path)
                record["digest_match"] = record["sha256"] == record["expected_sha256"]
            else:
                record["expected_git_blob_sha1"] = entry["blobId"]
                record["git_blob_sha1"] = git_blob_sha1(path)
                record["digest_match"] = (
                    record["git_blob_sha1"] == record["expected_git_blob_sha1"]
                )
        else:
            record.update({"size_match": False, "digest_match": False, "missing": True})
        source_records.append(record)
    expected_source_names = set(expected)
    actual_source_names = {path.name for path in source.iterdir() if path.is_file()}
    checks["source_expected_files"] = actual_source_names == expected_source_names
    checks["source_sizes_and_digests"] = all(
        item.get("size_match") and item.get("digest_match") for item in source_records
    )
    source_partials = partials(source)
    checks["source_zero_partials"] = not source_partials

    q6_index = json.loads((q6 / "model.safetensors.index.json").read_text())
    referenced_shards = sorted(set(q6_index["weight_map"].values()))
    q6_records = []
    for path in files(q6):
        q6_records.append(
            {"name": path.name, "size": path.stat().st_size, "sha256": sha256(path)}
        )
    actual_q6_shards = sorted(path.name for path in q6.glob("*.safetensors"))
    checks["q6_index_shards_complete"] = actual_q6_shards == referenced_shards
    q6_partials = partials(q6)
    checks["q6_zero_partials"] = not q6_partials

    source_config = json.loads((source / "config.json").read_text())
    q6_config = json.loads((q6 / "config.json").read_text())
    stripped_q6 = {
        key: value
        for key, value in q6_config.items()
        if key not in {"quantization", "quantization_config"}
    }
    checks["q6_config_is_pinned_source_plus_quantization"] = (
        stripped_q6 == source_config
    )
    expected_quant = {"group_size": 64, "bits": 6, "mode": "affine"}
    checks["q6_quantization"] = (
        q6_config.get("quantization") == expected_quant
        and q6_config.get("quantization_config") == expected_quant
    )
    try:
        inspection = inspect_artifact(q6)
        checks["mlx2_artifact_inspection"] = bool(
            inspection["quantized"]
            and inspection["hils_layers"] == 8
            and inspection["swa_layers"] == 24
            and not inspection["apcv2_qualified"]
        )
    except Exception as exc:  # noqa: BLE001 - keep a durable failure receipt
        inspection = None
        checks["mlx2_artifact_inspection"] = False
        errors.append(f"{type(exc).__name__}: {exc}")

    complete = all(checks.values())
    manifest = {
        "schema": "mlx2.hils-artifact-completion.v1",
        "status": "complete" if complete else "failed",
        "created_at": time.time(),
        "elapsed_seconds": time.time() - started,
        "upstream": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "api": API,
            "last_modified": upstream.get("lastModified"),
            "local_path": str(source),
            "acquisition_command": [
                "~/Desktop/mlx-uag/.venv/bin/hf",
                "download",
                REPO_ID,
                "--revision",
                REVISION,
                "--local-dir",
                str(source),
            ],
            "files": source_records,
            "partials": source_partials,
        },
        "q6_artifact": {
            "local_path": str(q6),
            "publication": "local affine q6 conversion; no upstream q6 repository asserted",
            "conversion_source_revision": "458e7ff1e6b521445537590f68625d39c28794f6",
            "conversion_source_path": "mlx_lm/models/olmo_hils.py",
            "quantization": expected_quant,
            "expected_files": [record["name"] for record in q6_records],
            "files": q6_records,
            "referenced_shards": referenced_shards,
            "partials": q6_partials,
            "mlx2_inspection": inspection,
        },
        "mlx2": {
            "source_revision": git("rev-parse", "HEAD"),
            "source_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        },
        "checks": checks,
        "errors": errors,
        "complete": complete,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    args.output.write_text(text)
    if args.write_model_manifest:
        (q6 / Q6_MANIFEST_NAME).write_text(text)
    print(json.dumps({"status": manifest["status"], "checks": checks}, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
