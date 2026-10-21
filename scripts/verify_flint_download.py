#!/usr/bin/env python3
"""Verify the pinned Flint Hugging Face snapshot and write an atomic receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlx2.adapters.flint import (
    SOURCE_INDEX_SHA256,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    inspect_artifact,
)

RECEIPT = ".mlx2-hf-download.json"


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> dict:
    root = root.expanduser().resolve(strict=True)
    tree_path = root / ".cache" / "huggingface" / "trees" / f"{SOURCE_REVISION}.json"
    if not tree_path.is_file():
        raise ValueError("missing pinned Hugging Face tree receipt")
    tree = json.loads(tree_path.read_text())
    files = tree.get("files") if isinstance(tree, dict) else None
    if tree.get("format_version") != 1 or not isinstance(files, dict):
        raise ValueError("malformed Hugging Face tree receipt")
    incomplete = list(
        (root / ".cache" / "huggingface" / "download").rglob("*.incomplete")
    )
    if incomplete:
        raise ValueError(f"snapshot still has {len(incomplete)} incomplete downloads")
    records = []
    total = 0
    for name, expected in sorted(files.items()):
        path = root / name
        if not path.is_file() or path.stat().st_size != expected.get("size"):
            raise ValueError(f"snapshot file is absent or has wrong size: {name}")
        if expected.get("lfs_sha256"):
            observed = _digest(path, "sha256")
            wanted = expected["lfs_sha256"]
            kind = "lfs_sha256"
        else:
            observed = _git_blob_sha1(path)
            wanted = expected.get("blob_id")
            kind = "git_blob_sha1"
        if observed != wanted:
            raise ValueError(f"snapshot digest differs: {name}")
        size = path.stat().st_size
        total += size
        records.append({"path": name, "bytes": size, kind: observed})
    artifact = inspect_artifact(root)
    if _digest(root / "model.safetensors.index.json", "sha256") != SOURCE_INDEX_SHA256:
        raise ValueError("source index differs after artifact inspection")
    output = root / RECEIPT
    prior = {}
    if output.is_file():
        value = json.loads(output.read_text())
        if isinstance(value, dict):
            prior = value
    receipt = {
        "schema": "mlx2.huggingface-snapshot-download.v1",
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "download_workers": 5,
        "authentication": "PAT read from external token file; value not retained",
        "files": records,
        "file_count": len(records),
        "bytes": total,
        "artifact_fingerprint": artifact["identity"]["fingerprint"],
        "complete": True,
        "conversion_started": prior.get("conversion_started") is True,
        "qualified": False,
        "selected": False,
        "observed_used": False,
    }
    for name in ("conversion_profile", "conversion_destination"):
        if receipt["conversion_started"] and isinstance(prior.get(name), str):
            receipt[name] = prior[name]
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    receipt = verify(args.root)
    print(
        json.dumps(
            {
                "receipt": str(args.root.expanduser().resolve() / RECEIPT),
                "file_count": receipt["file_count"],
                "bytes": receipt["bytes"],
                "artifact_fingerprint": receipt["artifact_fingerprint"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
