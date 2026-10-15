#!/usr/bin/env python3
"""Download and hash-verify the pinned BF16 FLUX 3 package (no GPU use)."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

from mlx2.adapters.flux3_image import snapshot_manifest


def download(root: Path, workers: int = 3):
    from huggingface_hub import snapshot_download

    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    expected = snapshot_manifest()
    with (root / ".mlx2-download.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = root / ".hf-download-manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != expected:
            raise ValueError("existing download manifest has another artifact identity")
        if not manifest_path.exists() and set(root.iterdir()) - {
            root / ".mlx2-download.lock"
        }:
            raise ValueError("refusing an unbound nonempty model directory")
        if not manifest_path.exists():
            temporary_manifest = manifest_path.with_suffix(".tmp")
            temporary_manifest.write_text(json.dumps(expected, indent=2) + "\n")
            os.replace(temporary_manifest, manifest_path)
        snapshot_download(
            expected["repo"],
            revision=expected["revision"],
            local_dir=root,
            allow_patterns=[e["path"] for e in expected["files"]],
            max_workers=workers,
        )
        for entry in expected["files"]:
            path = (root / entry["path"]).resolve()
            if not path.is_relative_to(root) or path.stat().st_size != entry["size"]:
                raise ValueError(f"missing or incomplete model file: {entry['path']}")
            digest = hashlib.sha256() if entry["sha256"] else hashlib.sha1()
            if not entry["sha256"]:
                digest.update(f"blob {entry['size']}\0".encode())
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != (entry["sha256"] or entry["git_blob"]):
                raise ValueError(f"model hash differs: {entry['path']}")
            print(f"verified {entry['path']}", flush=True)
        receipt = {
            "repo": expected["repo"],
            "revision": expected["revision"],
            "files": len(expected["files"]),
            "total_bytes": sum(e["size"] for e in expected["files"]),
        }
        marker = root / ".hf-download-complete.json"
        if not marker.exists() or json.loads(marker.read_text()) != receipt:
            temporary = marker.with_suffix(".tmp")
            temporary.write_text(json.dumps(receipt, indent=2) + "\n")
            os.replace(temporary, marker)
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(download(args.output, args.workers), indent=2))
