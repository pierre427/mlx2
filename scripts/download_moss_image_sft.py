#!/usr/bin/env python3
"""Download and hash-verify pinned MOSS-Image-SFT with five workers (CPU only)."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

from mlx2.adapters.moss_image import snapshot_manifest


def _digest(path, entry):
    digest = hashlib.sha256() if entry["sha256"] else hashlib.sha1()
    if not entry["sha256"]:
        digest.update(f"blob {entry['size']}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def download(root: Path, workers: int = 5):
    if workers != 5:
        raise ValueError("MOSS-Image acquisition requires exactly five workers")
    from huggingface_hub import snapshot_download

    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    expected = snapshot_manifest()
    with (root / ".mlx2-download.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = root / ".hf-download-manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != expected:
            raise ValueError("existing download manifest has another artifact identity")
        # The first acquisition may have been started by `hf download`. Adopt
        # only paths in the pinned tree plus Hugging Face's resumable cache.
        allowed = {entry["path"] for entry in expected["files"]}
        observed = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_relative_to(root / ".cache")
            and path.name
            not in {
                ".mlx2-download.lock",
                ".hf-download-manifest.json",
                ".hf-download-complete.json",
            }
        }
        if observed - allowed:
            raise ValueError("refusing unbound files in the model directory")
        if not manifest_path.exists():
            _atomic_json(manifest_path, expected)
        snapshot_download(
            expected["repo"],
            revision=expected["revision"],
            local_dir=root,
            max_workers=5,
        )
        for entry in expected["files"]:
            path = (root / entry["path"]).resolve()
            if not path.is_relative_to(root) or path.stat().st_size != entry["size"]:
                raise ValueError(f"missing or incomplete model file: {entry['path']}")
            if _digest(path, entry) != (entry["sha256"] or entry["git_blob"]):
                raise ValueError(f"model hash differs: {entry['path']}")
            print(f"verified {entry['path']}", flush=True)
        receipt = {
            "repo": expected["repo"],
            "revision": expected["revision"],
            "files": len(expected["files"]),
            "total_bytes": sum(entry["size"] for entry in expected["files"]),
        }
        _atomic_json(root / ".hf-download-complete.json", receipt)
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(download(args.output, args.workers), indent=2))
