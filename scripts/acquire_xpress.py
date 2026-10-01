#!/usr/bin/env python3
"""Acquire the pinned 4B XPress companion; metadata and hashes, no tensor imports.

The existing Qwen3-4B target can be reused. The duplicate pickle checkpoint is
intentionally absent from this serving manifest. Network bytes are bounded by
the pinned file sizes; the receipt is written only after reconciliation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

REPOSITORY = "UIUC-SSAIL/Qwen3-4B-XPress-b16"
REVISION = "fce84732637e1fb70bea3c70ca341131b8d49b61"
FILES = {
    "config.json": {
        "size": 1231,
        "git_blob": "c113d96e4b56ee6f0cb28534f5eb983b53616188",
    },
    "model.safetensors": {
        "size": 1235162528,
        "sha256": "d14de3fc431ee778675624d35f05b87bd458b5ac82d55e9080d81532780c8978",
    },
}


def verify_file(path: Path, expected: dict) -> dict:
    if not path.is_file() or path.stat().st_size != expected["size"]:
        raise ValueError(f"Size mismatch: {path.name}")
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {expected['size']}\0".encode())
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            sha.update(data)
            blob.update(data)
    if expected.get("sha256") and sha.hexdigest() != expected["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path.name}")
    if expected.get("git_blob") and blob.hexdigest() != expected["git_blob"]:
        raise ValueError(f"Git blob mismatch: {path.name}")
    return {"name": path.name, "size": expected["size"], "sha256": sha.hexdigest()}


def acquire(destination: Path) -> dict:
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    receipt = destination / ".mlx2-acquisition.json"
    # An old receipt never establishes completion for a failing retry.
    if receipt.exists():
        receipt.unlink()
    receipt_temp = receipt.with_suffix(".json.partial")
    if receipt_temp.exists():
        receipt_temp.unlink()
    records = []
    for name, expected in FILES.items():
        final = destination / name
        partial = destination / (name + ".partial")
        if final.exists():
            records.append(verify_file(final, expected))
            if partial.exists():
                partial.unlink()
            continue
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected["size"]:
            raise ValueError(f"Oversized partial: {name}")
        if offset < expected["size"]:
            url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
            headers = {"User-Agent": "mlx2-pinned-artifact-acquisition"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status != 206:
                    raise ValueError(f"Server did not honor resume range for {name}")
                if response.status == 206:
                    actual_range = response.headers.get("Content-Range", "")
                    if not actual_range.startswith(f"bytes {offset}-"):
                        raise ValueError(f"Wrong resume range for {name}")
                with partial.open("ab" if offset else "wb") as stream:
                    started = time.monotonic()
                    reported = started
                    while True:
                        data = response.read(8 << 20)
                        if not data:
                            break
                        if offset + len(data) > expected["size"]:
                            raise ValueError(f"Download exceeded pinned size: {name}")
                        stream.write(data)
                        offset += len(data)
                        if time.monotonic() - reported >= 15:
                            print(
                                f"{name}: {offset}/{expected['size']} bytes", flush=True
                            )
                            reported = time.monotonic()
                    stream.flush()
                    os.fsync(stream.fileno())
        record = verify_file(partial, expected)
        os.replace(partial, final)
        records.append({**record, "name": name})
        print(f"Verified {name}: {expected['size']} bytes", flush=True)
    partials = list(destination.glob("*.partial"))
    if partials:
        raise ValueError("Unreconciled partial files remain")
    result = {
        "schema": "mlx2.artifact-acquisition.v1",
        "repository": REPOSITORY,
        "revision": REVISION,
        "path": str(destination),
        "files": records,
        "downloaded": True,
        "hash_verified": True,
        "partials": 0,
        "model_loaded": False,
        "qualified": False,
    }
    temp = receipt.with_suffix(".json.partial")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temp, receipt)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path.home() / "mlx-models/Qwen3-4B-XPress-b16",
    )
    options = parser.parse_args()
    print(json.dumps(acquire(options.destination), indent=2))
