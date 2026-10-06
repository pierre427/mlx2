#!/usr/bin/env python3
"""Default-off prototype cache for payload SHA-256 digests.

This is research tooling, not an adapter fast path. A hit is accepted only
when both the lexical path entry and the opened file descriptor retain the
same device, inode, size, mtime_ns, and ctime_ns recorded with the digest.
The file descriptor is opened before validation so a path swap cannot change
the bytes being checked after admission.

Do not use this on filesystems that do not provide stable inode and nanosecond
mtime/ctime semantics. The cache cannot defend against a filesystem or local
principal that can forge those values; full hashing remains the adapter
default and the qualification reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

SCHEMA = "mlx2.payload-digest-cache.prototype.v1"
MAX_CACHE_BYTES = 16 << 20


def _binding(item: os.stat_result) -> list[int]:
    return [
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    ]


def _lexical_path(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(path)))


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {"schema": SCHEMA, "entries": {}}
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_nlink != 1
        or info.st_mode & 0o022
        or info.st_size > MAX_CACHE_BYTES
    ):
        raise ValueError("digest cache ownership, mode, link count, or size is unsafe")
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("digest cache is corrupt") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("digest cache schema is invalid")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise TypeError("digest cache entries are invalid")
    for key, value in entries.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, dict)
            or set(value) != {"path_binding", "target_binding", "sha256"}
            or not all(
                isinstance(binding, list)
                and len(binding) == 5
                and all(type(part) is int for part in binding)
                for binding in (value["path_binding"], value["target_binding"])
            )
            or not isinstance(value["sha256"], str)
            or len(value["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in value["sha256"])
        ):
            raise ValueError("digest cache contains an invalid entry")
    return payload


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _hash_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    while block := os.read(descriptor, 8 << 20):
        digest.update(block)
    return digest.hexdigest()


def _digest_one(path: Path, cached: dict | None) -> tuple[dict, bool]:
    lexical = _lexical_path(path)
    path_before = _binding(lexical.lstat())
    descriptor = os.open(lexical, os.O_RDONLY)
    try:
        target_before = _binding(os.fstat(descriptor))
        if target_before != _binding(lexical.stat()):
            raise ValueError(f"payload path changed while opening: {lexical}")
        if (
            cached is not None
            and cached["path_binding"] == path_before
            and cached["target_binding"] == target_before
        ):
            digest = cached["sha256"]
            hit = True
        else:
            digest = _hash_fd(descriptor)
            hit = False
        target_after = _binding(os.fstat(descriptor))
        path_after = _binding(lexical.lstat())
        followed_after = _binding(lexical.stat())
        if (
            target_before != target_after
            or target_before != followed_after
            or path_before != path_after
        ):
            raise ValueError(f"payload source changed during digest validation: {lexical}")
        return {
            "path_binding": path_after,
            "target_binding": target_after,
            "sha256": digest,
        }, hit
    finally:
        os.close(descriptor)


def digest_paths(paths: list[Path], cache_path: Path) -> dict:
    cache_path = _lexical_path(cache_path)
    payload = _load_cache(cache_path)
    entries = payload["entries"]
    results = []
    hits = 0
    for path in paths:
        lexical = _lexical_path(path)
        key = str(lexical)
        record, hit = _digest_one(lexical, entries.get(key))
        entries[key] = record
        results.append({"path": key, "sha256": record["sha256"], "cache_hit": hit})
        hits += int(hit)
    _write_cache(cache_path, payload)
    return {
        "schema": SCHEMA,
        "cache": str(cache_path),
        "hits": hits,
        "misses": len(results) - hits,
        "files": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(digest_paths(args.paths, args.cache), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
