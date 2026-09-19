# SPDX-License-Identifier: MIT
"""Exact, block-granular files for APCv2's process-local persistent tier."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path


_FORMAT = "mlx2-apcv2-blocks-v1"


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


# Snapshots are process-local scratch (the owning APCv2 sweeps them at
# startup), so the manifest MAC key lives only in this process.  A neighbour
# with write access to the cache directory can recompute plain digests for a
# swapped payload; it cannot forge this MAC.
_MANIFEST_KEY = secrets.token_bytes(32)


def _manifest_mac(signature: str, size: int, sha256: str) -> str:
    message = f"{signature}|{int(size)}|{sha256}".encode("utf-8")
    return hmac.new(_MANIFEST_KEY, message, hashlib.sha256).hexdigest()


def _contained_block_path(directory: Path, name: object) -> Path:
    """Resolve one manifest block without permitting traversal or symlinks."""
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("APCv2 persistent block name is not a strict basename")
    if name in {".", ".."} or not name.endswith(".block"):
        raise ValueError("APCv2 persistent block name is invalid")
    root = directory.resolve()
    candidate = (directory / name).resolve()
    if candidate.parent != root:
        raise ValueError("APCv2 persistent block path escapes its directory")
    return candidate


def encode_block_file(path: Path, *, block_bytes: int, signature: str) -> tuple[Path, ...]:
    """Replace a completed file with an atomic manifest and immutable blocks."""
    path = Path(path)
    width = int(block_bytes)
    if width <= 0:
        return (path,)
    raw = path.read_bytes()
    directory = path.with_suffix(path.suffix + ".blocks")
    temporary = directory.with_name(f".{directory.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    blocks = []
    try:
        for index, start in enumerate(range(0, len(raw), width)):
            payload = raw[start:start + width]
            name = f"{index:08d}-{_digest(payload)}.block"
            (temporary / name).write_bytes(payload)
            blocks.append({"name": name, "size": len(payload), "sha256": _digest(payload)})
        whole = _digest(raw)
        manifest = {
            "format": _FORMAT,
            "signature": str(signature),
            "size": len(raw),
            "sha256": whole,
            "mac": _manifest_mac(str(signature), len(raw), whole),
            "block_bytes": width,
            "blocks": blocks,
        }
        manifest_tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.manifest")
        manifest_tmp.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        os.replace(temporary, directory)
        os.replace(manifest_tmp, path)
    except BaseException:
        for item in temporary.glob("*") if temporary.exists() else ():
            item.unlink(missing_ok=True)
        if temporary.exists():
            temporary.rmdir()
        raise
    return (path, *sorted(directory.glob("*.block")))


def block_file_paths(path: Path) -> tuple[Path, ...]:
    path = Path(path)
    if not path.exists():
        return (path,)
    manifest = _read_block_manifest(path)
    if manifest is None:
        return (path,)
    directory = path.with_suffix(path.suffix + ".blocks")
    try:
        blocks = tuple(
            _contained_block_path(directory, item["name"])
            for item in manifest["blocks"]
        )
    except (KeyError, TypeError) as error:
        raise ValueError("APCv2 persistent block manifest is malformed") from error
    return (path, *blocks)


def _read_block_manifest(path: Path):
    """Return our JSON manifest without decoding an ordinary payload.

    Persistent APCv2 safetensors share the same filename suffix as a block
    manifest.  Reading the whole file as UTF-8 to distinguish them makes pin
    accounting O(entries * payload bytes) and adds an extra full payload pass
    to restore.  A generated block manifest starts with JSON; a safetensors
    file starts with its little-endian header length (normally including NULs).
    """
    try:
        with path.open("rb") as handle:
            prefix = handle.read(64)
        if b"\0" in prefix or not prefix.lstrip().startswith(b"{"):
            return None
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if manifest.get("format") != _FORMAT:
        return None
    return manifest


def remove_block_file(path: Path) -> int:
    try:
        paths = block_file_paths(path)
    except ValueError:
        # A corrupt manifest must never authorize deletion outside its own
        # directory. Removing the manifest makes the entry unusable; safe
        # remnants are reclaimed by startup cleanup.
        paths = (Path(path),)
    removed = 0
    for item in reversed(paths):
        try:
            removed += item.stat().st_size
            item.unlink()
        except OSError:
            pass
    directory = Path(path).with_suffix(Path(path).suffix + ".blocks")
    try:
        directory.rmdir()
    except OSError:
        pass
    return removed


@contextmanager
def materialize_block_file(path: Path, *, expected_signature: str):
    """Verify every block and full digest before exposing a temporary file."""
    path = Path(path)
    manifest = _read_block_manifest(path)
    if manifest is None:
        yield path
        return
    if manifest.get("signature") != str(expected_signature):
        raise ValueError("APCv2 persistent block signature mismatch")
    expected_mac = _manifest_mac(
        str(expected_signature), manifest.get("size", -1), str(manifest.get("sha256", ""))
    )
    if not hmac.compare_digest(str(manifest.get("mac", "")), expected_mac):
        raise ValueError("APCv2 persistent block manifest authentication failed")
    directory = path.with_suffix(path.suffix + ".blocks")
    chunks = []
    try:
        blocks = tuple(manifest["blocks"])
    except (KeyError, TypeError) as error:
        raise ValueError("APCv2 persistent block manifest is malformed") from error
    for item in blocks:
        try:
            block_path = _contained_block_path(directory, item["name"])
        except (KeyError, TypeError) as error:
            raise ValueError("APCv2 persistent block manifest is malformed") from error
        payload = block_path.read_bytes()
        if len(payload) != int(item["size"]) or _digest(payload) != item["sha256"]:
            raise ValueError("APCv2 persistent block checksum mismatch")
        chunks.append(payload)
    raw = b"".join(chunks)
    if len(raw) != int(manifest["size"]) or _digest(raw) != manifest["sha256"]:
        raise ValueError("APCv2 persistent snapshot checksum mismatch")
    # MLX dispatches by suffix, so the verified reconstruction must retain the
    # safetensors extension even though it is an ephemeral restore artifact.
    temporary = path.with_name(
        f".{path.stem}.{uuid.uuid4().hex}.restore{path.suffix}"
    )
    temporary.write_bytes(raw)
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)
