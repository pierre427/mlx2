"""Exact Vision-Exp DSpark sidecar identity, independent of download receipts.

The first preparation streams the complete file with before/after stat checks.
Serving inspections trust only that stat-bound mlx2 seal and the fixed SHA256,
then revalidate the bounded GGUF metadata and complete tensor index.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import asdict
from pathlib import Path

from ..runtime.drafters.deepseek_v4_dspark import SIDECAR_SHA256, DeepSeekV4DSparkConfig
from ..runtime.gguf.reader import GGUFFile

FILENAME = "DeepSeek-V4-Flash-Vision-Exp-DSpark-support.gguf"
SIZE = 5989114528
SCHEMA = "mlx2.deepseek-v4-dspark-integrity.v1"


def file_stat(path):
    value = path.stat()
    return {
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "inode": value.st_ino,
        "device": value.st_dev,
    }


def seal_path(path):
    return Path(str(path) + ".mlx2-integrity.json")


def _seal_bytes(path):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("DSpark integrity seal must be a regular nonsymlink file")
    # O_NOFOLLOW also covers replacement between the check and descriptor open.
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("DSpark integrity seal must be a regular file")
        return stream.read()


def _path(value):
    path = Path(value)
    if path.name != FILENAME or path.is_symlink() or not path.is_file():
        raise ValueError("exact regular Vision-Exp DSpark sidecar file required")
    if path.stat().st_size != SIZE:
        raise ValueError("DSpark sidecar size differs from exact checkpoint")
    return path


def _metadata(path):
    with GGUFFile(path) as index:
        config = DeepSeekV4DSparkConfig.from_index(index.metadata, index.tensors)
        return json.loads(json.dumps(asdict(config))), len(index.tensors)


def prepare_sidecar(value):
    """Hash actual complete bytes once; mint an atomic stat-bound integrity seal."""
    path = _path(value)
    before = file_stat(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    after = file_stat(path)
    if before != after:
        raise ValueError("DSpark file changed during complete SHA256 verification")
    actual = digest.hexdigest()
    if actual != SIDECAR_SHA256:
        raise ValueError("actual DSpark bytes do not match pinned SHA256")
    config, count = _metadata(path)
    if file_stat(path) != after:
        raise ValueError("DSpark file changed during metadata validation")
    seal = {
        "schema": SCHEMA,
        "filename": FILENAME,
        "sha256": actual,
        "weight_stat": after,
        "config": config,
        "tensor_count": count,
        "verification": "actual_full_file_stream_sha256_before_after_stat",
    }
    output = seal_path(path)
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise ValueError("DSpark integrity seal must be a regular nonsymlink file")
    # Exclusive random sibling creation avoids following a preexisting .tmp
    # link. Rename publishes atomically; finally removes failed staging files.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=output.parent,
                                         prefix=output.name + ".", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(seal, indent=2) + "\n")
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return inspect_sidecar(path)


def inspect_sidecar(value):
    """Bounded route inspection; fail closed on missing/stale/foreign seals."""
    path = _path(value)
    before = file_stat(path)
    seal_bytes = _seal_bytes(seal_path(path))
    seal = json.loads(seal_bytes)
    if (
        seal.get("schema") != SCHEMA
        or seal.get("filename") != FILENAME
        or seal.get("sha256") != SIDECAR_SHA256
        or seal.get("weight_stat") != before
        or seal.get("verification")
        != "actual_full_file_stream_sha256_before_after_stat"
    ):
        raise ValueError("DSpark integrity seal missing, stale or incompatible")
    config, count = _metadata(path)
    if (
        seal.get("config") != config
        or seal.get("tensor_count") != count
        or file_stat(path) != before
    ):
        raise ValueError("DSpark sealed index/config changed")
    return {
        "path": str(path),
        "sha256": SIDECAR_SHA256,
        "stat": before,
        "config": config,
        "tensor_count": count,
        "seal_path": str(seal_path(path)),
        "seal_sha256": hashlib.sha256(seal_bytes).hexdigest(),
        "verification": seal["verification"],
    }


def verified_source_identity(root=None):
    """Recompute loaded source bytes; environment/descriptor cannot forge binding."""
    import os
    import subprocess

    from ..desktop_runner import digest, source_manifest

    root = Path(__file__).resolve().parents[3] if root is None else Path(root)
    actual_manifest = source_manifest(root)
    snapshot = digest(actual_manifest)
    expected = os.environ.get("MLX2_SOURCE_SNAPSHOT_SHA256")
    revision = os.environ.get("MLX2_SOURCE_REVISION")
    if bool(expected) != bool(revision):
        raise ValueError("both frozen source revision and snapshot required")
    if expected:
        if snapshot != expected:
            raise ValueError("loaded DSpark source bytes differ from frozen snapshot")
        descriptor_path = Path(
            os.environ.get("MLX2_SOURCE_MANIFEST", root / "desktop-source.json")
        )
        descriptor = json.loads(descriptor_path.read_text())
        if (
            descriptor.get("source_revision") != revision
            or descriptor.get("source_sha256", descriptor.get("source_snapshot_sha256"))
            != snapshot
            or ("files" in descriptor and descriptor["files"] != actual_manifest)
        ):
            raise ValueError(
                "DSpark frozen source descriptor differs from actual bytes"
            )
    else:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    return {"source_revision": revision, "source_snapshot_sha256": snapshot}
