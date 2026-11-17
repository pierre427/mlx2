"""Containment of weight shards inside an artifact directory."""

from __future__ import annotations

import re
from pathlib import Path

_HUB_COMMIT = re.compile(r"[0-9a-f]{40}")
_HUB_BLOB = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


def shard_within_artifact(path: Path, resolved: Path) -> bool:
    """Whether a resolved shard belongs to the artifact at ``path``.

    A Hugging Face cache snapshot (``<repo>/snapshots/<rev>``) holds symlinks
    into its own ``<repo>/blobs``; those resolve outside the snapshot but stay
    inside the same repository.  Anything else outside ``path`` is foreign.
    """
    path = Path(path)
    if resolved.is_relative_to(path):
        return True
    return path.parent.name == "snapshots" and resolved.is_relative_to(
        path.parent.parent / "blobs"
    )


def hub_snapshot_revision(path: Path) -> str | None:
    """Return a Hub cache snapshot revision when ``path`` has that layout."""
    path = Path(path)
    if path.parent.name != "snapshots" or path.parent.parent.name == "":
        return None
    revision = path.name
    return revision if _HUB_COMMIT.fullmatch(revision) else None


def hub_blob_identity(snapshot: Path, resolved: Path) -> str | None:
    """Return the content-addressed Hub blob name for a snapshot member."""
    snapshot = Path(snapshot)
    resolved = Path(resolved)
    if snapshot.parent.name != "snapshots":
        return None
    blob_root = snapshot.parent.parent / "blobs"
    if not resolved.is_relative_to(blob_root):
        return None
    name = resolved.name
    return name if _HUB_BLOB.fullmatch(name) else None


def safetensors_shards(root: Path, pattern: str = "*.safetensors") -> list[Path]:
    """Sorted shard files under ``root`` matching ``pattern``, without dotfiles.

    ``Path.glob`` matches names that start with a dot. exFAT and other non-HFS
    volumes (the T7) carry an AppleDouble ``._<name>`` metadata file beside
    every file macOS writes there, so ``._model-00001-of-00009.safetensors``
    would otherwise be read as a shard.
    """
    return sorted(p for p in Path(root).glob(pattern) if not p.name.startswith("."))
