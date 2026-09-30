"""Containment of weight shards inside an artifact directory."""

from __future__ import annotations

from pathlib import Path


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
