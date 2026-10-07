"""Pinned, local-only MLX-VLM bridge for standalone candidate generation."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


def load_backend(source_root: str | Path, revision: str, source_paths: tuple[str, ...]):
    """Import only after exact local revision and source-file checks pass."""
    root = Path(source_root).expanduser().resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        from .mlx_vlm_pin import MLX_VLM_REVISION

        raise RuntimeError(
            f"MLX-VLM source revision mismatch: {root} is at {actual}, this "
            f"candidate requires {revision}"
            + (
                f" (the project pin installs {MLX_VLM_REVISION[:8]}, which is not "
                f"it; put a clean {revision[:8]} checkout first on PYTHONPATH)"
                if revision != MLX_VLM_REVISION
                else ""
            )
        )
    changed = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", *source_paths],
        check=False,
    )
    if changed.returncode != 0:
        raise RuntimeError("MLX-VLM pinned source files have local changes")
    loaded = sys.modules.get("mlx_vlm")
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(root):
        raise RuntimeError("another MLX-VLM checkout is already imported")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("mlx_vlm")
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError("MLX-VLM import did not resolve to pinned source")
    return module


def validate_media_paths(items: list[str | Path] | None, *, kind: str) -> list[str]:
    if not items:
        return []
    paths = []
    for item in items:
        path = Path(item).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"missing local {kind} input: {item}")
        paths.append(str(path))
    return paths
