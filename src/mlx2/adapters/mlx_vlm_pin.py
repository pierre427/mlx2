"""The mlx-vlm revision the Gemma 4, Gemma 3n, MiniCPM-o and Qwen-Image routes execute.

These routes import mlx-vlm model code at runtime, so the installed revision is
part of what a qualification run measured.  ``MLX_VLM_REVISION`` is Blaizzy
mlx-vlm main after the 0.7.3 release (0.7.3 plus the empty 3/5/6-bit quantized
KV width fix #2342, Qwen3.5 batched left padding #2357, small top-p / low
temperature sampling #2358 and interleaved image order #2362).  Adapters refuse
to load on any other revision or on an editable checkout with local changes
under ``mlx_vlm/``, and the resolved revision is recorded in the runtime
identity so receipts bind it.

The Qwen-Image direct adapter (``generative_media.py``) runs on this same
revision; it was re-qualified on it after its first qualification on the
local fork commit cc8b86f1.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path

MLX_VLM_VERSION = "0.7.3"
MLX_VLM_REVISION = "67599f2e8ec31bf35cbb7b02794114f20844f0bb"


def _git_head(root: Path) -> str | None:
    """HEAD of an editable checkout, suffixed ``+dirty`` when ``mlx_vlm/`` is
    modified: a local edit at the pinned HEAD is not the pinned code."""
    if not (root / ".git").exists():
        return None
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
        changes = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), "status",
             "--porcelain", "--untracked-files=normal", "--", "mlx_vlm"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{head}+dirty" if head and changes else head


def mlx_vlm_runtime() -> dict | None:
    """Describe the installed mlx-vlm: version, install source and revision.

    Returns ``None`` when mlx-vlm is not installed.  ``revision`` comes from
    the VCS install record, or from the checkout of an editable install; it is
    ``None`` when neither is available (a plain wheel).
    """
    try:
        dist = importlib.metadata.distribution("mlx-vlm")
    except importlib.metadata.PackageNotFoundError:
        return None
    direct = {}
    try:
        text = dist.read_text("direct_url.json")
        direct = json.loads(text) if text else {}
    except (OSError, ValueError):
        direct = {}
    revision = (direct.get("vcs_info") or {}).get("commit_id")
    if revision is None:
        spec = importlib.util.find_spec("mlx_vlm")
        if spec is not None and spec.origin is not None:
            revision = _git_head(Path(spec.origin).resolve().parents[1])
    return {
        "version": dist.version,
        "source": direct.get("url", "index"),
        "editable": bool((direct.get("dir_info") or {}).get("editable")),
        "revision": revision,
    }


def require_pinned_mlx_vlm(runtime: dict | None = None) -> dict:
    """Fail closed unless the installed mlx-vlm is the pinned revision."""
    runtime = mlx_vlm_runtime() if runtime is None else runtime
    if runtime is None:
        raise RuntimeError("multimodal adapters require the optional mlx-vlm runtime")
    if runtime["revision"] != MLX_VLM_REVISION:
        raise RuntimeError(
            "mlx-vlm revision %s (version %s) is not the pinned revision %s; "
            "install the multimodal extra (pip install -e '.[multimodal]')"
            % (runtime["revision"] or "unknown", runtime["version"], MLX_VLM_REVISION)
        )
    return runtime
