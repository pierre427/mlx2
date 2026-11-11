"""Production serving boundary: the reserved port and release-tree identity.

Production services run from immutable release trees that
``scripts/mlx2_deploy.py release`` builds: a ``git archive`` of one commit with
a ``RELEASE.json`` at its root. The production port belongs to those trees; a
development checkout or worktree must not bind it, so a test server can never
take over (or be mistaken for) the live service. A sanctioned takeover, such as
``scripts/run_with_service_paused.py`` running work while the service is
stopped, sets ``MLX2_ALLOW_PRODUCTION_PORT=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PRODUCTION_PORT = 8282
RELEASE_FILE = "RELEASE.json"
ALLOW_ENV = "MLX2_ALLOW_PRODUCTION_PORT"


def tree_root() -> Path:
    """Root of the tree this package was imported from (the parent of src/)."""
    return Path(__file__).resolve().parents[2]


def release_info(root: Path | None = None) -> dict | None:
    """The tree's RELEASE.json when it is a release tree, else None."""
    path = (root or tree_root()) / RELEASE_FILE
    try:
        info = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    sha = info.get("sha") if isinstance(info, dict) else None
    if not isinstance(sha, str) or len(sha) != 40:
        return None
    if path.parent.name != sha:  # a copy or a stray file in a dev tree is not a release
        return None
    return info


def port_refusal(port: int, *, allow: bool = False, root: Path | None = None) -> str | None:
    """Why this process may not bind ``port``, or None when it may."""
    if port != PRODUCTION_PORT or allow or os.environ.get(ALLOW_ENV) == "1":
        return None
    if release_info(root) is not None:
        return None
    return (f"port {PRODUCTION_PORT} is reserved for production release trees "
            f"(scripts/mlx2_deploy.py); this server runs from {root or tree_root()}, which is not a "
            f"release. Use another port, or pass --allow-production-port / set {ALLOW_ENV}=1 "
            "while the service is paused.")
