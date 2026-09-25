# SPDX-License-Identifier: MIT
"""Record the environment a tensor module saw when it was imported.

Several model modules read their ``MLX_QWEN4_*`` / ``MLX_GDN_*`` selections
once, at import. An adapter that pins a serving profile by editing
``os.environ`` only takes effect if it runs first; if the module was already
imported under a different environment, the profile is silently ignored (the
2026-09-25 triage lost fused GDN this way). Modules call ``snapshot`` before
reading their flags, and adapters call ``assert_profile_applied`` after
pinning, so a late profile fails closed instead of running a different route.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, Mapping, Optional

PREFIXES = ("MLX_QWEN4_", "MLX_GDN_")

_SNAPSHOTS: Dict[str, Dict[str, str]] = {}


def _scoped(environ: Mapping[str, str], prefixes: Iterable[str]) -> Dict[str, str]:
    prefixes = tuple(prefixes)
    return {k: v for k, v in environ.items() if k.startswith(prefixes)}


def snapshot(module_name: str, environ: Optional[Mapping[str, str]] = None) -> None:
    """Record the scoped environment ``module_name`` is being imported under."""
    _SNAPSHOTS[module_name] = _scoped(os.environ if environ is None else environ, PREFIXES)


class ImportOrderError(RuntimeError):
    """A tensor module was imported before its adapter pinned the profile."""


def assert_profile_applied(owner: str, environ: Optional[Mapping[str, str]] = None) -> None:
    """Refuse when an already imported module saw a different scoped environment."""
    current = _scoped(os.environ if environ is None else environ, PREFIXES)
    for module_name, seen in _SNAPSHOTS.items():
        if module_name not in sys.modules:
            continue
        differs = sorted(
            name for name in set(seen) | set(current) if seen.get(name) != current.get(name)
        )
        if differs:
            shown = ", ".join(
                f"{name}: {seen.get(name)!r} at import, {current.get(name)!r} now"
                for name in differs[:6]
            )
            more = f" (+{len(differs) - 6} more)" if len(differs) > 6 else ""
            raise ImportOrderError(
                f"{module_name} was imported before {owner} pinned its environment, so its "
                f"import-time flags do not match the profile: {shown}{more}. Construct the "
                f"adapter before importing mlx2.runtime model modules, or toggle features "
                f"through the module setters after load."
            )


__all__ = ["ImportOrderError", "PREFIXES", "assert_profile_applied", "snapshot"]
