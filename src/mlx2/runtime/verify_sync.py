# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from __future__ import annotations
import os
import threading
from contextlib import contextmanager

_LOCAL = threading.local()
_MAX_ROUNDS = 4096


def _env_enabled() -> bool:
    return os.getenv("MLX_LM_SYNC_TRACE") == "1"


def _state():
    state = getattr(_LOCAL, "state", None)
    if state is None:
        state = {"forced": False, "active": False, "rounds": [], "current": None}
        _LOCAL.state = state
    return state


def _enabled(state) -> bool:
    return state["forced"] or _env_enabled()


@contextmanager
def verify_sync_round():
    """Collect call-site counts for one verify transaction."""
    state = _state()
    if not _enabled(state):
        yield
        return
    if state["active"]:
        yield
        return
    counts = {}
    state["active"] = True
    state["current"] = counts
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        if len(state["rounds"]) == _MAX_ROUNDS:
            state["rounds"].pop(0)
        state["rounds"].append(
            {
                "round": len(state["rounds"]),
                "failed": failed,
                "total": sum(counts.values()),
                "sites": dict(sorted(counts.items())),
            }
        )
        state["active"] = False
        state["current"] = None


def record_verify_sync(site: str) -> None:
    """Record one named host synchronization when a round is active."""
    state = getattr(_LOCAL, "state", None)
    if state is None or not state["active"]:
        return
    counts = state["current"]
    counts[site] = counts.get(site, 0) + 1
