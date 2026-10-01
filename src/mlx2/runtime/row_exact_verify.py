# SPDX-License-Identifier: Apache-2.0
"""Model-neutral state of a row-exact speculative verify window.

A row-exact verify window computes every verify row with the arithmetic of
the one-token decode step at the same position, so greedy output with MTP on
is byte-identical to MTP off.  The model adapter that owns the arithmetic
opens the window (``window``) around its verify forward; shared runtime code
that splits work across rows (for example a segmented cache's attention
consumer) asks ``active()`` and runs its per-row form.  Outside a window
nothing changes.

Each window records which implementation every stage used.  A window is
row-exact only when no stage fell back to a width-dependent path
(``fail``); a caller that labels output "row-exact" must check ``exact``.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Optional

_WINDOW: ContextVar[Optional["Window"]] = ContextVar(
    "mlx2_row_exact_verify_window", default=None
)


class Window:
    __slots__ = ("rows", "stages", "failures")

    def __init__(self, rows: int):
        self.rows = int(rows)
        # stage -> route -> calls
        self.stages: Dict[str, Dict[str, int]] = {}
        self.failures: Dict[str, int] = {}

    def note(self, stage: str, route: str, count: int = 1) -> None:
        routes = self.stages.setdefault(stage, {})
        routes[route] = routes.get(route, 0) + int(count)

    def fail(self, reason: str) -> None:
        self.failures[reason] = self.failures.get(reason, 0) + 1

    @property
    def exact(self) -> bool:
        return not self.failures


def current() -> Optional[Window]:
    return _WINDOW.get()


def active() -> bool:
    return _WINDOW.get() is not None


@contextmanager
def window(record: Window):
    token = _WINDOW.set(record)
    try:
        yield record
    finally:
        _WINDOW.reset(token)
