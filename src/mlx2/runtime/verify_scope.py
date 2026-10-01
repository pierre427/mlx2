# SPDX-License-Identifier: Apache-2.0
"""Marks the target forward of a speculative verify window.

The scheduler opens the scope around the verify backbone forward (the rows
are the pending token plus the drafts of each lane). A model adapter may
read ``active()`` to give verify rows a different, explicitly selected
implementation (for example the Flash-Next routed MoE row window); outside
the scope nothing changes. Model-neutral: it carries no model identity.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_ACTIVE: ContextVar[bool] = ContextVar("mlx2_speculative_verify_forward", default=False)


def active() -> bool:
    return _ACTIVE.get()


@contextmanager
def verify_forward():
    token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
