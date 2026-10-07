# SPDX-License-Identifier: MIT
"""Process-global route selections that a live adapter depends on.

Two route selections are process globals that adapters set live after
load: the NAX sorted MoE gather mode (``moe_nax_gather.MODE``, set by the
Flash-Next and Qwen3.6 adapters) and the sorted-MoE rhs pad policy
(``switch_layers._RHS_PAD_POLICY``, set by Flash-Next).  Their receipts,
diagnostics and APCv2 namespaces also read the global, so binding them per
model would not keep a running adapter's identity truthful.  Instead this
registry fails closed (Codex review, flip-isolation 2026-10-02):

* ``claim`` refuses to construct an adapter whose selection would change a
  global that a live adapter depends on, unless the values are equal;
* the setters run only through the claim, and ``rollback`` puts the old
  values back when construction fails, so a failed load leaves nothing;
* ``commit`` records the selections against a weak reference to the
  adapter; ``release`` (adapter ``close``) or garbage collection ends the
  dependency.
"""

from __future__ import annotations

import gc
import threading
import weakref
from typing import Any, Callable, Mapping, Optional

MOE_NAX_GATHER = "moe_nax_gather"
MOE_RHS_PAD_POLICY = "moe_rhs_pad_policy"


class ProcessGlobalConflict(ValueError):
    """A live adapter depends on a different value of a process global."""


_LOCK = threading.RLock()
# id(adapter) -> (weakref to the adapter, owner label, {name: value})
_HOLDERS: dict = {}
_PENDING: dict = {}


def _live(*, include_pending: bool = True) -> list:
    with _LOCK:
        registries = (_HOLDERS, _PENDING) if include_pending else (_HOLDERS,)
        for registry in registries:
            for key, (ref, _, _) in list(registry.items()):
                if ref() is None:
                    del registry[key]
        return [
            (ref(), owner, dict(sel), registry is _PENDING)
            for registry in registries
            for ref, owner, sel in registry.values()
        ]


def live_selections() -> list:
    """``[(owner, {name: value})]`` for every live adapter (diagnostics, tests)."""
    return [(owner, sel) for obj, owner, sel, _ in _live(include_pending=False)
            if obj is not None]


def _conflicts(holder, selections: Mapping[str, Any]) -> list:
    found = []
    for obj, owner, held, pending in _live():
        if obj is None or obj is holder:
            continue
        for name, value in selections.items():
            # Equal live selections may coexist. An in-flight claim must
            # finish first: its rollback could otherwise restore the old
            # process value beneath another adapter's equal selection.
            if name in held and (pending or held[name] != value):
                found.append((owner, name, held[name], value))
    return found


class Claim:
    """One adapter construction's claim on process globals.

    ``selections`` maps a global's name to ``(value, setter)``; ``setter``
    takes the new value and returns the old one, or is None when the adapter
    only depends on the current value without setting it.
    """

    def __init__(
        self,
        holder: Any,
        owner: str,
        selections: Mapping[str, tuple[Any, Optional[Callable[[Any], Any]]]],
    ):
        self.holder = holder
        self.owner = owner
        self.selections = dict(selections)
        self._restore: list = []
        self._committed = False

    def apply(self) -> "Claim":
        values = {name: value for name, (value, _) in self.selections.items()}
        with _LOCK:
            found = _conflicts(self.holder, values)
            if found:
                # An adapter that is already unreachable must not block a load.
                gc.collect()
                found = _conflicts(self.holder, values)
            if found:
                shown = "; ".join(
                    f"{owner} runs {name}={held!r}, this load asks for {asked!r}"
                    for owner, name, held, asked in found
                )
                raise ProcessGlobalConflict(
                    f"{self.owner} conflicts with live or pending process-global "
                    f"route selections ({shown}). Close the live adapter or "
                    f"wait for the pending load before retrying."
                )
            _PENDING[id(self.holder)] = (
                weakref.ref(self.holder), self.owner, values
            )
            try:
                for name, (value, setter) in self.selections.items():
                    if setter is not None:
                        self._restore.append((setter, setter(value)))
            except BaseException:
                self.rollback()
                raise
        return self

    def rollback(self) -> None:
        """Restore every global this claim changed (failed construction)."""
        with _LOCK:
            while self._restore:
                setter, old = self._restore.pop()
                setter(old)
            pending = _PENDING.get(id(self.holder))
            if pending is not None and pending[0]() is self.holder:
                del _PENDING[id(self.holder)]
            if self._committed:
                release(self.holder)
                self._committed = False

    def commit(self) -> None:
        """Record the holder as live with these selections."""
        values = {name: value for name, (value, _) in self.selections.items()}
        with _LOCK:
            pending = _PENDING.get(id(self.holder))
            if pending is None or pending[0]() is not self.holder:
                raise RuntimeError("process-global claim was not applied")
            _HOLDERS[id(self.holder)] = (weakref.ref(self.holder), self.owner, values)
            del _PENDING[id(self.holder)]
            self._restore.clear()
            self._committed = True


def claim(holder, owner, selections) -> Claim:
    """Check and apply ``selections`` for ``holder``; see ``Claim``."""
    return Claim(holder, owner, selections).apply()


def claim_stock_moe(holder, owner: str) -> Claim:
    """Claim the stock sorted-MoE selections (NAX gather off, pad floor).

    For adapters whose model builds mlx2 ``switch_layers`` experts but runs
    neither the NAX gather nor a calibrated pad table: they read both globals
    on every sorted gather (and in receipts), so another live adapter must
    not change them underneath.  Call inside ``guarded_construction`` before
    tensors load, and ``release`` in ``close``.
    """
    from ..runtime.models import moe_nax_gather, switch_layers

    holder._process_claim = claim(
        holder,
        owner,
        {
            MOE_NAX_GATHER: ("off", moe_nax_gather.set_mode),
            MOE_RHS_PAD_POLICY: ("floor", switch_layers.set_pad_policy),
        },
    )
    return holder._process_claim


def guarded_construction(holder, build: Callable[[], None]) -> None:
    """Run ``build``; on failure undo its claim and its ``os.environ`` edits.

    ``build`` stores its claim on ``holder._process_claim``.  On success the
    claim is committed; on any failure its setters are rolled back and the
    process environment the profile pinned is put back as it was, so a
    refused or failed load leaves no global state behind.
    """
    import os

    # Profile application also edits os.environ. Keep the entire construction
    # atomic against another adapter's snapshot, claim, and rollback.
    with _LOCK:
        holder._process_claim = None
        saved = dict(os.environ)
        try:
            build()
        except BaseException:
            pending = getattr(holder, "_process_claim", None)
            if pending is not None:
                pending.rollback()
            for name in tuple(os.environ):
                if name not in saved:
                    del os.environ[name]
            for name, value in saved.items():
                if os.environ.get(name) != value:
                    os.environ[name] = value
            raise
        pending = getattr(holder, "_process_claim", None)
        if pending is not None:
            pending.commit()


def release(holder) -> None:
    """The holder no longer runs: its selections stop binding other loads."""
    with _LOCK:
        entry = _HOLDERS.get(id(holder))
        if entry is not None and entry[0]() in (holder, None):
            del _HOLDERS[id(holder)]


__all__ = [
    "Claim",
    "MOE_NAX_GATHER",
    "MOE_RHS_PAD_POLICY",
    "ProcessGlobalConflict",
    "claim",
    "claim_stock_moe",
    "guarded_construction",
    "live_selections",
    "release",
]
