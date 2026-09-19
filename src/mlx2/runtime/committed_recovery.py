"""Request-private recovery checkpoints for speculative execution routes.

The lifecycle is route-neutral, while snapshot and restore semantics remain
owned by the route.  A checkpoint may contain only state from a proven
committed token boundary; callers keep proposed suffixes outside this object.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class RecoveryCheckpointMismatch(ValueError):
    """A checkpoint was offered to the wrong route, revision, or boundary."""


@dataclass(frozen=True)
class CommittedRecoveryCheckpoint:
    route: str
    revision: str
    boundary: int
    generation: int
    _state: Any
    _restore: Callable[[Any], Any]

    def __post_init__(self) -> None:
        if not self.route or not self.revision:
            raise ValueError("recovery checkpoint route and revision are required")
        if self.boundary < 0 or self.generation < 1:
            raise ValueError("recovery checkpoint coordinates are invalid")
        if not callable(self._restore):
            raise TypeError("recovery checkpoint restore callback is required")

    def restore(self, *, route: str, revision: str, boundary: int) -> Any:
        expected = (self.route, self.revision, self.boundary)
        actual = (str(route), str(revision), int(boundary))
        if actual != expected:
            raise RecoveryCheckpointMismatch(
                "committed recovery checkpoint mismatch: "
                f"expected route/revision/boundary {expected!r}, got {actual!r}"
            )
        return self._restore(self._state)


class CommittedRecoverySlot:
    """One generation-stamped committed checkpoint owned by a request lane."""

    def __init__(self) -> None:
        self._checkpoint: CommittedRecoveryCheckpoint | None = None
        self._generation = 0
        self._lock = threading.RLock()
        self._counts = {
            "captures": 0,
            "restores": 0,
            "misses": 0,
            "mismatches": 0,
            "invalidations": 0,
        }

    def capture(
        self,
        *,
        route: str,
        revision: str,
        boundary: int,
        value: Any,
        snapshot: Callable[[Any], Any],
        restore: Callable[[Any], Any],
    ) -> CommittedRecoveryCheckpoint:
        if not callable(snapshot) or not callable(restore):
            raise TypeError("recovery snapshot and restore callbacks are required")
        frozen = snapshot(value)
        with self._lock:
            self._generation += 1
            checkpoint = CommittedRecoveryCheckpoint(
                str(route),
                str(revision),
                int(boundary),
                self._generation,
                frozen,
                restore,
            )
            self._checkpoint = checkpoint
            self._counts["captures"] += 1
            return checkpoint

    def restore(self, *, route: str, revision: str, boundary: int) -> Any | None:
        with self._lock:
            checkpoint = self._checkpoint
            if checkpoint is None:
                self._counts["misses"] += 1
                return None
            try:
                value = checkpoint.restore(
                    route=route, revision=revision, boundary=boundary
                )
            except RecoveryCheckpointMismatch:
                self._counts["mismatches"] += 1
                raise
            self._counts["restores"] += 1
            return value

    def invalidate(self) -> None:
        with self._lock:
            if self._checkpoint is not None:
                self._checkpoint = None
                self._generation += 1
                self._counts["invalidations"] += 1

    def status(self) -> dict[str, Any]:
        with self._lock:
            checkpoint = self._checkpoint
            return {
                **self._counts,
                "generation": self._generation,
                "available": checkpoint is not None,
                "route": None if checkpoint is None else checkpoint.route,
                "revision": None if checkpoint is None else checkpoint.revision,
                "boundary": None if checkpoint is None else checkpoint.boundary,
            }
