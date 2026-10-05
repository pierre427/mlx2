"""Opt-in, observation-only timing for one ragged verification round.

The default path has no observer and does not add evaluation fences.  Setting
``MLX2_RAGGED_VERIFY_TIMING=1`` creates one request-local observer and stage
helpers may materialize their outputs through the supplied evaluator.  Such a
receipt is diagnostic attribution, never a performance result.
"""

from __future__ import annotations

import contextvars
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterable


_CURRENT = contextvars.ContextVar("mlx2_ragged_verify_observer", default=None)


@dataclass
class RaggedVerifyObserver:
    evaluator: Callable[..., object] | None = None
    stages: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, name: str, *, elapsed_ns: int, rows: int) -> None:
        entry = self.stages.setdefault(
            name, {"calls": 0, "rows": 0, "elapsed_ns": 0}
        )
        entry["calls"] += 1
        entry["rows"] += int(rows)
        entry["elapsed_ns"] += int(elapsed_ns)

    def materialize(self, values: Iterable[object]) -> None:
        values = tuple(value for value in values if value is not None)
        if self.evaluator is not None and values:
            self.evaluator(*values)

    def receipt(self) -> dict:
        return {
            "schema": "mlx2.ragged-verify-observation.v1",
            "diagnostic_only": True,
            "evaluation_fences_inserted": self.evaluator is not None,
            "stages": {name: dict(values) for name, values in self.stages.items()},
            "total_elapsed_ns": sum(
                values["elapsed_ns"] for values in self.stages.values()
            ),
        }


def current_observer() -> RaggedVerifyObserver | None:
    return _CURRENT.get()


@contextmanager
def activate(observer: RaggedVerifyObserver | None):
    token = _CURRENT.set(observer)
    try:
        yield observer
    finally:
        _CURRENT.reset(token)


@contextmanager
def observed_stage(name: str, *, rows: int):
    observer = current_observer()
    if observer is None:
        yield None
        return
    started = time.perf_counter_ns()
    values = []

    def materialize(*items):
        values.extend(items)

    try:
        yield materialize
        observer.materialize(values)
    finally:
        observer.record(
            name, elapsed_ns=time.perf_counter_ns() - started, rows=rows
        )


def observer_from_environment(*, evaluator=None) -> RaggedVerifyObserver | None:
    if os.environ.get("MLX2_RAGGED_VERIFY_TIMING", "0") != "1":
        return None
    return RaggedVerifyObserver(evaluator=evaluator)


__all__ = [
    "RaggedVerifyObserver",
    "activate",
    "current_observer",
    "observed_stage",
    "observer_from_environment",
]
