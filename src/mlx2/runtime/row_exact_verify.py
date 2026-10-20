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
from dataclasses import dataclass
from typing import Dict, Optional

_WINDOW: ContextVar[Optional["Window"]] = ContextVar(
    "mlx2_row_exact_verify_window", default=None
)
_VALID_ROWS = ContextVar("mlx2_exact_valid_rows", default=None)
_STAGE_ROWS = ContextVar("mlx2_exact_stage_rows", default=None)


class StageRowTelemetry:
    """Launch-local logical/physical row accounting with no tensor values."""

    def __init__(self):
        self._rows = {}

    def record(self, stage, *, logical_rows, physical_rows):
        if (
            not isinstance(stage, str)
            or not stage
            or type(logical_rows) is not int
            or type(physical_rows) is not int
            or not 0 <= logical_rows <= physical_rows
        ):
            raise ValueError("stage row telemetry requires valid logical geometry")
        entry = self._rows.setdefault(
            stage,
            {"calls": 0, "logical_rows": 0, "physical_rows": 0, "padding_rows": 0},
        )
        entry["calls"] += 1
        entry["logical_rows"] += logical_rows
        entry["physical_rows"] += physical_rows
        entry["padding_rows"] += physical_rows - logical_rows

    def receipt(self):
        return {
            "schema": "mlx2.ast-stage-row-telemetry.v1",
            "scope": "sum_over_observed_stage_invocations",
            "stages": {
                name: dict(values) for name, values in sorted(self._rows.items())
            },
        }


@contextmanager
def collect_stage_rows():
    """Collect geometry for one target forward without changing execution."""

    telemetry = StageRowTelemetry()
    token = _STAGE_ROWS.set(telemetry)
    try:
        yield telemetry
    finally:
        _STAGE_ROWS.reset(token)


@contextmanager
def valid_rows(lengths, width):
    """Adapter-opted exact row compaction, scoped to one physical forward.

    Only row-independent stage mixins consume this scope. Attention/GDN/cache
    retain the rectangular ABI and authoritative per-request lengths.
    """
    lengths = tuple(lengths)
    if not lengths or any(type(n) is not int or not 1 <= n <= width for n in lengths):
        raise ValueError("exact valid-row scope has invalid lengths")
    value = None
    if any(n != width for n in lengths):
        import mlx.core as mx
        indices = [lane * width + position for lane, n in enumerate(lengths) for position in range(n)]
        value = (len(lengths), width, len(indices), mx.array(indices, mx.int32))
    token = _VALID_ROWS.set(value)
    try:
        yield
    finally:
        _VALID_ROWS.reset(token)


@dataclass(frozen=True)
class CompactedExactRows:
    values: object
    geometry: tuple

    def restore(self, output):
        import mlx.core as mx
        batch, width, live, indices = self.geometry
        if output.shape[:2] != (1, live):
            raise ValueError("exact compacted stage changed its token-row geometry")
        result = mx.zeros((batch * width, *output.shape[2:]), dtype=output.dtype)
        result[indices] = output.reshape(live, *output.shape[2:])
        return result.reshape(batch, width, *output.shape[2:])


def compacts_valid_rows(module) -> bool:
    """Whether ``module`` is a stage that consumes the valid-row scope
    (declared by the row-independent mixins, never inferred from shape)."""
    return bool(getattr(module, "_mlx2_compacts_valid_rows", False))


def compact_rows(value, *, stage):
    scope = _VALID_ROWS.get()
    record = current()
    if scope is None or record is None or value.ndim < 3 or tuple(value.shape[:2]) != scope[:2]:
        return None
    import mlx.core as mx
    batch, width, live, indices = scope
    telemetry = _STAGE_ROWS.get()
    if telemetry is not None:
        telemetry.record(stage, logical_rows=live, physical_rows=batch * width)
    values = mx.take(value.reshape(batch * width, *value.shape[2:]), indices, axis=0)
    record.note("valid_row_compaction", stage + "_executed", live)
    record.note("valid_row_compaction", stage + "_padding_skipped", batch * width - live)
    return CompactedExactRows(values[None], scope)


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
def window(record: Optional[Window]):
    """Open ``record`` for the block; ``None`` runs the block outside any
    window (for example a parity probe issued from inside one)."""
    token = _WINDOW.set(record)
    try:
        yield record
    finally:
        _WINDOW.reset(token)
