"""Per-request prefill slice schedules against the route's planned grid.

Prefill state is a deterministic function of the slice boundaries that
produced it, not of the tokens alone: the GDN chunked recurrence (and any
other order-dependent reduction) sums in an order that follows the slice
ends, so a 2048-row prompt prefilled as 4x512 and as 32x64 publishes
different bits (measured 2026-10-28 on Qwen3.8-27B and Sage, see
``docs/experiments/prefill-chunk-invariance-20261028``).  Same schedule twice
is bit-exact.

The *planned grid* of a request is the slicing an idle server would apply
to it from its start position: the route's prefill step (prompt-length
autoscale or a declared adapter plan evaluated with no decoding lanes), the
static depth budget, and the request's own planned cuts (interior, rolling,
junction, generation-prompt and media boundaries), each slice clamped to its
segment.  Everything a live scheduler adds on top of that plan (adaptive
slices, decode-time fairness caps, one-slice contention bounds, shared
decode-first budgets, mixed prefill+decode forwards, live memory caps,
admission halving, recovery re-prefill) is load dependent.

A schedule records every executed slice end and the first position at which
an executed slice departed from the plan.  Slice boundaries strictly below a
position decide the state at that position, so a checkpoint at ``p`` is
schedule-exact iff no deviation happened or ``p <= deviation``; once a slice
departs, every later state carries it, even if later slices realign.
"""

from __future__ import annotations

from typing import Optional

SCHEMA = "mlx2.prefill-slice-schedule.v1"
# Boundaries are evidence, not the decision input (``deviation`` is), so the
# list is bounded: a 128K prompt at 64-row slices still fits.
MAX_RECORDED_BOUNDARIES = 4096


class PrefillSliceSchedule:
    """One request's executed prefill slice ends and its first deviation."""

    __slots__ = (
        "start",
        "covered",
        "boundaries",
        "truncated",
        "slices",
        "deviation",
        "reason",
        "planned_rows",
        "executed_rows",
    )

    def __init__(self, start: int):
        self.start = int(start)
        self.covered = int(start)
        self.boundaries = []
        self.truncated = False
        self.slices = 0
        self.deviation: Optional[int] = None
        self.reason: Optional[str] = None
        self.planned_rows: Optional[int] = None
        self.executed_rows: Optional[int] = None

    def record(self, start: int, executed: int, planned: int, reason=None) -> None:
        """Note one executed slice ``[start, start + executed)``.

        ``planned`` is the row count the idle plan would run from ``start``;
        ``reason`` names a load-dependent mechanism that changed the forward
        even at the planned width (a mixed prefill+decode forward).
        """
        start, executed, planned = int(start), int(executed), int(planned)
        if executed <= 0:
            return
        if self.deviation is None:
            if start != self.covered:
                # A gap or overlap means a slice ran outside this record
                # (recovery or a path that does not report): fail closed.
                self._deviate(min(start, self.covered), "untracked_slice", planned, executed)
            elif reason is not None or executed != planned:
                self._deviate(start, reason or "width", planned, executed)
        self.covered = start + executed
        self.slices += 1
        if len(self.boundaries) < MAX_RECORDED_BOUNDARIES:
            self.boundaries.append(self.covered)
        else:
            self.truncated = True

    def deviate(self, position: int, reason: str) -> None:
        """Taint every state after ``position`` (no slice width to compare)."""
        if self.deviation is None or int(position) < self.deviation:
            self._deviate(int(position), reason, None, None)

    def _deviate(self, position, reason, planned, executed) -> None:
        self.deviation = int(position)
        self.reason = str(reason)
        self.planned_rows = planned
        self.executed_rows = executed

    def changed_at(self, position: int) -> bool:
        return self.deviation is not None and int(position) > self.deviation

    def receipt(self) -> dict:
        return {
            "schema": SCHEMA,
            "start": self.start,
            "covered": self.covered,
            "slices": self.slices,
            "boundaries": list(self.boundaries),
            "boundaries_truncated": self.truncated,
            "planned": self.deviation is None,
            "first_deviation": self.deviation,
            "deviation_reason": self.reason,
            "deviation_planned_rows": self.planned_rows,
            "deviation_executed_rows": self.executed_rows,
        }


def record_slice(registry, uid, start, executed, planned, reason=None) -> None:
    """Record one slice for ``uid`` in a ``{uid: PrefillSliceSchedule}`` map.

    The first slice opens the record at its own start (a warm request starts
    at its cached offset).  Sync-free host bookkeeping.
    """
    if registry is None or int(executed) <= 0:
        return
    uid = int(uid)
    schedule = registry.get(uid)
    if schedule is None:
        schedule = registry[uid] = PrefillSliceSchedule(start)
    schedule.record(start, executed, planned, reason)


def mark_deviation(registry, uid, position, reason) -> None:
    if registry is None:
        return
    uid = int(uid)
    schedule = registry.get(uid)
    if schedule is None:
        schedule = registry[uid] = PrefillSliceSchedule(position)
    schedule.deviate(position, reason)


def schedule_receipt(registry, uid, *, pop=False):
    if not registry:
        return None
    schedule = registry.pop(int(uid), None) if pop else registry.get(int(uid))
    return None if schedule is None else schedule.receipt()


def changed_at(receipt, position: int) -> bool:
    """Whether state at ``position`` departs from the planned grid."""
    if not receipt:
        return False
    deviation = receipt.get("first_deviation")
    return deviation is not None and int(position) > int(deviation)
