"""Prompt-state boundaries: where an exact prefill snapshot is worth taking.

A checkpointed hybrid target (recurrent + attention state) can only resume a
prompt at a position where an exact state snapshot already exists.  Three
independent reasons to take one share a single plan so a position is captured
once and chunked prefill stops exactly there:

* ``ROLLING``   -- disposable progress points every ``interval`` tokens.  A lane
  publishes each one immediately and retires its previous one; the last is
  retired when the committed prompt boundary is published.  A cancelled,
  failed or preempted lane keeps its latest point so a retry resumes there.
* ``INTERIOR``  -- budgeted power-of-two lattice points that survive the
  request (omlx#3456); published when the prompt finishes.
* ``JUNCTION``  -- the branch point of a longer cached path (item 6).

Design reference: Splash ``Request::StateBoundary`` (Engine.hpp, rev f58d36dd,
Apache-2.0); see ``provenance/splash-07-rolling-prefill-ckpt.json``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Iterable, Optional, Sequence, Tuple


class BoundaryPurpose(IntEnum):
    # Dedup keeps the maximum purpose at one position.
    ROLLING = 0
    INTERIOR = 1
    JUNCTION = 2


RETENTION_ROLE = {
    BoundaryPurpose.ROLLING: "prefill_rolling",
    BoundaryPurpose.INTERIOR: "interior_checkpoint",
    BoundaryPurpose.JUNCTION: "junction",
}


@dataclass(frozen=True)
class StateBoundary:
    position: int  # len(history) covered by the snapshot
    purpose: BoundaryPurpose


def _checked_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def plan_state_boundaries(
    *,
    prompt_tokens: int,
    cached_tokens: int,
    interior: Iterable[int] = (),
    junction: Optional[int] = None,
    rolling_interval: int = 0,
) -> Tuple[StateBoundary, ...]:
    """Merge every boundary source into one sorted, deduplicated plan.

    Only positions strictly inside ``(cached_tokens, prompt_tokens - 1)`` are
    kept: the cached prefix already exists, and the final prompt token is the
    generation boundary the committed prompt checkpoint covers.  Rolling
    positions are absolute multiples of ``rolling_interval`` so concurrent
    requests sharing a prefix land on the same keys.
    """
    prompt_tokens = _checked_int("prompt_tokens", prompt_tokens)
    cached_tokens = _checked_int("cached_tokens", cached_tokens)
    rolling_interval = _checked_int("rolling_interval", rolling_interval)
    if prompt_tokens < 0 or cached_tokens < 0 or rolling_interval < 0:
        raise ValueError("state boundary geometry must be non-negative")
    limit = prompt_tokens - 1
    purposes = {}

    def add(position, purpose):
        position = _checked_int("state boundary position", position)
        if cached_tokens < position < limit:
            purposes[position] = max(purposes.get(position, purpose), purpose)

    if rolling_interval:
        position = (cached_tokens // rolling_interval + 1) * rolling_interval
        while position < limit:
            add(position, BoundaryPurpose.ROLLING)
            position += rolling_interval
    for position in interior:
        add(position, BoundaryPurpose.INTERIOR)
    if junction is not None:
        add(junction, BoundaryPurpose.JUNCTION)
    return tuple(
        StateBoundary(position, BoundaryPurpose(purposes[position]))
        for position in sorted(purposes)
    )


def budget_state_boundaries(
    bounds: Sequence[StateBoundary],
    *,
    available_bytes,
    cache_projection: Optional[Callable[[int], float]],
) -> Tuple[Tuple[StateBoundary, ...], int]:
    """Keep the boundaries that fit measured headroom, by priority.

    JUNCTION first, then INTERIOR deepest-first (stopping at the first that
    does not fit, exactly as ``budget_interior_checkpoint_positions``), then
    ROLLING nearest-first.  A lane holds one rolling snapshot at a time (each
    publication retires the previous one), so rolling is charged as a single
    slot sized by the deepest rolling position kept.
    """
    bounds = tuple(bounds)
    if (
        isinstance(available_bytes, bool)
        or not isinstance(available_bytes, (int, float))
        or not math.isfinite(available_bytes)
        or available_bytes < 0
    ):
        raise ValueError("available checkpoint bytes must be finite and nonnegative")
    if not bounds or not callable(cache_projection):
        return (), 0

    def project(position):
        projected = cache_projection(position)
        if (
            isinstance(projected, bool)
            or not isinstance(projected, (int, float))
            or not math.isfinite(projected)
            or projected < 0
        ):
            raise ValueError("checkpoint cache projection must be finite and nonnegative")
        return int(projected)

    selected = []
    charged = 0
    for bound in bounds:
        if bound.purpose == BoundaryPurpose.JUNCTION:
            projected = project(bound.position)
            if charged + projected <= available_bytes:
                selected.append(bound)
                charged += projected
    for bound in reversed(bounds):
        if bound.purpose != BoundaryPurpose.INTERIOR:
            continue
        projected = project(bound.position)
        if charged + projected > available_bytes:
            break
        selected.append(bound)
        charged += projected
    rolling_slot = 0
    for bound in bounds:
        if bound.purpose != BoundaryPurpose.ROLLING:
            continue
        projected = max(rolling_slot, project(bound.position))
        if charged + projected > available_bytes:
            break
        selected.append(bound)
        rolling_slot = projected
    charged += rolling_slot
    return tuple(sorted(selected, key=lambda bound: bound.position)), charged
