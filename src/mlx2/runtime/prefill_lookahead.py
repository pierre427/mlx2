"""What the prompt loop guarantees about the rows after one prefill chunk.

``PromptProcessingBatch.prompt`` scopes ``prefill_lookahead`` around exactly
one plain prompt-chunk forward.  ``rows_after[i]`` is the number of lane
``i``'s prompt rows that the same call will still feed after this chunk, and
``step`` is the chunk width every later chunk of the call uses (the last one
may be shorter).  Rows a later call adds are not counted, so the value is a
lower bound on what follows: a model may rely on it, never on more.

A model that uses the lookahead may return ``StateOnlyPrefillOutput`` from a
chunk whose cache state stays incomplete until a later chunk of the same call
runs (DeepSeek-V4.1 CED skipping its decoder layers).  The loop accepts that
output only when every live lane has rows after the chunk, checks the given
hidden row for finiteness instead of logits, and the call's own final chunk
always returns ordinary logits, so the boundary APCv2 publishes is complete.

Outside a scope ``current()`` is None and nothing changes.

``width`` (the rows of the chunk this scope covers) and ``planned_after``
(prompt rows the scheduler plans to feed each lane in later calls) are
allocation hints only: ``allocation_rows_after`` lets an append-only cache
size one growth for the rest of the planned prompt instead of growing in
256-row steps (one whole-buffer copy per chunk).  Nothing may rely on them
for correctness; an unmet plan only leaves spare capacity, which is never
read and never exceeds the admitted prompt.

``MLX2_KV_PLANNED_GROWTH`` (read once at import; default ``1``) is the kill
switch: ``0`` restores the stepped policy, so ``allocation_rows_after`` plans
nothing, the scheduler scopes no plan, and the planned-capacity trims on batch
filter/extend/extract stay off.  ``planned_growth_enabled()`` reports the arm
(``scheduler.kv_planned_growth`` in the status receipt).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


def _planned_growth_from_env() -> bool:
    raw = os.environ.get("MLX2_KV_PLANNED_GROWTH", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


KV_PLANNED_GROWTH = _planned_growth_from_env()


def planned_growth_enabled() -> bool:
    """Whether prompt K/V growth follows the planned span (the kill switch)."""
    return KV_PLANNED_GROWTH


@dataclass(frozen=True)
class PrefillLookahead:
    rows_after: tuple[int, ...]
    step: int
    next_tokens: tuple[tuple[int, ...], ...] | None = None
    width: int | None = None
    planned_after: tuple[int, ...] | None = None


@dataclass(frozen=True)
class StateOnlyPrefillOutput:
    """An intermediate chunk that produced no logits; ``hidden`` is [batch, n]."""

    hidden: Any
    last_row_only: bool = True


_LOOKAHEAD: ContextVar[PrefillLookahead | None] = ContextVar(
    "mlx2_prefill_lookahead", default=None
)


@contextlib.contextmanager
def prefill_lookahead(rows_after: Sequence[int], *, step: int, next_tokens=None,
                      width: int | None = None, planned_after: Sequence[int] | None = None):
    rows = tuple(max(0, int(r)) for r in rows_after)
    tokens = None if next_tokens is None else tuple(tuple(int(t) for t in lane) for lane in next_tokens)
    if tokens is not None and (len(tokens) != len(rows) or any(len(t) > min(r, step) for t, r in zip(tokens, rows))):
        raise ValueError("prefill token lookahead exceeds the guaranteed next chunk")
    planned = None if planned_after is None else tuple(max(0, int(r)) for r in planned_after)
    if planned is not None and len(planned) != len(rows):
        raise ValueError("prefill allocation plan must name one entry per lane")
    value = PrefillLookahead(rows, int(step), tokens,
                             None if width is None else int(width), planned)
    token = _LOOKAHEAD.set(value)
    try:
        yield value
    finally:
        _LOOKAHEAD.reset(token)


def current() -> PrefillLookahead | None:
    return _LOOKAHEAD.get()


def allocation_rows_after(rows: int, batch: int) -> int:
    """Rows to reserve past an append of ``rows`` rows for ``batch`` lanes.

    Nonzero only inside a single-lane prompt-chunk scope that names its
    ``width``, for a B=1 append of exactly that width (the prompt cache of
    the chunk; a drafter or scratch cache appending other shapes keeps its
    own growth).  The value is the lane's remaining prompt rows: this call's
    later chunks plus the scheduler's plan.  Multi-lane scopes plan nothing:
    a batch shares one capacity, so a short lane would carry its longest
    sibling's plan into decode when it is split off.
    """
    if not KV_PLANNED_GROWTH:
        return 0
    value = _LOOKAHEAD.get()
    if (value is None or value.width is None or rows != value.width
            or batch != 1 or len(value.rows_after) != 1):
        return 0
    planned = value.planned_after[0] if value.planned_after else 0
    return value.rows_after[0] + planned


_PLANNED: ContextVar[tuple[int, ...] | None] = ContextVar(
    "mlx2_prefill_planned_rows", default=None
)


@contextlib.contextmanager
def planned_prompt_rows(rows_after: Sequence[int] | None):
    """Scope a scheduler's per-lane plan of prompt rows after one prompt call.

    Read by the prompt loop into ``planned_after``; an allocation hint only.
    ``None`` plans nothing.
    """
    planned = None if rows_after is None else tuple(max(0, int(r)) for r in rows_after)
    token = _PLANNED.set(planned)
    try:
        yield planned
    finally:
        _PLANNED.reset(token)


def planned_rows_for(lanes: int) -> tuple[int, ...] | None:
    """The scoped plan when it names exactly ``lanes`` lanes, else None."""
    value = _PLANNED.get()
    return value if value is not None and len(value) == lanes else None
