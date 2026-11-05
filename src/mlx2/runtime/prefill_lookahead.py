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
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrefillLookahead:
    rows_after: tuple[int, ...]
    step: int


@dataclass(frozen=True)
class StateOnlyPrefillOutput:
    """An intermediate chunk that produced no logits; ``hidden`` is [batch, n]."""

    hidden: Any
    last_row_only: bool = True


_LOOKAHEAD: ContextVar[PrefillLookahead | None] = ContextVar(
    "mlx2_prefill_lookahead", default=None
)


@contextlib.contextmanager
def prefill_lookahead(rows_after: Sequence[int], *, step: int):
    value = PrefillLookahead(tuple(max(0, int(r)) for r in rows_after), int(step))
    token = _LOOKAHEAD.set(value)
    try:
        yield value
    finally:
        _LOOKAHEAD.reset(token)


def current() -> PrefillLookahead | None:
    return _LOOKAHEAD.get()
