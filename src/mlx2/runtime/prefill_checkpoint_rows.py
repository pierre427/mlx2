"""Project only each lane's checkpoint row through the vocabulary head.

A prompt chunk's logits are discarded: prompt processing samples nothing, and
the first generated token comes from the decode step that follows.  The
prefill checkpoint guard (``generate._validate_prefill_checkpoint_output``)
still needs one row per lane -- the last real row, the one whose state the
chunk checkpoint commits -- to refuse non-finite state before it can be
published.  MLX does not push a slice down through a matmul, so indexing that
row out of the forward's ``[batch, sequence, vocabulary]`` logits evaluates
the whole ``sequence x vocabulary`` head projection on every chunk (sweep
2026-10-12 L2-02: about 594 MiB per row at 2048 x 151,936 bf16).

``prefill_checkpoint_rows`` scopes a request around exactly one prompt-chunk
forward.  A model's ``__call__`` passes the hidden state it is about to
project through the vocabulary head to ``select_checkpoint_rows``; while a
request is open, that returns one row per lane (``[batch, 1, hidden]``), so
the head and anything after it (scaling, soft-capping) run on ``batch`` rows
instead of ``batch x sequence``.  The request records that it was applied,
and only then does prompt processing treat the forward's output as
last-row-only.  A model that does not call the helper, or whose hidden state
does not have the request's ``[batch, width, hidden]`` geometry, is left
untouched and keeps the full-logits validation.

Outside a request the helper is the identity, so decode, verification,
drafting and every other forward are unchanged.  The selected row may round
differently from the same row of a full-width head (the head runs at a
different row count), which is immaterial: the guard checks finiteness only,
and prompt-chunk logits never reach a sampler.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from contextvars import ContextVar

import mlx.core as mx


class _CheckpointRowRequest:
    __slots__ = ("applied", "rows", "width")

    def __init__(self, rows: tuple[int, ...], width: int):
        self.rows = rows
        self.width = width
        self.applied = 0


_REQUEST: ContextVar[_CheckpointRowRequest | None] = ContextVar(
    "mlx2_prefill_checkpoint_rows", default=None
)


@contextlib.contextmanager
def prefill_checkpoint_rows(chunk_lengths: Sequence[int], *, width: int):
    """Open a checkpoint-row request for one prompt-chunk forward.

    ``chunk_lengths[i]`` is lane ``i``'s real token count in this chunk;
    lanes with no real tokens select row 0, which the guard skips.  Yields the
    request; ``request.applied`` counts how many head inputs were narrowed.
    """
    width = int(width)
    rows = tuple(max(0, min(int(length), width) - 1) for length in chunk_lengths)
    request = _CheckpointRowRequest(rows, width)
    token = _REQUEST.set(request)
    try:
        yield request
    finally:
        _REQUEST.reset(token)


def select_checkpoint_rows(hidden):
    """Narrow a vocabulary-head input to the open request's checkpoint rows.

    Identity unless a request is open, has not been applied yet, and
    ``hidden`` is ``[batch, width, hidden]`` for the request's batch and
    width.  Call it on the head input only, never on a hidden state the model
    also returns or caches.
    """
    request = _REQUEST.get()
    if request is None or request.applied:
        return hidden
    shape = getattr(hidden, "shape", None)
    if (
        shape is None
        or len(shape) != 3
        or int(shape[0]) != len(request.rows)
        or int(shape[1]) != request.width
    ):
        return hidden
    request.applied += 1
    if request.width == 1:
        return hidden
    index = mx.array(request.rows, dtype=mx.int32)[:, None, None]
    return mx.take_along_axis(hidden, index, axis=1)


__all__ = ["prefill_checkpoint_rows", "select_checkpoint_rows"]
