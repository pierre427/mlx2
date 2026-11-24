# SPDX-License-Identifier: Apache-2.0
# Mechanism after youssofal/MTPLX@3890be556160af48887d5bfbf5304c4ad5b9417f
# mtplx/models/inkling/fast/pass_messages.py (one read of a loop pass's
# messages for all its visits, proven per shape), Apache-2.0; reimplemented
# for mlx2 with MLX ops (no code copied), see
# provenance/inkling2-mtplx-launch.json.
"""``inkling2_pass_messages``: a loop pass's message reads in three launches.

Opt-in, default off, unqualified. On loop passes two and three each of the
twelve core visits reads its message from the pass's normalised bank of
prior states with its layer's four queries (``TextModel._forward_body``):

    scores = (query[4, D] @ bank^T) * D^-0.5;  softmax(scores) @ bank

The bank is fixed for the whole pass and the queries are weights, so nothing
a visit computes feeds another visit's read: they only wait on each other
because each sits in its own visit. Here a pass reads them together, with
the same ops per element:

- the scores of all twelve layers as one product, ``table[12 * 4, D] @
  bank^T``, and its scale;
- one softmax over the bank axis for every (visit, row, query);
- the mix as one product, ``probs[..., 48, N] @ bank``.

Each visit's message is then a slice of the result. About 5 launches per
visit (60 a pass, 120 a token at B1) become about 4 a pass.

Exactness: the scale and softmax are per element and per row, so they are
unchanged. Whether one product over 48 query rows gives each row the bits
of its own 4-row product depends on which kernel MLX picks for the shapes
(M = 4 versus 48: split-K, GEMM tiles, NAX), so it is decided on the first
call of each shape against the per-visit reads, bit for bit
(``library_kernels.run_proved``); the proving call returns the per-visit
messages, and a refused shape keeps them (counted). It runs on any device
(the CPU tests exercise the proof); it declines when the bank is the
one-row list that ``inkling2_fuse_residual_kernel``'s message kernel reads.

Counters: ``TextModel.pass_messages_counts`` (``calls``, ``proved``,
``refused``, ``refused_first``, ``unproven``, ``declined:<why>``), in
``kernel_receipt()["pass_messages"]``.
"""

from __future__ import annotations

import mlx.core as mx

from . import library_kernels as lib

QUERIES = 4
PROOFS = lib.Proofs("pass_messages")


def per_visit(bank, query, width):
    """The composed read of one visit (``TextModel._forward_body``)."""
    scores = (query @ bank.swapaxes(-1, -2)) * width ** -0.5
    return mx.softmax(scores, axis=-1) @ bank


def batched(bank, table, width):
    """Every visit's message from one scores product, one softmax, one mix."""
    visits = table.shape[0]
    scores = (table.reshape(visits * QUERIES, width) @ bank.swapaxes(-1, -2)) * width ** -0.5
    mixed = mx.softmax(scores, axis=-1) @ bank
    return [mixed[..., QUERIES * i : QUERIES * (i + 1), :] for i in range(visits)]


def rows(model, bank, table, width):
    """The pass's messages (one per core visit, in visit order), or None."""
    counts = "pass_messages_counts"
    if isinstance(bank, list):
        lib.count(model, counts, "declined:residual_kernel_messages")
        return None
    if table.ndim != 3 or table.shape[1] != QUERIES or table.shape[2] != width:
        lib.count(model, counts, "declined:layout")
        return None
    key = (tuple(bank.shape), str(bank.dtype), tuple(table.shape), str(table.dtype),
           str(mx.default_device()))
    out, _engaged = lib.run_proved(
        PROOFS, key, model, counts,
        lambda: batched(bank, table, width),
        lambda: [per_visit(bank, table[i], width) for i in range(table.shape[0])],
        lambda a, b: list(zip(a, b)))
    return out
