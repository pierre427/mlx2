# SPDX-License-Identifier: Apache-2.0
"""``inkling2_prefill_handover`` (default off, unqualified).

Original mlx2 code (2026-11-23), written from the idea described in the
MTPLX sweep (docs/research/2026-11-23-mtplx-inkling2-sweep): no MTPLX code
was copied. See provenance/inkling2-mtplx-prefill.json.

A prompt forward builds the graph of all 48 visits in Python before the
engine evaluates the chunk, so the GPU waits while the host builds. With the
switch on, the residual stream is submitted with ``mx.async_eval`` after
every ``HAND_OVER_VISITS`` visits of a prompt forward, and the GPU starts on
the early visits while the later ones are still being built. The graph and
kernels do not change, only when work is submitted, so the result is
bit-identical.

The same per-forward hook counts lane chunks executed at the resolved wide
prefill step (``inkling2_prefill_wide_chunk``; ``set_wide_rows``, bound by the
adapter's ``bind_prefill_step``), the route's engagement evidence:
``prefill_wide_chunks``.
"""

from __future__ import annotations

import mlx.core as mx

HAND_OVER_VISITS = 4
# Prompt forwards only: decode and DSpark verify rows keep one submission.
HAND_OVER_MIN_ROWS = 64


def handover_every(model, x):
    """Visits between submissions for this forward (0: switch off or not a prompt forward)."""
    wide = model.__dict__.get("_prefill_wide_rows")
    if wide and x.ndim == 3 and x.shape[1] == wide:
        # One count per lane of the forward.
        model.__dict__["prefill_wide_chunks"] = (
            model.__dict__.get("prefill_wide_chunks", 0) + int(x.shape[0]))
    if not model.__dict__.get("_prefill_handover"):
        return 0
    if x.ndim != 3 or x.shape[1] < HAND_OVER_MIN_ROWS:
        return 0
    return HAND_OVER_VISITS


def hand_over(model, x):
    """Submit the residual stream so far without waiting for it."""
    mx.async_eval(x)
    model.__dict__["prefill_handover_calls"] = model.__dict__.get("prefill_handover_calls", 0) + 1


def configure(model, *, prefill_handover=False):
    """Set the switch on a ``language.Model`` (called by configure_kernels)."""
    object.__setattr__(model.model, "_prefill_handover", bool(prefill_handover))
    model.kernel_policy["prefill_handover"] = bool(prefill_handover)


def set_wide_rows(model, rows):
    """Count executed prompt chunks of ``rows`` rows (None: stop counting)."""
    object.__setattr__(model.model, "_prefill_wide_rows", None if rows is None else int(rows))


def receipt(model):
    """Engagement counters for the kernel receipt."""
    inner = model.model.__dict__
    return {"prefill_handover_calls": inner.get("prefill_handover_calls", 0),
            "prefill_wide_chunks": inner.get("prefill_wide_chunks", 0)}
