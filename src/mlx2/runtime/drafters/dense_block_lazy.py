# SPDX-License-Identifier: Apache-2.0
"""Lazy drafts for dense-head mask-block drafters (Laguna DFlash, Muse DFlash).

Original mlx2 code: the ``speculative_device_round`` ``lazy_draft`` form of
the shared ``draft_distributions`` loop of ``laguna_dflash`` and
``muse_dflash`` (default off, unqualified).

Both drafters draft a block of the anchor plus mask tokens in one forward
(``_block_logits``); position ``j`` reads only the anchor, the mask tokens
and the committed context, never the drafts, so the per-position law is the
dense head row (after the lane's processors).  The host loop takes
``softmax(value, 0)`` (a one-hot at ``np.argmax``, the first maximum) and
``rng.sample`` (one uniform per position, landing on that one-hot).  Here
the same rows stay on the device: ``mx.argmax`` over the same float32 values
(the same first maximum), one ``rng.uniform()`` per position, so greedy
tokens and the lane RNG are bitwise the host's, and the draft cache takes the
same ``append_context`` rows.

A position the host loop would end early (no finite value after the
processors: a grammar dead end) or refuse (a NaN value, or no finite value
without processors, where ``softmax`` raises) makes every token
``DRAFT_LAZY_HOST_REDO``: the round is restored and redone on the host path.

``sampler(values [1, V]) -> [1]`` (``device_round.sampled_drafts``) draws
each position from its processed row instead; the device ratio rule then
verifies against the law it recorded (positions are independent, so that is
the drafter's exact law).  Without a sampler a sampled lane gets the greedy
point masses (the point-mass verifier keeps its output law exact).

One lane per call and one-lane cohorts only (``draft_lazy_width_one``): the
host path runs a cohort's rows through one batched ``fc``/MLP matmul, and a
per-lane draft at width two is not guaranteed bitwise equal to that on the
GPU (the matmul's row count can change its kernel).
"""

from __future__ import annotations

import mlx.core as mx


def dense_block_draft_lazy(model, anchor, hidden, cache, proposal_length, sampler=None, *,
                           rng=None, logits_processors=None, processor_history=None):
    """One lane's drafts as a lazy int32 ``[n]`` array, no host read."""
    from ..processor_probe import probe_logits_processors
    from ..proposal_composition import LazyDecline
    from .dflash2 import DRAFT_LAZY_HOST_REDO

    count = int(proposal_length)
    if count < 1:
        raise ValueError("dense-block lazy drafts need a positive proposal length")
    if hidden.shape[0] != 1:
        raise ValueError("dense-block lazy drafts run one lane at a time")
    rows = model._rows(cache)
    if len(rows) != 1:
        raise ValueError("dense-block lazy drafts run one lane at a time")
    processors = list(logits_processors or ())
    # Declines happen before any state moves (the round restores anyway).
    if isinstance(anchor, mx.array):
        # A pipelined round's unread anchor: the host prefix needs its value.
        raise LazyDecline("dense_block_device_anchor")
    if processors and callable(getattr(processor_history, "apply", None)):
        raise LazyDecline("dense_block_device_history")
    anchor = int(anchor)
    model.append_context(hidden, cache)
    dense = model._block_logits([anchor], rows, count).astype(mx.float32)
    model.stats["block_drafts"] += 1
    vocab = int(dense.shape[-1])
    prefix = (mx.array([*list(processor_history or ()), anchor], dtype=mx.int32)
              if processors else None)
    drafted, redo = [], mx.array(False)
    for position in range(count):
        value = dense[:, position]
        if processors:
            tokens = mx.concatenate([prefix, mx.stack(drafted)]) if drafted else prefix
            value = probe_logits_processors(processors, tokens, value).astype(mx.float32)
        redo = redo | ~mx.any(mx.isfinite(value)) | mx.any(mx.isnan(value))
        if sampler is None:
            if rng is not None:
                rng.uniform()  # ``rng.sample(one-hot)`` draws one uniform per position
            token = mx.argmax(value[0]).astype(mx.int32)
        else:
            # Clipped so a dead row's draw keeps later histories in the
            # vocabulary (the round is redone on the host there anyway).
            token = mx.clip(sampler(value).reshape(()).astype(mx.int32), 0, vocab - 1)
        drafted.append(token)
    out = mx.stack(drafted)
    return mx.where(redo, mx.full((count,), DRAFT_LAZY_HOST_REDO, dtype=mx.int32), out)


def install(cls):
    """Give a dense-block drafter class the lazy draft and its declarations."""

    def draft_lazy(self, anchor, hidden, cache, proposal_length, sampler=None, *, rng=None,
                   logits_processors=None, processor_history=None):
        return dense_block_draft_lazy(
            self, anchor, hidden, cache, proposal_length, sampler, rng=rng,
            logits_processors=logits_processors, processor_history=processor_history,
        )

    draft_lazy.__doc__ = dense_block_draft_lazy.__doc__
    cls.draft_lazy = draft_lazy
    cls.draft_lazy_consumes_rng = True
    cls.draft_lazy_processors = True
    cls.draft_lazy_width_one = True
    return cls


__all__ = ["dense_block_draft_lazy", "install"]
