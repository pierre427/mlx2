# SPDX-License-Identifier: MIT
"""Prompt-chunk tail skip: stop non-checkpoint rows at their last state write.

Opt-in (``MLX2_PREFILL_TAIL_SKIP``, default off).  Concept from
ashhart/TensorFold#2432 (see provenance/prefill-tail-skip-20261107.json); the
implementation is original and rides mlx2's checkpoint-row contract
(``prefill_checkpoint_rows``).

A prompt chunk's logits are never sampled; the prefill guard checks one
checkpoint row per lane for finiteness.  When every lane's checkpoint row is
the chunk's last (``claim_final_row_checkpoint``), the rows before it are
read by nothing after the stack's last state write.  So:

* Tier 1 (shared, no per-model math): a model's fresh ``make_cache()``
  entries carry ``apcv2_layer_index``.  The largest tag is the last layer that
  writes state; every layer after it, the final norm and the head run on the
  last row only.  Missing, duplicate or unordered tags, or a live cache whose
  entry count differs from the template, decline (the full chunk runs).  The
  template is used, not the live cache, because batch merges
  (``BatchKVCache.merge``) do not carry the tag and changing that would move
  APCv2 segment keys.
* Tier 2 (adapter-declared): when the last layer itself writes state
  (attention), the adapter runs that layer's K/V over every row and the rest
  (q, attention output, MLP) over the last row, like standard_decoder's
  ``Layer.tail``.
* Head trim: an adapter that did not project only the checkpoint row through
  its vocabulary head joins the contract (Gemma 4 via ``logits_to_keep``).

A forward that has a reader of the other rows never reaches this hook: MTP
prefill (``mtp_backbone``), capture taps (``forward_with_taps``), drafter
features, verify and decode forwards do not open a checkpoint-row request,
and a request whose checkpoint rows are not every lane's last row is not
claimed.  A reader that does open a request declines through ``refuse``
(Nemotron: a wrapped backbone, i.e. calibration's hidden-state observer).
Neither tier changes state arithmetic, so row-exact modes need no refusal.

Only the ordinary prompt batch (``generate.PromptProcessingBatch.prompt``)
opens checkpoint-row requests; the prompt-lookup, MTP and external-draft
prefills do not, so on those routes the switch is inert (counters stay 0).

Mode values: ``0`` off; ``1`` every tier the adapter supports; ``head`` the
head trim only (for A/B attribution).  Unset means off, except for families in
``FAMILY_DEFAULTS`` (Gemma 4: on since 2026-11-07, Pierre).
"""

from __future__ import annotations

import os
from collections import Counter

from .prefill_checkpoint_rows import claim_final_row_checkpoint, final_row_checkpoint_open

ENV = "MLX2_PREFILL_TAIL_SKIP"
_MODES = ("0", "1", "head")

STATS: Counter = Counter()


# Families whose adapter turns the skip on when the variable is unset.  Every
# other family stays off unless MLX2_PREFILL_TAIL_SKIP is set.  An explicit
# value (including 0) always wins.
#   gemma4: default 1 (Pierre, 2026-11-07), after per-chunk -5.5..-5.9% on
#   gemma-4-31B-it-MLX-8bit and identical greedy texts at 2K/16K
#   (qualification/runs/prefill-tail-skip-20261107).
FAMILY_DEFAULTS = {"gemma4": "1"}


def tail_skip_mode(family: str | None = None) -> str:
    raw = os.environ.get(ENV)
    if raw is None:
        return FAMILY_DEFAULTS.get(family, "0")
    value = raw.strip().lower()
    return value if value in _MODES else "0"


def tail_skip_status() -> dict:
    return {
        "configured": os.environ.get(ENV, "unset"),
        "default": "0",
        "family_defaults": dict(FAMILY_DEFAULTS),
        "qualification": "unqualified_default_gemma4_opt_in_elsewhere",
        "counts": dict(STATS),
    }


def last_state_writer(cache, layer_count: int):
    """Index of the last state-writing layer, or None when it cannot be proven.

    Every cache entry must be present and carry an integer
    ``apcv2_layer_index``, strictly increasing and inside ``[0, layer_count)``.
    Returns None also when the last writer is the last layer (no stateless
    tail to skip).
    """
    if not cache:
        return None
    previous = -1
    for entry in cache:
        index = getattr(entry, "apcv2_layer_index", None) if entry is not None else None
        if type(index) is not int or index <= previous or index >= layer_count:
            return None
        previous = index
    return previous if previous < layer_count - 1 else None


def template_state_writer(model, layer_count: int):
    """Memoized ``last_state_writer`` of the model's fresh ``make_cache()``."""
    cached = getattr(model, "_mlx2_tail_skip_writer", None)
    # Keyed by layer count so a structural change (excision) recomputes; the
    # caller also checks the live cache's entry count against the template.
    if cached is not None and cached[0] == layer_count:
        return cached[1], cached[2]
    make_cache = getattr(model, "make_cache", None)
    template = make_cache() if callable(make_cache) else None
    writer = last_state_writer(template, layer_count)
    entries = len(template) if template else 0
    try:
        object.__setattr__(model, "_mlx2_tail_skip_writer", (layer_count, writer, entries))
    except (AttributeError, TypeError):
        pass
    return writer, entries


def claim_stateless_tail(inputs, cache, model, layer_count: int, *,
                         refuse: str | None = None):
    """Tier 1: return the last writer's index and claim the request, or None.

    ``refuse`` names an adapter-specific reason to keep the full chunk (a
    row-exact mode, a reader); it is counted and nothing is claimed.  The
    request is claimed only after every other check passes, so a decline
    never leaves a claimed request with a full-width output.
    """
    if tail_skip_mode() != "1":
        return None
    if getattr(inputs, "ndim", 0) != 2 or inputs.shape[1] <= 1:
        return None
    # Count declines only for a forward that could have been claimed.
    if not final_row_checkpoint_open(inputs.shape[0], inputs.shape[1]):
        return None
    if refuse:
        STATS[f"declined_{refuse}"] += 1
        return None
    writer, entries = template_state_writer(model, layer_count)
    if writer is None:
        STATS["declined_no_stateless_tail"] += 1
        return None
    if cache is None or len(cache) != entries:
        STATS["declined_cache_layout"] += 1
        return None
    if not claim_final_row_checkpoint(inputs.shape[0], inputs.shape[1]):
        return None
    STATS["tier1_engaged"] += 1
    STATS["tier1_rows_skipped"] += int(inputs.shape[0]) * (int(inputs.shape[1]) - 1)
    return writer


def claim_adapter_tail(inputs, *, kind: str, refuse: str | None = None,
                       family: str | None = None) -> bool:
    """Tier 2 / head trim: claim the request for an adapter-owned tail.

    ``kind`` is ``"tail"`` (last state-writing layer split, needs mode 1) or
    ``"head"`` (head trim, mode 1 or ``head``).  ``family`` selects a
    ``FAMILY_DEFAULTS`` entry when the variable is unset.
    """
    mode = tail_skip_mode(family)
    if mode == "0" or (kind == "tail" and mode != "1"):
        return False
    if getattr(inputs, "ndim", 0) != 2 or inputs.shape[1] <= 1:
        return False
    # Count declines only for a forward that could have been claimed.
    if not final_row_checkpoint_open(inputs.shape[0], inputs.shape[1]):
        return False
    if refuse:
        STATS[f"declined_{refuse}"] += 1
        return False
    if not claim_final_row_checkpoint(inputs.shape[0], inputs.shape[1]):
        return False
    STATS[f"{kind}_engaged"] += 1
    return True


def structural_refusal(inputs, cache, *, trunk=None, trunk_class=None, parts=()):
    """Shared tier-2 decline checks, or None.

    ``trunk`` must be an instance of ``trunk_class`` (invariant-prefill's
    scoped subclass passes; any other wrapper does not); every
    ``(module, allowed_types)`` in ``parts`` must be exactly one of the
    allowed types (hooks such as residual-ablation capture swap
    ``__class__``); and the forward must not be inside a verify scope, a
    row-exact window or a speculating cache.  Families add their own checks.
    """
    if trunk_class is not None and not isinstance(trunk, trunk_class):
        return "trunk_wrapped"
    for module, allowed in parts:
        if type(module) not in tuple(allowed):
            return "wrapped_last_layer"
    from .models.invariant_prefill import prefill_decline_reason

    reason = prefill_decline_reason(inputs, cache)
    return None if reason in (None, "single_row") else reason


__all__ = [
    "ENV",
    "STATS",
    "claim_adapter_tail",
    "claim_stateless_tail",
    "last_state_writer",
    "structural_refusal",
    "FAMILY_DEFAULTS",
    "tail_skip_mode",
    "tail_skip_status",
    "template_state_writer",
]
