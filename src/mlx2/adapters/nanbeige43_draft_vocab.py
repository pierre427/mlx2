"""Explicit Nanbeige proposer-only shortlist with committed-context coverage.

The target head is never replaced. Prefix-id ordering is a candidate for this
SentencePiece vocabulary, not an assumed frequency ranking. Missing context,
constraints or any uncovered recent token select the original full head.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from threading import RLock

SCHEMA = "mlx2.nanbeige43-guarded-draft-vocab.v1"
SPECIAL_FROM = 166100
# The released SentencePiece BPE stores its basic-character region after
# merged pieces: id152343 is the word-boundary marker, then e,t,a,... .
# Preserve this complete tail, not just added role/EOS tokens. Pure id-prefix
# pruning would otherwise exclude ordinary English letters and punctuation.
BASE_TAIL_FROM = 152343


def validate(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {"rows", "coverage_window"} or "rows" not in value:
        raise ValueError("Nanbeige draft_vocab requires rows and optional coverage_window")
    rows, window = value["rows"], value.get("coverage_window", 16)
    if type(rows) is not int or not 1 <= rows < BASE_TAIL_FROM:
        raise ValueError("Nanbeige draft_vocab rows must precede the mandatory basic-character tail")
    if type(window) is not int or not 1 <= window <= 128:
        raise ValueError("Nanbeige draft_vocab coverage_window must be 1..128")
    return {"rows": rows, "coverage_window": window}


def coverage_reason(anchors, histories, processors, *, rows, special, vocab, window):
    count = len(anchors)
    if histories is None or processors is None or len(histories) != count or len(processors) != count:
        return "missing_context"
    if any(processors):
        return "constraints"
    for anchor, history in zip(anchors, histories, strict=True):
        if history is None:
            return "missing_context"
        for token in [*history, anchor][-window:]:
            token = int(token)
            if not (0 <= token < rows or special <= token < vocab):
                return "coverage"
    return None


def install(model, value, *, lane=False):
    """Wrap graph construction only; asynchronous graphs retain their operands."""
    policy = validate(value)
    if policy is None:
        return None
    from ..runtime.drafters.dflash2 import configure_draft_vocab

    receipt = configure_draft_vocab(model, policy["rows"], special_from=BASE_TAIL_FROM, lane=lane)
    receipt = {**receipt, "retained_tail_from": BASE_TAIL_FROM, "special_from": SPECIAL_FROM,
               "ordering": "merged-piece prefix plus complete base-character/special tail"}
    reduced = model.__dict__["_draft_vocab"]
    original = model.start_tree
    lock = RLock()
    counts = {"reduced": 0, "missing_context": 0, "constraints": 0, "coverage": 0}

    def start_tree(anchors, hidden, cache, max_nodes, *, proposal_histories=None,
                   proposal_has_processors=None, **options):
        reason = coverage_reason(anchors, proposal_histories, proposal_has_processors,
                                 rows=reduced["rows"], special=reduced["special"],
                                 vocab=reduced["vocab"], window=policy["coverage_window"])
        with lock:
            counts[reason or "reduced"] += 1
            # Selection is fixed while Python builds this graph. Its device
            # arrays remain retained after this pointer is restored.
            object.__setattr__(model, "_draft_vocab", reduced if reason is None else None)
            try:
                return original(anchors, hidden, cache, max_nodes, **options)
            finally:
                object.__setattr__(model, "_draft_vocab", reduced)

    def propose_tree(anchors, hidden, cache, max_nodes, *, mark=None, **options):
        state = start_tree(anchors, hidden, cache, max_nodes, **options)
        if mark is not None:
            mark("tree_draft_build")
        return model.finish_tree(state, mark=mark)

    object.__setattr__(model, "start_tree", start_tree)
    object.__setattr__(model, "propose_tree", propose_tree)
    object.__setattr__(model, "requires_tree_proposal_contexts", True)
    result = {**receipt, **policy, "schema": SCHEMA, "counts": counts,
              "fallback": "full head for constraints, missing context, or uncovered recent committed ids",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "target_head_changed": False, "qualified": False}
    # Runtime counters are diagnostics, never mutable arithmetic identity.
    object.__setattr__(model, "draft_vocab_receipt", {key: val for key, val in result.items() if key != "counts"})
    object.__setattr__(model, "_nanbeige_draft_vocab_counts", counts)
    return result
