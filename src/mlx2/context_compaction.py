"""Revision-bound transcript compaction control plane.

This module never labels summary replacement exact. The uncompacted transcript
remains a separate proposal-only plane; target state publication requires an
explicit compaction qualification profile.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from .contracts import Fidelity, QualifiedProfile, StatePlane
from .runtime.cache_planes import TranscriptLedgerPlane
from .state import RequestStateTransaction, StateManifest, StateOperation


COMPACTION_STRATEGIES = (
    "oldest_contiguous",
    "largest_first",
    "lowest_importance",
)


@dataclass(frozen=True)
class CompactionSelection:
    strategy: str
    segment_ids: tuple[str, ...]
    reclaimed_tokens: int
    target_tokens: int


def select_transcript_segments(
    plane: TranscriptLedgerPlane,
    target_tokens: int,
    *,
    protected_segment_ids: Collection[str] = (),
    strategy: str | None = None,
    importance_by_segment: Mapping[str, float] | None = None,
) -> CompactionSelection:
    """Choose whole immutable ledger segments without editing target state."""
    selected_strategy = strategy or plane.compaction_strategy
    if selected_strategy not in COMPACTION_STRATEGIES:
        raise ValueError(
            f"unknown compaction strategy {selected_strategy!r}; "
            f"expected one of {COMPACTION_STRATEGIES}"
        )
    if target_tokens < 0:
        raise ValueError("compaction target must be non-negative")
    protected = set(protected_segment_ids)
    if selected_strategy == "oldest_contiguous":
        # The oldest contiguous *unprotected* run: a protected leading prefix
        # (BOS/system attention sinks) is skipped, not treated as a wall.
        candidates = []
        for segment in plane.segments:
            if segment.segment_id in protected:
                if candidates:
                    break
                continue
            candidates.append(segment)
    elif selected_strategy == "largest_first":
        candidates = sorted(
            (segment for segment in plane.segments if segment.segment_id not in protected),
            key=lambda segment: (-len(segment.token_ids), segment.token_start),
        )
    else:
        if importance_by_segment is None:
            raise ValueError("lowest_importance compaction requires segment importance scores")
        eligible = [
            segment for segment in plane.segments if segment.segment_id not in protected
        ]
        missing = [
            segment.segment_id
            for segment in eligible
            if segment.segment_id not in importance_by_segment
        ]
        if missing:
            raise ValueError("missing segment importance scores for " + ", ".join(missing))
        candidates = sorted(
            eligible,
            key=lambda segment: (
                float(importance_by_segment[segment.segment_id]),
                segment.token_start,
            ),
        )
    selected = []
    reclaimed = 0
    for segment in candidates:
        if reclaimed >= target_tokens:
            break
        selected.append(segment.segment_id)
        reclaimed += len(segment.token_ids)
    return CompactionSelection(
        selected_strategy, tuple(selected), reclaimed, target_tokens
    )


@dataclass(frozen=True)
class TranscriptTurn:
    role: str
    token_ids: tuple[int, ...]
    content: str = ""
    tool_pair: str | None = None

    def __post_init__(self):
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("invalid transcript role")
        if not self.token_ids or any(type(token) is not int or token < 0 for token in self.token_ids):
            raise ValueError("transcript turns require nonnegative token IDs")


@dataclass(frozen=True)
class CompactionDecision:
    trigger: bool
    reason: str
    target_tokens: int


class CompactionPressurePolicy:
    def __init__(self, *, context_fraction=0.85, memory_fraction=0.85, target_fraction=0.60):
        values = (context_fraction, memory_fraction, target_fraction)
        if any(not 0 < value < 1 for value in values):
            raise ValueError("compaction fractions must be between zero and one")
        self.context_fraction = context_fraction
        self.memory_fraction = memory_fraction
        self.target_fraction = target_fraction

    def decide(self, *, context_tokens, max_context, footprint_bytes, memory_limit_bytes):
        if min(context_tokens, max_context, footprint_bytes, memory_limit_bytes) < 0 or not max_context or not memory_limit_bytes:
            raise ValueError("compaction pressure inputs are invalid")
        context_pressure = context_tokens / max_context
        memory_pressure = footprint_bytes / memory_limit_bytes
        trigger = context_pressure >= self.context_fraction or memory_pressure >= self.memory_fraction
        reason = "context" if context_pressure >= self.context_fraction else "memory" if trigger else "below_threshold"
        return CompactionDecision(trigger, reason, int(max_context * self.target_fraction))


def plan_turn_compaction(turns: Sequence[TranscriptTurn], *, target_tokens: int):
    """Select an oldest contiguous removable span while preserving invariants."""
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")
    turns = tuple(turns)
    total = sum(len(turn.token_ids) for turn in turns)
    if total <= target_tokens:
        return {"kept": turns, "removed": (), "removed_tokens": 0}
    protected = {index for index, turn in enumerate(turns) if turn.role == "system"}
    protected.update(range(max(0, len(turns) - 4), len(turns)))
    pairs = {}
    for index, turn in enumerate(turns):
        if turn.tool_pair:
            pairs.setdefault(turn.tool_pair, []).append(index)
    removed = []
    remaining = total
    for index, turn in enumerate(turns):
        if remaining <= target_tokens or index in protected:
            continue
        pair = pairs.get(turn.tool_pair, [index]) if turn.tool_pair else [index]
        if any(item in protected for item in pair):
            continue
        for item in pair:
            if item not in removed:
                removed.append(item)
                remaining -= len(turns[item].token_ids)
    removed_set = set(removed)
    return {
        "kept": tuple(turn for index, turn in enumerate(turns) if index not in removed_set),
        "removed": tuple(turn for index, turn in enumerate(turns) if index in removed_set),
        "removed_tokens": total - remaining,
    }


def publish_compacted_state(
    current: StateManifest,
    *,
    profile: QualifiedProfile,
    target_state,
    uncompacted_transcript,
):
    """Publish bounded compaction without overwriting the source transcript."""
    transaction = RequestStateTransaction(current, StateOperation.COMPACT, profile=profile)
    planes = dict(current.planes)
    planes[StatePlane.ATTENTION_KV] = target_state
    planes[StatePlane.TRANSCRIPT] = uncompacted_transcript
    prepared = transaction.prepare(planes, fidelity=Fidelity.NUMERICALLY_BOUNDED)
    return transaction.commit(current), prepared
