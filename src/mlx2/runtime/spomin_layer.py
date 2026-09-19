"""Revision-bound control plane for segmented live-context compaction.

Adapted from the lab-owned mlx-lm-unified revision recorded in provenance/.
The uncompacted transcript remains immutable and proposal-only; a backend must
declare every target-state operation it can safely perform.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from itertools import pairwise
from math import isfinite
from typing import Protocol

from ..context_compaction import (
    COMPACTION_STRATEGIES,
    CompactionSelection,
    select_transcript_segments,
)
from .cache_planes import TranscriptLedgerPlane, TranscriptLedgerSegment


class SpominCapabilityError(RuntimeError):
    pass


class SpominRevisionError(RuntimeError):
    pass


class SpominPlanError(RuntimeError):
    pass


class SpominBackendError(RuntimeError):
    pass


class SpominTrigger(str, Enum):
    BELOW_PRESSURE = "below_pressure"
    PRESSURE = "pressure"
    FORCED = "forced"


@dataclass(frozen=True)
class SpominConfig:
    capacity_tokens: int
    pressure_ratio: float = 0.70
    target_ratio: float = 0.65
    protect_recent_segments: int = 1
    strategy: str = "oldest_contiguous"

    def __post_init__(self):
        if self.capacity_tokens <= 0:
            raise ValueError("Spomin capacity_tokens must be positive")
        if not 0 < self.target_ratio < self.pressure_ratio <= 1:
            raise ValueError("Spomin ratios must satisfy 0 < target_ratio < pressure_ratio <= 1")
        if self.protect_recent_segments < 0:
            raise ValueError("protect_recent_segments must be non-negative")
        if self.strategy not in COMPACTION_STRATEGIES:
            raise ValueError(f"unknown Spomin strategy {self.strategy!r}")

    @property
    def pressure_tokens(self):
        return int(self.capacity_tokens * self.pressure_ratio)

    @property
    def target_tokens(self):
        return int(self.capacity_tokens * self.target_ratio)


@dataclass(frozen=True)
class SpominTargetState:
    revision: str
    target_tokens: int
    transcript: TranscriptLedgerPlane
    visible_segment_ids: tuple[str, ...]
    has_recurrent_state: bool = False
    has_mtp_state: bool = False

    def __post_init__(self):
        if not self.revision:
            raise ValueError("Spomin target state requires a revision")
        if self.target_tokens < 0:
            raise ValueError("target_tokens must be non-negative")
        order = {segment.segment_id: i for i, segment in enumerate(self.transcript.segments)}
        unknown = set(self.visible_segment_ids) - set(order)
        if unknown:
            raise ValueError("visible segments are absent from transcript: " + ", ".join(sorted(unknown)))
        if len(set(self.visible_segment_ids)) != len(self.visible_segment_ids):
            raise ValueError("visible segment identities must be unique")
        positions = [order[segment_id] for segment_id in self.visible_segment_ids]
        if positions != sorted(positions):
            raise ValueError("visible segments must retain transcript order")


@dataclass(frozen=True)
class SpominBackendCapabilities:
    simulation: bool = False
    exact_rebuild: bool = False
    attention_kv_edit: bool = False
    recurrent_state_repair: bool = False
    mtp_state_repair: bool = False
    noncontiguous_edit: bool = False


@dataclass(frozen=True)
class SpominPlan:
    source_revision: str
    source_transcript_digest: str
    trigger: SpominTrigger
    selection: CompactionSelection
    protected_segment_ids: tuple[str, ...]
    replacement_token_ids: tuple[int, ...]
    projected_target_tokens: int
    target_limit_tokens: int
    shortfall_tokens: int
    noncontiguous: bool

    @property
    def ready(self):
        return self.shortfall_tokens == 0 and bool(self.selection.segment_ids)


class SegmentImportanceScorer(Protocol):
    def score(self, transcript: TranscriptLedgerPlane) -> Mapping[str, float]: ...


class SpominBackend(Protocol):
    capabilities: SpominBackendCapabilities

    def apply(self, state: SpominTargetState, plan: SpominPlan) -> SpominTargetState: ...


@dataclass(frozen=True)
class StaticSegmentScorer:
    scores: Mapping[str, float]

    def score(self, transcript):
        return self.scores


@dataclass(frozen=True)
class HeadHotspotScorer:
    scores_by_head: Mapping[tuple[int, int], Sequence[float]]

    def score(self, transcript):
        count = len(transcript.token_ids)
        if not self.scores_by_head:
            raise ValueError("HeadHotspotScorer requires at least one head")
        for head, scores in self.scores_by_head.items():
            if len(scores) != count:
                raise ValueError(f"head {head} has {len(scores)} scores for {count} tokens")
        return {
            segment.segment_id: max(
                max(scores[segment.token_start : segment.token_stop])
                for scores in self.scores_by_head.values()
            )
            for segment in transcript.segments
        }


class SpominLayer:
    def __init__(self, config: SpominConfig, *, scorer: SegmentImportanceScorer | None = None):
        self.config = config
        self.scorer = scorer

    def proposal_transcript(self, state):
        return state.transcript

    def pressure_state(self, state, *, force=False):
        if force:
            return SpominTrigger.FORCED
        if state.target_tokens >= self.config.pressure_tokens:
            return SpominTrigger.PRESSURE
        return SpominTrigger.BELOW_PRESSURE

    @staticmethod
    def _visible_transcript(state):
        by_id = {segment.segment_id: segment for segment in state.transcript.segments}
        cursor = 0
        visible = []
        for segment_id in state.visible_segment_ids:
            source = by_id[segment_id]
            stop = cursor + len(source.token_ids)
            visible.append(TranscriptLedgerSegment(segment_id, cursor, stop, source.token_ids))
            cursor = stop
        return TranscriptLedgerPlane(
            tokenizer_identity=state.transcript.tokenizer_identity,
            tokenizer_version=state.transcript.tokenizer_version,
            revision=state.transcript.revision,
            segments=tuple(visible),
            compaction_strategy=state.transcript.compaction_strategy,
        )

    def plan(
        self,
        state,
        *,
        protected_segment_ids=(),
        replacement_token_ids=(),
        strategy=None,
        target_limit_tokens=None,
        force=False,
    ):
        trigger = self.pressure_state(state, force=force)
        if trigger is SpominTrigger.BELOW_PRESSURE:
            return None
        replacement = tuple(replacement_token_ids)
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in replacement):
            raise ValueError("replacement token IDs must be non-negative integers")
        known = {segment.segment_id for segment in state.transcript.segments}
        explicit = set(protected_segment_ids)
        unknown = explicit - known
        if unknown:
            raise ValueError("protected segments are absent from transcript: " + ", ".join(sorted(unknown)))
        recent = state.visible_segment_ids[-self.config.protect_recent_segments :] if self.config.protect_recent_segments else ()
        protected = explicit | set(recent)
        target_limit = self.config.target_tokens if target_limit_tokens is None else target_limit_tokens
        if isinstance(target_limit, bool) or not isinstance(target_limit, int) or target_limit < 0:
            raise ValueError("target_limit_tokens must be a non-negative integer")
        selected_strategy = strategy or self.config.strategy
        importance = None
        if selected_strategy == "lowest_importance":
            if self.scorer is None:
                raise ValueError("lowest_importance Spomin strategy requires a segment scorer")
            importance = self.scorer.score(state.transcript)
            invalid = [key for key, value in importance.items() if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value))]
            if invalid:
                raise ValueError("segment importance scores must be finite numbers: " + ", ".join(sorted(invalid)))
        gross_reclaim = max(state.target_tokens - target_limit + len(replacement), 0)
        selection = select_transcript_segments(
            self._visible_transcript(state), gross_reclaim,
            protected_segment_ids=protected,
            strategy=selected_strategy,
            importance_by_segment=importance,
        )
        projected = max(state.target_tokens - selection.reclaimed_tokens + len(replacement), 0)
        positions = sorted(i for i, segment_id in enumerate(state.visible_segment_ids) if segment_id in selection.segment_ids)
        noncontiguous = any(right != left + 1 for left, right in pairwise(positions))
        return SpominPlan(
            state.revision, state.transcript.fingerprint.digest, trigger, selection,
            tuple(sorted(protected)), replacement, projected, target_limit,
            max(projected - target_limit, 0), noncontiguous,
        )

    def apply(self, state, plan, backend):
        if state.revision != plan.source_revision:
            raise SpominRevisionError("target revision changed after planning")
        if state.transcript.fingerprint.digest != plan.source_transcript_digest:
            raise SpominRevisionError("uncompacted transcript changed after planning")
        if not plan.ready:
            raise SpominPlanError(f"compaction plan misses target by {plan.shortfall_tokens} tokens")
        capabilities = backend.capabilities
        if not capabilities.simulation and not capabilities.exact_rebuild:
            missing = []
            if not capabilities.attention_kv_edit:
                missing.append("attention_kv_edit")
            if state.has_recurrent_state and not capabilities.recurrent_state_repair:
                missing.append("recurrent_state_repair")
            if state.has_mtp_state and not capabilities.mtp_state_repair:
                missing.append("mtp_state_repair")
            if plan.noncontiguous and not capabilities.noncontiguous_edit:
                missing.append("noncontiguous_edit")
            if missing:
                raise SpominCapabilityError("backend cannot safely apply plan: " + ", ".join(missing))
        updated = backend.apply(state, plan)
        if updated.revision == state.revision:
            raise SpominBackendError("backend did not advance target revision")
        if updated.target_tokens != plan.projected_target_tokens:
            raise SpominBackendError("backend target token count disagrees with plan")
        if updated.transcript.fingerprint.digest != state.transcript.fingerprint.digest:
            raise SpominBackendError("backend mutated the uncompacted transcript")
        selected = set(plan.selection.segment_ids)
        expected = tuple(segment_id for segment_id in state.visible_segment_ids if segment_id not in selected)
        if updated.visible_segment_ids != expected:
            raise SpominBackendError("backend visible segments disagree with plan")
        return updated


class InMemorySpominBackend:
    capabilities = SpominBackendCapabilities(simulation=True)

    def apply(self, state, plan):
        selected = set(plan.selection.segment_ids)
        return replace(
            state,
            revision=f"{state.revision}:spomin",
            target_tokens=plan.projected_target_tokens,
            visible_segment_ids=tuple(segment_id for segment_id in state.visible_segment_ids if segment_id not in selected),
        )
