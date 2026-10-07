# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified revision 13eb83388750435bcc751d0b9e33857a38152544.
# See provenance/prompt-lookup-live.json and provenance/LICENSE.unified-MIT.
"""Live, fail-closed prompt-lookup decoding for the shared serving lifecycle."""

from __future__ import annotations

import math
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import mlx.core as mx

from .committed_recovery import CommittedRecoverySlot, RecoveryCheckpointMismatch
from .cow_cache import (
    restore_recovery_descriptors,
    snapshot_committed_cache,
    snapshot_recovery_descriptors,
)
from .generate import (
    ALLOCATOR_RECLAIM_STEP_INTERVAL,
    GenerationBatch,
    StopSequenceMatcher,
    _crossed_counter_interval,
    generation_stream,
)
from .models import cache as cache_module
from .models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache
from .prefill_plan import prompt_length_prefill_step
from .prompt_lookup import (
    AdaptiveLookback,
    CostAwarePLDLatch,
    HybridStats,
    IndexedPromptLookup,
    RecentCommittedSegmentStore,
    plan_proposal_around_verify_cliff,
)
from .rotating_replay import (
    RotatingReplayError,
    RotatingReplayPolicy,
    RotatingReplayTransaction,
)
from ..thinking_guard import stack_block_steer, verify_block_steer


class PromptLookupLaneFailure(RuntimeError):
    """A sampled output is invalid for one prompt-lookup request."""

    def __init__(self, uid, reason):
        super().__init__(reason)
        self.uid = uid
        self.reason = reason

    @property
    def failures(self):
        return (self,)


class PromptLookupRoundFailures(PromptLookupLaneFailure):
    """Lanes of a batched round that failed after their peers committed."""

    def __init__(self, failures):
        first = failures[0]
        super().__init__(first.uid, first.reason)
        self._failures = tuple(failures)

    @property
    def failures(self):
        return self._failures


def _output_reason(token, finite, vocab):
    """``_invalid_output_reason`` from host values read in a batched sync."""
    if token < 0 or token >= vocab:
        return f"sampled token {token} is outside vocabulary of size {vocab}"
    if not finite:
        return f"sampled token {token} has non-finite log probability"
    return None


def _finite_at(row, token):
    # Clipped: an out-of-vocabulary token is refused by its range check.
    return mx.isfinite(row[mx.clip(token, 0, row.shape[-1] - 1)])


def _walk_state(caches):
    return [entry.state for entry in caches]


def _steer_slice(steer, start, stop):
    """``steer`` restricted to verify positions ``[start, stop)``, or None."""
    if steer is None:
        return None
    start, stop = max(0, start), min(int(steer[1].shape[1]), stop)
    if stop <= start:
        return None
    return steer[0], steer[1][:, start:stop]


def _validate_cache(caches):
    def visit(entry):
        if isinstance(entry, CacheList):
            for child in entry.caches:
                visit(child)
        elif isinstance(entry, (ArraysCache, RotatingKVCache, KVCache)):
            return
        elif not (hasattr(entry, "offset") and hasattr(entry, "trim")):
            raise NotImplementedError(
                "prompt lookup requires exact rollback-capable caches; "
                f"{type(entry).__name__} is unsupported"
            )

    for entry in caches:
        visit(entry)


def _set_ordinary_b1_mask(caches, enabled):
    """Give request-private raw full-attention KV the ordinary B=1 mask."""
    stack = list(caches)
    while stack:
        entry = stack.pop()
        if isinstance(entry, CacheList):
            stack.extend(entry.caches)
        elif type(entry) is KVCache:
            entry._pld_ordinary_mask_padding = mx.array([0]) if enabled else None
            entry._pld_ordinary_mask_calls = 0


def _ordinary_b1_mask_calls(caches):
    count = 0
    stack = list(caches)
    while stack:
        entry = stack.pop()
        if isinstance(entry, CacheList):
            stack.extend(entry.caches)
        elif type(entry) is KVCache:
            count += getattr(entry, "_pld_ordinary_mask_calls", 0)
    return count


def _append_source_segment(
    source_segments,
    seen_segments,
    tokens,
    source_kind,
    source_id,
    *,
    source_limit,
    token_limit,
    response_limit,
    source_tokens,
):
    """Append one bounded, deduplicated hot source and return token usage."""
    if len(source_segments) >= source_limit:
        return source_tokens
    remaining = token_limit - source_tokens
    if remaining <= 0:
        return source_tokens
    values = tuple(int(token) for token in tokens)
    values = values[-min(response_limit, remaining) :]
    if values and values not in seen_segments:
        source_segments.append((values, source_kind, str(source_id)))
        seen_segments.add(values)
        source_tokens += len(values)
    return source_tokens


def _rotating_leaves(caches):
    """Every ``RotatingKVCache`` under ``caches``, in depth-first order.

    An explicit stack, not a self-recursive closure: a nested ``visit`` that
    refers to itself is a reference cycle holding its ``leaves`` cell, so a
    finished lane's rings (and their K/V) would stay alive until a cyclic GC.
    """
    leaves = []
    stack = list(reversed(list(caches)))
    while stack:
        entry = stack.pop()
        if isinstance(entry, CacheList):
            stack.extend(reversed(entry.caches))
        elif isinstance(entry, RotatingKVCache):
            leaves.append(entry)
    return leaves


def _snapshot(caches, *, skip_rotating=False):
    """Record the pre-verify state of every cache.

    ``skip_rotating`` leaves rotating rings to a caller that either owns them
    through a ``RotatingReplayTransaction`` or knows the round cannot roll
    back; ``_rewind`` then leaves those entries alone.
    """

    def one(entry):
        if isinstance(entry, CacheList):
            return ("list", [one(child) for child in entry.caches])
        if isinstance(entry, ArraysCache):
            if not entry.is_trimmable():
                raise RuntimeError("ArraysCache rollback epoch is not active")
            return ("array", entry.rollback_marker())
        if isinstance(entry, RotatingKVCache):
            if skip_rotating:
                return ("replay",)
            return (
                "rotating",
                None if entry.keys is None else mx.array(entry.keys),
                None if entry.values is None else mx.array(entry.values),
                entry.offset,
                getattr(entry, "_idx", None),
            )
        if isinstance(entry, KVCache) or (
            hasattr(entry, "offset") and hasattr(entry, "trim")
        ):
            return ("trim", entry.offset)
        raise NotImplementedError(type(entry).__name__)

    return [one(entry) for entry in caches]


def _rewind(caches, snapshots):
    def one(entry, snapshot):
        if snapshot[0] == "list":
            for child, child_snapshot in zip(entry.caches, snapshot[1]):
                one(child, child_snapshot)
        elif snapshot[0] == "array":
            entry.rewind_to_rollback_marker(snapshot[1])
        elif snapshot[0] == "trim":
            entry.trim(int(entry.offset) - int(snapshot[1]))
        elif snapshot[0] == "replay":
            return
        else:
            _, keys, values, offset, index = snapshot
            entry.keys, entry.values, entry.offset = keys, values, offset
            if index is not None:
                entry._idx = index

    for entry, snapshot in zip(caches, snapshots):
        one(entry, snapshot)


def _start_speculation(caches, rollback_window):
    started = []
    try:
        for entry in caches:
            try:
                entry.start_speculation(rollback_window=rollback_window)
            except TypeError:
                entry.start_speculation()
            started.append(entry)
    except Exception:
        for entry in reversed(started):
            with suppress(Exception):
                entry.stop_speculation()
        raise


def _stop_speculation(caches):
    first_error = None
    for entry in caches:
        try:
            entry.stop_speculation()
        except Exception as error:  # noqa: BLE001 - finish every cache before re-raising
            first_error = first_error or error
    if first_error is not None:
        raise first_error


def _argmax_sampler(value):
    return mx.argmax(value, axis=-1)


# Draws nothing: may run on verify rows past the accept point.
_argmax_sampler.deterministic = True


@dataclass
class _PlainInflight:
    """A dispatched plain round whose outputs the host has not read yet."""

    lanes: list
    steps: list
    transaction: Any
    logits: Any
    prepared: list
    # Rows rewound and dropped before the read (finished, failed, removed).
    dropped: set = field(default_factory=set)


@dataclass
class _Lane:
    uid: int
    remaining: deque
    cache: list
    history: list[int]
    lookup_history: list[int]
    proposer: IndexedPromptLookup
    lookback: AdaptiveLookback
    sampler: Any
    processors: list
    stop_matcher: StopSequenceMatcher
    maximum: int
    config: dict
    stats: HybridStats = field(default_factory=HybridStats)
    matcher_state: Any = None
    anchor: int | None = None
    generated: int = 0
    ready: deque = field(default_factory=deque)
    speculation_started: bool = False
    ordinary: bool = False
    admission_matches: int = 0
    admission_tokens: int = 0
    admission_consecutive: int = 0
    reprobe_at: int = 0
    cost_latch: CostAwarePLDLatch | None = None
    acceptance_window: deque = field(default_factory=deque)
    rotating: list = field(default_factory=list)
    recovery: CommittedRecoverySlot = field(default_factory=CommittedRecoverySlot)
    recovery_boundary: int | None = None
    # Widest committed verify forward; the receipt's ``target_width``.
    target_max_width: int = 1
    cost_width1_cohort_splits: int = 0
    # How this round's target rows ran: ``plain_decode`` (the ordinary
    # one-token step) or ``verify_rows`` (a verify forward: proposals, a
    # wider cohort, or a recurrent rollback epoch).  Receipts report it.
    round_target_path: str = "plain_decode"
    verify_path_rounds: int = 0
    # Topology of the lane's planes (recurrent plus KV), fixed at arming.
    hybrid_rows: bool = False
    # A pipelined plain round found a proposal it could not verify: the next
    # poll drains the pipeline so the round after it can propose.
    pipeline_blocked: bool = False
    # The finish reason of the round being committed (stop/length or None).
    round_finish: Any = None
    pipelined_rounds: int = 0
    source_scope: object = None
    source_stats: dict = field(default_factory=dict)
    apcv2_source_segments: int = 0
    recent_source_segments: int = 0


class PromptLookupBatchGenerator:
    """Prompt-lookup executor with the same seam as ``BatchGenerator``.

    Each lane owns an exact rollback epoch.  Lanes whose planes a per-round
    verify transaction owns (plain/rotating KV, or recurrent plus KV) share
    one target forward per round; the others verify B1 per lane, amortizing
    multiple predicted positions into one target forward.
    """

    # Set per instance; class defaults keep init-bypassing stubs serial.
    pipelined_plain = False
    _inflight = None
    # The hybrid cohort's verify owner, kept while its lanes are unchanged.
    _cohort_owner = None

    # Every policy key this generator reads.  ``ServingEngine`` rejects an
    # execution policy naming anything else, matching the adapters' own
    # unknown-key failure for the rest of the policy document.
    POLICY_KEYS = frozenset(
        {
            "num_draft",
            "ngram_min",
            "ngram_max",
            "hot_segments",
            "reject_ttl",
            "min_context_match",
            "max_sources",
            "recent_prompt_segments",
            "prompt_segment_tokens",
            "retrieval_segments",
            "lookback_ladder",
            "lookback_misses",
            "lookback_rejects",
            "adaptive",
            "adaptive_warmup",
            "adaptive_gate",
            "deferred_admission",
            "admission_window",
            "admission_gate",
            "admission_confirm_windows",
            "admission_reprobe_interval",
            "admission_probe_stride",
            "cost_aware_admission",
            "cost_shadow_span",
            "cost_probe_stride",
            "cost_shadow_window",
            "cost_shadow_gate",
            "cost_plain_rounds",
            "cost_explore_rounds",
            "cost_margin",
            "cost_reprobe_interval",
            "cost_park_rounds",
            "cliff_aware_span",
            "verify_cliff_start",
            "verify_cliff_end",
            "rotating_replay",
            "batched_verify",
            "pipelined_plain",
            "max_proposal_tokens",
            "memory_max_draft",
            "recent_committed_segments",
            "recent_committed_max_responses",
            "recent_committed_max_tokens",
            "recent_committed_response_tokens",
            "recent_committed_max_age_seconds",
        }
    )

    @classmethod
    def validate_policy(cls, config):
        if not isinstance(config, dict):
            raise ValueError("prompt_lookup policy must be an object")
        unknown = set(config) - cls.POLICY_KEYS
        if unknown:
            raise ValueError(
                "unknown prompt_lookup policy keys: " + ", ".join(sorted(map(str, unknown)))
            )
        validated = dict(config)
        integer_bounds = {
            "num_draft": 1,
            "ngram_min": 1,
            "ngram_max": 1,
            "hot_segments": 0,
            "reject_ttl": 0,
            "min_context_match": 0,
            "max_sources": 0,
            "recent_prompt_segments": 0,
            "prompt_segment_tokens": 1,
            "lookback_misses": 1,
            "lookback_rejects": 1,
            "adaptive_warmup": 0,
            "admission_window": 1,
            "admission_confirm_windows": 1,
            "admission_reprobe_interval": 0,
            "admission_probe_stride": 1,
            "cost_shadow_span": 2,
            "cost_probe_stride": 1,
            "cost_shadow_window": 1,
            "cost_plain_rounds": 1,
            "cost_explore_rounds": 1,
            "cost_reprobe_interval": 0,
            "cost_park_rounds": 1,
            "verify_cliff_start": 2,
            "verify_cliff_end": 1,
            "max_proposal_tokens": 1,
            "memory_max_draft": 0,
            "recent_committed_max_responses": 1,
            "recent_committed_max_tokens": 1,
            "recent_committed_response_tokens": 1,
        }
        for name, minimum in integer_bounds.items():
            if name not in validated:
                continue
            value = validated[name]
            if type(value) is not int or value < minimum:
                qualifier = "positive" if minimum else "nonnegative"
                raise ValueError(f"prompt_lookup {name} must be a {qualifier} integer")
        if validated.get("ngram_max", 6) < validated.get("ngram_min", 3):
            raise ValueError("prompt_lookup ngram_max must be at least ngram_min")
        if validated.get("min_context_match", 0) > 64:
            raise ValueError("prompt_lookup min_context_match must be at most 64")
        if validated.get("recent_prompt_segments", 0):
            if validated.get("retrieval_segments"):
                raise ValueError("prompt_lookup recent prompt segments cannot include external retrieval segments")
            if validated["recent_prompt_segments"] * validated.get("prompt_segment_tokens", 1024) > 65536:
                raise ValueError("prompt_lookup recent prompt source window exceeds 65536 tokens")
        for name in (
            "adaptive",
            "deferred_admission",
            "cost_aware_admission",
            "cliff_aware_span",
            "rotating_replay",
            "batched_verify",
            "pipelined_plain",
            "recent_committed_segments",
        ):
            if name in validated and type(validated[name]) is not bool:
                raise ValueError(f"prompt_lookup {name} must be a boolean")
        if "recent_committed_max_age_seconds" in validated:
            age = validated["recent_committed_max_age_seconds"]
            if (
                isinstance(age, bool)
                or not isinstance(age, (int, float))
                or not math.isfinite(float(age))
                or age <= 0
            ):
                raise ValueError(
                    "prompt_lookup recent_committed_max_age_seconds must be finite and positive"
                )
            validated["recent_committed_max_age_seconds"] = float(age)
        for name in ("adaptive_gate", "admission_gate", "cost_shadow_gate", "cost_margin"):
            if name not in validated:
                continue
            gate = validated[name]
            if type(gate) not in (int, float):
                raise ValueError(f"prompt_lookup {name} must be a finite probability")
            gate = float(gate)
            if not math.isfinite(gate) or not 0.0 <= gate <= 1.0:
                raise ValueError(f"prompt_lookup {name} must be a finite probability")
            validated[name] = gate
        if validated.get("cost_aware_admission") and validated.get("deferred_admission"):
            raise ValueError("prompt_lookup cost_aware_admission and deferred_admission are exclusive")
        if validated.get("cost_aware_admission") and validated.get("cost_shadow_span", 4) > validated.get("num_draft", 8):
            raise ValueError("prompt_lookup cost_shadow_span exceeds num_draft")
        if "lookback_ladder" in validated:
            ladder = validated["lookback_ladder"]
            if (
                not isinstance(ladder, (list, tuple))
                or not ladder
                or any(type(value) is not int or value < 1 for value in ladder)
                or any(right <= left for left, right in zip(ladder, ladder[1:]))
            ):
                raise ValueError(
                    "prompt_lookup lookback_ladder must contain strictly increasing positive integers"
                )
            validated["lookback_ladder"] = list(ladder)
        if "retrieval_segments" in validated:
            segments = validated["retrieval_segments"]
            if not isinstance(segments, (list, tuple)):
                raise ValueError(
                    "prompt_lookup retrieval_segments must contain nonempty integer arrays"
                )
            normalized = []
            for segment in segments:
                if (
                    not isinstance(segment, (list, tuple))
                    or not segment
                    or any(type(token) is not int or token < 0 for token in segment)
                ):
                    raise ValueError(
                        "prompt_lookup retrieval_segments must contain nonempty integer arrays"
                    )
                normalized.append(list(segment))
            validated["retrieval_segments"] = normalized
        if validated.get("verify_cliff_start", 9) > validated.get(
            "verify_cliff_end", 15
        ):
            raise ValueError("prompt_lookup verify cliff start must not exceed end")
        return validated

    def __init__(
        self,
        model,
        *,
        completion_batch_size=4,
        prefill_step_size=2048,
        prefill_step_autoscale=False,
        stop_tokens=(),
        prompt_lookup=None,
        decode_time_fairness=None,
        recent_source_store=None,
        **_kwargs,
    ):
        self.model = model
        self._recovery_revision = (
            f"{type(model).__module__}.{type(model).__qualname__}:{id(model)}"
        )
        self.capacity = int(completion_batch_size)
        self.prefill_step = int(prefill_step_size)
        if type(prefill_step_autoscale) is not bool:
            raise ValueError("prefill_step_autoscale must be boolean")
        self.prefill_step_autoscale = prefill_step_autoscale
        self.config = self.validate_policy(prompt_lookup or {})
        self.recent_source_store = recent_source_store
        if self.config.get("recent_committed_segments", False) and not isinstance(
            recent_source_store, RecentCommittedSegmentStore
        ):
            raise ValueError(
                "recent committed PLD segments require an engine-owned source store"
            )
        self.num_draft = self.config.get("num_draft", 8)
        # Default-off and generator-scoped: a per-request override never selects
        # the rotating replay transaction.  The bound counts proposed tokens;
        # the transaction also declares the anchor, hence the extra row.
        self.rotating_replay = bool(self.config.get("rotating_replay", False))
        # On unless disabled: verification stays target-exact either way, and
        # it only engages for lanes whose planes are all plain/rotating KV.
        self.batched_verify = bool(self.config.get("batched_verify", True))
        # On unless disabled: a proposal-free round of a hybrid cohort is
        # dispatched before the previous one is read, as ordinary batched
        # decode overlaps its host work with the device (``_advance_inflight``).
        self.pipelined_plain = bool(self.config.get("pipelined_plain", True))
        self._inflight = None
        self.rotating_replay_policy = RotatingReplayPolicy(
            enabled=self.rotating_replay,
            max_proposal_tokens=1
            + int(
                self.config.get(
                    "max_proposal_tokens",
                    max(
                        self.num_draft,
                        int(self.config.get("verify_cliff_end", 15)),
                    ),
                )
            ),
        )
        self.default_matcher = StopSequenceMatcher(stop_tokens or None)
        self.lanes = {}
        self._lane_failures = []
        self.next_uid = 0
        self.boundaries = {}
        # Decode rounds that delivered tokens; drives the same allocator
        # reclaim cadence as BatchGenerator (the PLD route had none).
        self._steps_counter = 0
        # Round-robin position over the lanes still prefilling.
        self._prefill_cursor = 0
        self.scheduler_stats = {
            "pld_cycles": 0,
            "pld_retrieval_cycles": 0,
            "pld_plain_cycles": 0,
            "pld_proposed": 0,
            "pld_accepted": 0,
            "pld_bonus": 0,
            "pld_fallbacks": 0,
            "pld_rollbacks": 0,
            "pld_prefill_rounds": 0,
            "pld_rotating_replay_rounds": 0,
            "pld_rotating_replay_replayed_tokens": 0,
            "pld_rotating_replay_refusals": 0,
            "pld_rotating_replay_rebuilds": 0,
            "pld_failed_round_rebuilds": 0,
            "pld_batched_rounds": 0,
            "pld_batched_lanes": 0,
            "pld_batched_max_width": 0,
            "pld_pipelined_rounds": 0,
            "pld_pipeline_drains": 0,
            "pld_pipeline_dropped_proposals": 0,
            "pld_pipeline_refuted_proposals": 0,
            "pld_pipeline_released_rows": 0,
            "pld_parked_direct_rounds": 0,
            "pld_cost_width1_cohort_splits": 0,
            "pld_recovery_checkpoint_captures": 0,
            "pld_recovery_checkpoint_restores": 0,
            "pld_recovery_checkpoint_failures": 0,
            "pld_recovery_full_rebuilds": 0,
            "target_max_width": 1,
        }
        from .adaptive_policy import DecodeTimeFairness

        # The serving route's decode-time fairness (the stall bound on a
        # prefill slice taken beside decoding lanes).  Absent, as for a
        # direct construction, it is constructed disabled and inert.
        self.decode_time_fairness = DecodeTimeFairness(
            **dict(decode_time_fairness or {})
        )
        self._sync_decode_fairness_stats()

    def _sync_decode_fairness_stats(self):
        fairness = self.decode_time_fairness
        if fairness.enabled:
            for key, value in fairness.counters.items():
                self.scheduler_stats[f"decode_fairness_{key}"] = int(value)

    def __len__(self):
        return len(self.lanes)

    @property
    def cache_nbytes(self):
        return sum(
            int(getattr(entry, "nbytes", 0))
            for lane in self.lanes.values()
            for entry in lane.cache
        )

    def insert(
        self,
        prompts,
        max_tokens=None,
        caches=None,
        all_tokens=None,
        samplers=None,
        logits_processors=None,
        stop_matchers=None,
        prompt_lookup_configs=None,
        lane_rngs=None,
        apc_interior_positions=None,
        state_boundaries=None,
        prefill_inputs=None,
        pld_source_scopes=None,
        apc_transcript_ledgers=None,
    ):
        count = len(prompts)

        def aligned(name, values, default):
            values = default if values is None else values
            if len(values) != count:
                raise ValueError(
                    f"prompt lookup {name} length {len(values)} does not match "
                    f"prompt batch length {count}"
                )
            return values

        # The serving seam passes these to every route. ``lane_rngs`` is
        # unused here because each sampler already carries its lane's RNG;
        # validate its shape nevertheless so request metadata cannot drift.
        aligned("lane_rngs", lane_rngs, [None] * count)
        prefill_inputs = aligned("prefill_inputs", prefill_inputs, [None] * count)
        apc_interior_positions = aligned(
            "apc_interior_positions", apc_interior_positions, [()] * count
        )
        state_boundaries = aligned(
            "state_boundaries", state_boundaries, [()] * count
        )
        if any(value is not None for value in prefill_inputs or ()):
            raise ValueError(
                "multimodal and neural-concept prefill are unavailable on prompt lookup"
            )
        if any(positions for positions in apc_interior_positions or ()):
            raise ValueError("APCv2 interior checkpoints are unavailable on prompt lookup")
        if any(bounds for bounds in state_boundaries or ()):
            raise ValueError("state checkpoints are unavailable on prompt lookup")
        if len(self.lanes) + len(prompts) > self.capacity:
            raise ValueError("prompt-lookup lane capacity exceeded")
        max_tokens = aligned("max_tokens", max_tokens, [128] * count)
        caches = aligned("caches", caches, [None] * count)
        all_tokens = aligned("all_tokens", all_tokens, [[] for _ in prompts])
        samplers = aligned("samplers", samplers, [None] * count)
        logits_processors = aligned(
            "logits_processors", logits_processors, [[] for _ in prompts]
        )
        stop_matchers = aligned(
            "stop_matchers", stop_matchers, [self.default_matcher] * count
        )
        prompt_lookup_configs = aligned(
            "prompt_lookup_configs", prompt_lookup_configs, [{} for _ in prompts]
        )
        pld_source_scopes = aligned(
            "pld_source_scopes", pld_source_scopes, [None] * count
        )
        apc_transcript_ledgers = aligned(
            "apc_transcript_ledgers", apc_transcript_ledgers, [None] * count
        )
        staged = []
        next_uid = self.next_uid
        for prompt, maximum, prompt_cache, prefix, sampler, processors, matcher, overrides, source_scope, apc_ledger in zip(
            prompts,
            max_tokens,
            caches,
            all_tokens,
            samplers,
            logits_processors,
            stop_matchers,
            prompt_lookup_configs,
            pld_source_scopes,
            apc_transcript_ledgers,
            strict=True,
        ):
            prompt = [int(token) for token in prompt]
            prefix = [int(token) for token in prefix]
            if not prompt or int(maximum) <= 0:
                raise ValueError("prompt lookup requires a nonempty tail and output budget")
            prompt_cache = prompt_cache or cache_module.make_prompt_cache(self.model)
            _validate_cache(prompt_cache)
            overrides = dict(overrides)
            policy = {
                **self.config,
                **{
                    name: value
                    for name, value in overrides.items()
                    if name in self.POLICY_KEYS
                },
            }
            config = {**overrides, **self.validate_policy(policy)}
            lookback_ladder = config.get(
                "lookback_ladder", (256, 1024, 4096, 16384)
            )
            recent_prompt_segments = config.get("recent_prompt_segments", 0)
            source_segments = []
            seen_segments = set()
            if config.get("recent_committed_segments", False):
                if source_scope is None:
                    raise ValueError(
                        "recent committed PLD segments require an APCv2-bound source scope"
                    )
                source_limit = config.get("recent_committed_max_responses", 64)
                token_limit = config.get("recent_committed_max_tokens", 131072)
                response_limit = config.get("recent_committed_response_tokens", 2048)
                source_tokens = 0

                if apc_ledger is not None:
                    for segment in reversed(tuple(getattr(apc_ledger, "segments", ()))):
                        source_tokens = _append_source_segment(
                            source_segments,
                            seen_segments,
                            segment.token_ids,
                            "apcv2_transcript",
                            segment.segment_id,
                            source_limit=source_limit,
                            token_limit=token_limit,
                            response_limit=response_limit,
                            source_tokens=source_tokens,
                        )
                for entry in self.recent_source_store.snapshot(source_scope):
                    source_tokens = _append_source_segment(
                        source_segments,
                        seen_segments,
                        entry.tokens,
                        "recent_committed",
                        entry.cache_id,
                        source_limit=source_limit,
                        token_limit=token_limit,
                        response_limit=response_limit,
                        source_tokens=source_tokens,
                    )
            _set_ordinary_b1_mask(
                prompt_cache, bool(config.get("cost_aware_admission", False))
            )
            proposer = IndexedPromptLookup(
                prefix + prompt,
                ngram_min=config.get("ngram_min", 3),
                ngram_max=config.get("ngram_max", 6),
                hot_segments=max(
                    config.get("hot_segments", 4),
                    len(config.get("retrieval_segments", ())) + len(source_segments),
                ),
                reject_ttl=config.get("reject_ttl", 8),
                recent_prompt_segments=recent_prompt_segments,
                prompt_segment_tokens=config.get("prompt_segment_tokens", 1024),
                index_window=(None if recent_prompt_segments else max(lookback_ladder)),
            )
            for segment in config.get("retrieval_segments", ()):
                proposer.add_hot_segment(segment)
            for segment, source_kind, source_id in source_segments:
                proposer.add_indexed_hot_segment(
                    self.recent_source_store.indexed_segment(
                        source_scope,
                        segment,
                        source_kind=source_kind,
                        source_id=source_id,
                        ngram_min=proposer.ngram_min,
                        ngram_max=proposer.ngram_max,
                        index_window=proposer.index_window,
                    )
                )
            lookback = AdaptiveLookback(
                lookback_ladder,
                misses=config.get("lookback_misses", 4),
                rejects=config.get("lookback_rejects", 2),
            )
            uid = next_uid
            next_uid += 1
            lane = _Lane(
                uid=uid,
                remaining=deque(prompt),
                cache=prompt_cache,
                history=prefix,
                lookup_history=prefix + prompt,
                proposer=proposer,
                lookback=lookback,
                sampler=sampler or _argmax_sampler,
                processors=list(processors or ()),
                stop_matcher=matcher,
                maximum=int(maximum),
                config=config,
                ordinary=bool(config.get("deferred_admission", False)),
                cost_latch=(CostAwarePLDLatch(
                    shadow_span=config.get("cost_shadow_span", 4),
                    probe_stride=config.get("cost_probe_stride", 4),
                    shadow_window=config.get("cost_shadow_window", 2),
                    shadow_gate=config.get("cost_shadow_gate", 0.75),
                    plain_rounds=config.get("cost_plain_rounds", 8),
                    explore_rounds=config.get("cost_explore_rounds", 2),
                    cost_margin=config.get("cost_margin", 0.05),
                    reprobe_interval=config.get("cost_reprobe_interval", 32),
                    park_rounds=config.get("cost_park_rounds", 4),
                ) if config.get("cost_aware_admission", False) else None),
                source_scope=source_scope,
                source_stats={
                    kind: {"cycles": 0, "proposed": 0, "accepted": 0}
                    for kind in (
                        "local",
                        "retrieval",
                        "apcv2_transcript",
                        "recent_committed",
                    )
                },
                apcv2_source_segments=sum(
                    source_kind == "apcv2_transcript"
                    for _segment, source_kind, _source_id in source_segments
                ),
                recent_source_segments=sum(
                    source_kind == "recent_committed"
                    for _segment, source_kind, _source_id in source_segments
                ),
            )
            if lane.cost_latch is not None:
                lane.ordinary = True
            lane.matcher_state = matcher.make_state()
            staged.append(lane)
        # Batch admission is atomic: no lane becomes schedulable until every
        # lane has passed validation and construction.
        self.lanes.update((lane.uid, lane) for lane in staged)
        self.next_uid = next_uid
        return [lane.uid for lane in staged]

    def _prefill(self, lane, *, contended=False):
        if len(lane.remaining) > 1:
            total = len(lane.history) + len(lane.remaining)
            step = (
                prompt_length_prefill_step(total, maximum=self.prefill_step)
                if self.prefill_step_autoscale
                else self.prefill_step
            )
            fairness = self.decode_time_fairness
            if contended:
                # Beside decoding lanes the slice is stall-bounded, as on the
                # ordinary route.  Before, it was the configured (or prompt-
                # length autoscaled) step whatever the decode lanes paid: 512
                # rows on a 12K prompt, 2048 at 32-64K and 8192 above 64K.
                step = fairness.floor_slice(
                    fairness.cap(step, contended=True, depth=len(lane.history)),
                    step,
                    contended=True,
                )
            count = min(step, len(lane.remaining) - 1)
            depth = len(lane.history)
            inputs = [lane.remaining.popleft() for _ in range(count)]
            tic = time.perf_counter()
            with mx.stream(generation_stream):
                self.model(mx.array([inputs], dtype=mx.uint32), cache=lane.cache)
                mx.eval(_walk_state(lane.cache))
            if fairness.enabled:
                fairness.observe_prefill(
                    count, time.perf_counter() - tic, contended=contended,
                    depth=depth,
                )
                self._sync_decode_fairness_stats()
            mx.clear_cache()
            lane.history.extend(inputs)
            self.scheduler_stats["pld_prefill_rounds"] += 1
            # As the ordinary prefill loop does: the recurrent and sliding
            # restore snapshots are what let a later prompt that shares only
            # a prefix of this one trim the stored boundary back to it.
            # Without them a hybrid model's boundary serves exact repeats only.
            cache_module.record_state_checkpoints(
                lane.cache,
                [len(lane.history)],
                force=len(lane.remaining) == 1,
            )
        done = len(lane.remaining) == 1
        # (done, span) over the whole prompt, as the ordinary generator's
        # prompt responses report it; the final token is consumed by decode.
        span = len(lane.history) + len(lane.remaining)
        progress = (span if done else len(lane.history), span)
        if done:
            lane.anchor = lane.remaining.popleft()
            self.boundaries[lane.uid] = {
                "committed_only": True,
                "tokens": list(lane.history),
                "target_cache": self._freeze_cache(lane.cache),
                "covered_tokens": len(lane.history),
            }
            self._capture_lane_recovery(lane)
            self._arm_lane_speculation(lane)
        return SimpleNamespace(
            uid=lane.uid,
            prompt_progress=(len(lane.history), len(lane.lookup_history)),
            progress=progress,
            end_of_prompt=done,
            end_of_segment=done,
        )

    def _freeze_cache(self, cache):
        """Independent committed cache for boundaries and finishes.

        Deep copy by default; ``MLX_LM_EXTERNAL_ROUND_COW=1`` selects
        descriptor COW, falling back to the deep copy for a graph with live
        speculation state.
        """
        frozen, _sidecar, mode = snapshot_committed_cache(cache)
        if mode == "descriptor_cow":
            key = "pld_cow_snapshots"
        elif mode == "deepcopy_fallback":
            key = "pld_cow_fallbacks"
        else:
            return frozen
        self.scheduler_stats[key] = self.scheduler_stats.get(key, 0) + 1
        return frozen

    @staticmethod
    def _snapshot_recovery_cache(cache):
        # Append-only KV planes keep only their fill level; an alias would
        # make every append of the next round copy the whole buffer.
        snapshot, _sidecar, borrowed = snapshot_recovery_descriptors(cache)
        return snapshot, borrowed

    @staticmethod
    def _restore_recovery_cache(frozen):
        snapshot, borrowed = frozen
        restored, _sidecar = restore_recovery_descriptors(snapshot, None, borrowed)
        return restored

    def _arm_lane_speculation(self, lane):
        lane.hybrid_rows = self._hybrid(lane)
        # An explicit rotating-replay policy selects that per-lane transaction.
        lane.batched = (
            self.batched_verify
            and not self.rotating_replay
            and self._batchable(lane.cache)
            and (
                getattr(self.model, "supports_speculative_rollback", False)
                or not lane.hybrid_rows
            )
        )
        if lane.batched:
            # The per-round segmented transaction is this lane's rollback; it
            # refuses planes that carry another speculation epoch.
            return
        _start_speculation(
            lane.cache,
            max(
                64,
                self.num_draft + 2,
                lane.config.get("verify_cliff_end", 15) + 2,
            ),
        )
        lane.speculation_started = True
        if self.rotating_replay:
            # Rotating rings use a per-round replay transaction rather than the
            # lane-long rollback epoch carried by the other cache planes.
            lane.rotating = _rotating_leaves(lane.cache)
            for entry in lane.rotating:
                entry.stop_speculation()

    @staticmethod
    def _hybrid(lane):
        from .hybrid_verify_rows import is_hybrid_rows

        return is_hybrid_rows([lane.cache])

    def _capture_lane_recovery(self, lane):
        """Publish the lane's latest exact, request-private committed boundary."""
        was_armed = lane.speculation_started
        if was_armed:
            _stop_speculation(lane.cache)
            lane.speculation_started = False
        try:
            lane.recovery.capture(
                route="prompt_lookup",
                revision=self._recovery_revision,
                boundary=len(lane.history),
                value=lane.cache,
                snapshot=self._snapshot_recovery_cache,
                restore=self._restore_recovery_cache,
            )
            self.scheduler_stats["pld_recovery_checkpoint_captures"] += 1
            lane.recovery_boundary = len(lane.history)
        except Exception:
            lane.recovery.invalidate()
            lane.recovery_boundary = None
            self.scheduler_stats["pld_recovery_checkpoint_failures"] += 1
        finally:
            if was_armed:
                self._arm_lane_speculation(lane)

    def _rebuild_lane_cache(
        self, lane, committed_inputs, taps=None, steer=None,
        counter="pld_rotating_replay_rebuilds",
    ):
        """Restore the last committed checkpoint, falling back to full re-prefill.

        The checkpoint may be older than ``lane.history`` (batched lanes keep
        only their prefill-end one): the committed tokens past its boundary
        are replayed after it.  ``steer`` covers ``committed_inputs`` position
        by position, so the replay recomputes the steered K/V the verify
        forward produced.
        """
        self.scheduler_stats[counter] = self.scheduler_stats.get(counter, 0) + 1
        with suppress(Exception):
            self._close_lane(lane)
        boundary = lane.recovery_boundary
        lane.cache = None
        if boundary is not None and boundary <= len(lane.history):
            try:
                lane.cache = lane.recovery.restore(
                    route="prompt_lookup",
                    revision=self._recovery_revision,
                    boundary=boundary,
                )
            except RecoveryCheckpointMismatch:
                lane.cache = None
        if lane.cache is not None:
            self.scheduler_stats["pld_recovery_checkpoint_restores"] += 1
            self._arm_lane_speculation(lane)
            tokens = list(lane.history[boundary:]) + list(committed_inputs)
        else:
            self.scheduler_stats["pld_recovery_full_rebuilds"] += 1
            lane.cache = cache_module.make_prompt_cache(self.model)
            tokens = list(lane.history) + list(committed_inputs)
        _set_ordinary_b1_mask(lane.cache, lane.cost_latch is not None)
        # The committed inputs are the tail of ``tokens``.  Earlier generated
        # positions (after an older checkpoint, or in a full re-prefill) are
        # replayed unsteered; this is the last-resort path after a failed
        # transaction.
        steer_from = len(tokens) - len(committed_inputs)
        with mx.stream(generation_stream):
            step = (
                prompt_length_prefill_step(len(tokens), maximum=self.prefill_step)
                if self.prefill_step_autoscale
                else self.prefill_step
            )
            for start in range(0, len(tokens), step):
                chunk = tokens[start : start + step]
                chunk_steer = _steer_slice(
                    steer, start - steer_from, start + len(chunk) - steer_from
                )
                if chunk_steer is not None:
                    taps.steer = chunk_steer
                try:
                    self.model(mx.array([chunk], dtype=mx.uint32), cache=lane.cache)
                    mx.eval(_walk_state(lane.cache))
                finally:
                    if chunk_steer is not None:
                        taps.steer = None
        if not lane.speculation_started:
            self._arm_lane_speculation(lane)

    @staticmethod
    def _processed_row(lane, logits, context=None):
        """The lane's processed log-softmax row, left lazy.

        ``context`` is the token array the processors see (lookup history and
        the row's tentative drafts); unused without processors.
        """
        value = logits[None]
        for processor in lane.processors:
            value = processor(context, value)
        row = value[0].astype(mx.float32)
        return row - mx.logsumexp(row)

    @classmethod
    def _prepare_plain_row(cls, lane, logits):
        """Lazy row, token and finiteness of an anchor-only round.

        The anchor's row is always sampled, so drawing it at dispatch takes
        the same single draw from the lane's stream as drawing it after the
        read; the token can then feed the next round before the host has it.
        """
        row = cls._processed_row(lane, logits[0])
        token = lane.sampler(row[None])[0]
        return {
            "rows": [row],
            "tokens": mx.stack([token]),
            "finite": mx.stack([_finite_at(row, token)]),
        }

    @classmethod
    def _prepare_rows(cls, lane, logits, proposal):
        """Lazy rows (and, for a deterministic sampler, tokens) of one lane.

        Built beside the verify forward and evaluated with it, so a lane
        without processors costs no host sync per verify row.  Its rows do
        not depend on which drafts are accepted, and a sampler that draws
        nothing (``deterministic``) may run on rows past the accept point.
        A lane with processors keeps the sequential per-row path: they see
        the tentative drafts and may count the rows they are shown.
        """
        if lane.processors:
            return None
        rows = [cls._processed_row(lane, logits[index]) for index in range(len(proposal) + 1)]
        prepared = {"rows": rows}
        if getattr(lane.sampler, "deterministic", False):
            tokens = [lane.sampler(row[None])[0] for row in rows]
            prepared["tokens"] = mx.stack(tokens)
            prepared["finite"] = mx.stack([_finite_at(row, token) for row, token in zip(rows, tokens)])
        return prepared

    @staticmethod
    def _prepared_arrays(prepared):
        return [
            value
            for item in prepared
            if item is not None
            for value in (item["rows"], item.get("tokens"), item.get("finite"))
            if value is not None
        ]

    def _round(self, lane):
        """One lane, one verify forward: snapshot, verify, rewind and replay."""
        lane.round_width = 1
        steps = self._round_steps(lane)
        inputs, proposal = next(steps)
        if proposal and lane.recovery_boundary != len(lane.history):
            # A proposal-free round verifies only its anchor, which is always
            # consumed, so it needs no recovery snapshot. Capture the exact
            # committed boundary immediately before the next verify attempt.
            self._capture_lane_recovery(lane)
        transaction = None
        if lane.rotating and proposal:
            try:
                transaction = RotatingReplayTransaction(
                    lane.rotating,
                    inputs,
                    request_id=str(lane.uid),
                    state_revision=f"pld:{lane.uid}:{len(lane.history)}",
                    policy=self.rotating_replay_policy,
                )
            except RotatingReplayError:
                # Refused rounds keep the exact copy snapshot below.
                self.scheduler_stats["pld_rotating_replay_refusals"] += 1
        # A proposal-free round verifies only the anchor, which is always
        # consumed, so its rotating rings need neither transaction nor copy.
        snapshots = _snapshot(
            lane.cache,
            skip_rotating=transaction is not None
            or (bool(lane.rotating) and not proposal),
        )
        # A lane-long rollback epoch puts even an anchor-only round of a
        # recurrent lane through the verify kernel.
        lane.round_target_path = (
            "verify_rows"
            if proposal or (lane.speculation_started and self._hybrid(lane))
            else "plain_decode"
        )
        taps, steer, commits = self._verify_steer([lane], [inputs])
        try:
            if steer is not None:
                taps.steer = steer
            try:
                with mx.stream(generation_stream):
                    logits = self.model(
                        mx.array([inputs], dtype=mx.uint32), cache=lane.cache
                    )[0]
                    prepared = self._prepare_rows(lane, logits, proposal)
                    mx.eval(logits, self._prepared_arrays([prepared]))
            finally:
                if steer is not None:
                    taps.steer = None
        except BaseException:
            if transaction is not None:
                with suppress(Exception):
                    transaction.rollback()
                for entry in lane.rotating:
                    entry.stop_speculation()
            raise
        try:
            consumed = steps.send((logits, prepared))
        except PromptLookupLaneFailure:
            # The verify forward touched this lane's cache, but no generated
            # token or checkpoint has been published.  Restore the snapshot
            # before the executor drops the failed request.
            if transaction is not None:
                with suppress(Exception):
                    transaction.rollback()
            with suppress(Exception):
                _rewind(lane.cache, snapshots)
            raise
        if consumed < len(inputs):
            self.scheduler_stats["pld_rollbacks"] += 1
            _rewind(lane.cache, snapshots)
            committed_inputs = inputs[:consumed]

            def replay(tokens):
                # The one replay forward of the round: it advances every cache
                # of the lane, rotating or not, over the committed inputs,
                # steered exactly as the verify forward steered them.
                replay_steer = _steer_slice(steer, 0, len(tokens))
                if replay_steer is not None:
                    taps.steer = replay_steer
                try:
                    with mx.stream(generation_stream):
                        self.model(
                            mx.array([list(tokens)], dtype=mx.uint32),
                            cache=lane.cache,
                        )
                        mx.eval(_walk_state(lane.cache))
                finally:
                    if replay_steer is not None:
                        taps.steer = None

            if transaction is not None:
                # Restores the rotating rings, then shares ``replay`` with the
                # caches ``_rewind`` just restored.
                try:
                    transaction.commit(consumed, replay)
                except RotatingReplayError:
                    # No ring copy exists to return to, and a failed lane must
                    # not take the worker with it. The token history is exact,
                    # so rebuild this lane's cache from it.
                    self._rebuild_lane_cache(lane, committed_inputs, taps, steer)
                else:
                    self.scheduler_stats[
                        "pld_rotating_replay_replayed_tokens"
                    ] += consumed
            elif committed_inputs:
                replay(committed_inputs)
        elif transaction is not None:
            try:
                transaction.commit_verified()
            except RotatingReplayError:
                self._rebuild_lane_cache(lane, inputs, taps, steer)
        if transaction is not None:
            self.scheduler_stats["pld_rotating_replay_rounds"] += 1
        for commit in commits:
            commit(consumed)
        with suppress(StopIteration):
            next(steps)

    def _verify_steer(self, lanes, blocks):
        """Residual steering for a verify forward (thinking guard alpha actuator).

        Per position, exactly what ordinary decode applies to the step that
        feeds that input: rows before the first thinking-close token are
        steered (the anchor included), the close and later rows are not.
        Returns ``(taps, steer, commits)``; ``commits[i](consumed)`` counts
        lane ``i``'s committed steered positions once the round commits.
        """
        taps = getattr(getattr(self.model, "model", None), "residual_taps", None)
        if taps is None:
            return None, None, []
        requests, commits = [], []
        for lane, inputs in zip(lanes, blocks):
            # ``history`` is every token before the anchor, the context the
            # ordinary step feeding the anchor saw.
            rows, commit = verify_block_steer(lane.processors, lane.history, inputs)
            requests.append(rows)
            commits.append(commit)
        steer = stack_block_steer(requests, max(len(inputs) for inputs in blocks))
        if steer is None:
            return None, None, commits
        return taps, steer, commits

    @staticmethod
    def _batchable(cache):
        """Lanes whose planes a per-round verify transaction owns exactly.

        Plain and rotating single-row KV planes share a segmented KV
        transaction; recurrent (``ArraysCache``) plus KV planes share the
        hybrid record-and-trim transaction the external route uses.
        """
        from .hybrid_verify_rows import is_hybrid_rows

        return (
            isinstance(cache, list)
            and bool(cache)
            and (
                all(type(entry) in (KVCache, RotatingKVCache) for entry in cache)
                or is_hybrid_rows([cache])
            )
        )

    def _begin_verify(self, lanes, lengths):
        """Open the verify transaction chosen by cache topology, never by model.

        A hybrid cohort keeps its owner (and so its batched views) from round
        to round while its lanes and their cache planes are the same objects.
        """
        from .hybrid_verify_rows import HybridVerifyRows, is_hybrid_rows
        from .segmented_rotating_kv import SegmentedKVRows

        rows = [lane.cache for lane in lanes]
        owner = self._cohort_owner
        if owner is None or not owner.holds(rows):
            owner = self._cohort_owner = None
            if is_hybrid_rows(rows):
                owner = self._cohort_owner = HybridVerifyRows(rows)
        if owner is not None:
            # A round of anchors only is a plain decode step: no rollback
            # epoch, no verify kernel, nothing to trim at commit.
            return owner.begin(lengths, plain_single_token=True)
        return SegmentedKVRows(rows).begin(lengths=lengths)

    def _round_batched(self, lanes):
        """Verify several lanes in one target forward.

        Each lane keeps its own request-private B1 caches; the segmented
        transaction appends every lane's verify block (ragged lengths), and the
        commit republishes exactly the consumed prefix per lane from the K/V it
        already computed, so a rejected tail costs no replay forward.  Hybrid
        lanes trim their recurrent rows the same way, from the rollback
        records the verify forward stages (``HybridVerifyRows``).
        """
        for lane in lanes:
            lane.round_width = len(lanes)
        steps = [self._round_steps(lane) for lane in lanes]
        plans = [next(step) for step in steps]
        # No per-round recovery capture: the transaction is the exact
        # rollback, and a failed round rebuilds from the prefill-end
        # checkpoint plus the committed tokens (``_recover_uncommitted_round``).
        # A capture walked every cache plane of every proposing lane, every
        # round, for a checkpoint the batched path never restored.
        lengths = [len(inputs) for inputs, _proposal in plans]
        width = max(lengths)
        if width == 1 and self._pipeline_eligible(lanes):
            # A proposal-free round of a hybrid cohort: dispatch it now and
            # read it next poll, after the round that follows is dispatched.
            self._inflight = self._dispatch_plain(
                lanes, steps, [inputs[0] for inputs, _proposal in plans]
            )
            return
        transaction = self._begin_verify(lanes, lengths)
        # One forward serves the cohort: a lane without a proposal still runs
        # as a verify row unless the whole round is one plain decode step.
        path = (
            "verify_rows"
            if width > 1 or not getattr(transaction, "plain", True)
            else "plain_decode"
        )
        for lane in lanes:
            lane.round_target_path = path
        try:
            padded = [inputs + [0] * (width - len(inputs)) for inputs, _proposal in plans]
            taps, steer, commits = self._verify_steer(
                lanes, [inputs for inputs, _proposal in plans]
            )
            if steer is not None:
                taps.steer = steer
            try:
                with mx.stream(generation_stream):
                    logits = self.model(
                        mx.array(padded, dtype=mx.uint32), cache=transaction.caches
                    )
                    # Every lane's rows (and deterministic tokens) evaluate
                    # with the forward: one host sync for the cohort.
                    prepared = [
                        self._prepare_rows(lane, logits[row], plans[row][1])
                        for row, lane in enumerate(lanes)
                    ]
                    mx.eval(logits, self._prepared_arrays(prepared))
            finally:
                if steer is not None:
                    taps.steer = None
            # Each row samples from its own lane's stream (LaneRNG never
            # rewinds) and runs its processors, so a peer's failure must not
            # discard the rows already sampled: that advanced the healthy
            # lanes' streams without committing their tokens, and their seeded
            # output then depended on the neighbour.  A failed row keeps only
            # its anchor (its cache is dropped with the lane); the rest commit.
            consumed, failures, failed_rows = [], [], set()
            for row, step in enumerate(steps):
                try:
                    consumed.append(step.send((logits[row], prepared[row])))
                except PromptLookupLaneFailure as error:
                    failures.append(error)
                    failed_rows.add(row)
                    consumed.append(1)
            transaction.commit(accepted_lengths=consumed)
            transaction = None
            for row, (commit, count) in enumerate(zip(commits, consumed)):
                if row not in failed_rows:
                    commit(count)
        except BaseException:
            # Nothing of this round is committed: every lane must be back at
            # its committed boundary before the error leaves, whether the
            # caller then drops one lane or retries the cohort.
            if transaction is not None:
                self._recover_uncommitted_round(lanes, transaction)
            raise
        self.scheduler_stats["pld_batched_rounds"] += 1
        self.scheduler_stats["pld_batched_lanes"] += len(lanes)
        self.scheduler_stats["pld_batched_max_width"] = max(
            self.scheduler_stats["pld_batched_max_width"], len(lanes)
        )
        self.scheduler_stats["pld_rollbacks"] += sum(
            int(done < length)
            for row, (done, length) in enumerate(zip(consumed, lengths))
            if row not in failed_rows
        )
        for row, step in enumerate(steps):
            if row not in failed_rows:
                with suppress(StopIteration):
                    next(step)
        if failures:
            raise PromptLookupRoundFailures(failures)

    def _pipeline_eligible(self, lanes):
        """Lanes whose anchor-only round may run before the host reads them.

        Hybrid planes (their plain transaction can release a finished row),
        no processors (nothing reads host tokens or steers the forward), no
        cost latch (it times one round at a time).
        """
        return self.pipelined_plain and all(
            lane.batched
            and lane.hybrid_rows
            and lane.cost_latch is None
            and not lane.processors
            for lane in lanes
        )

    def _dispatch_plain(self, lanes, steps, anchors):
        """Begin and dispatch one anchor-only round; nothing is read.

        ``anchors`` are host tokens or the previous round's lazy tokens.
        """
        transaction = self._begin_verify(lanes, [1] * len(lanes))
        try:
            if not getattr(transaction, "plain", False):
                raise RuntimeError("a pipelined round requires a plain transaction")
            with mx.stream(generation_stream):
                if isinstance(anchors, mx.array):
                    tokens = anchors.astype(mx.uint32)[:, None]
                else:
                    tokens = mx.array([[int(a)] for a in anchors], dtype=mx.uint32)
                logits = self.model(tokens, cache=transaction.caches)
                prepared = [
                    self._prepare_plain_row(lane, logits[row])
                    for row, lane in enumerate(lanes)
                ]
                mx.async_eval(logits, self._prepared_arrays(prepared))
        except BaseException:
            self._recover_uncommitted_round(lanes, transaction)
            raise
        return _PlainInflight(
            lanes=list(lanes),
            steps=list(steps),
            transaction=transaction,
            logits=logits,
            prepared=prepared,
        )

    def _release_inflight_row(self, inflight, row):
        """Rewind one lane out of a dispatched plain round."""
        if row in inflight.dropped:
            return
        inflight.dropped.add(row)
        self.scheduler_stats["pld_pipeline_released_rows"] += 1
        inflight.transaction.release_row(row)

    def _advance_inflight(self, together):
        """Dispatch the next plain round if the cohort allows, then read this one.

        Returns the lanes that still need a round this poll: none when the
        pipeline continues, else the cohort (and any newcomers) for an
        ordinary ``_round_batched`` after the drain.

        The pipelined round is committed before the next begins: a plain
        round consumes every anchor, so its commit needs no host token.  A
        lane that finishes (or fails) on the read round had already been fed
        to the next one; that row is released, back to the boundary the read
        round committed, before the finished lane's cache is frozen.
        """
        inflight, self._inflight = self._inflight, None
        live = [
            lane for row, lane in enumerate(inflight.lanes)
            if row not in inflight.dropped
        ]
        proceed = (
            self.pipelined_plain
            and len(live) == len(inflight.lanes)
            and {id(lane) for lane in together} == {id(lane) for lane in live}
            and not any(lane.pipeline_blocked for lane in live)
            # The read round must not finish a lane by length.
            and all(lane.generated + 2 <= lane.maximum for lane in live)
        )
        try:
            inflight.transaction.commit(accepted_lengths=[1] * len(inflight.lanes))
        except BaseException:
            # Unread, so the host never saw this round: back to before it.
            self._recover_uncommitted_round(live, inflight.transaction)
            raise
        following = dispatch_error = None
        if proceed:
            anchors = mx.concatenate(
                [prepared["tokens"] for prepared in inflight.prepared]
            )
            try:
                following = self._dispatch_plain(
                    live, [None] * len(live), anchors
                )
            except BaseException as error:  # noqa: BLE001 - read the round first
                dispatch_error = error
        else:
            self.scheduler_stats["pld_pipeline_drains"] += 1
        try:
            self._read_inflight(inflight, following)
        except PromptLookupLaneFailure:
            raise
        except BaseException:
            # The read round is committed to the caches but not (or not for
            # every lane) to the host: abandon the following round and
            # rebuild each lane from its committed history.
            self._inflight = None
            if following is not None and not following.transaction.closed:
                with suppress(Exception):
                    following.transaction.abort()
            for lane in live:
                if lane.uid in self.lanes:
                    self._rebuild_lane_cache(
                        lane, [], counter="pld_failed_round_rebuilds"
                    )
            raise
        if dispatch_error is not None:
            raise dispatch_error
        if self._inflight is not None:
            return []
        # A lane the drained round finished leaves this poll.
        return [
            lane for lane in together
            if lane.uid in self.lanes
            and not (lane.ready and lane.ready[-1].finish_reason)
        ]

    def _read_inflight(self, inflight, following):
        """Deliver a dispatched plain round; start ``following``'s lane rounds."""
        lanes = inflight.lanes
        mx.eval(inflight.logits, self._prepared_arrays(inflight.prepared))
        rows = [row for row in range(len(lanes)) if row not in inflight.dropped]
        # Phase 1: sample checks and stop/length for every lane, no commit.
        failures, sent = [], []
        for row in rows:
            lane = lanes[row]
            lane.round_width = len(lanes)
            lane.round_target_path = "plain_decode"
            try:
                inflight.steps[row].send(
                    (inflight.logits[row], inflight.prepared[row])
                )
            except PromptLookupLaneFailure as error:
                failures.append((row, error))
                continue
            sent.append(row)
        # Phase 2: lanes that stop here leave the following round first.
        if following is not None:
            leaving = {row for row, _error in failures} | {
                row for row in sent if lanes[row].round_finish is not None
            }
            try:
                for row, lane in enumerate(following.lanes):
                    if any(lane is lanes[other] for other in leaving):
                        self._release_inflight_row(following, row)
            except BaseException:  # noqa: BLE001 - fall back to the exact abort
                rows_alive = list(following.lanes)
                following.dropped.update(range(len(rows_alive)))
                try:
                    following.transaction.abort()
                except BaseException:  # noqa: BLE001 - rebuild instead
                    for lane in rows_alive:
                        # Before its bookkeeping the lane's history lacks this
                        # round's anchor, which its cache already holds.
                        self._rebuild_lane_cache(
                            lane, [lane.anchor], counter="pld_failed_round_rebuilds"
                        )
                following = None
        # Phase 3: bookkeeping (and, for a finished lane, its frozen cache).
        for row in sent:
            lanes[row].pipelined_rounds += 1
            with suppress(StopIteration):
                next(inflight.steps[row])
        self.scheduler_stats["pld_batched_rounds"] += 1
        self.scheduler_stats["pld_batched_lanes"] += len(lanes)
        self.scheduler_stats["pld_pipelined_rounds"] += 1
        self.scheduler_stats["pld_batched_max_width"] = max(
            self.scheduler_stats["pld_batched_max_width"], len(lanes)
        )
        # Phase 4: the following round's lanes start their own rounds.
        if following is not None:
            for row, lane in enumerate(following.lanes):
                if row in following.dropped:
                    continue
                following.steps[row] = self._round_steps(lane, plain=True)
                next(following.steps[row])
            if len(following.dropped) == len(following.lanes):
                following.transaction.abort()
                following = None
        self._inflight = following
        if failures:
            raise PromptLookupRoundFailures([error for _row, error in failures])

    def _recover_uncommitted_round(self, lanes, transaction):
        """Return every lane of a failed batched round to its committed boundary.

        A completed abort is exact (KV trimmed, recurrent rows restored).  A
        transaction that closed itself while failing (a commit error) or
        whose abort raised may have rewound some planes and not others, so
        each lane is rebuilt from its recovery checkpoint or re-prefilled
        from its committed history instead of being trusted.
        """
        self._cohort_owner = None
        if not transaction.closed:
            try:
                transaction.abort()
                return
            except Exception:  # noqa: BLE001, S110 - the rebuild below is authoritative
                pass
        for lane in lanes:
            self._rebuild_lane_cache(lane, [], counter="pld_failed_round_rebuilds")

    def _round_steps(self, lane, plain=False):
        """One lane's round; ``plain`` for a round already dispatched plain.

        A pipelined round's forward was built before this lane's anchor was
        on the host, so it verifies the anchor only.  Its lookup still runs
        (one per round, as always); a proposal it finds is dropped and blocks
        the pipeline, so the lane proposes again next round.  The round's own
        target token still tests the dropped proposal's first draft: a
        mismatch is a rejection the adaptive gate and the source TTL count,
        exactly as a verify round's first-position rejection; a match is no
        evidence yet (the next round's proposal continues it).
        """
        cost = lane.cost_latch
        lane.pipeline_blocked = False
        lane.round_finish = None
        cost_memory_cap = lane.config.get("memory_max_draft")
        cost_memory_ok = cost is None or cost_memory_cap is None or cost_memory_cap >= cost.shadow_span
        if cost is not None:
            if not cost_memory_ok:
                cost.suspend_for_memory(lane.generated)
                lane.ordinary = True
            elif lane.round_width != 1:
                cost.suspend_for_width(lane.generated)
                lane.ordinary = True
        round_started_ns = time.perf_counter_ns() if cost is not None and lane.round_width == 1 else None
        remaining_budget = lane.maximum - lane.generated
        nominal_proposal_budget = max(0, min(self.num_draft, remaining_budget - 1))
        proposal_budget = nominal_proposal_budget
        if lane.config.get("cliff_aware_span", False) and proposal_budget:
            proposal_budget = plan_proposal_around_verify_cliff(
                proposal_budget,
                max(remaining_budget - 1, 0),
                1,
                cliff_start=int(lane.config.get("verify_cliff_start", 9)),
                cliff_end=int(lane.config.get("verify_cliff_end", 15)),
            )
        deferred = bool(lane.config.get("deferred_admission", False))
        probing = (
            deferred
            and lane.ordinary
            and lane.generated >= lane.reprobe_at
            and lane.generated % lane.config.get("admission_probe_stride", 1) == 0
        )
        probe_candidate = (
            lane.proposer.propose(
                1, lookback=lane.lookback.current,
                min_context_match=lane.config.get("min_context_match", 0),
                max_sources=lane.config.get("max_sources", 0),
            )
            if probing
            else []
        )
        if cost is not None and cost_memory_ok and cost.should_probe(lane.generated) and lane.round_width == 1:
            shadow_candidate = lane.proposer.propose(
                cost.shadow_span,
                lookback=lane.lookback.current,
                min_context_match=lane.config.get("min_context_match", 0),
                max_sources=lane.config.get("max_sources", 0),
            )
            cost.start_shadow(shadow_candidate)
        proposal = [] if lane.ordinary else lane.proposer.propose(
            proposal_budget,
            lookback=lane.lookback.current,
            min_context_match=lane.config.get("min_context_match", 0),
            max_sources=lane.config.get("max_sources", 0),
        )
        dropped = list(proposal) if plain and proposal else None
        if dropped:
            lane.pipeline_blocked = True
            self.scheduler_stats["pld_pipeline_dropped_proposals"] += 1
            proposal = []
        proposal_source = lane.proposer.last_source_kind if proposal else None
        if proposal and len(proposal) > nominal_proposal_budget:
            lane.stats.span_extend_cycles += 1
            lane.stats.span_extend_tokens += len(proposal) - nominal_proposal_budget
        verify_rows = len(proposal) + 1
        cliff_start = int(lane.config.get("verify_cliff_start", 9))
        cliff_end = int(lane.config.get("verify_cliff_end", 15))
        if (
            lane.config.get("cliff_aware_span", False)
            and cliff_start <= verify_rows <= cliff_end
        ):
            safe = max(cliff_start - 2, 0)
            if len(proposal) > safe:
                lane.stats.span_snap_cycles += 1
                lane.stats.span_snap_tokens += len(proposal) - safe
                proposal = proposal[:safe]
        # Serving admission may seat a lane below the full verify span when
        # the host cannot hold num_draft + 1 rows; 0 decodes it at width one.
        memory_cap = lane.config.get("memory_max_draft")
        if memory_cap is not None and len(proposal) > memory_cap:
            proposal = proposal[:memory_cap]
        inputs = [lane.anchor] + proposal
        # The driver owns the verify forward and the cache transaction, so a
        # round can run alone or share one batched forward with other lanes.
        logits, prepared = yield inputs, proposal
        emitted = []
        accepted = 0
        vocab = int(logits.shape[-1])
        tokens = finite = context = None
        if prepared is not None and "tokens" in prepared:
            # Evaluated with the forward: plain host reads, no sync.
            tokens = prepared["tokens"].tolist()
            finite = prepared["finite"].tolist()
        elif prepared is None and lane.processors:
            # One context array per round; row ``index`` sees its prefix.
            context = mx.array(lane.lookup_history + proposal, dtype=mx.uint32)
        for index in range(len(proposal) + 1):
            if prepared is not None:
                row = prepared["rows"][index]
            else:
                row = self._processed_row(
                    lane, logits[index],
                    None if context is None
                    else context[: len(lane.lookup_history) + index],
                )
            if tokens is not None:
                token, token_finite = int(tokens[index]), bool(finite[index])
            else:
                # The sampler may draw: only rows up to the accept point,
                # one host sync each (token and its finiteness together).
                sampled = lane.sampler(row[None])[0]
                sampled_finite = _finite_at(row, sampled)
                mx.eval(row, sampled, sampled_finite)
                token, token_finite = int(sampled.item()), bool(sampled_finite.item())
            reason = _output_reason(token, token_finite, vocab)
            if reason is not None:
                raise PromptLookupLaneFailure(lane.uid, reason)
            from_draft = index < len(proposal) and token == proposal[index]
            emitted.append((token, row, from_draft))
            if not from_draft:
                break
            accepted += 1

        # Stop/length truncation is part of the cache transaction. The cache
        # must cover anchor + every emitted token except the final new anchor.
        finish_reason = None
        delivered = []
        matcher_state = lane.matcher_state
        for token, row, from_draft in emitted:
            generated = lane.generated + len(delivered) + 1
            matcher_state, matched = StopSequenceMatcher.match(
                matcher_state, lane.stop_matcher._trie, token
            )
            finish = "stop" if matched else "length" if generated >= lane.maximum else None
            delivered.append((token, row, from_draft))
            if finish:
                finish_reason = finish
                break
        consumed = len(delivered)
        lane.round_finish = finish_reason
        # Publish exactly ``consumed`` verified inputs to the lane's cache.
        yield consumed
        lane.history.extend(inputs[:consumed])
        lane.matcher_state = matcher_state
        lane.anchor = delivered[-1][0]
        lane.generated += len(delivered)
        lane.stats.cycles += 1
        lane.stats.verify_span_hist[len(inputs)] = lane.stats.verify_span_hist.get(len(inputs), 0) + 1
        _round_accepted = sum(item[2] for item in delivered)
        lane.stats.verify_accept_hist[_round_accepted] = (
            lane.stats.verify_accept_hist.get(_round_accepted, 0) + 1
        )
        probe_matched = int(
            bool(probing and delivered and probe_candidate and probe_candidate[0] == delivered[0][0])
        )
        feedback_proposed = len(probe_candidate) if probing else len(proposal)
        feedback_accepted = probe_matched if probing else min(accepted, len(delivered))
        refuted = bool(dropped) and delivered[0][0] != dropped[0]
        if refuted:
            lane.lookback.observe(len(dropped), 0)
            lane.proposer.feedback(len(dropped), 0)
            lane.acceptance_window.append(0.0)
            self.scheduler_stats["pld_pipeline_refuted_proposals"] += 1
        elif not dropped:
            lane.lookback.observe(feedback_proposed, feedback_accepted)
            lane.proposer.feedback(feedback_proposed, feedback_accepted)
        lane.stats.lookback_current = lane.lookback.current
        lane.stats.lookback_peak = max(lane.stats.lookback_peak, lane.lookback.current)
        lane.stats.lookback_widen_events = lane.lookback.widen_events
        lane.stats.lookback_narrow_events = lane.lookback.narrow_events
        if proposal:
            lane.stats.retrieval_cycles += 1
            lane.stats.retrieval_proposed += len(proposal)
            round_accepted = sum(item[2] for item in delivered)
            lane.stats.retrieval_accepted += round_accepted
            source_stats = lane.source_stats.setdefault(
                proposal_source or "unknown",
                {"cycles": 0, "proposed": 0, "accepted": 0},
            )
            source_stats["cycles"] += 1
            source_stats["proposed"] += len(proposal)
            source_stats["accepted"] += round_accepted
            lane.stats.bonus_tokens += sum(not item[2] for item in delivered)
            lane.acceptance_window.append(
                sum(item[2] for item in delivered) / max(len(proposal), 1)
            )
        else:
            lane.stats.plain_cycles += 1
            lane.stats.plain_tokens += len(delivered)
        if proposal or refuted:
            window = max(1, int(lane.config.get("admission_window", 8)))
            while len(lane.acceptance_window) > window:
                lane.acceptance_window.popleft()
        for token, _row, _from_draft in delivered:
            lane.lookup_history.append(token)
            lane.proposer.observe(token)
            if cost is not None and not proposal:
                cost.observe_plain_token(token)
        if cost is not None and round_started_ns is not None:
            duration_ns = time.perf_counter_ns() - round_started_ns
            cost.observe_round(duration_ns, len(delivered), len(proposal), lane.generated)
            lane.ordinary = cost.state == "parked"
            lane.stats.latched = lane.ordinary
        if probing and delivered:
            lane.admission_matches += probe_matched
            lane.admission_tokens += 1
            lane.stats.admission_matches += probe_matched
            lane.stats.admission_probe_tokens += 1
            window = int(lane.config.get("admission_window", 8))
            if lane.admission_tokens >= window:
                gate = float(lane.config.get("admission_gate", 0.5))
                fraction = lane.admission_matches / max(lane.admission_tokens, 1)
                lane.stats.admission_windows += 1
                if fraction >= gate:
                    lane.admission_consecutive += 1
                else:
                    lane.admission_consecutive = 0
                required = int(lane.config.get("admission_confirm_windows", 1))
                if lane.admission_consecutive >= required:
                    lane.ordinary = False
                    lane.stats.latched = False
                    lane.stats.admission_activations += 1
                    lane.acceptance_window.clear()
                    if lane.stats.admission_delatches:
                        lane.stats.admission_reentries += 1
                else:
                    lane.reprobe_at = lane.generated + int(
                        lane.config.get("admission_reprobe_interval", 16)
                    )
                lane.admission_matches = lane.admission_tokens = 0
        warmup = lane.config.get("adaptive_warmup", 48)
        gate = lane.config.get("adaptive_gate", 0.12)
        if (
            cost is None
            and lane.config.get("adaptive", True)
            and not lane.ordinary
            and lane.generated >= warmup
            and len(lane.acceptance_window)
            >= max(1, int(lane.config.get("admission_window", 8)))
            and sum(lane.acceptance_window) / len(lane.acceptance_window) < gate
        ):
            lane.ordinary = True
            lane.stats.latched = True
            lane.stats.admission_delatches += 1
            lane.reprobe_at = lane.generated + int(
                lane.config.get("admission_reprobe_interval", 16)
            )
            self.scheduler_stats["pld_fallbacks"] += 1

        self.scheduler_stats["pld_cycles"] += 1
        self.scheduler_stats["pld_retrieval_cycles"] += int(bool(proposal))
        self.scheduler_stats["pld_plain_cycles"] += int(not proposal)
        self.scheduler_stats["pld_proposed"] += len(proposal)
        self.scheduler_stats["pld_accepted"] += sum(item[2] for item in delivered)
        self.scheduler_stats["pld_bonus"] += sum(not item[2] for item in delivered)
        # Qualification reads ``target_width`` as the width the lane ran at,
        # as the external route reports it, not this round's width.
        lane.target_max_width = max(lane.target_max_width, getattr(lane, "round_width", 1))
        # ``ordinary_target`` only for rows that ran the ordinary decode step:
        # a proposal-free lane that shared a verify forward did not.
        verify_path = bool(proposal) or lane.round_target_path == "verify_rows"
        lane.verify_path_rounds += int(verify_path)
        receipt = {
            "schema": "mlx2.prompt-lookup-live.v1",
            "execution": (
                "prompt_lookup_verify"
                if lane.stats.retrieval_cycles or lane.verify_path_rounds
                else "ordinary_target"
            ),
            "current_execution": "prompt_lookup_verify" if verify_path else "ordinary_target",
            "current_target_path": "verify_rows" if verify_path else "plain_decode",
            "verify_path_rounds": lane.verify_path_rounds,
            # Anchor-only rounds dispatched before the host read their anchor.
            "pipelined_plain_rounds": lane.pipelined_rounds,
            "ordinary_fallback": lane.ordinary,
            "cycles": lane.stats.cycles,
            "retrieval_cycles": lane.stats.retrieval_cycles,
            "proposed": lane.stats.retrieval_proposed,
            "accepted": lane.stats.retrieval_accepted,
            "round_proposed": len(proposal),
            "round_accepted": sum(item[2] for item in delivered),
            "lookback": lane.lookback.current,
            "lookup_calls": lane.proposer.lookup_calls,
            "source_sites_scanned": lane.proposer.source_sites_scanned,
            "source_index_entries": lane.proposer.index_entries,
            "recent_prompt_segments": lane.proposer.recent_prompt_segments,
            "prompt_segment_tokens": lane.proposer.prompt_segment_tokens,
            "source_window_tokens": (
                min(lane.lookback.current, lane.proposer.index_window)
                if lane.proposer.index_window else lane.lookback.current
            ),
            "context_mismatch_sites": lane.proposer.context_mismatch_sites,
            "admission_windows": lane.stats.admission_windows,
            "admission_probe_tokens": lane.stats.admission_probe_tokens,
            "admission_activations": lane.stats.admission_activations,
            "admission_delatches": lane.stats.admission_delatches,
            "admission_reentries": lane.stats.admission_reentries,
            "verify_span_hist": dict(lane.stats.verify_span_hist),
            "verify_accept_hist": dict(lane.stats.verify_accept_hist),
            "span_snap_cycles": lane.stats.span_snap_cycles,
            "span_extend_cycles": lane.stats.span_extend_cycles,
            "lookup_index": lane.proposer.index_receipt(),
            "target_width": lane.target_max_width,
            "memory_max_draft": lane.config.get("memory_max_draft"),
            "qualification_authority": "serving_route",
        }
        if lane.config.get("recent_committed_segments", False):
            receipt.update(
                {
                    "proposal_sources": {
                        name: dict(values)
                        for name, values in lane.source_stats.items()
                    },
                    "apcv2_source_segments": lane.apcv2_source_segments,
                    "recent_committed_source_segments": lane.recent_source_segments,
                    "recent_committed_segments_enabled": True,
                    "recent_committed_store": self.recent_source_store.status(
                        lane.source_scope
                    ),
                }
            )
        if cost is not None:
            receipt["cost_latch"] = cost.receipt()
            receipt["cost_width1_cohort_splits"] = lane.cost_width1_cohort_splits
        if finish_reason:
            receipt["ordinary_b1_mask_calls"] = _ordinary_b1_mask_calls(lane.cache)
        if finish_reason and lane.speculation_started:
            _stop_speculation(lane.cache)
            lane.speculation_started = False
        committed_response_tokens = tuple(
            lane.lookup_history[-lane.generated :]
        ) if lane.generated else ()
        prior_response_tokens = committed_response_tokens[: -len(delivered)]
        delivered_response_tokens = []
        response_token_limit = int(
            lane.config.get("recent_committed_response_tokens", 2048)
        )
        for index, (token, row, from_draft) in enumerate(delivered):
            delivered_response_tokens.append(token)
            final = index == len(delivered) - 1
            response = GenerationBatch.Response(
                uid=lane.uid,
                token=token,
                logprobs=row,
                finish_reason=finish_reason if final else None,
                prompt_cache=(
                    self._freeze_cache(lane.cache) if final and finish_reason else None
                ),
                all_tokens=list(lane.history) if final and finish_reason else None,
                from_draft=from_draft,
                execution_width=getattr(lane, "round_width", 1),
            )
            response.speculative_receipt = receipt
            if lane.config.get("recent_committed_segments", False):
                # Attach exactly the prefix delivered through this response,
                # not the rest of a multi-token verify round. The serving
                # parser may stop between responses and publish only what the
                # client actually consumed.
                response.pld_committed_response_tokens = tuple(
                    (prior_response_tokens + tuple(delivered_response_tokens))[
                        -response_token_limit:
                    ]
                )
                response.pld_source_scope = lane.source_scope
            lane.ready.append(response)

    def next(self):
        prompts = []
        # At most one bounded prefill slice per poll, round-robin over the
        # lanes still prefilling, and lanes that already held an anchor
        # decode in the same poll, as on the ordinary and external routes.
        # Returning right after the prefills starved every decoding lane
        # until each waiting prompt had finished.  A lane whose prefill ends
        # in this poll starts decoding in the next one, as before.
        pending = [
            lane
            for lane in self.lanes.values()
            if lane.anchor is not None and not lane.ready
        ]
        waiting = [lane for lane in self.lanes.values() if lane.anchor is None]
        if waiting:
            lane = waiting[self._prefill_cursor % len(waiting)]
            self._prefill_cursor += 1
            contended = self.decode_time_fairness.enabled and any(
                other.anchor is not None for other in self.lanes.values()
            )
            prompts.append(self._prefill(lane, contended=contended))
        decode_started = time.perf_counter()
        # The cost latch measures one physical target row. Keep that geometry
        # when several PLD requests coexist: a width-two segmented target is
        # numerically distinct from the ordinary B1 path and has no matched
        # cost baseline. Other PLD lanes retain their batched verify route.
        together = [
            lane for lane in pending
            if getattr(lane, "batched", False) and lane.cost_latch is None
        ]
        if together or self._inflight is not None:
            # A lone batchable lane takes the same transactional path: it is
            # unarmed, so the snapshot/rewind driver is not its rollback.
            try:
                remaining = together
                if self._inflight is not None:
                    # A pipelined plain round is in flight: read it (having
                    # dispatched the next) or drain it into this poll's round.
                    remaining = self._advance_inflight(together)
                if remaining:
                    self._round_batched(remaining)
            except PromptLookupLaneFailure as error:
                # Healthy peers committed their rows; the failed lanes leave.
                for failure in error.failures:
                    self._lane_failures.append(
                        {"uid": failure.uid, "reason": failure.reason}
                    )
                    self.remove([failure.uid])
        for lane in pending:
            if lane not in together and lane.uid in self.lanes:
                try:
                    if lane.cost_latch is not None:
                        if len(pending) > 1:
                            lane.cost_width1_cohort_splits += 1
                            self.scheduler_stats["pld_cost_width1_cohort_splits"] += 1
                        if getattr(lane, "batched", False) and (
                            not lane.ordinary or self._hybrid(lane)
                        ):
                            # Keep the exact segmented transaction for a
                            # speculative verify, but with one target row.
                            # An unarmed hybrid lane has no snapshot epoch for
                            # the direct driver; its one-lane transaction is
                            # the ordinary B1 forward on its own caches.
                            self._round_batched([lane])
                        else:
                            if lane.ordinary:
                                self.scheduler_stats["pld_parked_direct_rounds"] += 1
                            self._round(lane)
                    else:
                        self._round(lane)
                except PromptLookupLaneFailure as error:
                    for failure in error.failures:
                        self._lane_failures.append(
                            {"uid": failure.uid, "reason": failure.reason}
                        )
                        self.remove([failure.uid])
        if pending and self.decode_time_fairness.enabled:
            # Decode rounds set the cost model's fixed per-forward cost.
            self.decode_time_fairness.observe_decode(
                time.perf_counter() - decode_started
            )
            self._sync_decode_fairness_stats()
        # Every token a round verified goes out in this poll, as on the
        # external route.  Handing out one per lane per poll kept a lane that
        # accepted k tokens out of the next k-1 rounds while it drained, so
        # the cohort fragmented into narrow forwards and B-lane PLD could not
        # exceed one token per lane per forward.
        responses = []
        for uid, lane in list(self.lanes.items()):
            while lane.ready:
                response = lane.ready.popleft()
                responses.append(response)
                if response.finish_reason:
                    # The cohort owner holds this lane's planes: let go.
                    self._cohort_owner = None
                    self._close_lane(lane)
                    del self.lanes[uid]
                    break
        if responses:
            previous_steps = self._steps_counter
            self._steps_counter += 1
            if _crossed_counter_interval(
                previous_steps, self._steps_counter, ALLOCATOR_RECLAIM_STEP_INTERVAL
            ):
                mx.clear_cache()
        return prompts, responses

    def scheduler_waiting_uids(self):
        """Lanes still prefilling: they wait on the prefill round-robin, not
        on memory, so the serving stall watchdog must not fail them."""
        return [uid for uid, lane in self.lanes.items() if lane.anchor is None]

    def take_lane_failures(self):
        failures, self._lane_failures = self._lane_failures, []
        return failures

    def pop_prompt_boundary(self, uid):
        return self.boundaries.pop(int(uid), None)

    def remove(self, uids):
        for uid in uids:
            lane = self.lanes.pop(int(uid), None)
            if lane is not None:
                self._leave_inflight(lane)
                self._cohort_owner = None
                self._close_lane(lane)
            self.boundaries.pop(int(uid), None)

    def _leave_inflight(self, lane):
        """Take a departing lane out of the dispatched plain round, if any."""
        inflight = self._inflight
        if inflight is None:
            return
        for row, member in enumerate(inflight.lanes):
            if member is lane and row not in inflight.dropped:
                try:
                    self._release_inflight_row(inflight, row)
                except Exception:  # noqa: BLE001, S110 - its cache leaves with it
                    pass
        if len(inflight.dropped) == len(inflight.lanes):
            self._inflight = None
            with suppress(Exception):
                inflight.transaction.abort()

    @staticmethod
    def _close_lane(lane):
        if lane.speculation_started:
            _stop_speculation(lane.cache)
            lane.speculation_started = False
        close = getattr(lane.cache, "close", None)
        if callable(close):
            close()

    def close(self):
        self.remove(tuple(self.lanes))
        self._cohort_owner = None
        self.boundaries.clear()
