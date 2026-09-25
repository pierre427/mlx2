# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified revision 13eb83388750435bcc751d0b9e33857a38152544.
# See provenance/prompt-lookup-live.json and provenance/LICENSE.unified-MIT.
"""Live, fail-closed prompt-lookup decoding for the shared serving lifecycle."""

from __future__ import annotations

import math
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import mlx.core as mx

from .committed_recovery import CommittedRecoverySlot
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
from .prompt_lookup import (
    AdaptiveLookback,
    HybridStats,
    IndexedPromptLookup,
    plan_proposal_around_verify_cliff,
)
from .rotating_replay import (
    RotatingReplayError,
    RotatingReplayPolicy,
    RotatingReplayTransaction,
)
from ..thinking_guard import stack_block_steer, verify_block_steer


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
    acceptance_window: deque = field(default_factory=deque)
    rotating: list = field(default_factory=list)
    recovery: CommittedRecoverySlot = field(default_factory=CommittedRecoverySlot)
    # Widest committed verify forward; the receipt's ``target_width``.
    target_max_width: int = 1


class PromptLookupBatchGenerator:
    """Prompt-lookup executor with the same seam as ``BatchGenerator``.

    Each lane owns an exact rollback epoch. Target verification is currently
    B1 per lane, which preserves model/cache compatibility while still
    amortizing multiple predicted positions into one target forward.
    """

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
            "cliff_aware_span",
            "verify_cliff_start",
            "verify_cliff_end",
            "rotating_replay",
            "batched_verify",
            "max_proposal_tokens",
            "memory_max_draft",
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
            "lookback_misses": 1,
            "lookback_rejects": 1,
            "adaptive_warmup": 0,
            "admission_window": 1,
            "admission_confirm_windows": 1,
            "admission_reprobe_interval": 0,
            "verify_cliff_start": 2,
            "verify_cliff_end": 1,
            "max_proposal_tokens": 1,
            "memory_max_draft": 0,
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
        for name in (
            "adaptive",
            "deferred_admission",
            "cliff_aware_span",
            "rotating_replay",
            "batched_verify",
        ):
            if name in validated and type(validated[name]) is not bool:
                raise ValueError(f"prompt_lookup {name} must be a boolean")
        for name in ("adaptive_gate", "admission_gate"):
            if name not in validated:
                continue
            gate = validated[name]
            if type(gate) not in (int, float):
                raise ValueError(f"prompt_lookup {name} must be a finite probability")
            gate = float(gate)
            if not math.isfinite(gate) or not 0.0 <= gate <= 1.0:
                raise ValueError(f"prompt_lookup {name} must be a finite probability")
            validated[name] = gate
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
        stop_tokens=(),
        prompt_lookup=None,
        **_kwargs,
    ):
        self.model = model
        self._recovery_revision = (
            f"{type(model).__module__}.{type(model).__qualname__}:{id(model)}"
        )
        self.capacity = int(completion_batch_size)
        self.prefill_step = int(prefill_step_size)
        self.config = self.validate_policy(prompt_lookup or {})
        self.num_draft = self.config.get("num_draft", 8)
        # Default-off and generator-scoped: a per-request override never selects
        # the rotating replay transaction.  The bound counts proposed tokens;
        # the transaction also declares the anchor, hence the extra row.
        self.rotating_replay = bool(self.config.get("rotating_replay", False))
        # On unless disabled: verification stays target-exact either way, and
        # it only engages for lanes whose planes are all plain/rotating KV.
        self.batched_verify = bool(self.config.get("batched_verify", True))
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
            "pld_batched_rounds": 0,
            "pld_batched_lanes": 0,
            "pld_batched_max_width": 0,
            "pld_recovery_checkpoint_captures": 0,
            "pld_recovery_checkpoint_restores": 0,
            "pld_recovery_checkpoint_failures": 0,
            "pld_recovery_full_rebuilds": 0,
            "target_max_width": 1,
        }

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
    ):
        # The serving seam passes these to every route.  ``lane_rngs`` is
        # unused here because each sampler already carries its lane's RNG;
        # the others would be dropped, so a non-neutral value is refused
        # rather than reported as applied.
        del lane_rngs
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
        count = len(prompts)
        max_tokens = max_tokens or [128] * count
        caches = caches or [None] * count
        all_tokens = all_tokens or [[] for _ in prompts]
        samplers = samplers or [None] * count
        logits_processors = logits_processors or [[] for _ in prompts]
        stop_matchers = stop_matchers or [self.default_matcher] * count
        prompt_lookup_configs = prompt_lookup_configs or [{} for _ in prompts]
        uids = []
        for prompt, maximum, prompt_cache, prefix, sampler, processors, matcher, overrides in zip(
            prompts,
            max_tokens,
            caches,
            all_tokens,
            samplers,
            logits_processors,
            stop_matchers,
            prompt_lookup_configs,
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
            proposer = IndexedPromptLookup(
                prefix + prompt,
                ngram_min=config.get("ngram_min", 3),
                ngram_max=config.get("ngram_max", 6),
                hot_segments=config.get("hot_segments", 4),
                reject_ttl=config.get("reject_ttl", 8),
            )
            for segment in config.get("retrieval_segments", ()):
                proposer.add_hot_segment(segment)
            lookback = AdaptiveLookback(
                config.get("lookback_ladder", (256, 1024, 4096, 16384)),
                misses=config.get("lookback_misses", 4),
                rejects=config.get("lookback_rejects", 2),
            )
            uid = self.next_uid
            self.next_uid += 1
            lane = _Lane(
                uid=uid,
                remaining=deque(prompt),
                cache=prompt_cache,
                history=prefix,
                lookup_history=prefix + prompt,
                proposer=proposer,
                lookback=lookback,
                sampler=sampler or (lambda value: mx.argmax(value, axis=-1)),
                processors=list(processors or ()),
                stop_matcher=matcher,
                maximum=int(maximum),
                config=config,
                ordinary=bool(config.get("deferred_admission", False)),
            )
            lane.matcher_state = matcher.make_state()
            self.lanes[uid] = lane
            uids.append(uid)
        return uids

    def _prefill(self, lane):
        if len(lane.remaining) > 1:
            count = min(self.prefill_step, len(lane.remaining) - 1)
            inputs = [lane.remaining.popleft() for _ in range(count)]
            with mx.stream(generation_stream):
                self.model(mx.array([inputs], dtype=mx.uint32), cache=lane.cache)
                mx.eval(_walk_state(lane.cache))
            mx.clear_cache()
            lane.history.extend(inputs)
            self.scheduler_stats["pld_prefill_rounds"] += 1
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
        # An explicit rotating-replay policy selects that per-lane transaction.
        lane.batched = (
            self.batched_verify
            and not self.rotating_replay
            and self._batchable(lane.cache)
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
        except Exception:
            lane.recovery.invalidate()
            self.scheduler_stats["pld_recovery_checkpoint_failures"] += 1
        finally:
            if was_armed:
                self._arm_lane_speculation(lane)

    def _rebuild_lane_cache(self, lane, committed_inputs, taps=None, steer=None):
        """Restore the last committed boundary, falling back to full re-prefill.

        ``steer`` covers ``committed_inputs`` position by position, so the
        replay recomputes the steered K/V the verify forward produced.
        """
        self.scheduler_stats["pld_rotating_replay_rebuilds"] = (
            self.scheduler_stats.get("pld_rotating_replay_rebuilds", 0) + 1
        )
        with suppress(Exception):
            self._close_lane(lane)
        lane.cache = lane.recovery.restore(
            route="prompt_lookup",
            revision=self._recovery_revision,
            boundary=len(lane.history),
        )
        if lane.cache is not None:
            self.scheduler_stats["pld_recovery_checkpoint_restores"] += 1
            self._arm_lane_speculation(lane)
            tokens = list(committed_inputs)
        else:
            self.scheduler_stats["pld_recovery_full_rebuilds"] += 1
            lane.cache = cache_module.make_prompt_cache(self.model)
            tokens = list(lane.history) + list(committed_inputs)
        # The committed inputs are the tail of ``tokens``.  A full re-prefill
        # (no recovery checkpoint) replays earlier generated positions
        # unsteered; it is the last-resort path after a failed transaction.
        steer_from = len(tokens) - len(committed_inputs)
        with mx.stream(generation_stream):
            for start in range(0, len(tokens), self.prefill_step):
                chunk = tokens[start : start + self.prefill_step]
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

    def _processed_row(self, lane, logits, tentative):
        value = logits[None]
        if lane.processors:
            tokens = mx.array(lane.lookup_history + tentative, dtype=mx.uint32)
            for processor in lane.processors:
                value = processor(tokens, value)
        row = value[0].astype(mx.float32)
        row = row - mx.logsumexp(row)
        mx.eval(row)
        return row

    def _round(self, lane):
        """One lane, one verify forward: snapshot, verify, rewind and replay."""
        steps = self._round_steps(lane)
        inputs, proposal = next(steps)
        lane.round_width = 1
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
        taps, steer, commits = self._verify_steer([lane], [inputs])
        try:
            if steer is not None:
                taps.steer = steer
            try:
                with mx.stream(generation_stream):
                    logits = self.model(
                        mx.array([inputs], dtype=mx.uint32), cache=lane.cache
                    )[0]
                    mx.eval(logits)
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
        consumed = steps.send(logits)
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
        """Plain and rotating single-row planes can share a segmented transaction."""
        return (
            isinstance(cache, list)
            and bool(cache)
            and all(type(entry) in (KVCache, RotatingKVCache) for entry in cache)
        )

    def _round_batched(self, lanes):
        """Verify several lanes in one target forward.

        Each lane keeps its own request-private B1 caches; the segmented
        transaction appends every lane's verify block (ragged lengths), and the
        commit republishes exactly the consumed prefix per lane from the K/V it
        already computed, so a rejected tail costs no replay forward.
        """
        from .segmented_rotating_kv import SegmentedKVRows

        steps = [self._round_steps(lane) for lane in lanes]
        plans = [next(step) for step in steps]
        lengths = [len(inputs) for inputs, _proposal in plans]
        width = max(lengths)
        for lane in lanes:
            lane.round_width = len(lanes)
        transaction = SegmentedKVRows([lane.cache for lane in lanes]).begin(lengths=lengths)
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
                    mx.eval(logits)
            finally:
                if steer is not None:
                    taps.steer = None
            consumed = [step.send(logits[row]) for row, step in enumerate(steps)]
            transaction.commit(accepted_lengths=consumed)
            transaction = None
            for commit, count in zip(commits, consumed):
                commit(count)
        finally:
            if transaction is not None:
                with suppress(Exception):
                    transaction.abort()
        self.scheduler_stats["pld_batched_rounds"] += 1
        self.scheduler_stats["pld_batched_lanes"] += len(lanes)
        self.scheduler_stats["pld_batched_max_width"] = max(
            self.scheduler_stats["pld_batched_max_width"], len(lanes)
        )
        self.scheduler_stats["pld_rollbacks"] += sum(
            int(done < length) for done, length in zip(consumed, lengths)
        )
        for step in steps:
            with suppress(StopIteration):
                next(step)

    def _round_steps(self, lane):
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
        probing = deferred and lane.ordinary and lane.generated >= lane.reprobe_at
        probe_candidate = (
            lane.proposer.propose(1, lookback=lane.lookback.current)
            if probing
            else []
        )
        proposal = [] if lane.ordinary else lane.proposer.propose(
            proposal_budget, lookback=lane.lookback.current
        )
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
        logits = yield inputs, proposal
        emitted = []
        accepted = 0
        for index in range(len(proposal) + 1):
            row = self._processed_row(lane, logits[index], proposal[:index])
            token = int(lane.sampler(row[None])[0].item())
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
        # Publish exactly ``consumed`` verified inputs to the lane's cache.
        yield consumed
        lane.history.extend(inputs[:consumed])
        lane.matcher_state = matcher_state
        lane.anchor = delivered[-1][0]
        lane.generated += len(delivered)
        self._capture_lane_recovery(lane)
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
        lane.lookback.observe(feedback_proposed, feedback_accepted)
        lane.proposer.feedback(feedback_proposed, feedback_accepted)
        lane.stats.lookback_current = lane.lookback.current
        lane.stats.lookback_peak = max(lane.stats.lookback_peak, lane.lookback.current)
        lane.stats.lookback_widen_events = lane.lookback.widen_events
        lane.stats.lookback_narrow_events = lane.lookback.narrow_events
        if proposal:
            lane.stats.retrieval_cycles += 1
            lane.stats.retrieval_proposed += len(proposal)
            lane.stats.retrieval_accepted += sum(item[2] for item in delivered)
            lane.stats.bonus_tokens += sum(not item[2] for item in delivered)
            lane.acceptance_window.append(
                sum(item[2] for item in delivered) / max(len(proposal), 1)
            )
            window = max(1, int(lane.config.get("admission_window", 8)))
            while len(lane.acceptance_window) > window:
                lane.acceptance_window.popleft()
        else:
            lane.stats.plain_cycles += 1
            lane.stats.plain_tokens += len(delivered)
        for token, _row, _from_draft in delivered:
            lane.lookup_history.append(token)
            lane.proposer.observe(token)
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
            lane.config.get("adaptive", True)
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
        receipt = {
            "schema": "mlx2.prompt-lookup-live.v1",
            "execution": "prompt_lookup_verify" if lane.stats.retrieval_cycles else "ordinary_target",
            "current_execution": "ordinary_target" if not proposal else "prompt_lookup_verify",
            "ordinary_fallback": lane.ordinary,
            "cycles": lane.stats.cycles,
            "retrieval_cycles": lane.stats.retrieval_cycles,
            "proposed": lane.stats.retrieval_proposed,
            "accepted": lane.stats.retrieval_accepted,
            "round_proposed": len(proposal),
            "round_accepted": sum(item[2] for item in delivered),
            "lookback": lane.lookback.current,
            "admission_windows": lane.stats.admission_windows,
            "admission_probe_tokens": lane.stats.admission_probe_tokens,
            "admission_activations": lane.stats.admission_activations,
            "admission_delatches": lane.stats.admission_delatches,
            "admission_reentries": lane.stats.admission_reentries,
            "verify_span_hist": dict(lane.stats.verify_span_hist),
            "verify_accept_hist": dict(lane.stats.verify_accept_hist),
            "span_snap_cycles": lane.stats.span_snap_cycles,
            "span_extend_cycles": lane.stats.span_extend_cycles,
            "target_width": lane.target_max_width,
            "memory_max_draft": lane.config.get("memory_max_draft"),
            "qualification_authority": "serving_route",
        }
        if finish_reason and lane.speculation_started:
            _stop_speculation(lane.cache)
            lane.speculation_started = False
        for index, (token, row, from_draft) in enumerate(delivered):
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
            prompts.append(self._prefill(lane))
        together = [lane for lane in pending if getattr(lane, "batched", False)]
        if together:
            # A lone batchable lane takes the same transactional path: it is
            # unarmed, so the snapshot/rewind driver is not its rollback.
            self._round_batched(together)
        for lane in pending:
            if lane not in together:
                self._round(lane)
        responses = []
        for uid, lane in list(self.lanes.items()):
            if lane.ready:
                response = lane.ready.popleft()
                responses.append(response)
                if response.finish_reason:
                    self._close_lane(lane)
                    del self.lanes[uid]
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

    def pop_prompt_boundary(self, uid):
        return self.boundaries.pop(int(uid), None)

    def remove(self, uids):
        for uid in uids:
            lane = self.lanes.pop(int(uid), None)
            if lane is not None:
                self._close_lane(lane)
            self.boundaries.pop(int(uid), None)

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
        self.boundaries.clear()
