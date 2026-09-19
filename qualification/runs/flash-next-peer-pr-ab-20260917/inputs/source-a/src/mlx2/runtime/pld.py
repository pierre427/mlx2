# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified revision 13eb83388750435bcc751d0b9e33857a38152544.
# See provenance/prompt-lookup-live.json and provenance/LICENSE.unified-MIT.
"""Live, fail-closed prompt-lookup decoding for the shared serving lifecycle."""

from __future__ import annotations

import copy
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import mlx.core as mx

from .generate import GenerationBatch, StopSequenceMatcher, generation_stream
from .models import cache as cache_module
from .models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache
from .prompt_lookup import AdaptiveLookback, HybridStats, IndexedPromptLookup


def _walk_state(caches):
    return [entry.state for entry in caches]


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


def _snapshot(caches):
    def one(entry):
        if isinstance(entry, CacheList):
            return ("list", [one(child) for child in entry.caches])
        if isinstance(entry, ArraysCache):
            if not entry.is_trimmable():
                raise RuntimeError("ArraysCache rollback epoch is not active")
            return ("array", entry.rollback_marker())
        if isinstance(entry, RotatingKVCache):
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


class PromptLookupBatchGenerator:
    """Prompt-lookup executor with the same seam as ``BatchGenerator``.

    Each lane owns an exact rollback epoch. Target verification is currently
    B1 per lane, which preserves model/cache compatibility while still
    amortizing multiple predicted positions into one target forward.
    """

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
        self.capacity = int(completion_batch_size)
        self.prefill_step = int(prefill_step_size)
        self.config = dict(prompt_lookup or {})
        self.num_draft = int(self.config.get("num_draft", 8))
        if self.num_draft < 1:
            raise ValueError("prompt-lookup num_draft must be positive")
        self.default_matcher = StopSequenceMatcher(stop_tokens or None)
        self.lanes = {}
        self.next_uid = 0
        self.boundaries = {}
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
        **_kwargs,
    ):
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
            config = {**self.config, **dict(overrides)}
            proposer = IndexedPromptLookup(
                prefix + prompt,
                ngram_min=int(config.get("ngram_min", 3)),
                ngram_max=int(config.get("ngram_max", 6)),
                hot_segments=int(config.get("hot_segments", 4)),
                reject_ttl=int(config.get("reject_ttl", 8)),
            )
            for segment in config.get("retrieval_segments", ()):
                proposer.add_hot_segment(segment)
            lookback = AdaptiveLookback(
                config.get("lookback_ladder", (256, 1024, 4096, 16384)),
                misses=int(config.get("lookback_misses", 4)),
                rejects=int(config.get("lookback_rejects", 2)),
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
            lane.history.extend(inputs)
            self.scheduler_stats["pld_prefill_rounds"] += 1
        done = len(lane.remaining) == 1
        if done:
            lane.anchor = lane.remaining.popleft()
            self.boundaries[lane.uid] = {
                "committed_only": True,
                "tokens": list(lane.history),
                "target_cache": copy.deepcopy(lane.cache),
                "covered_tokens": len(lane.history),
            }
            _start_speculation(lane.cache, max(64, self.num_draft + 2))
            lane.speculation_started = True
        return SimpleNamespace(
            uid=lane.uid,
            prompt_progress=(len(lane.history), len(lane.lookup_history)),
            end_of_prompt=done,
            end_of_segment=done,
        )

    def _processed_row(self, lane, logits, tentative):
        value = logits[None]
        if lane.processors:
            tokens = mx.array(lane.lookup_history + tentative, dtype=mx.uint32)
            for processor in lane.processors:
                value = processor(tokens, value)
        row = value[0] - mx.logsumexp(value[0])
        mx.eval(row)
        return row

    def _round(self, lane):
        remaining_budget = lane.maximum - lane.generated
        proposal_budget = max(0, min(self.num_draft, remaining_budget - 1))
        proposal = [] if lane.ordinary else lane.proposer.propose(
            proposal_budget, lookback=lane.lookback.current
        )
        inputs = [lane.anchor] + proposal
        snapshots = _snapshot(lane.cache)
        with mx.stream(generation_stream):
            logits = self.model(
                mx.array([inputs], dtype=mx.uint32), cache=lane.cache
            )[0]
            mx.eval(logits)
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
        if consumed < len(inputs):
            self.scheduler_stats["pld_rollbacks"] += 1
            _rewind(lane.cache, snapshots)
            committed_inputs = inputs[:consumed]
            if committed_inputs:
                with mx.stream(generation_stream):
                    self.model(
                        mx.array([committed_inputs], dtype=mx.uint32),
                        cache=lane.cache,
                    )
                    mx.eval(_walk_state(lane.cache))
        lane.history.extend(inputs[:consumed])
        lane.matcher_state = matcher_state
        lane.anchor = delivered[-1][0]
        lane.generated += len(delivered)
        lane.stats.cycles += 1
        lane.stats.verify_span_hist[len(inputs)] = lane.stats.verify_span_hist.get(len(inputs), 0) + 1
        lane.lookback.observe(len(proposal), min(accepted, len(delivered)))
        lane.proposer.feedback(len(proposal), min(accepted, len(delivered)))
        lane.stats.lookback_current = lane.lookback.current
        lane.stats.lookback_peak = max(lane.stats.lookback_peak, lane.lookback.current)
        lane.stats.lookback_widen_events = lane.lookback.widen_events
        lane.stats.lookback_narrow_events = lane.lookback.narrow_events
        if proposal:
            lane.stats.retrieval_cycles += 1
            lane.stats.retrieval_proposed += len(proposal)
            lane.stats.retrieval_accepted += sum(item[2] for item in delivered)
            lane.stats.bonus_tokens += sum(not item[2] for item in delivered)
        else:
            lane.stats.plain_cycles += 1
            lane.stats.plain_tokens += len(delivered)
        for token, _row, _from_draft in delivered:
            lane.lookup_history.append(token)
            lane.proposer.observe(token)
        warmup = int(lane.config.get("adaptive_warmup", 48))
        gate = float(lane.config.get("adaptive_gate", 0.12))
        if (
            lane.config.get("adaptive", True)
            and not lane.ordinary
            and lane.generated >= warmup
            and lane.stats.retrieval_accepted
            / max(lane.stats.retrieval_proposed, 1)
            < gate
        ):
            lane.ordinary = True
            lane.stats.latched = True
            self.scheduler_stats["pld_fallbacks"] += 1

        self.scheduler_stats["pld_cycles"] += 1
        self.scheduler_stats["pld_retrieval_cycles"] += int(bool(proposal))
        self.scheduler_stats["pld_plain_cycles"] += int(not proposal)
        self.scheduler_stats["pld_proposed"] += len(proposal)
        self.scheduler_stats["pld_accepted"] += sum(item[2] for item in delivered)
        self.scheduler_stats["pld_bonus"] += sum(not item[2] for item in delivered)
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
            "target_width": 1,
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
                prompt_cache=copy.deepcopy(lane.cache) if final and finish_reason else None,
                all_tokens=list(lane.history) if final and finish_reason else None,
                from_draft=from_draft,
                execution_width=1,
            )
            response.speculative_receipt = receipt
            lane.ready.append(response)

    def next(self):
        prompts = []
        for lane in list(self.lanes.values()):
            if lane.anchor is None:
                prompts.append(self._prefill(lane))
        if prompts:
            return prompts, []
        for lane in list(self.lanes.values()):
            if not lane.ready:
                self._round(lane)
        responses = []
        for uid, lane in list(self.lanes.items()):
            if lane.ready:
                response = lane.ready.popleft()
                responses.append(response)
                if response.finish_reason:
                    self._close_lane(lane)
                    del self.lanes[uid]
        return [], responses

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
