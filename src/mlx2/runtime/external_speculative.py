"""External draft/verify execution composed with mlx2 admission and APCv2.

Original implementation. Target verification uses per-lane segmented cache
transactions; proposal q is supplied by the bound external draft model.
No capability is qualified by importing or constructing this executor.
"""
from __future__ import annotations

import copy
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from .committed_recovery import CommittedRecoverySlot
from .processor_probe import VerifyWindow, copy_sharing, rollback_shared_memo
from .cow_cache import (
    COWCacheUnsupported,
    restore_recovery_descriptors,
    snapshot_committed_cache,
    snapshot_prompt_cache_descriptors,
    snapshot_recovery_descriptors,
)
from .generate import ALLOCATOR_RECLAIM_MTP_TOKEN_INTERVAL, _crossed_counter_interval
from .speculative_sampling import (
    FLyVerificationPolicy,
    RequestRNG,
    probability,
    verify_compact_proposals,
    verify_proposals,
)
from ..thinking_guard import stack_block_steer, verify_block_steer

_COUNTER_MAX = (1 << 63) - 1


def _bump(stats, key, amount=1):
    stats[key] = min(_COUNTER_MAX, int(stats.get(key, 0)) + int(amount))


# Default-off tree15 round-cost experiment gates (TensorFold parity bridge).
# Each maps to its environment spelling and the accepted value that enables it.
_TREE_GATES = {
    "cache_executor": ("MLX2_TENSORFOLD_CACHE_EXECUTOR", "1"),
    "codebook_cache": ("MLX2_DFLASH_TREE_CODEBOOK_CACHE", "1"),
    "batched_laws": ("MLX2_TREE_BATCHED_TARGET_LAWS", "1"),
    "logprobs_on_request": ("MLX2_TREE_LOGPROBS_ON_REQUEST", "1"),
    "single_fence": ("MLX2_TREE_SINGLE_FENCE", "1"),
    "pipeline_draft": ("MLX2_TREE_PIPELINE_DRAFT", "1"),
}
# Bounded integer engagement counters each enabled gate publishes.
_TREE_GATE_COUNTERS = {
    "cache_executor": (
        "external_tensorfold_executor_validations",
        "external_tensorfold_executor_cache_hits",
    ),
    "codebook_cache": ("external_tree_codebook_cache_hits",),
    "batched_laws": (
        "external_tree_batched_law_rounds",
        "external_tree_row_law_rounds",
        "external_tree_sparse_law_rows",
    ),
    "logprobs_on_request": ("external_tree_logprob_rows_skipped",),
    "single_fence": ("external_tree_single_fence_rounds",),
    "pipeline_draft": (
        "external_tree_pipelined_drafts",
        "external_tree_pipeline_discards",
    ),
}


def _tree_gates(topology, target_execution):
    """Read the tree experiment gates; fail closed on a gate the route lacks."""

    gates = {}
    for gate, (name, enabled) in _TREE_GATES.items():
        value = os.environ.get(name, "")
        if value not in ("", "0", enabled):
            raise ValueError(f"{name} must be unset, 0 or {enabled}")
        gates[gate] = value == enabled
    if any(gates.values()) and topology != "tree15":
        raise ValueError("tree experiment gates require MLX2_DFLASH_TOPOLOGY=tree15")
    if gates["cache_executor"] and target_execution != "tensorfold":
        raise ValueError(
            "MLX2_TENSORFOLD_CACHE_EXECUTOR requires MLX2_QWEN_TARGET_EXECUTION=tensorfold"
        )
    if gates["single_fence"] and not gates["batched_laws"]:
        raise ValueError(
            "MLX2_TREE_SINGLE_FENCE requires MLX2_TREE_BATCHED_TARGET_LAWS=1"
        )
    return gates


def _tensorfold_cohort_limit(target_execution, capacity):
    """Return the explicit bounded TensorFold cohort limit (singleton default)."""

    from .qwen38_tensorfold import MAX_COHORT_LANES

    name = "MLX2_TENSORFOLD_COHORT_LIMIT"
    raw = os.environ.get(name)
    if raw is None:
        return 1
    if target_execution != "tensorfold":
        raise ValueError(f"{name} requires MLX2_QWEN_TARGET_EXECUTION=tensorfold")
    allowed = {str(value) for value in range(1, MAX_COHORT_LANES + 1)}
    if raw not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(sorted(allowed))}")
    return min(int(capacity), int(raw))


# Laws with at most this many top-k survivors are read back sparsely.
_SPARSE_LAW_LIMIT = 256


class _PhaseClock:
    """Accumulate host time between existing sync points (round timing only)."""

    __slots__ = ("owner", "last")

    def __init__(self, owner):
        self.owner = owner
        self.last = time.perf_counter()

    def __call__(self, phase):
        self.last = self.owner._mark(phase, self.last)

    def skip(self):
        self.last = time.perf_counter()


def _context_pairing(history, following, length):
    """Return only the needed history suffix plus its following token."""
    if not length:
        return []
    return history[-(length - 1):] + [int(following)] if length > 1 else [int(following)]


class DraftUnavailable(RuntimeError):
    """A recoverable drafter-only failure; target ordinary path remains valid."""


class LaneFailure(RuntimeError):
    """One lane cannot continue, e.g. its target law is not a distribution.

    Only that request fails: the round restores every lane of its cohort,
    the executor drops the failed lane, and serving collects it through
    ``take_lane_failures``.
    """

    def __init__(self, uid, reason):
        super().__init__(reason)
        self.uid = uid
        self.reason = reason


@dataclass
class ExternalDraftState:
    state: object
    covered_tokens: int
    rng_key: object = None
    rng_draws: int = 0
    kind: str = "external_draft_v1"
    binding: str = ""

    @property
    def nbytes(self):
        return sum(int(getattr(c, "nbytes", 0)) for c in self.state[0]) + int(getattr(self.state[1], "nbytes", 0)) + int(getattr(self.rng_key, "nbytes", 0))

    def validate(self, binding, covered):
        if self.kind != "external_draft_v1" or self.binding != binding or self.covered_tokens != covered:
            raise ValueError("External draft sidecar revision/boundary mismatch")
        length = 0 if self.state[1] is None else int(self.state[1].shape[1])
        if any(int(c.offset) + length != covered for c in self.state[0]):
            raise ValueError("External draft context is not paired to target boundary")


@dataclass
class Lane:
    uid: int
    history: list
    remaining: deque
    cache: list
    draft_cache: list
    tail: object
    rng: RequestRNG
    maximum: int
    processors: list
    sampling: dict
    generated: int = 0
    anchor: int | None = None
    ready: deque = field(default_factory=deque)
    cancelled: bool = False
    ordinary: bool = False
    external_rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    target_max_width: int = 0
    draft_max_width: int = 0
    relaxed_accepts: int = 0
    # Per-round distributions over external verify rounds, ``{value: rounds}``.
    # Same contract as the self-MTP route (see prompt_lookup.HybridStats):
    # ``verify_accept_hist`` truncates to tau(k) for every k below the depth
    # actually run, ``verify_span_hist`` counts verified positions per round.
    verify_span_hist: dict = field(default_factory=dict)
    verify_accept_hist: dict = field(default_factory=dict)


# History changes only by append inside a decode round.  Keep its committed
# list and length as a journal, copying the prefix only if recovery is needed.
_LANE_PLANES = frozenset({"cache", "draft_cache", "tail"})
_LANE_JOURNAL = frozenset({"history"})


@dataclass(frozen=True)
class RoundSnapshot:
    """One lane's committed-boundary checkpoint for a single round."""
    slot: CommittedRecoverySlot
    boundary: int
    mode: str  # "descriptor_cow" or "deepcopy"
    processors: tuple  # authoritative objects retained by serving
    history: list  # committed prefix, append-only while this snapshot is live


@dataclass
class HostDraftRow:
    """One row's host-sampled draft: tokens and their dense proposal laws.

    Exposes the ``lengths``/``dense_laws`` surface of a one-row compact
    ``DraftBlock`` so the verify phase reads both the same way.
    """
    tokens: list
    laws: list
    width: int = 1

    @property
    def lengths(self):
        return (len(self.tokens),)

    def dense_laws(self, vocab):
        return [list(self.laws)]


@dataclass
class CompactDraftRow:
    """One pairwise row with candidate support kept sparse until rejection."""
    tokens: list
    candidate_ids: np.ndarray
    candidate_probs: np.ndarray
    width: int

    @property
    def lengths(self):
        return (len(self.tokens),)


@dataclass
class TreeDraftRow:
    """One best-first proposal tree; parents index ``tokens`` and use -1 as root."""

    tokens: list
    parents: list
    width: int = 1

    @property
    def lengths(self):
        return (len(self.tokens),)


@dataclass
class RoundDecision:
    """Verify outcome for one row, before commit.

    ``emitted`` is already stop-truncated; ``target_laws`` holds the dense
    target law per emitted token (``None`` when the verifier kept no dense
    law); ``relaxed`` counts FLy relaxed accepts.  ``response_logprobs``
    holds, per verify row, the log-probability row to publish instead of the
    law (``None`` entries fall back to the law).  ``verify_window`` holds
    the processor ledgers per verify row, settled at commit.
    """
    accepted: int
    emitted: list
    target_laws: object
    relaxed: int = 0
    response_logprobs: object = None
    verify_window: object = None
    commit_rows: object = None


def _block_row(block, vocab):
    """(draft tokens, dense proposal laws) of a one-row proposal block."""
    if block is None:
        return [], []
    if isinstance(block, HostDraftRow):
        return list(block.tokens), list(block.laws)
    if isinstance(block, CompactDraftRow):
        return list(block.tokens), block
    if isinstance(block, TreeDraftRow):
        return list(block.tokens), None
    length = int(block.lengths[0])
    tokens = [int(t) for t in np.asarray(block.tokens)[0, :length].tolist()]
    return tokens, (None if vocab is None else block.dense_laws(vocab)[0])


class _ReferenceTreeTransaction:
    """Commit a serial-tree accepted path through the ordinary target forward."""

    def __init__(self, owner, cache, inputs):
        self.owner = owner
        self.cache = cache
        self.inputs = list(inputs)
        self.closed = False

    def commit_paths(self, paths):
        if self.closed or len(paths) != 1 or not paths[0]:
            raise RuntimeError("reference tree commit requires one nonempty path")
        tokens = [self.inputs[index] for index in paths[0]]
        logits, features = self.owner.model.forward_with_taps(
            self.owner.mx.array([tokens]), self.cache, self.owner.layers
        )
        self.owner.mx.eval(logits, features)
        self.closed = True
        return [self.cache]

    def abort(self):
        self.closed = True


class ExternalDraftBatchGenerator:
    """Same request-lifecycle seam as BatchGenerator, distinct external protocol.

    The ordinary target trunk is evaluated at cohort width B. TensorFold keeps
    B1 state records but queues up to four lane trees behind one evaluation
    fence. Draft trunk forwards group lanes by pending-context length;
    attention remains per-row SDPA. Receipts report actual target and draft
    widths. No dense cache repacking or shared global random generator.
    """
    def __init__(self, model, *, draft_model, binding, completion_batch_size=4,
                 prefill_step_size=2048, num_draft=4, stop_tokens=(), memory_headroom=None,
                 reclaim_memory=None, evict_checkpoint=None, fly_verification=None,
                 pairwise_selection="host", ready_drain="one",
                 prefill_allocator_reclaim=False, **kwargs):
        import mlx.core as mx
        self.mx = mx; self.model = model; self.draft = draft_model
        self.memory_headroom = memory_headroom
        self.reclaim_memory = reclaim_memory; self.evict_checkpoint = evict_checkpoint
        self._schedule_cursor = 0; self._reclaims_left = 0; self._allocator_reclaimed = False
        # Responses returned by next(); drives the periodic allocator reclaim.
        self._emitted_responses = 0
        self.binding = binding; self.capacity = completion_batch_size
        self.fly_verification = FLyVerificationPolicy.from_value(fly_verification)
        self.prefill_step = prefill_step_size; self.num_draft = int(num_draft)
        if pairwise_selection not in ("host", "batched"):
            raise ValueError("pairwise_selection must be 'host' or 'batched'")
        self.pairwise_selection = pairwise_selection
        self.draft_topology = os.environ.get(
            "MLX2_DFLASH_TOPOLOGY", "chain"
        )
        self.target_execution = os.environ.get(
            "MLX2_QWEN_TARGET_EXECUTION", "reference"
        )
        if self.draft_topology not in ("chain", "tree15"):
            raise ValueError("MLX2_DFLASH_TOPOLOGY must be chain or tree15")
        # The reference tree remains one lane per round.  TensorFold cohorts
        # independent B1 records under its fixed four-lane ownership bound.
        if self.target_execution not in ("reference", "tensorfold"):
            raise ValueError(
                "MLX2_QWEN_TARGET_EXECUTION must be reference or tensorfold"
            )
        if self.target_execution == "tensorfold" and not os.environ.get(
            "MLX2_TENSORFOLD_SOURCE"
        ):
            raise ValueError(
                "TensorFold target execution requires MLX2_TENSORFOLD_SOURCE"
            )
        self.tree_gates = _tree_gates(self.draft_topology, self.target_execution)
        # "one" (default, unchanged): one ready token per lane per poll.  A
        # lane still draining a multi-token round sits out the next cohort,
        # so at B>1 a lane that accepted more waits one round of the others
        # per token it holds.  "all" returns every ready token of a lane in
        # the poll that produced it, as the self-MTP generator does, so
        # lanes stay in lockstep.  Opt-in per adapter (mlx2).
        if ready_drain not in ("one", "all"):
            raise ValueError("ready_drain must be 'one' or 'all'")
        self.ready_drain = ready_drain
        if not 1 <= self.num_draft < draft_model.config.block_size:
            raise ValueError("External draft count must fit trained block")
        if any(len(t) != 1 for t in stop_tokens):
            raise ValueError("External executor currently requires single-token stops")
        self.stops = {int(t[0]) for t in stop_tokens}
        self.layers = tuple(draft_model.config.target_layer_ids)
        self.lanes = {}; self.next_uid = 0; self.boundaries = {}
        self._lane_failures = []
        self.scheduler_stats = {"external_rounds": 0, "accepted_proposals": 0, "proposed_tokens": 0, "ordinary_rounds": 0, "cancelled": 0, "target_max_width": 0, "draft_max_width": 1, "prefill_rounds": 0, "paired_cache_resumes": 0, "segmented_transactions": 0, "segmented_rollbacks": 0, "draft_fallbacks": 0, "recovery_checkpoint_captures": 0, "recovery_checkpoint_restores": 0, "external_draft_masked_positions": 0, "external_ordinary_fast_path_rounds": 0, "external_ordinary_fast_path_lanes": 0, "external_draft_context_skipped": 0, "external_taps_skipped": 0, "external_transactions_skipped": 0, "fly_relaxed_accepts": 0, "external_context_token_pairings": 0, "external_verify_steer_rounds": 0, "external_verify_steered_lanes": 0, "external_allocator_reclaims": 0}
        # Drafters that fuse each target feature with the token that follows
        # it (EAGLE) opt in; DFlash-family drafters keep the original calls.
        self.pair_context_tokens = bool(getattr(draft_model, "requires_context_tokens", False))
        self.receipt_kind = str(getattr(draft_model, "receipt_kind", "external_dflash2"))
        # Default-off, unqualified candidate for direct-model A/Bs only (no
        # serving policy selects it): release the MLX pool after each
        # non-empty prefill chunk, once its target and draft state are
        # materialized, as the ordinary and prompt-lookup prefill already do.
        if type(prefill_allocator_reclaim) is not bool:
            raise ValueError("prefill_allocator_reclaim must be boolean")
        self.prefill_allocator_reclaim = prefill_allocator_reclaim
        if prefill_allocator_reclaim:
            self.scheduler_stats.update(external_prefill_allocator_reclaims=0)
        if pairwise_selection == "batched":
            # Default-off receipts keep their existing key set.
            self.scheduler_stats.update(external_pairwise_selection_groups=0, external_pairwise_selection_lanes=0)
        if self.draft_topology == "tree15":
            self.scheduler_stats.update(
                external_tree_rounds=0,
                external_tree_nodes=0,
                external_tree_accepted_edges=0,
            )
        if self.target_execution == "tensorfold":
            self.tensorfold_cohort_limit = _tensorfold_cohort_limit(
                self.target_execution, self.capacity
            )
            self.scheduler_stats.update(
                external_tensorfold_target_rounds=0,
                external_tensorfold_cohort_rounds=0,
                external_tensorfold_cohort_lanes=0,
                external_tensorfold_cohort_max_width=0,
                external_tensorfold_cohort_limit=self.tensorfold_cohort_limit,
            )
        else:
            self.tensorfold_cohort_limit = _tensorfold_cohort_limit(
                self.target_execution, self.capacity
            )
        for gate, counters in _TREE_GATE_COUNTERS.items():
            if self.tree_gates[gate]:
                self.scheduler_stats.update(dict.fromkeys(counters, 0))
        self._tree_clock = None
        # uid -> next round's lattice queued after this round's commit
        # (MLX2_TREE_PIPELINE_DRAFT). Never part of a lane or its snapshot.
        self._prelaunched = {}
        self._open = False
        # Host-time attribution for one external round.  Default off: a
        # perf_counter in this path is cheap but the block exists only for
        # tuning, and the bounded integer counters above stay always-on.
        self.round_timing = bool(os.environ.get("MLX2_EXTERNAL_ROUND_TIMING"))
        self.round_times = defaultdict(float)

    @property
    def _target_args(self):
        # Hybrid targets keep text geometry under ``speculative_args``.
        return getattr(self.model, "speculative_args", self.model.args)

    def _target_owner(self, rows):
        """Verify-transaction owner chosen by cache topology, never by model.

        Recurrent + KV lanes use the hybrid record-and-replay owner; plain
        and rotating KV lanes keep the segmented KV owner.
        """
        from .hybrid_verify_rows import HybridVerifyRows, is_hybrid_rows
        from .segmented_rotating_kv import SegmentedKVRows

        if is_hybrid_rows(rows):
            if not getattr(self.model, "supports_speculative_rollback", False):
                raise ValueError("target has recurrent state but declares no speculative rollback")
            _bump(self.scheduler_stats, "external_hybrid_transactions")
            return HybridVerifyRows(rows)
        return SegmentedKVRows(rows)

    @staticmethod
    def _is_hybrid(rows):
        from .hybrid_verify_rows import is_hybrid_rows

        return is_hybrid_rows(rows)

    def _empty_hidden(self):
        return self.mx.zeros((1, 0, len(self.layers)*self._target_args.hidden_size))

    def _append_context(self, lane, following):
        """Commit ``lane.tail`` to the draft plane.

        ``following`` is the token after the tail's last position; a pairing
        drafter receives ``(history + [following])[-L:]``, the token that
        follows each tail feature.
        """
        if not self.pair_context_tokens:
            return self.draft.append_context(lane.tail, lane.draft_cache)
        length = int(lane.tail.shape[1])
        tokens = _context_pairing(lane.history, following, length)
        _bump(self.scheduler_stats, "external_context_token_pairings", length)
        return self.draft.append_context(lane.tail, lane.draft_cache, context_tokens=[tokens])

    def _verify_steer(self, cohort, inputs, proposal_counts):
        """Thinking-guard residual steering for one verify forward.

        Mirrors the prompt-lookup route: per position, exactly what ordinary
        decode applies to the step that feeds that input, so rows before the
        first close token are steered (the anchor included) and the close
        and later rows are not.  The verified target law is then the steered
        law the ordinary route would sample from, so speculative exactness is
        unchanged.  Returns ``(taps, steer, commits)``; ``commits[i](consumed)``
        counts lane ``i``'s committed steered positions.
        """
        taps = getattr(getattr(self.model, "model", None), "residual_taps", None)
        if taps is None:
            return None, None, []
        requests, commits = [], []
        for index, lane in enumerate(cohort):
            block = inputs[index][: proposal_counts[index] + 1]
            # ``history`` is every token before the anchor, the context the
            # ordinary step feeding the anchor saw.
            rows, commit = verify_block_steer(lane.processors, lane.history, block)
            requests.append(rows)
            commits.append(commit)
        width = max((len(row) for row in inputs), default=1)
        steer = stack_block_steer(requests, width)
        if steer is None:
            return None, None, commits
        _bump(self.scheduler_stats, "external_verify_steer_rounds")
        _bump(
            self.scheduler_stats,
            "external_verify_steered_lanes",
            sum(any(request is not None for request in rows) for rows in requests),
        )
        return taps, steer, commits

    @staticmethod
    def _snapshot_draft_state(draft_cache, tail):
        draft_cache, tail, _ = snapshot_prompt_cache_descriptors(
            draft_cache, tail
        )
        return draft_cache, tail

    def insert(self, prompts, *, max_tokens, caches=None, all_tokens=None,
               cache_states=None, lane_rngs=None, sampling_configs=None,
               logits_processors=None, samplers=None, resume_rng=False,
               apc_interior_positions=None, prefill_inputs=None):
        if self._open:
            raise RuntimeError("Membership change during external transaction")
        apc_interior_positions = (
            [()] * len(prompts)
            if apc_interior_positions is None
            else apc_interior_positions
        )
        prefill_inputs = (
            [None] * len(prompts) if prefill_inputs is None else prefill_inputs
        )
        if len(apc_interior_positions) != len(prompts):
            raise ValueError(
                "apc_interior_positions must have one entry per external request"
            )
        if len(prefill_inputs) != len(prompts):
            raise ValueError("prefill_inputs must have one entry per external request")
        if any(positions for positions in apc_interior_positions):
            raise ValueError(
                "APCv2 interior checkpoints are unavailable on external draft routes"
            )
        if any(value is not None for value in prefill_inputs):
            raise ValueError("multimodal prefill is unavailable on external draft routes")
        uids = []
        for i, prompt in enumerate(prompts):
            prefix = list((all_tokens or [[]]*len(prompts))[i]); todo = list(prompt)
            if not todo or max_tokens[i] <= 0:
                raise ValueError("External request needs prompt tokens and positive budget")
            state = (cache_states or [None]*len(prompts))[i]
            target = (caches or [None]*len(prompts))[i]
            seed_key = (lane_rngs or [None]*len(prompts))[i]
            seed = np.asarray(seed_key.key).tolist() if seed_key is not None else self.next_uid
            rng = RequestRNG(seed)
            draft_cache = self.draft.make_cache(); tail = self._empty_hidden()
            if state is not None:
                state.validate(self.binding, len(prefix))
                # Recurrent planes carry no position; their KV siblings pin it.
                if target is None or any(
                    int(c.offset) != len(prefix)
                    for c in target
                    if getattr(c, "offset", None) is not None
                ):
                    raise ValueError("External target cache boundary mismatch")
                target, state, _ = snapshot_prompt_cache_descriptors(
                    target, state
                )
                state.validate(self.binding, len(prefix))
                draft_cache, tail = state.state
                self.scheduler_stats["paired_cache_resumes"] += 1
                if resume_rng and state.rng_key is not None:
                    rng = RequestRNG(state=json.loads(bytes(np.asarray(state.rng_key, dtype=np.uint8)).decode()))
            else:
                # An ordinary APC hit cannot fabricate draft context. Rebuild
                # both planes from transcript instead of pairing stale caches.
                # This includes a zero-length hit: its branch still holds the
                # APC's hooked cache objects, which the lane must not adopt.
                todo = prefix + todo; prefix = []; target = None
            uid = self.next_uid; self.next_uid += 1
            lane = Lane(uid, prefix, deque(todo), list(target) if target is not None else self.model.make_cache(), draft_cache, tail, rng, int(max_tokens[i]), (logits_processors or [[]]*len(prompts))[i] or [], dict((sampling_configs or [{}]*len(prompts))[i]))
            self.lanes[uid] = lane; uids.append(uid)
        return uids

    def _sidecar(self, lane):
        draft_cache, tail = self._snapshot_draft_state(
            lane.draft_cache, lane.tail
        )
        state = ExternalDraftState((draft_cache, tail), len(lane.history), self.mx.array(list(json.dumps(lane.rng.snapshot(), sort_keys=True).encode()), dtype=self.mx.uint8), lane.rng.draws, binding=self.binding)
        state.validate(self.binding, len(lane.history)); return state

    def _prefill(self, lane, *, step=None):
        if len(lane.remaining) > 1:
            n = min(self.prefill_step if step is None else step, len(lane.remaining)-1)
            if lane.tail.shape[1]:
                self._append_context(lane, lane.remaining[0])
            inputs = [lane.remaining.popleft() for _ in range(n)]
            lane.tail = self.model.prefill_body(self.mx.array([inputs]), lane.cache, self.layers)
            lane.history.extend(inputs)
            self.mx.eval(lane.tail, [c.state for c in lane.cache], [c.state for c in lane.draft_cache if c.offset])
            self.scheduler_stats["prefill_rounds"] += 1
            if self.prefill_allocator_reclaim:
                # After the materialization above; lane.tail stays referenced.
                self.mx.clear_cache()
                _bump(self.scheduler_stats, "external_prefill_allocator_reclaims")
        done = len(lane.remaining) == 1
        # (done, span) over the whole prompt; the final token is consumed by
        # the first decode round, so the prompt counts as done here.
        span = len(lane.history) + len(lane.remaining)
        progress = (span if done else len(lane.history), span)
        if done:
            lane.anchor = lane.remaining.popleft()
            if lane.history:
                # A pairing (EAGLE) drafter keeps the last prompt chunk pending:
                # its final feature must be drafted from in the first round.
                if lane.tail.shape[1] and not self.pair_context_tokens:
                    self._append_context(lane, lane.anchor)
                    lane.tail = lane.tail[:, :0]
                self.boundaries[lane.uid] = {"committed_only": True, "tokens": list(lane.history), "target_cache": self._freeze_cache(lane.cache), "covered_tokens": len(lane.history), "cache_sidecar": self._sidecar(lane)}
        return SimpleNamespace(uid=lane.uid, progress=progress, end_of_prompt=done)

    def _target_law(self, lane, logits, history, reachable=True, response_rows=None):
        """Return the verification law of one target row.

        When ``response_rows`` is a list, also append the log-probability row
        to publish for this position, or ``None`` to publish the law itself.
        """
        from .sample_utils import make_transformed_logprobs
        value = logits[None]
        tokens = self.mx.array(history, dtype=self.mx.int32)
        # A row that follows a draft token the target gives zero probability is
        # never used by verification.  Its history can already be outside a
        # structured-output grammar, and asking the processor about it would
        # latch a dead-end failure on a healthy lane (GPU: every structured
        # request on the DFlash2 route returned 502).
        for processor in (lane.processors if reachable else ()):
            value = processor(tokens, value)
        temp = float(lane.sampling.get("sampling_temp", 0))
        if temp == 0:
            # Greedy verification uses the one-hot law, but the logprobs API
            # reports the processed log-softmax, as ordinary decode does.
            row = value[0].astype(self.mx.float32)
            normalizer = self.mx.logsumexp(row)
            selected = self.mx.argmax(row)
            self.mx.eval(normalizer, selected)
            if not np.isfinite(float(normalizer.item())):
                raise LaneFailure(
                    lane.uid,
                    "external draft target law is not a probability distribution",
                )
            if response_rows is not None:
                response_rows.append(row - normalizer)
            p = np.zeros(value.shape[-1]); p[int(selected.item())] = 1; return p
        if response_rows is not None:
            response_rows.append(None)
        transform = make_transformed_logprobs(temp, top_p=lane.sampling.get("top_p", 0), top_k=lane.sampling.get("top_k", 0), min_p=lane.sampling.get("min_p", 0))
        try:
            return probability(np.asarray(self.mx.exp(transform(value)[0])))
        except ValueError as error:
            # Non-finite logits are this request's failure, not the
            # executor's: raising out of next() would stop the worker.
            raise LaneFailure(
                lane.uid,
                f"external draft target law is not a probability distribution: {error}",
            ) from error

    def _verification_receipt(self, lane):
        hists = {
            "verify_span_hist": dict(lane.verify_span_hist),
            "verify_accept_hist": dict(lane.verify_accept_hist),
        }
        if self.fly_verification.enabled and not lane.processors:
            return {
                "verification": "fly",
                "parameters": self.fly_verification.as_dict(),
                "relaxed_accepts": lane.relaxed_accepts,
                **hists,
            }
        result = {
            "verification": "exact",
            "relaxed_accepts": lane.relaxed_accepts,
            **hists,
        }
        if self.fly_verification.enabled and lane.processors:
            result["fly_disabled"] = "logits_processors"
        return result

    def _target_execution_receipt(self):
        if self.target_execution != "tensorfold":
            return {}
        return {
            "tensorfold_target": {
                "cohort_limit": self.tensorfold_cohort_limit,
                "cohort_max_width": self.scheduler_stats[
                    "external_tensorfold_cohort_max_width"
                ],
                "cohort_rounds": self.scheduler_stats[
                    "external_tensorfold_cohort_rounds"
                ],
            }
        }

    def _propose_pairwise(self, lanes, arguments):
        """Batched DFlash2 selection: one pair-table walk, one host read.

        Uniforms come off each lane's stream in position order, exactly the
        draws the sequential sampler makes, so RNG state and receipts match.
        """
        count = arguments[3]
        forbidden = []
        for lane in lanes:
            row = []
            for position in range(count):
                ids = set()
                length = len(lane.history) + 1 + position
                for processor in lane.processors:
                    rule = getattr(
                        processor, "forbidden_token_ids_at_length", None
                    )
                    if not callable(rule):
                        raise TypeError(
                            "batched pairwise selection requires compatible "
                            "forbidden-token processor metadata"
                        )
                    ids.update(int(token) for token in rule(length))
                row.append(tuple(sorted(ids)))
            forbidden.append(row)
        block = self.draft.propose_block(
            *arguments[:4],
            [[lane.rng.uniform() for _ in range(count)] for lane in lanes],
            arguments[5],
            forbidden_token_ids=forbidden,
        )
        _bump(self.scheduler_stats, "external_pairwise_selection_groups")
        _bump(self.scheduler_stats, "external_pairwise_selection_lanes", len(lanes))
        tokens = block.token_lists()
        ids = np.asarray(block.cand_ids)
        q = np.asarray(block.cand_q.astype(self.mx.float32)).astype(np.float64)
        return tokens, [
            CompactDraftRow(tokens[row], ids[row, :length], q[row, :length], len(lanes))
            for row, length in enumerate(block.lengths)
        ]

    def _freeze_lane(self, lane):
        """Freeze one lane at its committed boundary; returns (state, mode).

        The cache planes are captured as recovery descriptors: an
        append-only KV plane (target or draft ``KVCache``) keeps only its fill
        level and is borrowed back from the live cache on restore, and every
        other buffer keeps a descriptor alias.  An alias of an append-only
        buffer (a deep copy of an ``mx.array`` is one too) pinned it for the
        round, so every append copied the whole buffer: O(context) bytes per
        token on both the speculative and the ordinary path.  A graph the
        descriptors refuse (live transaction state) falls back to the deep
        copy.  Host-side fields (RNG, processors, ...) are deep-copied,
        except processors that re-sync from their next call's history, which
        stay shared.  History is journaled separately because rounds only
        append to it.
        """
        fields = vars(lane)
        # Processors that re-sync from the next call's history stay shared.
        shared = rollback_shared_memo(fields.get("processors"))
        # A graph carrying COW bookkeeping (an APC branch's hooked objects)
        # takes the descriptor route too: its segment tokens hold locks a
        # deep copy cannot pickle, as in ``snapshot_committed_cache``.
        try:
            cache, (draft_cache, tail), borrowed = snapshot_recovery_descriptors(
                fields["cache"], (fields["draft_cache"], fields["tail"])
            )
        except COWCacheUnsupported:
            _bump(self.scheduler_stats, "external_cow_fallbacks")
        else:
            host = copy_sharing(
                {
                    k: v for k, v in fields.items()
                    if k not in _LANE_PLANES | _LANE_JOURNAL
                },
                shared,
            )
            _bump(self.scheduler_stats, "external_cow_snapshots")
            return (host, cache, draft_cache, tail, borrowed), "descriptor_cow"
        host = {k: v for k, v in fields.items() if k not in _LANE_JOURNAL}
        return copy_sharing(host, shared), "deepcopy"

    @staticmethod
    def _thaw_lane(frozen):
        # Re-clone so the checkpoint itself stays pristine after a restore;
        # borrowed KV is re-read from the live caches at the captured level.
        host, cache, draft_cache, tail, borrowed = frozen
        cache, (draft_cache, tail) = restore_recovery_descriptors(
            cache, (draft_cache, tail), borrowed
        )
        return {
            **copy.deepcopy(host),
            "cache": cache,
            "draft_cache": draft_cache,
            "tail": tail,
        }

    def _freeze_cache(self, cache):
        """Independent committed target cache for boundaries and finishes."""
        frozen, _sidecar, mode = snapshot_committed_cache(cache)
        if mode == "descriptor_cow":
            _bump(self.scheduler_stats, "external_cow_snapshots")
        elif mode == "deepcopy_fallback":
            _bump(self.scheduler_stats, "external_cow_fallbacks")
        return frozen

    def _snapshot_round(self, cohort):
        """Capture every lane's committed boundary before the round mutates it.

        Must run outside ``SegmentedKVRows.begin/commit``: descriptor COW
        rejects live transactions, which then fall back to deep copies.
        """
        snapshots = []
        for lane in cohort:
            frozen, mode = self._freeze_lane(lane)
            slot = CommittedRecoverySlot()
            boundary = len(lane.history)
            slot.capture(
                route="external_dflash2",
                revision=self.binding,
                boundary=boundary,
                value=frozen,
                snapshot=lambda value: value,
                restore=(
                    self._thaw_lane if mode == "descriptor_cow" else copy.deepcopy
                ),
            )
            snapshots.append(
                RoundSnapshot(slot, boundary, mode, tuple(lane.processors), lane.history)
            )
        return snapshots

    def _restore_round(self, cohort, snapshots):
        for lane, snapshot in zip(cohort, snapshots):
            state = snapshot.slot.restore(
                route="external_dflash2",
                revision=self.binding,
                boundary=snapshot.boundary,
            )
            if len(snapshot.history) < snapshot.boundary:
                raise RuntimeError("committed history was truncated during external recovery")
            state["history"] = snapshot.history[:snapshot.boundary]
            # Serving and the lane share these processor objects.  Restore
            # their mutable round state without replacing that ownership;
            # otherwise a post-fallback grammar failure is invisible to the
            # serving layer's original failure latch.
            restored = state["processors"]
            if len(restored) != len(snapshot.processors):
                raise RuntimeError("processor count changed during external recovery")
            for original, saved in zip(snapshot.processors, restored):
                if original is not saved:
                    if not hasattr(original, "__dict__") or not hasattr(saved, "__dict__"):
                        raise TypeError("cannot restore processor identity in external recovery")
                    original.__dict__.clear()
                    original.__dict__.update(saved.__dict__)
            state["processors"] = list(snapshot.processors)
            lane.__dict__.clear(); lane.__dict__.update(state)

    def _propose(self, cohort):
        """Draft phase: one proposal block per cohort row, ``None`` for no draft.

        A zero-count round only appends pending draft context so the draft
        plane stays paired with the target boundary.
        """
        requested_count = min(
            15 if self.draft_topology == "tree15" else self.num_draft,
            min(l.maximum-l.generated-1 for l in cohort),
        )
        if any(l.ordinary for l in cohort): requested_count = 0
        blocks = [None]*len(cohort)
        if not requested_count:
            for lane in cohort:
                if lane.tail.shape[1]: self._append_context(lane, lane.anchor)
            return blocks
        groups = {}
        for row,lane in enumerate(cohort):
            groups.setdefault((int(lane.tail.shape[1]),str(lane.tail.dtype)),[]).append(row)
        for indices in groups.values():
            lanes = [cohort[i] for i in indices]
            processors = [lane.processors for lane in lanes]
            arguments = (
                [l.anchor for l in lanes],
                self.mx.concatenate([l.tail for l in lanes],axis=0),
                self.draft.batch_caches([l.draft_cache for l in lanes]),
                requested_count,
                [l.rng for l in lanes],
                [float(l.sampling.get("sampling_temp",0)) for l in lanes],
            )
            if self.pair_context_tokens:
                pairing = [
                    _context_pairing(l.history, l.anchor, int(l.tail.shape[1]))
                    for l in lanes
                ]
                _bump(
                    self.scheduler_stats,
                    "external_context_token_pairings",
                    sum(len(row) for row in pairing),
                )
            try:
                import inspect

                supports_processors = bool(
                    getattr(
                        self.draft,
                        "supports_logits_processors",
                        False,
                    )
                )
                try:
                    parameters = inspect.signature(
                        self.draft.draft_distributions
                    ).parameters
                    supports_processors = supports_processors or (
                        "logits_processors" in parameters
                        or any(
                            parameter.kind is inspect.Parameter.VAR_KEYWORD
                            for parameter in parameters.values()
                        )
                    )
                except (TypeError, ValueError):
                    pass
                extra = {"context_tokens": pairing} if self.pair_context_tokens else {}
                if self.draft_topology == "tree15":
                    if self.pair_context_tokens:
                        raise ValueError("tree15 is unavailable for paired-context drafters")
                    forbidden = [self._tree_forbidden(lane) for lane in lanes]
                    trees = None
                    if len(lanes) == 1:
                        trees = self._adopt_prelaunched(
                            lanes[0], requested_count, forbidden[0]
                        )
                    if trees is None:
                        tree_options = self._tree_options(count_hits=True)
                        if self._tree_clock is not None:
                            self._tree_clock("tree_draft_setup")
                            tree_options["mark"] = self._tree_clock
                        trees = self.draft.propose_tree(
                            *arguments[:4], forbidden_token_ids=forbidden, **tree_options
                        )
                    tokens = [tree[0] for tree in trees]
                    q = [
                        TreeDraftRow(tree[0], tree[1], len(lanes))
                        for tree in trees
                    ]
                else:
                    trees = None
                pairwise_processors = all(
                    all(
                        callable(
                            getattr(
                                processor,
                                "forbidden_token_ids_at_length",
                                None,
                            )
                        )
                        for processor in lane.processors
                    )
                    for lane in lanes
                )
                if trees is not None:
                    pass
                elif (
                    self.pairwise_selection == "batched"
                    and pairwise_processors
                    and not self.pair_context_tokens
                ):
                    # Stateless forbidden-token processors can mask the pair
                    # table without a per-position host read. Other processor
                    # kinds and pairing (EAGLE) drafters keep the sequential
                    # path.
                    tokens, q = self._propose_pairwise(lanes, arguments)
                elif any(processors) and supports_processors:
                    tokens,q = self.draft.draft_distributions(
                        *arguments,
                        logits_processors=processors,
                        processor_histories=[list(l.history) for l in lanes],
                        **extra,
                    )
                elif extra:
                    tokens,q = self.draft.draft_distributions(*arguments, **extra)
                else:
                    # Preserve the unstructured fast path byte-for-byte:
                    # no processor lists, prefixes or keyword handling.
                    tokens,q = self.draft.draft_distributions(*arguments)
            except DraftUnavailable as error:
                failed_rows = getattr(error, "failed_rows", None)
                if failed_rows is None:
                    error.failed_uids = tuple(lane.uid for lane in lanes)
                else:
                    error.failed_uids = tuple(
                        lanes[int(local_row)].uid for local_row in failed_rows
                    )
                raise
            _bump(
                self.scheduler_stats,
                "external_draft_masked_positions",
                requested_count * sum(bool(value) for value in processors),
            )
            self.scheduler_stats["draft_max_width"] = max(self.scheduler_stats["draft_max_width"],len(lanes))
            for j,row in enumerate(indices):
                blocks[row] = (
                    q[j] if isinstance(q[j], (CompactDraftRow, TreeDraftRow))
                    else HostDraftRow(tokens[j], q[j], len(lanes))
                )
        return blocks

    def _verify(self, cohort, blocks, logits):
        """Accept phase over the target logits of one verify forward."""
        vocab = int(logits.shape[-1])
        decisions = []
        for row, lane in enumerate(cohort):
            drafts, laws = _block_row(blocks[row], vocab)
            inputs = [lane.anchor] + drafts
            targets, reachable = [], True
            response_rows = [] if lane.sampling.get("emit_logprobs", True) else None
            count = len(drafts)
            # The processors run on every reachable row before the accept
            # point is known; the window rewinds the bookkeeping of the rows
            # past it (steps, tail bound, a latched budget overrun) at commit.
            window = VerifyWindow(lane.processors)
            for j in range(count+1):
                targets.append(self._target_law(lane, logits[row,j], lane.history + inputs[:j+1], reachable, response_rows))
                window.mark()
                if reachable and j < count and lane.processors and targets[-1][int(drafts[j])] <= 0:
                    reachable = False
            if isinstance(laws, CompactDraftRow):
                result = verify_compact_proposals(
                    drafts, laws.candidate_ids, laws.candidate_probs, targets,
                    lane.rng,
                    fly_verification=(
                        self.fly_verification
                        if self.fly_verification.enabled and not lane.processors
                        else None
                    ),
                )
            elif self.fly_verification.enabled and not lane.processors:
                result = verify_proposals(
                    drafts,
                    laws,
                    targets,
                    lane.rng,
                    fly_verification=self.fly_verification,
                )
            else:
                # Preserve exact/default-off verification, including its
                # RNG draw schedule and existing call seam.
                result = verify_proposals(
                    drafts, laws, targets, lane.rng
                )
            # Stop/length truncation is part of the same transaction.
            emitted = list(result.emitted)
            for j,t in enumerate(emitted):
                if t in self.stops:
                    emitted = emitted[:j+1]; break
            decisions.append(
                RoundDecision(
                    result.accepted,
                    emitted,
                    result.target_probabilities,
                    result.relaxed_accepts,
                    response_rows,
                    window,
                )
            )
        return decisions

    @staticmethod
    def _target_tree_parents(block):
        return [-1] + [0 if int(parent) < 0 else int(parent) + 1 for parent in block.parents]

    @staticmethod
    def _tree_paths(parents):
        from .drafters.dflash_tree import tree_paths

        return tree_paths(parents)

    @staticmethod
    def _copy_reference_cache(cache):
        """Clone a B1 cache and force KV appends onto fresh buffers."""

        cloned = copy.deepcopy(cache)
        for item in cloned:
            if (
                hasattr(item, "keys")
                and getattr(item, "keys", None) is not None
                and isinstance(getattr(item, "offset", None), int)
            ):
                offset = int(item.offset)
                item.keys = item.keys[..., :offset, :]
                item.values = item.values[..., :offset, :]
        return cloned

    def _reference_tree_forward(self, lane, inputs, parents):
        """Exact generic control: independently replay every root-to-node path."""

        logits, features = [], []
        for path in self._tree_paths(parents):
            cache = self._copy_reference_cache(lane.cache)
            row_inputs = self.mx.array([[inputs[index] for index in path]])
            row_logits, row_features = self.model.forward_with_taps(
                row_inputs, cache, self.layers
            )
            self.mx.eval(row_logits, row_features)
            logits.append(row_logits[:, -1:])
            features.append(row_features[:, -1:])
        return (
            self.mx.concatenate(logits, axis=1),
            self.mx.concatenate(features, axis=1),
            _ReferenceTreeTransaction(self, lane.cache, inputs),
        )

    def _target_tree_forward(self, lane, inputs, parents):
        if self.target_execution == "reference":
            return self._reference_tree_forward(lane, inputs, parents)
        from .qwen38_tensorfold import forward

        cached = self.tree_gates["cache_executor"]
        result = forward(
            self.model,
            inputs,
            parents,
            lane.cache,
            self.layers,
            os.environ["MLX2_TENSORFOLD_SOURCE"],
            cached=cached,
        )
        _bump(self.scheduler_stats, "external_tensorfold_target_rounds")
        if cached:
            _bump(
                self.scheduler_stats,
                "external_tensorfold_executor_cache_hits"
                if result[2].executor_cached
                else "external_tensorfold_executor_validations",
            )
        return result

    def _target_tree_forward_many(self, cohort, inputs, parents):
        """Queue one independent TensorFold tree per lane as one cohort."""

        if len(cohort) == 1:
            return self._target_tree_forward(cohort[0], inputs[0], parents[0])
        if self.target_execution != "tensorfold":
            raise ValueError("multi-lane tree execution requires TensorFold")
        if len(cohort) > self.tensorfold_cohort_limit:
            raise ValueError("TensorFold target cohort exceeds its bounded lane limit")
        from .qwen38_tensorfold import forward_many

        cached = self.tree_gates["cache_executor"]
        result = forward_many(
            self.model,
            inputs,
            parents,
            [lane.cache for lane in cohort],
            self.layers,
            os.environ["MLX2_TENSORFOLD_SOURCE"],
            cached=cached,
        )
        _bump(self.scheduler_stats, "external_tensorfold_target_rounds")
        _bump(self.scheduler_stats, "external_tensorfold_cohort_rounds")
        _bump(self.scheduler_stats, "external_tensorfold_cohort_lanes", len(cohort))
        self.scheduler_stats["external_tensorfold_cohort_max_width"] = max(
            self.scheduler_stats["external_tensorfold_cohort_max_width"], len(cohort)
        )
        if cached:
            _bump(
                self.scheduler_stats,
                "external_tensorfold_executor_cache_hits"
                if result[2].executor_cached
                else "external_tensorfold_executor_validations",
            )
        return result

    @staticmethod
    def _tree_forbidden(lane):
        """The proposer's forbidden ids at the lane's next position.

        Only when every processor declares the stateless mask; otherwise
        the lattice stays unmasked and verification alone enforces them.
        """

        ids = set()
        for processor in lane.processors:
            rule = getattr(processor, "forbidden_token_ids_at_length", None)
            if not callable(rule):
                return ()
            ids.update(int(token) for token in rule(len(lane.history) + 1))
        return tuple(sorted(ids))

    def _tree_options(self, *, count_hits):
        options = {}
        if self.tree_gates["codebook_cache"]:
            pred, succ, hit = self.draft.tree_codebooks()
            options["codebooks"] = (pred, succ)
            if hit:
                _bump(self.scheduler_stats, "external_tree_codebook_cache_hits")
        return options

    @staticmethod
    def _tree_count(lane):
        return min(15, lane.maximum - lane.generated - 1)

    def _prelaunch_tree(self, lane, decision):
        """Queue the next round's lattice right after this round's commit.

        The lattice runs on a descriptor copy of the draft cache, so the
        committed draft plane is untouched: the copy becomes the lane's only
        when the next round adopts it inside its own transaction, after its
        recovery snapshot, and only if the committed boundary still matches.
        Any failure here discards the work; the committed round stands.
        """

        self._prelaunched.pop(lane.uid, None)
        if (
            lane.ordinary
            or lane.cancelled
            or lane.maximum - lane.generated <= 1
            or (decision.emitted and decision.emitted[-1] in self.stops)
        ):
            return
        try:
            count = self._tree_count(lane)
            forbidden = self._tree_forbidden(lane)
            options = self._tree_options(count_hits=False)
            draft_cache, tail = self._snapshot_draft_state(lane.draft_cache, lane.tail)
            state = self.draft.start_tree(
                [lane.anchor],
                tail,
                self.draft.batch_caches([draft_cache]),
                count,
                forbidden_token_ids=[forbidden],
                **options,
            )
            self.mx.async_eval(*state["pending"])
        except Exception:
            _bump(self.scheduler_stats, "external_tree_pipeline_discards")
            return
        self._prelaunched[lane.uid] = SimpleNamespace(
            tail=lane.tail,
            key=(len(lane.history), lane.anchor, lane.generated, count, forbidden),
            options=options,
            state=state,
            draft_cache=draft_cache,
        )

    def _adopt_prelaunched(self, lane, count, forbidden):
        """Return the queued trees if they belong to this exact boundary."""

        record = self._prelaunched.pop(lane.uid, None)
        if record is None:
            return None
        if (
            record.tail is not lane.tail
            or record.key
            != (len(lane.history), lane.anchor, lane.generated, count, forbidden)
        ):
            _bump(self.scheduler_stats, "external_tree_pipeline_discards")
            return None
        if self._tree_clock is not None:
            self._tree_clock("tree_draft_setup")
        # Inside this round's transaction: a restore returns the committed
        # draft plane captured by the snapshot, never this copy.
        lane.draft_cache = record.draft_cache
        _bump(self.scheduler_stats, "external_tree_pipelined_drafts")
        return self.draft.finish_tree(record.state, mark=self._tree_clock)

    def _discard_prelaunched(self, uids):
        for uid in uids:
            if self._prelaunched.pop(uid, None) is not None:
                _bump(self.scheduler_stats, "external_tree_pipeline_discards")

    def _walk_tree(self, block, sample_row):
        """Follow target samples through matching children; otherwise correct.

        ``sample_row(row, prefix_rows)`` draws the target token at ``row``,
        whose root-to-node path is ``prefix_rows``. Rows are visited in path
        order, so the lane RNG advances once per visited row only.
        """

        inputs = [None] + list(block.tokens)
        parents = self._target_tree_parents(block)
        children = defaultdict(list)
        for row, parent in enumerate(parents):
            if parent >= 0:
                children[parent].append(row)
        row = 0
        commit_rows, emitted = [], []
        while True:
            token = int(sample_row(row, commit_rows + [row]))
            emitted.append(token)
            commit_rows.append(row)
            if token in self.stops:
                break
            child = next(
                (
                    candidate
                    for candidate in children.get(row, ())
                    if int(inputs[candidate]) == token
                ),
                None,
            )
            if child is None:
                break
            row = child
        return commit_rows, emitted

    def _verify_tree(self, lane, block, logits):
        """Row-wise reference: one processed target law and host read per row."""

        inputs = [lane.anchor] + list(block.tokens)
        target_laws = []
        response_rows = (
            []
            if lane.sampling.get("emit_logprobs", True)
            or not self.tree_gates["logprobs_on_request"]
            else None
        )

        def sample_row(row, prefix_rows):
            law = self._target_law(
                lane,
                logits[0, row],
                lane.history + [inputs[index] for index in prefix_rows],
                True,
                response_rows,
            )
            target_laws.append(law)
            return lane.rng.sample(law)

        commit_rows, emitted = self._walk_tree(block, sample_row)
        if response_rows is None:
            _bump(
                self.scheduler_stats,
                "external_tree_logprob_rows_skipped",
                len(commit_rows),
            )
        return RoundDecision(
            accepted=max(0, len(commit_rows) - 1),
            emitted=emitted,
            target_laws=target_laws,
            response_logprobs=response_rows,
            commit_rows=commit_rows,
        )

    @staticmethod
    def _batched_law_contract(lane):
        """True when every processor has a proven batched per-row form.

        Two declared kinds qualify. ``forbidden_token_ids_at_length(n)`` on a
        history-pure processor declares ``where(isin(vocab, ids), -inf,
        logits)`` for an ``n``-token history (``minimum_tokens_processor``).
        ``presence_window = (penalty, context_size, generation_start)``
        declares ``logits - penalty`` on the distinct tokens of that
        processor's generated window (``make_presence_penalty``). Any other
        processor keeps the row-wise reference verifier.
        """

        return all(
            getattr(processor, "presence_window", None) is not None
            or (
                callable(getattr(processor, "forbidden_token_ids_at_length", None))
                and getattr(processor, "history_pure", False)
            )
            for processor in lane.processors
        )

    def _launch_tree_laws(self, lane, block, logits):
        """Queue every tree row's processed target law as one batched tensor.

        Returns ``(fence, state)``: the lazy arrays to evaluate in the round's
        single device read, and what ``_verify_tree_batched`` needs after it.
        Mirrors ``_target_law`` operation for operation, batched over rows:
        each processor in order on each row's own history, then the greedy
        float32 argmax and normalizer, or the sampling transform.
        """

        mx = self.mx
        parents = self._target_tree_parents(block)
        paths = self._tree_paths(parents)
        width = len(paths)
        rows = logits[0, :width]
        vocab = int(rows.shape[-1])
        inputs = [lane.anchor] + list(block.tokens)
        histories = [
            lane.history + [inputs[index] for index in path] for path in paths
        ]
        for processor in lane.processors:
            presence = getattr(processor, "presence_window", None)
            if presence is not None:
                penalty, context_size, start = presence
                flat = []
                for row, history in enumerate(histories):
                    window = history if start is None else history[start:]
                    for token in set(window[-context_size:]):
                        flat.append(row * vocab + int(token))
                if flat:
                    mask = mx.zeros((width * vocab,), dtype=mx.bool_)
                    mask[mx.array(flat, dtype=mx.int32)] = True
                    rows = mx.where(mask.reshape(width, vocab), rows - penalty, rows)
                continue
            forbidden = [
                sorted({
                    int(token)
                    for token in processor.forbidden_token_ids_at_length(len(history))
                })
                for history in histories
            ]
            count = max((len(ids) for ids in forbidden), default=0)
            if count:
                table = mx.array(
                    [ids + [-1] * (count - len(ids)) for ids in forbidden],
                    dtype=mx.int32,
                )
                mask = mx.any(
                    mx.arange(vocab)[None, None, :] == table[:, :, None], axis=1
                )
                rows = mx.where(mask, -float("inf"), rows)
        temp = float(lane.sampling.get("sampling_temp", 0))
        state = {"temp": temp, "vocab": vocab, "width": width}
        if temp == 0:
            values = rows.astype(mx.float32)
            normalizer = mx.logsumexp(values, axis=-1)
            selected = mx.argmax(values, axis=-1)
            state.update(values=values, normalizer=normalizer, selected=selected)
            return (normalizer, selected), state
        from .sample_utils import make_transformed_logprobs

        transform = make_transformed_logprobs(
            temp,
            top_p=lane.sampling.get("top_p", 0),
            top_k=lane.sampling.get("top_k", 0),
            min_p=lane.sampling.get("min_p", 0),
        )
        law = mx.exp(transform(rows))
        finite = mx.all(mx.isfinite(law), axis=-1)
        top_k = int(lane.sampling.get("top_k", 0) or 0)
        if 0 < top_k <= _SPARSE_LAW_LIMIT and top_k < vocab:
            # Top-k leaves at most ``top_k`` nonzero entries per row, so the
            # host needs only their ids and values; ``survivors`` proves it.
            support = mx.argpartition(-law, kth=top_k - 1, axis=-1)[:, :top_k]
            values = mx.take_along_axis(law, support, axis=-1)
            survivors = mx.sum(law > 0, axis=-1)
            state.update(law=law, support=support, support_values=values,
                         survivors=survivors, finite=finite, top_k=top_k)
            return (support, values, survivors, finite), state
        state.update(law=law, finite=finite)
        return (law, finite), state

    def _verify_tree_batched(self, lane, block, state):
        """Host walk over laws already landed by the round's device read.

        Produces the same tokens, laws, RNG draws and failures as
        ``_verify_tree``: each visited row's dense law is rebuilt bit for bit
        and passed through the same ``probability`` and ``RequestRNG.sample``
        calls; a greedy row draws the one uniform the one-hot sample consumes.
        """

        from .speculative_sampling import probability

        emit = lane.sampling.get("emit_logprobs", True)
        keep_laws = emit or not self.tree_gates["logprobs_on_request"]
        target_laws = [] if keep_laws else None
        response_rows = []
        vocab = state["vocab"]
        if state["temp"] == 0:
            normalizer = np.asarray(state["normalizer"])
            selected = np.asarray(state["selected"])

            def sample_row(row, _prefix_rows):
                if not np.isfinite(float(normalizer[row])):
                    raise LaneFailure(
                        lane.uid,
                        "external draft target law is not a probability distribution",
                    )
                token = int(selected[row])
                # RequestRNG.sample on a one-hot law returns its index and
                # consumes exactly one uniform.
                lane.rng.uniform()
                if keep_laws:
                    law = np.zeros(vocab)
                    law[token] = 1
                    target_laws.append(law)
                    response_rows.append(
                        state["values"][row] - state["normalizer"][row]
                    )
                return token
        else:
            finite = np.asarray(state["finite"])
            sparse = "support" in state
            if sparse:
                support = np.asarray(state["support"])
                support_values = np.asarray(state["support_values"])
                survivors = np.asarray(state["survivors"])
                dense_law = None
            else:
                dense_law = np.asarray(state["law"])

            def sample_row(row, _prefix_rows):
                if sparse and int(survivors[row]) > state["top_k"]:
                    raise RuntimeError("top-k target law has more survivors than k")
                if sparse:
                    values = np.zeros(vocab, dtype=support_values.dtype)
                    values[support[row]] = support_values[row]
                    _bump(self.scheduler_stats, "external_tree_sparse_law_rows")
                else:
                    values = dense_law[row]
                try:
                    if not bool(finite[row]):
                        raise ValueError("Invalid probability distribution")
                    law = probability(values)
                except ValueError as error:
                    raise LaneFailure(
                        lane.uid,
                        "external draft target law is not a probability "
                        f"distribution: {error}",
                    ) from error
                if keep_laws:
                    target_laws.append(law)
                    response_rows.append(None)
                return lane.rng.sample(law)

        commit_rows, emitted = self._walk_tree(block, sample_row)
        if not keep_laws:
            response_rows = None
            _bump(
                self.scheduler_stats,
                "external_tree_logprob_rows_skipped",
                len(commit_rows),
            )
        return RoundDecision(
            accepted=max(0, len(commit_rows) - 1),
            emitted=emitted,
            target_laws=target_laws,
            response_logprobs=response_rows,
            commit_rows=commit_rows,
        )

    def _commit(self, cohort, decisions, features, *, blocks, transaction):
        """Commit accepted prefixes, then publish responses for every row."""
        proposal_counts = [0 if block is None else int(block.lengths[0]) for block in blocks]
        consumed = [
            min(decision.accepted+1, len(decision.emitted)) for decision in decisions
        ]
        clock = time.perf_counter() if self.round_timing else None
        paths = [
            (
                list(decision.commit_rows)
                if decision.commit_rows is not None
                else list(range(used))
            )
            for decision, used in zip(decisions, consumed)
        ]
        if any(decision.commit_rows is not None for decision in decisions):
            rows = transaction.commit_paths(paths)
        else:
            rows = transaction.commit(accepted_lengths=consumed)
        for decision, used in zip(decisions, consumed):
            # Delivered token j was drawn from verify row j: exactly ``used``
            # rows (accepted drafts plus the correction/bonus row, cut at a
            # stop) were used, and rows past them leave no trace.
            if decision.verify_window is not None:
                decision.verify_window.settle(used)
        self.scheduler_stats["segmented_transactions"] += len(decisions)
        self.scheduler_stats["segmented_rollbacks"] += sum(
            int(used < count + 1)
            for count,used in zip(proposal_counts,consumed)
        )
        if clock is not None:
            clock = self._mark("transaction_commit", clock)
        for row, (lane, decision) in enumerate(zip(cohort,decisions)):
            count = proposal_counts[row]
            emitted = decision.emitted
            drafts, _laws = _block_row(blocks[row], None)
            inputs = [lane.anchor] + drafts
            committed_inputs = [inputs[index] for index in paths[row]]
            lane.cache = rows[row]
            if decision.commit_rows is None:
                lane.tail = features[row:row+1,:consumed[row]]
            else:
                lane.tail = self.mx.take(
                    features[row:row+1],
                    self.mx.array(paths[row], dtype=self.mx.int32),
                    axis=1,
                )
            lane.history.extend(committed_inputs)
            lane.anchor = emitted[-1]
            round_accepted = min(decision.accepted, len(emitted)-1)
            self.scheduler_stats["accepted_proposals"] += round_accepted
            self.scheduler_stats["proposed_tokens"] += count
            lane.external_rounds += int(count > 0)
            lane.proposed += count
            lane.accepted += round_accepted
            if count:
                # Two host dict bumps on ints already in hand: no device work,
                # no eval, no per-round allocation beyond the at-most-(K+1)
                # integer keys each histogram ever holds.
                lane.verify_span_hist[count + 1] = (
                    lane.verify_span_hist.get(count + 1, 0) + 1
                )
                lane.verify_accept_hist[round_accepted] = (
                    lane.verify_accept_hist.get(round_accepted, 0) + 1
                )
            lane.relaxed_accepts += decision.relaxed
            _bump(
                self.scheduler_stats,
                "fly_relaxed_accepts",
                decision.relaxed,
            )
            lane.target_max_width = max(lane.target_max_width, len(cohort))
            if count:
                lane.draft_max_width = max(lane.draft_max_width, getattr(blocks[row], "width", 1))
            for j,token in enumerate(emitted):
                lane.generated += 1
                finish = "stop" if token in self.stops else "length" if lane.generated >= lane.maximum else None
                final = j == len(emitted)-1
                logp = (
                    self.mx.log(self.mx.array(decision.target_laws[j].astype(np.float32)))
                    if lane.sampling.get("emit_logprobs", True) and decision.target_laws is not None
                    else None
                )
                if logp is not None and decision.response_logprobs:
                    if j < len(decision.response_logprobs) and decision.response_logprobs[j] is not None:
                        logp = decision.response_logprobs[j]
                lane.ready.append(SimpleNamespace(uid=lane.uid, token=token, logprobs=logp, finish_reason=finish, execution_width=len(cohort), all_tokens=list(lane.history) if final else None, prompt_cache=self._freeze_cache(lane.cache) if finish else None, cache_sidecar=self._sidecar(lane) if finish else None, mtp_state=None, mtp_receipt=None, speculative_receipt={"kind":self.receipt_kind, "execution":"external_draft_verify" if lane.external_rounds else "ordinary_target", "current_execution":"ordinary_target" if count == 0 else "external_draft_verify", "ordinary_fallback":lane.ordinary, "external_rounds":lane.external_rounds, "accepted":lane.accepted,"proposed":lane.proposed,"round_accepted":round_accepted,"round_proposed":count,"target_width":lane.target_max_width,"draft_width":lane.draft_max_width,"qualification_authority":"serving_route", **self._target_execution_receipt(), **self._verification_receipt(lane)}))
        if clock is not None:
            self._mark("emit", clock)
        self.scheduler_stats[
            "external_rounds" if any(proposal_counts) else "ordinary_rounds"
        ] += 1
        self.scheduler_stats["target_max_width"] = max(self.scheduler_stats["target_max_width"],len(cohort))

    def _mark(self, phase, since):
        """Accumulate host time for ``phase``; returns the new reference time."""
        now = time.perf_counter()
        self.round_times[phase] += now - since
        # Receipt copy: integer nanoseconds per phase, present only when
        # MLX2_EXTERNAL_ROUND_TIMING is set.
        _bump(
            self.scheduler_stats,
            f"external_phase_{phase}_ns",
            int((now - since) * 1e9),
        )
        return now

    def _round(self, cohort):
        if cohort and all(lane.ordinary for lane in cohort):
            return self._ordinary_round(cohort)
        if self.draft_topology == "tree15" and all(
            lane.maximum - lane.generated > 1 for lane in cohort
        ):
            # A lane's last budget token takes the chain body's zero-count
            # round, which appends the draft context and publishes the
            # sidecar; the ordinary path reported ordinary_fallback for it.
            return self._tree_round(cohort)
        clock = time.perf_counter() if self.round_timing else None
        recovery = self._snapshot_round(cohort)
        self.scheduler_stats["recovery_checkpoint_captures"] += len(recovery)
        if clock is not None:
            clock = self._mark("recovery_capture", clock)
        stats_snapshot = dict(self.scheduler_stats)
        self._open = True
        transaction = None
        try:
            blocks = self._propose(cohort)
            if clock is not None:
                clock = self._mark("draft", clock)
            proposal_counts = [0 if block is None else int(block.lengths[0]) for block in blocks]
            verify_width = max(proposal_counts, default=0) + 1
            inputs = [
                [lane.anchor]+_block_row(block, None)[0]+[0]*(verify_width-count-1)
                for lane,block,count in zip(cohort,blocks,proposal_counts)
            ]
            taps, steer, steer_commits = self._verify_steer(
                cohort, inputs, proposal_counts
            )
            if steer is not None:
                taps.steer = steer
            try:
                if self.target_execution == "tensorfold":
                    parents = [
                        list(range(-1, len(row) - 1)) for row in inputs
                    ]
                    logits, features, transaction = self._target_tree_forward_many(
                        cohort, inputs, parents
                    )
                else:
                    owner = self._target_owner([l.cache for l in cohort])
                    transaction = owner.begin(
                        lengths=[count + 1 for count in proposal_counts]
                    )
                    logits, features = self.model.forward_with_taps(
                        self.mx.array(inputs), transaction.caches, self.layers
                    )
                self.mx.eval(logits, features)
            finally:
                if steer is not None:
                    taps.steer = None
            if clock is not None:
                clock = self._mark("verify_forward", clock)
            decisions = self._verify(cohort, blocks, logits)
            if clock is not None:
                self._mark("verify_laws", clock)
            self._commit(
                cohort, decisions, features, blocks=blocks, transaction=transaction
            )
            for commit, decision in zip(steer_commits, decisions):
                commit(min(decision.accepted + 1, len(decision.emitted)))
        except BaseException:
            if transaction is not None and not transaction.closed:
                try: transaction.abort()
                except BaseException:  # noqa: BLE001, S110 - authoritative snapshots restore below
                    pass
            self._restore_round(cohort, recovery)
            self.scheduler_stats = stats_snapshot
            self.scheduler_stats["recovery_checkpoint_restores"] += len(recovery)
            raise
        finally: self._open = False

    def _tree_round(self, cohort):
        if not cohort:
            return
        if len(cohort) > 1 and self.target_execution != "tensorfold":
            raise ValueError("multi-lane tree15 requires TensorFold target execution")
        if len(cohort) > self.tensorfold_cohort_limit:
            raise ValueError("tree15 cohort exceeds its bounded lane limit")
        clock = time.perf_counter() if self.round_timing else None
        phase = _PhaseClock(self) if self.round_timing else None
        recovery = self._snapshot_round(cohort)
        self.scheduler_stats["recovery_checkpoint_captures"] += len(recovery)
        if phase is not None:
            phase("tree_recovery_capture")
        stats_snapshot = dict(self.scheduler_stats)
        self._open = True
        transaction = None
        self._tree_clock = phase
        try:
            blocks = self._propose(cohort)
            if any(not isinstance(block, TreeDraftRow) for block in blocks):
                raise RuntimeError("tree15 proposal did not return a tree")
            inputs = [
                [lane.anchor] + list(block.tokens)
                for lane, block in zip(cohort, blocks)
            ]
            parents = [self._target_tree_parents(block) for block in blocks]
            logits, features, transaction = self._target_tree_forward_many(
                cohort, inputs, parents
            )
            batched = [
                self.tree_gates["batched_laws"]
                and self._batched_law_contract(lane)
                for lane in cohort
            ]
            states = [None] * len(cohort)
            fences = []
            if all(batched) and self.tree_gates["single_fence"]:
                # Every lane keeps its own law transform and RNG, but all lazy
                # target rows land at one cohort fence.
                for row, (lane, block) in enumerate(zip(cohort, blocks)):
                    fence, states[row] = self._launch_tree_laws(
                        lane, block, logits[row : row + 1]
                    )
                    fences.extend(fence)
                if phase is not None:
                    phase("tree_target_launch")
                self.mx.eval(*fences)
                _bump(
                    self.scheduler_stats,
                    "external_tree_single_fence_rounds",
                    len(cohort),
                )
                if phase is not None:
                    phase("tree_target_wait")
            else:
                if phase is not None:
                    phase("tree_target_launch")
                self.mx.eval(logits, features)
                if phase is not None:
                    phase("tree_target_wait")
                for row, (lane, block, enabled) in enumerate(
                    zip(cohort, blocks, batched)
                ):
                    if enabled:
                        fence, states[row] = self._launch_tree_laws(
                            lane, block, logits[row : row + 1]
                        )
                        fences.extend(fence)
                if fences:
                    self.mx.eval(*fences)
            decisions = []
            for row, (lane, block, enabled) in enumerate(
                zip(cohort, blocks, batched)
            ):
                if enabled:
                    decision = self._verify_tree_batched(
                        lane, block, states[row]
                    )
                    _bump(self.scheduler_stats, "external_tree_batched_law_rounds")
                else:
                    decision = self._verify_tree(
                        lane, block, logits[row : row + 1]
                    )
                    if self.tree_gates["batched_laws"]:
                        _bump(self.scheduler_stats, "external_tree_row_law_rounds")
                decisions.append(decision)
            if phase is not None:
                phase("tree_target_law")
            self._commit(
                cohort,
                decisions,
                features,
                blocks=blocks,
                transaction=transaction,
            )
            if phase is not None:
                phase.skip()
            if self.tree_gates["pipeline_draft"]:
                for lane, decision in zip(cohort, decisions):
                    self._prelaunch_tree(lane, decision)
                if phase is not None:
                    phase("tree_draft_prelaunch")
            _bump(self.scheduler_stats, "external_tree_rounds", len(cohort))
            _bump(
                self.scheduler_stats,
                "external_tree_nodes",
                sum(len(block.tokens) for block in blocks),
            )
            _bump(
                self.scheduler_stats,
                "external_tree_accepted_edges",
                sum(decision.accepted for decision in decisions),
            )
            if clock is not None:
                self._mark("tree_round", clock)
                _bump(self.scheduler_stats, "external_phase_rounds")
        except BaseException:
            if transaction is not None and not transaction.closed:
                try:
                    transaction.abort()
                except BaseException:
                    pass
            self._restore_round(cohort, recovery)
            self.scheduler_stats = stats_snapshot
            self.scheduler_stats["recovery_checkpoint_restores"] += len(recovery)
            self._discard_prelaunched([lane.uid for lane in cohort])
            raise
        finally:
            self._tree_clock = None
            self._open = False

    def _ordinary_round(self, cohort):
        """Advance permanently ordinary external lanes on target state only.

        ``Lane.ordinary`` is irreversible in this executor.  Unlike a final
        transient depth-zero round, these rows can never need DFlash context
        again, so avoid draft append, target taps and speculative rollback.
        """
        snapshots = self._snapshot_round(cohort)
        stats_snapshot = dict(self.scheduler_stats)
        self._open = True
        try:
            inputs = self.mx.array(
                [[lane.anchor] for lane in cohort], dtype=self.mx.int32
            )
            hybrid = None
            if len(cohort) == 1:
                # Preserve the authoritative row's allocation/capacity. A
                # merge/extract round-trip compacts it and perturbs admission.
                batched_cache = cohort[0].cache
            elif self._is_hybrid([lane.cache for lane in cohort]):
                # Recurrent + KV rows: segmented compute views over the
                # authoritative B1 rows, one token each, nothing trimmed.
                hybrid = self._target_owner([lane.cache for lane in cohort]).begin(
                    [1] * len(cohort)
                )
                batched_cache = hybrid.caches
            else:
                batched_cache = []
                for layer in range(len(cohort[0].cache)):
                    rows = [lane.cache[layer] for lane in cohort]
                    merge = getattr(rows[0], "merge", None)
                    if not callable(merge):
                        raise TypeError(
                            f"{type(rows[0]).__name__} cannot batch ordinary external lanes"
                        )
                    batched_cache.append(merge(rows))
            taps, steer, steer_commits = self._verify_steer(
                cohort, [[lane.anchor] for lane in cohort], [0] * len(cohort)
            )
            if steer is not None:
                taps.steer = steer
            try:
                logits = self.model(inputs, cache=batched_cache)
                if hybrid is not None:
                    self.mx.eval(logits)
                    hybrid.commit([1] * len(cohort))
                    self.mx.eval([c.state for lane in cohort for c in lane.cache])
                else:
                    self.mx.eval(logits, [cache.state for cache in batched_cache])
            except BaseException:
                if hybrid is not None and not hybrid.closed:
                    hybrid.abort()
                raise
            finally:
                if steer is not None:
                    taps.steer = None
            for row,lane in enumerate(cohort):
                if len(cohort) > 1 and hybrid is None:
                    lane.cache = [cache.extract(row) for cache in batched_cache]
                response_rows = (
                    [] if lane.sampling.get("emit_logprobs", True) else None
                )
                targets = [
                    self._target_law(
                        lane,
                        logits[row, 0],
                        lane.history + [lane.anchor],
                        True,
                        response_rows,
                    )
                ]
                result = verify_proposals([], [], targets, lane.rng)
                token = int(result.emitted[0])
                if row < len(steer_commits):
                    steer_commits[row](1)
                lane.history.append(lane.anchor)
                lane.anchor = token
                lane.generated += 1
                lane.target_max_width = max(lane.target_max_width, len(cohort))
                finish = (
                    "stop"
                    if token in self.stops
                    else "length"
                    if lane.generated >= lane.maximum
                    else None
                )
                logp = (
                    self.mx.log(
                        self.mx.array(result.target_probabilities[0].astype(np.float32))
                    )
                    if lane.sampling.get("emit_logprobs", True) else None
                )
                if logp is not None and response_rows and response_rows[0] is not None:
                    logp = response_rows[0]
                lane.ready.append(
                    SimpleNamespace(
                        uid=lane.uid,
                        token=token,
                        logprobs=logp,
                        finish_reason=finish,
                        execution_width=len(cohort),
                        # One token is this round's final response, even when
                        # the request continues. Match the multi-token round
                        # contract consumed by serving and downstream adapters.
                        all_tokens=list(lane.history),
                        prompt_cache=self._freeze_cache(lane.cache) if finish else None,
                        # The draft plane stopped at fallback and is stale by
                        # construction.  Publish this completion as target-only.
                        cache_sidecar=None,
                        mtp_state=None,
                        mtp_receipt=None,
                        speculative_receipt={
                            "kind":self.receipt_kind,
                            "execution":"external_draft_verify" if lane.external_rounds else "ordinary_target",
                            "current_execution":"ordinary_target",
                            "ordinary_fallback":True,
                            "external_rounds":lane.external_rounds,
                            "accepted":lane.accepted,
                            "proposed":lane.proposed,
                            "round_accepted":0,
                            "round_proposed":0,
                            "target_width":lane.target_max_width,
                            "draft_width":lane.draft_max_width,
                            "qualification_authority":"serving_route",
                            **self._target_execution_receipt(),
                            **self._verification_receipt(lane),
                        },
                    )
                )
            count = len(cohort)
            self.scheduler_stats["ordinary_rounds"] += 1
            _bump(self.scheduler_stats, "external_ordinary_fast_path_rounds")
            _bump(self.scheduler_stats, "external_ordinary_fast_path_lanes", count)
            _bump(self.scheduler_stats, "external_draft_context_skipped", count)
            if steer is None:
                _bump(self.scheduler_stats, "external_taps_skipped", count)
            _bump(self.scheduler_stats, "external_transactions_skipped", count)
            self.scheduler_stats["target_max_width"] = max(
                self.scheduler_stats["target_max_width"], len(cohort)
            )
        except BaseException:
            self._restore_round(cohort, snapshots)
            self.scheduler_stats = stats_snapshot
            raise
        finally:
            self._open = False

    def _admit(self, lanes, append, *, prefill=False):
        """Explicit reservation estimate; measured peak qualification still required.

        Global KV grows; sliding KV is window bounded but a prefill chunk can
        retain the whole appended span until attention completes. Four bytes
        per KV element conservatively covers the current FP32/BF16 paths.
        The server callback already excludes its mandatory process reserve.
        """
        if self.memory_headroom is None: return True
        target = self._target_args; draft = self.draft.config
        target_per_token = 2*target.num_key_value_heads*target.head_dim*4
        draft_per_token = 2*draft.num_key_value_heads*draft.head_dim*4
        required = 0
        for lane in lanes:
            target_bytes = sum(int(getattr(c,"nbytes",0)) for c in lane.cache)
            draft_bytes = sum(int(getattr(c,"nbytes",0)) for c in lane.draft_cache)
            required += 4*target_bytes + 3*draft_bytes
            required += append*(target.num_hidden_layers*target_per_token + draft.num_hidden_layers*draft_per_token)
            required += append*target.hidden_size*len(self.layers)*8
            required += append*(target.num_hidden_layers*(target.intermediate_size+target.hidden_size)*16 + draft.num_hidden_layers*(draft.intermediate_size+draft.hidden_size)*16)
            if not prefill:
                # Four coexisting float64 q/p originals+normalized copies,
                # native logits/exp/output buffers, selector and safety slack.
                required += append*target.vocab_size*128
        self.scheduler_stats["reservation_bytes"] = required
        if required > self.memory_headroom():
            self.scheduler_stats["memory_deferred"] = self.scheduler_stats.get("memory_deferred",0)+1
            return False
        return True

    def _reclaim_for_admission(self):
        """Pressure-only, bounded reclaim; never reclaim for a lane-count cap."""
        if not self._allocator_reclaimed and self.reclaim_memory is not None:
            self._allocator_reclaimed = True
            self.reclaim_memory()
            return True
        if self._reclaims_left and self.evict_checkpoint is not None:
            self._reclaims_left -= 1
            if self.evict_checkpoint():
                self.scheduler_stats["memory_pressure_evictions"] = self.scheduler_stats.get("memory_pressure_evictions", 0) + 1
                if self.reclaim_memory is not None:
                    self.reclaim_memory()
                return True
        return False

    def _fit_cohort(self, candidates, append):
        # A failed B4 reservation must not block B1/B2 work. Retain each lane's
        # own proposal count; changing width never changes its random schedule.
        while True:
            selected = []
            for lane in candidates:
                if self._admit(selected + [lane], append):
                    selected.append(lane)
            if selected or not self._reclaim_for_admission():
                return selected

    def disable_speculation(self, uid):
        if self._open: raise RuntimeError("Cannot change route inside a transaction")
        self.lanes[uid].ordinary = True
        self._discard_prelaunched([uid])

    def next(self):
        prompts, responses = [], []
        self._reclaims_left = 2; self._allocator_reclaimed = False
        ordered = list(self.lanes.values())
        if ordered:
            start = self._schedule_cursor % len(ordered)
            ordered = ordered[start:] + ordered[:start]
            self._schedule_cursor += 1
        # At most one bounded prefill slice; active decode progresses every poll.
        active_count = sum(l.anchor is not None for l in self.lanes.values())
        for lane in ordered:
            if lane.anchor is not None or active_count >= self.capacity:
                continue
            append = min(self.prefill_step, max(1, len(lane.remaining)-1))
            admitted = self._admit([lane], append, prefill=True)
            while not admitted and self._reclaim_for_admission():
                admitted = self._admit([lane], append, prefill=True)
            initial_append = append
            while not admitted and append > 1:
                append = max(1, append // 2)
                admitted = self._admit([lane], append, prefill=True)
            if admitted:
                if append < initial_append:
                    self.scheduler_stats["prefill_adaptive_slices"] = self.scheduler_stats.get("prefill_adaptive_slices", 0) + 1
                prompts.append(self._prefill(lane, step=append)); break
        ready = [l for l in ordered if l.anchor is not None and not l.ready and not l.cancelled][:self.capacity]
        groups = {}
        for lane in ready:
            # Another lane's token budget must not change this lane's proposal
            # count and random draw schedule. Cohort only compatible counts.
            count = 0 if lane.ordinary else min(self.num_draft,lane.maximum-lane.generated-1)
            # Permanent ordinary lanes must not share a zero-depth round with
            # transient final-budget rows, whose draft context remains exact.
            groups.setdefault((count, lane.ordinary),[]).append(lane)
        for (count, _ordinary), candidates in groups.items():
            if self.target_execution == "tensorfold":
                candidates = candidates[: self.tensorfold_cohort_limit]
            cohort = self._fit_cohort(candidates, count+1)
            if not cohort: continue
            if self.draft_topology == "tree15" and self.target_execution != "tensorfold":
                pending = [[lane] for lane in reversed(cohort)]
            else:
                pending = [cohort]
            while pending:
                group = pending.pop()
                try:
                    self._round(group)
                except DraftUnavailable as error:
                    failed_uids = set(
                        getattr(error, "failed_uids", ())
                        or (lane.uid for lane in group)
                    )
                    failed = [lane for lane in group if lane.uid in failed_uids]
                    retry = [lane for lane in group if lane.uid not in failed_uids]
                    if not failed:
                        raise
                    for lane in failed:
                        lane.ordinary = True
                    self.scheduler_stats["draft_fallbacks"] += len(failed)
                    # The ordinary fallback runs next, before the retry.
                    if retry:
                        pending.append(retry)
                    pending.append(failed)
                except LaneFailure as error:
                    # ``_round`` restored every lane of the group; drop the
                    # failed one and rerun the others from their boundary.
                    self._lane_failures.append({"uid": error.uid, "reason": error.reason})
                    _bump(self.scheduler_stats, "lane_failures")
                    self.remove([error.uid], cancelled=False)
                    retry = [lane for lane in group if lane.uid != error.uid]
                    if retry:
                        pending.append(retry)
        # A lane can finish in the poll that completes its prefill
        # (max_tokens=1, or a stop as the first token).  Its end_of_prompt
        # response is in this same return, so its committed boundary stays
        # for serving to pop, as the ordinary and PLD routes keep it.
        # Dropping it with the lane lost the boundary and failed n>1 fanout
        # siblings.  A boundary returned by an earlier poll was already
        # offered, so it still goes with its lane.
        prompt_ended = {prompt.uid for prompt in prompts if prompt.end_of_prompt}
        for lane in list(self.lanes.values()):
            while lane.ready:
                response = lane.ready.popleft(); responses.append(response)
                if response.finish_reason:
                    self.remove([lane.uid], cancelled=False, keep_boundary=lane.uid in prompt_ended)
                    break
                if self.ready_drain == "one":
                    break
        self._reclaim_after_emission(len(responses))
        return prompts, responses

    def _reclaim_after_emission(self, count):
        """Release the MLX buffer pool on the self-MTP emitted-token cadence.

        Each round frees verify-width KV and activation buffers that the pool
        keeps.  One poll can return several tokens (several lanes, or
        ``ready_drain="all"``), so the clear fires on crossing a multiple of
        the interval, not only on landing on one (Ollama #18510).  Admission
        reclaim stays pressure-only; like the self-MTP clear, this periodic
        one does not synchronize.
        """
        previous = self._emitted_responses
        self._emitted_responses = min(_COUNTER_MAX, previous + int(count))
        if _crossed_counter_interval(
            previous, self._emitted_responses, ALLOCATOR_RECLAIM_MTP_TOKEN_INTERVAL
        ):
            self.mx.clear_cache()
            _bump(self.scheduler_stats, "external_allocator_reclaims")

    def pop_prompt_boundary(self, uid): return self.boundaries.pop(uid, None)

    def take_lane_failures(self):
        """Transfer lanes that failed on their own; serving fails their requests."""
        failures, self._lane_failures = self._lane_failures, []
        return failures

    def remove(self, uids, return_prompt_caches=False, *, cancelled=True, keep_boundary=False):
        if self._open: raise RuntimeError("Cannot remove during external transaction")
        result = {}
        self._discard_prelaunched(uids)
        for uid in uids:
            lane = self.lanes.pop(uid,None)
            if not keep_boundary: self.boundaries.pop(uid,None)
            if lane is not None:
                lane.cancelled = bool(cancelled); self.scheduler_stats["cancelled"] += int(cancelled)
                if return_prompt_caches: result[uid] = self._freeze_cache(lane.cache)
                lane.ready.clear()
        return result

    def close(self):
        self.remove(list(self.lanes)); self.boundaries.clear()
