"""External draft/verify execution composed with mlx2 admission and APCv2.

Original implementation. Target verification uses per-lane segmented cache
transactions; proposal q is supplied by the bound external draft model.
No capability is qualified by importing or constructing this executor.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass, field
from itertools import pairwise
from types import SimpleNamespace

import numpy as np

from ..thinking_guard import stack_block_steer, verify_block_steer
from . import round_phases
from .cohort_ownership import declared_cohort_key
from .committed_recovery import CommittedRecoverySlot
from .ast_private_state import PrivateHistory, freeze_receipt, snapshot_receipt
from .cow_cache import (
    CommittedDescriptorPlan,
    COWCacheUnsupported,
    restore_recovery_descriptors,
    snapshot_committed_cache,
    snapshot_prompt_cache_descriptors,
    snapshot_recovery_descriptors,
)
from .generate import (
    ALLOCATOR_RECLAIM_MTP_TOKEN_INTERVAL,
    _crossed_counter_interval,
    _merge_caches,
    _right_pad_prompts,
)
from .prefill_plan import prompt_length_prefill_step
from .processor_probe import VerifyWindow, copy_sharing, rollback_shared_memo
from .progressive_external_verify import (
    PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION,
    PROGRESSIVE_EXTERNAL_VERIFY_VERSION,
    PROGRESSIVE_FIDELITY_EXACT,
    PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED,
    PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    execute_progressive_target_rows,
    progressive_target_protocol_binding,
    verify_greedy_proposal_tile,
    verify_proposal_tile,
)
from .proposal_providers import (
    InsufficientContinuationAlternatives,
    PointMassLaw,
)
from .ragged_verify_layout import RaggedVerifyLayout, padded_capabilities
from .speculative_sampling import (
    FLyVerificationPolicy,
    RequestRNG,
    count_delivered_relaxed_accepts,
    probability,
    validate_compact_proposal_laws,
    validate_proposal_laws,
    verify_block_proposals,
    verify_compact_block_proposals,
    verify_compact_proposals,
    verify_greedy_proposals,
    verify_proposals,
)
from .speculative_topology import (
    OrderedLiveFrontierPlanner,
    PerSessionSpeculationTopology,
    confidence_tree_branch_depths,
    confidence_tree_trunk_length,
    ordered_alternative_proposals,
    ordered_alternative_winner,
    ordered_branch_commit_counts,
    ordered_branch_outcomes,
    processors_branch_safe,
    staged_branch_indices,
    validate_ordered_alternative_strategy,
)

_COUNTER_MAX = (1 << 63) - 1


def _ast_artifact_fingerprint(value, *, binding):
    """Stable memo-only identity; never falls back to an object address."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        payload = json.dumps(
            {
                "binding": str(binding),
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    return hashlib.sha256(payload.encode()).hexdigest()


def _pad_cohort_trees(inputs, parents):
    """Pad a cohort's trees to one width for a uniform provider tile.

    A drafter may propose fewer nodes for one lane than another (prompt
    lookup finds fewer copies).  Each short tree gets extra leaves under its
    anchor, with token ids that are not already among the anchor's children,
    appended after its real nodes.  The walk and commit read only a lane's
    own tree (``logits[:, :len(tree)]`` and committed path indices), so the
    padding rows are computed and ignored.
    """
    width = max(len(row) for row in inputs)
    if all(len(row) == width for row in inputs):
        return inputs, parents
    padded_inputs, padded_parents = [], []
    for row, tree in zip(inputs, parents, strict=True):
        row, tree = list(row), list(tree)
        taken = {row[index] for index, parent in enumerate(tree) if parent == 0}
        filler = 0
        while len(row) < width:
            while filler in taken:
                filler += 1
            row.append(filler)
            tree.append(0)
            taken.add(filler)
        padded_inputs.append(row)
        padded_parents.append(tree)
    return padded_inputs, padded_parents


def _speculation_parked(lane) -> bool:
    """A reversible cost-handoff session that is parked and not re-probing."""
    state = getattr(lane, "cost_handoff_state", None)
    return state is not None and state.speculation_paused


class _CommittedPlainRows:
    """A plain transaction already committed before its host read.

    The pipelined parked round commits its one-token plain transaction before
    the following round begins (a plain round accepts every anchor, so its
    commit needs no host token); ``_commit`` then books the round against
    the rows that commit returned.
    """

    closed = True

    def __init__(self, rows):
        self._rows = rows

    def commit(self, accepted_lengths):
        if any(int(count) != 1 for count in accepted_lengths):
            raise RuntimeError("a pipelined parked round accepts exactly its anchor")
        return self._rows


class _RngMark:
    """A request RNG's exact stream position, restorable onto the same object.

    A parked round's undo record used a deep copy of the lane RNG, taken twice
    per round on the decode critical path.  ``RequestRNG`` is a NumPy PCG64
    stream plus a draw count: its ``bit_generator.state`` getter already
    returns a detached snapshot, and assigning it back rewinds the stream to
    exactly that position.  Other RNG types keep the deep copy.
    """

    __slots__ = ("rng", "state", "draws")

    def __init__(self, rng):
        self.rng = rng
        self.state = rng.generator.bit_generator.state
        self.draws = rng.draws

    def restore(self):
        self.rng.generator.bit_generator.state = self.state
        self.rng.draws = self.draws
        return self.rng


def _rng_mark(rng):
    if type(rng) is RequestRNG:
        return _RngMark(rng)
    return copy.deepcopy(rng)


@dataclass
class _ParkedInflight:
    """One dispatched, unread anchor-only round of a parked cohort."""

    lanes: list
    transaction: object
    logits: object
    features: object
    next_anchors: object
    normalizers: object
    logprob_rows: object
    recovery: list
    timing_start_ns: int
    generated_before: list

def _bump(stats, key, amount=1):
    stats[key] = min(_COUNTER_MAX, int(stats.get(key, 0)) + int(amount))


def _bump_map(stats, key, field, amount=1):
    """Bump ``stats[key][field]`` copy-on-write.

    The scheduler snapshot shares nested dicts with this generator; replacing
    the map keeps an exporter iterating the previous one consistent.
    """
    previous = stats.get(key) or {}
    stats[key] = {
        **previous,
        field: min(_COUNTER_MAX, int(previous.get(field, 0)) + int(amount)),
    }


# Per-uid draft-phase side tables; reset each draft phase and popped on remove.
_LAST_ALTERNATIVE_FIELDS = (
    "_last_alternative_blocks", "_last_alternative_prefix_tokens",
    "_last_alternative_latch", "_last_alternative_topology",
    "_last_alternative_downshift", "_last_alternative_prefix_search",
    "_last_alternative_prefix_diagnostics",
)


def _target_executor_stats():
    return {
        "schema": "mlx2.progressive-target-row-executor.v1",
        "scope": "physical_launches_shared_by_receipt_cohort",
        "launches": 0,
        "adapter_owned_launches": 0,
        "legacy_launches": 0,
        "row_exact_launches": 0,
        "nonexact_launches": 0,
        "physical_width_hist": {},
        "token_width_hist": {},
        "adapter_route_hist": {},
        "last": None,
    }


def _observe_target_executor(stats, receipt):
    stats["launches"] += 1
    execution = receipt["execution"]
    counter = (
        "adapter_owned_launches"
        if execution == "adapter_owned"
        else "legacy_launches"
    )
    stats[counter] += 1
    if receipt["row_exact"] is True:
        stats["row_exact_launches"] += 1
    elif receipt["row_exact"] is False:
        stats["nonexact_launches"] += 1
    for source, target in (
        ("physical_width", "physical_width_hist"),
        ("token_width", "token_width_hist"),
    ):
        key = str(int(receipt[source]))
        stats[target][key] = stats[target].get(key, 0) + 1
    route = receipt.get("adapter_detail", {}).get("route")
    if isinstance(route, str) and route:
        stats["adapter_route_hist"][route] = (
            stats["adapter_route_hist"].get(route, 0) + 1
        )
    stats["last"] = copy.deepcopy(receipt)


def _merge_target_executor_stats(*rows):
    result = _target_executor_stats()
    for row in rows:
        if not row:
            continue
        for key in (
            "launches",
            "adapter_owned_launches",
            "legacy_launches",
            "row_exact_launches",
            "nonexact_launches",
        ):
            result[key] += int(row.get(key, 0))
        for key in (
            "physical_width_hist",
            "token_width_hist",
            "adapter_route_hist",
        ):
            for width, count in row.get(key, {}).items():
                result[key][str(width)] = (
                    result[key].get(str(width), 0) + int(count)
                )
        if row.get("last") is not None:
            result["last"] = copy.deepcopy(row["last"])
    return result


def _first_tiles_pairwise_disjoint(token_rows, tile):
    """Prove proposal diversity without publishing request-private tokens."""
    first = [set(row[:tile]) for row in token_rows]
    return all(
        first[left].isdisjoint(first[right])
        for left in range(len(first))
        for right in range(left + 1, len(first))
    )


def _delivered_accepts(decision):
    """Accepted proposals that were delivered.  ``emitted`` normally ends
    with the target's correction or bonus token, but a stop can cut delivery
    on an accepted draft, and then every emitted token is an accepted one."""
    return min(int(decision.accepted), len(decision.emitted))


def _history_window(history, suffix, context_size, generation_start):
    """Slice a logical history+suffix without copying its unused prefix.

    Match ``history[generation_start:][-context_size:]``, including the
    full-window spelling ``context_size=0`` and Python's negative indices.
    The slices are applied to the combined sequence before splitting it.
    """
    boundary = len(history)
    length = boundary + len(suffix)
    start = slice(generation_start, None).indices(length)[0]
    start += slice(-context_size, None).indices(length - start)[0]
    return history[start:] + suffix if start < boundary else suffix[start - boundary:]


# Tree15 round-cost gates (target-tree provider bridge). Off on an explicit
# tree15 topology unless set; on by default for a selected bounded tree route
# (``dynamic_singleton_tree``, the Qwen3.8 DFlash2 default policy since
# 187c6aa23), where ``0`` turns one off.  Each maps to its environment
# spelling and the accepted value that enables it.
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
        "external_target_tree_executor_validations",
        "external_target_tree_executor_cache_hits",
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
        "external_tree_early_prelaunches",
        "external_tree_pipeline_reordered_adoptions",
    ),
}


def _chain_draft_cap(tree_budgets, max_tree_width, active_width):
    """Per-lane chain depth on the bounded tree route's B5+ chain leg.

    The adapter's tree budgets hold the verify forward near one row budget
    at every tree width (15/7/4/3 nodes: 16/16/15/16 target rows).  Past the
    widest tree the route switches to chains; uncapped that was
    ``num_draft + 1`` rows per lane (40 at B5, 64 at B8) on the unpacked
    reference target, a geometry no campaign measured and wider than the
    32-row built-in lane law.  Keep the same row budget instead:
    ``rows // lanes - 1`` drafts (2 at B5, 1 at B6-B8, 0 from B9).  ``None``
    when no cap applies (tree widths, and the B1-only tree route).
    """
    active_width = int(active_width)
    if max_tree_width != 4 or active_width <= max_tree_width:
        return None
    rows = max(lanes * (nodes + 1) for lanes, nodes in tree_budgets.items())
    return max(0, rows // active_width - 1)


def _tree_gates(topology, target_execution, *, default_on=False):
    """Read the tree experiment gates; fail closed on a gate the route lacks."""

    gates = {}
    for gate, (name, enabled) in _TREE_GATES.items():
        value = os.environ.get(name, "")
        if value not in ("", "0", enabled):
            raise ValueError(f"{name} must be unset, 0 or {enabled}")
        gates[gate] = value == enabled or (not value and default_on)
    if any(gates.values()) and topology != "tree15":
        raise ValueError("tree experiment gates require MLX2_DFLASH_TOPOLOGY=tree15")
    if gates["cache_executor"] and target_execution != "provider":
        raise ValueError(
            "MLX2_TENSORFOLD_CACHE_EXECUTOR requires target-tree provider execution"
        )
    if gates["single_fence"] and not gates["batched_laws"]:
        raise ValueError(
            "MLX2_TREE_SINGLE_FENCE requires MLX2_TREE_BATCHED_TARGET_LAWS=1"
        )
    return gates


def _target_tree_cohort_limit(
    target_execution, capacity, *, default=1, selected=None, maximum=4
):
    """Return the explicit bounded provider cohort limit (singleton default)."""

    name = "MLX2_TENSORFOLD_COHORT_LIMIT"
    raw = os.environ.get(name)
    if selected is not None:
        if raw is not None:
            raise ValueError(
                f"tensorfold_cohort_limit conflicts with explicit {name} override"
            )
        if type(selected) is not int:
            raise ValueError("tensorfold_cohort_limit must be an integer")
        raw = str(selected)
    if raw is None:
        return min(int(capacity), int(default))
    if target_execution != "provider":
        raise ValueError(f"{name} requires target-tree provider execution")
    allowed = {str(value) for value in range(1, maximum + 1)}
    if raw not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(sorted(allowed))}")
    return min(int(capacity), int(raw))


def _target_tree_execution_mode(*, default_provider=False):
    """Translate public and legacy ingress values into scheduler-owned modes."""

    public_name = "MLX2_TARGET_TREE_EXECUTION"
    legacy_name = "MLX2_QWEN_TARGET_EXECUTION"
    public = os.environ.get(public_name)
    legacy = os.environ.get(legacy_name)
    if public is not None and legacy is not None:
        translated_legacy = "provider" if legacy == "tensorfold" else legacy
        if public != translated_legacy:
            raise ValueError(
                f"{public_name} conflicts with legacy {legacy_name}"
            )
    raw = public if public is not None else legacy
    if raw is None:
        return "provider" if default_provider else "reference"
    mode = "provider" if raw == "tensorfold" else raw
    if mode not in ("reference", "provider"):
        raise ValueError(
            f"{public_name} must be reference or provider "
            f"({legacy_name}=tensorfold remains an ingress alias)"
        )
    return mode


# Laws with at most this many top-k survivors are read back sparsely.
_SPARSE_LAW_LIMIT = 256


class _PhaseClock:
    """Accumulate time between phase boundaries (diagnostic timing only)."""

    __slots__ = ("owner", "last")

    def __init__(self, owner):
        self.owner = owner
        self.last = owner._phase_time()

    def __call__(self, phase):
        self.last = self.owner._mark(phase, self.last)

    def skip(self):
        self.last = self.owner._phase_time()


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
    # Absolute prompt positions where prefill must end a slice.  These match
    # ordinary serving's numerical geometry but never request or publish an
    # APCv2 checkpoint on the external route.
    prefill_cuts: deque = field(default_factory=deque)
    prefill_inserted_at: float = field(default_factory=time.monotonic)
    prefill_coalesce_expired_recorded: bool = False
    generated: int = 0
    anchor: int | None = None
    ready: deque = field(default_factory=deque)
    cancelled: bool = False
    ordinary: bool = False
    # Why this lane decodes on the ordinary target path (None: it does not,
    # or the reason is a recoverable draft fallback counted elsewhere).
    ordinary_reason: str | None = None
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
    # Paired ``verified positions:accepted draft tokens`` histogram.  The two
    # marginal histograms above cannot prove that a particular short accept
    # occurred in a full tree round; this receipt makes planted and live
    # partial-commit gates unambiguous without retaining per-round records.
    verify_accept_span_hist: dict = field(default_factory=dict)
    # Lagged proposal-source scores belong to this request's verification
    # history. This state is copied by the normal round recovery snapshot.
    proposal_score_counts: dict = field(default_factory=dict)
    # Per-source trusted-PLD cadence/revocation state. Request ownership makes
    # it tenant-isolated; round snapshots restore it and lane removal discards it.
    proposal_trust_states: dict = field(default_factory=dict)
    # Committed target-tree geometry for this request. The bounded
    # trace makes allocator/formation-sensitive output drift diagnosable
    # without retaining tokens, prompts, timestamps, or unbounded history.
    target_tree_width_hist: dict = field(default_factory=dict)
    target_tree_width_trace: list = field(default_factory=list)
    target_tree_width_trace_overflow: int = 0
    cost_handoff_state: object | None = None
    cost_handoff_receipt: dict | None = None
    # Opt-in executor timing baseline captured at admission.  Phase counters
    # are process-local scheduler diagnostics, not request-exclusive clocks;
    # retaining the baseline on the lane makes a final request receipt useful
    # even when the periodic /v1/status snapshot has not refreshed yet.
    external_timing_start: dict = field(default_factory=dict)


# History changes only by append inside a decode round.  Keep its committed
# list and length as a journal, copying the prefix only if recovery is needed.
_LANE_PLANES = frozenset({"cache", "draft_cache", "tail"})
_LANE_JOURNAL = frozenset({"history"})
# These AST values are replaced, not mutated, inside a round. In particular,
# tuning is attached to the completed receipt only after _run_round returns.
# Alias their old versions in the recovery journal instead of recursively
# copying the same bounded diagnostic history on every decode round.
_LANE_AST_REPLACED = (
    "ast_tree_tiles_receipt", "ast_tree_tuning", "ast_tree_source_plan",
    "ast_source_budget",
)


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
    confidence_features: object = None
    proposal_source: str | None = None
    proposal_score_key: str | None = None
    proposal_score: float | None = None
    proposal_trust_key: str | None = None
    proposal_trusted: bool = False
    proposal_audit: bool = False
    proposal_periodic_audit: bool = False
    continuation_selection: object = None
    continuation_admission: object = None
    continuation_slate: object = None
    continuation_ast_provenance: object = None
    continuation_ast_admitted_provenance: object = None
    continuation_ast_branch_index: int | None = None
    continuation_ast_admitted_depths: object = None
    feedback_payload: object = None

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
    proposal_source: str | None = None
    proposal_score_key: str | None = None
    proposal_score: float | None = None
    proposal_trust_key: str | None = None
    proposal_trusted: bool = False
    proposal_audit: bool = False
    proposal_periodic_audit: bool = False

    @property
    def lengths(self):
        return (len(self.tokens),)


@dataclass
class TreeDraftRow:
    """One best-first proposal tree; parents index ``tokens`` and use -1 as root."""

    tokens: list
    parents: list
    width: int = 1
    # Calibrated drafters report per node its depth, own log q and the
    # calibrated path log-probability (``calibrated_best_first_tree``).
    info: object = None

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
    continuation_outcome: object = None
    continuation_semantic_index: int | None = None
    commit_count: int | None = None
    committed_inputs: object = None
    relaxed_positions: tuple[int, ...] = ()
    # Tree rounds only: the committed feature rows, built once before commit
    # so the next round's lattice can start from the very same array.
    committed_tail: object = None


def _greedy_proposal_supported(block, laws, drafts, vocab):
    """Validate q as the dense/compact verifier would, before a greedy compare.

    The greedy verifier never reads q again, so a malformed law (negative,
    non-finite or zero-sum mass, wrong shape or count, duplicate or
    out-of-range compact ids) must be refused here, before any RNG draw,
    exactly where ``verify_proposals``/``verify_compact_proposals`` refuse it.
    """
    if not drafts:
        return
    if isinstance(laws, (list, tuple)) and all(
        isinstance(law, PointMassLaw) for law in laws
    ):
        if len(laws) != len(drafts) or any(
            law.token != int(token) or not 0 <= law.token < vocab
            for token, law in zip(drafts, laws, strict=True)
        ):
            raise ValueError("invalid sparse point-mass proposal laws")
    elif isinstance(laws, CompactDraftRow):
        validate_compact_proposal_laws(
            drafts, laws.candidate_ids, laws.candidate_probs, vocab
        )
    elif laws is None:
        # A compact ``DraftBlock`` row read without its dense laws.
        import mlx.core as mx

        length = len(drafts)
        ids = np.asarray(block.cand_ids)[0, :length]
        probs = np.asarray(block.cand_q.astype(mx.float32))[0, :length]
        validate_compact_proposal_laws(drafts, list(ids), list(probs), vocab)
    else:
        validate_proposal_laws(drafts, laws, vocab)


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


def _proposal_collision_probabilities(laws):
    """Distribution-only AST features; this API cannot inspect draft tokens."""
    from .acceptance_estimator import proposal_law_features

    if isinstance(laws, CompactDraftRow):
        rows = laws.candidate_probs
    elif isinstance(laws, TreeDraftRow):
        raise TypeError(
            "AST collision probability requires proposal distributions"
        )
    else:
        rows = laws
    result = []
    for law in rows:
        result.append(
            1.0
            if isinstance(law, PointMassLaw)
            else proposal_law_features(law).collision_probability
        )
    if not result:
        raise ValueError("AST collision probability requires a nonempty law row")
    return tuple(result)


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

    The ordinary target trunk is evaluated at cohort width B. The provider keeps
    B1 state records but queues up to four lane trees behind one evaluation
    fence. Draft trunk forwards group lanes by pending-context length;
    attention remains per-row SDPA. Receipts report actual target and draft
    widths. No dense cache repacking or shared global random generator.
    """
    # Class default so instances built without __init__ (test doubles that
    # bind only the lane methods) keep the off behaviour.
    prefill_allocator_reclaim = False
    external_varlen_prefill = False
    progressive_alternative_branches = 1
    progressive_ast_rootless_fallback = True
    progressive_cross_session_pack_rows = None
    progressive_cross_session_pack_mode = None
    adaptive_policy = None
    ast_collision_probability = False
    ast_stage_row_telemetry = False
    ast_physical_target = False
    multilane_draft_cap = None

    def __init__(self, model, *, draft_model, binding,
                 target_artifact_fingerprint=None, completion_batch_size=4,
                 prefill_step_size=2048, prefill_step_autoscale=False,
                 num_draft=4, stop_tokens=(), memory_headroom=None,
                 reclaim_memory=None, evict_checkpoint=None, fly_verification=None,
                 pairwise_selection="host", ready_drain=None,
                 prefill_allocator_reclaim=False, adaptive_verification=None,
                 continuation_pool=None, continuation_verification_strategy=None,
                 continuation_pool_ast_bridge=False,
                 dynamic_singleton_tree=False,
                 dynamic_tree_max_width=1, external_varlen_prefill=False,
                 tree_node_budget_by_lanes=None,
                 tensorfold_cohort_limit=None,
                 target_tree_executor_provider=None,
                 ast_target_tree_executor_provider=None,
                 ast_b1_tile_policy=None,
                 minimum_draft_proposals=None,
                 external_prefill_coalesce_ms=0,
                 external_prefill_coalesce_min_tokens=1,
                 exact_verification="token",
                 progressive_verification_tile=None,
                 progressive_verification_tile_policy=None,
                 progressive_terminal_input_widths=None,
                 progressive_alternative_branches=1,
                 progressive_alternative_strategy="distinct_root",
                 progressive_alternative_depth_policy="full",
                 progressive_ast_rootless_fallback=True,
                 progressive_cross_session_pack_rows=None,
                 progressive_cross_session_pack_mode=None,
                 multilane_draft_cap=None,
                 progressive_verification_fidelity=None,
                 progressive_target_protocol=None,
                 progressive_protocol_binding=None,
                 ast_collision_probability=False,
                 ast_stage_row_telemetry=False,
                 ast_physical_target=False,
                 ast_memo=None,
                 decode_time_fairness=None, decode_first=None,
                 cost_per_commit_handoff=None,
                 draft_tail_equalize=None,
                 ast_topology=None,
                 ast_adaptive_rows=None,
                 ast_tree_calibration=None,
                 ast_batch_row_sharing=False,
                 **kwargs):
        import mlx.core as mx
        self.mx = mx; self.model = model; self.draft = draft_model
        # Tree-tile proposal topology for drafters that build paths rather
        # than their own lattice ("3x5", "4/4/4/3", "chain"); validated here.
        if ast_topology is not None:
            from .ast_tree_proposals import parse_topology

            parse_topology(ast_topology)
        self.ast_topology = ast_topology
        if ast_adaptive_rows not in (None, True, False):
            raise ValueError("ast_adaptive_rows must be a boolean")
        self._ast_adaptive_rows = ast_adaptive_rows
        from .ast_tree_calibration import parse_mode as _calibration_mode

        self._ast_tree_calibration_mode = _calibration_mode(ast_tree_calibration)
        round_phases.initialize(self, decode_first)
        self.memory_headroom = memory_headroom
        self.reclaim_memory = reclaim_memory; self.evict_checkpoint = evict_checkpoint
        self._schedule_cursor = 0; self._reclaims_left = 0; self._allocator_reclaimed = False
        # Responses returned by next(); drives the periodic allocator reclaim.
        self._emitted_responses = 0
        self.binding = binding; self.capacity = completion_batch_size
        # ``binding`` identifies the composed target+drafter execution route.
        # Physical-target discovery must instead compare the provider against
        # the target artifact whose tensors it executes.
        self.target_artifact_fingerprint = (
            binding
            if target_artifact_fingerprint is None
            else target_artifact_fingerprint
        )
        self._generator_revision = uuid.uuid4().hex
        self.ast_memo = ast_memo
        # Physical-target discovery is an installed semantic capability, not
        # an adapter tuning declaration.  Discovery is read-only and cannot
        # select an executor or open a transaction in this integration slice.
        self.ast_physical_target_provider = getattr(
            model, "ast_physical_target_provider", None
        )
        if type(ast_physical_target) is not bool:
            raise ValueError("ast_physical_target must be boolean")
        self.ast_physical_target = ast_physical_target
        if ast_physical_target and self.ast_physical_target_provider is None:
            raise ValueError(
                "ast_physical_target requires an adapter-owned provider"
            )
        if type(ast_stage_row_telemetry) is not bool:
            raise ValueError("ast_stage_row_telemetry must be boolean")
        self.ast_stage_row_telemetry = ast_stage_row_telemetry
        # Immutable semantic discovery is reused only for an equal owner
        # contract. APC metadata does not replace this hot-path-local cache.
        self._ast_execution_capabilities = {}
        from .cost_per_commit_handoff import CostPerCommitHandoffSettings

        explicit_cost_handoff = cost_per_commit_handoff is not None
        self.ast_cost_self_configure = (
            ast_physical_target and not explicit_cost_handoff
        )
        # Mutable performance feedback stays local to this exact generator
        # binding.  It is neither APC metadata nor adapter configuration.
        self._ast_cost_prior = None
        if self.ast_cost_self_configure and not explicit_cost_handoff:
            self.cost_per_commit_handoff = (
                CostPerCommitHandoffSettings.ast_self_configuring_defaults()
            )
            self.cost_per_commit_handoff_authority = (
                "runtime_ast_self_configuring_default"
            )
        else:
            self.cost_per_commit_handoff = (
                CostPerCommitHandoffSettings.from_value(cost_per_commit_handoff)
            )
            self.cost_per_commit_handoff_authority = (
                "explicit_policy"
                if self.cost_per_commit_handoff is not None
                else None
            )
        self._feedback_outbox = None
        self.continuation_policy = None
        if type(continuation_pool_ast_bridge) is not bool:
            raise ValueError("continuation_pool_ast_bridge must be boolean")
        self.continuation_pool_ast_bridge = continuation_pool_ast_bridge
        if continuation_pool is not None:
            from .proposal_providers import ContinuationPoolPolicy
            self.continuation_policy = (continuation_pool if isinstance(continuation_pool, ContinuationPoolPolicy)
                                        else ContinuationPoolPolicy.from_value(continuation_pool))
            if not hasattr(draft_model, "last_continuation_selections") or self.continuation_policy != draft_model.policy:
                raise ValueError("continuation_pool requires the matching bound proposal provider")
        elif hasattr(draft_model, "last_continuation_selections"):
            self.continuation_policy = draft_model.policy
        if getattr(self, "continuation_pool_ast_bridge", False) and self.continuation_policy is None:
            raise ValueError(
                "continuation_pool_ast_bridge requires a bound continuation pool"
            )
        from .continuation_strategy import ContinuationStrategy

        self.continuation_strategy = ContinuationStrategy.from_value(
            continuation_verification_strategy
        )
        if (
            self.continuation_strategy.algorithm
            != "parallel_complete_paths_v1"
            and self.continuation_policy is None
        ):
            raise ValueError(
                "continuation verification strategy requires a bound continuation pool"
            )
        if (
            self.continuation_strategy.cache_layout is not None
            and getattr(model, "apc_v2_layout", None)
            != self.continuation_strategy.cache_layout
        ):
            raise ValueError("continuation strategy cache layout mismatch")
        self.fly_verification = FLyVerificationPolicy.from_value(fly_verification)
        self.prefill_step = prefill_step_size; self.num_draft = int(num_draft)
        from .draft_capabilities import maximum_proposal_length

        if not 1 <= self.num_draft <= maximum_proposal_length(draft_model):
            raise ValueError("External draft count exceeds adapter execution capacity")
        if type(prefill_step_autoscale) is not bool:
            raise ValueError("prefill_step_autoscale must be boolean")
        self.prefill_step_autoscale = prefill_step_autoscale
        if minimum_draft_proposals is None:
            declared_floor = getattr(
                draft_model, "minimum_proposal_length", 1
            )
            if type(declared_floor) is not int:
                raise ValueError(
                    "draft minimum_proposal_length must be an integer"
                )
            minimum_draft_proposals = min(declared_floor, self.num_draft)
        if (
            type(minimum_draft_proposals) is not int
            or not 1 <= minimum_draft_proposals <= self.num_draft
        ):
            raise ValueError(
                "minimum_draft_proposals must be an integer from 1 to num_draft"
            )
        self.minimum_draft_proposals = minimum_draft_proposals
        if type(prefill_step_autoscale) is not bool:
            raise ValueError("prefill_step_autoscale must be boolean")
        self.prefill_step_autoscale = prefill_step_autoscale
        if type(external_varlen_prefill) is not bool:
            raise ValueError("external_varlen_prefill must be boolean")
        if external_varlen_prefill and not callable(
            getattr(model, "prefill_row_context", None)
        ):
            raise ValueError(
                "external_varlen_prefill requires adapter-declared prefill row geometry"
            )
        self.external_varlen_prefill = external_varlen_prefill
        if (
            type(external_prefill_coalesce_ms) is not int
            or not 0 <= external_prefill_coalesce_ms <= 1000
        ):
            raise ValueError("external_prefill_coalesce_ms must be an integer from 0 to 1000")
        if external_prefill_coalesce_ms and not external_varlen_prefill:
            raise ValueError(
                "external_prefill_coalesce_ms requires external_varlen_prefill"
            )
        if (
            type(external_prefill_coalesce_min_tokens) is not int
            or external_prefill_coalesce_min_tokens < 1
        ):
            raise ValueError(
                "external_prefill_coalesce_min_tokens must be a positive integer"
            )
        self.external_prefill_coalesce_ms = external_prefill_coalesce_ms
        self.external_prefill_coalesce_min_tokens = (
            external_prefill_coalesce_min_tokens
        )
        if pairwise_selection not in ("host", "batched"):
            raise ValueError("pairwise_selection must be 'host' or 'batched'")
        self.pairwise_selection = pairwise_selection
        if type(dynamic_singleton_tree) is not bool:
            raise ValueError("dynamic_singleton_tree must be boolean")
        if type(dynamic_tree_max_width) is not int or dynamic_tree_max_width not in (1, 4):
            raise ValueError("dynamic_tree_max_width must be 1 or the bounded candidate width 4")
        if dynamic_tree_max_width != 1 and not dynamic_singleton_tree:
            raise ValueError("bounded dynamic tree requires an enabled tree route")
        self.dynamic_singleton_tree = dynamic_singleton_tree
        self.dynamic_tree_max_width = dynamic_tree_max_width
        if tree_node_budget_by_lanes is None:
            tree_node_budget_by_lanes = {
                width: 15 for width in range(1, dynamic_tree_max_width + 1)
            }
        if not isinstance(tree_node_budget_by_lanes, dict):
            raise ValueError("tree_node_budget_by_lanes must be a mapping")
        normalized_tree_budgets = {}
        for width, nodes in tree_node_budget_by_lanes.items():
            try:
                width = int(width)
            except (TypeError, ValueError) as exc:
                raise ValueError("tree node budget widths must be integers") from exc
            if type(nodes) is not int:
                raise ValueError("tree node budgets must be integers from 1 to 31")
            normalized_tree_budgets[width] = nodes
        tree_node_budget_by_lanes = normalized_tree_budgets
        expected_widths = set(range(1, dynamic_tree_max_width + 1))
        if set(tree_node_budget_by_lanes) != expected_widths:
            raise ValueError(
                "tree_node_budget_by_lanes must declare every enabled tree width"
            )
        # Up to two 16-row tiles per round; a stream holds 32 rows.
        if any(not 1 <= nodes <= 31 for nodes in tree_node_budget_by_lanes.values()):
            raise ValueError("tree node budgets must be integers from 1 to 31")
        ordered_budgets = [
            tree_node_budget_by_lanes[width]
            for width in range(1, dynamic_tree_max_width + 1)
        ]
        if any(
            left < right
            for left, right in zip(ordered_budgets, ordered_budgets[1:])
        ):
            raise ValueError("tree node budgets must not increase as lanes fill")
        if not dynamic_singleton_tree and ordered_budgets != [15]:
            raise ValueError("tree node budgets require an enabled tree route")
        self.tree_node_budget_by_lanes = tree_node_budget_by_lanes
        # Measured tile size within the knob maximum (default on for the
        # tree route; ``ast_adaptive_rows: false`` keeps the knob budget).
        self.tile_budget = None
        if dynamic_singleton_tree and self._ast_adaptive_rows is not False:
            from .ast_tile_budget import TileBudgetController

            self.tile_budget = TileBudgetController()
        # Per-depth drafter calibration (Kapitsa KfBudget): orders, and with
        # "order_size" sizes, each lattice's tree from learned acceptance.
        # Proposal-only; the verification law is unchanged.
        self.tree_calibration = None
        if self._ast_tree_calibration_mode is not None:
            if not dynamic_singleton_tree or not getattr(
                draft_model, "supports_tree_calibration", False
            ):
                raise ValueError(
                    "ast_tree_calibration requires the tree route and a drafter "
                    "that reports calibrated trees (DFlash2)"
                )
            if self._ast_tree_calibration_mode == "order_size" and self.tile_budget is None:
                raise ValueError(
                    "ast_tree_calibration order_size needs measured tile costs "
                    "(ast_adaptive_rows must stay on)"
                )
            from .ast_tree_calibration import DepthCalibration

            self.tree_calibration = DepthCalibration(self._ast_tree_calibration_mode)
        # Rows per request in batched tree rounds (Kapitsa #207): a cohort's
        # tile rows go to its lanes' nodes in decreasing calibrated survival
        # instead of an equal split; lanes then verify ragged trees.
        if type(ast_batch_row_sharing) is not bool:
            raise ValueError("ast_batch_row_sharing must be a boolean")
        if ast_batch_row_sharing and (
            self.tree_calibration is None
            or not getattr(target_tree_executor_provider, "supports_ragged_cohort", False)
        ):
            raise ValueError(
                "ast_batch_row_sharing requires ast_tree_calibration and a target-tree "
                "provider that verifies ragged cohorts"
            )
        self.ast_batch_row_sharing = ast_batch_row_sharing
        self._auto_last_mode = None
        self._auto_active_width = 0
        self._auto_cohort_width = 0
        self.draft_topology = os.environ.get(
            "MLX2_DFLASH_TOPOLOGY", "tree15" if dynamic_singleton_tree else "chain"
        )
        self.target_execution = _target_tree_execution_mode(
            default_provider=dynamic_singleton_tree
        )
        if dynamic_singleton_tree and (
            self.draft_topology != "tree15" or self.target_execution != "provider"
        ):
            raise ValueError("dynamic singleton tree requires tree15 and a target-tree provider")
        if dynamic_singleton_tree and not explicit_cost_handoff:
            # A tree tile that commits fewer tokens per second than plain
            # decode (an MoE target on prose) parks its session on the
            # ordinary path and re-probes; the same reversible controller
            # the AST planner installs.  ``cost_per_commit_handoff: false``
            # keeps the tree unconditionally.
            from dataclasses import replace as _replace

            # The first tree round also ingests the prompt into the draft
            # plane: settle it, and decide on two speculative samples.  A
            # losing re-probe costs a catch-up round and two tree rounds
            # (about 33 ms each on Qwen3.6-35B-A3B chat, against a 5.4 ms
            # plain step), so failed re-probes back off to 256 parked tokens
            # rather than 64; the first re-probe still comes after 16.
            self.cost_per_commit_handoff = _replace(
                CostPerCommitHandoffSettings.ast_self_configuring_defaults(),
                speculative_samples=2, settle_rounds=1, reprobe_max_tokens=256,
            )
            self.cost_per_commit_handoff_authority = "runtime_tree_tile_default"
        if self.draft_topology not in ("chain", "tree15"):
            raise ValueError("MLX2_DFLASH_TOPOLOGY must be chain or tree15")
        if hasattr(draft_model, "last_proposal_sources") and self.draft_topology != "chain":
            raise ValueError("proposal composition requires chain verification")
        if self.continuation_policy is not None and (self.target_execution != "reference" or self.fly_verification.enabled or self.pairwise_selection != "host"):
            raise ValueError("complete continuation verification requires exact reference target execution and host proposals")
        # The reference tree remains one lane per round. Providers may cohort
        # independent B1 records under their declared ownership bound.
        if self.target_execution not in ("reference", "provider"):
            raise ValueError(
                "target-tree execution must be reference or provider"
            )
        if self.target_execution == "provider":
            from .target_tree_executor import TargetTreeExecutorProvider

            if target_tree_executor_provider is None:
                raise ValueError(
                    "target-tree execution requires an adapter-owned provider"
                )
            if not isinstance(
                target_tree_executor_provider, TargetTreeExecutorProvider
            ):
                raise TypeError("incomplete target-tree executor provider")
            target_tree_executor_provider.validate_cohort(1, 1)
        elif target_tree_executor_provider is not None:
            raise ValueError(
                "target-tree executor provider requires provider execution"
            )
        self.target_tree_executor_provider = target_tree_executor_provider
        self.tree_gates = _tree_gates(
            self.draft_topology, self.target_execution,
            default_on=dynamic_singleton_tree,
        )
        from .acceptance_estimator import (
            AdaptiveVerificationPolicy,
            OnlineAcceptanceEstimator,
        )
        self.adaptive_policy = AdaptiveVerificationPolicy.from_value(
            adaptive_verification, self.num_draft
        )
        if (self.adaptive_policy is not None and self.adaptive_policy.continuation_costs
                and self.continuation_policy is None):
            raise ValueError("continuation costs require a bound complete continuation route")
        self.acceptance_estimator = None
        if self.adaptive_policy is not None:
            if self.draft_topology != "chain" or self.fly_verification.enabled:
                raise ValueError("adaptive verification requires exact chain verification")
            self.acceptance_estimator = OnlineAcceptanceEstimator(
                self.num_draft, refit_interval=self.adaptive_policy.refit_interval
            )
        # Exact chain verification rule.  "token" (default) is the token-wise
        # residual rule; "block" is block verification (Sun et al. 2024): the
        # same target law, a never-shorter expected accepted prefix for
        # stochastic drafts, and its own RNG schedule.  Block decisions are
        # defined for one linear chain under the exact law only.
        if exact_verification not in ("token", "block"):
            raise ValueError("exact_verification must be 'token' or 'block'")
        if exact_verification == "block" and (
            self.fly_verification.enabled
            or self.draft_topology != "chain"
            or dynamic_singleton_tree
            or self.continuation_policy is not None
            or self.adaptive_policy is not None
        ):
            raise ValueError(
                "block verification requires exact linear chain verification: "
                "it refuses FLy, tree topologies, the continuation pool and "
                "adaptive verification"
            )
        self.exact_verification = exact_verification
        monolithic_progressive = (
            progressive_verification_tile is not None
            and progressive_verification_tile == self.num_draft
        )
        if progressive_verification_tile is not None and (
            type(progressive_verification_tile) is not int
            or not 1 <= progressive_verification_tile <= self.num_draft
            or monolithic_progressive
            and progressive_verification_tile_policy != "auto"
        ):
            raise ValueError(
                "progressive_verification_tile must be an integer from 1 below "
                "num_draft, except an admitted automatic terminal tile"
            )
        if progressive_verification_tile_policy not in (None, "auto"):
            raise ValueError("progressive_verification_tile_policy must be auto")
        if (
            progressive_verification_tile_policy is not None
            and progressive_verification_tile is None
        ):
            raise ValueError(
                "progressive_verification_tile_policy requires a resolved tile"
            )
        if progressive_verification_tile_policy == "auto":
            from ..adapters.external_draft_policy import (
                progressive_external_terminal_input_widths,
            )

            declared_terminal_widths = progressive_external_terminal_input_widths(
                model,
                max_lanes=completion_batch_size,
                num_draft=self.num_draft,
                fidelity=progressive_verification_fidelity or "exact",
            )
            if (
                not isinstance(
                    progressive_terminal_input_widths, (tuple, list, range)
                )
                or not progressive_terminal_input_widths
                or any(
                    type(width) is not int or width < 2
                    for width in progressive_terminal_input_widths
                )
            ):
                raise ValueError(
                    "automatic progressive verification requires declared "
                    "terminal input widths"
                )
            if (
                not isinstance(declared_terminal_widths, (tuple, list, range))
                or not declared_terminal_widths
                or any(
                    type(width) is not int or width < 2
                    for width in declared_terminal_widths
                )
            ):
                raise ValueError(
                    "automatic progressive verification target has no "
                    "terminal input width capability"
                )
            required_terminal_widths = range(
                2, int(progressive_verification_tile) + 2
            )
            if any(
                width not in progressive_terminal_input_widths
                or width not in declared_terminal_widths
                for width in required_terminal_widths
            ):
                raise ValueError(
                    "automatic progressive verification terminal input width "
                    + (
                        "is not declared exact"
                        if progressive_verification_fidelity in (None, "exact")
                        else "is not declared for the selected fidelity"
                    )
                )
        if monolithic_progressive and (
            progressive_alternative_branches != 1
            or continuation_pool is not None
            or progressive_cross_session_pack_rows is not None
            or progressive_cross_session_pack_mode is not None
        ):
            raise ValueError(
                "automatic terminal K-wide verification requires a simple "
                "primary chain"
            )
        if progressive_verification_tile is None and any(
            value is not None
            for value in (
                progressive_verification_fidelity,
                progressive_target_protocol,
                progressive_protocol_binding,
            )
        ):
            raise ValueError(
                "progressive protocol selectors require progressive_verification_tile"
            )
        if progressive_verification_tile is not None and (
            self.target_execution != "reference"
            or self.draft_topology != "chain"
            or self.pairwise_selection != "host"
            or self.fly_verification.enabled
            or self.exact_verification != "token"
            or (
                self.continuation_policy is not None
                and not getattr(self, "continuation_pool_ast_bridge", False)
            )
            or self.adaptive_policy is not None
            or dynamic_singleton_tree
            or (
                hasattr(draft_model, "last_proposal_sources")
                and not getattr(self, "continuation_pool_ast_bridge", False)
            )
        ):
            raise ValueError(
                "progressive verification requires an exact reference chain "
                "with host proposals and no FLy, tree, continuation, composition, "
                "or adaptive route"
            )
        self.ast_packed_pattern = getattr(self.continuation_policy, "packed_pattern", None)
        if (
            type(progressive_alternative_branches) is not int
            or not 1 <= progressive_alternative_branches <= (5 if self.ast_packed_pattern else 4)
        ):
            raise ValueError(
                "progressive_alternative_branches must be an integer from 1 to 4"
            )
        if progressive_cross_session_pack_rows is not None and (
            progressive_cross_session_pack_rows not in {4, 8}
            or progressive_alternative_strategy != "confidence_tree"
            or progressive_alternative_branches <= 1
            or not 2 <= completion_batch_size <= 4
        ):
            raise ValueError(
                "progressive_cross_session_pack_rows must be 4 or 8 and requires "
                "two to four confidence-tree sessions"
            )
        if progressive_cross_session_pack_mode not in {None, "ordered_live_frontier"}:
            raise ValueError(
                "progressive_cross_session_pack_mode must be ordered_live_frontier"
            )
        if (
            progressive_cross_session_pack_mode == "ordered_live_frontier"
            and progressive_cross_session_pack_rows is None
        ):
            raise ValueError(
                "ordered_live_frontier requires progressive cross-session packing"
            )
        if progressive_alternative_branches > 1 and (
            progressive_verification_tile is None
            or (
                completion_batch_size != 1
                and progressive_cross_session_pack_rows is None
                and not getattr(ast_target_tree_executor_provider, "ast_tree_tile_rows", 0)
            )
            or getattr(draft_model, "proposal_distribution", None)
            != "deterministic_point_mass"
        ):
            raise ValueError(
                "ordered alternative branches require B1 or explicit cross-session "
                "packing, exact progressive verification, and deterministic "
                "point-mass proposals"
            )
        validate_ordered_alternative_strategy(
            progressive_alternative_strategy, progressive_alternative_branches
        )
        if progressive_alternative_depth_policy not in {"full", "confidence_tail"}:
            raise ValueError("unsupported progressive alternative depth policy")
        if progressive_alternative_depth_policy == "confidence_tail" and (
            progressive_alternative_strategy != "confidence_tree"
            or progressive_alternative_branches < 3
        ):
            raise ValueError(
                "confidence_tail requires confidence_tree with at least three branches"
            )
        if type(progressive_ast_rootless_fallback) is not bool:
            raise ValueError("progressive_ast_rootless_fallback must be boolean")
        if not progressive_ast_rootless_fallback and (
            not getattr(self, "continuation_pool_ast_bridge", False)
            or progressive_alternative_strategy != "confidence_tree"
            or progressive_alternative_branches <= 1
        ):
            raise ValueError(
                "progressive_ast_rootless_fallback=false requires the exact "
                "continuation-pool confidence-tree bridge"
            )
        if progressive_alternative_branches > 1 and not callable(
            getattr(draft_model, "ordered_alternative_proposals", None)
        ):
            raise ValueError(
                "ordered alternative branches require a capable proposal adapter"
            )
        primary_near_tie_ast = (
            progressive_verification_fidelity == PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED
            and type(progressive_verification_tile) is int
            and progressive_verification_tile_policy in (None, "auto")
            and progressive_alternative_branches == 1
            and self.continuation_policy is not None
            and self.continuation_policy.ast_planner
            and self.continuation_policy.limit == 1
            and self.continuation_policy.sources == ("external",)
            and ast_target_tree_executor_provider is None
        )
        self.primary_near_tie_ast = bool(
            self.continuation_pool_ast_bridge and primary_near_tie_ast
        )
        primary_exact_ast = (
            progressive_verification_fidelity in (None, PROGRESSIVE_FIDELITY_EXACT)
            and progressive_alternative_branches == 1
            and self.continuation_policy is not None
            and self.continuation_policy.ast_planner
            and self.continuation_policy.limit == 1
            and self.continuation_policy.sources == ("external",)
        )
        if getattr(self, "continuation_pool_ast_bridge", False) and (
            progressive_verification_tile is None
            or (
                not (primary_near_tie_ast or primary_exact_ast)
                and (
                    progressive_alternative_branches <= 1
                    or progressive_verification_fidelity
                    not in (None, PROGRESSIVE_FIDELITY_EXACT)
                )
            )
            or getattr(draft_model, "proposal_distribution", None)
            != "deterministic_point_mass"
            or not callable(
                getattr(draft_model, "ordered_alternative_proposals", None)
            )
        ):
            raise ValueError(
                "continuation_pool_ast_bridge requires exact ordered alternatives "
                "or explicit primary-only near-tie AST from a deterministic "
                "point-mass continuation wrapper"
            )
        self.progressive_verification_fidelity = None
        self.progressive_target_protocol = None
        self.progressive_protocol_binding = None
        if progressive_verification_tile is not None:
            fidelity = (
                PROGRESSIVE_FIDELITY_EXACT
                if progressive_verification_fidelity is None
                else progressive_verification_fidelity
            )
            resolved = progressive_target_protocol_binding(model, fidelity)
            if progressive_target_protocol is not None and (
                progressive_target_protocol != resolved["target_protocol"]
            ):
                raise ValueError("progressive target protocol binding mismatch")
            if fidelity == PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED:
                if progressive_target_protocol != (
                    PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION
                ):
                    raise ValueError(
                        "near-tie-bounded progressive verification requires its "
                        "explicit target protocol"
                    )
                if progressive_protocol_binding != resolved:
                    raise ValueError(
                        "near-tie-bounded progressive verification requires an "
                        "exact source-hashed protocol binding"
                    )
            elif progressive_protocol_binding is not None and (
                progressive_protocol_binding != resolved
            ):
                raise ValueError("progressive protocol source binding mismatch")
            self.progressive_verification_fidelity = fidelity
            self.progressive_target_protocol = resolved["target_protocol"]
            self.progressive_protocol_binding = resolved
        self.progressive_verification_tile = progressive_verification_tile
        if self.ast_physical_target and (
            self.progressive_verification_tile is None
            or self.progressive_verification_fidelity
            != PROGRESSIVE_FIDELITY_EXACT
            or self.progressive_target_protocol
            != PROGRESSIVE_TARGET_PROTOCOL_VERSION
            or not callable(
                getattr(self.model, "progressive_external_target_rows", None)
            )
        ):
            raise ValueError(
                "ast_physical_target requires selected exact progressive "
                "verification and an adapter-owned row-exact target"
            )
        self.progressive_terminal_input_widths = (
            tuple(progressive_terminal_input_widths)
            if progressive_terminal_input_widths is not None
            else None
        )
        self.progressive_verification_tile_policy = (
            progressive_verification_tile_policy
        )
        self.progressive_alternative_branches = progressive_alternative_branches
        self.progressive_alternative_strategy = progressive_alternative_strategy
        self.progressive_alternative_depth_policy = (
            progressive_alternative_depth_policy
        )
        self.progressive_ast_rootless_fallback = progressive_ast_rootless_fallback
        self.progressive_cross_session_pack_rows = (
            progressive_cross_session_pack_rows
        )
        self.progressive_cross_session_pack_mode = progressive_cross_session_pack_mode
        self.progressive_topology_latch = (
            PerSessionSpeculationTopology(progressive_alternative_branches)
            if progressive_alternative_strategy == "confidence_tree"
            and progressive_alternative_branches > 1
            and self.ast_packed_pattern is None
            else None
        )
        self.ast_execution_planner = None
        if self.continuation_pool_ast_bridge and self.continuation_policy.ast_planner:
            from ..adapters.ast_execution_profile import build_execution_profile
            from .ast_planner import ASTExecutionPlanner

            profile = build_execution_profile(
                model, draft_model, binding=str(binding),
                tile_cap=progressive_verification_tile,
                max_lanes=progressive_cross_session_pack_rows or 4,
                num_draft=self.num_draft,
                alternative_provider=ast_target_tree_executor_provider,
                fidelity=self.progressive_verification_fidelity,
                automatic=progressive_verification_tile_policy == "auto",
            )
            self.ast_execution_planner = ASTExecutionPlanner(
                profile, tile_cap=progressive_verification_tile,
                fixed_tile=progressive_verification_tile_policy != "auto",
                source_limit=self.continuation_policy.limit,
                minimum_depth=self.minimum_draft_proposals,
                cohort_sources=True,
                ordered_branches=(
                    progressive_cross_session_pack_mode == "ordered_live_frontier"
                ),
            )
            if not explicit_cost_handoff:
                # Measured AST economics do not depend on selecting a
                # specialized physical target. The padded exact verifier
                # needs the same reversible ordinary-path control.
                self.ast_cost_self_configure = True
                self.cost_per_commit_handoff = (
                    CostPerCommitHandoffSettings.ast_self_configuring_defaults()
                )
                self.cost_per_commit_handoff_authority = (
                    "runtime_ast_self_configuring_default"
                )
        elif ast_target_tree_executor_provider is not None:
            raise ValueError(
                "AST target-tree backend requires an adapter-bound AST planner"
            )
        self.ast_target_tree_executor_provider = ast_target_tree_executor_provider
        self.ast_tree_tile_rows = getattr(
            ast_target_tree_executor_provider, "ast_tree_tile_rows", 0
        )
        if self.ast_tree_tile_rows and (
            type(self.ast_tree_tile_rows) is not int
            or not 2 <= self.ast_tree_tile_rows <= 16
            or not self.continuation_pool_ast_bridge
            or self.progressive_alternative_strategy != "confidence_tree"
        ):
            raise ValueError("AST tree tiles require private continuation ownership")
        self._ast_tree_tuners = {}
        if self.ast_packed_pattern is not None and (
            completion_batch_size != 1 or self.num_draft != max(self.ast_packed_pattern)
            or self.progressive_alternative_branches != len(self.ast_packed_pattern)
            or self.ast_tree_tile_rows != 16 or not self.continuation_pool_ast_bridge
            or cost_per_commit_handoff is not False
        ):
            raise ValueError("packed_pattern requires a B1 sixteen-row AST provider and fixed source geometry")
        if ast_b1_tile_policy is not None and (
            ast_b1_tile_policy != "four_path15"
            or completion_batch_size != 1 or self.num_draft != 15
            or self.ast_tree_tile_rows != 16
            or self.progressive_alternative_branches != 4
            or not getattr(self.draft, "supports_continuation_tree_reuse", False)
            or not getattr(self.draft, "requires_proposal_contexts", False)
            or getattr(self.draft, "fold_policy", None) is not None
            or cost_per_commit_handoff is not False
        ):
            raise ValueError("four_path15 requires B1 K15/four-path TensorFold and cost handoff off")
        self.ast_b1_tile_policy = ast_b1_tile_policy
        self._ast_tree_ready_sessions = 1
        if type(ast_collision_probability) is not bool:
            raise ValueError("ast_collision_probability must be boolean")
        if ast_collision_probability and self.ast_execution_planner is None:
            raise ValueError(
                "ast_collision_probability requires an adapter-bound AST planner"
            )
        self.ast_collision_probability = ast_collision_probability
        if multilane_draft_cap is not None and (
            type(multilane_draft_cap) is not int
            or not self.minimum_draft_proposals
            <= multilane_draft_cap
            <= self.num_draft
        ):
            raise ValueError(
                "multilane_draft_cap must be an integer from "
                "minimum_draft_proposals to num_draft"
            )
        self.multilane_draft_cap = multilane_draft_cap
        # AST owns a live speculative cohort, not one token of each lane's
        # already-committed block per poll. Draining those blocks piecemeal
        # excluded successful lanes from the next draft/verify batch.
        # Preserve explicit choices and non-AST behavior.
        ast_batching = self.progressive_verification_tile is not None
        if ready_drain is None:
            ready_drain = "all" if ast_batching else "one"
        if ready_drain not in ("one", "all"):
            raise ValueError("ready_drain must be 'one' or 'all'")
        self.ready_drain = ready_drain
        # A cohort's lanes carry different pending-context tails (each
        # accepted a different number of tokens), and one drafter call needs
        # one tail length.  With this set, the longer tails' leading part is
        # committed to each lane's own draft plane first (the depth-zero
        # append), so every lane enters one block call with the shortest
        # tail.  Draft-plane only: target rows and verification are unchanged.
        if draft_tail_equalize is None:
            draft_tail_equalize = bool(
                ast_batching and not getattr(self.draft, "requires_context_tokens", False)
                and callable(getattr(self.draft, "append_context", None))
            )
        if type(draft_tail_equalize) is not bool:
            raise ValueError("draft_tail_equalize must be a boolean")
        self.draft_tail_equalize = draft_tail_equalize
        self.speculative_batching = {
            "ready_drain": ready_drain,
            "draft_tail_equalize": draft_tail_equalize,
            "scope": "live_ast_cohort" if ast_batching else "legacy_cohort",
        }
        if any(len(t) != 1 for t in stop_tokens):
            raise ValueError("External executor currently requires single-token stops")
        self.stops = {int(t[0]) for t in stop_tokens}
        self.layers = tuple(draft_model.config.target_layer_ids)
        self.lanes = {}; self.next_uid = 0; self.boundaries = {}
        # A parked cohort's dispatched, unread round (pipelined parked decode).
        self._parked_inflight = None
        self._parked_last_read_ns = None
        # The multi-lane plain step's hybrid owner (``_parked_target_owner``).
        self._parked_owner = None
        # Cohort widths at which every session parked: the tree lost to plain
        # decode there (``_record_width_park_prior``).
        self._width_park_priors = {}
        # A permanent B1 ordinary lane can use the target model's native
        # one-token pipeline after a one-way handoff.  Keep this separate from
        # parked speculation: no draft plane or speculative transaction owns
        # the state after the handoff.
        self._ordinary_inflight = None
        self._lane_failures = []
        # Client-declared cohorts own every member until their first complete
        # decode frontier.  Cancellation/removal before that boundary fails
        # the surviving owners together through ServingEngine's existing
        # atomic-failure channel.
        self._atomic_cohort_failures = []
        self._released_atomic_cohorts = set()
        self._memory_waiting_uids = set()
        self._decode_fit_waiting_uids = set()
        self._atomic_prefill_waiting_uids = set()
        self._decode_service_epoch = 0
        self.scheduler_stats = {"external_rounds": 0, "accepted_proposals": 0, "proposed_tokens": 0, "ordinary_rounds": 0, "cancelled": 0, "target_max_width": 0, "draft_max_width": 1, "prefill_rounds": 0, "paired_cache_resumes": 0, "segmented_transactions": 0, "segmented_rollbacks": 0, "draft_fallbacks": 0, "recovery_checkpoint_captures": 0, "recovery_checkpoint_restores": 0, "external_draft_masked_positions": 0, "external_ordinary_fast_path_rounds": 0, "external_ordinary_fast_path_lanes": 0, "external_draft_context_skipped": 0, "external_taps_skipped": 0, "external_transactions_skipped": 0, "fly_relaxed_accepts": 0, "external_context_token_pairings": 0, "external_verify_steer_rounds": 0, "external_verify_steered_lanes": 0, "external_allocator_reclaims": 0, "external_idle_allocator_reclaims": 0, "external_atomic_cohort_prefill_holds": 0, "external_atomic_cohort_releases": 0, "external_atomic_cohort_failures": 0}
        self.scheduler_stats.update(
            external_minimum_draft_proposals=self.minimum_draft_proposals,
            external_proposal_floor_raises=0,
        )
        if getattr(self, "dynamic_singleton_tree", False):
            # The tree tile's row mode and proposal knobs, published from the
            # start (plain numbers and strings, for metrics).  The adaptive
            # controller's per-round choices are in external_tree_tile_budget.
            calibration = getattr(self, "tree_calibration", None)
            self.scheduler_stats["external_tree_tile_mode"] = {
                "mode": "adaptive" if getattr(self, "tile_budget", None) else "fixed",
                "tile_rows": int(self.tree_node_budget_by_lanes.get(1, 15)) + 1,
                "maximum_nodes_by_lanes": {
                    str(lanes): int(nodes)
                    for lanes, nodes in sorted(self.tree_node_budget_by_lanes.items())
                },
                "calibration": None if calibration is None else calibration.mode,
                "batch_row_sharing": bool(getattr(self, "ast_batch_row_sharing", False)),
            }
            if calibration is not None:
                self.scheduler_stats["external_tree_calibration"] = calibration.receipt()
            if getattr(self, "ast_batch_row_sharing", False):
                self.scheduler_stats["external_tree_row_sharing_rounds"] = 0
                self.scheduler_stats["external_tree_row_sharing_lane_nodes"] = {}
        if self.progressive_verification_tile is not None:
            self.scheduler_stats.update(
                external_progressive_verify_rounds=0,
                external_progressive_verify_launches=0,
                external_progressive_verify_target_rows=0,
                external_progressive_verify_rejections=0,
                external_progressive_verify_full_tiles=0,
                external_progressive_verify_tile_checkpoints=0,
                external_progressive_verify_physical_launches=0,
                external_progressive_verify_ragged_launches=0,
                external_progressive_verify_padding_rows=0,
                external_progressive_target_executor_adapter_owned_launches=0,
                external_progressive_target_executor_legacy_launches=0,
                external_progressive_target_executor_row_exact_launches=0,
                external_progressive_target_executor_nonexact_launches=0,
                external_progressive_verify_batched_rounds=0,
                external_progressive_verify_batched_lanes=0,
                external_progressive_verify_b1_aligned_tile_launches=0,
                external_progressive_verify_b1_aligned_continued_launches=0,
                external_progressive_verify_b2_aligned_tile_launches=0,
                external_progressive_verify_b2_aligned_continued_launches=0,
                external_progressive_verify_b3_aligned_tile_launches=0,
                external_progressive_verify_b3_aligned_continued_launches=0,
                external_progressive_verify_b4_aligned_tile_launches=0,
                external_progressive_verify_b4_aligned_continued_launches=0,
                external_progressive_verify_b4_disjoint_first_tiles=0,
                external_progressive_verify_max_cohort_width=0,
                external_progressive_alternative_rounds=0,
                external_progressive_alternative_sampled_primary_rounds=0,
                external_progressive_alternative_stop_guard_rounds=0,
                external_progressive_ast_geometry_guard_rounds=0,
                external_progressive_alternative_branches=0,
                external_progressive_alternative_full_accepts=0,
                external_progressive_alternative_b2_short_circuits=0,
                external_progressive_alternative_shared_prefix_rounds=0,
                external_progressive_alternative_shared_prefix_tokens=0,
                external_progressive_alternative_unique_target_rows=0,
                external_progressive_cross_session_ast_rounds=0,
                external_progressive_cross_session_ast_sessions=0,
                external_progressive_cross_session_ast_packs=0,
                external_progressive_cross_session_ast_b8_launches=0,
                external_progressive_apt_frontier_rows_admitted=0,
                external_progressive_apt_frontier_sessions_retired=0,
                external_progressive_apt_frontier_branches_unvisited=0,
                external_progressive_apt_frontier_proposal_tokens_not_scheduled=0,
                external_progressive_apt_frontier_refills=0,
                external_progressive_apt_frontier_occupancy_numerator=0,
                external_progressive_apt_frontier_occupancy_denominator=0,
                external_progressive_apt_frontier_partial_packs=0,
                external_progressive_apt_frontier_full_packs=0,
                external_progressive_apt_frontier_max_service_gap=0,
                deprecated_external_progressive_apt_frontier_alias_for=(
                    "external_progressive_ast_frontier_*"
                ),
                external_progressive_ast_frontier_rows_admitted=0,
                external_progressive_ast_frontier_sessions_retired=0,
                external_progressive_ast_frontier_branches_unvisited=0,
                external_progressive_ast_frontier_proposal_tokens_not_scheduled=0,
                external_progressive_ast_frontier_refills=0,
                external_progressive_ast_frontier_occupancy_numerator=0,
                external_progressive_ast_frontier_occupancy_denominator=0,
                external_progressive_ast_frontier_partial_packs=0,
                external_progressive_ast_frontier_full_packs=0,
                external_progressive_ast_frontier_max_service_gap=0,
                external_progressive_ast_primary_fallback_insufficient_branches_rounds=0,
                external_progressive_ast_primary_fallback_depth_not_above_tile_rounds=0,
                external_progressive_ast_primary_fallback_rootless_refused_rounds=0,
                external_progressive_ast_branch_downshift_rounds=0,
                external_progressive_ast_branch_downshift_lanes=0,
                external_progressive_ast_rootless_rounds=0,
                external_progressive_ast_rootless_lanes=0,
                external_progressive_ast_rootless_refusal_rounds=0,
                external_progressive_ast_rootless_refusal_sessions=0,
                external_progressive_ast_adaptive_prefix_rounds=0,
                external_progressive_ast_adaptive_prefix_lanes=0,
                external_progressive_ast_utility_downshift_rounds=0,
                external_progressive_ast_utility_downshift_sessions=0,
                external_progressive_ast_primary_only_rounds=0,
                external_progressive_ast_primary_only_sessions=0,
                external_progressive_topology_observation_failures=0,
                external_progressive_verify_tile=self.progressive_verification_tile,
                external_multilane_draft_cap=self.multilane_draft_cap,
                external_multilane_draft_cap_rounds=0,
                external_multilane_draft_cap_lanes=0,
            )
        if getattr(self, "external_varlen_prefill", False):
            self.scheduler_stats.update(
                external_batched_prefill_rounds=0,
                external_batched_prefill_lanes=0,
                external_batched_prefill_max_cohort_width=0,
                external_batched_prefill_max_token_width=0,
                external_batched_prefill_padding_rows=0,
                external_prefill_coalesce_deferrals=0,
                external_prefill_coalesce_expirations=0,
            )
        if dynamic_singleton_tree:
            self.scheduler_stats.update(
                external_auto_tree_rounds=0,
                external_auto_chain_rounds=0,
                external_auto_mode_switches=0,
                external_auto_tree_max_width=0,
                external_auto_chain_max_width=0,
            )
        if self.adaptive_policy is not None:
            self.scheduler_stats.update(
                external_adaptive_rounds=0,
                external_adaptive_trimmed_rounds=0,
                external_adaptive_trimmed_target_rows=0,
                external_adaptive_target_rows=0,
                external_adaptive_round_depth=0,
                external_adaptive_verify_width=0,
                external_adaptive_verification_groups=0,
                external_adaptive_per_request_rounds=0,
                external_adaptive_current_confidence_rounds=0,
            )
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
        from .adaptive_policy import DecodeTimeFairness

        # The serving route's decode-time fairness (the stall bound on a
        # prefill slice taken beside decoding lanes).  Absent, as for a
        # direct construction, it is constructed disabled and inert.
        self.decode_time_fairness = DecodeTimeFairness(
            **dict(decode_time_fairness or {})
        )
        self._sync_decode_fairness_stats()
        if pairwise_selection == "batched":
            # Default-off receipts keep their existing key set.
            self.scheduler_stats.update(external_pairwise_selection_groups=0, external_pairwise_selection_lanes=0)
        if self.draft_topology == "tree15":
            self.scheduler_stats.update(
                external_tree_rounds=0,
                external_tree_nodes=0,
                external_tree_accepted_edges=0,
                external_tree_node_budget_last=0,
                external_tree_node_budget_histogram={},
            )
        if self.target_execution == "provider":
            self.target_tree_selected = True
            self.target_tree_cohort_limit = _target_tree_cohort_limit(
                self.target_execution, self.capacity,
                default=self.dynamic_tree_max_width if dynamic_singleton_tree else 1,
                selected=tensorfold_cohort_limit,
                maximum=self.target_tree_executor_provider.maximum_cohort_width,
            )
            self.scheduler_stats.update(
                external_target_tree_target_rounds=0,
                external_target_tree_cohort_rounds=0,
                external_target_tree_cohort_lanes=0,
                external_target_tree_cohort_max_width=0,
                external_target_tree_cohort_limit=self.target_tree_cohort_limit,
                external_target_tree_packed_target_rounds=0,
                external_target_tree_packed_target_lanes=0,
                external_target_tree_physical_target_forwards=0,
            )
            for width in range(1, self.target_tree_cohort_limit + 1):
                self.scheduler_stats[
                    f"external_target_tree_target_width_{width}_rounds"
                ] = 0
        else:
            self.target_tree_selected = False
            self.target_tree_cohort_limit = _target_tree_cohort_limit(
                self.target_execution, self.capacity
            )
        for gate, counters in _TREE_GATE_COUNTERS.items():
            if self.tree_gates[gate]:
                self.scheduler_stats.update(dict.fromkeys(counters, 0))
        self._tree_clock = None
        # uid -> next round's lattice queued after this round's commit
        # (MLX2_TREE_PIPELINE_DRAFT). Never part of a lane or its snapshot.
        self._prelaunched = {}
        # (lane, draft plane, its descriptor copy) taken during the tile wait.
        self._draft_presnapshot = None
        self._open = False
        # Host-time attribution for one external round.  Default off: a
        # perf_counter in this path is cheap but the block exists only for
        # tuning, and the bounded integer counters above stay always-on.
        self.round_timing = bool(os.environ.get("MLX2_EXTERNAL_ROUND_TIMING"))
        # Diagnostic-only: fence the MLX queue at every timing boundary so
        # queued device work is charged to the phase that submitted it.  This
        # deliberately serializes the measured route and is never default-on.
        self.round_timing_sync = self.round_timing and bool(
            os.environ.get("MLX2_EXTERNAL_ROUND_TIMING_SYNC")
        )
        self.round_times = defaultdict(float)

    @property
    def _target_args(self):
        # Hybrid targets keep text geometry under ``speculative_args``.
        return getattr(self.model, "speculative_args", self.model.args)

    def _cohort_target_forward(self, inputs, transaction, cohort, *, site, taps=True):
        """Target forward for one cohort transaction: lane-exact when owned.

        A batched hybrid transaction at B>1 otherwise runs the plain batched
        forward: multi-row SDPA and batched Mamba/MoE/projection kernels whose
        bits depend on the cohort's width and padding.  When the route is
        configured for exact progressive verification and the target exposes
        its adapter-owned row-exact executor (the same one progressive verify
        launches), every row runs with its own lane's one-token decode
        arithmetic instead, so a lane's stream no longer depends on which
        lanes shared its round.  ``site`` names the scheduler path in the
        counters and in each lane's receipt.  ``taps=False`` (a logits-only
        caller) keeps the plain logits-only model call whenever the lane-exact
        executor is not selected, and returns ``None`` features.

        Permanent ordinary cohorts may instead use an adapter's terminal
        stock-forward capability. That is ``ordinary_native``, not a
        row-exact launch; reversible parked/probe rounds cannot select it.
        Other paths are classified by the cohort's lane count, whatever the
        transaction type: ``lane_exact``, ``batched`` or ``single_lane``.
        """
        width = int(inputs.shape[1])
        batch = int(inputs.shape[0])
        if getattr(self, "ast_tree_tile_rows", 0) and width > 1:
            raise RuntimeError(
                "AST tree tiles require the admitted target-tree provider; "
                "ordinary multi-row verification is not a tile fallback"
            )
        executor = getattr(self.model, "progressive_external_target_rows", None)
        # A target may declare that its exact executor needs more than one
        # row per lane (one-token batched launches are not on its row law);
        # narrower cohort forwards then take the plain batched path.
        min_width = int(
            getattr(self.model, "progressive_external_lane_exact_min_width", 1)
        )
        from .progressive_external_verify import cohort_exact_lane_limit

        lane_exact = bool(
            executor is not None
            and self.progressive_verification_fidelity == PROGRESSIVE_FIDELITY_EXACT
            and getattr(transaction, "batched", False)
            and 2 <= batch <= cohort_exact_lane_limit(self.model)
            and max(1, min_width) <= width <= 16
        )
        ordinary_forward = getattr(self.model, "ordinary_external_target_forward", None)
        terminal_ordinary = bool(
            not taps
            and width == 1
            and cohort
            and all(lane.ordinary for lane in cohort)
            and getattr(transaction, "plain", False)
            and callable(ordinary_forward)
            and getattr(self.model, "ordinary_external_target_protocol", None)
            == "committed-stock-ordinary-terminal-v1"
        )
        if terminal_ordinary:
            # Irreversible handoff owns no future draft/verify transaction.
            # Use the adapter's genuine ordinary arithmetic, not the exact
            # speculative executor's more expensive B1-per-row law.
            logits = ordinary_forward(inputs, transaction.caches)
            features, detail, path = None, None, "ordinary_native"
        elif lane_exact:
            logits, features, detail = execute_progressive_target_rows(
                self.model,
                inputs,
                transaction.caches,
                self.layers,
                protocol=self.progressive_target_protocol,
                fidelity=PROGRESSIVE_FIDELITY_EXACT,
                input_lengths=transaction.lengths,
            )
            path = "lane_exact"
            if detail["row_exact"] is not True:
                raise RuntimeError("cohort target executor left its row-exact path")
        else:
            if taps:
                logits, features = self.model.forward_with_taps(
                    inputs, transaction.caches, self.layers
                )
            else:
                logits, features = self.model(inputs, cache=transaction.caches), None
            detail = None
            path = "batched" if batch >= 2 else "single_lane"
        _bump(self.scheduler_stats, f"external_cohort_target_{path}_{site}_rounds")
        for lane in cohort:
            counts = getattr(lane, "cohort_target_execution", None)
            if counts is None:
                counts = lane.cohort_target_execution = {}
            key = f"{site}:{path}"
            counts[key] = counts.get(key, 0) + 1
        return logits, features

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
        # A drafter whose feature is not a concatenation of hidden-size layer
        # taps (a native MTP head's multi-stream seed) declares its width.
        width = getattr(self.draft.config, "feature_width", None)
        if width is None:
            width = len(self.layers) * self._target_args.hidden_size
        return self.mx.zeros((1, 0, int(width)))

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

    def _equalize_draft_tails(self, cohort):
        """Bring every lane's draft tail to the cohort's shortest length.

        Commits the leading ``len(tail) - shortest`` features of each longer
        tail to that lane's own draft plane, exactly as a depth-zero round
        commits a whole tail, so one drafter call can cover the cohort.  A
        pairing drafter's context tokens are per position of the whole tail,
        so it keeps its per-length groups. Empty-tail coalescence additionally
        requires the drafter's explicit committed-context capability.
        """
        if self.pair_context_tokens:
            return
        dtypes = {str(lane.tail.dtype) for lane in cohort}
        lengths = [int(lane.tail.shape[1]) for lane in cohort]
        if len(dtypes) != 1 or len(set(lengths)) == 1:
            return
        shortest = min(lengths)
        if shortest == 0:
            _bump(self.scheduler_stats, "external_draft_zero_tail_mixed_rounds")
            if (
                self.progressive_verification_tile is None
                or getattr(self.draft, "supports_empty_target_context", False)
                is not True
            ):
                _bump(self.scheduler_stats, "external_draft_zero_tail_declined_rounds")
                return
            _bump(self.scheduler_stats, "external_draft_zero_tail_equalized_rounds")
        for lane, length in zip(cohort, lengths, strict=True):
            excess = length - shortest
            if not excess:
                continue
            self.draft.append_context(lane.tail[:, :excess], lane.draft_cache)
            lane.tail = lane.tail[:, excess:]
            _bump(self.scheduler_stats, "external_draft_tail_equalized_lanes")
            _bump(self.scheduler_stats, "external_draft_tail_equalized_positions", excess)
            if shortest == 0:
                _bump(self.scheduler_stats, "external_draft_zero_tail_equalized_lanes")
                _bump(self.scheduler_stats, "external_draft_zero_tail_equalized_positions", excess)

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
               apc_interior_positions=None, prefill_inputs=None,
               prefill_cuts=None, session_keys=None,
               ast_memo_layout_revisions=None):
        if self._open:
            raise RuntimeError("Membership change during external transaction")
        if session_keys is not None and (len(session_keys) != len(prompts) or any(
                value is not None and (type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value))
                for value in session_keys)):
            raise ValueError("session_keys must contain one pinned privacy hash per request")
        if ast_memo_layout_revisions is not None and (
            len(ast_memo_layout_revisions) != len(prompts)
            or any(
                type(value) is not str
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
                for value in ast_memo_layout_revisions
            )
        ):
            raise ValueError(
                "ast_memo_layout_revisions must contain one SHA-256 per request"
            )
        apc_interior_positions = (
            [()] * len(prompts)
            if apc_interior_positions is None
            else apc_interior_positions
        )
        prefill_inputs = (
            [None] * len(prompts) if prefill_inputs is None else prefill_inputs
        )
        prefill_cuts = (
            [None] * len(prompts) if prefill_cuts is None else prefill_cuts
        )
        if len(apc_interior_positions) != len(prompts):
            raise ValueError(
                "apc_interior_positions must have one entry per external request"
            )
        if len(prefill_inputs) != len(prompts):
            raise ValueError("prefill_inputs must have one entry per external request")
        if len(prefill_cuts) != len(prompts):
            raise ValueError("prefill_cuts must have one entry per external request")
        if any(positions for positions in apc_interior_positions):
            raise ValueError(
                "APCv2 interior checkpoints are unavailable on external draft routes"
            )
        if any(value is not None for value in prefill_inputs):
            raise ValueError("multimodal prefill is unavailable on external draft routes")
        for name, rows in (
            ("max_tokens", max_tokens), ("caches", caches), ("all_tokens", all_tokens),
            ("cache_states", cache_states), ("lane_rngs", lane_rngs),
            ("sampling_configs", sampling_configs), ("logits_processors", logits_processors),
        ):
            if rows is not None and len(rows) != len(prompts):
                raise ValueError(f"{name} must have one entry per external request")
        # Build the entire cohort privately. A later row can fail artifact,
        # sidecar, or allocation checks; publishing earlier rows before that
        # point strands live requests whose caller never received their UIDs.
        pending_lanes = []
        resumed = 0
        for i, prompt in enumerate(prompts):
            prefix = list((all_tokens or [[]]*len(prompts))[i]); todo = list(prompt)
            if not todo or max_tokens[i] <= 0:
                raise ValueError("External request needs prompt tokens and positive budget")
            state = (cache_states or [None]*len(prompts))[i]
            target = (caches or [None]*len(prompts))[i]
            seed_key = (lane_rngs or [None]*len(prompts))[i]
            uid = self.next_uid + len(pending_lanes)
            seed = np.asarray(seed_key.key).tolist() if seed_key is not None else uid
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
                resumed += 1
                if resume_rng and state.rng_key is not None:
                    rng = RequestRNG(state=json.loads(bytes(np.asarray(state.rng_key, dtype=np.uint8)).decode()))
            else:
                # An ordinary APC hit cannot fabricate draft context. Rebuild
                # both planes from transcript instead of pairing stale caches.
                # This includes a zero-length hit: its branch still holds the
                # APC's hooked cache objects, which the lane must not adopt.
                todo = prefix + todo; prefix = []; target = None
            cuts = tuple(prefill_cuts[i] or ())
            if any(
                isinstance(position, bool) or not isinstance(position, int)
                for position in cuts
            ) or tuple(sorted(set(cuts))) != cuts:
                raise ValueError("invalid prefill cut positions")
            passed = len(prefix)
            total = passed + len(todo)
            cuts = tuple(
                position for position in cuts
                if passed < position < total - 1
            )
            lane = Lane(
                uid,
                prefix,
                deque(todo),
                list(target) if target is not None else self.model.make_cache(),
                draft_cache,
                tail,
                rng,
                int(max_tokens[i]),
                (logits_processors or [[]] * len(prompts))[i] or [],
                dict((sampling_configs or [{}] * len(prompts))[i]),
                prefill_cuts=deque(cuts),
            )
            if self.round_timing:
                lane.external_timing_start = {
                    key: int(value)
                    for key, value in self.scheduler_stats.items()
                    if key.startswith("external_phase_")
                    and (
                        key.endswith("_ns")
                        or key == "external_phase_synchronizations"
                    )
                    and type(value) is int
                }
            if self._sampled_lane_needs_ordinary(lane):
                # Point-mass continuation laws can only be verified greedily
                # (progressive add_state refuses them under sampling): serve
                # this request on the exact ordinary target path, labelled in
                # its receipt, instead of failing it mid-round.  Ordinary
                # lanes cohort apart, so greedy lanes keep speculating.
                lane.ordinary = True
                lane.ordinary_reason = "sampled_request_on_greedy_only_point_mass_ast"
                _bump(self.scheduler_stats, "external_sampled_point_mass_ordinary_lanes")
            lane.proposal_request_id = self._generator_revision + ":" + str(uid)
            lane.session_scope_hash = ((session_keys or [None] * len(prompts))[i]
                or hashlib.sha256(json.dumps([self.binding, lane.proposal_request_id]).encode()).hexdigest())
            lane.ast_memo_named_scope = bool(
                (session_keys or [None] * len(prompts))[i]
            )
            lane.ast_memo_layout_revision = (
                None
                if ast_memo_layout_revisions is None
                else ast_memo_layout_revisions[i]
            )
            if self.cost_per_commit_handoff is not None:
                from .cost_per_commit_handoff import CostPerCommitLaneState

                lane.cost_handoff_state = CostPerCommitLaneState(
                    self.cost_per_commit_handoff,
                    lane_uid=uid,
                    owner_scope=(self.binding, lane.session_scope_hash),
                )
            pending_lanes.append(lane)
        if self.ast_memo is not None:
            registered = []
            for lane in pending_lanes:
                if lane.ast_memo_named_scope:
                    continue
                if self.ast_memo.register_scope(lane.session_scope_hash):
                    registered.append(lane.session_scope_hash)
                    continue
                for scope in registered:
                    self.ast_memo.advance(
                        (scope,), "anonymous_lane_end", purge=True
                    )
                for pending in pending_lanes:
                    pending.ast_memo_layout_revision = None
                break
        self.lanes.update((lane.uid, lane) for lane in pending_lanes)
        self.next_uid += len(pending_lanes)
        self.scheduler_stats["paired_cache_resumes"] += resumed
        planned_cuts = sum(len(lane.prefill_cuts) for lane in pending_lanes)
        if planned_cuts:
            _bump(
                self.scheduler_stats,
                "external_prefill_cuts_planned",
                planned_cuts,
            )
        return [lane.uid for lane in pending_lanes]

    def _sampled_lane_needs_ordinary(self, lane):
        """A sampled lane on a progressive route whose proposals are point-mass
        continuation laws (``continuation_pool``), which verify greedy-only."""
        return (
            float(lane.sampling.get("sampling_temp", 0)) != 0.0
            and self.progressive_verification_tile is not None
            and self.continuation_policy is not None
        )

    def _sidecar(self, lane):
        draft_cache, tail = self._snapshot_draft_state(
            lane.draft_cache, lane.tail
        )
        # A durable row/tail view can pin a whole prefill/verify allocation,
        # even when it has zero elements. Copy only its visible values and
        # submit the copy now: an idle APC entry may never evaluate it again.
        # contiguous() can alias an already contiguous row slice. Keep the
        # recovery/prelaunch descriptor snapshots above zero-copy instead.
        tail = self.mx.array(tail)
        self.mx.async_eval(tail)
        state = ExternalDraftState((draft_cache, tail), len(lane.history), self.mx.array(list(json.dumps(lane.rng.snapshot(), sort_keys=True).encode()), dtype=self.mx.uint8), lane.rng.draws, binding=self.binding)
        state.validate(self.binding, len(lane.history)); return state

    def _prefill_limit(self, lane):
        step = (
            prompt_length_prefill_step(
                len(lane.history) + len(lane.remaining), maximum=self.prefill_step
            )
            if self.prefill_step_autoscale else self.prefill_step
        )
        while lane.prefill_cuts and lane.prefill_cuts[0] <= len(lane.history):
            lane.prefill_cuts.popleft()
            _bump(self.scheduler_stats, "external_prefill_cuts_reached")
        if lane.prefill_cuts:
            step = min(step, lane.prefill_cuts[0] - len(lane.history))
        return step

    def _sync_decode_fairness_stats(self):
        fairness = self.decode_time_fairness
        if fairness.enabled:
            for key, value in fairness.counters.items():
                self.scheduler_stats[f"decode_fairness_{key}"] = int(value)

    def _decoding_beside_prefill(self):
        """Whether a lane decodes in this poll beside its prefill slice.

        A lane held at its prompt boundary for atomic-cohort peers waits for
        this prefill rather than decoding, so it does not make a slice
        contended (the check is the pure form of
        ``_atomic_cohort_prefill_ready``).
        """
        for lane in self.lanes.values():
            if lane.anchor is None or lane.cancelled:
                continue
            key = self._batch_cohort_key(lane)
            if key is None:
                return True
            members = [
                candidate
                for candidate in self.lanes.values()
                if not candidate.cancelled
                and self._batch_cohort_key(candidate) == key
            ]
            if len(members) != key[2] or all(
                candidate.anchor is not None for candidate in members
            ):
                return True
        return False

    def _fair_step(self, step, *, contended, rows=1, depth=0):
        """Stall-bound a prefill slice taken beside decoding lanes.

        The same cost-model bound and contended slice floor as the ordinary
        route.  Before it the external route sliced at the configured (or
        prompt-length autoscaled) step whatever the decode lanes paid: 512
        rows on a 12K prompt, 2048 at 32-64K and 8192 above 64K.
        """
        fairness = self.decode_time_fairness
        capped = fairness.cap(step, contended=contended, rows=rows, depth=depth)
        return fairness.floor_slice(capped, step, contended=contended)

    def _observe_prefill(self, width, seconds, *, contended, rows, depth):
        fairness = self.decode_time_fairness
        if fairness.enabled:
            fairness.observe_prefill(
                width, seconds, contended=contended, rows=rows, depth=depth
            )
            self._sync_decode_fairness_stats()

    def _prefill_response(self, lane, *, external_prefill_width=1):
        """Publish progress and the paired committed boundary after a chunk."""
        done = len(lane.remaining) == 1
        span = len(lane.history) + len(lane.remaining)
        progress = (span if done else len(lane.history), span)
        if done:
            lane.anchor = lane.remaining.popleft()
            if lane.history:
                if (lane.tail.shape[1] and not self.pair_context_tokens
                        and not getattr(self.draft, "requires_pending_context_at_prefill", False)):
                    self._append_context(lane, lane.anchor)
                    lane.tail = lane.tail[:, :0]
                self.boundaries[lane.uid] = {"committed_only": True, "tokens": list(lane.history), "target_cache": self._freeze_cache(lane.cache), "covered_tokens": len(lane.history), "cache_sidecar": self._sidecar(lane)}
        return SimpleNamespace(
            uid=lane.uid,
            progress=progress,
            end_of_prompt=done,
            external_prefill_width=external_prefill_width,
        )

    def _prefill(self, lane, *, step=None, contended=False):
        if len(lane.remaining) > 1:
            n = min(
                self._prefill_limit(lane) if step is None else step,
                len(lane.remaining) - 1,
            )
            depth = len(lane.history)
            tic = time.perf_counter()
            if lane.tail.shape[1]:
                self._append_context(lane, lane.remaining[0])
            inputs = [lane.remaining.popleft() for _ in range(n)]
            lane.tail = self.model.prefill_body(self.mx.array([inputs]), lane.cache, self.layers)
            lane.history.extend(inputs)
            self.mx.eval(lane.tail, [c.state for c in lane.cache], [c.state for c in lane.draft_cache if c.offset])
            self._observe_prefill(
                n, time.perf_counter() - tic, contended=contended, rows=1,
                depth=depth,
            )
            self.scheduler_stats["prefill_rounds"] += 1
            if self.prefill_allocator_reclaim:
                # After the materialization above; lane.tail stays referenced.
                self.mx.clear_cache()
                _bump(self.scheduler_stats, "external_prefill_allocator_reclaims")
        return self._prefill_response(lane)

    def _prefill_many(self, lanes, *, step=None, contended=False):
        """Advance several external target prompts through one padded slab.

        Target cache padding is handled by the same cache contract as ordinary
        prefill. Draft state remains request-private: each lane consumes its
        preceding target taps before the new target slab is formed.
        """
        lanes = [lane for lane in lanes if len(lane.remaining) > 1]
        if len(lanes) < 2:
            return [
                self._prefill(lane, step=step, contended=contended)
                for lane in lanes
            ]
        lengths = [
            min(
                self._prefill_limit(lane)
                if step is None
                else min(step, self._prefill_limit(lane)),
                len(lane.remaining) - 1,
            )
            for lane in lanes
        ]
        width = max(lengths)
        depth = max(len(lane.history) for lane in lanes)
        tic = time.perf_counter()
        chunks = []
        for lane, length in zip(lanes, lengths):
            if lane.tail.shape[1]:
                self._append_context(lane, lane.remaining[0])
            chunks.append([lane.remaining.popleft() for _ in range(length)])

        padding = [width - length for length in lengths]
        cache = _merge_caches([lane.cache for lane in lanes])
        for plane in cache:
            plane.prepare(lengths=lengths, right_padding=padding)
        tokens = _right_pad_prompts(chunks, max_length=width)
        with self.model.prefill_row_context(lengths, width=width):
            tails = self.model.prefill_body(tokens, cache, self.layers)
        self.mx.eval(tails, [plane.state for plane in cache])
        for plane in cache:
            plane.finalize()
        self.mx.eval([plane.state for plane in cache])

        for row, (lane, chunk, length) in enumerate(zip(lanes, chunks, lengths)):
            lane.cache = [plane.extract(row) for plane in cache]
            lane.tail = tails[row : row + 1, :length]
            lane.history.extend(chunk)
            self.mx.eval(
                lane.tail,
                [plane.state for plane in lane.cache],
                [plane.state for plane in lane.draft_cache if plane.offset],
            )
        self._observe_prefill(
            width, time.perf_counter() - tic, contended=contended,
            rows=len(lanes), depth=depth,
        )
        self.scheduler_stats["prefill_rounds"] += 1
        _bump(self.scheduler_stats, "external_batched_prefill_rounds")
        _bump(self.scheduler_stats, "external_batched_prefill_lanes", len(lanes))
        self.scheduler_stats["external_batched_prefill_max_cohort_width"] = max(
            self.scheduler_stats["external_batched_prefill_max_cohort_width"],
            len(lanes),
        )
        self.scheduler_stats["external_batched_prefill_max_token_width"] = max(
            self.scheduler_stats["external_batched_prefill_max_token_width"], width
        )
        _bump(
            self.scheduler_stats,
            "external_batched_prefill_padding_rows",
            sum(padding),
        )
        if self.prefill_allocator_reclaim:
            self.mx.clear_cache()
            _bump(self.scheduler_stats, "external_prefill_allocator_reclaims")
        return [
            self._prefill_response(
                lane, external_prefill_width=len(lanes)
            )
            for lane in lanes
        ]

    def _target_law(
        self, lane, logits, history, reachable=True, response_rows=None,
        greedy_token=False, *, history_suffix=(),
    ):
        """Return the verification law of one target row.

        When ``response_rows`` is a list, also append the log-probability row
        to publish for this position, or ``None`` to publish the law itself.
        ``greedy_token`` returns a greedy row's argmax instead of its dense
        one-hot law (``verify_greedy_proposals``).
        Keep ``history_suffix`` separate until a reachable processor needs
        tokens; ordinary laws need neither a full host copy nor an MLX array.
        """
        from .sample_utils import make_transformed_logprobs
        # Processors compute in float32, as ordinary decode applies them.
        value = logits[None].astype(self.mx.float32)
        # A row that follows a draft token the target gives zero probability is
        # never used by verification.  Its history can already be outside a
        # structured-output grammar, and asking the processor about it would
        # latch a dead-end failure on a healthy lane (GPU: every structured
        # request on the DFlash2 route returned 502).
        if reachable and lane.processors:
            tokens = self.mx.array(
                [*history, *history_suffix] if len(history_suffix) else history,
                dtype=self.mx.int32,
            )
            for processor in lane.processors:
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
            if greedy_token:
                return int(selected.item())
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
            "verify_accept_span_hist": dict(lane.verify_accept_span_hist),
        }
        if getattr(lane, "trusted_pld_rounds", 0):
            return {
                "verification": "mixed_exact_and_trusted_pld",
                "approximate": True,
                "trusted_pld_rounds": lane.trusted_pld_rounds,
                "trusted_pld_tokens": getattr(lane, "trusted_pld_tokens", 0),
                "copied_row_target_draws": 0,
                "bonus_row_target_draws": 1,
                "relaxed_accepts": lane.relaxed_accepts,
                **hists,
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
        ast_planner = getattr(self, "ast_execution_planner", None)
        if ast_planner is not None:
            result["ast_execution_plan"] = {
                **ast_planner.receipt(),
                "source_budget": getattr(lane, "ast_source_budget", None),
                "last_launch": getattr(lane, "ast_last_launch", None),
            }
        if getattr(lane, "ast_tree_tiles_receipt", None) is not None:
            result["ast_tree_tiles"] = snapshot_receipt(lane.ast_tree_tiles_receipt)
        if self.progressive_verification_tile is not None:
            approximate = (
                self.progressive_verification_fidelity
                == PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED
            )
            if approximate:
                result["verification"] = "approximate_progressive_near_tie_bounded"
                result["approximate"] = True
            result["progressive_verification"] = {
                "algorithm": PROGRESSIVE_EXTERNAL_VERIFY_VERSION,
                "target_protocol": self.progressive_target_protocol,
                "fidelity": self.progressive_verification_fidelity,
                "source_hashes": self.progressive_protocol_binding[
                    "source_hashes"
                ],
                "qualification_gate_schema": self.progressive_protocol_binding[
                    "qualification_gate_schema"
                ],
                "implemented": True,
                "qualified": False,
                "selected": True,
                "observed_used": (
                    getattr(lane, "progressive_verify_rounds", 0) > 0
                ),
                "verification_tile": self.progressive_verification_tile,
                **(
                    {
                        "verification_tile_policy": "auto",
                        "verification_tile_resolution_max_lanes": self.capacity,
                    }
                    if self.progressive_verification_tile_policy == "auto"
                    else {}
                ),
                **(
                    {
                        "terminal_input_widths": list(
                            self.progressive_terminal_input_widths
                        )
                    }
                    if self.progressive_verification_tile_policy == "auto"
                    else {}
                ),
                "rounds": getattr(lane, "progressive_verify_rounds", 0),
                "target_launches": getattr(
                    lane, "progressive_verify_launches", 0
                ),
                "target_rows": getattr(
                    lane, "progressive_verify_target_rows", 0
                ),
                "padding_rows": getattr(
                    lane, "progressive_verify_padding_rows", 0
                ),
                "ragged_launches": getattr(
                    lane, "progressive_verify_ragged_launches", 0
                ),
                "rejections": getattr(
                    lane, "progressive_verify_rejections", 0
                ),
                "full_nonfinal_tiles": getattr(
                    lane, "progressive_verify_full_tiles", 0
                ),
                "tile_checkpoints": getattr(
                    lane, "progressive_verify_tile_checkpoints", 0
                ),
                "last_tile_input_rows": getattr(
                    lane, "progressive_verify_last_tile_input_rows", []
                ),
                "last_tile_checkpoints": getattr(
                    lane, "progressive_verify_last_tile_checkpoints", []
                ),
                "last_rollback_base_tile": getattr(
                    lane, "progressive_verify_last_rollback_base_tile", 0
                ),
                "last_cohort_width": getattr(
                    lane, "progressive_verify_last_cohort_width", 0
                ),
                "max_cohort_width": getattr(
                    lane, "progressive_verify_max_cohort_width", 0
                ),
                "last_proposal_sha256": getattr(
                    lane, "progressive_verify_last_proposal_sha256", None
                ),
                "last_ragged_layout": getattr(
                    lane, "progressive_verify_last_ragged_layout", None
                ),
                "target_row_executor": snapshot_receipt(
                    getattr(
                        lane,
                        "progressive_target_row_executor",
                        _target_executor_stats(),
                    )
                ),
                # Non-progressive cohort forwards (standard verify after a
                # progressive decline, plain/parked/ordinary rounds), by
                # ``site:path``: ``lane_exact`` ran the adapter-owned
                # row-exact executor, ``batched`` the width-sensitive batched
                # forward, ``single_lane`` a B1 transaction.
                "cohort_target_execution": dict(
                    getattr(lane, "cohort_target_execution", None) or {}
                ),
                "ordered_alternatives": {
                    "display_name": "Adaptive Speculation Topology",
                    "mechanism_id": "adaptive_speculation_topology",
                    "mechanism_short_name": "AST",
                    "deprecated_mechanism_aliases": [
                        "adaptive_proposal_topology",
                        "APT",
                    ],
                    "mechanism": "adaptive_speculation_topology",
                    "configured_branches": self.progressive_alternative_branches,
                    "last_active_branches": getattr(
                        lane,
                        "progressive_alternative_last_active_branches",
                        1,
                    ),
                    "configured_strategy": self.progressive_alternative_strategy,
                    "configured_depth_policy": (
                        self.progressive_alternative_depth_policy
                    ),
                    "winner_contract": (
                        "exact_max_visited_committed_prefix_then_full_rank"
                        if getattr(self, "continuation_pool_ast_bridge", False)
                        else "first_nonlosing_full_branch_else_b1"
                    ),
                    "rootless_fallback_allowed": (
                        self.progressive_ast_rootless_fallback
                    ),
                    "rootless_refusal_rounds": getattr(
                        lane, "progressive_ast_rootless_refusal_rounds", 0
                    ),
                    "rounds": getattr(
                        lane, "progressive_alternative_rounds", 0
                    ),
                    "primary_fallback_rounds": getattr(
                        lane, "progressive_ast_primary_fallback_rounds", 0
                    ),
                    "primary_fallback_reasons": snapshot_receipt(
                        getattr(lane, "progressive_ast_primary_fallback_reasons", {})
                    ),
                    "last_primary_fallback_reason": getattr(
                        lane, "progressive_ast_last_primary_fallback_reason", None
                    ),
                    "branch_downshift_rounds": getattr(
                        lane, "progressive_ast_branch_downshift_rounds", 0
                    ),
                    "last_branch_downshift": snapshot_receipt(
                        getattr(lane, "progressive_ast_last_branch_downshift", None)
                    ),
                    "rootless_rounds": getattr(
                        lane, "progressive_ast_rootless_rounds", 0
                    ),
                    "last_topology": getattr(
                        lane, "progressive_ast_last_topology", None
                    ),
                    "adaptive_prefix_rounds": getattr(
                        lane, "progressive_ast_adaptive_prefix_rounds", 0
                    ),
                    "last_prefix_search": snapshot_receipt(
                        getattr(lane, "progressive_ast_last_prefix_search", None)
                    ),
                    "last_prefix_diagnostics": snapshot_receipt(
                        getattr(
                            lane,
                            "progressive_ast_last_prefix_diagnostics",
                            None,
                        )
                    ),
                    "last_proposed_prefix_diagnostics": snapshot_receipt(
                        getattr(
                            lane,
                            "progressive_ast_last_proposed_prefix_diagnostics",
                            None,
                        )
                    ),
                    "sampled_primary_rounds": getattr(
                        lane, "progressive_alternative_sampled_primary_rounds", 0
                    ),
                    "stop_guard_primary_rounds": getattr(
                        lane, "progressive_alternative_stop_guard_rounds", 0
                    ),
                    "last_branch_accepts": getattr(
                        lane,
                        "progressive_alternative_last_branch_accepts",
                        [],
                    ),
                    "last_branch_proposal_counts": getattr(
                        lane,
                        "progressive_alternative_last_branch_proposal_counts",
                        [],
                    ),
                    "last_branch_outcomes": getattr(
                        lane,
                        "progressive_alternative_last_branch_outcomes",
                        [],
                    ),
                    "last_branch_sha256": getattr(
                        lane,
                        "progressive_alternative_last_branch_hashes",
                        [],
                    ),
                    "last_full_accept_branches": getattr(
                        lane,
                        "progressive_alternative_last_full_branches",
                        [],
                    ),
                    "last_selected_branch": getattr(
                        lane,
                        "progressive_alternative_last_selected_branch",
                        None,
                    ),
                    "last_branch_committed_tokens": getattr(
                        lane,
                        "progressive_alternative_last_branch_committed_tokens",
                        [],
                    ),
                    "last_selected_token_uplift": getattr(
                        lane,
                        "progressive_alternative_last_selected_token_uplift",
                        0,
                    ),
                    "last_discarded_branches": getattr(
                        lane,
                        "progressive_alternative_last_discarded_branches",
                        [],
                    ),
                    "last_short_circuit_after_branch": getattr(
                        lane,
                        "progressive_alternative_last_short_circuit",
                        None,
                    ),
                    "last_strategy": getattr(
                        lane,
                        "progressive_alternative_last_strategy",
                        None,
                    ),
                    "last_shared_prefix_tokens": getattr(
                        lane,
                        "progressive_alternative_last_shared_prefix_tokens",
                        0,
                    ),
                    "last_unique_target_rows": getattr(
                        lane,
                        "progressive_alternative_last_unique_target_rows",
                        0,
                    ),
                    "geometry_totals": {
                        "schema": "mlx2.ast-geometry.v2",
                        "scope": "request_session_all_progressive_ast_modes",
                        "provider_rank_basis": "pool_selection_rank",
                        "counterfactual_scope": "verified_partial_branches_only",
                        "verified_branches": getattr(
                            lane, "progressive_ast_verified_branches_total", 0
                        ),
                        "unique_target_rows": getattr(
                            lane, "progressive_ast_unique_target_rows_total", 0
                        ),
                        "shared_prefix_target_rows": getattr(
                            lane,
                            "progressive_ast_shared_prefix_target_rows_total",
                            0,
                        ),
                        "branch_suffix_target_rows": getattr(
                            lane,
                            "progressive_ast_branch_suffix_target_rows_total",
                            0,
                        ),
                        "selected_committed_inputs": getattr(
                            lane,
                            "progressive_ast_selected_committed_inputs_total",
                            0,
                        ),
                        "selected_token_uplift": getattr(
                            lane, "progressive_ast_selected_token_uplift_total", 0
                        ),
                        "counterfactual_partial_uplift_rounds": getattr(
                            lane,
                            "progressive_ast_counterfactual_partial_uplift_rounds",
                            0,
                        ),
                        "counterfactual_partial_uplift_tokens": getattr(
                            lane,
                            "progressive_ast_counterfactual_partial_uplift_tokens",
                            0,
                        ),
                        "counterfactual_best_branch_hist": snapshot_receipt(
                            getattr(
                                lane,
                                "progressive_ast_counterfactual_best_branch_hist",
                                {},
                            )
                        ),
                        "provider_rank_verified_hist": snapshot_receipt(
                            getattr(
                                lane,
                                "progressive_ast_provider_rank_verified_hist",
                                {},
                            )
                        ),
                        "provider_rank_committed_inputs": snapshot_receipt(
                            getattr(
                                lane,
                                "progressive_ast_provider_rank_committed_inputs",
                                {},
                            )
                        ),
                        "provider_rank_full_accept_hist": snapshot_receipt(
                            getattr(
                                lane,
                                "progressive_ast_provider_rank_full_accept_hist",
                                {},
                            )
                        ),
                        "alternative_divergence_position_hist": snapshot_receipt(
                            getattr(
                                lane,
                                "progressive_ast_alternative_divergence_position_hist",
                                {},
                            )
                        ),
                        "prefix_rejected_rounds": getattr(
                            lane, "progressive_ast_prefix_rejected_rounds", 0
                        ),
                        "active_branch_hist": snapshot_receipt(
                            getattr(lane, "progressive_ast_active_branch_hist", {})
                        ),
                        "shared_prefix_token_hist": snapshot_receipt(
                            getattr(
                                lane,
                                "progressive_ast_shared_prefix_token_hist",
                                {},
                            )
                        ),
                    },
                    "b2_full_accept_short_circuits": getattr(
                        lane,
                        "progressive_alternative_b2_short_circuits",
                        0,
                    ),
                    "winner_selection": (
                        "exact_max_visited_committed_prefix_then_full_rank"
                        if getattr(self, "continuation_pool_ast_bridge", False)
                        else "first_full_branch_with_at_least_b1_committed_tokens"
                    ),
                    "fallback_when_none_full": (
                        "commit_best_exact_visited_prefix"
                        if getattr(self, "continuation_pool_ast_bridge", False)
                        else "commit_b1_exact_result"
                    ),
                    "fallback_when_no_useful_full": (
                        "commit_best_exact_visited_prefix"
                        if getattr(self, "continuation_pool_ast_bridge", False)
                        else "commit_b1_exact_result"
                    ),
                    "next_round_base": "selected_branch_committed_state",
                    "committed_physical_width": getattr(
                        lane,
                        "progressive_alternative_last_physical_width",
                        0,
                    ),
                    "cross_shape_state_equivalence": (
                        "unqualified_shape_sensitive_float_arithmetic"
                    ),
                    "cross_session_ast_packing": {
                        "configured_pack_rows": (
                            self.progressive_cross_session_pack_rows
                        ),
                        "configured_mode": (
                            self.progressive_cross_session_pack_mode
                        ),
                        "observed_used": bool(
                            getattr(
                                lane,
                                "progressive_cross_session_ast_last_packs",
                                [],
                            )
                        ),
                        "last_pack_rows": getattr(
                            lane,
                            "progressive_cross_session_ast_last_pack_rows",
                            None,
                        ),
                        "last_packs": getattr(
                            lane,
                            "progressive_cross_session_ast_last_packs",
                            [],
                        ),
                        "mode": getattr(
                            lane,
                            "progressive_cross_session_ast_last_mode",
                            None,
                        ),
                        "live_frontier": getattr(
                            lane,
                            "progressive_cross_session_ast_last_frontier",
                            None,
                        ),
                        "state_scope": "request_private",
                        "physical_scope": "shared_target_launch_only",
                    },
                    "cross_session_apt_packing": {
                        "deprecated_alias_for": "cross_session_ast_packing",
                        "configured_pack_rows": self.progressive_cross_session_pack_rows,
                        "configured_mode": self.progressive_cross_session_pack_mode,
                        "observed_used": bool(
                            getattr(
                                lane,
                                "progressive_cross_session_ast_last_packs",
                                [],
                            )
                        ),
                        "last_pack_rows": getattr(
                            lane,
                            "progressive_cross_session_ast_last_pack_rows",
                            None,
                        ),
                        "last_packs": getattr(
                            lane,
                            "progressive_cross_session_ast_last_packs",
                            [],
                        ),
                        "mode": getattr(
                            lane,
                            "progressive_cross_session_ast_last_mode",
                            None,
                        ),
                        "live_frontier": getattr(
                            lane,
                            "progressive_cross_session_ast_last_frontier",
                            None,
                        ),
                        "state_scope": "request_private",
                        "physical_scope": "shared_target_launch_only",
                    },
                    "adaptive_latch": {
                        "scope": "request_session",
                        "decision_uses": "this_session_prior_completed_batches_only",
                        "last_decision": getattr(
                            lane,
                            "progressive_topology_latch_last_decision",
                            None,
                        ),
                        "last_observation": getattr(
                            lane,
                            "progressive_topology_latch_last_observation",
                            None,
                        ),
                    },
                },
                "b4_disjoint_first_tiles": getattr(
                    lane, "progressive_verify_b4_disjoint_first_tiles", 0
                ),
                "checkpoint_state": (
                    "joint_attention_kv_and_recurrent_layer_state"
                ),
                "rollback_policy": "last_accepted_tile_then_exact_prefix",
                "state_promotion": "request_private_then_atomic_lane_publish",
                "performance_claim": False,
            }
        if getattr(self, "exact_verification", "token") == "block":
            # Greedy rows verify by token compare under either rule (block
            # and token-wise decisions coincide for one-hot targets).
            result["verification"] = (
                "exact_block" if getattr(lane, "block_verified_rounds", 0) else "exact"
            )
            result["exact_verification"] = {
                "selected": "block",
                "algorithm": "block-verification-sun-2024-alg2-v1",
                "block_verified_rounds": getattr(lane, "block_verified_rounds", 0),
                "qualified": False,
            }
        if self.fly_verification.enabled and lane.processors:
            result["fly_disabled"] = "logits_processors"
        return result

    def _target_execution_receipt(self, lane=None):
        model_receipt = getattr(self.model, "external_execution_receipt", None)
        result = {"target_protocol": dict(model_receipt)} if model_receipt else {}
        cohort_key = self._batch_cohort_key(lane) if lane is not None else None
        if cohort_key is not None:
            result["declared_cohort_frontier"] = {
                "implemented": True,
                "selected": True,
                "qualified": False,
                "observed_used": cohort_key in self._released_atomic_cohorts,
                "declared_width": cohort_key[2],
                "ownership": "atomic_until_complete_decode_frontier",
            }
        if lane is not None and lane.cost_handoff_state is not None:
            result["cost_per_commit_handoff"] = self._cost_handoff_receipt(lane)
        if getattr(self, "external_varlen_prefill", False):
            result["external_varlen_prefill"] = {
                "implemented": True,
                "selected": True,
                "qualified": False,
                "observed_used": (
                    self.scheduler_stats["external_batched_prefill_rounds"] > 0
                ),
                "rounds": self.scheduler_stats["external_batched_prefill_rounds"],
                "lanes": self.scheduler_stats["external_batched_prefill_lanes"],
                "max_cohort_width": self.scheduler_stats[
                    "external_batched_prefill_max_cohort_width"
                ],
                "max_token_width": self.scheduler_stats[
                    "external_batched_prefill_max_token_width"
                ],
                "padding_rows": self.scheduler_stats[
                    "external_batched_prefill_padding_rows"
                ],
                "coalesce_ms": self.external_prefill_coalesce_ms,
                "coalesce_min_tokens": self.external_prefill_coalesce_min_tokens,
                "coalesce_deferrals": self.scheduler_stats[
                    "external_prefill_coalesce_deferrals"
                ],
                "coalesce_expirations": self.scheduler_stats[
                    "external_prefill_coalesce_expirations"
                ],
                "coalescing": {
                    "implemented": True,
                    "selected": self.external_prefill_coalesce_ms > 0,
                    "qualified": False,
                    "observed_used": self.scheduler_stats[
                        "external_prefill_coalesce_deferrals"
                    ] > 0,
                    "maximum_wait_ms": self.external_prefill_coalesce_ms,
                    "minimum_prompt_tokens": (
                        self.external_prefill_coalesce_min_tokens
                    ),
                },
            }
        if self.dynamic_singleton_tree:
            tree_budgets = getattr(
                self,
                "tree_node_budget_by_lanes",
                {
                    width: 15
                    for width in range(1, self.dynamic_tree_max_width + 1)
                },
            )
            budget_width = min(
                max(1, int(self._auto_active_width)), self.dynamic_tree_max_width
            )
            result["batch_size_route"] = {
                "policy": (
                    "tree15_b1_b4_chain_b5plus_v1"
                    if self.dynamic_tree_max_width == 4
                    else "tree15_b1_chain_b2plus_v1"
                ),
                "max_tree_width": self.dynamic_tree_max_width,
                "tree_node_budget_by_lanes": {
                    str(width): nodes
                    for width, nodes in tree_budgets.items()
                },
                "current_tree_node_budget": tree_budgets[budget_width],
                "active_lanes": self._auto_active_width,
                "cohort_width": self._auto_cohort_width,
                "current": (
                    "tree15_provider"
                    if self.draft_topology == "tree15"
                    else "chain_reference"
                ),
                "qualified": False,
            }
            if self.dynamic_tree_max_width == 4:
                result["batch_size_route"]["chain_draft_cap"] = _chain_draft_cap(
                    tree_budgets, 4, self._auto_active_width
                )
        provider_selected = getattr(
            self,
            "target_tree_selected",
            self.target_execution == "provider",
        )
        if not provider_selected:
            return result
        width_rounds = {
            str(width): self.scheduler_stats.get(
                f"external_target_tree_target_width_{width}_rounds", 0
            )
            for width in range(1, self.target_tree_cohort_limit + 1)
        }
        result["target_tree"] = {
            "cohort_limit": self.target_tree_cohort_limit,
            "cohort_max_width": self.scheduler_stats[
                "external_target_tree_cohort_max_width"
            ],
            "cohort_rounds": self.scheduler_stats[
                "external_target_tree_cohort_rounds"
            ],
            "packed_target_rounds": self.scheduler_stats.get(
                "external_target_tree_packed_target_rounds", 0
            ),
            "packed_target_lanes": self.scheduler_stats.get(
                "external_target_tree_packed_target_lanes", 0
            ),
            "physical_target_forwards": self.scheduler_stats.get(
                "external_target_tree_physical_target_forwards", 0
            ),
            "target_width_rounds": width_rounds,
        }
        result["target_tree"].update(
            self.target_tree_executor_provider.receipt()
        )
        if lane is not None:
            result["target_tree"].update(
                lane_width_histogram={
                    str(width): int(rounds)
                    for width, rounds in sorted(
                        lane.target_tree_width_hist.items()
                    )
                },
                lane_width_trace=list(lane.target_tree_width_trace),
                lane_width_trace_overflow=(
                    lane.target_tree_width_trace_overflow
                ),
            )
        # Receipt compatibility only. Scheduler state and counters stay
        # provider-neutral; older observers may still consume this key.
        result["tensorfold_target"] = snapshot_receipt(result["target_tree"])
        return result

    def _cost_handoff_receipt(self, lane):
        receipt = (
            lane.cost_handoff_state.receipt()
            if lane.cost_handoff_receipt is None
            else snapshot_receipt(lane.cost_handoff_receipt)
        )
        authority = getattr(
            self, "cost_per_commit_handoff_authority", "explicit_policy"
        )
        receipt["selection_authority"] = authority
        receipt["defaulted"] = (
            authority == "runtime_ast_self_configuring_default"
        )
        return receipt

    def _apply_target_tree_dispatch_metrics(self, cohort, event):
        """Apply provider-described metrics without delegating scheduler state.

        The provider callback used to mutate this scheduler after returning an
        open target-cache transaction.  A callback failure then prevented the
        caller from receiving (and aborting) that transaction.  Validate and
        stage every update here before publishing any of them.
        """
        from .target_tree_executor import TargetTreeDispatchMetrics

        if not isinstance(event, TargetTreeDispatchMetrics):
            raise TypeError("target-tree provider returned invalid dispatch metrics")
        width = event.width
        if type(width) is not int or width != len(cohort) or width <= 0:
            raise ValueError("target-tree dispatch metric width disagrees with cohort")
        key = f"external_target_tree_target_width_{width}_rounds"
        if key not in self.scheduler_stats:
            raise RuntimeError(
                f"target-tree width {width} exceeds configured receipt bounds"
            )
        required = {
            key,
            "external_target_tree_target_rounds",
            "external_target_tree_physical_target_forwards",
        }
        if width > 1:
            required.update({
                "external_target_tree_cohort_rounds",
                "external_target_tree_cohort_lanes",
                "external_target_tree_cohort_max_width",
            })
            if event.physical_packed:
                required.update({
                    "external_target_tree_packed_target_rounds",
                    "external_target_tree_packed_target_lanes",
                })
        cache_key = None
        if event.cached:
            cache_key = (
                "external_target_tree_executor_cache_hits"
                if event.executor_cached
                else "external_target_tree_executor_validations"
            )
            required.add(cache_key)
        missing = required.difference(self.scheduler_stats)
        if missing:
            raise RuntimeError(
                f"target-tree dispatch metrics are not declared: {sorted(missing)!r}"
            )
        for name in required:
            if type(self.scheduler_stats[name]) is not int:
                raise TypeError(f"target-tree scheduler metric {name} is not an integer")
        lane_updates = []
        for lane in cohort:
            histogram = dict(lane.target_tree_width_hist)
            histogram[width] = histogram.get(width, 0) + 1
            trace = list(lane.target_tree_width_trace)
            overflow = int(lane.target_tree_width_trace_overflow)
            if len(trace) < 128:
                trace.append(width)
            else:
                overflow += 1
            lane_updates.append((lane, histogram, trace, overflow))

        stats = dict(self.scheduler_stats)
        stats[key] += 1
        stats["external_target_tree_target_rounds"] += 1
        stats["external_target_tree_physical_target_forwards"] += 1
        if width > 1:
            stats["external_target_tree_cohort_rounds"] += 1
            stats["external_target_tree_cohort_lanes"] += width
            stats["external_target_tree_cohort_max_width"] = max(
                stats["external_target_tree_cohort_max_width"], width
            )
            if event.physical_packed:
                stats["external_target_tree_packed_target_rounds"] += 1
                stats["external_target_tree_packed_target_lanes"] += width
        if cache_key is not None:
            stats[cache_key] += 1

        previous_stats = self.scheduler_stats
        previous_lanes = [
            (
                lane,
                lane.target_tree_width_hist,
                lane.target_tree_width_trace,
                lane.target_tree_width_trace_overflow,
            )
            for lane in cohort
        ]
        try:
            self.scheduler_stats = stats
            for lane, histogram, trace, overflow in lane_updates:
                lane.target_tree_width_hist = histogram
                lane.target_tree_width_trace = trace
                lane.target_tree_width_trace_overflow = overflow
        except BaseException:
            self.scheduler_stats = previous_stats
            for lane, histogram, trace, overflow in previous_lanes:
                lane.target_tree_width_hist = histogram
                lane.target_tree_width_trace = trace
                lane.target_tree_width_trace_overflow = overflow
            raise

    def _finish_target_tree_dispatch(
        self,
        cohort,
        result,
        *,
        cached,
        provider=None,
        publish_tree_metrics=True,
    ):
        """Publish metrics or abort the still-open target transaction."""
        provider = self.target_tree_executor_provider if provider is None else provider
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError("target-tree executor must return logits, features, transaction")
        transaction = result[2]
        from .target_tree_executor import TargetTreeTransaction

        if not isinstance(transaction, TargetTreeTransaction):
            abort = getattr(transaction, "abort", None)
            # An invalid provider may also supply a malformed truthy `closed`
            # value. Only exact True proves cleanup already completed.
            if callable(abort) and getattr(transaction, "closed", None) is not True:
                try:
                    abort()
                except BaseException:
                    pass
            raise TypeError("target-tree executor returned an invalid transaction")
        if type(transaction.closed) is not bool:
            try:
                transaction.abort()
            except BaseException:
                # Preserve the protocol diagnostic. The caller's authoritative
                # round snapshot remains the final recovery boundary.
                pass
            raise TypeError("target-tree transaction closed state must be boolean")
        if transaction.closed:
            raise RuntimeError("target-tree executor returned a closed transaction")
        if not (
            callable(getattr(transaction, "commit", None))
            or callable(getattr(transaction, "commit_paths", None))
        ):
            try:
                transaction.abort()
            except BaseException:
                pass
            raise TypeError("target-tree transaction has no commit operation")
        if (
            transaction.provider_protocol
            != provider.protocol
            or transaction.provider_revision_token
            != provider.source_revision_token
        ):
            try:
                transaction.abort()
            except BaseException:
                pass
            raise RuntimeError("target-tree transaction provider binding mismatch")
        try:
            event = provider.dispatch_metrics(
                len(cohort), transaction, cached=cached
            )
            if event.provider_protocol != provider.protocol:
                raise RuntimeError("target-tree dispatch provider identity mismatch")
            if type(event.cached) is not bool or event.cached is not cached:
                raise RuntimeError("target-tree dispatch cache selection mismatch")
            if type(event.executor_cached) is not bool:
                raise TypeError("target-tree executor cache metric must be boolean")
            if event.executor_cached and not event.cached:
                raise RuntimeError("uncached target-tree dispatch reported a cache hit")
            if type(event.physical_packed) is not bool:
                raise TypeError("target-tree packed-dispatch metric must be boolean")
            if publish_tree_metrics:
                self._apply_target_tree_dispatch_metrics(cohort, event)
            else:
                from .target_tree_executor import TargetTreeDispatchMetrics

                if not isinstance(event, TargetTreeDispatchMetrics):
                    raise TypeError(
                        "target-tree provider returned invalid dispatch metrics"
                    )
                if (
                    type(event.width) is not int
                    or event.width != len(cohort)
                    or event.width > getattr(provider, "ast_tree_max_sessions", 1)
                    or event.width > 1 and not event.physical_packed
                ):
                    raise ValueError(
                        "AST target-tree dispatch exceeds its admitted session shape"
                    )
                _bump(
                    self.scheduler_stats,
                    "external_ast_target_tree_provider_launches",
                )
                if event.width > 1:
                    _bump(self.scheduler_stats, "external_ast_target_tree_cohort_launches")
                    _bump(self.scheduler_stats, "external_ast_target_tree_cohort_sessions", event.width)
        except BaseException:
            # Provider callbacks are untrusted and may corrupt `closed` before
            # failing. Only exact True is sufficient evidence that abort is no
            # longer required.
            if (
                transaction is not None
                and getattr(transaction, "closed", None) is not True
            ):
                try:
                    transaction.abort()
                except BaseException:
                    pass
            raise
        return result

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
        if getattr(self, "ast_tree_tile_rows", 0):
            shared.update(
                (id(fields[name]), fields[name])
                for name in _LANE_AST_REPLACED if name in fields
            )
        # A graph carrying COW bookkeeping (an APC branch's hooked objects)
        # takes the descriptor route too: its segment tokens hold locks a
        # deep copy cannot pickle, as in ``snapshot_committed_cache``.
        try:
            captured = None
            if getattr(self, "ast_tree_tile_rows", 0):
                captured = self._ast_descriptor_capture(
                    [*fields["cache"], *fields["draft_cache"]], recovery=True
                )
            if captured is None:
                cache, (draft_cache, tail), borrowed = snapshot_recovery_descriptors(
                    fields["cache"], (fields["draft_cache"], fields["tail"])
                )
            else:
                clones, borrowed = captured
                split = len(fields["cache"])
                cache, draft_cache = clones[:split], clones[split:]
                tail = self.mx.stop_gradient(fields["tail"])
                _bump(self.scheduler_stats, "external_ast_lean_recovery_snapshots")
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

    def _snapshot_scheduler_stats(self):
        """Copy mutable histogram values as well as scalar counters.

        Round rollback previously used a shallow dictionary copy.  A failed
        tree round could therefore leak an in-place node-budget histogram bump
        even though every authoritative lane and scalar counter was restored.
        """
        snapshot = dict(self.scheduler_stats)
        for key, value in self.scheduler_stats.items():
            if isinstance(value, dict):
                snapshot[key] = copy.deepcopy(value)
        return snapshot

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

    def _draft_accepts_processors(self):
        # Inspect only a path that will actually supply processor keywords.
        # Lazy binding also supports host/source harnesses using __new__.
        from .call_contracts import KeywordSupportCache

        support = getattr(self, "_draft_processor_support", None)
        if support is None:
            support = self._draft_processor_support = KeywordSupportCache(
                "logits_processors"
            )
        return support.accepts(
            self.draft.draft_distributions,
            declared=bool(getattr(self.draft, "supports_logits_processors", False)),
        )

    @staticmethod
    def _progressive_processors_supported(lane):
        """Require an explicit law-and-side-effect-safe private-branch contract."""

        return processors_branch_safe(lane.processors)

    def _continuation_ast_layout(self, lanes, tokens, prefixes, widths, fallback):
        """Admit each session's feasible slate without narrowing its peers.

        This is a host-only projection of already produced paths. No target
        state or proposal pool selection is replaced by the row-local search.
        """
        if getattr(self, "ast_tree_tile_rows", 0):
            from .ast_tree_verify import project_tree_slate

            return project_tree_slate(self, lanes, tokens, widths)
        rows, actual_prefixes, provenance, search_diagnostics = [], [], [], []
        refused_sessions = 0
        for row, lane in enumerate(lanes):
            requested = int(widths[lane.uid])
            suggested = int(prefixes[row])
            chosen = None
            refused = False
            diagnostics = None
            for width in range(requested, 1, -1):
                try:
                    candidate = self.draft.ordered_alternative_proposals(
                        [tokens[row]], branch_count=width,
                        strategy="confidence_tree", shared_prefix_tokens=[suggested],
                        allow_adaptive_prefix_search=True, row_indices=[row],
                    )[0]
                except InsufficientContinuationAlternatives:
                    candidate = None
                details = self.draft.last_ordered_alternative_search_diagnostics
                if details:
                    diagnostics = copy.deepcopy(details[0])
                if candidate is None:
                    continue
                prefix = int(self.draft.last_ordered_alternative_prefix_tokens[0])
                if prefix == 0 and not self.progressive_ast_rootless_fallback:
                    refused = True
                    continue
                chosen = candidate
                branch_provenance = self.draft.last_ordered_alternative_provenance[0]
                break
            if diagnostics is not None:
                self._last_alternative_prefix_diagnostics[lane.uid] = diagnostics
            search_diagnostics.append(diagnostics)
            if refused:
                lane.progressive_ast_rootless_refusal_rounds = (
                    getattr(lane, "progressive_ast_rootless_refusal_rounds", 0) + 1
                )
                refused_sessions += 1
            if chosen is None:
                rows.append([list(tokens[row])])
                actual_prefixes.append(0)
                provenance.append(())
                fallback([lane], "adaptive_utility_latch" if requested == 1
                         else "rootless_refused" if refused else "insufficient_branches")
                continue
            rows.append(chosen)
            actual_prefixes.append(prefix)
            provenance.append(branch_provenance)
            topology = (
                "rootless_sibling_fanout" if prefix == 0 else
                "adaptive_shared_trunk" if prefix != suggested else "shared_trunk"
            )
            self._last_alternative_topology[lane.uid] = topology
            if prefix != suggested:
                self._last_alternative_prefix_search[lane.uid] = {
                    "suggested": suggested, "active": prefix,
                }
            if len(chosen) < requested:
                self._last_alternative_downshift[lane.uid] = {
                    "configured": requested, "active": len(chosen),
                    "reason": "insufficient_root_sibling_branches" if prefix == 0
                    else "insufficient_shared_prefix_branches",
                }
        self.draft.last_ordered_alternative_provenance = tuple(provenance)
        self.draft.last_ordered_alternative_prefix_tokens = tuple(actual_prefixes)
        self.draft.last_ordered_alternative_search_diagnostics = tuple(search_diagnostics)
        if refused_sessions:
            _bump(self.scheduler_stats, "external_progressive_ast_rootless_refusal_rounds")
            _bump(self.scheduler_stats, "external_progressive_ast_rootless_refusal_sessions", refused_sessions)
        return rows, tuple(actual_prefixes)

    def _propose(self, cohort, *, adaptive_depth=None, force_zero=False):
        """Draft phase: one proposal block per cohort row, ``None`` for no draft.

        A zero-count round only appends pending draft context so the draft
        plane stays paired with the target boundary.  A forced zero-count
        round (the cost-per-commit ordinary probe) defers that append: the
        pending context stays in ``lane.tail`` and ``_commit`` extends it, so
        the probe times only the target step the ordinary route performs.
        """
        self._last_alternative_blocks = {}
        self._last_alternative_prefix_tokens = {}
        self._last_alternative_latch = {}
        self._last_alternative_topology = {}
        self._last_alternative_downshift = {}
        self._last_alternative_prefix_search = {}
        self._last_alternative_prefix_diagnostics = {}
        ast_planner = getattr(self, "ast_execution_planner", None)
        source_plans = {}

        def record_ast_primary_fallback(lanes, reason):
            _bump(
                self.scheduler_stats,
                "external_progressive_ast_primary_fallback_rounds",
                len(lanes),
            )
            _bump(
                self.scheduler_stats,
                f"external_progressive_ast_primary_fallback_{reason}_rounds",
                len(lanes),
            )
            for lane in lanes:
                # This round executed only the exact primary chain.  Keep
                # cumulative AST history, but do not expose a proposal-only
                # branch width as the last observed execution shape.
                lane.progressive_alternative_last_active_branches = 1
                lane.progressive_ast_primary_fallback_rounds = (
                    getattr(lane, "progressive_ast_primary_fallback_rounds", 0)
                    + 1
                )
                reasons = getattr(
                    lane, "progressive_ast_primary_fallback_reasons", {}
                )
                reasons[reason] = reasons.get(reason, 0) + 1
                lane.progressive_ast_primary_fallback_reasons = reasons
                lane.progressive_ast_last_primary_fallback_reason = reason
                prefix_diagnostics = self._last_alternative_prefix_diagnostics.get(
                    lane.uid
                )
                if prefix_diagnostics is not None:
                    lane.progressive_ast_last_proposed_prefix_diagnostics = (
                        copy.deepcopy(prefix_diagnostics)
                    )
        # Host doubles bind _propose without __init__; off unless configured.
        branches = getattr(self, "progressive_alternative_branches", 1)
        if (
            branches > 1
            and len(cohort) != 1
            and getattr(self, "progressive_cross_session_pack_rows", None) is None
            and not (len(cohort) <= getattr(self, "_ast_tree_cohort_limit", lambda: 1)())
        ):
            raise ValueError(
                "ordered alternative proposals require one request lane or "
                "explicit cross-session packing"
            )
        tree_budget = (
            self._tree_node_budget() if self.draft_topology == "tree15" else None
        )
        chain_cap = None if tree_budget is not None else self._chain_draft_cap()
        requested_count = min(
            tree_budget if tree_budget is not None
            else self.num_draft if chain_cap is None
            else min(self.num_draft, chain_cap),
            (max if ast_planner is not None else min)(l.maximum-l.generated-1 for l in cohort),
        )
        multilane_draft_cap = getattr(self, "multilane_draft_cap", None)
        active_width = 1
        if multilane_draft_cap is not None:
            lanes = getattr(self, "lanes", None)
            active_width = (
                sum(not lane.cancelled for lane in lanes.values())
                if lanes is not None
                else len(cohort)
            )
        # Preserve the long B1 chain, but reduce the fixed-verifier row slab
        # while multiple lanes compete for the same bounded headroom.  The
        # explicit policy is identity-bound and this branch is receipt-counted.
        if (
            multilane_draft_cap is not None
            and active_width > 1
            and requested_count > multilane_draft_cap
        ):
            requested_count = multilane_draft_cap
            _bump(self.scheduler_stats, "external_multilane_draft_cap_rounds")
            _bump(
                self.scheduler_stats,
                "external_multilane_draft_cap_lanes",
                len(cohort),
            )
        if tree_budget is not None:
            self.scheduler_stats["external_tree_node_budget_last"] = requested_count
            histogram = self.scheduler_stats["external_tree_node_budget_histogram"]
            key = str(requested_count)
            histogram[key] = int(histogram.get(key, 0)) + len(cohort)
        if force_zero:
            requested_count = 0
        elif adaptive_depth is not None:
            adaptive_count = min(requested_count, int(adaptive_depth))
            requested_count = self._proposal_depth(adaptive_count, requested_count)
        if (all if ast_planner is not None else any)(l.ordinary or _speculation_parked(l) for l in cohort):
            requested_count = 0
        if getattr(self, "ast_b1_tile_policy", None) == "four_path15" and requested_count:
            if (len(cohort) != 1 or float(cohort[0].sampling.get("sampling_temp", 0)) != 0.0
                or any(not getattr(processor, "history_pure", False)
                       or not callable(getattr(processor, "forbidden_token_ids_at_length", None))
                       for processor in cohort[0].processors)):
                raise ValueError("four_path15 requires one unconstrained greedy request")
            requested_count = 15
        if getattr(self, "ast_packed_pattern", None) and requested_count:
            if len(cohort) != 1 or float(cohort[0].sampling.get("sampling_temp", 0)) != 0.0:
                raise ValueError("packed_pattern requires one greedy request")
            requested_count = max(self.ast_packed_pattern)
        alternative_round = branches > 1 and not (
            (
                self.progressive_alternative_strategy == "paired_continuations"
                and requested_count
                <= int(self.progressive_verification_tile or 0) + 1
            )
            or (
                self.progressive_alternative_strategy == "confidence_tree"
                and requested_count < 2
            )
            or (
                # The shared prefix diverges at the tile; a short terminal
                # round has no proposal position there to branch from.
                self.progressive_alternative_strategy == "shared_prefix"
                and requested_count
                <= int(self.progressive_verification_tile or 0)
            )
        )
        if alternative_round and requested_count and any(
            float(l.sampling.get("sampling_temp", 0)) != 0.0 for l in cohort
        ):
            # Alternative verification is greedy-only.  A sampled lane keeps
            # the primary progressive chain, counted so the receipt separates
            # the configured option from the rounds that actually used it.
            alternative_round = False
            _bump(
                self.scheduler_stats,
                "external_progressive_alternative_sampled_primary_rounds",
            )
            for lane in cohort:
                lane.progressive_alternative_sampled_primary_rounds = (
                    getattr(lane, "progressive_alternative_sampled_primary_rounds", 0)
                    + 1
                )
        if alternative_round and any(
            not self._progressive_processors_supported(lane) for lane in cohort
        ):
            alternative_round = False
            if getattr(self, "continuation_pool_ast_bridge", False):
                record_ast_primary_fallback(cohort, "unsupported_processors")
            # Name the processor kinds that blocked the branches, so a receipt
            # says which request feature kept AST on its primary chain.
            for lane in cohort:
                for processor in lane.processors:
                    if not processors_branch_safe([processor]):
                        kind = type(processor).__name__
                        if kind == "function":
                            kind = getattr(processor, "__qualname__", kind)
                        _bump(
                            self.scheduler_stats,
                            "external_progressive_unsupported_processor:"
                            + str(kind)[:64],
                        )
        # Alternative rows are valid only for the proposal round that created
        # them.  Clear the prior round before deciding whether this round has
        # enough suffix depth to branch; short terminal paired rounds then use
        # the primary progressive chain instead of replaying stale branches.
        for lane in cohort:
            self._last_alternative_blocks.pop(lane.uid, None)
            self._last_alternative_prefix_tokens.pop(lane.uid, None)
        if (
            getattr(self, "continuation_pool_ast_bridge", False)
            and ast_planner is None
            and active_width > 1
            and self.progressive_cross_session_pack_rows is not None
            and requested_count
            and requested_count <= int(self.progressive_verification_tile or 0)
            and all(
                float(lane.sampling.get("sampling_temp", 0)) == 0.0
                for lane in cohort
            )
        ):
            alternative_round = False
            record_ast_primary_fallback(cohort, "depth_not_above_tile")
        blocks = [None]*len(cohort)
        if not requested_count:
            if force_zero:
                return blocks
            for lane in cohort:
                if lane.tail.shape[1] and not self._defer_ast_parked_context(lane):
                    self._append_context(lane, lane.anchor)
            return blocks
        # Generators built without __init__ (tests, restored owners) predate
        # the flag; it is opt-in, so a missing attribute means off.
        if getattr(self, "draft_tail_equalize", False) and len(cohort) > 1:
            self._equalize_draft_tails([
                lane for lane in cohort
                if not self._defer_ast_parked_context(lane)
            ])
        if ast_planner is not None:
            from .ast_planner import RequestWork

            work = []
            for lane in cohort:
                depth = max(0, min(requested_count, lane.maximum - lane.generated - 1))
                if lane.ordinary or _speculation_parked(lane):
                    depth = 0
                choice = (self.progressive_topology_latch.session(lane.uid).choose(
                    0, max(1, depth), self.progressive_verification_tile)
                    if self.progressive_topology_latch is not None else None)
                # Preview before source work; commit alone advances the latch.
                if choice is not None:
                    self._last_alternative_latch[lane.uid] = dict(choice.__dict__)
                # The planner owns branch width from its learned uplift; the
                # latch contributes only its reprobe clock and trunk hint.
                work.append(RequestWork(
                    int(lane.uid), depth, len(lane.history),
                    branches if alternative_round else 1,
                    ast_planner.acceptance(int(lane.uid)), choice.reprobe if choice is not None else False,
                    trunk=int(choice.trunk_tokens) if choice is not None else 1,
                ))
            source_plans = (
                {} if getattr(self, "ast_tree_tile_rows", 0)
                else {plan.uid: plan for plan in ast_planner.source_budgets(work)}
            )
            if getattr(self, "ast_tree_tile_rows", 0):
                from .ast_planner import SourceBudget

                # Exact parent-tree economics are measured separately from
                # the rectangular verifier's launch curve.
                source_plans = {
                    item.uid: SourceBudget(
                        item.uid,
                        item.maximum_depth,
                        min(item.branch_limit, self.continuation_policy.limit, 4),
                        min(
                            60,
                            max(
                                15,
                                item.maximum_depth * min(item.branch_limit, 4),
                            ),
                        ),
                        self.progressive_verification_tile,
                        0.0,
                        0.0,
                        "fixed_tree_source_budget",
                    )
                    for item in work
                }
                if getattr(self, "ast_packed_pattern", None):
                    from .ast_tree_tuning import ASTTreeSourcePlan

                    lane = cohort[0]
                    depth, branches = max(self.ast_packed_pattern), len(self.ast_packed_pattern)
                    plan = ASTTreeSourcePlan(depth, branches, 16, 1, 1, 1,
                                            "fixed_packed_proposal_tile", 0,
                                            lane.maximum - lane.generated < depth + 1, 1)
                    lane.ast_tree_source_plan = plan
                    lane.ast_tree_tuning = {"policy": "packed_pattern", "fixed": True,
                                            "pattern": self.ast_packed_pattern, "defaulted": False}
                    source_plans[int(lane.uid)] = SourceBudget(int(lane.uid), depth, branches,
                                                               15, depth, 0.0, 0.0, plan.reason)
                elif getattr(self, "ast_b1_tile_policy", None) == "four_path15":
                    from .ast_tree_tuning import ASTTreeSourcePlan

                    lane = cohort[0]
                    plan = ASTTreeSourcePlan(15, 4, 16, 1, 1, 1,
                                             "fixed_b1_horizon15_four_path_tile", 0,
                                             lane.maximum - lane.generated < 16, 1)
                    lane.ast_tree_source_plan = plan
                    lane.ast_tree_tuning = {"policy": "four_path15", "fixed": True,
                                            "defaulted": False, "optimality_claim": False}
                    source_plans[int(lane.uid)] = SourceBudget(int(lane.uid), 15, 4, 60, 15,
                                                               0.0, 0.0, plan.reason)
                elif self.progressive_verification_tile_policy == "auto":
                    from .ast_tree_tuning import ASTTreeTuner

                    for lane, item in zip(cohort, work, strict=True):
                        if item.maximum_depth == 0:
                            lane.ast_tree_source_plan = None
                            continue
                        tuner = self._ast_tree_tuners.get(item.uid)
                        if tuner is None:
                            tuner = self._ast_tree_tuners[item.uid] = ASTTreeTuner(
                                (ast_planner.profile.binding, item.uid),
                                max_depth=self.num_draft,
                                minimum_depth=self.minimum_draft_proposals,
                                max_branches=min(self.continuation_policy.limit, 4),
                                max_tile_rows=self.ast_tree_tile_rows,
                            )
                        plan = tuner.plan(
                            remaining_tokens=item.maximum_depth + 1,
                            ready_sessions=max(self._ast_tree_ready_sessions, len(cohort)),
                            target_sessions=min(len(cohort), self._ast_tree_cohort_limit()),
                        )
                        lane.ast_tree_source_plan = plan
                        lane.ast_tree_tuning = tuner.receipt()
                        source_plans[item.uid] = SourceBudget(
                            item.uid, plan.depth, plan.branches,
                            plan.depth * plan.branches, plan.max_tile_rows - 1,
                            0.0, 0.0, plan.reason,
                        )
            for lane in cohort:
                lane.ast_source_budget = source_plans[int(lane.uid)].receipt()
                _bump_map(
                    self.scheduler_stats, "ast_planner_source_budgets",
                    str(source_plans[int(lane.uid)].reason),
                )
                if lane.uid in self._last_alternative_latch:
                    plan = source_plans[int(lane.uid)]
                    self._last_alternative_latch[lane.uid].update(
                        planner_branches=plan.branches,
                        planner_depth=plan.depth,
                        planner_reason=plan.reason,
                    )
        groups = {}
        for row,lane in enumerate(cohort):
            depth = source_plans[int(lane.uid)].depth if ast_planner is not None else requested_count
            groups.setdefault((int(lane.tail.shape[1]),str(lane.tail.dtype),depth),[]).append(row)
        _bump(self.scheduler_stats, "external_draft_cohort_calls", len(groups))
        tail_groups = len({(tail, dtype) for tail, dtype, _depth in groups})
        if any(tail == 0 for tail, _dtype, _depth in groups) and any(
            tail > 0 for tail, _dtype, _depth in groups
        ):
            _bump(self.scheduler_stats, "external_draft_zero_tail_split_rounds")
            _bump(
                self.scheduler_stats,
                "external_draft_zero_tail_split_extra_calls",
                tail_groups - 1,
            )
        if tail_groups > 1:
            _bump(self.scheduler_stats, "external_draft_tail_split_rounds")
            _bump(self.scheduler_stats, "external_draft_tail_split_extra_calls", tail_groups - 1)
        if len(groups) > tail_groups:
            _bump(self.scheduler_stats, "external_draft_depth_split_rounds")
            _bump(self.scheduler_stats, "external_draft_depth_split_extra_calls", len(groups) - tail_groups)
        cohort_alternative_round = alternative_round
        for (_tail, _dtype, group_depth), indices in groups.items():
            # A previous tail/dtype group's parked latch is not this group's
            # admission decision.
            alternative_round = cohort_alternative_round
            session_alternative_branches = None
            lanes = [cohort[i] for i in indices]
            requested_count = group_depth
            if self.progressive_alternative_strategy == "confidence_tree" and requested_count < 2:
                alternative_round = False
            if not requested_count:
                for lane in lanes:
                    if lane.tail.shape[1] and not self._defer_ast_parked_context(lane):
                        self._append_context(lane, lane.anchor)
                continue
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
                extra = {"context_tokens": pairing} if self.pair_context_tokens else {}
                if getattr(self, "ast_b1_tile_policy", None) is not None:
                    extra["ast_b1_tile_policy"] = self.ast_b1_tile_policy
                if ast_planner is not None:
                    extra["ast_source_budgets"] = [
                        source_plans[int(lane.uid)].branches
                        if getattr(self, "ast_tree_tile_rows", 0)
                        else 1 if source_plans[int(lane.uid)].branches == 1
                        else source_plans[int(lane.uid)].node_budget for lane in lanes
                    ]
                    if getattr(self, "ast_tree_tile_rows", 0):
                        extra["ast_tree_node_budgets"] = [
                            source_plans[int(lane.uid)].node_budget for lane in lanes
                        ]
                if getattr(self.draft, "requires_proposal_contexts", False):
                    from .proposal_providers import continuation_context_revision
                    extra["proposal_contexts"] = [{
                        "request_id": l.proposal_request_id,
                        "round_id": l.external_rounds,
                        "context_revision": continuation_context_revision(self.draft.session.session_revision, l.history, l.anchor),
                        "session_scope_hash": l.session_scope_hash,
                    } for l in lanes]
                needs_histories = bool(getattr(self.draft, "requires_processor_histories", False))
                if needs_histories:
                    extra["processor_histories"] = [list(l.history) for l in lanes]
                if getattr(self.draft, "requires_cohort_width", False):
                    # Draft groups split a cohort by tail length; B1-only
                    # proposal modes must see the verify cohort's width.
                    extra["cohort_width"] = len(cohort)
                if getattr(self.draft, "requires_proposal_score_states", False):
                    extra["proposal_score_states"] = [
                        lane.proposal_score_counts for lane in lanes
                    ]
                if getattr(self.draft, "requires_proposal_trust_states", False):
                    extra["proposal_trust_states"] = [
                        lane.proposal_trust_states for lane in lanes
                    ]
                if self.draft_topology == "tree15":
                    if self.pair_context_tokens and not getattr(
                        self.draft, "tree_topology_aware", False
                    ):
                        raise ValueError("tree15 is unavailable for paired-context drafters")
                    forbidden = [self._tree_forbidden(lane) for lane in lanes]
                    trees = None
                    if len(lanes) == 1:
                        trees = self._adopt_prelaunched(
                            lanes[0], requested_count, forbidden[0]
                        )
                    else:
                        trees = self._adopt_prelaunched_cohort(
                            lanes, requested_count, forbidden
                        )
                    if trees is None:
                        tree_options = self._tree_options(count_hits=True, lanes=lanes)
                        if self._tree_clock is not None:
                            self._tree_clock("tree_draft_setup")
                            tree_options["mark"] = self._tree_clock
                        if self.pair_context_tokens:
                            tree_options["context_tokens"] = pairing
                        if needs_histories and getattr(
                            self.draft, "tree_topology_aware", False
                        ):
                            tree_options["processor_histories"] = extra[
                                "processor_histories"
                            ]
                        trees = self.draft.propose_tree(
                            *arguments[:4], forbidden_token_ids=forbidden, **tree_options
                        )
                    tokens = [tree[0] for tree in trees]
                    q = [
                        TreeDraftRow(
                            tree[0], tree[1], len(lanes),
                            info=tree[2] if len(tree) > 2 else None,
                        )
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
                    # Request-bound histories/contexts are part of proposal
                    # production, not merely processor metadata.  Bypassing
                    # their wrapper through the backend's compact block API
                    # would skip source arbitration and leave its row receipts
                    # empty or stale.
                    and not extra
                ):
                    # Stateless forbidden-token processors can mask the pair
                    # table without a per-position host read. Other processor
                    # kinds and pairing (EAGLE) drafters keep the sequential
                    # path.
                    tokens, q = self._propose_pairwise(lanes, arguments)
                elif any(processors) and self._draft_accepts_processors():
                    tokens,q = self.draft.draft_distributions(
                        *arguments,
                        logits_processors=processors,
                        **{**extra, "processor_histories": [list(l.history) for l in lanes]},
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
            deterministic_proposals = (
                getattr(self.draft, "proposal_distribution", None)
                == "deterministic_point_mass"
            )
            confidence = None
            if deterministic_proposals:
                confidence = getattr(self.draft, "adaptive_confidence_features", None)
                if confidence is not None and len(confidence) != len(lanes):
                    raise ValueError("deterministic drafter confidence rows mismatch")
            deterministic_admission = (
                self.adaptive_policy is not None
                and self.adaptive_policy.mode == "per_request"
                and deterministic_proposals
            )
            sources = getattr(self.draft, "last_proposal_sources", None)
            scored_composition = hasattr(self.draft, "last_proposal_score_keys")
            score_keys = getattr(self.draft, "last_proposal_score_keys", None)
            scores = getattr(self.draft, "last_proposal_scores", None)
            trust_keys = getattr(self.draft, "last_proposal_trust_keys", None)
            trusted_rows = getattr(self.draft, "last_proposal_trusted", None)
            audit_rows = getattr(self.draft, "last_proposal_audits", None)
            periodic_audit_rows = getattr(
                self.draft, "last_proposal_periodic_audits", None
            )
            selections = getattr(self.draft, "last_continuation_selections", None) if self.continuation_policy is not None else None
            admissions = getattr(self.draft, "last_continuation_admissions", None) if self.continuation_policy is not None else None
            slates = getattr(self.draft, "last_continuation_slates", None) if self.continuation_policy is not None else None
            source_diagnostics = getattr(
                self.draft, "last_continuation_source_diagnostics", None
            ) if self.continuation_policy is not None else None
            payloads = getattr(self.draft, "draft_feedback_payloads", None)
            if selections is not None and len(selections) != len(lanes):
                raise ValueError("continuation selection rows mismatch")
            if admissions is not None and len(admissions) != len(lanes):
                raise ValueError("continuation admission rows mismatch")
            if slates is not None and len(slates) != len(lanes):
                raise ValueError("continuation slate rows mismatch")
            if source_diagnostics is not None and len(source_diagnostics) != len(lanes):
                raise ValueError("continuation source diagnostic rows mismatch")
            if source_diagnostics is not None:
                for lane, diagnostics in zip(lanes, source_diagnostics, strict=True):
                    lane.continuation_source_geometry_last = copy.deepcopy(diagnostics)
                    totals = getattr(lane, "continuation_source_geometry_totals", {})
                    for source_id, geometry in diagnostics.items():
                        source_totals = totals.setdefault(
                            source_id,
                            {
                                "calls": 0,
                                "nodes": 0,
                                "roots": 0,
                                "leaves": 0,
                                "full_depth_nodes": 0,
                                "full_depth_leaves": 0,
                                "depth_hist": {},
                                "leaf_depth_hist": {},
                            },
                        )
                        source_totals["calls"] += 1
                        for key in (
                            "nodes",
                            "roots",
                            "leaves",
                            "full_depth_nodes",
                            "full_depth_leaves",
                        ):
                            source_totals[key] += int(geometry.get(key, 0))
                        for hist_key in ("depth_hist", "leaf_depth_hist"):
                            target_hist = source_totals[hist_key]
                            for bucket, count in geometry.get(hist_key, {}).items():
                                target_hist[str(bucket)] = (
                                    target_hist.get(str(bucket), 0) + int(count)
                                )
                    lane.continuation_source_geometry_totals = totals
            if selections is not None:
                self._continuation_open_selections.extend(selection for selection in selections if selection is not None)
            if payloads is not None and len(payloads) != len(lanes):
                raise ValueError("draft feedback payload rows mismatch")
            if sources is not None and (
                    len(sources) != len(lanes)
                    or any(type(source) is not str or source not in {"prompt_lookup", "native_mtp", "external"}
                           for source in sources)):
                raise ValueError("invalid composed proposal source rows")
            if scored_composition and (
                score_keys is None
                or scores is None
                or len(score_keys) != len(lanes)
                or len(scores) != len(lanes)
                or trust_keys is None
                or len(trust_keys) != len(lanes)
                or trusted_rows is None
                or audit_rows is None
                or periodic_audit_rows is None
                or len(trusted_rows) != len(lanes)
                or len(audit_rows) != len(lanes)
                or len(periodic_audit_rows) != len(lanes)
                or any(type(key) is not str or not key for key in score_keys)
                or any(not np.isfinite(score) for score in scores)
                or any(key is not None and (type(key) is not str or not key) for key in trust_keys)
                or any(type(value) is not bool for value in trusted_rows)
                or any(type(value) is not bool for value in audit_rows)
                or any(type(value) is not bool for value in periodic_audit_rows)
                or any(audit and trusted for audit, trusted in zip(audit_rows, trusted_rows))
                or any(periodic and not audit for periodic, audit in zip(periodic_audit_rows, audit_rows))
            ):
                raise ValueError("invalid composed proposal score rows")
            alternatives = None
            alternative_prefixes = None
            active_alternative_branches = getattr(self, "progressive_alternative_branches", 1)
            if ast_planner is not None and self.progressive_alternative_strategy != "confidence_tree":
                active_alternative_branches = min(
                    source_plans[int(lane.uid)].branches for lane in lanes)
                alternative_round = alternative_round and active_alternative_branches > 1
            latch_decisions = None
            if alternative_round and getattr(self, "ast_packed_pattern", None):
                from .ast_tree_verify import project_tree_slate

                alternatives, alternative_prefixes = project_tree_slate(
                    self, lanes, tokens, {int(lane.uid): len(self.ast_packed_pattern) for lane in lanes})
            elif alternative_round:
                if self.progressive_alternative_strategy == "confidence_tree":
                    if confidence is None:
                        raise ValueError(
                            "confidence tree requires per-token confidence features"
                        )
                    # This is each source plan's trunk limit, not the
                    # server-wide maximum tile. A one-column physical tile
                    # may still execute a one-token shared trunk separately;
                    # the confidence helper's two-input minimum encodes that
                    # anchor-plus-child semantic contract, not a launch size.
                    alternative_tiles = tuple(
                        max(2, source_plans[int(lane.uid)].tile)
                        if ast_planner is not None else int(self.progressive_verification_tile or 0)
                        for lane in lanes)
                    alternative_prefixes = tuple(
                        confidence_tree_trunk_length(
                            confidence[row],
                            len(tokens[row]),
                            alternative_tiles[row],
                        )
                        for row in range(len(lanes))
                    )
                    latch_decisions = tuple(
                        self.progressive_topology_latch.session(lane.uid).choose(
                            prefix, len(row), tile)
                        for lane, prefix, row, tile in zip(
                            lanes, alternative_prefixes, tokens, alternative_tiles, strict=True))
                    # Each session owns its width. Complete-path slates are
                    # projected row-locally; the generic logits provider uses
                    # the widest slate and trims before target verification.
                    session_alternative_branches = {
                        int(lane.uid): source_plans[int(lane.uid)].branches
                        if ast_planner is not None else int(decision.active_branches)
                        for lane, decision in zip(lanes, latch_decisions, strict=True)
                    }
                    active_alternative_branches = max(
                        session_alternative_branches.values()
                    )
                    suggested_alternative_prefixes = tuple(
                        decision.trunk_tokens for decision in latch_decisions
                    )
                    if self.progressive_alternative_depth_policy == "confidence_tail":
                        # A verification tile counts its current anchor.  Keep
                        # enough room for one distinct child while preserving
                        # the requested physical tile width.  For tile4 this
                        # locks the shared trunk to at most two proposals, so
                        # the first suffix launch is 4+4 rather than a five-row
                        # terminal slab when three confidence features are
                        # strong.
                        alternative_prefixes = tuple(
                            min(value, max(1, tile - 2))
                            for value, tile in zip(suggested_alternative_prefixes, alternative_tiles, strict=True)
                        )
                    else:
                        alternative_prefixes = suggested_alternative_prefixes
                    alternative_round = active_alternative_branches > 1
                    for lane, decision, suggested, actual in zip(
                        lanes,
                        latch_decisions,
                        suggested_alternative_prefixes,
                        alternative_prefixes,
                        strict=True,
                    ):
                        payload = dict(decision.__dict__)
                        if ast_planner is not None:
                            plan = source_plans[int(lane.uid)]
                            payload.update(
                                planner_branches=plan.branches,
                                planner_depth=plan.depth,
                                planner_reason=plan.reason,
                            )
                        payload["suggested_trunk_tokens"] = int(suggested)
                        payload["trunk_tokens"] = int(actual)
                        payload["snapshot_before"] = (
                            self.progressive_topology_latch.snapshot(lane.uid)
                        )
                        self._last_alternative_latch[lane.uid] = payload
                    downshifted = [
                        lane for lane in lanes
                        if session_alternative_branches[int(lane.uid)]
                        < self.progressive_alternative_branches
                    ]
                    if downshifted:
                        _bump(
                            self.scheduler_stats,
                            "external_progressive_ast_utility_downshift_rounds",
                        )
                        _bump(
                            self.scheduler_stats,
                            "external_progressive_ast_utility_downshift_sessions",
                            len(downshifted),
                        )
                    if active_alternative_branches == 1:
                        record_ast_primary_fallback(
                            lanes, "adaptive_utility_latch"
                        )
                    elif any(
                        width == 1 for width in session_alternative_branches.values()
                    ):
                        _bump(
                            self.scheduler_stats,
                            "external_progressive_ast_mixed_width_rounds",
                        )
                if alternative_round:
                    if (
                        getattr(self, "continuation_pool_ast_bridge", False)
                        and self.progressive_alternative_strategy == "confidence_tree"
                    ):
                        alternatives, alternative_prefixes = self._continuation_ast_layout(
                            lanes, tokens, alternative_prefixes,
                            session_alternative_branches, record_ast_primary_fallback,
                        )
                    else:
                        try:
                            alternatives = ordered_alternative_proposals(
                                self.draft, tokens,
                                branch_count=active_alternative_branches,
                                strategy=self.progressive_alternative_strategy,
                                shared_prefix_tokens=(
                                    alternative_prefixes if alternative_prefixes is not None
                                    else self.progressive_verification_tile
                                    if self.progressive_alternative_strategy
                                    in {"shared_prefix", "paired_continuations"} else None
                                ),
                            )
                        except InsufficientContinuationAlternatives:
                            if not getattr(self, "continuation_pool_ast_bridge", False):
                                raise
                            alternative_round = False
                            record_ast_primary_fallback(lanes, "insufficient_branches")
                        else:
                            if getattr(self, "continuation_pool_ast_bridge", False):
                                alternative_prefixes = self.draft.last_ordered_alternative_prefix_tokens
                            for lane in lanes:
                                self._last_alternative_topology[lane.uid] = "shared_trunk"
            for j,row in enumerate(indices):
                feature_row = None
                if deterministic_admission:
                    # Every deterministic row must uphold its law contract,
                    # including PLD rows without a current confidence proxy.
                    # Their lagged admission still happens after proposal work.
                    if isinstance(q[j], CompactDraftRow):
                        row_laws = list(zip(q[j].candidate_ids, q[j].candidate_probs))
                    else:
                        row_laws = [(None, law) for law in q[j]]
                    if len(row_laws) != len(tokens[j]):
                        raise ValueError("deterministic proposal law lengths mismatch")
                    for token, (ids, law) in zip(tokens[j], row_laws):
                        if isinstance(law, PointMassLaw):
                            if law.token != int(token):
                                raise ValueError(
                                    "sparse point-mass law differs from proposal token"
                                )
                            continue
                        law = probability(law)
                        index = int(token)
                        if ids is not None:
                            positions = np.flatnonzero(np.asarray(ids) == index)
                            if len(positions) != 1:
                                raise ValueError("deterministic proposal token missing from compact law")
                            index = int(positions[0])
                        if not 0 <= index < len(law) or law[index] != 1.0 or np.count_nonzero(law) != 1:
                            raise ValueError("current confidence requires deterministic point-mass proposals")
                if confidence is not None and confidence[j] is not None:
                    feature_row = np.asarray(confidence[j], dtype=np.float64)
                    if (feature_row.ndim != 1 or len(feature_row) < len(tokens[j])
                            or not np.isfinite(feature_row).all()):
                        raise ValueError("invalid deterministic drafter confidence features")
                    feature_row = np.clip(feature_row[:len(tokens[j])], -40.0, 40.0).tolist()
                blocks[row] = (
                    q[j] if isinstance(q[j], (CompactDraftRow, TreeDraftRow))
                    else HostDraftRow(tokens[j], q[j], len(lanes), feature_row)
                )
                if alternative_round and (
                    not getattr(self, "continuation_pool_ast_bridge", False)
                    or len(alternatives[j]) > 1
                ):
                    active_alternative_branches = len(alternatives[j])
                    provenance_rows = getattr(
                        self.draft,
                        "last_ordered_alternative_provenance",
                        None,
                    )
                    if (
                        alternatives is None
                        or len(alternatives) != len(lanes)
                        or len(alternatives[j])
                        != active_alternative_branches
                    ):
                        raise ValueError("drafter did not publish ordered alternatives")
                    if getattr(self, "continuation_pool_ast_bridge", False) and (
                        provenance_rows is None
                        or len(provenance_rows) != len(lanes)
                        or len(provenance_rows[j])
                        != active_alternative_branches
                    ):
                        raise ValueError(
                            "AST bridge drafter did not publish branch provenance"
                        )
                    row_blocks = []
                    if self.progressive_alternative_depth_policy == "confidence_tail":
                        branch_prefix = (
                            int(alternative_prefixes[j])
                            if alternative_prefixes is not None
                            else int(self.progressive_verification_tile or 0)
                        )
                        branch_depths = (
                            tuple(
                                len(branch_tokens)
                                for branch_tokens in alternatives[j]
                            )
                            if getattr(self, "continuation_pool_ast_bridge", False)
                            and self.progressive_alternative_strategy
                            == "confidence_tree"
                            and branch_prefix == 0
                            else confidence_tree_branch_depths(
                                len(tokens[j]),
                                active_alternative_branches,
                                branch_prefix,
                            )
                        )
                    elif getattr(self, "ast_tree_tile_rows", 0):
                        branch_depths = tuple(
                            len(branch_tokens) for branch_tokens in alternatives[j]
                        )
                    else:
                        branch_depths = (
                            (len(tokens[j]),) * active_alternative_branches
                        )
                    admitted_depths = tuple(
                        min(len(branch_tokens), branch_depths[branch_index])
                        for branch_index, branch_tokens in enumerate(
                            alternatives[j]
                        )
                    )
                    for branch_index, branch_tokens in enumerate(alternatives[j]):
                        branch_tokens = list(
                            branch_tokens[: admitted_depths[branch_index]]
                        )
                        maximum_branch_depth = (
                            requested_count
                            if getattr(self, "continuation_pool_ast_bridge", False)
                            and self.progressive_alternative_strategy
                            == "confidence_tree"
                            and int(alternative_prefixes[j]) == 0
                            else len(tokens[j])
                        )
                        if not 1 <= len(branch_tokens) <= maximum_branch_depth:
                            raise ValueError("ordered alternative proposal depth is invalid")
                        branch_laws = []
                        for position, token in enumerate(branch_tokens):
                            if getattr(self, "continuation_pool_ast_bridge", False):
                                if not 0 <= int(token) < self._target_args.vocab_size:
                                    raise ValueError(
                                        "AST bridge token is outside the target vocabulary"
                                    )
                                law = PointMassLaw(int(token))
                            elif isinstance(q[j][position], PointMassLaw):
                                law = PointMassLaw(int(token))
                            else:
                                law = np.zeros_like(
                                    np.asarray(q[j][position], dtype=np.float64)
                                )
                                if not 0 <= int(token) < len(law):
                                    raise ValueError(
                                        "ordered alternative token is outside proposal law"
                                    )
                                law[int(token)] = 1.0
                            branch_laws.append(law)
                        row_blocks.append(
                            HostDraftRow(
                                list(branch_tokens),
                                branch_laws,
                                active_alternative_branches,
                                continuation_selection=(
                                    None if selections is None else selections[j]
                                ),
                                continuation_admission=(
                                    None if admissions is None else admissions[j]
                                ),
                                continuation_slate=(
                                    None if slates is None else slates[j]
                                ),
                                continuation_ast_provenance=copy.deepcopy(
                                    provenance_rows[j][branch_index]
                                    if getattr(self, "continuation_pool_ast_bridge", False)
                                    else None
                                ),
                                continuation_ast_admitted_provenance=copy.deepcopy(
                                    provenance_rows[j]
                                    if getattr(self, "continuation_pool_ast_bridge", False)
                                    else None
                                ),
                                continuation_ast_branch_index=(
                                    branch_index
                                    if getattr(self, "continuation_pool_ast_bridge", False)
                                    else None
                                ),
                                continuation_ast_admitted_depths=(
                                    admitted_depths
                                    if getattr(self, "continuation_pool_ast_bridge", False)
                                    else None
                                ),
                                feedback_payload=(
                                    None if payloads is None else payloads[j]
                                ),
                            )
                        )
                    if row_blocks[0].tokens != list(tokens[j]):
                        raise ValueError(
                            "ordered alternative B1 differs from primary proposal"
                        )
                    own_width = (
                        session_alternative_branches.get(int(lanes[j].uid))
                        if session_alternative_branches is not None
                        else None
                    )
                    if own_width is not None and own_width < len(row_blocks):
                        row_blocks = row_blocks[:own_width]
                    self._last_alternative_blocks[lanes[j].uid] = row_blocks
                    self._last_alternative_prefix_tokens[lanes[j].uid] = (
                        int(alternative_prefixes[j])
                        if alternative_prefixes is not None
                        else int(self.progressive_verification_tile or 0)
                    )
                if sources is not None:
                    if not isinstance(blocks[row], (HostDraftRow, CompactDraftRow)):
                        raise ValueError("composition requires supported chain proposal blocks")
                    blocks[row].proposal_source = sources[j]
                    if scored_composition:
                        blocks[row].proposal_score_key = score_keys[j]
                        blocks[row].proposal_score = float(scores[j])
                        blocks[row].proposal_trust_key = trust_keys[j]
                        blocks[row].proposal_trusted = trusted_rows[j]
                        blocks[row].proposal_audit = audit_rows[j]
                        blocks[row].proposal_periodic_audit = periodic_audit_rows[j]
                if selections is not None:
                    if not isinstance(blocks[row], HostDraftRow) or selections[j] is None:
                        raise ValueError("continuation requires host complete-path selections")
                    blocks[row].continuation_selection = selections[j]
                    blocks[row].continuation_admission = admissions[j]
                    blocks[row].continuation_slate = slates[j]
                if payloads is not None and isinstance(blocks[row], HostDraftRow):
                    blocks[row].feedback_payload = payloads[j]
        if getattr(self, "continuation_pool_ast_bridge", False) and self._last_alternative_blocks:
            # Feasibility is request-local. A peer without alternatives joins
            # the physical frontier as its exact primary chain, not as a
            # reason to discard another session's useful branches.
            for lane, block in zip(cohort, blocks, strict=True):
                if lane.uid not in self._last_alternative_blocks and block is not None:
                    self._last_alternative_blocks[lane.uid] = [block]
                    self._last_alternative_prefix_tokens[lane.uid] = 0
        return blocks

    def _record_ast_geometry(self, lane, blocks, accepts, counts, committed, winner, geometry):
        """One request-local accounting law for B1 and cross-session AST."""
        def add(name, value):
            key = "progressive_ast_" + name
            setattr(lane, key, getattr(lane, key, 0) + int(value))

        def histogram(name, bucket, value=1):
            key = "progressive_ast_" + name
            values = getattr(lane, key, {})
            bucket = str(bucket)
            values[bucket] = values.get(bucket, 0) + int(value)
            setattr(lane, key, values)

        rows = int(geometry.get("unique_target_rows", geometry["physical_target_rows"]))
        shared = int(geometry.get("shared_prefix_target_rows", 0))
        add("verified_branches_total", sum(value >= 0 for value in accepts))
        add("unique_target_rows_total", rows)
        add("shared_prefix_target_rows_total", shared)
        add("branch_suffix_target_rows_total", geometry.get("branch_suffix_target_rows", rows - shared))
        add("selected_committed_inputs_total", committed[winner])
        add("selected_token_uplift_total", max(0, committed[winner] - committed[0]))
        partial = [index for index, (accepted, count) in enumerate(zip(accepts, counts, strict=True))
                   if 0 <= accepted < count]
        best = max(partial, key=lambda index: (committed[index], -index), default=0)
        uplift = max(0, committed[best] - committed[0]) if partial else 0
        add("counterfactual_partial_uplift_rounds", uplift > 0)
        add("counterfactual_partial_uplift_tokens", uplift)
        if uplift:
            histogram("counterfactual_best_branch_hist", best + 1)
        for index, (block, accepted, count, commit) in enumerate(zip(blocks, accepts, counts, committed, strict=True)):
            if accepted < 0:
                continue
            if getattr(self, "continuation_pool_ast_bridge", False):
                provenance = getattr(block, "continuation_ast_provenance", None)
                rank = provenance.get("rank") if provenance else (0 if len(blocks) == 1 else None)
                if type(rank) is not int or rank < 0:
                    raise RuntimeError("AST bridge verification lacks provider rank provenance")
                histogram("provider_rank_verified_hist", rank)
                histogram("provider_rank_committed_inputs", rank, commit)
                histogram("provider_rank_full_accept_hist", rank, accepted == count)
            if index:
                primary = blocks[0].tokens
                divergence = next((position for position, (a, b) in enumerate(zip(primary, block.tokens))
                                   if a != b), min(len(primary), len(block.tokens)))
                histogram("alternative_divergence_position_hist", divergence)
        add("prefix_rejected_rounds", bool(geometry.get("prefix_rejected", False)))
        histogram("active_branch_hist", len(blocks))
        histogram("shared_prefix_token_hist", geometry.get("shared_prefix_tokens", 0))

    def _record_ast_topology_execution(self, lanes):
        """Publish content topology only after its target rows executed."""

        rootless = [
            lane
            for lane in lanes
            if self._last_alternative_topology.get(lane.uid)
            == "rootless_sibling_fanout"
        ]
        if rootless:
            _bump(
                self.scheduler_stats,
                "external_progressive_ast_rootless_rounds",
            )
            _bump(
                self.scheduler_stats,
                "external_progressive_ast_rootless_lanes",
                len(rootless),
            )
        downshifted = [
            lane for lane in lanes if lane.uid in self._last_alternative_downshift
        ]
        if downshifted:
            _bump(
                self.scheduler_stats,
                "external_progressive_ast_branch_downshift_rounds",
            )
            _bump(
                self.scheduler_stats,
                "external_progressive_ast_branch_downshift_lanes",
                len(downshifted),
            )
        prefix_searched = [
            lane for lane in lanes if lane.uid in self._last_alternative_prefix_search
        ]
        if prefix_searched:
            _bump(
                self.scheduler_stats,
                "external_progressive_ast_adaptive_prefix_rounds",
            )
            _bump(
                self.scheduler_stats,
                "external_progressive_ast_adaptive_prefix_lanes",
                len(prefix_searched),
            )
        for lane in lanes:
            topology = self._last_alternative_topology.get(lane.uid)
            if topology is not None:
                lane.progressive_ast_last_topology = topology
            if topology == "rootless_sibling_fanout":
                lane.progressive_ast_rootless_rounds = (
                    getattr(lane, "progressive_ast_rootless_rounds", 0) + 1
                )
            downshift = self._last_alternative_downshift.get(lane.uid)
            if downshift is not None:
                lane.progressive_ast_branch_downshift_rounds = (
                    getattr(lane, "progressive_ast_branch_downshift_rounds", 0) + 1
                )
                lane.progressive_ast_last_branch_downshift = copy.deepcopy(downshift)
            prefix_search = self._last_alternative_prefix_search.get(lane.uid)
            if prefix_search is not None:
                lane.progressive_ast_adaptive_prefix_rounds = (
                    getattr(lane, "progressive_ast_adaptive_prefix_rounds", 0) + 1
                )
                lane.progressive_ast_last_prefix_search = copy.deepcopy(prefix_search)
            prefix_diagnostics = self._last_alternative_prefix_diagnostics.get(
                lane.uid
            )
            if prefix_diagnostics is not None:
                lane.progressive_ast_last_prefix_diagnostics = copy.deepcopy(
                    prefix_diagnostics
                )
                lane.progressive_ast_last_proposed_prefix_diagnostics = (
                    copy.deepcopy(prefix_diagnostics)
                )

    def _proposal_depth(self, requested, maximum):
        """Apply the configured floor without exceeding terminal headroom."""

        requested = min(int(requested), int(maximum))
        floor = min(int(maximum), self.minimum_draft_proposals)
        if requested < floor:
            _bump(self.scheduler_stats, "external_proposal_floor_raises")
        return max(floor, requested)

    def _defer_ast_parked_context(self, lane):
        return bool(
            getattr(self, "ast_execution_planner", None) is not None
            and _speculation_parked(lane)
            and int(lane.tail.shape[1])
            < lane.cost_handoff_state.settings.deferred_context_cap
        )

    def _progressive_verify_many(
        self,
        cohort,
        blocks,
        *,
        terminal=True,
        max_physical_rows=4,
        queue_rows=False,
        row_metadata=None,
        row_refill=None,
    ):
        """Verify bounded independent proposal chains in aligned tiles.

        Each lane owns private target state and RNG.  A physical target launch
        batches only the lanes still advancing at the same tile boundary; a
        rejection commits that lane's exact in-tile prefix and removes it from
        later launches without changing another lane's checkpoint.
        """

        if getattr(self, "ast_tree_tile_rows", 0):
            raise RuntimeError(
                "AST tree tiles require the admitted target-tree provider; "
                "legacy progressive verification is not a tile fallback"
            )
        if self.progressive_verification_tile is None:
            raise RuntimeError("progressive verification is not selected")
        if max_physical_rows not in {4, 8}:
            raise RuntimeError("progressive physical row cap must be 4 or 8")
        logical_row_cap = 16 if queue_rows else max_physical_rows
        if not 1 <= len(cohort) == len(blocks) <= logical_row_cap:
            raise RuntimeError(
                "progressive verification exceeds its logical row cap"
            )
        if row_metadata is not None and len(row_metadata) != len(cohort):
            raise RuntimeError("progressive row metadata does not match its cohort")
        if queue_rows and row_metadata is None:
            raise RuntimeError(
                "queued progressive verification requires row metadata"
            )
        if row_refill is not None and not queue_rows:
            raise RuntimeError("progressive row refill requires queued verification")

        states = []
        proposal_counts = []
        cohort = list(cohort)
        blocks = list(blocks)
        preserve_private_capacity = bool(
            self.ast_physical_target
            and getattr(
                self.ast_physical_target_provider,
                "requires_qualified_capability",
                False,
            )
        )
        if row_metadata is not None:
            row_metadata = list(row_metadata)

        def add_state(lane, block, metadata=None):
            if len(states) >= logical_row_cap:
                raise RuntimeError("progressive verification exceeds its logical row cap")
            if (
                self.progressive_verification_fidelity
                == PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED
                and float(lane.sampling.get("sampling_temp", 0)) != 0.0
            ):
                raise LaneFailure(
                    lane.uid,
                    "near-tie-bounded progressive verification is greedy-only",
                )
            if lane.processors and not self._progressive_processors_supported(lane):
                raise LaneFailure(
                    lane.uid,
                    "progressive verification requires branch-safe logits processors",
                )
            if not isinstance(block, HostDraftRow):
                raise LaneFailure(
                    lane.uid,
                    "progressive verification requires dense host proposal laws",
                )
            drafts = list(block.tokens)
            laws = list(block.laws)
            if (not drafts and not terminal) or len(laws) != len(drafts):
                raise LaneFailure(
                    lane.uid,
                    "progressive verification requires one law per proposal token",
                )
            sparse_point_mass = bool(laws) and all(
                isinstance(law, PointMassLaw) for law in laws
            )
            if sparse_point_mass:
                if float(lane.sampling.get("sampling_temp", 0)) != 0.0:
                    raise LaneFailure(
                        lane.uid,
                        "continuation AST point-mass verification is greedy-only",
                    )
                vocab = int(self.draft.config.vocab_size)
                _greedy_proposal_supported(block, laws, drafts, vocab)
            elif laws:
                vocab = len(laws[0])
                validate_proposal_laws(drafts, laws, vocab)
            proposal_counts.append(len(drafts))
            cohort.append(lane)
            blocks.append(block)
            if row_metadata is not None:
                if metadata is None:
                    raise RuntimeError("refilled progressive row requires metadata")
                row_metadata.append(dict(metadata))
            states.append(
                {
                    "lane": lane,
                    "drafts": drafts,
                    "laws": laws,
                    "sparse_point_mass": sparse_point_mass,
                    "private_cache": self._copy_reference_cache(
                        lane.cache,
                        preserve_capacity=preserve_private_capacity,
                    ),
                    "private_rng": RequestRNG(state=lane.rng.snapshot()),
                    # Processors read the full token history; otherwise
                    # only its length and private suffix are used, so pin
                    # the committed prefix instead of copying it per round.
                    "local_history": (
                        list(lane.history) if lane.processors
                        else PrivateHistory(lane.history)
                    ),
                    "anchor": int(lane.anchor),
                    "offset": 0,
                    "accepted": 0,
                    "emitted": [],
                    "committed_inputs": [],
                    "feature_parts": [],
                    "target_laws": [],
                    "response_rows": (
                        [] if lane.sampling.get("emit_logprobs", True) else None
                    ),
                    "window": VerifyWindow(lane.processors),
                    "launches": 0,
                    "target_rows": 0,
                    "rejected": False,
                    "full_tiles": 0,
                    "tile_input_rows": [],
                    "tile_checkpoints": [],
                    "rollback_base_tile": 0,
                    "done": False,
                }
            )
            return len(states) - 1

        initial_cohort = list(cohort)
        initial_blocks = list(blocks)
        initial_metadata = None if row_metadata is None else list(row_metadata)
        cohort.clear()
        blocks.clear()
        if row_metadata is not None:
            row_metadata.clear()
        for index, (lane, block) in enumerate(
            zip(initial_cohort, initial_blocks, strict=True)
        ):
            add_state(
                lane,
                block,
                None if initial_metadata is None else initial_metadata[index],
            )
        physical_launches = 0
        ragged_launches = 0
        padding_rows = 0
        last_layout_receipt = None
        max_cohort_width = 0
        target_executor = _target_executor_stats()
        aligned_tile_launches = {
            width: 0 for width in range(1, max_physical_rows + 1)
        }
        aligned_continued_launches = {
            width: 0 for width in range(1, max_physical_rows + 1)
        }
        first_tile_pairwise_disjoint = _first_tiles_pairwise_disjoint(
            [state["drafts"] for state in states],
            self.progressive_verification_tile,
        )
        ready = deque(range(len(states)))
        launch_schedule = []
        packed_min_width = max(
            1, int(getattr(self.model, "progressive_external_lane_exact_min_width", 1))
        )
        ast_planner = getattr(self, "ast_execution_planner", None)
        while ready:
            launch_plan = None
            tile_width = self.progressive_verification_tile
            if ast_planner is not None:
                from .ast_planner import ReadyRow

                launch_plan = ast_planner.plan_launch([
                    ReadyRow(i, int(states[i]["lane"].uid),
                             len(states[i]["drafts"]) - states[i]["offset"],
                             len(states[i]["local_history"]),
                             ast_planner.acceptance(int(states[i]["lane"].uid),
                                 getattr(states[i]["lane"], "ast_source_budget", {}).get("depth")))
                    for i in ready
                ], terminal=terminal, max_lanes=max_physical_rows)
                active = list(launch_plan.indices)
                tile_width = launch_plan.tile
                selected_indices = set(active)
                ready = deque(i for i in ready if i not in selected_indices)
            else:
                active = [ready.popleft() for _ in range(min(max_physical_rows, len(ready)))]
            inputs = []
            stage = []
            for index in active:
                state = states[index]
                remaining = len(state["drafts"]) - state["offset"]
                final = terminal and remaining <= tile_width
                count = min(remaining, tile_width)
                stage_tokens = state["drafts"][
                    state["offset"] : state["offset"] + count
                ]
                stage_laws = state["laws"][
                    state["offset"] : state["offset"] + count
                ]
                row_inputs = (
                    [state["anchor"], *stage_tokens]
                    if final
                    else [state["anchor"], *stage_tokens[:-1]]
                )
                state["rollback_base_tile"] = state["full_tiles"]
                inputs.append(row_inputs)
                stage.append((index, final, count, stage_tokens, stage_laws))
            # A legal configured tile can still produce a shorter shared
            # trunk or final remainder. Honor the adapter's *actual launch*
            # boundary, not merely the load-time tile bound. Narrow trunks
            # retain exact B1 execution; wider suffixes remain packable.
            packing_split = bool(
                self.progressive_verification_fidelity == PROGRESSIVE_FIDELITY_EXACT
                and len(active) > 1
                and max(map(len, inputs)) < packed_min_width
            )
            if packing_split:
                ready.extendleft(reversed(active[1:]))
                active, inputs, stage = active[:1], inputs[:1], stage[:1]
            request_lane_uids = [states[index]["lane"].uid for index in active]
            layout_lane_uids = (
                request_lane_uids
                if len(set(request_lane_uids)) == len(request_lane_uids)
                else list(range(len(active)))
            )
            layout = RaggedVerifyLayout.from_draft_depths(
                layout_lane_uids,
                [len(row) - 1 for row in inputs],
            )
            plan = layout.select_backend(
                padded_capabilities(max_lanes=max_physical_rows)
            )
            last_layout_receipt = layout.receipt(plan)
            last_layout_receipt["request_lane_uids"] = request_lane_uids
            last_layout_receipt["physical_lane_uids"] = layout_lane_uids
            last_layout_receipt["packed_min_query_len"] = packed_min_width
            last_layout_receipt["split_for_packed_min_query_len"] = packing_split
            if launch_plan is not None:
                if tuple(map(len, inputs)) != launch_plan.input_lengths or packing_split:
                    raise RuntimeError("AST physical launch disagrees with its exact plan")
                last_layout_receipt["ast_plan"] = launch_plan.receipt()
                _bump_map(
                    self.scheduler_stats, "ast_planner_launches",
                    "primary" if launch_plan.backend == ast_planner.profile.backend
                    else "alternative",
                )
                for row_kind in ("logical", "physical", "padding"):
                    _bump_map(
                        self.scheduler_stats, "ast_planner_launch_rows", row_kind,
                        last_layout_receipt["ast_plan"][f"{row_kind}_rows"],
                    )
                for index in active:
                    states[index]["lane"].ast_last_launch = launch_plan.receipt()
            provider_backend = bool(
                launch_plan is not None
                and launch_plan.backend != ast_planner.profile.backend
            )
            if provider_backend and self.ast_physical_target:
                raise RuntimeError(
                    "AST physical target conflicts with target-tree backend"
                )
            physical_target_discovery = None
            qualified_capability_discovery = None
            requires_qualification = bool(
                self.ast_physical_target
                and getattr(
                    self.ast_physical_target_provider,
                    "requires_qualified_capability",
                    False,
                )
            )
            qualification_available = not requires_qualification
            descriptor = None
            live_rows = None
            if self.ast_physical_target and requires_qualification:
                from .ast_qualified_capability import (
                    normalize_ast_qualified_capability_discovery,
                )
                from .flat_segmented_ast import FlatSegmentDescriptor

                descriptor = FlatSegmentDescriptor.from_query_lengths(
                    layout.query_lengths
                )
                live_rows = [states[index]["private_cache"] for index in active]
                qualified_discover = getattr(
                    self.ast_physical_target_provider,
                    "discover_qualified_serving_request",
                    None,
                )
                try:
                    raw_qualified_discovery = (
                        qualified_discover(
                            rows=live_rows,
                            descriptor=descriptor,
                            capture_layers=tuple(self.layers),
                        )
                        if callable(qualified_discover)
                        else None
                    )
                except Exception:  # noqa: BLE001 - ordinary fallback is safe
                    raw_qualified_discovery = None
                qualified_capability_discovery = (
                    normalize_ast_qualified_capability_discovery(
                        raw_qualified_discovery
                    )
                )
                last_layout_receipt["qualified_capability_discovery"] = (
                    qualified_capability_discovery
                )
                qualification_available = (
                    qualified_capability_discovery.get("available") is True
                )
            if self.ast_physical_target and qualification_available:
                from .ast_physical_target import (
                    discover_ast_physical_target_from_layout,
                )

                physical_target_discovery = (
                    discover_ast_physical_target_from_layout(
                        self.ast_physical_target_provider,
                        provider_getter=lambda: getattr(
                            self.model, "ast_physical_target_provider", None
                        ),
                        query_lengths=layout.query_lengths,
                        capture_layers=tuple(self.layers),
                        fidelity=self.progressive_verification_fidelity,
                        fallback_target_protocol=self.progressive_target_protocol,
                        artifact_fingerprint=str(self.target_artifact_fingerprint),
                        fallback_plan=tuple(sorted(plan.receipt().items())),
                    )
                )
                last_layout_receipt["physical_target_discovery"] = (
                    physical_target_discovery
                )
            input_lengths = list(layout.query_lengths)
            launch_width = layout.max_query_len
            launch_padding = plan.padding_rows
            physical_binding = None
            physical_target_admission = None
            if self.ast_physical_target and qualification_available:
                if (
                    physical_target_discovery is None
                    or physical_target_discovery.get("candidate_implemented") is not True
                ):
                    reasons = (
                        []
                        if physical_target_discovery is None
                        else physical_target_discovery.get("decline_reasons", [])
                    )
                    raise RuntimeError(
                        "requested AST physical target is unavailable: "
                        + (", ".join(reasons) or "discovery unavailable")
                    )
                bind = getattr(
                    self.ast_physical_target_provider,
                    "bind_serving_physical_target",
                    None,
                )
                if not callable(bind) and not requires_qualification:
                    raise RuntimeError(
                        "requested AST physical target lacks a serving binding"
                    )
                session_scope_sha256 = hashlib.sha256(
                    json.dumps(
                        {
                            "generator_revision": self._generator_revision,
                            "scopes": [
                                states[index]["lane"].session_scope_hash
                                for index in active
                            ],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                admit = getattr(
                    self.ast_physical_target_provider,
                    "admit_serving_physical_target",
                    None,
                )
                validate_provider = getattr(
                    self.ast_physical_target_provider, "validate_revision", None
                )
                if not callable(admit):
                    raise RuntimeError(
                        "requested AST physical target lacks live admission"
                    )
                try:
                    from .ast_physical_target import (
                        validate_ast_serving_admission,
                    )

                    if descriptor is None:
                        from .flat_segmented_ast import FlatSegmentDescriptor

                        descriptor = FlatSegmentDescriptor.from_query_lengths(
                            layout.query_lengths
                        )
                        live_rows = [
                            states[index]["private_cache"] for index in active
                        ]

                    if (
                        getattr(self.model, "ast_physical_target_provider", None)
                        is not self.ast_physical_target_provider
                    ):
                        raise RuntimeError("physical-target provider binding drifted")
                    if not callable(validate_provider):
                        raise RuntimeError("physical-target provider validator drifted")
                    validate_provider()

                    if qualification_available:
                        if not callable(bind):
                            raise RuntimeError(
                                "qualified AST physical target lacks a serving binding"
                            )
                        physical_target_admission = admit(
                            rows=live_rows,
                            descriptor=descriptor,
                            capture_layers=tuple(self.layers),
                            session_scope_sha256=session_scope_sha256,
                            discovery=physical_target_discovery,
                        )
                    if (
                        getattr(self.model, "ast_physical_target_provider", None)
                        is not self.ast_physical_target_provider
                    ):
                        raise RuntimeError("physical-target provider binding drifted")
                    validate_provider()
                    if qualification_available:
                        physical_target_admission = validate_ast_serving_admission(
                            physical_target_admission,
                            physical_target_discovery,
                            expected_session_scope_sha256=session_scope_sha256,
                        )

                        bind_kwargs = {
                            "rows": live_rows,
                            "descriptor": descriptor,
                            "capture_layers": tuple(self.layers),
                            "flat_inputs": tuple(
                                token for row in inputs for token in row
                            ),
                            "session_scope_sha256": session_scope_sha256,
                            "discovery": physical_target_discovery,
                            "admission": physical_target_admission,
                        }
                        if requires_qualification:
                            bind_kwargs["qualified_discovery"] = (
                                qualified_capability_discovery
                            )
                        physical_binding = bind(**bind_kwargs)
                    if (
                        getattr(self.model, "ast_physical_target_provider", None)
                        is not self.ast_physical_target_provider
                    ):
                        raise RuntimeError("physical-target provider binding drifted")
                    validate_provider()
                except Exception as error:
                    raise RuntimeError(
                        "requested AST physical target admission or binding failed"
                    ) from error
            native_selected = physical_binding is not None
            if native_selected:
                launch_padding = 0
                last_layout_receipt["fallback_backend_mode"] = (
                    last_layout_receipt["backend_mode"]
                )
                last_layout_receipt["executed_backend_mode"] = (
                    "native_flat_logical_rows"
                )
                last_layout_receipt["physical_target_serving"] = {
                    "implemented": True,
                    "qualified": bool(
                        qualified_capability_discovery is not None
                        and qualified_capability_discovery.get("available") is True
                        and qualified_capability_discovery.get("qualified") is True
                    ),
                    "eligible": True,
                    "selected": True,
                    "default": False,
                    "observed_used": False,
                    "physical_layout": (
                        "whole_layer_ragged_native_flat"
                        if qualified_capability_discovery is not None
                        else "native_flat_logical_rows"
                    ),
                    "performance_claim": False,
                }
                last_layout_receipt["physical_target_admission"] = (
                    physical_target_admission
                )
            else:
                last_layout_receipt["executed_backend_mode"] = (
                    last_layout_receipt["backend_mode"]
                )
            ragged = not native_selected and launch_padding > 0
            if ragged:
                # Cache/state planes consume the exact lengths. Token values in
                # the right-padding tail are therefore inert and never commit.
                padded_inputs = [
                    row + [row[-1]] * (launch_width - len(row)) for row in inputs
                ]
                ragged_launches += 1
                padding_rows += launch_padding
            elif not native_selected:
                padded_inputs = inputs
            else:
                padded_inputs = None
            active_width = len(active)
            if row_metadata is not None:
                launch_schedule.append(
                    {
                        "session_slots": [
                            int(row_metadata[index]["session_slot"])
                            for index in active
                        ],
                        "branch_numbers": [
                            int(row_metadata[index]["branch_number"])
                            for index in active
                        ],
                        "logical_rows": active_width,
                        "max_physical_width": active_width,
                        "physical_launches": 1,
                        "padding_rows": int(launch_padding),
                        "continued": [
                            bool(states[index]["offset"] > 0) for index in active
                        ],
                    }
                )
            if not ragged and not native_selected:
                aligned_tile_launches[active_width] += 1
                if all(states[index]["offset"] > 0 for index in active):
                    aligned_continued_launches[active_width] += 1

            if provider_backend and len(active) != 1:
                raise RuntimeError(
                    "AST target-tree backend is single-session only"
                )
            transaction = None
            transaction_committed = False
            results = []
            participating_ast_scopes = tuple(
                dict.fromkeys(
                    states[index]["lane"].session_scope_hash for index in active
                )
            )
            ast_memo_identity = None
            ast_memo_debug_identity = None
            ast_memo_debug = None
            try:
                planning_started = None
                if provider_backend:
                    if launch_plan is not None and not self.ast_stage_row_telemetry:
                        planning_started = time.perf_counter()
                    logits, features, transaction = self._ast_target_tree_forward(
                        states[active[0]]["lane"],
                        states[active[0]]["private_cache"],
                        padded_inputs[0],
                    )
                    _observe_target_executor(
                        target_executor,
                        {
                            "execution": "adapter_owned",
                            "row_exact": True,
                            "physical_width": 1,
                            "token_width": len(padded_inputs[0]),
                            "adapter_detail": {
                                "route": launch_plan.backend,
                                "provider_protocol": (
                                    self.ast_target_tree_executor_provider.protocol
                                ),
                            },
                        },
                    )
                elif native_selected:
                    if (
                        getattr(self.model, "ast_physical_target_provider", None)
                        is not self.ast_physical_target_provider
                    ):
                        raise RuntimeError("physical-target provider binding drifted")
                    validate_provider()
                    native_result = physical_binding.execute()
                    execution_receipt = getattr(physical_binding, "receipt", None)
                    physical_execution_receipt = (
                        execution_receipt() if callable(execution_receipt) else {}
                    )
                    native_execution_mode = physical_execution_receipt.get(
                        "execution_mode"
                    ) or "native_flat_logical_rows"
                    last_layout_receipt["executed_backend_mode"] = (
                        native_execution_mode
                    )
                    if (
                        getattr(self.model, "ast_physical_target_provider", None)
                        is not self.ast_physical_target_provider
                    ):
                        raise RuntimeError("physical-target provider binding drifted")
                    validate_provider()
                    logits = features = None
                    _observe_target_executor(
                        target_executor,
                        {
                            "execution": "adapter_owned",
                            "row_exact": True,
                            "physical_width": len(active),
                            "token_width": layout.max_query_len,
                            "adapter_detail": {
                                "route": native_execution_mode,
                                "geometry_sha256": layout.geometry_sha256,
                                "logical_rows": layout.logical_rows,
                                "physical_rows": layout.logical_rows,
                                "padding_rows": 0,
                            },
                        },
                    )
                else:
                    owner = self._target_owner(
                        [states[index]["private_cache"] for index in active]
                    )
                    from .ast_execution_capability import (
                        query_ast_execution_capability,
                        transaction_owner_contract,
                    )

                    owner_contract = transaction_owner_contract(owner)
                    owner_contract_sha256 = (
                        hashlib.sha256(
                            json.dumps(
                                owner_contract,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest()
                        if owner_contract is not None
                        else None
                    )
                    capability = (
                        None
                        if owner_contract_sha256 is None
                        else self._ast_execution_capabilities.get(
                            owner_contract_sha256
                        )
                    )
                    if capability is None:
                        capability = query_ast_execution_capability(
                            self.model,
                            owner,
                            fidelity=self.progressive_verification_fidelity,
                            protocol_binding=self.progressive_protocol_binding,
                            execution_profile=(
                                None if ast_planner is None else ast_planner.profile
                            ),
                            artifact_identity=str(self.binding),
                        ).receipt()
                        if owner_contract_sha256 is not None:
                            self._ast_execution_capabilities[
                                owner_contract_sha256
                            ] = capability
                    memo_snapshot = (
                        None
                        if self.ast_memo is None
                        else self.ast_memo.snapshot(participating_ast_scopes)
                    )
                    memo_enabled = bool(
                        memo_snapshot is not None and memo_snapshot.get("enabled")
                    )
                    memo_outcome = None
                    layout_revisions = {
                        getattr(
                            states[index]["lane"],
                            "ast_memo_layout_revision",
                            None,
                        )
                        for index in active
                    }
                    if (
                        memo_enabled
                        and owner_contract_sha256 is not None
                        and len(layout_revisions) == 1
                        and None not in layout_revisions
                    ):
                        source_build_device_sha256 = hashlib.sha256(
                            json.dumps(
                                capability["identity"],
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest()
                        policy_sha256 = hashlib.sha256(
                            json.dumps(
                                {
                                    "fidelity": self.progressive_verification_fidelity,
                                    "protocol_binding": self.progressive_protocol_binding,
                                    "tile": self.progressive_verification_tile,
                                    "tile_policy": self.progressive_verification_tile_policy,
                                    "terminal_input_widths": self.progressive_terminal_input_widths,
                                    "alternative_branches": self.progressive_alternative_branches,
                                    "alternative_strategy": self.progressive_alternative_strategy,
                                    "alternative_depth_policy": self.progressive_alternative_depth_policy,
                                    "rootless_fallback": self.progressive_ast_rootless_fallback,
                                    "cross_session_pack_rows": self.progressive_cross_session_pack_rows,
                                    "cross_session_pack_mode": self.progressive_cross_session_pack_mode,
                                    "collision_probability": self.ast_collision_probability,
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest()
                        ast_memo_identity = {
                            "schema": "mlx2.apcv2.ast-memo.v1",
                            "value_kind": "ragged_geometry",
                            "apc_semantic_epoch": memo_snapshot[
                                "apc_semantic_epoch"
                            ],
                            "scope_epochs": (),
                            "layout_revision_sha256": next(iter(layout_revisions)),
                            "target_fingerprint": _ast_artifact_fingerprint(
                                getattr(self.model, "identity", self.model),
                                binding=self.binding,
                            ),
                            "drafter_fingerprint": _ast_artifact_fingerprint(
                                getattr(self.draft, "identity", self.draft),
                                binding=self.binding,
                            ),
                            "source_build_device_sha256": source_build_device_sha256,
                            "policy_sha256": policy_sha256,
                            "owner_semantic_contract_sha256": owner_contract_sha256,
                            "capability_manifest_sha256": capability[
                                "manifest_sha256"
                            ],
                            "cohort_geometry_sha256": layout.geometry_sha256,
                        }
                        ast_memo_debug_identity = {
                            **ast_memo_identity,
                            "scope_epochs": tuple(
                                tuple(item)
                                for item in memo_snapshot["scope_epochs"]
                            ),
                        }
                        live_geometry = {
                            "schema": "mlx2.ast-ragged-geometry.v1",
                            "query_lengths": list(layout.query_lengths),
                            "cu_query_lengths": list(layout.cu_query_lengths),
                            "right_padding": list(layout.right_padding),
                            "token_to_lane": list(layout.token_to_lane),
                            **plan.receipt(),
                        }
                        from .apc_ast_metadata import shadow_checked_metadata

                        comparison = shadow_checked_metadata(
                            self.ast_memo, ast_memo_identity, live_geometry
                        )
                        memo_outcome = comparison["outcome"]
                        ast_memo_debug = {
                            "lookup_input_sha256": layout.geometry_sha256,
                            "candidate_backends_and_shapes": [
                                {
                                    "backend": plan.mode.value,
                                    "logical_rows": plan.logical_rows,
                                    "physical_rows": plan.physical_rows,
                                    "max_query_len": plan.max_query_len,
                                }
                            ],
                            "selected_plan": plan.receipt(),
                            "decline_reasons": (
                                ["shadow_mismatch_live_authoritative"]
                                if comparison["outcome"] == "shadow_mismatch"
                                else []
                            ),
                            "memo_outcome": comparison["outcome"],
                            "invalidation_cause": None,
                            "shadow_comparison": comparison,
                        }
                        self.ast_memo.debug_append(
                            ast_memo_debug_identity, ast_memo_debug
                        )
                    last_layout_receipt["execution_capability"] = capability
                    last_layout_receipt["memo_identity_material"] = {
                        "capability_manifest_sha256": capability[
                            "manifest_sha256"
                        ],
                        "cohort_geometry_sha256": layout.geometry_sha256,
                        "owner_semantic_contract_sha256": (
                            owner_contract_sha256 or "unknown"
                        ),
                        "authoritative_cache": "APCv2",
                        "memo_implemented": True,
                        "memo_enabled": memo_enabled,
                        "memo_value_kind": (
                            None
                            if ast_memo_identity is None
                            else ast_memo_identity["value_kind"]
                        ),
                        "memo_selected_for_execution": False,
                        "memo_observed_used": memo_outcome is not None,
                        "memo_outcome": memo_outcome,
                    }
                    # Discovery and optional row telemetry are diagnostics, not
                    # execution cost.  Start the planner sample only after
                    # discovery; diagnostic launches do not update feedback.
                    if launch_plan is not None and not self.ast_stage_row_telemetry:
                        planning_started = time.perf_counter()
                    transaction = owner.begin(lengths=input_lengths)
                    (
                        logits,
                        features,
                        _target_executor_detail,
                    ) = execute_progressive_target_rows(
                        self.model,
                        self.mx.array(padded_inputs),
                        transaction.caches,
                        self.layers,
                        protocol=self.progressive_target_protocol,
                        fidelity=self.progressive_verification_fidelity,
                        input_lengths=input_lengths,
                        include_stage_rows=self.ast_stage_row_telemetry,
                    )
                    _target_executor_detail["capability_manifest_sha256"] = (
                        capability["manifest_sha256"]
                    )
                    _target_executor_detail["cohort_geometry_sha256"] = (
                        layout.geometry_sha256
                    )
                    _observe_target_executor(
                        target_executor, _target_executor_detail
                    )
                # Greedy, processor-free rows need only the target argmax and
                # (when requested) the ordinary float32 log-softmax row.  Keep
                # the whole physical slab on device and cross the host/device
                # boundary once: calling ``_target_law`` below once per row
                # would otherwise launch and synchronize logsumexp/argmax for
                # every row and allocate one vocabulary-sized NumPy one-hot
                # law for each of them.
                batched_greedy = all(
                    not states[index]["lane"].processors
                    and float(
                        states[index]["lane"].sampling.get("sampling_temp", 0)
                    )
                    == 0.0
                    for index in active
                )
                greedy_selected = None
                greedy_normalizers = None
                greedy_response_rows = None
                greedy_lane_offsets = None
                if batched_greedy:
                    if native_selected:
                        lane_row_counts = []
                        flat_logits = []
                        for lane in range(len(active)):
                            lane_logits, _lane_features = native_result.lane(lane)
                            lane_row_counts.append(len(lane_logits))
                            flat_logits.extend(lane_logits)
                        rows32 = self.mx.stack(flat_logits).astype(self.mx.float32)
                        offsets = [0]
                        for count in lane_row_counts:
                            offsets.append(offsets[-1] + count)
                        greedy_lane_offsets = tuple(offsets)
                    else:
                        rows32 = logits.astype(self.mx.float32)
                    normalizers = self.mx.logsumexp(
                        rows32, axis=-1, keepdims=True
                    )
                    selected = self.mx.argmax(rows32, axis=-1)
                    if any(
                        states[index]["response_rows"] is not None
                        for index in active
                    ):
                        greedy_response_rows = rows32 - normalizers
                        values = [selected, normalizers, greedy_response_rows]
                    else:
                        values = [selected, normalizers]
                    if not native_selected:
                        values.insert(0, features)
                    self.mx.eval(*values)
                    selected_values = selected.tolist()
                    normalizer_values = normalizers.tolist()
                    if native_selected:
                        greedy_selected = [
                            selected_values[start:stop]
                            for start, stop in pairwise(greedy_lane_offsets)
                        ]
                        greedy_normalizers = [
                            normalizer_values[start:stop]
                            for start, stop in pairwise(greedy_lane_offsets)
                        ]
                    else:
                        greedy_selected = selected_values
                        greedy_normalizers = normalizer_values
                elif native_selected:
                    # The physical target evaluates its result before returning;
                    # sampled rows still need the ordinary per-row target law,
                    # but must not force the tensors a second time here.
                    pass
                else:
                    self.mx.eval(logits, features)
                if planning_started is not None:
                    ast_planner.observe_launch(
                        launch_plan, time.perf_counter() - planning_started
                    )
                for physical_row, (
                    index,
                    final,
                    count,
                    stage_tokens,
                    stage_laws,
                ) in enumerate(stage):
                    state = states[index]
                    expected = count + int(final)
                    if native_selected:
                        lane_logits, lane_features = native_result.lane(physical_row)
                        if (
                            len(lane_logits) < expected
                            or len(lane_features) < len(inputs[physical_row])
                        ):
                            raise RuntimeError(
                                "native progressive target returned the wrong row count"
                            )
                    else:
                        lane_logits = logits[physical_row]
                        lane_features = features[physical_row]
                        if int(logits.shape[1]) < expected:
                            raise RuntimeError(
                                "progressive target stage returned the wrong row count"
                            )
                        if int(features.shape[1]) < len(inputs[physical_row]):
                            raise RuntimeError(
                                "progressive target stage returned the wrong feature count"
                            )
                    stage_response_rows = (
                        [] if state["response_rows"] is not None else None
                    )
                    if batched_greedy:
                        stage_target_laws = [
                            int(token)
                            for token in greedy_selected[physical_row][:expected]
                        ]
                        for row in range(expected):
                            normalizer = float(
                                greedy_normalizers[physical_row][row][0]
                            )
                            if not np.isfinite(normalizer):
                                raise LaneFailure(
                                    state["lane"].uid,
                                    "external draft target law is not a "
                                    "probability distribution",
                                )
                            if stage_response_rows is not None:
                                if native_selected:
                                    stage_response_rows.append(
                                        greedy_response_rows[
                                            greedy_lane_offsets[physical_row] + row
                                        ]
                                    )
                                else:
                                    stage_response_rows.append(
                                        greedy_response_rows[physical_row, row]
                                    )
                            state["window"].mark()
                    else:
                        stage_target_laws = []
                        for row in range(expected):
                            stage_target_laws.append(
                                self._target_law(
                                    state["lane"],
                                    lane_logits[row],
                                    state["local_history"],
                                    True,
                                    stage_response_rows,
                                    greedy_token=state["sparse_point_mass"],
                                    history_suffix=inputs[physical_row][: row + 1],
                                )
                            )
                            state["window"].mark()
                    if state["sparse_point_mass"]:
                        mismatch = next(
                            (
                                position
                                for position, (draft, target) in enumerate(
                                    zip(
                                        stage_tokens,
                                        stage_target_laws,
                                        strict=not final,
                                    )
                                )
                                if int(draft) != int(target)
                            ),
                            None,
                        )
                        if mismatch is not None:
                            emitted = [
                                *stage_tokens[:mismatch],
                                int(stage_target_laws[mismatch]),
                            ]
                            accepted = mismatch
                            rejected = True
                        elif final:
                            emitted = [
                                *stage_tokens,
                                int(stage_target_laws[len(stage_tokens)]),
                            ]
                            accepted = len(stage_tokens)
                            rejected = False
                        else:
                            emitted = list(stage_tokens)
                            accepted = len(stage_tokens)
                            rejected = False
                        outcome = SimpleNamespace(
                            accepted=accepted,
                            emitted=tuple(emitted),
                            target_probabilities=tuple(None for _ in emitted),
                            rejected=rejected,
                        )
                    elif batched_greedy:
                        outcome = verify_greedy_proposal_tile(
                            stage_tokens,
                            stage_target_laws,
                            state["private_rng"],
                            final=final,
                        )
                    else:
                        outcome = verify_proposal_tile(
                            stage_tokens,
                            stage_laws,
                            stage_target_laws,
                            state["private_rng"],
                            final=final,
                        )
                    stage_emitted = list(outcome.emitted)
                    stopped = False
                    for emitted_index, token in enumerate(stage_emitted):
                        if token in self.stops:
                            stage_emitted = stage_emitted[: emitted_index + 1]
                            stopped = True
                            break
                    consumed = min(outcome.accepted + 1, len(stage_emitted))
                    results.append(
                        (
                            index,
                            final,
                            count,
                            outcome,
                            stage_emitted,
                            stopped,
                            consumed,
                            stage_target_laws,
                            stage_response_rows,
                            (
                                self.mx.stack(
                                    [lane_features[row] for row in range(consumed)]
                                )[None]
                                if native_selected
                                else features[
                                    physical_row : physical_row + 1, :consumed
                                ]
                            ),
                            inputs[physical_row],
                        )
                    )
                accepted_lengths = [result[6] for result in results]
                if native_selected:
                    if (
                        getattr(self.model, "ast_physical_target_provider", None)
                        is not self.ast_physical_target_provider
                    ):
                        raise RuntimeError("physical-target provider binding drifted")
                    validate_provider()
                committed = (
                    physical_binding.commit(accepted_lengths)
                    if native_selected
                    else transaction.commit(accepted_lengths=accepted_lengths)
                )
                transaction_committed = True
                if native_selected:
                    last_layout_receipt["physical_target_serving"] = (
                        physical_binding.receipt()
                    )
                if self.ast_memo is not None:
                    if (
                        ast_memo_debug_identity is not None
                        and ast_memo_debug is not None
                    ):
                        self.ast_memo.debug_append(
                            ast_memo_debug_identity,
                            {
                                **ast_memo_debug,
                                "memo_outcome": None,
                                "invalidation_cause": "transaction_commit",
                                "shadow_comparison": None,
                            },
                        )
                    self.ast_memo.advance(
                        participating_ast_scopes,
                        "transaction_commit",
                    )
            except BaseException:
                if native_selected and not transaction_committed:
                    try:
                        physical_binding.abort()
                    except BaseException:  # noqa: BLE001,S110 - preserve original
                        pass
                if transaction is not None and not transaction_committed:
                    if not transaction.closed:
                        try:
                            transaction.abort()
                        except BaseException:
                            # Preserve the original model/commit exception.
                            # The memo epoch is still advanced below.
                            pass
                if self.ast_memo is not None:
                    if (
                        ast_memo_debug_identity is not None
                        and ast_memo_debug is not None
                    ):
                        self.ast_memo.debug_append(
                            ast_memo_debug_identity,
                            {
                                **ast_memo_debug,
                                "memo_outcome": None,
                                "invalidation_cause": "transaction_abort",
                                "shadow_comparison": None,
                            },
                        )
                    self.ast_memo.advance(
                        participating_ast_scopes,
                        "transaction_abort",
                    )
                raise

            physical_launches += 1
            max_cohort_width = max(max_cohort_width, len(active))
            for private_cache, result in zip(committed, results, strict=True):
                (
                    index,
                    final,
                    count,
                    outcome,
                    stage_emitted,
                    stopped,
                    consumed,
                    stage_target_laws,
                    stage_response_rows,
                    feature_part,
                    row_inputs,
                ) = result
                state = states[index]
                state["private_cache"] = private_cache
                state["launches"] += 1
                state["target_rows"] += len(row_inputs)
                state["tile_input_rows"].append(len(row_inputs))
                state["feature_parts"].append(feature_part)
                committed_inputs = row_inputs[:consumed]
                state["committed_inputs"].extend(committed_inputs)
                state["local_history"].extend(committed_inputs)
                state["emitted"].extend(stage_emitted)
                # A stop can truncate an otherwise accepted proposal tile.
                # Count only accepted proposal tokens that are part of the
                # emitted prefix; proposals after the stop never commit.
                state["accepted"] += min(
                    int(outcome.accepted), len(stage_emitted)
                )
                state["target_laws"].extend(
                    outcome.target_probabilities[: len(stage_emitted)]
                )
                if state["response_rows"] is not None:
                    state["response_rows"].extend(
                        stage_response_rows[: len(stage_emitted)]
                    )
                state["anchor"] = int(stage_emitted[-1])
                if outcome.rejected or stopped:
                    state["rejected"] = bool(outcome.rejected)
                    state["done"] = True
                    continue
                state["offset"] += count
                if final:
                    state["done"] = True
                    continue
                state["full_tiles"] += 1
                state["tile_checkpoints"].append(
                    {
                        "tile": state["full_tiles"],
                        "committed_inputs": len(state["committed_inputs"]),
                        "accepted_proposals": state["accepted"],
                    }
                )
                if state["offset"] == len(state["drafts"]):
                    state["done"] = True
            refill_indices = []
            if row_refill is not None:
                for index in active:
                    state = states[index]
                    if not state["done"]:
                        continue
                    refill = row_refill(
                        index,
                        state,
                        dict(row_metadata[index]),
                    )
                    if refill is not None:
                        lane, block, metadata = refill
                        refill_indices.append(add_state(lane, block, metadata))
            for index in active:
                if not states[index]["done"]:
                    ready.append(index)
            ready.extend(refill_indices)

        decisions = []
        feature_rows = []
        stats = []
        for state in states:
            decisions.append(
                RoundDecision(
                    accepted=state["accepted"],
                    emitted=state["emitted"],
                    target_laws=state["target_laws"],
                    response_logprobs=state["response_rows"],
                    verify_window=state["window"],
                    commit_count=len(state["committed_inputs"]),
                    committed_inputs=state["committed_inputs"],
                )
            )
            feature_rows.append(
                self.mx.concatenate(state["feature_parts"], axis=1)
            )
            stats.append(
                {
                    "launches": state["launches"],
                    "target_rows": state["target_rows"],
                    "rejected": int(state["rejected"]),
                    "full_tiles": state["full_tiles"],
                    "tile_checkpoints": state["tile_checkpoints"],
                    "tile_input_rows": state["tile_input_rows"],
                    "rollback_base_tile": state["rollback_base_tile"],
                    "cohort_width": len(cohort),
                    "ast_last_launch": getattr(state["lane"], "ast_last_launch", None),
                }
            )
        width = max(int(row.shape[1]) for row in feature_rows)
        padded = []
        for row in feature_rows:
            missing = width - int(row.shape[1])
            if missing:
                row = self.mx.concatenate(
                    [
                        row,
                        self.mx.zeros(
                            (1, missing, int(row.shape[2])), dtype=row.dtype
                        ),
                    ],
                    axis=1,
                )
            padded.append(row)
        return (
            decisions,
            self.mx.concatenate(padded, axis=0),
            [state["private_cache"] for state in states],
            [state["private_rng"] for state in states],
            stats,
            {
                "physical_launches": physical_launches,
                "physical_target_rows": sum(
                    int(state["target_rows"]) for state in states
                ),
                "ragged_launches": ragged_launches,
                "padding_rows": padding_rows,
                "last_ragged_layout": last_layout_receipt,
                "max_cohort_width": max_cohort_width,
                "target_row_executor": target_executor,
                "aligned_tile_launches": aligned_tile_launches,
                "aligned_continued_launches": aligned_continued_launches,
                "b2_aligned_tile_launches": aligned_tile_launches[2],
                "b2_aligned_continued_launches": aligned_continued_launches[2],
                "b4_aligned_tile_launches": aligned_tile_launches[4],
                "b4_aligned_continued_launches": aligned_continued_launches[4],
                "b8_aligned_tile_launches": aligned_tile_launches.get(8, 0),
                "b8_aligned_continued_launches": aligned_continued_launches.get(
                    8, 0
                ),
                "physical_row_cap": max_physical_rows,
                "first_tile_pairwise_disjoint": first_tile_pairwise_disjoint,
                "launch_schedule": launch_schedule,
                "row_metadata": (
                    None if row_metadata is None else copy.deepcopy(row_metadata)
                ),
            },
        )

    def _progressive_verify_paired_suffix_family(self, lanes, blocks):
        """Verify one family parent once, then fan out its two continuations."""

        if len(lanes) != 2 or len(blocks) != 2:
            raise RuntimeError("paired continuation family requires two rows")
        if any(len(block.tokens) < 2 for block in blocks):
            raise RuntimeError("paired continuation family requires child suffixes")
        if int(blocks[0].tokens[0]) != int(blocks[1].tokens[0]):
            raise RuntimeError("paired continuation family parents differ")
        parent = HostDraftRow(
            [int(blocks[0].tokens[0])], [blocks[0].laws[0]], 1
        )
        (
            parent_decisions,
            parent_features,
            parent_caches,
            parent_rngs,
            parent_stats,
            parent_physical,
        ) = self._progressive_verify_many([lanes[0]], [parent], terminal=False)
        parent_decision = parent_decisions[0]
        if int(parent_decision.accepted) != 1:
            decisions = [copy.copy(parent_decision), copy.copy(parent_decision)]
            features = self.mx.concatenate(
                [parent_features, parent_features], axis=0
            )
            caches = [
                parent_caches[0], self._copy_reference_cache(parent_caches[0])
            ]
            rngs = [
                RequestRNG(state=parent_rngs[0].snapshot()),
                RequestRNG(state=parent_rngs[0].snapshot()),
            ]
            stats = [dict(parent_stats[0]), dict(parent_stats[0])]
            physical = dict(parent_physical)
            physical["paired_parent_rows_reused"] = 1
            return decisions, features, caches, rngs, stats, physical

        child_lanes = []
        child_blocks = []
        for index, (lane, block) in enumerate(zip(lanes, blocks, strict=True)):
            child = copy.copy(lane)
            child.cache = (
                parent_caches[0]
                if index == 0
                else self._copy_reference_cache(parent_caches[0])
            )
            child.rng = RequestRNG(state=parent_rngs[0].snapshot())
            child.history = list(lane.history) + list(
                parent_decision.committed_inputs
            )
            child.anchor = int(parent_decision.emitted[-1])
            child_lanes.append(child)
            child_blocks.append(
                HostDraftRow(
                    list(block.tokens[1:]), list(block.laws[1:]), len(blocks)
                )
            )
        (
            child_decisions,
            child_features,
            child_caches,
            child_rngs,
            child_stats,
            child_physical,
        ) = self._progressive_verify_many(child_lanes, child_blocks)
        decisions = []
        stats = []
        feature_rows = []
        for index, (decision, child_stat) in enumerate(
            zip(child_decisions, child_stats, strict=True)
        ):
            decisions.append(
                RoundDecision(
                    accepted=1 + int(decision.accepted),
                    emitted=list(parent_decision.emitted) + list(decision.emitted),
                    target_laws=list(parent_decision.target_laws)
                    + list(decision.target_laws),
                    response_logprobs=(
                        None
                        if parent_decision.response_logprobs is None
                        else list(parent_decision.response_logprobs)
                        + list(decision.response_logprobs)
                    ),
                    verify_window=decision.verify_window,
                    commit_count=int(parent_decision.commit_count)
                    + int(decision.commit_count),
                    committed_inputs=list(parent_decision.committed_inputs)
                    + list(decision.committed_inputs),
                )
            )
            feature_rows.append(
                self.mx.concatenate(
                    [parent_features, child_features[index : index + 1]],
                    axis=1,
                )
            )
            stats.append(
                {
                    "launches": parent_stats[0]["launches"]
                    + child_stat["launches"],
                    "target_rows": parent_stats[0]["target_rows"]
                    + child_stat["target_rows"],
                    "rejected": parent_stats[0]["rejected"]
                    + child_stat["rejected"],
                    "full_tiles": parent_stats[0]["full_tiles"]
                    + child_stat["full_tiles"],
                    "tile_checkpoints": list(
                        parent_stats[0]["tile_checkpoints"]
                    )
                    + list(child_stat["tile_checkpoints"]),
                    "tile_input_rows": list(parent_stats[0]["tile_input_rows"])
                    + list(child_stat["tile_input_rows"]),
                    "rollback_base_tile": child_stat["rollback_base_tile"],
                    "cohort_width": 2,
                }
            )
        aligned = {
            width: parent_physical["aligned_tile_launches"][width]
            + child_physical["aligned_tile_launches"][width]
            for width in range(1, 5)
        }
        continued = {
            width: parent_physical["aligned_continued_launches"][width]
            + child_physical["aligned_continued_launches"][width]
            for width in range(1, 5)
        }
        physical = {
            "physical_launches": parent_physical["physical_launches"]
            + child_physical["physical_launches"],
            "physical_target_rows": parent_physical["physical_target_rows"]
            + child_physical["physical_target_rows"],
            "ragged_launches": parent_physical.get("ragged_launches", 0)
            + child_physical.get("ragged_launches", 0),
            "padding_rows": parent_physical.get("padding_rows", 0)
            + child_physical.get("padding_rows", 0),
            "last_ragged_layout": child_physical.get("last_ragged_layout")
            or parent_physical.get("last_ragged_layout"),
            "max_cohort_width": max(
                parent_physical["max_cohort_width"],
                child_physical["max_cohort_width"],
            ),
            "target_row_executor": _merge_target_executor_stats(
                parent_physical.get("target_row_executor"),
                child_physical.get("target_row_executor"),
            ),
            "aligned_tile_launches": aligned,
            "aligned_continued_launches": continued,
            "b2_aligned_tile_launches": aligned[2],
            "b2_aligned_continued_launches": continued[2],
            "b4_aligned_tile_launches": aligned[4],
            "b4_aligned_continued_launches": continued[4],
            "first_tile_pairwise_disjoint": False,
            "paired_parent_rows_reused": 1,
        }
        return (
            decisions,
            self.mx.concatenate(feature_rows, axis=0),
            child_caches,
            child_rngs,
            stats,
            physical,
        )

    def _progressive_verify_shared_prefix(self, lane, blocks, *, prefix_count=None):
        """Verify one common tile once, then only the unique branch suffixes.

        The prefix transaction is request-private and nonterminal, so a full
        prefix accept emits no bonus token.  Each suffix starts from an exact
        copy of that joint attention/recurrent checkpoint and RNG state.  A
        prefix rejection is authoritative for every branch because their
        proposal tokens and laws are identical up to the divergence point.
        """

        prefix_count = int(
            self.progressive_verification_tile
            if prefix_count is None
            else prefix_count
        )
        if prefix_count == 0:
            if not blocks:
                raise RuntimeError("zero-trunk AST requires proposal branches")
            decisions, features, caches, rngs, stats, physical = (
                self._progressive_verify_many(
                    [copy.copy(lane) for _ in blocks],
                    list(blocks),
                )
            )
            physical = dict(physical)
            physical.update(
                shared_prefix_tokens=0,
                shared_prefix_target_rows=0,
                branch_suffix_target_rows=physical["physical_target_rows"],
                unique_target_rows=physical["physical_target_rows"],
                prefix_rejected=False,
            )
            return decisions, features, caches, rngs, stats, physical
        if not blocks or any(len(block.tokens) <= prefix_count for block in blocks):
            raise RuntimeError("shared-prefix alternatives require a suffix")
        prefix_tokens = list(blocks[0].tokens[:prefix_count])
        prefix_laws = list(blocks[0].laws[:prefix_count])
        if any(list(block.tokens[:prefix_count]) != prefix_tokens for block in blocks):
            raise RuntimeError("shared-prefix alternatives diverged before the tile")
        prefix_block = HostDraftRow(prefix_tokens, prefix_laws, 1)
        (
            prefix_decisions,
            prefix_features,
            prefix_caches,
            prefix_rngs,
            prefix_stats,
            prefix_physical,
        ) = self._progressive_verify_many(
            [lane], [prefix_block], terminal=False
        )
        prefix_decision = prefix_decisions[0]
        branch_count = len(blocks)
        if int(prefix_decision.accepted) != prefix_count:
            decisions = [copy.copy(prefix_decision) for _ in blocks]
            features = self.mx.concatenate(
                [prefix_features for _ in blocks], axis=0
            )
            caches = [
                prefix_caches[0]
                if index == 0
                else self._copy_reference_cache(prefix_caches[0])
                for index in range(branch_count)
            ]
            rngs = [
                RequestRNG(state=prefix_rngs[0].snapshot())
                for _ in blocks
            ]
            zero = {
                "launches": 0,
                "target_rows": 0,
                "rejected": 0,
                "full_tiles": 0,
                "tile_checkpoints": [],
                "tile_input_rows": [],
                "rollback_base_tile": 0,
                "cohort_width": branch_count,
            }
            stats = [prefix_stats[0], *[dict(zero) for _ in blocks[1:]]]
            physical = dict(prefix_physical)
            physical.update(
                shared_prefix_tokens=prefix_count,
                shared_prefix_target_rows=prefix_physical["physical_target_rows"],
                branch_suffix_target_rows=0,
                unique_target_rows=prefix_physical["physical_target_rows"],
                prefix_rejected=True,
            )
            return decisions, features, caches, rngs, stats, physical

        suffix_lanes = []
        suffix_blocks = []
        for block in blocks:
            branch = copy.copy(lane)
            branch.cache = prefix_caches[0]
            branch.rng = prefix_rngs[0]
            branch.history = list(lane.history) + list(
                prefix_decision.committed_inputs
            )
            branch.anchor = int(prefix_decision.emitted[-1])
            suffix_lanes.append(branch)
            suffix_blocks.append(
                HostDraftRow(
                    list(block.tokens[prefix_count:]),
                    list(block.laws[prefix_count:]),
                    branch_count,
                )
            )
        decisions = [None] * branch_count
        feature_rows = [None] * branch_count
        committed_caches = [None] * branch_count
        committed_rngs = [None] * branch_count
        combined_stats = [None] * branch_count
        aligned = dict(prefix_physical["aligned_tile_launches"])
        continued = dict(prefix_physical["aligned_continued_launches"])
        physical_launches = int(prefix_physical["physical_launches"])
        ragged_launches = int(prefix_physical.get("ragged_launches", 0))
        padding_rows = int(prefix_physical.get("padding_rows", 0))
        last_ragged_layout = prefix_physical.get("last_ragged_layout")
        unique_target_rows = int(prefix_physical["physical_target_rows"])
        max_cohort_width = int(prefix_physical["max_cohort_width"])
        target_executor = _merge_target_executor_stats(
            prefix_physical.get("target_row_executor")
        )
        short_circuit_after_branch = None
        proposal_counts = tuple(len(block.tokens) for block in blocks)
        for indices in staged_branch_indices(
            branch_count, self.progressive_alternative_strategy, 2
        ):
            (
                cohort_decisions,
                cohort_features,
                cohort_caches,
                cohort_rngs,
                cohort_stats,
                cohort_physical,
            ) = (
                self._progressive_verify_paired_suffix_family(
                    [suffix_lanes[index] for index in indices],
                    [suffix_blocks[index] for index in indices],
                )
                if self.progressive_alternative_strategy
                == "paired_continuations"
                and len(indices) == 2
                else self._progressive_verify_many(
                    [suffix_lanes[index] for index in indices],
                    [suffix_blocks[index] for index in indices],
                )
            )
            physical_launches += int(cohort_physical["physical_launches"])
            ragged_launches += int(cohort_physical.get("ragged_launches", 0))
            padding_rows += int(cohort_physical.get("padding_rows", 0))
            last_ragged_layout = (
                cohort_physical.get("last_ragged_layout") or last_ragged_layout
            )
            unique_target_rows += int(cohort_physical["physical_target_rows"])
            max_cohort_width = max(
                max_cohort_width, int(cohort_physical["max_cohort_width"])
            )
            target_executor = _merge_target_executor_stats(
                target_executor, cohort_physical.get("target_row_executor")
            )
            for width in range(1, 5):
                aligned[width] += cohort_physical["aligned_tile_launches"][width]
                continued[width] += cohort_physical[
                    "aligned_continued_launches"
                ][width]
            for local, (decision, stats) in enumerate(
                zip(cohort_decisions, cohort_stats, strict=True)
            ):
                index = indices[local]
                combined = RoundDecision(
                    accepted=prefix_count + int(decision.accepted),
                    emitted=list(prefix_decision.emitted) + list(decision.emitted),
                    target_laws=list(prefix_decision.target_laws)
                    + list(decision.target_laws),
                    response_logprobs=(
                        None
                        if prefix_decision.response_logprobs is None
                        else list(prefix_decision.response_logprobs)
                        + list(decision.response_logprobs)
                    ),
                    verify_window=decision.verify_window,
                    commit_count=int(prefix_decision.commit_count)
                    + int(decision.commit_count),
                    committed_inputs=list(prefix_decision.committed_inputs)
                    + list(decision.committed_inputs),
                )
                decisions[index] = combined
                feature_rows[index] = self.mx.concatenate(
                    [prefix_features, cohort_features[local : local + 1]],
                    axis=1,
                )
                committed_caches[index] = cohort_caches[local]
                committed_rngs[index] = cohort_rngs[local]
                combined_stats[index] = {
                    "launches": prefix_stats[0]["launches"] + stats["launches"],
                    "target_rows": prefix_stats[0]["target_rows"]
                    + stats["target_rows"],
                    "rejected": prefix_stats[0]["rejected"] + stats["rejected"],
                    "full_tiles": prefix_stats[0]["full_tiles"]
                    + stats["full_tiles"],
                    "tile_checkpoints": list(prefix_stats[0]["tile_checkpoints"])
                    + list(stats["tile_checkpoints"]),
                    "tile_input_rows": list(prefix_stats[0]["tile_input_rows"])
                    + list(stats["tile_input_rows"]),
                    "rollback_base_tile": stats["rollback_base_tile"],
                    "cohort_width": len(indices),
                }
            full = [
                index
                for index in indices
                if int(decisions[index].accepted) == proposal_counts[index]
            ]
            if full:
                short_circuit_after_branch = full[0] + 1
                break

        template = next(row for row in feature_rows if row is not None)
        zero_stats = {
            "launches": 0,
            "target_rows": 0,
            "rejected": 0,
            "full_tiles": 0,
            "tile_checkpoints": [],
            "tile_input_rows": [],
            "rollback_base_tile": 0,
            "cohort_width": 0,
        }
        for index in range(branch_count):
            if decisions[index] is not None:
                continue
            decisions[index] = RoundDecision(
                accepted=-1,
                emitted=[],
                target_laws=[],
                commit_count=0,
                committed_inputs=[],
            )
            feature_rows[index] = self.mx.zeros_like(template)
            committed_caches[index] = prefix_caches[0]
            committed_rngs[index] = prefix_rngs[0]
            combined_stats[index] = dict(zero_stats)

        feature_width = max(int(row.shape[1]) for row in feature_rows)
        padded_features = []
        for row in feature_rows:
            missing = feature_width - int(row.shape[1])
            if missing:
                row = self.mx.concatenate(
                    [
                        row,
                        self.mx.zeros(
                            (1, missing, int(row.shape[2])), dtype=row.dtype
                        ),
                    ],
                    axis=1,
                )
            padded_features.append(row)
        physical = {
            "physical_launches": physical_launches,
            "physical_target_rows": unique_target_rows,
            "ragged_launches": ragged_launches,
            "padding_rows": padding_rows,
            "last_ragged_layout": last_ragged_layout,
            "max_cohort_width": max_cohort_width,
            "target_row_executor": target_executor,
            "aligned_tile_launches": aligned,
            "aligned_continued_launches": continued,
            "b2_aligned_tile_launches": aligned[2],
            "b2_aligned_continued_launches": continued[2],
            "b4_aligned_tile_launches": aligned[4],
            "b4_aligned_continued_launches": continued[4],
            "first_tile_pairwise_disjoint": False,
            "shared_prefix_tokens": prefix_count,
            "shared_prefix_target_rows": int(
                prefix_physical["physical_target_rows"]
            ),
            "branch_suffix_target_rows": int(
                unique_target_rows - prefix_physical["physical_target_rows"]
            ),
            "unique_target_rows": unique_target_rows,
            "prefix_rejected": False,
            "short_circuit_after_branch": short_circuit_after_branch,
        }
        return (
            decisions,
            self.mx.concatenate(padded_features, axis=0),
            committed_caches,
            committed_rngs,
            combined_stats,
            physical,
        )

    def _progressive_verify_cross_session_ast(
        self, cohort, block_groups, prefix_counts, *, pack_rows
    ):
        """Verify private AST sessions while packing their suffix rows together.

        The shared trunk is evaluated once per request in a B1-semantic row.
        Accepted trunks fan out into request-private branch states.  Only those
        branch rows share a physical target launch; cache, RNG, history,
        selection, rollback, and final promotion remain request-scoped.
        """

        if self.progressive_alternative_strategy != "confidence_tree":
            raise RuntimeError("cross-session AST packing requires confidence_tree")
        if pack_rows not in {4, 8}:
            raise RuntimeError("cross-session AST pack rows must be 4 or 8")
        if not 2 <= len(cohort) == len(block_groups) == len(prefix_counts) <= 4:
            raise RuntimeError("cross-session AST packing requires two to four sessions")
        if any(not 1 <= len(blocks) <= 4 for blocks in block_groups) or all(
            len(blocks) == 1 for blocks in block_groups
        ):
            # A session latched to its primary chain packs as one trunk plus
            # one suffix branch; at least one session must still branch.
            raise RuntimeError(
                "cross-session AST requires one to four branches per session "
                "and at least one branching session"
            )

        prefix_blocks = []
        for blocks, prefix_count in zip(block_groups, prefix_counts, strict=True):
            prefix_count = int(prefix_count)
            if prefix_count < 0 or any(
                len(block.tokens) <= prefix_count for block in blocks
            ):
                raise RuntimeError("cross-session AST requires a nonempty branch suffix")
            if prefix_count == 0:
                prefix_blocks.append(None)
                continue
            prefix_tokens = list(blocks[0].tokens[:prefix_count])
            prefix_laws = list(blocks[0].laws[:prefix_count])
            if any(
                list(block.tokens[:prefix_count]) != prefix_tokens
                for block in blocks
            ):
                raise RuntimeError("cross-session AST branches diverged before trunk")
            prefix_blocks.append(HostDraftRow(prefix_tokens, prefix_laws, 1))

        positive_prefix_rows = [
            index for index, block in enumerate(prefix_blocks) if block is not None
        ]
        prefix_decisions = [None] * len(cohort)
        prefix_features = [None] * len(cohort)
        prefix_caches = [None] * len(cohort)
        prefix_rngs = [None] * len(cohort)
        prefix_stats = [None] * len(cohort)
        prefix_physical = {
            "physical_launches": 0,
            "physical_target_rows": 0,
            "ragged_launches": 0,
            "padding_rows": 0,
            "last_ragged_layout": None,
            "max_cohort_width": 0,
            "target_row_executor": _target_executor_stats(),
            "aligned_tile_launches": {width: 0 for width in range(1, 5)},
            "aligned_continued_launches": {width: 0 for width in range(1, 5)},
        }
        if positive_prefix_rows:
            (
                positive_decisions,
                positive_features,
                positive_caches,
                positive_rngs,
                positive_stats,
                prefix_physical,
            ) = self._progressive_verify_many(
                [cohort[index] for index in positive_prefix_rows],
                [prefix_blocks[index] for index in positive_prefix_rows],
                terminal=False,
                max_physical_rows=4,
            )
            for local, session_index in enumerate(positive_prefix_rows):
                prefix_decisions[session_index] = positive_decisions[local]
                prefix_features[session_index] = positive_features[local : local + 1]
                prefix_caches[session_index] = positive_caches[local]
                prefix_rngs[session_index] = positive_rngs[local]
                prefix_stats[session_index] = positive_stats[local]
        for session_index, prefix_count in enumerate(prefix_counts):
            if int(prefix_count) != 0:
                continue
            lane = cohort[session_index]
            prefix_decisions[session_index] = RoundDecision(
                accepted=0,
                emitted=[],
                target_laws=[],
                response_logprobs=(
                    [] if lane.sampling.get("emit_logprobs", True) else None
                ),
                commit_count=0,
                committed_inputs=[],
            )
            prefix_caches[session_index] = lane.cache
            prefix_rngs[session_index] = lane.rng
            prefix_stats[session_index] = {
                "launches": 0,
                "target_rows": 0,
                "rejected": 0,
                "full_tiles": 0,
                "tile_checkpoints": [],
                "tile_input_rows": [],
                "rollback_base_tile": 0,
            }

        branch_results = {}
        pending = []
        live_frontier = (
            self.progressive_cross_session_pack_mode == "ordered_live_frontier"
        )
        frontier_planner = None

        def branch_row(session_index, branch_index):
            lane = cohort[session_index]
            blocks = block_groups[session_index]
            prefix_count = int(prefix_counts[session_index])
            branch = copy.copy(lane)
            branch.cache = prefix_caches[session_index]
            branch.rng = prefix_rngs[session_index]
            branch.history = list(lane.history) + list(
                prefix_decisions[session_index].committed_inputs
            )
            branch.anchor = (
                int(lane.anchor)
                if prefix_count == 0
                else int(prefix_decisions[session_index].emitted[-1])
            )
            block = blocks[branch_index]
            suffix = HostDraftRow(
                list(block.tokens[prefix_count:]),
                list(block.laws[prefix_count:]),
                len(blocks),
            )
            return branch_index, session_index, branch, suffix

        if live_frontier:
            frontier_planner = OrderedLiveFrontierPlanner(
                [
                    [len(block.tokens) - int(prefix_count) for block in blocks]
                    for blocks, prefix_count in zip(
                        block_groups, prefix_counts, strict=True
                    )
                ],
                maximize_committed_prefix=bool(
                    getattr(self, "continuation_pool_ast_bridge", False)
                ),
            )
            eligible = [
                session_index
                for session_index, prefix_count in enumerate(prefix_counts)
                if int(prefix_decisions[session_index].accepted) == int(prefix_count)
            ]
            for session_index in range(len(cohort)):
                if session_index not in eligible:
                    frontier_planner.retire_unadmitted(session_index)
            pending = [
                branch_row(admission.session_slot, admission.branch_index)
                for admission in frontier_planner.start(eligible)
            ]
        else:
            for session_index, (blocks, prefix_count) in enumerate(
                zip(block_groups, prefix_counts, strict=True)
            ):
                if int(prefix_decisions[session_index].accepted) != int(prefix_count):
                    continue
                pending.extend(
                    branch_row(session_index, branch_index)
                    for branch_index in range(len(blocks))
                )
        # Interleave by branch rank so every full pack draws from all live
        # sessions instead of filling one session's branches first.
        pending.sort(key=lambda item: (item[0], item[1]))

        physical_launches = int(prefix_physical["physical_launches"])
        physical_target_rows = int(prefix_physical["physical_target_rows"])
        ragged_launches = int(prefix_physical.get("ragged_launches", 0))
        padding_rows = int(prefix_physical.get("padding_rows", 0))
        max_cohort_width = int(prefix_physical["max_cohort_width"])
        target_executor = _merge_target_executor_stats(
            prefix_physical.get("target_row_executor")
        )
        aligned = {width: 0 for width in range(1, pack_rows + 1)}
        continued = {width: 0 for width in range(1, pack_rows + 1)}
        for width, value in prefix_physical["aligned_tile_launches"].items():
            aligned[width] += value
        for width, value in prefix_physical["aligned_continued_launches"].items():
            continued[width] += value
        uid_to_slot = {int(lane.uid): index for index, lane in enumerate(cohort)}

        def receipt_layout(layout):
            if layout is None:
                return None
            result = copy.deepcopy(layout)
            result["request_lane_uids"] = [
                uid_to_slot[int(uid)] for uid in result["request_lane_uids"]
            ]
            result["request_lane_identity"] = "cohort_local_session_slot"
            return result

        last_ragged_layout = receipt_layout(
            prefix_physical.get("last_ragged_layout")
        )
        packs = []
        if pending:
            def refill(_row_index, state, metadata):
                if frontier_planner is None:
                    return None
                session_index = int(metadata["session_slot"])
                branch_index = int(metadata["branch_number"]) - 1
                admission = frontier_planner.complete(
                    session_index,
                    branch_index,
                    full=int(state["accepted"]) == len(state["drafts"]),
                )
                if admission is None:
                    return None
                row = branch_row(admission.session_slot, admission.branch_index)
                return (
                    row[2],
                    row[3],
                    {
                        "session_slot": int(row[1]),
                        "branch_number": int(row[0]) + 1,
                    },
                )

            (
                decisions,
                features,
                caches,
                rngs,
                stats,
                physical,
            ) = self._progressive_verify_many(
                [row[2] for row in pending],
                [row[3] for row in pending],
                max_physical_rows=pack_rows,
                queue_rows=True,
                row_metadata=[
                    {
                        "session_slot": int(row[1]),
                        "branch_number": int(row[0]) + 1,
                    }
                    for row in pending
                ],
                row_refill=refill if live_frontier else None,
            )
            physical_launches += int(physical["physical_launches"])
            physical_target_rows += int(physical["physical_target_rows"])
            ragged_launches += int(physical.get("ragged_launches", 0))
            padding_rows += int(physical.get("padding_rows", 0))
            max_cohort_width = max(
                max_cohort_width, int(physical["max_cohort_width"])
            )
            target_executor = _merge_target_executor_stats(
                target_executor, physical.get("target_row_executor")
            )
            last_ragged_layout = (
                receipt_layout(physical.get("last_ragged_layout"))
                or last_ragged_layout
            )
            for width in aligned:
                aligned[width] += physical["aligned_tile_launches"].get(width, 0)
                continued[width] += physical["aligned_continued_launches"].get(
                    width, 0
                )
            packs = list(physical["launch_schedule"])
            result_metadata = physical["row_metadata"]
            for local, metadata in enumerate(result_metadata):
                key = (
                    int(metadata["session_slot"]),
                    int(metadata["branch_number"]) - 1,
                )
                branch_results[key] = (
                    decisions[local],
                    features[local : local + 1],
                    caches[local],
                    rngs[local],
                    stats[local],
                )

        selected_decisions = []
        selected_features = []
        selected_caches = []
        selected_rngs = []
        selected_stats = []
        selected_blocks = []
        selections = []
        for session_index, (lane, blocks, prefix_count) in enumerate(
            zip(cohort, block_groups, prefix_counts, strict=True)
        ):
            prefix_count = int(prefix_count)
            prefix = prefix_decisions[session_index]
            proposal_counts = tuple(len(block.tokens) for block in blocks)
            if int(prefix.accepted) != prefix_count:
                decisions = [copy.copy(prefix) for _ in blocks]
                winner, full = ordered_alternative_winner(
                    decisions,
                    proposal_counts,
                    allow_partial_uplift=bool(
                        getattr(self, "continuation_pool_ast_bridge", False)
                    ),
                )
                committed_tokens = ordered_branch_commit_counts(decisions)
                selected_feature = prefix_features[session_index][
                    :, : int(prefix.commit_count)
                ]
                selected_cache = prefix_caches[session_index]
                selected_rng = prefix_rngs[session_index]
                stats = dict(prefix_stats[session_index])
                branch_accepts = [int(prefix.accepted)] * len(blocks)
                unique_target_rows = int(prefix_stats[session_index]["target_rows"])
            else:
                decisions = []
                features = []
                caches = []
                rngs = []
                stats_rows = []
                session_template = next(
                    (
                        result[1]
                        for (slot, _branch), result in branch_results.items()
                        if slot == session_index
                    ),
                    None,
                )
                if session_template is None:
                    raise RuntimeError("AST session admitted no branch rows")
                for branch_index in range(len(blocks)):
                    result = branch_results.get((session_index, branch_index))
                    if result is None:
                        decisions.append(
                            RoundDecision(
                                accepted=-1,
                                emitted=[],
                                target_laws=[],
                                commit_count=0,
                                committed_inputs=[],
                            )
                        )
                        features.append(
                            self.mx.zeros_like(session_template)[:, :0]
                        )
                        caches.append(prefix_caches[session_index])
                        rngs.append(prefix_rngs[session_index])
                        stats_rows.append(
                            {
                                "launches": 0,
                                "target_rows": 0,
                                "rejected": 0,
                                "full_tiles": 0,
                                "tile_checkpoints": [],
                                "tile_input_rows": [],
                                "rollback_base_tile": 0,
                            }
                        )
                        continue
                    decision, feature, cache, rng, stats = result
                    decisions.append(
                        RoundDecision(
                            accepted=prefix_count + int(decision.accepted),
                            emitted=list(prefix.emitted) + list(decision.emitted),
                            target_laws=list(prefix.target_laws)
                            + list(decision.target_laws),
                            response_logprobs=(
                                None
                                if prefix.response_logprobs is None
                                else list(prefix.response_logprobs)
                                + list(decision.response_logprobs)
                            ),
                            verify_window=decision.verify_window,
                            commit_count=int(prefix.commit_count)
                            + int(decision.commit_count),
                            committed_inputs=list(prefix.committed_inputs)
                            + list(decision.committed_inputs),
                        )
                    )
                    branch_feature = feature[:, : int(decision.commit_count)]
                    if prefix_count:
                        branch_feature = self.mx.concatenate(
                            [
                                prefix_features[session_index][
                                    :, : int(prefix.commit_count)
                                ],
                                branch_feature,
                            ],
                            axis=1,
                        )
                    features.append(branch_feature)
                    caches.append(cache)
                    rngs.append(rng)
                    stats_rows.append(stats)
                winner, full = ordered_alternative_winner(
                    decisions,
                    proposal_counts,
                    allow_partial_uplift=bool(
                        getattr(self, "continuation_pool_ast_bridge", False)
                    ),
                )
                committed_tokens = ordered_branch_commit_counts(decisions)
                selected_feature = features[winner]
                selected_cache = caches[winner]
                selected_rng = rngs[winner]
                stats = {
                    **stats_rows[winner],
                    "launches": int(prefix_stats[session_index]["launches"])
                    + sum(int(row["launches"]) for row in stats_rows),
                    "target_rows": int(prefix_stats[session_index]["target_rows"])
                    + sum(int(row["target_rows"]) for row in stats_rows),
                    "rejected": int(prefix_stats[session_index]["rejected"])
                    + sum(int(row["rejected"]) for row in stats_rows),
                    "full_tiles": int(prefix_stats[session_index]["full_tiles"])
                    + sum(int(row["full_tiles"]) for row in stats_rows),
                    "tile_checkpoints": list(
                        prefix_stats[session_index]["tile_checkpoints"]
                    )
                    + list(stats_rows[winner]["tile_checkpoints"]),
                    "tile_input_rows": list(
                        prefix_stats[session_index]["tile_input_rows"]
                    )
                    + list(stats_rows[winner]["tile_input_rows"]),
                }
                branch_accepts = [int(decision.accepted) for decision in decisions]
                unique_target_rows = int(stats["target_rows"])
            selected_decisions.append(decisions[winner])
            selected_features.append(selected_feature)
            selected_caches.append(selected_cache)
            selected_rngs.append(selected_rng)
            selected_stats.append(stats)
            selected_blocks.append(blocks[winner])
            self._record_ast_geometry(
                lane, blocks, branch_accepts, proposal_counts, committed_tokens, winner,
                {
                    "physical_target_rows": unique_target_rows,
                    "shared_prefix_target_rows": prefix_stats[session_index]["target_rows"],
                    "shared_prefix_tokens": prefix_count,
                    "prefix_rejected": int(prefix.accepted) != prefix_count,
                },
            )
            selections.append(
                {
                    "session_uid": int(lane.uid),
                    "branch_accepts": branch_accepts,
                    "proposal_counts": list(proposal_counts),
                    "full": list(full),
                    "winner": int(winner),
                    "branch_committed_tokens": list(committed_tokens),
                    "selected_token_uplift": max(
                        0, int(committed_tokens[winner]) - int(committed_tokens[0])
                    ),
                    "unique_target_rows": unique_target_rows,
                }
            )

        feature_width = max(int(row.shape[1]) for row in selected_features)
        padded_features = []
        for row in selected_features:
            missing = feature_width - int(row.shape[1])
            if missing:
                row = self.mx.concatenate(
                    [
                        row,
                        self.mx.zeros(
                            (1, missing, int(row.shape[2])), dtype=row.dtype
                        ),
                    ],
                    axis=1,
                )
            padded_features.append(row)
        frontier = None
        if frontier_planner is not None:
            frontier = frontier_planner.snapshot()
            if not frontier["finished"]:
                raise RuntimeError("ordered live frontier did not retire every session")
            occupancy_numerator = sum(int(pack["logical_rows"]) for pack in packs)
            occupancy_denominator = len(packs) * pack_rows
            last_service = {}
            max_service_gap = 0
            for launch_index, pack in enumerate(packs):
                for session_slot in pack["session_slots"]:
                    if session_slot in last_service:
                        max_service_gap = max(
                            max_service_gap,
                            launch_index - last_service[session_slot],
                        )
                    last_service[session_slot] = launch_index
            frontier.update(
                {
                    "occupancy_numerator": occupancy_numerator,
                    "occupancy_denominator": occupancy_denominator,
                    "partial_packs": sum(
                        int(pack["logical_rows"] < pack_rows) for pack in packs
                    ),
                    "full_packs": sum(
                        int(pack["logical_rows"] == pack_rows) for pack in packs
                    ),
                    "max_service_gap": max_service_gap,
                }
            )
        physical = {
            "physical_launches": physical_launches,
            "physical_target_rows": physical_target_rows,
            "ragged_launches": ragged_launches,
            "padding_rows": padding_rows,
            "last_ragged_layout": last_ragged_layout,
            "max_cohort_width": max_cohort_width,
            "target_row_executor": target_executor,
            "aligned_tile_launches": aligned,
            "aligned_continued_launches": continued,
            "b2_aligned_tile_launches": aligned[2],
            "b2_aligned_continued_launches": continued[2],
            "b4_aligned_tile_launches": aligned[4],
            "b4_aligned_continued_launches": continued[4],
            "b8_aligned_tile_launches": aligned.get(8, 0),
            "b8_aligned_continued_launches": continued.get(8, 0),
            "first_tile_pairwise_disjoint": False,
            "physical_row_cap": pack_rows,
            "cross_session_ast": True,
            "cross_session_packs": packs,
            "cross_session_ast_mode": (
                "ordered_live_frontier" if live_frontier else "eager_all_branches"
            ),
            "cross_session_apt_mode": (
                "ordered_live_frontier" if live_frontier else "eager_all_branches"
            ),
            "deprecated_cross_session_apt_mode_alias_for": (
                "cross_session_ast_mode"
            ),
            "live_frontier": frontier,
        }
        return (
            selected_decisions,
            self.mx.concatenate(padded_features, axis=0),
            selected_caches,
            selected_rngs,
            selected_stats,
            physical,
            selections,
            selected_blocks,
        )

    def _progressive_verify(self, lane, block):
        """Compatibility wrapper for the exact B1 progressive contract."""

        decisions, features, caches, rngs, stats, _physical = (
            self._progressive_verify_many([lane], [block])
        )
        return decisions[0], features, caches[0], rngs[0], stats[0]


    def _verify(self, cohort, blocks, logits):
        """Accept phase over the target logits of one verify forward."""
        vocab = int(logits.shape[-1])
        decisions = []
        for row, lane in enumerate(cohort):
            fly = self.fly_verification.enabled and not lane.processors
            # A greedy row's target law is one-hot: verify by token compare
            # instead of 2K+1 dense vocabulary laws (3.6 ms per lane-round at
            # K=7 and a 248K vocabulary).  Same decisions, same RNG schedule.
            greedy = float(lane.sampling.get("sampling_temp", 0)) == 0 and not fly
            drafts, laws = _block_row(
                blocks[row],
                None if greedy and hasattr(blocks[row], "cand_ids") else vocab,
            )
            if greedy:
                _greedy_proposal_supported(blocks[row], laws, drafts, vocab)
            inputs = [lane.anchor] + drafts
            targets, reachable = [], True
            response_rows = [] if lane.sampling.get("emit_logprobs", True) else None
            count = len(drafts)
            # The processors run on every reachable row before the accept
            # point is known; the window rewinds the bookkeeping of the rows
            # past it (steps, tail bound, a latched budget overrun) at commit.
            window = VerifyWindow(lane.processors)
            for j in range(count+1):
                targets.append(self._target_law(
                    lane, logits[row, j], lane.history, reachable, response_rows,
                    greedy_token=greedy, history_suffix=inputs[:j + 1],
                ))
                window.mark()
                if reachable and j < count and lane.processors and (
                    targets[-1] != int(drafts[j]) if greedy
                    else targets[-1][int(drafts[j])] <= 0
                ):
                    reachable = False
            if greedy:
                result = verify_greedy_proposals(drafts, targets, lane.rng)
            elif isinstance(laws, (list, tuple)) and all(
                isinstance(law, PointMassLaw) for law in laws
            ):
                candidate_ids = np.asarray(
                    [[law.token] for law in laws], dtype=np.int64
                )
                candidate_probs = np.ones((len(laws), 1), dtype=np.float64)
                result = verify_compact_proposals(
                    drafts,
                    candidate_ids,
                    candidate_probs,
                    targets,
                    lane.rng,
                )
            elif getattr(self, "exact_verification", "token") == "block":
                # A draft the processed target forbids zeroes P from its row
                # on, so the unprocessed rows past it cannot change tau.
                if isinstance(laws, CompactDraftRow):
                    result = verify_compact_block_proposals(
                        drafts, laws.candidate_ids, laws.candidate_probs,
                        targets, lane.rng,
                    )
                else:
                    result = verify_block_proposals(drafts, laws, targets, lane.rng)
                lane.block_verified_rounds = getattr(lane, "block_verified_rounds", 0) + 1
            elif isinstance(laws, CompactDraftRow):
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
                    # A greedy row publishes its processed log-softmax rows.
                    None if greedy else result.target_probabilities,
                    result.relaxed_accepts,
                    response_rows,
                    window,
                    relaxed_positions=result.relaxed_positions,
                )
            )
        return decisions

    def _verify_trusted_pld(self, lane, block, logits):
        """Approximate PLD: trust copied rows and sample only the target bonus."""
        if (
            not isinstance(block, HostDraftRow)
            or block.proposal_source != "prompt_lookup"
            or not block.proposal_trusted
            or block.proposal_audit
            or lane.processors
            or logits.shape[0] != 1
            or logits.shape[1] != 1
        ):
            raise ValueError("trusted PLD requires one unprocessed host copy row")
        drafts = list(block.tokens)
        response_rows = [] if lane.sampling.get("emit_logprobs", True) else None
        law = self._target_law(
            lane,
            logits[0, 0],
            lane.history,
            True,
            response_rows,
            history_suffix=[lane.anchor, *drafts],
        )
        bonus = lane.rng.sample(law)
        emitted = [*drafts, bonus]
        for index, token in enumerate(emitted):
            if token in self.stops:
                emitted = emitted[: index + 1]
                break
        laws = [None] * len(drafts) + [law]
        rows = (
            None
            if response_rows is None
            else [None] * len(drafts) + list(response_rows)
        )
        return RoundDecision(
            len(drafts),
            emitted,
            laws[: len(emitted)],
            response_logprobs=(None if rows is None else rows[: len(emitted)]),
            verify_window=VerifyWindow(()),
        )

    @staticmethod
    def _target_tree_parents(block):
        return [-1] + [0 if int(parent) < 0 else int(parent) + 1 for parent in block.parents]

    @staticmethod
    def _tree_paths(parents):
        from .drafters.dflash_tree import tree_paths

        return tree_paths(parents)

    @staticmethod
    def _copy_reference_cache(cache, *, preserve_capacity=False):
        """Clone a B1 cache and force KV appends onto fresh buffers."""

        cloned = copy.deepcopy(cache)
        for item in cloned:
            if (
                hasattr(item, "keys")
                and getattr(item, "keys", None) is not None
                and isinstance(getattr(item, "offset", None), int)
            ):
                offset = int(item.offset)
                if preserve_capacity:
                    # Qualified request classes bind the allocation capacity,
                    # not only the written prefix.  Detach the full arrays so
                    # the private transaction preserves that exact schema.
                    item.keys = item.keys * 1
                    item.values = item.values * 1
                else:
                    item.keys = item.keys[..., :offset, :]
                    item.values = item.values[..., :offset, :]
        return cloned

    def _copy_ast_private_cache(self, cache):
        """Fork only committed COW descriptors, never a general deep copy.

        Private cache objects remain necessary: a failed later tile must not
        publish changes to the original lane. Immutable tensor storage is
        aliased, with the same prefix-view append geometry as the reference.
        Unsupported descriptor graphs fail closed before native mutation.
        """
        captured = self._ast_descriptor_capture(cache)
        if captured is None:
            cloned, _sidecar, _receipt = snapshot_prompt_cache_descriptors(cache)
        else:
            cloned, _borrowed = captured
            _bump(self.scheduler_stats, "external_ast_lean_private_forks")
        from .cow_cache import lend_append_only_kv

        for live, item in zip(cache, cloned, strict=True):
            if (
                getattr(item, "keys", None) is not None
                and isinstance(getattr(item, "offset", None), int)
                and not lend_append_only_kv(live, item)
            ):
                offset = int(item.offset)
                item.keys = item.keys[..., :offset, :]
                item.values = item.values[..., :offset, :]
        _bump(self.scheduler_stats, "external_ast_private_descriptor_forks")
        return cloned

    def _ast_descriptor_capture(self, cache, *, recovery=False):
        # This generator is the serving-session owner. Plans retain only
        # types/operations, never lane objects, buffers, or mutable feedback.
        plans = getattr(self, "_ast_descriptor_plans", None)
        if plans is None:
            plans = self._ast_descriptor_plans = {}
        key = tuple(type(item) for item in cache)
        scope = self.binding
        plan = plans.get(key)
        if plan is None or plan.scope != scope:
            if len(plans) >= 8:
                plans.clear()
            plan = plans[key] = CommittedDescriptorPlan(cache, scope)
        result = plan.capture(cache, scope, recovery=recovery)
        if result is None:
            # A changed class operation/layout may become supported again;
            # no stale plan is allowed to silently retain admission.
            plans.pop(key, None)
            _bump(self.scheduler_stats, "external_ast_lean_descriptor_declines")
        return result

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
        cached = self.tree_gates["cache_executor"]
        result = self.target_tree_executor_provider.forward(
            self.model,
            inputs,
            parents,
            lane.cache,
            self.layers,
            cached=cached,
        )
        return self._finish_target_tree_dispatch([lane], result, cached=cached)

    def _target_tree_forward_many(self, cohort, inputs, parents):
        """Queue one independent provider tree per lane as one cohort."""

        if len(cohort) == 1:
            return self._target_tree_forward(cohort[0], inputs[0], parents[0])
        if self.target_execution != "provider":
            raise ValueError("multi-lane tree execution requires a target-tree provider")
        if len(cohort) > self.target_tree_cohort_limit:
            raise ValueError("target-tree cohort exceeds its bounded lane limit")
        cached = self.tree_gates["cache_executor"]
        ragged = bool(getattr(self, "ast_batch_row_sharing", False))
        if not ragged:
            inputs, parents = _pad_cohort_trees(inputs, parents)
        result = self.target_tree_executor_provider.forward_many(
            self.model,
            inputs,
            parents,
            [lane.cache for lane in cohort],
            self.layers,
            cached=cached,
            configured_limit=self.target_tree_cohort_limit,
            **({"ragged": True} if ragged else {}),
        )
        return self._finish_target_tree_dispatch(cohort, result, cached=cached)

    def _ast_target_tree_forward(self, lane, cache, inputs, *, parents=None):
        """One private chain or admitted parent-closed AST tile.

        This deliberately accepts one lane and one private cache only. There
        is no sibling cache copy/share operation in the phase-1 contract.
        """
        if (getattr(self, "ast_b1_tile_policy", None) == "four_path15"
            or getattr(self, "ast_packed_pattern", None)) and len(inputs) < 2:
            raise ValueError("four_path15 cannot submit an anchor-only TensorFold tile")

        provider = self.ast_target_tree_executor_provider
        if provider is None:
            raise RuntimeError("AST target-tree backend has no adapter provider")
        provider.validate_cohort(1, 1)
        if parents is None:
            parents = list(range(-1, len(inputs) - 1))
        elif (
            not getattr(self, "ast_tree_tile_rows", 0)
            or len(parents) != len(inputs)
            or len(inputs) > self.ast_tree_tile_rows
        ):
            raise ValueError("AST provider has no matching tree-tile capability")
        # AST admission already binds this immutable provider and its validated
        # module set. Legacy ordinary-tree experiment gates do not govern it.
        cached = True
        result = provider.forward(
            self.model,
            inputs,
            parents,
            cache,
            self.layers,
            cached=cached,
        )
        return self._finish_target_tree_dispatch(
            [lane],
            result,
            cached=cached,
            provider=provider,
            publish_tree_metrics=False,
        )

    def _ast_tree_cohort_limit(self):
        """Request ownership capacity is distinct from a target tile's rows."""
        provider = getattr(self, "ast_target_tree_executor_provider", None)
        maximum = getattr(provider, "ast_tree_max_sessions", 1)
        rows = getattr(self, "ast_tree_tile_rows", 0)
        if (
            not rows or type(maximum) is not int or not 1 <= maximum <= 4
            or not callable(getattr(provider, "ast_tree_cohort_input_widths", None))
        ):
            return 1
        floor = max(1, min(
            getattr(self, "num_draft", 1), getattr(self, "minimum_draft_proposals", 1)
        ))
        return max(1, min(maximum, rows // (floor + 1)))

    def _ast_target_tree_forward_many(self, lanes, caches, inputs, *, parents):
        """Dispatch an admitted equal-node cohort with separate private state."""
        provider = self.ast_target_tree_executor_provider
        count = len(lanes)
        if count == 1:
            return self._ast_target_tree_forward(
                lanes[0], caches[0], inputs[0], parents=parents[0]
            )
        widths = getattr(provider, "ast_tree_cohort_input_widths", None)
        maximum = (
            self._ast_tree_anchor_limit()
            if inputs and all(len(row) == 1 for row in inputs)
            else self._ast_tree_cohort_limit()
        )
        if (
            not callable(widths) or not 2 <= count <= maximum
            or len(caches) != count or len(inputs) != count or len(parents) != count
            or len({len(row) for row in inputs}) != 1
            or len(inputs[0]) not in widths(count)
            or sum(map(len, inputs)) > self.ast_tree_tile_rows
            or any(len(row) != len(parent) for row, parent in zip(inputs, parents, strict=True))
        ):
            raise ValueError("AST target-tree cohort has no admitted physical geometry")
        provider.validate_cohort(count, provider.ast_tree_max_sessions)
        result = provider.forward_many(
            self.model, inputs, parents, caches, self.layers,
            cached=True, configured_limit=provider.ast_tree_max_sessions,
        )
        return self._finish_target_tree_dispatch(
            lanes, result, cached=True, provider=provider, publish_tree_metrics=False,
        )

    def _ast_tree_anchor_limit(self):
        """Admitted private anchor rows need no proposal-depth allocation."""
        provider = getattr(self, "ast_target_tree_executor_provider", None)
        maximum = getattr(provider, "ast_tree_max_sessions", 1)
        widths = getattr(provider, "ast_tree_cohort_input_widths", None)
        if (
            not getattr(self, "ast_tree_tile_rows", 0)
            or type(maximum) is not int or not 1 <= maximum <= 4
            or not callable(widths)
        ):
            return 1
        return max(
            (width for width in range(2, maximum + 1) if 1 in widths(width)),
            default=1,
        )

    def _ast_anchor_cohort_eligible(self, cohort):
        return (
            2 <= len(cohort) <= self._ast_tree_anchor_limit()
            and self.target_execution == "reference"
            and getattr(getattr(self.model, "model", None), "residual_taps", None) is None
            and all(
                not lane.ordinary
                and float(lane.sampling.get("sampling_temp", 0)) == 0.0
                and self._progressive_processors_supported(lane)
                and self._batched_law_contract(lane)
                for lane in cohort
            )
        )

    def _ast_verify_primary_tree_cohort(self, lanes, blocks):
        """Collapse independent primary results into the unchanged commit ABI."""
        from .ast_tree_verify import verify_tree_tiles_many

        results = verify_tree_tiles_many(self, lanes, [[block] for block in blocks])
        if len(results) != len(lanes) or any(len(row[0]) != 1 for row in results):
            raise RuntimeError("AST primary cohort returned a foreign lane layout")
        feature_width = max(int(row[1].shape[1]) for row in results)
        features = []
        for result in results:
            row = result[1]
            missing = feature_width - int(row.shape[1])
            if missing:
                row = self.mx.concatenate(
                    [row, self.mx.zeros((1, missing, int(row.shape[2])), dtype=row.dtype)],
                    axis=1,
                )
            features.append(row)
        physical_rows = [row[5] for row in results]
        physical = {
            name: sum(row.get(name, 0) for row in physical_rows)
            for name in ("physical_launches", "physical_target_rows", "unique_target_rows", "padding_rows", "ragged_launches")
        }
        physical.update(
            max_cohort_width=max(row["max_cohort_width"] for row in physical_rows),
            target_row_executor=_merge_target_executor_stats(
                *(row.get("target_row_executor") for row in physical_rows)
            ),
            first_tile_pairwise_disjoint=False,
        )
        for name in ("aligned_tile_launches", "aligned_continued_launches"):
            physical[name] = {
                width: sum(row[name].get(width, 0) for row in physical_rows)
                for width in range(1, 5)
            }
        return (
            [row[0][0] for row in results], self.mx.concatenate(features, axis=0),
            [row[2][0] for row in results], [row[3][0] for row in results],
            [row[4][0] for row in results], physical,
        )

    @staticmethod
    def _tree_forbidden(lane, history_length=None):
        """The proposer's forbidden ids at the lane's next position.

        Only when every processor declares the stateless mask; otherwise
        the lattice stays unmasked and verification alone enforces them.
        ``history_length`` names a committed length other than the lane's
        current one (the boundary a pending commit is about to publish).
        """

        length = len(lane.history) if history_length is None else int(history_length)
        ids = set()
        for processor in lane.processors:
            rule = getattr(processor, "forbidden_token_ids_at_length", None)
            if not callable(rule):
                return ()
            ids.update(int(token) for token in rule(length + 1))
        return tuple(sorted(ids))

    @staticmethod
    def _calibration_key(lane):
        """Greedy and sampled lanes accept by different events: own tables."""

        sampling = getattr(lane, "sampling", None) or {}
        return "sampled" if float(sampling.get("sampling_temp", 0) or 0) != 0.0 else "greedy"

    def _tree_options(self, *, count_hits, lanes=None):
        options = {}
        calibration = getattr(self, "tree_calibration", None)
        if calibration is not None and lanes is not None:
            options["calibration"] = [
                calibration.params(self._calibration_key(lane)) for lane in lanes
            ]
        if getattr(self.draft, "tree_topology_aware", False):
            from .ast_tree_proposals import DEFAULT_TOPOLOGY

            options["topology"] = self.ast_topology or DEFAULT_TOPOLOGY
        if self.tree_gates["codebook_cache"] and callable(
            getattr(self.draft, "tree_codebooks", None)
        ):
            pred, succ, hit = self.draft.tree_codebooks()
            options["codebooks"] = (pred, succ)
            if hit:
                _bump(self.scheduler_stats, "external_tree_codebook_cache_hits")
        return options

    def _tree_node_budget(self, active_width=None):
        width = max(
            1,
            int(self._auto_active_width if active_width is None else active_width),
        )
        width = min(width, self.dynamic_tree_max_width)
        budgets = getattr(
            self,
            "tree_node_budget_by_lanes",
            {candidate: 15 for candidate in range(1, self.dynamic_tree_max_width + 1)},
        )
        if width > 1 and getattr(self, "ast_batch_row_sharing", False):
            # Each lane drafts deep enough to take the cohort's whole share;
            # ``_share_cohort_rows`` then cuts the trees to the tile.
            return min(15, self._shared_cohort_rows(width) - width)
        controller = getattr(self, "tile_budget", None)
        if controller is not None:
            return controller.choose(width, budgets[width])
        return budgets[width]

    def _shared_cohort_rows(self, width):
        """Physical rows of a ``width``-lane tile: the equal split's total."""

        budgets = self.tree_node_budget_by_lanes
        return int(width) * (int(budgets[int(width)]) + 1)

    def _share_cohort_rows(self, cohort, blocks):
        """Give the tile's rows to the cohort's most likely nodes.

        Every lane keeps its anchor.  The other rows go to nodes in
        decreasing calibrated path probability across the cohort (ties: lane
        then node order), so each lane keeps a best-first prefix of its own
        tree.  Which rows are verified depends only on the drafter's trees,
        never on the target, so the verification law is unchanged.
        """

        if len(cohort) < 2 or any(block.info is None for block in blocks):
            return blocks
        capacity = self._shared_cohort_rows(len(cohort)) - len(cohort)
        ranked = sorted(
            (-value, lane, node)
            for lane, block in enumerate(blocks)
            for node, value in enumerate(block.info["path_log_p"])
        )
        keep = [0] * len(blocks)
        for _value, lane, _node in ranked[:capacity]:
            keep[lane] += 1
        shared = [
            TreeDraftRow(
                list(block.tokens[:count]), list(block.parents[:count]), block.width,
                info={key: list(value[:count]) for key, value in block.info.items()},
            )
            for block, count in zip(blocks, keep)
        ]
        _bump(self.scheduler_stats, "external_tree_row_sharing_rounds")
        for count in keep:
            _bump_map(self.scheduler_stats, "external_tree_row_sharing_lane_nodes", int(count))
        return shared

    def _chain_draft_cap(self, active_width=None):
        width = self._auto_active_width if active_width is None else active_width
        if not getattr(self, "dynamic_singleton_tree", False):
            return None
        return _chain_draft_cap(
            self.tree_node_budget_by_lanes, self.dynamic_tree_max_width, width
        )

    def verify_rows_by_width(self, max_lanes):
        """The widest target verify call, by cohort width, this route can issue.

        A tree round verifies the width's node budget plus the root a lane --
        one lane a forward under reference execution, up to the provider's
        cohort limit otherwise: the fixed ``tree15`` topology at every width,
        the dynamic route only up to its widest tree.  A chain verifies ``num_draft + 1`` rows a lane,
        capped past the widest tree as ``_chain_draft_cap`` caps it.  A
        continuation-pool round verifies one lane's paths in one forward --
        up to ``limit`` paths of ``num_draft`` tokens plus the anchor -- or,
        longest-first, one path a forward.  Upper
        bounds: a round may draft fewer.  Lane ``exact`` mode refuses a route
        whose window does not hold them.
        """
        from .proposal_providers import LONGEST_FIRST_EXACT_PREFIX

        tree_rows = getattr(self, "ast_tree_tile_rows", 0)
        if tree_rows:
            # Source depth and proposal count do not widen a physical tile;
            # each admitted private request runs through this same bound.
            return {str(width): tree_rows for width in range(1, int(max_lanes) + 1)}
        dynamic = bool(getattr(self, "dynamic_singleton_tree", False))
        continuation = getattr(self, "continuation_policy", None)
        continuation_rows = 0
        if continuation is not None and not getattr(self, "continuation_pool_ast_bridge", False):
            # Parallel verification forwards every path of one lane together
            # (max_depth=num_draft); longest-first verifies one path a launch.
            strategy = getattr(getattr(self, "continuation_strategy", None), "algorithm", None)
            serial = LONGEST_FIRST_EXACT_PREFIX in (
                strategy, getattr(continuation, "verification_algorithm", None)
            )
            continuation_rows = (1 if serial else int(continuation.limit)) * (self.num_draft + 1)
        rows = {}
        for lanes in range(1, int(max_lanes) + 1):
            lane_continuation_rows = (
                self._continuation_ast_bridge_geometry(lanes)[
                    "maximum_target_rows_per_launch"
                ]
                if continuation is not None
                and getattr(self, "continuation_pool_ast_bridge", False)
                else continuation_rows
            )
            tree = (
                lanes <= self.dynamic_tree_max_width
                if dynamic
                else self.draft_topology == "tree15"
            )
            if tree:
                # Reference execution runs each lane's tree alone; a provider
                # packs at most its cohort limit of lanes into one forward.
                tree_lanes = lanes if getattr(self, "target_execution", None) == "provider" else 1
                tree_lanes = min(tree_lanes, int(getattr(self, "target_tree_cohort_limit", lanes)))
                if tree_lanes > 1 and getattr(self, "ast_batch_row_sharing", False):
                    # Shared rows: the cohort's tile keeps the equal split's total.
                    width_rows = self._shared_cohort_rows(tree_lanes)
                else:
                    width_rows = tree_lanes * (self._tree_node_budget(lanes) + 1)
            else:
                cap = self._chain_draft_cap(lanes)
                depth = self.num_draft if cap is None else min(self.num_draft, cap)
                width_rows = lanes * (depth + 1)
            rows[str(lanes)] = max(width_rows, lane_continuation_rows)
        return rows

    def _continuation_ast_bridge_geometry(self, max_lanes):
        """Bound live AST rows and retained request-private checkpoints."""

        if not getattr(self, "continuation_pool_ast_bridge", False):
            raise ValueError("continuation AST geometry requires the selected bridge")
        sessions = max(1, min(int(max_lanes), int(self.capacity), 4))
        branches = int(self.progressive_alternative_branches)
        tree_rows = getattr(self, "ast_tree_tile_rows", 0)
        if tree_rows:
            return {
                "schema": "mlx2.continuation-ast-bridge-geometry.v1",
                "sessions": sessions,
                "branches_per_session": branches,
                "physical_path_rows": 1,
                "tile_rows_per_path": tree_rows,
                "maximum_target_rows_per_launch": tree_rows,
                "private_target_checkpoints": sessions * branches,
                "private_provider_cache_rows": sessions,
                "live_frontier": True,
                "dispatch": "fair_private_request_trees",
            }
        pack_rows = self.progressive_cross_session_pack_rows
        live_frontier = (
            self.progressive_cross_session_pack_mode == "ordered_live_frontier"
        )
        if branches == 1:
            # Primary-only AST batches independent requests, not siblings.
            live_paths = sessions
        elif pack_rows is None:
            sessions = 1
            live_paths = branches
        elif live_frontier:
            live_paths = sessions
        else:
            live_paths = min(int(pack_rows), sessions * branches)
        tile_rows = int(self.progressive_verification_tile) + 1
        return {
            "schema": "mlx2.continuation-ast-bridge-geometry.v1",
            "sessions": sessions,
            "branches_per_session": branches,
            "physical_path_rows": live_paths,
            "tile_rows_per_path": tile_rows,
            "maximum_target_rows_per_launch": live_paths * tile_rows,
            "private_target_checkpoints": sessions * branches,
            "private_provider_cache_rows": sessions,
            "live_frontier": live_frontier,
        }

    def _tree_count(self, lane):
        return min(self._tree_node_budget(), lane.maximum - lane.generated - 1)

    def _lattice_submit_option(self):
        """Queue a prelaunched lattice's draft layers as they are built.

        A prelaunch is evaluated asynchronously anyway; submitting layer by
        layer lets the device start on the first draft layer while the host
        still builds the others.  Only drafters that declare the option.
        """
        return {"submit": True} if getattr(self.draft, "start_tree_submits", False) else {}

    def _presnapshot_draft(self, cohort):
        """Copy a B1 lane's draft plane while the host waits on the tile.

        The next lattice runs on a descriptor copy of the draft plane, which
        nothing in a tree round writes after the proposal is adopted.  Taking
        the copy here moves it off the path between the tile and the next
        lattice; ``_start_lattice`` uses it only for the same lane and the
        same draft-plane object, and copies afresh otherwise.
        """

        self._draft_presnapshot = None
        if (
            len(cohort) != 1
            or not self.tree_gates["pipeline_draft"]
            or not callable(getattr(self.draft, "start_tree", None))
            or self.pair_context_tokens
        ):
            return
        lane = cohort[0]
        try:
            draft_cache, _tail = self._snapshot_draft_state(lane.draft_cache, None)
        except Exception:
            return
        self._draft_presnapshot = (lane, lane.draft_cache, draft_cache)

    def _start_lattice(self, lane, anchor, tail, count, forbidden):
        """Snapshot the draft plane and queue one lattice on the copy."""

        mark = self._tree_clock
        options = self._tree_options(count_hits=False, lanes=[lane])
        if mark is not None:
            mark("tree_prelaunch_setup")
        pre = getattr(self, "_draft_presnapshot", None)
        self._draft_presnapshot = None
        if pre is not None and pre[0] is lane and pre[1] is lane.draft_cache:
            draft_cache, tail_copy = pre[2], tail
        else:
            draft_cache, tail_copy = self._snapshot_draft_state(lane.draft_cache, tail)
        if mark is not None:
            mark("tree_prelaunch_snapshot")
        state = self.draft.start_tree(
            [anchor],
            tail_copy,
            self.draft.batch_caches([draft_cache]),
            count,
            forbidden_token_ids=[forbidden],
            **options,
            **self._lattice_submit_option(),
        )
        if mark is not None:
            mark("tree_prelaunch_build")
        self.mx.async_eval(*state["pending"])
        if mark is not None:
            mark("tree_prelaunch_submit")
        return SimpleNamespace(
            tail=tail,
            options=options,
            state=state,
            draft_cache=draft_cache,
        )

    def _prelaunch_tree(self, lane, decision):
        """Queue the next round's lattice right after this round's commit.

        The lattice runs on a descriptor copy of the draft cache, so the
        committed draft plane is untouched: the copy becomes the lane's only
        when the next round adopts it inside its own transaction, after its
        recovery snapshot, and only if the committed boundary still matches.
        Any failure here discards the work; the committed round stands.
        """

        self._discard_prelaunched([lane.uid])
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
            record = self._start_lattice(lane, lane.anchor, lane.tail, count, forbidden)
        except Exception:
            _bump(self.scheduler_stats, "external_tree_pipeline_discards")
            return
        record.key = (len(lane.history), lane.anchor, lane.generated, count, forbidden)
        self._prelaunched[lane.uid] = record

    def _prelaunch_tree_early(self, lane, decision, features):
        """Queue the next lattice from a verified, not yet committed round.

        Everything the lattice reads is already fixed by the walk: the new
        anchor (the last emitted token), the committed feature rows (the
        accepted path's rows of ``features``, the exact array ``_commit``
        then publishes as ``lane.tail``) and the draft plane, which a target
        commit never touches.  Submitting it before the commit lets the
        drafter run on the device while the host commits and publishes.
        The record is bound to the post-commit boundary it predicts and is
        kept only if ``_bind_early_prelaunch`` finds that boundary exactly.
        """

        rows = decision.commit_rows
        emitted = decision.emitted
        if rows is None or not emitted:
            return None
        if len(rows) != len(emitted):
            return None
        tail = self.mx.take(
            features[0:1], self.mx.array(list(rows), dtype=self.mx.int32), axis=1
        )
        decision.committed_tail = tail
        generated = lane.generated + len(emitted)
        if (
            lane.ordinary
            or lane.cancelled
            or lane.maximum - generated <= 1
            or emitted[-1] in self.stops
        ):
            return None
        history = len(lane.history) + len(rows)
        anchor = emitted[-1]
        try:
            count = min(self._tree_node_budget(), lane.maximum - generated - 1)
            forbidden = self._tree_forbidden(lane, history)
            record = self._start_lattice(lane, anchor, tail, count, forbidden)
        except Exception:
            _bump(self.scheduler_stats, "external_tree_pipeline_discards")
            return None
        record.key = (history, anchor, generated, count, forbidden)
        return record

    def _bind_early_prelaunch(self, lane, decision, record):
        """Keep an early lattice only at the exact boundary it predicted."""

        self._discard_prelaunched([lane.uid])
        if (
            record is not None
            and not lane.ordinary
            and not lane.cancelled
            and record.tail is lane.tail
            and record.key == (
                len(lane.history), lane.anchor, lane.generated,
                self._tree_count(lane), self._tree_forbidden(lane),
            )
        ):
            self._prelaunched[lane.uid] = record
            _bump(self.scheduler_stats, "external_tree_early_prelaunches")
            return
        if record is not None:
            _bump(self.scheduler_stats, "external_tree_pipeline_discards")
        self._prelaunch_tree(lane, decision)

    def _adopt_prelaunched(self, lane, count, forbidden):
        """Return the queued trees if they belong to this exact boundary."""

        record = self._prelaunched.pop(lane.uid, None)
        if record is None:
            # A multi-lane round queues one lattice per equal-tail group; a
            # lane alone in its group comes back as a singleton group here.
            group = self._prelaunched.pop(("cohort", lane.uid), None)
            if group is None:
                return None
            record = SimpleNamespace(
                tail=group.tails[0], key=group.keys[0],
                state=group.state, draft_cache=group.draft_caches[0],
            )
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

    def _prelaunch_tree_cohort(self, cohort, decisions, views=None):
        """Queue the next round's batched lattices for a multi-lane cohort.

        Lanes are grouped as ``_propose`` groups them (equal draft tails), so
        each group's lattice is exactly the batched draft the next round
        would build; it is adopted only if the same lanes return at the same
        committed boundaries (see ``_adopt_prelaunched_cohort``).

        ``views`` (one per lane: anchor, tail, history and generated
        lengths) names a boundary a pending commit is about to publish; the
        records are then returned for ``_bind_early_cohort`` instead of
        being queued.
        """

        early = views is not None
        if not early:
            self._discard_prelaunched([lane.uid for lane in cohort])
            views = [
                SimpleNamespace(
                    anchor=lane.anchor, tail=lane.tail,
                    history=len(lane.history), generated=lane.generated,
                )
                for lane in cohort
            ]
        records = {}
        if any(
            lane.ordinary or lane.cancelled or lane.maximum - view.generated <= 1
            or (decision.emitted and decision.emitted[-1] in self.stops)
            or _speculation_parked(lane)
            for lane, decision, view in zip(cohort, decisions, views)
        ):
            return records
        groups = {}
        for lane, view in zip(cohort, views):
            groups.setdefault(
                (int(view.tail.shape[1]), str(view.tail.dtype)), []
            ).append((lane, view))
        budget = self._tree_node_budget()
        for members in groups.values():
            lanes = [lane for lane, _view in members]
            try:
                count = min(budget, min(l.maximum - v.generated - 1 for l, v in members))
                forbidden = [self._tree_forbidden(lane, view.history) for lane, view in members]
                options = self._tree_options(count_hits=False, lanes=lanes)
                snapshots = [
                    self._snapshot_draft_state(lane.draft_cache, view.tail)
                    for lane, view in members
                ]
                state = self.draft.start_tree(
                    [view.anchor for _lane, view in members],
                    self.mx.concatenate([tail for _cache, tail in snapshots], axis=0),
                    self.draft.batch_caches([cache for cache, _tail in snapshots]),
                    count,
                    forbidden_token_ids=forbidden,
                    **options,
                    **self._lattice_submit_option(),
                )
                self.mx.async_eval(*state["pending"])
            except Exception:
                _bump(self.scheduler_stats, "external_tree_pipeline_discards")
                continue
            records[("cohort",) + tuple(lane.uid for lane in lanes)] = SimpleNamespace(
                tails=[view.tail for _lane, view in members],
                keys=[
                    (view.history, view.anchor, view.generated, count, tuple(f))
                    for (_lane, view), f in zip(members, forbidden)
                ],
                state=state,
                draft_caches=[cache for cache, _tail in snapshots],
            )
        if not early:
            self._prelaunched.update(records)
        return records

    def _prelaunch_tree_cohort_early(self, cohort, decisions, features):
        """Queue a cohort's next lattices from its verified, uncommitted round."""

        views = []
        for row, (lane, decision) in enumerate(zip(cohort, decisions)):
            rows, emitted = decision.commit_rows, decision.emitted
            if rows is None or not emitted or len(rows) != len(emitted):
                return None
            tail = self.mx.take(
                features[row:row + 1],
                self.mx.array(list(rows), dtype=self.mx.int32),
                axis=1,
            )
            decision.committed_tail = tail
            views.append(SimpleNamespace(
                anchor=emitted[-1], tail=tail,
                history=len(lane.history) + len(rows),
                generated=lane.generated + len(emitted),
            ))
        return self._prelaunch_tree_cohort(cohort, decisions, views)

    def _bind_early_cohort(self, cohort, decisions, records):
        """Keep early cohort lattices only at the exact boundaries predicted."""

        if not records:
            self._prelaunch_tree_cohort(cohort, decisions)
            return
        self._discard_prelaunched([lane.uid for lane in cohort])
        lanes = {lane.uid: lane for lane in cohort}
        budget = self._tree_node_budget()
        exact = not any(
            lane.ordinary or lane.cancelled or _speculation_parked(lane)
            for lane in cohort
        )
        for key, record in records.items():
            members = [lanes[uid] for uid in key[1:]]
            count = min(budget, min(l.maximum - l.generated - 1 for l in members))
            if not exact or any(
                tail is not lane.tail
                or expected != (
                    len(lane.history), lane.anchor, lane.generated, count,
                    tuple(self._tree_forbidden(lane)),
                )
                for lane, tail, expected in zip(members, record.tails, record.keys)
            ):
                exact = False
                break
        if exact:
            self._prelaunched.update(records)
            _bump(
                self.scheduler_stats,
                "external_tree_early_prelaunches",
                sum(len(key) - 1 for key in records),
            )
            return
        if records:
            _bump(self.scheduler_stats, "external_tree_pipeline_discards")
        self._prelaunch_tree_cohort(cohort, decisions)

    def _adopt_prelaunched_cohort(self, lanes, count, forbidden):
        """The queued batched lattice if this exact group returns unchanged."""

        for lane in lanes:
            # A singleton lattice queued before the lanes joined is stale.
            if self._prelaunched.pop(lane.uid, None) is not None:
                _bump(self.scheduler_stats, "external_tree_pipeline_discards")
        uids = tuple(lane.uid for lane in lanes)
        key = ("cohort",) + uids
        record = self._prelaunched.pop(key, None)
        if record is None:
            # The scheduler rotates its lane order every poll, so the same
            # group may return in another order: its rows are the same lanes'
            # independent lattice rows, read back in the group's new order.
            key = next(
                (
                    queued for queued in self._prelaunched
                    if isinstance(queued, tuple) and queued[:1] == ("cohort",)
                    and len(queued) == len(key) and set(queued[1:]) == set(uids)
                ),
                None,
            )
            record = None if key is None else self._prelaunched.pop(key)
        if record is None:
            return None
        rows = [key[1:].index(uid) for uid in uids]
        if (
            any(record.tails[row] is not lane.tail for row, lane in zip(rows, lanes))
            or [record.keys[row] for row in rows] != [
                (len(lane.history), lane.anchor, lane.generated, count, tuple(f))
                for lane, f in zip(lanes, forbidden)
            ]
        ):
            _bump(self.scheduler_stats, "external_tree_pipeline_discards")
            return None
        if self._tree_clock is not None:
            self._tree_clock("tree_draft_setup")
        for row, lane in zip(rows, lanes):
            lane.draft_cache = record.draft_caches[row]
        _bump(self.scheduler_stats, "external_tree_pipelined_drafts", len(lanes))
        if rows != sorted(rows):
            _bump(self.scheduler_stats, "external_tree_pipeline_reordered_adoptions")
        trees = self.draft.finish_tree(record.state, mark=self._tree_clock)
        return [trees[row] for row in rows]

    def _discard_prelaunched(self, uids):
        uids = set(uids)
        for key in list(self._prelaunched):
            members = key[1:] if isinstance(key, tuple) and key[:1] == ("cohort",) else (key,)
            if uids.intersection(members) and self._prelaunched.pop(key, None) is not None:
                _bump(self.scheduler_stats, "external_tree_pipeline_discards")

    def _walk_tree(self, block, sample_row, *, max_emitted=None):
        """Follow target samples through matching children; otherwise correct.

        ``sample_row(row, prefix_rows)`` draws the target token at ``row``,
        whose root-to-node path is ``prefix_rows``. Rows are visited in path
        order, so the lane RNG advances once per visited row only.
        """

        inputs = [None] + list(block.tokens)
        if max_emitted is not None and (type(max_emitted) is not int or max_emitted < 1):
            raise ValueError("tree response budget must be a positive integer")
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
            if token in self.stops or max_emitted is not None and len(emitted) >= max_emitted:
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
                lane.history,
                True,
                response_rows,
                history_suffix=[inputs[index] for index in prefix_rows],
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
        rows = logits[0, :width].astype(mx.float32)
        vocab = int(rows.shape[-1])
        inputs = [lane.anchor] + list(block.tokens)
        # Length-only processors need no token copies. Presence penalties
        # inspect only their declared window, rather than N full prompts.
        lengths = [len(lane.history) + len(path) for path in paths]
        for processor in lane.processors:
            presence = getattr(processor, "presence_window", None)
            if presence is not None:
                penalty, context_size, start = presence
                flat = []
                for row, path in enumerate(paths):
                    window = _history_window(
                        lane.history, [inputs[index] for index in path],
                        context_size, start,
                    )
                    for token in set(window):
                        flat.append(row * vocab + int(token))
                if flat:
                    mask = mx.zeros((width * vocab,), dtype=mx.bool_)
                    mask[mx.array(flat, dtype=mx.int32)] = True
                    rows = mx.where(mask.reshape(width, vocab), rows - penalty, rows)
                continue
            forbidden = [
                sorted({
                    int(token)
                    for token in processor.forbidden_token_ids_at_length(length)
                })
                for length in lengths
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

    def _verify_tree_batched(self, lane, block, state, *, logprobs_on_request=None):
        """Host walk over laws already landed by the round's device read.

        Produces the same tokens, laws, RNG draws and failures as
        ``_verify_tree``: each visited row's dense law is rebuilt bit for bit
        and passed through the same ``probability`` and ``RequestRNG.sample``
        calls; a greedy row draws the one uniform the one-hot sample consumes.
        """

        from .speculative_sampling import probability

        emit = lane.sampling.get("emit_logprobs", True)
        if logprobs_on_request is None:
            logprobs_on_request = self.tree_gates["logprobs_on_request"]
        keep_laws = emit or not logprobs_on_request
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

        budget = state.get("max_emitted")
        options = {"max_emitted": budget} if budget is not None and budget < len(block.tokens) + 1 else {}
        commit_rows, emitted = self._walk_tree(block, sample_row, **options)
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

    def _commit(
        self,
        cohort,
        decisions,
        features,
        *,
        blocks,
        transaction,
        committed_rows=None,
        committed_rngs=None,
        deferred_context=False,
        deferred_context_uids=(),
    ):
        """Commit accepted prefixes, then publish responses for every row.

        ``deferred_context`` means this round's proposal phase did not append
        the pending ``lane.tail`` to the draft plane; the committed features
        then extend that tail instead of replacing it, keeping
        ``draft offset + tail length == committed boundary``.
        """
        proposal_counts = [
            0 if block is None else int(block.lengths[0]) for block in blocks
        ]
        consumed = [
            (
                int(decision.commit_count)
                if decision.commit_count is not None
                else min(decision.accepted + 1, len(decision.emitted))
            )
            for decision in decisions
        ]
        clock = self._phase_time() if self.round_timing else None
        paths = [
            (
                list(decision.commit_rows)
                if decision.commit_rows is not None
                else list(range(used))
            )
            for decision, used in zip(decisions, consumed)
        ]
        bridge_provenance = [
            self._ast_bridge_commit_provenance(block) if count else None
            for block, count in zip(blocks, proposal_counts, strict=True)
        ]
        if committed_rows is not None:
            if transaction is not None or len(committed_rows) != len(cohort):
                raise RuntimeError("invalid private-cache promotion")
            if committed_rngs is None or len(committed_rngs) != len(cohort):
                raise RuntimeError("private-cache promotion requires private RNG state")
            if any(decision.commit_rows is not None for decision in decisions):
                raise RuntimeError("private-cache promotion cannot commit tree paths")
            for lane, promoted, used in zip(
                cohort, committed_rows, consumed, strict=True
            ):
                if len(promoted) != len(lane.cache):
                    raise RuntimeError("private-cache promotion changed cache geometry")
                for current, candidate in zip(
                    lane.cache, promoted, strict=True
                ):
                    if type(current) is not type(candidate):
                        raise RuntimeError(
                            "private-cache promotion changed cache type"
                        )
                    old_offset = getattr(current, "offset", None)
                    new_offset = getattr(candidate, "offset", None)
                    if (
                        isinstance(old_offset, int)
                        and isinstance(new_offset, int)
                        and new_offset - old_offset != used
                    ):
                        raise RuntimeError(
                            "private-cache promotion advanced the wrong boundary"
                        )
            rows = list(committed_rows)
        elif any(decision.commit_rows is not None for decision in decisions):
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
            if decision.committed_inputs is None:
                inputs = [lane.anchor] + drafts
                committed_inputs = [inputs[index] for index in paths[row]]
            else:
                committed_inputs = list(decision.committed_inputs)
                if len(committed_inputs) != consumed[row]:
                    raise RuntimeError(
                        "continuation committed-input count differs from transaction"
                    )
            lane.cache = rows[row]
            if committed_rngs is not None:
                lane.rng = committed_rngs[row]
            if decision.commit_rows is None:
                tail = features[row:row+1,:consumed[row]]
            elif decision.committed_tail is not None:
                tail = decision.committed_tail
            else:
                tail = self.mx.take(
                    features[row:row+1],
                    self.mx.array(paths[row], dtype=self.mx.int32),
                    axis=1,
                )
            if (deferred_context or lane.uid in deferred_context_uids) and lane.tail.shape[1]:
                tail = self.mx.concatenate([lane.tail, tail], axis=1)
            lane.tail = tail
            lane.history.extend(committed_inputs)
            lane.anchor = emitted[-1]
            round_accepted = _delivered_accepts(decision)
            self.scheduler_stats["accepted_proposals"] += round_accepted
            self.scheduler_stats["proposed_tokens"] += count
            lane.external_rounds += int(count > 0)
            lane.proposed += count
            lane.accepted += round_accepted
            source = getattr(blocks[row], "proposal_source", None)
            if hasattr(self.draft, "last_proposal_sources"):
                lane.proposal_composition_current_source = source if count else "ordinary"
                lane.proposal_composition_current_score_key = (
                    blocks[row].proposal_score_key if count else None
                )
                lane.proposal_composition_current_score = (
                    blocks[row].proposal_score if count else None
                )
                lane.proposal_composition_current_trust_key = (
                    blocks[row].proposal_trust_key if count else None
                )
                lane.proposal_composition_current_trusted = bool(
                    count and getattr(blocks[row], "proposal_trusted", False)
                )
                lane.proposal_composition_current_audit = bool(
                    count and getattr(blocks[row], "proposal_audit", False)
                )
                lane.proposal_composition_current_periodic_audit = bool(
                    count
                    and getattr(blocks[row], "proposal_periodic_audit", False)
                )
                if source is not None and count:
                    counters = getattr(lane, "proposal_composition_counts", {})
                    record = counters.setdefault(source, {"verified_rounds": 0, "proposed_tokens": 0, "accepted_tokens": 0})
                    record["verified_rounds"] += 1
                    record["proposed_tokens"] += count
                    record["accepted_tokens"] += round_accepted
                    lane.proposal_composition_counts = counters
                    _bump(self.scheduler_stats, "external_composed_" + source + "_rounds")
                    if getattr(blocks[row], "proposal_trusted", False):
                        lane.trusted_pld_rounds = getattr(lane, "trusted_pld_rounds", 0) + 1
                        lane.trusted_pld_tokens = getattr(lane, "trusted_pld_tokens", 0) + count
                        _bump(self.scheduler_stats, "external_composed_trusted_pld_rounds")
                        _bump(
                            self.scheduler_stats,
                            "external_composed_trusted_pld_tokens",
                            count,
                        )
                    elif getattr(blocks[row], "proposal_audit", False):
                        _bump(
                            self.scheduler_stats,
                            "external_composed_trusted_pld_audit_rounds",
                        )
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
                pair = f"{count + 1}:{round_accepted}"
                lane.verify_accept_span_hist[pair] = (
                    lane.verify_accept_span_hist.get(pair, 0) + 1
                )
            committed_relaxed = count_delivered_relaxed_accepts(
                decision.relaxed_positions,
                decision.relaxed,
                delivered=len(emitted),
                accepted=decision.accepted,
            )
            lane.relaxed_accepts += committed_relaxed
            _bump(
                self.scheduler_stats,
                "fly_relaxed_accepts",
                committed_relaxed,
            )
            lane.target_max_width = max(lane.target_max_width, len(cohort))
            committed_bridge = bridge_provenance[row]
            if committed_bridge is not None and count:
                selection = blocks[row].continuation_selection
                selected_index = committed_bridge["selected_rank"]
                admitted_indices = committed_bridge["admitted_ranks"]
                self._record_continuation_round(
                    lane,
                    blocks[row],
                    selection,
                    admitted_width=len(admitted_indices),
                    admitted_indices=admitted_indices,
                    selected_index=selected_index,
                    proposed_depth=count,
                    accepted=round_accepted,
                )
                lane.continuation_rounds = (
                    getattr(lane, "continuation_rounds", 0) + 1
                )
                lane.continuation_verification_algorithm = (
                    "progressive_exact_target_draw_then_matching_continuation_prefix"
                )
                lane.continuation_semantic_selected = len(selection.paths)
                lane.continuation_semantic_admitted = len(admitted_indices)
                lane.continuation_physical_width = len(admitted_indices)
                lane.continuation_physical_span = committed_bridge[
                    "physical_span"
                ]
                lane.continuation_physical_to_semantic = tuple(admitted_indices)
                semantic_to_physical = [None] * len(selection.paths)
                for physical, semantic in enumerate(admitted_indices):
                    semantic_to_physical[semantic] = physical
                lane.continuation_semantic_to_physical = tuple(
                    semantic_to_physical
                )
                lane.continuation_selected_path = selected_index
                lane.continuation_selected_physical_path = committed_bridge[
                    "physical_branch"
                ]
                chosen = selection.paths[selected_index]
                lane.continuation_selected_sources = tuple(
                    (contributor.source_id, contributor.source_revision)
                    for contributor in chosen.contributors
                )
                lane.continuation_last_ast_provenance = copy.deepcopy(
                    committed_bridge["selected_provenance"]
                )
                if committed_bridge["ast_engaged"]:
                    lane.continuation_ast_committed_rounds = (
                        getattr(lane, "continuation_ast_committed_rounds", 0) + 1
                    )
                else:
                    lane.continuation_ast_primary_fallback_committed_rounds = (
                        getattr(
                            lane,
                            "continuation_ast_primary_fallback_committed_rounds",
                            0,
                        )
                        + 1
                    )
            if decision.continuation_outcome is not None:
                outcome = decision.continuation_outcome
                lane.continuation_rounds = getattr(lane, "continuation_rounds", 0) + 1
                lane.continuation_physical_width = outcome.physical_width
                lane.continuation_physical_span = outcome.physical_span
                lane.continuation_strategy = outcome.algorithm
                lane.continuation_strategy_launches = (
                    getattr(lane, "continuation_strategy_launches", 0)
                    + outcome.launches
                )
                lane.continuation_pruned_siblings = (
                    getattr(lane, "continuation_pruned_siblings", 0)
                    + outcome.pruned_siblings
                )
                lane.continuation_shared_prefix_reused_tokens = (
                    getattr(lane, "continuation_shared_prefix_reused_tokens", 0)
                    + outcome.shared_prefix_reused_tokens
                )
                semantic_index = (outcome.selected_index if decision.continuation_semantic_index is None
                                  else decision.continuation_semantic_index)
                lane.continuation_selected_path = semantic_index
                lane.continuation_selected_physical_path = outcome.selected_index
                chosen = blocks[row].continuation_selection.paths[semantic_index]
                lane.continuation_selected_sources = tuple((path.source_id, path.source_revision)
                    for path in chosen.contributors)
                lane.continuation_target_rows = (
                    getattr(lane, "continuation_target_rows", 0)
                    + outcome.executed_target_rows
                )
                lane.target_max_width = max(lane.target_max_width, outcome.physical_width)
            if self._feedback_outbox is not None:
                payload = getattr(blocks[row], "feedback_payload", None)
                selection = getattr(blocks[row], "continuation_selection", None)
                score_key = getattr(blocks[row], "proposal_score_key", None)
                if payload is not None or selection is not None or score_key is not None:
                    self._feedback_outbox.append((lane, blocks[row], decision, lane.external_rounds - 1))
            if count:
                lane.draft_max_width = max(lane.draft_max_width, getattr(blocks[row], "width", 1))
            round_receipt = {
                "kind": self.receipt_kind,
                "execution": "external_draft_verify" if lane.external_rounds else "ordinary_target",
                "current_execution": "ordinary_target" if count == 0 else "trusted_pld_teacher_force" if getattr(blocks[row], "proposal_trusted", False) else "external_draft_verify",
                "ordinary_fallback": lane.ordinary,
                "external_rounds": lane.external_rounds,
                "accepted": lane.accepted, "proposed": lane.proposed,
                "round_accepted": round_accepted, "round_proposed": count,
                "target_width": lane.target_max_width, "draft_width": lane.draft_max_width,
                "qualification_authority": "serving_route",
                **self._target_execution_receipt(lane),
                **self._verification_receipt(lane),
                **self._adaptive_receipt(lane),
                **self._draft_settings_receipt(),
                **self._proposal_composition_receipt(lane),
                **self._continuation_receipt(lane),
            }
            for j,token in enumerate(emitted):
                lane.generated += 1
                finish = "stop" if token in self.stops else "length" if lane.generated >= lane.maximum else None
                final = j == len(emitted)-1
                # The published row when one was kept, else the law itself.
                logp = None
                if lane.sampling.get("emit_logprobs", True):
                    if (
                        decision.response_logprobs
                        and j < len(decision.response_logprobs)
                        and decision.response_logprobs[j] is not None
                    ):
                        logp = decision.response_logprobs[j]
                    elif (
                        decision.target_laws is not None
                        and decision.target_laws[j] is not None
                    ):
                        logp = self.mx.log(self.mx.array(decision.target_laws[j].astype(np.float32)))
                lane.ready.append(SimpleNamespace(uid=lane.uid, token=token, logprobs=logp, finish_reason=finish, execution_width=(decision.continuation_outcome.physical_width if decision.continuation_outcome is not None else len(cohort)), all_tokens=list(lane.history) if final else None, prompt_cache=self._freeze_cache(lane.cache) if finish else None, cache_sidecar=self._sidecar(lane) if finish else None, mtp_state=None, mtp_receipt=None, speculative_receipt=dict(round_receipt)))
        if clock is not None:
            self._mark("emit", clock)
            # The responses are published only after this round returns, so
            # refresh their diagnostic snapshot once the emit phase itself is
            # accounted.  This does not change tokens, cache, or scheduling.
            for lane in cohort:
                timing = self._round_timing_receipt(lane)
                for response in lane.ready:
                    response.speculative_receipt.update(timing)
        self.scheduler_stats[
            "external_rounds" if any(proposal_counts) else "ordinary_rounds"
        ] += 1
        self.scheduler_stats["target_max_width"] = max(self.scheduler_stats["target_max_width"],len(cohort))

    def _ast_bridge_commit_provenance(self, block):
        """Validate branch-bound continuation identity before target commit."""

        if not getattr(self, "continuation_pool_ast_bridge", False) or block is None:
            return None
        selection = getattr(block, "continuation_selection", None)
        if selection is None or not selection.paths:
            raise RuntimeError("AST bridge proposal lacks a continuation selection")
        selected = getattr(block, "continuation_ast_provenance", None)
        admitted = getattr(block, "continuation_ast_admitted_provenance", None)
        if selected is None:
            selected_rank = 0
            admitted_ranks = (0,)
            physical_branch = 0
            ast_engaged = False
            selected_payload = None
            admitted_depths = (len(block.tokens),)
        else:
            if not isinstance(selected, dict) or not isinstance(
                admitted, (list, tuple)
            ):
                raise RuntimeError("AST bridge branch provenance is malformed")
            selected_rank = selected.get("rank")
            physical_branch = getattr(block, "continuation_ast_branch_index", None)
            admitted_ranks = tuple(item.get("rank") for item in admitted)
            admitted_depths = getattr(
                block, "continuation_ast_admitted_depths", None
            )
            if (
                type(selected_rank) is not int
                or type(physical_branch) is not int
                or any(type(rank) is not int for rank in admitted_ranks)
                or not isinstance(admitted_depths, (list, tuple))
                or len(admitted_depths) != len(admitted_ranks)
                or any(type(depth) is not int or depth < 1 for depth in admitted_depths)
                or len(set(admitted_ranks)) != len(admitted_ranks)
                or not 0 <= physical_branch < len(admitted_ranks)
                or admitted_ranks[physical_branch] != selected_rank
                or admitted_depths[physical_branch] != len(block.tokens)
            ):
                raise RuntimeError("AST bridge branch/rank mapping is invalid")
            admitted_depths = tuple(admitted_depths)
            ast_engaged = True
            selected_payload = selected
        if (
            not admitted_ranks
            or any(not 0 <= rank < len(selection.paths) for rank in admitted_ranks)
            or selected_rank not in admitted_ranks
        ):
            raise RuntimeError("AST bridge admitted ranks are invalid")
        chosen = selection.paths[selected_rank]
        if tuple(block.tokens) != tuple(chosen.tokens[: len(block.tokens)]):
            raise RuntimeError("AST bridge winning branch differs from selected path")
        if selected_payload is not None:
            contributors = tuple(
                (
                    contributor.source_id,
                    contributor.source_revision,
                    contributor.candidate_id,
                )
                for contributor in chosen.contributors
            )
            if (
                selected_payload.get("path_id") != chosen.path_id
                or tuple(selected_payload.get("contributors", ())) != contributors
            ):
                raise RuntimeError("AST bridge winning provenance differs from selection")
        return {
            "selected_rank": selected_rank,
            "admitted_ranks": admitted_ranks,
            "physical_branch": physical_branch,
            "physical_span": sum(admitted_depths),
            "ast_engaged": ast_engaged,
            "selected_provenance": selected_payload,
        }

    def _continuation_receipt(self, lane):
        if self.continuation_policy is None:
            return {}
        # A transient depth-zero terminal token has no feedback outbox entry.
        # Retain evidence from prior committed pool rounds; this locked query
        # changes neither labels nor tickets. Speculative rounds refresh their
        # ready responses after the entire cohort commits, as before.
        try:
            ranking = {"ranking": copy.deepcopy(self.draft.proposal_pool.receipt(lane.session_scope_hash))}
        except Exception as error:  # noqa: BLE001 - optional diagnostics cannot invalidate committed inference
            ranking = {"ranking_error": str(error)}
        configured_algorithm = self.continuation_policy.verification_algorithm
        verification_algorithm = getattr(
            lane, "continuation_verification_algorithm", "not_executed"
        )
        strategy_algorithm = getattr(lane, "continuation_strategy", "not_executed")
        strategy_value = strategy_algorithm
        if self.continuation_strategy.routed_experts_per_token is not None:
            attempts = getattr(lane, "continuation_strategy_launches", 0)
            target_rows = getattr(lane, "continuation_target_rows", 0)
            strategy_value = {
                "algorithm": strategy_algorithm,
                "mode": "longest_first_exact_prefix",
                "cache_layout": self.continuation_strategy.cache_layout,
                "proposal_state": self.continuation_strategy.proposal_state,
                "target_state": self.continuation_strategy.target_state,
                "implemented": True,
                "qualified": self.continuation_strategy.qualified,
                "selected": (
                    self.continuation_strategy.algorithm
                    == "longest_first_exact_prefix_v1"
                ),
                "observed_used": attempts > 0,
                "attempts": attempts,
                "sibling_proposals_pruned": getattr(
                    lane, "continuation_pruned_siblings", 0
                ),
                "shared_prefix_tokens_reused": getattr(
                    lane, "continuation_shared_prefix_reused_tokens", 0
                ),
                "target_rows": target_rows,
                "routed_expert_rows": (
                    target_rows
                    * self.continuation_strategy.routed_experts_per_token
                    * self.continuation_strategy.routed_moe_layers
                ),
                "routed_experts_per_token": (
                    self.continuation_strategy.routed_experts_per_token
                ),
                "routed_moe_layers": self.continuation_strategy.routed_moe_layers,
                "proposal_state_authoritative": False,
                "apcv2_publication": False,
            }
        bridge_rounds = getattr(lane, "continuation_ast_committed_rounds", 0)
        verification = (
            "progressive_exact_target_draw_then_matching_continuation_prefix"
            if getattr(self, "continuation_pool_ast_bridge", False)
            else "processed_target_draw_then_matching_continuation_prefix"
        )
        return {"verification": verification,
                "continuation_pool": {"implemented": True, "selected": True, "qualified": False,
                    **ranking,
                    "verification_role": (
                        "proposal_only_ast_bridge"
                        if getattr(self, "continuation_pool_ast_bridge", False)
                        else "complete_path_verifier"
                    ),
                    "verification_algorithm": verification_algorithm,
                    "configured_verification_algorithm": configured_algorithm,
                    "observed_used": (
                        getattr(lane, "continuation_rounds", 0) > 0
                        if getattr(self, "continuation_pool_ast_bridge", False)
                        else getattr(lane, "continuation_rounds", 0) > 0
                    ),
                    "ast_observed_used": (
                        bridge_rounds > 0
                        if getattr(self, "continuation_pool_ast_bridge", False)
                        else False
                    ),
                    "unit": "complete_continuation_sequences", "limit": self.continuation_policy.limit,
                    "committed_rounds": getattr(lane, "continuation_rounds", 0),
                    "ast_committed_rounds": bridge_rounds,
                    "ast_primary_fallback_committed_rounds": getattr(
                        lane,
                        "continuation_ast_primary_fallback_committed_rounds",
                        0,
                    ),
                    "physical_width": getattr(lane, "continuation_physical_width", 0),
                    "physical_span": getattr(lane, "continuation_physical_span", 0),
                    "strategy": strategy_value,
                    "strategy_algorithm": strategy_algorithm,
                    "strategy_qualified": self.continuation_strategy.qualified,
                    "strategy_selected": (
                        self.continuation_strategy.algorithm
                        == "longest_first_exact_prefix_v1"
                    ),
                    "strategy_observed_used": getattr(
                        lane, "continuation_strategy_launches", 0
                    ) > 0,
                    "strategy_launches": getattr(lane, "continuation_strategy_launches", 0),
                    "shared_prefix_reused_tokens": getattr(
                        lane, "continuation_shared_prefix_reused_tokens", 0
                    ),
                    "ranked_complete_sequences": getattr(lane, "continuation_semantic_selected", 0),
                    "admitted_complete_sequences": getattr(lane, "continuation_semantic_admitted", 0),
                    "prefix_dedup_enabled": getattr(lane, "continuation_prefix_dedup_enabled", False),
                    "prefix_dedup_observed_used": getattr(lane, "continuation_deduplicated_sequences", 0) > 0,
                    "deduplicated_sequences": getattr(lane, "continuation_deduplicated_sequences", 0),
                    "cascade_attempts": getattr(lane, "continuation_cascade_attempts", 0),
                    "pruned_siblings": getattr(lane, "continuation_pruned_siblings", 0),
                    "shared_prefix_tokens_reused": getattr(
                        lane, "continuation_shared_prefix_tokens_reused", 0
                    ),
                    "shared_prefix_reuse_observed_used": getattr(
                        lane, "continuation_shared_prefix_tokens_reused", 0
                    ) > 0,
                    "physical_to_semantic": list(getattr(lane, "continuation_physical_to_semantic", ())),
                    "semantic_to_physical": list(getattr(lane, "continuation_semantic_to_physical", ())),
                    "target_rows": getattr(lane, "continuation_target_rows", 0),
                    "selected_path": getattr(lane, "continuation_selected_path", None),
                    "selected_physical_path": getattr(lane, "continuation_selected_physical_path", None),
                    "selected_sources": list(getattr(lane, "continuation_selected_sources", ())),
                    "selected_ast_provenance": copy.deepcopy(
                        getattr(lane, "continuation_last_ast_provenance", None)
                    ),
                    # Scores and slate predictions come from the shared
                    # ranking critic, which every tenant's committed labels
                    # move; every lane's receipt is scoped (an unkeyed request
                    # gets a private scope), so they stay internal, as
                    # ProposalPool.receipt drops the shared counts.
                    "provider_stats": {
                        source: {
                            key: copy.deepcopy(value)
                            for key, value in row.items()
                            if key not in ("score_sum", "score_observations")
                        }
                        for source, row in getattr(lane, "continuation_provider_stats", {}).items()
                    },
                    "source_tree_geometry": {
                        "last": copy.deepcopy(
                            getattr(lane, "continuation_source_geometry_last", {})
                        ),
                        "totals": copy.deepcopy(
                            getattr(lane, "continuation_source_geometry_totals", {})
                        ),
                    },
                    "winning_combinations": copy.deepcopy(
                        getattr(lane, "continuation_winning_combinations", {})
                    ),
                    "last_admission": copy.deepcopy(
                        getattr(lane, "continuation_last_admission", None)
                    ),
                    "admission_totals": copy.deepcopy(
                        getattr(lane, "continuation_admission_totals", {})
                    ),
                    "last_candidate_slate": tuple(
                        {
                            key: copy.deepcopy(value)
                            for key, value in item.items()
                            if key not in ("conditional_acceptance", "expected_accepted_tokens", "score")
                        }
                        for item in getattr(lane, "continuation_last_slate", ())
                    ),
                    "session_scope_hash": lane.session_scope_hash,
                    "adaptive_cost_binding": "one_request_physical_path_width_and_depth_plus_bonus",
                    "coverage_calibration": "lagged_top_width_reached_prefix_beta_frequency"}}

    def _record_continuation_round(
        self,
        lane,
        block,
        selection,
        *,
        admitted_width,
        selected_index,
        proposed_depth,
        accepted,
        admitted_indices=None,
    ):
        """Accumulate bounded host stats after target selection, before commit."""
        admission = copy.deepcopy(block.continuation_admission or {})
        slate = tuple(copy.deepcopy(block.continuation_slate or ()))
        admitted_indices = (
            set(range(int(admitted_width)))
            if admitted_indices is None
            else {int(index) for index in admitted_indices}
        )
        if (
            not admitted_indices
            or any(not 0 <= index < len(selection.paths) for index in admitted_indices)
            or int(selected_index) not in admitted_indices
        ):
            raise ValueError("continuation admitted provenance is invalid")
        lane.continuation_last_admission = admission
        admission_totals = getattr(lane, "continuation_admission_totals", {})
        for budget_field in (
            "lane_slots_used",
            "lane_slots_available",
            "tile_rows_used",
            "tile_rows_available",
        ):
            admission_totals[budget_field] = admission_totals.get(
                budget_field, 0
            ) + max(
                0, int(admission.get(budget_field, 0))
            )
        admission_totals["rounds"] = admission_totals.get("rounds", 0) + 1
        lane.continuation_admission_totals = admission_totals
        lane.continuation_last_slate = tuple(
            {
                **item,
                "admitted": int(item.get("rank", -1)) in admitted_indices,
                "target_selected": int(item.get("rank", -1)) == int(selected_index),
            }
            for item in slate
        )
        stats = getattr(lane, "continuation_provider_stats", {})
        admitted = set(admission.get("admitted") or ())
        for source_id, source in self.draft.source_records.items():
            row = stats.setdefault(
                source_id,
                {
                    "mechanism": source.mechanism,
                    "enabled_rounds": 0,
                    "admitted_rounds": 0,
                    "selected_rounds": 0,
                    "contributed_rounds": 0,
                    "candidate_paths": 0,
                    "proposed_tokens": 0,
                    "accepted_tokens": 0,
                    "zero_rounds": 0,
                    "partial_rounds": 0,
                    "full_rounds": 0,
                    "proposal_depths": {},
                    "accepted_depths": {},
                    "first_rejected_positions": {},
                    "selected_ranks": {},
                    "score_sum": 0.0,
                    "score_observations": 0,
                    "parked_rounds": 0,
                    "fallback_rounds": 0,
                },
            )
            row["enabled_rounds"] += 1
            row["admitted_rounds"] += int(source_id in admitted)
            row["parked_rounds"] += int(
                any(
                    item.get("source_id") == source_id
                    for item in admission.get("parked", ())
                )
            )
            # A runtime fallback is charged to the fallback provider alone; the
            # admitted model drafter that came back empty did not serve as one.
            runtime_fallback = admission.get("runtime_fallback")
            row["fallback_rounds"] += int(
                runtime_fallback == source_id
                if runtime_fallback is not None
                else bool(admission.get("fallback_only") and source_id in admitted)
            )
        for ranked in selection.paths:
            for source_id in {item.source_id for item in ranked.contributors}:
                stats[source_id]["candidate_paths"] += 1
        chosen = selection.paths[selected_index]
        proposed = max(1, min(int(proposed_depth), len(chosen.tokens)))
        accepted = max(0, min(int(accepted), proposed))
        mechanisms = []
        contributor_ids = {
            contributor.source_id for contributor in chosen.contributors
        }
        for source_id in contributor_ids:
            row = stats[source_id]
            mechanisms.append(row["mechanism"])
            row["selected_rounds"] += int(
                source_id == chosen.representative.source_id
            )
            row["contributed_rounds"] += 1
            row["proposed_tokens"] += proposed
            row["accepted_tokens"] += accepted
            row["zero_rounds"] += int(accepted == 0)
            row["partial_rounds"] += int(0 < accepted < proposed)
            row["full_rounds"] += int(accepted == proposed)
            for key, value in (
                ("proposal_depths", proposed),
                ("accepted_depths", accepted),
                ("selected_ranks", selected_index),
            ):
                bucket = str(value)
                row[key][bucket] = row[key].get(bucket, 0) + 1
            if accepted < proposed:
                bucket = str(accepted)
                row["first_rejected_positions"][bucket] = (
                    row["first_rejected_positions"].get(bucket, 0) + 1
                )
            row["score_sum"] += float(chosen.score)
            row["score_observations"] += 1
        combinations = getattr(lane, "continuation_winning_combinations", {})
        key = "+".join(sorted(mechanisms)) or "none"
        combinations[key] = combinations.get(key, 0) + 1
        lane.continuation_winning_combinations = combinations
        lane.continuation_provider_stats = stats

    def _phase_time(self):
        """Return a phase boundary, optionally after a diagnostic device fence."""
        if self.round_timing_sync:
            self.mx.synchronize()
            _bump(self.scheduler_stats, "external_phase_synchronizations")
        return time.perf_counter()

    def _mark(self, phase, since):
        """Accumulate phase time; returns the next boundary."""
        now = self._phase_time()
        self.round_times[phase] += now - since
        # Receipt copy: integer nanoseconds per phase, present only when
        # MLX2_EXTERNAL_ROUND_TIMING is set.
        _bump(
            self.scheduler_stats,
            f"external_phase_{phase}_ns",
            int((now - since) * 1e9),
        )
        return now

    def _round_timing_receipt(self, lane):
        """Publish opt-in phase deltas without claiming per-request isolation.

        ``/v1/status`` is a periodic worker snapshot.  A short diagnostic
        request can finish before that snapshot refreshes, losing the only
        externally visible phase breakdown.  The scheduler counters already
        exist behind ``MLX2_EXTERNAL_ROUND_TIMING``; bind their admission-to-
        receipt deltas into the request receipt as executor-scoped evidence.

        Concurrent requests may share a physical phase, so these deltas are
        deliberately labelled executor-scoped and unsuitable as controlled
        performance evidence unless the campaign owns a single request lane.
        """
        if not self.round_timing:
            return {}
        baseline = getattr(lane, "external_timing_start", {})
        phases = {
            key.removeprefix("external_phase_").removesuffix("_ns"): (
                int(value) - int(baseline.get(key, 0))
            )
            for key, value in self.scheduler_stats.items()
            if key.startswith("external_phase_")
            and key.endswith("_ns")
            and type(value) is int
            and int(value) >= int(baseline.get(key, 0))
        }
        return {
            "round_timing": {
                "enabled": True,
                "device_synchronized": self.round_timing_sync,
                "device_synchronizations": int(self.scheduler_stats.get(
                    "external_phase_synchronizations", 0
                )) - int(baseline.get("external_phase_synchronizations", 0)),
                "scope": "executor_shared_since_request_admission",
                "request_exclusive": False,
                "performance_claim": False,
                "phase_nanoseconds": dict(sorted(phases.items())),
                "phase_total_nanoseconds": sum(phases.values()),
            }
        }

    def _proposal_composition_receipt(self, lane, *, current_source=None):
        if not hasattr(self.draft, "last_proposal_sources"):
            return {}
        counters = getattr(lane, "proposal_composition_counts", {})
        scoring = getattr(self.draft, "proposal_composition_receipt", {})
        preselection_ranking = {
            "order": "score_then_span_then_source",
            "authority": "lagged_committed_verification_only",
            "score_counts": {
                key: {
                    "weighted_accepted_tokens": value[0],
                    "weighted_proposed_tokens": value[1],
                }
                for key, value in sorted(lane.proposal_score_counts.items())
            },
        }
        return {"proposal_composition": {
            "selected": True, "qualified": False,
            "observed_used": any(counters.get(source, {}).get("verified_rounds", 0)
                                 for source in ("prompt_lookup", "native_mtp")),
            "current_source": current_source or getattr(lane, "proposal_composition_current_source", "not_executed"),
            "current_score_key": getattr(
                lane, "proposal_composition_current_score_key", None
            ),
            "current_score": getattr(
                lane, "proposal_composition_current_score", None
            ),
            "current_trust_key": getattr(
                lane, "proposal_composition_current_trust_key", None
            ),
            "current_trusted": getattr(
                lane, "proposal_composition_current_trusted", False
            ),
            "current_audit": getattr(
                lane, "proposal_composition_current_audit", False
            ),
            "current_periodic_audit": getattr(
                lane, "proposal_composition_current_periodic_audit", False
            ),
            # Compatibility alias plus an explicit phase name. Both are frozen
            # before this round's committed feedback is published.
            "ranking": copy.deepcopy(preselection_ranking),
            "preselection_ranking": preselection_ranking,
            "trusted_pld": {
                "scope": "request_lane",
                "sources": copy.deepcopy(lane.proposal_trust_states),
            },
            "adapter_diagnostics": copy.deepcopy(
                scoring.get("adapter_diagnostics", {})
            ),
            # A depth-zero terminal response has no new outbox item; carry the
            # latest committed learning snapshot so its final receipt is not
            # stale or empty.
            "postcommit_learning": copy.deepcopy(
                getattr(lane, "proposal_postcommit_learning", None)
            ),
            "committed_verification": copy.deepcopy(counters),
        }}

    @staticmethod
    def _proposal_postcommit_snapshot(lane, block, *, proposed, accepted):
        return {
            "scope": "request_lane",
            "observed_proposed_tokens": proposed,
            "observed_accepted_tokens": accepted,
            "score_key": block.proposal_score_key,
            "trust_key": block.proposal_trust_key,
            "audited": bool(block.proposal_audit),
            "periodic_audit": bool(block.proposal_periodic_audit),
            "trusted": bool(block.proposal_trusted),
            "ranking": {
                "authority": "postcommit_observed_prefix",
                "score_counts": {
                    key: {
                        "weighted_accepted_tokens": value[0],
                        "weighted_proposed_tokens": value[1],
                    }
                    for key, value in sorted(lane.proposal_score_counts.items())
                },
            },
            "trusted_pld": {
                "scope": "request_lane",
                "sources": copy.deepcopy(lane.proposal_trust_states),
            },
        }

    def _attach_postcommit_proposal_receipts(
        self, lane, block, decision, *, proposed, accepted
    ):
        if not decision.emitted:
            return
        snapshot = self._proposal_postcommit_snapshot(
            lane, block, proposed=proposed, accepted=accepted
        )
        lane.proposal_postcommit_learning = copy.deepcopy(snapshot)
        for response in list(lane.ready)[-len(decision.emitted):]:
            composition = response.speculative_receipt.get("proposal_composition")
            if composition is not None:
                composition["postcommit_learning"] = copy.deepcopy(snapshot)

    def _draft_settings_receipt(self):
        settings = getattr(getattr(self, "draft", None), "receipt_settings", None)
        if settings is None:
            settings = {}
        if not isinstance(settings, dict):
            raise ValueError("drafter receipt_settings must be a dictionary")  # noqa: TRY004
        # Freeze settings with the response: later drafter tuning cannot change
        # the configuration attributed to an already committed token.
        batching = getattr(self, "speculative_batching", {})
        cached = getattr(self, "_draft_receipt_settings_snapshot", None)
        if (cached is None or cached[0] is not self.draft
                or cached[1] != settings or cached[2] != batching):
            json.dumps(settings, allow_nan=False)
            json.dumps(batching, allow_nan=False)
            cached = self._draft_receipt_settings_snapshot = (
                self.draft, freeze_receipt(settings), freeze_receipt(batching)
            )
        return {
            "draft_settings": {
                **cached[1],
                "minimum_proposal_length": self.minimum_draft_proposals,
                "proposal_floor_raises": self.scheduler_stats[
                    "external_proposal_floor_raises"
                ],
                "terminal_exhaustion_may_shorten": True,
                "batching": cached[2],
                "multilane_draft_cap": self.multilane_draft_cap,
                "multilane_draft_cap_rounds": self.scheduler_stats.get(
                    "external_multilane_draft_cap_rounds", 0
                ),
                "multilane_draft_cap_lanes": self.scheduler_stats.get(
                    "external_multilane_draft_cap_lanes", 0
                ),
            }
        }

    def _adaptive_receipt(self, lane=None):
        if self.adaptive_policy is None:
            return {}
        return {"adaptive_verification": {
            "implemented": True,
            "qualified": False,
            "selected": True,
            "observed_used": (
                getattr(lane, "adaptive_trimmed_rounds", 0) > 0 if lane is not None
                else self.scheduler_stats["external_adaptive_trimmed_rounds"] > 0
            ),
            "policy": ("continuation_physical_shape_cost_model" if self.continuation_policy is not None
                       else "per_request_grouped_cost_model" if self.adaptive_policy.mode == "per_request"
                       else "lagged_cohort_cost_model"),
            "confidence": ("lagged_top_width_prefix_coverage" if self.continuation_policy is not None
                           else "deterministic_refined_logit_proxy_or_lagged_q"
                           if self.adaptive_policy.mode == "per_request" else "lagged_q"),
            "calibration": ("censored_conditional_beta_frequency" if self.continuation_policy is not None
                            else "shared_online_censored_logistic"),
            "current_policy": getattr(lane, "adaptive_round_policy", "not_executed"),
            "current_feature_source": getattr(lane, "adaptive_feature_source", "not_executed"),
            "per_request_rounds": self.scheduler_stats["external_adaptive_per_request_rounds"],
            "cost_model_cohort_sizes": (sorted({self.adaptive_policy.cohort_size,
                *(size for size, _ in self.adaptive_policy.verification_costs_by_cohort)})
                if self.adaptive_policy.verification_costs else []),
            "verification_groups": self.scheduler_stats["external_adaptive_verification_groups"],
            "cost_model_cohort_size": self.adaptive_policy.cohort_size if self.adaptive_policy.verification_costs else None,
            "cost_model_source": "caller_supplied",
            "continuation_cost_path_widths": [width for width, _ in self.adaptive_policy.continuation_costs],
            "round_proposal_depth": self.scheduler_stats["external_adaptive_round_depth"],
            "round_verify_width": self.scheduler_stats["external_adaptive_verify_width"],
            "target_rows": self.scheduler_stats["external_adaptive_target_rows"],
            "trimmed_target_rows": self.scheduler_stats["external_adaptive_trimmed_target_rows"],
        }}

    def _adaptive_observe(self, blocks, decisions, cohort=None):
        from .acceptance_estimator import proposal_feature
        for row, (block, decision) in enumerate(zip(blocks, decisions)):
            if block is None:
                continue
            tokens, laws = _block_row(block, self._target_args.vocab_size)
            confidence = getattr(block, "confidence_features", None)
            if confidence is not None:
                features = list(confidence)
            elif isinstance(laws, CompactDraftRow):
                features = [proposal_feature(row) for row in laws.candidate_probs]
            else:
                features = [proposal_feature(row) for row in laws]
            if cohort is not None:
                lane = cohort[row]
                means = np.asarray(getattr(lane, "adaptive_feature_means", np.zeros(self.num_draft)), dtype=float)
                counts = np.asarray(getattr(lane, "adaptive_feature_counts", np.zeros(self.num_draft)), dtype=np.int64)
                size = len(features)
                counts[:size] += 1
                means[:size] += (np.asarray(features) - means[:size]) / counts[:size]
                lane.adaptive_feature_means = means.tolist()
                lane.adaptive_feature_counts = counts.tolist()
            # A stop can cut an accepted sequence. Only delivered accepts and
            # the first rejection reached before that stop are labeled.
            accepted = _delivered_accepts(decision)
            rejected = decision.accepted < len(tokens) and decision.accepted < len(decision.emitted)
            self.acceptance_estimator.observe(features, accepted, rejected=rejected)
        self.acceptance_estimator.finish_round()

    @staticmethod
    def _trim_adaptive_block(block, depth):
        if block is None:
            return None
        if isinstance(block, CompactDraftRow):
            return CompactDraftRow(block.tokens[:depth], block.candidate_ids[:depth],
                                   block.candidate_probs[:depth], block.width,
                                   proposal_source=block.proposal_source,
                                   proposal_score_key=block.proposal_score_key,
                                   proposal_score=block.proposal_score,
                                   proposal_trust_key=block.proposal_trust_key,
                                   proposal_trusted=block.proposal_trusted,
                                   proposal_audit=block.proposal_audit,
                                   proposal_periodic_audit=block.proposal_periodic_audit)
        if isinstance(block, HostDraftRow):
            confidence = block.confidence_features
            return HostDraftRow(block.tokens[:depth], block.laws[:depth], block.width,
                                None if confidence is None else confidence[:depth],
                                proposal_source=block.proposal_source,
                                proposal_score_key=block.proposal_score_key,
                                proposal_score=block.proposal_score,
                                proposal_trust_key=block.proposal_trust_key,
                                proposal_trusted=block.proposal_trusted,
                                proposal_audit=block.proposal_audit,
                                proposal_periodic_audit=block.proposal_periodic_audit,
                                continuation_selection=block.continuation_selection,
                                feedback_payload=block.feedback_payload)
        raise ValueError("adaptive chain trimming requires a supported proposal block")

    def _adaptive_physical_groups(self, blocks):
        groups = {}
        for row, block in enumerate(blocks):
            depth = 0 if block is None else int(block.lengths[0])
            groups.setdefault(depth, []).append(row)
        result = []
        for indices in groups.values():
            while indices:
                width = max(size for size in range(1, len(indices) + 1)
                            if self.adaptive_policy.costs(size) is not None)
                result.append(indices[:width])
                indices = indices[width:]
        return result

    def _request_adaptive_round(self, cohort):
        """Atomic variable-depth round with physical forwards grouped by depth.

        Stochastic admission precedes proposal RNG draws. Deterministic full
        blocks may be trimmed using their own confidence hook. All lanes retain
        one common recovery boundary until every subgroup and estimator commits.
        """
        recovery = self._snapshot_round(cohort)
        self.scheduler_stats["recovery_checkpoint_captures"] += len(recovery)
        stats_snapshot = self._snapshot_scheduler_stats()
        estimator_snapshot = copy.deepcopy(self.acceptance_estimator)
        self._open = True
        transaction = None
        try:
            maxima = [min(self.num_draft, lane.maximum - lane.generated - 1)
                      if not lane.ordinary else 0 for lane in cohort]
            deterministic = getattr(self.draft, "proposal_distribution", None) == "deterministic_point_mass"
            blocks = [None] * len(cohort)
            features = []
            for lane, maximum in zip(cohort, maxima):
                means = getattr(lane, "adaptive_feature_means", None)
                counts = getattr(lane, "adaptive_feature_counts", None)
                features.append(means if counts is not None and all(counts[:maximum]) else None)
            if deterministic:
                depths = maxima
            else:
                # This choice cannot see any current sampled proposal token.
                depths = self.adaptive_policy.request_depths(self.acceptance_estimator, maxima, features)
            depths = [
                self._proposal_depth(depth, maximum) if maximum else 0
                for depth, maximum in zip(depths, maxima)
            ]
            draft_groups = {}
            for row, depth in enumerate(depths):
                draft_groups.setdefault(depth, []).append(row)
            for depth, indices in draft_groups.items():
                proposed = self._propose([cohort[row] for row in indices], adaptive_depth=depth)
                for row, block in zip(indices, proposed):
                    blocks[row] = block
            if deterministic:
                current_features = [getattr(block, "confidence_features", None) for block in blocks]
                if any(feature is not None for feature in current_features):
                    _bump(self.scheduler_stats, "external_adaptive_current_confidence_rounds")
                features = [current if current is not None else lagged
                            for current, lagged in zip(current_features, features)]
                depths = self.adaptive_policy.request_depths(self.acceptance_estimator, maxima, features)
                depths = [
                    self._proposal_depth(depth, maximum) if maximum else 0
                    for depth, maximum in zip(depths, maxima)
                ]
                blocks = [self._trim_adaptive_block(block, depth) for block, depth in zip(blocks, depths)]
            # This round verifies every row exactly; it has no trusted-PLD
            # path.  A "trusted" copy that reaches it is an exact audit, so
            # book and receipt it as one and let its labels feed revocation.
            demoted = False
            for block in blocks:
                if block is not None and bool(getattr(block, "proposal_trusted", False)):
                    block.proposal_trusted = False
                    block.proposal_audit = True
                    demoted = True
            if demoted:
                _bump(self.scheduler_stats, "external_composed_trusted_pld_adaptive_guard_rounds")
            decisions = [None] * len(cohort)
            physical = self._adaptive_physical_groups(blocks)
            _bump(self.scheduler_stats, "external_adaptive_rounds")
            _bump(self.scheduler_stats, "external_adaptive_per_request_rounds")
            _bump(self.scheduler_stats, "external_adaptive_verification_groups", len(physical))
            baseline = min(maxima, default=0)
            saved = max(0, baseline * len(cohort) - sum(depths))
            _bump(self.scheduler_stats, "external_adaptive_trimmed_rounds", int(saved > 0))
            _bump(self.scheduler_stats, "external_adaptive_trimmed_target_rows", saved)
            for lane, depth, block in zip(cohort, depths, blocks):
                lane.adaptive_round_policy = "per_request_grouped"
                lane.adaptive_feature_source = (
                    "deterministic_refined_logit_proxy"
                    if deterministic and getattr(block, "confidence_features", None) is not None else "lagged_q"
                )
                if depth < baseline:
                    lane.adaptive_trimmed_rounds = getattr(lane, "adaptive_trimmed_rounds", 0) + 1
            for indices in physical:
                lanes = [cohort[row] for row in indices]
                selected = [blocks[row] for row in indices]
                counts = [0 if block is None else int(block.lengths[0]) for block in selected]
                verify_width = counts[0] + 1
                self.scheduler_stats["external_adaptive_round_depth"] = counts[0]
                self.scheduler_stats["external_adaptive_verify_width"] = verify_width
                _bump(self.scheduler_stats, "external_adaptive_target_rows", verify_width * len(lanes))
                inputs = [[lane.anchor] + _block_row(block, None)[0] for lane, block in zip(lanes, selected)]
                taps, steer, commits = self._verify_steer(lanes, inputs, counts)
                if steer is not None:
                    taps.steer = steer
                try:
                    if self.target_execution == "provider":
                        parents = [list(range(-1, len(row) - 1)) for row in inputs]
                        logits, hidden, transaction = self._target_tree_forward_many(lanes, inputs, parents)
                    else:
                        transaction = self._target_owner([lane.cache for lane in lanes]).begin(
                            lengths=[count + 1 for count in counts])
                        logits, hidden = self._cohort_target_forward(
                            self.mx.array(inputs), transaction, lanes,
                            site="adaptive_grouped_verify")
                    self.mx.eval(logits, hidden)
                finally:
                    if steer is not None:
                        taps.steer = None
                verified = self._verify(lanes, selected, logits)
                self._commit(lanes, verified, hidden, blocks=selected, transaction=transaction)
                for commit, decision in zip(commits, verified):
                    commit(min(decision.accepted + 1, len(decision.emitted)))
                for row, decision in zip(indices, verified):
                    decisions[row] = decision
            self._adaptive_observe(blocks, decisions, cohort)
        except BaseException:
            if transaction is not None and not transaction.closed:
                try:
                    transaction.abort()
                except BaseException:  # noqa: BLE001, S110 - original boundary is authoritative
                    pass
            self._restore_round(cohort, recovery)
            self.scheduler_stats = stats_snapshot
            self.acceptance_estimator = estimator_snapshot
            self.scheduler_stats["recovery_checkpoint_restores"] += len(recovery)
            raise
        finally:
            self._open = False

    def _topology_observation_outcome(
        self,
        lane,
        decision,
        branch_accepts,
        proposal_counts,
        selected_branch,
    ):
        """Classify whether a committed AST result can train the latch."""

        reasons = []
        selected = int(selected_branch) - 1
        selected_full = (
            int(branch_accepts[selected]) == int(proposal_counts[selected])
        )
        if not selected_full and any(int(value) < 0 for value in branch_accepts):
            reasons.append("unvisited_alternative")
        if any(int(token) in self.stops for token in decision.emitted):
            reasons.append("target_stop")
        # Observations are assembled before _commit updates lane.generated.
        if int(lane.generated) + len(decision.emitted) >= int(lane.maximum):
            reasons.append("length_boundary")
        if lane.processors and not self._progressive_processors_supported(lane):
            reasons.append("processor_limited")
        return {
            "outcome": (
                "committed_censored" if reasons else "committed_conclusive"
            ),
            "censor_reasons": reasons,
            "budget_limited": "length_boundary" in reasons,
        }

    def _round(self, cohort, *, forced_depth=None):
        """Publish learning observations only after every request commits."""
        planner = (
            getattr(self.draft, "admission_planner", None)
            if self.continuation_policy is not None else None
        )
        admission_snapshot = planner.snapshot() if planner is not None else None
        self._feedback_outbox = []
        self._topology_observation_outbox = []
        self._continuation_open_selections = []
        ready_starts = {int(lane.uid): len(lane.ready) for lane in cohort}
        try:
            if forced_depth is None:
                # Keep compatibility with lightweight observers that wrap the
                # historical one-argument call shape. A requested forced
                # depth is always passed through below, so signature errors
                # remain visible instead of being mistaken for compatibility.
                self._run_round(cohort)
            else:
                self._run_round(cohort, forced_depth=forced_depth)
        except BaseException:
            if planner is not None:
                planner.restore(admission_snapshot)
            if self.continuation_policy is not None:
                pool = self.draft.proposal_pool
                selections = list(self._continuation_open_selections)
                selections.extend(getattr(self.draft, "last_continuation_selections", ()))
                seen = set()
                for selection in selections:
                    if selection is not None and id(selection) not in seen:
                        seen.add(id(selection))
                        try:
                            pool.discard(selection)
                        except ValueError:
                            pass  # Already consumed/discarded tickets carry no new labels.
            raise
        else:
            # This is outside the transaction's restore path. A diagnostic
            # feedback failure must never rewind already delivered target state.
            lane_by_uid = {int(lane.uid): lane for lane in cohort}
            observed_topology_uids = {
                int(uid)
                for _token, rows in self._topology_observation_outbox
                for uid, _observation in rows
            }
            primary_only = {}
            latch_payloads = getattr(self, "_last_alternative_latch", {})
            for lane in cohort:
                uid = int(lane.uid)
                payload = latch_payloads.get(uid)
                if (
                    uid not in observed_topology_uids
                    and payload is not None
                    and len(lane.ready) > ready_starts[uid]
                ):
                    primary_only.setdefault(
                        int(payload["decision_token"]), []
                    ).append(lane)
            for token, lane in (
                (token, lane) for token, lanes in primary_only.items() for lane in lanes
            ):
                lanes = [lane]
                try:
                    before = self.progressive_topology_latch.snapshot(lane.uid)
                    payload = latch_payloads[lane.uid]
                    unavailability_reason = (
                        "planner_zero_depth" if payload.get("planner_depth") == 0
                        else "planner_declined_alternatives"
                        if payload.get("planner_branches") == 1 else None
                    )
                    self.progressive_topology_latch.observe_primary_only(
                        session_uids=[lane.uid],
                        decision_tokens=[token],
                        unavailable=(
                            unavailability_reason is not None
                            or int(payload["active_branches"]) > 1
                        ),
                    )
                    snapshot = self.progressive_topology_latch.snapshot(lane.uid)
                    _bump(
                        self.scheduler_stats,
                        "external_progressive_ast_primary_only_rounds",
                    )
                    _bump(
                        self.scheduler_stats,
                        "external_progressive_ast_primary_only_sessions",
                        len(lanes),
                    )
                    for lane in lanes:
                        uid = int(lane.uid)
                        decision_payload = copy.deepcopy(
                            latch_payloads[uid]
                        )
                        decision_payload["application"] = "committed_primary_only"
                        lane.progressive_topology_latch_last_decision = (
                            decision_payload
                        )
                        observation = {
                            "outcome": "committed_primary_only",
                            "executed_alternative_branches": 0,
                            "accepted_token_uplift": 0,
                            "unavailability_reason": unavailability_reason,
                            "decision_token": int(token),
                            "snapshot_before": copy.deepcopy(before),
                            "snapshot_after": copy.deepcopy(snapshot),
                        }
                        lane.progressive_topology_latch_last_observation = observation
                        for response in list(lane.ready)[ready_starts[uid] :]:
                            alternatives = response.speculative_receipt.get(
                                "progressive_verification", {}
                            ).get("ordered_alternatives")
                            if alternatives is not None:
                                alternatives["adaptive_latch"][
                                    "last_decision"
                                ] = copy.deepcopy(decision_payload)
                                alternatives["adaptive_latch"][
                                    "last_observation"
                                ] = copy.deepcopy(observation)
                except Exception as error:  # noqa: BLE001 - learning is post-commit
                    _bump(
                        self.scheduler_stats,
                        "external_progressive_topology_observation_failures",
                    )
                    self.last_topology_observation_error = str(error)
            for token, rows in self._topology_observation_outbox:
                try:
                    self.progressive_topology_latch.observe_many(
                        [observation for _uid, observation in rows],
                        session_uids=[uid for uid, _observation in rows],
                        decision_tokens=[token] * len(rows),
                    )
                    for uid, observation in rows:
                        snapshot = self.progressive_topology_latch.snapshot(uid)
                        lane = lane_by_uid[uid]
                        decision_payload = copy.deepcopy(
                            self._last_alternative_latch[uid]
                        )
                        decision_payload["application"] = "committed"
                        lane.progressive_topology_latch_last_decision = (
                            decision_payload
                        )
                        lane.progressive_topology_latch_last_observation = {
                            **copy.deepcopy(observation),
                            "decision_token": int(token),
                            "snapshot_after": copy.deepcopy(snapshot),
                        }
                        for response in list(lane.ready)[ready_starts[uid] :]:
                            alternatives = response.speculative_receipt.get(
                                "progressive_verification", {}
                            ).get("ordered_alternatives")
                            if alternatives is not None:
                                alternatives["adaptive_latch"][
                                    "last_decision"
                                ] = copy.deepcopy(decision_payload)
                                alternatives["adaptive_latch"][
                                    "last_observation"
                                ] = copy.deepcopy(
                                    lane.progressive_topology_latch_last_observation
                                )
                except Exception as error:  # noqa: BLE001 - learning is post-commit
                    _bump(
                        self.scheduler_stats,
                        "external_progressive_topology_observation_failures",
                    )
                    self.last_topology_observation_error = str(error)
                    for uid, observation in rows:
                        lane = lane_by_uid[uid]
                        decision_payload = copy.deepcopy(
                            self._last_alternative_latch[uid]
                        )
                        decision_payload["application"] = "committed_learning_error"
                        lane.progressive_topology_latch_last_decision = (
                            decision_payload
                        )
                        degraded = {
                            **copy.deepcopy(observation),
                            "decision_token": int(token),
                            "application": "error_degraded",
                            "error": str(error),
                        }
                        lane.progressive_topology_latch_last_observation = degraded
                        for response in list(lane.ready)[ready_starts[uid] :]:
                            alternatives = response.speculative_receipt.get(
                                "progressive_verification", {}
                            ).get("ordered_alternatives")
                            if alternatives is not None:
                                alternatives["adaptive_latch"][
                                    "last_decision"
                                ] = copy.deepcopy(decision_payload)
                                alternatives["adaptive_latch"][
                                    "last_observation"
                                ] = copy.deepcopy(degraded)
            manager = getattr(self.draft, "feedback_manager", None)
            primary_outcomes = {
                int(uid): observation
                for _token, rows in self._topology_observation_outbox
                for uid, observation in rows
            }
            for lane, block, decision, round_id in self._feedback_outbox:
                selection = block.continuation_selection
                try:
                    observer = getattr(self.draft, "observe_proposal_feedback", None)
                    trusted_observer = getattr(
                        self.draft, "observe_trusted_pld", None
                    )
                    accepted = _delivered_accepts(decision)
                    # Stop/budget truncation censors an unobserved proposal
                    # suffix. Learn only from the draft prefix the committed
                    # response actually reached.
                    observed_proposed = min(
                        len(block.tokens), len(decision.emitted)
                    )
                    ast_planner = getattr(self, "ast_execution_planner", None)
                    if ast_planner is not None and observed_proposed:
                        if self.ast_collision_probability:
                            _tokens, proposal_laws = _block_row(
                                block, self._target_args.vocab_size
                            )
                            ast_planner.observe_collision_probabilities(
                                int(lane.uid),
                                _proposal_collision_probabilities(proposal_laws),
                            )
                            _bump(
                                self.scheduler_stats,
                                "external_ast_collision_feature_rows",
                            )
                        primary = primary_outcomes.get(int(lane.uid))
                        tested = observed_proposed
                        primary_accepted = accepted
                        if primary is not None:
                            # Do not teach the primary-chain prior the best
                            # of several branches: that would make wider
                            # searches manufacture their own confidence.
                            tested = min(tested, int(primary["proposal_counts"][0]))
                            primary_accepted = int(primary["branch_accepts"][0])
                        if primary_accepted >= 0:
                            ast_planner.observe_commit(
                                int(lane.uid), min(primary_accepted, tested), tested,
                                censored=primary_accepted >= tested,
                                source_depth=getattr(lane, "ast_source_budget", {}).get("depth"),
                            )
                        if primary is not None and len(primary["branch_accepts"]) > 1:
                            # What the extra branch rows bought this round:
                            # committed tokens beyond the primary chain, per
                            # target row beyond the primary's own rows.
                            accepts = [int(a) for a in primary["branch_accepts"]]
                            committed = [max(1, a + 1) if a >= 0 else 0 for a in accepts]
                            winner = int(primary["selected_branch"]) - 1
                            primary_rows = int(primary["proposal_counts"][0]) + 1
                            ast_planner.observe_branches(
                                int(lane.uid),
                                max(0, int(primary["unique_target_rows"]) - primary_rows),
                                max(0, committed[winner] - committed[0]),
                            )
                    if (
                        callable(observer)
                        and block.proposal_score_key is not None
                        and not block.proposal_trusted
                        and observed_proposed
                    ):
                        observer(
                            lane.proposal_score_counts,
                            block.proposal_score_key,
                            observed_proposed,
                            accepted,
                            self.draft.policy.score_decay_per_token,
                        )
                    if callable(trusted_observer) and (
                        block.proposal_trusted or block.proposal_audit
                    ) and observed_proposed:
                        trusted_observer(
                            lane.proposal_trust_states,
                            block.proposal_trust_key,
                            proposed=observed_proposed,
                            audited=bool(block.proposal_audit),
                            periodic_audit=bool(block.proposal_periodic_audit),
                            accepted=accepted if block.proposal_audit else 0,
                        )
                    self._attach_postcommit_proposal_receipts(
                        lane,
                        block,
                        decision,
                        proposed=observed_proposed,
                        accepted=accepted,
                    )
                    if selection is not None:
                        coverage = getattr(lane, "continuation_coverage", {})
                        # Each top-W subset is ranked before target draws. A
                        # first mismatch is observed; later positions censored.
                        for width in range(1, len(selection.paths) + 1):
                            active = [path.tokens for path in selection.paths[:width]]
                            for position, teacher in enumerate(decision.emitted):
                                eligible = [path for path in active if len(path) > position]
                                if not eligible:
                                    break
                                matches = [path for path in eligible if path[position] == teacher]
                                yes, no = coverage.get((width, position), (0, 0))
                                coverage[(width, position)] = (yes + int(bool(matches)), no + int(not matches))
                                if not matches:
                                    break
                                active = matches
                        lane.continuation_coverage = coverage
                        self.draft.proposal_pool.commit_feedback(selection, decision.emitted)
                    if manager is not None and block.feedback_payload is not None:
                        # ``emitted`` ends with the bonus token on a full
                        # acceptance; only a draft rejected before any stop
                        # cut is a rejection boundary inside the lattice.
                        first_rejected = (
                            decision.accepted
                            if decision.accepted
                            < min(len(block.tokens), len(decision.emitted))
                            else None
                        )
                        manager.submit_verified(block.feedback_payload, decision.emitted,
                            first_rejected_position=first_rejected,
                            request_id=lane.proposal_request_id, round_id=round_id)
                except Exception as error:  # noqa: BLE001 - optional learning cannot alter committed inference
                    _bump(self.scheduler_stats, "external_feedback_failures")
                    self.last_feedback_error = str(error)
                    if selection is not None:
                        try:
                            self.draft.proposal_pool.discard(selection)
                        except ValueError:
                            pass
            if manager is not None and self._feedback_outbox:
                try:
                    manager.settle_round()
                    receipt = manager.receipt()
                    for lane, _block, _decision, _round_id in self._feedback_outbox:
                        for response in lane.ready:
                            response.speculative_receipt["lilicorr_feedback"] = copy.deepcopy(receipt)
                except Exception as error:  # noqa: BLE001 - optional learning cannot alter committed inference
                    _bump(self.scheduler_stats, "external_feedback_failures")
                    self.last_feedback_error = str(error)
            if self.continuation_policy is not None:
                for lane, _block, _decision, _round_id in self._feedback_outbox:
                    try:
                        ranking = self.draft.proposal_pool.receipt(lane.session_scope_hash)
                        for response in lane.ready:
                            response.speculative_receipt["continuation_pool"]["ranking"] = copy.deepcopy(ranking)
                    except Exception as error:  # noqa: BLE001 - diagnostics cannot invalidate committed inference
                        _bump(self.scheduler_stats, "external_feedback_failures")
                        self.last_feedback_error = str(error)
        finally:
            self._feedback_outbox = None
            self._topology_observation_outbox = None
            self._continuation_open_selections = []

    def _round_at_batch_route(self, cohort, *, concurrent, active_width=None):
        """Select a route only between transactions, never during a round.

        Tree and chain share each lane's committed draft/target cache boundary.
        A prelaunched tree proposal belongs to the previous boundary and must
        be discarded before a concurrent chain round can advance it.
        """
        states = [lane.cost_handoff_state for lane in cohort]
        width = len(cohort)
        logical_width = width if active_width is None else int(active_width)
        _bump(self.scheduler_stats, f"external_round_cohort_width_{width}")
        if cohort and all(lane.ordinary for lane in cohort):
            # A permanent ordinary decision is authoritative.  In particular,
            # an explicitly disabled lane must not run the cost controller's
            # initial depth-zero sampling rounds before reaching the direct
            # ordinary pipeline.
            inflight = getattr(self, "_parked_inflight", None)
            if inflight is not None:
                drained = list(inflight.lanes)
                self._drain_parked_inflight()
                if any(lane.ready for lane in drained if lane in cohort):
                    return
            self._round(cohort)
            return
        from .cost_per_commit_handoff import cohort_needs_ordinary_probe

        parked_uids = {lane.uid for lane in cohort if _speculation_parked(lane)}
        mixed_parked_uids = set()
        inflight = getattr(self, "_parked_inflight", None)
        if inflight is not None and (
            len(parked_uids) != len(cohort)
            or {lane.uid for lane in cohort}
            != {lane.uid for lane in inflight.lanes}
        ):
            # Another cohort owns this round: finish the parked one first.
            drained = list(inflight.lanes)
            self._drain_parked_inflight()
            inflight = None
            if any(lane.ready for lane in drained if lane in cohort):
                # Those lanes just produced this poll's token.
                return
        if inflight is not None:
            # The scheduler rotates its service order every poll; the
            # dispatched round's rows keep the order they were launched in.
            cohort = list(inflight.lanes)
        if (
            parked_uids
            and len(parked_uids) == len(cohort)
            and (inflight is not None or self._parked_pipeline_eligible(cohort))
        ):
            if inflight is None:
                # A tree round may have queued this cohort's next lattice
                # (MLX2_TREE_PIPELINE_DRAFT) at the boundary the parked rounds
                # now move; release it before the first dispatch.
                self._discard_prelaunched([lane.uid for lane in cohort])
            self._parked_pipeline_round(cohort)
            return
        if parked_uids and len(parked_uids) == len(cohort):
            # A parked cohort times only the target step and defers the draft
            # append, until a session's deferred context reaches its cap; that
            # flush round appends it and is not timing evidence.
            flush = any(
                int(lane.tail.shape[1])
                >= lane.cost_handoff_state.settings.deferred_context_cap
                for lane in cohort
            )
            ordinary_probe = not flush
            ordinary_probe_uids = set() if flush else set(parked_uids)
            _bump(
                self.scheduler_stats,
                "external_cost_parked_flush_rounds" if flush
                else "external_cost_parked_rounds",
            )
        else:
            if getattr(self, "ast_execution_planner", None) is not None:
                mixed_parked_uids = set(parked_uids)
            parked_uids = set()
            ordinary_probe = cohort_needs_ordinary_probe(states, width)
            ordinary_probe_uids = {
                lane.uid
                for lane, state in zip(cohort, states, strict=True)
                if state is not None and state.needs_ordinary_probe(width)
            }
            if ordinary_probe:
                ordinary_probe_uids.update(mixed_parked_uids)
                mixed_parked_uids.clear()
            elif mixed_parked_uids:
                _bump(self.scheduler_stats, "external_ast_mixed_parked_rounds")
                _bump(self.scheduler_stats, "external_ast_mixed_parked_lanes",
                      len(mixed_parked_uids))
        if (
            ordinary_probe
            and self.dynamic_singleton_tree
            and not mixed_parked_uids
            and self._plain_target_step_eligible(cohort)
        ):
            # The tree route's ordinary probe and its unpipelined parked
            # rounds time the plain target step the ordinary route runs, not
            # a depth-zero external round through the tree provider.
            self._plain_target_round(
                cohort,
                probe_uids=ordinary_probe_uids,
                execution_width=logical_width,
            )
            return
        # Keep the default route free of timing calls.  ``states`` has one
        # entry per lane even when the capability was not selected.
        started_ns = (
            time.perf_counter_ns()
            if any(state is not None for state in states)
            else None
        )
        generated_before = [lane.generated for lane in cohort]
        if not self.dynamic_singleton_tree:
            self._round(cohort, forced_depth=0 if ordinary_probe else None)
            self._observe_cost_handoff_round(
                cohort,
                generated_before,
                started_ns,
                ordinary_probe=ordinary_probe or bool(parked_uids),
                ordinary_probe_uids=ordinary_probe_uids,
                prior_execution_width=logical_width,
                mixed_parked_uids=mixed_parked_uids,
            )
            return
        mode = "chain" if concurrent else "tree15"
        saved = (self.draft_topology, self.target_execution)
        saved_widths = (self._auto_active_width, self._auto_cohort_width)
        self._auto_active_width = len(self.lanes) if active_width is None else int(active_width)
        self._auto_cohort_width = len(cohort)
        if concurrent:
            self._discard_prelaunched([lane.uid for lane in self.lanes.values()])
            self.draft_topology = "chain"
            self.target_execution = "reference"
        try:
            self._round(cohort, forced_depth=0 if ordinary_probe else None)
            self._observe_cost_handoff_round(
                cohort,
                generated_before,
                started_ns,
                ordinary_probe=ordinary_probe,
                ordinary_probe_uids=ordinary_probe_uids,
                prior_execution_width=logical_width,
            )
            if mode != self._auto_last_mode and self._auto_last_mode is not None:
                _bump(self.scheduler_stats, "external_auto_mode_switches")
            self._auto_last_mode = mode
            _bump(
                self.scheduler_stats,
                "external_auto_chain_rounds" if concurrent else "external_auto_tree_rounds",
            )
            maximum = "external_auto_chain_max_width" if concurrent else "external_auto_tree_max_width"
            self.scheduler_stats[maximum] = max(
                self.scheduler_stats[maximum], len(cohort)
            )
        finally:
            self.draft_topology, self.target_execution = saved
            self._auto_active_width, self._auto_cohort_width = saved_widths

    def _parked_pipeline_eligible(self, cohort, *, following=False):
        """Whether a parked cohort's anchor-only round may run before its read.

        Only the whole executor: every live lane is in this greedy, processor-
        free, hybrid parked cohort (no other lane may see a transaction across
        the publication point), every lane has budget for another token, and
        the deferred draft context stays below its cap.

        The tree route (``dynamic_singleton_tree``, provider target) is
        included: a parked round there is the same plain hybrid target step
        on the lane's own cache (the provider never sees it), its draft
        append is deferred exactly as on the chain route, and the caller
        releases the cohort's prelaunched lattices before the first dispatch.
        The hybrid-cache, residual-tap and fly-verification guards remain:
        only the hybrid owner has a plain one-token transaction, and steering
        or relaxed verification change the step itself.
        """
        settings = self.cost_per_commit_handoff
        if (
            settings is None
            or self._ast_anchor_cohort_eligible(cohort)
            or not settings.reversible
            or not settings.pipelined_park
            or self.fly_verification.enabled
            or getattr(getattr(self.model, "model", None), "residual_taps", None)
            is not None
        ):
            return False
        live = [lane for lane in self.lanes.values() if not lane.cancelled]
        if {id(lane) for lane in live} != {id(lane) for lane in cohort}:
            return False
        if not self._is_hybrid([lane.cache for lane in cohort]):
            return False
        reserve = 2 if following else 1
        for lane in cohort:
            if (
                not _speculation_parked(lane)
                or lane.ordinary
                or lane.processors
                or float(lane.sampling.get("sampling_temp", 0)) != 0.0
                or lane.maximum - lane.generated <= reserve
                or int(lane.tail.shape[1]) + reserve + 1
                >= lane.cost_handoff_state.settings.deferred_context_cap
            ):
                return False
        return True

    def _plain_target_step_eligible(self, cohort):
        """Whether a tree-route anchor-only round may be one plain target step.

        On the tree route a depth-zero external round (an ordinary probe, or
        a parked round that cannot pipeline) otherwise runs the full external
        round: recovery snapshot, zero-count proposal and a one-row provider
        tree.  That costs about twice the plain decode step it stands for, so
        the cost controller compared the tree against an inflated ordinary
        baseline.  The plain step is the parked pipeline's own round, run
        synchronously: greedy, processor-free lanes of a hybrid target with
        no residual taps or fly verification.  Sampled or processed rows keep
        the exact external round and its RNG schedule.  A parked lane also
        needs its deferred draft context below the cap (the flush round
        appends it).  Each lane needs budget beyond this token: a final
        token keeps the round that publishes the draft plane.
        """
        if (
            not cohort
            or self._open
            or self.fly_verification.enabled
            or getattr(getattr(self.model, "model", None), "residual_taps", None)
            is not None
            or self._ast_anchor_cohort_eligible(cohort)
            or not self._is_hybrid([lane.cache for lane in cohort])
        ):
            return False
        for lane in cohort:
            state = lane.cost_handoff_state
            if (
                lane.ordinary
                or lane.cancelled
                or lane.processors
                or float(lane.sampling.get("sampling_temp", 0)) != 0.0
                or lane.maximum - lane.generated <= 1
                or (
                    _speculation_parked(lane)
                    and int(lane.tail.shape[1])
                    >= state.settings.deferred_context_cap
                )
            ):
                return False
        return True

    def _plain_target_round(self, cohort, *, probe_uids, execution_width):
        """One synchronous plain target step for an anchor-only tree cohort.

        The cohort's prelaunched lattices belong to the boundary this step
        moves, so they are released first.  Timing covers dispatch, commit
        and the host read, as the depth-zero external round's did; the draft
        append stays deferred.  Only ``probe_uids`` record ordinary evidence.
        """
        self._discard_prelaunched([lane.uid for lane in cohort])
        probe = any(not _speculation_parked(lane) for lane in cohort)
        inflight = self._dispatch_parked(
            cohort,
            [lane.anchor for lane in cohort],
            timing_start_ns=time.perf_counter_ns(),
        )
        try:
            rows = inflight.transaction.commit(
                accepted_lengths=[1] * len(cohort)
            )
        except BaseException:
            if not inflight.transaction.closed:
                inflight.transaction.abort()
            self._restore_parked(inflight.recovery)
            raise
        self._read_parked(
            inflight,
            rows,
            None,
            pipelined=False,
            probe_uids=probe_uids,
            prior_execution_width=execution_width,
        )
        _bump(self.scheduler_stats, "external_cost_plain_target_rounds")
        if probe:
            _bump(self.scheduler_stats, "external_cost_plain_probe_rounds")

    def _dispatch_parked(self, cohort, anchors, *, timing_start_ns):
        """Begin and dispatch one anchor-only round; nothing is read."""
        phase = _PhaseClock(self) if self.round_timing else None
        owner = self._parked_target_owner(cohort)
        recovery = self._parked_recovery(
            cohort, recurrent_rows=getattr(owner, "_recurrent_rows", None)
        )
        if phase is not None:
            phase("parked_snapshot")
        transaction = owner.begin(
            lengths=[1] * len(cohort), plain_single_token=True
        )
        try:
            if not getattr(transaction, "plain", False):
                raise RuntimeError("a pipelined parked round requires a plain transaction")
            if isinstance(anchors, self.mx.array):
                inputs = anchors.astype(self.mx.int32)[:, None]
            else:
                inputs = self.mx.array([[int(a)] for a in anchors], dtype=self.mx.int32)
            logits, features = self._cohort_target_forward(
                inputs, transaction, cohort, site="parked_plain"
            )
            # Everything the host read needs is built here, before the
            # following round launches: an op created on these logits after
            # that launch would queue behind the following forward and wait
            # for it (Rapid-MLX #4238).  Same float32 argmax and log-softmax
            # as ``_target_law``.
            # Per-row slices too: even a reshape or row slice made at read
            # time is a new op behind the following round.
            rows32 = logits[:, -1, :].astype(self.mx.float32)
            normalizer_rows = self.mx.logsumexp(rows32, axis=-1, keepdims=True)
            logprob_rows = [
                rows32[row] - normalizer_rows[row] for row in range(len(cohort))
            ]
            normalizers = normalizer_rows.reshape(-1)
            next_anchors = self.mx.argmax(rows32, axis=-1)
            if phase is not None:
                phase("parked_build")
            self.mx.async_eval(
                logits, features, next_anchors, normalizers, *logprob_rows
            )
            if phase is not None:
                phase("parked_launch")
        except BaseException:
            if not transaction.closed:
                transaction.abort()
            raise
        return _ParkedInflight(
            lanes=list(cohort),
            transaction=transaction,
            logits=logits,
            features=features,
            next_anchors=next_anchors,
            normalizers=normalizers,
            logprob_rows=logprob_rows,
            recovery=recovery,
            timing_start_ns=timing_start_ns,
            generated_before=[lane.generated for lane in cohort],
        )

    def _parked_target_owner(self, cohort):
        """The plain step's hybrid owner, kept while the cohort is unchanged.

        A multi-lane plain step runs on the owner's batched views (segmented
        KV rows, joined recurrent state).  Rebuilding the owner every round
        re-validated every plane and re-joined every recurrent state from its
        rows; a kept owner resynchronizes its views instead (KV views re-read
        row offsets, a recurrent view re-joins only when a row no longer holds
        the arrays it split into it), exactly as the PLD cohort owner does.
        A B1 owner has no views; keeping it skips the per-plane validation.
        """
        from .hybrid_verify_rows import HybridVerifyRows

        rows = [lane.cache for lane in cohort]
        owner = getattr(self, "_parked_owner", None)
        if (
            owner is not None
            and owner.holds(rows)
            and (owner._active is None or owner._active.closed)
        ):
            _bump(self.scheduler_stats, "external_hybrid_transactions")
            _bump(self.scheduler_stats, "external_parked_owner_reuses")
            return owner
        owner = self._target_owner(rows)
        self._parked_owner = owner if isinstance(owner, HybridVerifyRows) else None
        return owner

    def _parked_recovery(self, cohort, *, recurrent_rows=None):
        """A one-token plain step's exact undo record, without a deep snapshot.

        The pipelined round advances each lane by exactly one anchor, so its
        undo is the plain transaction's own: trim every KV plane back to its
        offset and put each recurrent plane's state list back.  Host fields
        are restored from a shallow copy; the lists the round appends to
        (history, ready) are cut back to their lengths and the RNG position is
        recorded.  ``recurrent_rows`` is the owner's per-plane classification
        of exactly these lanes, when the caller holds one.
        """
        from .hybrid_verify_rows import _kv_offset, _recurrent

        if recurrent_rows is None or len(recurrent_rows) != len(cohort):
            recurrent_rows = [
                [_recurrent(cache) for cache in lane.cache] for lane in cohort
            ]
        records = []
        for lane, row_flags in zip(cohort, recurrent_rows):
            planes = []
            for cache, recurrent in zip(lane.cache, row_flags, strict=True):
                if recurrent:
                    planes.append(
                        ("recurrent", cache, list(cache.cache), cache.lengths, cache.left_padding)
                    )
                else:
                    planes.append(("kv", cache, _kv_offset(cache)))
            records.append([lane, *self._parked_host_record(lane), planes])
        return records

    @staticmethod
    def _parked_host_record(lane):
        return dict(vars(lane)), len(lane.history), len(lane.ready), _rng_mark(lane.rng)

    def _refresh_parked_host(self, records):
        """Re-take the host half after the previous round is booked.

        A following round is dispatched (and its cache planes recorded) before
        the round ahead of it is read, so its host fields must be captured at
        the same boundary as those planes: after that read.
        """
        for record in records:
            record[1:5] = self._parked_host_record(record[0])

    def _restore_parked(self, records):
        from .hybrid_verify_rows import _kv_offset

        for lane, fields, history, ready, rng, planes in records:
            for plane in planes:
                if plane[0] == "recurrent":
                    _kind, cache, state, lengths, left_padding = plane
                    cache.cache = list(state)
                    cache.lengths, cache.left_padding = lengths, left_padding
                else:
                    _kind, cache, offset = plane
                    drop = _kv_offset(cache) - offset
                    if drop < 0 or (drop and int(cache.trim(drop)) != drop):
                        raise RuntimeError("parked recovery could not rewind KV")
            lane.__dict__.clear()
            lane.__dict__.update(fields)
            while len(lane.history) > history:
                lane.history.pop()
            while len(lane.ready) > ready:
                lane.ready.pop()
            lane.rng = rng.restore() if isinstance(rng, _RngMark) else rng
        _bump(self.scheduler_stats, "recovery_checkpoint_restores", len(records))

    def _abort_parked_inflight(self):
        inflight, self._parked_inflight = self._parked_inflight, None
        self._parked_last_read_ns = None
        if inflight is not None and not inflight.transaction.closed:
            inflight.transaction.abort()
            _bump(self.scheduler_stats, "external_parked_pipeline_aborts")

    def _drain_parked_inflight(self):
        """Read the dispatched parked round without dispatching another."""
        inflight, self._parked_inflight = self._parked_inflight, None
        if inflight is None:
            return
        rows = inflight.transaction.commit(accepted_lengths=[1] * len(inflight.lanes))
        _bump(self.scheduler_stats, "external_parked_pipeline_drains")
        self._read_parked(inflight, rows, None)
        self._parked_last_read_ns = None

    def _parked_decisions(self, cohort, inflight):
        """``_verify`` of zero-draft greedy rows from precomputed device values.

        Identical decisions to ``_verify``: the float32 argmax token, the
        processed (here: unprocessed) log-softmax response row, the same
        greedy verifier call and its RNG schedule, and stop truncation.
        """
        tokens = [int(value) for value in inflight.next_anchors.tolist()]
        normalizers = inflight.normalizers.tolist()
        decisions = []
        for row, lane in enumerate(cohort):
            if not np.isfinite(float(normalizers[row])):
                raise LaneFailure(
                    lane.uid,
                    "external draft target law is not a probability distribution",
                )
            response_rows = (
                [inflight.logprob_rows[row]]
                if lane.sampling.get("emit_logprobs", True)
                else None
            )
            window = VerifyWindow(lane.processors)
            window.mark()
            result = verify_greedy_proposals([], [tokens[row]], lane.rng)
            emitted = list(result.emitted)
            for index, token in enumerate(emitted):
                if token in self.stops:
                    emitted = emitted[: index + 1]
                    break
            decisions.append(
                RoundDecision(
                    result.accepted,
                    emitted,
                    None,
                    result.relaxed_accepts,
                    response_rows,
                    window,
                    relaxed_positions=result.relaxed_positions,
                )
            )
        return decisions

    def _parked_pipeline_round(self, cohort):
        """One poll of pipelined parked decode (the PLD plain-round pipeline).

        Commit the dispatched round (no host token needed), dispatch the
        following round from its lazy argmax, then read the committed round on
        the host while the device runs the following one.  A first poll only
        dispatches.
        """
        inflight = self._parked_inflight
        if inflight is None:
            self._parked_inflight = self._dispatch_parked(
                cohort,
                [lane.anchor for lane in cohort],
                timing_start_ns=time.perf_counter_ns(),
            )
            _bump(self.scheduler_stats, "external_parked_pipeline_starts")
            return
        self._parked_inflight = None
        phase = _PhaseClock(self) if self.round_timing else None
        try:
            rows = inflight.transaction.commit(
                accepted_lengths=[1] * len(inflight.lanes)
            )
            if phase is not None:
                phase("parked_commit")
        except BaseException:
            if not inflight.transaction.closed:
                inflight.transaction.abort()
            self._restore_parked(inflight.recovery)
            raise
        following = None
        if self._parked_pipeline_eligible(cohort, following=True):
            try:
                following = self._dispatch_parked(
                    cohort,
                    inflight.next_anchors,
                    timing_start_ns=0,
                )
            except BaseException:  # noqa: BLE001 - the read below still books the round
                following = None
        self._read_parked(inflight, rows, following)

    def _read_parked(
        self,
        inflight,
        rows,
        following,
        *,
        pipelined=True,
        probe_uids=None,
        prior_execution_width=None,
    ):
        """Book a committed parked round; keep ``following`` only if it is exact.

        ``pipelined=False`` books a synchronous plain step
        (``_plain_target_round``): no pipeline state is read or left behind,
        and only ``probe_uids`` record ordinary evidence.
        """
        cohort = inflight.lanes
        phase = _PhaseClock(self) if self.round_timing else None
        self._open = True
        try:
            self.mx.eval(
                inflight.features,
                inflight.next_anchors,
                inflight.normalizers,
                *inflight.logprob_rows,
            )
            if phase is not None:
                phase("parked_wait")
            blocks = [None] * len(cohort)
            decisions = self._parked_decisions(cohort, inflight)
            if phase is not None:
                phase("parked_decide")
            lazy = [int(value) for value in inflight.next_anchors.tolist()]
            keep = following is not None
            for row, (lane, decision) in enumerate(zip(cohort, decisions)):
                token = int(decision.emitted[0])
                if (
                    token != lazy[row]
                    or token in self.stops
                    or lane.generated + 1 >= lane.maximum
                ):
                    keep = False
                    if token != lazy[row]:
                        _bump(self.scheduler_stats, "external_parked_pipeline_mismatches")
            if following is not None and not keep:
                # The following round used an anchor this round does not
                # commit, or a lane finishes here and its cache must freeze at
                # this boundary: rewind it exactly before booking.
                following.transaction.abort()
                following = None
                _bump(self.scheduler_stats, "external_parked_pipeline_drains")
            self._commit(
                cohort,
                decisions,
                inflight.features,
                blocks=blocks,
                transaction=_CommittedPlainRows(rows),
                deferred_context=True,
            )
            # Before the following round's host record is retaken: its
            # recovery must describe the tail this flush leaves.
            self._flush_parked_context(cohort)
            if following is not None:
                self._refresh_parked_host(following.recovery)
                # The following round was dispatched before this read booked
                # its token; its evidence is its own token only.  Counting
                # from the dispatch boundary credited two tokens per read
                # interval and halved the parked ordinary baseline.
                following.generated_before = [
                    lane.generated for lane in following.lanes
                ]
            if phase is not None:
                phase("parked_book")
        except BaseException:
            if following is not None and not following.transaction.closed:
                with suppress(Exception):
                    following.transaction.abort()
            self._restore_parked(inflight.recovery)
            if pipelined:
                self._parked_last_read_ns = None
            raise
        finally:
            self._open = False
        now = time.perf_counter_ns()
        start = (
            self._parked_last_read_ns
            if pipelined and self._parked_last_read_ns is not None
            else inflight.timing_start_ns
        )
        self._observe_cost_handoff_round(
            cohort,
            inflight.generated_before,
            start,
            ordinary_probe=True,
            ordinary_probe_uids=(
                {lane.uid for lane in cohort}
                if probe_uids is None
                else set(probe_uids)
            ),
            prior_execution_width=prior_execution_width,
        )
        if not pipelined:
            return
        _bump(self.scheduler_stats, "external_cost_parked_rounds")
        _bump(self.scheduler_stats, "external_parked_pipelined_rounds")
        self._parked_last_read_ns = now if following is not None else None
        self._parked_inflight = following

    def _flush_parked_context(self, cohort):
        """Commit a parked lane's deferred draft context before it reaches its cap.

        A parked lane defers its draft append; at the cap the scheduler used
        to stop the pipeline and run a flush round, a full depth-zero external
        round (recovery snapshot, provider tree, draft append) whose only job
        was the append, at about twice a plain step and then a pipeline
        restart.  The append is the same call that round's proposal phase
        makes at this boundary (``_append_context(lane, lane.anchor)``), so
        the draft plane is identical; it is queued behind the in-flight plain
        step and the pipeline keeps running.  A lane that finished this round
        keeps its tail: its response already published the paired sidecar.
        """
        for lane in cohort:
            state = lane.cost_handoff_state
            if (
                state is None
                # The AST planner owns its parked context schedule.
                or getattr(self, "ast_execution_planner", None) is not None
                or not _speculation_parked(lane)
                or int(lane.tail.shape[1]) + 3 < state.settings.deferred_context_cap
                or any(
                    getattr(response, "finish_reason", None) is not None
                    for response in lane.ready
                )
            ):
                continue
            self._append_context(lane, lane.anchor)
            _bump(self.scheduler_stats, "external_parked_context_flushes")
            _bump(
                self.scheduler_stats,
                "external_parked_context_flush_positions",
                int(lane.tail.shape[1]),
            )
            lane.tail = lane.tail[:, :0]

    def _observe_cost_handoff_round(
        self,
        cohort,
        generated_before,
        started_ns,
        *,
        ordinary_probe,
        ordinary_probe_uids=(),
        prior_execution_width=None,
        mixed_parked_uids=(),
    ):
        """Publish request-local timing only after a complete round commit."""
        if started_ns is None:
            return
        elapsed_ns = max(time.perf_counter_ns() - started_ns, 1)
        width = len(cohort)
        reprobed = reengaged = False
        for lane, before in zip(cohort, generated_before, strict=True):
            state = lane.cost_handoff_state
            if state is None or lane.cancelled:
                continue
            if ordinary_probe and lane.uid not in ordinary_probe_uids:
                continue
            if not ordinary_probe and lane.uid in ordinary_probe_uids:
                # A bounded speculative round granted to established peers
                # is not evidence for a lane without a baseline.
                continue
            if lane.uid in mixed_parked_uids:
                decision = state.observe_parked_progress(
                    committed_tokens=lane.generated - before,
                    lane_uid=lane.uid,
                    owner_scope=(self.binding, lane.session_scope_hash),
                )
            else:
                decision = state.observe(
                    kind="ordinary" if ordinary_probe else "speculative",
                    elapsed_ns=elapsed_ns,
                    committed_tokens=lane.generated - before,
                    execution_width=width,
                    lane_uid=lane.uid,
                    owner_scope=(self.binding, lane.session_scope_hash),
                )
            if decision is None:
                continue
            if state.settings.reversible:
                # A park only changes how the next closed boundary groups
                # this session; its draft plane stays exact throughout.
                prior_width = prior_execution_width
                if prior_width is None and state.session_prior_active:
                    prior_width = 1
                if prior_width is not None:
                    self._update_ast_cost_prior(
                        decision, execution_width=prior_width
                    )
                reprobed = reprobed or decision["event"] == "reprobe"
                reengaged = reengaged or decision["event"] == "reengage"
                _bump(self.scheduler_stats, f"external_cost_{decision['event']}_events")
                lane.cost_handoff_receipt = state.receipt()
                lane.cost_handoff_receipt = self._cost_handoff_receipt(lane)
                for response in lane.ready:
                    response.speculative_receipt["cost_per_commit_handoff"] = (
                        self._cost_handoff_receipt(lane)
                    )
                continue
            if not any(
                getattr(response, "finish_reason", None) is not None
                for response in lane.ready
            ):
                # The target and draft transaction already committed.
                # Marking this lane ordinary only affects the next closed
                # boundary.  A lane that finished this round never leaves
                # speculation: its receipt keeps the decision only.
                lane.ordinary = True
                state.mark_handoff_performed()
                _bump(self.scheduler_stats, "external_cost_handoff_events")
                _bump(self.scheduler_stats, "external_cost_handoff_lanes")
            lane.cost_handoff_receipt = state.receipt()
            lane.cost_handoff_receipt = self._cost_handoff_receipt(lane)
            for response in lane.ready:
                response.speculative_receipt["cost_per_commit_handoff"] = (
                    self._cost_handoff_receipt(lane)
                )
        if not ordinary_probe and width >= 2:
            reengaged = (
                self._reconcile_cohort_costs(cohort, generated_before, elapsed_ns)
                or reengaged
            )
        if reprobed:
            self._align_cohort_reprobes()
        self._record_width_park_prior(reengaged=reengaged)

    def _publish_cost_receipt(self, lane):
        lane.cost_handoff_receipt = lane.cost_handoff_state.receipt()
        lane.cost_handoff_receipt = self._cost_handoff_receipt(lane)
        for response in lane.ready:
            response.speculative_receipt["cost_per_commit_handoff"] = (
                self._cost_handoff_receipt(lane)
            )

    def _reconcile_cohort_costs(self, cohort, generated_before, elapsed_ns):
        """Keep a whole speculating cohort on one route at width two or more.

        After a speculative round over every live session, some sessions may
        have parked (or re-parked after a probe) while others keep speculating
        or re-engage.  The decode phase would then split into a speculative
        round beside an unpipelined plain step every poll, and the lone
        speculating sessions would be timed at a width the cohort no longer
        runs.  The cohort instead adopts the aggregate decision: this round's
        cost per committed token across the cohort (``elapsed * width /
        committed``, the per-session unit) against the sessions' mean ordinary
        baseline, with the sessions' loss ratio.
        """
        live = self._live_reversible_lanes()
        if (
            not live
            or len(live) < 2
            or {lane.uid for lane in live} != {lane.uid for lane in cohort}
        ):
            return False
        states = [lane.cost_handoff_state for lane in cohort]
        paused = [state.speculation_paused for state in states]
        if all(paused) or not any(paused):
            return False
        baselines = [state.ordinary_baseline() for state in states]
        committed = sum(
            lane.generated - before for lane, before in zip(cohort, generated_before)
        )
        if any(value is None for value in baselines) or committed <= 0:
            return False
        ordinary = sum(baselines) / len(baselines)
        speculative = elapsed_ns * len(cohort) / committed
        park = speculative >= ordinary / states[0].settings.loss_ratio
        backoff = max(
            (state.reprobe_after_tokens for state in states if state.speculation_paused),
            default=None,
        )
        _bump(
            self.scheduler_stats,
            "external_cost_cohort_park_decisions" if park
            else "external_cost_cohort_reengage_decisions",
        )
        for lane, state in zip(cohort, states):
            decision = state.adopt_cohort_decision(
                park=park,
                ordinary_cost=ordinary,
                speculative_cost=speculative,
                reprobe_after=backoff,
            )
            if decision is None:
                continue
            _bump(self.scheduler_stats, f"external_cost_{decision['event']}_events")
            self._publish_cost_receipt(lane)
        return not park

    def _live_reversible_lanes(self):
        """Live decoding lanes when every one carries reversible cost state.

        ``None`` under the AST planner, which packs parked anchors into its own
        cohorts, or when any live lane has no reversible state.
        """
        if getattr(self, "ast_execution_planner", None) is not None:
            return None
        live = [
            lane for lane in getattr(self, "lanes", {}).values()
            if not lane.cancelled and lane.anchor is not None
        ]
        if any(
            lane.cost_handoff_state is None
            or not lane.cost_handoff_state.settings.reversible
            or lane.cost_handoff_state.handed_off
            for lane in live
        ):
            return None
        return live

    def _record_width_park_prior(self, *, reengaged):
        """Remember, per cohort width, that the tree lost to plain decode.

        When every live session at width two or more is parked (and not
        probing), speculation lost at that width; a session that re-engages
        at a width clears it.  The record is token-free timing local to this
        generator binding, like the B1 ``_ast_cost_prior``.
        """
        live = self._live_reversible_lanes()
        if not live or len(live) < 2:
            return
        priors = self.__dict__.setdefault("_width_park_priors", {})
        width = len(live)
        if reengaged:
            if priors.pop(width, None) is not None:
                _bump(self.scheduler_stats, "external_cost_width_prior_clears")
            return
        states = [lane.cost_handoff_state for lane in live]
        if not all(state.speculation_paused for state in states):
            return
        backoff = max(state.reprobe_after_tokens for state in states)
        if width in priors:
            # Failed cohort re-probes at this width carry their backoff.
            priors[width]["reprobe_after"] = max(
                priors[width]["reprobe_after"], backoff
            )
            return
        baselines = [state.ordinary_baseline() for state in states]
        speculative = [state.last_park_speculative_cost for state in states]
        baselines = [value for value in baselines if value is not None]
        speculative = [value for value in speculative if value is not None]
        if not baselines or not speculative:
            return
        _bump(self.scheduler_stats, "external_cost_width_prior_records")
        priors[width] = {
            "ordinary_ns_per_committed_token": sum(baselines) / len(baselines),
            "speculative_ns_per_committed_token": max(speculative),
            "reprobe_after": backoff,
        }

    def _apply_width_park_prior(self, active_width):
        """Park new or speculating sessions at a width where the tree lost.

        A cohort that already learned (this generator binding) that the tree
        loses to plain decode at this width does not re-measure it with every
        new set of requests: their sessions start parked, build their own
        ordinary baselines from the pipelined plain steps, and re-probe with
        the cohort after the re-probe backoff the earlier cohorts at this
        width reached (failed re-probes double it, as for one session).
        Sessions already parked or probing keep their state; a re-engagement
        at the width clears the record.
        """
        priors = getattr(self, "_width_park_priors", None)
        if not priors or active_width < 2:
            return
        prior = priors.get(active_width)
        live = self._live_reversible_lanes()
        if prior is None or not live or len(live) != active_width:
            return
        for lane in live:
            state = lane.cost_handoff_state
            if state.parked:
                continue
            decision = state.adopt_cohort_decision(
                park=True,
                ordinary_cost=prior["ordinary_ns_per_committed_token"],
                speculative_cost=prior["speculative_ns_per_committed_token"],
                reprobe_after=prior["reprobe_after"],
                reason="width_prior",
            )
            if decision is None:
                continue
            _bump(self.scheduler_stats, "external_cost_park_events")
            _bump(self.scheduler_stats, "external_cost_width_prior_parks")
            self._publish_cost_receipt(lane)

    def _align_cohort_reprobes(self):
        """Re-probe every parked peer with the session whose re-probe fired.

        At width two or more a lone re-probing session splits the decode
        phase into its one-lane speculative round and an unpipelined plain
        step of its parked peers, and its probe is timed at a width the
        cohort never runs.  Aligned, the peers re-probe in the same rounds:
        the probe runs at the cohort's own width, each session decides from
        its own probe against its own ordinary baseline, and the parked
        cohort leaves and re-forms its batched plain pipeline once.
        """
        live = self._live_reversible_lanes()
        if not live or len(live) < 2:
            return
        for lane in live:
            decision = lane.cost_handoff_state.begin_cohort_reprobe()
            if decision is None:
                continue
            _bump(self.scheduler_stats, "external_cost_reprobe_events")
            _bump(self.scheduler_stats, "external_cost_cohort_reprobe_events")
            self._publish_cost_receipt(lane)

    def _apply_ast_cost_prior_scope(self, cohort, *, execution_width):
        """Keep a measured slow-route hint inside its B1 execution scope."""
        if not getattr(self, "ast_cost_self_configure", False):
            return
        if execution_width != 1:
            for lane in cohort:
                state = lane.cost_handoff_state
                if state is None:
                    continue
                if not state.revoke_session_prior_for_width_growth(
                    execution_width
                ):
                    continue
                lane.cost_handoff_receipt = state.receipt()
                _bump(
                    self.scheduler_stats,
                    "external_ast_cost_prior_width_revocations",
                )
            return
        if getattr(self, "_ast_cost_prior", None) is None or len(cohort) != 1:
            return
        state = cohort[0].cost_handoff_state
        if state is not None and state.session_prior_eligible:
            state.seed_reversible_park(self._ast_cost_prior)

    def _update_ast_cost_prior(self, decision, *, execution_width):
        """Retain token-free B1 slow-route feedback for this generator binding."""
        if not getattr(self, "ast_cost_self_configure", False) or execution_width != 1:
            return
        event = decision.get("event")
        if event == "reengage":
            self._ast_cost_prior = None
            return
        if event not in {"park", "repark"}:
            return
        self._ast_cost_prior = {
            name: float(decision[name])
            for name in (
                "ordinary_ns_per_committed_token",
                "speculative_ns_per_committed_token",
                "speculative_to_ordinary_cost_ratio",
            )
        }

    def _pool_round(self, cohort):
        from .continuation_verification import (
            SelectedContinuationTransaction,
            prepare_continuations,
            sample_continuations,
            verify_longest_first_continuations,
        )
        from .proposal_providers import LONGEST_FIRST_EXACT_PREFIX
        recovery = self._snapshot_round(cohort)
        self.scheduler_stats["recovery_checkpoint_captures"] += len(recovery)
        stats_snapshot = self._snapshot_scheduler_stats()
        self._open = True
        transaction = None
        try:
            blocks = self._propose(cohort)
            for lane, original_block in zip(cohort, blocks):
                selection = original_block.continuation_selection
                selection.require_verification_contract("target_draw_then_prefix_match")
                maximum_width = len(selection.paths)
                maximum_depth = max(len(path.tokens) for path in selection.paths)
                dedup_enabled = (
                    getattr(self.model, "supports_contextual_prefix_equivalence", False) is True
                    and getattr(getattr(self.model, "model", None), "residual_taps", None) is None
                )
                physical_widths = {}
                for semantic_width in range(1, maximum_width + 1):
                    for candidate_depth in range(1, maximum_depth + 1):
                        candidates = tuple(path.tokens[:candidate_depth]
                            for path in selection.paths[:semantic_width])
                        physical_widths[semantic_width, candidate_depth] = (
                            len(set(candidates)) if dedup_enabled else semantic_width)
                width, depth = maximum_width, maximum_depth
                if self.adaptive_policy is not None:
                    width, depth = self.adaptive_policy.choose_continuation_shape(
                        maximum_width, maximum_depth, getattr(lane, "continuation_coverage", {}),
                        getattr(lane, "continuation_rounds", 0), [len(path.tokens) for path in selection.paths],
                        physical_widths=physical_widths)
                semantic_paths = tuple(path.tokens[:depth] for path in selection.paths[:width])
                semantic_order = tuple(range(len(semantic_paths)))
                if (
                    self.continuation_strategy.algorithm
                    == "longest_first_exact_prefix_v1"
                ):
                    semantic_order = tuple(
                        index
                        for index, _path in sorted(
                            enumerate(semantic_paths),
                            key=lambda item: (-len(item[1]), item[0]),
                        )
                    )
                    semantic_paths = tuple(
                        semantic_paths[index] for index in semantic_order
                    )
                paths, representatives, semantic_to_physical = self._unique_continuation_paths(
                    semantic_paths, enabled=dedup_enabled)
                laws = []
                response_rows = [] if lane.sampling.get("emit_logprobs", True) else None
                window = VerifyWindow(lane.processors)

                def sample_row(logits_row, prefix, *, _lane=lane,
                               _response_rows=response_rows, _window=window, _laws=laws):
                    law = self._target_law(
                        _lane, logits_row, _lane.history, True, _response_rows,
                        history_suffix=[_lane.anchor, *prefix],
                    )
                    _window.mark()
                    _laws.append(law)
                    return _lane.rng.sample(law)
                strategy_longest_first = (
                    self.continuation_strategy.algorithm
                    == LONGEST_FIRST_EXACT_PREFIX
                )
                policy_longest_first = (
                    self.continuation_policy.verification_algorithm
                    == LONGEST_FIRST_EXACT_PREFIX
                )
                if (strategy_longest_first or policy_longest_first) and not dedup_enabled:
                    raise RuntimeError(
                        "longest-first exact-prefix verification requires "
                        "adapter-attested contextual prefix equivalence"
                    )

                if strategy_longest_first:
                    inputs = [[lane.anchor, *path] for path in paths]
                    counts = [len(path) for path in paths]
                    taps, steer, commits = self._verify_steer(
                        [lane] * len(paths), inputs, counts
                    )
                    if steer is not None or commits:
                        raise ValueError(
                            "longest-first continuation reuse is unavailable "
                            "with residual steering"
                        )
                    from .continuation_verification import (
                        prepare_longest_first_continuations,
                    )

                    outcome, hidden, transaction = prepare_longest_first_continuations(
                        self.model,
                        self.mx,
                        lane.cache,
                        lane.anchor,
                        paths,
                        self.layers,
                        self._target_owner,
                        sample_row,
                        maximum=lane.maximum - lane.generated,
                        stop_tokens=self.stops,
                        max_sequences=self.continuation_policy.limit,
                        max_depth=self.num_draft,
                    )
                    commit_features = hidden
                    continuation_transaction = transaction
                    commits = ()
                    lane.continuation_verification_algorithm = LONGEST_FIRST_EXACT_PREFIX
                    lane.continuation_cascade_attempts = getattr(
                        lane, "continuation_cascade_attempts", 0
                    ) + outcome.launches
                    lane.continuation_shared_prefix_tokens_reused = getattr(
                        lane, "continuation_shared_prefix_tokens_reused", 0
                    ) + outcome.shared_prefix_reused_tokens
                elif policy_longest_first:
                    def prepare_attempt(cache, anchor, suffix, *, _lane=lane):
                        inputs = [[anchor, *suffix]]
                        taps, steer, steer_commits = self._verify_steer(
                            [_lane], inputs, [len(suffix)]
                        )
                        if steer is not None:
                            taps.steer = steer
                        try:
                            _paths, logits, hidden, attempt = prepare_continuations(
                                self.model,
                                self.mx,
                                cache,
                                anchor,
                                (suffix,),
                                self.layers,
                                self._target_owner,
                                max_sequences=1,
                                max_depth=self.num_draft,
                            )
                        finally:
                            if steer is not None:
                                taps.steer = None
                        settle = steer_commits[0] if steer_commits else None
                        return logits, hidden, attempt, settle

                    outcome, hidden_slices, transaction = (
                        verify_longest_first_continuations(
                            paths,
                            lane.cache,
                            lane.anchor,
                            prepare_attempt,
                            sample_row,
                            maximum=lane.maximum - lane.generated,
                            stop_tokens=self.stops,
                        )
                    )
                    hidden = self.mx.concatenate(hidden_slices, axis=1)
                    commit_features = hidden
                    commits = ()
                    continuation_transaction = transaction
                    lane.continuation_verification_algorithm = LONGEST_FIRST_EXACT_PREFIX
                    lane.continuation_cascade_attempts = getattr(
                        lane, "continuation_cascade_attempts", 0
                    ) + len(outcome.attempted_indices)
                    lane.continuation_shared_prefix_tokens_reused = getattr(
                        lane, "continuation_shared_prefix_tokens_reused", 0
                    ) + outcome.shared_prefix_tokens_reused
                else:
                    inputs = [[lane.anchor, *path] for path in paths]
                    counts = [len(path) for path in paths]
                    taps, steer, commits = self._verify_steer(
                        [lane] * len(paths), inputs, counts
                    )
                    if steer is not None:
                        taps.steer = steer
                    try:
                        paths, logits, hidden, transaction = prepare_continuations(
                            self.model, self.mx, lane.cache, lane.anchor, paths, self.layers,
                            self._target_owner, max_sequences=self.continuation_policy.limit,
                            max_depth=self.num_draft)
                    finally:
                        if steer is not None:
                            taps.steer = None
                    outcome = sample_continuations(paths, logits, sample_row,
                        maximum=lane.maximum - lane.generated, stop_tokens=self.stops)
                    commit_features = hidden[
                        outcome.selected_index : outcome.selected_index + 1
                    ]
                    continuation_transaction = SelectedContinuationTransaction(
                        transaction, outcome.selected_index, len(paths)
                    )
                    lane.continuation_verification_algorithm = (
                        self.continuation_policy.verification_algorithm
                    )
                physical_index = outcome.selected_index
                semantic_index = semantic_order[representatives[physical_index]]
                physical_to_semantic = tuple(
                    semantic_order[index] for index in representatives
                )
                semantic_to_physical_original = [None] * len(semantic_order)
                for reordered_index, physical in enumerate(semantic_to_physical):
                    semantic_to_physical_original[
                        semantic_order[reordered_index]
                    ] = physical
                chosen = selection.paths[semantic_index]
                source = chosen.representative.source_id
                self._record_continuation_round(
                    lane,
                    original_block,
                    selection,
                    admitted_width=width,
                    selected_index=semantic_index,
                    proposed_depth=depth,
                    accepted=outcome.accepted,
                )
                selected = HostDraftRow(list(paths[outcome.selected_index]), [], original_block.width,
                    proposal_source=source if source in {"prompt_lookup", "native_mtp"} else "external",
                    continuation_selection=selection,
                    continuation_admission=original_block.continuation_admission,
                    continuation_slate=original_block.continuation_slate,
                    feedback_payload=original_block.feedback_payload)
                decision = RoundDecision(outcome.accepted, list(outcome.emitted), laws,
                    response_logprobs=response_rows, verify_window=window, continuation_outcome=outcome,
                    continuation_semantic_index=semantic_index,
                    **(
                        {
                            "commit_count": len(outcome.emitted),
                            "committed_inputs": [lane.anchor, *outcome.emitted[:-1]],
                        }
                        if outcome.algorithm == "longest_first_exact_prefix_v1"
                        else {}
                    ))
                lane.continuation_semantic_selected = maximum_width
                lane.continuation_semantic_admitted = width
                lane.continuation_physical_to_semantic = physical_to_semantic
                lane.continuation_semantic_to_physical = tuple(
                    semantic_to_physical_original
                )
                lane.continuation_prefix_dedup_enabled = dedup_enabled
                lane.continuation_deduplicated_sequences = getattr(lane, "continuation_deduplicated_sequences", 0) + width - len(paths)
                if self.adaptive_policy is not None:
                    saved = (
                        maximum_width * (maximum_depth + 1)
                        - outcome.executed_target_rows
                    )
                    lane.adaptive_round_policy = ("continuation_lagged_coverage_cost_model" if saved
                        else "continuation_fixed_depth_cost_backstop")
                    lane.adaptive_feature_source = "pool_lagged_source_labels"
                    _bump(self.scheduler_stats, "external_adaptive_rounds")
                    _bump(self.scheduler_stats, "external_adaptive_trimmed_rounds", int(saved > 0))
                    _bump(self.scheduler_stats, "external_adaptive_trimmed_target_rows", saved)
                    if saved:
                        lane.adaptive_trimmed_rounds = getattr(lane, "adaptive_trimmed_rounds", 0) + 1
                    _bump(
                        self.scheduler_stats,
                        "external_adaptive_target_rows",
                        outcome.executed_target_rows,
                    )
                    _bump(self.scheduler_stats, "external_adaptive_verification_groups")
                    self.scheduler_stats["external_adaptive_round_depth"] = outcome.physical_span - 1
                    self.scheduler_stats["external_adaptive_verify_width"] = outcome.physical_span
                self._commit([lane], [decision], commit_features,
                    blocks=[selected], transaction=continuation_transaction)
                if commits:
                    commits[physical_index](min(outcome.accepted + 1, len(outcome.emitted)))
                _bump(self.scheduler_stats, "external_continuation_rounds")
                _bump(
                    self.scheduler_stats,
                    "external_continuation_sequences",
                    outcome.launches,
                )
                _bump(self.scheduler_stats, "external_continuation_ranked_complete_sequences", maximum_width)
                _bump(self.scheduler_stats, "external_continuation_admitted_complete_sequences", width)
                _bump(self.scheduler_stats, "external_continuation_deduplicated_sequences", width - len(paths))
                _bump(
                    self.scheduler_stats,
                    "external_continuation_target_rows",
                    outcome.executed_target_rows,
                )
                if outcome.algorithm == "longest_first_exact_prefix_v1":
                    _bump(
                        self.scheduler_stats,
                        "external_continuation_longest_first_attempts",
                        outcome.launches,
                    )
                    _bump(
                        self.scheduler_stats,
                        "external_continuation_prefix_pruned",
                        outcome.pruned_siblings,
                    )
                self.scheduler_stats["target_max_width"] = max(self.scheduler_stats["target_max_width"], outcome.physical_width)
        except BaseException:
            if transaction is not None and not transaction.closed:
                try:
                    transaction.abort()
                except BaseException:  # noqa: BLE001, S110 - original boundary is authoritative
                    pass
            self._restore_round(cohort, recovery)
            self.scheduler_stats = stats_snapshot
            self.scheduler_stats["recovery_checkpoint_restores"] += len(recovery)
            raise
        finally:
            self._open = False

    @staticmethod
    def _unique_continuation_paths(paths, *, enabled):
        """Stable physical rows; each uses its first ranked semantic representative."""
        unique, representatives, inverse, seen = [], [], [], {}
        for semantic, path in enumerate(paths):
            key = tuple(path)
            physical = seen.get(key) if enabled else None
            if physical is None:
                physical = len(unique)
                seen[key] = physical
                unique.append(key)
                representatives.append(semantic)
            inverse.append(physical)
        return tuple(unique), tuple(representatives), tuple(inverse)

    def _run_round(self, cohort, *, forced_depth=None):
        if (getattr(self, "ast_b1_tile_policy", None) == "four_path15"
            or getattr(self, "ast_packed_pattern", None)):
            if len(cohort) != 1:
                raise ValueError("four_path15 admits only one active request")
            if forced_depth == 0 or cohort[0].maximum - cohort[0].generated <= 1:
                _bump(self.scheduler_stats, "external_ast_b1_explicit_ordinary_terminal_rounds")
                return self._ordinary_round(cohort)
        if forced_depth not in (None, 0):
            raise ValueError("external forced draft depth must be zero or absent")
        if forced_depth is None and cohort and all(lane.ordinary for lane in cohort):
            inflight = self._ordinary_inflight
            if inflight is not None or self._ordinary_pipeline_eligible(cohort):
                return self._ordinary_pipeline_round(cohort)
            return self._ordinary_round(cohort)
        if (
            forced_depth is None
            and self.continuation_policy is not None
            and not getattr(self, "continuation_pool_ast_bridge", False)
            and all(
                not lane.ordinary and lane.maximum - lane.generated > 1
                for lane in cohort
            )
        ):
            return self._pool_round(cohort)
        if forced_depth is None and self.draft_topology == "tree15" and all(
            lane.maximum - lane.generated > 1
            # A parked or ordinary lane drafts nothing this round; a mixed
            # cohort takes the chain body, which handles zero-count lanes.
            and not lane.ordinary and not _speculation_parked(lane)
            for lane in cohort
        ):
            # A lane's last budget token takes the chain body's zero-count
            # round, which appends the draft context and publishes the
            # sidecar; the ordinary path reported ordinary_fallback for it.
            return self._tree_round(cohort)
        if (self.adaptive_policy is not None and self.adaptive_policy.mode == "per_request"
                and self.adaptive_policy.costs(1) is not None
                and self.adaptive_policy.costs(len(cohort)) is not None):
            return self._request_adaptive_round(cohort)
        clock = self._phase_time() if self.round_timing else None
        recovery = self._snapshot_round(cohort)
        self.scheduler_stats["recovery_checkpoint_captures"] += len(recovery)
        if clock is not None:
            clock = self._mark("recovery_capture", clock)
        stats_snapshot = self._snapshot_scheduler_stats()
        adaptive_snapshot = (
            copy.deepcopy(self.acceptance_estimator) if self.adaptive_policy is not None else None
        )
        self._open = True
        transaction = None
        try:
            deferred_context_uids = {
                lane.uid for lane in cohort
                if self._defer_ast_parked_context(lane)
            }
            if forced_depth is not None:
                blocks = self._propose(cohort, force_zero=True)
                fixed_depth = forced_depth
            elif self.adaptive_policy is None:
                blocks = self._propose(cohort)
                fixed_depth = None
            else:
                fixed_depth = min(self.num_draft, min(l.maximum - l.generated - 1 for l in cohort))
                if any(l.ordinary for l in cohort):
                    fixed_depth = 0
                depth = self.adaptive_policy.choose_depth(
                    self.acceptance_estimator, fixed_depth, len(cohort)
                )
                blocks = self._propose(cohort, adaptive_depth=depth)
            if clock is not None:
                clock = self._mark("draft", clock)
            proposal_counts = [0 if block is None else int(block.lengths[0]) for block in blocks]
            verify_width = max(proposal_counts, default=0) + 1
            if (
                getattr(self, "ast_execution_planner", None) is not None
                and any(proposal_counts)
                and any(block is None for block in blocks)
            ):
                # A parked/final-budget row still owns one ordinary anchor
                # law. It can share the active rows' exact ragged transaction.
                blocks = [
                    HostDraftRow([], [], 1) if block is None else block
                    for block in blocks
                ]
            for block in blocks:
                if (
                    bool(getattr(block, "proposal_trusted", False))
                    and any(int(token) in self.stops for token in block.tokens)
                ):
                    # Blindly copying a role/end marker can terminate a chat
                    # with a previous assistant turn. Keep the candidate, but
                    # obtain an exact committed label for this round.
                    block.proposal_trusted = False
                    block.proposal_audit = True
                    _bump(
                        self.scheduler_stats,
                        "external_composed_trusted_pld_stop_guard_rounds",
                    )
            trusted_round = any(
                bool(getattr(block, "proposal_trusted", False))
                for block in blocks
                if block is not None
            )
            if trusted_round and (
                len(cohort) != 1
                or len(blocks) != 1
                or self.target_execution != "reference"
                or not bool(getattr(self.model, "supports_trusted_pld", False))
                or self.fly_verification.enabled
            ):
                # Trusted PLD needs a supported B1 reference target. A row
                # that reaches any other round is verified exactly instead:
                # raising here escaped next() and stopped the serving worker.
                for block in blocks:
                    if block is not None and bool(
                        getattr(block, "proposal_trusted", False)
                    ):
                        block.proposal_trusted = False
                        block.proposal_audit = True
                _bump(
                    self.scheduler_stats,
                    "external_composed_trusted_pld_b1_guard_rounds",
                )
                trusted_round = False
            progressive_proposal_span = max(
                [*proposal_counts]
                + [
                    len(alternative.tokens)
                    for lane in cohort
                    for alternative in self._last_alternative_blocks.get(
                        lane.uid, ()
                    )
                ],
                default=0,
            )
            if (
                self.progressive_verification_tile is not None
                and 1 <= len(cohort) == len(blocks) <= 4
                and all(block is not None for block in blocks)
                and (
                    progressive_proposal_span > self.progressive_verification_tile
                    or getattr(self, "ast_execution_planner", None) is not None
                    or self.progressive_verification_tile_policy == "auto"
                    and self.progressive_verification_tile == self.num_draft
                    and progressive_proposal_span
                    == self.progressive_verification_tile
                )
                and all(
                    self._progressive_processors_supported(lane)
                    for lane in cohort
                )
                and (getattr(self, "ast_b1_tile_policy", None) == "four_path15"
                     or getattr(self, "ast_packed_pattern", None)
                     or len(cohort) > 1 and self._ast_tree_cohort_limit() >= len(cohort) or not any(
                    int(token) in self.stops
                    for block in blocks
                    for token in block.tokens
                ))
            ):
                cross_session_ast_available = bool(
                    not getattr(self, "ast_tree_tile_rows", 0)
                    and self.progressive_cross_session_pack_rows is not None
                    and len(cohort) > 1
                    and all(
                        lane.uid in self._last_alternative_blocks
                        for lane in cohort
                    )
                    and any(
                        len(self._last_alternative_blocks[lane.uid]) > 1
                        for lane in cohort
                    )
                )
                cross_session_ast_stop_guard = bool(
                    cross_session_ast_available
                    and any(
                        int(token) in self.stops
                        for lane in cohort
                        for block in self._last_alternative_blocks[lane.uid]
                        for token in block.tokens
                    )
                )
                # The ordered live frontier accepts 1-4 branches per session,
                # each with a positive suffix past its shared prefix.  A
                # processor-masked draft can end a branch at the prefix (a
                # forbidden token left no finite candidate), so such a round
                # keeps the exact primary progressive path instead.
                cross_session_ast_geometry_guard = bool(
                    cross_session_ast_available
                    and not cross_session_ast_stop_guard
                    and self.progressive_cross_session_pack_mode
                    == "ordered_live_frontier"
                    and any(
                        not 1 <= len(self._last_alternative_blocks[lane.uid]) <= 4
                        or any(
                            len(block.tokens)
                            - int(self._last_alternative_prefix_tokens.get(lane.uid, 0))
                            < 1
                            for block in self._last_alternative_blocks[lane.uid]
                        )
                        for lane in cohort
                    )
                )
                if cross_session_ast_geometry_guard:
                    _bump(
                        self.scheduler_stats,
                        "external_progressive_ast_geometry_guard_rounds",
                        len(cohort),
                    )
                cross_session_ast_mode = (
                    cross_session_ast_available
                    and not cross_session_ast_stop_guard
                    and not cross_session_ast_geometry_guard
                )
                if cross_session_ast_stop_guard:
                    _bump(
                        self.scheduler_stats,
                        "external_progressive_alternative_stop_guard_rounds",
                        len(cohort),
                    )
                    for lane in cohort:
                        lane.progressive_alternative_stop_guard_rounds = (
                            getattr(
                                lane,
                                "progressive_alternative_stop_guard_rounds",
                                0,
                            )
                            + 1
                        )
                alternative_mode = (
                    not cross_session_ast_mode
                    and len(cohort) == 1
                    and self.progressive_alternative_branches > 1
                    and cohort[0].uid in self._last_alternative_blocks
                )
                verify_cohort = cohort
                verify_blocks = blocks
                if cross_session_ast_mode:
                    block_groups = [
                        self._last_alternative_blocks[lane.uid] for lane in cohort
                    ]
                    prefix_counts = [
                        self._last_alternative_prefix_tokens[lane.uid]
                        for lane in cohort
                    ]
                    (
                        decisions,
                        features,
                        committed_caches,
                        committed_rngs,
                        progressive_stats,
                        physical_stats,
                        cross_session_selections,
                        selected_blocks,
                    ) = self._progressive_verify_cross_session_ast(
                        cohort,
                        block_groups,
                        prefix_counts,
                        pack_rows=self.progressive_cross_session_pack_rows,
                    )
                    self._record_ast_topology_execution(cohort)
                    blocks = selected_blocks
                    topology_rows = []
                    for lane, selection, lane_blocks, decision in zip(
                        cohort,
                        cross_session_selections,
                        block_groups,
                        decisions,
                        strict=True,
                    ):
                        branch_accepts = selection["branch_accepts"]
                        proposal_counts = selection["proposal_counts"]
                        full = selection["full"]
                        winner = selection["winner"]
                        lane.progressive_alternative_rounds = (
                            getattr(lane, "progressive_alternative_rounds", 0) + 1
                        )
                        lane.progressive_alternative_last_branch_accepts = list(
                            branch_accepts
                        )
                        lane.progressive_alternative_last_active_branches = len(
                            lane_blocks
                        )
                        lane.progressive_alternative_last_branch_proposal_counts = list(
                            proposal_counts
                        )
                        lane.progressive_alternative_last_branch_outcomes = list(
                            ordered_branch_outcomes(branch_accepts, proposal_counts)
                        )
                        lane.progressive_alternative_last_branch_hashes = [
                            hashlib.sha256(
                                json.dumps(
                                    [int(token) for token in block.tokens],
                                    separators=(",", ":"),
                                ).encode()
                            ).hexdigest()
                            for block in lane_blocks
                        ]
                        lane.progressive_alternative_last_full_branches = [
                            index + 1 for index in full
                        ]
                        lane.progressive_alternative_last_selected_branch = winner + 1
                        lane.progressive_alternative_last_branch_committed_tokens = list(
                            selection["branch_committed_tokens"]
                        )
                        lane.progressive_alternative_last_selected_token_uplift = int(
                            selection["selected_token_uplift"]
                        )
                        lane.progressive_alternative_last_discarded_branches = [
                            index + 1
                            for index in range(len(lane_blocks))
                            if index != winner
                        ]
                        lane.progressive_alternative_last_short_circuit = None
                        lane.progressive_alternative_last_physical_width = (
                            physical_stats["max_cohort_width"]
                        )
                        lane.progressive_alternative_last_strategy = (
                            self.progressive_alternative_strategy
                        )
                        lane.progressive_alternative_last_shared_prefix_tokens = int(
                            self._last_alternative_prefix_tokens[lane.uid]
                        )
                        lane.progressive_alternative_last_unique_target_rows = int(
                            selection["unique_target_rows"]
                        )
                        lane.progressive_cross_session_ast_last_pack_rows = int(
                            self.progressive_cross_session_pack_rows
                        )
                        lane.progressive_cross_session_ast_last_packs = list(
                            physical_stats["cross_session_packs"]
                        )
                        lane.progressive_cross_session_ast_last_mode = physical_stats[
                            "cross_session_ast_mode"
                        ]
                        lane.progressive_cross_session_ast_last_frontier = copy.deepcopy(
                            physical_stats.get("live_frontier")
                        )
                        if self.progressive_topology_latch is not None:
                            observation = {
                                "branch_accepts": list(branch_accepts),
                                "proposal_counts": list(proposal_counts),
                                "selected_branch": winner + 1,
                                "unique_target_rows": selection[
                                    "unique_target_rows"
                                ],
                                **self._topology_observation_outcome(
                                    lane,
                                    decision,
                                    branch_accepts,
                                    proposal_counts,
                                    winner + 1,
                                ),
                            }
                            topology_rows.append(
                                (int(lane.uid), copy.deepcopy(observation))
                            )
                    for uid, observation in topology_rows:
                        # Sessions own their decision clocks; a session that
                        # executed only its primary chain is recorded by the
                        # primary-only path after commit.
                        payload = self._last_alternative_latch.get(uid)
                        if payload is None or len(observation["branch_accepts"]) == 1:
                            continue
                        self._topology_observation_outbox.append(
                            (int(payload["decision_token"]), [(uid, observation)])
                        )
                elif alternative_mode:
                    lane = cohort[0]
                    if float(lane.sampling.get("sampling_temp", 0)) != 0.0:
                        raise LaneFailure(
                            lane.uid,
                            "ordered alternative verification is greedy-only",
                        )
                    verify_blocks = self._last_alternative_blocks.get(lane.uid)
                    if verify_blocks is None:
                        raise LaneFailure(
                            lane.uid,
                            "ordered alternative proposal rows are unavailable",
                        )
                    if (getattr(self, "ast_b1_tile_policy", None) != "four_path15"
                        and not getattr(self, "ast_packed_pattern", None)) and any(
                        int(token) in self.stops
                        for block in verify_blocks
                        for token in block.tokens
                    ):
                        # The primary chain already passed the stop-token
                        # gate above. Alternative rows must not bypass it;
                        # use the primary progressive chain for this round.
                        alternative_mode = False
                        verify_blocks = blocks
                        _bump(
                            self.scheduler_stats,
                            "external_progressive_alternative_stop_guard_rounds",
                        )
                        lane.progressive_alternative_stop_guard_rounds = (
                            getattr(
                                lane,
                                "progressive_alternative_stop_guard_rounds",
                                0,
                            )
                            + 1
                        )
                    else:
                        verify_cohort = [
                            copy.copy(lane) for _ in range(len(verify_blocks))
                        ]
                if cross_session_ast_mode:
                    pass
                elif (
                    len(cohort) > 1
                    and self._ast_tree_cohort_limit() >= len(cohort)
                    and all(
                        block.tokens and float(lane.sampling.get("sampling_temp", 0)) == 0.0
                        and self._batched_law_contract(lane)
                        and len(self._last_alternative_blocks.get(lane.uid, (block,))) == 1
                        for lane, block in zip(cohort, blocks, strict=True)
                    )
                ):
                    (
                        decisions, features, committed_caches, committed_rngs,
                        progressive_stats, physical_stats,
                    ) = self._ast_verify_primary_tree_cohort(cohort, blocks)
                elif (
                    getattr(self, "ast_tree_tile_rows", 0)
                    and len(cohort) == 1
                    and float(cohort[0].sampling.get("sampling_temp", 0)) == 0.0
                    and self._progressive_processors_supported(cohort[0])
                    and self._batched_law_contract(cohort[0])
                ):
                    from .ast_tree_verify import verify_tree_tiles

                    (
                        decisions,
                        features,
                        committed_caches,
                        committed_rngs,
                        progressive_stats,
                        physical_stats,
                    ) = verify_tree_tiles(self, cohort[0], verify_blocks)
                elif getattr(self, "ast_tree_tile_rows", 0):
                    raise RuntimeError(
                        "AST tree tiles have no admitted target-tree dispatch "
                        "for this request; legacy verification is not a fallback"
                    )
                elif (
                    alternative_mode
                    and self.progressive_alternative_strategy
                    in {
                        "shared_prefix",
                        "paired_continuations",
                        "confidence_tree",
                    }
                ):
                    (
                        decisions,
                        features,
                        committed_caches,
                        committed_rngs,
                        progressive_stats,
                        physical_stats,
                    ) = self._progressive_verify_shared_prefix(
                        cohort[0],
                        verify_blocks,
                        prefix_count=self._last_alternative_prefix_tokens.get(
                            cohort[0].uid
                        ),
                    )
                else:
                    (
                        decisions,
                        features,
                        committed_caches,
                        committed_rngs,
                        progressive_stats,
                        physical_stats,
                    ) = self._progressive_verify_many(
                        verify_cohort, verify_blocks
                    )
                if alternative_mode:
                    self._record_ast_topology_execution(cohort)
                    proposal_counts = tuple(
                        int(block.lengths[0]) for block in verify_blocks
                    )
                    winner, full = ordered_alternative_winner(
                        decisions,
                        proposal_counts,
                        allow_partial_uplift=bool(
                            getattr(self, "continuation_pool_ast_bridge", False)
                        ),
                    )
                    committed_tokens = ordered_branch_commit_counts(decisions)
                    branch_accepts = [
                        int(decision.accepted) for decision in decisions
                    ]
                    branch_outcomes = list(
                        ordered_branch_outcomes(branch_accepts, proposal_counts)
                    )
                    branch_hashes = [
                        hashlib.sha256(
                            json.dumps(
                                [int(token) for token in block.tokens],
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest()
                        for block in verify_blocks
                    ]
                    lane = cohort[0]
                    lane.progressive_alternative_rounds = (
                        getattr(lane, "progressive_alternative_rounds", 0) + 1
                    )
                    lane.progressive_alternative_last_active_branches = len(
                        verify_blocks
                    )
                    lane.progressive_alternative_last_branch_accepts = branch_accepts
                    lane.progressive_alternative_last_branch_proposal_counts = list(
                        proposal_counts
                    )
                    lane.progressive_alternative_last_branch_outcomes = (
                        branch_outcomes
                    )
                    lane.progressive_alternative_last_branch_hashes = branch_hashes
                    lane.progressive_alternative_last_full_branches = [
                        index + 1 for index in full
                    ]
                    lane.progressive_alternative_last_selected_branch = winner + 1
                    lane.progressive_alternative_last_branch_committed_tokens = list(
                        committed_tokens
                    )
                    lane.progressive_alternative_last_selected_token_uplift = max(
                        0,
                        int(committed_tokens[winner]) - int(committed_tokens[0]),
                    )
                    lane.progressive_alternative_last_discarded_branches = [
                        index + 1
                        for index in range(len(decisions))
                        if index != winner
                    ]
                    lane.progressive_alternative_last_short_circuit = (
                        physical_stats.get("short_circuit_after_branch")
                        if physical_stats.get("short_circuit_after_branch") is not None
                        else (full[0] + 1 if full else None)
                    )
                    lane.progressive_alternative_last_physical_width = (
                        physical_stats["max_cohort_width"]
                    )
                    lane.progressive_alternative_last_strategy = (
                        self.progressive_alternative_strategy
                    )
                    lane.progressive_alternative_last_shared_prefix_tokens = int(
                        physical_stats.get("shared_prefix_tokens", 0)
                    )
                    lane.progressive_alternative_last_unique_target_rows = int(
                        physical_stats.get(
                            "unique_target_rows",
                            physical_stats["physical_target_rows"],
                        )
                    )
                    self._record_ast_geometry(
                        lane, verify_blocks, branch_accepts, proposal_counts,
                        committed_tokens, winner, physical_stats,
                    )
                    if self.progressive_topology_latch is not None:
                        observation = {
                            "branch_accepts": list(branch_accepts),
                            "proposal_counts": list(proposal_counts),
                            "selected_branch": winner + 1,
                            "unique_target_rows": (
                                lane.progressive_alternative_last_unique_target_rows
                            ),
                            **self._topology_observation_outcome(
                                lane,
                                decisions[winner],
                                branch_accepts,
                                proposal_counts,
                                winner + 1,
                            ),
                        }
                        self._topology_observation_outbox.append(
                            (
                                self._last_alternative_latch[lane.uid][
                                    "decision_token"
                                ],
                                [(int(lane.uid), copy.deepcopy(observation))],
                            )
                        )
                    lane.progressive_alternative_b2_short_circuits = (
                        getattr(
                            lane,
                            "progressive_alternative_b2_short_circuits",
                            0,
                        )
                        + int(bool(full) and full[0] == 1)
                    )
                    lane.target_max_width = max(
                        lane.target_max_width, physical_stats["max_cohort_width"]
                    )
                    selected_stats = dict(progressive_stats[winner])
                    selected_stats["launches"] = physical_stats[
                        "physical_launches"
                    ]
                    selected_stats["target_rows"] = int(
                        physical_stats["physical_target_rows"]
                    )
                    selected_stats["rejected"] = sum(
                        int(stats["rejected"]) for stats in progressive_stats
                    )
                    selected_stats["full_tiles"] = sum(
                        int(stats["full_tiles"]) for stats in progressive_stats
                    )
                    decisions = [decisions[winner]]
                    features = features[winner : winner + 1]
                    committed_caches = [committed_caches[winner]]
                    committed_rngs = [committed_rngs[winner]]
                    progressive_stats = [selected_stats]
                    blocks = [verify_blocks[winner]]
                for lane, block, lane_stats in zip(
                    cohort, blocks, progressive_stats, strict=True
                ):
                    lane.progressive_verify_rounds = (
                        getattr(lane, "progressive_verify_rounds", 0) + 1
                    )
                    lane.progressive_verify_launches = (
                        getattr(lane, "progressive_verify_launches", 0)
                        + lane_stats["launches"]
                    )
                    lane.progressive_verify_target_rows = (
                        getattr(lane, "progressive_verify_target_rows", 0)
                        + lane_stats["target_rows"]
                    )
                    lane.progressive_verify_padding_rows = (
                        getattr(lane, "progressive_verify_padding_rows", 0)
                        + physical_stats.get("padding_rows", 0)
                    )
                    lane.progressive_verify_ragged_launches = (
                        getattr(lane, "progressive_verify_ragged_launches", 0)
                        + physical_stats.get("ragged_launches", 0)
                    )
                    lane.progressive_verify_rejections = (
                        getattr(lane, "progressive_verify_rejections", 0)
                        + lane_stats["rejected"]
                    )
                    lane.progressive_verify_full_tiles = (
                        getattr(lane, "progressive_verify_full_tiles", 0)
                        + lane_stats["full_tiles"]
                    )
                    lane.progressive_verify_tile_checkpoints = (
                        getattr(lane, "progressive_verify_tile_checkpoints", 0)
                        + len(lane_stats["tile_checkpoints"])
                    )
                    lane.progressive_verify_last_tile_input_rows = list(
                        lane_stats["tile_input_rows"]
                    )
                    lane.progressive_verify_last_tile_checkpoints = list(
                        lane_stats["tile_checkpoints"]
                    )
                    lane.progressive_verify_last_rollback_base_tile = int(
                        lane_stats["rollback_base_tile"]
                    )
                    lane.progressive_verify_last_cohort_width = len(cohort)
                    lane.progressive_verify_max_cohort_width = max(
                        getattr(lane, "progressive_verify_max_cohort_width", 0),
                        physical_stats["max_cohort_width"],
                    )
                    lane.progressive_verify_last_proposal_sha256 = hashlib.sha256(
                        json.dumps(
                            [int(token) for token in block.tokens],
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest()
                    lane.progressive_verify_last_ragged_layout = physical_stats.get(
                        "last_ragged_layout"
                    )
                    if lane_stats.get("ast_last_launch") is not None:
                        lane.ast_last_launch = lane_stats["ast_last_launch"]
                    lane.progressive_target_row_executor = (
                        _merge_target_executor_stats(
                            getattr(
                                lane,
                                "progressive_target_row_executor",
                                None,
                            ),
                            physical_stats.get("target_row_executor"),
                        )
                    )
                    if len(cohort) == 4 and physical_stats[
                        "first_tile_pairwise_disjoint"
                    ]:
                        lane.progressive_verify_b4_disjoint_first_tiles = (
                            getattr(
                                lane,
                                "progressive_verify_b4_disjoint_first_tiles",
                                0,
                            )
                            + 1
                        )
                    _bump(
                        self.scheduler_stats,
                        "external_progressive_verify_launches",
                        lane_stats["launches"],
                    )
                    _bump(
                        self.scheduler_stats,
                        "external_progressive_verify_target_rows",
                        lane_stats["target_rows"],
                    )
                    _bump(
                        self.scheduler_stats,
                        "external_progressive_verify_rejections",
                        lane_stats["rejected"],
                    )
                    _bump(
                        self.scheduler_stats,
                        "external_progressive_verify_full_tiles",
                        lane_stats["full_tiles"],
                    )
                    _bump(
                        self.scheduler_stats,
                        "external_progressive_verify_tile_checkpoints",
                        len(lane_stats["tile_checkpoints"]),
                    )
                _bump(
                    self.scheduler_stats,
                    "external_progressive_verify_rounds",
                    len(cohort),
                )
                _bump(
                    self.scheduler_stats,
                    "external_progressive_verify_physical_launches",
                    physical_stats["physical_launches"],
                )
                _bump(
                    self.scheduler_stats,
                    "external_progressive_verify_ragged_launches",
                    physical_stats.get("ragged_launches", 0),
                )
                _bump(
                    self.scheduler_stats,
                    "external_progressive_verify_padding_rows",
                    physical_stats.get("padding_rows", 0),
                )
                executor_stats = physical_stats.get("target_row_executor", {})
                for source, suffix in (
                    ("adapter_owned_launches", "adapter_owned_launches"),
                    ("legacy_launches", "legacy_launches"),
                    ("row_exact_launches", "row_exact_launches"),
                    ("nonexact_launches", "nonexact_launches"),
                ):
                    _bump(
                        self.scheduler_stats,
                        f"external_progressive_target_executor_{suffix}",
                        executor_stats.get(source, 0),
                    )
                if len(cohort) > 1:
                    _bump(
                        self.scheduler_stats,
                        "external_progressive_verify_batched_rounds",
                    )
                    _bump(
                        self.scheduler_stats,
                        "external_progressive_verify_batched_lanes",
                        len(cohort),
                    )
                    if cross_session_ast_mode:
                        _bump(
                            self.scheduler_stats,
                            "external_progressive_cross_session_ast_rounds",
                        )
                        _bump(
                            self.scheduler_stats,
                            "external_progressive_cross_session_ast_sessions",
                            len(cohort),
                        )
                        _bump(
                            self.scheduler_stats,
                            "external_progressive_cross_session_ast_packs",
                            len(physical_stats["cross_session_packs"]),
                        )
                        _bump(
                            self.scheduler_stats,
                            "external_progressive_cross_session_ast_b8_launches",
                            physical_stats.get("b8_aligned_tile_launches", 0),
                        )
                        frontier = physical_stats.get("live_frontier")
                        if frontier is not None:
                            for suffix in (
                                "rows_admitted",
                                "sessions_retired",
                                "branches_unvisited",
                                "proposal_tokens_not_scheduled",
                                "refills",
                                "occupancy_numerator",
                                "occupancy_denominator",
                                "partial_packs",
                                "full_packs",
                            ):
                                _bump(
                                    self.scheduler_stats,
                                    f"external_progressive_ast_frontier_{suffix}",
                                    frontier[suffix],
                                )
                                self.scheduler_stats[
                                    f"external_progressive_apt_frontier_{suffix}"
                                ] = self.scheduler_stats[
                                    f"external_progressive_ast_frontier_{suffix}"
                                ]
                            self.scheduler_stats[
                                "external_progressive_ast_frontier_max_service_gap"
                            ] = max(
                                self.scheduler_stats[
                                    "external_progressive_ast_frontier_max_service_gap"
                                ],
                                frontier["max_service_gap"],
                            )
                            self.scheduler_stats[
                                "external_progressive_apt_frontier_max_service_gap"
                            ] = self.scheduler_stats[
                                "external_progressive_ast_frontier_max_service_gap"
                            ]
                        _bump(
                            self.scheduler_stats,
                            "external_progressive_alternative_rounds",
                            len(cohort),
                        )
                        _bump(
                            self.scheduler_stats,
                            "external_progressive_alternative_branches",
                            sum(
                                len(group)
                                for group in block_groups
                            ),
                        )
                        _bump(
                            self.scheduler_stats,
                            "external_progressive_alternative_full_accepts",
                            sum(
                                len(selection["full"])
                                for selection in cross_session_selections
                            ),
                        )
                    for width in range(1, 5):
                        _bump(
                            self.scheduler_stats,
                            f"external_progressive_verify_b{width}_aligned_tile_launches",
                            physical_stats["aligned_tile_launches"][width],
                        )
                        _bump(
                            self.scheduler_stats,
                            f"external_progressive_verify_b{width}_aligned_continued_launches",
                            physical_stats["aligned_continued_launches"][width],
                        )
                    if len(cohort) == 4 and physical_stats[
                        "first_tile_pairwise_disjoint"
                    ]:
                        _bump(
                            self.scheduler_stats,
                            "external_progressive_verify_b4_disjoint_first_tiles",
                        )
                elif alternative_mode:
                    _bump(
                        self.scheduler_stats,
                        "external_progressive_alternative_rounds",
                    )
                    _bump(
                        self.scheduler_stats,
                        "external_progressive_alternative_branches",
                        len(verify_blocks),
                    )
                    _bump(
                        self.scheduler_stats,
                        "external_progressive_alternative_full_accepts",
                        len(full),
                    )
                    _bump(
                        self.scheduler_stats,
                        "external_progressive_alternative_b2_short_circuits",
                        int(bool(full) and full[0] == 1),
                    )
                    if self.progressive_alternative_strategy in {
                        "shared_prefix",
                        "confidence_tree",
                        "paired_continuations",
                    }:
                        _bump(
                            self.scheduler_stats,
                            "external_progressive_alternative_shared_prefix_rounds",
                        )
                        _bump(
                            self.scheduler_stats,
                            "external_progressive_alternative_shared_prefix_tokens",
                            physical_stats.get("shared_prefix_tokens", 0),
                        )
                        _bump(
                            self.scheduler_stats,
                            "external_progressive_alternative_unique_target_rows",
                            physical_stats.get("unique_target_rows", 0),
                        )
                    for width in range(1, 5):
                        _bump(
                            self.scheduler_stats,
                            f"external_progressive_verify_b{width}_aligned_tile_launches",
                            physical_stats["aligned_tile_launches"][width],
                        )
                        _bump(
                            self.scheduler_stats,
                            f"external_progressive_verify_b{width}_aligned_continued_launches",
                            physical_stats["aligned_continued_launches"][width],
                        )
                self.scheduler_stats[
                    "external_progressive_verify_max_cohort_width"
                ] = max(
                    self.scheduler_stats[
                        "external_progressive_verify_max_cohort_width"
                    ],
                    physical_stats["max_cohort_width"],
                )
                if clock is not None:
                    clock = self._mark("progressive_verify", clock)
                self._commit(
                    cohort,
                    decisions,
                    features,
                    blocks=blocks,
                    transaction=None,
                    committed_rows=committed_caches,
                    committed_rngs=committed_rngs,
                    deferred_context_uids=deferred_context_uids,
                )
                return
            if getattr(self, "ast_tree_tile_rows", 0) and any(proposal_counts):
                raise RuntimeError(
                    "AST tree tiles have no admitted target-tree dispatch "
                    "for this request; ordinary verification is not a tile fallback"
                )
            if self.adaptive_policy is not None:
                effective_depth = max(proposal_counts, default=0)
                _bump(self.scheduler_stats, "external_adaptive_rounds")
                saved = max(0, fixed_depth - effective_depth) * len(cohort)
                for lane in cohort:
                    lane.adaptive_round_policy = "lagged_cohort"
                    lane.adaptive_feature_source = "lagged_q"
                    if saved:
                        lane.adaptive_trimmed_rounds = getattr(lane, "adaptive_trimmed_rounds", 0) + 1
                _bump(self.scheduler_stats, "external_adaptive_trimmed_rounds", int(saved > 0))
                _bump(self.scheduler_stats, "external_adaptive_trimmed_target_rows", saved)
                _bump(self.scheduler_stats, "external_adaptive_target_rows", verify_width * len(cohort))
                self.scheduler_stats["external_adaptive_round_depth"] = effective_depth
                self.scheduler_stats["external_adaptive_verify_width"] = verify_width
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
                if (
                    all(count == 0 for count in proposal_counts)
                    and self._ast_anchor_cohort_eligible(cohort)
                ):
                    logits, features, transaction = self._ast_target_tree_forward_many(
                        cohort, [lane.cache for lane in cohort], inputs,
                        parents=[[-1] for _lane in cohort],
                    )
                    _bump(self.scheduler_stats, "external_ast_anchor_target_launches")
                    _bump(self.scheduler_stats, "external_ast_anchor_target_rows", len(cohort))
                    for lane in cohort:
                        counts = getattr(lane, "cohort_target_execution", None)
                        if counts is None:
                            counts = lane.cohort_target_execution = {}
                        key = "ast_anchor:target_tree_packed"
                        counts[key] = counts.get(key, 0) + 1
                elif self.target_execution == "provider":
                    parents = [
                        list(range(-1, len(row) - 1)) for row in inputs
                    ]
                    logits, features, transaction = self._target_tree_forward_many(
                        cohort, inputs, parents
                    )
                else:
                    owner = self._target_owner([l.cache for l in cohort])
                    lengths = [count + 1 for count in proposal_counts]
                    if all(count == 0 for count in proposal_counts) and hasattr(
                        owner, "views"
                    ):
                        # Anchors only (a parked session, an ordinary probe, a
                        # final-budget row): an ordinary decode step with no
                        # rollback epoch, as the PLD executor runs it.
                        _bump(self.scheduler_stats, "external_plain_single_token_rounds")
                        transaction = owner.begin(lengths=lengths, plain_single_token=True)
                    else:
                        transaction = owner.begin(lengths=lengths)
                    if trusted_round:
                        logits, features = self.model.forward_with_taps(
                            self.mx.array(inputs),
                            transaction.caches,
                            self.layers,
                            last_logits_only=True,
                        )
                    else:
                        logits, features = self._cohort_target_forward(
                            self.mx.array(inputs),
                            transaction,
                            cohort,
                            site=(
                                "plain_single_token"
                                if getattr(transaction, "plain", False)
                                else "standard_verify"
                            ),
                        )
                self.mx.eval(logits, features)
            finally:
                if steer is not None:
                    taps.steer = None
            if clock is not None:
                clock = self._mark("verify_forward", clock)
            decisions = (
                [self._verify_trusted_pld(cohort[0], blocks[0], logits)]
                if trusted_round
                else self._verify(cohort, blocks, logits)
            )
            if clock is not None:
                self._mark("verify_laws", clock)
            self._commit(
                cohort,
                decisions,
                features,
                blocks=blocks,
                transaction=transaction,
                deferred_context=forced_depth is not None,
                deferred_context_uids=deferred_context_uids,
            )
            for commit, decision in zip(steer_commits, decisions):
                commit(min(decision.accepted + 1, len(decision.emitted)))
            if self.adaptive_policy is not None:
                self._adaptive_observe(blocks, decisions, cohort)
        except BaseException:
            if transaction is not None and not transaction.closed:
                try: transaction.abort()
                except BaseException:  # noqa: BLE001, S110 - authoritative snapshots restore below
                    pass
            self._restore_round(cohort, recovery)
            self.scheduler_stats = stats_snapshot
            if self.adaptive_policy is not None:
                self.acceptance_estimator = adaptive_snapshot
            self.scheduler_stats["recovery_checkpoint_restores"] += len(recovery)
            raise
        finally: self._open = False

    def _tree_round(self, cohort):
        if not cohort:
            return
        if len(cohort) > 1 and self.target_execution != "provider":
            raise ValueError("multi-lane tree15 requires target-tree provider execution")
        if len(cohort) > self.target_tree_cohort_limit:
            raise ValueError("tree15 cohort exceeds its bounded lane limit")
        clock = self._phase_time() if self.round_timing else None
        phase = _PhaseClock(self) if self.round_timing else None
        if phase is not None:
            # Wall time between the previous tree round's end and this one's
            # start: scheduling, response publication, ordinary probes.
            ended, uids = getattr(self, "_tree_round_ended", None) or (None, None)
            if ended is not None and uids == [lane.uid for lane in cohort] \
                    and clock - ended < 1.0:
                self.round_times["tree_gap"] += clock - ended
                _bump(
                    self.scheduler_stats,
                    "external_phase_tree_gap_ns",
                    int((clock - ended) * 1e9),
                )
        tile_started = time.perf_counter() if getattr(self, "tile_budget", None) else None
        recovery = self._snapshot_round(cohort)
        self.scheduler_stats["recovery_checkpoint_captures"] += len(recovery)
        if phase is not None:
            phase("tree_recovery_capture")
        stats_snapshot = self._snapshot_scheduler_stats()
        if phase is not None:
            phase("tree_stats_snapshot")
        budget_backup = None
        calibration = getattr(self, "tree_calibration", None)
        calibration_backup = None if calibration is None else calibration.snapshot()
        self._open = True
        transaction = None
        self._tree_clock = phase
        try:
            blocks = self._propose(cohort)
            if any(not isinstance(block, TreeDraftRow) for block in blocks):
                raise RuntimeError("tree15 proposal did not return a tree")
            if calibration is not None and calibration.sizes:
                blocks = self._calibrated_live_blocks(cohort, blocks)
            if len(cohort) > 1 and getattr(self, "ast_batch_row_sharing", False):
                blocks = self._share_cohort_rows(cohort, blocks)
            # Verified nodes per lane-round, after any calibrated cut or row
            # sharing (keys are node counts, at most 31).
            for block in blocks:
                _bump_map(self.scheduler_stats, "external_tree_verified_nodes", len(block.tokens))
            inputs = [
                [lane.anchor] + list(block.tokens)
                for lane, block in zip(cohort, blocks)
            ]
            parents = [self._target_tree_parents(block) for block in blocks]
            if phase is not None:
                phase("tree_inputs")
            logits, features, transaction = self._target_tree_forward_many(
                cohort, inputs, parents
            )
            if phase is not None:
                phase("tree_target_forward_build")
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
                self._presnapshot_draft(cohort)
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
                self._presnapshot_draft(cohort)
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
            pipelined = (
                self.tree_gates["pipeline_draft"]
                and callable(getattr(self.draft, "start_tree", None))
                and not self.pair_context_tokens
            )
            if tile_started is not None:
                # Observed at the verified boundary, before the commit: the
                # next lattice's node budget must already be the one the
                # next round will ask for when it is queued below.
                budget_backup = self.tile_budget.snapshot()
                seconds = time.perf_counter() - tile_started
                if any(getattr(lane, "external_rounds", 0) < 2 for lane in cohort):
                    seconds = None          # prompt ingest is not tile cost
                for block, decision in zip(blocks, decisions):
                    rows = decision.commit_rows or [0]
                    self.tile_budget.observe(
                        len(cohort), len(block.tokens),
                        [row - 1 for row in rows[1:]], seconds,
                    )
                self.scheduler_stats["external_tree_tile_budget"] = self.tile_budget.receipt()
            if calibration is not None:
                self._observe_tree_calibration(
                    cohort, blocks, decisions,
                    None if tile_started is None else time.perf_counter() - tile_started,
                )
            if phase is not None:
                phase("tree_tile_budget")
            early = None
            if pipelined:
                # The drafter reads nothing the target commit writes; queue
                # it first so it runs while the host commits and publishes.
                early = (
                    self._prelaunch_tree_early(cohort[0], decisions[0], features)
                    if len(cohort) == 1
                    else self._prelaunch_tree_cohort_early(cohort, decisions, features)
                )
            self._commit(
                cohort,
                decisions,
                features,
                blocks=blocks,
                transaction=transaction,
            )
            if phase is not None:
                phase("tree_commit")
            # ``_propose`` adopts a queued lattice only for a singleton group,
            # so a multi-lane cohort's prelaunches were built and thrown away
            # every round (SPEC-01, packed provider B2-B4).
            if pipelined:
                if len(cohort) == 1:
                    self._bind_early_prelaunch(cohort[0], decisions[0], early)
                else:
                    self._bind_early_cohort(cohort, decisions, early)
                if phase is not None:
                    phase("tree_draft_prelaunch")
            else:
                self._discard_prelaunched([lane.uid for lane in cohort])
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
                self._tree_round_ended = (
                    self._mark("tree_round", clock), [lane.uid for lane in cohort]
                )
                _bump(self.scheduler_stats, "external_phase_rounds")
        except BaseException:
            if transaction is not None and not transaction.closed:
                try:
                    transaction.abort()
                except BaseException:  # noqa: S110 - preserve the original verification failure
                    pass
            self._restore_round(cohort, recovery)
            self.scheduler_stats = stats_snapshot
            self.scheduler_stats["recovery_checkpoint_restores"] += len(recovery)
            self._discard_prelaunched([lane.uid for lane in cohort])
            if budget_backup is not None:
                self.tile_budget.restore(budget_backup)
            if calibration_backup is not None:
                calibration.restore(calibration_backup)
            raise
        finally:
            self._tree_clock = None
            self._draft_presnapshot = None
            self._open = False

    def _calibrated_live_blocks(self, cohort, blocks):
        """Kapitsa's live-row rule: keep the prefix that pays for its rows.

        A calibrated best-first tree's prefixes are parent-closed and its
        path probabilities give each prefix's expected tokens; the tile cost
        per verified size comes from the measured tile budget.  Only a lone
        lane is cut here (a cohort's trees are padded to one width).  Which
        rows are verified depends only on the drafter's tree, never on the
        target, so the verification law is unchanged.
        """

        from .ast_tree_calibration import live_nodes

        calibration = self.tree_calibration
        controller = getattr(self, "tile_budget", None)
        if len(cohort) != 1 or controller is None:
            return blocks
        block = blocks[0]
        info = block.info
        if info is None or len(block.tokens) < 2:
            return blocks
        curve = controller.cost_curve(1)
        if curve is None:
            return blocks
        intercept, slope = curve
        live = live_nodes(
            info["path_log_p"], lambda n: intercept + slope * (n + 1),
            rate=calibration.rate(),
        )
        if not 1 <= live < len(block.tokens):
            return blocks
        calibration.truncated_rounds += 1
        calibration.truncated_nodes += len(block.tokens) - live
        return [TreeDraftRow(
            list(block.tokens[:live]), list(block.parents[:live]), block.width,
            info={key: list(value[:live]) for key, value in info.items()},
        )]

    def _observe_tree_calibration(self, cohort, blocks, decisions, seconds):
        """Feed each verified tree's trials to the per-depth calibration."""

        from .ast_tree_calibration import expected_tokens, tree_records

        calibration = self.tree_calibration
        committed = 0
        for lane, block, decision in zip(cohort, blocks, decisions):
            rows = decision.commit_rows or [0]
            committed += len(decision.emitted)
            info = block.info
            if info is None or not block.tokens:
                continue
            calibration.observe(
                self._calibration_key(lane),
                tree_records(block.parents, info["depths"], info["log_q"], rows),
                predicted=expected_tokens(info["path_log_p"], len(block.tokens)),
                realized=len(rows),
            )
        if seconds is not None and not any(
            getattr(lane, "external_rounds", 0) < 2 for lane in cohort
        ):
            calibration.observe_rate(committed, seconds)
        self.scheduler_stats["external_tree_calibration"] = calibration.receipt()

    def _ordinary_pipeline_eligible(self, cohort, *, following=False):
        """Whether a permanent B1 lane may use native target pipelining.

        Diagnostic phase timing stays on the synchronous implementation so a
        boundary fence continues to measure exactly the phase that submitted
        work.  The production pipeline is deliberately narrow: one permanent
        greedy lane, no processor or residual-steering state, the ordinary
        target executor, and the hybrid cache family whose exact one-token
        undo record is already used by parked decode.
        """
        if (
            self.round_timing
            or len(cohort) != 1
            or self.target_execution == "provider"
            or self.fly_verification.enabled
            or getattr(getattr(self.model, "model", None), "residual_taps", None)
            is not None
            or not self._is_hybrid([cohort[0].cache])
        ):
            return False
        lane = cohort[0]
        live = [candidate for candidate in self.lanes.values() if not candidate.cancelled]
        if [candidate.uid for candidate in live] != [lane.uid]:
            return False
        return (
            lane.ordinary
            and not lane.cancelled
            and not lane.processors
            and float(lane.sampling.get("sampling_temp", 0)) == 0.0
            and lane.maximum - lane.generated >= 2
        )

    def _dispatch_ordinary_pipeline(self, cohort, anchors):
        """Dispatch a direct-cache B1 ordinary step without a host wait."""
        recovery = self._parked_recovery(cohort)
        lane = cohort[0]
        inputs = (
            anchors.astype(self.mx.int32)[:, None]
            if isinstance(anchors, self.mx.array)
            else self.mx.array([[int(anchors[0])]], dtype=self.mx.int32)
        )
        try:
            logits = self.model(inputs, cache=lane.cache)
            rows32 = logits[:, -1, :].astype(self.mx.float32)
            normalizer_rows = self.mx.logsumexp(
                rows32, axis=-1, keepdims=True
            )
            logprob_rows = (
                [rows32[0] - normalizer_rows[0]]
                if lane.sampling.get("emit_logprobs", True)
                else []
            )
            normalizers = normalizer_rows.reshape(-1)
            next_anchors = self.mx.argmax(rows32, axis=-1)
            self.mx.async_eval(
                logits, next_anchors, normalizers, *logprob_rows
            )
        except BaseException:
            self._restore_parked(recovery)
            raise
        return _ParkedInflight(
            lanes=list(cohort),
            transaction=None,
            logits=logits,
            features=None,
            next_anchors=next_anchors,
            normalizers=normalizers,
            logprob_rows=logprob_rows,
            recovery=recovery,
            timing_start_ns=0,
            generated_before=[lane.generated],
        )

    def _abort_ordinary_inflight(self):
        inflight, self._ordinary_inflight = self._ordinary_inflight, None
        if inflight is not None:
            self._restore_parked(inflight.recovery)
            _bump(self.scheduler_stats, "external_ordinary_pipeline_aborts")

    def _ordinary_pipeline_round(self, cohort):
        """Read one B1 ordinary result while its successor runs on device."""
        inflight = self._ordinary_inflight
        if inflight is None:
            self._ordinary_inflight = self._dispatch_ordinary_pipeline(
                cohort, [cohort[0].anchor]
            )
            _bump(self.scheduler_stats, "external_ordinary_pipeline_starts")
            return
        if [lane.uid for lane in cohort] != [lane.uid for lane in inflight.lanes]:
            self._abort_ordinary_inflight()
            return self._ordinary_round(cohort)
        self._ordinary_inflight = None
        following = None
        try:
            if self._ordinary_pipeline_eligible(cohort, following=True):
                following = self._dispatch_ordinary_pipeline(
                    cohort, inflight.next_anchors
                )
            self.mx.eval(
                inflight.next_anchors,
                inflight.normalizers,
                *inflight.logprob_rows,
            )
            decisions = self._parked_decisions(cohort, inflight)
            lane = cohort[0]
            decision = decisions[0]
            token = int(decision.emitted[0])
            lazy = int(inflight.next_anchors.item())
            finish = (
                "stop"
                if token in self.stops
                else "length"
                if lane.generated + 1 >= lane.maximum
                else None
            )
            if following is not None and (token != lazy or finish is not None):
                self._restore_parked(following.recovery)
                following = None
                _bump(self.scheduler_stats, "external_ordinary_pipeline_drains")
            lane.history.append(lane.anchor)
            lane.anchor = token
            lane.generated += 1
            lane.target_max_width = max(lane.target_max_width, 1)
            if self.adaptive_policy is not None:
                self.scheduler_stats["external_adaptive_round_depth"] = 0
                self.scheduler_stats["external_adaptive_verify_width"] = 1
                _bump(self.scheduler_stats, "external_adaptive_target_rows")
                lane.adaptive_round_policy = "ordinary_target"
                lane.adaptive_feature_source = "not_executed"
            logp = (
                decision.response_logprobs[0]
                if lane.sampling.get("emit_logprobs", True)
                else None
            )
            lane.ready.append(
                self._ordinary_response(
                    lane,
                    token=token,
                    logp=logp,
                    finish=finish,
                    execution_width=1,
                )
            )
            if following is not None:
                self._refresh_parked_host(following.recovery)
            count = len(cohort)
            self.scheduler_stats["ordinary_rounds"] += 1
            _bump(self.scheduler_stats, "external_ordinary_fast_path_rounds")
            _bump(self.scheduler_stats, "external_ordinary_fast_path_lanes", count)
            _bump(self.scheduler_stats, "external_draft_context_skipped", count)
            _bump(self.scheduler_stats, "external_taps_skipped", count)
            _bump(self.scheduler_stats, "external_transactions_skipped", count)
            _bump(self.scheduler_stats, "external_ordinary_pipelined_rounds")
            self.scheduler_stats["target_max_width"] = max(
                self.scheduler_stats["target_max_width"], 1
            )
            self._ordinary_inflight = following
        except BaseException:
            if following is not None:
                with suppress(Exception):
                    self._restore_parked(following.recovery)
            self._restore_parked(inflight.recovery)
            raise

    def _ordinary_round(self, cohort):
        """Advance permanently ordinary external lanes on target state only.

        ``Lane.ordinary`` is irreversible in this executor.  Unlike a final
        transient depth-zero round, these rows can never need DFlash context
        again, so avoid draft append, target taps and speculative rollback.
        """
        phase = _PhaseClock(self) if self.round_timing else None
        residual_taps = getattr(
            getattr(self.model, "model", None), "residual_taps", None
        )
        light_recovery = residual_taps is None and all(
            not lane.processors
            and float(lane.sampling.get("sampling_temp", 0)) == 0.0
            for lane in cohort
        )
        snapshots = (
            self._parked_recovery(cohort)
            if light_recovery
            else self._snapshot_round(cohort)
        )
        if phase is not None:
            phase("ordinary_snapshot")
        stats_snapshot = self._snapshot_scheduler_stats()
        self._open = True
        try:
            if self.adaptive_policy is not None:
                self.scheduler_stats["external_adaptive_round_depth"] = 0
                self.scheduler_stats["external_adaptive_verify_width"] = 1
                _bump(self.scheduler_stats, "external_adaptive_target_rows", len(cohort))
                for lane in cohort:
                    lane.adaptive_round_policy = "ordinary_target"
                    lane.adaptive_feature_source = "not_executed"
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
                    [1] * len(cohort),
                    plain_single_token=(
                        callable(getattr(self.model, "ordinary_external_target_forward", None))
                        and getattr(self.model, "ordinary_external_target_protocol", None)
                        == "committed-stock-ordinary-terminal-v1"
                    ),
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
            if phase is not None:
                phase("ordinary_setup")
            if steer is not None:
                taps.steer = steer
            try:
                if hybrid is not None:
                    logits, _ = self._cohort_target_forward(
                        inputs, hybrid, cohort, site="ordinary_round", taps=False
                    )
                else:
                    logits = self.model(inputs, cache=batched_cache)
                if hybrid is not None:
                    self.mx.eval(logits)
                    hybrid.commit([1] * len(cohort))
                    self.mx.eval([c.state for lane in cohort for c in lane.cache])
                else:
                    self.mx.eval(logits, [cache.state for cache in batched_cache])
                if phase is not None:
                    phase("ordinary_forward")
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
                if float(lane.sampling.get("sampling_temp", 0)) == 0.0:
                    selected = self._target_law(
                        lane,
                        logits[row, 0],
                        lane.history,
                        True,
                        response_rows,
                        greedy_token=True,
                        history_suffix=[lane.anchor],
                    )
                    # A dense one-hot law is almost 2 MiB for Qwen3.6's
                    # vocabulary.  The compact verifier is bit- and RNG-exact
                    # for greedy rows and avoids allocating/normalizing that
                    # host vector on every permanently ordinary token.
                    result = verify_greedy_proposals([], [selected], lane.rng)
                else:
                    targets = [
                        self._target_law(
                            lane,
                            logits[row, 0],
                            lane.history,
                            True,
                            response_rows,
                            history_suffix=[lane.anchor],
                        )
                    ]
                    result = verify_proposals([], [], targets, lane.rng)
                token = int(result.emitted[0])
                if phase is not None:
                    phase("ordinary_law")
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
                logp = None
                if lane.sampling.get("emit_logprobs", True):
                    if response_rows and response_rows[0] is not None:
                        logp = response_rows[0]
                    else:
                        logp = self.mx.log(
                            self.mx.array(
                                result.target_probabilities[0].astype(np.float32)
                            )
                        )
                lane.ready.append(
                    self._ordinary_response(
                        lane,
                        token=token,
                        logp=logp,
                        finish=finish,
                        execution_width=len(cohort),
                    )
                )
                if phase is not None:
                    phase("ordinary_publish")
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
            if light_recovery:
                self._restore_parked(snapshots)
            else:
                self._restore_round(cohort, snapshots)
            self.scheduler_stats = stats_snapshot
            raise
        finally:
            self._open = False

    def _ordinary_response(self, lane, *, token, logp, finish, execution_width):
        """Publish one permanent target-only token with the external receipt."""
        return SimpleNamespace(
            uid=lane.uid,
            token=int(token),
            logprobs=logp,
            finish_reason=finish,
            execution_width=int(execution_width),
            # One token is this round's final response, even when the request
            # continues. Match the multi-token external round contract.
            all_tokens=list(lane.history),
            prompt_cache=self._freeze_cache(lane.cache) if finish else None,
            # The draft plane stopped at fallback and is stale by construction.
            cache_sidecar=None,
            mtp_state=None,
            mtp_receipt=None,
            speculative_receipt={
                "kind": self.receipt_kind,
                "execution": (
                    "external_draft_verify"
                    if lane.external_rounds
                    else "ordinary_target"
                ),
                "current_execution": "ordinary_target",
                "ordinary_fallback": True,
                **(
                    {"ordinary_fallback_reason": lane.ordinary_reason}
                    if lane.ordinary_reason
                    else {}
                ),
                "external_rounds": lane.external_rounds,
                "accepted": lane.accepted,
                "proposed": lane.proposed,
                "round_accepted": 0,
                "round_proposed": 0,
                "target_width": lane.target_max_width,
                "draft_width": lane.draft_max_width,
                "qualification_authority": "serving_route",
                **self._target_execution_receipt(lane),
                **self._verification_receipt(lane),
                **self._adaptive_receipt(lane),
                **self._draft_settings_receipt(),
                **self._proposal_composition_receipt(
                    lane, current_source="ordinary"
                ),
                **self._continuation_receipt(lane),
                **(
                    self._round_timing_receipt(lane)
                    if self.round_timing
                    else {}
                ),
            },
        )

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
        # A model-free drafter (prompt lookup) has no draft KV to reserve.
        draft_per_token = (
            2 * int(getattr(draft, "num_key_value_heads", 0) or 0)
            * int(getattr(draft, "head_dim", 0) or 0) * 4
        )
        required = 0
        extra_required = 0
        ast_private_required = 0
        ast_transient_per_lane = []
        reserve_draft = getattr(self.draft, "extra_memory_reservation", None)
        for lane in lanes:
            lane_append = (
                min(append, max(0, len(lane.remaining) - 1))
                if prefill
                else append
            )
            target_bytes = sum(int(getattr(c,"nbytes",0)) for c in lane.cache)
            draft_bytes = sum(int(getattr(c,"nbytes",0)) for c in lane.draft_cache)
            required += 4*target_bytes + 3*draft_bytes
            required += lane_append*(target.num_hidden_layers*target_per_token + int(getattr(draft, "num_hidden_layers", 0) or 0)*draft_per_token)
            feature_width = getattr(draft, "feature_width", None) or target.hidden_size*len(self.layers)
            required += lane_append*int(feature_width)*8
            if callable(reserve_draft):
                extra = reserve_draft(
                    lane.draft_cache, append=lane_append,
                    pending_tokens=int(lane.tail.shape[1]), prefill=prefill,
                )
                if type(extra) is not int or extra < 0:
                    raise ValueError("drafter memory reservation must be nonnegative bytes")
                required += extra
                extra_required += extra
            if prefill:
                # Target prefill walks layers serially.  Only one layer's MLP
                # scratch is live at once, and the draft model does not run;
                # multiplying this transient by every target and draft layer
                # priced a skewed B4 as tens of GiB and split it into B2 slabs.
                required += lane_append*(target.intermediate_size+target.hidden_size)*16
            else:
                transient = lane_append*(target.num_hidden_layers*(target.intermediate_size+target.hidden_size)*16 + draft.num_hidden_layers*(draft.intermediate_size+draft.hidden_size)*16)
                required += transient
                ast_transient_per_lane.append(transient)
            if not prefill:
                # Four coexisting float64 q/p originals+normalized copies,
                # native logits/exp/output buffers, selector and safety slack.
                required += append*target.vocab_size*128
            if not prefill and getattr(self, "continuation_pool_ast_bridge", False):
                ast_private_required += (
                    (self.progressive_alternative_branches - 1) * target_bytes
                    + draft_bytes
                )
        if not prefill and getattr(self, "continuation_pool_ast_bridge", False):
            geometry = self._continuation_ast_bridge_geometry(len(lanes))
            concurrent_per_lane = max(
                1,
                (geometry["physical_path_rows"] + len(lanes) - 1)
                // len(lanes),
            )
            ast_private_required += (concurrent_per_lane - 1) * sum(
                ast_transient_per_lane
            )
            required += ast_private_required
            self.scheduler_stats["continuation_ast_bridge_geometry"] = geometry
            self.scheduler_stats["continuation_ast_bridge_private_bytes"] = (
                ast_private_required
            )
        elif not prefill and self.continuation_policy is not None:
            # Conservative admission includes all private path branches and
            # provider recomputation; it must not reserve as though B15 were B1.
            required *= self.continuation_policy.limit
        self.scheduler_stats["reservation_bytes"] = required
        self.scheduler_stats["draft_extra_reservation_bytes"] = extra_required
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

    def _fit_cohort(self, candidates, append, *, prefill=False):
        # A failed B4 reservation must not block B1/B2 work. Retain each lane's
        # own proposal count; changing width never changes its random schedule.
        while True:
            selected = []
            for lane in candidates:
                if self._admit(selected + [lane], append, prefill=prefill):
                    selected.append(lane)
            if selected or not self._reclaim_for_admission():
                return selected

    @staticmethod
    def _batch_cohort_key(lane):
        return declared_cohort_key(lane.sampling.get("batch_cohort"))

    def _record_atomic_cohort_failure(self, uids, cohort, reason):
        uids = tuple(int(uid) for uid in uids)
        if not uids:
            return
        self._atomic_cohort_failures.append(
            {"uids": uids, "cohort": dict(cohort), "reason": str(reason)}
        )
        _bump(self.scheduler_stats, "external_atomic_cohort_failures")

    def take_atomic_cohort_failures(self):
        failures, self._atomic_cohort_failures = (
            self._atomic_cohort_failures,
            [],
        )
        return failures

    def _atomic_cohort_prefill_ready(self, lane):
        """Hold a completed short prompt until its declared peers are ready.

        Ingress publishes a complete ``batch_cohort`` atomically, but uneven
        external prompts reach their final prompt token on different polls.
        Letting the short member decode immediately turns the cohort into a
        prefill/decode staircase and leaves the provider no sustained batch to
        execute.  The ordinary/self-MTP path already applies this boundary
        hold; external DFlash must honor the same request contract.

        A cohort that loses a member before this boundary is failed as a whole
        by ``remove``.  Once the complete frontier exists, normal dynamic
        decode membership resumes and early completion may shrink width.
        """
        key = self._batch_cohort_key(lane)
        if key is None or key in self._released_atomic_cohorts:
            return True
        members = [
            candidate
            for candidate in self.lanes.values()
            if not candidate.cancelled and self._batch_cohort_key(candidate) == key
        ]
        if len(members) != key[2]:
            # Ingress normally makes this impossible.  Hold rather than run a
            # smaller width if a direct/runtime caller violates publication;
            # explicit cancellation takes the fail-closed path in ``remove``.
            self._atomic_prefill_waiting_uids.add(lane.uid)
            _bump(self.scheduler_stats, "external_atomic_cohort_prefill_holds")
            return False
        if all(candidate.anchor is not None for candidate in members):
            if key not in self._released_atomic_cohorts:
                self._released_atomic_cohorts.add(key)
                _bump(self.scheduler_stats, "external_atomic_cohort_releases")
            return True
        self._atomic_prefill_waiting_uids.add(lane.uid)
        _bump(self.scheduler_stats, "external_atomic_cohort_prefill_holds")
        return False

    def _refuse_incomplete_atomic_cohorts(self):
        """Retire declarations that cannot own their promised frontier.

        Serving attaches every member before polling the executor.  A short
        group here therefore means either a direct caller supplied an
        incomplete declaration or membership was lost outside ``remove``.
        Both must fail closed rather than wait forever or decode at a smaller
        physical width.
        """

        groups = {}
        for lane in self.lanes.values():
            key = self._batch_cohort_key(lane)
            if key is not None and key not in self._released_atomic_cohorts:
                groups.setdefault(key, []).append(lane)
        failed = set()
        for key, members in groups.items():
            if len(members) == key[2]:
                continue
            uids = tuple(sorted(lane.uid for lane in members))
            cohort = members[0].sampling["batch_cohort"]
            # Temporarily mark the identity released so the retirement itself
            # does not recursively classify the same incomplete declaration.
            self._released_atomic_cohorts.add(key)
            self.remove(uids, cancelled=False)
            self._released_atomic_cohorts.discard(key)
            self._record_atomic_cohort_failure(
                uids,
                cohort,
                "declared external cohort cannot form its complete decode frontier",
            )
            failed.update(uids)
        return failed

    def scheduler_waiting_uids(self):
        """Lanes intentionally paused by scheduling, not memory admission.

        Uneven atomic cohorts can hold a short row at its prompt boundary for
        longer than the serving stall watchdog while the longest row keeps
        prefilling.  Report that phase wait explicitly.  Unprocessed rows are
        also scheduler-waiting unless their own minimum prefill slice failed
        memory admission in the most recent poll.
        """
        waiting = set(self._atomic_prefill_waiting_uids)
        # A lane omitted from a nonempty partial decode fit can run once the
        # admitted siblings publish and release scratch.  Refresh it for this
        # single scheduling turn; a wholly unfit group remains a real memory
        # wait and is intentionally left to the bounded serving watchdog.
        waiting.update(self._decode_fit_waiting_uids)
        waiting.update(
            lane.uid
            for lane in self.lanes.values()
            if lane.anchor is None and lane.uid not in self._memory_waiting_uids
        )
        return tuple(sorted(waiting))

    def disable_speculation(self, uid):
        if self._open: raise RuntimeError("Cannot change route inside a transaction")
        inflight = getattr(self, "_parked_inflight", None)
        if inflight is not None and any(lane.uid == uid for lane in inflight.lanes):
            # Close the previous numerical schedule before making this
            # request terminal ordinary. Its deferred draft state is still
            # exact at this publication boundary.
            self._drain_parked_inflight()
        self.lanes[uid].ordinary = True
        self._discard_prelaunched([uid])

    def next(self):
        mode = self.decode_first.mode()
        self._decode_first_round_mode = mode
        return round_phases.publish_round(
            self, mode, lambda: self._scheduler_round(mode)
        )

    def _prefill_phase(self, candidates):
        # A published phase may be resumed after cancellation or new arrivals.
        ordered = [self.lanes[uid] for uid in candidates if uid in self.lanes]
        self._memory_waiting_uids.difference_update(
            lane.uid for lane in ordered if lane.anchor is None
        )
        prompts = []
        # At most one bounded prefill slice; active decode progresses every poll.
        active_count = sum(l.anchor is not None for l in self.lanes.values())
        defer_serial_prefill = False
        # Beside a decoding lane the slice is also stall-bounded.
        contended = (
            self.decode_time_fairness.enabled and self._decoding_beside_prefill()
        )
        if self.external_varlen_prefill and active_count < self.capacity:
            candidates = [
                lane for lane in ordered
                if lane.anchor is None and len(lane.remaining) > 1
            ][: self.capacity - active_count]
            if self._decode_first_round_mode == "all":
                budget = self.decode_first.prefill_token_budget or self.prefill_step
                if len(candidates) > budget:
                    self.decode_first.bump("budget_deferred_rows", len(candidates) - budget)
                    candidates = candidates[:budget]
            pristine = [lane for lane in candidates if not lane.history]
            if (
                self.external_prefill_coalesce_ms
                and self.capacity - active_count > 1
                and 0 < len(candidates) < self.capacity - active_count
                and len(pristine) == len(candidates)
                and all(
                    len(lane.remaining) >= self.external_prefill_coalesce_min_tokens
                    for lane in pristine
                )
            ):
                oldest = min(pristine, key=lambda lane: lane.prefill_inserted_at)
                age_ms = (time.monotonic() - oldest.prefill_inserted_at) * 1000
                if age_ms < self.external_prefill_coalesce_ms:
                    defer_serial_prefill = True
                    _bump(
                        self.scheduler_stats,
                        "external_prefill_coalesce_deferrals",
                    )
                elif not oldest.prefill_coalesce_expired_recorded:
                    oldest.prefill_coalesce_expired_recorded = True
                    _bump(
                        self.scheduler_stats,
                        "external_prefill_coalesce_expirations",
                    )
            append = max(
                (
                    min(self._prefill_limit(lane), len(lane.remaining) - 1)
                    for lane in candidates
                ),
                default=0,
            )
            slab_step = None
            if contended and append:
                slab_step = self._fair_step(
                    max(self._prefill_limit(lane) for lane in candidates),
                    contended=True,
                    rows=len(candidates),
                    depth=max(len(lane.history) for lane in candidates),
                )
                append = min(append, slab_step)
            if candidates and self._decode_first_round_mode == "all":
                budget_step = round_phases.prefill_width(
                    self, max(self._prefill_limit(lane) for lane in candidates),
                    len(candidates),
                )
                slab_step = budget_step if slab_step is None else min(slab_step, budget_step)
                append = min(append, slab_step)
            cohort = (
                self._fit_cohort(candidates, append, prefill=True)
                if append and not defer_serial_prefill else []
            )
            if len(cohort) > 1:
                prompts.extend(
                    self._prefill_many(cohort, step=slab_step, contended=contended)
                )
        # A multirow slab is this poll's one bounded physical prefill slice.
        # If it could not form, retain the unchanged serial admission path.
        for lane in (() if prompts or defer_serial_prefill else ordered):
            if lane.anchor is not None or active_count >= self.capacity:
                continue
            limit = self._prefill_limit(lane)
            if contended:
                limit = self._fair_step(
                    limit, contended=True, depth=len(lane.history)
                )
            limit = round_phases.prefill_width(self, limit)
            append = min(limit, max(1, len(lane.remaining)-1))
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
                prompts.append(
                    self._prefill(lane, step=append, contended=contended)
                )
                break
            self._memory_waiting_uids.add(lane.uid)
        return prompts

    def _drain_ready(self, prompts):
        responses = []
        receipt_memo = {}
        # A lane can finish in the poll that completes its prefill
        # (max_tokens=1, or a stop as the first token).  Its end_of_prompt
        # response is in this same return, so its committed boundary stays
        # for serving to pop, as the ordinary and PLD routes keep it.
        # Dropping it with the lane lost the boundary and failed n>1 fanout
        # siblings.  A boundary returned by an earlier poll was already
        # offered, so it still goes with its lane.
        prompt_ended = {prompt.uid for prompt in prompts if prompt.end_of_prompt}
        prompt_ended.update(getattr(self, "_decode_first_resumed_prompt_uids", ()))
        for lane in list(self.lanes.values()):
            while lane.ready:
                response = lane.ready.popleft(); responses.append(response)
                if getattr(self, "ast_tree_tile_rows", 0):
                    # Feedback/round diagnostics have settled. Publish an
                    # immutable snapshot; siblings share only frozen groups.
                    response.speculative_receipt = freeze_receipt(
                        response.speculative_receipt, receipt_memo
                    )
                if response.finish_reason:
                    self.remove([lane.uid], cancelled=False, keep_boundary=lane.uid in prompt_ended)
                    break
                if self.ready_drain == "one":
                    break
        return responses

    def _scheduler_round(self, mode):
        prompts, responses = [], []
        # Ingress attaches a declared group completely before the first poll.
        # Refuse any incomplete direct/runtime declaration before it can run a
        # prefill forward or publish a prompt boundary.
        self._refuse_incomplete_atomic_cohorts()
        # A pending prefill may have just failed admission. Keep that evidence
        # until this lane's next admission attempt; clearing it here would hide
        # memory stalls behind the decode-first publication boundary.
        if mode == "off":
            self._memory_waiting_uids.clear()
        else:
            self._memory_waiting_uids.intersection_update(self.lanes)
        self._decode_fit_waiting_uids.clear()
        self._atomic_prefill_waiting_uids.clear()
        self._reclaims_left = 2; self._allocator_reclaimed = False
        ordered = list(self.lanes.values())
        if ordered:
            start = self._schedule_cursor % len(ordered)
            ordered = ordered[start:] + ordered[:start]
            self._schedule_cursor += 1
        candidates = tuple(lane.uid for lane in ordered)
        if mode == "off":
            prompts.extend(self._prefill_phase(candidates))
        self._decode_phase(ordered)
        responses.extend(self._drain_ready(prompts))
        self._reclaim_after_emission(len(responses))
        if mode != "off":
            # No speculative transaction is open across the publication point.
            ordered.clear()
            yield responses
            prompts.extend(self._prefill_phase(candidates))
        return prompts, responses

    def _observe_ast_tree_round(self, cohort, before, seconds):
        """Learn complete successful request turns at the publication boundary."""
        for lane in cohort:
            tuner = getattr(self, "_ast_tree_tuners", {}).get(lane.uid)
            plan = getattr(lane, "ast_tree_source_plan", None)
            previous = before[lane.uid]
            receipt = getattr(lane, "ast_tree_tiles_receipt", {})
            launches = receipt.get("launches", 0) - previous[2].get("launches", 0)
            if tuner is None or plan is None or launches <= 0:
                continue
            tuner.observe(
                plan, seconds=seconds,
                emitted=lane.generated - previous[0],
                accepted=lane.accepted - previous[1],
                unique_rows=receipt.get("target_rows", 0)
                - previous[2].get("target_rows", 0),
                launches=launches,
                terminal=previous[3] < self.num_draft + 1
                or receipt.get("last", {}).get("termination_reason") in ("stop", "limit"),
            )
            lane.ast_tree_tuning = tuner.receipt()
            receipt["tuning"] = lane.ast_tree_tuning
            for response in lane.ready:
                payload = getattr(response, "speculative_receipt", None)
                if payload is not None and "ast_tree_tiles" in payload:
                    payload["ast_tree_tiles"]["tuning"] = copy.deepcopy(lane.ast_tree_tuning)
            _bump(self.scheduler_stats, "external_ast_tree_completed_turns")

    def _decode_phase(self, ordered):
        self._memory_waiting_uids.difference_update(
            lane.uid for lane in ordered if lane.anchor is not None
        )
        ready = [
            lane
            for lane in ordered
            if lane.uid in self.lanes
            and lane.anchor is not None
            and not lane.ready
            and not lane.cancelled
            and self._atomic_cohort_prefill_ready(lane)
        ][:self.capacity]
        draining = sum(
            1
            for lane in ordered
            if lane.uid in self.lanes
            and lane.anchor is not None
            and lane.ready
            and not lane.cancelled
        )
        if draining:
            # Lanes sitting out this decode phase while they emit queued
            # tokens (ready_drain="one"): the cohort-split cause at B>1.
            _bump(self.scheduler_stats, "external_decode_draining_lanes_skipped", draining)
        active_width = (
            sum(lane.anchor is not None and not lane.cancelled for lane in self.lanes.values())
            if self.dynamic_tree_max_width == 4 else len(self.lanes)
        )
        inflight = getattr(self, "_parked_inflight", None)
        if (
            active_width != 1
            and inflight is not None
            and any(
                lane.cost_handoff_state is not None
                and lane.cost_handoff_state.session_prior_active
                for lane in inflight.lanes
            )
        ):
            # The dispatched B1 target transaction and its recovery snapshot
            # own the seeded state until this exact boundary is closed.  Book
            # it first; the next poll revokes the prior before regrouping.
            self._drain_parked_inflight()
            return
        self._apply_ast_cost_prior_scope(ready, execution_width=active_width)
        self._apply_width_park_prior(active_width)
        concurrent = self.dynamic_singleton_tree and active_width > self.dynamic_tree_max_width
        chain_cap = self._chain_draft_cap(active_width) if concurrent else None
        depth = self.num_draft if chain_cap is None else min(self.num_draft, chain_cap)
        multilane_draft_cap = getattr(self, "multilane_draft_cap", None)
        if multilane_draft_cap is not None and active_width > 1:
            # Match the grouping key to the count _propose will actually ask
            # for, so capped lanes can form one physical target cohort.
            depth = min(depth, multilane_draft_cap)
        mixed_ast = (
            getattr(self, "ast_execution_planner", None) is not None
            and self.target_execution == "reference"
            and not self.dynamic_singleton_tree
            and self.adaptive_policy is None
            and (
                self.progressive_verification_fidelity == PROGRESSIVE_FIDELITY_EXACT
                or getattr(self, "primary_near_tie_ast", False)
            )
        )
        ast_candidates = [lane for lane in ready if not lane.ordinary]
        ast_count = max(
            (0 if _speculation_parked(lane)
             else min(depth, lane.maximum - lane.generated - 1)
             for lane in ast_candidates),
            default=0,
        )
        ast_all_parked = bool(ast_candidates) and all(
            _speculation_parked(lane) for lane in ast_candidates
        )
        groups = {}
        for lane in ready:
            # The ordinary verifier groups compatible counts. The AST
            # verifier below owns per-request ragged bounds and draws.
            parked = _speculation_parked(lane)
            count = 0 if lane.ordinary or parked else min(depth,lane.maximum-lane.generated-1)
            # Permanent ordinary lanes must not share a zero-depth round with
            # transient final-budget rows, whose draft context remains exact.
            # Without AST, parked sessions retain their separate target-only
            # rounds and deferred draft append.
            if mixed_ast and not lane.ordinary:
                # AST owns ragged per-request depths, including the parked
                # anchor-only row. Do not fragment the physical cohort by a
                # reversible controller decision before AST can pack it.
                count, parked = ast_count, ast_all_parked
            groups.setdefault((count, lane.ordinary, parked),[]).append(lane)
        if ready:
            _bump(self.scheduler_stats, "external_decode_phases")
            _bump(self.scheduler_stats, f"external_decode_phase_ready_width_{len(ready)}")
            if len(groups) > 1:
                _bump(self.scheduler_stats, "external_decode_group_split_phases")
                if len({key[2] for key in groups}) > 1:
                    _bump(self.scheduler_stats, "external_decode_parked_split_phases")
        decode_started = time.perf_counter()
        for (count, _ordinary, _parked), candidates in groups.items():
            candidates.sort(
                key=lambda lane: getattr(lane, "decode_service_epoch", -1)
            )
            group_candidates = candidates
            if self.target_execution == "provider" and not concurrent:
                candidates = group_candidates[: self.target_tree_cohort_limit]
                self._decode_fit_waiting_uids.update(
                    lane.uid for lane in group_candidates[len(candidates):]
                )
            # Bound total target rows as lane pressure rises.  The adapter
            # declares the tree-node budget; admission prices that exact
            # verify shape rather than assuming 16 rows for every lane.
            append = (
                self._tree_node_budget(active_width) + 1
                if self.dynamic_singleton_tree and not concurrent and count
                else count + 1
            )
            cohort = self._fit_cohort(candidates, append)
            if not cohort:
                self._memory_waiting_uids.update(lane.uid for lane in candidates)
                continue
            self._decode_fit_waiting_uids.update(
                lane.uid for lane in group_candidates if lane not in cohort
            )
            if getattr(self, "ast_tree_tile_rows", 0):
                # Anchor-only work uses the same admitted private provider,
                # not an accidental B1 bypass around the physical cohort.
                limit = (
                    self._ast_tree_cohort_limit()
                    if self.progressive_verification_tile_policy == "auto" else 1
                )
                anchor_limit = (
                    self._ast_tree_anchor_limit()
                    if self.progressive_verification_tile_policy == "auto" else 1
                )
                turns, compatible = [], {}
                for lane in cohort:
                    state = lane.cost_handoff_state
                    packable = (
                        anchor_limit > 1 and not lane.ordinary
                        and float(lane.sampling.get("sampling_temp", 0)) == 0.0
                        and self._progressive_processors_supported(lane)
                        and self._batched_law_contract(lane)
                        and (state is None or state.settings.reversible)
                    )
                    if not packable:
                        turns.append([lane])
                    else:
                        kind = (
                            "parked" if _speculation_parked(lane)
                            else "final" if lane.maximum - lane.generated <= 1
                            else "speculative"
                        )
                        compatible.setdefault(kind, []).append(lane)
                for kind, lanes in compatible.items():
                    capacity = limit if kind == "speculative" else anchor_limit
                    turns.extend(
                        lanes[index:index + capacity]
                        for index in range(0, len(lanes), capacity)
                    )
                turns.sort(key=lambda group: min(
                    getattr(lane, "decode_service_epoch", -1) for lane in group
                ))
                pending = list(reversed(turns))
            elif self.draft_topology == "tree15" and self.target_execution != "provider":
                pending = [[lane] for lane in reversed(cohort)]
            else:
                pending = [cohort]
            while pending:
                group = pending.pop()
                try:
                    tree_timing = bool(getattr(self, "ast_tree_tile_rows", 0))
                    if tree_timing:
                        self._ast_tree_ready_sessions = max(1, len(ready))
                        tree_before = {
                            lane.uid: (lane.generated, lane.accepted,
                                dict(getattr(lane, "ast_tree_tiles_receipt", {})),
                                lane.maximum - lane.generated)
                            for lane in group
                        }
                        tree_started = time.perf_counter()
                    self._round_at_batch_route(
                        group, concurrent=concurrent, active_width=active_width
                    )
                    if tree_timing:
                        self._observe_ast_tree_round(
                            group, tree_before, time.perf_counter() - tree_started
                        )
                    self._decode_service_epoch += 1
                    for lane in group:
                        lane.decode_service_epoch = self._decode_service_epoch
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
        if groups and self.decode_time_fairness.enabled:
            # Decode rounds set the cost model's fixed per-forward cost.
            self.decode_time_fairness.observe_decode(
                time.perf_counter() - decode_started
            )
            self._sync_decode_fairness_stats()

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
        requested_uids = {int(uid) for uid in uids}
        inflight = getattr(self, "_parked_inflight", None)
        if inflight is not None and any(
            int(lane.uid) in requested_uids for lane in inflight.lanes
        ):
            # Unread, so the host never saw it: rewind exactly, then remove.
            self._abort_parked_inflight()
        # A kept plain-step owner pins its lanes' caches and joined views.
        self._parked_owner = None
        ordinary_inflight = getattr(self, "_ordinary_inflight", None)
        if ordinary_inflight is not None and any(
            int(lane.uid) in requested_uids for lane in ordinary_inflight.lanes
        ):
            # The unread token was never published.  Rewind its direct target
            # step before returning a cache or retiring the lane.
            self._abort_ordinary_inflight()
        uids = set(requested_uids)
        # Before the first complete decode frontier, membership is owned by
        # the declaration rather than by individual lanes.  Removing one
        # member retires every surviving sibling and reports those survivors
        # through the atomic-failure channel.  After release, ordinary dynamic
        # width shrink is legal and remains unchanged.
        owned_failures = []
        for uid in tuple(uids):
            lane = self.lanes.get(uid)
            if lane is None:
                continue
            key = self._batch_cohort_key(lane)
            if key is None or key in self._released_atomic_cohorts:
                continue
            members = {
                candidate.uid
                for candidate in self.lanes.values()
                if self._batch_cohort_key(candidate) == key
            }
            survivors = members - uids
            if survivors:
                owned_failures.append(
                    (
                        tuple(sorted(survivors)),
                        lane.sampling["batch_cohort"],
                        "declared external cohort lost a member before its decode frontier",
                    )
                )
            uids.update(members)
        result = {}
        removed = False
        self._discard_prelaunched(uids)
        for uid in uids:
            lane = self.lanes.pop(uid,None)
            if not keep_boundary: self.boundaries.pop(uid,None)
            if lane is not None:
                removed = True
                if (
                    self.ast_memo is not None
                    and not getattr(lane, "ast_memo_named_scope", False)
                ):
                    self.ast_memo.advance(
                        (lane.session_scope_hash,),
                        "anonymous_lane_end",
                        purge=True,
                    )
                if lane.cost_handoff_state is not None:
                    lane.cost_handoff_state.cancel()
                latches = getattr(self, "progressive_topology_latch", None)
                if latches is not None and hasattr(latches, "release"):
                    latches.release(uid)
                ast_planner = getattr(self, "ast_execution_planner", None)
                if ast_planner is not None:
                    ast_planner.release(uid)
                getattr(self, "_ast_tree_tuners", {}).pop(uid, None)
                for name in _LAST_ALTERNATIVE_FIELDS:
                    getattr(self, name, {}).pop(uid, None)
                lane.cancelled = bool(cancelled); self.scheduler_stats["cancelled"] += int(cancelled)
                if return_prompt_caches and uid in requested_uids:
                    result[uid] = self._freeze_cache(lane.cache)
                lane.ready.clear()
        for failed_uids, cohort, reason in owned_failures:
            self._record_atomic_cohort_failure(failed_uids, cohort, reason)
        live_keys = {
            key
            for lane in self.lanes.values()
            if (key := self._batch_cohort_key(lane)) is not None
        }
        self._released_atomic_cohorts.intersection_update(live_keys)
        if removed and not self.lanes:
            # No request state can reference verify/prefill scratch now.  Free
            # the allocator pool before the next atomic cohort is admitted;
            # otherwise a completed long-context cell can make its successor
            # fail admission on stale pages that a later rejection immediately
            # reclaims anyway.
            self.mx.clear_cache()
            _bump(self.scheduler_stats, "external_idle_allocator_reclaims")
        return result

    def close(self):
        round_phases.close_pending(self)
        self.remove(list(self.lanes)); self.boundaries.clear()
