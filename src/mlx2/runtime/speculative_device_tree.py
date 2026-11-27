"""Tree rounds of the device round (device_round v2 slices 1b-a and 1b-b).

Original mlx2 implementation (no mined code).  Design:
``docs/research/2026-11-25-gpu-resident-spec-rounds-design.md`` section 4.10.

Default off; with neither switch set the engine never calls into this module.
Switches (``device_round`` policy object):

``tree_lazy`` (slice 1b-a)
    A ``tree15`` round at width one drafts its tree lazily: the drafter's
    ``draft_tree_lazy`` returns a ``LazyTreeProposal`` (device int32 tokens,
    static parents), the verify input ``[anchor] + tokens`` is built on the
    device, and the target-tree forward (the generic reference control, or a
    provider declaring ``device_tree_inputs``) runs on it.  The round reads
    the tokens together with the verify logits and features once; the host
    decision (``_verify_tree`` / ``_verify_tree_batched``) and the commit are
    the switch-off round's, so greedy and sampled output are unchanged.
    The tree's geometry (target parents, root-to-node paths, the ancestor and
    sibling masks the decision uses) is a function of the static parents
    alone and is built once per topology (``geometry``), the section 3.5
    shape key; no per-round host mask building.
``tree_decide`` (slice 1b-b; needs ``tree_lazy``)
    A greedy lane's accepted path is selected on the device: each row's
    float32 argmax, a node accepted when its token equals its parent row's
    argmax and no earlier sibling's does (the host walk's first-match
    tie-break), the path closed over ancestors with static masks, and one
    packet ``[path rows by depth (-1 past the end), selected, finite]`` read
    with the tokens and verify output.  The host walks only that path:
    finiteness in visit order, one lane uniform per visited row, the first
    stop ends it, exactly as ``_walk_tree`` under the row-wise or batched
    host verifier; the ``RoundDecision`` takes the form of whichever
    verifier the host would run (``batched_laws`` gate), so the commit and
    the responses are unchanged.  Sampled lanes keep the host decision
    (counted under ``external_device_tree_decide_skips``).

What a round cannot serve (width above one, processors, tree calibration,
draft pipelining, a verify trace, a provider without device inputs, a drafter
without ``draft_tree_lazy``) runs the host tree round, counted under
``external_device_tree_decline_reasons``.  A drafter that cannot draft a
given round lazily raises ``LazyDecline``; the round is restored and redone on
the host (counted under the drafter's reason).
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass

import numpy as np

VERSION = "speculative-device-tree-v1"
SWITCHES = ("tree_lazy", "tree_decide")
COUNTERS = (
    "external_device_tree_rounds",
    "external_device_tree_lazy_lanes",
    "external_device_tree_host_reads",
    "external_device_tree_decide_lanes",
    "external_device_tree_host_decide_lanes",
    "external_device_tree_declines",
)
DECLINES = "external_device_tree_decline_reasons"
# Device tree rounds whose decision stayed on the host under ``tree_decide``.
DECIDE_SKIPS = "external_device_tree_decide_skips"


@dataclass(frozen=True)
class LazyTreeProposal:
    """A drafter's tree with its tokens still on the device.

    ``tokens``: int32 ``[n]`` device array in node order; ``parents``: the
    static parent-first topology (``-1`` = a child of the anchor), the same
    ``(tokens, parents)`` contract as ``propose_tree``."""

    tokens: object
    parents: tuple


def chain_branch_parents(pattern):
    """Parents of ``len(pattern)`` disjoint chains from the anchor, in order."""
    parents = []
    for depth in pattern:
        base = len(parents)
        parents.extend([-1] + [base + index for index in range(int(depth) - 1)])
    return tuple(parents)


@dataclass(frozen=True)
class TreeGeometry:
    """Per-topology constants of a verify tile (host numpy, built once)."""

    target_parents: tuple  # row-indexed: row 0 = anchor, row i = node i - 1
    rows: int
    max_depth: int
    paths: tuple  # root-to-row row paths, one per row
    parent_rows: np.ndarray  # [N] int32: the target row of each node's parent
    earlier_siblings: np.ndarray  # [N, N] bool: j is an earlier sibling of i
    ancestors: np.ndarray  # [N, N] bool: j is i or an ancestor of i (nodes)
    depth_select: np.ndarray  # [D + 1, R] bool: row r sits at depth k


@functools.lru_cache(maxsize=128)
def geometry(parents):
    parents = tuple(int(parent) for parent in parents)
    count = len(parents)
    for node, parent in enumerate(parents):
        if not -1 <= parent < node:
            raise ValueError("tree parents must be parent-first with -1 for the anchor")
    target = (-1, *(0 if parent < 0 else parent + 1 for parent in parents))
    rows = count + 1
    depth = [0] * rows
    paths = [(0,)]
    for row in range(1, rows):
        depth[row] = depth[target[row]] + 1
        paths.append((*paths[target[row]], row))
    parent_rows = np.array(target[1:], dtype=np.int32)
    earlier = np.zeros((count, count), dtype=bool)
    ancestors = np.zeros((count, count), dtype=bool)
    for node in range(count):
        for other in range(node):
            earlier[node, other] = parents[other] == parents[node]
        for row in paths[node + 1][1:]:
            ancestors[node, row - 1] = True
    max_depth = max(depth)
    select = np.zeros((max_depth + 1, rows), dtype=bool)
    for row in range(rows):
        select[depth[row], row] = True
    return TreeGeometry(target, rows, max_depth, tuple(paths), parent_rows, earlier,
                        ancestors, select)


def _bump(stats, key, amount=1):
    stats[key] = stats.get(key, 0) + amount


def decline(owner, reason, lanes=1):
    _bump(owner.scheduler_stats, "external_device_tree_declines", lanes)
    reasons = owner.scheduler_stats.setdefault(DECLINES, {})
    reasons[reason] = reasons.get(reason, 0) + lanes


def engine_reason(owner, cohort):
    """Why this tree round keeps the host path, or None."""
    from .external_speculative import _VERIFY_TRACE_PATH

    if len(cohort) != 1:
        return "width"
    execution = getattr(owner, "target_execution", "reference")
    if execution == "provider":
        if getattr(owner.target_tree_executor_provider, "device_tree_inputs", False) is not True:
            return "provider_without_device_inputs"
    elif execution != "reference":
        return "target_execution"
    draft = owner.draft
    if not callable(getattr(draft, "draft_tree_lazy", None)):
        return "drafter_without_lazy_tree"
    if _twin_overridden(draft):
        # A subclass replaced a host method the lazy tree mirrors (e.g. its
        # own ``_lazy_tree_paths``) but inherited the lazy hook unchanged.
        return "lazy_tree_twin_overridden"
    if getattr(owner, "tree_calibration", None) is not None:
        # Calibrated trees are reordered on host tokens before the forward.
        return "tree_calibration"
    if owner._draft_pipelines():
        # Pipelined lattices are host trees adopted from the previous round.
        return "pipeline_draft"
    if _VERIFY_TRACE_PATH is not None:
        return "verify_trace"
    if getattr(draft, "requires_tree_proposal_contexts", False) is True:
        return "proposal_contexts"
    if getattr(draft, "requires_target_caches", False):
        return "target_caches"
    if getattr(owner, "ast_batch_row_sharing", False):
        return "batch_row_sharing"
    for lane in cohort:
        if lane.processors:
            # Forbidden roots and processed drafts change the host tree.
            return "processors"
    return None


def _defining_class(cls, name):
    return next((klass for klass in cls.__mro__ if name in vars(klass)), None)


def _twin_overridden(draft):
    """Whether a host twin named by ``draft_tree_lazy_host_twins`` is defined
    below the class that defines ``draft_tree_lazy``."""
    cls = type(draft)
    owner = _defining_class(cls, "draft_tree_lazy")
    twins = getattr(draft, "draft_tree_lazy_host_twins", ("propose_tree",))
    return any(_defining_class(cls, twin) not in (None, owner) and issubclass(
        _defining_class(cls, twin), owner) for twin in twins)


def run(runtime, owner, cohort):
    """``device_round.run_tree``: serve one ``tree15`` round (always served)."""
    from .proposal_composition import LazyDecline

    if runtime.device_commit:
        # The tree round commits on host-form caches.
        runtime.leave_device_form(owner, cohort)
    reason = engine_reason(owner, cohort)
    if reason is not None:
        decline(owner, reason, len(cohort))
        return owner._tree_round(cohort)
    draft_stats = getattr(owner.draft, "stats", None)
    draft_snapshot = dict(draft_stats) if type(draft_stats) is dict else None
    try:
        return owner._tree_round(cohort, device_tree=DeviceTree(runtime))
    except LazyDecline as error:
        # ``_tree_round`` restored the lanes and statistics before re-raising.
        if draft_snapshot is not None:
            owner.draft.stats.clear()
            owner.draft.stats.update(draft_snapshot)
        decline(owner, error.reason, len(cohort))
        return owner._tree_round(cohort)


class DeviceTree:
    """One device tree round's hooks into ``_propose`` and ``_tree_round``."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.policy = runtime.policy

    # -- the proposal (``_propose``'s ``propose_tree`` call) ---------------------
    def propose(self, owner, lanes, arguments, forbidden, options):
        from .proposal_composition import LazyDecline

        if len(lanes) != 1:
            raise RuntimeError("device tree proposals run one lane at a time")
        if any(forbidden):
            raise LazyDecline("forbidden_root")
        accepted = getattr(owner.draft, "draft_tree_lazy_options", frozenset())
        unknown = sorted(set(options) - set(accepted))
        if unknown:
            raise LazyDecline(f"tree_option_{unknown[0]}")
        anchors, hidden, cache, count = arguments[:4]
        proposal = owner.draft.draft_tree_lazy(anchors[0], hidden, cache, count, **options)
        parents = tuple(int(parent) for parent in proposal.parents)
        tokens = proposal.tokens
        if tokens.ndim != 1 or int(tokens.shape[0]) != len(parents) or not parents:
            raise RuntimeError("lazy tree proposal tokens and parents differ")
        geometry(parents)  # validates parent-first order
        _bump(owner.scheduler_stats, "external_device_tree_lazy_lanes")
        self.runtime._lane_count(lanes[0], "tree_lazy_rounds")
        return [(tokens, list(parents))]

    # -- the forward ------------------------------------------------------------
    def forward(self, owner, cohort, blocks):
        """The verify input on the device and the target-tree forward on it.

        Returns ``(inputs, logits, features, transaction)``."""
        from .external_speculative import _amoeba_rows

        mx = owner.mx
        (lane,), (block,) = cohort, blocks
        geo = geometry(tuple(block.parents))
        inputs = mx.concatenate([
            mx.array([int(lane.anchor)], dtype=mx.int32),
            _device_tokens(mx, block.tokens),
        ])
        if owner.target_execution == "reference":
            logits, features, transaction = self._reference_forward(owner, lane, inputs, geo)
        else:
            cached = owner.tree_gates["cache_executor"]
            with _amoeba_rows(owner, [lane], provider=True):
                result = owner.target_tree_executor_provider.forward(
                    owner.model, inputs, list(geo.target_parents), lane.cache, owner.layers,
                    cached=cached,
                )
            logits, features, transaction = owner._finish_target_tree_dispatch(
                [lane], result, cached=cached)
        return [inputs], logits, features, transaction

    @staticmethod
    def _reference_forward(owner, lane, inputs, geo):
        """``_reference_tree_forward`` with device inputs: every root-to-node
        path replayed on a private cache copy, gathered on the device."""
        from .external_speculative import _amoeba_rows, _ReferenceTreeTransaction

        mx = owner.mx
        logits, features = [], []
        for path in geo.paths:
            cache = owner._copy_reference_cache(lane.cache)
            row_inputs = mx.take(inputs, mx.array(path, dtype=mx.int32))[None]
            with _amoeba_rows(owner, [lane]):
                row_logits, row_features = owner.model.forward_with_taps(
                    row_inputs, cache, owner.layers)
            logits.append(row_logits[:, -1:])
            features.append(row_features[:, -1:])
        # The commit replays host tokens; ``decide`` fills them after the read.
        transaction = _ReferenceTreeTransaction(owner, lane.cache, [], uid=lane.uid)
        return mx.concatenate(logits, axis=1), mx.concatenate(features, axis=1), transaction

    # -- the decision -----------------------------------------------------------
    def decide(self, owner, cohort, blocks, logits, features, transaction, phase):
        """One read (tokens with the verify output, and the ``tree_decide``
        packet), then the round decision.

        Returns host ``TreeDraftRow`` blocks and the ``RoundDecision``s."""
        from .external_speculative import _ReferenceTreeTransaction
        from .proposal_composition import LazyDecline

        mx = owner.mx
        (lane,), (block,) = cohort, blocks
        lazy = isinstance(block.tokens, mx.array)
        plan = None
        if self.policy.tree_decide:
            if float(lane.sampling.get("sampling_temp", 0) or 0) != 0.0:
                # Served lazily with the host decision (not a decline).
                skips = owner.scheduler_stats.setdefault(DECIDE_SKIPS, {})
                skips["sampled_lane"] = skips.get("sampled_lane", 0) + 1
            else:
                plan = self._plan(owner, lane, block, logits)
        if plan is not None:
            # The host decision marks these itself (``_tree_verify``).
            if phase is not None:
                phase("tree_target_launch")
            owner._presnapshot_draft(cohort)
        mx.eval(logits, features, *([block.tokens] if lazy else []),
                *([plan.packet] if plan is not None else []))
        _bump(owner.scheduler_stats, "external_device_tree_host_reads")
        if plan is not None and phase is not None:
            phase("tree_target_wait")
        tokens = ([int(token) for token in np.asarray(block.tokens).tolist()] if lazy
                  else [int(token) for token in block.tokens])
        vocab = int(logits.shape[-1])
        if any(not 0 <= token < vocab for token in tokens):
            # Restored by ``_tree_round``; the host round serves it as before.
            raise LazyDecline("draft_outside_vocabulary")
        host = dataclasses.replace(block, tokens=tokens, parents=list(block.parents))
        if isinstance(transaction, _ReferenceTreeTransaction):
            transaction.inputs = [int(lane.anchor), *tokens]
        if plan is None:
            decisions = owner._tree_verify(cohort, [host], logits, features, phase)
            _bump(owner.scheduler_stats, "external_device_tree_host_decide_lanes")
        else:
            decisions = [self._finish(owner, lane, host, plan, logits)]
            if phase is not None:
                phase("tree_target_law")
            _bump(owner.scheduler_stats, "external_device_tree_decide_lanes")
            self.runtime._lane_count(lane, "tree_decide_rounds")
        _bump(owner.scheduler_stats, "external_device_tree_rounds")
        self.runtime._lane_count(lane, "tree_rounds")
        return [host], decisions

    def _plan(self, owner, lane, block, logits):
        """The lazy greedy decision packet (``tree_decide``)."""
        from types import SimpleNamespace

        mx = owner.mx
        int32, f32 = mx.int32, mx.float32
        geo = geometry(tuple(block.parents))
        rows = geo.rows
        batched = bool(owner.tree_gates["batched_laws"] and owner._batched_law_contract(lane))
        emit = lane.sampling.get("emit_logprobs", True)
        keep = bool(emit or not owner.tree_gates["logprobs_on_request"])
        plan = SimpleNamespace(geo=geo, batched=batched, keep=keep, values=None, norms=None)
        if batched:
            # ``_launch_tree_laws``' greedy arithmetic.
            values = logits[0, :rows].astype(f32).astype(f32)
            selected = mx.argmax(values, axis=-1)
            if keep:
                norms = mx.logsumexp(values, axis=-1)
                valid = mx.isfinite(norms)
                plan.values, plan.norms = values, norms
            else:
                peak = mx.take_along_axis(values, selected[:, None], axis=-1)[:, 0]
                valid = mx.isfinite(peak) & ~mx.any(mx.isnan(values), axis=-1)
        else:
            # ``_target_law``'s per-row arithmetic.
            values, norms, picks = [], [], []
            for row in range(rows):
                value = logits[0, row][None].astype(f32)[0].astype(f32)
                values.append(value)
                norms.append(mx.logsumexp(value))
                picks.append(mx.argmax(value))
            selected = mx.stack(picks)
            valid = mx.isfinite(mx.stack(norms))
            plan.values, plan.norms = values, norms
        selected = selected.astype(int32)
        drafts = _device_tokens(mx, block.tokens)
        # A node is on the path when its token is its parent row's argmax,
        # no earlier sibling's is (``_walk_tree`` takes the first match), and
        # the same holds for every ancestor.
        match = drafts == mx.take(selected, mx.array(geo.parent_rows))
        earlier = mx.array(geo.earlier_siblings)
        first = match & ~mx.any(earlier & match[None, :], axis=1)
        on_path = mx.all(mx.where(mx.array(geo.ancestors), first[None, :], True), axis=1)
        on_rows = mx.concatenate([mx.array([True]), on_path])
        at_depth = mx.array(geo.depth_select) & on_rows[None, :]
        index = mx.arange(rows, dtype=int32)
        path = mx.where(mx.any(at_depth, axis=1),
                        mx.sum(mx.where(at_depth, index[None, :], 0), axis=1),
                        -1).astype(int32)
        plan.packet = mx.concatenate([path, selected, valid.astype(int32)])
        return plan

    def _finish(self, owner, lane, block, plan, logits):
        """The host walk of the packet's path: ``_verify_tree``'s decision."""
        from .external_speculative import (
            LaneFailure,
            RoundDecision,
            _tree_accepted_rows,
        )
        from .processor_probe import VerifyWindow

        geo = plan.geo
        packet = [int(value) for value in np.asarray(plan.packet).tolist()]
        depth = geo.max_depth + 1
        path = packet[:depth]
        selected = packet[depth: depth + geo.rows]
        valid = packet[depth + geo.rows:]
        vocab = int(logits.shape[-1])
        commit_rows, emitted = [], []
        for row in path:
            if row < 0:
                break
            if not valid[row]:
                raise LaneFailure(
                    lane.uid, "external draft target law is not a probability distribution")
            token = selected[row]
            if not 0 <= token < vocab:
                raise ValueError("device-tree token outside the vocabulary")
            # ``RequestRNG.sample`` on a one-hot law consumes one uniform.
            lane.rng.uniform()
            commit_rows.append(row)
            emitted.append(token)
            if token in owner.stops:
                break
        accepted_rows = _tree_accepted_rows(block, commit_rows, emitted)

        def one_hot(token):
            law = np.zeros(vocab)
            law[token] = 1
            return law

        if plan.batched:
            target_laws = [one_hot(t) for t in emitted] if plan.keep else None
            response = ([plan.values[row] - plan.norms[row] for row in commit_rows]
                        if plan.keep else None)
            window = None
        else:
            target_laws = [one_hot(t) for t in emitted]
            response = ([plan.values[row] - plan.norms[row] for row in commit_rows]
                        if plan.keep else None)
            window = VerifyWindow(lane.processors)
            for _ in commit_rows:
                window.mark()
        if not plan.keep:
            _bump(owner.scheduler_stats, "external_tree_logprob_rows_skipped", len(commit_rows))
        return RoundDecision(
            accepted=len(accepted_rows), accepted_rows=accepted_rows, emitted=emitted,
            target_laws=target_laws, response_logprobs=response, commit_rows=commit_rows,
            verify_window=window,
        )


def _device_tokens(mx, tokens):
    if isinstance(tokens, mx.array):
        return tokens.reshape(-1).astype(mx.int32)
    # A host tree (e.g. adopted from a previous round) is lifted as is.
    return mx.array([int(token) for token in tokens], dtype=mx.int32)


__all__ = ["COUNTERS", "DECIDE_SKIPS", "DECLINES", "SWITCHES", "VERSION", "DeviceTree", "LazyTreeProposal",
           "TreeGeometry", "chain_branch_parents", "decline", "engine_reason", "geometry", "run"]
