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
``tree_fused`` (needs ``tree_decide``; ``speculative_device_tree_fused``)
    The verify input is one Metal kernel reading the drafter's branch grid
    (``LazyTreeProposal.grid``/``index``; any other lazy or adopted tree
    reads its token row), and a greedy lane's decision is one Metal kernel
    over the row law (argmax and finiteness); the host walks the path from
    the read argmaxes.  About 35 small device operations per round become 3
    (``scripts/count_device_tree_ops.py``); tokens, RNG draws and response
    rows are the ``tree_decide`` round's.  Counted under
    ``external_device_tree_fused_lanes``.

Host-searched lazy trees (DFlash2, ``drafters/dflash2.py``): a drafter whose
tree topology is a data-dependent host search returns ``LazyTreeProposal``
with ``host_search=True``: the tokens are the host search's, lifted to the
device, so the round keeps that one draft-side read (counted under
``external_device_tree_host_search_lanes``).  Such a drafter may declare
``draft_tree_lazy_pipelines``: its pipelined lattices (``pipeline_draft``,
adopted by the next round as host trees) are then the same contract and are
served by the device round (``external_device_tree_adopted_lanes``) instead
of declining.  ``MLX2_TREE_PIPELINE_DRAFT=0`` is the serial control arm.  A
drafter declaring ``draft_tree_lazy_greedy_only`` declines sampled lanes
(``sampled_lane``).

``processors`` (the chain round's switch, admitted here with ``tree_lazy``)
    Lanes whose logits processors are all stateless and history-pure
    (``processors_ok``: the presence, frequency and repetition penalties of
    vendor sampling profiles) keep the device tree round.  Under
    ``tree_lazy`` alone the host decision applies them after the one read, as
    before; under ``tree_decide`` each row's law is processed on the device
    on that row's own history (lane history plus its root-to-node inputs,
    the host verifier's ``history_suffix``), in whichever form the host
    verifier would use (``_launch_tree_laws`` batched masks or
    ``_target_law``'s per-row processor calls).  Forbidden roots are the
    host's (a nonempty set declines the lazy draft, ``forbidden_root``).

What a round cannot serve (width above one, processors without the
``processors`` switch or that are not stateless, tree calibration,
draft pipelining by a drafter that does not declare it, a verify trace, a
provider without device inputs, a drafter without ``draft_tree_lazy``) runs
the host tree round, counted under ``external_device_tree_decline_reasons``.
A drafter that cannot draft a given round lazily raises ``LazyDecline``; the
round is restored and redone on the host (counted under the drafter's reason).
The drafter's ``stats`` and the integer counters of each attribute it names in
``draft_tree_lazy_counter_maps`` are restored with it.
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass

import numpy as np

from .technologies.tree_verify import TopologySpec, TreeTopology

VERSION = "speculative-device-tree-v1"
SWITCHES = ("tree_lazy", "tree_decide", "tree_fused")
COUNTERS = (
    "external_device_tree_rounds",
    "external_device_tree_lazy_lanes",
    "external_device_tree_host_reads",
    "external_device_tree_decide_lanes",
    "external_device_tree_host_decide_lanes",
    "external_device_tree_declines",
    # Lazy proposals whose tokens came from a host search (``host_search``),
    # and pipelined host trees adopted by a device round (see above).
    "external_device_tree_host_search_lanes",
    "external_device_tree_adopted_lanes",
)
DECLINES = "external_device_tree_decline_reasons"
# Device tree rounds whose decision stayed on the host under ``tree_decide``.
DECIDE_SKIPS = "external_device_tree_decide_skips"
_TOPOLOGY = TopologySpec(max_rows=None)


@dataclass(frozen=True)
class LazyTreeProposal:
    """A drafter's tree with its tokens still on the device.

    ``tokens``: int32 ``[n]`` device array in node order; ``parents``: the
    static parent-first topology (``-1`` = a child of the anchor), the same
    ``(tokens, parents)`` contract as ``propose_tree``."""

    tokens: object
    parents: tuple
    # The topology came from a host search over the drafter's device output
    # (one draft-side read, e.g. DFlash2's best-first lattice search).
    host_search: bool = False
    # Optional (``tree_fused``): the drafter's own token grid and the flat
    # position of each node in it, so ``tokens`` is ``grid.reshape(-1)[index]``
    # and the fused verify-row kernel reads the grid directly (native MTP:
    # the ``[branches, depth]`` branch grid; the drafter's slices,
    # concatenate and cast are then never evaluated).
    grid: object = None
    index: tuple = ()


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
    topology = TreeTopology(target, _TOPOLOGY)
    rows = len(topology)
    depth, paths = topology.depths, topology.paths
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
    return TreeGeometry(target, rows, max_depth, paths, parent_rows, earlier,
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
    if owner._draft_pipelines() and getattr(draft, "draft_tree_lazy_pipelines", False) is not True:
        # Pipelined lattices are host trees adopted from the previous round;
        # only a drafter whose lazy tree is a host tree anyway declares them.
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
        if lane.processors and not processors_ok(owner, lane):
            # Without the ``processors`` switch, or for a processor that is not
            # stateless and history-pure (a grammar, a verify ledger), the
            # host tree round applies them.
            return "processors"
        if getattr(draft, "draft_tree_lazy_greedy_only", False) is True and float(
                lane.sampling.get("sampling_temp", 0) or 0) != 0.0:
            return "sampled_lane"
    return None


def processors_ok(owner, lane):
    """``processors`` switch on a tree round: every processor is stateless.

    A stateless processor (``sample_utils._probe_safe``: ``history_pure`` and
    its own ``probe``, e.g. the presence, frequency and repetition penalties)
    or one declaring the stateless branch contract (``progressive_branch_safe``
    and ``history_pure``) is a pure function of the history it is given, so
    the round applies it to each tree row's own root-to-node history (the
    host tree verifier's ``history_suffix``), with the draft tokens still on
    the device.  Its forbidden roots stay the host's (``_tree_forbidden``; a
    nonempty set declines the lazy draft with ``forbidden_root``).  A verify
    ledger snapshots per-row state through ``VerifyWindow``: declined.
    """
    policy = getattr(getattr(owner, "device_round", None), "policy", None)
    if policy is None or not policy.processors:
        return False
    return all(
        getattr(processor, "history_pure", False) is True
        and (getattr(processor, "probe", None) is processor
             or getattr(processor, "progressive_branch_safe", False) is True)
        and not callable(getattr(processor, "verify_ledger", None))
        for processor in lane.processors
    )


def _twin_overridden(draft):
    """A more specific host operation needs its own matching lazy hook."""
    instance = getattr(draft, "__dict__", {})
    if "draft_tree_lazy" in instance:
        return False
    twins = getattr(draft, "draft_tree_lazy_host_twins", ("propose_tree",))
    if any(name in instance for name in twins):
        return True
    for cls in type(draft).__mro__:
        if "draft_tree_lazy" in vars(cls):
            return False
        if any(name in vars(cls) for name in twins):
            return True
    # A dynamically forwarded hook has no declared host twin on this owner.
    return True


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
    counter_maps = _counter_maps(owner.draft)
    counter_snapshots = [{key: value for key, value in state.items() if type(value) is int}
                         for state in counter_maps]
    try:
        return owner._tree_round(cohort, device_tree=DeviceTree(runtime))
    except LazyDecline as error:
        # ``_tree_round`` restored the lanes and statistics before re-raising.
        if draft_snapshot is not None:
            owner.draft.stats.clear()
            owner.draft.stats.update(draft_snapshot)
        for counters, snapshot in zip(counter_maps, counter_snapshots, strict=True):
            counters.update(snapshot)
        decline(owner, error.reason, len(cohort))
        return owner._tree_round(cohort)


def _counter_maps(draft):
    """The drafter's declared counter maps (e.g. DFlash2's ``_draft_vocab``
    reduced/full head rounds), restored on a ``LazyDecline`` like ``stats``."""
    names = getattr(draft, "draft_tree_lazy_counter_maps", ())
    states = [getattr(draft, "__dict__", {}).get(name) for name in names]
    return [state for state in states if type(state) is dict]


class DeviceTree:
    """One device tree round's hooks into ``_propose`` and ``_tree_round``."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.policy = runtime.policy
        # ``tree_fused``: the proposal's grid and node index, and the raw
        # drafts the fused verify row read (the host reads these).
        self.fused = bool(getattr(self.policy, "tree_fused", False))
        self._fused_source = None
        self._fused_raw = None

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
        if tokens.ndim != 1 or int(tokens.shape[0]) != len(parents):
            raise RuntimeError("lazy tree proposal tokens and parents differ")
        geometry(parents)  # validates parent-first order
        grid = getattr(proposal, "grid", None)
        if self.fused and grid is not None:
            index = tuple(int(value) for value in proposal.index)
            if len(index) != len(parents) or any(
                    not 0 <= value < int(grid.size) for value in index):
                raise RuntimeError("lazy tree proposal grid index and parents differ")
            self._fused_source = (grid, index)
        _bump(owner.scheduler_stats, "external_device_tree_lazy_lanes")
        if getattr(proposal, "host_search", False):
            _bump(owner.scheduler_stats, "external_device_tree_host_search_lanes")
        self.runtime._lane_count(lanes[0], "tree_lazy_rounds")
        return [(tokens, list(parents))]

    # -- the forward ------------------------------------------------------------
    def forward(self, owner, cohort, blocks):
        """The verify input on the device and the target-tree forward on it.

        Returns ``(inputs, logits, features, transaction)``."""
        from .external_speculative import _amoeba_rows

        mx = owner.mx
        (lane,), (block,) = cohort, blocks
        if not isinstance(block.tokens, mx.array):
            # Only a pipelined lattice (``draft_tree_lazy_pipelines``) reaches
            # the device round as a host tree.
            _bump(owner.scheduler_stats, "external_device_tree_adopted_lanes")
            self.runtime._lane_count(lane, "tree_adopted_rounds")
        geo = geometry(tuple(block.parents))
        if self.fused:
            # One launch: [anchor] + clip(drafts) and the raw drafts.
            from . import speculative_device_tree_fused as fused

            source = self._fused_source
            if source is None or len(source[1]) != len(block.parents) or not isinstance(
                    block.tokens, mx.array):
                # No grid, or the block is not the proposal's tree any more
                # (an adopted host tree): its own token row.
                count = len(block.parents)
                source = (_device_tokens(mx, block.tokens), tuple(range(count)))
            inputs, self._fused_raw = fused.verify_row(
                mx, source[0], source[1], int(lane.anchor),
                owner._target_token_vocab_size - 1)
        else:
            inputs = mx.concatenate([
                mx.array([int(lane.anchor)], dtype=mx.int32),
                mx.clip(_device_tokens(mx, block.tokens), 0, owner._target_token_vocab_size - 1),
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
        transaction = _ReferenceTreeTransaction(owner, lane.cache, [], geo.paths, uid=lane.uid)
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
        # ``tree_fused``: the verify row's raw drafts stand for the
        # drafter's token array (which then never runs).
        raw = self._fused_raw if lazy else None
        mx.eval(logits, features, *([block.tokens if raw is None else raw] if lazy else []),
                *([plan.packet, *getattr(plan, "fence", ())] if plan is not None else []))
        _bump(owner.scheduler_stats, "external_device_tree_host_reads")
        if plan is not None and phase is not None:
            phase("tree_target_wait")
        tokens = ([int(token) for token in np.asarray(
                       block.tokens if raw is None else raw).tolist()] if lazy
                  else [int(token) for token in block.tokens])
        vocab = min(owner._target_token_vocab_size, int(logits.shape[-1]))
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
            if plan.fused:
                from .speculative_device_tree_fused import COUNTER

                _bump(owner.scheduler_stats, COUNTER)
        if any(not 0 <= token < vocab for d in decisions for token in d.emitted):
            raise ValueError("device-tree token outside the vocabulary")
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
        fused = self.fused
        plan = SimpleNamespace(geo=geo, batched=batched, keep=keep, values=None, norms=None,
                               fused=fused)
        processors = getattr(lane, "processors", None) or ()
        # ``tree_fused``: the fused verify row's raw drafts (same values as
        # the drafter's token array, which then never runs).
        drafts = (self._fused_raw if fused and self._fused_raw is not None
                  else _device_tokens(mx, block.tokens))
        # Row r's verify input path is [anchor, node tokens...]; its processor
        # history is the lane history plus those tokens (``history_suffix``).
        # Clipped like the verify input: an out-of-vocabulary draft declines the
        # whole round after the read (``draft_outside_vocabulary``), and must
        # not index outside the logits before it.
        inputs = mx.concatenate([
            mx.array([int(lane.anchor)], dtype=int32),
            mx.clip(drafts, 0, int(logits.shape[-1]) - 1),
        ]) if processors else None
        if fused:
            return self._plan_fused(owner, lane, plan, logits, inputs, processors)
        if batched:
            # ``_launch_tree_laws``' greedy arithmetic.
            values = logits[0, :rows].astype(f32)
            if processors:
                values = _batched_processed_rows(mx, lane, geo, inputs, values)
            values = values.astype(f32)
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
            history = mx.array(list(lane.history), dtype=int32) if processors else None
            for row in range(rows):
                value = logits[0, row][None].astype(f32)
                if processors:
                    # ``_target_law``: the lane's processors in order on
                    # ``history + path tokens``, the drafts still lazy.
                    tokens = mx.concatenate([
                        history, mx.take(inputs, mx.array(geo.paths[row], dtype=int32))])
                    for processor in processors:
                        value = processor(tokens, value)
                value = value[0].astype(f32)
                values.append(value)
                norms.append(mx.logsumexp(value))
                picks.append(mx.argmax(value))
            selected = mx.stack(picks)
            valid = mx.isfinite(mx.stack(norms))
            plan.values, plan.norms = values, norms
        selected = selected.astype(int32)
        path = self._accepted_path(mx, geo, drafts, selected)
        plan.packet = mx.concatenate([path, selected, valid.astype(int32)])
        return plan

    @staticmethod
    def _accepted_path(mx, geo, tokens, selected):
        if geo.rows == 1:
            # A model-free proposal can have no matching n-gram.
            return mx.array([0], dtype=mx.int32)
        drafts = _device_tokens(mx, tokens)
        # Match the parent argmax and keep the first matching sibling.
        match = drafts == mx.take(selected, mx.array(geo.parent_rows))
        earlier = mx.array(geo.earlier_siblings)
        first = match & ~mx.any(earlier & match[None, :], axis=1)
        on_path = mx.all(mx.where(mx.array(geo.ancestors), first[None, :], True), axis=1)
        on_rows = mx.concatenate([mx.array([True]), on_path])
        at_depth = mx.array(geo.depth_select) & on_rows[None, :]
        index = mx.arange(geo.rows, dtype=mx.int32)
        return mx.where(mx.any(at_depth, axis=1),
                        mx.sum(mx.where(at_depth, index[None, :], 0), axis=1),
                        -1).astype(mx.int32)

    @staticmethod
    def _plan_fused(owner, lane, plan, logits, inputs, processors):
        """``_plan``'s values with the decision as one ``decide_rows`` launch.

        The row law the kernel reads is the unfused round's (bf16 logits
        without processors, whose float32 argmax and finiteness are the
        cast's; the processed float32 rows otherwise).  Response rows keep
        the unfused lazy ``values - logsumexp`` arrays (only when ``keep``),
        evaluated when a response reads them."""
        from . import speculative_device_tree_fused as fused

        mx = owner.mx
        f32 = mx.float32
        geo = plan.geo
        rows = geo.rows
        raw = logits[0, :rows]
        if plan.batched:
            values = raw.astype(f32)
            if processors:
                values = _batched_processed_rows(mx, lane, geo, inputs, values).astype(f32)
            law = values if processors else raw
            if plan.keep:
                plan.values, plan.norms = values, mx.logsumexp(values, axis=-1)
                # In the round's one read, as the unfused packet has them.
                plan.fence = (plan.norms,)
        else:
            values = []
            history = mx.array(list(lane.history), dtype=mx.int32) if processors else None
            for row in range(rows):
                value = logits[0, row][None].astype(f32)
                if processors:
                    tokens = mx.concatenate([
                        history, mx.take(inputs, mx.array(geo.paths[row], dtype=mx.int32))])
                    for processor in processors:
                        value = processor(tokens, value)
                values.append(value[0].astype(f32))
            law = mx.stack(values) if processors else raw
            if plan.keep:
                plan.values, plan.norms = values, [mx.logsumexp(value) for value in values]
                plan.fence = tuple(plan.norms)
        plan.packet = fused.decide_rows(mx, law)
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
        if getattr(plan, "fused", False):
            from .speculative_device_tree_fused import walk

            selected, valid = packet[:geo.rows], packet[geo.rows:]
            path = walk(geo.target_parents, [int(token) for token in block.tokens], selected)
        else:
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


def _batched_processed_rows(mx, lane, geo, inputs, rows):
    """``_launch_tree_laws``' processor arithmetic with the path tokens on the device.

    The lane's processors in order, each on every row's own history (lane
    history plus the row's root-to-node inputs): a presence window
    (``presence_window``) subtracts its penalty on the distinct tokens of
    ``_history_window(history, path tokens, context, start)``, the history
    part from host ids and the path part gathered from ``inputs``; a
    length-only processor masks its forbidden ids per row length.  Only the
    two kinds ``_batched_law_contract`` admits reach here.  The window bounds
    are ``external_speculative._history_window``'s."""
    width, vocab = geo.rows, int(rows.shape[-1])
    history = list(lane.history)
    boundary = len(history)
    lengths = [boundary + len(path) for path in geo.paths]
    for processor in lane.processors:
        presence = getattr(processor, "presence_window", None)
        if presence is not None:
            penalty, context_size, start = presence
            host_flat, device_rows, device_columns = [], [], []
            for row, path in enumerate(geo.paths):
                length = boundary + len(path)
                # The window's first index in history + suffix (as
                # ``_history_window`` slices it).
                first = slice(start, None).indices(length)[0]
                first += slice(-context_size, None).indices(length - first)[0]
                if first < boundary:
                    host_flat.extend(row * vocab + int(token) for token in set(history[first:]))
                for position in range(max(0, first - boundary), len(path)):
                    device_rows.append(row)
                    device_columns.append(path[position])
            pieces = []
            if host_flat:
                pieces.append(mx.array(host_flat, dtype=mx.int32))
            if device_rows:
                pieces.append(mx.array(device_rows, dtype=mx.int32) * vocab
                              + mx.take(inputs, mx.array(device_columns, dtype=mx.int32)))
            if pieces:
                mask = mx.zeros((width * vocab,), dtype=mx.bool_)
                mask[mx.concatenate(pieces)] = True
                rows = mx.where(mask.reshape(width, vocab), rows - penalty, rows)
            continue
        forbidden = [sorted({int(token) for token in processor.forbidden_token_ids_at_length(length)})
                     for length in lengths]
        count = max((len(ids) for ids in forbidden), default=0)
        if count:
            table = mx.array([ids + [-1] * (count - len(ids)) for ids in forbidden], dtype=mx.int32)
            mask = mx.any(mx.arange(vocab)[None, None, :] == table[:, :, None], axis=1)
            rows = mx.where(mask, -float("inf"), rows)
    return rows


def _device_tokens(mx, tokens):
    if isinstance(tokens, mx.array):
        return tokens.reshape(-1).astype(mx.int32)
    # A host tree (e.g. adopted from a previous round) is lifted as is.
    return mx.array([int(token) for token in tokens], dtype=mx.int32)


__all__ = ["COUNTERS", "DECIDE_SKIPS", "DECLINES", "SWITCHES", "VERSION", "DeviceTree", "LazyTreeProposal",
           "TreeGeometry", "chain_branch_parents", "decline", "engine_reason", "geometry", "run"]
