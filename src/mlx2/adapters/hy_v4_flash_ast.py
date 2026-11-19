"""Hy4-flash AST target-tree executor: one fused forward per tree chain.

Original mlx2 code; provider protocol of ``runtime/target_tree_executor.py``,
transaction shape of ``runtime/hybrid_tree_executor.py``.

A tile's rooted tree is split into maximal parent-to-child chains.  Each
chain runs as one ordinary multi-row forward (fused MLA, batched MoE) on a
private copy of the cache state of its source node, so a chain topology
verifies in a single target pass instead of one pass per node.  Positions
along a chain are contiguous, which keeps the compressed indexer's token
groups and the sliding window exactly the ordinary-decode ones.  A child
chain that leaves mid-chain starts from a copy of the parent chain trimmed
back to its source node (Hy4-flash caches trim exactly).  Committing a path
copies its terminal chain's state, trimmed to the terminal node, into the
live caches.

Cohorts (two to four lanes) and sibling chains run jointly: chains are
grouped into waves by chain depth (a wave's chains depend only on earlier
waves), and each wave with more than one chain runs as one forward over a
``HY4FlashBatchLayerCache`` whose rows are the chains' private caches,
right-padded with ``prepare(lengths=, right_padding=)``.  Attention stays per
row (each reads only its own history and span); projections, MoE, Engram and
the head run on the joined rows, so expert weights are read once per wave.
Commit and abort stay per lane and exact.  ``joint=False`` keeps the serial
reference (one forward per chain).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mlx.core as mx

from ..runtime.target_tree_executor import TargetTreeDispatchMetrics

PROTOCOL = "hy4_flash_chain_packed_exact_v1"


def clone_caches(caches, *, lend=False):
    """Private copies; ``lend`` hands the append-only buffers over (one fork)."""
    from ..runtime.models.hy_v4_flash import LAYER_CACHE_TYPES, HY4FlashBatchLayerCache

    out = []
    for cache in caches:
        if type(cache) is HY4FlashBatchLayerCache:
            out.append(HY4FlashBatchLayerCache([row.clone(lend=lend) for row in cache.rows]))
        elif type(cache) in LAYER_CACHE_TYPES:  # native or fp8 latent rows
            out.append(cache.clone(lend=lend))
        else:
            raise TypeError("Hy4-flash tree executor needs Hy4-flash layer caches")
    return out


def _replace(destination, source):
    if type(destination) is not type(source):
        raise TypeError("Hy4-flash tree cache type changed")
    destination.__dict__.clear()
    destination.__dict__.update(source.__dict__)


def chains(parents):
    """``[(source_node, [nodes...])]`` covering a parent-first rooted tree."""
    children = {}
    for row, parent in enumerate(parents):
        if type(parent) is not int or (row == 0 and parent != -1) or (row > 0 and not 0 <= parent < row):
            raise ValueError("AST parents must form one parent-first rooted tree")
        children.setdefault(parent, []).append(row)
    out, pending = [], [(-1, 0)]
    while pending:
        source, node = pending.pop(0)
        chain = [node]
        while children.get(chain[-1]):
            first, *others = children[chain[-1]]
            pending.extend((chain[-1], other) for other in others)
            chain.append(first)
        out.append((source, chain))
    return out


class Hy4ChainTransaction:
    def __init__(self, live, states, where, *, token, parents):
        self.live = live
        self.states = states  # chain index -> cache list after the whole chain
        self.where = where  # node -> (chain index, position in chain, chain length)
        self.parents = [int(p) for p in parents]
        self.closed = False
        self.executor_cached = False
        self.physical_packed = False
        self.provider_protocol = PROTOCOL
        self.provider_revision_token = token

    def validate_path(self, path):
        """A commit path is exactly the root-first ancestry of its last node."""
        if self.closed:
            raise RuntimeError("Hy4-flash tree transaction is closed")
        path = [int(n) for n in path]
        if not path or path[0] != 0 or any(not 0 <= n < len(self.parents) for n in path) or any(
                self.parents[child] != parent for parent, child in zip(path, path[1:])):
            raise ValueError("Hy4-flash tree commit path is not a root path of the tile")
        return path

    def commit_paths(self, paths):
        if self.closed or len(paths) != 1:
            raise RuntimeError("Hy4-flash tree transaction needs one nonempty path")
        terminal = self.validate_path(paths[0])[-1]
        try:
            chain, index, length = self.where[terminal]
            # The chain states are private to this transaction: hand them over.
            selected = self.states[chain]
            for cache in selected:
                cache.trim(length - 1 - index)
            for destination, source in zip(self.live, selected, strict=True):
                _replace(destination, source)
        finally:
            self.closed = True
        return [self.live]

    def commit(self, *, accepted_lengths):
        if len(accepted_lengths) != 1:
            raise RuntimeError("Hy4-flash tree transaction is single-lane")
        return self.commit_paths([list(range(int(accepted_lengths[0])))])

    def abort(self):
        self.closed = True


class Hy4ChainCohortTransaction:
    def __init__(self, transactions, *, token):
        self.transactions = transactions
        self.closed = False
        self.executor_cached = False
        self.physical_packed = False
        self.plain = False
        self.provider_protocol = PROTOCOL
        self.provider_revision_token = token

    def commit_paths(self, paths):
        if self.closed or len(paths) != len(self.transactions):
            raise RuntimeError("Hy4-flash tree cohort path count mismatch")
        caches = []
        try:
            # Every lane's path is checked before any lane commits.
            for transaction, path in zip(self.transactions, paths, strict=True):
                transaction.validate_path(path)
            for transaction, path in zip(self.transactions, paths, strict=True):
                caches.extend(transaction.commit_paths([path]))
        finally:
            self.closed = True
        return caches

    def commit(self, *, accepted_lengths):
        return self.commit_paths([list(range(int(n))) for n in accepted_lengths])

    def abort(self):
        if not self.closed:
            for transaction in self.transactions:
                transaction.abort()
            self.closed = True


def _plan(parents):
    """``[(source, nodes, depth)]``: chains with their wave (chain depth)."""
    plan = chains([int(p) for p in parents])
    owner, out = {}, []
    for number, (source, nodes) in enumerate(plan):
        depth = 0 if source < 0 else out[owner[source]][2] + 1
        for node in nodes:
            owner[node] = number
        out.append((source, nodes, depth))
    return out


def _row_caches(caches):
    """The per-layer ``HY4FlashLayerCache`` rows of one chain's private state."""
    from ..runtime.models.hy_v4_flash import HY4FlashBatchLayerCache

    rows = []
    for cache in caches:
        if type(cache) is HY4FlashBatchLayerCache:
            if len(cache.rows) != 1:
                raise ValueError("Hy4-flash tree lane cache must hold exactly one row")
            cache = cache.rows[0]
        rows.append(cache)
    return rows


def _wave_forward(model, jobs, layers, joint):
    """Run one wave's chains; ``jobs`` are ``(tokens, private caches)``.

    Returns ``([(logits [1, L, V], features [1, L, D])], packed)``.
    """
    from ..runtime.models.hy_v4_flash import HY4FlashBatchLayerCache

    if len(jobs) == 1 or not joint:
        return [model.forward_with_taps(mx.array([tokens], dtype=mx.uint32), private, layers)
                for tokens, private in jobs], False
    lengths = [len(tokens) for tokens, _ in jobs]
    width = max(lengths)
    # Padding repeats the row's last real token; padded positions are never
    # attended, written to a row cache, hashed into Engram, or read back.
    padded = [list(tokens) + [tokens[-1]] * (width - len(tokens)) for tokens, _ in jobs]
    rows = [_row_caches(private) for _, private in jobs]
    batch = []
    for layer in range(len(rows[0])):
        cache = HY4FlashBatchLayerCache([lane[layer] for lane in rows])
        cache.prepare(lengths=lengths, right_padding=[width - n for n in lengths])
        batch.append(cache)
    logits, features = model.forward_with_taps(mx.array(padded, dtype=mx.uint32), batch, layers)
    for cache in batch:
        cache.finalize()
    return [(logits[j : j + 1, :n], features[j : j + 1, :n]) for j, n in enumerate(lengths)], True


def _check_independent_lanes(lives):
    """Each root chain lends its lane's append-only buffers and writes past
    that lane's fill level; two lanes sharing a buffer object would write the
    same rows.  Refuse a cohort whose lanes share any cache or lent buffer."""
    from ..runtime.models.hy_v4_flash import LAYER_CACHE_TYPES

    caches, buffers = set(), set()
    for lane in lives:
        for cache in _row_caches(lane):
            if type(cache) not in LAYER_CACHE_TYPES:  # native or fp8 latent rows
                raise TypeError("Hy4-flash tree executor needs Hy4-flash layer caches")
            if id(cache) in caches:
                raise ValueError("Hy4-flash tree cohort lanes share a cache")
            caches.add(id(cache))
            for name, _, _ in cache._RECOVERY_APPEND_ONLY_FIELDS:
                data = getattr(cache, name, None)
                if data is not None:
                    if id(data) in buffers:
                        raise ValueError("Hy4-flash tree cohort lanes share an append-only buffer")
                    buffers.add(id(data))


def cohort_forward(model, tokens, parents, lives, layers, token, *, joint=True):
    """Verify each lane's tile; returns ``(logits, features, transactions, stats)``.

    ``logits``/``features`` are per lane ``[1, rows, ...]`` in node order.
    """
    if len(tokens) != len(parents) or len(tokens) != len(lives) or not tokens:
        raise ValueError("Hy4-flash AST cohort inputs disagree")
    for row, tree in zip(tokens, parents, strict=True):
        if not 1 <= len(row) <= 32 or len(row) != len(tree):
            raise ValueError("Hy4-flash AST tile must contain 1..32 matching rows")
    _check_independent_lanes(lives)
    plans = [_plan(tree) for tree in parents]
    states = [[None] * len(plan) for plan in plans]
    where = [{} for _ in plans]
    logits = [{} for _ in plans]
    features = [{} for _ in plans]
    stats = {"waves": 0, "joint_waves": 0, "joint_chains": 0, "cross_lane_waves": 0,
             "chain_forwards": 0}
    for depth in range(max(d for plan in plans for _, _, d in plan) + 1):
        jobs, owners = [], []
        for lane, plan in enumerate(plans):
            for number, (source, nodes, d) in enumerate(plan):
                if d != depth:
                    continue
                if source < 0:
                    # The root chain borrows the live append-only buffers: it
                    # writes only past the live fill level, so an abort leaves
                    # the lane exact and its appends copy nothing.  Child
                    # chains alias (copy on write).
                    private = clone_caches(lives[lane], lend=True)
                else:
                    chain, index, length = where[lane][source]
                    private = clone_caches(states[lane][chain])
                    for cache in private:
                        cache.trim(length - 1 - index)
                jobs.append(([int(tokens[lane][n]) for n in nodes], private))
                owners.append((lane, number, nodes))
        results, packed = _wave_forward(model, jobs, layers, joint)
        stats["waves"] += 1
        stats["chain_forwards"] += 1 if packed else len(jobs)
        if packed:
            stats["joint_waves"] += 1
            stats["joint_chains"] += len(jobs)
            # A wave holding chains of several lanes: the cohort's projection,
            # MoE and head work ran in one forward.
            stats["cross_lane_waves"] += len({lane for lane, _, _ in owners}) > 1
        for (lane, number, nodes), (row_logits, row_features), (_, private) in zip(
                owners, results, jobs, strict=True):
            for i, node in enumerate(nodes):
                logits[lane][node] = row_logits[:, i : i + 1]
                features[lane][node] = row_features[:, i : i + 1]
                where[lane][node] = (number, i, len(nodes))
            states[lane][number] = private
    out_logits = [mx.concatenate([logits[lane][n] for n in range(len(tokens[lane]))], axis=1)
                  for lane in range(len(tokens))]
    out_features = [mx.concatenate([features[lane][n] for n in range(len(tokens[lane]))], axis=1)
                    for lane in range(len(tokens))]
    transactions = [Hy4ChainTransaction(lives[lane], states[lane], where[lane], token=token,
                                        parents=parents[lane])
                    for lane in range(len(tokens))]
    return out_logits, out_features, transactions, stats


def _stack(rows):
    """Concatenate per-lane ``[1, n_i, ...]`` rows, zero-padding to the widest."""
    width = max(row.shape[1] for row in rows)
    return mx.concatenate([row if row.shape[1] == width else mx.pad(
        row, [(0, 0), (0, width - row.shape[1])] + [(0, 0)] * (row.ndim - 2)) for row in rows], axis=0)


def chain_forward(model, tokens, parents, live, layers, token, *, joint=True):
    logits, features, transactions, _ = cohort_forward(
        model, [tokens], [parents], [live], layers, token, joint=joint)
    return logits[0], features[0], transactions[0]


class Hy4ChainTreeExecutor:
    """Target-tree provider: widths 1..4, waves packed jointly by default."""

    protocol = PROTOCOL
    maximum_cohort_width = 4
    # The engine's anchor-only (ordinary probe / parked) rounds run here too:
    # the generic hybrid owner has no segmented view of Hy4 private rows.
    plain_anchor_protocol = "private-ordinary-anchor-v1"
    plain_anchor_max_width = 4
    # Lanes' trees arrive unpadded (no filler leaves, which would each be an
    # extra chain and wave); output rows are zero-padded to the widest lane
    # and never read past a lane's own tree.
    accepts_unequal_tree_widths = True

    def __init__(self, *, joint=True):
        if type(joint) is not bool:
            raise ValueError("Hy4-flash tree executor joint must be a boolean")
        self.joint = joint
        self.source_revision_token = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.source_identity = (("module", "mlx2.adapters.hy_v4_flash_ast"),
                                ("sha256", self.source_revision_token),
                                ("joint", str(joint).lower()))
        self.dispatches = 0
        self.cohort_dispatches = 0
        self.cohort_lanes = 0
        self.chain_forwards = 0
        self.joint_waves = 0
        self.joint_chains = 0
        self.joint_cohort_waves = 0
        self.rows = 0
        self.tree_cohort_packed_dispatches = 0  # tree cohorts that packed several lanes
        self.plain_anchor_dispatches = 0
        self.plain_anchor_lanes = 0
        self.plain_anchor_joint_steps = 0

    @classmethod
    def create(cls, *, joint=True):
        return cls(joint=joint)

    def validate_revision(self):
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != self.source_revision_token:
            raise RuntimeError("Hy4-flash tree executor source changed")

    def validate_cohort(self, width, configured_limit):
        if type(width) is not int or not 1 <= width <= min(self.maximum_cohort_width, configured_limit):
            raise ValueError("Hy4-flash tree cohort exceeds its bound")

    def _run(self, model, inputs, parents, caches, layers, *, plain=False):
        logits, features, transactions, stats = cohort_forward(
            model, inputs, parents, caches, layers, self.source_revision_token, joint=self.joint)
        self.chain_forwards += stats["chain_forwards"]
        self.joint_waves += stats["joint_waves"]
        self.joint_chains += stats["joint_chains"]
        if plain:
            self.plain_anchor_joint_steps += stats["cross_lane_waves"]
        else:
            self.joint_cohort_waves += stats["cross_lane_waves"]
        packed = stats["cross_lane_waves"] > 0
        self.rows += sum(len(row) for row in inputs)
        return logits, features, transactions, packed

    def forward(self, model, inputs, parents, cache, layers, *, cached):
        self.validate_revision()
        logits, features, transactions, _ = self._run(model, [inputs], [parents], [cache], layers)
        self.dispatches += 1
        return logits[0], features[0], transactions[0]

    def forward_many(self, model, inputs, parents, caches, layers, *, cached, configured_limit,
                     ragged=False):
        self.validate_revision()
        if ragged:
            raise ValueError("Hy4-flash tree executor has no ragged row-sharing layout")
        self.validate_cohort(len(caches), configured_limit)
        logits, features, transactions, packed = self._run(model, inputs, parents, caches, layers)
        self.dispatches += 1
        self.cohort_dispatches += 1
        self.cohort_lanes += len(caches)
        transaction = Hy4ChainCohortTransaction(transactions, token=self.source_revision_token)
        transaction.physical_packed = packed
        self.tree_cohort_packed_dispatches += packed
        return _stack(logits), _stack(features), transaction

    def forward_plain_anchors(self, model, inputs, caches, layers):
        """One ordinary decode step for 1..4 lanes (anchors ``[n, 1]``), jointly."""
        self.validate_revision()
        count = len(caches)
        if not 1 <= count <= self.plain_anchor_max_width or tuple(inputs.shape) != (count, 1):
            raise ValueError("Hy4-flash plain anchors need one token per bounded lane")
        identities = [id(item) for cache in caches for item in cache]
        if len(set(identities)) != len(identities):
            raise ValueError("Hy4-flash plain anchors need independent lane caches")
        tokens = [[int(t)] for t in inputs[:, 0].tolist()]
        logits, features, transactions, packed = self._run(
            model, tokens, [[-1]] * count, caches, layers, plain=True)
        self.plain_anchor_dispatches += 1
        self.plain_anchor_lanes += count
        transaction = Hy4ChainCohortTransaction(transactions, token=self.source_revision_token)
        transaction.physical_packed = packed
        transaction.plain = True
        return mx.concatenate(logits, axis=0), mx.concatenate(features, axis=0), transaction

    def dispatch_metrics(self, width, transaction, *, cached):
        self.validate_cohort(width, self.maximum_cohort_width)
        packed = width > 1 and getattr(transaction, "physical_packed", False) is True
        return TargetTreeDispatchMetrics(width=width, physical_packed=packed, executor_cached=False,
                                         cached=bool(cached), provider_protocol=self.protocol)

    def receipt(self):
        return {
            "provider": self.protocol,
            "source_identity": dict(self.source_identity),
            "source_revision_token": self.source_revision_token,
            "joint": self.joint,
            "dispatches": self.dispatches,
            "cohort_dispatches": self.cohort_dispatches,
            "cohort_lanes": self.cohort_lanes,
            "chain_forwards": self.chain_forwards,
            "joint_waves": self.joint_waves,
            "joint_chains": self.joint_chains,
            "joint_cohort_waves": self.joint_cohort_waves,
            "rows": self.rows,
            "tree_cohort_packed_dispatches": self.tree_cohort_packed_dispatches,
            "plain_anchor_dispatches": self.plain_anchor_dispatches,
            "plain_anchor_lanes": self.plain_anchor_lanes,
            "plain_anchor_joint_steps": self.plain_anchor_joint_steps,
            "implemented": True,
            "qualified": False,
            "selected": self.dispatches > 0,
            "observed_used": self.dispatches > 0,
        }
