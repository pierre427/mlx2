"""Exact serial target-tree executor for mixed KV/recurrent model caches.

This is the correctness-first AST reference path.  Every node executes from a
private descriptor snapshot of its parent's exact state.  It deliberately makes no
throughput claim; a future parent-indexed fused kernel can replace it behind
the same provider contract after qualification.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .target_tree_executor import TargetTreeDispatchMetrics
from .technologies.tree_verify import (
    SnapshotBranchOps,
    TECHNOLOGY_SOURCES,
    SourceBoundRevision,
    TreeLaneTransaction,
    TreeTopology,
    legacy_topology,
    revision_token,
)

PROTOCOL = "hybrid_tree_serial_exact_v1"
TOPOLOGY = legacy_topology("hybrid_serial")
_SNAPSHOT = SnapshotBranchOps()


class _SerialLane:
    """One request's private per-node snapshots (the serial reference)."""

    def __init__(self, live, branches):
        self.live = live
        self.branches = branches

    def validate_path(self, path):
        # The serial lane publishes its terminal node's snapshot; it only
        # requires a nonempty path that ends inside the tile.
        if not path:
            raise RuntimeError("hybrid tree transaction needs one nonempty path")
        if not 0 <= int(path[-1]) < len(self.branches):
            raise ValueError("hybrid tree commit path is outside the tile")

    def prepare_commit(self, path):
        return _SNAPSHOT.prepare_commit(self.live, self.branches, int(path[-1]))


class HybridTreeTransaction(TreeLaneTransaction):
    protocol = PROTOCOL
    count_error = "hybrid tree transaction needs one nonempty path"

    def __init__(self, live, branches, *, token):
        self.live = live
        self.branches = branches
        super().__init__([_SerialLane(live, branches)], token)

    def commit(self, *, accepted_lengths):
        if len(accepted_lengths) != 1:
            raise RuntimeError("hybrid tree transaction is single-lane")
        return self.commit_paths([list(range(int(accepted_lengths[0])))])


class HybridTreeCohortTransaction(TreeLaneTransaction):
    """All lanes validate and prepare before any lane publishes."""

    protocol = PROTOCOL
    count_error = "hybrid tree cohort path count mismatch"
    # Legacy: lanes were validated inside the cohort's try/finally.
    close_on_refused_path = True

    def __init__(self, transactions, *, token):
        self.transactions = transactions
        super().__init__([lane for item in transactions for lane in item.trees], token)

    def _close(self):
        super()._close()
        for transaction in self.transactions:
            transaction._close()


def tree_forward(model, tokens, parents, live, layers, token):
    if len(tokens) != len(parents):
        raise ValueError(TOPOLOGY.rows_error)
    topology = TreeTopology(parents, TOPOLOGY)
    logits, features, branches = _SNAPSHOT.forward_rows(
        model, tokens, topology.parents, live, layers
    )
    return logits, features, HybridTreeTransaction(live, branches, token=token)


class HybridTreeExecutor(SourceBoundRevision):
    protocol = PROTOCOL
    maximum_cohort_width = 4
    _identity_style = "module"
    _source_module = "mlx2.runtime.hybrid_tree_executor"
    _revision_error = "hybrid tree executor source changed"

    def __init__(self):
        self._bind_sources((Path(__file__), *TECHNOLOGY_SOURCES))
        self.source_revision_token = revision_token(self.source_identity)
        self.dispatches = 0

    @classmethod
    def create(cls):
        return cls()

    def validate_cohort(self, width, configured_limit):
        if type(width) is not int or not 1 <= width <= min(
            self.maximum_cohort_width, configured_limit
        ):
            raise ValueError("hybrid tree cohort exceeds its bound")

    def forward(self, model, inputs, parents, cache, layers, *, cached):
        self.validate_revision()
        result = tree_forward(
            model, inputs, parents, cache, layers, self.source_revision_token
        )
        self.dispatches += 1
        return result

    def forward_many(
        self, model, inputs, parents, caches, layers, *, cached, configured_limit
    ):
        self.validate_revision()
        self.validate_cohort(len(caches), configured_limit)
        outputs, transactions = [], []
        try:
            for row, tree, cache in zip(inputs, parents, caches, strict=True):
                result = tree_forward(
                    model, row, tree, cache, layers, self.source_revision_token
                )
                outputs.append(result[:2])
                transactions.append(result[2])
        except BaseException:
            for transaction in transactions:
                transaction.abort()
            raise
        self.dispatches += 1
        return (
            mx.concatenate([item[0] for item in outputs], axis=0),
            mx.concatenate([item[1] for item in outputs], axis=0),
            HybridTreeCohortTransaction(transactions, token=self.source_revision_token),
        )

    def dispatch_metrics(self, width, transaction, *, cached):
        self.validate_cohort(width, self.maximum_cohort_width)
        return TargetTreeDispatchMetrics(
            width=width,
            physical_packed=False,
            executor_cached=False,
            cached=bool(cached),
            provider_protocol=self.protocol,
        )

    def receipt(self):
        return {
            "provider": self.protocol,
            "source_identity": dict(self.source_identity),
            "source_revision_token": self.source_revision_token,
            "dispatches": self.dispatches,
            "implemented": True,
            "qualified": False,
            "selected": self.dispatches > 0,
            "observed_used": self.dispatches > 0,
        }


__all__ = ["HybridTreeExecutor", "HybridTreeTransaction", "tree_forward"]
