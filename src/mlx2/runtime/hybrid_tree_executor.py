"""Exact serial target-tree executor for mixed KV/recurrent model caches.

This is the correctness-first AST reference path.  Every node executes from a
private deep copy of its parent's exact state.  It deliberately makes no
throughput claim; a future parent-indexed fused kernel can replace it behind
the same provider contract after qualification.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import mlx.core as mx

from .target_tree_executor import TargetTreeDispatchMetrics

PROTOCOL = "hybrid_tree_serial_exact_v1"


def _replace_cache(destination, source):
    if type(destination) is not type(source):
        raise TypeError("hybrid tree cache type changed")
    destination.__dict__.clear()
    destination.__dict__.update(copy.deepcopy(source.__dict__))


class HybridTreeTransaction:
    def __init__(self, live, branches, *, token):
        self.live = live
        self.branches = branches
        self.closed = False
        self.executor_cached = False
        self.physical_packed = False
        self.provider_protocol = PROTOCOL
        self.provider_revision_token = token

    def commit_paths(self, paths):
        if self.closed or len(paths) != 1 or not paths[0]:
            raise RuntimeError("hybrid tree transaction needs one nonempty path")
        terminal = int(paths[0][-1])
        if not 0 <= terminal < len(self.branches):
            raise ValueError("hybrid tree commit path is outside the tile")
        try:
            selected = self.branches[terminal]
            for destination, source in zip(self.live, selected, strict=True):
                _replace_cache(destination, source)
        finally:
            self.closed = True
        return [self.live]

    def commit(self, *, accepted_lengths):
        if len(accepted_lengths) != 1:
            raise RuntimeError("hybrid tree transaction is single-lane")
        return self.commit_paths([list(range(int(accepted_lengths[0])))])

    def abort(self):
        self.closed = True


class HybridTreeCohortTransaction:
    def __init__(self, transactions, *, token):
        self.transactions = transactions
        self.closed = False
        self.executor_cached = False
        self.physical_packed = False
        self.provider_protocol = PROTOCOL
        self.provider_revision_token = token

    def commit_paths(self, paths):
        if self.closed or len(paths) != len(self.transactions):
            raise RuntimeError("hybrid tree cohort path count mismatch")
        caches = []
        try:
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


def tree_forward(model, tokens, parents, live, layers, token):
    if not 1 <= len(tokens) <= 16 or len(tokens) != len(parents):
        raise ValueError("hybrid AST tile must contain 1..16 matching rows")
    branches, logits, features = [], [], []
    for row, (value, parent) in enumerate(zip(tokens, parents, strict=True)):
        if (
            type(parent) is not int
            or (row == 0 and parent != -1)
            or (row > 0 and not 0 <= parent < row)
        ):
            raise ValueError(
                "hybrid AST parents must form one parent-first rooted tree"
            )
        source = live if parent < 0 else branches[parent]
        private = copy.deepcopy(source)
        row_logits, row_features = model.forward_with_taps(
            mx.array([[int(value)]], dtype=mx.uint32), private, layers
        )
        branches.append(private)
        logits.append(row_logits)
        features.append(row_features)
    return (
        mx.concatenate(logits, axis=1),
        mx.concatenate(features, axis=1),
        HybridTreeTransaction(live, branches, token=token),
    )


class HybridTreeExecutor:
    protocol = PROTOCOL
    maximum_cohort_width = 4

    def __init__(self):
        self.source_revision_token = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        self.source_identity = (
            ("module", "mlx2.runtime.hybrid_tree_executor"),
            ("sha256", self.source_revision_token),
        )
        self.dispatches = 0

    @classmethod
    def create(cls):
        return cls()

    def validate_revision(self):
        if (
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            != self.source_revision_token
        ):
            raise RuntimeError("hybrid tree executor source changed")

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
