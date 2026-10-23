"""Adapter-owned packed Ling execution through AST's provider contract.

The generic lane installer (including TensorFold) owns projection arithmetic.
Recurrent siblings share a depth call; requests retain independent caches.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mlx.core as mx

from .hybrid_tree_executor import HybridTreeExecutor
from .models.ling_tree import LingTree
from .target_tree_executor import TargetTreeDispatchMetrics

PROTOCOL = "ling_layerwise_packed_tree_v1"


class LingTreeTransaction:
    executor_cached = False
    physical_packed = False  # no physical cross-request packing

    def __init__(self, trees, token):
        self.trees = trees
        self.closed = False
        self.provider_protocol = PROTOCOL
        self.provider_revision_token = token

    def commit_paths(self, paths):
        if self.closed or len(paths) != len(self.trees):
            raise RuntimeError("Ling tree commit lane count mismatch or closed transaction")
        for tree, path in zip(self.trees, paths, strict=True):
            tree.validate_path(path)
        try:
            prepared = []
            for tree, path in zip(self.trees, paths, strict=True):
                prepared.extend(tree.prepare_commit(path))
            LingTree.publish(prepared)
            return [tree.live for tree in self.trees]
        finally:
            self.closed = True

    def commit(self, *, accepted_lengths):
        return self.commit_paths([list(range(int(length))) for length in accepted_lengths])

    def abort(self):
        self.closed = True


class LingTreeExecutor(HybridTreeExecutor):
    protocol = PROTOCOL

    def __init__(self):
        root = Path(__file__).parent
        self._sources = (Path(__file__), root / "models/ling_tree.py",
                         root / "models/bailing_hybrid.py", root / "hybrid_tree_executor.py")
        self.source_identity = tuple(
            (path.name, hashlib.sha256(path.read_bytes()).hexdigest()) for path in self._sources
        )
        self.source_revision_token = hashlib.sha256(repr(self.source_identity).encode()).hexdigest()
        self.dispatches = 0
        self.projection_rows = 0

    def validate_revision(self):
        current = tuple((path.name, hashlib.sha256(path.read_bytes()).hexdigest()) for path in self._sources)
        if current != self.source_identity:
            raise RuntimeError("Ling packed tree source changed")

    def forward(self, model, inputs, parents, cache, layers, *, cached):
        logits, features, transaction = self.forward_many(
            model, [inputs], [parents], [cache], layers, cached=cached, configured_limit=1,
        )
        return logits, features, transaction

    def forward_many(self, model, inputs, parents, caches, layers, *, cached, configured_limit):
        self.validate_revision()
        self.validate_cohort(len(caches), configured_limit)
        if not len(inputs) == len(parents) == len(caches):
            raise ValueError("Ling packed tree cohort dimensions differ")
        identities = [id(item) for cache in caches for item in cache]
        if len(set(identities)) != len(identities):
            raise ValueError("Ling tree cohorts require independent cache descriptors")
        trees, logits, features = [], [], []
        for tokens, ancestry, cache in zip(inputs, parents, caches, strict=True):
            if len(tokens) != len(ancestry):
                raise ValueError("Ling packed tree tokens/parents differ")
            tree = LingTree(ancestry, cache)
            output, taps = model.forward_tree_with_taps(mx.array([tokens], dtype=mx.uint32), tree, layers)
            trees.append(tree)
            logits.append(output)
            features.append(taps)
            self.projection_rows += len(tokens)
        self.dispatches += 1
        return mx.concatenate(logits, axis=0), mx.concatenate(features, axis=0), LingTreeTransaction(
            trees, self.source_revision_token,
        )

    def dispatch_metrics(self, width, transaction, *, cached):
        self.validate_cohort(width, self.maximum_cohort_width)
        return TargetTreeDispatchMetrics(width, False, False, bool(cached), self.protocol)

    def receipt(self):
        return {
            **super().receipt(), "provider": PROTOCOL,
            "within_request_projection_packing": True,
            "cross_request_physical_packing": False,
            "projection_rows": self.projection_rows,
            "kda_execution": "parent_indexed_depth_cohorts",
            "mla_execution": "absorbed_exact_ancestor_rows",
        }
