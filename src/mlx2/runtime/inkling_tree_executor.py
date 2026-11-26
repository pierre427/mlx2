"""Adapter-owned packed Inkling execution through AST's provider contract.

The generic lane installer (including TensorFold) owns projection arithmetic.
Convolution histories follow ancestry; requests retain independent caches.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .cow_cache import CommittedDescriptorPlan
from .hybrid_tree_executor import HybridTreeExecutor
from .models.inkling.tree import InklingTree
from .target_tree_executor import TargetTreeDispatchMetrics
from .technologies.tree_verify import (
    TECHNOLOGY_SOURCES,
    TreeLaneTransaction,
    revision_token,
    source_stat,
)

PROTOCOL = "inkling_layerwise_packed_tree_v1"


# Device, inode, size, mtime and ctime: same-sized writes with restored
# mtimes still trigger exact content validation.
_source_stat = source_stat


class InklingTreeTransaction(TreeLaneTransaction):
    protocol = PROTOCOL
    count_error = "Inkling tree commit lane count mismatch or closed transaction"

    def __init__(self, trees, token, *, physical_packed=False, plain=False):
        super().__init__(trees, token, publish=InklingTree.publish)
        self.physical_packed = physical_packed
        self.plain = plain


class InklingTreeExecutor(HybridTreeExecutor):
    protocol = PROTOCOL
    _identity_style = "relative"
    _source_root = Path(__file__).parent
    _revision_error = "Inkling packed tree source changed"
    plain_anchor_protocol = "private-ordinary-anchor-v1"
    plain_anchor_max_width = 4

    def __init__(self):
        root = Path(__file__).parent
        self._sources = (
            Path(__file__),
            root / "models/inkling/tree.py",
            root / "models/inkling/language.py",
            root / "models/inkling/attention.py",
            root / "models/inkling/attention_nax.py",
            root / "models/inkling/cache.py",
            root / "models/inkling/config.py",
            root / "models/inkling/experts.py",
            root / "models/inkling/ancestry.py",
            root / "models/inkling/tree_attention.py",
            root / "models/qwen36_moe_grouped.py",
            root / "models/qwen4_routed_decode.py",
            root / "hybrid_tree_executor.py",
            *TECHNOLOGY_SOURCES,
            root / "cow_cache.py",
            root / "models/cache.py",
            root / "technologies/window_attention/__init__.py",
            root / "technologies/window_attention/vector_sources.py",
            root / "technologies/window_attention/vector_launch.py",
            root / "technologies/dispatch.py",
        )
        self._bind_sources(self._sources)
        self.source_revision_token = revision_token(self.source_identity)
        self.dispatches = 0
        self.projection_rows = 0
        self.physical_packed_dispatches = 0
        self.global_kv_lends = 0
        self.cross_request_packed_dispatches = 0
        self.ordinary_anchor_dispatches = 0
        self.descriptor_plan_captures = 0
        self.descriptor_plan_declines = 0
        self._descriptor_plans = {}

    def _new_tree(self, parents, cache):
        key = tuple(type(item) for item in cache)
        plan = self._descriptor_plans.get(key)
        if plan is None:
            if len(self._descriptor_plans) >= 8:
                self._descriptor_plans.clear()
            plan = self._descriptor_plans[key] = CommittedDescriptorPlan(
                cache, self.source_revision_token
            )
        tree = InklingTree(
            parents,
            cache,
            descriptor_plan=plan,
            descriptor_scope=self.source_revision_token,
        )
        if tree.descriptor_plan_used:
            self.descriptor_plan_captures += 1
        else:
            self.descriptor_plan_declines += 1
            self._descriptor_plans.pop(key, None)
        return tree

    def forward_plain_anchors(self, model, inputs, caches, layers):
        """Accept lazy anchors from the generic ordinary parked pipeline."""
        self.validate_revision()
        if not 1 <= len(caches) <= self.plain_anchor_max_width or inputs.shape != (
            len(caches),
            1,
        ):
            raise ValueError("Inkling plain anchors require one token per bounded lane")
        identities = [id(item) for cache in caches for item in cache]
        if len(set(identities)) != len(identities):
            raise ValueError("Inkling ordinary cohorts require independent descriptors")
        trees = [self._new_tree([-1], cache) for cache in caches]
        return self._forward_anchor(model, inputs, trees, layers)

    @staticmethod
    def plain_anchor_cache_reservation(cache):
        """Future cache overlap of an ordinary private step, beyond residency.

        Global KV lends its append-only buffers instead of merging a slab.
        Reserve a complete growth copy plus two allocation quanta for the
        committed/unread pipeline pair. Local rings and recurrent histories
        retain the generic four-copy bound. No speculative-tree rebate.
        """
        from .models.cache import KVCache, RotatingKVCache
        from .models.inkling.cache import InklingConvolutionCache

        if len(cache) % 2 or any(
            type(item) not in (KVCache, RotatingKVCache)
            if index % 2 == 0
            else type(item) is not InklingConvolutionCache
            for index, item in enumerate(cache)
        ):
            return None
        required = 0
        for item in cache:
            size = int(item.nbytes)
            if type(item) is not KVCache:
                required += 4 * size
                continue
            if item.keys is None or item.values is None or item.keys.shape[2] == 0:
                return None
            capacity = int(item.keys.shape[2])
            if item.values.shape[2] != capacity or type(item.offset) is not int:
                return None
            # Reserve larger explicit growth too, rather than assuming the
            # caller's capacity always follows the default 256-token quantum.
            extra = max(2 * item.step, getattr(item, "_reserve_tokens", 0) - capacity)
            required += size + extra * (size // capacity)
        return required

    def _forward_anchor(self, model, inputs, trees, layers):
        packed = len(trees) > 1 and getattr(model, "tree_cohort", False)
        if packed:
            logits, features = InklingTree.forward_anchor_cohort(
                model, trees, inputs, layers
            )
        elif len(trees) == 1:
            logits, features = trees[0].forward_anchor(model, inputs, layers)
        else:
            rows = [
                tree.forward_anchor(model, inputs[row : row + 1], layers)
                for row, tree in enumerate(trees)
            ]
            logits = mx.concatenate([row[0] for row in rows], axis=0)
            features = mx.concatenate([row[1] for row in rows], axis=0)
        # Anchor rounds lend global KV too (sweep 2026-11-06: they were uncounted).
        self.global_kv_lends += sum(tree.lent_global_kv for tree in trees)
        self.dispatches += 1
        self.projection_rows += len(trees)
        self.ordinary_anchor_dispatches += 1
        self.physical_packed_dispatches += int(packed)
        self.cross_request_packed_dispatches += int(packed)
        return (
            logits,
            features,
            InklingTreeTransaction(
                trees, self.source_revision_token, plain=True, physical_packed=packed
            ),
        )

    def forward(self, model, inputs, parents, cache, layers, *, cached):
        logits, features, transaction = self.forward_many(
            model,
            [inputs],
            [parents],
            [cache],
            layers,
            cached=cached,
            configured_limit=1,
        )
        return logits, features, transaction

    def forward_many(
        self, model, inputs, parents, caches, layers, *, cached, configured_limit
    ):
        self.validate_revision()
        self.validate_cohort(len(caches), configured_limit)
        if not len(inputs) == len(parents) == len(caches):
            raise ValueError("Inkling packed tree cohort dimensions differ")
        identities = [id(item) for cache in caches for item in cache]
        if len(set(identities)) != len(identities):
            raise ValueError(
                "Inkling tree cohorts require independent cache descriptors"
            )
        trees, logits, features = [], [], []
        for tokens, ancestry, cache in zip(inputs, parents, caches, strict=True):
            if len(tokens) != len(ancestry):
                raise ValueError("Inkling packed tree tokens/parents differ")
            trees.append(self._new_tree(ancestry, cache))
        if all(len(tokens) == 1 for tokens in inputs):
            # Generic hybrid plain-step ownership deliberately excludes wrapped
            # rotating KV. Preserve this adapter's private commit/abort contract
            # while running the ordinary tensor path for probes/parked anchors.
            return self._forward_anchor(
                model, mx.array(inputs, dtype=mx.uint32), trees, layers
            )
        if getattr(model, "tree_cohort", False):
            if len({len(row) for row in inputs}) != 1:
                raise ValueError(
                    "Inkling physical cohorts require equal padded tree widths"
                )
            logits, features = InklingTree.forward_cohort(model, trees, inputs, layers)
            self.global_kv_lends += sum(tree.lent_global_kv for tree in trees)
            self.dispatches += 1
            self.projection_rows += sum(map(len, inputs))
            self.physical_packed_dispatches += 1
            self.cross_request_packed_dispatches += int(len(trees) > 1)
            return (
                logits,
                features,
                InklingTreeTransaction(
                    trees, self.source_revision_token, physical_packed=True
                ),
            )
        for tokens, tree in zip(inputs, trees, strict=True):
            output, taps = tree.forward(
                model, mx.array([tokens], dtype=mx.uint32), layers
            )
            logits.append(output)
            features.append(taps)
            self.global_kv_lends += tree.lent_global_kv
            self.projection_rows += len(tokens)
        self.dispatches += 1
        return (
            mx.concatenate(logits, axis=0),
            mx.concatenate(features, axis=0),
            InklingTreeTransaction(
                trees,
                self.source_revision_token,
            ),
        )

    def dispatch_metrics(self, width, transaction, *, cached):
        self.validate_cohort(width, self.maximum_cohort_width)
        return TargetTreeDispatchMetrics(
            width, transaction.physical_packed, False, bool(cached), self.protocol
        )

    def receipt(self):
        return {
            **super().receipt(),
            "provider": PROTOCOL,
            "within_request_projection_packing": True,
            "cross_request_physical_packing": self.cross_request_packed_dispatches > 0,
            "cross_request_packed_dispatches": self.cross_request_packed_dispatches,
            "physical_packed_dispatches": self.physical_packed_dispatches,
            "global_kv_lends": self.global_kv_lends,
            "projection_rows": self.projection_rows,
            "ordinary_anchor_dispatches": self.ordinary_anchor_dispatches,
            "descriptor_plan_captures": self.descriptor_plan_captures,
            "descriptor_plan_declines": self.descriptor_plan_declines,
            "convolution_execution": "ancestry_indexed_private_histories; kernel counters identify packed dispatches",
            "attention_execution": "singleton_arithmetic_exact_ancestor_rows; kernel counters identify mapped dispatches",
        }
