# SPDX-License-Identifier: MIT
"""Adapter-owned exact Nemotron-H tree execution (target-tree provider).

The provider protocol and transaction shape follow ``LingTreeExecutor`` /
``KimiTreeExecutor``.  The tree math lives with the model
(``Model.forward_tree_with_taps``, ``NemotronHMamba2Mixer.tree_forward``)
and in ``models/nemotron_tree.py``; requests keep independent caches and
nothing is packed across requests.  Provenance:
provenance/nemotron-h-mamba-tree.json.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .hybrid_tree_executor import HybridTreeExecutor
from .models.nemotron_mamba_tree import TREE_KERNEL_LAW, parity_status
from .models.nemotron_tree import NemotronTree
from .target_tree_executor import TargetTreeDispatchMetrics
from .technologies.tree_verify import TreeLaneTransaction, revision_token
from .technologies.tree_verify import TECHNOLOGY_SOURCES, _TECHNOLOGIES

PROTOCOL = "nemotron_h_layerwise_exact_tree_v1"
# Per-node law: row n equals one-token decode along paths[n] on the same
# route; the commit equals the exact chain commit of the accepted path.
TREE_LAW = "nemotron-h-exact-tree-v1:one-token-decode-per-path"

# Every source the tree forward and commit execute, relative to runtime/.
SOURCES = (
    "nemotron_tree_executor.py", "hybrid_tree_executor.py", "target_tree_executor.py",
    "technologies/tree_verify.py", "technologies/dispatch.py",
    "models/nemotron_tree.py", "models/nemotron_mamba_tree.py", "models/nemotron_h.py",
    "technologies/moe_router.py", 
    "models/nemotron_mamba_conv.py", "models/nemotron_mamba_ssm.py",
    "models/nemotron_mamba_norm.py", "models/served_exp.py", "models/ssm.py",
    "models/base.py", "models/cache.py", "models/qsdpa_decode_metal.py",
    "technologies/window_attention/__init__.py", "technologies/window_attention/vector_sources.py",
    "technologies/window_attention/vector_launch.py", 
    "models/fp_verify_tensor.py", "models/fp_row_exact_qmv.py", "models/row_exact_qmv.py",
    "models/switch_layers.py", "models/moe_nax_gather.py", "models/nemotron_routed_decode.py",
    "row_exact/ops.py", "row_exact/recipes.py", "row_exact/canary.py", "row_exact_verify.py",
)

# Bind every technology module (runtime/technologies/**), deduplicated.
SOURCES = tuple(dict.fromkeys(SOURCES + tuple(
    "technologies/" + path.relative_to(_TECHNOLOGIES).as_posix() for path in TECHNOLOGY_SOURCES)))


class NemotronTreeTransaction(TreeLaneTransaction):
    protocol = PROTOCOL
    count_error = "Nemotron tree commit lane count mismatch or closed transaction"
    close_on_refused_path = True  # legacy: validated inside its try/finally

    def __init__(self, trees, token):
        super().__init__(trees, token, publish=NemotronTree.publish)


class NemotronTreeExecutor(HybridTreeExecutor):
    protocol = PROTOCOL
    maximum_cohort_width = 1
    # ``forward`` takes a device int32 token row as ``inputs`` as well as a
    # host list (``speculative_device_tree``, device_round ``tree_lazy``).
    device_tree_inputs = True
    _identity_style = "relative"
    _source_root = Path(__file__).parent
    _revision_error = "Nemotron tree source changed"

    def __init__(self):
        root = Path(__file__).parent
        self._bind_sources(tuple(root / name for name in SOURCES))
        self.source_revision_token = revision_token(
            (self.source_identity, TREE_LAW, TREE_KERNEL_LAW)
        )
        self.dispatches = 0
        self.rows = 0
        self.backends = {}
        # True only once an adapter route binds this provider for serving.
        self.selected = False

    @classmethod
    def create(cls, *, selected=False):
        executor = cls()
        executor.selected = bool(selected)
        return executor

    def forward(self, model, inputs, parents, cache, layers, *, cached):
        return self.forward_many(
            model, [inputs], [parents], [cache], layers, cached=cached, configured_limit=1,
        )

    def forward_many(self, model, inputs, parents, caches, layers, *, cached, configured_limit):
        self.validate_revision()
        self.validate_cohort(len(caches), configured_limit)
        if not len(inputs) == len(parents) == len(caches):
            raise ValueError("Nemotron tree cohort dimensions differ")
        identities = [id(item) for cache in caches for item in cache]
        if len(set(identities)) != len(identities):
            raise ValueError("Nemotron tree cohorts require independent cache descriptors")
        trees, logits, features = [], [], []
        try:
            for tokens, ancestry, cache in zip(inputs, parents, caches, strict=True):
                if len(tokens) != len(ancestry):
                    raise ValueError("Nemotron tree tokens/parents differ")
                tree = NemotronTree(ancestry, cache)
                trees.append(tree)
                # A device round's verify input is a lazy int32 row (the
                # same token ids, ``device_tree_inputs``).
                row = (tokens.reshape(1, -1).astype(mx.uint32) if isinstance(tokens, mx.array)
                       else mx.array([list(tokens)], dtype=mx.uint32))
                output, taps = model.forward_tree_with_taps(row, tree, layers)
                logits.append(output)
                features.append(taps)
                self.rows += len(tokens)
                for stage, routes in tree.backends.items():
                    bucket = self.backends.setdefault(stage, {})
                    for route, count in routes.items():
                        bucket[route] = bucket.get(route, 0) + count
        except BaseException:
            for tree in trees:
                tree.release()
            raise
        self.dispatches += 1
        return (mx.concatenate(logits, axis=0), mx.concatenate(features, axis=0),
                NemotronTreeTransaction(trees, self.source_revision_token))

    def dispatch_metrics(self, width, transaction, *, cached):
        self.validate_cohort(width, self.maximum_cohort_width)
        return TargetTreeDispatchMetrics(width, False, False, bool(cached), self.protocol)

    def receipt(self):
        fused = all(
            set(self.backends.get(stage, {})) <= {"fused_tree"}
            for stage in ("mamba_conv", "mamba_ssm")
        ) and self.dispatches > 0
        return {
            "provider": PROTOCOL,
            "tree_law": TREE_LAW,
            "kernel_law": TREE_KERNEL_LAW,
            "source_identity": dict(self.source_identity),
            "source_revision_token": self.source_revision_token,
            "dispatches": self.dispatches,
            "rows": self.rows,
            "backends": {stage: dict(routes) for stage, routes in self.backends.items()},
            "mamba_backend": (
                None if not self.dispatches else "fused_tree" if fused else "serial_or_mixed"
            ),
            "kernel_parity": parity_status(),
            "within_request_projection_packing": True,
            "cross_request_physical_packing": False,
            "mamba_execution": "parent_slot_tree_kernels_or_serial_one_token",
            "attention_execution": "one_query_exact_ancestor_keys_decode_mask_law",
            "commit": "chain_rollback_replay_of_accepted_path",
            "implemented": True,
            "qualified": False,
            "selected": self.selected,
            "observed_used": self.dispatches > 0,
        }


__all__ = ["NemotronTreeExecutor", "NemotronTreeTransaction", "PROTOCOL", "TREE_LAW"]
