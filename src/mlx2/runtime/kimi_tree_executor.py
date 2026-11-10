# SPDX-License-Identifier: MIT
"""Adapter-owned packed Kimi Linear tree execution (AST provider contract).

Mirrors ``LingTreeExecutor`` (provenance/ling3-packed-tree.json) with the
Kimi KDA/NoPE-MLA tree math of ``models/kimi_tree.py``.  Requests keep
independent caches; nothing is packed across requests.  Provenance:
provenance/kimi-linear-speculative.json.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mlx.core as mx

from .ling_tree_executor import LingTreeExecutor, LingTreeTransaction
from .models.kimi_tree import KimiTree

PROTOCOL = "kimi_linear_layerwise_packed_tree_v1"


class KimiTreeTransaction(LingTreeTransaction):
    def __init__(self, trees, token):
        super().__init__(trees, token)
        self.provider_protocol = PROTOCOL


# Every source the tree forward executes: the tree contract, the Kimi model
# (projections, KDA recurrence, norm-gate, MLA attend, router, MoE) and the
# kernel modules those laws dispatch to.  Relative to runtime/.
SOURCES = (
    "kimi_tree_executor.py", "ling_tree_executor.py", "hybrid_tree_executor.py",
    "models/kimi_tree.py", "models/ling_tree.py", "models/kimi_linear.py",
    "models/bailing_hybrid.py", "models/base.py", "models/qsdpa_decode_metal.py", "models/cache.py",
    "models/gated_delta.py", "models/gdn_chunk32.py", "models/kda_chunk_wy.py",
    "models/kimi_linear_fused.py", "models/qwen4_fused_gdn_prefill.py",
    "models/kimi_linear_mla_flash.py", "models/kimi_linear_mla_fused.py",
    # Donor of latent_fused's partial/combine kernels (kernel_spec()).
    "models/glm53_sparse_attention.py",
    # Default MoE execution and its NAX gather / invariant-prefill gates.
    "models/switch_layers.py", "models/moe_nax_gather.py", "models/invariant_prefill.py",
    # dense_weight untiles raw TensorFold banks for the absorbed MLA and router.
    "tensorfold_qwen38/lane_qmm.py",
)
# Package imports of the Kimi tree/model/kernel modules left unbound, with why.
UNBOUND_IMPORTS = {
    "served_exp": "device-fault classification in exception handlers; no numerics",
    "prefill_checkpoint_rows": "prompt-chunk checkpoint row selection; not on the tree path",
    "prefill_tail_skip": "opt-in prompt-chunk tail claim in Model.__call__; not on the tree path",
    "segmented_plain_kv": "type check for multi-lane segmented verify; tree MLA reads its own ancestry keys",
}


def _stat(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


class KimiTreeExecutor(LingTreeExecutor):
    protocol = PROTOCOL

    def __init__(self):
        root = Path(__file__).parent
        self._sources = tuple(root / name for name in SOURCES)
        self.source_identity = self._identity()
        self._source_stats = tuple(_stat(path) for path in self._sources)
        self.source_revision_token = hashlib.sha256(repr(self.source_identity).encode()).hexdigest()
        self.dispatches = 0
        self.projection_rows = 0

    def _identity(self):
        return tuple((path.name, hashlib.sha256(path.read_bytes()).hexdigest()) for path in self._sources)

    def validate_revision(self):
        # Re-hash only when a source's inode, size or mtime moved: the bound
        # set is ~0.8 MB, too much to hash on every tree dispatch.
        stats = tuple(_stat(path) for path in self._sources)
        if stats == self._source_stats:
            return
        if self._identity() != self.source_identity:
            raise RuntimeError("Kimi packed tree source changed")
        self._source_stats = stats

    def forward_many(self, model, inputs, parents, caches, layers, *, cached, configured_limit):
        self.validate_revision()
        self.validate_cohort(len(caches), configured_limit)
        if not len(inputs) == len(parents) == len(caches):
            raise ValueError("Kimi packed tree cohort dimensions differ")
        identities = [id(item) for cache in caches for item in cache]
        if len(set(identities)) != len(identities):
            raise ValueError("Kimi tree cohorts require independent cache descriptors")
        trees, logits, features = [], [], []
        for tokens, ancestry, cache in zip(inputs, parents, caches, strict=True):
            if len(tokens) != len(ancestry):
                raise ValueError("Kimi packed tree tokens/parents differ")
            tree = KimiTree(ancestry, cache)
            output, taps = model.forward_tree_with_taps(mx.array([tokens], dtype=mx.uint32), tree, layers)
            trees.append(tree)
            logits.append(output)
            features.append(taps)
            self.projection_rows += len(tokens)
        self.dispatches += 1
        return (mx.concatenate(logits, axis=0), mx.concatenate(features, axis=0),
                KimiTreeTransaction(trees, self.source_revision_token))

    def receipt(self):
        return {
            **super().receipt(), "provider": PROTOCOL,
            "kda_execution": "kimi_parent_indexed_depth_cohorts_one_token_law",
            "mla_execution": "nope_latent_exact_ancestor_rows_one_query_law",
        }
