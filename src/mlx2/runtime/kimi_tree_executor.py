# SPDX-License-Identifier: MIT
"""Adapter-owned packed Kimi Linear tree execution (AST provider contract).

Mirrors ``LingTreeExecutor`` (provenance/ling3-packed-tree.json) with the
Kimi KDA/NoPE-MLA tree math of ``models/kimi_tree.py``.  Requests keep
independent caches; nothing is packed across requests.  Provenance:
provenance/kimi-linear-speculative.json.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .ling_tree_executor import LingTreeExecutor, LingTreeTransaction
from .models.kimi_tree import KimiTree
from .technologies.tree_verify import revision_token
from .technologies.tree_verify import TECHNOLOGY_SOURCES, _TECHNOLOGIES

PROTOCOL = "kimi_linear_layerwise_packed_tree_v1"


class KimiTreeTransaction(LingTreeTransaction):
    protocol = PROTOCOL


# Every source the tree forward executes: the tree contract, the Kimi model
# (projections, KDA recurrence, norm-gate, MLA attend, router, MoE) and the
# kernel modules those laws dispatch to.  Relative to runtime/.
SOURCES = (
    "kimi_tree_executor.py", "ling_tree_executor.py", "hybrid_tree_executor.py",
    "technologies/tree_verify.py", "technologies/dispatch.py",
    "models/kimi_tree.py", "models/ling_tree.py", "models/kimi_linear.py",
    "models/bailing_hybrid.py", "models/base.py", "models/qsdpa_decode_metal.py", "models/cache.py",
    "technologies/window_attention/__init__.py", "technologies/window_attention/vector_sources.py",
    "technologies/window_attention/vector_launch.py", "technologies/dispatch.py",
    "models/gated_delta.py", "models/gdn_chunk32.py", "models/kda_chunk_wy.py",
    "models/kimi_linear_fused.py", "models/qwen4_fused_gdn_prefill.py",
    "models/kimi_linear_mla_flash.py", "models/kimi_linear_mla_fused.py",
    # Donor of latent_fused's partial/combine kernels (kernel_spec()).
    "models/glm53_sparse_attention.py",
    # Default MoE execution and its NAX gather / invariant-prefill gates.
    "models/switch_layers.py", "models/moe_nax_gather.py", "models/invariant_prefill.py",
    # dense_weight untiles raw TensorFold banks for the absorbed MLA and router.
    "tensorfold_qwen38/lane_qmm.py",
    # The affine adapter can install row matvecs on dense tree projections.
    "models/flash_tensorfold_qmv.py",
    # The technologies the Kimi/Ling laws call: the MLA reference, the
    # linear-recurrence KDA laws and recurrence dispatch, their shared
    # dispatch, and the gradient guard it asks.
    "technologies/__init__.py", "technologies/linear_recurrence.py", 
    "technologies/mla.py", "models/differentiable.py",
)

# Bind every technology module (runtime/technologies/**), deduplicated.
SOURCES = tuple(dict.fromkeys(SOURCES + tuple(
    "technologies/" + path.relative_to(_TECHNOLOGIES).as_posix() for path in TECHNOLOGY_SOURCES)))
# Package imports of the Kimi tree/model/kernel modules left unbound, with why.
UNBOUND_IMPORTS = {
    "served_exp": "device-fault classification in exception handlers; no numerics",
    "prefill_checkpoint_rows": "prompt-chunk checkpoint row selection; not on the tree path",
    "prefill_tail_skip": "opt-in prompt-chunk tail claim in Model.__call__; not on the tree path",
    "segmented_plain_kv": "type check for multi-lane segmented verify; tree MLA reads its own ancestry keys",
    "device_offset_kv": "type check for a device-mode KV cache's verify mask (spec-round DC-5); "
                        "tree rounds never run on device-mode caches",
}


class KimiTreeExecutor(LingTreeExecutor):
    protocol = PROTOCOL
    device_tree_inputs = True
    _identity_style = "name"
    # Revalidation re-hashes only when a source's metadata moved: the bound
    # set is ~0.8 MB, too much to hash on every tree dispatch.
    _revision_error = "Kimi packed tree source changed"

    def __init__(self):
        root = Path(__file__).parent
        self._bind_sources(tuple(root / name for name in SOURCES))
        self.source_revision_token = revision_token(self.source_identity)
        self.dispatches = 0
        self.projection_rows = 0

    @staticmethod
    def decode_round_cache_reservation(cache, *, append, child_chains, lanes=1):
        from .models.cache import ArraysCache, KVCache, plain_kv_append_copy_bytes

        if lanes != 1 or any(type(item) not in (ArraysCache, KVCache) for item in cache):
            return None
        copied = branch = 0
        for item in cache:
            if type(item) is ArraysCache:
                # Each node can hold an independent recurrent terminal state.
                copied += max(4, append + 1) * item.nbytes
            elif item.keys is not None and item.keys.shape[2]:
                per_token = item.nbytes // item.keys.shape[2]
                copied += plain_kv_append_copy_bytes(item, append, bytes_per_token=per_token)
                branch = max(branch, (item.offset + append) * per_token)
        # Branch views are consumed by their layer before the next layer's
        # attention. Committed KV still needs one full copy-on-write image.
        return copied + child_chains * branch

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
            if isinstance(tokens, mx.array):
                if tokens.ndim != 1 or tokens.dtype not in (mx.int32, mx.uint32):
                    raise ValueError("Kimi packed tree needs one integer token row")
                row = tokens[None].astype(mx.uint32)
            else:
                row = mx.array([tokens], dtype=mx.uint32)
            tree = KimiTree(ancestry, cache)
            output, taps = model.forward_tree_with_taps(row, tree, layers)
            trees.append(tree)
            logits.append(output)
            features.append(taps)
        self.projection_rows += sum(len(tokens) for tokens in inputs)
        self.dispatches += 1
        return (mx.concatenate(logits, axis=0), mx.concatenate(features, axis=0),
                KimiTreeTransaction(trees, self.source_revision_token))

    def receipt(self):
        return {
            **super().receipt(), "provider": PROTOCOL,
            "device_tree_inputs": self.device_tree_inputs,
            "kda_execution": "kimi_parent_indexed_depth_cohorts_one_token_law",
            "mla_execution": "nope_latent_exact_ancestor_rows_one_query_law",
        }
