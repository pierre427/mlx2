"""Ling 3 UBTree drafter runtime.

The five-layer proposer is the released DFlash backbone.  Its learned selector
uses target-embedding factorizer MLPs and the paper's depth-calibrated,
width-pruned tree construction; it is deliberately separate from DFlash2.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from mlx import nn

from .dflash_base import DFlashDraftModel, append_context_kv
from .ubtree_config import UBTreeConfig
from .ubtree_tree import build_ubtree


class TokenFactorizer(nn.Module):
    def __init__(self, config: UBTreeConfig):
        super().__init__()
        self.input = nn.Linear(
            config.target_embedding_dim
            if hasattr(config, "target_embedding_dim")
            else config.hidden_size,
            config.factorizer_hidden_width,
            bias=False,
        )
        self.output = nn.Linear(
            config.factorizer_hidden_width, config.selector_rank, bias=False
        )

    def __call__(self, values):
        return self.output(nn.silu(self.input(values)))


class UBTreeSelector(nn.Module):
    def __init__(self, config: UBTreeConfig):
        super().__init__()
        self.hidden_projection = nn.Linear(
            config.hidden_size, config.selector_rank, bias=False
        )
        self.predecessor_factorizer = TokenFactorizer(config)
        self.successor_factorizer = TokenFactorizer(config)
        self.depth_conditioner = nn.Module()
        self.depth_conditioner.unary_log_scale = mx.zeros(
            (config.depth_condition_count,)
        )
        self.depth_conditioner.correction_log_scale = mx.zeros(
            (config.depth_condition_count,)
        )


class UBTreeDraftModel(DFlashDraftModel):
    proposal_distribution = "tree_only"
    prefer_requested_block_size = True
    minimum_proposal_length = 1
    maximum_proposal_length = 15
    supports_empty_target_context = True
    supports_continuation_tree_reuse = True
    receipt_kind = "external_ubtree"

    def __init__(self, config: UBTreeConfig):
        super().__init__(config)
        self.ubtree_selector = UBTreeSelector(config)

    @property
    def receipt_settings(self):
        return {
            "algorithm": "ubtree-pruned-v1",
            "candidate_k": self.config.selector_top_k,
            "frontier_width": self.config.tree_supertree_width,
            "checkpoint_tree_budget": self.config.tree_budget,
            "served_tree_budget": 15,
            "depth_bonus": 0.0,
            "ast_projection": "top15-prefix-closed-from-checkpoint-supertree",
        }

    def validate_target_compatibility(self, target_model):
        args = getattr(target_model, "speculative_args", target_model.args)
        for field in ("hidden_size", "vocab_size", "num_hidden_layers"):
            expected = (
                self.config.num_target_layers
                if field == "num_hidden_layers"
                else getattr(self.config, field)
            )
            if getattr(args, field) != expected:
                raise ValueError(f"UBTree target {field} mismatch")

    def bind(self, target_model):
        self.validate_target_compatibility(target_model)
        inner = getattr(target_model, "model", target_model)
        embedding = getattr(inner, "word_embeddings", None)
        if embedding is None:
            embedding = getattr(inner, "embed_tokens", None)
        if embedding is None:
            raise AttributeError("UBTree target has no token embedding")
        self.embed_tokens = embedding
        self.embed_scale = getattr(embedding, "embed_scale", 1.0)
        self.lm_head = getattr(target_model, "lm_head", None) or embedding.as_linear
        return self

    def _embed_input_tokens(self, inputs):
        return self.embed_tokens(inputs) * self.embed_scale

    def _logits(self, hidden):
        return self.lm_head(hidden) * self.config.output_multiplier

    def append_context(self, hidden, cache):
        """Commit target context to the drafter KV without drafting tokens."""
        if hidden.shape[1] == 0:
            return
        context = self.hidden_norm(self.fc(hidden))
        batch, length, _ = context.shape
        for layer, entry in zip(self.layers, cache):
            attention = layer.self_attn
            keys = attention.k_norm(
                attention.k_proj(context).reshape(
                    batch,
                    length,
                    attention.n_kv_heads,
                    attention.head_dim,
                )
            ).transpose(0, 2, 1, 3)
            values = attention.v_proj(context).reshape(
                batch,
                length,
                attention.n_kv_heads,
                attention.head_dim,
            ).transpose(0, 2, 1, 3)
            append_context_kv(entry, self.rope(keys, offset=entry.offset), values)

    def batch_caches(self, rows):
        if not rows or any(len(row) != len(self.layers) for row in rows):
            raise ValueError("UBTree cache row topology mismatch")
        return SimpleNamespace(rows=[list(row) for row in rows])

    def _lattice(self, anchors, hidden, cache, positions):
        anchors = mx.array([int(value) for value in anchors], dtype=mx.int32)
        inputs = mx.concatenate(
            [
                anchors[:, None],
                mx.full(
                    (anchors.shape[0], positions),
                    self.config.mask_token_id,
                    dtype=mx.int32,
                ),
            ],
            axis=1,
        )
        features = self._hidden(inputs, hidden, cache)[:, 1:]
        logits = self._logits(features)
        count = min(self.config.selector_top_k, int(logits.shape[-1]))
        candidates = mx.argpartition(logits, -count, axis=-1)[..., -count:]
        unary = mx.take_along_axis(logits, candidates, axis=-1).astype(mx.float32)
        projected = self.ubtree_selector.hidden_projection(features).astype(mx.float32)
        return candidates.astype(mx.int32), unary, projected

    def propose_tree(self, anchors, hidden, cache, max_nodes, *, mark=None, **options):
        cache_rows = getattr(cache, "rows", None)
        if cache_rows is not None:
            if len(cache_rows) != len(anchors) or hidden.shape[0] != len(anchors):
                raise ValueError("UBTree batched cache rows must match anchors")
            forbidden = options.get("forbidden_token_ids")
            if forbidden is not None and len(forbidden) != len(anchors):
                raise ValueError("UBTree forbidden-token rows must match the batch")
            trees = []
            for row, (anchor, row_cache) in enumerate(zip(anchors, cache_rows)):
                row_options = dict(options)
                if forbidden is not None:
                    row_options["forbidden_token_ids"] = [forbidden[row]]
                trees.extend(
                    self.propose_tree(
                        [anchor],
                        hidden[row : row + 1],
                        row_cache,
                        max_nodes,
                        mark=mark,
                        **row_options,
                    )
                )
            return trees
        if options:
            allowed = {
                "forbidden_token_ids",
                "lattice_positions",
                "submit",
                "codebooks",
            }
            if set(options) - allowed:
                raise TypeError(
                    f"unsupported UBTree tree options: {sorted(set(options) - allowed)}"
                )
        if options.get("codebooks") is not None:
            raise ValueError("UBTree derives codebooks from the bound target embedding")
        positions = (
            min(
                int(options.get("lattice_positions", self.config.block_size)),
                self.config.block_size,
            )
            - 1
        )
        if not 1 <= int(max_nodes) <= 15 or positions < 1:
            raise ValueError("mlx2 UBTree AST serves 1..15 non-root nodes")
        candidates, unary, projected = self._lattice(anchors, hidden, cache, positions)
        forbidden = options.get("forbidden_token_ids")
        if forbidden is not None:
            if len(forbidden) != len(anchors):
                raise ValueError("UBTree forbidden-token rows must match the batch")
            width = max((len(row) for row in forbidden), default=0)
            if width:
                blocked = mx.array(
                    [list(row) + [-1] * (width - len(row)) for row in forbidden],
                    dtype=mx.int32,
                )
                masked = mx.any(
                    candidates[:, :, :, None] == blocked[:, None, None, :],
                    axis=-1,
                )
                unary = mx.where(masked, -float("inf"), unary)
        if mark is not None:
            mark("tree_draft_build")
        mx.eval(candidates, unary, projected)
        if mark is not None:
            mark("tree_draft_wait")
        candidate_rows = np.asarray(candidates)
        ids = np.unique(
            np.concatenate(
                ([int(value) for value in anchors], candidate_rows.reshape(-1).tolist())
            )
        ).astype(np.int32)
        embeddings = self.embed_tokens(mx.array(ids, dtype=mx.int32)).astype(mx.float32)
        predecessor = self.ubtree_selector.predecessor_factorizer(embeddings).astype(
            mx.float32
        )
        successor = self.ubtree_selector.successor_factorizer(embeddings).astype(
            mx.float32
        )
        unary_scale = self.ubtree_selector.depth_conditioner.unary_log_scale.astype(
            mx.float32
        )
        correction_scale = (
            self.ubtree_selector.depth_conditioner.correction_log_scale.astype(
                mx.float32
            )
        )
        mx.eval(predecessor, successor, unary_scale, correction_scale)
        result = []
        for row, anchor in enumerate(anchors):
            result.append(
                build_ubtree(
                    candidate_rows[row],
                    np.asarray(unary)[row],
                    np.asarray(projected)[row],
                    int(anchor),
                    np.asarray(predecessor),
                    np.asarray(successor),
                    np.asarray(unary_scale),
                    np.asarray(correction_scale),
                    codebook_ids=ids,
                    frontier_width=self.config.tree_supertree_width,
                    max_nodes=int(max_nodes),
                    depth_bonus=0.0,
                )
            )
        if mark is not None:
            mark("tree_search")
        return result

    def sanitize(self, weights):
        normalized = {}
        for key, value in weights.items():
            key = key.removeprefix("model.")
            key = key.replace(
                "ubtree_selector.tied_token_factorizer.predecessor_mlp.input_projection.",
                "ubtree_selector.predecessor_factorizer.input.",
            )
            key = key.replace(
                "ubtree_selector.tied_token_factorizer.predecessor_mlp.output_projection.",
                "ubtree_selector.predecessor_factorizer.output.",
            )
            key = key.replace(
                "ubtree_selector.tied_token_factorizer.successor_mlp.input_projection.",
                "ubtree_selector.successor_factorizer.input.",
            )
            key = key.replace(
                "ubtree_selector.tied_token_factorizer.successor_mlp.output_projection.",
                "ubtree_selector.successor_factorizer.output.",
            )
            normalized[key] = value
        return normalized


Model = UBTreeDraftModel

__all__ = ["Model", "UBTreeDraftModel", "UBTreeSelector"]
