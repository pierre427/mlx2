# SPDX-License-Identifier: MIT
"""Kimi Linear DFlash block drafter (Moonlight556 Kimi-Linear-48B-A3B-DFlash-240k).

The checkpoint is a plain z-lab-style ``DFlashDraftModel``: five bidirectional
Qwen3 attention layers (q/k RMSNorm, default full-head RoPE at theta 1e4,
head_dim 72) over ``hidden_norm(fc(concat(target taps)))`` context and an
anchor-plus-mask block, with the target's embedding and LM head.  That is
exactly ``dflash_base.DFlashDraftModel``'s math; this module adds the external
route contract the DFlash2/Laguna drafters implement (context append, row
batching, per-position laws, best-first trees) and a strict config parser.
Provenance: provenance/kimi-linear-speculative.json (source dflash.py read as
data; no remote code is imported).

Because the block inputs after the anchor are mask tokens and attention is
bidirectional, position ``j`` does not depend on the drafts at positions
``< j``: ``softmax(logits_j / T)`` is the exact law of the drafted token, and a
tree may combine per-depth candidates freely.  No capability is qualified by
importing or constructing this module.
"""
from __future__ import annotations

import heapq
from collections.abc import Mapping
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from .base_config import DFlashConfig
from .dflash_base import DFlashDraftModel

ARCHITECTURE = "DFlashDraftModel"


def parse_config(config: Mapping) -> DFlashConfig:
    """Strict parse; refuse every geometry this drafter would silently ignore."""
    if config.get("architectures") != [ARCHITECTURE] or config.get("model_type") != "qwen3":
        raise ValueError("Kimi DFlash expects a qwen3 DFlashDraftModel artifact")
    dflash = config.get("dflash_config")
    if not isinstance(dflash, Mapping):
        raise ValueError("Kimi DFlash requires a dflash_config object")
    if bool(dflash.get("causal_head", False)):
        raise ValueError("Kimi DFlash serves the bidirectional (causal_head=false) block only")
    if config.get("attention_bias", False) or config.get("tie_word_embeddings", False):
        raise ValueError("Kimi DFlash requires bias-free attention and an untied target head")
    if config.get("hidden_act", "silu") != "silu":
        # The draft MLP is the SwiGLU of dflash_base, whatever the config says.
        raise ValueError("Kimi DFlash supports hidden_act='silu' only")
    layers = config.get("layer_types") or []
    if (len(layers) != config.get("num_hidden_layers")
            or any(kind != "full_attention" for kind in layers)
            or config.get("sliding_window") is not None or config.get("use_sliding_window")):
        raise ValueError("Kimi DFlash requires full-attention draft layers")
    rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
    if not isinstance(rope, Mapping) or rope.get("rope_type", "default") != "default":
        raise ValueError("Kimi DFlash supports default full-head RoPE only")
    theta = rope.get("rope_theta", config.get("rope_theta"))
    if not isinstance(theta, (int, float)) or theta <= 0:
        raise ValueError("Kimi DFlash requires an explicit rope_theta")
    flat = dict(config)
    flat["rope_theta"] = float(theta)
    flat["rope_scaling"] = None
    args = DFlashConfig.from_dict(flat)
    ids = list(args.target_layer_ids)
    if (not ids or ids != sorted(set(ids)) or ids[0] < 0
            or ids[-1] >= args.num_target_layers):
        raise ValueError("Kimi DFlash target_layer_ids must be increasing target layers")
    if not 0 <= int(args.mask_token_id) < args.vocab_size:
        raise ValueError("Kimi DFlash mask_token_id must be inside the vocabulary")
    if args.block_size < 2:
        raise ValueError("Kimi DFlash block_size must exceed one")
    return args


def expected_weight_shapes(args: DFlashConfig) -> dict[str, list[int]]:
    hidden = args.hidden_size
    heads = args.num_attention_heads * args.head_dim
    kv = args.num_key_value_heads * args.head_dim
    shapes = {
        "fc.weight": [hidden, len(args.target_layer_ids) * hidden],
        "hidden_norm.weight": [hidden],
        "norm.weight": [hidden],
    }
    for index in range(args.num_hidden_layers):
        p = f"layers.{index}."
        shapes.update({
            p + "input_layernorm.weight": [hidden],
            p + "post_attention_layernorm.weight": [hidden],
            p + "self_attn.q_proj.weight": [heads, hidden],
            p + "self_attn.k_proj.weight": [kv, hidden],
            p + "self_attn.v_proj.weight": [kv, hidden],
            p + "self_attn.o_proj.weight": [hidden, heads],
            p + "self_attn.q_norm.weight": [args.head_dim],
            p + "self_attn.k_norm.weight": [args.head_dim],
            p + "mlp.gate_proj.weight": [args.intermediate_size, hidden],
            p + "mlp.up_proj.weight": [args.intermediate_size, hidden],
            p + "mlp.down_proj.weight": [hidden, args.intermediate_size],
        })
    return shapes


def best_first_tree(log_probs, candidates, max_nodes, *, children=4, forbidden_root=()):
    """Best-first ``(tokens, parents)`` over independent per-depth laws.

    ``log_probs``/``candidates`` are ``[depth, K]`` (descending).  A node's
    score is its path's summed log-probability; parents precede children and
    ``-1`` is the anchor.  Deterministic for equal scores (insertion order).
    """
    depth_count, width = candidates.shape
    children = max(1, min(int(children), width))
    blocked = {int(token) for token in forbidden_root}
    heap, order = [], 0
    for index in range(width):
        token = int(candidates[0, index])
        if token in blocked:
            continue
        heapq.heappush(heap, (-float(log_probs[0, index]), order, -1, token, 0))
        order += 1
        if order >= children:
            break
    tokens, parents = [], []
    while heap and len(tokens) < int(max_nodes):
        negative, _, parent, token, depth = heapq.heappop(heap)
        tokens.append(token)
        parents.append(parent)
        node = len(tokens) - 1
        if depth + 1 < depth_count:
            for index in range(children):
                heapq.heappush(heap, (negative - float(log_probs[depth + 1, index]), order,
                                      node, int(candidates[depth + 1, index]), depth + 1))
                order += 1
    return tokens, parents


class KimiDFlashDraftModel(DFlashDraftModel):
    receipt_kind = "external_kimi_dflash"
    supports_logits_processors = True
    proposal_distribution = "stochastic_exact_law"
    # Committed context may be appended before a proposal call; an empty tail
    # reads the same committed draft KV without padding.
    supports_empty_target_context = True
    tree_children = 4
    # Receives the AST ``topology`` (BxD distinct-root branches, chain, drafter).
    tree_topology_aware = True

    def __init__(self, config: DFlashConfig):
        super().__init__(config)
        self.maximum_proposal_length = int(config.block_size) - 1
        self.stats = {"block_drafts": 0, "tree_lattices": 0, "context_appends": 0}

    @property
    def receipt_settings(self):
        return {
            "kind": "kimi_dflash",
            "trained_block_size": int(self.config.block_size),
            "maximum_proposal_length": self.maximum_proposal_length,
            "target_layer_ids": list(self.config.target_layer_ids),
            "mask_token_id": int(self.config.mask_token_id),
            "block_attention": "bidirectional",
            "proposal_distribution": "independent_position_laws_plus_best_first_tree",
            "tree_children": self.tree_children,
            "draft_context": "full_committed_context_kv",
            "qualified": False,
        }

    def validate_target_compatibility(self, target_model):
        args = getattr(target_model, "speculative_args", None) or target_model.args
        for name in ("hidden_size", "vocab_size"):
            if getattr(args, name) != getattr(self.config, name):
                raise ValueError(f"Kimi DFlash target {name} mismatch")
        if args.num_hidden_layers != self.config.num_target_layers:
            raise ValueError("Kimi DFlash target layer count mismatch")

    def bind(self, target_model):
        self.validate_target_compatibility(target_model)
        return super().bind(target_model)

    def make_cache(self):
        caches = super().make_cache()
        # Empty planes stay serializable at a one-token committed boundary.
        for cache in caches:
            cache.keys = mx.zeros((1, self.config.num_key_value_heads, 0, self.config.head_dim))
            cache.values = mx.zeros_like(cache.keys)
        return caches

    def batch_caches(self, rows):
        if not rows or any(len(row) != len(self.layers) for row in rows):
            raise ValueError("Kimi DFlash cache row topology mismatch")
        return [SimpleNamespace(rows=[row[layer] for row in rows]) for layer in range(len(self.layers))]

    def append_context(self, hidden, cache):
        """Commit target context to the draft KV without drafting a block."""
        from .dflash_base import append_context_kv

        if hidden.shape[1] == 0:
            return
        context = self.hidden_norm(self.fc(self.target_activation(hidden)))
        batch, length, _ = context.shape
        for layer, entry in zip(self.layers, cache):
            attn = layer.self_attn
            keys = attn.k_norm(attn.k_proj(context).reshape(
                batch, length, attn.n_kv_heads, attn.head_dim)).transpose(0, 2, 1, 3)
            values = attn.v_proj(context).reshape(
                batch, length, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
            entries = entry.rows if hasattr(entry, "rows") else [entry]
            for row, item in enumerate(entries):
                rows = slice(row, row + 1) if hasattr(entry, "rows") else slice(None)
                append_context_kv(item, self.rope(keys[rows], offset=item.offset), values[rows])
        self.stats["context_appends"] += 1

    def _block_logits(self, anchors, hidden, cache, count):
        count = int(count)
        if not 1 <= count < int(self.config.block_size):
            raise ValueError("Kimi DFlash proposal length must be below the trained block size")
        # The trained block is always anchor plus block_size - 1 masks (the
        # reference dflash.py shape); attention over it is bidirectional, so a
        # shorter block would change every position's logits.  Slice after.
        # Acceptance impact of this repair is GPU-unmeasured.
        anchors = mx.array([int(value) for value in anchors], dtype=mx.int32)
        inputs = mx.concatenate([anchors[:, None], mx.full(
            (anchors.shape[0], int(self.config.block_size) - 1), int(self.config.mask_token_id),
            dtype=mx.int32)], axis=1)
        return self._logits(self._hidden(inputs, hidden, cache)[:, 1:1 + count])

    def draft_distributions(self, anchors, hidden, cache, proposal_length, rngs, temperatures, *,
                            logits_processors=None, processor_histories=None, **_unused):
        from ..processor_probe import probe_logits_processors
        from ..proposal_providers import PointMassLaw
        from ..speculative_sampling import softmax

        batch = len(anchors)
        count = int(proposal_length)
        if not 1 <= count <= self.maximum_proposal_length:
            raise ValueError("Kimi DFlash proposal length exceeds the trained block")
        logits_processors = logits_processors or [[] for _ in range(batch)]
        processor_histories = processor_histories or [[] for _ in range(batch)]
        logits = self._block_logits(anchors, hidden, cache, count).astype(mx.float32)
        self.stats["block_drafts"] += batch
        greedy = [float(t) == 0 and not logits_processors[row] for row, t in enumerate(temperatures)]
        argmax = np.asarray(mx.argmax(logits, axis=-1)) if any(greedy) else None
        dense = None if all(greedy) else np.asarray(logits)
        tokens = [[] for _ in range(batch)]
        laws = [[] for _ in range(batch)]
        for row in range(batch):
            if greedy[row]:
                tokens[row] = [int(token) for token in argmax[row]]
                laws[row] = [PointMassLaw(token) for token in tokens[row]]
                continue
            for position in range(count):
                value = dense[row, position]
                if logits_processors[row]:
                    prefix = mx.array(list(processor_histories[row]) + [int(anchors[row])] + tokens[row],
                                      dtype=mx.int32)
                    value = np.asarray(probe_logits_processors(
                        logits_processors[row], prefix, mx.array(value)[None])[0].astype(mx.float32))
                    if not np.isfinite(value).any():
                        break  # grammar dead end: a shorter round, not an outage
                q = softmax(value, temperatures[row])
                token = int(rngs[row].sample(q))
                tokens[row].append(token)
                laws[row].append(q)
        return tokens, laws

    def propose_tree(self, anchors, hidden, cache, max_nodes, *, forbidden_token_ids=None,
                     topology=None, mark=None, **_options):
        """Per-row trees from one block pass (``-1`` = anchor).

        ``topology`` follows ``ast_tree_proposals``: ``BxD`` is B
        distinct-root branches of depth D (fitted to ``max_nodes``), as the
        native-MTP head builds them; ``chain`` is one greedy path;
        ``drafter`` (or None) is the head's own best-first search over its
        independent per-depth laws.  Positions after the anchor are mask
        inputs, so a branch's tail is the same per-depth argmax for every
        root; the branches differ at the root.
        """
        if not 1 <= int(max_nodes) <= 31:
            raise ValueError("Kimi DFlash AST serves 1..31 non-root nodes")
        if topology not in (None, "drafter"):
            return self._pattern_trees(anchors, hidden, cache, int(max_nodes), topology,
                                       forbidden_token_ids, mark)
        depth = min(int(max_nodes), self.maximum_proposal_length)
        logits = self._block_logits(anchors, hidden, cache, depth).astype(mx.float32)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        width = self.tree_children + 1
        candidates = mx.argpartition(-log_probs, width - 1, axis=-1)[..., :width]
        scores = mx.take_along_axis(log_probs, candidates, axis=-1)
        order = mx.argsort(-scores, axis=-1)
        candidates = mx.take_along_axis(candidates, order, axis=-1)
        scores = mx.take_along_axis(scores, order, axis=-1)
        if mark is not None:
            mark("tree_draft_build")
        mx.eval(candidates, scores)
        if mark is not None:
            mark("tree_draft_wait")
        candidates, scores = np.asarray(candidates), np.asarray(scores)
        forbidden = forbidden_token_ids or [() for _ in anchors]
        result = [best_first_tree(scores[row], candidates[row], max_nodes,
                                  children=self.tree_children, forbidden_root=forbidden[row])
                  for row in range(len(anchors))]
        self.stats["tree_lattices"] += len(result)
        self.stats["block_drafts"] += len(result)
        if mark is not None:
            mark("tree_search")
        return result

    def _pattern_trees(self, anchors, hidden, cache, max_nodes, topology, forbidden, mark):
        from ..ast_tree_proposals import paths_to_tree, pattern_for_budget

        pattern = [min(depth, self.maximum_proposal_length)
                   for depth in pattern_for_budget(topology, max_nodes)]
        if not pattern:
            return [([], []) for _ in anchors]
        branches, depth = len(pattern), max(pattern)
        forbidden = forbidden or [() for _ in anchors]
        logits = self._block_logits(anchors, hidden, cache, depth).astype(mx.float32)
        width = min(branches + max((len(row) for row in forbidden), default=0), logits.shape[-1])
        roots = mx.argpartition(-logits[:, 0], width - 1, axis=-1)[:, :width]
        roots = mx.take_along_axis(roots, mx.argsort(-mx.take_along_axis(logits[:, 0], roots, -1), -1), -1)
        tails = mx.argmax(logits, axis=-1)
        if mark is not None:
            mark("tree_draft_build")
        mx.eval(roots, tails)
        if mark is not None:
            mark("tree_draft_wait")
        roots, tails = np.asarray(roots), np.asarray(tails)
        result = []
        for row in range(len(anchors)):
            blocked = {int(token) for token in forbidden[row]}
            heads = [int(token) for token in roots[row] if int(token) not in blocked][:branches]
            paths = [[head] + [int(token) for token in tails[row, 1:length]]
                     for head, length in zip(heads, pattern)]
            result.append(paths_to_tree(paths, max_nodes))
        self.stats["tree_lattices"] += len(result)
        self.stats["block_drafts"] += len(result)
        if mark is not None:
            mark("tree_search")
        return result


Model = KimiDFlashDraftModel

__all__ = ["ARCHITECTURE", "KimiDFlashDraftModel", "Model", "best_first_tree",
           "expected_weight_shapes", "parse_config"]
