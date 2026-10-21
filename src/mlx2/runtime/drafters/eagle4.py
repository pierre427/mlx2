# Copyright 2026 Eagle AI Corp.
# Modified 2026-10-21 by mlx2 contributors: ported the dense Qwen3.8 draft
# architecture and candidate law to MLX and mlx2-owned scheduling/state.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Eagle 4 Source-Available License (not an OSI-approved open-source license):
# see third_party/eagle4/ (LICENSE, NOTICE, MODIFICATIONS.md, README.md).
# The draft layer follows eagle_harness/sglang/models/llama_eagle4.py, which
# derives from SGLang (SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team)
# python/sglang/srt/models/llama_eagle3.py, itself adapted from SafeAILab EAGLE
# (SPDX-FileCopyrightText: Copyright 2025 SafeAI Lab (SAIL)) eagle/model/cnets.py,
# both Apache-2.0; see third_party/eagle4/THIRD_PARTY_NOTICES.md.
"""EAGLE-4 dense drafter for Qwen3.8 external chain and AST tree routes.

This is an MLX implementation of the released checkpoint contract, not an
engine port.  The scheduler, target-tree verifier and state publication remain
mlx2-owned.  See ``provenance/eagle4-drafter.json`` for the source and license
boundary.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np
from mlx import nn

from .cohere_eagle import EagleWindowKVCache, _Rows
from .eagle3 import Eagle3Layer

ARCHITECTURE = "LlamaForCausalLMEagle4"
EAGLE4_NUMERICS_VERSION = "mlx2.eagle4-dense.v1"


@dataclass
class Eagle4Config:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    sliding_window: int
    target_hidden_size: int
    num_target_layers: int
    aux_layer_ids: list[int]
    candidate_rank: int
    num_candidates: int
    num_hidden_layers: int = 1
    target_layer_ids: list[int] = field(default_factory=list)
    block_size: int = 16
    norm_before_residual: bool = False
    norm_output: bool = True
    fc_norm: bool = True
    has_embed_tokens: bool = True
    model_type: str = "eagle4"
    maximum_proposal_length: int = 31

    def __post_init__(self):
        if not self.target_layer_ids:
            # The release config names post-block outputs. mlx2 captures the
            # same post-block residual stream by zero-based layer index.
            self.target_layer_ids = list(self.aux_layer_ids)
        self.validate()

    @property
    def feature_width(self):
        return len(self.target_layer_ids) * self.target_hidden_size

    def validate(self):
        positive = (
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
            "sliding_window",
            "target_hidden_size",
            "num_target_layers",
            "candidate_rank",
            "num_candidates",
            "num_hidden_layers",
            "block_size",
            "maximum_proposal_length",
        )
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) <= 0
            for name in positive
        ):
            raise ValueError("EAGLE-4 integer geometry must be positive")
        if self.num_hidden_layers != 1:
            raise ValueError("EAGLE-4 supports exactly one dense draft layer")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("EAGLE-4 attention heads must divide KV heads")
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError("EAGLE-4 attention width must equal hidden_size")
        if len(self.aux_layer_ids) != 3 or self.aux_layer_ids != sorted(
            set(self.aux_layer_ids)
        ):
            raise ValueError("EAGLE-4 needs three increasing target auxiliary layers")
        if self.target_layer_ids != self.aux_layer_ids:
            raise ValueError(
                "EAGLE-4 target taps must be the declared post-block outputs"
            )
        if (
            self.aux_layer_ids[0] < 0
            or self.aux_layer_ids[-1] >= self.num_target_layers
        ):
            raise ValueError("EAGLE-4 auxiliary layer is outside the target")
        if not all(
            type(value) is bool
            for value in (
                self.norm_before_residual,
                self.norm_output,
                self.fc_norm,
                self.has_embed_tokens,
            )
        ):
            raise ValueError("EAGLE-4 structural switches must be boolean")
        if not self.fc_norm or not self.norm_output or not self.has_embed_tokens:
            raise ValueError(
                "The released dense EAGLE-4 contract requires fc_norm, norm_output and embeddings"
            )
        if not isinstance(self.rms_norm_eps, (int, float)) or self.rms_norm_eps <= 0:
            raise ValueError("EAGLE-4 rms_norm_eps must be positive")

    @classmethod
    def from_hf(cls, config: dict, *, num_target_layers: int):
        if config.get("architectures") != [ARCHITECTURE]:
            raise ValueError(f"Expected a {ARCHITECTURE} draft artifact")
        head = config.get("efficient_lm_head_config")
        eagle = config.get("eagle_config")
        rope = config.get("rope_parameters") or {}
        required = {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "mlp_bias": False,
            "num_hidden_layers": 1,
            "fc_norm": True,
            "norm_output": True,
            "use_sliding_window": True,
            "layer_types": ["sliding_attention"],
            "tie_word_embeddings": False,
        }
        for name, wanted in required.items():
            if config.get(name) != wanted:
                raise ValueError(f"EAGLE-4 {name} must be {wanted!r}")
        if not isinstance(eagle, dict) or eagle.get("use_aux_hidden_state") is not True:
            raise ValueError("EAGLE-4 requires auxiliary target hidden states")
        if set(eagle) != {"use_aux_hidden_state", "eagle_aux_hidden_state_layer_ids"}:
            raise ValueError("Unsupported EAGLE-4 eagle_config keys")
        if not isinstance(head, dict) or set(head) != {
            "type",
            "rank",
            "num_candidates",
        }:
            raise ValueError("Unsupported EAGLE-4 efficient head config")
        if head.get("type") != "low_rank":
            raise ValueError("EAGLE-4 requires a low_rank candidate head")
        if rope.get("rope_type", "default") != "default" or set(rope) - {
            "rope_type",
            "rope_theta",
        }:
            raise ValueError("Unsupported EAGLE-4 RoPE contract")
        if int(config.get("draft_vocab_size", config["vocab_size"])) != int(
            config["vocab_size"]
        ):
            raise ValueError("EAGLE-4 reduced draft vocabularies are not implemented")
        return cls(
            hidden_size=int(config["hidden_size"]),
            intermediate_size=int(config["intermediate_size"]),
            num_attention_heads=int(config["num_attention_heads"]),
            num_key_value_heads=int(config["num_key_value_heads"]),
            head_dim=int(config["head_dim"]),
            vocab_size=int(config["vocab_size"]),
            rms_norm_eps=float(config["rms_norm_eps"]),
            rope_theta=float(rope.get("rope_theta", config.get("rope_theta", 10000.0))),
            max_position_embeddings=int(config["max_position_embeddings"]),
            sliding_window=int(config["sliding_window"]),
            target_hidden_size=int(
                config.get("target_hidden_size", config["hidden_size"])
            ),
            num_target_layers=int(num_target_layers),
            aux_layer_ids=list(eagle["eagle_aux_hidden_state_layer_ids"]),
            candidate_rank=int(head["rank"]),
            num_candidates=int(head["num_candidates"]),
        )


def expected_weight_schema(config: Eagle4Config) -> dict[str, list[int]]:
    h, inter = config.hidden_size, config.intermediate_size
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    p = "layers.0."
    return {
        "efficient_lm_head.weight": [config.vocab_size, config.candidate_rank],
        "efficient_lm_head_down_proj.weight": [config.candidate_rank, h],
        "embed_tokens.weight": [config.vocab_size, h],
        "fc.weight": [h, config.feature_width],
        **{
            f"fc_norm.{index}.weight": [config.target_hidden_size] for index in range(3)
        },
        p + "hidden_norm.weight": [h],
        p + "input_layernorm.weight": [h],
        p + "mlp.down_proj.weight": [h, inter],
        p + "mlp.gate_proj.weight": [inter, h],
        p + "mlp.up_proj.weight": [inter, h],
        p + "post_attention_layernorm.weight": [h],
        p + "self_attn.k_proj.weight": [kv, 2 * h],
        p + "self_attn.o_proj.weight": [h, q],
        p + "self_attn.q_proj.weight": [q, 2 * h],
        p + "self_attn.v_proj.weight": [kv, 2 * h],
        "lm_head.weight": [config.vocab_size, h],
        "norm.weight": [h],
    }


class Eagle4DraftModel(nn.Module):
    requires_context_tokens = True
    tree_topology_aware = True
    supports_logits_processors = True
    receipt_kind = "external_eagle4"
    proposal_distribution = "candidate_restricted_exact_law"
    maximum_proposal_length = 31

    def __init__(self, config: Eagle4Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.fc_norm = [
            nn.RMSNorm(config.target_hidden_size, eps=config.rms_norm_eps)
            for _ in range(3)
        ]
        self.fc = nn.Linear(config.feature_width, config.hidden_size, bias=False)
        self.layers = [Eagle3Layer(config)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.efficient_lm_head = nn.Linear(
            config.candidate_rank, config.vocab_size, bias=False
        )
        self.efficient_lm_head_down_proj = nn.Linear(
            config.hidden_size, config.candidate_rank, bias=False
        )
        self.stats = {
            "committed_positions": 0,
            "chain_steps": 0,
            "draft_calls": 0,
            "tree_calls": 0,
            "tree_nodes": 0,
        }

    @property
    def receipt_settings(self):
        c = self.config
        return {
            "algorithm": "eagle4_dense",
            "numerics": EAGLE4_NUMERICS_VERSION,
            "aux_layer_ids": list(c.aux_layer_ids),
            "target_layer_ids": list(c.target_layer_ids),
            "sliding_window": c.sliding_window,
            "candidate_rank": c.candidate_rank,
            "candidate_count": c.num_candidates,
            "tree_children": 4,
            "tree_ranking": "best_first_cumulative_candidate_log_probability",
            "proposal_distribution": "softmax_over_low_rank_candidates_exactly_rescored_by_full_head",
        }

    def validate_target_compatibility(self, target_model):
        args = getattr(target_model, "speculative_args", target_model.args)
        for name in ("hidden_size", "vocab_size"):
            if getattr(args, name) != getattr(
                self.config, name if name != "hidden_size" else "target_hidden_size"
            ):
                raise ValueError(f"EAGLE-4 target {name} mismatch")
        if args.num_hidden_layers != self.config.num_target_layers:
            raise ValueError("EAGLE-4 target layer count mismatch")

    def bind(self, target_model):
        self.validate_target_compatibility(target_model)
        # EAGLE-4 serving shares the target embedding even though the release
        # artifact carries an export-time copy.  Bind the live target table so
        # a converted/quantized target and its drafter cannot silently drift.
        trunk = getattr(target_model, "model", None)
        target_embedding = getattr(trunk, "embed_tokens", None)
        if target_embedding is None:
            raise ValueError("EAGLE-4 target does not expose its token embedding")
        self.embed_tokens = target_embedding
        return self

    def make_cache(self):
        c = self.config
        return [EagleWindowKVCache(c.sliding_window, c.num_key_value_heads, c.head_dim)]

    def batch_caches(self, rows):
        if not rows or any(len(row) != 1 for row in rows):
            raise ValueError("EAGLE-4 cache row topology mismatch")
        return _Rows([list(row) for row in rows])

    @staticmethod
    def _rows(cache):
        return cache.rows if isinstance(cache, _Rows) else [cache]

    def _project_features(self, features):
        width = self.config.target_hidden_size
        chunks = [
            features[..., index * width : (index + 1) * width] for index in range(3)
        ]
        normalized = mx.concatenate(
            [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks, strict=True)],
            axis=-1,
        )
        return self.fc(normalized.astype(self.fc.weight.dtype))

    def _forward(self, tokens, features, rows, *, commit, chain=None, fused=False):
        layer = self.layers[0]
        embeds = self.embed_tokens(tokens)
        if not fused:
            features = self._project_features(features)
        features = features.astype(embeds.dtype)
        e = layer.input_layernorm(embeds)
        g = layer.hidden_norm(features)
        residual = g if self.config.norm_before_residual else features
        q, k, v = layer.self_attn.project(mx.concatenate([e, g], axis=-1))
        outputs = []
        for row, caches in enumerate(rows):
            cache = caches[0]
            prior_k, prior_v = cache.keys_and_values()
            local = None if commit else chain[row]
            # The trained window includes the current context position. Keep
            # W committed rows for subsequent tree steps, but a new committed
            # context query reads only W-1 prior rows before adding itself.
            if commit and prior_k.shape[2] >= self.config.sliding_window:
                keep = self.config.sliding_window - 1
                prior_k = prior_k[:, :, -keep:] if keep else prior_k[:, :, :0]
                prior_v = prior_v[:, :, -keep:] if keep else prior_v[:, :, :0]
            if local is not None and local[0] is not None:
                prior_k = (
                    mx.concatenate([prior_k, local[0]], axis=2)
                    if prior_k.shape[2]
                    else local[0]
                )
                prior_v = (
                    mx.concatenate([prior_v, local[1]], axis=2)
                    if prior_v.shape[2]
                    else local[1]
                )
            position = cache.offset + (
                0 if local is None or local[0] is None else local[0].shape[2]
            )
            out, rotated = layer.self_attn.attend(
                q[row : row + 1],
                k[row : row + 1],
                v[row : row + 1],
                position=position,
                prior_k=prior_k,
                prior_v=prior_v,
            )
            outputs.append(out)
            if commit:
                cache.append(rotated, v[row : row + 1])
            elif chain[row][0] is None:
                chain[row] = (rotated, v[row : row + 1])
            else:
                old = chain[row]
                chain[row] = (
                    mx.concatenate([old[0], rotated], axis=2),
                    mx.concatenate([old[1], v[row : row + 1]], axis=2),
                )
        out = (
            mx.concatenate(outputs, axis=0)
            .transpose(0, 2, 1, 3)
            .reshape(*features.shape)
        )
        hidden = residual + layer.self_attn.o_proj(out)
        summed = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        normed = self.norm(summed)
        return normed, normed if self.config.norm_output else summed

    def _candidate_rows(self, hidden, *, forbidden=()):
        """Candidate ids, exact logits and their normalized law for each row."""
        low = self.efficient_lm_head_down_proj(
            hidden.astype(self.efficient_lm_head_down_proj.weight.dtype)
        )
        scores = self.efficient_lm_head(low).astype(mx.float32)
        count = min(self.config.num_candidates, int(scores.shape[-1]))
        ids = mx.argpartition(scores, -count, axis=-1)[..., -count:]
        full_rows = self.lm_head.weight[ids]
        logits = mx.sum(
            full_rows.astype(mx.float32) * hidden[..., None, :].astype(mx.float32),
            axis=-1,
        )
        if forbidden:
            blocked = mx.array(tuple(int(token) for token in forbidden), dtype=mx.int32)
            logits = mx.where(
                mx.any(ids[..., None] == blocked, axis=-1), -float("inf"), logits
            )
        if bool(mx.any(mx.all(mx.isneginf(logits), axis=-1)).item()):
            raise ValueError("EAGLE-4 candidate set is entirely forbidden")
        logits = logits - mx.max(logits, axis=-1, keepdims=True)
        probs = mx.exp(logits)
        probs = probs / mx.sum(probs, axis=-1, keepdims=True)
        return ids, logits, probs

    def append_context(self, hidden, cache, *, context_tokens=None):
        if hidden.shape[1] == 0:
            return None
        rows = self._rows(cache)
        if context_tokens is None:
            raise ValueError("EAGLE-4 needs the token following each feature")
        tokens = mx.array(
            np.asarray(context_tokens, dtype=np.int32).reshape(len(rows), -1)
        )
        if tokens.shape[1] != hidden.shape[1]:
            raise ValueError("EAGLE-4 context tokens must pair with features")
        result = self._forward(tokens, hidden, rows, commit=True)
        self.stats["committed_positions"] += int(hidden.shape[1]) * len(rows)
        return result

    def draft_distributions(
        self,
        anchors,
        hidden,
        cache,
        proposal_length,
        rngs,
        temperatures,
        *,
        context_tokens=None,
        logits_processors=None,
        processor_histories=None,
    ):
        from ..processor_probe import probe_logits_processors
        from ..speculative_sampling import softmax

        rows = self._rows(cache)
        batch = len(rows)
        anchors = [int(value) for value in anchors]
        if hidden.shape[1] == 0:
            return [[] for _ in rows], [[] for _ in rows]
        if context_tokens is None:
            context_tokens = [[anchor] for anchor in anchors]
        normed, feedback = self.append_context(
            hidden, cache, context_tokens=context_tokens
        )
        normed, feedback = normed[:, -1:], feedback[:, -1:]
        chain = [(None, None) for _ in rows]
        tokens, laws = [[] for _ in rows], [[] for _ in rows]
        processors = logits_processors or [[] for _ in rows]
        histories = processor_histories or [[] for _ in rows]
        self.stats["draft_calls"] += 1
        for step in range(int(proposal_length)):
            if step:
                previous = mx.array([[row[-1]] for row in tokens], dtype=mx.int32)
                normed, feedback = self._forward(
                    previous, feedback, rows, commit=False, chain=chain, fused=True
                )
                self.stats["chain_steps"] += 1
            ids, _, probs = self._candidate_rows(normed[:, -1])
            mx.eval(ids, probs)
            id_values, prob_values = (
                np.asarray(ids),
                np.asarray(probs).astype(np.float64),
            )
            for row in range(batch):
                dense = np.zeros(self.config.vocab_size, dtype=np.float64)
                dense[id_values[row]] = prob_values[row]
                if processors[row]:
                    prefix = mx.array(
                        histories[row] + [anchors[row]] + tokens[row], dtype=mx.int32
                    )
                    processed = np.asarray(
                        probe_logits_processors(
                            processors[row],
                            prefix,
                            mx.array(
                                np.log(
                                    dense,
                                    where=dense > 0,
                                    out=np.full_like(dense, -np.inf),
                                )
                            )[None],
                        )[0]
                    )
                    if np.isfinite(processed).any():
                        dense = softmax(processed, temperatures[row])
                elif temperatures[row] not in (0, 0.0):
                    candidate_logits = np.log(
                        dense, where=dense > 0, out=np.full_like(dense, -np.inf)
                    )
                    dense = softmax(candidate_logits, temperatures[row])
                token = (
                    int(np.argmax(dense))
                    if temperatures[row] in (0, 0.0)
                    else int(rngs[row].sample(dense))
                )
                tokens[row].append(token)
                laws[row].append(dense)
        return tokens, laws

    def propose_tree(
        self,
        anchors,
        hidden,
        cache,
        max_nodes,
        *,
        context_tokens=None,
        forbidden_token_ids=None,
        mark=None,
        children=4,
        **_options,
    ):
        """Build parent-first best-first trees from the trained candidate head."""
        if type(max_nodes) is not int or max_nodes <= 0:
            raise ValueError("EAGLE-4 tree max_nodes must be a positive integer")
        if type(children) is not int or not 1 <= children <= self.config.num_candidates:
            raise ValueError("EAGLE-4 tree children must fit the candidate set")
        rows = self._rows(cache)
        anchors = [int(value) for value in anchors]
        if hidden.shape[1] == 0:
            return [([], []) for _ in rows]
        if context_tokens is None:
            context_tokens = [[anchor] for anchor in anchors]
        normed, feedback = self.append_context(
            hidden, cache, context_tokens=context_tokens
        )
        normed, feedback = normed[:, -1:], feedback[:, -1:]
        forbidden_token_ids = forbidden_token_ids or [() for _ in rows]
        results = []
        serial = 0
        for row_index, row_cache in enumerate(rows):
            ids, _, probs = self._candidate_rows(
                normed[row_index, -1], forbidden=forbidden_token_ids[row_index]
            )
            mx.eval(ids, probs)
            ids = np.asarray(ids).reshape(-1)
            probs = np.asarray(probs).reshape(-1)
            heap = []
            top = np.argsort(-probs)[: int(children)]
            for index in top:
                heapq.heappush(
                    heap,
                    (
                        -float(
                            np.log(max(float(probs[index]), np.finfo(np.float32).tiny))
                        ),
                        serial,
                        -1,
                        int(ids[index]),
                        0,
                        feedback[row_index : row_index + 1],
                        (None, None),
                    ),
                )
                serial += 1
            tokens, parents = [], []
            while heap and len(tokens) < int(max_nodes):
                (
                    negative,
                    _serial,
                    parent,
                    token,
                    depth,
                    parent_feedback,
                    parent_chain,
                ) = heapq.heappop(heap)
                tokens.append(token)
                parents.append(parent)
                me = len(tokens) - 1
                local = [parent_chain]
                next_normed, next_feedback = self._forward(
                    mx.array([[token]], dtype=mx.int32),
                    parent_feedback,
                    [row_cache],
                    commit=False,
                    chain=local,
                    fused=True,
                )
                if (
                    len(tokens) + len(heap) >= int(max_nodes)
                    or depth + 1 >= self.maximum_proposal_length
                ):
                    continue
                child_ids, _, child_probs = self._candidate_rows(
                    next_normed[:, -1], forbidden=forbidden_token_ids[row_index]
                )
                mx.eval(child_ids, child_probs)
                child_ids = np.asarray(child_ids).reshape(-1)
                child_probs = np.asarray(child_probs).reshape(-1)
                for index in np.argsort(-child_probs)[: int(children)]:
                    heapq.heappush(
                        heap,
                        (
                            negative
                            - float(
                                np.log(
                                    max(
                                        float(child_probs[index]),
                                        np.finfo(np.float32).tiny,
                                    )
                                )
                            ),
                            serial,
                            me,
                            int(child_ids[index]),
                            depth + 1,
                            next_feedback,
                            local[0],
                        ),
                    )
                    serial += 1
            results.append((tokens, parents))
            self.stats["tree_nodes"] += len(tokens)
        self.stats["tree_calls"] += len(rows)
        if mark is not None:
            mark("tree_search")
        return results


Model = Eagle4DraftModel

__all__ = [
    "ARCHITECTURE",
    "EAGLE4_NUMERICS_VERSION",
    "Eagle4Config",
    "Eagle4DraftModel",
    "Model",
    "expected_weight_schema",
]
