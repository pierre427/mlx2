# SPDX-License-Identifier: MIT
# Draft math mined from Blaizzy/mlx-vlm mlx_vlm/speculative/drafters/laguna_dflash
# (MIT); see provenance/laguna-dflash.json.  The external-route contract
# (exact per-position laws, committed-only context KV, row batching) is
# original mlx2 work.
"""Causal Laguna DFlash block drafter for the external draft/verify route.

``poolside/Laguna-XS-2.1-DFlash`` differs from the DFlash2 closure in three
ways that make it a separate drafter rather than a DFlash2 configuration:
the proposal block is *causal*, there is no candidate selector (each position
samples from the dense head), and every target tap has its own RMSNorm plus
per-head softplus attention gating.  One block forward yields ``K``
position laws; because the block inputs are mask tokens, position ``j`` does
not depend on drafts ``< j``, so ``softmax(logits_j / T)`` is the exact law
``verify_proposals`` needs.  Only target-backed context enters the draft KV.

No capability is qualified by importing or constructing this module.
"""
from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .cohere_eagle import EagleWindowKVCache, _Rows

_ARCHITECTURE = "DFlashLagunaForCausalLM"


@dataclass
class LagunaDFlashConfig:
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 5
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 100352
    rms_norm_eps: float = 1e-6
    rope_theta: float = 500000.0
    sliding_window: int = 512
    max_position_embeddings: int = 262144
    layer_types: list[str] = field(default_factory=list)
    block_size: int = 16
    mask_token_id: int = 12
    target_layer_ids: list[int] = field(default_factory=lambda: [1, 13, 25, 33, 39])
    num_target_layers: int = 40
    causal: bool = True
    model_type: str = "laguna_dflash"

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = ["sliding_attention"] * self.num_hidden_layers
        self.validate()

    def validate(self):
        for key in (
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "vocab_size", "sliding_window", "block_size", "num_target_layers",
        ):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Laguna DFlash requires a positive integer {key}")
        if self.sliding_window < 2 or self.block_size < 2:
            raise ValueError("Laguna DFlash window and block must exceed one")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Laguna DFlash heads must be divisible by KV heads")
        if len(self.layer_types) != self.num_hidden_layers or any(
            kind != "sliding_attention" for kind in self.layer_types
        ):
            raise ValueError("Laguna DFlash requires sliding-attention draft layers")
        ids = list(self.target_layer_ids)
        if not ids or ids != sorted(set(ids)) or ids[0] < 0 or ids[-1] >= self.num_target_layers:
            raise ValueError("Laguna DFlash target_layer_ids must be increasing target layers")
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError("Laguna DFlash mask_token_id must be inside the vocabulary")
        if not self.causal:
            raise ValueError("Laguna DFlash requires causal block attention")

    @classmethod
    def from_hf(cls, config: Mapping):
        if config.get("architectures") != [_ARCHITECTURE]:
            raise ValueError(f"Expected a {_ARCHITECTURE} draft artifact")
        dflash = config.get("dflash_config")
        if not isinstance(dflash, Mapping):
            raise ValueError("Laguna DFlash requires a dflash_config object")
        if config.get("gating") != "per-head" or config.get("attention_bias", False):
            raise ValueError("Laguna DFlash requires per-head gating without attention bias")
        if config.get("draft_vocab_size", config.get("vocab_size")) != config.get("vocab_size"):
            raise ValueError("Laguna DFlash requires the target vocabulary")
        if int(config.get("num_experts", 0) or 0):
            raise ValueError("Laguna DFlash draft layers must be dense")
        aux = config.get("eagle_aux_hidden_state_layer_ids")
        if aux is not None and list(aux) != [i + 1 for i in dflash.get("target_layer_ids", [])]:
            # Aux boundary ids are post-block taps shifted by one (HF hidden_states).
            raise ValueError("Laguna DFlash aux boundaries disagree with target_layer_ids")
        flat = dict(config)
        flat.pop("model_type", None)
        for key in ("block_size", "mask_token_id", "target_layer_ids", "num_target_layers", "causal"):
            if key in dflash:
                flat[key] = dflash[key]
        signature = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in signature})


def expected_weight_shapes(config: LagunaDFlashConfig) -> dict[str, list[int]]:
    hidden = config.hidden_size
    qkv = (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim
    shapes = {
        "fc.weight": [hidden, len(config.target_layer_ids) * hidden],
        "hidden_norm.weight": [hidden],
        "norm.weight": [hidden],
    }
    for index in range(len(config.target_layer_ids)):
        shapes[f"aux_hidden_norms.{index}.weight"] = [hidden]
    for index in range(config.num_hidden_layers):
        p = f"layers.{index}."
        shapes.update({
            p + "input_layernorm.weight": [hidden],
            p + "post_attention_layernorm.weight": [hidden],
            p + "self_attn.qkv_proj.weight": [qkv, hidden],
            p + "self_attn.o_proj.weight": [hidden, config.num_attention_heads * config.head_dim],
            p + "self_attn.g_proj.weight": [config.num_attention_heads, hidden],
            p + "self_attn.q_norm.weight": [config.head_dim],
            p + "self_attn.k_norm.weight": [config.head_dim],
            p + "mlp.gate_proj.weight": [config.intermediate_size, hidden],
            p + "mlp.up_proj.weight": [config.intermediate_size, hidden],
            p + "mlp.down_proj.weight": [hidden, config.intermediate_size],
        })
    return shapes


class LagunaDFlashAttention(nn.Module):
    def __init__(self, config: LagunaDFlashConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        dim = config.hidden_size
        self.qkv_proj = nn.Linear(dim, (self.n_heads + 2 * self.n_kv_heads) * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.g_proj = nn.Linear(dim, self.n_heads, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def split(self, x):
        batch, length, _ = x.shape
        qkv = self.qkv_proj(x)
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        q = qkv[..., :q_size].reshape(batch, length, self.n_heads, self.head_dim)
        k = qkv[..., q_size:q_size + kv_size].reshape(batch, length, self.n_kv_heads, self.head_dim)
        v = qkv[..., q_size + kv_size:].reshape(batch, length, self.n_kv_heads, self.head_dim)
        return (
            self.q_norm(q).transpose(0, 2, 1, 3),
            self.k_norm(k).transpose(0, 2, 1, 3),
            v.transpose(0, 2, 1, 3),
        )

    def context_kv(self, context, cache):
        """Commit one row's context K/V (RoPE at absolute positions)."""
        _, k, v = self.split(context)
        cache.append(self.rope(k, offset=cache.offset), v)

    def block(self, x, cache):
        """One row's causal proposal block over committed context (not stored)."""
        batch, length, _ = x.shape
        q, k, v = self.split(x)
        q = self.rope(q, offset=cache.offset)
        k = self.rope(k, offset=cache.offset)
        ctx_k, ctx_v = cache.keys_and_values()
        keys = mx.concatenate([ctx_k.astype(k.dtype), k], axis=2) if ctx_k.shape[2] else k
        values = mx.concatenate([ctx_v.astype(v.dtype), v], axis=2) if ctx_v.shape[2] else v
        mask = mx.concatenate(
            [mx.ones((length, ctx_k.shape[2]), dtype=mx.bool_),
             mx.tril(mx.ones((length, length), dtype=mx.bool_))], axis=-1
        )
        out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        gate = nn.softplus(self.g_proj(x).astype(mx.float32)).astype(out.dtype)
        out = (out.reshape(batch, length, self.n_heads, self.head_dim) * gate[..., None]).reshape(
            batch, length, -1
        )
        return self.o_proj(out)


class LagunaDFlashMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LagunaDFlashLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LagunaDFlashAttention(config)
        self.mlp = LagunaDFlashMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LagunaDFlashDraftModel(nn.Module):
    supports_logits_processors = True
    receipt_kind = "external_laguna_dflash"

    def __init__(self, config: LagunaDFlashConfig):
        super().__init__()
        self.config = config
        self.layers = [LagunaDFlashLayer(config) for _ in range(config.num_hidden_layers)]
        self.aux_hidden_norms = [
            nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            for _ in config.target_layer_ids
        ]
        self.fc = nn.Linear(len(config.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_tokens = None
        self.lm_head = None
        self.stats = {"committed_positions": 0, "block_drafts": 0}

    def validate_target_compatibility(self, target_model):
        args = target_model.args
        for name in ("hidden_size", "vocab_size"):
            if getattr(args, name) != getattr(self.config, name):
                raise ValueError(f"Laguna DFlash target {name} mismatch")
        if args.num_hidden_layers != self.config.num_target_layers:
            raise ValueError("Laguna DFlash target layer count mismatch")

    def bind(self, target_model):
        self.validate_target_compatibility(target_model)
        self.embed_tokens = target_model.model.embed_tokens
        head = getattr(target_model, "lm_head", None)
        self.lm_head = head if head is not None else self.embed_tokens.as_linear
        return self

    def sanitize(self, weights):
        out = {}
        for key, value in weights.items():
            key = key.removeprefix("model.")
            if key in out:
                raise ValueError(f"Duplicate Laguna DFlash weight after sanitize: {key}")
            out[key] = value
        return out

    def make_cache(self):
        c = self.config
        # Context attends the last sliding_window - 1 committed positions.
        return [
            EagleWindowKVCache(c.sliding_window - 1, c.num_key_value_heads, c.head_dim)
            for _ in self.layers
        ]

    def batch_caches(self, rows):
        if not rows or any(len(row) != len(self.layers) for row in rows):
            raise ValueError("Laguna DFlash cache row topology mismatch")
        return _Rows([list(row) for row in rows])

    @staticmethod
    def _rows(cache):
        return cache.rows if isinstance(cache, _Rows) else [cache]

    def _context(self, hidden):
        parts = mx.split(hidden, len(self.aux_hidden_norms), axis=-1)
        normed = [norm(part) for norm, part in zip(self.aux_hidden_norms, parts)]
        return self.hidden_norm(self.fc(mx.concatenate(normed, axis=-1)))

    def append_context(self, hidden, cache):
        if hidden.shape[1] == 0:
            return
        rows = self._rows(cache)
        context = self._context(hidden)
        for row, caches in enumerate(rows):
            for layer, entry in zip(self.layers, caches):
                layer.self_attn.context_kv(context[row:row + 1], entry)
        self.stats["committed_positions"] += int(hidden.shape[1]) * len(rows)

    def _block_logits(self, anchors, rows, count):
        tokens = mx.concatenate(
            [mx.array(anchors, dtype=mx.int32)[:, None],
             mx.full((len(anchors), count), self.config.mask_token_id, dtype=mx.int32)], axis=1
        )
        x = self.embed_tokens(tokens)
        for index, layer in enumerate(self.layers):
            h = layer.input_layernorm(x)
            attn = mx.concatenate(
                [layer.self_attn.block(h[row:row + 1], caches[index]) for row, caches in enumerate(rows)],
                axis=0,
            )
            x = x + attn
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.lm_head(self.norm(x)[:, 1:])

    def draft_distributions(
        self, anchors, hidden, cache, proposal_length, rngs, temperatures, *,
        logits_processors=None, processor_histories=None,
    ):
        from ..processor_probe import probe_logits_processors
        from ..speculative_sampling import softmax

        rows = self._rows(cache)
        batch = len(rows)
        anchor_values = [int(a) for a in anchors]
        logits_processors = logits_processors or [[] for _ in range(batch)]
        processor_histories = processor_histories or [[] for _ in range(batch)]
        self.append_context(hidden, cache)
        dense = np.asarray(self._block_logits(anchor_values, rows, int(proposal_length)).astype(mx.float32))
        self.stats["block_drafts"] += batch
        tokens = [[] for _ in range(batch)]
        laws = [[] for _ in range(batch)]
        for row in range(batch):
            for position in range(int(proposal_length)):
                value = dense[row, position]
                if logits_processors[row]:
                    prefix = mx.array(
                        list(processor_histories[row]) + [anchor_values[row]] + tokens[row],
                        dtype=mx.int32,
                    )
                    value = np.asarray(probe_logits_processors(
                        logits_processors[row], prefix, mx.array(value)[None]
                    )[0].astype(mx.float32))
                    if not np.isfinite(value).any():
                        break  # grammar dead end: shorter round, not an outage
                q = softmax(value, temperatures[row])
                token = int(rngs[row].sample(q))
                tokens[row].append(token)
                laws[row].append(q)
        return tokens, laws


Model = LagunaDFlashDraftModel

__all__ = ["LagunaDFlashConfig", "LagunaDFlashDraftModel", "Model", "expected_weight_shapes"]
