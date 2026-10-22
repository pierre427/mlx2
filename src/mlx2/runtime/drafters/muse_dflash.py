# SPDX-License-Identifier: Apache-2.0
# Checkpoint normalization mined from bstnxbt/dflash-mlx (Apache-2.0,
# models/muse_glimmer_draft.py); the block mask law and the proposal logits
# follow the Hugging Face transformers reference (Apache-2.0,
# models/muse_glimmer_assistant and generation/candidate_generator.py
# DFlashTokenCandidateGenerator); see provenance/muse-dflash-assistant.json.
# The external-route contract (committed-only context KV, exact per-position
# laws, row batching) follows mlx2's own laguna_dflash drafter.
"""Muse Glimmer's original DFlash drafter (``MuseGlimmerAssistantModel``).

Meta ships Muse Glimmer 30B with a five-layer DFlash block drafter (BF16,
Q8 and Q4 checkpoints).  It is the stock DFlash geometry -- per-head q/k
RMSNorm, SwiGLU MLP, NeoX RoPE, every layer sliding-window 2048 -- with the
target-tap projection named ``encoder.fc`` / ``encoder.output_norm_enc`` and
``target_layer_ids`` / ``mask_token_id`` / ``block_size`` at the config root.
It has no vocabulary head: proposals are projected through the target's
raw ``lm_head`` -- not the target's logits tail.  The reference candidate
generator (transformers ``DFlashTokenCandidateGenerator``) applies only the
output embedding; the drafter was trained against those logits.  Muse's
tail multiplies by ``output_multiplier`` (~0.196) and softcaps, which leaves
the argmax unchanged but flattens the sampled draft law about five-fold: at
the model card's default sampling the drafter then put ~0.1% of its mass on
the target's support and no sampled draft was ever accepted.

The proposal block is the anchor plus ``block_size - 1`` mask tokens with
bidirectional attention inside the block (the reference's
``create_bidirectional_sliding_window_mask``); rows past ``num_draft`` are
computed and discarded.  Block inputs are fixed (anchor and mask tokens), so
position ``j`` never depends on the drafts sampled at other positions and
``softmax(logits_j / T)`` is the exact law ``verify_proposals`` needs.

It is distinct from the DFlash2 closure (no candidate selector, no dynamic
convolutions).

No capability is qualified by importing or constructing this module.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .cohere_eagle import EagleWindowKVCache, _Rows

ARCHITECTURE = "MuseGlimmerAssistantModel"
# v2: raw lm_head logits and the bidirectional full block (reference law);
# v1 projected through the target's logits tail with a causal block.
MUSE_ASSISTANT_LAW_VERSION = "muse-assistant-raw-head-bidirectional-v2"
MODEL_TYPE = "muse_glimmer_assistant"
_ENCODER_KEY_REMAP = (
    ("encoder.fc.", "fc."),
    ("encoder.output_norm_enc.", "hidden_norm."),
)
_LINEARS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


@dataclass
class MuseDFlashConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    num_target_layers: int
    target_layer_ids: list[int]
    mask_token_id: int
    block_size: int
    sliding_window: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    quantization: dict | None = None
    layer_types: list[str] = field(default_factory=list)
    model_type: str = MODEL_TYPE

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = ["sliding_attention"] * self.num_hidden_layers
        self.validate()

    def validate(self):
        for key in (
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "vocab_size", "num_target_layers", "block_size", "sliding_window",
        ):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Muse DFlash requires a positive integer {key}")
        if self.block_size < 2 or self.sliding_window < 2:
            raise ValueError("Muse DFlash block and window must exceed one")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Muse DFlash heads must be divisible by KV heads")
        if len(self.layer_types) != self.num_hidden_layers or any(
            kind != "sliding_attention" for kind in self.layer_types
        ):
            raise ValueError("Muse DFlash requires sliding-attention draft layers")
        ids = list(self.target_layer_ids)
        if not ids or ids != sorted(set(ids)) or ids[0] < 0 or ids[-1] >= self.num_target_layers:
            raise ValueError("Muse DFlash target_layer_ids must be increasing target layers")
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError("Muse DFlash mask_token_id must be inside the target vocabulary")
        if self.quantization is not None:
            quant = self.quantization
            if (
                not isinstance(quant, Mapping)
                or set(quant) - {"group_size", "bits", "mode"}
                or quant.get("mode", "affine") != "affine"
                or quant.get("bits") not in (2, 3, 4, 5, 6, 8)
                or quant.get("group_size") not in (32, 64, 128)
            ):
                raise ValueError("Muse DFlash supports uniform affine quantization only")

    @classmethod
    def from_hf(cls, config: Mapping, *, target_text: Mapping):
        """Normalize a root-keyed assistant config against its target."""
        if config.get("architectures") != [ARCHITECTURE] or config.get("model_type") != MODEL_TYPE:
            raise ValueError(f"Expected a {ARCHITECTURE} draft artifact")
        if "dflash_config" in config:
            raise ValueError("Muse assistant checkpoints carry DFlash keys at the config root")
        for key in ("target_layer_ids", "mask_token_id", "block_size"):
            # dflash-mlx: a missing key would silently default (mask id 0,
            # evenly spaced taps) and collapse acceptance; refuse instead.
            if config.get(key) is None:
                raise ValueError(f"Muse assistant config is missing {key!r}")
        if config.get("attention_bias") or config.get("hidden_act", "silu") != "silu":
            raise ValueError("Muse DFlash requires bias-free SwiGLU layers")
        rope = config.get("rope_parameters") or {}
        if not isinstance(rope, Mapping) or rope.get("rope_type", "default") != "default":
            raise ValueError("Muse DFlash supports default full-head RoPE only")
        if config.get("partial_rotary_factor", 1.0) not in (None, 1, 1.0):
            raise ValueError("Muse DFlash supports full-head RoPE only")
        if int(target_text["hidden_size"]) != int(config["hidden_size"]):
            raise ValueError("Muse DFlash target hidden_size mismatch")
        return cls(
            hidden_size=int(config["hidden_size"]),
            intermediate_size=int(config["intermediate_size"]),
            num_hidden_layers=int(config["num_hidden_layers"]),
            num_attention_heads=int(config["num_attention_heads"]),
            num_key_value_heads=int(config["num_key_value_heads"]),
            head_dim=int(config["head_dim"]),
            vocab_size=int(target_text["vocab_size"]),
            num_target_layers=int(target_text["num_hidden_layers"]),
            target_layer_ids=[int(v) for v in config["target_layer_ids"]],
            mask_token_id=int(config["mask_token_id"]),
            block_size=int(config["block_size"]),
            sliding_window=int(config["sliding_window"]),
            rms_norm_eps=float(config.get("rms_norm_eps", 1e-5)),
            rope_theta=float(rope.get("rope_theta", config.get("rope_theta", 500000.0))),
            quantization=config.get("quantization"),
            layer_types=list(config.get("layer_types") or []),
        )


def expected_weights(config: MuseDFlashConfig) -> dict[str, tuple[str, list[int]]]:
    """Checkpoint tensor names (artifact spelling) -> (dtype class, shape)."""
    hidden = config.hidden_size
    quant = config.quantization
    out: dict[str, tuple[str, list[int]]] = {}

    def linear(prefix: str, rows: int, cols: int):
        if quant is None:
            out[prefix + ".weight"] = ("float", [rows, cols])
            return
        bits, group = int(quant["bits"]), int(quant["group_size"])
        out[prefix + ".weight"] = ("U32", [rows, cols * bits // 32])
        out[prefix + ".scales"] = ("float", [rows, cols // group])
        out[prefix + ".biases"] = ("float", [rows, cols // group])

    linear("encoder.fc", hidden, len(config.target_layer_ids) * hidden)
    out["encoder.output_norm_enc.weight"] = ("float", [hidden])
    out["norm.weight"] = ("float", [hidden])
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    for index in range(config.num_hidden_layers):
        p = f"layers.{index}."
        out[p + "input_layernorm.weight"] = ("float", [hidden])
        out[p + "post_attention_layernorm.weight"] = ("float", [hidden])
        out[p + "self_attn.q_norm.weight"] = ("float", [config.head_dim])
        out[p + "self_attn.k_norm.weight"] = ("float", [config.head_dim])
        linear(p + "self_attn.q_proj", q, hidden)
        linear(p + "self_attn.k_proj", kv, hidden)
        linear(p + "self_attn.v_proj", kv, hidden)
        linear(p + "self_attn.o_proj", hidden, q)
        linear(p + "mlp.gate_proj", config.intermediate_size, hidden)
        linear(p + "mlp.up_proj", config.intermediate_size, hidden)
        linear(p + "mlp.down_proj", hidden, config.intermediate_size)
    return out


class MuseDFlashAttention(nn.Module):
    def __init__(self, config: MuseDFlashConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.window = config.sliding_window
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def _kv(self, x):
        batch, length, _ = x.shape
        k = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        return self.k_norm(k).transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)

    def context_kv(self, context, cache):
        """Commit one row's context K/V (RoPE at absolute positions)."""
        k, v = self._kv(context)
        cache.append(self.rope(k, offset=cache.offset), v)

    def block(self, x, cache):
        """One row's proposal block over committed context (not stored).

        Reference sliding-layer law: a context key at position ``p`` is
        visible to a query at ``q`` only when ``q - p < window``; every block
        key is visible to every block query (bidirectional; the block is far
        shorter than the window).  The resident ring holds the last
        ``window - 1`` keys.
        """
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        q = self.rope(self.q_norm(q).transpose(0, 2, 1, 3), offset=cache.offset)
        k, v = self._kv(x)
        k = self.rope(k, offset=cache.offset)
        ctx_k, ctx_v = cache.keys_and_values()
        resident = int(ctx_k.shape[2])
        keys = mx.concatenate([ctx_k.astype(k.dtype), k], axis=2) if resident else k
        values = mx.concatenate([ctx_v.astype(v.dtype), v], axis=2) if resident else v
        if resident:
            # Context column c holds absolute position offset - resident + c.
            rows = mx.arange(length)[:, None]
            columns = mx.arange(resident)[None, :]
            context = columns > rows + resident - self.window
            block = mx.ones((length, length), dtype=mx.bool_)
            mask = mx.concatenate([context, block], axis=-1)
        else:
            mask = None
        out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class MuseDFlashMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MuseDFlashLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MuseDFlashAttention(config)
        self.mlp = MuseDFlashMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class MuseDFlashDraftModel(nn.Module):
    supports_logits_processors = True
    receipt_kind = "external_muse_dflash_assistant"
    proposal_distribution = "stochastic_exact_law"

    def __init__(self, config: MuseDFlashConfig):
        super().__init__()
        self.config = config
        self.layers = [MuseDFlashLayer(config) for _ in range(config.num_hidden_layers)]
        self.fc = nn.Linear(len(config.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._target = None
        self.stats = {"committed_positions": 0, "block_drafts": 0}

    @property
    def receipt_settings(self):
        return {
            "kind": self.receipt_kind,
            "trained_block_size": self.config.block_size,
            "target_layer_ids": list(self.config.target_layer_ids),
            "sliding_window": self.config.sliding_window,
            "block_attention": "bidirectional_full_block",
            "proposal_logits": "target_lm_head_raw",
            "proposal_distribution": "dense_target_head_law",
            "law_version": MUSE_ASSISTANT_LAW_VERSION,
            "quantization": self.config.quantization,
        }

    def validate_target_compatibility(self, target_model):
        args = target_model.args
        for name in ("hidden_size", "vocab_size"):
            if getattr(args, name) != getattr(self.config, name):
                raise ValueError(f"Muse DFlash target {name} mismatch")
        if len(target_model.model.layers) != self.config.num_target_layers:
            raise ValueError("Muse DFlash target layer count mismatch")

    def bind(self, target_model):
        self.validate_target_compatibility(target_model)
        # Keep the target out of this module's parameter tree.
        object.__setattr__(self, "_target", target_model)
        return self

    def sanitize(self, weights):
        out = {}
        for key, value in weights.items():
            key = key.removeprefix("model.")
            for prefix, replacement in _ENCODER_KEY_REMAP:
                if key.startswith(prefix):
                    key = replacement + key[len(prefix):]
                    break
            if key.startswith("encoder."):
                raise ValueError(f"Unrecognized Muse assistant encoder weight: {key}")
            if key in out:
                raise ValueError(f"Duplicate Muse DFlash weight after sanitize: {key}")
            out[key] = value
        return out

    def make_cache(self):
        c = self.config
        return [
            EagleWindowKVCache(c.sliding_window - 1, c.num_key_value_heads, c.head_dim)
            for _ in self.layers
        ]

    def batch_caches(self, rows):
        if not rows or any(len(row) != len(self.layers) for row in rows):
            raise ValueError("Muse DFlash cache row topology mismatch")
        return _Rows([list(row) for row in rows])

    @staticmethod
    def _rows(cache):
        return cache.rows if isinstance(cache, _Rows) else [cache]

    def append_context(self, hidden, cache):
        if hidden.shape[1] == 0:
            return
        rows = self._rows(cache)
        context = self.hidden_norm(self.fc(hidden))
        for row, caches in enumerate(rows):
            for layer, entry in zip(self.layers, caches):
                layer.self_attn.context_kv(context[row:row + 1], entry)
        self.stats["committed_positions"] += int(hidden.shape[1]) * len(rows)

    def _logits(self, hidden):
        """The reference candidate logits: the target's output embedding only.

        Not the target's logits tail: ``output_multiplier`` and the final
        softcap belong to the target's own head, not to the drafter's.
        """
        target = self._target
        if target.tie_word_embeddings:
            return target.model.embed_tokens.as_linear(hidden)
        return target.lm_head(hidden)

    def _block_logits(self, anchors, rows, count):
        count = int(count)
        if not 1 <= count < self.config.block_size:
            raise ValueError("Muse DFlash proposal length must be below the trained block size")
        # The trained block: anchor plus block_size - 1 mask tokens.
        tokens = mx.concatenate(
            [mx.array(anchors, dtype=mx.int32)[:, None],
             mx.full((len(anchors), self.config.block_size - 1), self.config.mask_token_id,
                     dtype=mx.int32)], axis=1
        )
        # The drafter consumes the target's raw token embedding (the target's
        # scaleless embedding RMSNorm belongs to the target trunk).
        x = self._target.model.embed_tokens(tokens)
        for index, layer in enumerate(self.layers):
            h = layer.input_layernorm(x)
            attn = mx.concatenate(
                [layer.self_attn.block(h[row:row + 1], caches[index]) for row, caches in enumerate(rows)],
                axis=0,
            )
            x = x + attn
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self._logits(self.norm(x)[:, 1:1 + count])

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


Model = MuseDFlashDraftModel

__all__ = ["ARCHITECTURE", "MUSE_ASSISTANT_LAW_VERSION", "Model", "MuseDFlashConfig", "MuseDFlashDraftModel", "expected_weights"]
