# SPDX-License-Identifier: MIT
# Architecture follows vLLM vllm/model_executor/models/cohere_eagle.py and
# commandr.py (Apache-2.0); see provenance/north-cohere-eagle.json. The chain
# drafting loop, cache and exact-law export are original mlx2 work.
"""Cohere EAGLE chain drafter for the external draft/verify route.

The official North Mini Code drafter (``EagleCohereForCausalLM``) is an
EAGLE-1 style head: each draft position fuses the embedding of the *next*
token with one target feature (the target's final-norm hidden state),

    x_t = fc(cat(embed(token_{t+1}), feature_t))          # fc has a bias
    x   = x + attn(rmsnorm(x)) + mlp(rmsnorm(x))          # 3 parallel blocks
    h_t = layernorm(x)                                    # explicit final norm
    logits = lm_head_target(h_t) * logit_scale            # shared vocab head

and proposes a chain: the draft's own ``h_t`` replaces the target feature for
the following step.  Only positions backed by a real target feature ever
enter the draft KV cache; chain positions live in round-local KV and are
discarded, so the draft plane stays paired with the committed target
boundary exactly as ``ExternalDraftState.validate`` requires.

No capability is qualified by importing or constructing this module.
"""
from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..models.cache import _BaseCache

_ARCHITECTURE = "EagleCohereForCausalLM"


@dataclass
class CohereEagleConfig:
    hidden_size: int = 2048
    intermediate_size: int = 768
    num_hidden_layers: int = 3
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    vocab_size: int = 262144
    rms_norm_eps: float = 1e-6
    layer_norm_eps: float = 1e-5
    rope_theta: float = 50000.0
    sliding_window: int = 4096
    max_position_embeddings: int = 5_000_000
    logit_scale: float = 1.0
    layer_types: list[str] = field(default_factory=list)
    # Target contract.  EAGLE-1 consumes one feature: the target's final-norm
    # hidden state, addressed by the tap sentinel ``num_target_layers``.
    num_target_layers: int = 49
    target_layer_ids: list[int] = field(default_factory=list)
    # The executor requires ``num_draft < block_size``; a chain has no trained
    # block, so this is the operator ceiling on proposals per round.
    block_size: int = 8
    model_type: str = "cohere_eagle"

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = ["sliding_attention"] * self.num_hidden_layers
        if not self.target_layer_ids:
            self.target_layer_ids = [self.num_target_layers]
        self.validate()

    def validate(self):
        for key in (
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "vocab_size", "sliding_window", "num_target_layers", "block_size",
        ):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Cohere EAGLE requires a positive integer {key}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Cohere EAGLE heads must be divisible by KV heads")
        if len(self.layer_types) != self.num_hidden_layers or any(
            kind != "sliding_attention" for kind in self.layer_types
        ):
            raise ValueError("Cohere EAGLE supports sliding-attention draft layers only")
        if list(self.target_layer_ids) != [self.num_target_layers]:
            raise ValueError(
                "Cohere EAGLE-1 consumes exactly the target final-norm feature"
            )
        if self.block_size < 2:
            raise ValueError("Cohere EAGLE block_size must allow one proposal")

    @classmethod
    def from_hf(cls, config: Mapping, *, num_target_layers: int, block_size: int = 8):
        """Strict mapping of the published ``config.json``."""
        if config.get("architectures") != [_ARCHITECTURE]:
            raise ValueError(f"Expected a {_ARCHITECTURE} draft artifact")
        required = {
            "transformer_block_type": "parallel",
            "use_qk_norm": False,
            "position_embedding_type": "rope_gptj",
            "rope_scaling": None,
            "attention_bias": False,
            "hidden_act": "silu",
            "use_gated_activation": True,
            # Pinned, not decorative: this one key is the drafter's whole
            # statement about which norm it was trained with, and RMSNorm and
            # LayerNorm have identical parameter shapes here, so a drafter that
            # said something else would load clean and mis-normalize.
            "norm_type": "rms_norm",
        }
        for key, value in required.items():
            if config.get(key) != value:
                raise ValueError(f"Cohere EAGLE {key} must be {value!r}")
        flat = dict(config)
        flat["num_target_layers"] = int(num_target_layers)
        flat["block_size"] = int(block_size)
        flat.pop("model_type", None)
        flat.setdefault("logit_scale", 1.0)
        signature = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in signature})


def expected_weight_shapes(config: CohereEagleConfig) -> dict[str, list[int]]:
    """Raw published schema (``model.``-prefixed), before ``sanitize``."""
    hidden = config.hidden_size
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    inter = config.intermediate_size
    shapes = {
        "model.fc.weight": [hidden, 2 * hidden],
        "model.fc.bias": [hidden],
        "model.norm.weight": [hidden],
    }
    for index in range(config.num_hidden_layers):
        p = f"model.layers.{index}."
        shapes.update({
            p + "input_layernorm.weight": [hidden],
            p + "self_attn.q_proj.weight": [q, hidden],
            p + "self_attn.k_proj.weight": [kv, hidden],
            p + "self_attn.v_proj.weight": [kv, hidden],
            p + "self_attn.o_proj.weight": [hidden, q],
            p + "self_attn.o_proj.bias": [hidden],
            p + "mlp.gate_proj.weight": [inter, hidden],
            p + "mlp.gate_proj.bias": [inter],
            p + "mlp.up_proj.weight": [inter, hidden],
            p + "mlp.up_proj.bias": [inter],
            p + "mlp.down_proj.weight": [hidden, inter],
            p + "mlp.down_proj.bias": [hidden],
        })
    return shapes


class EagleWindowKVCache(_BaseCache):
    """Committed draft KV in temporal order, bounded by the sliding window.

    ``offset`` counts committed positions (target features consumed).  Only
    the last ``window`` positions are resident.  Registered as a model-local
    cache class, so APCv2 disk persistence resolves it by ``module:Class``.
    """

    def __init__(self, window: int, n_kv_heads: int = 1, head_dim: int = 1):
        self.window = int(window)
        self.offset = 0
        self.keys = mx.zeros((1, n_kv_heads, 0, head_dim))
        self.values = mx.zeros_like(self.keys)

    def append(self, keys, values):
        if self.keys.shape[2] == 0:
            joined_k, joined_v = keys, values
        else:
            joined_k = mx.concatenate([self.keys, keys.astype(self.keys.dtype)], axis=2)
            joined_v = mx.concatenate([self.values, values.astype(self.values.dtype)], axis=2)
        if joined_k.shape[2] > self.window:
            joined_k = joined_k[:, :, -self.window:]
            joined_v = joined_v[:, :, -self.window:]
        self.keys, self.values = joined_k, joined_v
        self.offset += int(keys.shape[2])

    def keys_and_values(self):
        return self.keys, self.values

    def size(self):
        return self.offset

    @property
    def state(self):
        return (self.keys, self.values)

    @state.setter
    def state(self, value):
        self.keys, self.values = value

    @property
    def meta_state(self):
        return (str(self.offset), str(self.window))

    @meta_state.setter
    def meta_state(self, value):
        self.offset, self.window = int(value[0]), int(value[1])

    def empty(self):
        return self.offset == 0

    @property
    def nbytes(self):
        return int(self.keys.nbytes + self.values.nbytes)


class _Rows:
    """Per-lane cache rows for one batched draft call (lanes keep own planes)."""

    __slots__ = ("rows",)

    def __init__(self, rows):
        self.rows = rows


class CohereEagleAttention(nn.Module):
    def __init__(self, config: CohereEagleConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        # vLLM: a Cohere SWA layer sees [pos - sliding_window, pos].
        self.window = config.sliding_window
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=True)
        # Cohere pairs RoPE dimensions GPT-J style (interleaved).
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=config.rope_theta)

    def project(self, x):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        return q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)

    def attend(self, q, k, v, *, position, prior_k, prior_v, prior_start):
        """One row.  ``q/k/v``: [1, H, L, D] for positions ``position..+L-1``;
        ``prior_*`` are earlier keys starting at absolute ``prior_start``."""
        length = q.shape[2]
        q = self.rope(q, offset=position)
        k = self.rope(k, offset=position)
        keys = mx.concatenate([prior_k, k.astype(prior_k.dtype)], axis=2) if prior_k.shape[2] else k
        values = mx.concatenate([prior_v, v.astype(prior_v.dtype)], axis=2) if prior_v.shape[2] else v
        key_pos = mx.arange(prior_start, position + length)
        query_pos = mx.arange(position, position + length)
        delta = query_pos[:, None] - key_pos[None, :]
        mask = (delta >= 0) & (delta <= self.window)
        out = mx.fast.scaled_dot_product_attention(
            q, keys.astype(q.dtype), values.astype(q.dtype), scale=self.scale, mask=mask
        )
        return out, k


class CohereEagleMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=True)
        self.up_proj = nn.Linear(dim, hidden, bias=True)
        self.down_proj = nn.Linear(hidden, dim, bias=True)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class CohereEagleLayer(nn.Module):
    def __init__(self, config: CohereEagleConfig):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = CohereEagleAttention(config)
        self.mlp = CohereEagleMLP(config.hidden_size, config.intermediate_size)


class CohereEagleDraftModel(nn.Module):
    """External-draft contract: ``append_context``, ``draft_distributions``,
    ``make_cache``, ``batch_caches``; ``requires_context_tokens`` asks the
    executor for the token that follows each committed target feature."""

    requires_context_tokens = True
    supports_logits_processors = True
    receipt_kind = "external_cohere_eagle"

    def __init__(self, config: CohereEagleConfig):
        super().__init__()
        self.config = config
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=True)
        self.layers = [CohereEagleLayer(config) for _ in range(config.num_hidden_layers)]
        # RMSNorm, not LayerNorm (fixed 2026-09-20 with the target's norm; see
        # runtime/models/cohere2_moe._norm_layer).  Three reasons, none of them
        # the weights -- ``model.norm.weight`` is a bare ``[hidden]`` gamma with
        # no bias, which both norms would accept:
        #   1. the drafter's published config declares ``norm_type: "rms_norm"``
        #      once for the whole model, and from_hf now pins that value;
        #   2. this drafter's own decoder layers already build RMSNorm from the
        #      same config (CohereEagleLayer.input_layernorm) -- a LayerNorm
        #      here was inconsistent within one model;
        #   3. structural: EAGLE-1's final hidden is projected by the *bound
        #      target's* lm_head (see bind()).  That head reads hiddens the
        #      target's own final norm produced, and the target's final norm is
        #      RMSNorm.  A mean-centred draft hidden is in a different geometry
        #      than the head it is fed to.
        # Caveat: vLLM's cohere_eagle.py, which this port follows, is not on
        # this machine, so the reference was not re-read for this line.
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_tokens = None
        self.lm_head = None
        # Default-on, sync-free mechanism counters (Python ints only).
        self.stats = {"committed_positions": 0, "chain_steps": 0, "draft_calls": 0}

    # -- binding ---------------------------------------------------------
    def validate_target_compatibility(self, target_model):
        args = target_model.args
        for name in ("hidden_size", "vocab_size"):
            if getattr(args, name) != getattr(self.config, name):
                raise ValueError(f"Cohere EAGLE target {name} mismatch")
        if args.num_hidden_layers != self.config.num_target_layers:
            raise ValueError("Cohere EAGLE target layer count mismatch")

    def bind(self, target_model):
        self.validate_target_compatibility(target_model)
        inner = target_model.model
        self.embed_tokens = inner.embed_tokens
        head = getattr(target_model, "lm_head", None)
        self.lm_head = head if head is not None else inner.embed_tokens.as_linear
        return self

    def sanitize(self, weights):
        out = {}
        for key, value in weights.items():
            key = key.removeprefix("model.")
            if key.startswith(("embed_tokens.", "lm_head.")):
                continue  # shared with the bound target
            if key in out:
                raise ValueError(f"Duplicate Cohere EAGLE weight after sanitize: {key}")
            out[key] = value
        return out

    # -- cache contract ----------------------------------------------------
    def make_cache(self):
        c = self.config
        return [
            EagleWindowKVCache(c.sliding_window, c.num_key_value_heads, c.head_dim)
            for _ in self.layers
        ]

    def batch_caches(self, rows):
        if not rows or any(len(row) != len(self.layers) for row in rows):
            raise ValueError("Cohere EAGLE cache row topology mismatch")
        return _Rows([list(row) for row in rows])

    @staticmethod
    def _rows(cache):
        return cache.rows if isinstance(cache, _Rows) else [cache]

    # -- math --------------------------------------------------------------
    def _logits(self, hidden):
        return self.lm_head(hidden) * self.config.logit_scale

    def _forward(self, tokens, features, rows, *, commit, chain=None):
        """Run the draft stack for B rows at L positions.

        ``tokens``: [B, L] next-token ids; ``features``: [B, L, D].
        ``commit``: write this KV into the committed row caches.  Otherwise
        ``chain`` is a per-row list of per-layer ``(k, v)`` round-local KV,
        extended in place.  Returns the final-norm hidden [B, L, D].
        """
        embeds = self.embed_tokens(tokens)
        x = self.fc(mx.concatenate([embeds, features.astype(embeds.dtype)], axis=-1))
        batch, length, _ = x.shape
        for index, layer in enumerate(self.layers):
            h = layer.input_layernorm(x)
            attn = layer.self_attn
            q, k, v = attn.project(h)
            outputs = []
            for row, caches in enumerate(rows):
                cache = caches[index]
                prior_k, prior_v = cache.keys_and_values()
                local = None if commit else chain[row][index]
                if local is not None and local[0] is not None:
                    prior_k = mx.concatenate([prior_k, local[0]], axis=2) if prior_k.shape[2] else local[0]
                    prior_v = mx.concatenate([prior_v, local[1]], axis=2) if prior_v.shape[2] else local[1]
                position = cache.offset + (0 if local is None or local[0] is None else local[0].shape[2])
                out, rotated = attn.attend(
                    q[row:row + 1], k[row:row + 1], v[row:row + 1],
                    position=position, prior_k=prior_k, prior_v=prior_v,
                    prior_start=position - prior_k.shape[2],
                )
                outputs.append(out)
                if commit:
                    cache.append(rotated, v[row:row + 1])
                else:
                    old = chain[row][index]
                    if old[0] is None:
                        chain[row][index] = (rotated, v[row:row + 1])
                    else:
                        chain[row][index] = (
                            mx.concatenate([old[0], rotated], axis=2),
                            mx.concatenate([old[1], v[row:row + 1]], axis=2),
                        )
            o = mx.concatenate(outputs, axis=0).transpose(0, 2, 1, 3).reshape(batch, length, -1)
            x = x + attn.o_proj(o) + layer.mlp(h)
        return self.norm(x)

    def append_context(self, hidden, cache, *, context_tokens=None):
        """Commit target features (paired with their next tokens) to draft KV."""
        if hidden.shape[1] == 0:
            return None
        if context_tokens is None:
            raise ValueError("Cohere EAGLE needs the token following each feature")
        rows = self._rows(cache)
        tokens = mx.array(np.asarray(context_tokens, dtype=np.int32).reshape(len(rows), -1))
        if tokens.shape[1] != hidden.shape[1]:
            raise ValueError("Cohere EAGLE context tokens must pair with features")
        out = self._forward(tokens, hidden, rows, commit=True)
        self.stats["committed_positions"] += int(hidden.shape[1]) * len(rows)
        return out

    def draft_distributions(
        self, anchors, hidden, cache, proposal_length, rngs, temperatures, *,
        context_tokens=None, logits_processors=None, processor_histories=None,
    ):
        """Chain-draft ``proposal_length`` tokens; return tokens and exact q laws."""
        from ..processor_probe import probe_logits_processors
        from ..speculative_sampling import softmax

        rows = self._rows(cache)
        batch = len(rows)
        anchor_values = [int(a) for a in anchors]
        if context_tokens is None:
            # A pending tail is always paired with the anchor as its last token.
            context_tokens = [[a] for a in anchor_values] if hidden.shape[1] else None
        if hidden.shape[1] == 0:
            # No target feature yet (one-token prompt): nothing to chain from.
            # Zero proposals make this an ordinary verify round, never a guess.
            return [[] for _ in range(batch)], [[] for _ in range(batch)]
        logits_processors = logits_processors or [[] for _ in range(batch)]
        processor_histories = processor_histories or [[] for _ in range(batch)]
        top = self.append_context(hidden, cache, context_tokens=context_tokens)[:, -1:]
        self.stats["draft_calls"] += 1
        chain = [[(None, None) for _ in self.layers] for _ in range(batch)]
        tokens = [[] for _ in range(batch)]
        laws = [[] for _ in range(batch)]
        for step in range(int(proposal_length)):
            if step:
                previous = mx.array([[row[-1]] for row in tokens], dtype=mx.int32)
                top = self._forward(previous, top, rows, commit=False, chain=chain)
                self.stats["chain_steps"] += 1
            logits = self._logits(top[:, -1, :]).astype(mx.float32)
            dense = np.asarray(logits)
            for row in range(batch):
                value = dense[row]
                if logits_processors[row]:
                    prefix = mx.array(
                        list(processor_histories[row]) + [anchor_values[row]] + tokens[row],
                        dtype=mx.int32,
                    )
                    value = np.asarray(
                        probe_logits_processors(
                            logits_processors[row], prefix, mx.array(value)[None]
                        )[0].astype(mx.float32)
                    )
                    if not np.isfinite(value).any():
                        value = dense[row]  # grammar dead end: q stays a valid law
                q = softmax(value, temperatures[row])
                token = int(rngs[row].sample(q))
                tokens[row].append(token)
                laws[row].append(q)
        return tokens, laws


Model = CohereEagleDraftModel

__all__ = [
    "CohereEagleConfig",
    "CohereEagleDraftModel",
    "EagleWindowKVCache",
    "Model",
    "expected_weight_shapes",
]
