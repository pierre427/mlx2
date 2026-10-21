# SPDX-License-Identifier: MIT
# Architecture follows vLLM vllm/model_executor/models/llama_eagle3.py and the
# speculators Eagle3Speculator reference (both Apache-2.0); see
# provenance/eagle3-drafter.json.  The chain drafting loop, committed-only
# draft cache and exact-law export are original mlx2 work.
"""EAGLE-3 chain drafter for the external draft/verify route.

EAGLE-3 fuses three target hidden states (low/mid/high) and the embedding of
the token that *follows* each feature through one llama-style decoder layer:

    f_t     = fc(cat(h_low_t, h_mid_t, h_high_t))            # no bias
    e_t     = input_layernorm(embed(token_{t+1}))
    g, res  = hidden_norm(f_t), hidden_norm(f_t)             # norm_before_residual
            | hidden_norm(f_t), f_t                          # otherwise
    x       = res + o_proj(attn(cat(e_t, g)))                # q/k/v read 2*hidden
    p_t     = x + mlp(post_attention_layernorm(x))           # "prenorm"
    logits  = lm_head(norm(p_t))                             # draft vocabulary
    target  = draft_id + d2t[draft_id]                       # reduced -> target ids

A chain step feeds the draft's own ``p_t`` (already hidden-size, no fc) back
as the next feature, exactly as vLLM's proposer feeds ``hidden_prenorm``.
Only target-backed positions enter the committed draft KV; chain positions
live in round-local KV and are discarded, so the draft plane stays paired with
the committed target boundary.

Tap convention: an EAGLE-3 auxiliary layer id ``k`` (vLLM) is the residual
stream after decoder layer ``k - 1``; mlx2 taps name the decoder layer whose
output they capture, so ``target_layer_ids = [k - 1 for k in aux_ids]`` (the
same mapping vLLM's speculators config uses for DFlash).

No capability is qualified by importing or constructing this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np
from mlx import nn

from .cohere_eagle import EagleWindowKVCache, _Rows

ARCHITECTURE = "Eagle3Speculator"
EAGLE3_NUMERICS_VERSION = "mlx2.eagle3-chain.v1"


def default_aux_layer_ids(num_target_layers: int) -> list[int]:
    """vLLM ``get_eagle3_default_aux_hidden_state_layers``: (2, L // 2, L - 3)."""
    return [2, num_target_layers // 2, num_target_layers - 3]


@dataclass
class Eagle3Config:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 1
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    draft_vocab_size: int = 32000
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 40960
    norm_before_residual: bool = True
    target_hidden_size: int | None = None
    logit_scale: float = 1.0
    # Target contract.  ``aux_layer_ids`` uses the EAGLE-3/vLLM convention;
    # ``target_layer_ids`` is the derived mlx2 tap list the executor reads.
    num_target_layers: int = 36
    aux_layer_ids: list[int] = field(default_factory=list)
    target_layer_ids: list[int] = field(default_factory=list)
    # Executor requires ``num_draft < block_size``; a chain has no trained
    # block, so this is the operator ceiling on proposals per round.
    block_size: int = 8
    has_embed_tokens: bool = True
    model_type: str = "eagle3"

    def __post_init__(self):
        if self.target_hidden_size is None:
            self.target_hidden_size = self.hidden_size
        if not self.aux_layer_ids:
            self.aux_layer_ids = default_aux_layer_ids(self.num_target_layers)
        derived = [int(k) - 1 for k in self.aux_layer_ids]
        if not self.target_layer_ids:
            self.target_layer_ids = derived
        self.validate()

    @property
    def feature_width(self) -> int:
        return len(self.target_layer_ids) * int(self.target_hidden_size)

    def validate(self):
        for key in (
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "vocab_size", "draft_vocab_size", "max_position_embeddings",
            "num_target_layers", "block_size", "target_hidden_size",
        ):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"EAGLE-3 requires a positive integer {key}")
        if self.num_hidden_layers != 1:
            raise ValueError("EAGLE-3 supports exactly one draft decoder layer")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("EAGLE-3 heads must be divisible by KV heads")
        if self.draft_vocab_size > self.vocab_size:
            raise ValueError("EAGLE-3 draft vocabulary exceeds the target vocabulary")
        if type(self.norm_before_residual) is not bool:
            raise ValueError("EAGLE-3 norm_before_residual must be boolean")
        ids = list(self.aux_layer_ids)
        if (
            len(ids) != 3
            or any(type(k) is not int for k in ids)
            or ids != sorted(set(ids))
            or ids[0] < 1
            or ids[-1] > self.num_target_layers
        ):
            raise ValueError("EAGLE-3 needs three increasing auxiliary layer ids")
        if list(self.target_layer_ids) != [k - 1 for k in ids]:
            raise ValueError("EAGLE-3 target taps must be auxiliary ids minus one")
        if self.block_size < 2:
            raise ValueError("EAGLE-3 block_size must allow one proposal")
        for key in ("rms_norm_eps", "rope_theta", "logit_scale"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"Invalid EAGLE-3 {key}")

    @classmethod
    def from_speculators(
        cls, config, *, num_target_layers: int, block_size: int = 8,
        has_embed_tokens: bool = True,
    ):
        """Strict mapping of a ``speculators`` (``Eagle3Speculator``) config.

        Variants this port does not implement (norm_before_fc, fc_norm,
        norm_output, parallel drafting, non-llama layers, biases, RoPE
        scaling) fail closed instead of loading with silently wrong math.
        """
        if not isinstance(config, dict) or config.get("architectures") != [ARCHITECTURE]:
            raise ValueError(f"Expected a {ARCHITECTURE} draft artifact")
        if config.get("speculators_model_type") != "eagle3":
            raise ValueError("EAGLE-3 speculators_model_type must be 'eagle3'")
        algorithm = (config.get("speculators_config") or {}).get("algorithm")
        if algorithm != "eagle3":
            raise ValueError("EAGLE-3 speculators_config.algorithm must be 'eagle3'")
        for key in ("norm_before_fc", "fc_norm", "norm_output", "parallel_drafting"):
            if config.get(key, False) not in (False, None):
                raise ValueError(f"EAGLE-3 variant {key} is not implemented")
        layer = config.get("transformer_layer_config")
        if not isinstance(layer, dict) or layer.get("model_type") != "llama":
            raise ValueError("EAGLE-3 supports llama draft decoder layers only")
        required = {
            "attention_bias": False,
            "mlp_bias": False,
            "hidden_act": "silu",
            "rope_scaling": None,
            "num_hidden_layers": 1,
        }
        for key, value in required.items():
            if layer.get(key, value if key == "mlp_bias" else None) != value:
                raise ValueError(f"EAGLE-3 draft layer {key} must be {value!r}")
        heads = int(layer["num_attention_heads"])
        aux = config.get("eagle_aux_hidden_state_layer_ids") or (
            (config.get("eagle_config") or {}).get("eagle_aux_hidden_state_layer_ids")
        )
        return cls(
            hidden_size=int(layer["hidden_size"]),
            intermediate_size=int(layer["intermediate_size"]),
            num_hidden_layers=1,
            num_attention_heads=heads,
            num_key_value_heads=int(layer.get("num_key_value_heads", heads)),
            head_dim=int(layer.get("head_dim", int(layer["hidden_size"]) // heads)),
            vocab_size=int(layer["vocab_size"]),
            draft_vocab_size=int(config.get("draft_vocab_size") or layer["vocab_size"]),
            rms_norm_eps=float(layer["rms_norm_eps"]),
            rope_theta=float(layer.get("rope_theta", 10000.0)),
            max_position_embeddings=int(layer["max_position_embeddings"]),
            # vLLM's speculators translation defaults this to True.
            norm_before_residual=bool(config.get("norm_before_residual", True)),
            target_hidden_size=config.get("target_hidden_size"),
            logit_scale=float(layer.get("logit_scale", 1.0)),
            num_target_layers=int(num_target_layers),
            aux_layer_ids=list(aux) if aux else [],
            block_size=int(block_size),
            has_embed_tokens=bool(has_embed_tokens),
        )


def expected_weight_schema(config: Eagle3Config) -> dict[str, tuple[str, list[int]]]:
    """Published speculators tensor schema: name -> (dtype code, shape).

    Floating tensors carry the checkpoint dtype (filled in by the inspector
    as ``None``); ``d2t`` is int64 and ``t2d`` is bool.
    """
    h = config.hidden_size
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    inter = config.intermediate_size
    p = "layers.0."
    schema = {
        "fc.weight": (None, [h, config.feature_width]),
        "norm.weight": (None, [h]),
        "lm_head.weight": (None, [config.draft_vocab_size, h]),
        p + "input_layernorm.weight": (None, [h]),
        p + "hidden_norm.weight": (None, [h]),
        p + "post_attention_layernorm.weight": (None, [h]),
        p + "self_attn.q_proj.weight": (None, [q, 2 * h]),
        p + "self_attn.k_proj.weight": (None, [kv, 2 * h]),
        p + "self_attn.v_proj.weight": (None, [kv, 2 * h]),
        p + "self_attn.o_proj.weight": (None, [h, q]),
        p + "mlp.gate_proj.weight": (None, [inter, h]),
        p + "mlp.up_proj.weight": (None, [inter, h]),
        p + "mlp.down_proj.weight": (None, [h, inter]),
    }
    if config.draft_vocab_size != config.vocab_size:
        schema["d2t"] = ("I64", [config.draft_vocab_size])
        schema["t2d"] = ("BOOL", [config.vocab_size])
    if config.has_embed_tokens:
        schema["embed_tokens.weight"] = (None, [config.vocab_size, h])
    return schema


def draft_to_target_ids(d2t, t2d, *, vocab_size: int) -> np.ndarray:
    """Validate the reduced-vocabulary maps and return target id per draft id.

    ``target = draft + d2t[draft]`` (speculators/vLLM); ``t2d`` must mark
    exactly that image.  A map that disagrees with itself, leaves the target
    vocabulary, or sends two draft ids to one target id fails closed.
    """
    d2t = np.asarray(d2t).astype(np.int64).reshape(-1)
    targets = np.arange(d2t.shape[0], dtype=np.int64) + d2t
    if targets.min(initial=0) < 0 or targets.max(initial=0) >= vocab_size:
        raise ValueError("EAGLE-3 d2t maps outside the target vocabulary")
    if np.unique(targets).shape[0] != targets.shape[0]:
        raise ValueError("EAGLE-3 d2t maps two draft ids to one target id")
    if t2d is not None:
        mask = np.asarray(t2d).astype(bool).reshape(-1)
        if mask.shape[0] != vocab_size:
            raise ValueError("EAGLE-3 t2d must cover the target vocabulary")
        image = np.zeros(vocab_size, dtype=bool)
        image[targets] = True
        if not np.array_equal(image, mask):
            raise ValueError("EAGLE-3 t2d disagrees with d2t")
    return targets


class Eagle3Attention(nn.Module):
    def __init__(self, config: Eagle3Config):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(2 * dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(2 * dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(2 * dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        # Llama rotary embedding (rotate_half / NeoX pairing).
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def project(self, x):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        return q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)

    def attend(self, q, k, v, *, position, prior_k, prior_v):
        """One row, full causal attention over ``prior`` + the new positions."""
        length = q.shape[2]
        q = self.rope(q, offset=position)
        k = self.rope(k, offset=position)
        keys = mx.concatenate([prior_k, k.astype(prior_k.dtype)], axis=2) if prior_k.shape[2] else k
        values = mx.concatenate([prior_v, v.astype(prior_v.dtype)], axis=2) if prior_v.shape[2] else v
        start = position - prior_k.shape[2]
        key_pos = mx.arange(start, position + length)
        query_pos = mx.arange(position, position + length)
        mask = key_pos[None, :] <= query_pos[:, None]
        out = mx.fast.scaled_dot_product_attention(
            q, keys.astype(q.dtype), values.astype(q.dtype), scale=self.scale, mask=mask
        )
        return out, k


class Eagle3MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Eagle3Layer(nn.Module):
    def __init__(self, config: Eagle3Config):
        super().__init__()
        h, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(h, eps=eps)
        self.hidden_norm = nn.RMSNorm(h, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(h, eps=eps)
        self.self_attn = Eagle3Attention(config)
        self.mlp = Eagle3MLP(h, config.intermediate_size)


class Eagle3DraftModel(nn.Module):
    """External-draft contract: ``append_context``, ``draft_distributions``,
    ``make_cache``, ``batch_caches``; ``requires_context_tokens`` asks the
    executor for the token that follows each committed target feature."""

    requires_context_tokens = True
    supports_logits_processors = True
    receipt_kind = "external_eagle3"

    def __init__(self, config: Eagle3Config):
        super().__init__()
        self.config = config
        self.fc = nn.Linear(config.feature_width, config.hidden_size, bias=False)
        self.layers = [Eagle3Layer(config)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)
        if config.has_embed_tokens:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = None
        identity = np.arange(config.draft_vocab_size, dtype=np.int64)
        self._target_ids = identity  # replaced by load_vocab_map
        self.stats = {"committed_positions": 0, "chain_steps": 0, "draft_calls": 0}

    # -- vocabulary map ------------------------------------------------------
    def load_vocab_map(self, d2t=None, t2d=None):
        c = self.config
        if c.draft_vocab_size == c.vocab_size and d2t is None:
            self._target_ids = np.arange(c.vocab_size, dtype=np.int64)
            return self
        if d2t is None:
            raise ValueError("EAGLE-3 reduced vocabulary needs a d2t map")
        self._target_ids = draft_to_target_ids(d2t, t2d, vocab_size=c.vocab_size)
        return self

    @property
    def target_ids(self) -> np.ndarray:
        return self._target_ids

    @property
    def receipt_settings(self):
        c = self.config
        return {
            "algorithm": "eagle3_chain",
            "numerics": EAGLE3_NUMERICS_VERSION,
            "aux_layer_ids": list(c.aux_layer_ids),
            "target_layer_ids": list(c.target_layer_ids),
            "draft_vocab_size": c.draft_vocab_size,
            "norm_before_residual": c.norm_before_residual,
            "embed_tokens": "draft_checkpoint" if c.has_embed_tokens else "bound_target",
            "operator_block_size": c.block_size,
            "proposal_distribution": "exact_draft_softmax_on_reduced_vocab",
        }

    # -- binding ---------------------------------------------------------
    def validate_target_compatibility(self, target_model):
        args = target_model.args
        if args.hidden_size != self.config.target_hidden_size:
            raise ValueError("EAGLE-3 target hidden_size mismatch")
        if args.vocab_size != self.config.vocab_size:
            raise ValueError("EAGLE-3 target vocab_size mismatch")
        if args.num_hidden_layers != self.config.num_target_layers:
            raise ValueError("EAGLE-3 target layer count mismatch")

    def bind(self, target_model):
        self.validate_target_compatibility(target_model)
        if self.embed_tokens is None:
            if self.config.target_hidden_size != self.config.hidden_size:
                raise ValueError("EAGLE-3 cannot share a target embedding of another width")
            self.embed_tokens = target_model.model.embed_tokens
        return self

    def sanitize(self, weights):
        """Split vocabulary maps from parameters; return (params, d2t, t2d)."""
        params = dict(weights)
        d2t = params.pop("d2t", None)
        t2d = params.pop("t2d", None)
        if any(key.startswith("midlayer.") for key in params):
            raise ValueError("EAGLE-3 midlayer.* naming is not the speculators format")
        return params, d2t, t2d

    # -- cache contract ----------------------------------------------------
    def make_cache(self):
        c = self.config
        return [
            EagleWindowKVCache(c.max_position_embeddings, c.num_key_value_heads, c.head_dim)
        ]

    def batch_caches(self, rows):
        if not rows or any(len(row) != 1 for row in rows):
            raise ValueError("EAGLE-3 cache row topology mismatch")
        return _Rows([list(row) for row in rows])

    @staticmethod
    def _rows(cache):
        return cache.rows if isinstance(cache, _Rows) else [cache]

    # -- math --------------------------------------------------------------
    def _forward(self, tokens, features, rows, *, commit, chain=None, fused=False):
        """Run the draft layer for B rows at L positions.

        ``tokens``: [B, L] following-token ids (target vocabulary).
        ``features``: [B, L, 3*H_target] raw target taps, or, with ``fused``,
        [B, L, H] draft prenorm features from the previous chain step.
        ``commit``: write this KV to the committed caches; otherwise extend the
        per-row round-local ``chain`` KV.  Returns ``(normed, prenorm)``.
        """
        layer = self.layers[0]
        embeds = self.embed_tokens(tokens)
        if not fused:
            features = self.fc(features.astype(self.fc.weight.dtype))
        features = features.astype(embeds.dtype)
        e = layer.input_layernorm(embeds)
        if self.config.norm_before_residual:
            g = layer.hidden_norm(features)
            residual = g
        else:
            residual = features
            g = layer.hidden_norm(features)
        x = mx.concatenate([e, g], axis=-1)
        attn = layer.self_attn
        q, k, v = attn.project(x)
        batch, length, _ = x.shape
        outputs = []
        for row, caches in enumerate(rows):
            cache = caches[0]
            prior_k, prior_v = cache.keys_and_values()
            local = None if commit else chain[row]
            if local is not None and local[0] is not None:
                prior_k = mx.concatenate([prior_k, local[0]], axis=2) if prior_k.shape[2] else local[0]
                prior_v = mx.concatenate([prior_v, local[1]], axis=2) if prior_v.shape[2] else local[1]
            position = cache.offset + (0 if local is None or local[0] is None else local[0].shape[2])
            out, rotated = attn.attend(
                q[row:row + 1], k[row:row + 1], v[row:row + 1],
                position=position, prior_k=prior_k, prior_v=prior_v,
            )
            outputs.append(out)
            if commit:
                cache.append(rotated, v[row:row + 1])
            elif chain[row][0] is None:
                chain[row] = (rotated, v[row:row + 1])
            else:
                old = chain[row]
                chain[row] = (
                    mx.concatenate([old[0], rotated], axis=2),
                    mx.concatenate([old[1], v[row:row + 1]], axis=2),
                )
        o = mx.concatenate(outputs, axis=0).transpose(0, 2, 1, 3).reshape(batch, length, -1)
        hidden = residual + attn.o_proj(o)
        prenorm = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return self.norm(prenorm), prenorm

    def draft_logits(self, normed):
        """Reduced-vocabulary logits ``[..., draft_vocab_size]``."""
        logits = self.lm_head(normed)
        if self.config.logit_scale != 1.0:
            logits = logits * self.config.logit_scale
        return logits

    def target_logits(self, normed):
        """Draft logits scattered into the target vocabulary (-inf elsewhere)."""
        small = np.asarray(self.draft_logits(normed).astype(mx.float32))
        full = np.full(small.shape[:-1] + (self.config.vocab_size,), -np.inf, dtype=np.float32)
        full[..., self._target_ids] = small
        return full

    def append_context(self, hidden, cache, *, context_tokens=None):
        """Commit target features (paired with their next tokens) to draft KV."""
        if hidden.shape[1] == 0:
            return None
        if context_tokens is None:
            raise ValueError("EAGLE-3 needs the token following each feature")
        rows = self._rows(cache)
        tokens = mx.array(np.asarray(context_tokens, dtype=np.int32).reshape(len(rows), -1))
        if tokens.shape[1] != hidden.shape[1]:
            raise ValueError("EAGLE-3 context tokens must pair with features")
        out = self._forward(tokens, hidden, rows, commit=True)
        self.stats["committed_positions"] += int(hidden.shape[1]) * len(rows)
        return out

    def draft_distributions(
        self, anchors, hidden, cache, proposal_length, rngs, temperatures, *,
        context_tokens=None, logits_processors=None, processor_histories=None,
    ):
        """Chain-draft ``proposal_length`` tokens; return tokens and exact q laws.

        Each law is the draft softmax over the reduced vocabulary, placed on
        the mapped target ids (zero mass on target ids outside the draft
        vocabulary), so rejection verification is exact for sampled targets.
        """
        from ..processor_probe import probe_logits_processors
        from ..speculative_sampling import softmax

        rows = self._rows(cache)
        batch = len(rows)
        anchor_values = [int(a) for a in anchors]
        if context_tokens is None:
            context_tokens = [[a] for a in anchor_values] if hidden.shape[1] else None
        if hidden.shape[1] == 0:
            # No target feature yet: zero proposals make this an ordinary
            # verify round, never a guess.
            return [[] for _ in range(batch)], [[] for _ in range(batch)]
        logits_processors = logits_processors or [[] for _ in range(batch)]
        processor_histories = processor_histories or [[] for _ in range(batch)]
        normed, prenorm = self.append_context(hidden, cache, context_tokens=context_tokens)
        normed, prenorm = normed[:, -1:], prenorm[:, -1:]
        self.stats["draft_calls"] += 1
        chain = [(None, None) for _ in range(batch)]
        tokens = [[] for _ in range(batch)]
        laws = [[] for _ in range(batch)]
        for step in range(int(proposal_length)):
            if step:
                previous = mx.array([[row[-1]] for row in tokens], dtype=mx.int32)
                normed, prenorm = self._forward(
                    previous, prenorm, rows, commit=False, chain=chain, fused=True
                )
                self.stats["chain_steps"] += 1
            dense = self.target_logits(normed[:, -1, :])
            for row in range(batch):
                value = dense[row]
                if logits_processors[row]:
                    prefix = mx.array(
                        list(processor_histories[row]) + [anchor_values[row]] + tokens[row],
                        dtype=mx.int32,
                    )
                    processed = np.asarray(
                        probe_logits_processors(
                            logits_processors[row], prefix, mx.array(value)[None]
                        )[0].astype(mx.float32)
                    )
                    if np.isfinite(processed).any():
                        value = processed  # else grammar dead end: keep a valid law
                q = softmax(value, temperatures[row])
                token = int(rngs[row].sample(q))
                if q[token] <= 0:
                    # Float round-off at the top of the cumulative sum can
                    # land past the last supported id; stay on the support.
                    token = int(np.flatnonzero(q)[-1])
                tokens[row].append(token)
                laws[row].append(q)
        return tokens, laws


Model = Eagle3DraftModel

__all__ = [
    "ARCHITECTURE",
    "EAGLE3_NUMERICS_VERSION",
    "Eagle3Config",
    "Eagle3DraftModel",
    "Model",
    "default_aux_layer_ids",
    "draft_to_target_ids",
    "expected_weight_schema",
]
