# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see provenance/mellum21.json and .NOTICE.
"""Mellum 2/2.1 sparse-MoE tensor model.

The ordinary model is the reference path.  It keeps Mellum's mixed
sliding/full-attention layout and packs the 64 per-layer experts into
``SwitchGLU`` tables once at load time; scheduling and APCv2 remain shared
mlx2 concerns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
from mlx import nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "mellum"
    hidden_size: int = 2304
    num_hidden_layers: int = 28
    intermediate_size: int = 7168
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    num_experts: int = 64
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 896
    rms_norm_eps: float = 1e-6
    vocab_size: int = 98304
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 131072
    norm_topk_prob: bool = True
    sliding_window: int = 1024
    layer_types: list[str] = field(default_factory=list)
    rope_parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = [
                "full_attention" if index % 4 == 3 else "sliding_attention"
                for index in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers or any(
            kind not in {"full_attention", "sliding_attention"}
            for kind in self.layer_types
        ):
            raise ValueError("Mellum layer_types must describe every attention layer")
        if set(self.rope_parameters) != {"full_attention", "sliding_attention"}:
            raise ValueError("Mellum requires full and sliding RoPE parameters")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Mellum query heads must be divisible by KV heads")
        if not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError("Mellum expert routing geometry is invalid")


def _rope_for(layer_type: str, args: ModelArgs):
    params = args.rope_parameters[layer_type]
    base = params["rope_theta"]
    rope_type = params.get("rope_type", "default")
    if rope_type in {"default", "linear"}:
        return initialize_rope(args.head_dim, base=base, traditional=False)
    scaling = dict(params)
    scaling["type"] = rope_type
    return initialize_rope(
        args.head_dim,
        base=base,
        traditional=False,
        scaling_config=scaling,
        max_position_embeddings=args.max_position_embeddings,
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_index: int):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim**-0.5
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = _rope_for(args.layer_types[layer_index], args)

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        y = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        return self.o_proj(y.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class SparseMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )

    def __call__(self, x):
        scores = mx.softmax(self.gate(x), axis=-1, precise=True)
        indices = mx.argpartition(scores, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        scores = mx.take_along_axis(scores, indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        return mx.sum(self.switch_mlp(x, indices) * scores[..., None], axis=-2)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_index: int):
        super().__init__()
        self.self_attn = Attention(args, layer_index)
        self.mlp = SparseMoE(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None):
        hidden = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Decoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, index) for index in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._first_full = args.layer_types.index("full_attention")
        self._first_sliding = args.layer_types.index("sliding_attention")

    def __call__(self, inputs, cache=None):
        hidden = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        full_mask = create_attention_mask(hidden, cache[self._first_full])
        sliding_mask = create_attention_mask(
            hidden,
            cache[self._first_sliding],
            window_size=self.args.sliding_window,
        )
        for layer, layer_cache, layer_type in zip(
            self.layers, cache, self.args.layer_types, strict=True
        ):
            mask = full_mask if layer_type == "full_attention" else sliding_mask
            hidden = layer(hidden, mask, layer_cache)
        return self.norm(hidden)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Decoder(args)
        self.completed_forwards = 0
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        hidden = self.model(inputs, cache)
        logits = (
            self.model.embed_tokens.as_linear(hidden)
            if self.args.tie_word_embeddings
            else self.lm_head(hidden)
        )
        self.completed_forwards += 1
        return logits

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights, *, eager=False):
        """Pack official per-expert tensors into the SwitchGLU layout.

        ``eager=True`` evaluates and releases one layer at a time.  The BF16
        checkpoint is 24.3 GB, so this avoids retaining a second checkpoint's
        worth of lazy stack expressions during load.
        """
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        for layer in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer}.mlp"
            packed = []
            for projection in ("up_proj", "down_proj", "gate_proj"):
                first = f"{prefix}.experts.0.{projection}.weight"
                if first not in weights:
                    continue
                names = [
                    f"{prefix}.experts.{expert}.{projection}.weight"
                    for expert in range(self.args.num_experts)
                ]
                if any(name not in weights for name in names):
                    raise ValueError(
                        f"Mellum layer {layer} has an incomplete {projection} expert table"
                    )
                value = mx.stack([weights.pop(name) for name in names])
                weights[f"{prefix}.switch_mlp.{projection}.weight"] = value
                packed.append(value)
            if eager and packed:
                mx.eval(packed)
                mx.clear_cache()
        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def make_cache(self):
        return [
            KVCache()
            if layer_type == "full_attention"
            else RotatingKVCache(max_size=self.args.sliding_window)
            for layer_type in self.args.layer_types
        ]
