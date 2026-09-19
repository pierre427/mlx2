# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see provenance/north-mini-code.json and .NOTICE.
"""Cohere2 MoE tensor model for North Mini Code.

This module contains model-specific tensor math only. Serving, APCv2,
admission, scheduling, and route selection stay in their shared layers.
"""

from dataclasses import dataclass

import mlx.core as mx
from mlx import nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "cohere2_moe"
    hidden_size: int = 2048
    head_dim: int = 128
    num_hidden_layers: int = 49
    intermediate_size: int = 768
    prefix_dense_intermediate_size: int = 3072
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    vocab_size: int = 262144
    rope_theta: float = 50000.0
    layer_norm_eps: float = 1e-5
    logit_scale: float = 1.0
    attention_bias: bool = False
    sliding_window: int = 4096
    max_position_embeddings: int = 500000
    tie_word_embeddings: bool = True
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int = 0
    norm_topk_prob: bool = False
    first_k_dense_replace: int = 1
    expert_selection_fn: str = "sigmoid"
    layer_types: list[str] | None = None
    use_parallel_block: bool = True
    use_qk_norm: bool = False

    def __post_init__(self):
        # The converted North artifacts encode this field as JSON null while
        # omitting lm_head.weight. Upstream North uses the embedding as its
        # output projection, so null resolves to the tied architecture.
        if self.tie_word_embeddings is None:
            self.tie_word_embeddings = True
        elif self.tie_word_embeddings is not True:
            raise ValueError("North requires tied word embeddings")
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if i % 4 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must match num_hidden_layers")
        if any(t not in {"full_attention", "sliding_attention"} for t in self.layer_types):
            raise ValueError("unsupported North attention type")
        if not self.use_parallel_block or self.use_qk_norm:
            raise ValueError("North requires a parallel block without QK norm")
        if self.expert_selection_fn != "sigmoid":
            raise ValueError("North requires sigmoid expert selection")
        if self.num_shared_experts != 0 or self.norm_topk_prob:
            raise ValueError("North checkpoint has no shared or normalized router")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.first_k_dense_replace < 0 or self.first_k_dense_replace > self.num_hidden_layers:
            raise ValueError("invalid dense prefix")
        if self.sliding_window <= 0:
            raise ValueError("sliding_window must be positive")


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Cohere2MoeSparseBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.intermediate_size, args.num_experts
        )

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        scores = mx.sigmoid(self.gate(x).astype(mx.float32))
        inds = mx.stop_gradient(
            mx.argpartition(-scores, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        )
        weights = mx.take_along_axis(scores, inds, axis=-1).astype(dtype)
        return mx.sum(self.switch_mlp(x, inds) * weights[..., None], axis=-2)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"
        dim = args.hidden_size
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.rope = (
            nn.RoPE(self.head_dim, traditional=True, base=args.rope_theta)
            if self.is_sliding
            else None
        )

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        q, k, v = (
            q.transpose(0, 2, 1, 3),
            k.transpose(0, 2, 1, 3),
            v.transpose(0, 2, 1, 3),
        )
        if self.rope is not None:
            offset = cache.offset if cache is not None else 0
            q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = (
            MLP(args.hidden_size, args.prefix_dense_intermediate_size)
            if layer_idx < args.first_k_dense_replace
            else Cohere2MoeSparseBlock(args)
        )
        self.input_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps, bias=False
        )
        self.attention_type = args.layer_types[layer_idx]

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        h = self.input_layernorm(x)
        return x + self.self_attn(h, mask, cache) + self.mlp(h)


class ResidualTaps:
    """Optional residual-stream taps for one decoder layer band.

    ``steer`` is ``(layer_index, vector)``: the vector (broadcastable to the
    residual, e.g. ``[B, 1, D]`` with zero rows for unsteered lanes) is added
    to that layer's output.  ``capture`` maps layer indices to the residual
    they produced on the last forward (calibration only).  Both are None in
    ordinary serving, and the forward is then byte-for-byte the plain one.
    A plain object, so neither is ever mistaken for a model parameter.
    """

    __slots__ = ("steer", "capture")

    def __init__(self):
        self.steer = None
        self.capture = None


class Cohere2MoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps, bias=False)
        self.full_index = args.layer_types.index("full_attention")
        self.sliding_index = args.layer_types.index("sliding_attention")
        self._taps = ResidualTaps()

    @property
    def residual_taps(self):
        return self._taps

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        if len(cache) != len(self.layers):
            raise ValueError("North cache layer count mismatch")
        masks = {
            "full_attention": create_attention_mask(h, cache[self.full_index]),
            "sliding_attention": create_attention_mask(
                h, cache[self.sliding_index], window_size=self.args.sliding_window
            ),
        }
        steer, capture = self._taps.steer, self._taps.capture
        if steer is None and capture is None:
            for layer, layer_cache in zip(self.layers, cache):
                h = layer(h, masks[layer.attention_type], layer_cache)
            return self.norm(h)
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            h = layer(h, masks[layer.attention_type], layer_cache)
            if steer is not None and index == steer[0]:
                h = h + steer[1].astype(h.dtype)
            if capture is not None and index in capture:
                capture[index] = h
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Cohere2MoeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def apc_v2_layout(self):
        return "north-mini-code-layer-segments-v1"

    def __call__(self, inputs: mx.array, cache=None):
        hidden = self.model(inputs, cache)
        logits = (
            self.model.embed_tokens.as_linear(hidden)
            if self.args.tie_word_embeddings
            else self.lm_head(hidden)
        )
        return logits * self.args.logit_scale

    def make_cache(self):
        return [
            KVCache()
            if kind == "full_attention"
            else RotatingKVCache(max_size=self.args.sliding_window, keep=0)
            for kind in self.args.layer_types
        ]

    def sanitize(self, weights):
        if any(k.startswith("language_model.") for k in weights):
            weights = {
                k.removeprefix("language_model."): value
                for k, value in weights.items()
            }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return {
            k: value
            for k, value in weights.items()
            if "rotary_emb.inv_freq" not in k
        }

    @property
    def quant_predicate(self):
        def predicate(path, _):
            return {"group_size": 64, "bits": 8} if path.endswith("mlp.gate") else True

        return predicate

    @property
    def layers(self):
        return self.model.layers
