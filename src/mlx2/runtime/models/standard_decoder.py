# SPDX-License-Identifier: MIT
"""Ordinary full-attention decoder math for Qwen3, Qwen3 MoE, Qwen2 and Llama.

Mined from the pinned unified implementations; see provenance/standard-decoder.json.
No speculative or model-name-dependent scheduler behavior lives here.
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    tie_word_embeddings: bool
    head_dim: Optional[int] = None
    rope_scaling: Optional[dict] = None
    rope_traditional: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    mlp_only_layers: Optional[list[int]] = None
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = False

    def __post_init__(self):
        if self.model_type not in {"qwen3", "qwen3_moe", "qwen2", "llama"}:
            raise ValueError("unsupported standard decoder family")
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.mlp_only_layers is None:
            self.mlp_only_layers = []


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim, heads, kv_heads, head_dim = (
            args.hidden_size, args.num_attention_heads,
            args.num_key_value_heads, args.head_dim,
        )
        self.n_heads, self.n_kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.scale = head_dim**-0.5
        qkv_bias = args.model_type == "qwen2" or args.attention_bias
        self.q_proj = nn.Linear(dim, heads * head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, kv_heads * head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, kv_heads * head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(heads * head_dim, dim, bias=args.attention_bias and args.model_type == "llama")
        if args.model_type in {"qwen3", "qwen3_moe"}:
            self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            head_dim, base=args.rope_theta,
            traditional=args.rope_traditional, scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        if hasattr(self, "q_norm"):
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = (z.transpose(0, 2, 1, 3) for z in (q, k, v))
        offset = cache.offset if cache is not None else 0
        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        y = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        return self.o_proj(y.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim, hidden = args.hidden_size, args.intermediate_size
        bias = args.model_type == "llama" and args.mlp_bias
        self.gate_proj = nn.Linear(dim, hidden, bias=bias)
        self.up_proj = nn.Linear(dim, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, dim, bias=bias)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoE(nn.Module):
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
        indices = mx.argpartition(scores, kth=-self.top_k, axis=-1)[..., -self.top_k:]
        scores = mx.take_along_axis(scores, indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        return mx.sum(self.switch_mlp(x, indices) * scores[..., None], axis=-2)


class Layer(nn.Module):
    def __init__(self, args: ModelArgs, index: int):
        super().__init__()
        self.self_attn = Attention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        sparse = (
            args.model_type == "qwen3_moe"
            and index not in args.mlp_only_layers
            and (index + 1) % args.decoder_sparse_step == 0
        )
        self.mlp = MoE(args) if sparse else MLP(args)

    def __call__(self, x, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class Decoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Layer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None, *, capture_layers=(), hidden_sink=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask, layer_cache)
            if hidden_sink is not None and index in capture_layers:
                hidden_sink.append(h)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Decoder(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        out = self.model(inputs, cache)
        return self.logits(out)

    def logits(self, hidden):
        return (self.model.embed_tokens.as_linear(hidden)
                if self.args.tie_word_embeddings else self.lm_head(hidden))

    def make_cache(self):
        from .cache import KVCache

        return [KVCache() for _ in self.layers]

    def forward_with_taps(self, inputs, cache, capture_layers, *, body_only=False):
        """Post-layer features before final norm, paired with consumed cache tokens."""
        capture_layers = tuple(capture_layers)
        if (not capture_layers
                or any(type(index) is not int for index in capture_layers)
                or tuple(sorted(set(capture_layers))) != capture_layers
                or capture_layers[0] < 0 or capture_layers[-1] >= len(self.layers)):
            raise ValueError("Invalid target capture layers")
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("Target cache must have one entry per layer")
        taps = []
        hidden = self.model(inputs, cache=cache, capture_layers=capture_layers,
                            hidden_sink=taps)
        features = mx.concatenate(taps, axis=-1)
        return (None if body_only else self.logits(hidden)), features

    def prefill_body(self, inputs, cache, capture_layers):
        return self.forward_with_taps(inputs, cache, capture_layers, body_only=True)[1]

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        # Qwen2 and Llama checkpoints may retain a precomputed rotary table;
        # the rotary module builds its own frequencies from the configuration.
        return {
            key: value for key, value in weights.items()
            if "self_attn.rotary_emb.inv_freq" not in key
        }

    @property
    def layers(self):
        return self.model.layers
