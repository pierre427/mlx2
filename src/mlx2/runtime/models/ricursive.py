# SPDX-License-Identifier: Apache-2.0
# Adapted from the pinned Ricursive model; see provenance/ripple-65b.json.
"""Ripple tensor math; ordinary BF16 activations over source NVFP4 MLPs.

The source's W4A4 activation quantizer is deliberately not inferred from its
weight storage. The serving adapter requires explicit W4A16 selection.
"""

from dataclasses import dataclass

import mlx.core as mx
from mlx import nn

from ..prefill_checkpoint_rows import select_checkpoint_rows
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .modelopt_nvfp4 import ModelOptNVFP4Linear


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "ricursive"
    hidden_size: int = 5120
    intermediate_size: int = 40960
    num_hidden_layers: int = 80
    num_attention_heads: int = 80
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 100352
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    sliding_window: int = 1024
    layer_types: list[str] | None = None
    embed_scale: float | None = 71.55417527999327
    ffn_clamp_threshold: float | None = 20.0
    use_gated_attention: bool = True
    gate_full_precision: bool = True
    use_valres: bool = True
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if i % 5 == 4 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers or any(
            t not in {"full_attention", "sliding_attention"} for t in self.layer_types
        ):
            raise ValueError("invalid Ripple layer types")
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError("invalid Ripple attention geometry")
        if self.sliding_window < 1 or self.tie_word_embeddings:
            raise ValueError("invalid Ripple window or tied embeddings")


class RMSNorm(nn.Module):
    """Source rounding: normalize FP32, cast, then multiply the norm weight."""

    def __init__(self, dims, eps):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        y = x.astype(mx.float32)
        y = y * mx.rsqrt(mx.mean(y * y, axis=-1, keepdims=True) + self.eps)
        return y.astype(x.dtype) * self.weight


def rotary(x, offset, theta):
    # Source materializes cos/sin in the activation dtype before the FP32
    # rotate. mx.fast.rope would skip that BF16 trigonometric rounding.
    dims = x.shape[-1]
    inv = 1.0 / (theta ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
    if isinstance(offset, mx.array) and offset.ndim:
        positions = mx.arange(x.shape[-2], dtype=mx.float32)[None, :] + offset[:, None]
        phase = positions[:, None, :, None] * inv
    else:
        positions = mx.arange(x.shape[-2], dtype=mx.float32) + offset
        phase = positions[:, None] * inv
    phase = mx.concatenate((phase, phase), axis=-1)
    cos = mx.cos(phase).astype(x.dtype).astype(mx.float32)
    sin = mx.sin(phase).astype(x.dtype).astype(mx.float32)
    y = x.astype(mx.float32)
    half = mx.concatenate((-y[..., dims // 2:], y[..., :dims // 2]), axis=-1)
    return (y * cos + half * sin).astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, args, index):
        super().__init__()
        self.args = args
        self.local = args.layer_types[index] == "sliding_attention"
        qdim = args.num_attention_heads * args.head_dim
        kvdim = args.num_key_value_heads * args.head_dim
        self.q_proj = nn.Linear(args.hidden_size, qdim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, kvdim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, kvdim, bias=False)
        self.o_proj = nn.Linear(qdim, args.hidden_size, bias=False)
        self.q_norm = RMSNorm(args.head_dim, args.rms_norm_eps)
        self.k_norm = RMSNorm(args.head_dim, args.rms_norm_eps)
        if args.use_gated_attention:
            self.g_proj = nn.Linear(args.hidden_size, qdim, bias=False)
        if args.use_valres:
            self.value_res_lambda = mx.zeros((1,))

    def __call__(self, x, seed, cache=None):
        args = self.args
        b, length, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(b, length, args.num_attention_heads, args.head_dim))
        k = self.k_norm(self.k_proj(x).reshape(b, length, args.num_key_value_heads, args.head_dim))
        value_input = x
        if args.use_valres:
            blend = mx.sigmoid(self.value_res_lambda[0]).astype(x.dtype)
            # torch.lerp computes pointwise in opmath FP32 for BF16 inputs.
            value_input = (seed.astype(mx.float32) + blend.astype(mx.float32)
                           * (x.astype(mx.float32) - seed.astype(mx.float32))).astype(x.dtype)
        v = self.v_proj(value_input).reshape(b, length, args.num_key_value_heads, args.head_dim)
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        mask = create_attention_mask(x, cache, window_size=args.sliding_window if self.local else None)
        if self.local:
            offset = cache.offset if cache is not None else 0
            q, k = rotary(q, offset, args.rope_theta), rotary(k, offset, args.rope_theta)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=args.head_dim ** -0.5, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
        if args.use_gated_attention:
            gate = self.g_proj(x)
            if args.gate_full_precision:
                gate = gate.astype(mx.float32)
            out = out * mx.sigmoid(gate).astype(out.dtype)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, args, *, nvfp4):
        super().__init__()
        linear = ModelOptNVFP4Linear if nvfp4 else nn.Linear
        self.gate_proj = linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = linear(args.intermediate_size, args.hidden_size, bias=False)
        self.clamp = args.ffn_clamp_threshold

    def __call__(self, x):
        gate, up = self.gate_proj(x), self.up_proj(x)
        if self.clamp:
            gate = mx.minimum(gate, self.clamp)
            up = mx.clip(up, -self.clamp, self.clamp)
        # Match source's separately rounded SiLU and multiplication.
        silu = (gate.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(gate.dtype)
        return self.down_proj(silu * up)


class DecoderLayer(nn.Module):
    def __init__(self, args, index, *, nvfp4):
        super().__init__()
        self.self_attn = Attention(args, index)
        self.mlp = MLP(args, nvfp4=nvfp4)
        self.input_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(self, x, seed, cache):
        x = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), seed, cache))
        return x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))


class Backbone(nn.Module):
    def __init__(self, args, *, nvfp4):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i, nvfp4=nvfp4) for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(self, tokens, cache=None, input_embeddings=None):
        h = self.embed_tokens(tokens) if input_embeddings is None else input_embeddings
        if self.args.embed_scale is not None:
            h = h * self.args.embed_scale
        seed = h
        if cache is None:
            cache = [None] * len(self.layers)
        if len(cache) != len(self.layers):
            raise ValueError("Ripple cache must contain exactly one entry per layer")
        for layer, entry in zip(self.layers, cache, strict=True):
            h = layer(h, seed, entry)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args, *, nvfp4=True):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.forward_calls = 0
        self.model = Backbone(args, nvfp4=nvfp4)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers

    def __call__(self, tokens, cache=None, input_embeddings=None):
        self.forward_calls += 1
        h = self.model(tokens, cache, input_embeddings)
        return self.lm_head(select_checkpoint_rows(h))

    def make_cache(self):
        return [RotatingKVCache(self.args.sliding_window, keep=0)
                if kind == "sliding_attention" else KVCache()
                for kind in self.args.layer_types]
