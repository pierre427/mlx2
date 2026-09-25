# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see provenance/muse-glimmer.json and .NOTICE.
"""Muse Glimmer text tower using mlx2 attention and cache contracts."""

import mlx.core as mx
import mlx.nn as nn
from ...adapters.muse_glimmer_config import ModelArgs
from .base import create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache


class CenteredRMSNorm(nn.Module):
    """RMSNorm with a zero-centered scale: out = norm(x) * (1 + weight)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.qk_scale_factor = args.qk_scale_factor

        self.layer_type = args.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.use_rope = bool(args.layer_rope_theta[layer_idx])

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)

        self.eps = args.rms_norm_eps
        if self.use_rope:
            self.rope = nn.RoPE(
                self.head_dim, traditional=False, base=args.layer_rope_theta[layer_idx]
            )

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        # Scaleless QK-norm over head_dim; Q additionally scaled.
        q = mx.fast.rms_norm(q, None, self.eps) * self.qk_scale_factor
        k = mx.fast.rms_norm(k, None, self.eps)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if self.use_rope:
            offset = cache.offset if cache is not None else 0
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)

        # Glimmer's output gate.
        out = out * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.input_layernorm = CenteredRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = CenteredRMSNorm(
            args.hidden_size, args.post_norm_eps
        )
        self.pre_feedforward_layernorm = CenteredRMSNorm(
            args.hidden_size, args.rms_norm_eps
        )
        self.post_feedforward_layernorm = CenteredRMSNorm(
            args.hidden_size, args.post_norm_eps
        )

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask, cache)
        h = self.post_attention_layernorm(h)
        h = residual + h

        residual = h
        h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        return residual + h


class MuseGlimmerModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.embed_eps = args.rms_norm_eps
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.window = args.sliding_window

    def _masks(self, h, cache):
        made, out = {}, []
        for layer, c in zip(self.layers, cache):
            t = layer.self_attn.layer_type
            if t not in made:
                if t == "sliding_attention":
                    made[t] = create_attention_mask(h, c, window_size=self.window)
                else:
                    made[t] = create_attention_mask(h, c)
            out.append(made[t])
        return out

    def __call__(self, inputs: mx.array, cache=None, capture_layers=(), hidden_sink=None):
        h = self.embed_tokens(inputs)
        # Scaleless RMSNorm on the embeddings (not Gemma's sqrt scaling).
        h = mx.fast.rms_norm(h, None, self.embed_eps)

        if cache is None:
            cache = [None] * len(self.layers)

        if len(cache) != len(self.layers):
            raise ValueError("Muse cache layer count mismatch")
        masks = self._masks(h, cache)
        for index, (layer, c, mask) in enumerate(zip(self.layers, cache, masks)):
            h = layer(h, mask, c)
            if hidden_sink is not None and index in capture_layers:
                hidden_sink.append(h)
        return self.norm(h)


class Model(nn.Module):
    @property
    def apc_v2_layout(self):
        return self.args.cache_layout

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MuseGlimmerModel(args)
        self.tie_word_embeddings = args.tie_word_embeddings
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out = self.model(inputs, cache=cache)
        if self.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        out = out * self.args.output_multiplier
        cap = self.args.final_logit_softcapping
        return mx.tanh(out / cap) * cap

    def forward_with_taps(self, inputs, cache, capture_layers, *, body_only=False):
        """Post-block target taps in ascending layer order, before final norm.

        Body-only prefill avoids the expensive vocabulary projection. Tap
        tensors remain paired with exactly the cache tokens consumed here.
        """
        capture_layers = tuple(capture_layers)
        if not capture_layers or tuple(sorted(set(capture_layers))) != capture_layers or capture_layers[-1] >= len(self.layers) or capture_layers[0] < 0:
            raise ValueError("Invalid target capture layers")
        taps = []
        hidden = self.model(inputs, cache=cache, capture_layers=capture_layers, hidden_sink=taps)
        features = mx.concatenate(taps, axis=-1)
        if body_only:
            return None, features
        logits = self.model.embed_tokens.as_linear(hidden) if self.tie_word_embeddings else self.lm_head(hidden)
        logits = logits * self.args.output_multiplier
        cap = self.args.final_logit_softcapping
        return mx.tanh(logits / cap) * cap, features

    def prefill_body(self, inputs, cache, capture_layers):
        return self.forward_with_taps(inputs, cache, capture_layers, body_only=True)[1]

    def sanitize(self, weights):
        out = {}
        vision = ("vision_tower", "vision_adapter", "vision_projection")
        for k, v in weights.items():
            # Drop the vision tower — this is a text-only port. The original
            # Hugging Face checkpoint nests it under ``model.`` (as it does the
            # text tower); the MLX conversion keeps it at the top level.
            if k.startswith(vision) or k.startswith(tuple("model." + p for p in vision)):
                continue
            # Meta/MLX nest the text tower under language_model.*
            if k.startswith("language_model.model."):
                k = "model." + k[len("language_model.model.") :]
            elif k.startswith("language_model.lm_head."):
                k = "lm_head." + k[len("language_model.lm_head.") :]
            elif k.startswith("model.language_model."):
                k = "model." + k[len("model.language_model.") :]
            out[k] = v
        return out

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    def make_cache(self):
        caches = []
        for lt in self.args.layer_types:
            if lt == "sliding_attention":
                caches.append(
                    RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                )
            else:
                caches.append(KVCache())
        return caches
