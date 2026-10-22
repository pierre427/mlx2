"""LFM2.5 24B-A2B hybrid short-convolution/MoE decoder.

This is the ordinary-decode reference implementation for the exact LiquidAI
24B-A2B topology.  It is adapted from the corrected sigmoid-routing model in
the pinned local mlx-lm-unified source; see
``provenance/lfm25-24b-a2b.json``.  Model-specific tensor math stays here,
outside scheduling, request lifecycle, and APCv2.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from mlx import nn

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from . import moe_nax_gather as _moe_nax
from .cache import ArraysCache, KVCache
from .switch_layers import SwitchGLU

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    use_expert_bias: bool
    num_dense_layers: int
    norm_eps: float
    conv_bias: bool
    conv_L_cache: int
    tie_word_embeddings: bool = True
    rope_theta: float = 1_000_000.0
    routed_scaling_factor: float = 1.0
    rope_parameters: dict | None = None
    full_attn_idxs: list[int] | None = None
    layer_types: list[str] | None = None

    def __post_init__(self):
        if self.rope_parameters is not None:
            self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)
        if self.layer_types is None:
            raise ValueError("LFM2.5 requires an explicit hybrid layer order")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("LFM2.5 layer_types length does not match the model")
        unknown = set(self.layer_types) - {"conv", "full_attention"}
        if unknown:
            raise ValueError(f"unsupported LFM2.5 layer types: {sorted(unknown)}")
        derived = [
            index
            for index, layer_type in enumerate(self.layer_types)
            if layer_type == "full_attention"
        ]
        if self.full_attn_idxs is not None and self.full_attn_idxs != derived:
            raise ValueError("LFM2.5 attention indices disagree with layer_types")
        self.full_attn_idxs = derived

    @property
    def head_dim(self):
        # Attention geometry, also read by the external-draft KV admission estimate.
        return self.hidden_size // self.num_attention_heads


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.rope = nn.RoPE(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
        )

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = self.q_layernorm(
            queries.reshape(batch, length, self.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        keys = self.k_layernorm(
            keys.reshape(batch, length, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(batch, length, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )
        if cache is None:
            queries = self.rope(queries)
            keys = self.rope(keys)
        else:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            mask=mask,
            scale=self.scale,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.out_proj(output)


class ShortConv(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.history = args.conv_L_cache - 1
        self.conv = nn.Conv1d(
            in_channels=args.hidden_size,
            out_channels=args.hidden_size,
            kernel_size=args.conv_L_cache,
            groups=args.hidden_size,
            bias=args.conv_bias,
        )
        self.in_proj = nn.Linear(
            args.hidden_size, 3 * args.hidden_size, bias=args.conv_bias
        )
        self.out_proj = nn.Linear(
            args.hidden_size, args.hidden_size, bias=args.conv_bias
        )

    def __call__(self, x, mask=None, cache=None):
        return self.out_proj(self._convolve_projected(self.in_proj(x), mask, cache))

    def _convolve_projected(self, projected, mask, cache):
        """``C * conv(B * x)`` from the ``in_proj`` output; owns the cache write.

        This is the reference path and the seam the fused one-token decode
        kernel replaces (``adapters/lfm25_fused_shortconv.py``).
        """
        gate_b, gate_c, value = mx.split(projected, 3, axis=-1)
        gated_value = gate_b * value
        if mask is not None:
            gated_value = mx.where(mask[..., None], gated_value, 0)
        if cache is None:
            gated_value = mx.pad(gated_value, [(0, 0), (self.history, 0), (0, 0)])
        else:
            previous = cache[0]
            state = previous
            if state is None:
                state = mx.zeros(
                    (gated_value.shape[0], self.history, self.hidden_size),
                    dtype=gated_value.dtype,
                )
            gated_value = mx.concatenate([state, gated_value], axis=1)
            token_count = projected.shape[1]
            if cache.lengths is None:
                cache[0] = gated_value[:, -self.history :, :]
            else:
                ends = mx.clip(cache.lengths, 0, token_count)
                cache[0] = _history_rows(gated_value, ends, self.history)
            spans = (
                cache.rollback_spans(token_count, mask)
                if getattr(cache, "speculating", False)
                else None
            )
            if spans is not None:
                # Exact closed form: the state after the first ``m`` tokens is
                # the two rows of ``[state, B*x]`` starting at ``m``.  A
                # rejected proposal restores it without any replay.
                history = self.history

                def rollback(m, padded=gated_value):
                    return [mx.contiguous(padded[:, m : m + history, :])]

                def rollback_rows(lengths, padded=gated_value):
                    ends = mx.array(list(lengths), dtype=mx.int32)
                    return [mx.contiguous(_history_rows(padded, ends, history))]

                cache.record_rollback(
                    token_count,
                    rollback,
                    [previous],
                    per_row_fn=rollback_rows,
                    depths=spans or None,
                )
            cache.advance(token_count)
        return gate_c * self.conv(gated_value)


def _history_rows(padded, ends, history):
    """Row ``b``'s ``padded[b, ends[b] : ends[b] + history]``."""
    positions = (ends[:, None] + mx.arange(history))[..., None]
    return mx.take_along_axis(padded, positions, axis=1)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class SparseMoE(nn.Module):
    """Sigmoid-routed top-k experts with selection-only balancing bias."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.use_expert_bias = args.use_expert_bias
        self.routed_scaling_factor = args.routed_scaling_factor
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        if self.use_expert_bias:
            self.expert_bias = mx.zeros((args.num_experts,))

    def routing(self, x):
        routing_weights = mx.sigmoid(self.gate(x))
        selection_scores = routing_weights.astype(mx.float32)
        if self.use_expert_bias:
            selection_scores = selection_scores + self.expert_bias
        indices = mx.argpartition(selection_scores, kth=-self.top_k, axis=-1)[
            ..., -self.top_k :
        ]
        scores = mx.take_along_axis(routing_weights, indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-6)
        scores = scores * self.routed_scaling_factor
        return indices, scores.astype(x.dtype)

    def __call__(self, x):
        indices, scores = self.routing(x)
        expert_outputs = self.switch_mlp(x, indices)
        return mx.sum(expert_outputs * scores[..., None], axis=-2)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, index: int):
        super().__init__()
        self.is_attention_layer = index in args.full_attn_idxs
        if self.is_attention_layer:
            self.self_attn = Attention(args)
        else:
            self.conv = ShortConv(args)
        self.feed_forward = (
            MLP(args) if index < args.num_dense_layers else SparseMoE(args)
        )
        self.operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(self, x, mask=None, cache=None):
        normalized = self.operator_norm(x)
        residual = (
            self.self_attn(normalized, mask=mask, cache=cache)
            if self.is_attention_layer
            else self.conv(normalized, mask=mask, cache=cache)
        )
        hidden = x + residual
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class LFM25Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, index) for index in range(args.num_hidden_layers)
        ]
        self.embedding_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.attention_index = args.full_attn_idxs[0]
        self.conv_index = next(
            index
            for index, layer in enumerate(self.layers)
            if not layer.is_attention_layer
        )

    def __call__(self, inputs, cache=None, input_embeddings=None, capture=None):
        # The NAX sorted expert gather (MLX2_MOE_NAX_GATHER=gather, default
        # off) is prefill-only; the trunk forward decides the phase once.
        # Decode, speculating caches and prepared verify blocks are not
        # prefill.  A no-op while the mode is off.
        with _moe_nax.forward_scope(
            inputs if inputs is not None else input_embeddings, cache
        ):
            return self._forward(inputs, cache, input_embeddings, capture)

    def _forward(self, inputs, cache, input_embeddings, capture=None):
        hidden = (
            self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        )
        if cache is None:
            cache = [None] * len(self.layers)
        attention_mask = create_attention_mask(hidden, cache[self.attention_index])
        convolution_mask = create_ssm_mask(hidden, cache[self.conv_index])
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            mask = attention_mask if layer.is_attention_layer else convolution_mask
            hidden = layer(hidden, mask=mask, cache=layer_cache)
            # Post-block taps for external drafters (layer k = output of layer k).
            if capture is not None and index in capture:
                capture[index] = hidden
        return self.embedding_norm(hidden)


class Model(nn.Module):
    apc_v2_layout = "lfm25-24b-a2b-hybrid-layer-segments-v1"
    # ShortConv stages an exact closed-form rollback record on every
    # speculating forward (``_convolve_projected``), so hybrid verify rows may
    # trim it instead of replaying.  CPU evidence:
    # tests/test_lfm25_moe_model_cpu.py::test_prompt_lookup_matches_greedy_with_rejections.
    supports_speculative_rollback = True

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LFM25Model(args)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        hidden = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
        return self.model.embed_tokens.as_linear(hidden)

    def forward_with_taps(self, inputs, cache, capture_layers, *, body_only=False):
        """Target features for an external drafter, paired with the cache
        tokens consumed; ascending post-block taps, before the final norm
        (the pinned mlx-vlm LFM2 ``hidden_sink`` convention)."""
        capture_layers = tuple(int(i) for i in capture_layers)
        if (
            not capture_layers
            or tuple(sorted(set(capture_layers))) != capture_layers
            or capture_layers[0] < 0
            or capture_layers[-1] >= len(self.model.layers)
        ):
            raise ValueError("Invalid target capture layers")
        taps = {i: None for i in capture_layers}
        hidden = self.model(inputs, cache=cache, capture=taps)
        features = mx.concatenate([taps[i] for i in capture_layers], axis=-1)
        if body_only:
            return None, features
        return self.model.embed_tokens.as_linear(hidden), features

    def prefill_body(self, inputs, cache, capture_layers):
        return self.forward_with_taps(inputs, cache, capture_layers, body_only=True)[1]

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            KVCache() if layer.is_attention_layer else ArraysCache(size=1)
            for layer in self.layers
        ]

    def sanitize(self, weights):
        sanitized = {}
        for name, value in weights.items():
            if "conv.weight" in name and value.shape[-1] > value.shape[1]:
                value = value.transpose(0, 2, 1)
            replacements = {
                "w1.weight": "gate_proj.weight",
                "w2.weight": "down_proj.weight",
                "w3.weight": "up_proj.weight",
            }
            for source, target in replacements.items():
                if source in name:
                    name = name.replace(source, target)
            sanitized[name] = value
        for index in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{index}.feed_forward"
            for projection in ("gate_proj", "down_proj", "up_proj"):
                legacy = [
                    f"{prefix}.experts.{expert}.{projection}.weight"
                    for expert in range(self.args.num_experts)
                ]
                if legacy[0] in sanitized:
                    sanitized[f"{prefix}.switch_mlp.{projection}.weight"] = mx.stack(
                        [sanitized.pop(name) for name in legacy]
                    )
        return sanitized

    @property
    def cast_predicate(self):
        return lambda name: "expert_bias" not in name
