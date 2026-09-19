# SPDX-License-Identifier: MIT
# Mined from local mlx-lm-unified; see provenance/laguna-xs21.json and .NOTICE.
"""Laguna 2.1 sparse-MoE tensor model.

Only model-specific tensor math lives here. Serving, APCv2, scheduling and
route selection remain in shared mlx2 layers.
"""

import os
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
from mlx import nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


def _kernel_enabled(name: str) -> bool:
    return os.environ.get(name, "stock").strip().lower() in {
        "1", "true", "on", "yes", "candidate", "auto",
    }


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "laguna"
    vocab_size: int = 100352
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 40
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    qkv_bias: bool = False
    attention_bias: bool = False
    gating: bool | str = "per-head"
    tie_word_embeddings: bool = False
    rope_theta: float = 500000.0
    rope_parameters: dict[str, Any] | None = None
    rope_scaling: dict[str, Any] | None = None
    partial_rotary_factor: float | None = None
    sliding_window: int | None = 512
    layer_types: list[str] | None = None
    num_attention_heads_per_layer: list[int] | None = None
    mlp_layer_types: list[str] | None = None
    swa_rope_parameters: dict[str, Any] | None = None
    swa_attention_sink_enabled: bool = False
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: list[int] = field(default_factory=lambda: [0])
    moe_routed_scaling_factor: float = 2.5
    moe_apply_router_weight_on_input: bool = False
    moe_router_logit_softcapping: float = 0.0
    moe_router_use_sigmoid: bool = True

    def __post_init__(self):
        if self.gating is True:
            self.gating = "per-head"
        if self.gating not in (False, "per-head", "per-element"):
            raise ValueError("unsupported Laguna attention gating")
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if len(self.layer_types) != self.num_hidden_layers or any(
            kind not in {"full_attention", "sliding_attention"}
            for kind in self.layer_types
        ):
            raise ValueError("invalid Laguna layer_types")
        if self.num_attention_heads_per_layer is None:
            self.num_attention_heads_per_layer = [self.num_attention_heads] * self.num_hidden_layers
        if len(self.num_attention_heads_per_layer) != self.num_hidden_layers:
            raise ValueError("num_attention_heads_per_layer must match num_hidden_layers")
        if any(heads % self.num_key_value_heads for heads in self.num_attention_heads_per_layer):
            raise ValueError("query-head counts must be divisible by KV heads")
        if self.mlp_layer_types is not None and (
            len(self.mlp_layer_types) != self.num_hidden_layers
            or any(kind not in {"dense", "sparse"} for kind in self.mlp_layer_types)
        ):
            raise ValueError("invalid Laguna mlp_layer_types")
        if "sliding_attention" in self.layer_types and not self.sliding_window:
            raise ValueError("sliding Laguna layers require a window")

        rope = dict(self.rope_parameters or self.rope_scaling or {
            "rope_type": "default", "rope_theta": self.rope_theta
        })
        layer_rope = {
            key: value for key, value in rope.items()
            if key in set(self.layer_types) and isinstance(value, dict)
        }
        if layer_rope:
            common = {key: value for key, value in rope.items() if key not in layer_rope}
            def for_layer(kind):
                value = dict(layer_rope.get(kind, {}))
                for key, item in common.items():
                    value.setdefault(key, item)
                return value
            self.rope_parameters = for_layer("full_attention")
            if self.swa_rope_parameters is None and "sliding_attention" in layer_rope:
                self.swa_rope_parameters = for_layer("sliding_attention")
        else:
            self.rope_parameters = rope
        self.rope_parameters.setdefault("rope_type", "default")
        if self.swa_rope_parameters is not None:
            self.swa_rope_parameters = dict(self.swa_rope_parameters)
            self.swa_rope_parameters.setdefault("rope_type", "default")


def _rope_dims(args, config):
    return int(args.head_dim * float(config.get("partial_rotary_factor", 1.0)))


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class LagunaTopKRouter(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.use_sigmoid = args.moe_router_use_sigmoid
        self.softcap = args.moe_router_logit_softcapping
        self.proj = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.num_experts,))
        self.fused_mode = _kernel_enabled("MLX_LAGUNA_FUSED_ROUTER")
        self.fused_calls = 0
        self.fused_fallbacks = 0
        self.last_fallback = None

    def __call__(self, x):
        dtype = x.dtype
        logits = self.proj(x).astype(mx.float32)
        if self.softcap > 0:
            logits = mx.tanh(logits / self.softcap) * self.softcap
        if self.fused_mode:
            from .laguna_fused_moe import (
                CANDIDATE_TOKEN_WIDTHS,
                admit_laguna_router,
                laguna_fused_router,
            )

            admission = admit_laguna_router(
                logits,
                self.e_score_correction_bias,
                top_k=self.top_k,
                norm_topk_prob=self.norm_topk_prob,
                use_sigmoid=self.use_sigmoid,
                softcap=self.softcap,
                candidate_token_widths=CANDIDATE_TOKEN_WIDTHS,
            )
            if admission.accepted:
                result = laguna_fused_router(
                    logits,
                    self.e_score_correction_bias,
                    top_k=self.top_k,
                    norm_topk_prob=self.norm_topk_prob,
                    use_sigmoid=self.use_sigmoid,
                    softcap=self.softcap,
                    output_dtype=dtype,
                    candidate_token_widths=CANDIDATE_TOKEN_WIDTHS,
                )
                self.fused_calls += 1
                self.last_fallback = None
                return result
            self.fused_fallbacks += 1
            self.last_fallback = admission.reason
        scores = mx.sigmoid(logits) if self.use_sigmoid else mx.softmax(logits, axis=-1)
        corrected = scores + self.e_score_correction_bias.astype(scores.dtype)
        inds = mx.stop_gradient(mx.argpartition(-corrected, kth=self.top_k - 1, axis=-1)[..., :self.top_k])
        weights = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob:
            weights = weights / mx.sum(weights, axis=-1, keepdims=True)
        return inds, weights.astype(dtype)


class LagunaSparseMoeBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.moe_apply_router_weight_on_input:
            raise ValueError("router-weight-on-input is unsupported")
        self.scale = args.moe_routed_scaling_factor
        self.top_k = args.num_experts_per_tok
        self.gate = LagunaTopKRouter(args)
        self.switch_mlp = SwitchGLU(args.hidden_size, args.moe_intermediate_size, args.num_experts)
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)
        self.fused_down_mode = _kernel_enabled("MLX_LAGUNA_FUSED_DOWN")
        self.fused_down_calls = 0
        self.fused_down_fallbacks = 0
        self.fused_down_last_fallback = None

    def __call__(self, x):
        inds, weights = self.gate(x)
        if self.fused_down_mode and inds.size // self.top_k > 8:
            self.fused_down_fallbacks += 1
            self.fused_down_last_fallback = "token width exceeds candidate set"
        elif self.fused_down_mode:
            from .laguna_fused_moe import (
                CANDIDATE_TOKEN_WIDTHS,
                admit_laguna_fused_down,
                laguna_fused_down,
            )
            from .switch_layers import QuantizedSwitchLinear

            down = self.switch_mlp.down_proj
            if isinstance(down, QuantizedSwitchLinear) and not down.training:
                expanded = mx.expand_dims(x, (-2, -3))
                hidden = self.switch_mlp.activation(
                    self.switch_mlp.up_proj(expanded, inds, sorted_indices=False),
                    self.switch_mlp.gate_proj(expanded, inds, sorted_indices=False),
                ).squeeze(-2)
                admission = admit_laguna_fused_down(
                    hidden,
                    inds,
                    weights,
                    down["weight"],
                    down["scales"],
                    down.get("biases"),
                    num_experts=down.num_experts,
                    group_size=down.group_size,
                    bits=down.bits,
                    mode=down.mode,
                    candidate_token_widths=CANDIDATE_TOKEN_WIDTHS,
                )
                if admission.accepted:
                    routed = laguna_fused_down(
                        hidden,
                        inds,
                        weights,
                        down["weight"],
                        down["scales"],
                        down.get("biases"),
                        num_experts=down.num_experts,
                        group_size=down.group_size,
                        bits=down.bits,
                        mode=down.mode,
                        candidate_token_widths=CANDIDATE_TOKEN_WIDTHS,
                    )
                    self.fused_down_calls += 1
                    self.fused_down_last_fallback = None
                    return routed * self.scale + self.shared_expert(x)
                self.fused_down_fallbacks += 1
                self.fused_down_last_fallback = admission.reason
            else:
                self.fused_down_fallbacks += 1
                self.fused_down_last_fallback = "down projection is not frozen q8"
        routed = mx.sum(self.switch_mlp(x, inds) * weights[..., None], axis=-2)
        return routed * self.scale + self.shared_expert(x)


class Attention(nn.Module):
    def __init__(self, args, layer_idx):
        super().__init__()
        self.n_heads = args.num_attention_heads_per_layer[layer_idx]
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5
        self.gate_per_head = args.gating == "per-head"
        self.gating = bool(args.gating)
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"
        dim = args.hidden_size
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=args.qkv_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=args.qkv_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=args.qkv_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=args.attention_bias)
        if self.gating:
            gate_dim = self.n_heads if self.gate_per_head else self.n_heads * self.head_dim
            self.g_proj = nn.Linear(dim, gate_dim, bias=False)
        self.sink = mx.zeros((self.n_heads,)) if self.is_sliding and args.swa_attention_sink_enabled else None
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        rope = args.swa_rope_parameters if self.is_sliding and args.swa_rope_parameters else args.rope_parameters
        self.rope = initialize_rope(
            _rope_dims(args, rope), base=float(rope.get("rope_theta", args.rope_theta)),
            traditional=False, scaling_config=rope,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask, sinks=self.sink)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        if self.gating:
            gate = nn.softplus(self.g_proj(x).astype(mx.float32)).astype(out.dtype)
            if self.gate_per_head:
                out = (out.reshape(batch, length, self.n_heads, self.head_dim) * gate[..., None]).reshape(out.shape)
            else:
                out = out * gate
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, args, layer_idx):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        sparse = (args.mlp_layer_types[layer_idx] == "sparse") if args.mlp_layer_types else (
            layer_idx not in args.mlp_only_layers and args.num_experts > 0 and (layer_idx + 1) % args.decoder_sparse_step == 0
        )
        self.mlp = LagunaSparseMoeBlock(args) if sparse else MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attention_type = args.layer_types[layer_idx]

    def __call__(self, x, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class LagunaModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, index) for index in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.full_index = args.layer_types.index("full_attention")
        self.sliding_index = args.layer_types.index("sliding_attention") if "sliding_attention" in args.layer_types else None

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        cache = [None] * len(self.layers) if cache is None else cache
        if len(cache) != len(self.layers):
            raise ValueError("Laguna cache layer count mismatch")
        masks = {"full_attention": create_attention_mask(h, cache[self.full_index])}
        if self.sliding_index is not None:
            masks["sliding_attention"] = create_attention_mask(h, cache[self.sliding_index], window_size=self.args.sliding_window)
        for layer, layer_cache in zip(self.layers, cache):
            h = layer(h, masks[layer.attention_type], layer_cache)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LagunaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def apc_v2_layout(self):
        return "laguna-xs21-layer-segments-v1"

    def __call__(self, inputs, cache=None, input_embeddings=None):
        hidden = self.model(inputs, cache, input_embeddings)
        return self.model.embed_tokens.as_linear(hidden) if self.args.tie_word_embeddings else self.lm_head(hidden)

    def make_cache(self):
        return [
            RotatingKVCache(max_size=self.args.sliding_window, keep=0)
            if kind == "sliding_attention" else KVCache()
            for kind in self.args.layer_types
        ]

    def sanitize(self, weights):
        if any(key.startswith("language_model.") for key in weights):
            weights = {key.removeprefix("language_model."): value for key, value in weights.items()}
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return {
            key: value for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key
            and not key.endswith(".self_attn.k_scale")
            and not key.endswith(".self_attn.v_scale")
        }

    @property
    def quant_predicate(self):
        def predicate(path, _):
            return {"group_size": 64, "bits": 8} if path.endswith("mlp.gate.proj") else True
        return predicate

    @property
    def cast_predicate(self):
        return lambda key: "e_score_correction_bias" not in key

    @property
    def layers(self):
        return self.model.layers

    def set_fused_moe_modes(self, *, down: bool, router: bool):
        for layer in self.layers:
            block = layer.mlp
            if isinstance(block, LagunaSparseMoeBlock):
                block.fused_down_mode = bool(down)
                block.gate.fused_mode = bool(router)

    def fused_moe_stats(self):
        result = {
            "down_calls": 0,
            "down_fallbacks": 0,
            "router_calls": 0,
            "router_fallbacks": 0,
            "down_last_fallback": None,
            "router_last_fallback": None,
        }
        for layer in self.layers:
            block = layer.mlp
            if not isinstance(block, LagunaSparseMoeBlock):
                continue
            result["down_calls"] += block.fused_down_calls
            result["down_fallbacks"] += block.fused_down_fallbacks
            result["router_calls"] += block.gate.fused_calls
            result["router_fallbacks"] += block.gate.fused_fallbacks
            result["down_last_fallback"] = (
                block.fused_down_last_fallback or result["down_last_fallback"]
            )
            result["router_last_fallback"] = (
                block.gate.last_fallback or result["router_last_fallback"]
            )
        return result
