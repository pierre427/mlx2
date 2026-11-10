# SPDX-License-Identifier: MIT
"""Native Gemma 4 tensor math: 31B dense and 26B-A4B sparse text, vision tower.

Ported from mlx-vlm 67599f2e8ec31bf35cbb7b02794114f20844f0bb (MIT, Prince
Canuma): ``mlx_vlm/models/gemma4/{language,gemma4,vision,config}.py`` and the
helpers they import.  See provenance/gemma4-native-tensors.json.

The module tree and every parameter path match mlx-vlm's exactly, so the local
artifacts (already in MLX layout) load unchanged with a strict key match, and
path-keyed mlx2 machinery (int8 prefill ``language_model.model.layers.*``,
LoRA, residual ablation) sees the same names.  mlx2 modifications:

- attention goes through mlx2's ``scaled_dot_product_attention``, so a
  segmented prompt-lookup verify view reduces its own rows (no attention
  monkeypatch) and mlx2's decode/verify kernels apply where they qualify;
- the vocabulary head honours ``select_checkpoint_rows`` (a prompt chunk
  projects one row per lane through the 262,144-entry soft-capped head) and
  ``logits_to_keep`` trims before the final norm;
- ``DecoderLayer.tail`` computes K/V over every row but the rest of the
  layer over the last row only (a ``Layer.tail``-style split, engaged by the
  adapter's prompt-chunk tail skip through ``tail_last``);
- MoE experts use mlx2's ``SwitchGLU``;
- topologies outside the served pair fail closed: KV-shared layers, per-layer
  inputs (Gemma 3n-style), an audio tower, unknown RoPE types;
- the loader is mlx2's own (config + generation_config eos, sanitize, the
  artifact's quantization incl. per-module overrides, strict load).

The mlx-vlm processor (text/image/video preprocessing) is still used by the
adapter; it is not tensor math.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..prefill_checkpoint_rows import select_checkpoint_rows
from ..prefill_tail_skip import STATS as _TAIL_STATS
from .base import BaseModelArgs, create_causal_mask, scaled_dot_product_attention
from .rope_utils import ProportionalRoPE
from .switch_layers import SwitchGLU

SOURCE_REVISION = "67599f2e8ec31bf35cbb7b02794114f20844f0bb"


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class VisionConfig(BaseModelArgs):
    model_type: str = "gemma4_vision"
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    global_head_dim: int = 64
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 131072
    attention_bias: bool = False
    use_bidirectional_attention: str = "vision"
    layer_types: Optional[List[str]] = None
    rope_parameters: Optional[Dict] = None
    default_output_length: int = 280
    patch_size: int = 16
    position_embedding_size: int = 10240
    pooling_kernel_size: int = 3
    use_clipped_linears: bool = False
    standardize: bool = False

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": 100.0, "rope_type": "default"}


@dataclass
class TextConfig(BaseModelArgs):
    model_type: str = "gemma4_text"
    hidden_size: int = 1536
    num_hidden_layers: int = 35
    intermediate_size: int = 6144
    num_attention_heads: int = 8
    head_dim: int = 256
    global_head_dim: int = 512
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262144
    num_key_value_heads: int = 1
    num_global_key_value_heads: Optional[int] = None
    # mlx-vlm's defaults: a config that omits them describes an E2B/E4B-style
    # model, which the native constructor then refuses.
    num_kv_shared_layers: int = 20
    pad_token_id: int = 0
    hidden_activation: str = "gelu_pytorch_tanh"
    hidden_size_per_layer_input: int = 256
    rope_traditional: bool = False
    rope_parameters: Optional[Dict] = None
    sliding_window: int = 512
    sliding_window_pattern: int = 5
    max_position_embeddings: int = 131072
    attention_bias: bool = False
    attention_k_eq_v: bool = False
    use_bidirectional_attention: Optional[str] = None
    final_logit_softcapping: Optional[float] = 30.0
    enable_moe_block: bool = False
    num_experts: Optional[int] = None
    top_k_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    layer_types: Optional[List[str]] = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {
                    "partial_rotary_factor": 1.0,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional",
                },
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
            }
        if self.layer_types is None:
            pattern = ["sliding_attention"] * (self.sliding_window_pattern - 1) + [
                "full_attention"
            ]
            self.layer_types = (pattern * (self.num_hidden_layers // len(pattern) + 1))[
                : self.num_hidden_layers
            ]


@dataclass
class ModelConfig(BaseModelArgs):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    audio_config: Optional[dict] = None
    model_type: str = "gemma4"
    vocab_size: int = 262144
    image_token_id: int = 258880
    audio_token_id: int = 258881
    video_token_id: Optional[int] = None
    boi_token_id: int = 255999
    eoi_token_id: int = 258882
    pad_token_id: int = 0
    eos_token_id: Optional[Any] = None
    tie_word_embeddings: bool = True

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        text = params.get("text_config")
        vision = params.get("vision_config")
        params["text_config"] = (
            text if isinstance(text, TextConfig) else TextConfig.from_dict(text or {})
        )
        params["vision_config"] = (
            vision
            if isinstance(vision, VisionConfig)
            else VisionConfig.from_dict(vision or {})
        )
        return super().from_dict(params)


# --------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------


@partial(mx.compile, shapeless=True)
def geglu(gate, x):
    return nn.gelu_approx(gate) * x


@partial(mx.compile, shapeless=True)
def logit_softcap(softcap, x):
    return mx.tanh(x / softcap) * softcap


class RMSNormNoScale(nn.Module):
    """RMSNorm without a learnable scale."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, None, self.eps)


def _cache_mask(h, cache, *, window_size=None, return_array=False):
    """mlx-vlm's ``create_attention_mask``: ask the cache whenever there is one."""
    N = h.shape[1]
    if cache is not None and not isinstance(cache, mx.array) and hasattr(
        cache, "make_mask"
    ):
        return cache.make_mask(N, return_array=return_array, window_size=window_size)
    if N == 1:
        return None
    if return_array or (window_size and N > window_size):
        return create_causal_mask(N, window_size=window_size)
    return "causal"


def _rope(head_dim, params, traditional):
    rope_type = params.get("type") or params.get("rope_type", "default")
    base = params.get("rope_theta", 10000.0)
    if rope_type in ("default", "linear"):
        scale = 1 / params["factor"] if rope_type == "linear" else 1.0
        return nn.RoPE(head_dim, traditional=traditional, base=base, scale=scale)
    if rope_type == "proportional":
        partial_factor = params.get("partial_rotary_factor", 1.0)
        rotated = 2 * int(partial_factor * head_dim // 2)
        if rotated <= 0:
            raise ValueError("Gemma 4 proportional RoPE needs rotated dimensions")
        return ProportionalRoPE(
            dims=head_dim,
            rotated_dims=rotated,
            traditional=traditional,
            base=base,
            factor=params.get("factor", 1.0),
        )
    raise ValueError(f"Gemma 4 RoPE type {rope_type!r} is not ported")


# --------------------------------------------------------------------------
# Text model
# --------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        dim, hidden = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(geglu(self.gate_proj(x), self.up_proj(x)))


class Router(nn.Module):
    """Expert router: scaled RMS norm -> project -> top-k -> softmax -> scale."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.top_k = config.top_k_experts
        self.eps = config.rms_norm_eps
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = mx.ones((config.hidden_size,))
        self.per_expert_scale = mx.ones((config.num_experts,))
        self._root_size = config.hidden_size**-0.5

    def __call__(self, x):
        x = mx.fast.rms_norm(x, self.scale * self._root_size, self.eps)
        scores = self.proj(x)
        indices = mx.argpartition(scores, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        weights = mx.softmax(mx.take_along_axis(scores, indices, axis=-1), axis=-1)
        return indices, weights * self.per_expert_scale[indices]


class GeGLU(nn.Module):
    """GELU-gated unit in ``SwitchGLU``'s ``activation(x_up, x_gate)`` order."""

    def __call__(self, x, gate):
        return geglu(gate, x)


class Experts(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.switch_glu = SwitchGLU(
            input_dims=config.hidden_size,
            hidden_dims=config.moe_intermediate_size,
            num_experts=config.num_experts,
            activation=GeGLU(),
            bias=False,
        )

    def __call__(self, x, indices, weights):
        return (mx.expand_dims(weights, -1) * self.switch_glu(x, indices)).sum(-2)


class Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        # Present for code that inspects mlx-vlm's attention; KV sharing is
        # refused at model construction.
        self.kv_shared_only = False
        self.is_kv_shared_layer = False
        full = self.layer_type == "full_attention"
        self.head_dim = (
            config.global_head_dim if full and config.global_head_dim else config.head_dim
        )
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        # Full-attention layers of the 26B/31B share K and V: values are the
        # raw k_proj output (before k_norm), and there is no v_proj.
        self.use_k_eq_v = bool(config.attention_k_eq_v) and not self.is_sliding
        if self.use_k_eq_v and config.num_global_key_value_heads is not None:
            self.n_kv_heads = config.num_global_key_value_heads
        else:
            self.n_kv_heads = config.num_key_value_heads
        self.scale = 1.0
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        if not self.use_k_eq_v:
            self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNormNoScale(self.head_dim, eps=config.rms_norm_eps)
        key = "sliding_attention" if self.is_sliding else "full_attention"
        self.rope = _rope(
            self.head_dim, config.rope_parameters.get(key, {}), config.rope_traditional
        )

    def _keys_values(self, x, cache):
        B, L, _ = x.shape
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        values = (
            keys
            if self.use_k_eq_v
            else self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        )
        offset = mx.array(cache.offset) if cache is not None else 0
        keys = self.rope(self.k_norm(keys).transpose(0, 2, 1, 3), offset=offset)
        values = self.v_norm(values).transpose(0, 2, 1, 3)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        return keys, values, offset

    def __call__(self, x, mask=None, cache=None, shared_kv=None, offset=None):
        if shared_kv is not None:
            raise ValueError("native Gemma 4 does not port KV-shared layers")
        B, L, _ = x.shape
        queries = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim))
        keys, values, offset = self._keys_values(x, cache)
        queries = self.rope(queries.transpose(0, 2, 1, 3), offset=offset)
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values), offset

    def tail(self, x, mask=None, cache=None):
        """K/V over every row (cache updated as usual); output for the last row."""
        B, L, _ = x.shape
        queries = self.q_norm(
            self.q_proj(x[:, -1:]).reshape(B, 1, self.n_heads, self.head_dim)
        )
        keys, values, offset = self._keys_values(x, cache)
        queries = self.rope(queries.transpose(0, 2, 1, 3), offset=offset + (L - 1))
        if isinstance(mask, str):
            mask = None  # one query aligned with the last key sees every key
        elif mask is not None:
            mask = mask[..., -1:, :]
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, 1, -1))


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        eps = config.rms_norm_eps
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.enable_moe = bool(config.enable_moe_block)
        if self.enable_moe:
            self.router = Router(config)
            self.experts = Experts(config)
            self.post_feedforward_layernorm_1 = nn.RMSNorm(config.hidden_size, eps=eps)
            self.post_feedforward_layernorm_2 = nn.RMSNorm(config.hidden_size, eps=eps)
            self.pre_feedforward_layernorm_2 = nn.RMSNorm(config.hidden_size, eps=eps)
        self.per_layer_input_gate = None
        self.layer_scalar = mx.ones((1,))

    def _feed_forward(self, h):
        residual = h
        if self.enable_moe:
            dense = self.post_feedforward_layernorm_1(
                self.mlp(self.pre_feedforward_layernorm(h))
            )
            indices, weights = self.router(h)
            sparse = self.post_feedforward_layernorm_2(
                self.experts(self.pre_feedforward_layernorm_2(h), indices, weights)
            )
            h = dense + sparse
        else:
            h = self.mlp(self.pre_feedforward_layernorm(h))
        h = residual + self.post_feedforward_layernorm(h)
        return h * self.layer_scalar

    def __call__(
        self, x, mask=None, cache=None, per_layer_input=None, shared_kv=None, offset=None
    ):
        """mlx-vlm's layer contract: ``(hidden, (keys, values), offset)``."""
        if per_layer_input is not None:
            raise ValueError("native Gemma 4 does not port per-layer inputs")
        attended, kvs, offset = self.self_attn(
            self.input_layernorm(x), mask, cache, shared_kv=shared_kv, offset=offset
        )
        h = x + self.post_attention_layernorm(attended)
        return self._feed_forward(h), kvs, offset

    def tail(self, x, mask=None, cache=None):
        """The layer for the last row only; K/V still cover (and cache) all rows."""
        attended = self.self_attn.tail(self.input_layernorm(x), mask, cache)
        return self._feed_forward(x[:, -1:] + self.post_attention_layernorm(attended))


class Gemma4TextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        if int(config.num_kv_shared_layers or 0):
            raise ValueError("native Gemma 4 does not port KV-shared layers")
        if int(config.hidden_size_per_layer_input or 0):
            raise ValueError("native Gemma 4 does not port per-layer inputs")
        if config.enable_moe_block and not (
            config.num_experts and config.top_k_experts and config.moe_intermediate_size
        ):
            raise ValueError("Gemma 4 MoE block needs experts, top-k and width")
        self.config = config
        self.vocab_size = config.vocab_size
        self.window_size = config.sliding_window
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = config.hidden_size**0.5
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # mlx-vlm attribute names other mlx2 code reads.
        self.first_kv_shared_layer_idx = config.num_hidden_layers
        self.hidden_size_per_layer_input = 0

    def _make_masks(self, h, cache, mm_token_type_ids=None):
        """Masks per layer type, as mlx-vlm builds them (vision overlay included)."""
        mask, masks = {}, []
        has_visual = (
            mm_token_type_ids is not None
            and int(mx.sum((mm_token_type_ids == 1) | (mm_token_type_ids == 2)).item())
            > 0
        )
        bidirectional = (
            self.config.use_bidirectional_attention == "vision"
            and has_visual
            and h.shape[1] > 1
        )
        for layer, c in zip(self.layers, cache):
            kind = layer.layer_type
            if kind not in mask:
                if kind == "full_attention":
                    mask[kind] = _cache_mask(
                        h,
                        c,
                        return_array=bidirectional
                        or getattr(c, "left_padding", None) is not None,
                    )
                else:
                    return_array = (
                        h.shape[1] > 1
                        and c is not None
                        and int(mx.max(mx.array(c.offset)).item()) > 0
                    ) or bidirectional
                    mask[kind] = _cache_mask(
                        h, c, window_size=self.window_size, return_array=return_array
                    )
                if bidirectional and isinstance(mask[kind], str) and mask[kind] == "causal":
                    window = self.window_size if kind == "sliding_attention" else None
                    mask[kind] = create_causal_mask(h.shape[1], window_size=window)
                if bidirectional and isinstance(mask[kind], mx.array):
                    mask[kind] = _bidirectional_vision_overlay(mask[kind], mm_token_type_ids)
            masks.append(mask[kind])
        return masks

    def __call__(
        self,
        inputs=None,
        inputs_embeds=None,
        mask=None,
        cache=None,
        mm_token_type_ids=None,
        token_type_ids=None,
        final_norm=True,
        tail_last=False,
    ):
        """``tail_last`` (prompt-chunk tail skip, claimed by the adapter): the
        last layer runs ``DecoderLayer.tail`` and one row per lane returns."""
        if inputs_embeds is None:
            h = self.embed_tokens(inputs) * self.embed_scale
        else:
            h = inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        elif len(cache) != len(self.layers):
            raise ValueError("Gemma 4 cache list does not match the decoder depth")
        if mask is None:
            if mm_token_type_ids is None:
                mm_token_type_ids = token_type_ids
            masks = self._make_masks(h, cache, mm_token_type_ids)
        else:
            masks = [mask] * len(self.layers)
        last = len(self.layers) - 1
        for index, (layer, c, m) in enumerate(zip(self.layers, cache, masks)):
            if tail_last and index == last and type(layer) is DecoderLayer:
                h = layer.tail(h, m, c)
                # Counted where the split is decided (sweep 2026-11-09
                # GEM-1): a swapped last-layer class runs the full layer.
                _TAIL_STATS["gemma4_native_tail_split"] += 1
            else:
                if tail_last and index == last:
                    _TAIL_STATS["gemma4_native_tail_full"] += 1
                h, _, _ = layer(h, m, c)
        return self.norm(h) if final_norm else h


def _bidirectional_vision_overlay(base_mask, mm_token_type_ids):
    """Let each contiguous image/video soft-token block attend within itself."""
    if mm_token_type_ids.shape[1] != base_mask.shape[-1]:
        return base_mask
    if base_mask.shape[-2] != base_mask.shape[-1]:
        return base_mask
    is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
    prev = mx.concatenate([mx.zeros_like(is_vision[:, :1]), is_vision[:, :-1]], axis=1)
    starts = is_vision & ~prev
    group = mx.cumsum(starts.astype(mx.int32), axis=1) - 1
    blocks = mx.where(is_vision, group, mx.zeros_like(group) - 1)
    q_blocks, k_blocks = mx.expand_dims(blocks, -1), mx.expand_dims(blocks, -2)
    same = (q_blocks != -1) & (q_blocks == k_blocks)
    return base_mask | mx.expand_dims(same, 1)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Gemma4TextModel(config)
        self.final_logit_softcapping = config.final_logit_softcapping

    def logits_from_hidden(self, hidden):
        logits = self.model.embed_tokens.as_linear(hidden)
        if self.final_logit_softcapping is not None:
            logits = logit_softcap(self.final_logit_softcapping, logits)
        return logits

    def __call__(
        self,
        inputs=None,
        inputs_embeds=None,
        mask=None,
        cache=None,
        logits_to_keep=None,
        mm_token_type_ids=None,
        token_type_ids=None,
        tail_last=False,
    ):
        hidden = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            mm_token_type_ids=mm_token_type_ids,
            token_type_ids=token_type_ids,
            final_norm=False,
            tail_last=tail_last,
        )
        # The final norm is row-wise, so narrowing first changes no kept row.
        if logits_to_keep:
            hidden = hidden[:, -int(logits_to_keep) :, :]
        else:
            hidden = select_checkpoint_rows(hidden)
        return self.logits_from_hidden(self.model.norm(hidden))

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    def make_cache(self):
        from .cache import KVCache, RotatingKVCache

        return [
            KVCache()
            if kind == "full_attention"
            else RotatingKVCache(max_size=self.config.sliding_window, keep=0)
            for kind in self.config.layer_types
        ]


# --------------------------------------------------------------------------
# Vision tower
# --------------------------------------------------------------------------


class ClippableLinear(nn.Module):
    """Linear with optional checkpoint-supplied input/output clamps."""

    def __init__(self, in_features, out_features, bias=False, use_clipping=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.use_clipping = use_clipping
        if use_clipping:
            self.input_min = mx.array(float("-inf"))
            self.input_max = mx.array(float("inf"))
            self.output_min = mx.array(float("-inf"))
            self.output_max = mx.array(float("inf"))

    def __call__(self, x):
        if self.use_clipping:
            x = mx.clip(x, self.input_min, self.input_max)
        x = self.linear(x)
        if self.use_clipping:
            x = mx.clip(x, self.output_min, self.output_max)
        return x


def _one_hot(indices, num_classes):
    return (mx.expand_dims(indices, -1) == mx.arange(num_classes)).astype(mx.float32)


class VisionRMSNorm(nn.Module):
    """Scaled RMS norm computed in float32."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x):
        xf = x.astype(mx.float32)
        normed = xf * mx.rsqrt(mx.mean(xf**2, axis=-1, keepdims=True) + self.eps)
        return (normed * self.weight.astype(mx.float32)).astype(x.dtype)


class VisionRMSNormNoScale(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x):
        xf = x.astype(mx.float32)
        return (xf * mx.rsqrt(mx.mean(xf**2, axis=-1, keepdims=True) + self.eps)).astype(
            x.dtype
        )


def _rotate_half(x):
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _rope_tables(positions, channels, base, dtype):
    exponents = (2.0 / channels) * mx.arange(0, channels // 2).astype(mx.float32)
    angles = positions[..., None].astype(mx.float32) / mx.power(base, exponents)
    cos, sin = mx.cos(angles), mx.sin(angles)
    cos = mx.expand_dims(mx.concatenate([cos, cos], axis=-1).astype(dtype), 2)
    sin = mx.expand_dims(mx.concatenate([sin, sin], axis=-1).astype(dtype), 2)
    return cos, sin


def apply_multidimensional_rope(inputs, positions, base_frequency=100.0):
    """2-D RoPE: rotate_half independently within each spatial axis' channels.

    ``inputs`` is ``[B, L, N, H]``; ``positions`` is ``[B, L, 2]`` (or
    ``[B, L]`` for one axis).
    """
    head_dim = inputs.shape[-1]
    if positions.ndim == 2:
        cos, sin = _rope_tables(positions, head_dim, base_frequency, inputs.dtype)
        return inputs * cos + _rotate_half(inputs) * sin
    ndim = positions.shape[-1]
    channels = 2 * (head_dim // (2 * ndim))
    parts = []
    for d in range(ndim):
        part = inputs[..., d * channels : (d + 1) * channels]
        cos, sin = _rope_tables(positions[..., d], channels, base_frequency, inputs.dtype)
        parts.append(part * cos + _rotate_half(part) * sin)
    return mx.concatenate(parts, axis=-1)


def _padded_fused_sdpa(q, k, v, scale, mask=None):
    """Pad head_dim up to a fused SDPA size (64/80/128), then drop the padding."""
    d = q.shape[-1]
    target = next((t for t in (64, 80, 128) if d <= t), d)
    if target != d:
        pad = [(0, 0)] * (q.ndim - 1) + [(0, target - d)]
        q, k, v = mx.pad(q, pad), mx.pad(k, pad), mx.pad(v, pad)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)[
        ..., :d
    ]


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.rope_base_frequency = config.rope_parameters["rope_theta"]
        clip = bool(config.use_clipped_linears)
        hd, dim = self.head_dim, self.hidden_size
        self.q_proj = ClippableLinear(dim, self.num_heads * hd, use_clipping=clip)
        self.k_proj = ClippableLinear(dim, self.num_kv_heads * hd, use_clipping=clip)
        self.v_proj = ClippableLinear(dim, self.num_kv_heads * hd, use_clipping=clip)
        self.o_proj = ClippableLinear(self.num_heads * hd, dim, use_clipping=clip)
        self.q_norm = VisionRMSNorm(hd)
        self.k_norm = VisionRMSNorm(hd)
        self._v_norm = VisionRMSNormNoScale()

    def __call__(self, x, positions, mask=None):
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim))
        v = self._v_norm(self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim))
        q = apply_multidimensional_rope(q, positions, self.rope_base_frequency)
        k = apply_multidimensional_rope(k, positions, self.rope_base_frequency)
        q, k, v = (z.transpose(0, 2, 1, 3) for z in (q, k, v))
        out = _padded_fused_sdpa(q, k, v, scale=1.0, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        clip = bool(config.use_clipped_linears)
        dim, hidden = config.hidden_size, config.intermediate_size
        self.gate_proj = ClippableLinear(dim, hidden, use_clipping=clip)
        self.up_proj = ClippableLinear(dim, hidden, use_clipping=clip)
        self.down_proj = ClippableLinear(hidden, dim, use_clipping=clip)

    def __call__(self, x):
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class VisionTransformerBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        eps = config.rms_norm_eps
        self.self_attn = VisionAttention(config)
        self.mlp = VisionMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)

    def __call__(self, x, positions, mask=None):
        h = x + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x), positions, mask)
        )
        return h + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(h))
        )


class VisionPatchEmbedder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.position_embedding_size = config.position_embedding_size
        self.input_proj = nn.Linear(3 * self.patch_size**2, self.hidden_size, bias=False)
        self.position_embedding_table = mx.ones(
            (2, self.position_embedding_size, self.hidden_size)
        )

    def _position_embeddings(self, patch_positions, padding_positions):
        oh = _one_hot(patch_positions, self.position_embedding_size)
        oh = oh.transpose(0, 2, 1, 3).astype(self.position_embedding_table.dtype)
        embeddings = (oh @ self.position_embedding_table).sum(axis=1)
        return mx.where(mx.expand_dims(padding_positions, -1), 0.0, embeddings)

    def _patchify(self, pixel_values):
        B, C, H, W = pixel_values.shape
        p = self.patch_size
        patches = pixel_values.reshape(B, C, H // p, p, W // p, p)
        patches = patches.transpose(0, 2, 4, 3, 5, 1).reshape(
            B, (H // p) * (W // p), C * p * p
        )
        patches = 2 * (patches - 0.5)
        return self.input_proj(patches.astype(self.input_proj.weight.dtype))

    def embed_patches(self, pixel_values, patch_positions, padding_positions):
        """Pre-patchified ``[B, L, 3*p*p]`` input (the processor's packed form)."""
        hidden = self.input_proj((2 * (pixel_values - 0.5)).astype(self.input_proj.weight.dtype))
        return hidden + self._position_embeddings(patch_positions, padding_positions)

    def __call__(self, pixel_values, patch_positions, padding_positions):
        return self._patchify(pixel_values) + self._position_embeddings(
            patch_positions, padding_positions
        )


class VisionPooler(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.default_output_length = config.default_output_length
        self.root_hidden_size = self.hidden_size**0.5

    def _avg_pool_by_positions(self, x, patch_positions, length):
        k = int((x.shape[1] // length) ** 0.5)
        clamped = mx.clip(patch_positions, 0, None)
        max_x = mx.max(clamped[..., 0], axis=-1, keepdims=True) + 1
        kernel = mx.floor(clamped.astype(mx.float32) / k).astype(mx.int32)
        kernel = kernel[..., 0] + (max_x // k) * kernel[..., 1]
        weights = _one_hot(kernel, length).astype(mx.float32) / (k**2)
        output = mx.einsum("bLl,bLd->bld", weights, x).astype(x.dtype)
        return output, mx.logical_not(mx.all(weights == 0, axis=1))

    def __call__(self, hidden_states, patch_positions, padding_positions, output_length=None):
        hidden_states = mx.where(mx.expand_dims(padding_positions, -1), 0.0, hidden_states)
        length = output_length or self.default_output_length
        if hidden_states.shape[1] == length:
            mask = padding_positions
        else:
            hidden_states, mask = self._avg_pool_by_positions(
                hidden_states, patch_positions, length
            )
        return hidden_states * self.root_hidden_size, mask


class VisionTransformerModel(nn.Module):
    """The transformer stack (``vision_tower.encoder`` in the weights)."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [VisionTransformerBlock(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, hidden_states, positions, mask):
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, mask)
        return hidden_states


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.patch_size = config.patch_size
        self.pooling_kernel_size = config.pooling_kernel_size
        self.default_output_length = config.default_output_length
        self.max_patches = self.default_output_length * self.pooling_kernel_size**2
        self.patch_embedder = VisionPatchEmbedder(config)
        self.encoder = VisionTransformerModel(config)
        self.pooler = VisionPooler(config)
        if config.standardize:
            self.std_bias = mx.zeros((config.hidden_size,))
            self.std_scale = mx.ones((config.hidden_size,))

    def _patch_positions_single(self, H, W, max_patches):
        pH, pW = H // self.patch_size, W // self.patch_size
        num_patches = pH * pW
        gx, gy = np.meshgrid(np.arange(pW), np.arange(pH), indexing="xy")
        real = np.stack([gx.flatten(), gy.flatten()], axis=-1)
        num_padding = max_patches - num_patches
        if num_padding > 0:
            positions = np.concatenate(
                [real, np.full((num_padding, 2), -1, dtype=np.int64)], axis=0
            )
        else:
            positions = real[:max_patches]
        padding = np.zeros(max_patches, dtype=bool)
        if num_padding > 0:
            padding[num_patches:] = True
        return positions.astype(np.int32), padding

    def __call__(self, pixel_values, pixel_position_ids=None):
        if isinstance(pixel_values, list):
            items = (
                pixel_position_ids
                if isinstance(pixel_position_ids, list)
                else [None] * len(pixel_values)
            )
            real = []
            for image, position in zip(pixel_values, items):
                image = image if isinstance(image, mx.array) else mx.array(image)
                if position is not None and not isinstance(position, mx.array):
                    position = mx.array(position)
                if position is None and image.ndim == 3:
                    image = image[None]
                real.append(self(image, position)[0])
            return mx.concatenate(real, axis=0)[None]

        if not isinstance(pixel_values, mx.array):
            pixel_values = mx.array(pixel_values)
        pool_sq = self.pooling_kernel_size**2
        if pixel_position_ids is not None:
            if not isinstance(pixel_position_ids, mx.array):
                pixel_position_ids = mx.array(pixel_position_ids)
            if pixel_values.ndim == 2:
                pixel_values = pixel_values[None]
            if pixel_position_ids.ndim == 2:
                pixel_position_ids = pixel_position_ids[None]
            if pixel_values.ndim == 4:
                bsz, frames, seq_len, patch_dim = pixel_values.shape
                pixel_values = pixel_values.reshape(bsz * frames, seq_len, patch_dim)
                pixel_position_ids = pixel_position_ids.reshape(bsz * frames, seq_len, 2)
            patch_positions = pixel_position_ids
            padding_positions = mx.all(patch_positions == -1, axis=-1)
            B = pixel_values.shape[0]
            output_length = pixel_values.shape[1] // pool_sq
            embeds = self.patch_embedder.embed_patches(
                pixel_values, patch_positions, padding_positions
            )
        else:
            B, _, H, W = pixel_values.shape
            num_real = (H // self.patch_size) * (W // self.patch_size)
            output_length = num_real // pool_sq
            positions, padding = self._patch_positions_single(H, W, num_real)
            patch_positions = mx.array(np.tile(positions[None], (B, 1, 1)))
            padding_positions = mx.array(np.tile(padding[None], (B, 1)))
            embeds = self.patch_embedder(pixel_values, patch_positions, padding_positions)

        # Bidirectional [B, 1, L, L] additive mask; -1e4 (not -inf) keeps
        # all-padding query rows finite.
        valid = ~padding_positions
        allowed = mx.expand_dims(valid, 1) * mx.expand_dims(valid, 2)
        attn_mask = mx.where(
            allowed,
            mx.array(0.0, dtype=embeds.dtype),
            mx.array(-1e4, dtype=embeds.dtype),
        )
        hidden = self.encoder(embeds, patch_positions, mx.expand_dims(attn_mask, 1))
        pooled, pool_mask = self.pooler(
            hidden, patch_positions, padding_positions, output_length=output_length
        )
        valid = pool_mask if pool_mask.shape[1] == output_length else ~pool_mask
        real = []
        for i in range(B):
            count = int(valid[i].astype(mx.int32).sum().item())
            real.append(pooled[i, :count])
        hidden = mx.concatenate(real, axis=0)[None]
        if self.config.standardize:
            hidden = (hidden - self.std_bias) * self.std_scale
        return hidden


class MultimodalEmbedder(nn.Module):
    """Projects vision soft tokens into the text model's hidden space."""

    def __init__(self, embedding_dim, text_hidden_size, eps=1e-6):
        super().__init__()
        self.embedding_projection = nn.Linear(embedding_dim, text_hidden_size, bias=False)
        self.embedding_pre_projection_norm = RMSNormNoScale(embedding_dim, eps=eps)

    def __call__(self, inputs_embeds):
        return self.embedding_projection(self.embedding_pre_projection_norm(inputs_embeds))


def masked_scatter(input_tensor, mask, source):
    mask_flat = mask.flatten().astype(mx.int32)
    indices = mx.cumsum(mask_flat) - 1
    aligned = source.flatten()[indices % source.size]
    return mx.where(mask_flat, aligned, input_tensor.flatten()).reshape(input_tensor.shape)


# --------------------------------------------------------------------------
# Multimodal model
# --------------------------------------------------------------------------


class Model(nn.Module):
    """Gemma 4 vision/video + text, mlx-vlm's module tree, mlx2-owned math."""

    mlx2_native_tensors = True

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.audio_config is not None:
            raise ValueError("native Gemma 4 does not port the audio tower")
        self.model_type = config.model_type
        self.config = config
        self.language_model = LanguageModel(config.text_config)
        self.vocab_size = config.text_config.vocab_size
        self.vision_tower = VisionModel(config.vision_config)
        self.embed_vision = MultimodalEmbedder(
            embedding_dim=config.vision_config.hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            eps=config.vision_config.rms_norm_eps,
        )

    def encode_image(self, pixel_values, image_position_ids=None):
        return self.embed_vision(self.vision_tower(pixel_values, image_position_ids))

    def get_input_embeddings(
        self,
        input_ids,
        pixel_values=None,
        image_position_ids=None,
        pixel_values_videos=None,
        video_position_ids=None,
        **kwargs,
    ):
        text = self.language_model.model
        embeds = text.embed_tokens(input_ids) * text.embed_scale

        def scatter(embeds, source, token_id, positions, cached_kw, key_kw):
            if source is None or token_id is None:
                return embeds
            vision_cache = kwargs.get("vision_cache")
            key = kwargs.get(key_kw)
            cached = kwargs.get(cached_kw)
            if cached is None and vision_cache is not None:
                cached = vision_cache.get(key)
            if cached is not None:
                features = cached.astype(embeds.dtype)
            else:
                features = self.encode_image(source, positions).astype(embeds.dtype)
                if vision_cache is not None and key is not None:
                    mx.eval(features)
                    vision_cache.put(key, features)
            selected = mx.broadcast_to(
                mx.expand_dims(input_ids == token_id, -1), embeds.shape
            )
            return masked_scatter(embeds, selected, features)

        embeds = scatter(
            embeds,
            pixel_values,
            self.config.image_token_id,
            image_position_ids,
            "cached_image_features",
            "_image_key",
        )
        return scatter(
            embeds,
            pixel_values_videos,
            self.config.video_token_id,
            video_position_ids,
            "cached_video_features",
            "_video_key",
        )

    # mlx-vlm speculative hooks this port does not implement; refuse them
    # rather than silently returning plain logits.
    _UNPORTED_KWARGS = frozenset({
        "capture_layer_ids", "hidden_sink", "shared_kv_sink", "return_hidden",
        "return_shared_kv", "speculative_verify", "per_layer_inputs",
        "audio_features", "input_features",
    })

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        """Logits ``[batch, rows, vocab]``; processor extras are ignored as in mlx-vlm."""
        tail_last = bool(kwargs.pop("tail_last", False))
        unported = self._UNPORTED_KWARGS.intersection(kwargs)
        if unported:
            raise TypeError(f"native Gemma 4 does not port {sorted(unported)}")
        embeds = self.get_input_embeddings(input_ids, pixel_values=pixel_values, **kwargs)
        return self.language_model(
            inputs_embeds=embeds,
            mask=mask,
            cache=cache,
            logits_to_keep=kwargs.get("logits_to_keep"),
            mm_token_type_ids=kwargs.get("mm_token_type_ids"),
            token_type_ids=kwargs.get("token_type_ids"),
            tail_last=tail_last,
        )

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def sanitize(self, weights):
        """mlx-vlm's Gemma 4 key mapping (HF names and MLX layout both accepted)."""
        clipped = bool(self.config.vision_config.use_clipped_linears)
        out = {}
        for key, value in weights.items():
            if any(s in key for s in ("input_max", "input_min", "output_max", "output_min")):
                if "vision_tower" not in key or not clipped:
                    continue
            if "rotary_emb" in key:
                continue
            if "audio_tower" in key or "embed_audio" in key:
                raise ValueError("Gemma 4 artifact carries audio weights; not ported")
            new = key[len("model.") :] if key.startswith("model.") else key
            if new.startswith("language_model.") and not new.startswith(
                "language_model.model."
            ):
                new = "language_model.model." + new[len("language_model.") :]
            if new.endswith(".experts.down_proj"):
                new = new.replace(".experts.down_proj", ".experts.switch_glu.down_proj.weight")
            if new.endswith(".experts.gate_up_proj"):
                value = value.swapaxes(-1, -2)
                mid = value.shape[-1] // 2
                base = new[: -len(".experts.gate_up_proj")]
                out[base + ".experts.switch_glu.gate_proj.weight"] = value[
                    ..., :mid
                ].swapaxes(-1, -2)
                out[base + ".experts.switch_glu.up_proj.weight"] = value[
                    ..., mid:
                ].swapaxes(-1, -2)
                continue
            out[new] = value
        return out


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

_GENERATION_KEYS = ("eos_token_id", "temperature", "top_p", "top_k", "do_sample")


def load_config(model_path) -> dict:
    """``config.json`` with the generation-config keys mlx-vlm merges in."""
    path = Path(model_path)
    config = json.loads((path / "config.json").read_text())
    generation = path / "generation_config.json"
    if generation.is_file():
        try:
            values = json.loads(generation.read_text())
        except json.JSONDecodeError:
            values = None
        if isinstance(values, dict) and values:
            config["generation_config"] = values
            for key in _GENERATION_KEYS:
                if key in values:
                    config[key] = values[key]
    return config


def _quantization(config):
    quant = config.get("quantization")
    if quant is None:
        if config.get("quantization_config") is not None:
            raise ValueError(
                "native Gemma 4 loads MLX-quantized artifacts only "
                "(config.quantization); quantization_config alone is not ported"
            )
        return None
    return quant


def quantize_like_artifact(model, quant, weights):
    """Quantize exactly the modules the checkpoint stores quantized.

    Per-module dict overrides in ``quant`` (the 26B's 8-bit router) win; a
    ``False`` override keeps that module dense; every other module is
    quantized with the global settings iff the weights carry its ``scales``
    (mlx-vlm's legacy rule, including the ``weight.size % 64`` skip).
    """
    modules = dict(model.named_modules())
    unknown = sorted(
        key
        for key, value in quant.items()
        if isinstance(value, (dict, bool)) and key not in modules
    )
    if unknown:
        raise ValueError(f"quantization overrides name unknown modules: {unknown[:4]}")

    def predicate(path, module):
        override = quant.get(path)
        if isinstance(override, dict):
            return override
        if override is False:
            return False
        if not hasattr(module, "to_quantized"):
            return False
        if hasattr(module, "weight") and module.weight.size % 64 != 0:
            return False
        return f"{path}.scales" in weights

    nn.quantize(
        model,
        group_size=quant["group_size"],
        bits=quant["bits"],
        mode=quant.get("mode", "affine"),
        class_predicate=predicate,
    )


def load(model_path, *, lazy=False):
    """Build the native model from a local Gemma 4 artifact, strict key match."""
    path = Path(model_path)
    config = load_config(path)
    if str(config.get("model_type", "")).replace("-", "_") != "gemma4":
        raise ValueError("native Gemma 4 loader got a non-Gemma 4 artifact")
    model = Model(ModelConfig.from_dict(config))
    index = path / "model.safetensors.index.json"
    if index.is_file():
        # Only the shards the index names (a stale extra shard with duplicate
        # keys would otherwise overwrite real tensors unseen by strict load).
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        files = [str(path / shard) for shard in shards]
        missing = [name for name in files if not Path(name).is_file()]
        if missing:
            raise FileNotFoundError(f"index names missing shards: {missing[:3]}")
    else:
        files = sorted(
            name
            for name in glob.glob(str(path / "*.safetensors"))
            if Path(name).name != "consolidated.safetensors"
        )
    if not files:
        raise FileNotFoundError(f"No safetensors in {path}")
    weights = {}
    for name in files:
        weights.update(mx.load(name))
    weights = model.sanitize(weights)
    quant = _quantization(config)
    if quant is not None:
        quantize_like_artifact(model, quant, weights)
    model.load_weights(list(weights.items()), strict=True)
    if not lazy:
        mx.eval(model.parameters())
    model.eval()
    return model


__all__ = [
    "LanguageModel",
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "load",
    "load_config",
]
