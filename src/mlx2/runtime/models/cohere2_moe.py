# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see provenance/north-mini-code.json and .NOTICE.
"""Cohere2 MoE tensor model for North Mini Code.

This module contains model-specific tensor math only. Serving, APCv2,
admission, scheduling, and route selection stay in their shared layers.
"""

import os
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
    # Upstream Cohere2Moe picks the norm *class* from this field: RMSNorm when
    # it is present, the mean-centred Cohere LayerNorm when it is None.  See
    # _norm_layer below.  It must stay a declared field or BaseModelArgs.from_dict
    # drops it and the checkpoint silently runs the wrong normalization.
    rms_norm_eps: float | None = None
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
    # Upstream Cohere2Moe rotates the dense-prefix layers when this is 1 even
    # though their layer_types entry is full_attention (``force_rope``).  Like
    # rms_norm_eps it must stay a declared field or from_dict drops it; the
    # default matches the reference config class.
    prefix_dense_sliding_window_pattern: int = 1
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


# Bounded, sync-free construction counters.  A norm is built once per layer at
# load time, so counting constructions (not forwards) costs nothing in the hot
# path and still lets an A/B harness refuse an arm whose norm never engaged.
NORM_COUNTERS = {"rms": 0, "layernorm": 0, "legacy_override": 0}


def norm_counters() -> dict[str, int]:
    """Which normalization the loaded North body actually built."""
    return dict(NORM_COUNTERS)


def _norm_layer(args: ModelArgs):
    """Build North's normalization exactly as the reference implementation does.

    Rationale (pinned by tests/test_north_norm_choice.py):

    HF transformers ``modeling_cohere2_moe.py`` builds both the per-layer
    ``input_layernorm`` and the final ``model.norm`` as::

        Cohere2MoeRMSNorm(hidden_size, eps=config.rms_norm_eps)
        if config.rms_norm_eps is not None
        else Cohere2MoeLayerNorm(hidden_size, eps=config.layer_norm_eps)

    i.e. ``rms_norm_eps`` selects the norm *class*, not just an epsilon.
    North-Mini-Code-1.0's ``config.json`` sets ``rms_norm_eps: 1e-06``, so the
    reference runs **RMSNorm with eps 1e-6**.  The ``layer_norm_eps: 1e-05``
    that sits beside it is only the config class's default being serialized
    (``Cohere2MoeConfig.layer_norm_eps = 1e-5``), and is inert for this
    checkpoint.  mlx-vlm's independent cohere2_moe port makes the same choice.

    The two variants differ by mean-centring, and have *identical* parameter
    shapes -- one ``[hidden]`` gamma, no bias, in both cases.  The North
    checkpoint carries exactly 49 ``input_layernorm.weight`` plus one
    ``model.norm.weight``, all ``[2048]``, and no ``*.bias`` anywhere, so a
    wrong choice loads cleanly and is invisible to any shape check.  That is
    why mlx2 ran mean-centred LayerNorm on this model until 2026-09-20.

    ``MLX2_NORTH_NORM=legacy_layernorm`` restores the old (incorrect) norm.  It
    exists only so the qualification A/B can interleave both arms in one
    process; it is not a serving lever and defaults off.
    """
    if os.environ.get("MLX2_NORTH_NORM", "").strip() == "legacy_layernorm":
        NORM_COUNTERS["legacy_override"] += 1
        NORM_COUNTERS["layernorm"] += 1
        return nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps, bias=False)
    if args.rms_norm_eps is not None:
        NORM_COUNTERS["rms"] += 1
        return nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    NORM_COUNTERS["layernorm"] += 1
    return nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps, bias=False)


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
        # The reference applies RoPE to sliding layers and, when the prefix
        # pattern is 1, to the dense-prefix layers as well: North's layer 0 is
        # a full-attention layer with RoPE, not a NoPE layer.  Running it
        # unrotated raised the real 4-bit artifact's perplexity from 8.9 to
        # 50.8 on held-out code (2026-09-23).
        force_rope = (
            layer_idx < args.first_k_dense_replace
            and args.prefix_dense_sliding_window_pattern == 1
        )
        self.rope = (
            nn.RoPE(self.head_dim, traditional=True, base=args.rope_theta)
            if self.is_sliding or force_rope
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
        self.input_layernorm = _norm_layer(args)
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
        self.norm = _norm_layer(args)
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

    def _project(self, hidden):
        logits = (
            self.model.embed_tokens.as_linear(hidden)
            if self.args.tie_word_embeddings
            else self.lm_head(hidden)
        )
        return logits * self.args.logit_scale

    def forward_with_taps(self, inputs, cache, capture_layers, *, body_only=False):
        """Target features for external drafters, paired with consumed tokens.

        Tap ``k < num_hidden_layers`` is the residual after decoder layer ``k``
        (vLLM #49819 / DFlash ``target_layer_ids`` semantics); the sentinel
        ``k == num_hidden_layers`` is the final-norm hidden state an EAGLE-1
        head consumes.  Features concatenate in ascending tap order.  Uses the
        same ``ResidualTaps`` seam as calibration, so a steering vector set by
        the caller applies to this forward too.
        """
        capture_layers = tuple(int(i) for i in capture_layers)
        count = len(self.model.layers)
        if (
            not capture_layers
            or tuple(sorted(set(capture_layers))) != capture_layers
            or capture_layers[0] < 0
            or capture_layers[-1] > count
        ):
            raise ValueError("Invalid target capture layers")
        taps = self.model.residual_taps
        if taps.capture is not None:
            raise RuntimeError("North residual capture is already in use")
        residual = {i: None for i in capture_layers if i < count}
        taps.capture = residual or None
        try:
            hidden = self.model(inputs, cache)
        finally:
            taps.capture = None
        features = mx.concatenate(
            [hidden if i == count else residual[i] for i in capture_layers], axis=-1
        )
        if body_only:
            return None, features
        return self._project(hidden), features

    def prefill_body(self, inputs, cache, capture_layers):
        return self.forward_with_taps(inputs, cache, capture_layers, body_only=True)[1]

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
