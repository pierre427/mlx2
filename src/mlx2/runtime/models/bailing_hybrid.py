# SPDX-License-Identifier: MIT
"""MLX model adapter for inclusionAI/Ling-3.0-flash.

Mined from TensorFold/Ling-3.0-flash-oQ4 at the revision recorded in
``provenance/ling3-ubtree.json`` (MIT).  Imports are mapped onto mlx2's
self-contained runtime and target taps are added for the external drafter.
"""

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx import nn

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_update
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU
from .xing4_0 import MultiLinear


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int
    n_group: int
    topk_group: int
    first_k_dense_replace: int
    layer_group_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    routed_scaling_factor: float
    head_dim: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    short_conv_kernel_size: int = 4
    moe_shared_expert_intermediate_size: int | None = None
    q_lora_rank: int | None = None
    rope_interleave: bool = True
    rope_scaling: dict[str, float | str] | None = None
    use_qkv_bias: bool = False
    use_bias: bool = False
    norm_topk_prob: bool = True
    score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = True
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = 0
    no_kda_lora: bool = False
    kda_safe_gate: bool = False
    kda_lower_bound: float | None = None
    gated_attention_proj_granularity_type: str | None = None


def _cache_offset(cache: Any | None) -> int | mx.array:
    offset = cache.offset if cache is not None else 0
    if isinstance(offset, mx.array):
        if offset.size == 1:
            return offset.item()
        return mx.array(offset.tolist(), dtype=offset.dtype)
    return offset


class MLP(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        intermediate_size: int | None = None,
    ):
        super().__init__()
        hidden = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(args.hidden_size, hidden, bias=args.use_bias)
        self.up_proj = nn.Linear(args.hidden_size, hidden, bias=args.use_bias)
        self.down_proj = nn.Linear(hidden, args.hidden_size, bias=args.use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MultiLatentAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.q_lora_rank = args.q_lora_rank
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.v_head_dim = args.v_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.gate_type = args.gated_attention_proj_granularity_type
        self.scale = self.qk_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                args.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = nn.Linear(
                args.hidden_size,
                self.q_lora_rank,
                bias=args.use_qkv_bias,
            )
            self.q_a_layernorm = nn.RMSNorm(
                self.q_lora_rank,
                eps=args.rms_norm_eps,
            )
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=args.use_qkv_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(
            self.kv_lora_rank,
            eps=args.rms_norm_eps,
        )
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim,
            self.kv_lora_rank,
            self.num_heads,
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank,
            self.v_head_dim,
            self.num_heads,
        )

        if self.gate_type is None:
            self.g_proj = None
        elif self.gate_type == "head_wise":
            self.g_proj = nn.Linear(args.hidden_size, self.num_heads, bias=False)
        elif self.gate_type == "element_wise":
            self.g_proj = nn.Linear(
                args.hidden_size,
                self.num_heads * self.v_head_dim,
                bias=False,
            )
        else:
            raise ValueError(f"Unsupported attention gate type: {self.gate_type}")

        self.dense = nn.Linear(
            self.num_heads * self.v_head_dim,
            args.hidden_size,
            bias=args.use_qkv_bias,
        )

        if args.rope_scaling is not None:
            mscale_all_dim = args.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = args.rope_scaling.get("factor", 1)
            if mscale_all_dim and scaling_factor > 1:
                mscale = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                self.scale *= mscale * mscale

        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=args.rope_interleave,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        batch, length, _ = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(batch, length, self.num_heads, self.qk_head_dim)
        q = q.transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(
            compressed_kv,
            [self.kv_lora_rank],
            axis=-1,
        )
        k_pe = k_pe.reshape(batch, length, 1, self.qk_rope_head_dim)
        k_pe = k_pe.transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = _cache_offset(cache)
        q_pe = self.rope(q_pe, offset=offset)
        k_pe = self.rope(k_pe, offset=offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)
        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if length == 1:
            q_nope = self.embed_q(q_nope)
            keys = values = kv_latent
        else:
            keys = self.embed_q(kv_latent, transpose=False)
            values = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=pe_scores,
        )
        if length == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3)
        if self.g_proj is not None:
            gate = mx.sigmoid(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
            if self.gate_type == "head_wise":
                output = output * gate[..., None]
            else:
                output = output * gate.reshape(
                    batch,
                    length,
                    self.num_heads,
                    self.v_head_dim,
                )

        output = output.reshape(batch, length, -1)
        return self.dense(output)


class ShortConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            bias=False,
            groups=channels,
            padding=0,
        )

    def __call__(
        self,
        x: mx.array,
        state: mx.array | None,
        mask: mx.array | None,
        lengths: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)

        if state is None:
            state = mx.zeros(
                (x.shape[0], self.kernel_size - 1, x.shape[-1]),
                dtype=x.dtype,
            )
        conv_input = mx.concatenate([state, x], axis=1)
        output = nn.silu(self.conv(conv_input))
        n_keep = self.kernel_size - 1

        if lengths is not None:
            ends = mx.clip(lengths, 0, x.shape[1])
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            new_state = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            new_state = mx.contiguous(conv_input[:, -n_keep:, :])

        return output, new_state


class KimiDeltaAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.conv_kernel = args.short_conv_kernel_size
        self.projection_dim = self.num_heads * self.head_dim
        self.no_kda_lora = args.no_kda_lora
        self.scale = float(self.head_dim) ** -0.5

        hidden = args.hidden_size
        self.q_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.q_conv1d = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.k_conv1d = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.v_conv1d = ShortConv1d(self.projection_dim, self.conv_kernel)

        if self.no_kda_lora:
            self.f_proj = nn.Linear(hidden, self.projection_dim, bias=False)
            self.g_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        else:
            self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
            self.f_b_proj = nn.Linear(
                self.head_dim,
                self.projection_dim,
                bias=False,
            )
            self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(
                self.head_dim,
                self.projection_dim,
                bias=False,
            )

        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.A_log = mx.log(
            mx.random.uniform(low=1.0, high=16.0, shape=(self.num_heads,))
        )
        self.dt_bias = mx.zeros((self.projection_dim,))
        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: ArraysCache | None = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        dtype = x.dtype

        if cache is not None:
            q_state, k_state, v_state, recurrent_state = cache
            lengths = cache.lengths
        else:
            q_state = k_state = v_state = recurrent_state = None
            lengths = None

        if q_state is None:
            state = mx.zeros(
                (batch, self.conv_kernel - 1, self.projection_dim),
                dtype=dtype,
            )
            q_state = k_state = v_state = state

        q, q_state = self.q_conv1d(self.q_proj(x), q_state, mask, lengths)
        k, k_state = self.k_conv1d(self.k_proj(x), k_state, mask, lengths)
        v, v_state = self.v_conv1d(self.v_proj(x), v_state, mask, lengths)

        if cache is not None:
            cache[0] = q_state
            cache[1] = k_state
            cache[2] = v_state

        q = q.reshape(batch, length, self.num_heads, self.head_dim)
        k = k.reshape(batch, length, self.num_heads, self.head_dim)
        v = v.reshape(batch, length, self.num_heads, self.head_dim)

        q = (self.scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = self.scale * mx.fast.rms_norm(k, None, 1e-6)

        if self.no_kda_lora:
            decay_logits = self.f_proj(x)
        else:
            decay_logits = self.f_b_proj(self.f_a_proj(x))
        decay_logits = decay_logits.reshape(
            batch,
            length,
            self.num_heads,
            self.head_dim,
        )
        beta_logits = self.b_proj(x).reshape(batch, length, self.num_heads)

        output, recurrent_state = gated_delta_update(
            q,
            k,
            v,
            decay_logits,
            beta_logits,
            self.A_log.reshape(self.num_heads, 1),
            self.dt_bias.reshape(self.num_heads, self.head_dim),
            state=recurrent_state,
            mask=mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[3] = recurrent_state
            cache.advance(length)

        if self.no_kda_lora:
            gate = self.g_proj(x)
        else:
            gate = self.g_b_proj(self.g_a_proj(x))
        gate = gate.reshape(batch, length, self.num_heads, self.head_dim)
        output = self.o_norm(output) * mx.sigmoid(gate)
        return self.o_proj(output.reshape(batch, length, -1))


@mx.compile
def group_expert_select(
    gates: mx.array,
    expert_bias: mx.array | None,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    score_function: str,
) -> tuple[mx.array, mx.array]:
    input_type = gates.dtype
    if score_function == "sigmoid":
        scores = mx.sigmoid(gates.astype(mx.float32))
    elif score_function == "softmax":
        scores = mx.softmax(gates.astype(mx.float32), axis=-1, precise=True)
    else:
        raise ValueError(f"Unsupported MoE score function: {score_function}")

    original_scores = scores
    if expert_bias is not None:
        scores = scores + expert_bias

    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        n_drop = n_group - topk_group
        group_idx = mx.argpartition(
            group_scores,
            kth=n_drop - 1,
            axis=-2,
        )[..., :n_drop, :]
        scores = mx.put_along_axis(
            scores,
            mx.stop_gradient(group_idx),
            mx.array(0.0, dtype=scores.dtype),
            axis=-2,
        )
        scores = mx.flatten(scores, -2, -1)

    indices = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(original_scores, indices, axis=-1)
    if top_k > 1 and norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return indices, weights.astype(input_type)


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling_factor = args.routed_scaling_factor
        self.score_function = args.score_function
        self.gate_proj = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = (
            mx.zeros((args.num_experts,), dtype=mx.float32)
            if args.moe_router_enable_expert_bias
            else None
        )

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        return group_expert_select(
            self.gate_proj(x),
            self.expert_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
            self.score_function,
        )


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            bias=args.use_bias,
        )
        self.gate = Gate(args)
        shared_size = (
            args.moe_shared_expert_intermediate_size or args.moe_intermediate_size
        ) * args.num_shared_experts
        self.shared_experts = (
            MLP(args, intermediate_size=shared_size)
            if args.num_shared_experts > 0
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        indices, weights = self.gate(x)
        output = self.switch_mlp(x, indices)
        output = (output * weights[..., None]).sum(axis=-2)
        if self.shared_experts is not None:
            output = output + self.shared_experts(x)
        return output


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        n_layers = args.num_hidden_layers
        group_size = args.layer_group_size
        self.is_global = (layer_idx + 1) % group_size == 0 or layer_idx >= (
            n_layers // group_size
        ) * group_size
        self.attention = (
            MultiLatentAttention(args) if self.is_global else KimiDeltaAttention(args)
        )
        self.mlp = (
            SparseMoeBlock(args)
            if args.num_experts is not None and layer_idx >= args.first_k_dense_replace
            else MLP(args)
        )
        self.input_layernorm = nn.RMSNorm(
            args.hidden_size,
            eps=args.rms_norm_eps,
        )
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size,
            eps=args.rms_norm_eps,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        attention_output = self.attention(
            self.input_layernorm(x),
            mask,
            cache,
        )
        hidden = x + attention_output
        mlp_output = self.mlp(self.post_attention_layernorm(hidden))
        return hidden + mlp_output


class LanguageModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx) for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._attention_idx = next(
            (i for i, layer in enumerate(self.layers) if layer.is_global),
            0,
        )
        self._kda_idx = next(
            (i for i, layer in enumerate(self.layers) if not layer.is_global),
            0,
        )

    @property
    def embed_tokens(self):
        """Common target-embedding seam used by external drafters."""

        return self.word_embeddings

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
    ) -> mx.array:
        hidden = self.word_embeddings(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        attention_mask = create_attention_mask(
            hidden,
            cache[self._attention_idx],
            return_array=True,
        )
        kda_mask = create_ssm_mask(hidden, cache[self._kda_idx])

        for layer, layer_cache in zip(self.layers, cache):
            mask = attention_mask if layer.is_global else kda_mask
            hidden = layer(hidden, mask=mask, cache=layer_cache)
        return self.norm(hidden)


class Model(nn.Module):
    # ArraysCache record/trim plus KVCache trim are the generic
    # HybridVerifyRows exact transaction contract.  This capability is owned
    # by the cache topology, independent of the Ling adapter or route name.
    supports_speculative_rollback = True

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LanguageModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.hidden_size,
                args.vocab_size,
                bias=False,
            )

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
    ) -> mx.array:
        output = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.word_embeddings.as_linear(output)
        return self.lm_head(output)

    @property
    def layers(self):
        return self.model.layers

    @property
    def speculative_args(self):
        return self.args

    def forward_with_taps(
        self,
        inputs,
        cache,
        capture_layers,
        *,
        body_only=False,
        last_logits_only=False,
    ):
        """Return logits and concatenated post-layer residual-stream taps."""

        capture_layers = tuple(int(value) for value in capture_layers)
        if (
            not capture_layers
            or tuple(sorted(set(capture_layers))) != capture_layers
            or capture_layers[0] < 0
            or capture_layers[-1] >= len(self.layers)
        ):
            raise ValueError("invalid Ling target capture layers")
        if body_only and last_logits_only:
            raise ValueError("body_only and last_logits_only are mutually exclusive")
        hidden = self.model.word_embeddings(inputs)
        cache = [None] * len(self.layers) if cache is None else cache
        attention_mask = create_attention_mask(
            hidden, cache[self.model._attention_idx], return_array=True
        )
        kda_mask = create_ssm_mask(hidden, cache[self.model._kda_idx])
        wanted = frozenset(capture_layers)
        taps = []
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            mask = attention_mask if layer.is_global else kda_mask
            hidden = layer(hidden, mask=mask, cache=layer_cache)
            if index in wanted:
                taps.append(hidden)
        if len(taps) != len(capture_layers):
            raise RuntimeError("Ling target tap count does not match capture layers")
        features = mx.concatenate(taps, axis=-1)
        hidden = self.model.norm(hidden)
        if body_only:
            return None, features
        projected = hidden[:, -1:] if last_logits_only else hidden
        logits = (
            self.model.word_embeddings.as_linear(projected)
            if self.args.tie_word_embeddings
            else self.lm_head(projected)
        )
        return logits, features

    def prefill_body(self, inputs, cache, capture_layers):
        return self.forward_with_taps(inputs, cache, capture_layers, body_only=True)[1]

    def make_cache(self) -> list[Any]:
        return [
            KVCache() if layer.is_global else ArraysCache(size=4)
            for layer in self.layers
        ]

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        n_layers = self.args.num_hidden_layers
        weights = {
            key: value
            for key, value in weights.items()
            if not (
                key.startswith("model.layers.")
                and key.split(".")[2].isdigit()
                and int(key.split(".")[2]) >= n_layers
            )
        }

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for layer_idx, layer in enumerate(self.layers):
            prefix = f"model.layers.{layer_idx}"

            if isinstance(layer.mlp, SparseMoeBlock):
                for projection in ("gate_proj", "down_proj", "up_proj"):
                    for suffix in ("weight", "scales", "biases"):
                        first = f"{prefix}.mlp.experts.0.{projection}.{suffix}"
                        if first in weights:
                            stacked = [
                                weights.pop(
                                    f"{prefix}.mlp.experts.{expert}.{projection}.{suffix}"
                                )
                                for expert in range(self.args.num_experts)
                            ]
                            weights[
                                f"{prefix}.mlp.switch_mlp.{projection}.{suffix}"
                            ] = mx.stack(stacked)

                gate_key = f"{prefix}.mlp.gate.weight"
                if gate_key in weights:
                    weights[f"{prefix}.mlp.gate.gate_proj.weight"] = weights.pop(
                        gate_key
                    )

            if layer.is_global:
                kv_b_key = f"{prefix}.attention.kv_b_proj.weight"
                if kv_b_key in weights:
                    value = weights.pop(kv_b_key)
                    combined = self.args.qk_nope_head_dim + self.args.v_head_dim
                    value = value.reshape(
                        self.args.num_attention_heads,
                        combined,
                        -1,
                    )
                    key_weight = mx.contiguous(
                        value[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2)
                    )
                    value_weight = mx.contiguous(
                        value[:, self.args.qk_nope_head_dim :, :]
                    )
                    weights[f"{prefix}.attention.embed_q.weight"] = key_weight
                    weights[f"{prefix}.attention.unembed_out.weight"] = value_weight
            else:
                for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                    source = f"{prefix}.attention.{name}.weight"
                    if source in weights:
                        value = weights.pop(source)
                        if value.ndim == 3:
                            value = value.moveaxis(2, 1)
                        weights[f"{prefix}.attention.{name}.conv.weight"] = value

        return weights

    @property
    def quant_predicate(self):
        def predicate(path: str, _module: nn.Module):
            if path.endswith("mlp.gate.gate_proj"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if path.endswith(("A_log", "dt_bias")):
                return False
            return "expert_bias" not in path

        return predicate
