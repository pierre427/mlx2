# SPDX-License-Identifier: MIT
# Mined from local mlx-lm-unified; provenance/qwen38-27b.json and .NOTICE.
"""Dense Qwen3.8 27B text and embedded MTP model; no qualified routes yet."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import mlx.core as mx
import mlx.nn as nn
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .pipeline import PipelineMixin
from .qwen3_5 import TextModelArgs, GatedDeltaNet
from .qwen3_next import Qwen3NextMLP as MLP
from .precise_ops import gate_sigmoid
from .rope_utils import initialize_rope


class Qwen3NextAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** (-0.5)
        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None
    ) -> mx.array:
        (B, L, D) = x.shape
        q_proj_output = self.q_proj(x)
        (queries, gate) = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)
        (keys, values) = (self.k_proj(x), self.v_proj(x))
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            (keys, values) = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * gate_sigmoid(gate))


Attention = Qwen3NextAttention


def _apply_deep_concept_memory(hidden_states: mx.array, memory: dict) -> mx.array:
    """Attend from the final prompt state into request-scoped concept memory.

    The operation changes no sequence or cache geometry.  It is intentionally
    bounded to the final token of an isolated prefill; later decoder layers
    consume the amended state and write ordinary Qwen cache planes.
    """
    keys = memory.get("keys")
    values = memory.get("values")
    temperature = memory.get("temperature")
    gate = memory.get("gate")
    hidden = hidden_states.shape[-1]
    if not isinstance(keys, mx.array) or not isinstance(values, mx.array):
        raise ValueError("deep concept memory requires device key/value arrays")
    if keys.ndim != 2 or values.shape != keys.shape or keys.shape[1] != hidden:
        raise ValueError("deep concept memory does not match Qwen hidden geometry")
    if not 1 <= keys.shape[0] <= 32:
        raise ValueError("deep concept memory requires 1..32 concepts")
    if not isinstance(temperature, (int, float)) or not 0.001 <= float(temperature) <= 1.0:
        raise ValueError("deep concept memory temperature is out of bounds")
    if not isinstance(gate, (int, float)) or not 0.0 <= float(gate) <= 1.0:
        raise ValueError("deep concept memory gate is out of bounds")
    query = hidden_states[:, -1:, :].astype(mx.float32)
    query_norm = mx.maximum(mx.linalg.norm(query, axis=-1, keepdims=True), 1e-6)
    query = query / query_norm
    keys = keys.astype(mx.float32)
    values = values.astype(mx.float32)
    keys = keys / mx.maximum(mx.linalg.norm(keys, axis=-1, keepdims=True), 1e-6)
    values = values / mx.maximum(mx.linalg.norm(values, axis=-1, keepdims=True), 1e-6)
    weights = mx.softmax((query @ keys.T) / float(temperature), axis=-1)
    # ``gate`` is a relative hidden-state norm, not an absolute embedding
    # delta.  This keeps the bridge meaningful at different depths while
    # bounding it to at most one current-state norm.
    residual = (float(gate) * query_norm * (weights @ values)).astype(
        hidden_states.dtype
    )
    return mx.concatenate(
        [hidden_states[:, :-1, :], hidden_states[:, -1:, :] + residual], axis=1
    )


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        if args.num_experts > 0:
            raise ValueError("Qwen3.8 27B adapter requires a dense model")
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen3_5TextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def pipeline(self, group):
        super().pipeline(group)
        self.ssm_idx = None
        self.fa_idx = None
        for e, l in enumerate(self.pipeline_layers):
            if self.ssm_idx is None and l.is_linear:
                self.ssm_idx = e
            elif self.fa_idx is None and (not l.is_linear):
                self.fa_idx = e
            if self.ssm_idx is not None and self.fa_idx is not None:
                break

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        deep_concept_memory: Optional[dict] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)
        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        if cache is None:
            cache = [None] * len(self.pipeline_layers)
        fa_mask = None
        ssm_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        if self.ssm_idx is not None:
            ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])
        if pipeline_rank < pipeline_size - 1:
            hidden_states = mx.distributed.recv_like(hidden_states, pipeline_rank + 1)
        injection_layer = None
        if deep_concept_memory is not None:
            if pipeline_size != 1:
                raise ValueError("deep concept memory is not qualified with pipeline parallelism")
            if hidden_states.shape[0] != 1:
                raise ValueError("deep concept memory requires an isolated B=1 prefill")
            injection_layer = deep_concept_memory.get("layer")
            if isinstance(injection_layer, bool) or not isinstance(injection_layer, int):
                raise ValueError("deep concept memory layer must be an integer")
            if not 0 <= injection_layer < len(self.pipeline_layers):
                raise ValueError("deep concept memory layer is outside the Qwen trunk")
        for layer_index, (layer, c) in enumerate(zip(self.pipeline_layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)
            if layer_index == injection_layer:
                hidden_states = _apply_deep_concept_memory(
                    hidden_states, deep_concept_memory
                )
        if pipeline_rank != 0:
            hidden_states = mx.distributed.send(
                hidden_states, (pipeline_rank - 1) % pipeline_size
            )
            if cache[-1] is not None:
                if hasattr(cache[-1], "keys"):
                    cache[-1].keys = mx.depends(cache[-1].keys, hidden_states)
                else:
                    cache[-1][0] = mx.depends(cache[-1][0], hidden_states)
        if pipeline_size > 1:
            hidden_states = mx.distributed.all_gather(hidden_states)[
                : hidden_states.shape[0]
            ]
        return self.norm(hidden_states)


class MTPModule(nn.Module):
    """Qwen3.5 multi-token-prediction head, present in "-mtp" checkpoints:
    fc([norm(embed(t_{p+1})); norm(hidden_p)]) -> full-attention decoder
    layer (own KV cache) -> norm -> the target's own lm_head. Enables
    self-speculative decoding with no external draft model."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.fc = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layers = [
            DecoderLayer(args, layer_idx=args.full_attention_interval - 1)
            for _ in range(args.mtp_num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.mtp = None
        self.model = Qwen3_5TextModel(args)
        if args.mtp_num_hidden_layers > 0:
            self.mtp = MTPModule(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        deep_concept_memory: Optional[dict] = None,
    ) -> mx.array:
        out = self.model(
            inputs,
            cache,
            input_embeddings=input_embeddings,
            deep_concept_memory=deep_concept_memory,
        )
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.pipeline_layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def logits(self, hidden: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)

    def make_mtp_cache(self):
        return [KVCache() for _ in self.mtp.layers]

    def mtp_step(self, hidden, tokens, mtp_cache):
        """One MTP forward over S positions.

        hidden: [B, S, H] post-final-norm hiddens at positions p..p+S-1
        (from the trunk, or from a previous mtp_step when chaining draft
        depth). tokens: [B, S] the tokens at positions p+1..p+S (the
        committed or drafted token FOLLOWING each hidden's position).
        Returns (logits [B, S, V], post_norm_hidden [B, S, H]).

        The MTP KV cache offset counts pairs fed, i.e. rope positions are
        uniformly shifted by -1 vs absolute; the shift cancels in q·k
        since the MTP layer only attends within its own cache.
        """
        e = self.mtp.pre_fc_norm_embedding(self.model.embed_tokens(tokens))
        h = self.mtp.pre_fc_norm_hidden(hidden)
        x = self.mtp.fc(mx.concatenate([e, h], axis=-1))
        mask = create_attention_mask(x, mtp_cache[0])
        x = self.mtp.layers[0](x, mask=mask, cache=mtp_cache[0])
        post = self.mtp.norm(x)
        return (self.logits(post), post)

    def sanitize(self, weights):
        has_unsanitized_conv1d = any(
            ("conv1d.weight" in k and v.shape[-1] != 1 for (k, v) in weights.items())
        )
        should_shift_norm_weights = has_unsanitized_conv1d
        has_mtp_weights = any(("mtp." in k for k in weights))
        # ``mtp`` is always an attribute (None when the model was built
        # without a head), so test its value: an "-mtp" checkpoint loaded
        # into a head-less trunk must drop the head's tensors.
        if not (has_mtp_weights and getattr(self, "mtp", None) is not None):
            weights = {k: v for (k, v) in weights.items() if "mtp." not in k}
            self.mtp = None
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            "mtp.norm.weight",
            ".pre_fc_norm_embedding.weight",
            ".pre_fc_norm_hidden.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if has_unsanitized_conv1d and any((k.endswith(sfx) for sfx in norm_keys)):
                if v.ndim == 1:
                    weights[k] = v + 1.0
        return weights

    @property
    def quant_predicate(self):
        if self.args.num_experts <= 0:
            return None

        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):

        def predicate(path: str):
            if path.endswith("A_log"):
                return False
            return True

        return predicate


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    apc_v2_layout = "qwen38-27b-hybrid-layer-segments-v1"
    supports_speculative_rollback = True

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextModelArgs.from_dict(args.text_config))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        deep_concept_memory: Optional[dict] = None,
    ):
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            deep_concept_memory=deep_concept_memory,
        )

    @property
    def model(self):
        return self.language_model.model

    @property
    def mtp(self):
        return self.language_model.mtp

    def logits(self, hidden: mx.array) -> mx.array:
        return self.language_model.logits(hidden)

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    def mtp_step(self, hidden, tokens, mtp_cache):
        return self.language_model.mtp_step(hidden, tokens, mtp_cache)

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif key.startswith("language_model."):
                pass
            else:
                key = "language_model." + key
            sanitized[key] = value
        return self.language_model.sanitize(sanitized)

    @property
    def layers(self):
        return self.language_model.model.pipeline_layers

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
