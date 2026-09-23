# SPDX-License-Identifier: MIT
# Mined from local mlx-lm-unified; see provenance/qwen36-35b.json and .NOTICE.
"""Sparse Qwen3.6 35B-A3B text model on the shared hybrid runtime."""

from __future__ import annotations

import os
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from . import qwen3_next
from .qwen3_5 import GatedDeltaNet as ReferenceGatedDeltaNet, TextModelArgs
from .qwen3_next import Qwen3NextSparseMoeBlock, transform_moe_weights
from .qwen4_fused_gdn import (
    admit_qwen4_fused_gdn_decode,
    fused_gdn_runtime_supported,
    probe_qwen4_fused_gdn_decode,
    qwen4_fused_gdn_decode,
)
from .pipeline import PipelineMixin
from .qwen38_27b import (
    ModelArgs,
    Qwen3NextAttention,
    Qwen3_5TextModel as DenseTextModel,
    TextModel as DenseTextWrapper,
)


_FUSED_GDN_DECODE = os.environ.get("MLX_QWEN36_FUSED_GDN_DECODE", "0") == "1"


class GatedDeltaNet(ReferenceGatedDeltaNet):
    """Qwen3.6 decode specialization; ordinary Qwen3.5 math stays the reference."""

    def __init__(self, args: TextModelArgs):
        super().__init__(args)
        self.fused_gdn_decode_mode = "fused" if _FUSED_GDN_DECODE else "stock"
        self.fused_gdn_decode_calls = 0
        self.fused_gdn_decode_fallbacks = 0
        self.fused_gdn_decode_last_fallback = None
        object.__setattr__(self, "fused_gdn_decode_fallback_reasons", {})

    def set_fused_gdn_decode_mode(self, mode: str):
        if mode not in ("stock", "fused"):
            raise ValueError("Qwen3.6 fused GDN mode must be 'stock' or 'fused'")
        self.fused_gdn_decode_mode = mode

    def _fallback(self, reason: str):
        self.fused_gdn_decode_fallbacks += 1
        self.fused_gdn_decode_last_fallback = reason
        reasons = self.fused_gdn_decode_fallback_reasons
        reasons[reason] = reasons.get(reason, 0) + 1
        return None

    def _try_fused_decode(self, qkv, z, b, a, mask, cache):
        if self.fused_gdn_decode_mode == "stock":
            return None
        if qkv.shape[1] > 1 and not bool(getattr(cache, "speculating", False)):
            # A prefill chunk is not a decode candidate; see Qwen4's gate.
            return None
        if cache is None or cache[0] is None or cache[1] is None:
            return self._fallback("uninitialized cache")
        describe = getattr(cache, "rollback_spans", None)
        spans = describe(int(qkv.shape[1]), mask) if callable(describe) else ()
        admission = admit_qwen4_fused_gdn_decode(
            qkv=qkv,
            z=z,
            b=b,
            a=a,
            conv_state=cache[0],
            recurrent_state=cache[1],
            conv_weight=self.conv1d.weight,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            norm_weight=self.norm.weight,
            mask=mask,
            spans=spans,
            speculating=bool(getattr(cache, "speculating", False)),
            training=bool(self.training),
            sharded=self.sharding_group is not None,
            num_key_heads=self.num_k_heads,
            num_value_heads=self.num_v_heads,
            key_head_dim=self.head_k_dim,
            value_head_dim=self.head_v_dim,
            conv_kernel=self.conv_kernel_size,
            gate_activation="swish",
            architecture="qwen35",
        )
        if not admission.accepted:
            return self._fallback(admission.reason)
        if not fused_gdn_runtime_supported():
            return self._fallback("Metal runtime unavailable")
        try:
            threadgroup_y = probe_qwen4_fused_gdn_decode(qkv.dtype)
            if threadgroup_y is None:
                return self._fallback("Metal kernel probe declined")
            (out, conv_state, recurrent_state) = qwen4_fused_gdn_decode(
                qkv,
                z,
                b,
                a,
                cache[0],
                self.conv1d.weight,
                self.A_log,
                self.dt_bias,
                cache[1],
                self.norm.weight,
                self.norm.eps,
                threadgroup_y=threadgroup_y,
                architecture="qwen35",
                num_key_heads=self.num_k_heads,
                num_value_heads=self.num_v_heads,
                key_head_dim=self.head_k_dim,
                value_head_dim=self.head_v_dim,
                conv_kernel=self.conv_kernel_size,
            )
        except Exception as exc:
            return self._fallback(
                f"Metal kernel dispatch failed: {type(exc).__name__}"
            )
        cache[0] = conv_state
        cache[1] = recurrent_state
        cache.advance(1)
        self.fused_gdn_decode_calls += 1
        self.fused_gdn_decode_last_fallback = None
        return self.out_proj(out)


def qwen36_fused_gdn_stats(model: nn.Module, *, reset: bool = False) -> dict[str, Any]:
    report = {"mode": "stock", "layers": 0, "fused_calls": 0, "fallbacks": 0, "reasons": {}}
    for _, module in model.named_modules():
        if not isinstance(module, GatedDeltaNet):
            continue
        report["layers"] += 1
        if module.fused_gdn_decode_mode == "fused":
            report["mode"] = "fused"
        report["fused_calls"] += module.fused_gdn_decode_calls
        report["fallbacks"] += module.fused_gdn_decode_fallbacks
        for reason, count in module.fused_gdn_decode_fallback_reasons.items():
            report["reasons"][reason] = report["reasons"].get(reason, 0) + count
        if reset:
            module.fused_gdn_decode_calls = 0
            module.fused_gdn_decode_fallbacks = 0
            module.fused_gdn_decode_last_fallback = None
            module.fused_gdn_decode_fallback_reasons.clear()
    return report


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        if args.num_experts <= 0:
            raise ValueError("Qwen3.6 35B-A3B requires sparse expert weights")
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Qwen3NextAttention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = Qwen3NextSparseMoeBlock(args)

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None
    ) -> mx.array:
        if self.is_linear:
            residual = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            residual = self.self_attn(self.input_layernorm(x), mask, cache)
        hidden = x + residual
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Qwen36TextModel(DenseTextModel):
    def __init__(self, args: TextModelArgs):
        PipelineMixin.__init__(self)
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args=args, layer_idx=index)
            for index in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1


class MTPModule(nn.Module):
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


class TextModel(DenseTextWrapper):
    def __init__(self, args: TextModelArgs):
        nn.Module.__init__(self)
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen36TextModel(args)
        self.mtp = MTPModule(args) if args.mtp_num_hidden_layers > 0 else None
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)


class Model(nn.Module):
    apc_v2_layout = "qwen36-35b-a3b-hybrid-layer-segments-v1"
    supports_speculative_rollback = True

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextModelArgs.from_dict(args.text_config))

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None):
        return self.language_model(inputs, cache=cache, input_embeddings=input_embeddings)

    @property
    def model(self):
        return self.language_model.model

    @property
    def mtp(self):
        return self.language_model.mtp

    @property
    def layers(self):
        return self.language_model.model.pipeline_layers

    def make_cache(self):
        return self.language_model.make_cache()

    def logits(self, hidden):
        return self.language_model.logits(hidden)

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    def mtp_step(self, hidden, tokens, mtp_cache):
        return self.language_model.mtp_step(hidden, tokens, mtp_cache)

    @staticmethod
    def shard_prune(weights):
        """Drop, per shard, the tensors ``sanitize`` discards outright.

        This is the key-local subset of ``sanitize``'s first rule, split out
        so a shard-streaming loader can discard the vision tower *before*
        materialising it. ``sanitize`` still applies the same filter to the
        full dict, so it remains authoritative: under-dropping here is merely
        a missed saving, and over-dropping fails loudly in ``load_weights``.
        """
        return {
            key: value
            for key, value in weights.items()
            if not key.startswith(("vision_tower", "model.visual"))
        }

    def sanitize(self, weights):
        normalized = {}
        for key, value in weights.items():
            if key.startswith(("vision_tower", "model.visual")):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif not key.startswith("language_model."):
                key = "language_model." + key
            normalized[key] = value

        args = self.language_model.args
        prefixes = [
            f"language_model.model.layers.{index}.mlp"
            for index in range(args.num_hidden_layers)
        ]
        if self.language_model.mtp is not None:
            prefixes.extend(
                f"language_model.mtp.layers.{index}.mlp"
                for index in range(args.mtp_num_hidden_layers)
            )
        for prefix in prefixes:
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key not in normalized:
                continue
            gate_up = normalized.pop(gate_up_key)
            if qwen3_next._MOE_FUSED_GATE_UP:
                normalized[f"{prefix}.switch_mlp.gate_up_proj.weight"] = gate_up
            else:
                midpoint = gate_up.shape[-2] // 2
                normalized[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[
                    ..., :midpoint, :
                ]
                normalized[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[
                    ..., midpoint:, :
                ]
            normalized[f"{prefix}.switch_mlp.down_proj.weight"] = normalized.pop(
                f"{prefix}.experts.down_proj"
            )
        transform_moe_weights(
            normalized,
            prefixes,
            fuse_gate_up=qwen3_next._MOE_FUSED_GATE_UP,
            fold_shared=qwen3_next._MOE_SHARED_IN_GATHER,
        )
        return self.language_model.sanitize(normalized)

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
