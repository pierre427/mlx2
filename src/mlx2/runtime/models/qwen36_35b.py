# SPDX-License-Identifier: MIT
# Mined from local mlx-lm-unified; see provenance/qwen36-35b.json and .NOTICE.
"""Sparse Qwen3.6 35B-A3B text model on the shared hybrid runtime."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from . import moe_nax_gather as _moe_nax
from . import qwen3_next
from . import qwen4_fused_gdn_verify as _verify
from .gdn_state import check_state
from .import_env import snapshot as _import_env_snapshot
from .pipeline import PipelineMixin
from .qwen3_5 import GatedDeltaNet as ReferenceGatedDeltaNet
from .qwen3_5 import TextModelArgs
from .qwen3_next import transform_moe_weights
from .qwen4_fused_gdn import (
    _THREADGROUP_Y_CANDIDATES,
    admit_qwen4_fused_gdn_batch_decode,
    admit_qwen4_fused_gdn_decode,
    fused_gdn_runtime_supported,
    qwen4_fused_gdn_batch_decode,
    qwen4_fused_gdn_decode,
    served_silu_refusal,
)
from .served_exp import is_device_fault
from .qwen36_moe_decode import Qwen36SparseMoeBlock as Qwen3NextSparseMoeBlock
from .qwen38_27b import (
    ModelArgs,
    Qwen3NextAttention,
)
from .qwen38_27b import (
    Qwen3_5TextModel as DenseTextModel,
)
from .qwen38_27b import (
    TextModel as DenseTextWrapper,
)

_import_env_snapshot(__name__)
_FUSED_GDN_DECODE = os.environ.get("MLX_QWEN36_FUSED_GDN_DECODE", "0") == "1"


@lru_cache(maxsize=64)
def probe_qwen36_gdn(dtype, state_dtype, *, verify=False, rows=1, steps=1):
    """Compile the selected 32-head contract, not the sibling's 48-head one.
    Runtime capability is checked before any tensor or kernel construction."""
    if not fused_gdn_runtime_supported():
        return None
    cd, vd, hv = 8192, 4096, 32
    qkv = mx.zeros((rows, steps, cd), dtype)
    z = mx.zeros((rows, steps, vd), dtype)
    gate = mx.zeros((rows, steps, hv), dtype)
    conv = mx.zeros((rows, 3, cd), dtype)
    weight = mx.zeros((cd, 4, 1), dtype)
    state = mx.zeros((rows, hv, 128, 128), state_dtype)
    args = (
        qkv,
        z,
        gate,
        gate,
        conv,
        weight,
        mx.zeros((hv,), mx.float32),
        mx.zeros((hv,), dtype),
        state,
        mx.ones((128,), dtype),
        1e-6,
    )
    fn = (
        (
            _verify.qwen4_fused_gdn_batch_verify
            if rows > 1
            else _verify.qwen4_fused_gdn_verify
        )
        if verify
        else (qwen4_fused_gdn_batch_decode if rows > 1 else qwen4_fused_gdn_decode)
    )
    extra = dict(architecture="qwen35", num_value_heads=32)
    if verify and rows > 1:
        args += ((steps,) * rows,)
    for ty in _THREADGROUP_Y_CANDIDATES:
        try:
            mx.eval(*fn(*args, threadgroup_y=ty, **extra))
            return ty
        except (RuntimeError, ValueError) as exc:
            if is_device_fault(exc):
                # Not a kernel refusal: lru_cache does not cache a raise, so
                # serving recovers and the next call probes again.
                raise
            continue
    return None


class GatedDeltaNet(ReferenceGatedDeltaNet):
    """Qwen3.6 decode specialization; ordinary Qwen3.5 math stays the reference."""

    def __init__(self, args: TextModelArgs):
        super().__init__(args)
        self.fused_gdn_decode_mode = "fused" if _FUSED_GDN_DECODE else "stock"
        for choice in ("batch_decode", "verify", "batch_verify"):
            enabled = (
                os.environ.get("MLX_QWEN36_FUSED_GDN_" + choice.upper(), "0") == "1"
            )
            setattr(
                self, "fused_gdn_" + choice + "_mode", "row_exact" if enabled else "off"
            )
            setattr(self, "fused_gdn_" + choice + "_calls", 0)
            setattr(self, "fused_gdn_" + choice + "_fallbacks", 0)
            setattr(self, "fused_gdn_" + choice + "_last_fallback", None)
            object.__setattr__(self, "fused_gdn_" + choice + "_fallback_reasons", {})
        self.fused_gdn_decode_calls = 0
        self.fused_gdn_decode_fallbacks = 0
        self.fused_gdn_decode_last_fallback = None
        object.__setattr__(self, "fused_gdn_decode_fallback_reasons", {})

    def set_fused_gdn_decode_mode(self, mode: str):
        if mode not in ("stock", "fused"):
            raise ValueError("Qwen3.6 fused GDN mode must be 'stock' or 'fused'")
        self.fused_gdn_decode_mode = mode

    def _set_row_mode(self, choice, mode):
        if mode not in ("off", "row_exact"):
            raise ValueError("Qwen3.6 GDN row mode must be off or row_exact")
        setattr(self, "fused_gdn_" + choice + "_mode", mode)

    def set_fused_gdn_batch_decode_mode(self, mode):
        self._set_row_mode("batch_decode", mode)

    def set_fused_gdn_verify_mode(self, mode):
        self._set_row_mode("verify", mode)

    def set_fused_gdn_batch_verify_mode(self, mode):
        self._set_row_mode("batch_verify", mode)

    def _row_fallback(self, choice, reason):
        prefix = "fused_gdn_" + choice
        setattr(self, prefix + "_fallbacks", getattr(self, prefix + "_fallbacks") + 1)
        setattr(self, prefix + "_last_fallback", reason)
        reasons = getattr(self, prefix + "_fallback_reasons")
        if reason not in reasons and len(reasons) >= 16:
            reason = "other"
        reasons[reason] = reasons.get(reason, 0) + 1
        return None

    def _fallback(self, reason: str):
        self.fused_gdn_decode_fallbacks += 1
        self.fused_gdn_decode_last_fallback = reason
        reasons = self.fused_gdn_decode_fallback_reasons
        reasons[reason] = reasons.get(reason, 0) + 1
        return None

    def _try_fused_decode(self, qkv, z, b, a, mask, cache):
        if cache is not None:
            # A state of the class this layer did not select never runs.
            check_state(cache[1], getattr(self, "_gdn_state_dtype", None))
        rows, steps = qkv.shape[:2]
        if steps > 1 and bool(getattr(cache, "speculating", False)):
            choice = "batch_verify" if rows > 1 else "verify"
            if getattr(self, "fused_gdn_" + choice + "_mode") == "row_exact":
                return self._try_fused_verify(qkv, z, b, a, mask, cache, choice)
            return None
        batch = rows > 1 and steps == 1
        if batch:
            if self.fused_gdn_batch_decode_mode == "off":
                return None
        elif self.fused_gdn_decode_mode == "stock":
            return None
        fallback = (
            (lambda reason: self._row_fallback("batch_decode", reason))
            if batch
            else self._fallback
        )
        if qkv.shape[1] > 1 and not bool(getattr(cache, "speculating", False)):
            # A prefill chunk is not a decode candidate; see Qwen4's gate.
            return None
        if cache is None or cache[0] is None or cache[1] is None:
            return fallback("uninitialized cache")
        describe = getattr(cache, "rollback_spans", None)
        spans = describe(int(qkv.shape[1]), mask) if callable(describe) else ()
        admit = (
            admit_qwen4_fused_gdn_batch_decode
            if batch
            else admit_qwen4_fused_gdn_decode
        )
        admission = admit(
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
            return fallback(admission.reason)
        if not fused_gdn_runtime_supported():
            return fallback("Metal runtime unavailable")
        refusal = served_silu_refusal()
        if refusal is not None:
            return fallback(refusal)
        try:
            threadgroup_y = probe_qwen36_gdn(qkv.dtype, cache[1].dtype, rows=rows)
            if threadgroup_y is None:
                return fallback("Metal kernel probe declined")
            kernel = qwen4_fused_gdn_batch_decode if batch else qwen4_fused_gdn_decode
            (out, conv_state, recurrent_state) = kernel(
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
            if is_device_fault(exc):
                raise
            return fallback(f"Metal kernel dispatch failed: {type(exc).__name__}")
        cache[0] = conv_state
        cache[1] = recurrent_state
        cache.advance(1)
        if batch:
            self.fused_gdn_batch_decode_calls += 1
            self.fused_gdn_batch_decode_last_fallback = None
        else:
            self.fused_gdn_decode_calls += 1
            self.fused_gdn_decode_last_fallback = None
        return self.out_proj(out)

    def _try_fused_verify(self, qkv, z, b, a, mask, cache, choice):
        fallback = lambda reason: self._row_fallback(choice, reason)
        if cache is None or cache[0] is None or cache[1] is None:
            return fallback("uninitialized cache")
        describe = getattr(cache, "rollback_spans", None)
        if not callable(describe) or not callable(
            getattr(cache, "record_rollback", None)
        ):
            return fallback("cache lacks rollback records")
        rows, steps = qkv.shape[:2]
        batch = rows > 1
        spans = describe(steps, mask)
        admit = (
            _verify.admit_qwen4_fused_gdn_batch_verify
            if batch
            else _verify.admit_qwen4_fused_gdn_verify
        )
        admission = admit(
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
            speculating=True,
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
            return fallback(admission.reason)
        if not fused_gdn_runtime_supported():
            return fallback("Metal runtime unavailable")
        refusal = served_silu_refusal()
        if refusal is not None:
            return fallback(refusal)
        counts = (
            _verify.batch_verify_row_steps(spans, mask, rows, steps)
            if batch
            else (steps,)
        )
        initial = [cache[0], cache[1]]
        try:
            ty = probe_qwen36_gdn(
                qkv.dtype, cache[1].dtype, verify=True, rows=rows, steps=steps
            )
            if ty is None:
                return fallback("Metal kernel probe declined")
            fn = (
                _verify.qwen4_fused_gdn_batch_verify
                if batch
                else _verify.qwen4_fused_gdn_verify
            )
            args = (
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
            )
            if batch:
                args += (counts,)
            out, conv, state, states, convs = fn(
                *args, threadgroup_y=ty, architecture="qwen35", num_value_heads=32
            )
        except Exception as exc:
            if is_device_fault(exc):
                raise
            return fallback(f"Metal kernel dispatch failed: {type(exc).__name__}")

        def restore_rows(lengths):
            # Include checkpoint and final state: zero/full/ragged acceptance
            # must never read an unwritten padded snapshot.
            ends = mx.broadcast_to(mx.array(lengths, mx.int32).reshape(-1), (rows,))
            ends = mx.minimum(mx.maximum(ends, 0), mx.array(counts, mx.int32))
            idx = mx.minimum(mx.maximum(ends - 1, 0), steps - 2)

            def pick(snaps, before, final):
                index = idx.reshape((rows, 1) + (1,) * (snaps.ndim - 2))
                selected = mx.take_along_axis(snaps, index, axis=1).squeeze(1)
                full = (ends == mx.array(counts)).reshape(
                    (rows,) + (1,) * (before.ndim - 1)
                )
                zero = (ends == 0).reshape((rows,) + (1,) * (before.ndim - 1))
                return mx.contiguous(
                    mx.where(zero, before, mx.where(full, final, selected))
                )

            return [pick(convs, initial[0], conv), pick(states, initial[1], state)]

        def restore(m):
            return restore_rows(m if isinstance(m, mx.array) else [m] * rows)

        cache.record_rollback(steps, restore, initial, per_row_fn=restore_rows)
        cache[0], cache[1] = conv, state
        cache.advance(steps)
        prefix = "fused_gdn_" + choice
        setattr(self, prefix + "_calls", getattr(self, prefix + "_calls") + 1)
        setattr(self, prefix + "_last_fallback", None)
        return self.out_proj(out)


def qwen36_fused_gdn_stats(model: nn.Module, *, reset: bool = False) -> dict[str, Any]:
    report = {
        "mode": "stock",
        "layers": 0,
        "fused_calls": 0,
        "fallbacks": 0,
        "reasons": {},
    }
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

    def __call__(self, inputs, cache=None, input_embeddings=None, *args, **kwargs):
        # The NAX MoE gather (MLX2_MOE_NAX_GATHER, default off) is
        # prefill-only: the trunk forward decides the phase once, as
        # Flash-Next's does.  Decode, MTP verify (verify scope or a
        # speculating cache) and prepared verify blocks are not prefill; the
        # MTP head runs outside this scope; the invariant prefill lane is
        # excluded at the MoE call sites.  A no-op while the mode is off.
        with _moe_nax.forward_scope(
            inputs if inputs is not None else input_embeddings, cache
        ):
            return super().__call__(
                inputs, cache, input_embeddings, *args, **kwargs
            )


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
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

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


def qwen36_decode_wins_stats(model, *, reset=False):
    report = {
        "gdn": {},
        "moe": {
            "calls": 0,
            "fallbacks": 0,
            "reasons": {},
            "routed_gate_up_calls": 0,
            "topk_launch_calls": 0,
            "window_calls": 0,
            "shared_calls": 0,
        },
    }
    for _, layer in model.named_modules():
        if isinstance(layer, GatedDeltaNet):
            for choice in ("batch_decode", "verify", "batch_verify"):
                prefix = "fused_gdn_" + choice
                cell = report["gdn"].setdefault(
                    choice, {"calls": 0, "fallbacks": 0, "reasons": {}}
                )
                for key in ("calls", "fallbacks"):
                    cell[key] += getattr(layer, prefix + "_" + key)
                    if reset:
                        setattr(layer, prefix + "_" + key, 0)
                reasons = getattr(layer, prefix + "_fallback_reasons")
                for key, value in reasons.items():
                    cell["reasons"][key] = cell["reasons"].get(key, 0) + value
                if reset:
                    reasons.clear()
        if isinstance(layer, Qwen3NextSparseMoeBlock):
            cell = report["moe"]
            cell["calls"] += layer.qwen36_decode_calls
            cell["fallbacks"] += layer.qwen36_decode_fallbacks
            cell["routed_gate_up_calls"] += layer.switch_mlp.routed_decode_calls
            cell["topk_launch_calls"] += layer.moe_topk_calls["launch"]
            cell["window_calls"] += sum(layer.moe_window_calls.values())
            cell["shared_calls"] += layer.shared_fold_calls
            for key, value in layer.qwen36_decode_reasons.items():
                cell["reasons"][key] = cell["reasons"].get(key, 0) + value
            if reset:
                layer.qwen36_decode_calls = layer.qwen36_decode_fallbacks = 0
                layer.qwen36_decode_reasons.clear()
                layer.switch_mlp.routed_decode_calls = 0
                layer.moe_topk_calls = {"launch": 0, "fold": 0}
                layer.moe_window_calls = {key: 0 for key in layer.moe_window_calls}
                layer.shared_fold_calls = 0
    return report
