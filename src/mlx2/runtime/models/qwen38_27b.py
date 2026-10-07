# SPDX-License-Identifier: MIT
# Mined from local mlx-lm-unified; provenance/qwen38-27b.json and .NOTICE.
"""Dense Qwen3.8 27B text and embedded MTP model; no qualified routes yet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .. import round_levers as _lv
from . import invariant_prefill as _invariant
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .pipeline import PipelineMixin
from .qwen3_5 import TextModelArgs
from .qwen38_fused_gdn import GatedDeltaNet
from .qwen3_next import Qwen3NextMLP as MLP
from .precise_ops import gate_sigmoid
from .rope_utils import initialize_rope
from ..ragged_verify_observation import current_observer, observed_stage


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
        rows = B * L
        observer = current_observer()
        if observer is None:
            output = self._attend(
                self.q_proj(x), self.k_proj(x), self.v_proj(x), B, L, mask, cache
            )
            return self.o_proj(output)
        with observed_stage("attention.qkv_projections", rows=rows) as materialize:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            materialize(q, k, v)
        with observed_stage("attention.read_eval", rows=rows) as materialize:
            output = self._attend(q, k, v, B, L, mask, cache)
            materialize(output)
        with observed_stage("attention.out_projection", rows=rows) as materialize:
            output = self.o_proj(output)
            materialize(output)
        return output

    def _attend(self, q_proj_output, keys, values, B, L, mask, cache):
        """Everything between the input and output projections (gated output).

        Shared by ``__call__`` and :meth:`mixed`.
        """
        (queries, gate) = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)
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
        lane_refusal = (
            _invariant.sdpa_refusal(queries, keys, values, mask)
            if _invariant.active()
            else "off"
        )
        if lane_refusal is None:
            output = _invariant.sdpa(queries, keys, values, scale=self.scale, mask=mask)
        else:
            if lane_refusal != "off":
                _invariant.not_invariant("sdpa_" + lane_refusal)
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return output * gate_sigmoid(gate)

    def mixed(self, x, parts):
        """Packed projections, per-segment RoPE/cache/attention, packed o_proj.

        See ``GatedDeltaNet.mixed`` for ``parts``.  Each segment keeps its own
        cache, positions and mask; no variable-length kernel is needed because
        MLX's own attention kernels run once per segment.
        """
        observer = current_observer()
        packed_rows = x.shape[0] * x.shape[1]
        if observer is None:
            q_all, k_all, v_all = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        else:
            with observed_stage(
                "attention.qkv_projections", rows=packed_rows
            ) as materialize:
                q_all, k_all, v_all = self.q_proj(x), self.k_proj(x), self.v_proj(x)
                materialize(q_all, k_all, v_all)
        outs = []
        for rows, length, start, cache, mask in parts:
            span = slice(start, start + rows * length)
            if observer is None:
                out = self._attend(
                    q_all[:, span].reshape(rows, length, -1),
                    k_all[:, span].reshape(rows, length, -1),
                    v_all[:, span].reshape(rows, length, -1),
                    rows, length, mask, cache,
                )
            else:
                with observed_stage(
                    "attention.read_eval", rows=rows * length
                ) as materialize:
                    out = self._attend(
                        q_all[:, span].reshape(rows, length, -1),
                        k_all[:, span].reshape(rows, length, -1),
                        v_all[:, span].reshape(rows, length, -1),
                        rows, length, mask, cache,
                    )
                    materialize(out)
            outs.append(out.reshape(1, rows * length, -1))
        joined = mx.concatenate(outs, axis=1)
        if observer is None:
            return self.o_proj(joined)
        with observed_stage(
            "attention.out_projection", rows=packed_rows
        ) as materialize:
            output = self.o_proj(joined)
            materialize(output)
        return output


Attention = Qwen3NextAttention


def _deep_memory_components(memory: dict) -> tuple[dict, ...]:
    components = memory.get("components")
    if components is None:
        return (memory,)
    if not isinstance(components, (list, tuple)) or not components:
        raise ValueError("deep concept memory components must be a nonempty ordered list")
    if any(not isinstance(component, dict) or "components" in component for component in components):
        raise ValueError("deep concept memory components must be flat mappings")
    return tuple(components)


def _validate_deep_concept_component(
    hidden_states: mx.array, memory: dict, *, layer_count: int
) -> int:
    """Validate all device geometry before any decoder layer can write cache."""
    if hidden_states.shape[0] != 1:
        raise ValueError("deep concept memory requires an isolated B=1 prefill")
    injection_layer = memory.get("layer")
    if isinstance(injection_layer, bool) or not isinstance(injection_layer, int):
        raise ValueError("deep concept memory layer must be an integer")
    if not 0 <= injection_layer < layer_count:
        raise ValueError("deep concept memory layer is outside the Qwen trunk")
    operation = memory.get("operation", "cross_attention_memory")
    if operation not in {
        "directional_residual",
        "continuous_prefix",
        "cross_attention_memory",
    }:
        raise ValueError("unsupported deep concept memory operation")
    values = memory.get("values")
    gate = memory.get("gate")
    hidden = hidden_states.shape[-1]
    if not isinstance(values, mx.array) or values.ndim != 2 or values.shape[1] != hidden:
        raise ValueError("deep concept memory values do not match Qwen hidden geometry")
    if not 1 <= values.shape[0] <= 32:
        raise ValueError("deep concept memory requires 1..32 values")
    if not isinstance(gate, (int, float)) or not 0.0 <= float(gate) <= 1.0:
        raise ValueError("deep concept memory gate is out of bounds")
    if operation == "directional_residual":
        if values.shape[0] != 1:
            raise ValueError("directional residual requires exactly one value")
        return injection_layer
    keys = memory.get("keys")
    temperature = memory.get("temperature")
    if not isinstance(keys, mx.array) or keys.shape != values.shape:
        raise ValueError("deep concept memory keys do not match its values")
    if not isinstance(temperature, (int, float)) or not 0.001 <= float(temperature) <= 1.0:
        raise ValueError("deep concept memory temperature is out of bounds")
    order_bias = memory.get("order_bias")
    if order_bias is not None and (
        not isinstance(order_bias, mx.array)
        or order_bias.ndim != 1
        or order_bias.shape[0] != keys.shape[0]
    ):
        raise ValueError("deep concept memory order bias geometry mismatch")
    return injection_layer


def _apply_deep_concept_memory(hidden_states: mx.array, memory: dict) -> mx.array:
    """Attend from the final prompt state into request-scoped concept memory.

    The operation changes no sequence or cache geometry.  It is intentionally
    bounded to the final token of an isolated prefill; later decoder layers
    consume the amended state and write ordinary Qwen cache planes.
    """
    operation = memory.get("operation", "cross_attention_memory")
    gate = float(memory["gate"])
    # A disabled learned gate is an exact object-level bypass.  Validation was
    # already completed before the layer loop, so this does not hide a stale
    # capsule binding or malformed tensor.
    if gate == 0.0:
        return hidden_states
    keys = memory.get("keys")
    values = memory.get("values")
    temperature = memory.get("temperature")
    query = hidden_states[:, -1:, :].astype(mx.float32)
    query_norm = mx.maximum(mx.linalg.norm(query, axis=-1, keepdims=True), 1e-6)
    values = values.astype(mx.float32)
    values = values / mx.maximum(mx.linalg.norm(values, axis=-1, keepdims=True), 1e-6)
    if operation == "directional_residual":
        residual_direction = values[None, :1, :]
    else:
        query = query / query_norm
        keys = keys.astype(mx.float32)
        keys = keys / mx.maximum(mx.linalg.norm(keys, axis=-1, keepdims=True), 1e-6)
        logits = (query @ keys.T) / float(temperature)
        order_bias = memory.get("order_bias")
        if order_bias is not None:
            logits = logits + order_bias.astype(mx.float32)[None, None, :]
        weights = mx.softmax(logits, axis=-1)
        residual_direction = weights @ values
    # ``gate`` is a relative hidden-state norm, not an absolute embedding
    # delta.  This keeps the bridge meaningful at different depths while
    # bounding it to at most one current-state norm.
    residual = (gate * query_norm * residual_direction).astype(
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

    def mixed(self, x: mx.array, parts) -> mx.array:
        """``__call__`` over a packed multi-segment stream (see ``Qwen3_5TextModel.mixed``)."""
        mixer = self.linear_attn if self.is_linear else self.self_attn
        h = x + mixer.mixed(self.input_layernorm(x), parts)
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen3_5TextModel(PipelineMixin, nn.Module):
    # Per-layer eager dispatch (MTPLX #579; same mechanism as Flash-Next's
    # ``MLX_QWEN4_EAGER_DISPATCH``): on forwards of at most ``max_rows`` rows,
    # ``mx.async_eval`` the residual every ``stride`` layers so the GPU starts
    # while the host still builds the later layers.  Graph and kernels are
    # unchanged.  0 = off; selected only through ``set_eager_dispatch``
    # (adapter execution policy ``eager_dispatch_stride``).
    eager_dispatch_stride = 0
    eager_dispatch_max_rows = 64

    def set_eager_dispatch(self, stride: int, max_rows: int = 64) -> None:
        if type(stride) is not int or stride < 0:
            raise ValueError("eager_dispatch_stride must be a non-negative integer")
        if type(max_rows) is not int or max_rows < 1:
            raise ValueError("eager_dispatch_max_rows must be a positive integer")
        self.eager_dispatch_stride = stride
        self.eager_dispatch_max_rows = max_rows

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
        capture_layers=(),
        hidden_sink=None,
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
        injection_layers = {}
        if deep_concept_memory is not None:
            if pipeline_size != 1:
                raise ValueError("deep concept memory is not qualified with pipeline parallelism")
            # Validate every component before the first decoder layer.  Some
            # layers write attention or recurrent cache state, so late
            # validation would leave an invalid request partially committed.
            for component in _deep_memory_components(deep_concept_memory):
                layer = _validate_deep_concept_component(
                    hidden_states, component, layer_count=len(self.pipeline_layers)
                )
                injection_layers.setdefault(layer, []).append(component)
        stride = self.eager_dispatch_stride if pipeline_size == 1 else 0
        if stride:
            if hidden_states.shape[0] * hidden_states.shape[1] <= self.eager_dispatch_max_rows:
                _lv.bump("eager_dispatch_forwards")
            else:
                _lv.bump("eager_dispatch_row_declines")
                stride = 0
        last = len(self.pipeline_layers) - 1
        for layer_index, (layer, c) in enumerate(zip(self.pipeline_layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)
            if stride and (layer_index == last or (layer_index + 1) % stride == 0):
                mx.async_eval(hidden_states)
                _lv.bump("eager_async_evals")
            for component in injection_layers.get(layer_index, ()):
                hidden_states = _apply_deep_concept_memory(hidden_states, component)
            if hidden_sink is not None and layer_index in capture_layers:
                # Post-block residual stream, before the final norm: the
                # tap an external block drafter conditions on.
                hidden_sink.append(hidden_states)
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


def _mixed_plan(trunk, segments):
    """Validate ``segments`` and lay them out in one packed stream."""
    if trunk.pipeline_size != 1:
        raise ValueError("mixed forward is not qualified with pipeline parallelism")
    if not segments:
        raise ValueError("mixed forward needs at least one segment")
    n_layers = len(trunk.layers)
    embeds, plan, start = [], [], 0
    for tokens, caches in segments:
        if not isinstance(tokens, mx.array) or tokens.ndim != 2 or tokens.size == 0:
            raise ValueError("each mixed segment needs a nonempty [rows, length] token array")
        if caches is None or len(caches) != n_layers:
            raise ValueError("each mixed segment needs one cache per layer")
        if any(bool(getattr(c, "speculating", False)) for c in caches):
            raise ValueError("mixed forward cannot run inside a speculative transaction")
        (rows, length) = tokens.shape
        e = trunk.embed_tokens(tokens)
        plan.append((
            rows, length, start, caches,
            create_attention_mask(e, caches[trunk.fa_idx]),
            create_ssm_mask(e, caches[trunk.ssm_idx]),
        ))
        embeds.append(e.reshape(1, rows * length, -1))
        start += rows * length
    return mx.concatenate(embeds, axis=1), plan


def _trunk_mixed(trunk, segments):
    """``Qwen3_5TextModel.__call__`` over several segments in one forward.

    Per-token work (embedding, norms, every projection, the MLP) runs once on
    the packed stream, so a decode row costs a fraction of a percent of a
    prefill slice's forward instead of its own full weight read; attention
    and the gated-delta recurrence run per segment against that segment's
    cache.  On the served 27B the mixed forward costs what the slice alone
    costs when the packed width is 64-row aligned (docs/audits/
    2026-09-30-basics-sweep/prefill-contention).  Returns the final-normed
    hidden states per segment as ``[rows, length, D]``.
    """
    (x, plan) = _mixed_plan(trunk, segments)
    for index, layer in enumerate(trunk.layers):
        parts = [
            (rows, length, start, caches[index], ssm_mask if layer.is_linear else fa_mask)
            for (rows, length, start, caches, fa_mask, ssm_mask) in plan
        ]
        x = layer.mixed(x, parts)
    x = trunk.norm(x)
    return [
        x[:, start : start + rows * length].reshape(rows, length, -1)
        for (rows, length, start, *_rest) in plan
    ]


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

    def forward_with_taps(
        self,
        inputs,
        cache,
        capture_layers,
        *,
        body_only=False,
        last_logits_only=False,
    ):
        """Logits plus post-block target taps in ascending layer order.

        Taps are the residual stream after each listed decoder layer, before
        the final norm, concatenated on the feature axis.  ``body_only``
        skips the vocabulary projection (prefill).  External draft routes
        only; the ordinary and self-MTP paths never call this.
        """
        capture_layers = tuple(int(value) for value in capture_layers)
        if (
            not capture_layers
            or tuple(sorted(set(capture_layers))) != capture_layers
            or capture_layers[0] < 0
            or capture_layers[-1] >= len(self.model.layers)
        ):
            raise ValueError("Invalid target capture layers")
        if self.model.pipeline_size != 1:
            raise ValueError("target taps are not implemented under pipeline parallelism")
        taps = []
        hidden = self.model(
            inputs, cache, capture_layers=frozenset(capture_layers), hidden_sink=taps
        )
        if len(taps) != len(capture_layers):
            raise RuntimeError("target tap count does not match capture layers")
        features = mx.concatenate(taps, axis=-1)
        if body_only and last_logits_only:
            raise ValueError("body_only and last_logits_only are mutually exclusive")
        if body_only:
            return None, features
        projected = hidden[:, -1:] if last_logits_only else hidden
        return self.logits(projected), features

    def prefill_body(self, inputs, cache, capture_layers):
        return self.forward_with_taps(inputs, cache, capture_layers, body_only=True)[1]

    @property
    def speculative_args(self):
        return self.args

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def mixed_forward(self, segments):
        """One forward over prefill slices and decode cohorts; see ``_trunk_mixed``.

        ``segments`` is a sequence of ``(tokens [rows, length], caches)``.
        Logits are left to the caller (``self.logits`` on the rows it needs),
        as the ordinary prefill leaves them unevaluated.
        """
        return _trunk_mixed(self.model, segments)

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
        # The +1 fold is decided per group: the layout trigger decides (and
        # the pooled check verifies) the trunk, and every MTP-head norm is
        # decided on its own evidence, because converters such as oQ skip the
        # fold on some head norms only. Ambiguity raises (see norm_repair).
        from ...adapters.norm_repair import resolve_norm_convention

        self.norm_convention = resolve_norm_convention(
            weights, fold_suffixes=norm_keys, trunk_raw=should_shift_norm_weights
        )
        # After the fold (done in the stored float32), round float32 norm
        # gammas to the compute dtype: a float32 gamma makes rms_norm return
        # float32 and promotes the residual stream and KV cache (CRACK
        # conversions). GDN A_log/dt_bias/linear_attn.norm do not promote and
        # stay as stored. See dtype_normalize.
        from .dtype_normalize import normalize_norm_dtypes, resolve_compute_dtype

        self.compute_dtype = resolve_compute_dtype(
            getattr(self.args, "dtype", None), weights
        )
        # A dict attribute would join the nn.Module parameter tree.
        object.__setattr__(
            self,
            "dtype_normalization",
            normalize_norm_dtypes(weights, norm_keys, self.compute_dtype),
        )
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
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
    supports_trusted_pld = True

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

    def mixed_forward(self, segments):
        return self.language_model.mixed_forward(segments)

    def prefill_row_context(self, lengths, *, width):
        """Adapter-installed live-row scope for ordinary padded prefill."""
        from .varlen_dense_mlp import prefill_row_context

        return prefill_row_context(self, lengths, width=width)

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

    @property
    def speculative_args(self):
        """Text geometry for external draft executors (hidden size, heads)."""
        return self.language_model.args

    def forward_with_taps(
        self,
        inputs,
        cache,
        capture_layers,
        *,
        body_only=False,
        last_logits_only=False,
    ):
        return self.language_model.forward_with_taps(
            inputs,
            cache,
            capture_layers,
            body_only=body_only,
            last_logits_only=last_logits_only,
        )

    def prefill_body(self, inputs, cache, capture_layers):
        return self.forward_with_taps(inputs, cache, capture_layers, body_only=True)[1]

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
