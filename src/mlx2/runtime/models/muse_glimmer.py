# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see provenance/muse-glimmer.json and .NOTICE.
"""Muse Glimmer text tower using mlx2 attention and cache contracts."""

import mlx.core as mx
from mlx import nn

from ...adapters.muse_glimmer_config import ModelArgs
from .base import create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache

TARGET_VERIFY_ROW_EXACT_VERSION = "muse-q4-native-b1s1-causal-prefix-v1"


def _row_exact_project(module, value, *, call=None):
    """Project every logical row with ordinary one-row reduction order."""
    from .row_exact_qmv import quantized_linear

    return quantized_linear(module, value, call=call)[0]


def _row_exact_call(call, value):
    """Apply a last-axis operation with the ordinary one-row outer shape."""
    from .row_exact_qmv import per_row

    return per_row(call, value)


def _row_exact_group(modules, value):
    """Same-input projections with one-row bits, grouped when Metal admits it."""
    from .row_exact_qmv import quantized_linears

    return quantized_linears(modules, value)[0]


class CenteredRMSNorm(nn.Module):
    """RMSNorm with a zero-centered scale: out = norm(x) * (1 + weight)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # HF and mlx-vlm apply the centered scale in fp32 and cast back; in
        # bf16, 1 + w rounds to steps of 2^-7 and changed decode choices.
        scale = 1.0 + self.weight.astype(mx.float32)
        return mx.fast.rms_norm(x.astype(mx.float32), scale, self.eps).astype(x.dtype)


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
        self.window = args.sliding_window
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

    def _row_exact_rope(self, q, k, cache):
        if not self.use_rope:
            return q, k
        if hasattr(cache, "rows"):
            offsets = [int(row.offset) for row in cache.rows]
            array_offsets = True
        else:
            if q.shape[0] != 1 or not isinstance(cache.offset, int):
                raise ValueError("Muse row-exact RoPE requires independent B1 caches")
            offsets = [cache.offset]
            array_offsets = False
        q_batches, k_batches = [], []
        for batch, offset in enumerate(offsets):
            q_rows, k_rows = [], []
            for position in range(q.shape[2]):
                current = offset + position
                rope_offset = (
                    mx.array([current], dtype=mx.int32) if array_offsets else current
                )
                q_rows.append(
                    self.rope(
                        q[batch : batch + 1, :, position : position + 1],
                        offset=rope_offset,
                    )
                )
                k_rows.append(
                    self.rope(
                        k[batch : batch + 1, :, position : position + 1],
                        offset=rope_offset,
                    )
                )
            q_batches.append(mx.concatenate(q_rows, axis=2))
            k_batches.append(mx.concatenate(k_rows, axis=2))
        return mx.concatenate(q_batches), mx.concatenate(k_batches)

    def _row_exact_attention(self, q, k, v, cache):
        segmented = getattr(cache, "row_exact_attention", None)
        if callable(segmented):
            return segmented(
                q,
                k,
                v,
                scale=self.scale,
                window_size=self.window if self.is_sliding else None,
            )
        if q.shape[0] != 1:
            raise ValueError("Muse row-exact attention requires independent B1 caches")
        outputs = []
        for position in range(q.shape[2]):
            row_mask = cache.make_mask(
                1,
                window_size=self.window if self.is_sliding else None,
                return_array=True,
            )
            keys, values = cache.update_and_fetch(
                k[..., position : position + 1, :],
                v[..., position : position + 1, :],
            )
            outputs.append(
                scaled_dot_product_attention(
                    q[..., position : position + 1, :],
                    keys,
                    values,
                    cache=cache,
                    scale=self.scale,
                    mask=row_mask,
                )
            )
        return mx.concatenate(outputs, axis=2)

    def __call__(
        self, x: mx.array, mask=None, cache=None, *, row_exact=False
    ) -> mx.array:
        B, L, _ = x.shape
        if row_exact:
            q, k, v, gate = _row_exact_group(
                (self.q_proj, self.k_proj, self.v_proj, self.gate_proj), x
            )
        else:
            q, k, v, gate = (
                self.q_proj(x),
                self.k_proj(x),
                self.v_proj(x),
                self.gate_proj(x),
            )
        q = q.reshape(B, L, self.n_heads, self.head_dim)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim)

        # Scaleless QK-norm over head_dim; Q additionally scaled.
        def norm(value):
            return mx.fast.rms_norm(value, None, self.eps)
        q = (
            (_row_exact_call(norm, q) if row_exact else norm(q)).astype(mx.float32)
            * self.qk_scale_factor
        ).astype(q.dtype)
        k = _row_exact_call(norm, k) if row_exact else norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if row_exact:
            q, k = self._row_exact_rope(q, k, cache)
        elif self.use_rope:
            offset = cache.offset if cache is not None else 0
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)

        if row_exact:
            out = self._row_exact_attention(q, k, v, cache)
        else:
            if cache is not None:
                k, v = cache.update_and_fetch(k, v)

            out = scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scale, mask=mask
            )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)

        # Glimmer's output gate.
        out = out * mx.sigmoid(gate)
        return _row_exact_project(self.o_proj, out) if row_exact else self.o_proj(out)


def lane_projection_groups():
    """Lane groups: ``Attention.__call__`` feeds its one ``x`` to each member.

    The output gate reads the attention input, not the attention output, so
    it stacks with q/k/v (tests/test_lane_projection_groups.py pins the call
    site).  The MLP's gate/up stay on the default group.
    """
    from ..lane import ProjectionGroup

    return (
        ProjectionGroup(
            "muse-attn-qkv-gate", Attention, ("q_proj", "k_proj", "v_proj", "gate_proj")
        ),
    )


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array, *, row_exact=False) -> mx.array:
        if not row_exact:
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
        gate, up = _row_exact_group((self.gate_proj, self.up_proj), x)
        return _row_exact_project(self.down_proj, nn.silu(gate) * up)


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

    def __call__(
        self, x: mx.array, mask=None, cache=None, *, row_exact=False
    ) -> mx.array:
        residual = x
        h = (
            _row_exact_call(self.input_layernorm, x)
            if row_exact
            else self.input_layernorm(x)
        )
        h = self.self_attn(h, mask, cache, row_exact=row_exact)
        h = (
            _row_exact_call(self.post_attention_layernorm, h)
            if row_exact
            else self.post_attention_layernorm(h)
        )
        h = residual + h

        residual = h
        h = (
            _row_exact_call(self.pre_feedforward_layernorm, h)
            if row_exact
            else self.pre_feedforward_layernorm(h)
        )
        h = self.mlp(h, row_exact=row_exact)
        h = (
            _row_exact_call(self.post_feedforward_layernorm, h)
            if row_exact
            else self.post_feedforward_layernorm(h)
        )
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

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        capture_layers=(),
        hidden_sink=None,
        *,
        row_exact=False,
    ):
        h = self.embed_tokens(inputs)
        # Scaleless RMSNorm on the embeddings (not Gemma's sqrt scaling).
        def embed_norm(value):
            return mx.fast.rms_norm(value, None, self.embed_eps)
        h = _row_exact_call(embed_norm, h) if row_exact else embed_norm(h)

        if cache is None:
            cache = [None] * len(self.layers)

        if len(cache) != len(self.layers):
            raise ValueError("Muse cache layer count mismatch")
        masks = self._masks(h, cache)
        for index, (layer, c, mask) in enumerate(zip(self.layers, cache, masks)):
            h = layer(h, mask, c, row_exact=row_exact)
            if hidden_sink is not None and index in capture_layers:
                hidden_sink.append(h)
        return _row_exact_call(self.norm, h) if row_exact else self.norm(h)


class Model(nn.Module):
    supports_trusted_pld = True

    @property
    def apc_v2_layout(self):
        return self.args.cache_layout

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MuseGlimmerModel(args)
        self._target_verify_row_exact = False
        self._row_exact_forwards = 0
        self._row_exact_query_rows = 0
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

    def forward_with_taps(
        self,
        inputs,
        cache,
        capture_layers,
        *,
        body_only=False,
        last_logits_only=False,
    ):
        """Post-block target taps in ascending layer order, before final norm.

        Body-only prefill avoids the expensive vocabulary projection. Tap
        tensors remain paired with exactly the cache tokens consumed here.
        """
        capture_layers = tuple(capture_layers)
        if not capture_layers or tuple(sorted(set(capture_layers))) != capture_layers or capture_layers[-1] >= len(self.layers) or capture_layers[0] < 0:
            raise ValueError("Invalid target capture layers")
        taps = []
        if body_only and last_logits_only:
            raise ValueError("body-only forward cannot request last-row logits")
        row_exact = self._target_verify_row_exact and not body_only
        hidden = self.model(
            inputs,
            cache=cache,
            capture_layers=capture_layers,
            hidden_sink=taps,
            row_exact=row_exact,
        )
        features = mx.concatenate(taps, axis=-1)
        if body_only:
            return None, features
        projected = hidden[:, -1:] if last_logits_only else hidden
        head = self.model.embed_tokens if self.tie_word_embeddings else self.lm_head
        if row_exact:
            logits = _row_exact_project(
                head,
                projected,
                call=head.as_linear if self.tie_word_embeddings else None,
            )
        else:
            logits = (
                head.as_linear(projected)
                if self.tie_word_embeddings
                else head(projected)
            )
        logits = logits * self.args.output_multiplier
        cap = self.args.final_logit_softcapping
        logits = mx.tanh(logits / cap) * cap
        if row_exact:
            mx.eval(logits, features)
            self._row_exact_forwards += 1
            self._row_exact_query_rows += inputs.shape[0] * inputs.shape[1]
        return logits, features

    def configure_target_verify_row_exact(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("target_verify_row_exact must be a boolean")
        self._target_verify_row_exact = enabled

    @property
    def supports_contextual_prefix_equivalence(self):
        return bool(self._target_verify_row_exact)

    @property
    def progressive_external_verify_protocol(self):
        if not self._target_verify_row_exact:
            return None
        from ..progressive_external_verify import (
            PROGRESSIVE_TARGET_PROTOCOL_VERSION,
        )

        return PROGRESSIVE_TARGET_PROTOCOL_VERSION

    @property
    def external_execution_receipt(self):
        if not self._target_verify_row_exact:
            return None
        return {
            "target_verify_row_exact": {
                "algorithm": TARGET_VERIFY_ROW_EXACT_VERSION,
                "implemented": True,
                "qualified": False,
                "selected": True,
                "performance_claim": False,
                "observed_used": self._row_exact_forwards > 0,
                "executed_forwards": self._row_exact_forwards,
                "physical_query_rows": self._row_exact_query_rows,
            }
        }

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

    @staticmethod
    def lora_module_key(key):
        """Resolve checkpoint LoRA names to the loaded text model's modules."""
        if key.startswith("language_model.model."):
            return "model." + key[len("language_model.model.") :]
        if key.startswith("language_model.lm_head."):
            return "lm_head." + key[len("language_model.lm_head.") :]
        if key.startswith("model.language_model."):
            return "model." + key[len("model.language_model.") :]
        return key

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
