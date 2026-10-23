# SPDX-License-Identifier: Apache-2.0
"""MLX reference implementation of Recursive Flint.

The model-specific numerical law lives here: hybrid full/sliding attention,
per-head QK normalization and learned sink logits, RoPE-free full-attention
layers, sigmoid top-k routing, and an always-active shared expert.  The
released SGLang implementation is the source of truth; see
``provenance/flint.json``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from mlx import nn

from ..prefill_checkpoint_rows import select_checkpoint_rows
from . import moe_nax_gather as _moe_nax
from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    max_position_embeddings: int
    rope_theta: float
    layer_types: list[str]
    no_rope_layer_indices: list[int]
    mlp_only_layers: list[int]
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    sliding_window: int
    tie_word_embeddings: bool
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    norm_topk_prob: bool = True
    router_score_function: str = "sigmoid"
    softmax_type: str = "learnable"

    def __post_init__(self):
        if self.model_type != "flint":
            raise ValueError("Flint model_type must be 'flint'")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("Flint layer_types length differs from num_hidden_layers")


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, *, use_rope: bool):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim**-0.5
        self.use_rope = use_rope
        self.q_proj = nn.Linear(
            args.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=False
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        # The checkpoint calls these learned sink logits ``softmax_offset``.
        self.softmax_offset = mx.zeros((self.n_heads,))
        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=None,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        if self.use_rope:
            offset = cache.offset if cache is not None else 0
            q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        y = scaled_dot_product_attention(
            q,
            k,
            v,
            cache=cache,
            scale=self.scale,
            mask=mask,
            sinks=self.softmax_offset.astype(q.dtype),
        )
        return self.o_proj(y.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class SparseMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)
        # One-token routed experts from one-expert views (flint_routed_decode):
        # "off" (stock gathers), "gate_up" or "gate_up_down".  The adapter
        # selects it; the kernels prove themselves bitwise against the stock
        # gathers on the first call, else every call stays stock.
        self.routed_decode = "off"
        self.routed_decode_calls = 0
        self.routed_decode_fallbacks = 0
        self.routed_decode_last_fallback = None

    def _routed_rows(self, x, indices):
        if self.routed_decode == "off" or x.size != x.shape[-1]:
            return None
        from . import flint_routed_decode as FR

        reason = FR.admission(x, self.switch_mlp)
        rows = None if reason else FR.routed_rows(x, indices, self.switch_mlp, self.routed_decode)
        if rows is None:
            self.routed_decode_fallbacks += 1
            self.routed_decode_last_fallback = reason or FR.status()["verdict"]
            if reason is not None and reason not in FR.TRANSIENT_REFUSALS:
                # The expert tables (or the runtime) can never admit it: stop
                # re-checking on every token; the stock gathers serve.
                self.routed_decode = "off"
            return None
        self.routed_decode_calls += 1
        return rows.reshape(*indices.shape, -1)

    def __call__(self, x):
        # SGLang routes in fp32: bias-free sigmoid scores choose the experts,
        # and the chosen raw sigmoid weights are renormalized to sum to one.
        scores = mx.sigmoid(self.gate(x).astype(mx.float32))
        indices = mx.argpartition(scores, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        expert_rows = self._routed_rows(x, indices)
        if expert_rows is None:
            expert_rows = self.switch_mlp(x, indices)
        routed = mx.sum(expert_rows * weights[..., None], axis=-2)
        return routed.astype(x.dtype) + self.shared_expert(x)


class Layer(nn.Module):
    def __init__(self, args: ModelArgs, index: int):
        super().__init__()
        self.layer_type = args.layer_types[index]
        self.self_attn = Attention(
            args, use_rope=index not in frozenset(args.no_rope_layer_indices)
        )
        self.mlp = (
            MLP(args.hidden_size, args.intermediate_size)
            if index in frozenset(args.mlp_only_layers)
            else SparseMoE(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None):
        hidden = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Decoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Layer(args, index) for index in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._full = args.layer_types.index("full_attention")
        self._sliding = args.layer_types.index("sliding_attention")

    def __call__(self, inputs, cache=None, capture=None, *, decode=False):
        # The M5 NAX sorted expert gather (bit-identical to the stock sorted
        # kernel) is prefill-only: the trunk forward decides the phase once.
        # A no-op while its mode is off.
        with _moe_nax.forward_scope(inputs, cache):
            return self._forward(inputs, cache, capture, decode=decode)

    def _forward(self, inputs, cache, capture, *, decode=False):
        hidden = self.embed_tokens(inputs)
        fused = getattr(self, "fused_blocks", None)
        if fused and inputs.shape[0] != 1:
            raise ValueError("Flint fused decode candidate requires B1")
        if fused and decode and inputs.shape[1] == 1 and not _moe_nax.prefill_active():
            if capture is not None or cache is None or any(c is None for c in cache):
                raise ValueError("Flint fused decode requires ordinary cache state and no taps")
            offset = 0
            for block in fused:
                hidden = block(hidden, cache[offset:offset+len(block.layers)])
                offset += len(block.layers)
            return self.norm(hidden)
        if cache is None:
            cache = [None] * len(self.layers)
        full_mask = create_attention_mask(hidden, cache[self._full])
        sliding_mask = create_attention_mask(
            hidden, cache[self._sliding], window_size=self.args.sliding_window
        )
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache, strict=True)):
            mask = full_mask if layer.layer_type == "full_attention" else sliding_mask
            hidden = layer(hidden, mask, layer_cache)
            if capture is not None and index in capture:
                # Post-block residual stream, before the final norm.
                capture[index] = hidden
        return self.norm(hidden)


class Model(nn.Module):
    supports_speculative_rollback = True
    exact_prefix_reuse_algorithm = "flint-full-swa-segmented-kv-v1"

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Decoder(args)
        self.completed_forwards = 0
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        # The serving prompt path calls prefill_forward explicitly. A bare
        # cold call also remains ordinary; populated-cache calls are decode.
        decode = (
            bool(getattr(self.model, "fused_blocks", None))
            and inputs.shape[0] == 1 and bool(cache)
            and all(c is not None and c.offset > 0 for c in cache)
        )
        hidden = select_checkpoint_rows(self.model(inputs, cache, decode=decode))
        logits = self._head(hidden)
        self.completed_forwards += 1
        return logits

    def prefill_forward(self, inputs, cache=None):
        """Prompt chunks, including one-token tails, use reference arithmetic."""
        with _moe_nax.prefill_scope(True):
            return self(inputs, cache=cache)

    @property
    def layers(self):
        return self.model.layers

    @property
    def speculative_args(self):
        return self.args

    def _head(self, hidden):
        return (
            self.model.embed_tokens.as_linear(hidden)
            if self.args.tie_word_embeddings
            else self.lm_head(hidden)
        )

    def forward_with_taps(self, inputs, cache, capture_layers, *, body_only=False):
        """Logits plus ascending post-block taps for external/tree drafting."""
        capture_layers = tuple(int(index) for index in capture_layers)
        if (
            not capture_layers
            or tuple(sorted(set(capture_layers))) != capture_layers
            or capture_layers[0] < 0
            or capture_layers[-1] >= len(self.model.layers)
        ):
            raise ValueError("Invalid target capture layers")
        taps = {index: None for index in capture_layers}
        hidden = self.model(inputs, cache, capture=taps)
        features = mx.concatenate([taps[index] for index in capture_layers], axis=-1)
        if body_only:
            return None, features
        logits = self._head(hidden)
        self.completed_forwards += 1
        return logits, features

    def prefill_body(self, inputs, cache, capture_layers):
        return self.forward_with_taps(inputs, cache, capture_layers, body_only=True)[1]

    def sanitize(self, weights, *, eager=False):
        """Pack 256 per-expert tensors into the native SwitchGLU layout."""
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        for layer in range(self.args.num_hidden_layers):
            if layer in self.args.mlp_only_layers:
                continue
            prefix = f"model.layers.{layer}.mlp"
            packed = []
            for projection in ("gate_proj", "up_proj", "down_proj"):
                names = [
                    f"{prefix}.experts.{expert}.{projection}.weight"
                    for expert in range(self.args.num_experts)
                ]
                present = [name in weights for name in names]
                if not any(present):
                    continue
                if not all(present):
                    raise ValueError(
                        f"Flint {prefix} has an incomplete {projection} expert table"
                    )
                parts = [weights.pop(name) for name in names]
                value = mx.stack(parts)
                from ..weight_stream import record_stack

                record_stack(value, parts)
                weights[f"{prefix}.switch_mlp.{projection}.weight"] = value
                packed.append(value)
                # A converted artifact may keep one quantized record per HF
                # expert so conversion remains bounded by one source tensor.
                # Stack its affine metadata in the same expert order as the
                # packed weights; mixed or incomplete records fail closed.
                for suffix in ("scales", "biases"):
                    metadata_names = [
                        name.removesuffix(".weight") + f".{suffix}" for name in names
                    ]
                    present_metadata = [name in weights for name in metadata_names]
                    if any(present_metadata) and not all(present_metadata):
                        raise ValueError(
                            f"Flint {prefix} has incomplete expert {suffix}"
                        )
                    if all(present_metadata):
                        metadata_parts = [weights.pop(name) for name in metadata_names]
                        metadata = mx.stack(metadata_parts)
                        record_stack(metadata, metadata_parts)
                        weights[f"{prefix}.switch_mlp.{projection}.{suffix}"] = metadata
                        packed.append(metadata)
            if eager and packed:
                mx.eval(packed)
                mx.clear_cache()
        return weights

    def make_cache(self, max_kv_size=None):
        result = []
        for kind in self.args.layer_types:
            if kind == "sliding_attention":
                result.append(RotatingKVCache(max_size=self.args.sliding_window))
            elif kind == "full_attention":
                result.append(
                    RotatingKVCache(max_size=max_kv_size) if max_kv_size else KVCache()
                )
            else:
                raise ValueError(f"unsupported Flint attention kind: {kind}")
        return result

    def exact_prefix_reuse_contract(self, cache):
        from numbers import Integral

        if not isinstance(cache, (list, tuple)) or len(cache) != len(
            self.args.layer_types
        ):
            raise ValueError("Flint exact prefix reuse needs one cache per layer")
        offsets = set()
        full_geometries = set()
        full = sliding = 0
        for index, (kind, entry) in enumerate(
            zip(self.args.layer_types, cache, strict=True)
        ):
            if kind == "sliding_attention":
                if (
                    type(entry) is not RotatingKVCache
                    or entry.max_size != self.args.sliding_window
                    or entry.keep != 0
                ):
                    raise ValueError(
                        f"Flint sliding layer {index} has incompatible cache geometry"
                    )
                sliding += 1
            elif kind == "full_attention":
                if type(entry) not in (KVCache, RotatingKVCache):
                    raise ValueError(
                        f"Flint full layer {index} has incompatible cache geometry"
                    )
                if type(entry) is KVCache:
                    full_geometries.add(("unbounded", None, 0))
                elif entry.keep != 0 or entry.max_size <= 0:
                    raise ValueError(
                        f"Flint full layer {index} has incompatible rotating geometry"
                    )
                else:
                    full_geometries.add(("rotating", entry.max_size, entry.keep))
                full += 1
            else:
                raise ValueError(f"unsupported Flint attention kind: {kind}")
            if (
                isinstance(entry.offset, bool)
                or not isinstance(entry.offset, Integral)
                or entry.offset < 0
                or getattr(entry, "speculating", False)
            ):
                raise ValueError("Flint exact prefix reuse needs idle host offsets")
            offsets.add(int(entry.offset))
        if len(offsets) != 1:
            raise ValueError("Flint cache layers disagree on the committed position")
        if len(full_geometries) != 1:
            raise ValueError("Flint full-attention caches disagree on their geometry")
        full_kind, full_max_size, full_keep = full_geometries.pop()
        return {
            "algorithm": self.exact_prefix_reuse_algorithm,
            "full_layers": full,
            "full_cache": {
                "kind": full_kind,
                "max_size": full_max_size,
                "keep": full_keep,
            },
            "sliding_layers": sliding,
            "sliding_window": self.args.sliding_window,
            "committed_offset": offsets.pop(),
            "transaction": "SegmentedKVRows exact rotating-window snapshot/replay",
        }

    @property
    def quant_predicate(self):
        def predicate(path, module):
            # Keep the routing choice and learned attention law in source
            # precision.  The conversion profile may independently protect
            # embeddings, heads, attention, dense and shared projections.
            return not path.endswith((".gate", ".q_norm", ".k_norm"))

        return predicate
