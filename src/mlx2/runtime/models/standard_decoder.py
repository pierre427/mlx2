# SPDX-License-Identifier: MIT
"""Ordinary full-attention decoder math for Qwen3, Qwen3 MoE, Qwen2 and Llama.

Mined from the pinned unified implementations; see provenance/standard-decoder.json.
No speculative or model-name-dependent scheduler behavior lives here.
"""

import os
from dataclasses import dataclass

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU

TARGET_VERIFY_ROW_EXACT_VERSION = "dense-qwen3-native-b1s1-causal-prefix-v1"


def _project(module, value, row_exact=False):
    """Keep the original projector and its native B1/S1 reduction geometry."""
    if not row_exact:
        return module(value)
    return mx.concatenate(
        [
            mx.concatenate(
                [module(value[b : b + 1, s : s + 1]) for s in range(value.shape[1])],
                axis=1,
            )
            for b in range(value.shape[0])
        ],
        axis=0,
    )


def _prefix_attention(q, k, v, cache, scale, mask):
    """Each valid query sees precisely the ordinary S1 causal key prefix."""

    def attend(query, keys, values):
        prior = keys.shape[2] - query.shape[2]
        if prior < 0:
            raise ValueError("row-exact attention has a shorter key prefix")
        return mx.concatenate(
            [
                mx.fast.scaled_dot_product_attention(
                    query[:, :, j : j + 1],
                    keys[:, :, : prior + j + 1],
                    values[:, :, : prior + j + 1],
                    scale=scale,
                    mask=None,
                )
                for j in range(query.shape[2])
            ],
            axis=2,
        )

    if hasattr(cache, "row_views"):
        outputs = []
        for index, valid, keys, values, _ in cache.row_views(mask):
            if valid:
                value = attend(q[index : index + 1, :, :valid], keys, values)
                if valid < q.shape[2]:
                    value = mx.pad(
                        value, [(0, 0), (0, 0), (0, q.shape[2] - valid), (0, 0)]
                    )
            else:
                value = mx.zeros((1, q.shape[1], q.shape[2], q.shape[3]), dtype=q.dtype)
            outputs.append(value)
        cache.note_attention()
    else:
        outputs = [
            attend(q[b : b + 1], k[b : b + 1], v[b : b + 1]) for b in range(q.shape[0])
        ]
    return mx.concatenate(outputs, axis=0)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    tie_word_embeddings: bool
    head_dim: int | None = None
    rope_scaling: dict | None = None
    rope_traditional: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    mlp_only_layers: list[int] | None = None
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = False

    def __post_init__(self):
        if self.model_type not in {"qwen3", "qwen3_moe", "qwen2", "llama"}:
            raise ValueError("unsupported standard decoder family")
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.mlp_only_layers is None:
            self.mlp_only_layers = []


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim, heads, kv_heads, head_dim = (
            args.hidden_size,
            args.num_attention_heads,
            args.num_key_value_heads,
            args.head_dim,
        )
        self.n_heads, self.n_kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.scale = head_dim**-0.5
        qkv_bias = args.model_type == "qwen2" or args.attention_bias
        self.q_proj = nn.Linear(dim, heads * head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, kv_heads * head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, kv_heads * head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(
            heads * head_dim,
            dim,
            bias=args.attention_bias and args.model_type == "llama",
        )
        if args.model_type in {"qwen3", "qwen3_moe"}:
            self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=args.rope_traditional,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(self, x, mask=None, cache=None, *, row_exact=False):
        batch, length, _ = x.shape
        q = _project(self.q_proj, x, row_exact).reshape(
            batch, length, self.n_heads, self.head_dim
        )
        k = _project(self.k_proj, x, row_exact).reshape(
            batch, length, self.n_kv_heads, self.head_dim
        )
        v = _project(self.v_proj, x, row_exact).reshape(
            batch, length, self.n_kv_heads, self.head_dim
        )
        if hasattr(self, "q_norm"):
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = (z.transpose(0, 2, 1, 3) for z in (q, k, v))
        offset = cache.offset if cache is not None else 0
        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        y = (
            _prefix_attention(q, k, v, cache, self.scale, mask)
            if row_exact
            else scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scale, mask=mask
            )
        )
        return _project(
            self.o_proj, y.transpose(0, 2, 1, 3).reshape(batch, length, -1), row_exact
        )


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim, hidden = args.hidden_size, args.intermediate_size
        bias = args.model_type == "llama" and args.mlp_bias
        self.gate_proj = nn.Linear(dim, hidden, bias=bias)
        self.up_proj = nn.Linear(dim, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, dim, bias=bias)

    def __call__(self, x, *, row_exact=False):
        return _project(
            self.down_proj,
            swiglu(
                _project(self.gate_proj, x, row_exact),
                _project(self.up_proj, x, row_exact),
            ),
            row_exact,
        )


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )

    def __call__(self, x):
        scores = mx.softmax(self.gate(x), axis=-1, precise=True)
        indices = mx.argpartition(scores, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        scores = mx.take_along_axis(scores, indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        return mx.sum(self.switch_mlp(x, indices) * scores[..., None], axis=-2)


class Layer(nn.Module):
    def __init__(self, args: ModelArgs, index: int):
        super().__init__()
        self.self_attn = Attention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        sparse = (
            args.model_type == "qwen3_moe"
            and index not in args.mlp_only_layers
            and (index + 1) % args.decoder_sparse_step == 0
        )
        self.mlp = MoE(args) if sparse else MLP(args)

    def __call__(self, x, mask=None, cache=None, *, row_exact=False):
        h = x + self.self_attn(
            self.input_layernorm(x), mask, cache, row_exact=row_exact
        )
        normalized = self.post_attention_layernorm(h)
        return h + (
            self.mlp(normalized, row_exact=True) if row_exact else self.mlp(normalized)
        )


class Decoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Layer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs,
        cache=None,
        *,
        capture_layers=(),
        hidden_sink=None,
        row_exact=False,
    ):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask, layer_cache, row_exact=row_exact)
            if hidden_sink is not None and index in capture_layers:
                hidden_sink.append(h)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Decoder(args)
        self._target_verify_row_exact = False
        self._row_exact_backbone_dtype = None
        self._row_exact_forwards = 0
        self._row_exact_query_rows = 0
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        out = self.model(inputs, cache)
        return self.logits(out)

    def paged_embed(self, token_ids):
        """Flattened packed rows; ordinary ``__call__`` remains the reference."""
        return self.model.embed_tokens(mx.array(token_ids, dtype=mx.int32))

    def paged_project(self, index, hidden, row_counts, offsets, *,
                      vector_q1_rope=False):
        """Qwen3-owned Q/K/V, including per-lane absolute RoPE positions."""
        layer = self.layers[index]
        attention = layer.self_attn
        normalized = layer.input_layernorm(hidden)
        rows = hidden.shape[0]
        q = attention.q_proj(normalized).reshape(rows, attention.n_heads, attention.head_dim)
        k = attention.k_proj(normalized).reshape(rows, attention.n_kv_heads, attention.head_dim)
        v = attention.v_proj(normalized).reshape(rows, attention.n_kv_heads, attention.head_dim)
        q, k = attention.q_norm(q), attention.k_norm(k)
        if vector_q1_rope:
            if (tuple(row_counts) != (1, 1) or len(offsets) != 2 or
                    any(type(offset) is not int or not 0 <= offset < 2**31
                        for offset in offsets)):
                raise ValueError("vector Q1 RoPE requires two bounded one-row lanes")
            # MLX RoPE accepts one offset per batch item. Keep the two rows
            # separate in the batch dimension, avoiding four lane-local RoPE
            # calls and two concatenations in every model layer.
            positions = mx.array(offsets, dtype=mx.int32)
            q = attention.rope(q[:, :, None, :], offset=positions)[:, :, 0, :]
            k = attention.rope(k[:, :, None, :], offset=positions)[:, :, 0, :]
            return q, k, v
        q_rows, k_rows = [], []
        begin = 0
        for count, offset in zip(row_counts, offsets):
            end = begin + count
            q_lane = q[begin:end].transpose(1, 0, 2)[None, ...]
            k_lane = k[begin:end].transpose(1, 0, 2)[None, ...]
            q_rows.append(attention.rope(q_lane, offset=offset)[0].transpose(1, 0, 2))
            k_rows.append(attention.rope(k_lane, offset=offset)[0].transpose(1, 0, 2))
            begin = end
        return mx.concatenate(q_rows), mx.concatenate(k_rows), v

    def paged_finish_layer(self, index, hidden, attended):
        layer = self.layers[index]
        h = hidden + layer.self_attn.o_proj(attended.reshape(hidden.shape[0], -1))
        return h + layer.mlp(layer.post_attention_layernorm(h))

    def paged_logits(self, hidden):
        return self.logits(self.model.norm(hidden))

    def logits(self, hidden, *, row_exact=False):
        projector = (
            self.model.embed_tokens.as_linear
            if self.args.tie_word_embeddings
            else self.lm_head
        )
        return _project(projector, hidden, row_exact)

    def make_cache(self):
        from .cache import KVCache

        return [KVCache() for _ in self.layers]

    def configure_target_verify_row_exact(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("target_verify_row_exact must be a boolean")
        if enabled:
            if (
                self.args.model_type != "qwen3"
                or self.args.num_experts
                or self.args.rope_scaling is not None
            ):
                raise ValueError(
                    "row-exact verification requires a known dense Qwen3 topology"
                )
            if os.environ.get("MLX2_FP_DECODE_KERNEL", "0") == "1":
                raise ValueError(
                    "row-exact verification does not support MLX2_FP_DECODE_KERNEL"
                )
            for layer in self.layers:
                if (
                    type(layer) is not Layer
                    or type(layer.self_attn) is not Attention
                    or type(layer.mlp) is not MLP
                ):
                    raise ValueError(
                        "row-exact verification requires the native dense module topology"
                    )
                projectors = [
                    getattr(layer.self_attn, name)
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                ]
                projectors += [
                    getattr(layer.mlp, name)
                    for name in ("gate_proj", "up_proj", "down_proj")
                ]
                if any(type(module) is not nn.Linear for module in projectors):
                    raise ValueError(
                        "row-exact verification requires native unquantized Linear modules"
                    )
            if type(self.model.embed_tokens) is not nn.Embedding or (
                not self.args.tie_word_embeddings
                and type(self.lm_head) is not nn.Linear
            ):
                raise ValueError(
                    "row-exact verification requires a native unquantized output head"
                )
            dtype = self._homogeneous_backbone_dtype()
            self._row_exact_backbone_dtype = dtype
        self._target_verify_row_exact = enabled

    def _homogeneous_backbone_dtype(self):
        # An existing KVCache stores future appends in its allocated dtype.
        # Mixed backbone math would need a separately evidenced per-layer
        # contract; the current candidate admits one native floating dtype.
        dtypes = {value.dtype for _, value in tree_flatten(self.model.parameters())}
        if len(dtypes) != 1 or not dtypes.issubset(
            {mx.float16, mx.bfloat16, mx.float32}
        ):
            raise ValueError(
                "row-exact verification requires a homogeneous floating backbone dtype"
            )
        return next(iter(dtypes))

    @property
    def supports_contextual_prefix_equivalence(self):
        """BF16 S1 candidate contract; qualification stays separate."""
        return bool(
            self._target_verify_row_exact
            and self._row_exact_backbone_dtype == mx.bfloat16
        )

    @property
    def external_execution_receipt(self):
        if not self._target_verify_row_exact:
            return None
        return {
            "target_verify_row_exact": {
                "algorithm": TARGET_VERIFY_ROW_EXACT_VERSION,
                "backbone_dtype": str(self._row_exact_backbone_dtype),
                "implemented": True,
                "qualified": False,
                "selected": True,
                "performance_claim": False,
                "observation_scope": "model_lifetime_not_per_request",
                "observed_used": self._row_exact_forwards > 0,
                "executed_forwards": self._row_exact_forwards,
                "physical_query_rows": self._row_exact_query_rows,
            }
        }

    def _validate_row_exact_cache(self, inputs, cache):
        from ..segmented_plain_kv import SegmentedBatchKVCache
        from ..segmented_rotating_kv import SegmentedKVView
        from .cache import KVCache

        if os.environ.get("MLX2_FP_DECODE_KERNEL", "0") == "1":
            raise ValueError(
                "row-exact verification does not support MLX2_FP_DECODE_KERNEL"
            )
        if self._homogeneous_backbone_dtype() != self._row_exact_backbone_dtype:
            raise ValueError(
                "row-exact backbone dtype changed after configuration; reconfigure before allocating caches"
            )
        # Serving installs some projection policies after adapter loading.
        # Dtype alone cannot admit replacement modules into this math contract.
        self.configure_target_verify_row_exact(True)
        if inputs.ndim != 2 or min(inputs.shape) <= 0 or cache is None:
            raise ValueError(
                "row-exact verification requires nonempty batched tokens and plain KV caches"
            )
        if len(cache) != len(self.layers):
            raise ValueError("row-exact verification requires every cache layer")
        batch, width = inputs.shape
        if type(cache[0]) is SegmentedKVView:
            transaction = cache[0].transaction
            if (
                transaction is None
                or len(transaction.caches) != len(cache)
                or any(
                    value is not canonical
                    for value, canonical in zip(cache, transaction.caches, strict=True)
                )
            ):
                raise ValueError(
                    "row-exact verification requires one canonical layer transaction"
                )
        geometries, leaves = [], []
        for layer in cache:
            if type(layer) is KVCache:
                rows, lengths = [layer], (width,) * batch
                offsets = (layer.offset,) * batch
            elif type(layer) is SegmentedKVView:
                layer._check()
                if layer._updated:
                    raise ValueError(
                        "row-exact verification requires an unconsumed transaction"
                    )
                rows, lengths = layer.rows, tuple(layer.lengths)
                offsets = tuple(row.offset for row in rows)
                if layer.width != width or layer._geometry[1:] != (offsets, lengths):
                    raise ValueError(
                        "row-exact verification has stale transaction geometry"
                    )
            elif type(layer) is SegmentedBatchKVCache:
                rows = layer.rows
                lengths = tuple(layer._step_lengths or ())
                offsets = tuple(row.offset for row in rows)
                if layer._updated or offsets != tuple(layer._base_lengths):
                    raise ValueError(
                        "row-exact verification has stale prepared geometry"
                    )
            else:
                raise ValueError("row-exact verification requires plain KV owners")
            if (
                len(lengths) != batch
                or max(lengths, default=0) != width
                or any(type(n) is not int or not 0 <= n <= width for n in lengths)
                or (type(layer) is not KVCache and len(rows) != batch)
            ):
                raise ValueError("row-exact verification has invalid ragged geometry")
            for row in rows:
                if (
                    type(row) is not KVCache
                    or type(row.offset) is not int
                    or row.offset < 0
                    or getattr(row, "_pld_ordinary_mask_padding", None) is not None
                ):
                    raise ValueError(
                        "row-exact verification requires original plain KV owners and masks"
                    )
                expected_batch = batch if type(layer) is KVCache else 1
                if row.keys is not None and (
                    row.values is None
                    or type(row.keys) is not mx.array
                    or type(row.values) is not mx.array
                    or row.keys.ndim != 4
                    or row.values.ndim != 4
                    or row.keys.shape[:2]
                    != (expected_batch, self.args.num_key_value_heads)
                    or row.values.shape != row.keys.shape
                    or row.keys.dtype != row.values.dtype
                    or row.keys.dtype not in (mx.float16, mx.bfloat16, mx.float32)
                    or row.keys.dtype != self._row_exact_backbone_dtype
                    or row.keys.shape[-1] != self.args.head_dim
                    or row.values.shape[-1] != self.args.head_dim
                    or row.offset > min(row.keys.shape[2], row.values.shape[2])
                ):
                    raise ValueError(
                        "row-exact verification has invalid plain KV storage"
                    )
                if row.keys is None and (row.values is not None or row.offset):
                    raise ValueError(
                        "row-exact verification has incomplete plain KV storage"
                    )
                leaves.append(id(row))
            geometries.append((type(layer), offsets, lengths))
        if len(set(geometries)) != 1 or len(set(leaves)) != len(leaves):
            raise ValueError(
                "row-exact verification requires matching independent layer geometry"
            )

    def forward_with_taps(self, inputs, cache, capture_layers, *, body_only=False):
        """Post-layer features before final norm, paired with consumed cache tokens."""
        capture_layers = tuple(capture_layers)
        if (
            not capture_layers
            or any(type(index) is not int for index in capture_layers)
            or tuple(sorted(set(capture_layers))) != capture_layers
            or capture_layers[0] < 0
            or capture_layers[-1] >= len(self.layers)
        ):
            raise ValueError("Invalid target capture layers")
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("Target cache must have one entry per layer")
        row_exact = self._target_verify_row_exact and not body_only
        if row_exact:
            self._validate_row_exact_cache(inputs, cache)
        taps = []
        hidden = self.model(
            inputs,
            cache=cache,
            capture_layers=capture_layers,
            hidden_sink=taps,
            row_exact=row_exact,
        )
        features = mx.concatenate(taps, axis=-1)
        logits = None if body_only else self.logits(hidden, row_exact=row_exact)
        if row_exact:
            # Count executed forwards, including failed requests whose outer
            # transaction later rolls back, rather than lazy graph construction.
            mx.eval(logits, features)
            self._row_exact_forwards += 1
            self._row_exact_query_rows += inputs.shape[0] * inputs.shape[1]
        return logits, features

    def prefill_body(self, inputs, cache, capture_layers):
        return self.forward_with_taps(inputs, cache, capture_layers, body_only=True)[1]

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        # Qwen2 and Llama checkpoints may retain a precomputed rotary table;
        # the rotary module builds its own frequencies from the configuration.
        return {
            key: value
            for key, value in weights.items()
            if "self_attn.rotary_emb.inv_freq" not in key
        }

    @property
    def layers(self):
        return self.model.layers
