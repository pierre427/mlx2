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
from .switch_layers import QuantizedSwitchLinear, SwitchGLU

NORTH_BATCH_ROW_EXACT_Q4_VERSION = "north-batch-row-exact-q4-v1"


class BatchRowExactQ4:
    """Explicit North decode-only q4 row arithmetic policy."""

    def __init__(self):
        self.selected = False
        self.counts = {
            "started_forwards": 0,
            "complete_forwards": 0,
            "physical_rows": 0,
            "dense_projection_calls": 0,
            "dense_projection_rows": 0,
            "head_calls": 0,
            "head_rows": 0,
            "ordinary_q8_router_calls": 0,
            "ordinary_q8_router_rows": 0,
            "ordinary_expert_calls": 0,
            "ordinary_expert_rows": 0,
            "kernel": 0,
            "group_kernel": 0,
            "per_row": 0,
            "one_row": 0,
            "refusals": 0,
        }
        self.geometry_audit = None
        self._forward_start = None

    @staticmethod
    def _geometry(module, input_dims, output_dims, *, bits=4, embedding=False):
        expected = nn.QuantizedEmbedding if embedding else nn.QuantizedLinear
        if type(module) is not expected:
            raise ValueError(
                "North batch row-exact q4 requires native quantized modules"
            )
        if (
            module.bits != bits
            or module.group_size != 64
            or module.mode != "affine"
            or module.weight.dtype != mx.uint32
            or module.scales.dtype != mx.bfloat16
            or module.biases is None
            or module.biases.dtype != mx.bfloat16
            or (not embedding and "bias" in module)
            or tuple(module.weight.shape) != (output_dims, input_dims * bits // 32)
            or tuple(module.scales.shape) != (output_dims, input_dims // 64)
            or tuple(module.biases.shape) != (output_dims, input_dims // 64)
        ):
            raise ValueError(
                f"North batch row-exact q4 requires affine q{bits} group-64 BF16 geometry"
            )

    @staticmethod
    def _expert_geometry(module, experts, input_dims, output_dims):
        if (
            type(module) is not QuantizedSwitchLinear
            or module.bits != 4
            or module.group_size != 64
            or module.mode != "affine"
            or module.weight.dtype != mx.uint32
            or module.scales.dtype != mx.bfloat16
            or module.biases is None
            or module.biases.dtype != mx.bfloat16
            or tuple(module.weight.shape) != (experts, output_dims, input_dims // 8)
            or tuple(module.scales.shape) != (experts, output_dims, input_dims // 64)
            or tuple(module.biases.shape) != (experts, output_dims, input_dims // 64)
            or "bias" in module
        ):
            raise ValueError("North batch row-exact q4 requires ordinary q4 expert geometry")

    def configure(self, model, enabled):
        if type(enabled) is not bool:
            raise ValueError("batch_row_exact_q4 must be a boolean")
        if enabled:
            if (
                model.args.model_type != "cohere2_moe"
                or not model.args.tie_word_embeddings
            ):
                raise ValueError("batch_row_exact_q4 requires tied North Cohere2-MoE")
            hidden = model.args.hidden_size
            heads = model.args.num_attention_heads * model.args.head_dim
            kv_heads = model.args.num_key_value_heads * model.args.head_dim
            self._geometry(
                model.model.embed_tokens,
                hidden,
                model.args.vocab_size,
                embedding=True,
            )
            audit = {
                "dense_q4_linears": 0,
                "tied_q4_heads": 1,
                "ordinary_q8_routers": 0,
                "ordinary_q4_expert_tables": 0,
            }
            for index, layer in enumerate(model.layers):
                for name, input_dims, output_dims in (
                    ("q_proj", hidden, heads),
                    ("k_proj", hidden, kv_heads),
                    ("v_proj", hidden, kv_heads),
                    ("o_proj", heads, hidden),
                ):
                    self._geometry(getattr(layer.self_attn, name), input_dims, output_dims)
                    audit["dense_q4_linears"] += 1
                if index < model.args.first_k_dense_replace:
                    if type(layer.mlp) is not MLP:
                        raise ValueError("North dense-prefix topology changed")
                    dense = model.args.prefix_dense_intermediate_size
                    for name, input_dims, output_dims in (
                        ("gate_proj", hidden, dense),
                        ("up_proj", hidden, dense),
                        ("down_proj", dense, hidden),
                    ):
                        self._geometry(getattr(layer.mlp, name), input_dims, output_dims)
                        audit["dense_q4_linears"] += 1
                elif type(layer.mlp) is not Cohere2MoeSparseBlock:
                    raise ValueError("North sparse-layer topology changed")
                else:
                    gate = layer.mlp.gate
                    self._geometry(
                        gate, hidden, model.args.num_experts, bits=8
                    )
                    audit["ordinary_q8_routers"] += 1
                    intermediate = model.args.intermediate_size
                    for name, input_dims, output_dims in (
                        ("gate_proj", hidden, intermediate),
                        ("up_proj", hidden, intermediate),
                        ("down_proj", intermediate, hidden),
                    ):
                        expert = getattr(layer.mlp.switch_mlp, name)
                        self._expert_geometry(
                            expert,
                            model.args.num_experts,
                            input_dims,
                            output_dims,
                        )
                        audit["ordinary_q4_expert_tables"] += 1
            self.geometry_audit = audit
        else:
            self.geometry_audit = None
        self.selected = enabled

    def begin_tokens(self, tokens):
        if (
            not self.selected
            or tokens.ndim != 2
            or tokens.shape[1] != 1
            or tokens.shape[0] <= 1
        ):
            return False
        from .row_exact_qmv import MAX_ROWS

        if tokens.shape[0] > MAX_ROWS:
            raise ValueError("North batch row-exact q4 exceeds its 32-lane capability")
        self.counts["started_forwards"] += 1
        self.counts["physical_rows"] += int(tokens.shape[0])
        self._forward_start = {
            key: self.counts[key]
            for key in (
                "dense_projection_calls",
                "ordinary_q8_router_calls",
                "ordinary_expert_calls",
                "head_calls",
            )
        }
        self._forward_start["rows"] = int(tokens.shape[0])
        self._forward_start["dense_projection_rows"] = self.counts[
            "dense_projection_rows"
        ]
        self._forward_start["ordinary_q8_router_rows"] = self.counts[
            "ordinary_q8_router_rows"
        ]
        self._forward_start["ordinary_expert_rows"] = self.counts[
            "ordinary_expert_rows"
        ]
        self._forward_start["head_rows"] = self.counts["head_rows"]
        return True

    def note_ordinary_sparse(self, x):
        self.counts["ordinary_q8_router_calls"] += 1
        self.counts["ordinary_q8_router_rows"] += int(x.shape[0])
        self.counts["ordinary_expert_calls"] += 1
        self.counts["ordinary_expert_rows"] += int(x.shape[0])

    def linear(self, module, x):
        from .row_exact_qmv import quantized_linear

        output, route = quantized_linear(module, x)
        self.counts[route] += 1
        self.counts["dense_projection_calls"] += 1
        self.counts["dense_projection_rows"] += int(x.shape[0])
        if route == "per_row":
            self.counts["refusals"] += 1
            raise RuntimeError("North batch row-exact q4 native projection declined")
        return output

    def linears(self, modules, x):
        from .row_exact_qmv import quantized_linears

        outputs, route = quantized_linears(modules, x)
        self.counts[route] += 1
        self.counts["dense_projection_calls"] += len(modules)
        self.counts["dense_projection_rows"] += int(x.shape[0]) * len(modules)
        if route == "per_row":
            self.counts["refusals"] += 1
            raise RuntimeError("North batch row-exact q4 native projection group declined")
        return outputs

    def head(self, embedding, x):
        from .row_exact_qmv import quantized_linear

        output, route = quantized_linear(embedding, x, call=embedding.as_linear)
        self.counts[route] += 1
        self.counts["head_calls"] += 1
        self.counts["head_rows"] += int(x.shape[0])
        if route == "per_row":
            self.counts["refusals"] += 1
            raise RuntimeError("North batch row-exact q4 native tied head declined")
        start = self._forward_start
        rows = None if start is None else start["rows"]
        audit = self.geometry_audit or {}
        dense_calls = audit.get("dense_q4_linears")
        sparse_calls = audit.get("ordinary_q8_routers")
        if start is None or (
            self.counts["dense_projection_calls"] - start["dense_projection_calls"]
            != dense_calls
            or self.counts["ordinary_q8_router_calls"]
            - start["ordinary_q8_router_calls"]
            != sparse_calls
            or self.counts["ordinary_expert_calls"] - start["ordinary_expert_calls"]
            != sparse_calls
            or self.counts["head_calls"] - start["head_calls"] != 1
            or self.counts["dense_projection_rows"]
            - start["dense_projection_rows"]
            != dense_calls * rows
            or self.counts["ordinary_q8_router_rows"]
            - start["ordinary_q8_router_rows"]
            != sparse_calls * rows
            or self.counts["ordinary_expert_rows"] - start["ordinary_expert_rows"]
            != sparse_calls * rows
            or self.counts["head_rows"] - start["head_rows"] != rows
        ):
            self.counts["refusals"] += 1
            raise RuntimeError("North batch row-exact q4 forward engagement was incomplete")
        self.counts["complete_forwards"] += 1
        self._forward_start = None
        return output

    def status(self):
        return {
            "schema": "mlx2.north-batch-row-exact-q4.v1",
            "algorithm": NORTH_BATCH_ROW_EXACT_Q4_VERSION,
            "implemented": True,
            "qualified": False,
            "selected": self.selected,
            "observed_used": (
                self.counts["complete_forwards"] > 0
                and self.counts["refusals"] == 0
                and self.counts["kernel"] > 0
                and self.counts["group_kernel"] > 0
            ),
            "scope": "multi-lane one-token decode only",
            "max_rows": 32,
            "geometry": "affine-q4-group64-bfloat16",
            "geometry_audit": self.geometry_audit,
            "counts": dict(self.counts),
        }


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

    def __call__(self, x: mx.array, row_exact=None) -> mx.array:
        if row_exact is not None:
            gate, up = row_exact.linears((self.gate_proj, self.up_proj), x)
            return row_exact.linear(self.down_proj, swiglu(gate, up))
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Cohere2MoeSparseBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.intermediate_size, args.num_experts
        )

    def __call__(self, x: mx.array, row_exact=None) -> mx.array:
        if row_exact is not None:
            row_exact.note_ordinary_sparse(x)
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

    def __call__(self, x: mx.array, mask=None, cache=None, row_exact=None) -> mx.array:
        batch, length, _ = x.shape
        if row_exact is None:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        else:
            q, k, v = row_exact.linears((self.q_proj, self.k_proj, self.v_proj), x)
        q = q.reshape(batch, length, self.n_heads, self.head_dim)
        k = k.reshape(batch, length, self.n_kv_heads, self.head_dim)
        v = v.reshape(batch, length, self.n_kv_heads, self.head_dim)
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
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return (
            self.o_proj(out)
            if row_exact is None
            else row_exact.linear(self.o_proj, out)
        )


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

    def __call__(self, x: mx.array, mask=None, cache=None, row_exact=None) -> mx.array:
        h = self.input_layernorm(x)
        return x + self.self_attn(h, mask, cache, row_exact) + self.mlp(h, row_exact)


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

    def __call__(self, inputs: mx.array, cache=None, row_exact=None):
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
                h = layer(h, masks[layer.attention_type], layer_cache, row_exact)
            return self.norm(h)
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            h = layer(h, masks[layer.attention_type], layer_cache, row_exact)
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
        self._batch_row_exact_q4 = BatchRowExactQ4()
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def apc_v2_layout(self):
        return "north-mini-code-layer-segments-v1"

    def __call__(self, inputs: mx.array, cache=None):
        row_exact = (
            self._batch_row_exact_q4
            if self._batch_row_exact_q4.begin_tokens(inputs)
            else None
        )
        hidden = self.model(inputs, cache, row_exact)
        return self._project_with_policy(hidden, row_exact)

    def _project(self, hidden):
        logits = (
            self.model.embed_tokens.as_linear(hidden)
            if self.args.tie_word_embeddings
            else self.lm_head(hidden)
        )
        return logits * self.args.logit_scale

    def _project_with_policy(self, hidden, row_exact):
        if row_exact is None:
            return self._project(hidden)
        if not self.args.tie_word_embeddings:
            raise ValueError("North batch row-exact q4 requires its tied head")
        return row_exact.head(self.model.embed_tokens, hidden) * self.args.logit_scale

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
        if (
            body_only
            and self._batch_row_exact_q4.selected
            and inputs.ndim == 2
            and inputs.shape[1] == 1
            and inputs.shape[0] > 1
        ):
            raise ValueError(
                "North batch row-exact q4 requires tied-head completion"
            )
        taps = self.model.residual_taps
        if taps.capture is not None:
            raise RuntimeError("North residual capture is already in use")
        residual = {i: None for i in capture_layers if i < count}
        taps.capture = residual or None
        try:
            row_exact = (
                self._batch_row_exact_q4
                if self._batch_row_exact_q4.begin_tokens(inputs)
                else None
            )
            hidden = self.model(inputs, cache, row_exact)
        finally:
            taps.capture = None
        features = mx.concatenate(
            [hidden if i == count else residual[i] for i in capture_layers], axis=-1
        )
        if body_only:
            return None, features
        return self._project_with_policy(hidden, row_exact), features

    def prefill_body(self, inputs, cache, capture_layers):
        return self.forward_with_taps(inputs, cache, capture_layers, body_only=True)[1]

    def configure_batch_row_exact_q4(self, enabled):
        self._batch_row_exact_q4.configure(self, enabled)

    @property
    def batch_row_exact_q4_status(self):
        return self._batch_row_exact_q4.status()

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
