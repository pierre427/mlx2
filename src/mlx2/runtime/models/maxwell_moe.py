# SPDX-License-Identifier: Apache-2.0
"""MLX reference implementation of Maxwell-V1-Flash (``maxwell_moe``).

The model-specific numerical law lives here (see ``provenance/maxwell-moe.json``;
the source of truth is the checkpoint's own ``modeling_maxwell_moe.py``):

* mHC residual streams: the hidden state is ``n`` (4) copies of the model width.
  Each sub-block computes ``h_pre`` (sigmoid + eps), ``h_post`` (2 sigmoid) and
  a doubly-stochastic ``h_res`` (softmax, then Sinkhorn column/row passes) from
  an RMS-scaled fp32 projection of all streams, collapses the streams with
  ``h_pre``, runs the sub-block and writes ``h_post * out + h_res^T @ streams``.
  A learned ``hc_head`` contracts the streams before the final RMSNorm.
* Attention: GQA with per-head QK RMSNorm and RoPE (base 2e7).  The checkpoint
  routes every layer through Block Sparse GQA: fixed prefix blocks + local
  blocks + 16 indexer-selected remote blocks of 128 tokens.  Whenever the
  context is at most ``(prefix + local + remote) * block`` = 4,096 tokens the
  route covers every causal block, so it equals dense causal attention
  exactly; this module implements that regime and fails closed beyond it.
  The indexer weights are loaded (strict closure) but not evaluated.
* MoE (layers not in ``mlp_only_layers``): fp32 sigmoid scores, top-k chosen on
  ``scores + expert_bias`` (fp32), weights are the chosen *unbiased* scores
  renormalised to sum to one, plus an ungated shared expert.
* Logits: an fp32 head over the post-norm hidden state.

The native MTP layer is not implemented (its weights are dropped at load).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mlx.core as mx
from mlx import nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache
from .switch_layers import SwitchGLU

HC_EPS = 1e-6


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_head_size: int
    rms_norm_eps: float
    rope_theta: float
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    mlp_only_layers: list = field(default_factory=list)
    max_position_embeddings: int = 524288
    num_residual_streams: int = 4
    mhc_sinkhorn_iterations: int = 20
    block_sparse_gqa_block_size: int = 128
    block_sparse_gqa_prefix_window_size: int = 1024
    block_sparse_gqa_local_window_size: int = 1024
    block_sparse_gqa_num_remote_blocks: int = 16
    block_sparse_gqa_indexer_n_heads: int = 4
    block_sparse_gqa_indexer_head_dim: int = 128
    tie_word_embeddings: bool = False
    qk_layernorm: bool = True
    enable_hyper_connections: bool = True
    moe_router_enable_expert_bias: bool = True
    router_score_function: str = "sigmoid"
    use_shared_expert_gate: bool = False
    hidden_act: str = "silu"
    rope_scaling: dict | None = None

    def __post_init__(self):
        if self.model_type != "maxwell_moe":
            raise ValueError("Maxwell model_type must be 'maxwell_moe'")
        if not (self.qk_layernorm and self.enable_hyper_connections
                and self.moe_router_enable_expert_bias):
            raise ValueError("Maxwell requires QK norm, mHC and expert bias")
        if self.router_score_function != "sigmoid" or self.use_shared_expert_gate:
            raise ValueError("Maxwell requires sigmoid routing and an ungated shared expert")
        if self.hidden_act != "silu" or self.rope_scaling is not None:
            raise ValueError("Maxwell requires silu and unscaled RoPE")
        if self.tie_word_embeddings:
            raise ValueError("Maxwell has an untied lm_head")

    @property
    def dense_context_limit(self) -> int:
        """Largest context where the Block Sparse GQA route is dense causal."""
        b = self.block_sparse_gqa_block_size
        return (self.block_sparse_gqa_prefix_window_size // b
                + self.block_sparse_gqa_local_window_size // b
                + self.block_sparse_gqa_num_remote_blocks) * b


def rms_norm(x, weight, eps):
    """HF Maxwell RMSNorm: fp32 normalise, fp32 weight multiply, one rounding."""
    xf = x.astype(mx.float32)
    xf = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (weight.astype(mx.float32) * xf).astype(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return rms_norm(x, self.weight, self.eps)


class MLP(nn.Module):
    """SwiGLU; checkpoint w1/w3/w2 are renamed gate/up/down at load."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Indexer(nn.Module):
    """Block Sparse GQA index branch; loaded, not evaluated in the dense regime."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        g, d = args.block_sparse_gqa_indexer_n_heads, args.block_sparse_gqa_indexer_head_dim
        self.linear_q_proj = nn.Linear(args.hidden_size, g * d, bias=False)
        self.linear_k_proj = nn.Linear(args.hidden_size, d, bias=False)
        self.q_layernorm = RMSNorm(d, args.rms_norm_eps)
        self.k_layernorm = RMSNorm(d, args.rms_norm_eps)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.attention_head_size
        self.scale = self.head_dim ** -0.5
        d = args.hidden_size
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.q_layernorm = RMSNorm(self.head_dim, args.rms_norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, args.rms_norm_eps)
        self.indexer = Indexer(args)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)

    def __call__(self, x, mask=None, cache=None):
        b, t, _ = x.shape
        q = self.q_layernorm(self.q_proj(x).reshape(b, t, self.n_heads, self.head_dim))
        k = self.k_layernorm(self.k_proj(x).reshape(b, t, self.n_kv_heads, self.head_dim))
        v = self.v_proj(x).reshape(b, t, self.n_kv_heads, self.head_dim)
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        offset = cache.offset if cache is not None else 0
        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        y = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        return self.o_proj(y.transpose(0, 2, 1, 3).reshape(b, t, -1))


class Router(nn.Module):
    """fp32 sigmoid scores; top-k on scores + expert_bias; renormalised unbiased weights.

    Returns ``(ids, weights)`` using the Laguna router contract.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.expert_bias = mx.zeros((args.num_experts,), dtype=mx.float32)

    def __call__(self, x):
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        scores = mx.sigmoid(logits)
        biased = scores + self.expert_bias.astype(mx.float32)
        ids = mx.stop_gradient(mx.argpartition(-biased, kth=self.top_k - 1, axis=-1)[..., : self.top_k])
        w = mx.take_along_axis(scores, ids, axis=-1)
        return ids, w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)


class SparseMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.scale = 1.0           # routed-output scale (none in Maxwell); carve contract
        self.gate = Router(args)
        self.switch_mlp = SwitchGLU(args.hidden_size, args.moe_intermediate_size, args.num_experts)
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)

    def __call__(self, x):
        ids, w = self.gate(x)
        rows = self.switch_mlp(x, ids)
        w = w.astype(mx.float32) * self.scale
        routed = (rows.astype(mx.float32) * w[..., None]).sum(axis=-2)
        return routed.astype(x.dtype) + self.shared_expert(x)


class HyperConnection(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n = n = args.num_residual_streams
        self.hidden_size = args.hidden_size
        self.iterations = args.mhc_sinkhorn_iterations
        self.mapping_proj = nn.Linear(n * args.hidden_size, n * n + 2 * n, bias=False)
        self.alpha_pre = mx.zeros((1,))
        self.alpha_post = mx.zeros((1,))
        self.alpha_res = mx.zeros((1,))
        self.bias = mx.zeros((n * n + 2 * n,))

    def mappings(self, x):
        b, t, nc = x.shape
        n = self.n
        xf = x.astype(mx.float32)
        norm = mx.sqrt(mx.sum(xf * xf, axis=-1, keepdims=True))
        r = 1.0 / (norm / math.sqrt(nc) + HC_EPS)
        proj = xf @ self.mapping_proj.weight.astype(mx.float32).T
        alpha = mx.concatenate([
            mx.broadcast_to(self.alpha_pre, (n,)), mx.broadcast_to(self.alpha_post, (n,)),
            mx.broadcast_to(self.alpha_res, (n * n,)),
        ]).astype(mx.float32)
        h = r * proj * alpha + self.bias.astype(mx.float32)
        h_pre = (mx.sigmoid(h[..., :n]) + HC_EPS).astype(x.dtype)
        h_post = (mx.sigmoid(h[..., n:2 * n]) * 2).astype(x.dtype)
        m = mx.softmax(h[..., 2 * n:].reshape(b, t, n, n), axis=-1)
        m = m + HC_EPS
        m = m / (mx.sum(m, axis=-2, keepdims=True) + HC_EPS)
        for _ in range(self.iterations - 1):
            m = m / (mx.sum(m, axis=-1, keepdims=True) + HC_EPS)
            m = m / (mx.sum(m, axis=-2, keepdims=True) + HC_EPS)
        return h_pre, h_post, m.astype(x.dtype)

    def aggregate(self, x, h_pre):
        b, t, _ = x.shape
        s = x.reshape(b, t, self.n, self.hidden_size).astype(mx.float32)
        return (s * h_pre.astype(mx.float32)[..., None]).sum(axis=2).astype(x.dtype)

    def expand_and_mix(self, out, residual, h_post, h_res):
        b, t, c = out.shape
        res = residual.reshape(b, t, self.n, c).astype(mx.float32)
        mixed = mx.swapaxes(h_res.astype(mx.float32), -1, -2) @ res
        expanded = h_post.astype(mx.float32)[..., None] * out.astype(mx.float32)[:, :, None, :]
        return (expanded + mixed).astype(residual.dtype).reshape(b, t, self.n * c)


class Layer(nn.Module):
    def __init__(self, args: ModelArgs, index: int):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = (MLP(args.hidden_size, args.intermediate_size)
                    if index in set(args.mlp_only_layers) else SparseMoE(args))
        self.input_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.self_attn_hyper_connection = HyperConnection(args)
        self.mlp_hyper_connection = HyperConnection(args)

    def __call__(self, x, mask=None, cache=None):
        hc = self.self_attn_hyper_connection
        h_pre, h_post, h_res = hc.mappings(x)
        a = self.self_attn(self.input_layernorm(hc.aggregate(x, h_pre)), mask, cache)
        x = hc.expand_and_mix(a, x, h_post, h_res)
        hc = self.mlp_hyper_connection
        h_pre, h_post, h_res = hc.mappings(x)
        m = self.mlp(self.post_attention_layernorm(hc.aggregate(x, h_pre)))
        return hc.expand_and_mix(m, x, h_post, h_res)


class FinalNorm(nn.Module):
    """The learned mHC head (n streams -> 1) followed by the final RMSNorm.

    One module so that ``norm(multi-stream hidden)`` is the post-norm hidden
    state, as structural callers (layer replay) expect."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        n, c = args.num_residual_streams, args.hidden_size
        self.n, self.hidden_size, self.eps = n, c, args.rms_norm_eps
        self.weight = mx.ones((c,))
        self.hc_head_fn = mx.zeros((n, n * c))
        self.hc_head_base = mx.zeros((n,))
        self.hc_head_scale = mx.ones((1,))

    def contract(self, x):
        b, t, _ = x.shape
        xf = x.astype(mx.float32)
        rs = mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + HC_EPS)
        mixes = (xf @ self.hc_head_fn.astype(mx.float32).T) * rs
        w = mx.sigmoid(mixes * self.hc_head_scale.astype(mx.float32)
                       + self.hc_head_base.astype(mx.float32)) + HC_EPS
        s = x.reshape(b, t, self.n, self.hidden_size).astype(mx.float32)
        return (s * w[..., None]).sum(axis=2).astype(x.dtype)

    def __call__(self, x):
        return rms_norm(self.contract(x), self.weight, self.eps)


class Decoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Layer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = FinalNorm(args)

    def expand(self, inputs):
        h = self.embed_tokens(inputs)
        b, t, c = h.shape
        n = self.args.num_residual_streams
        return mx.broadcast_to(h[:, :, None, :], (b, t, n, c)).reshape(b, t, n * c)

    def __call__(self, inputs, cache=None, capture=None):
        if cache is None:
            cache = [None] * len(self.layers)
        offset = cache[0].offset if cache[0] is not None else 0
        limit = self.args.dense_context_limit
        if offset + inputs.shape[1] > limit:
            raise ValueError(
                f"Maxwell Block Sparse GQA is implemented only in its dense regime "
                f"(context <= {limit} tokens); refusing {offset + inputs.shape[1]}")
        h = self.expand(inputs)
        mask = create_attention_mask(h, cache[0])
        for i, (layer, c) in enumerate(zip(self.layers, cache, strict=True)):
            h = layer(h, mask, c)
            if capture is not None and i in capture:
                capture[i] = h          # multi-stream residual after block i
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Decoder(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.completed_forwards = 0

    def logits_from_hidden(self, h):
        """The reference head: fp32 linear over the post-norm hidden state."""
        return h.astype(mx.float32) @ self.lm_head.weight.astype(mx.float32).T

    def __call__(self, inputs, cache=None):
        out = self.logits_from_hidden(self.model(inputs, cache))
        self.completed_forwards += 1
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self, max_kv_size=None):
        return [KVCache() for _ in self.model.layers]

    def sanitize(self, weights, *, eager=False):
        """Checkpoint names -> this module tree; per-expert tensors -> SwitchGLU tables.

        Drops the native MTP layer (not implemented)."""
        a = self.args
        proj = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
        for name in [k for k in weights if k.startswith("mtp_layers.")]:
            weights.pop(name)
        for suffix in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            key = f"model.hc_head.{suffix}"
            if key in weights:
                weights[f"model.norm.{suffix}"] = weights.pop(key)
        for layer in range(a.num_hidden_layers):
            src = f"model.layers.{layer}.block_sparse_moe"
            dst = f"model.layers.{layer}.mlp"
            if layer in set(a.mlp_only_layers):
                for w, p in proj.items():
                    if f"{src}.{w}.weight" in weights:
                        weights[f"{dst}.{p}.weight"] = weights.pop(f"{src}.{w}.weight")
                continue
            for key, new in ((f"{src}.gate.weight", f"{dst}.gate.weight"),
                             (f"{src}.expert_bias", f"{dst}.gate.expert_bias")):
                if key in weights:
                    weights[new] = weights.pop(key)
            bias = f"{dst}.gate.expert_bias"
            if bias in weights:
                # stored fp32; its per-expert spread is below one bf16 step
                weights[bias] = weights[bias].astype(mx.float32)
            for w, p in proj.items():
                key = f"{src}.shared_expert.{w}.weight"
                if key in weights:
                    weights[f"{dst}.shared_expert.{p}.weight"] = weights.pop(key)
            packed = []
            for w, p in proj.items():
                names = [f"{src}.experts.{e}.{w}.weight" for e in range(a.num_experts)]
                present = [n in weights for n in names]
                if not any(present):
                    continue
                if not all(present):
                    raise ValueError(f"Maxwell layer {layer} has an incomplete {w} expert table")
                parts = [weights.pop(n) for n in names]
                value = mx.stack(parts)
                weights[f"{dst}.switch_mlp.{p}.weight"] = value
                packed.append(value)
            if eager and packed:
                mx.eval(packed)
                mx.clear_cache()
        return weights
