# SPDX-License-Identifier: MIT
"""Kimi-K3.1 text math for layer-at-a-time execution; provenance/kimi-k31-layer-stream.json.

Kimi-K3.1-Pro ships no modeling file. The equations follow the Moonshot K3
reference (moonshotai/Kimi-K3@9f62e4e modeling_kimi_linear.py) for the shared
K3 body, and the K3.1 gated-DSA indexer as vLLM #66145 and llama.cpp #31779
implement it. Every function here is a pure function of a flat weight mapping
with the checkpoint's own tensor names (prefix stripped), so one decoder layer
can be loaded, run and released on its own.

What crosses a layer boundary (``ResidualState``) is the attention-residual
stream, not a single hidden vector: the list of banked block representations
and the current block's partial sum. Nothing else crosses layers in a single
prompt forward (no KV or recurrent cache is kept; this is prefill only).

Weights may be bf16 or MLX affine q8 (``X.weight`` uint32 + ``X.scales`` +
``X.biases``); routed experts are compressed-tensors MXFP4 mapped onto MLX's
native ``mxfp4`` mode without changing a bit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import mlx.core as mx
import numpy as np

from ..technologies import linear_recurrence as _lr
from ..technologies.linear_recurrence import compute_lower_bound_g  # noqa: F401
from ..technologies import sparse_selection as _sparse
from ..technologies import mla as _mla
from ..technologies.moe_router import cached_spec, route as _route_reference

# Vendor KimiRMSNorm default eps; the K3 reference builds q_a_layernorm and
# kv_a_layernorm without passing config.rms_norm_eps (vLLM and llama.cpp use
# rms_norm_eps=1e-5 there instead: an open disagreement, see provenance).
MLA_LATENT_NORM_EPS = 1e-6
# FLA l2norm epsilon inside chunk_kda(use_qk_l2norm_in_kernel=True).
KDA_L2_EPS = 1e-6
# K3.1 q/k law only: FLA's fp32 l2 q/k (epsilon on the SUM).  It carries no
# gate: kda_attention applies the lower-bounded sigmoid gate with the
# checkpoint's cfg.gate_lower_bound (``_lr.decay("lower_bound_sigmoid", ...)``)
# and runs the vector recurrence through the shared dispatcher.
_K31_QK = _lr.LinearRecurrenceSpec(num_k_heads=1, num_v_heads=1, gate_kind="vector",
                                   beta_law="sigmoid_float32", qk_norm="l2_rsqrt",
                                   qk_norm_dtype="float32", qk_eps=KDA_L2_EPS,
                                   conv_layout="separate_qkv", core_scan=False)
FP4_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
MXFP4_GROUP = 32
Q8 = {"bits": 8, "group_size": 64, "mode": "affine"}


@dataclass(frozen=True)
class K31Config:
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    attn_res_block_size: int
    first_k_dense_replace: int
    moe_layer_freq: int
    kda_layers: frozenset          # 0-based
    full_attn_layers: frozenset    # 0-based
    dsa_layers: frozenset          # 0-based
    kda_heads: int
    kda_head_dim: int
    conv_kernel: int
    gate_lower_bound: float
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    mla_use_output_gate: bool
    index_topk: int
    index_heads: int
    index_head_dim: int
    num_experts: int
    top_k: int
    num_shared_experts: int
    moe_intermediate_size: int
    routed_expert_hidden_size: int
    latent_moe_use_norm: bool
    routed_scaling_factor: float
    moe_renormalize: bool
    situ_beta: float
    situ_linear_beta: float | None
    mla_latent_norm_eps: float = MLA_LATENT_NORM_EPS

    @classmethod
    def from_dict(cls, config: dict, **overrides) -> "K31Config":
        t = config.get("text_config", config)
        lin, dsa = t["linear_attn_config"], t.get("dsa_config") or {}
        # Layer lists are 1-based in the checkpoint (configuration_kimi_k3.py
        # is_kda_layer: ``(layer_idx + 1) in kda_layers``).
        kda = frozenset(i - 1 for i in lin["kda_layers"])
        full = frozenset(i - 1 for i in lin["full_attn_layers"])
        n = t["num_hidden_layers"]
        if kda | full != frozenset(range(n)) or kda & full:
            raise ValueError("kda_layers and full_attn_layers must partition the layers")
        refuse = []
        if t.get("hidden_act") != "situ":
            refuse.append(f"hidden_act={t.get('hidden_act')!r}")
        if t.get("moe_router_activation_func") != "sigmoid":
            refuse.append("router activation is not sigmoid")
        if t.get("topk_method") != "noaux_tc":
            refuse.append(f"topk_method={t.get('topk_method')!r}")
        if t.get("num_expert_group", 1) != 1 or t.get("topk_group", 1) != 1:
            refuse.append("grouped expert selection")
        if not t.get("mla_use_nope", False):
            refuse.append("MLA with RoPE")
        if not lin.get("use_full_rank_gate", False):
            refuse.append("low-rank KDA output gate")
        if dsa and (dsa.get("index_rope_dim", 0) != 0 or dsa.get("index_topk_freq", 1) != 1
                    or dsa.get("index_k_norm_type", "layernorm") != "layernorm"):
            refuse.append("DSA indexer variant other than NoPE/freq-1/layernorm")
        if dsa and sorted(dsa.get("dsa_layers", [])) != sorted(lin["full_attn_layers"]):
            refuse.append("dsa_layers differ from full_attn_layers")
        if dsa and dsa.get("dsa_layers") and not dsa.get("index_topk", 0) > 0:
            refuse.append("DSA layers without a positive index_topk")
        if dsa and dsa.get("index_skip_topk_offset", 1) != 1:
            # Read by no public stack (vLLM keeps it for schema parity); only
            # the shipped value is admitted.
            refuse.append(f"index_skip_topk_offset={dsa.get('index_skip_topk_offset')}")
        if t.get("attn_res_block_size") is None or t.get("routed_expert_hidden_size") is None:
            refuse.append("K3.1 requires attention residuals and latent MoE")
        if refuse:
            raise ValueError("Kimi-K3.1 layer stream refuses: " + "; ".join(refuse))
        values = dict(
            hidden_size=t["hidden_size"], num_hidden_layers=n, vocab_size=t["vocab_size"],
            rms_norm_eps=t["rms_norm_eps"], intermediate_size=t["intermediate_size"],
            attn_res_block_size=t["attn_res_block_size"],
            first_k_dense_replace=t["first_k_dense_replace"],
            moe_layer_freq=t.get("moe_layer_freq", 1),
            kda_layers=kda, full_attn_layers=full,
            dsa_layers=frozenset(i - 1 for i in dsa.get("dsa_layers", [])),
            kda_heads=lin["num_heads"], kda_head_dim=lin["head_dim"],
            conv_kernel=lin["short_conv_kernel_size"],
            gate_lower_bound=lin["gate_lower_bound"],
            num_attention_heads=t["num_attention_heads"], q_lora_rank=t["q_lora_rank"],
            kv_lora_rank=t["kv_lora_rank"], qk_nope_head_dim=t["qk_nope_head_dim"],
            qk_rope_head_dim=t["qk_rope_head_dim"], v_head_dim=t["v_head_dim"],
            mla_use_output_gate=bool(t.get("mla_use_output_gate", False)),
            index_topk=dsa.get("index_topk", 0), index_heads=dsa.get("index_num_heads", 0),
            index_head_dim=dsa.get("index_head_dim", 0),
            num_experts=t["num_experts"], top_k=t["num_experts_per_token"],
            num_shared_experts=t["num_shared_experts"],
            moe_intermediate_size=t["moe_intermediate_size"],
            routed_expert_hidden_size=t["routed_expert_hidden_size"],
            latent_moe_use_norm=bool(t["latent_moe_use_norm"]),
            routed_scaling_factor=t["routed_scaling_factor"],
            moe_renormalize=bool(t["moe_renormalize"]),
            situ_beta=t.get("activation_situ_beta") or 1.0,
            situ_linear_beta=t.get("activation_situ_linear_beta"),
        )
        values.update(overrides)
        return cls(**values)

    def is_moe(self, i: int) -> bool:
        return i >= self.first_k_dense_replace and i % self.moe_layer_freq == 0


# --------------------------------------------------------------------------
# primitives


def linear(w: dict, name: str, x: mx.array, quant: dict = Q8) -> mx.array:
    """``x @ W.T`` for a bf16 weight or an MLX affine quantized one.

    A tensor is quantized exactly when ``name.scales`` is present; its
    geometry must agree with ``quant`` or the call fails closed.
    """
    weight = w[name + ".weight"]
    scales = w.get(name + ".scales")
    if scales is None:
        if not mx.issubdtype(weight.dtype, mx.floating):
            raise ValueError(f"{name}: {weight.dtype} weight without scales (missing q8 tensors?)")
        return x @ weight.T.astype(x.dtype)
    gs, bits = quant["group_size"], quant["bits"]
    if weight.dtype != mx.uint32 or weight.shape[-1] * 32 != scales.shape[-1] * gs * bits:
        raise ValueError(f"{name}: packed shape {weight.shape} / scales {scales.shape} "
                         f"do not match affine bits={bits} group={gs}")
    if _on_cpu():
        wd = mx.dequantize(weight, scales, w[name + ".biases"], group_size=gs, bits=bits,
                           mode=quant.get("mode", "affine"))
        return _cpu_fp32_matmul(x, wd)
    return mx.quantized_matmul(x, weight, scales, w[name + ".biases"], transpose=True,
                               group_size=gs, bits=bits, mode=quant.get("mode", "affine"))


def _on_cpu() -> bool:
    return mx.default_device() == mx.cpu


def _cpu_fp32_matmul(x: mx.array, w_dequantized: mx.array) -> mx.array:
    """CPU route for quantized weights: same dequantized values, fp32 accumulate via BLAS.

    MLX's CPU quantized_matmul is ~150x slower than dequantize + fp32 matmul at prompt
    widths (31 s vs 0.2 s for 66x7168 @ 7168x12288 on M5 Max, mlx 0.32.4)."""
    return (x.astype(mx.float32) @ w_dequantized.astype(mx.float32).T).astype(x.dtype)


def embed(w: dict, name: str, tokens: mx.array, dtype, quant: dict = Q8) -> mx.array:
    weight = w[name + ".weight"]
    scales = w.get(name + ".scales")
    if scales is None:
        return weight[tokens].astype(dtype)
    return mx.dequantize(weight[tokens], scales[tokens], w[name + ".biases"][tokens],
                         group_size=quant["group_size"], bits=quant["bits"],
                         mode=quant.get("mode", "affine")).astype(dtype)


def rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """KimiRMSNorm: FP32 normalize, cast to the input dtype, then the weight."""
    xf = x.astype(mx.float32)
    normed = (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)).astype(x.dtype)
    return weight * normed


def situ(gate: mx.array, up: mx.array, beta: float, linear_beta: float | None) -> mx.array:
    """SituAndMul: beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta), FP32."""
    dtype = gate.dtype
    g, u = gate.astype(mx.float32), up.astype(mx.float32)
    a = beta * mx.tanh(g / beta) * mx.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * mx.tanh(u / linear_beta)
    return (a * u).astype(dtype)


def situ_mlp(w: dict, prefix: str, x: mx.array, cfg: K31Config, quant: dict = Q8) -> mx.array:
    a = situ(linear(w, prefix + ".gate_proj", x, quant), linear(w, prefix + ".up_proj", x, quant),
             cfg.situ_beta, cfg.situ_linear_beta)
    return linear(w, prefix + ".down_proj", a, quant)


def attn_res_mix(blocks: list, partial: mx.array, norm_w: mx.array, proj_w: mx.array,
                 eps: float) -> mx.array:
    """Depth attention over [banked blocks..., current partial sum] (K3 _apply_attn_res).

    Each source is RMS-normalised (no weight) and scored by the fused vector
    ``norm_w * proj_w``; a softmax over sources mixes the *raw* sources.
    """
    if not blocks:
        return partial
    score_w = norm_w.astype(mx.float32) * proj_w.reshape(-1).astype(mx.float32)
    t = partial.shape[0]
    # Per-token math, so row chunks are exact; ~0.5 GiB of FP32 per source
    # chunk keeps a 32K x 7168 x 9-source mix from materialising ~30 GB.
    rows = max(1, min(t, (1 << 27) // max(1, partial.shape[-1] * (len(blocks) + 1))))
    outs = []
    for r0 in range(0, t, rows):
        v = mx.stack([b[r0:r0 + rows] for b in (*blocks, partial)]).astype(mx.float32)  # [S, r, D]
        k = v * mx.rsqrt(mx.mean(v * v, axis=-1, keepdims=True) + eps)
        probs = mx.softmax(mx.sum(k * score_w, axis=-1), axis=0)                         # [S, r]
        outs.append(mx.sum(probs[..., None] * v, axis=0).astype(partial.dtype))
        if rows < t:
            mx.eval(outs[-1])
    return outs[0] if len(outs) == 1 else mx.concatenate(outs)


# --------------------------------------------------------------------------
# attention


def _short_conv(x: mx.array, weight: mx.array, tail: mx.array | None = None):
    """FLA ShortConvolution(activation='silu'): causal depthwise over [T, P].

    ``tail`` holds the previous K-1 inputs (FP32), zeros at sequence start.
    Returns (output, new tail) so a decode step continues the convolution."""
    t = x.shape[0]
    kernel = weight.shape[-1]
    w = weight.reshape(weight.shape[0], kernel).astype(mx.float32)    # [P, K]
    if tail is None:
        tail = mx.zeros((kernel - 1, x.shape[1]), mx.float32)
    xp = mx.concatenate([tail, x.astype(mx.float32)])
    out = sum(xp[j:j + t] * w[:, j] for j in range(kernel))
    return (out * mx.sigmoid(out)).astype(x.dtype), xp[-(kernel - 1):]


def kda_attention(w: dict, x: mx.array, cfg: K31Config, quant: dict = Q8,
                  cache: dict | None = None) -> mx.array:
    """KDA with the K3 lower-bounded gate; x is the normed layer input [T, D].

    With ``cache`` (a per-layer dict), the conv tails and the recurrent state
    are read from it and written back, so prompt + decode steps equal one
    forward over the whole sequence."""
    t = x.shape[0]
    h, d = cfg.kda_heads, cfg.kda_head_dim
    c = cache if cache is not None else {}
    q, tq = _short_conv(linear(w, "self_attn.q_proj", x, quant), w["self_attn.q_conv1d.weight"], c.get("q_tail"))
    k, tk = _short_conv(linear(w, "self_attn.k_proj", x, quant), w["self_attn.k_conv1d.weight"], c.get("k_tail"))
    v, tv = _short_conv(linear(w, "self_attn.v_proj", x, quant), w["self_attn.v_conv1d.weight"], c.get("v_tail"))
    q, k = q.astype(mx.float32).reshape(1, t, h, d), k.astype(mx.float32).reshape(1, t, h, d)
    q, k = _lr.normalize_qk(_K31_QK, q, k)
    a = linear(w, "self_attn.f_b_proj", linear(w, "self_attn.f_a_proj", x, quant), quant)
    # FLA's in-kernel gate (chunk_kda use_gate_in_kernel=True) reads A_log[i_h]:
    # the checkpoint's [head_dim]-long A_log is consumed per head, first H entries.
    a_log = w["self_attn.A_log"].reshape(-1)[:h].reshape(h, 1)
    decay = _lr.decay("lower_bound_sigmoid", a_log, a.reshape(1, t, h, d),
                      w["self_attn.dt_bias"].reshape(h, d).astype(mx.float32),
                      cfg.gate_lower_bound)
    beta = _lr.beta("sigmoid_float32", linear(w, "self_attn.b_proj", x, quant)).reshape(1, t, h)
    v = v.astype(mx.float32).reshape(1, t, h, d)
    s0 = c.get("state")
    out, s1 = _lr.recurrence(q, k, v, decay, beta,
                             s0 if s0 is not None else mx.zeros((1, h, d, d), mx.float32), None,
                             allow_core=False)
    if cache is not None:
        mx.eval(s1, tq, tk, tv)
        cache.update(state=s1, q_tail=tq, k_tail=tk, v_tail=tv)
    out = out.reshape(t, h, d).astype(x.dtype).astype(mx.float32)
    gate = linear(w, "self_attn.g_proj", x, quant).reshape(t, h, d).astype(mx.float32)
    normed = out * mx.rsqrt(mx.mean(out * out, -1, keepdims=True) + cfg.rms_norm_eps)
    normed = normed * w["self_attn.o_norm.weight"].astype(mx.float32)
    out = (normed * mx.sigmoid(gate)).astype(x.dtype).reshape(t, h * d)
    return linear(w, "self_attn.o_proj", out, quant)


def _layer_norm(x: mx.array, weight: mx.array, bias: mx.array, eps: float) -> mx.array:
    xf = x.astype(mx.float32)
    mu = mx.mean(xf, -1, keepdims=True)
    var = mx.mean((xf - mu) ** 2, -1, keepdims=True)
    y = (xf - mu) * mx.rsqrt(var + eps) * weight.astype(mx.float32) + bias.astype(mx.float32)
    return y.astype(x.dtype)


def _row_chunk(heads: int, keys: int, budget: int = 1 << 28) -> int:
    """Query rows per chunk so an FP32 [heads, rows, keys] block stays ~1 GiB."""
    return max(1, min(keys, budget // max(1, heads * keys)))


# K3.1 DSA indexer laws (historical values, sparse_selection spec): weights
# pre-scaled, einsum in the operands' dtype, token-causal keys, ``argsort``
# top-k (exact ties unspecified; ``stable_chronological`` is the proposed fix).
DSA_SPEC = _sparse.SelectionSpec(
    score_layout="lhd_einsum", score_scale_site="weights", score_cast="none",
    tie_break="argsort", visibility="token_causal",
)


def dsa_keep(index: dict, r0: int, r1: int, topk: int, offset: int = 0) -> mx.array:
    """Lightning-indexer top-k mask for query rows [r0, r1) over keys [0, offset + r1).

    Rows index the current chunk (absolute positions offset + r0 ...); keys
    cover the whole cached sequence. score[t, s] = sum_h w[t, h] *
    relu(q[t, h] . k[s]), with w pre-scaled by (index_head_dim *
    index_heads)^-1/2, over causal keys; each row keeps its min(topk, t + 1)
    best keys.
    """
    n = offset + r1
    iq, ik, iw = index["q"][r0:r1], index["k"][:n], index["w"][r0:r1]
    score = _sparse.score(DSA_SPEC, iq, ik, iw)                         # [rows, n]
    causal = _sparse.visible(DSA_SPEC, mx.arange(offset + r0, offset + r1), n)
    score = mx.where(causal, score, -mx.inf)
    # Rows with fewer causal keys than top-k pick up -inf columns, which the
    # causal mask drops again: they keep every causal key.
    order = _sparse.topk(DSA_SPEC, score, topk)
    keep = mx.put_along_axis(mx.zeros(score.shape, mx.bool_), order,
                             mx.ones(order.shape, mx.bool_), axis=-1)
    return keep & causal


def _index_keys(w: dict, x: mx.array, cfg: K31Config, quant: dict = Q8) -> mx.array:
    return _layer_norm(linear(w, "self_attn.indexer.wk", x, quant),
                       w["self_attn.indexer.k_norm.weight"], w["self_attn.indexer.k_norm.bias"],
                       cfg.rms_norm_eps).astype(mx.float32)


def dsa_index(w: dict, x: mx.array, qr: mx.array, cfg: K31Config, quant: dict = Q8,
              cache: dict | None = None, offset: int = 0):
    """Indexer queries/keys/weights, or None while the sequence is dense.

    Called for DSA layers only. A sequence no longer than index_topk keeps
    every causal key in every row, so the indexer is skipped: exactly dense
    attention. With ``cache``, indexer keys are always appended (they are
    needed once the sequence grows past index_topk).
    """
    t = x.shape[0]
    if "self_attn.indexer.wk.weight" not in w:
        raise KeyError("DSA layer without its indexer weights (self_attn.indexer.*)")
    keys = None
    if cache is not None:
        new = _index_keys(w, x, cfg, quant)
        keys = new if offset == 0 or "index_k" not in cache else mx.concatenate([cache["index_k"], new])
        if keys.shape[0] != offset + t:
            raise ValueError(f"indexer key cache holds {keys.shape[0]} rows, expected {offset + t}")
        mx.eval(keys)
        cache["index_k"] = keys
    if offset + t <= cfg.index_topk:
        return None
    hi, di = cfg.index_heads, cfg.index_head_dim
    return {
        "q": linear(w, "self_attn.indexer.wq_b", qr, quant).astype(mx.float32).reshape(t, hi, di),
        "k": keys if keys is not None else _index_keys(w, x, cfg, quant),
        "w": linear(w, "self_attn.indexer.weights_proj", x, quant).astype(mx.float32) * (di * hi) ** -0.5,
    }


@lru_cache(maxsize=16)
def mla_spec(cfg: K31Config) -> "_mla.MLASpec":
    """K3.1 MLA: gated NoPE, expanded only (concat_key, FP32 scores).

    Built once per (frozen, hashable) config, not per attention call."""
    return _mla.MLASpec.from_config(
        cfg, position="nope", mscale_rule="none", absorbed_rule="never",
        pe_form="split_fp32", expanded_pe_form="concat_key")


def mla_attention(w: dict, x: mx.array, cfg: K31Config, quant: dict = Q8, row_chunk: int | None = None,
                  *, dsa: bool = True, cache: dict | None = None, offset: int = 0):
    """Gated NoPE MLA (with the K3.1 DSA top-k mask on indexer layers).

    Expanded (non-absorbed) form, processed in query-row chunks so a 32K
    prompt never materialises a full [H, T, T] score tensor. With ``cache``,
    the normed latent and k_pe rows (576 values per token) are appended and
    keys/values are re-expanded over the whole cached sequence; the current
    rows sit at absolute positions ``offset ...``.
    """
    t = x.shape[0]
    h, dn, dr, dv = (cfg.num_attention_heads, cfg.qk_nope_head_dim,
                     cfg.qk_rope_head_dim, cfg.v_head_dim)
    eps = cfg.mla_latent_norm_eps
    qr = rms_norm(linear(w, "self_attn.q_a_proj", x, quant), w["self_attn.q_a_layernorm.weight"], eps)
    q = linear(w, "self_attn.q_b_proj", qr, quant).reshape(t, h, dn + dr)
    ckv = linear(w, "self_attn.kv_a_proj_with_mqa", x, quant)
    latent = rms_norm(ckv[:, :cfg.kv_lora_rank], w["self_attn.kv_a_layernorm.weight"], eps)
    k_pe = ckv[:, cfg.kv_lora_rank:]                                    # NoPE: no rotation
    if cache is not None:
        if offset and "latent" in cache:
            latent = mx.concatenate([cache["latent"], latent])
            k_pe = mx.concatenate([cache["k_pe"], k_pe])
        if latent.shape[0] != offset + t:
            raise ValueError(f"MLA cache holds {latent.shape[0]} rows, expected {offset + t}")
        mx.eval(latent, k_pe)
        cache.update(latent=latent, k_pe=k_pe)
    n = offset + t
    kv = linear(w, "self_attn.kv_b_proj", latent, quant).reshape(n, h, dn + dv)
    index = dsa_index(w, x, qr, cfg, quant, cache=cache, offset=offset) if dsa else None
    rows = row_chunk or _row_chunk(max(h, cfg.index_heads or 1), n)
    keep = (None if index is None
            else lambda r0, r1: dsa_keep(index, r0, r1, cfg.index_topk, offset))
    out = _mla.expanded_rows_softmax(
        q, kv, k_pe, nope=dn, scale=mla_spec(cfg).softmax_scale, offset=offset, rows=rows,
        out_dtype=x.dtype, keep=keep).reshape(t, h * dv)
    if cfg.mla_use_output_gate:
        out = out * mx.sigmoid(linear(w, "self_attn.g_proj", x, quant))
    return linear(w, "self_attn.o_proj", out, quant), index is not None


# --------------------------------------------------------------------------
# latent MoE


@dataclass
class Routing:
    """Router output for one layer. ``weights`` are renormalised over the top-k."""

    ids: mx.array        # [T, K] int32, in top-k order
    weights: mx.array    # [T, K] f32, renormalised * routed_scaling_factor
    scores: mx.array     # [T, K] f32 raw sigmoid scores of the chosen experts

    def aggregate(self) -> dict:
        ids, wts = np.array(self.ids), np.array(self.weights)
        total, count = {}, {}
        for e, v in zip(ids.reshape(-1).tolist(), wts.reshape(-1).tolist()):
            total[e] = total.get(e, 0.0) + v
            count[e] = count.get(e, 0) + 1
        return {"weight": dict(sorted(total.items(), key=lambda kv: -kv[1])), "tokens": count}


def route(w: dict, x: mx.array, cfg: K31Config) -> Routing:
    """noaux_tc: choose by sigmoid + e_score_correction_bias, weight by sigmoid."""
    gate = w["block_sparse_moe.gate.weight"]
    if not mx.issubdtype(gate.dtype, mx.floating) or "block_sparse_moe.gate.scales" in w:
        raise ValueError("router gate must be a float matrix (it is never quantized)")
    logits = x.astype(mx.float32) @ gate.astype(mx.float32).T
    # Top-k by choice score, descending (exact ties are ordered arbitrarily).
    spec = cached_spec(
        top_k=cfg.top_k, score_func="sigmoid", use_bias=True, bias_cast="to_scores",
        topk_method="argsort_neg", norm_topk_prob=bool(cfg.moe_renormalize),
        norm_eps=1e-20, norm_min_top_k=2,
        routed_scaling_factor=float(cfg.routed_scaling_factor), index_dtype="int32",
    )
    ids, wts, chosen = _route_reference(
        logits, spec, w["block_sparse_moe.gate.e_score_correction_bias"], with_scores=True
    )
    return Routing(ids, wts, chosen)


def restrict(routing: Routing, present, cfg: K31Config, *, choice: mx.array | None = None):
    """Restrict each token's top-k to the present experts and renormalise.

    Returns (slot_weights [T, K] f32 with zeros on absent slots, kept [T] f32):
    ``kept`` is the share of the router's original renormalised top-k weight
    whose experts are present. Kept slots are renormalised to sum to the
    original routed_scaling_factor (moe_renormalize semantics); a token with
    no present expert gets an all-zero row.
    """
    present = mx.array(sorted(int(e) for e in present), dtype=mx.int32)
    is_present = mx.any(routing.ids[..., None] == present, axis=-1)
    base = routing.weights / cfg.routed_scaling_factor
    kept = mx.sum(mx.where(is_present, base, 0), axis=-1)
    raw = mx.where(is_present, routing.scores, 0)
    if cfg.moe_renormalize:
        raw = raw / (mx.sum(raw, -1, keepdims=True) + 1e-20)
    return raw * cfg.routed_scaling_factor, kept


@dataclass
class Expert:
    """One routed expert in MLX mxfp4 layout (bit-identical to the checkpoint)."""

    w1: tuple   # (packed uint32 [I, H/8], scales uint8 [I, H/32]) gate
    w2: tuple   # down [H, I]
    w3: tuple   # up

    @classmethod
    def from_compressed(cls, tensors: dict, prefix: str = "") -> "Expert":
        def one(name):
            packed = tensors[f"{prefix}{name}.weight_packed"]
            scale = tensors[f"{prefix}{name}.weight_scale"]
            if packed.dtype != mx.uint8 or scale.dtype != mx.uint8:
                raise ValueError(f"{name}: expected U8 packed/scale, got {packed.dtype}/{scale.dtype}")
            if packed.shape[-1] * 2 != scale.shape[-1] * MXFP4_GROUP or packed.shape[-1] % 4:
                raise ValueError(f"{name}: shapes {packed.shape}/{scale.shape} are not mxfp4 g32")
            return mx.contiguous(packed).view(mx.uint32), scale
        return cls(one("w1"), one("w2"), one("w3"))


def mxfp4_matmul(x: mx.array, wq: tuple) -> mx.array:
    if _on_cpu():
        return _cpu_fp32_matmul(x, mx.dequantize(wq[0], wq[1], None, group_size=MXFP4_GROUP,
                                                 bits=4, mode="mxfp4"))
    return mx.quantized_matmul(x, wq[0], wq[1], None, transpose=True,
                               group_size=MXFP4_GROUP, bits=4, mode="mxfp4")


def mxfp4_dequantize_reference(packed_u8: np.ndarray, scale_u8: np.ndarray) -> np.ndarray:
    """compressed-tensors mxfp4-pack-quantized dequant, independent of MLX.

    Low nibble = even column, high nibble = odd column; E2M1 code with sign
    in bit 3; one E8M0 scale 2^(s-127) per 32 consecutive columns.
    """
    codes = np.stack([packed_u8 & 0xF, packed_u8 >> 4], -1).reshape(packed_u8.shape[0], -1)
    scale = np.exp2(scale_u8.astype(np.float32) - 127.0)
    return FP4_E2M1[codes] * np.repeat(scale, MXFP4_GROUP, axis=-1)


EXPERT_Q2 = {"bits": 2, "group_size": 64, "mode": "affine"}


def expert_matmul(x: mx.array, wq: tuple) -> mx.array:
    """An expert matrix is either shipped MXFP4 ``(packed, scales)`` or a derived
    affine-q2 ``(packed, scales, biases)`` (see ``expert_to_q2``)."""
    if len(wq) == 2:
        return mxfp4_matmul(x, wq)
    gs, bits = EXPERT_Q2["group_size"], EXPERT_Q2["bits"]
    if _on_cpu():
        return _cpu_fp32_matmul(x, mx.dequantize(wq[0], wq[1], wq[2], group_size=gs, bits=bits))
    return mx.quantized_matmul(x, wq[0], wq[1], wq[2], transpose=True, group_size=gs, bits=bits)


def expert_to_q2(e: Expert) -> Expert:
    """Requantize a shipped MXFP4 expert to MLX affine q2 (group 64).

    Approximate by construction: an experimental storage tier, never the
    checkpoint's own bytes."""
    def one(wq):
        w = mx.dequantize(wq[0], wq[1], None, group_size=MXFP4_GROUP, bits=4, mode="mxfp4")
        return tuple(mx.quantize(w.astype(mx.bfloat16), group_size=EXPERT_Q2["group_size"],
                                 bits=EXPERT_Q2["bits"]))
    return Expert(one(e.w1), one(e.w2), one(e.w3))


def _expert_out(e: Expert, x: mx.array, cfg: K31Config) -> mx.array:
    a = situ(expert_matmul(x, e.w1), expert_matmul(x, e.w3), cfg.situ_beta, cfg.situ_linear_beta)
    return expert_matmul(a, e.w2)


def routed_experts(experts: dict, xl: mx.array, routing: Routing, slot_w: mx.array,
                   cfg: K31Config, rows: int = 4096) -> mx.array:
    """Sum_k slot_w[t, k] * expert_{ids[t, k]}(xl[t]) over present experts.

    Token chunks bound the [rows, K, latent] slot buffer (~0.5 GB at 4096 rows
    and latent 3584); within a chunk every expert runs once on its tokens and
    one scatter places the outputs. As in the vendor moe_infer, expert outputs
    stay in the activation dtype, are weighted in FP32, summed over slots in
    top-k order and cast back.
    """
    t, k = routing.ids.shape
    ids_np = np.array(routing.ids)
    present = set(experts)
    outs = []
    for r0 in range(0, t, rows):
        r1 = min(t, r0 + rows)
        chunk = ids_np[r0:r1]
        pos, parts = [], []
        for e in sorted(present.intersection(np.unique(chunk).tolist())):
            tok, slot = np.nonzero(chunk == e)
            parts.append(_expert_out(experts[e], xl[mx.array(r0 + tok)], cfg))
            pos.append(tok * k + slot)
        slot_out = mx.zeros(((r1 - r0) * k, xl.shape[-1]), xl.dtype)
        if parts:
            slot_out[mx.array(np.concatenate(pos))] = mx.concatenate(parts)
        slot_out = slot_out.reshape(r1 - r0, k, -1).astype(mx.float32)
        outs.append(mx.sum(slot_out * slot_w[r0:r1, :, None], axis=1).astype(xl.dtype))
        mx.eval(outs[-1])
    return outs[0] if len(outs) == 1 else mx.concatenate(outs)


def routed_full(experts: dict, xl: mx.array, routing: Routing, cfg: K31Config) -> mx.array:
    """Ordinary route: every selected expert must be present (fails closed)."""
    missing = set(np.array(routing.ids).reshape(-1).tolist()) - set(experts)
    if missing:
        raise KeyError(f"full route needs every selected expert; missing {sorted(missing)[:8]}"
                       f"{'...' if len(missing) > 8 else ''} ({len(missing)} total)")
    return routed_experts(experts, xl, routing, routing.weights, cfg)


def routed_restricted(experts: dict, xl: mx.array, routing: Routing, cfg: K31Config):
    """Present-expert route: absent slots dropped, kept slots renormalised."""
    slot_w, kept = restrict(routing, experts.keys(), cfg)
    return routed_experts(experts, xl, routing, slot_w, cfg), kept


def latent_moe(w: dict, x: mx.array, cfg: K31Config, routing: Routing, experts: dict, *,
               full_route: bool, quant: dict = Q8):
    """down_proj -> routed experts (latent) -> weighted sum -> RMSNorm -> up_proj, + shared."""
    xl = linear(w, "block_sparse_moe.routed_expert_down_proj", x, quant)
    if full_route:
        y, kept = routed_full(experts, xl, routing, cfg), mx.ones((x.shape[0],), mx.float32)
    else:
        y, kept = routed_restricted(experts, xl, routing, cfg)
    if cfg.latent_moe_use_norm:
        y = rms_norm(y, w["block_sparse_moe.routed_expert_norm.weight"], cfg.rms_norm_eps)
    y = linear(w, "block_sparse_moe.routed_expert_up_proj", y, quant)
    if cfg.num_shared_experts:
        y = y + situ_mlp(w, "block_sparse_moe.shared_experts", x, cfg, quant)
    return y, kept


# --------------------------------------------------------------------------
# layer and model


@dataclass
class ResidualState:
    """Everything a later layer needs from earlier ones in a prompt forward."""

    tokens: mx.array
    blocks: list = field(default_factory=list)   # banked [T, D] block inputs
    partial: mx.array | None = None              # current block partial sum [T, D]
    next_layer: int = 0
    pending: dict | None = None                  # attention half of next_layer, once routed
    cache: dict | None = None                    # layer -> attention cache (KDA state / MLA latent)
    offset: int = 0                              # absolute position of tokens[0]


def attention_half(w: dict, i: int, state: ResidualState, cfg: K31Config, quant: dict = Q8) -> dict:
    """Run layer i up to the FFN input. Does not mutate ``state``."""
    eps = cfg.rms_norm_eps
    blocks, partial = list(state.blocks), state.partial
    h = attn_res_mix(blocks, partial, w["self_attention_res_norm.weight"],
                     w["self_attention_res_proj.weight"], eps)
    if i % cfg.attn_res_block_size == 0:
        blocks.append(partial)
        partial = None
    xn = rms_norm(h, w["input_layernorm.weight"], eps)
    sparse = False
    cache = None if state.cache is None else state.cache.setdefault(i, {})
    if cache is None and state.offset:
        raise ValueError("a nonzero offset needs the attention cache of the earlier tokens")
    if i in cfg.kda_layers:
        y = kda_attention(w, xn, cfg, quant, cache=cache)
    else:
        y, sparse = mla_attention(w, xn, cfg, quant, dsa=i in cfg.dsa_layers, cache=cache,
                                  offset=state.offset)
    partial = y if partial is None else partial + y
    h = attn_res_mix(blocks, partial, w["mlp_res_norm.weight"], w["mlp_res_proj.weight"], eps)
    ffn_in = rms_norm(h, w["post_attention_layernorm.weight"], eps)
    return {"layer": i, "blocks": blocks, "partial": partial, "ffn_in": ffn_in,
            "attention": "kda" if i in cfg.kda_layers else ("dsa_sparse" if sparse else "mla_dense")}


def ffn_half(w: dict, half: dict, cfg: K31Config, *, experts: dict | None, routing: Routing | None,
             full_route: bool, quant: dict = Q8):
    i, x = half["layer"], half["ffn_in"]
    if not cfg.is_moe(i):
        return situ_mlp(w, "mlp", x, cfg, quant), None
    if experts is None:
        raise ValueError(f"layer {i} is MoE: pass the present experts (an explicit {{}} runs "
                         "with no routed experts)")
    return latent_moe(w, x, cfg, routing, experts, full_route=full_route, quant=quant)


def output_logits(g: dict, state: ResidualState, cfg: K31Config, quant: dict = Q8,
                  last_only: bool = True) -> mx.array:
    blocks, partial = state.blocks, state.partial
    if last_only:  # per-token mix: slicing first is exact
        blocks, partial = [b[-1:] for b in blocks], partial[-1:]
    h = attn_res_mix(blocks, partial, g["output_attn_res_norm.weight"],
                     g["output_attn_res_proj.weight"], cfg.rms_norm_eps)
    h = rms_norm(h, g["norm.weight"], cfg.rms_norm_eps)
    return linear(g, "lm_head", h, quant).astype(mx.float32)


def begin(g: dict, tokens, cfg: K31Config, dtype=mx.bfloat16, quant: dict = Q8,
          cache: dict | None = None, offset: int = 0) -> ResidualState:
    """Start a forward over ``tokens`` at absolute position ``offset``.

    Pass the same ``cache`` dict to every chunk of one sequence (prompt, then
    each decode step); an empty dict starts a new cached sequence. Attention
    residual state is per token, so nothing else crosses steps."""
    tokens = mx.array(tokens, dtype=mx.int32).reshape(-1)
    return ResidualState(tokens=tokens, partial=embed(g, "embed_tokens", tokens, dtype, quant),
                         cache=cache, offset=int(offset))
