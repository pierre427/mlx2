# SPDX-License-Identifier: MIT
"""MiMo-V2 text decoder math for layer-at-a-time execution; provenance/mimo-v2-layer-stream.json.

Equations follow Xiaomi's reference ``modeling_mimo_v2.py`` shipped with
XiaomiMiMo/MiMo-V2.6-Pro-RL@a066ba00 (Apache-2.0 header): the text backbone
only (no audio, vision, MTP or DFlash). Every function is a pure function of
a flat per-layer weight mapping in HF names with the ``model.layers.<l>.``
prefix stripped and q/k/v already split (see ``mimo_v2_weights``), so one
decoder layer can be loaded, run and released on its own. Forward only, one
prompt per row, positions ``0..T-1``, no KV cache.

What the reference does, and this module repeats:

* hybrid attention per ``hybrid_layer_pattern`` (1 = sliding window, 0 =
  full); each kind has its own head counts, q/k and v head dims, RoPE base and
  optional per-head attention sink logit; values are multiplied by
  ``attention_value_scale`` before attention; partial RoPE on the first
  ``int(head_dim * partial_rotary_factor)`` dims (rotate-half, not
  interleaved); scores ``q k^T * head_dim^-0.5``; window ``q - window < kv <= q``;
  softmax in float32 over ``[scores, sink]`` with the sink column dropped.
* MiMoV2RMSNorm: float32 normalize, cast back, then times the weight.
* ``moe_layer_freq`` picks MoE layers; others are dense SwiGLU MLPs.
* router: float32 ``x W^T``, sigmoid, ``e_score_correction_bias`` added for
  the choice only (noaux_tc), group-limited top-k (top-2 sum per group), the
  gathered *unbiased* sigmoid scores, ``/ (sum + 1e-20)`` when
  ``norm_topk_prob``, times ``routed_scaling_factor`` (1.0 when null). No
  shared experts exist in the reference.
* expert outputs are combined in float32 and cast back to the hidden dtype.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .mimo_v2_weights import MXFP4_GROUP


@dataclass(frozen=True)
class AttnSpec:
    sliding: bool
    heads: int
    kv_heads: int
    head_dim: int
    v_head_dim: int
    rope_dim: int
    rope_theta: float
    sink: bool
    window: int | None


@dataclass(frozen=True)
class MiMoV2Config:
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    eps: float
    intermediate_size: int
    moe_intermediate_size: int
    n_routed_experts: int | None
    top_k: int
    n_group: int
    topk_group: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    value_scale: float | None
    layer_pattern: tuple          # 1 = sliding window attention
    moe_layers: tuple             # bool per layer
    full: AttnSpec
    swa: AttnSpec
    projection_layout: str
    tie_word_embeddings: bool

    @classmethod
    def from_dict(cls, c: dict) -> "MiMoV2Config":
        refuse = []
        if c.get("model_type") != "mimo_v2":
            refuse.append(f"model_type={c.get('model_type')!r}")
        if c.get("hidden_act", "silu") != "silu":
            refuse.append(f"hidden_act={c.get('hidden_act')!r}")
        if c.get("attention_bias", False):
            refuse.append("attention_bias")
        n_exp = c.get("n_routed_experts")
        if n_exp is not None:
            if c.get("scoring_func", "sigmoid") != "sigmoid":
                refuse.append(f"scoring_func={c.get('scoring_func')!r}")
            if c.get("topk_method", "noaux_tc") != "noaux_tc":
                refuse.append(f"topk_method={c.get('topk_method')!r}")
        rope = c.get("rope_parameters") or c.get("rope_scaling") or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            refuse.append(f"rope_type={rope.get('rope_type')!r}")
        prf = float(c.get("partial_rotary_factor", 1.0))
        if float(rope.get("partial_rotary_factor", prf)) != prf:
            # The reference sizes cos/sin from rope_parameters and splits q/k
            # by config.partial_rotary_factor; unequal values cannot run there.
            refuse.append("rope_parameters.partial_rotary_factor differs from partial_rotary_factor")
        if c.get("attention_projection_layout", "split") not in ("split", "fused_qkv"):
            refuse.append(f"attention_projection_layout={c.get('attention_projection_layout')!r}")
        if refuse:
            raise ValueError("MiMo-V2 layer stream refuses: " + "; ".join(refuse))

        n = c["num_hidden_layers"]
        heads = c["num_attention_heads"]
        kv = c.get("num_key_value_heads") or heads
        hd = c.get("head_dim") or c["hidden_size"] // heads
        vhd = c.get("v_head_dim") or hd
        theta = float(rope.get("rope_theta", c.get("rope_theta", 10000.0)))

        def spec(sliding, h, k, d, v, base, sink, window):
            rd = int(d * prf)
            if rd % 2:
                raise ValueError(f"odd rotary dim {rd} (head_dim={d}, partial_rotary_factor={prf})")
            return AttnSpec(sliding, h, k, d, v, rd, base, bool(sink), window)

        full = spec(False, heads, kv, hd, vhd, theta, c.get("add_full_attention_sink_bias", False), None)
        s_heads = c.get("swa_num_attention_heads") or heads
        s_hd = c.get("swa_head_dim") or hd
        swa = spec(True, s_heads, c.get("swa_num_key_value_heads") or kv, s_hd,
                   c.get("swa_v_head_dim") or s_hd, float(c.get("swa_rope_theta") or theta),
                   c.get("add_swa_attention_sink_bias", False),
                   c.get("sliding_window") or c.get("sliding_window_size"))
        pattern = c.get("hybrid_layer_pattern")
        if pattern is None and c.get("hybrid_block_size") is not None:
            b = c["hybrid_block_size"]
            pattern = [0 if (i + 1) % b == 0 else 1 for i in range(n)]
        pattern = tuple(int(p) for p in (pattern or [0] * n))
        freq = c.get("moe_layer_freq")
        if isinstance(freq, int):
            freq = [freq > 0 and i % freq == 0 for i in range(n)]
        moe = tuple(bool(f) and n_exp is not None for f in (freq or [False] * n))
        if len(pattern) != n or len(moe) != n:
            raise ValueError("hybrid_layer_pattern / moe_layer_freq length must equal num_hidden_layers")
        if any(pattern) and not swa.window:
            raise ValueError("sliding layers without sliding_window")
        if n_exp is not None and n_exp % (c.get("n_group") or 1):
            raise ValueError("n_routed_experts not divisible by n_group")
        return cls(
            hidden_size=c["hidden_size"], num_hidden_layers=n, vocab_size=c["vocab_size"],
            eps=float(c.get("layernorm_epsilon", 1e-6)), intermediate_size=c["intermediate_size"],
            moe_intermediate_size=c.get("moe_intermediate_size") or c["intermediate_size"],
            n_routed_experts=n_exp, top_k=c.get("num_experts_per_tok") or 0,
            n_group=c.get("n_group") or 1, topk_group=c.get("topk_group") or 1,
            norm_topk_prob=bool(c.get("norm_topk_prob", True)),
            routed_scaling_factor=float(c.get("routed_scaling_factor") or 1.0),
            value_scale=c.get("attention_value_scale"), layer_pattern=pattern, moe_layers=moe,
            full=full, swa=swa, projection_layout=c.get("attention_projection_layout", "split"),
            tie_word_embeddings=bool(c.get("tie_word_embeddings", False)))

    def attn(self, layer: int) -> AttnSpec:
        return self.swa if self.layer_pattern[layer] == 1 else self.full

    def is_moe(self, layer: int) -> bool:
        return self.moe_layers[layer]


# ---------------------------------------------------------------- primitives

def rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    xf = x.astype(mx.float32)
    normed = (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)).astype(x.dtype)
    return weight * normed


def rope_cos_sin(T: int, rope_dim: int, base: float, dtype) -> tuple:
    """Default RoPE as the reference builds it: float32 ``pos x inv_freq``, ``cat(f, f)``, cast."""
    inv = 1.0 / (np.float32(base) ** (np.arange(0, rope_dim, 2, dtype=np.int64).astype(np.float32)
                                     / np.float32(rope_dim)))
    freqs = mx.arange(T, dtype=mx.float32)[:, None] * mx.array(inv.astype(np.float32))[None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


def _rotate_half(x: mx.array) -> mx.array:
    h = x.shape[-1] // 2
    return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def apply_partial_rope(x: mx.array, cos: mx.array, sin: mx.array, rope_dim: int) -> mx.array:
    """``x`` [B, H, T, D]: rotate-half RoPE on ``x[..., :rope_dim]``, the rest passes through."""
    r, rest = x[..., :rope_dim], x[..., rope_dim:]
    return mx.concatenate([r * cos + _rotate_half(r) * sin, rest], axis=-1)


def linear(x: mx.array, w: mx.array) -> mx.array:
    return x @ w.T


def attention_mask(T: int, window: int | None) -> mx.array:
    """Additive [T, T] mask: causal, and ``kv > q - window`` for a sliding layer."""
    q = mx.arange(T)[:, None]
    k = mx.arange(T)[None, :]
    keep = k <= q
    if window is not None:
        keep = keep & (k > q - window)
    return mx.where(keep, mx.array(0.0, mx.float32), mx.array(-mx.inf, mx.float32))


def attention(w: dict, x: mx.array, spec: AttnSpec, cfg: MiMoV2Config, *,
              row_chunk: int | None = None) -> mx.array:
    """Self attention of one layer (input already normalized). ``x`` [B, T, d]."""
    B, T, _ = x.shape
    q = linear(x, w["self_attn.q_proj.weight"]).reshape(B, T, spec.heads, spec.head_dim).transpose(0, 2, 1, 3)
    k = linear(x, w["self_attn.k_proj.weight"]).reshape(B, T, spec.kv_heads, spec.head_dim).transpose(0, 2, 1, 3)
    v = linear(x, w["self_attn.v_proj.weight"]).reshape(B, T, spec.kv_heads, spec.v_head_dim).transpose(0, 2, 1, 3)
    if cfg.value_scale is not None:
        v = v * cfg.value_scale
    cos, sin = rope_cos_sin(T, spec.rope_dim, spec.rope_theta, x.dtype)
    q = apply_partial_rope(q, cos, sin, spec.rope_dim)
    k = apply_partial_rope(k, cos, sin, spec.rope_dim)
    groups = spec.heads // spec.kv_heads
    if groups > 1:
        k = mx.repeat(k, groups, axis=1)
        v = mx.repeat(v, groups, axis=1)
    mask = attention_mask(T, spec.window if spec.sliding else None)
    sink = w.get("self_attn.attention_sink_bias") if spec.sink else None
    if spec.sink and sink is None:
        raise ValueError("layer declares an attention sink but has no attention_sink_bias")
    scale = spec.head_dim ** -0.5
    step = row_chunk or T
    outs = []
    for r0 in range(0, T, step):
        r1 = min(T, r0 + step)
        s = (q[:, :, r0:r1] @ k.transpose(0, 1, 3, 2)) * scale
        s = s.astype(mx.float32) + mask[r0:r1]
        if sink is not None:
            col = mx.broadcast_to(sink.astype(mx.float32).reshape(1, -1, 1, 1), (B, spec.heads, r1 - r0, 1))
            s = mx.concatenate([s, col], axis=-1)
        p = mx.softmax(s - mx.max(s, axis=-1, keepdims=True), axis=-1).astype(x.dtype)
        if sink is not None:
            p = p[..., :-1]
        outs.append(p @ v)
    o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)
    o = o.transpose(0, 2, 1, 3).reshape(B, T, spec.heads * spec.v_head_dim)
    return linear(o, w["self_attn.o_proj.weight"])


def dense_mlp(w: dict, x: mx.array, prefix: str = "mlp.") -> tuple:
    """SwiGLU MLP. Returns ``(out, act)`` where ``act`` is the down_proj input."""
    act = nn_silu(linear(x, w[prefix + "gate_proj.weight"])) * linear(x, w[prefix + "up_proj.weight"])
    return linear(act, w[prefix + "down_proj.weight"]), act


def nn_silu(x: mx.array) -> mx.array:
    """SiLU evaluated in float32 and rounded once (torch's opmath for bf16 inputs)."""
    xf = x.astype(mx.float32)
    return (xf * mx.sigmoid(xf)).astype(x.dtype)


# ---------------------------------------------------------------- router

def route(w: dict, x: mx.array, cfg: MiMoV2Config) -> tuple:
    """``(inds int32 [B,T,k], weights float32 [B,T,k])`` exactly as MiMoV2MoEGate.

    Slots are ordered by descending biased choice score (the reference's
    ``topk(sorted=False)`` order is unspecified; compare as sets)."""
    B, T, d = x.shape
    E, k = cfg.n_routed_experts, cfg.top_k
    logits = x.reshape(-1, d).astype(mx.float32) @ w["mlp.gate.weight"].astype(mx.float32).T
    scores = mx.sigmoid(logits)
    choice = scores + w["mlp.gate.e_score_correction_bias"].astype(mx.float32)[None, :]
    if cfg.n_group > 1:
        g = choice.reshape(-1, cfg.n_group, E // cfg.n_group)
        top2 = mx.sort(g, axis=-1)[..., -2:].sum(-1)                         # [N, n_group]
        keep_g = mx.argsort(-top2, axis=-1)[:, :cfg.topk_group]
        gmask = mx.zeros(top2.shape, mx.bool_)
        gmask = mx.put_along_axis(gmask, keep_g, mx.array(True), axis=-1)
        emask = mx.repeat(gmask, E // cfg.n_group, axis=-1)
        choice = mx.where(emask, choice, mx.array(-mx.inf, mx.float32))
    inds = mx.argsort(-choice, axis=-1)[:, :k]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if k > 1 and cfg.norm_topk_prob:
        weights = weights / (weights.sum(-1, keepdims=True) + 1e-20)
    weights = weights * cfg.routed_scaling_factor
    return inds.astype(mx.int32).reshape(B, T, k), weights.reshape(B, T, k)


# ---------------------------------------------------------------- experts

@dataclass
class ExpertBank:
    """Routed experts held for one layer.

    ``ids`` are the global expert ids present (sorted); ``gate``/``up``/``down``
    are either MXFP4 ``(packed uint32 [n, out, in/8], scales uint8 [n, out, in/32])``
    or one float array ``(w [n, out, in],)``. The loop path also accepts per-expert
    lists in place of the stacked arrays (no stacking copy)."""

    ids: np.ndarray
    gate: tuple
    up: tuple
    down: tuple

    def local(self, inds: mx.array, n_experts: int) -> mx.array:
        if len(self.ids) == n_experts and np.array_equal(self.ids, np.arange(n_experts)):
            return inds
        table = np.full(n_experts, -1, np.int32)
        table[self.ids] = np.arange(len(self.ids), dtype=np.int32)
        loc = mx.array(table)[inds]
        missing = np.setdiff1d(np.unique(np.array(inds)), self.ids)
        if missing.size:
            raise KeyError(f"routed experts not loaded: {missing[:8].tolist()}")
        return loc


def _matmul_slots(x: mx.array, wq: tuple, rhs: mx.array) -> mx.array:
    """``x`` [N, 1, 1, in] (or [N, k, 1, in]) times expert ``rhs`` [N, k] -> [N, k, 1, out]."""
    if len(wq) == 2:
        return mx.gather_qmm(x, wq[0], wq[1], None, rhs_indices=rhs, transpose=True,
                             group_size=MXFP4_GROUP, bits=4, mode="mxfp4")
    return mx.gather_mm(x, wq[0].swapaxes(-1, -2), rhs_indices=rhs)


def expert_rows_gather(bank: ExpertBank, x: mx.array, inds: mx.array, n_experts: int) -> mx.array:
    """Per-slot expert outputs [B,T,k,d] with gather kernels (all of it lazy; the GPU path)."""
    B, T, d = x.shape
    k = inds.shape[-1]
    rhs = bank.local(inds, n_experts).reshape(B * T, k)
    xs = x.reshape(B * T, 1, 1, d)
    g = _matmul_slots(xs, bank.gate, rhs)
    u = _matmul_slots(xs, bank.up, rhs)
    a = nn_silu(g) * u                                                   # [N, k, 1, I]
    y = _matmul_slots(a, bank.down, rhs)                                 # [N, k, 1, d]
    return y.reshape(B, T, k, d)


def _dense_of(wq: tuple, e: int, dtype) -> mx.array:
    if len(wq) == 2:
        return mx.dequantize(wq[0][e], wq[1][e], None, group_size=MXFP4_GROUP, bits=4,
                             mode="mxfp4").astype(dtype)
    return wq[0][e].astype(dtype)


def expert_rows_loop(bank: ExpertBank, x: mx.array, inds: mx.array, n_experts: int) -> mx.array:
    """Per-slot expert outputs [B,T,k,d], one dequantized expert at a time (bounded CPU memory)."""
    B, T, d = x.shape
    k = inds.shape[-1]
    loc = np.array(bank.local(inds, n_experts)).reshape(-1)              # [N*k]
    xs = x.reshape(B * T, d)
    rows = np.zeros((B * T * k, d), np.float32)
    for e in np.unique(loc):
        slots = np.nonzero(loc == e)[0]
        xe = xs[mx.array(slots // k)]
        g = linear(xe, _dense_of(bank.gate, int(e), x.dtype))
        u = linear(xe, _dense_of(bank.up, int(e), x.dtype))
        y = linear(nn_silu(g) * u, _dense_of(bank.down, int(e), x.dtype))
        mx.eval(y)
        rows[slots] = np.array(y.astype(mx.float32))
    return mx.array(rows).astype(x.dtype).reshape(B, T, k, d)


def combine(rows: mx.array, weights: mx.array, dtype) -> mx.array:
    """The reference's float32 ``index_add`` of ``expert_out * weight``, cast to the hidden dtype."""
    return (rows.astype(mx.float32) * weights[..., None]).sum(axis=-2).astype(dtype)


# ---------------------------------------------------------------- layer

def attention_half(w: dict, layer: int, h: mx.array, cfg: MiMoV2Config,
                   row_chunk: int | None = None) -> tuple:
    """Residual attention sub-block. Returns ``(h_after_attention, mlp_input)``."""
    a = attention(w, rms_norm(h, w["input_layernorm.weight"], cfg.eps), cfg.attn(layer), cfg,
                  row_chunk=row_chunk)
    h = h + a
    return h, rms_norm(h, w["post_attention_layernorm.weight"], cfg.eps)


def decoder_layer(w: dict, layer: int, h: mx.array, cfg: MiMoV2Config, *,
                  experts=None, rows_fn=expert_rows_gather, row_chunk: int | None = None) -> tuple:
    """One decoder layer. ``experts`` is an ``ExpertBank`` or a callable ``inds -> ExpertBank``
    (lazy loading of only the routed experts). Returns ``(h_next, payload)``."""
    h, x = attention_half(w, layer, h, cfg, row_chunk=row_chunk)
    if not cfg.is_moe(layer):
        out, act = dense_mlp(w, x)
        return h + out, {"act": act, "x": x}
    inds, weights = route(w, x, cfg)
    bank = experts(inds) if callable(experts) else experts
    if bank is None:
        raise ValueError(f"layer {layer} is MoE but no experts were provided")
    rows = rows_fn(bank, x, inds, cfg.n_routed_experts)
    return h + combine(rows, weights, h.dtype), {"inds": inds, "w": weights, "rows": rows, "x": x}


def final_logits(norm_w: mx.array, head_w: mx.array, h: mx.array, cfg: MiMoV2Config) -> mx.array:
    return linear(rms_norm(h, norm_w, cfg.eps), head_w)
