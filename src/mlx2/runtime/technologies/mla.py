# SPDX-License-Identifier: MIT
# MultiLinear, QuantizedMultiLinear and absorbed_max_query were mined from
# local mlx-lm-unified mla.py (MIT) via xing4_0.py; see
# provenance/xing4-0-model.json ("moved_to"). The rest is original.
"""Multi-head latent attention (MLA): one reference for every MLA model.

MLA caches one normalized latent row ``c`` (``kv_lora_rank`` wide) and one
shared positional key row ``k_pe`` (``qk_rope_head_dim`` wide, rotated under
RoPE, raw under NoPE) per token. Per head ``h``, ``kv_b_proj`` expands the
latent into a NoPE key ``W_uk[h] c`` and a value ``W_uv[h] c``. Attention can
run in two algebraically equal forms:

* **expanded**: build per-head keys/values from the latent, then ordinary
  multi-head attention (cheap for wide prompt chunks over a short prefix);
* **absorbed**: fold ``W_uk`` into the query (``q_latent = q_nope W_uk``),
  attend one latent head whose key and value are both the latent (plus the
  positional channels), then unabsorb the output through ``W_uv`` (cheap for
  decode and verify rows: the latent is read once for every head).

This module owns that math. A model builds an :class:`MLASpec` (usually via
:meth:`MLASpec.from_config`) and calls the reference that matches its cache
layout; the different historical op orders that model identities depend on
(how the positional term enters the score, which SDPA dispatcher runs, the
FP32 softmax form) are spec fields, never model-name branches. Caches are not
unified: callers pass the dense latent/positional views they already fetch
(a "latent source view"), and sparse models pass their selected rows or
masks from the sparse-selection technology.

Fast Metal paths are registered in :data:`ABSORBED_DECODE`,
:data:`EXPANDED_PREFILL` and :data:`LATENT_GATHERED` with ``admits`` over
geometry, dtype and device; see ``docs/technologies/mla.md``.
"""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

import mlx.core as mx
from mlx import nn

from .dispatch import FastPath, KernelRegistry, spec_from_config

# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------

POSITION_MODES = ("rope", "nope")
# How the positional (rope/shared) channel enters the attention score.
#   score_bias:   (q_pe*scale) @ k_pe^T is passed to SDPA as an additive mask
#                 (Xing, Ling);
#   concat_key:   k_pe is concatenated onto every head's key (Kimi expanded,
#                 Kimi K3.1);
#   split_fp32:   the latent and positional dot products are computed
#                 separately and summed in FP32 (Kimi latent laws);
#   fused_buffer: the cache stores latent || pe as one buffer which is both
#                 key and value; trailing pe value lanes are dropped
#                 (HY-V4-Flash, GLM-5.3).
PE_FORMS = ("score_bias", "concat_key", "split_fp32", "fused_buffer")
# Which absorbed layout a score_bias model uses.
#   fold_heads: [B,H,L,D] queries attend as [B,1,L*H,D] (Xing);
#   broadcast:  SDPA broadcasts the one latent head over H query heads (Ling).
ABSORBED_LAYOUTS = ("fold_heads", "broadcast")
# Which SDPA implementation runs the score_bias forms.
#   fast: mx.fast.scaled_dot_product_attention directly (Xing: a cache's
#         bucketed hook reads cache.values, which is k_pe for an MLA cache);
#   base: models.base.scaled_dot_product_attention(cache=...) with its
#         cache hooks and decode kernels (Ling).
SDPA_ROUTES = ("fast", "base")
# Absorbed-versus-expanded selection rule.
#   cost_model:  absorbed while L <= absorbed_max_query(...) (Xing);
#   single_row:  absorbed for one-row forwards only (Ling default);
#   policy:      the model's own serving policy decides (Kimi auto/latent_*);
#   always:      absorbed only (HY, GLM-5.3: weights are folded, no kv_b);
#   never:       expanded only (Kimi K3.1).
ABSORBED_RULES = ("cost_model", "single_row", "policy", "always", "never")
# yarn mscale application.
#   nondefault_rope_type: only when rope_scaling names a non-default type
#                         (Xing, HF DeepseekV3 parity);
#   any_rope_type:        whenever mscale_all_dim and factor > 1 (Ling);
#   none:                 never (Kimi, K3.1, HY, GLM).
MSCALE_RULES = ("nondefault_rope_type", "any_rope_type", "none")
# Float order of the yarn mscale product: scale*s*s ("sequential", Xing) or
# scale*(s*s) ("squared", Ling). These round differently in float64.
MSCALE_ORDERS = ("sequential", "squared")


@dataclass(frozen=True)
class MLASpec:
    n_heads: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    q_lora_rank: Optional[int] = None
    position: str = "rope"
    sinks: bool = False
    # Softmax scale. ``from_config`` derives it from the head dims and yarn
    # mscale; an explicit value wins.
    softmax_scale: Optional[float] = None
    mscale_rule: str = "none"
    mscale_order: str = "sequential"
    absorbed_rule: str = "single_row"
    absorbed_layout: str = "broadcast"
    pe_form: str = "score_bias"
    # The expanded form's positional law (score_bias or concat_key); the
    # absorbed form's is ``pe_form``.
    expanded_pe_form: str = "score_bias"
    sdpa_route: str = "fast"
    # Opt-in: short multi-row forwards (<= absorbed_verify_rows) take the
    # absorbed formula under the single_row rule (Ling attribution law).
    absorbed_verify_rows: int = 0

    def __post_init__(self):
        for name, allowed in (("position", POSITION_MODES), ("pe_form", PE_FORMS),
                              ("expanded_pe_form", ("score_bias", "concat_key")),
                              ("absorbed_layout", ABSORBED_LAYOUTS), ("sdpa_route", SDPA_ROUTES),
                              ("absorbed_rule", ABSORBED_RULES), ("mscale_rule", MSCALE_RULES),
                              ("mscale_order", MSCALE_ORDERS)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"MLASpec.{name}={getattr(self, name)!r} not in {allowed}")
        if self.softmax_scale is None:
            object.__setattr__(self, "softmax_scale", self.qk_head_dim ** -0.5)

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def absorbed_geometry(self) -> tuple[int, int, int]:
        return (self.kv_lora_rank, self.qk_nope_head_dim, self.v_head_dim)

    @classmethod
    def from_config(cls, config: Any, *, rope_scaling: Any = None, **overrides) -> "MLASpec":
        """Derive from HF DeepSeek-style fields.

        ``num_attention_heads``, ``kv_lora_rank``, ``q_lora_rank``,
        ``qk_nope_head_dim``, ``qk_rope_head_dim``, ``v_head_dim``;
        ``mla_use_nope`` selects NoPE; the softmax scale is
        ``qk_head_dim**-0.5`` times the yarn ``mscale**2`` under
        ``mscale_rule`` (``rope_scaling`` overrides the config's).
        """
        mapping = {
            "n_heads": "num_attention_heads",
            "kv_lora_rank": "kv_lora_rank",
            "q_lora_rank": "q_lora_rank",
            "qk_nope_head_dim": "qk_nope_head_dim",
            "qk_rope_head_dim": "qk_rope_head_dim",
            "v_head_dim": "v_head_dim",
            "position": lambda c: "nope" if _get(c, "mla_use_nope", False) else None,
        }
        explicit_scale = overrides.pop("softmax_scale", None)
        spec = spec_from_config(cls, config, mapping, **overrides)
        if explicit_scale is not None:
            return replace(spec, softmax_scale=float(explicit_scale))
        scaling = rope_scaling if rope_scaling is not None else _get(config, "rope_scaling", None)
        return replace(spec, softmax_scale=softmax_scale(
            spec.qk_head_dim, scaling, rule=spec.mscale_rule, order=spec.mscale_order))


def _get(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def yarn_mscale(scaling, rule: str) -> Optional[float]:
    """The yarn ``mscale`` the rule applies for ``rope_scaling``, or None."""
    if scaling is None or rule == "none":
        return None
    if rule == "nondefault_rope_type":
        rope_type = scaling.get("type") or scaling.get("rope_type", "default")
        mscale_all_dim = scaling.get("mscale_all_dim", 0)
        factor = scaling.get("factor", 1.0)
        if rope_type != "default" and mscale_all_dim and factor > 1:
            return 0.1 * mscale_all_dim * math.log(factor) + 1.0
        return None
    mscale_all_dim = scaling.get("mscale_all_dim", 0)
    factor = scaling.get("factor", 1)
    if mscale_all_dim and factor > 1:
        return 0.1 * mscale_all_dim * math.log(factor) + 1.0
    return None


def softmax_scale(qk_head_dim: int, scaling=None, *, rule: str = "none",
                  order: str = "sequential") -> float:
    scale = qk_head_dim ** -0.5
    s = yarn_mscale(scaling, rule)
    if s is None:
        return scale
    if order == "sequential":
        return scale * s * s
    scale *= s * s
    return scale


# --------------------------------------------------------------------------
# Absorbed-versus-expanded cost rule
# --------------------------------------------------------------------------

ABSORBED_UNBOUNDED = 1 << 30


def absorbed_max_query(kv_lora_rank, qk_nope_head_dim, v_head_dim, cache_len=None) -> int:
    """Largest query width L for which absorbed MLA is cheaper than expanded.

    absorbed: 2*L*S*r + L*r*D; expanded: S*r*D + L*S*D (D = dn + dv). S is the
    attended cache length; the asymptotic form (``cache_len=None``) overstates
    the limit for cold prefill.
    """
    d = qk_nope_head_dim + v_head_dim
    if cache_len is None:
        numerator = kv_lora_rank * d
        denom = 2 * kv_lora_rank - d
    else:
        numerator = cache_len * kv_lora_rank * d
        denom = cache_len * (2 * kv_lora_rank - d) + kv_lora_rank * d
    if denom <= 0:
        return ABSORBED_UNBOUNDED
    return max(1, numerator // denom)


def use_absorbed(spec: MLASpec, query_len: int, cache_len: int, *,
                 max_query_override: Optional[int] = None) -> bool:
    """Apply ``spec.absorbed_rule`` (``policy`` is the caller's decision)."""
    rule = spec.absorbed_rule
    if rule == "always":
        return True
    if rule == "never":
        return False
    if rule == "cost_model":
        limit = absorbed_max_query(*spec.absorbed_geometry, cache_len=cache_len)
        if max_query_override is not None:
            limit = max_query_override
        return query_len <= limit
    if rule == "single_row":
        return query_len == 1 or (spec.absorbed_verify_rows > 0
                                  and query_len <= spec.absorbed_verify_rows)
    raise ValueError("MLA absorbed_rule 'policy' is decided by the model's serving policy")


# --------------------------------------------------------------------------
# Absorb / unabsorb weights
# --------------------------------------------------------------------------


class MultiLinear(nn.Module):
    """Per-head linear with weight ``[H, out, in]``.

    ``embed_q = MultiLinear(dn, r, H)`` holds ``W_uk``: ``embed_q(q_nope)``
    absorbs it into the query and ``embed_q(latent, transpose=False)``
    expands NoPE keys. ``unembed_out = MultiLinear(r, dv, H)`` holds
    ``W_uv``. ``init="uniform"`` draws the usual Linear init; ``"zeros"``
    allocates zeros (weights always come from the checkpoint).

    This class is not quantizable; :class:`QuantizableMultiLinear` adds
    ``to_quantized`` for models whose loaders quantize the folded weights.
    """

    def __init__(self, input_dims: int, output_dims: int, num_heads: int, *, init: str = "uniform"):
        super().__init__()
        if init == "uniform":
            scale = math.sqrt(1.0 / input_dims)
            self.weight = mx.random.uniform(
                low=-scale, high=scale, shape=(num_heads, output_dims, input_dims)
            )
        elif init == "zeros":
            self.weight = mx.zeros((num_heads, output_dims, input_dims))
        else:
            raise ValueError(f"MultiLinear init {init!r}")

    def __call__(self, x, transpose=True):
        if transpose:
            return x @ self.weight.swapaxes(-1, -2)
        return x @ self.weight


class QuantizableMultiLinear(MultiLinear):
    def to_quantized(self, group_size: int, bits: int, mode: str = "affine"):
        num_heads, output_dims, input_dims = self.weight.shape
        ql = QuantizedMultiLinear(input_dims, output_dims, num_heads, group_size, bits, mode)
        ql.weight, ql.scales, *biases = mx.quantize(self.weight, group_size, bits, mode=mode)
        ql.biases = biases[0] if biases else None
        return ql


class QuantizedMultiLinear(nn.Module):
    def __init__(self, input_dims, output_dims, num_heads, group_size, bits, mode):
        super().__init__()
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        scale = math.sqrt(1 / input_dims)
        weight = mx.random.uniform(low=-scale, high=scale, shape=(num_heads, output_dims, input_dims))
        self.weight, self.scales, *biases = mx.quantize(weight, group_size, bits, mode=mode)
        self.biases = biases[0] if biases else None
        self.freeze()

    def __call__(self, x, transpose=True):
        return mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            biases=self.get("biases"),
            transpose=transpose,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )


def kv_b_absorb(q_nope, w, nope: int):
    """Absorb from a dense ``kv_b_proj`` weight ``w [H, dn+dv, r]``."""
    return q_nope @ w[:, :nope]


def kv_b_unabsorb(out, w, nope: int):
    """Unabsorb a latent output ``[.., r]`` through ``w [H, dn+dv, r]``."""
    return out @ w[:, nope:].swapaxes(-1, -2)


# --------------------------------------------------------------------------
# Reference attention
# --------------------------------------------------------------------------


def _sdpa(spec: MLASpec, q, k, v, *, scale, mask, cache=None):
    if spec.sdpa_route == "base":
        from ..models.base import scaled_dot_product_attention

        return scaled_dot_product_attention(q, k, v, cache=cache, scale=scale, mask=mask)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def _masked_min(scores, mask):
    return mx.where(mask, scores, mx.array(mx.finfo(scores.dtype).min, scores.dtype))


def pe_score_bias(q_pe, k_pe, mask, *, scale):
    """score_bias form: ``(q_pe*scale) @ k_pe^T``, masked keys at finfo.min."""
    scores = (q_pe * scale) @ k_pe.swapaxes(-1, -2)
    if mask is not None:
        scores = _masked_min(scores, mask)
    return scores


def absorbed_attention(spec: MLASpec, q_latent, q_pe, kv_latent, k_pe, mask, *,
                       scale=None, cache=None):
    """Absorbed score_bias MLA over one latent head -> latent ``[B,H,L,r]``.

    ``q_latent [B,H,L,r]`` (``W_uk`` already absorbed), ``q_pe [B,H,L,p]``,
    ``kv_latent [B,1,S,r]``, ``k_pe [B,1,S,p]``; ``mask`` None or a boolean
    array broadcastable to ``[B,H,L,S]``. Under ``fold_heads`` the heads
    become query rows (rows ordered (position, head); a per-position mask is
    repeated per head); under ``broadcast`` SDPA broadcasts the latent head.
    """
    scale = spec.softmax_scale if scale is None else scale
    if spec.absorbed_layout == "fold_heads":
        B, H, L, D = q_latent.shape
        q = q_latent.transpose(0, 2, 1, 3).reshape(B, 1, L * H, D)
        q_pe = (q_pe * scale).transpose(0, 2, 1, 3).reshape(B, 1, L * H, q_pe.shape[-1])
        pe_scores = q_pe @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            rows = mx.repeat(mask, H, axis=-2)
            pe_scores = _masked_min(pe_scores, rows)
        output = _sdpa(spec, q, kv_latent, kv_latent, scale=scale, mask=pe_scores, cache=cache)
        return output.reshape(B, L, H, D).transpose(0, 2, 1, 3)
    pe_scores = pe_score_bias(q_pe, k_pe, mask, scale=scale)
    return _sdpa(spec, q_latent, kv_latent, kv_latent, scale=scale, mask=pe_scores, cache=cache)


def expanded_attention(spec: MLASpec, q_nope, q_pe, keys, values, k_pe, mask, *,
                       scale=None, cache=None):
    """Expanded score_bias MLA -> ``[B,H,L,dv]``.

    ``keys = W_uk c`` and ``values = W_uv c`` per head; the positional term
    enters as the additive SDPA mask, exactly as in the absorbed form.
    """
    scale = spec.softmax_scale if scale is None else scale
    pe_scores = pe_score_bias(q_pe, k_pe, mask, scale=scale)
    return _sdpa(spec, q_nope, keys, values, scale=scale, mask=pe_scores, cache=cache)


def score_bias_attend(spec: MLASpec, q_nope, q_pe, kv_latent, k_pe, mask, *, embed_q, unembed_out,
                      absorbed: bool, scale=None, cache=None):
    """Full score_bias MLA (absorb, attend, unabsorb) -> ``[B,H,L,dv]``."""
    if absorbed:
        return unembed_out(absorbed_attention(
            spec, embed_q(q_nope), q_pe, kv_latent, k_pe, mask, scale=scale, cache=cache))
    keys = embed_q(kv_latent, transpose=False)
    values = unembed_out(kv_latent)
    return expanded_attention(spec, q_nope, q_pe, keys, values, k_pe, mask, scale=scale, cache=cache)


def latent_attention(q, kv, mask, *, scale, sinks=None, value_dims=None,
                     score_budget=None, bounded=False):
    """fused_buffer MLA: one latent head whose key and value are both ``kv``.

    ``q [N,H,L,D]`` (absorbed latent query || roped pe), ``kv [N,1,T,D]``
    (normed latent || roped key pe, the cache buffer itself, so no K/V copy).
    ``value_dims`` keeps the first output lanes (the latent; pe lanes are
    dropped). ``sinks`` (per head) are SDPA's extra softmax column. With
    ``score_budget`` and an array or None mask, queries run in row chunks of
    at most ``score_budget // (N*H*T)``; ``bounded`` evaluates each chunk
    (same kernels and shapes: bit-identical).
    """
    def run(qq, m):
        if sinks is None:
            out = mx.fast.scaled_dot_product_attention(qq, kv, kv, scale=scale, mask=m)
        else:
            out = mx.fast.scaled_dot_product_attention(qq, kv, kv, scale=scale, mask=m, sinks=sinks)
        return out if value_dims is None else out[..., :value_dims]

    N, H, L, _ = q.shape
    T = kv.shape[2]
    if sinks is not None:
        sinks = sinks.astype(q.dtype)
    if score_budget is None:
        return run(q, mask)
    rows = L if isinstance(mask, str) else max(1, score_budget // max(1, N * H * T))
    if rows >= L:
        return run(q, mask)
    outs = []
    for a in range(0, L, rows):
        b = min(L, a + rows)
        m = mask if mask is None else mask[..., a:b, :]
        outs.append(run(q[:, :, a:b], m))
        if bounded:
            mx.eval(outs[-1])
    return mx.concatenate(outs, axis=2)


def _visible(offset, r0, r1, k0, k1):
    return mx.arange(offset + r0, offset + r1)[:, None] >= mx.arange(k0, k1)[None, :]


def latent_single_fits(q_latent, latent, score_bytes: int) -> bool:
    """Whether the FP32 ``[B,H,L,S]`` score block fits ``score_bytes``."""
    return q_latent.size // q_latent.shape[-1] * latent.shape[2] * 4 <= score_bytes


def latent_softmax_single(q_latent, q_pe, latent, k_pe, mask, *, scale, out_dtype):
    """split_fp32 absorbed MLA in one pass -> latent ``[B,H,L,r]``.

    Latent and positional scores in FP32, summed then scaled; a string mask
    is bottom-right causal built from positions; a fully masked row yields
    zeros. ``q_pe``/``k_pe`` are the shared (NoPE) or rotated channels.
    """
    causal = isinstance(mask, str)
    offset = latent.shape[2] - q_latent.shape[2]
    score = ((q_latent @ latent.swapaxes(-1, -2)).astype(mx.float32)
             + (q_pe @ k_pe.swapaxes(-1, -2)).astype(mx.float32)) * scale
    if causal:
        score = mx.where(_visible(offset, 0, q_latent.shape[2], 0, latent.shape[2]), score, -float("inf"))
    elif mask is not None:
        score = mx.where(mask, score, -float("inf"))
    top = score.max(-1, keepdims=True)
    prob = mx.exp(score - mx.where(mx.isfinite(top), top, 0))
    total = mx.maximum(prob.sum(-1, keepdims=True), 1e-30)
    return ((prob @ latent.astype(mx.float32)) / total).astype(out_dtype)


def latent_softmax_tiled(q_latent, q_pe, latent, k_pe, mask, *, scale, out_dtype,
                         key_tile: int, row_tile: int = 64, finish: Callable | None = None):
    """split_fp32 absorbed MLA with an online softmax over key tiles.

    Bounded score storage at any context. ``finish`` (the unabsorb) is
    applied to each row tile before concatenation.
    """
    causal = isinstance(mask, str)
    offset = latent.shape[2] - q_latent.shape[2]
    rank = latent.shape[-1]
    pieces = []
    for start in range(0, q_latent.shape[2], row_tile):
        end = min(start + row_tile, q_latent.shape[2])
        qn_tile, qs_tile = q_latent[:, :, start:end], q_pe[:, :, start:end]
        maximum = mx.full((*qn_tile.shape[:-1], 1), -float("inf"), dtype=mx.float32)
        total = mx.zeros_like(maximum)
        accum = mx.zeros((*qn_tile.shape[:-1], rank), dtype=mx.float32)
        for k0 in range(0, latent.shape[2], key_tile):
            k1 = min(k0 + key_tile, latent.shape[2])
            score = ((qn_tile @ latent[:, :, k0:k1].swapaxes(-1, -2)).astype(mx.float32)
                     + (qs_tile @ k_pe[:, :, k0:k1].swapaxes(-1, -2)).astype(mx.float32)) * scale
            if causal:
                score = mx.where(_visible(offset, start, end, k0, k1), score, -float("inf"))
            elif mask is not None:
                score = mx.where(mask[..., start:end, k0:k1], score, -float("inf"))
            new_max = mx.maximum(maximum, score.max(-1, keepdims=True))
            safe_max = mx.where(mx.isfinite(new_max), new_max, 0)
            alpha = mx.exp(maximum - safe_max)
            prob = mx.exp(score - safe_max)
            accum = accum * alpha + prob @ latent[:, :, k0:k1].astype(mx.float32)
            total = total * alpha + prob.sum(-1, keepdims=True)
            maximum = new_max
        result = (accum / mx.maximum(total, 1e-30)).astype(out_dtype)
        pieces.append(result if finish is None else finish(result))
    return mx.concatenate(pieces, axis=2)


def expanded_concat_attention(q, kv, k_pe, mask, *, scale, nope: int, pad_to: int | None = None):
    """concat_key expanded MLA -> ``[B,H,L,dv]``.

    ``kv [B,H,S,dn+dv]`` the head-major ``kv_b_proj`` expansion, ``k_pe
    [B,1,S,p]`` the shared key broadcast to every head. ``pad_to`` zero-pads
    q/k/v to that head dim for a fused kernel (exact for zero columns) and
    slices the value lanes back.
    """
    b, h = kv.shape[0], kv.shape[1]
    keys, values = mx.split(kv, [nope], -1)
    v = values.shape[-1]
    shared = mx.broadcast_to(k_pe, (b, h, k_pe.shape[2], k_pe.shape[-1]))
    if pad_to is not None:
        def pad(z, d):
            return mx.concatenate([z, mx.zeros((*z.shape[:-1], d), z.dtype)], -1)
        width = nope + k_pe.shape[-1]
        keys = mx.concatenate([keys, shared, mx.zeros((*keys.shape[:-1], pad_to - width), keys.dtype)], -1)
        return mx.fast.scaled_dot_product_attention(
            pad(q, pad_to - width), keys, pad(values, pad_to - v), scale=scale, mask=mask,
            force_fused=True)[..., :v]
    from ..models.base import scaled_dot_product_attention

    keys = mx.concatenate([keys, shared], -1)
    return scaled_dot_product_attention(q, keys, values, cache=None, scale=scale, mask=mask)


def expanded_rows_softmax(q, kv, k_pe, *, nope: int, scale, offset: int, rows: int,
                          out_dtype, keep: Callable | None = None):
    """concat_key expanded MLA over token-major rows, FP32 scores -> ``[T,H,dv]``.

    ``q [T,H,dn+p]``, ``kv [N,H,dn+dv]`` (``kv_b_proj`` over the whole
    cached latent), ``k_pe [N,p]`` (unrotated under NoPE). Query rows run
    in chunks of ``rows`` (each evaluated) under a causal mask at absolute
    positions ``offset...``; ``keep(r0, r1)`` (e.g. a sparse top-k mask)
    is ANDed in.
    """
    n, h = kv.shape[0], kv.shape[1]
    dr = k_pe.shape[-1]
    t = q.shape[0]
    k = mx.concatenate([kv[..., :nope], mx.broadcast_to(k_pe[:, None, :], (n, h, dr))], -1)
    k, v = k.astype(mx.float32), kv[..., nope:]
    q = q.astype(mx.float32) * scale
    outs = []
    for r0 in range(0, t, rows):
        r1 = min(t, r0 + rows)
        nk = offset + r1
        scores = mx.einsum("thd,shd->hts", q[r0:r1], k[:nk])           # [H, rows, nk]
        mask = mx.arange(nk)[None, :] <= mx.arange(offset + r0, offset + r1)[:, None]
        if keep is not None:
            mask = mask & keep(r0, r1)
        probs = mx.softmax(mx.where(mask, scores, -mx.inf), axis=-1).astype(out_dtype)
        outs.append(mx.einsum("hts,shd->thd", probs, v[:nk]))
        mx.eval(outs[-1])
    return mx.concatenate(outs)


# --------------------------------------------------------------------------
# Fast-path registries
# --------------------------------------------------------------------------
#
# Registered kernels keep their own modules, probes and counters; ``admits``
# reports why one refuses a spec/shape so a policy that names it fails closed
# and anything else falls back to the reference. Shapes are passed as
# keywords: ``heads``, ``rank``, ``rope``, ``rows``, ``batch``, ``context``,
# ``dtype``, ``mask`` ("none" | "causal" | "array"), ``device`` ("gpu"|"cpu").

# Refusal codes (the bounded Prometheus label; detail follows the colon):
#   device    not on the GPU / no Metal;    dtype     operand dtype;
#   geometry  spec heads/latent/rope/value dims or positional form;
#   shape     rows/batch/context of this call;   mask   mask kind;
#   scale     the kernel compiles a different softmax scale.
_GEOMETRY_KEYS = {"n_heads", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
                  "pe_form"}


def _shape_refusal(spec, want: dict, shape: dict) -> str | None:
    for key, value in want.items():
        got = shape.get(key, getattr(spec, key, None))
        code = "geometry" if key in _GEOMETRY_KEYS else "mask" if key == "mask" else "shape"
        if callable(value):
            if not value(got):
                return f"{code}: {key}={got!r}"
        elif got != value:
            return f"{code}: {key}={got!r} needs {value!r}"
    return None


def _device(shape):
    if shape.get("device", "gpu") != "gpu":
        return "device: needs the GPU"
    return None


def _gpu_half(shape):
    if _device(shape):
        return _device(shape)
    if shape.get("dtype") not in (mx.float16, mx.bfloat16):
        return f"dtype: {shape.get('dtype')} needs fp16/bf16"
    return None


def _admits_xing_split_k(spec, **shape):
    """xing4_0_mla_metal: score_bias absorbed, rank 512, rope 64, 1..4 rows."""
    reason = _gpu_half(shape) or _shape_refusal(spec, {
        "kv_lora_rank": 512, "qk_rope_head_dim": 64, "pe_form": "score_bias",
        "rows": lambda r: r is not None and 1 <= r <= 4,
        "context": lambda c: c is not None and c <= 131072,
    }, shape)
    if reason is None and shape.get("rows") != 1 and shape.get("mask") == "none":
        reason = "mask: multi-row needs an explicit array mask"
    return reason


def _compiled_scale(scale: float, text: str):
    def refusal(spec):
        if spec.softmax_scale != scale:
            return f"scale: kernel compiles SCALE={text}"
        return None
    return refusal


def _fused_split_k_admits(*, compiled_scale: float, scale_text: str):
    """kimi_linear_mla_fused (patched GLM donor): one unmasked B1 decode row,
    32 heads x 512 latent x 64 shared; the entry carries the compiled scale."""
    scale = _compiled_scale(compiled_scale, scale_text)

    def admits(spec, **shape):
        return (_device(shape) or _shape_refusal(spec, {
            "n_heads": 32, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
            "rows": 1, "batch": 1, "mask": "none",
        }, shape) or scale(spec))
    return admits


def _latent_flash_admits(*, default_heads: frozenset):
    """kimi_linear_mla_flash.absorbed: NAX, M5 probe. Admits the entry's
    ``default_heads`` (the original two-group kernel's 32) plus any count the
    caller passes as ``probed_heads`` (a model's own probe, e.g. Raon's
    NAX_DECODE_HEADS); the model's probe still gates the run."""
    def admits(spec, **shape):
        heads = frozenset(default_heads) | frozenset(shape.get("probed_heads", ()))
        return _gpu_half(shape) or _shape_refusal(spec, {
            "n_heads": lambda h: h in heads, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
            "v_head_dim": 128, "kv_lora_rank": 512,
            "mask": lambda m: m in ("none", "causal", "left_padded"),
        }, shape)
    return admits


def _admits_kimi_flash192(spec, **shape):
    """kimi_linear_mla_flash.expanded: qk192/v128 prompt chunks, any head count
    (Raon admits 40 under its own probe)."""
    return _gpu_half(shape) or _shape_refusal(spec, {
        "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128, "kv_lora_rank": 512,
        "mask": lambda m: m in ("none", "causal"),
    }, shape)


def _glm53_gathered_admits(*, compiled_scale: float, scale_text: str):
    """glm53_sparse_attention(_v2): gathered latent rows, 64 heads x 512; the
    entry carries the compiled scale; ids come from sparse selection."""
    scale = _compiled_scale(compiled_scale, scale_text)

    def admits(spec, **shape):
        return (_device(shape) or _shape_refusal(
            spec, {"n_heads": 64, "kv_lora_rank": 512, "pe_form": "fused_buffer"}, shape)
            or scale(spec))
    return admits


def _lazy(load, name: str):
    """``load`` is a zero-argument literal import, so the source-dependency
    analysis can resolve the kernel module this path runs."""

    def run(*args, **kwargs):
        return getattr(load(), name)(*args, **kwargs)
    run.__name__ = name
    run.load = load
    return run


ABSORBED_DECODE = KernelRegistry("mla.absorbed_decode", absorbed_attention)
ABSORBED_DECODE.register(FastPath("xing_split_k", _admits_xing_split_k,
                                  _lazy(lambda: importlib.import_module("mlx2.runtime.models.xing4_0_mla_metal"), "attend")))
ABSORBED_DECODE.register(FastPath("kimi_fused_split_k", _fused_split_k_admits(
    compiled_scale=192 ** -0.5, scale_text="192**-0.5"),
                                  _lazy(lambda: importlib.import_module("mlx2.runtime.models.kimi_linear_mla_fused"), "attend")))
ABSORBED_DECODE.register(FastPath("kimi_latent_flash", _latent_flash_admits(
    default_heads=frozenset({32})),
                                  _lazy(lambda: importlib.import_module("mlx2.runtime.models.kimi_linear_mla_flash"), "absorbed")))

EXPANDED_PREFILL = KernelRegistry("mla.expanded_prefill", expanded_concat_attention)
EXPANDED_PREFILL.register(FastPath("kimi_flash192", _admits_kimi_flash192,
                                   _lazy(lambda: importlib.import_module("mlx2.runtime.models.kimi_linear_mla_flash"), "expanded")))

LATENT_GATHERED = KernelRegistry("mla.latent_gathered", latent_attention)
_GLM53_GATHERED = _glm53_gathered_admits(compiled_scale=256 ** -0.5, scale_text="256**-0.5")
LATENT_GATHERED.register(FastPath("glm53_gathered_v1", _GLM53_GATHERED,
                                  _lazy(lambda: importlib.import_module("mlx2.runtime.models.glm53_sparse_attention"), "sparse_decode")))
LATENT_GATHERED.register(FastPath("glm53_fused_v2", _GLM53_GATHERED,
                                  _lazy(lambda: importlib.import_module("mlx2.runtime.models.glm53_sparse_attention_v2"), "launch")))


__all__ = [
    "ABSORBED_DECODE", "ABSORBED_UNBOUNDED", "EXPANDED_PREFILL", "LATENT_GATHERED", "MLASpec",
    "MultiLinear", "QuantizableMultiLinear", "QuantizedMultiLinear", "absorbed_attention",
    "absorbed_max_query", "expanded_attention", "expanded_concat_attention",
    "expanded_rows_softmax", "kv_b_absorb", "kv_b_unabsorb", "latent_attention",
    "latent_single_fits", "latent_softmax_single", "latent_softmax_tiled", "pe_score_bias",
    "score_bias_attend", "softmax_scale", "use_absorbed", "yarn_mscale",
]
