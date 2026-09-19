# SPDX-License-Identifier: MIT
# MLA/MoE mined from local mlx-lm-unified deepseek_v3.py + mla.py (MIT); mHC
# semantics ported from the Xing4.0 HF reference (Apache-2.0). See
# provenance/xing4-0-model.json and provenance/xing4-0-model.NOTICE.
"""Xing4.0-29B-A4B: DeepSeek-V3 MLA + noaux_tc MoE with mHC residual streams.

Architecture (HF ``modeling_xing4_0.py`` is the reference):

* Every layer is MLA attention cached as the compressed latent
  (``kv_lora_rank`` + ``qk_rope_head_dim`` per token) in a standard
  :class:`KVCache` -- keys hold the normed latent ``[B, 1, S, r]`` and values
  hold the roped key ``[B, 1, S, d_rope]``. Nothing else carries cross-token
  state, so APCv2 prefix reuse, COW branching, trim/rollback and the batched
  caches (``BatchKVCache`` via ``KVCache.merge``) work unchanged.
* Layers ``< first_k_dense_replace`` use a dense SwiGLU MLP; the rest use a
  sigmoid ``noaux_tc`` router (fp32) with ``e_score_correction_bias``, routed
  SwitchGLU experts and one shared expert.
* The residual is ``hc_mult`` parallel streams mixed by Manifold-constrained
  Hyper-Connections (mHC): per sub-block, a doubly-stochastic ``comb`` from 20
  Sinkhorn iterations, a ``pre`` collapse and a ``post`` expansion. The
  embedding is copied into every stream; the output is the stream mean, then
  the final RMSNorm.
* The MTP head (checkpoint ``model.layers.{num_hidden_layers}.*``) is a plain
  DeepSeek-V3 MTP block without mHC. Its seed is the trunk's post-final-norm
  hidden, the same tensor the LM head consumes (vLLM and SGLang both hand the
  draft ``norm(output_contract(streams))``).

mHC math is fp32 throughout (HF rounds the ``hc_fn`` projection and the stream
update to the model dtype; vLLM/SGLang keep fp32 operands -- we follow the
fp32 path and store the streams in the model dtype like all three). The
compiled mHC path is algebraically the eager reference path; set
``MLX2_XING_COMPILE_MHC=0`` or :func:`set_compile_mhc` to force eager.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask
from .cache import KVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class ModelArgs(BaseModelArgs):
    # Defaults mirror HF ``Xing4_0Config.__init__``; config.json overrides.
    model_type: str = "xing4_0"
    vocab_size: int = 131072
    hidden_size: int = 3584
    intermediate_size: int = 9216
    moe_intermediate_size: int = 1024
    num_hidden_layers: int = 40
    num_nextn_predict_layers: int = 1
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    n_shared_experts: Optional[int] = 1
    n_routed_experts: Optional[int] = 64
    routed_scaling_factor: float = 2.0
    kv_lora_rank: int = 512
    q_lora_rank: Optional[int] = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    topk_method: str = "noaux_tc"
    n_group: int = 8
    topk_group: int = 4
    num_experts_per_tok: int = 4
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 2
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_interleave: bool = True
    attention_bias: bool = False
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    mhc_h_res_clamp_min: float = -30.0
    mhc_h_res_clamp_max: float = 30.0

    def __post_init__(self):
        # Fail closed on semantics this port does not implement.
        if self.topk_method != "noaux_tc":
            raise ValueError(f"Xing4.0 port supports topk_method='noaux_tc' only, got {self.topk_method!r}")
        if self.scoring_func != "sigmoid":
            raise ValueError(f"Xing4.0 port supports scoring_func='sigmoid' only, got {self.scoring_func!r}")
        if self.hidden_act != "silu":
            raise ValueError(f"Xing4.0 port supports hidden_act='silu' only, got {self.hidden_act!r}")
        if self.hc_mult < 1:
            raise ValueError("hc_mult must be >= 1")
        if self.tie_word_embeddings:
            raise ValueError("Xing4.0 checkpoints have an untied lm_head")
        if self.num_attention_heads != self.num_key_value_heads:
            raise ValueError("Xing4.0 MLA expects num_key_value_heads == num_attention_heads")
        if self.num_nextn_predict_layers not in (0, 1):
            raise ValueError("Xing4.0 port supports at most one MTP layer")


# --------------------------------------------------------------------------
# MLA helpers (mined from mlx-lm-unified mla.py)
# --------------------------------------------------------------------------

#: Test/A-B override of the absorbed-path query-width limit (``None`` keeps
#: the geometric gate, ``0`` forces the expanded branch, a large value forces
#: the absorbed branch). The two branches are the same math.
ABSORBED_MAX_QUERY_OVERRIDE: Optional[int] = None
_ABSORBED_UNBOUNDED = 1 << 30


def set_absorbed_max_query_override(value: Optional[int]) -> None:
    global ABSORBED_MAX_QUERY_OVERRIDE
    ABSORBED_MAX_QUERY_OVERRIDE = None if value is None else max(0, int(value))


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
        return _ABSORBED_UNBOUNDED
    return max(1, numerator // denom)


def use_absorbed_path(query_len: int, cache_len: int, geometry) -> bool:
    limit = absorbed_max_query(*geometry, cache_len=cache_len)
    if ABSORBED_MAX_QUERY_OVERRIDE is not None:
        limit = ABSORBED_MAX_QUERY_OVERRIDE
    return query_len <= limit


class MultiLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, num_heads: int) -> None:
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale, high=scale, shape=(num_heads, output_dims, input_dims)
        )

    def __call__(self, x, transpose=True):
        if transpose:
            return x @ self.weight.swapaxes(-1, -2)
        return x @ self.weight

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


# --------------------------------------------------------------------------
# Attention (MLA, latent cache)
# --------------------------------------------------------------------------


class Xing4_0Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.q_lora_rank = args.q_lora_rank
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.v_head_dim = args.v_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.scale = self.q_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(args.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(args.hidden_size, self.q_lora_rank, bias=args.attention_bias)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=args.attention_bias
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)
        # kv_b_proj is folded into these two at sanitize time.
        self.embed_q = MultiLinear(self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads)
        self.unembed_out = MultiLinear(self.kv_lora_rank, self.v_head_dim, self.num_heads)
        self.absorbed_geometry = (self.kv_lora_rank, self.qk_nope_head_dim, self.v_head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, args.hidden_size, bias=args.attention_bias)

        scaling = args.rope_scaling
        if scaling is not None:
            rope_type = scaling.get("type") or scaling.get("rope_type", "default")
            mscale_all_dim = scaling.get("mscale_all_dim", 0)
            factor = scaling.get("factor", 1.0)
            # HF: only non-default rope types with a truthy mscale_all_dim.
            if rope_type != "default" and mscale_all_dim and factor > 1:
                s = 0.1 * mscale_all_dim * math.log(factor) + 1.0
                self.scale = self.scale * s * s
        # HF's interleaved apply (rope_interleave=True) is MLX traditional RoPE
        # up to a fixed permutation of the rope dims that cancels in q_pe.k_pe.
        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=bool(args.rope_interleave),
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_scaling,
        )

    def _absorbed(self, q_latent, q_pe, kv_latent, k_pe, mask):
        """Absorbed MLA with the heads folded into the query rows.

        All heads share one latent key/value head, so ``[B, H, L, D]`` queries
        attend as ``[B, 1, L*H, D]`` against ``[B, 1, S, D]``: the latent is
        read once for every head (2.5-7.5x faster decode/verify than the
        broadcast GQA form on M5).  Rows are ordered (position, head); a
        per-position mask is repeated per head to match.
        """
        B, H, L, D = q_latent.shape
        q = q_latent.transpose(0, 2, 1, 3).reshape(B, 1, L * H, D)
        q_pe = (q_pe * self.scale).transpose(0, 2, 1, 3).reshape(B, 1, L * H, q_pe.shape[-1])
        pe_scores = q_pe @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            rows = mx.repeat(mask, H, axis=-2)
            pe_scores = mx.where(rows, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype))
        output = mx.fast.scaled_dot_product_attention(q, kv_latent, kv_latent, scale=self.scale, mask=pe_scores)
        return output.reshape(B, L, H, D).transpose(0, 2, 1, 3)

    def _attend(self, q_nope, q_pe, kv_latent, k_pe, mask):
        """MLA over one latent history; returns [B, H, L, v_head_dim]."""
        L = q_nope.shape[2]
        if mask is not None and isinstance(mask, str):
            raise ValueError("Xing4.0 MLA requires an array attention mask")
        if use_absorbed_path(L, k_pe.shape[-2], self.absorbed_geometry):
            return self.unembed_out(self._absorbed(self.embed_q(q_nope), q_pe, kv_latent, k_pe, mask))
        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(mask, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype))

        # Call SDPA directly: the latent is both key and value, so a cache's
        # bucketed-attention hook (which reads cache.values) must not be used.
        k = self.embed_q(kv_latent, transpose=False)
        v = self.unembed_out(kv_latent)
        return mx.fast.scaled_dot_product_attention(q_nope, k, v, scale=self.scale, mask=pe_scores)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        B, L, _ = x.shape
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)
        kv_latent = mx.expand_dims(kv_latent, axis=1)
        if cache is not None and hasattr(cache, "row_views"):
            # Segmented batch: history stays in per-row caches; attend each
            # row's own latent with the same MLA math.
            cache.update_and_fetch(kv_latent, k_pe)
            outputs = []
            for index, valid, row_latent, row_pe, row_mask in cache.row_views(mask):
                if not valid:
                    outputs.append(mx.zeros((1, self.num_heads, L, self.v_head_dim), dtype=q_nope.dtype))
                    continue
                output = self._attend(
                    q_nope[index : index + 1, :, :valid], q_pe[index : index + 1, :, :valid],
                    row_latent, row_pe, row_mask,
                )
                if valid < L:
                    output = mx.pad(output, [(0, 0), (0, 0), (0, L - valid), (0, 0)])
                outputs.append(output)
            cache.note_attention()
            output = mx.concatenate(outputs, axis=0)
        else:
            if cache is not None:
                kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)
            if not isinstance(k_pe, mx.array):
                # Fail closed: a quantized latent cache is not qualified here.
                raise NotImplementedError("Xing4.0 MLA does not support a quantized KV cache")
            output = self._attend(q_nope, q_pe, kv_latent, k_pe, mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


# --------------------------------------------------------------------------
# MLP / MoE
# --------------------------------------------------------------------------


class Xing4_0MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


def router_select(x, weight, e_score_correction_bias, top_k, routed_scaling_factor, norm_topk_prob):
    """HF ``Xing4_0TopkRouter`` in fp32: sigmoid scores, top-k on
    ``scores + bias``, weights gathered from unbiased scores, optional
    ``/(sum + 1e-20)`` normalisation, times ``routed_scaling_factor``.

    HF ignores ``n_group``/``topk_group`` (no group-limited selection); so do we.
    """
    logits = x.astype(mx.float32) @ weight.astype(mx.float32).T
    scores = mx.sigmoid(logits)
    biased = scores + e_score_correction_bias.astype(mx.float32)
    inds = mx.argpartition(-biased, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    return inds, weights * routed_scaling_factor


class MoEGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling_factor = args.routed_scaling_factor
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((args.n_routed_experts,), dtype=mx.float32)

    def __call__(self, x):
        return router_select(
            x,
            self.weight,
            self.e_score_correction_bias,
            self.top_k,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class Xing4_0MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.switch_mlp = SwitchGLU(args.hidden_size, args.moe_intermediate_size, args.n_routed_experts)
        self.gate = MoEGate(args)
        self.has_shared = bool(args.n_shared_experts)
        if self.has_shared:
            self.shared_experts = Xing4_0MLP(
                args.hidden_size, args.moe_intermediate_size * args.n_shared_experts
            )

    def __call__(self, x):
        inds, scores = self.gate(x)
        routed = self.switch_mlp(x, inds)
        y = (routed * scores[..., None]).sum(axis=-2).astype(x.dtype)
        if self.has_shared:
            y = y + self.shared_experts(x)
        return y


def _make_mlp(args: ModelArgs, layer_idx: int) -> nn.Module:
    # HF: MoE iff layer_idx >= first_k_dense_replace (moe_layer_freq unused).
    if args.n_routed_experts and layer_idx >= args.first_k_dense_replace:
        return Xing4_0MoE(args)
    return Xing4_0MLP(args.hidden_size, args.intermediate_size)


# --------------------------------------------------------------------------
# mHC (Manifold-constrained Hyper-Connections)
# --------------------------------------------------------------------------

_COMPILE_MHC = os.environ.get("MLX2_XING_COMPILE_MHC", "1") == "1"
# Fused Metal mHC kernels (``xing4_0_mhc_metal``); GPU only, compiled path otherwise.
_MHC_KERNEL = os.environ.get("MLX2_XING_MHC_KERNEL", "1") == "1"
_MHC_STATS = {
    "compiled_calls": 0,
    "eager_calls": 0,
    "fallbacks": 0,
    "last_fallback": None,
    "kernel_calls": 0,
    "kernel_fallbacks": 0,
    "last_kernel_fallback": None,
}
_MHC_COMPILED: Dict[Any, Any] = {}


def set_compile_mhc(enabled: bool) -> None:
    global _COMPILE_MHC
    _COMPILE_MHC = bool(enabled)


def compile_mhc_enabled() -> bool:
    return _COMPILE_MHC


def set_mhc_kernel(enabled: bool) -> None:
    global _MHC_KERNEL
    _MHC_KERNEL = bool(enabled)
    _MHC_QUALIFIED.clear()


def mhc_kernel_enabled() -> bool:
    return _MHC_KERNEL


def _kernel_usable(array) -> bool:
    return (
        _MHC_KERNEL
        and mx.default_device() == mx.gpu
        and array.dtype in (mx.bfloat16, mx.float16, mx.float32)
    )


_MHC_QUALIFIED = set()
_MHC_QUALIFIED_LIMIT = 4096


def _run_kernel(key, fn, *args, **kwargs):
    """Run a fused mHC kernel or return None (fall back to the reference path).

    Kernel outputs are lazy, so a Metal build or dispatch failure would only
    surface at evaluation, outside this guard.  Each specialization ``key``
    (kernel, sizes, dtypes, template constants) is therefore evaluated inside
    the guard the first time it is used; only then is it trusted lazily.
    """
    global _MHC_KERNEL
    try:
        out = fn(*args, **kwargs)
        if key not in _MHC_QUALIFIED:
            mx.eval(out)
            if len(_MHC_QUALIFIED) >= _MHC_QUALIFIED_LIMIT:
                _MHC_QUALIFIED.clear()
            _MHC_QUALIFIED.add(key)
    except Exception as exc:  # disable the kernel for the process lifetime
        _MHC_KERNEL = False
        _MHC_QUALIFIED.clear()
        _MHC_STATS["kernel_fallbacks"] += 1
        _MHC_STATS["last_kernel_fallback"] = f"{key[0]}: {type(exc).__name__}: {exc}"
        return None
    _MHC_STATS["kernel_calls"] += 1
    return out


def mhc_stats(reset: bool = False) -> dict:
    out = dict(_MHC_STATS)
    if reset:
        _MHC_STATS.update(
            compiled_calls=0, eager_calls=0, fallbacks=0, last_fallback=None,
            kernel_calls=0, kernel_fallbacks=0, last_kernel_fallback=None,
        )
        _MHC_QUALIFIED.clear()
    return out


def _mhc_mix_logits(streams, hc_fn, norm_eps):
    """Unweighted fp32 RMSNorm over the flattened streams, then ``@ hc_fn.T``."""
    flat = streams.reshape(*streams.shape[:-2], -1).astype(mx.float32)
    flat = mx.fast.rms_norm(flat, None, norm_eps)
    return flat @ hc_fn.astype(mx.float32).T


def _mhc_coeffs_body(streams, pre_w, post_w, comb_w, scale, pre_b, post_b, comb_b, iters, eps, lo, hi):
    """fp32 pre/post/comb + Sinkhorn + collapse. Every input shape-agnostic."""
    pre = mx.sigmoid(pre_w * scale[0] + pre_b)
    post = 2.0 * mx.sigmoid(post_w * scale[1] + post_b)
    comb = mx.clip(comb_w * scale[2] + comb_b, lo, hi)
    comb = mx.exp(comb - comb.max(axis=-1, keepdims=True))
    for _ in range(iters):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    collapsed = (mx.expand_dims(pre, -1) * streams.astype(mx.float32)).sum(axis=-2)
    return post, comb, collapsed.astype(streams.dtype)


def _mhc_update_body(streams, out, post, comb):
    """``post[..., None] * out[..., None, :] + comb @ streams`` in fp32 (HF orientation)."""
    s32 = streams.astype(mx.float32)
    new = mx.expand_dims(post, -1) * mx.expand_dims(out.astype(mx.float32), -2) + comb @ s32
    return new.astype(streams.dtype)


def _compiled(key, builder):
    fn = _MHC_COMPILED.get(key)
    if fn is None:
        fn = mx.compile(builder(), shapeless=True)
        _MHC_COMPILED[key] = fn
    return fn


def _run_compiled(key, builder, *args):
    """Run a compiled mHC span or return None (fail closed to eager math)."""
    if not _COMPILE_MHC or _MHC_COMPILED.get(key, 0) is None:
        return None
    try:
        out = _compiled(key, builder)(*args)
    except Exception as exc:  # demote this span for the process lifetime
        _MHC_COMPILED[key] = None
        _MHC_STATS["fallbacks"] += 1
        _MHC_STATS["last_fallback"] = f"{key[0]}: {type(exc).__name__}: {exc}"
        return None
    _MHC_STATS["compiled_calls"] += 1
    return out


class HyperConnection(nn.Module):
    """HF ``Xing4_0HyperConnection``; ``hc_fn/hc_base/hc_scale`` stay unquantized."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        hc = args.hc_mult
        mix = (2 + hc) * hc
        self.hc_mult = hc
        self.iters = int(args.hc_sinkhorn_iters)
        self.eps = float(args.hc_eps)
        self.norm_eps = float(args.rms_norm_eps)
        self.clamp_min = float(args.mhc_h_res_clamp_min)
        self.clamp_max = float(args.mhc_h_res_clamp_max)
        self.hc_fn = mx.zeros((mix, hc * args.hidden_size))
        self.hc_base = mx.zeros((mix,))
        self.hc_scale = mx.ones((3,))

    def _split(self, streams):
        hc = self.hc_mult
        mix = _mhc_mix_logits(streams, self.hc_fn, self.norm_eps)
        base = self.hc_base.astype(mx.float32)
        return (
            mix[..., :hc],
            mix[..., hc : 2 * hc],
            mix[..., 2 * hc :].reshape(*mix.shape[:-1], hc, hc),
            self.hc_scale.astype(mx.float32),
            base[:hc],
            base[hc : 2 * hc],
            base[2 * hc :].reshape(hc, hc),
        )

    def reference(self, streams):
        """Uncompiled reference: returns (post, comb, collapsed)."""
        return _mhc_coeffs_body(
            streams, *self._split(streams), self.iters, self.eps, self.clamp_min, self.clamp_max
        )

    def _kernel_params(self):
        from .xing4_0_mhc_metal import pack_params

        key = (id(self.hc_scale), id(self.hc_base))
        cached = getattr(self, "_packed", None)
        if cached is None or cached[0] != key:
            packed = pack_params(
                self.norm_eps, self.eps, self.clamp_min, self.clamp_max, self.hc_scale, self.hc_base
            )
            object.__setattr__(self, "_packed", (key, packed))
            return packed
        return cached[1]

    def __call__(self, streams):
        if _kernel_usable(streams):
            from .xing4_0_mhc_metal import mhc_pre

            key = ("mhc_pre", streams.shape[-1], streams.dtype, self.hc_fn.dtype, self.iters)
            out = _run_kernel(
                key, mhc_pre, streams, self.hc_fn, self._kernel_params(), iters=self.iters
            )
            if out is not None:
                return out
        parts = self._split(streams)
        iters, eps, lo, hi = self.iters, self.eps, self.clamp_min, self.clamp_max
        out = _run_compiled(
            ("mhc_coeffs", iters, eps, lo, hi),
            lambda: lambda *a: _mhc_coeffs_body(*a, iters, eps, lo, hi),
            streams,
            *parts,
        )
        if out is None:
            _MHC_STATS["eager_calls"] += 1
            out = _mhc_coeffs_body(streams, *parts, iters, eps, lo, hi)
        return out

    @staticmethod
    def update_reference(streams, out, post, comb):
        return _mhc_update_body(streams, out, post, comb)

    @staticmethod
    def update(streams, out, post, comb):
        if _kernel_usable(streams):
            from .xing4_0_mhc_metal import mhc_update

            key = ("mhc_update", streams.shape[-1], streams.dtype, out.dtype)
            fused = _run_kernel(key, mhc_update, streams, out, post, comb)
            if fused is not None:
                return fused
        new = _run_compiled(("mhc_update",), lambda: _mhc_update_body, streams, out, post, comb)
        if new is None:
            _MHC_STATS["eager_calls"] += 1
            new = _mhc_update_body(streams, out, post, comb)
        return new


# --------------------------------------------------------------------------
# Decoder / trunk
# --------------------------------------------------------------------------


def _span_body(streams, out, post, comb, hc_fn, params, norm_weight, *, iters, norm_eps):
    from .xing4_0_mhc_metal import mhc_pre, mhc_update

    streams = mhc_update(streams, out, post, comb)
    post, comb, x = mhc_pre(streams, hc_fn, params, iters=iters)
    return streams, post, comb, mx.fast.rms_norm(x, norm_weight, norm_eps)


def _mhc_span(streams, out, post, comb, hc, norm):
    """update(streams) -> next HyperConnection pre -> RMS norm, compiled per shape."""
    key = ("span", hc.iters, float(norm.eps))
    fn = _MHC_COMPILED.get(key)
    if fn is None:
        iters, eps = hc.iters, float(norm.eps)
        fn = mx.compile(
            lambda s, o, p, c, w, pr, nw: _span_body(s, o, p, c, w, pr, nw, iters=iters, norm_eps=eps)
        )
        _MHC_COMPILED[key] = fn
    return fn(streams, out, post, comb, hc.hc_fn, hc._kernel_params(), norm.weight)


def _mhc_last_update(streams, out, post, comb):
    from .xing4_0_mhc_metal import mhc_update

    return mhc_update(streams, out, post, comb)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Xing4_0Attention(args)
        self.mlp = _make_mlp(args, layer_idx)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attn_hc = HyperConnection(args)
        self.ffn_hc = HyperConnection(args)

    def __call__(self, streams: mx.array, mask=None, cache=None, reference: bool = False) -> mx.array:
        """``streams``: [B, L, hc_mult, H] -> [B, L, hc_mult, H]."""
        attn_hc = self.attn_hc.reference if reference else self.attn_hc
        update = HyperConnection.update_reference if reference else HyperConnection.update
        post, comb, x = attn_hc(streams)
        a = self.self_attn(self.input_layernorm(x), mask, cache)
        streams = update(streams, a, post, comb)
        ffn_hc = self.ffn_hc.reference if reference else self.ffn_hc
        post, comb, x = ffn_hc(streams)
        m = self.mlp(self.post_attention_layernorm(x))
        return update(streams, m, post, comb)


class Xing4_0Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, idx) for idx in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        reference: bool = False,
    ) -> mx.array:
        """Post-final-norm hidden [B, L, H] (LM-head input and MTP seed)."""
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        if len(cache) != len(self.layers):
            raise ValueError("Xing4.0 cache layer count mismatch")
        mask = create_attention_mask(h, cache[0], return_array=True)
        B, L, H = h.shape
        streams = mx.broadcast_to(mx.expand_dims(h, 2), (B, L, self.args.hc_mult, H))
        if not reference and _kernel_usable(streams):
            fused = self._fused_trunk(streams, mask, cache)
            if fused is not None:
                return fused
        for layer, layer_cache in zip(self.layers, cache):
            streams = layer(streams, mask, layer_cache, reference=reference)
        h = streams.astype(mx.float32).mean(axis=2).astype(streams.dtype)
        return self.norm(h)

    def _fused_trunk(self, streams, mask, cache):
        """Kernel path: every update -> next pre -> RMS norm runs as one compiled span.

        Same math as ``DecoderLayer.__call__``; the spans chain across layer
        boundaries so each layer costs attention, MLP and two compiled spans.
        """
        from .xing4_0_mhc_metal import mhc_pre

        first = self.layers[0]

        key = (
            "fused_trunk", tuple(streams.shape), streams.dtype, first.attn_hc.hc_fn.dtype,
            first.attn_hc.iters, float(first.input_layernorm.eps),
        )

        def start():
            post, comb, x = mhc_pre(
                mx.contiguous(streams), first.attn_hc.hc_fn, first.attn_hc._kernel_params(),
                iters=first.attn_hc.iters,
            )
            # For a new specialization the whole chain (both kernels and the
            # compiled span at this shape) is built and evaluated before any
            # cache is written: a failure here may fall back to the generic
            # path, a failure after the first attention append must propagate.
            probe = None
            if key not in _MHC_QUALIFIED:
                probe = _mhc_span(
                    streams, mx.zeros_like(x), post, comb, first.ffn_hc,
                    first.post_attention_layernorm,
                )
            return post, comb, first.input_layernorm(x), probe

        began = _run_kernel(key, start)
        if began is None:
            return None
        post, comb, x, _ = began
        count = len(self.layers)
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            a = layer.self_attn(x, mask, layer_cache)
            streams, post, comb, x = _mhc_span(
                streams, a, post, comb, layer.ffn_hc, layer.post_attention_layernorm
            )
            m = layer.mlp(x)
            if index + 1 < count:
                following = self.layers[index + 1]
                streams, post, comb, x = _mhc_span(
                    streams, m, post, comb, following.attn_hc, following.input_layernorm
                )
            else:
                streams = _mhc_last_update(streams, m, post, comb)
        h = streams.astype(mx.float32).mean(axis=2).astype(streams.dtype)
        return self.norm(h)


# --------------------------------------------------------------------------
# MTP head
# --------------------------------------------------------------------------


class SharedHead(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)


class MTPLayer(nn.Module):
    """DeepSeek-V3 MTP block (no mHC). Tensor names match checkpoint layer 40.

    ``embed_tokens`` / ``shared_head.head`` are removed by :meth:`Model.sanitize`
    when the checkpoint copy is proven equal to the trunk tensor (or already
    absent), in which case the trunk embedding / lm_head are used.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        H = args.hidden_size
        self.embed_tokens = nn.Embedding(args.vocab_size, H)
        self.enorm = nn.RMSNorm(H, eps=args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(H, eps=args.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * H, H, bias=False)
        self.self_attn = Xing4_0Attention(args)
        self.mlp = _make_mlp(args, layer_idx)
        self.input_layernorm = nn.RMSNorm(H, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(H, eps=args.rms_norm_eps)
        self.shared_head = SharedHead(args)

    def block(self, x, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class MTPModule(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layers = [
            MTPLayer(args, args.num_hidden_layers + i) for i in range(args.num_nextn_predict_layers)
        ]


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class Model(nn.Module):
    apc_v2_layout = "xing4-0-mla-latent-layer-segments-v1"
    supports_speculative_rollback = True

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Xing4_0Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.mtp = MTPModule(args) if args.num_nextn_predict_layers > 0 else None

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None):
        return self.lm_head(self.model(inputs, cache, input_embeddings=input_embeddings))

    @property
    def layers(self):
        return self.model.layers

    def logits(self, hidden: mx.array) -> mx.array:
        return self.lm_head(hidden)

    def make_cache(self):
        return [KVCache() for _ in self.model.layers]

    def mtp_backbone(self, inputs: mx.array, cache=None):
        """(LM-head hidden, MTP seed hidden): both the post-final-norm hidden."""
        hidden = self.model(inputs, cache=cache)
        return hidden, hidden

    def make_mtp_cache(self):
        if self.mtp is None:
            raise RuntimeError("Xing4.0 MTP head is not loaded")
        return [KVCache() for _ in self.mtp.layers]

    def mtp_step(self, hidden, tokens, mtp_cache):
        """One MTP forward over S positions.

        hidden: [B, S, H] post-final-norm trunk hiddens at positions p..p+S-1
        (or the ``post`` of a previous step when chaining). tokens: [B, S] the
        tokens at p+1..p+S. Returns (logits [B, S, V], post [B, S, H]) where
        ``post`` is ``shared_head.norm`` output (SGLang NextN convention).
        MTP rope positions are the pair count, a uniform -1 shift of absolute
        positions that cancels in q.k because the layer only attends its own
        cache.
        """
        if self.mtp is None:
            raise RuntimeError("Xing4.0 MTP head is not loaded")
        layer = self.mtp.layers[0]
        embed = layer.embed_tokens if "embed_tokens" in layer else self.model.embed_tokens
        e = layer.enorm(embed(tokens))
        h = layer.hnorm(hidden.astype(e.dtype))
        x = layer.eh_proj(mx.concatenate([e, h], axis=-1))
        cache = mtp_cache[0]
        mask = create_attention_mask(x, cache, return_array=True)
        x = layer.block(x, mask, cache)
        post = layer.shared_head.norm(x)
        head = layer.shared_head.head if "head" in layer.shared_head else self.lm_head
        return head(post), post

    # ---------------------------------------------------------------- weights

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """HF or already-converted weights -> this module tree. Idempotent."""
        args = self.args
        n_layers = args.num_hidden_layers
        if any("weight_scale_inv" in k or k.endswith("weight_packed") for k in weights):
            raise ValueError("Xing4.0 port expects bf16/fp32 or MLX-affine weights")

        out: Dict[str, mx.array] = {}
        for key, value in weights.items():
            if "rotary_emb" in key:
                continue
            if key.startswith("model.layers."):
                index = int(key.split(".")[2])
                if index >= n_layers:
                    rest = key.split(".", 3)[3]
                    key = f"mtp.layers.{index - n_layers}.{rest}"
            out[key] = value

        has_mtp = any(k.startswith("mtp.layers.") for k in out)
        if self.mtp is None or not has_mtp:
            out = {k: v for k, v in out.items() if not k.startswith("mtp.")}
            self.mtp = None

        prefixes = [f"model.layers.{i}" for i in range(n_layers)]
        if self.mtp is not None:
            prefixes += [f"mtp.layers.{i}" for i in range(len(self.mtp.layers))]
        for prefix in prefixes:
            self._stack_experts(out, prefix)
            self._fold_kv_b(out, f"{prefix}.self_attn")

        if self.mtp is not None:
            for i, layer in enumerate(self.mtp.layers):
                self._dedupe_mtp(out, layer, f"mtp.layers.{i}.embed_tokens", "model.embed_tokens", "embed_tokens", layer)
                self._dedupe_mtp(
                    out, layer, f"mtp.layers.{i}.shared_head.head", "lm_head", "head", layer.shared_head
                )
        return out

    def _stack_experts(self, weights, prefix):
        if not self.args.n_routed_experts:
            return
        for name in ("gate_proj", "down_proj", "up_proj"):
            for kind in ("weight", "scales", "biases"):
                first = f"{prefix}.mlp.experts.0.{name}.{kind}"
                if first not in weights:
                    continue
                keys = [f"{prefix}.mlp.experts.{e}.{name}.{kind}" for e in range(self.args.n_routed_experts)]
                missing = [k for k in keys if k not in weights]
                if missing:
                    raise ValueError(f"missing expert tensors, e.g. {missing[0]}")
                weights[f"{prefix}.mlp.switch_mlp.{name}.{kind}"] = mx.stack([weights.pop(k) for k in keys])

    def _fold_kv_b(self, weights, prefix):
        key = f"{prefix}.kv_b_proj.weight"
        if key not in weights:
            return
        args = self.args
        v = weights.pop(key)
        quantized = f"{prefix}.kv_b_proj.scales" in weights
        if quantized:
            scales = weights.pop(f"{prefix}.kv_b_proj.scales")
            biases = weights.pop(f"{prefix}.kv_b_proj.biases", None)
            mode = (getattr(args, "quantization", None) or {}).get("mode", "affine")
            if biases is None or mode != "affine":
                raise ValueError("MLA kv_b_proj folding supports affine quantization only")
            dims = args.kv_lora_rank
            bits = (v.shape[-1] * 32) // dims
            group_size = dims // scales.shape[-1]
            v = mx.dequantize(v, scales, biases, bits=bits, group_size=group_size)
        head_dim = args.qk_nope_head_dim + args.v_head_dim
        v = v.reshape(args.num_attention_heads, head_dim, -1)
        wk = mx.contiguous(v[:, : args.qk_nope_head_dim, :].swapaxes(-1, -2))
        wv = mx.contiguous(v[:, args.qk_nope_head_dim :, :])
        if quantized:
            wk, wk_s, wk_b = mx.quantize(wk, bits=bits, group_size=group_size)
            wv, wv_s, wv_b = mx.quantize(wv, bits=bits, group_size=group_size)
            weights[f"{prefix}.embed_q.scales"] = wk_s
            weights[f"{prefix}.embed_q.biases"] = wk_b
            weights[f"{prefix}.unembed_out.scales"] = wv_s
            weights[f"{prefix}.unembed_out.biases"] = wv_b
        weights[f"{prefix}.embed_q.weight"] = wk
        weights[f"{prefix}.unembed_out.weight"] = wv

    @staticmethod
    def _dedupe_mtp(weights, layer, mtp_key, trunk_key, attr, owner):
        """Drop the MTP copy only when proven equal to the trunk tensor.

        Absent (already deduped) -> share the trunk module. Present and equal
        in every component (weight/scales/biases) -> drop and share. Present
        and different -> keep the MTP's own module.
        """
        parts = ("weight", "scales", "biases")
        present = [p for p in parts if f"{mtp_key}.{p}" in weights]
        if present:
            for p in parts:
                mine, trunk = weights.get(f"{mtp_key}.{p}"), weights.get(f"{trunk_key}.{p}")
                if (mine is None) != (trunk is None):
                    return
                if mine is None:
                    continue
                if mine.shape != trunk.shape or mine.dtype != trunk.dtype:
                    return
                if not bool(mx.array_equal(mine, trunk).item()):
                    return
            for p in present:
                weights.pop(f"{mtp_key}.{p}")
        if attr in owner:
            delattr(owner, attr)

    # ------------------------------------------------------------- predicates

    @property
    def quant_predicate(self):
        def predicate(path, _module=None):
            # mHC operands and the router stay full precision. (They are plain
            # arrays, never nn.quantize targets; this is belt and braces.)
            if "attn_hc" in path or "ffn_hc" in path:
                return False
            if path.endswith("mlp.gate"):
                return False
            return True

        return predicate

    @property
    def cast_predicate(self):
        keep = ("e_score_correction_bias", "hc_fn", "hc_base", "hc_scale", "mlp.gate.weight")

        def predicate(path: str):
            return not any(path.endswith(k) for k in keep)

        return predicate
