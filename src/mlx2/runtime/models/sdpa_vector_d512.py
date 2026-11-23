"""Decode attention at head_dim 512 with a two-key unroll (opt-in, exact vs stock).

MLX 0.32.4 compiles ``sdpa_vector_2pass_1`` at head_dim 512 with ``TK = 4``
(``sdpa_vector.h``): every thread prefetches four K and four V rows of 16
values before scoring, which spills registers.  ml-explore/mlx #4941 (open,
read only) uses ``TK = 2`` above 256.  On the M5 Max (2026-11-22,
``scripts/bench_sdpa_vector_tk.py``) pass 1 at TK 2 took 0.21-0.34x the TK 4
time at 4K-128K for Gemma 4's 32q/4kv and 16q/2kv global-attention shapes.

This module is the 0.32.4 kernel pair (``sdpa_vector_2pass_1`` without mask,
causal or sinks, one query row; ``sdpa_vector_2pass_2`` verbatim; MIT,
Copyright Apple Inc., see docs/PROVENANCE.md) compiled through
``mx.fast.metal_kernel`` with ``TK = 2``.  The unroll does not change the key
visit order, and the block count copies MLX's own choice for the device
class, so the output is bitwise the stock result (checked by
``tests/test_sdpa_vector_d512.py`` on Metal).  It is therefore not a new
numerical law and needs no APCv2 namespace of its own.  mlx2 changes: every
input is addressed through its strides, including the element stride, with
64-bit token offsets (MLX copies such K/V first; the values read are the
same), so there is no ``kv_copy`` and no query transposition flag; the scale
is read from a buffer; TK 2.

Selection: ``MLX2_SDPA_D512_TK2=1``.  Only geometry stock MLX would itself
send to the two-pass kernel is taken.

LEFT OFF: not helpful for served Gemma 4 decode (2026-11-22,
gemma-4-31B-it-MLX-8bit@sha256:f58e1d0a, mlx2 8f802f4ca, M5 Max, greedy B1
--ordinary, 4K/16K/64K, one repetition under heavy system contention:
sanity, not publishable statistics).  Served decode hands these layers an
all-true array mask, so stock MLX runs its unfused path and never reaches
``sdpa_vector``; this kernel needs ``MLX2_DECODE_MASK=none`` to engage.  With
it, the kernel engaged (3,140-3,580 calls per cell) and was token-identical
to MLX's own TK 4 vector kernel, but both were slower than today's unfused
path at every context (decode tok/s stock / maskless TK 4 / maskless TK 2:
13.8 / 13.4 / 13.7 at 4K, 12.5 / 11.3 / 12.1 at 16K, 9.3 / 7.0 / 8.3 at 64K).
The TK 2 gain over TK 4 is real (also 1.17-1.57x in the module bench) but
does not beat the unfused path.  Receipt:
~/mlx-models/.download-state/recon1122-8f802f4ca-r1-gemma3/ab/receipt.json.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

from .cache import DECODE_MASK_STATS, decode_mask_mode

VERSION = "sdpa-vector-d512-tk2-v1"
TK = 2
HEAD_DIM = 512
STATS = {"d512_decode_calls": 0}

_PASS1 = r"""
    constexpr int BD = 32;
    constexpr int qk_per_thread = D / BD;
    constexpr int v_per_thread = D / BD;
    typedef float U;

    thread U q[qk_per_thread];
    thread U o[v_per_thread] = {0};

    const int kv_head_idx = threadgroup_position_in_grid.x;
    const int batch_idx = threadgroup_position_in_grid.y;
    const int block_idx = threadgroup_position_in_grid.z;
    const int blocks = threadgroups_per_grid.z;
    const int gqa_factor = threads_per_threadgroup.y;
    const int q_head_idx = gqa_factor * kv_head_idx + thread_position_in_threadgroup.y;
    const int num_kv_heads = threadgroups_per_grid.x;
    const int num_q_heads = num_kv_heads * gqa_factor;
    const int o_offset = batch_idx * num_q_heads + q_head_idx;
    const uint simd_lid = thread_index_in_simdgroup;
    const int N = keys_shape[2];
    const int64_t k_seq_stride = keys_strides[2];
    const int64_t v_seq_stride = values_strides[2];
    const int64_t qd = queries_strides[3];
    const int64_t kd = keys_strides[3];
    const int64_t vd = values_strides[3];

    const device T* qp = queries + batch_idx * queries_strides[0]
        + q_head_idx * queries_strides[1] + simd_lid * qk_per_thread * qd;
    const device T* kp = keys + batch_idx * keys_strides[0] + kv_head_idx * keys_strides[1]
        + block_idx * k_seq_stride + simd_lid * qk_per_thread * kd;
    const device T* vp = values + batch_idx * values_strides[0]
        + kv_head_idx * values_strides[1] + block_idx * v_seq_stride
        + simd_lid * v_per_thread * vd;
    device T* op = out + o_offset * blocks * D + block_idx * D + simd_lid * v_per_thread;

    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = static_cast<U>(scale[0]) * qp[i * qd];
    }

    U max_score = Limits<U>::finite_min;
    U sum_exp_score = 0;

    const int64_t stride_k_tok = blocks * k_seq_stride;
    const int64_t stride_v_tok = blocks * v_seq_stride;

    int i = block_idx;
    for (; i + (TK - 1) * blocks < N; i += blocks * TK) {
        T k_reg[TK][qk_per_thread];
        T v_reg[TK][v_per_thread];
        for (int t = 0; t < TK; t++) {
            const device T* kt = kp + t * stride_k_tok;
            for (int j = 0; j < qk_per_thread; j++) { k_reg[t][j] = kt[j * kd]; }
        }
        for (int t = 0; t < TK; t++) {
            const device T* vt = vp + t * stride_v_tok;
            for (int j = 0; j < v_per_thread; j++) { v_reg[t][j] = vt[j * vd]; }
        }
        for (int t = 0; t < TK; t++) {
            U score = 0;
            for (int j = 0; j < qk_per_thread; j++) {
                score += q[j] * static_cast<U>(k_reg[t][j]);
            }
            score = simd_sum(score);
            U new_max = max(max_score, score);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            for (int j = 0; j < v_per_thread; j++) {
                o[j] = o[j] * factor + exp_score * static_cast<U>(v_reg[t][j]);
            }
        }
        kp += TK * stride_k_tok;
        vp += TK * stride_v_tok;
    }

    for (; i < N; i += blocks) {
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) { score += q[j] * kp[j * kd]; }
        score = simd_sum(score);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int j = 0; j < v_per_thread; j++) {
            o[j] = o[j] * factor + exp_score * vp[j * vd];
        }
        kp += stride_k_tok;
        vp += stride_v_tok;
    }

    if (simd_lid == 0) {
        sums[o_offset * blocks + block_idx] = sum_exp_score;
        maxs[o_offset * blocks + block_idx] = max_score;
    }
    for (int j = 0; j < v_per_thread; j++) {
        op[j] = static_cast<T>(o[j]);
    }
"""

_PASS2 = r"""
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int elem_per_thread = D / BD;
    typedef float U;

    thread U o[elem_per_thread] = {0};
    threadgroup U outputs[BN * BD];

    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int q_offset = threadgroup_position_in_grid.x;
    const device T* pp = partials + q_offset * BLOCKS * D + simd_gid * D + simd_lid * elem_per_thread;
    const device float* sp = sums + q_offset * BLOCKS;
    const device float* mp = maxs + q_offset * BLOCKS;
    device T* op = out + q_offset * D + simd_gid * elem_per_thread;

    U sum_exp_score = 0.0;
    U max_score = Limits<U>::finite_min;

    for (int b = 0; b < BLOCKS / BN; ++b) {
        max_score = max(max_score, mp[simd_lid + BN * b]);
    }
    max_score = simd_max(max_score);

    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(mp[simd_lid + BN * b] - max_score);
        sum_exp_score += factor * sp[simd_lid + BN * b];
    }
    sum_exp_score = simd_sum(sum_exp_score);

    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(mp[simd_gid] - max_score);
        for (int i = 0; i < elem_per_thread; i++) {
            o[i] += factor * static_cast<U>(pp[i]);
        }
        mp += BN;
        sp += BN;
        pp += BN * D;
    }

    for (int i = 0; i < elem_per_thread; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (int i = 0; i < elem_per_thread; i++) {
            op[i] = static_cast<T>(o[i]);
        }
    }
"""


@lru_cache(maxsize=None)
def _pass1():
    return mx.fast.metal_kernel(
        name="mlx2_sdpa_vector_d512_pass1", input_names=["queries", "keys", "values", "scale"],
        output_names=["out", "sums", "maxs"], source=_PASS1, ensure_row_contiguous=False)


@lru_cache(maxsize=None)
def _pass2():
    return mx.fast.metal_kernel(
        name="mlx2_sdpa_vector_d512_pass2", input_names=["partials", "sums", "maxs"],
        output_names=["out"], source=_PASS2)


@lru_cache(maxsize=None)
def device_class() -> str:
    """Last letter of the Metal architecture name (MLX's ``devc``)."""
    return str(mx.device_info().get("architecture", ""))[-1:]


def stock_two_pass(n: int, gqa: int, devc: str) -> bool:
    """Whether MLX 0.32.4 routes this one-row decode to ``sdpa_vector_2pass``."""
    return (devc in ("d", "s") and n >= 1024) or (gqa > 1 and n >= 4096)


def stock_blocks(n: int, gqa: int, devc: str) -> int:
    """MLX 0.32.4 ``sdpa_vector_2pass`` block count (one query row).

    Copied so the partials, and hence the result, are bitwise the stock ones.
    ``MLX_SDPA_BLOCKS`` overrides it in MLX, so it does here too.
    """
    n_simds = gqa
    if devc == "s":
        blocks = 64
        if n > 1024 and n_simds > 4:
            blocks = 128 if n <= 8192 else 256 if n <= 32768 else 512 if n <= 65536 else 1024
    elif devc == "d":
        blocks = 128
        if n_simds <= 2 and n > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= n < 65536:
                blocks = 512
            elif n >= 65536:
                blocks = 1024
    else:
        blocks = 64 if n_simds >= 4 else 32
    env = int(os.environ.get("MLX_SDPA_BLOCKS", "0") or 0)
    if env > 0:
        blocks = ((env + 31) // 32) * 32
    return blocks


def selected() -> bool:
    return os.environ.get("MLX2_SDPA_D512_TK2", "0") == "1"


def stock_vector_d512(n: int, gqa: int) -> bool:
    """Whether MLX 0.32.4 uses its vector kernels for a one-row D=512 call.

    ``has_fused_kernel`` admits head_dim 512 only at GQA 8 (and no array
    mask) unless ``MLX_SDPA_D512_MIN_KL=0``, and never below that many keys;
    the vector kernels also need ``gqa <= 32``.  Everything else takes the
    unfused matmul/softmax path, which this kernel does not reproduce.
    """
    min_kl = int(os.environ.get("MLX_SDPA_D512_MIN_KL", "1024") or 1024)
    return gqa <= 32 and (min_kl == 0 or gqa == 8) and n >= min_kl


def supported(queries, keys, values, *, mask=None, sinks=None) -> bool:
    """Geometry this kernel reproduces exactly; everything else stays stock."""
    if sinks is not None or not (mask is None or (isinstance(mask, str) and mask == "causal")):
        return False
    if mx.default_device() != mx.gpu:
        return False
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return False
    B, Hq, L, D = queries.shape
    if L != 1 or D != HEAD_DIM or keys.shape != values.shape or keys.shape[0] != B:
        return False
    Hkv, N = keys.shape[1], keys.shape[2]
    if keys.shape[3] != D or min(B, Hq, Hkv, N) <= 0 or Hq % Hkv:
        return False
    if not (queries.dtype == keys.dtype == values.dtype) or queries.dtype not in (
            mx.bfloat16, mx.float16):
        return False
    gqa = Hq // Hkv
    return stock_vector_d512(N, gqa) and stock_two_pass(N, gqa, device_class())


def use_kernel(queries, keys, values, *, mask=None, sinks=None) -> bool:
    return selected() and supported(queries, keys, values, mask=mask, sinks=sinks)


def status():
    return {"implemented": True, "selected": selected(), "version": VERSION, "tk": TK,
            "head_dim": HEAD_DIM,
            "exact_vs_stock": "by construction; evidence is the Metal parity test",
            "basis": "selected_host_graph_branch",
            "observed_host_dispatch": STATS["d512_decode_calls"] > 0,
            # Stock MLX takes its D=512 vector kernels (and this one) only
            # without an array mask; MLX2_DECODE_MASK=none drops the all-true
            # single-token mask on an unpadded BatchKVCache.
            "decode_mask_mode": decode_mask_mode(),
            "counts": {**STATS, **DECODE_MASK_STATS}}


def attention(queries, keys, values, *, scale: float, blocks: int | None = None, tk: int = TK):
    """One-row decode attention; bitwise stock at the stock ``blocks``."""
    B, Hq, _, D = queries.shape
    Hkv, N = keys.shape[1], keys.shape[2]
    gqa = Hq // Hkv
    blocks = stock_blocks(N, gqa, device_class()) if blocks is None else blocks
    if blocks <= 0 or blocks % 32:
        raise ValueError("blocks must be a positive multiple of 32 (the pass-2 width)")
    STATS["d512_decode_calls"] += 1
    partials, sums, maxs = _pass1()(
        inputs=[queries, keys, values, mx.array([scale], dtype=mx.float32)],
        template=[("T", queries.dtype), ("D", D), ("TK", tk)],
        grid=(32 * Hkv, gqa * B, blocks), threadgroup=(32, gqa, 1),
        output_shapes=[(B * Hq, blocks, D), (B * Hq, blocks), (B * Hq, blocks)],
        output_dtypes=[queries.dtype, mx.float32, mx.float32],
    )
    return _pass2()(
        inputs=[partials, sums, maxs],
        template=[("T", queries.dtype), ("D", D), ("BLOCKS", blocks)],
        grid=(1024 * B * Hq, 1, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(B, Hq, 1, D)], output_dtypes=[queries.dtype],
    )[0]
