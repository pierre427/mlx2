"""GQA-aware decode attention over affine-quantized K/V (Metal, flash-decoding).

One threadgroup owns one KV head and one block of the sequence. Its
simdgroups take 32-token tiles:

* each lane dequantizes one whole K row once and scores it against the
  query heads it owns (queries broadcast from threadgroup memory), so the
  online-softmax max/sum reductions run once per tile, not per token;
* lanes then split the head dimension, dequantize each V row once, and
  accumulate it into every owned head.

K/V bytes, loads and dequantization therefore happen once per KV head (or
once per head subset when ``head_split`` > 1), not once per query head.
That once-per-query-head cost kept both mlx2's composed path and MLX's
``quant_sdpa_vector_2pass`` behind fp16 SDPA at head_dim 256, GQA 4-8
(2026-09-23, ``qualification/runs/kvq-decode-20260923``). Scores use
``scale_k * (q . codes) + bias_k * sum(q)`` per quantization group.

Each simdgroup writes one normalized partial (o/l, l, m) per head; a small
merge in MLX ops combines them.

The split-sequence, per-row-reuse structure follows MLX's
``sdpa_vector_2pass_1_gqa`` (``mlx/backend/metal/kernels/sdpa_vector.h``,
MIT; see docs/PROVENANCE.md). The tiled layout, quantized loads and
dequantization, separate key and value bit widths, head split, strided
cache addressing and merge are mlx2's own.

Covered: decode (one query row), no mask or sinks, head_dim 128 or 256
(4-bit needs 256), group_size 64, 1-8 query heads per KV head, bf16/fp16.
Everything else stays on the composed path; see ``use_decode_kernel``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import mlx.core as mx

_SOURCE = r"""
    constexpr int EPL = D / 32;
    constexpr int NG = D / GS;
    constexpr int KWG = GS * KB / 32;       // K words per quantization group
    constexpr int VW = EPL * VB / 32;
    constexpr uint KMASK = (1u << KB) - 1u;
    constexpr uint VMASK = (1u << VB) - 1u;

    threadgroup float q_sh[G * D];
    threadgroup float qg_sh[G * NG];
    constexpr int GL = G / HS;              // heads per simdgroup
    threadgroup float p_sh[NS * GL * 32];

    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint flat = sg * 32 + lane;
    const uint bh = threadgroup_position_in_grid.y;
    const uint block = threadgroup_position_in_grid.z;
    const uint blocks = threadgroups_per_grid.z;

    const int Hkv = kw_shape[1];
    const int N = kw_shape[2];
    const int Hq = q_shape[1];
    const int b = bh / Hkv;
    const int hkv = bh % Hkv;

    for (uint idx = flat; idx < uint(G * D); idx += NS * 32) {
        const int j = idx / D, d = idx % D;
        q_sh[idx] = float(q[b * q_strides[0] + (hkv * G + j) * q_strides[1] + d * q_strides[3]]) * scale[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = flat; idx < uint(G * NG); idx += NS * 32) {
        const int j = idx / NG, g = idx % NG;
        float acc = 0.0f;
        for (int d = 0; d < GS; d++) acc += q_sh[j * D + g * GS + d];
        qg_sh[idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int chunk = (N + int(blocks) - 1) / int(blocks);
    const int kstart = int(block) * chunk;
    const int kend = min(N, kstart + chunk);
    constexpr int NT = NS / HS;             // token sub-ranges per block
    const int hs = int(sg) % HS;            // this simdgroup's head subset
    const int h0 = hs * GL;
    const int sub = (chunk + NT - 1) / NT;
    const int s0 = kstart + (int(sg) / HS) * sub;
    const int s1 = min(kend, s0 + sub);

    const device uint32_t* kwb = kw + b * kw_strides[0] + hkv * kw_strides[1];
    const device T* ksb = ks + b * ks_strides[0] + hkv * ks_strides[1];
    const device T* kbb = kb + b * kb_strides[0] + hkv * kb_strides[1];
    const int vgroup = (int(lane) * EPL) / GS;
    const device uint32_t* vwb = vw + b * vw_strides[0] + hkv * vw_strides[1] + lane * VW;
    const device T* vsb = vs + b * vs_strides[0] + hkv * vs_strides[1] + vgroup;
    const device T* vbb = vb + b * vb_strides[0] + hkv * vb_strides[1] + vgroup;
    threadgroup float* my_p = p_sh + sg * GL * 32;

    float mmax[GL], lsum[GL], o[GL][EPL];
    for (int j = 0; j < GL; j++) {
        mmax[j] = -INFINITY; lsum[j] = 0.0f;
        for (int i = 0; i < EPL; i++) o[j][i] = 0.0f;
    }

    for (int tile = s0; tile < s1; tile += 32) {
        const int t = tile + int(lane);
        const bool valid = t < s1;
        float sc[GL];
        for (int j = 0; j < GL; j++) sc[j] = 0.0f;
        if (valid) {
            const device uint32_t* krow = kwb + t * kw_strides[2];
            for (int g = 0; g < NG; g++) {
                float part[GL];
                for (int j = 0; j < GL; j++) part[j] = 0.0f;
                for (int w = 0; w < KWG; w++) {
                    const uint word = krow[g * KWG + w];
                    for (int e = 0; e < 32 / KB; e++) {
                        const float c = float((word >> (e * KB)) & KMASK);
                        const int d = g * GS + w * (32 / KB) + e;
                        for (int j = 0; j < GL; j++) part[j] += q_sh[(h0 + j) * D + d] * c;
                    }
                }
                const float ksc = float(ksb[t * ks_strides[2] + g]);
                const float kbi = float(kbb[t * kb_strides[2] + g]);
                for (int j = 0; j < GL; j++) sc[j] += ksc * part[j] + kbi * qg_sh[(h0 + j) * NG + g];
            }
        }
        for (int j = 0; j < GL; j++) {
            const float tmax = simd_max(valid ? sc[j] : -INFINITY);
            const float nm = max(mmax[j], tmax);
            const float f = mmax[j] == -INFINITY ? 0.0f : fast::exp(mmax[j] - nm);
            const float pj = valid ? fast::exp(sc[j] - nm) : 0.0f;
            lsum[j] = lsum[j] * f + simd_sum(pj);
            mmax[j] = nm;
            for (int i = 0; i < EPL; i++) o[j][i] *= f;
            my_p[j * 32 + lane] = pj;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const int tn = min(32, s1 - tile);
        for (int u = 0; u < tn; u++) {
            const int tt = tile + u;
            const float vsc = float(vsb[tt * vs_strides[2]]);
            const float vbi = float(vbb[tt * vb_strides[2]]);
            float vv[EPL];
            for (int w = 0; w < VW; w++) {
                const uint word = vwb[tt * vw_strides[2] + w];
                for (int e = 0; e < 32 / VB; e++)
                    vv[w * (32 / VB) + e] = vsc * float((word >> (e * VB)) & VMASK) + vbi;
            }
            for (int j = 0; j < GL; j++) {
                const float pj = my_p[j * 32 + u];
                for (int i = 0; i < EPL; i++) o[j][i] += pj * vv[i];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int P = int(blocks) * NT;
    const int p = int(block) * NT + int(sg) / HS;
    for (int j = 0; j < GL; j++) {
        const int row = (b * Hq + hkv * G + h0 + j) * P + p;
        const float inv = lsum[j] > 0.0f ? 1.0f / lsum[j] : 0.0f;
        for (int i = 0; i < EPL; i++) part_o[size_t(row) * D + lane * EPL + i] = o[j][i] * inv;
        if (lane == 0) {
            part_l[row] = lsum[j];
            part_m[row] = lsum[j] > 0.0f ? mmax[j] : -1.0e30f;
        }
    }
"""

@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="mlx2_gqa_quantized_decode_attention",
        input_names=["q", "kw", "ks", "kb", "vw", "vs", "vb", "scale"],
        output_names=["part_o", "part_l", "part_m"],
        source=_SOURCE,
        ensure_row_contiguous=False,
    )


# Below this many cached positions the composed path wins (launch + merge
# overhead); measured 2026-09-23, qualification/runs/kvq-decode-20260923.
MIN_CONTEXT = 16384


def decode_kernel_config(n: int, gqa: int) -> tuple[int, int, int]:
    """(simdgroups, head_split, blocks) from the 2026-09-23 sweep on M5 Max.

    GQA 8 splits the group across two simdgroup halves (one lane scoring 8
    heads x 256 codes per token was too much work per thread); GQA <= 6
    keeps the group whole.
    """
    if gqa >= 8:
        return 4, 2, (256 if n >= 65536 else 128)
    return 2, 1, 64


def decode_attention_supported(queries, q_keys, q_values, *, group_size, key_bits,
                               value_bits, mask=None, sinks=None) -> bool:
    """True when the kernel covers this call; otherwise use the composed path."""
    if mask is not None or sinks is not None or mx.default_device() != mx.gpu:
        return False
    if len(q_keys) != 3 or len(q_values) != 3:  # normalized caches carry extra planes
        return False
    B, Hq, L, D = queries.shape
    Hkv = q_keys[0].shape[1]
    if L != 1 or Hq % Hkv or not 1 <= Hq // Hkv <= 8 or group_size != 64:
        return False
    for bits in (key_bits, value_bits):
        if bits not in (4, 8) or (D // 32) * bits % 32 or D not in (128, 256):
            return False
    return queries.dtype in (mx.bfloat16, mx.float16) and q_keys[1].dtype == queries.dtype


def use_decode_kernel(queries, q_keys, q_values, *, group_size, key_bits, value_bits,
                      mask=None, sinks=None) -> bool:
    """Supported, long enough to win, and not disabled by MLX2_QSDPA_DECODE_KERNEL=0."""
    if os.environ.get("MLX2_QSDPA_DECODE_KERNEL", "1") == "0":
        return False
    if q_keys[0].shape[2] < MIN_CONTEXT:
        return False
    return decode_attention_supported(queries, q_keys, q_values, group_size=group_size,
                                      key_bits=key_bits, value_bits=value_bits,
                                      mask=mask, sinks=sinks)


def gqa_quantized_decode_attention(
    queries: mx.array,
    q_keys: tuple,
    q_values: tuple,
    *,
    scale: float,
    group_size: int = 64,
    key_bits: int = 8,
    value_bits: int = 8,
    blocks: Optional[int] = None,
    simdgroups: Optional[int] = None,
    head_split: Optional[int] = None,
) -> mx.array:
    """``(B, Hq, 1, D)`` attention of one query row over quantized K/V views.

    Unset tuning arguments come from ``decode_kernel_config``.
    """
    B, Hq, _, D = queries.shape
    Hkv, N = q_keys[0].shape[1], q_keys[0].shape[2]
    G = Hq // Hkv
    tuned = decode_kernel_config(N, G)
    simdgroups = simdgroups or tuned[0]
    head_split = head_split or tuned[1]
    blocks = blocks or tuned[2]
    if G % head_split or simdgroups % head_split:
        raise ValueError("head_split must divide the GQA group and the simdgroup count")
    P = blocks * simdgroups // head_split
    part_o, part_l, part_m = _kernel()(
        inputs=[queries, *q_keys, *q_values, mx.array([scale], dtype=mx.float32)],
        template=[("T", queries.dtype), ("D", D), ("G", G), ("KB", key_bits),
                  ("VB", value_bits), ("NS", simdgroups), ("GS", group_size),
                  ("HS", head_split)],
        grid=(32 * simdgroups, B * Hkv, blocks),
        threadgroup=(32 * simdgroups, 1, 1),
        output_shapes=[(B, Hq, P, D), (B, Hq, P), (B, Hq, P)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    top = part_m.max(axis=-1, keepdims=True)
    weight = part_l * mx.exp(part_m - top)
    out = (weight[..., None] * part_o).sum(axis=-2) / weight.sum(axis=-1, keepdims=True)
    return out.astype(queries.dtype)[:, :, None, :]


# --- Unquantized (fp16/bf16) K/V: the same tiled design, no dequantization ---
# Experimental: benchmarked against mx.fast SDPA before any routing decision.

_SOURCE_FP = r"""
    constexpr int EPL = D / 32;

    threadgroup float q_sh[G * D];
    constexpr int GL = G / HS;
    threadgroup float p_sh[NS * GL * 32];

    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint flat = sg * 32 + lane;
    const uint bh = threadgroup_position_in_grid.y;
    const uint block = threadgroup_position_in_grid.z;
    const uint blocks = threadgroups_per_grid.z;

    const int Hkv = k_shape[1];
    const int N = k_shape[2];
    const int Hq = q_shape[1];
    const int b = bh / Hkv;
    const int hkv = bh % Hkv;

    for (uint idx = flat; idx < uint(G * D); idx += NS * 32) {
        const int j = idx / D, d = idx % D;
        q_sh[idx] = float(q[b * q_strides[0] + (hkv * G + j) * q_strides[1] + d * q_strides[3]]) * scale[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr int NT = NS / HS;
    const int hs = int(sg) % HS;
    const int h0 = hs * GL;
    const int chunk = (N + int(blocks) - 1) / int(blocks);
    const int kstart = int(block) * chunk;
    const int kend = min(N, kstart + chunk);
    const int sub = (chunk + NT - 1) / NT;
    const int s0 = kstart + (int(sg) / HS) * sub;
    const int s1 = min(kend, s0 + sub);

    const device T* kb0 = k + b * k_strides[0] + hkv * k_strides[1];
    const device T* vb0 = v + b * v_strides[0] + hkv * v_strides[1] + lane * EPL;
    threadgroup float* my_p = p_sh + sg * GL * 32;

    float mmax[GL], lsum[GL], o[GL][EPL];
    for (int j = 0; j < GL; j++) {
        mmax[j] = -INFINITY; lsum[j] = 0.0f;
        for (int i = 0; i < EPL; i++) o[j][i] = 0.0f;
    }

    for (int tile = s0; tile < s1; tile += 32) {
        const int t = tile + int(lane);
        const bool valid = t < s1;
        float sc[GL];
        for (int j = 0; j < GL; j++) sc[j] = 0.0f;
        if (valid) {
            const device vec<T, 4>* krow = (const device vec<T, 4>*)(kb0 + t * k_strides[2]);
            for (int d4 = 0; d4 < D / 4; d4++) {
                const float4 kv = float4(krow[d4]);
                for (int j = 0; j < GL; j++) {
                    const threadgroup float* qj = q_sh + (h0 + j) * D + d4 * 4;
                    sc[j] += qj[0] * kv.x + qj[1] * kv.y + qj[2] * kv.z + qj[3] * kv.w;
                }
            }
        }
        for (int j = 0; j < GL; j++) {
            const float tmax = simd_max(valid ? sc[j] : -INFINITY);
            const float nm = max(mmax[j], tmax);
            const float f = mmax[j] == -INFINITY ? 0.0f : fast::exp(mmax[j] - nm);
            const float pj = valid ? fast::exp(sc[j] - nm) : 0.0f;
            lsum[j] = lsum[j] * f + simd_sum(pj);
            mmax[j] = nm;
            for (int i = 0; i < EPL; i++) o[j][i] *= f;
            my_p[j * 32 + lane] = pj;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const int tn = min(32, s1 - tile);
        for (int u = 0; u < tn; u++) {
            const device T* vrow = vb0 + (tile + u) * v_strides[2];
            float vv[EPL];
            for (int i = 0; i < EPL; i++) vv[i] = float(vrow[i]);
            for (int j = 0; j < GL; j++) {
                const float pj = my_p[j * 32 + u];
                for (int i = 0; i < EPL; i++) o[j][i] += pj * vv[i];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int P = int(blocks) * NT;
    const int p = int(block) * NT + int(sg) / HS;
    for (int j = 0; j < GL; j++) {
        const int row = (b * Hq + hkv * G + h0 + j) * P + p;
        const float inv = lsum[j] > 0.0f ? 1.0f / lsum[j] : 0.0f;
        for (int i = 0; i < EPL; i++) part_o[size_t(row) * D + lane * EPL + i] = o[j][i] * inv;
        if (lane == 0) {
            part_l[row] = lsum[j];
            part_m[row] = lsum[j] > 0.0f ? mmax[j] : -1.0e30f;
        }
    }
"""


# Opt-in until an end-to-end A/B qualifies it: this would replace
# mx.fast SDPA on the default decode path of every model. The microbenchmark
# crossover (2026-09-23) is between 32K (kernel loses) and 128K (16-30% win).
FP_MIN_CONTEXT = 65536


def use_fp_decode_kernel(queries, keys, values, *, mask=None, sinks=None) -> bool:
    """Opt-in (MLX2_FP_DECODE_KERNEL=1) tiled decode for unquantized K/V."""
    if os.environ.get("MLX2_FP_DECODE_KERNEL", "0") != "1":
        return False
    if mask is not None or sinks is not None or mx.default_device() != mx.gpu:
        return False
    B, Hq, L, D = queries.shape
    Hkv, N = keys.shape[1], keys.shape[2]
    if L != 1 or N < FP_MIN_CONTEXT or Hq % Hkv or not 1 <= Hq // Hkv <= 8:
        return False
    if D not in (128, 256) or values.shape[-1] != D:
        return False
    return queries.dtype in (mx.bfloat16, mx.float16) and keys.dtype == queries.dtype == values.dtype

@lru_cache(maxsize=None)
def _kernel_fp():
    return mx.fast.metal_kernel(
        name="mlx2_gqa_decode_attention_fp",
        input_names=["q", "k", "v", "scale"],
        output_names=["part_o", "part_l", "part_m"],
        source=_SOURCE_FP,
        ensure_row_contiguous=False,
    )


def gqa_decode_attention_fp(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    blocks: Optional[int] = None,
    simdgroups: Optional[int] = None,
    head_split: Optional[int] = None,
) -> mx.array:
    """Tiled decode attention over unquantized K/V views (experimental)."""
    B, Hq, _, D = queries.shape
    Hkv, N = keys.shape[1], keys.shape[2]
    G = Hq // Hkv
    tuned = decode_kernel_config(N, G)
    simdgroups = simdgroups or tuned[0]
    head_split = head_split or tuned[1]
    blocks = blocks or tuned[2]
    if G % head_split or simdgroups % head_split:
        raise ValueError("head_split must divide the GQA group and the simdgroup count")
    P = blocks * simdgroups // head_split
    part_o, part_l, part_m = _kernel_fp()(
        inputs=[queries, keys, values, mx.array([scale], dtype=mx.float32)],
        template=[("T", queries.dtype), ("D", D), ("G", G), ("NS", simdgroups), ("HS", head_split)],
        grid=(32 * simdgroups, B * Hkv, blocks),
        threadgroup=(32 * simdgroups, 1, 1),
        output_shapes=[(B, Hq, P, D), (B, Hq, P), (B, Hq, P)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    top = part_m.max(axis=-1, keepdims=True)
    weight = part_l * mx.exp(part_m - top)
    out = (weight[..., None] * part_o).sum(axis=-2) / weight.sum(axis=-1, keepdims=True)
    return out.astype(queries.dtype)[:, :, None, :]
