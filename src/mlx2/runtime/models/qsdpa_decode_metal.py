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

# Host-side engagement count (read by qualification tracers).
STATS = {"decode_kernel_calls": 0, "fp_decode_kernel_calls": 0,
         "fp_decode_gqa16_calls": 0}


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
    STATS["decode_kernel_calls"] += 1
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
            const device T* krow = kb0 + t * k_strides[2];
            for (int d4 = 0; d4 < D / 4; d4++) {
                const int d = d4 * 4;
                const float4 kv = float4(float(krow[d * k_strides[3]]),
                    float(krow[(d + 1) * k_strides[3]]),
                    float(krow[(d + 2) * k_strides[3]]),
                    float(krow[(d + 3) * k_strides[3]]));
                for (int j = 0; j < GL; j++) {
                    const threadgroup float* qj = q_sh + (h0 + j) * D + d4 * 4;
                    sc[j] += qj[0] * kv.x + qj[1] * kv.y + qj[2] * kv.z + qj[3] * kv.w;
                }
            }
        }
        for (int j = 0; j < GL; j++) {
            const bool visible = valid && mask[b * mask_strides[0]
                + (hkv * G + h0 + j) * mask_strides[1] + t * mask_strides[3]];
            const float tmax = simd_max(visible ? sc[j] : -INFINITY);
            const float nm = max(mmax[j], tmax);
            const float f = mmax[j] == -INFINITY ? 0.0f : fast::exp(mmax[j] - nm);
            const float pj = visible ? fast::exp(sc[j] - nm) : 0.0f;
            lsum[j] = lsum[j] * f + simd_sum(pj);
            mmax[j] = nm;
            for (int i = 0; i < EPL; i++) o[j][i] *= f;
            my_p[j * 32 + lane] = pj;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const int tn = min(32, s1 - tile);
        for (int u = 0; u < tn; u++) {
            const device T* vrow = v + b * v_strides[0] + hkv * v_strides[1]
                + (tile + u) * v_strides[2];
            float vv[EPL];
            for (int i = 0; i < EPL; i++)
                vv[i] = float(vrow[(lane * EPL + i) * v_strides[3]]);
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


# One merge launch avoids materializing broadcast weights and weighted output
# tensors across several generic reductions for the GQA16 candidate.
_SOURCE_FP_MERGE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.x;
    constexpr int EPL = D / 32;
    threadgroup float weights[P];
    const device float* ml = part_m + row * P;
    const device float* ll = part_l + row * P;
    float local_max = -INFINITY;
    for (int p = lane; p < P; p += 32)
        local_max = max(local_max, ml[p]);
    const float top = simd_max(local_max);
    float local_sum = 0.0f;
    for (int p = lane; p < P; p += 32) {
        const float w = ll[p] * fast::exp(ml[p] - top);
        weights[p] = w;
        local_sum += w;
    }
    const float total = simd_sum(local_sum);
    const float inv = total > 0.0f ? 1.0f / total : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float accum[EPL];
    for (int i = 0; i < EPL; ++i) accum[i] = 0.0f;
    for (int p = 0; p < P; ++p) {
        const float w = weights[p];
        for (int i = 0; i < EPL; ++i)
            accum[i] += w * part_o[(size_t(row) * P + p) * D + lane * EPL + i];
    }
    for (int i = 0; i < EPL; ++i)
        output[row * D + lane * EPL + i] = T(accum[i] * inv);
"""


@lru_cache(maxsize=None)
def _kernel_fp_merge():
    return mx.fast.metal_kernel(
        name="mlx2_gqa16_fp_decode_merge",
        input_names=["part_o", "part_l", "part_m"],
        output_names=["output"], source=_SOURCE_FP_MERGE,
    )


# Opt-in until an end-to-end A/B qualifies it: this would replace
# mx.fast SDPA on the default decode path of every model. The microbenchmark
# crossover (2026-09-23) is between 32K (kernel loses) and 128K (16-30% win).
# That historical crossover does not transfer to GQA16: the 2026-11-03 M5
# Max D128/B1 sweep at 64K/128K was slower than stock, including a fused
# partial merge and eight launch schedules. Muse 4bit at 66,068 tokens passed
# ordinary-reference and cold/warm APCv2 checks, but no serving speedup was
# established. Keep this candidate off; evidence: qualification/runs/
# muse-fp-gqa16-20261103-r4/result.json (source a74315043).
FP_MIN_CONTEXT = 65536
FP_KERNEL_VERSION = "fp-gqa-strided-masked-v4"


def fp_decode_status():
    """Host dispatch counters; evaluated GPU evidence is recorded by probes."""
    return {"implemented": True, "selected": os.environ.get("MLX2_FP_DECODE_KERNEL", "0") == "1",
            "version": FP_KERNEL_VERSION, "min_context": FP_MIN_CONTEXT,
            "basis": "selected_host_graph_branch",
            "observed_host_dispatch": STATS["fp_decode_kernel_calls"] > 0,
            "counts": {key: value for key, value in STATS.items() if key.startswith("fp_")}}


def fp_decode_attention_supported(queries, keys, values, *, mask=None, sinks=None) -> bool:
    """Geometry covered by the unquantized kernel, independent of selection.

    Muse's 32Q/2KV geometry is admitted explicitly. Ratios 9--15 remain
    outside this contract; several cannot use the split-head schedule.
    All inputs may be strided, including along the head dimension.
    """
    if sinks is not None or mx.default_device() != mx.gpu:
        return False
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return False
    B, Hq, L, D = queries.shape
    Bkv, Hkv, N, Dkv = keys.shape
    if min(B, Hq, Hkv, N) <= 0 or L != 1 or Bkv != B:
        return False
    if Hq % Hkv or Hq // Hkv not in (*range(1, 9), 16):
        return False
    if D not in (128, 256) or Dkv != D or values.shape != keys.shape:
        return False
    if mask is not None:
        if not isinstance(mask, mx.array) or mask.dtype != mx.bool_ or mask.ndim > 4:
            return False
        shape = (1,) * (4 - mask.ndim) + tuple(mask.shape)
        if any(given not in (1, expected) for given, expected in zip(shape, (B, Hq, 1, N))):
            return False
    return queries.dtype in (mx.bfloat16, mx.float16) and keys.dtype == queries.dtype == values.dtype


def fp_decode_kernel_config(n: int, gqa: int) -> tuple[int, int, int]:
    """Eight heads per simdgroup at GQA16, from the bounded M5 sweep.

    This is an execution schedule, not a claim of a serving speedup. Keep
    the existing schedules for GQA <= 8 and the route opt-in.
    """
    if gqa == 16:
        return 4, 2, (256 if n >= 65536 else 128)
    return decode_kernel_config(n, gqa)


def use_fp_decode_kernel(queries, keys, values, *, mask=None, sinks=None) -> bool:
    """Opt-in (MLX2_FP_DECODE_KERNEL=1) tiled decode for unquantized K/V."""
    if os.environ.get("MLX2_FP_DECODE_KERNEL", "0") != "1":
        return False
    if not fp_decode_attention_supported(queries, keys, values, mask=mask, sinks=sinks):
        return False
    return keys.shape[2] >= FP_MIN_CONTEXT

@lru_cache(maxsize=None)
def _kernel_fp():
    return mx.fast.metal_kernel(
        name="mlx2_gqa_decode_attention_fp",
        input_names=["q", "k", "v", "scale", "mask"],
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
    mask=None,
    blocks: Optional[int] = None,
    simdgroups: Optional[int] = None,
    head_split: Optional[int] = None,
) -> mx.array:
    """Tiled decode attention over unquantized K/V views (experimental)."""
    if not fp_decode_attention_supported(queries, keys, values, mask=mask):
        raise ValueError("Unsupported floating-point decode attention geometry")
    B, Hq, _, D = queries.shape
    Hkv, N = keys.shape[1], keys.shape[2]
    G = Hq // Hkv
    tuned = fp_decode_kernel_config(N, G)
    simdgroups = tuned[0] if simdgroups is None else simdgroups
    head_split = tuned[1] if head_split is None else head_split
    blocks = tuned[2] if blocks is None else blocks
    if any(type(value) is not int or value <= 0 for value in (simdgroups, head_split, blocks)):
        raise ValueError("decode launch dimensions must be positive integers")
    if simdgroups > 32:
        raise ValueError("decode launch exceeds 1024 threads")
    if G % head_split or simdgroups % head_split:
        raise ValueError("head_split must divide the GQA group and the simdgroup count")
    if (G * D + simdgroups * (G // head_split) * 32) * 4 > 32768:
        raise ValueError("decode launch exceeds the 32 KiB threadgroup memory bound")
    P = blocks * simdgroups // head_split
    if G == 16 and P > 8192:
        raise ValueError("decode merge exceeds the 32 KiB threadgroup memory bound")
    STATS["fp_decode_kernel_calls"] += 1
    STATS["fp_decode_gqa16_calls"] += int(G == 16)
    part_o, part_l, part_m = _kernel_fp()(
        inputs=[queries, keys, values, mx.array([scale], dtype=mx.float32),
                mx.broadcast_to(mx.array(True) if mask is None else mask, (B, Hq, 1, N))],
        template=[("T", queries.dtype), ("D", D), ("G", G), ("NS", simdgroups), ("HS", head_split)],
        grid=(32 * simdgroups, B * Hkv, blocks),
        threadgroup=(32 * simdgroups, 1, 1),
        output_shapes=[(B, Hq, P, D), (B, Hq, P), (B, Hq, P)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    if G == 16:
        return _kernel_fp_merge()(
            inputs=[part_o, part_l, part_m],
            template=[("T", queries.dtype), ("D", D), ("P", P)],
            grid=(32 * B * Hq, 1, 1), threadgroup=(32, 1, 1),
            output_shapes=[(B, Hq, 1, D)], output_dtypes=[queries.dtype],
        )[0]
    top = part_m.max(axis=-1, keepdims=True)
    weight = part_l * mx.exp(part_m - top)
    denominator = weight.sum(axis=-1, keepdims=True)
    out = (weight[..., None] * part_o).sum(axis=-2) / mx.where(denominator > 0, denominator, 1)
    return out.astype(queries.dtype)[:, :, None, :]
