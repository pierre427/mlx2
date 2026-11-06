# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
# Law v2 (SOURCE_V2_*): the 16-byte row loads with a shared Q/K dim permutation,
# the 16x32x32 cooperative-tensor layouts and the fp16 hi+lo P packing follow
# jundot/omlx #5084 at b0541976f5e5031c8ecd4c728f5a5810b6af4fdf (Apache-2.0);
# see provenance/qsa-nax-law-v2.json.
import os

import mlx.core as mx

from .import_env import snapshot as _import_env_snapshot

_import_env_snapshot(__name__)

BLOCK = 4
MTILE = 16
HEADER = "\n#include <metal_stdlib>\n#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\n"
SOURCE = "\n    constexpr int NSG = 4;\n    constexpr int DS  = D / NSG;\n    constexpr int NKS = DS / 16;\n    constexpr int NNT = DS / 32;\n\n    // Query length, KV length and ids stride are runtime uniforms, not template\n    // constants: every prefill-chunk width then shares ONE compiled pipeline.\n    const int L   = dims[0];   // query length\n    const int TOT = dims[1];   // KV length (physical_width)\n    const int U   = dims[2];   // per-token ids stride (u_width)\n\n    const ushort sg   = simdgroup_index_in_threadgroup;\n    const ushort lane = thread_index_in_simdgroup;\n    const uint  tok   = threadgroup_position_in_grid.y;   // query token\n    const uint  bkv   = threadgroup_position_in_grid.z;   // b * NKVH + hkv\n    const uint  b     = bkv / NKVH;\n    const uint  hkv   = bkv % NKVH;\n\n    const short qid = lane >> 2;\n    const short fm0 = ((qid & 4) | ((lane >> 1) & 3));    // row = query head\n    const short fn0 = ((qid & 2) | (lane & 1)) * 4;\n\n    threadgroup float sS[NSG * 32 * 16];\n\n    const size_t k_head =\n        (size_t)b * k_strides[0] + (size_t)hkv * k_strides[1];\n    const size_t v_head =\n        (size_t)b * v_strides[0] + (size_t)hkv * v_strides[1];\n\n    const int  lpad      = left_pad[b];\n    const uint tile_base = (b * L + tok) * U;\n    const uint cnt       = counts[b * L + tok];\n    const uint nsel      = n_sel[b * L + tok];\n\n    const int qp       = qpos[b * L + tok];\n    const int complete = ((qp + 1) / BS) * BS;\n\n    // Row r is query head hkv*GQA + r.  Rows GQA..15 are idle.\n    int head_of[2];\n    bool row_live[2];\n    for (short rr = 0; rr < 2; rr++) {\n        int r = fm0 + rr * 8;\n        row_live[rr] = (r < GQA);\n        head_of[rr] = (int)hkv * GQA + r;\n    }\n\n    typedef metal::vec<T, 8> frag_t;\n    frag_t qf[NKS];\n    for (short ks = 0; ks < NKS; ks++) {\n        for (short i = 0; i < 8; i++) {\n            short rr = (i >> 2);\n            int c = sg * DS + ks * 16 + fn0 + (i % 4);\n            qf[ks][i] = row_live[rr]\n                ? q[((size_t)(b * NQH + head_of[rr]) * L + tok) * D + c]\n                : T(0);\n        }\n    }\n\n    float of[NNT][16];\n    for (short t = 0; t < NNT; t++)\n        for (short i = 0; i < 16; i++) of[t][i] = 0.0f;\n    float rmax_r[2] = {-INFINITY, -INFINITY};\n    float rsum_r[2] = {0.0f, 0.0f};\n\n    constexpr auto desc_qk = mpp::tensor_ops::matmul2d_descriptor(\n        16, 32, 16, false, true, true,\n        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);\n    constexpr auto desc_pv = mpp::tensor_ops::matmul2d_descriptor(\n        16, 32, 16, false, false, true,\n        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);\n\n    const float qk_scale = scale[0];\n\n    for (uint u0 = 0; u0 < cnt; u0 += 8) {\n\n        uint physical_rows[4];\n        for (short qq = 0; qq < 4; qq++) {\n            int n = fm0 + qq * 8;\n            uint u = u0 + (uint)(n >> 2);\n            int bid = (u < cnt) ? (int)ids[tile_base + u] : 0;\n            int phys = lpad + bid * BS + (n & 3);\n            physical_rows[qq] = (uint)metal::clamp(phys, 0, (int)TOT - 1);\n        }\n\n        {\n            mpp::tensor_ops::matmul2d<desc_qk, metal::execution_simdgroup> op;\n            auto ca = op.get_left_input_cooperative_tensor<T, T, float>();\n            auto cb = op.get_right_input_cooperative_tensor<T, T, float>();\n            auto cc = op.get_destination_cooperative_tensor<\n                metal::remove_addrspace_t<decltype(ca)>,\n                metal::remove_addrspace_t<decltype(cb)>, float>();\n            for (short i = 0; i < 16; i++) cc[i] = 0.0f;\n            for (short ks = 0; ks < NKS; ks++) {\n                for (short i = 0; i < 8; i++) ca[i] = qf[ks][i];\n                for (short i = 0; i < 8; i++) {\n                    int c = sg * DS + ks * 16 + fn0 + (i % 4);\n                    const uint physical = physical_rows[(i >> 2)];\n                    const uint physical_2 = physical_rows[2 + (i >> 2)];\n                    cb[i] = k[\n                        k_head + (size_t)physical * k_strides[2]\n                        + (size_t)c * k_strides[3]\n                    ];\n                    cb[8 + i] = k[\n                        k_head + (size_t)physical_2 * k_strides[2]\n                        + (size_t)c * k_strides[3]\n                    ];\n                }\n                op.run(ca, cb, cc);\n            }\n            for (short i = 0; i < 16; i++)\n                sS[(sg * 32 + lane) * 16 + i] = cc[i];\n        }\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n        float s[16];\n        for (short i = 0; i < 16; i++) {\n            s[i] = (sS[(0 * 32 + lane) * 16 + i] + sS[(1 * 32 + lane) * 16 + i]\n                  + sS[(2 * 32 + lane) * 16 + i] + sS[(3 * 32 + lane) * 16 + i])\n                 * qk_scale;\n        }\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n        // One selection for the whole tile: the mask has NO row dependence.\n        bool colok[8];\n        for (short f = 0; f < 2; f++) {\n            for (short co = 0; co < 4; co++) {\n                int n = f * 16 + fn0 + co;\n                uint u = u0 + (uint)(n >> 2);\n                bool inb = (u < cnt);\n                int bid = inb ? (int)ids[tile_base + u] : 0;\n                int kl = bid * BS + (n & 3);\n                colok[f * 4 + co] = inb && (kl >= 0) && (kl <= qp)\n                                    && ((u < nsel) || (kl >= complete));\n            }\n        }\n        for (short f = 0; f < 2; f++)\n            for (short rr = 0; rr < 2; rr++)\n                for (short co = 0; co < 4; co++)\n                    if (!colok[f * 4 + co]) s[f * 8 + rr * 4 + co] = -INFINITY;\n\n        // The mask is shared across rows; the scores are not.  Max and sum\n        // stay per-row.\n        float rowmax[2];\n        for (short rr = 0; rr < 2; rr++) {\n            float m = -INFINITY;\n            for (short f = 0; f < 2; f++)\n                for (short co = 0; co < 4; co++)\n                    m = metal::max(m, s[f * 8 + rr * 4 + co]);\n            m = metal::max(m, metal::simd_shuffle_xor(m, ushort(1)));\n            m = metal::max(m, metal::simd_shuffle_xor(m, ushort(8)));\n            rowmax[rr] = m;\n        }\n\n        float alpha[2], p[16], newmax[2], newsum[2];\n        for (short rr = 0; rr < 2; rr++) {\n            float mn = metal::max(rmax_r[rr], rowmax[rr]);\n            bool live = mn > -INFINITY;\n            alpha[rr] = live ? metal::fast::exp(rmax_r[rr] - mn) : 1.0f;\n            float sm = 0.0f;\n            for (short f = 0; f < 2; f++) {\n                for (short co = 0; co < 4; co++) {\n                    short idx = f * 8 + rr * 4 + co;\n                    float pv = live ? metal::fast::exp(s[idx] - mn) : 0.0f;\n                    p[idx] = pv;\n                    sm += pv;\n                }\n            }\n            sm += metal::simd_shuffle_xor(sm, ushort(1));\n            sm += metal::simd_shuffle_xor(sm, ushort(8));\n            newmax[rr] = mn;\n            newsum[rr] = rsum_r[rr] * alpha[rr] + sm;\n        }\n        for (short rr = 0; rr < 2; rr++) {\n            rmax_r[rr] = newmax[rr];\n            rsum_r[rr] = newsum[rr];\n        }\n\n        for (short t = 0; t < NNT; t++) {\n            for (short f = 0; f < 2; f++)\n                for (short rr = 0; rr < 2; rr++)\n                    for (short co = 0; co < 4; co++)\n                        of[t][f * 8 + rr * 4 + co] *= alpha[rr];\n\n            int nbase = sg * DS + t * 32;\n            mpp::tensor_ops::matmul2d<desc_pv, metal::execution_simdgroup> op;\n            auto ca = op.get_left_input_cooperative_tensor<T, T, float>();\n            auto cb = op.get_right_input_cooperative_tensor<T, T, float>();\n            auto cc = op.get_destination_cooperative_tensor<\n                metal::remove_addrspace_t<decltype(ca)>,\n                metal::remove_addrspace_t<decltype(cb)>, float>();\n            for (short i = 0; i < 16; i++) cc[i] = of[t][i];\n            for (short ks = 0; ks < 2; ks++) {\n                for (short i = 0; i < 8; i++) ca[i] = (T)p[ks * 8 + i];\n                for (short i = 0; i < 8; i++) {\n                    uint physical = physical_rows[ks * 2 + (i >> 2)];\n                    int co = nbase + fn0 + (i % 4);\n                    cb[i] = v[\n                        v_head + (size_t)physical * v_strides[2]\n                        + (size_t)co * v_strides[3]\n                    ];\n                    cb[8 + i] = v[\n                        v_head + (size_t)physical * v_strides[2]\n                        + (size_t)(co + 16) * v_strides[3]\n                    ];\n                }\n                op.run(ca, cb, cc);\n            }\n            for (short i = 0; i < 16; i++) of[t][i] = cc[i];\n        }\n    }\n\n    for (short t = 0; t < NNT; t++) {\n        for (short f = 0; f < 2; f++) {\n            for (short rr = 0; rr < 2; rr++) {\n                if (!row_live[rr]) continue;\n                float inv = (rsum_r[rr] > 0.0f) ? (1.0f / rsum_r[rr]) : 0.0f;\n                for (short co = 0; co < 4; co++) {\n                    int c = sg * DS + t * 32 + f * 16 + fn0 + co;\n                    out[((size_t)(b * NQH + head_of[rr]) * L + tok) * D + c] =\n                        of[t][f * 8 + rr * 4 + co] * inv;\n                }\n            }\n        }\n    }\n"
_KERNEL = mx.fast.metal_kernel(
    name="nax_qsa_attn_b",
    input_names=[
        "q",
        "k",
        "v",
        "ids",
        "counts",
        "n_sel",
        "qpos",
        "left_pad",
        "scale",
        "dims",
    ],
    output_names=["out"],
    header=HEADER,
    source=SOURCE,
    ensure_row_contiguous=False,
)


# ---------------------------------------------------------------------------
# Law v2.  The same grid, threadgroup, inputs, ids ABI, masks, 4-simdgroup D
# split and fp32 online softmax as law v1 (SOURCE).  What changes:
#   * rows load 8 contiguous dims at a time (``fcol = 2 * fn0``): Q and K use
#     the same dim permutation, so the dot products are unchanged, and the
#     output store applies the inverse permutation;
#   * QK and PV run as 16x32x32 ops (K-width 32) instead of 16x32x16;
#   * P enters PV as bf16 (``v2``, today's P precision) or as fp16 hi + lo
#     (``v2_hilo``, two K-halves per PV op: ~fp32 P).
# Rows still own their threadgroup, so chunk invariance and per-lane batch
# independence hold as in v1.  Microbench (qualification/runs/
# qsa-nax-kernel-ab-20261105, M5 Max, synthetic Flash-Next and 27B
# geometries): v2 1.72-2.13x v1's kernel time at equal error, v2_hilo
# 1.54-1.90x at ~2,400x lower error against fp64.
# A K/V/Q layout that cannot take 16-byte loads (a non-unit last stride, a
# row stride not a multiple of 8 elements, or a misaligned base) takes an
# in-kernel scalar path with the same arithmetic, so the law does not depend
# on the cache's view layout.
HEADER_V2 = HEADER + r"""
template <typename T>
inline metal::vec<T, 8> qsa_nax_load8(const device T* p, int64_t step, bool vec) {
    if (vec) {
        return *(const device metal::vec<T, 8>*)p;
    }
    metal::vec<T, 8> r;
    for (short e = 0; e < 8; e++) r[e] = p[e * step];
    return r;
}

template <typename T>
inline bool qsa_nax_vec_ok(const device T* base, const constant int64_t* strides) {
    return strides[3] == 1 && (strides[2] % 8) == 0 && (strides[1] % 8) == 0
        && (strides[0] % 8) == 0 && ((reinterpret_cast<ulong>(base) & 15ul) == 0);
}
"""

_SOURCE_V2_HEAD = r"""
    constexpr int NSG = 4;
    constexpr int DS  = D / NSG;
    constexpr int NJ  = DS / 32;      // 16x32x32 QK ops per 32-key step
    constexpr int NNT = DS / 32;      // 32-wide output tiles

    const int L   = dims[0];
    const int TOT = dims[1];
    const int U   = dims[2];

    const ushort sg   = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const uint  tok   = threadgroup_position_in_grid.y;
    const uint  bkv   = threadgroup_position_in_grid.z;
    const uint  b     = bkv / NKVH;
    const uint  hkv   = bkv % NKVH;

    const short qid = lane >> 2;
    const short fm0 = ((qid & 4) | ((lane >> 1) & 3));
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;
    const short fcol = 2 * fn0;   // 8 contiguous dims feed a fragment pair

    threadgroup float sS[NSG * 32 * 16];

    const int64_t sk2 = k_strides[2], sk3 = k_strides[3];
    const int64_t sv2 = v_strides[2], sv3 = v_strides[3];
    const bool kvec = qsa_nax_vec_ok(k, k_strides);
    const bool vvec = qsa_nax_vec_ok(v, v_strides);
    const device T* kbase = k + (int64_t)b * k_strides[0] + (int64_t)hkv * k_strides[1]
                              + (int64_t)(sg * DS + fcol) * sk3;
    const device T* vbase = v + (int64_t)b * v_strides[0] + (int64_t)hkv * v_strides[1]
                              + (int64_t)(sg * DS + fcol) * sv3;

    const int  lpad      = left_pad[b];
    const uint tile_base = (b * L + tok) * U;
    const uint cnt       = counts[b * L + tok];
    const uint nsel      = n_sel[b * L + tok];
    const int qp       = qpos[b * L + tok];
    const int complete = ((qp + 1) / BS) * BS;

    // Row r is query head hkv*GQA + r.  Rows GQA..15 are idle.
    int head_of[2];
    bool row_live[2];
    for (short rr = 0; rr < 2; rr++) {
        int r = fm0 + rr * 8;
        row_live[rr] = (r < GQA);
        head_of[rr] = (int)hkv * GQA + r;
    }

    // q is row-contiguous (the wrapper makes it so) and D % 32 == 0; its
    // base may still be misaligned (a row-contiguous view at an odd offset).
    const bool qvec = ((reinterpret_cast<ulong>(q) & 15ul) == 0);
    metal::vec<T, 8> qa[NJ][2];
    for (short jj = 0; jj < NJ; jj++) {
        for (short rr = 0; rr < 2; rr++) {
            qa[jj][rr] = row_live[rr]
                ? qsa_nax_load8(q + ((size_t)(b * NQH + head_of[rr]) * L + tok) * D
                                  + sg * DS + 32 * jj + fcol, (int64_t)1, qvec)
                : metal::vec<T, 8>(T(0));
        }
    }

    float of[NNT][16];
    for (short t = 0; t < NNT; t++)
        for (short i = 0; i < 16; i++) of[t][i] = 0.0f;
    float rmax_r[2] = {-INFINITY, -INFINITY};
    float rsum_r[2] = {0.0f, 0.0f};

    constexpr auto desc_qk = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto desc_pv = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 32, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    const float qk_scale = scale[0];

    for (uint u0 = 0; u0 < cnt; u0 += 8) {
        // Keys fm0, fm0+8, fm0+16, fm0+24 of this 32-key step.
        uint prow[4];
        for (short qq = 0; qq < 4; qq++) {
            int n = fm0 + qq * 8;
            uint u = u0 + (uint)(n >> 2);
            int bid = (u < cnt) ? (int)ids[tile_base + u] : 0;
            int phys = lpad + bid * BS + (n & 3);
            prow[qq] = (uint)metal::clamp(phys, 0, (int)TOT - 1);
        }
        {
            mpp::tensor_ops::matmul2d<desc_qk, metal::execution_simdgroup> op;
            auto ca = op.template get_left_input_cooperative_tensor<T, T, float>();
            auto cb = op.template get_right_input_cooperative_tensor<T, T, float>();
            auto cc = op.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(ca)>,
                metal::remove_addrspace_t<decltype(cb)>, float>();
            for (short i = 0; i < 16; i++) cc[i] = 0.0f;
            for (short jj = 0; jj < NJ; jj++) {
                metal::vec<T, 8> kr[4];
                for (short r = 0; r < 4; r++)
                    kr[r] = qsa_nax_load8(kbase + (int64_t)prow[r] * sk2 + (int64_t)(32 * jj) * sk3, sk3, kvec);
                for (short j = 0; j < 4; j++) {
                    ca[j] = qa[jj][0][j];
                    ca[4 + j] = qa[jj][1][j];
                    ca[8 + j] = qa[jj][0][4 + j];
                    ca[12 + j] = qa[jj][1][4 + j];
                    cb[j] = kr[0][j];
                    cb[4 + j] = kr[1][j];
                    cb[8 + j] = kr[2][j];
                    cb[12 + j] = kr[3][j];
                    cb[16 + j] = kr[0][4 + j];
                    cb[20 + j] = kr[1][4 + j];
                    cb[24 + j] = kr[2][4 + j];
                    cb[28 + j] = kr[3][4 + j];
                }
                op.run(ca, cb, cc);
            }
            for (short i = 0; i < 16; i++)
                sS[(sg * 32 + lane) * 16 + i] = cc[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float s[16];
        for (short i = 0; i < 16; i++) {
            s[i] = (sS[(0 * 32 + lane) * 16 + i] + sS[(1 * 32 + lane) * 16 + i]
                  + sS[(2 * 32 + lane) * 16 + i] + sS[(3 * 32 + lane) * 16 + i])
                 * qk_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // One selection for the whole tile: the mask has NO row dependence.
        bool colok[8];
        for (short f = 0; f < 2; f++) {
            for (short co = 0; co < 4; co++) {
                int n = f * 16 + fn0 + co;
                uint u = u0 + (uint)(n >> 2);
                bool inb = (u < cnt);
                int bid = inb ? (int)ids[tile_base + u] : 0;
                int kl = bid * BS + (n & 3);
                colok[f * 4 + co] = inb && (kl >= 0) && (kl <= qp)
                                    && ((u < nsel) || (kl >= complete));
            }
        }
        for (short f = 0; f < 2; f++)
            for (short rr = 0; rr < 2; rr++)
                for (short co = 0; co < 4; co++)
                    if (!colok[f * 4 + co]) s[f * 8 + rr * 4 + co] = -INFINITY;

        float rowmax[2];
        for (short rr = 0; rr < 2; rr++) {
            float m = -INFINITY;
            for (short f = 0; f < 2; f++)
                for (short co = 0; co < 4; co++)
                    m = metal::max(m, s[f * 8 + rr * 4 + co]);
            m = metal::max(m, metal::simd_shuffle_xor(m, ushort(1)));
            m = metal::max(m, metal::simd_shuffle_xor(m, ushort(8)));
            rowmax[rr] = m;
        }
        float alpha[2], p[16];
        for (short rr = 0; rr < 2; rr++) {
            float mn = metal::max(rmax_r[rr], rowmax[rr]);
            bool live = mn > -INFINITY;
            alpha[rr] = live ? metal::fast::exp(rmax_r[rr] - mn) : 1.0f;
            float sm = 0.0f;
            for (short f = 0; f < 2; f++) {
                for (short co = 0; co < 4; co++) {
                    short idx = f * 8 + rr * 4 + co;
                    float pv = live ? metal::fast::exp(s[idx] - mn) : 0.0f;
                    p[idx] = pv;
                    sm += pv;
                }
            }
            sm += metal::simd_shuffle_xor(sm, ushort(1));
            sm += metal::simd_shuffle_xor(sm, ushort(8));
            rmax_r[rr] = mn;
            rsum_r[rr] = rsum_r[rr] * alpha[rr] + sm;
        }

        for (short t = 0; t < NNT; t++) {
            for (short f = 0; f < 2; f++)
                for (short rr = 0; rr < 2; rr++)
                    for (short co = 0; co < 4; co++)
                        of[t][f * 8 + rr * 4 + co] *= alpha[rr];
            mpp::tensor_ops::matmul2d<desc_pv, metal::execution_simdgroup> op;
"""

_SOURCE_V2_PV_BF16 = r"""
            auto ca = op.template get_left_input_cooperative_tensor<T, T, float>();
            auto cb = op.template get_right_input_cooperative_tensor<T, T, float>();
            auto cc = op.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(ca)>,
                metal::remove_addrspace_t<decltype(cb)>, float>();
            for (short i = 0; i < 16; i++) cc[i] = of[t][i];
            metal::vec<T, 8> vr[4];
            for (short r = 0; r < 4; r++)
                vr[r] = qsa_nax_load8(vbase + (int64_t)prow[r] * sv2 + (int64_t)(32 * t) * sv3, sv3, vvec);
            for (short i = 0; i < 16; i++) ca[i] = (T)p[i];
            for (short j = 0; j < 4; j++) {
                cb[j] = vr[0][j];
                cb[4 + j] = vr[1][j];
                cb[8 + j] = vr[0][4 + j];
                cb[12 + j] = vr[1][4 + j];
                cb[16 + j] = vr[2][j];
                cb[20 + j] = vr[3][j];
                cb[24 + j] = vr[2][4 + j];
                cb[28 + j] = vr[3][4 + j];
            }
            op.run(ca, cb, cc);
            for (short i = 0; i < 16; i++) of[t][i] = cc[i];
        }
    }
"""

_SOURCE_V2_PV_HILO = r"""
            auto ca = op.template get_left_input_cooperative_tensor<half, T, float>();
            auto cb = op.template get_right_input_cooperative_tensor<half, T, float>();
            auto cc = op.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(ca)>,
                metal::remove_addrspace_t<decltype(cb)>, float>();
            for (short i = 0; i < 16; i++) cc[i] = of[t][i];
            for (short f = 0; f < 2; f++) {
                const metal::vec<T, 8> ra = qsa_nax_load8(
                    vbase + (int64_t)prow[2 * f] * sv2 + (int64_t)(32 * t) * sv3, sv3, vvec);
                const metal::vec<T, 8> rb = qsa_nax_load8(
                    vbase + (int64_t)prow[2 * f + 1] * sv2 + (int64_t)(32 * t) * sv3, sv3, vvec);
                for (short j = 0; j < 4; j++) {
                    cb[j] = ra[j];
                    cb[4 + j] = rb[j];
                    cb[8 + j] = ra[4 + j];
                    cb[12 + j] = rb[4 + j];
                    cb[16 + j] = ra[j];
                    cb[20 + j] = rb[j];
                    cb[24 + j] = ra[4 + j];
                    cb[28 + j] = rb[4 + j];
                }
                for (short i = 0; i < 8; i++) {
                    const half hi = half(p[f * 8 + i]);
                    ca[i] = hi;
                    ca[8 + i] = half(p[f * 8 + i] - float(hi));
                }
                op.run(ca, cb, cc);
            }
            for (short i = 0; i < 16; i++) of[t][i] = cc[i];
        }
    }
"""

_SOURCE_V2_TAIL = r"""
    for (short t = 0; t < NNT; t++) {
        for (short f = 0; f < 2; f++) {
            for (short rr = 0; rr < 2; rr++) {
                if (!row_live[rr]) continue;
                float inv = (rsum_r[rr] > 0.0f) ? (1.0f / rsum_r[rr]) : 0.0f;
                for (short co = 0; co < 4; co++) {
                    int c = sg * DS + t * 32 + fcol + f * 4 + co;
                    out[((size_t)(b * NQH + head_of[rr]) * L + tok) * D + c] =
                        of[t][f * 8 + rr * 4 + co] * inv;
                }
            }
        }
    }
"""

SOURCE_V2_BF16 = _SOURCE_V2_HEAD + _SOURCE_V2_PV_BF16 + _SOURCE_V2_TAIL
SOURCE_V2_HILO = _SOURCE_V2_HEAD + _SOURCE_V2_PV_HILO + _SOURCE_V2_TAIL

_INPUT_NAMES = [
    "q", "k", "v", "ids", "counts", "n_sel", "qpos", "left_pad", "scale", "dims",
]

# Law ids, as receipts and the APCv2 numerics identity carry them.  "v1" is
# the default and the only qualified law; v2 and v2_hilo are opt-in
# (MLX_QWEN4_QSA_NAX_LAW, FlashNextPolicy.qsa_nax_law) until they pass the
# model test flow and Pierre makes one a default.
LAW_ENV = "MLX_QWEN4_QSA_NAX_LAW"
DEFAULT_LAW = "v1"
LAWS = {
    "v1": "qsa-nax-v1",
    "v2": "qsa-nax-v2-bf16p",
    "v2_hilo": "qsa-nax-v2-hilop",
}
_LAW_KERNELS = {"v1": _KERNEL}
_LAW_SOURCES = {"v2": SOURCE_V2_BF16, "v2_hilo": SOURCE_V2_HILO}


def resolve_law(raw):
    """The law name ``raw`` selects; empty or None is the default.  Fails
    closed on an unknown value instead of running another kernel."""
    value = (raw or "").strip().lower() or DEFAULT_LAW
    if value not in LAWS:
        raise ValueError(f"{LAW_ENV} must be one of {sorted(LAWS)}, got {raw!r}")
    return value


_LAW = resolve_law(os.environ.get(LAW_ENV))


def qsa_nax_law():
    """The law ``nax_qsa_attention`` runs when no ``law`` is passed."""
    return _LAW


def _law_kernel(law):
    kernel = _LAW_KERNELS.get(law)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"nax_qsa_attn_{law}",
            input_names=_INPUT_NAMES,
            output_names=["out"],
            header=HEADER_V2,
            source=_LAW_SOURCES[law],
            ensure_row_contiguous=False,
        )
        _LAW_KERNELS[law] = kernel
    return kernel


def block_sparse_layout_supported(head_dim, n_heads, n_kv_heads, block_size):
    """Whether the head-tiled NAX kernel can serve this attention geometry.

    Constraints come from the tile shape, not the KV length: the D axis is
    split across NSG=4 simdgroups then 16/32-wide NAX tiles (so DS = D/4 must
    be a multiple of 32), one KV head's query heads must fit a 16-row tile,
    and the block size is fixed at 4 (the QSA compression ratio).
    """
    if n_heads <= 0 or n_kv_heads <= 0 or n_heads % n_kv_heads:
        return False
    ds = head_dim // 4
    return (
        head_dim % 4 == 0
        and ds % 32 == 0
        and (block_size == BLOCK)
        and (n_heads // n_kv_heads <= MTILE)
    )


_NAX_AVAILABLE = None


def nax_kernel_available():
    """True iff the MPP/NAX kernel is validated and runs on this device.

    The minimal zero probe only establishes compilation: on M3 Pro the kernel
    compiles and returns finite zeros but corrupts nonzero production geometry.
    Keep every caller (including the direct-decode and stage-one experimental
    routes) inside the M5 envelope that passed the dense numerical oracle.
    Within that envelope the memoized probe still catches missing Metal 4 / MPP
    support so callers fall back instead of crashing a forward pass.
    """
    global _NAX_AVAILABLE
    if _NAX_AVAILABLE is not None:
        return _NAX_AVAILABLE
    if not mx.metal.is_available():
        _NAX_AVAILABLE = False
        return False
    try:
        if "M5" not in str(mx.device_info().get("device_name", "")):
            _NAX_AVAILABLE = False
            return False
    except (AttributeError, RuntimeError, TypeError):
        _NAX_AVAILABLE = False
        return False
    try:
        (total, dim) = (8, 256)
        q = mx.zeros((1, 2, 2, dim), dtype=mx.bfloat16)
        k = mx.zeros((1, 1, total, dim), dtype=mx.bfloat16)
        v = mx.zeros((1, 1, total, dim), dtype=mx.bfloat16)
        ids = mx.zeros((1, 2, 8), dtype=mx.uint32)
        counts = mx.ones((1, 2), dtype=mx.uint32)
        n_sel = mx.zeros((1, 2), dtype=mx.uint32)
        qpos = mx.array([[0, 1]], dtype=mx.int32)
        left_pad = mx.zeros((1,), dtype=mx.int32)
        out = nax_qsa_attention(
            q,
            k,
            v,
            ids,
            counts,
            n_sel,
            qpos,
            left_pad,
            scale=1.0 / 16.0,
            u_width=8,
            total=total,
            n_kv_heads=1,
        )
        mx.eval(out)
        _NAX_AVAILABLE = True
    except Exception as exc:
        from .served_exp import is_device_fault

        if is_device_fault(exc):
            # Not a kernel refusal: let serving recover, probe again.
            raise
        _NAX_AVAILABLE = False
    return _NAX_AVAILABLE


def compact_blocks_to_kernel_inputs(cb):
    """Turn a ``QSACompactBlocks`` into the kernel's per-token block list.

    ``cb.block_ids`` is the sorted, prefix-packed set of genuinely SELECTED
    logical blocks (``cb.block_counts`` long, suffix zeroed).  The kernel also
    needs the incomplete TAIL block appended at slot ``n_sel``; a slot at or
    past ``n_sel`` is admitted only by ``kl >= complete`` (the causal tail),
    never by membership.  The tail block is ``q_pos // block_size`` and is
    dropped from ``counts`` when it duplicates the last selected block
    (``q_pos == block_size - 1 mod block_size`` with that block selected), so
    the block list stays a set.

    Returns ``(ids, counts, n_sel, u_width, q_pos, left_pad, total)`` ready for
    ``nax_qsa_attention``.  The width is derived from the static block-id
    width, so this never syncs the device.
    """
    block_ids = cb.block_ids.astype(mx.int32)
    n_sel = cb.block_counts.astype(mx.int32)
    bs = cb.block_size
    total = cb.physical_width
    (batch, length, k_width) = block_ids.shape
    n_ext = -(-total // bs)
    q_pos = (cb.tail_stop - 1).astype(mx.int32)
    if q_pos.shape != (batch, length):
        q_pos = mx.broadcast_to(q_pos, (batch, length))
    tail = mx.clip(q_pos // bs, 0, n_ext - 1)
    last = mx.take_along_axis(
        block_ids, mx.clip(n_sel - 1, 0, k_width - 1)[..., None], axis=-1
    )[..., 0]
    dup = (n_sel > 0) & (last == tail)
    counts = n_sel + mx.where(dup, 0, 1)
    u_width = max(8, (k_width + 1 + 7) // 8 * 8)
    pad = u_width - k_width
    ids_full = mx.concatenate(
        [block_ids, mx.full((batch, length, pad), n_ext, dtype=mx.int32)], axis=-1
    )
    ids_full = mx.put_along_axis(ids_full, n_sel[..., None], tail[..., None], axis=-1)
    ids = mx.where(ids_full < n_ext, ids_full, 0).astype(mx.uint32)
    left_pad = cb.left_padding
    if left_pad is None:
        left_pad = mx.zeros((batch,), dtype=mx.int32)
    else:
        left_pad = left_pad.astype(mx.int32)
    return (
        ids,
        counts.astype(mx.uint32),
        n_sel.astype(mx.uint32),
        u_width,
        q_pos,
        left_pad,
        total,
    )


def compact_token_validity(cb):
    """Return compact token coordinates and the shared validity predicate."""
    (ids, counts, n_sel, u_width, q_pos, left_pad, total) = (
        compact_blocks_to_kernel_inputs(cb)
    )
    block_size = int(cb.block_size)
    logical = ids.astype(mx.int32)[..., None] * block_size + mx.arange(
        block_size, dtype=mx.int32
    )
    slots = mx.arange(u_width, dtype=mx.int32)[None, None, :, None]
    present = slots < counts.astype(mx.int32)[..., None, None]
    selected = slots < n_sel.astype(mx.int32)[..., None, None]
    tail = (logical >= cb.tail_start[..., None, None]) & (
        logical < cb.tail_stop[..., None, None]
    )
    valid = present & (selected | tail)
    physical = logical + left_pad[:, None, None, None]
    valid = valid & (physical >= 0) & (physical < total)
    valid = valid & (logical <= q_pos[..., None, None])
    physical = mx.clip(physical, 0, total - 1)
    if cb.causal_mask is not None:
        (batch, length) = ids.shape[:2]
        causal = mx.broadcast_to(cb.causal_mask, (batch, 1, length, total))[:, 0]
        gathered = mx.take_along_axis(
            causal, physical.reshape(batch, length, -1), axis=-1
        ).reshape(physical.shape)
        valid = valid & gathered
    return (ids, counts, n_sel, u_width, q_pos, left_pad, total, physical, valid)


def nax_qsa_attention(
    q, k, v, ids, counts, n_sel, q_pos, left_pad, *, scale, u_width, total, n_kv_heads,
    law=None,
):
    """Block-sparse QSA attention on NAX.  q [B,H,L,D], k/v [B,HKV,T,D] -> fp32.

    ``ids``/``counts``/``n_sel``/``q_pos``/``left_pad`` come from
    ``compact_blocks_to_kernel_inputs``.  Output is fp32 [B,H,L,D]; the caller
    casts to the attention dtype.  ``law`` (a ``LAWS`` key) defaults to the
    process law ``qsa_nax_law()``.
    """
    law = _LAW if law is None else resolve_law(law)
    if law != "v1" and q.dtype not in (mx.bfloat16, mx.float16):
        # The 16-byte load checks assume 2-byte elements.
        raise ValueError(f"NAX QSA law {law} needs bf16/fp16 q/k/v, got {q.dtype}")
    (batch, nqh, length, dim) = q.shape
    if nqh <= 0 or n_kv_heads <= 0 or nqh % n_kv_heads:
        # GQA >= 1: the kernel's q-head clamp min(head_of, NQH - 1) needs NQH >= 1.
        raise ValueError(f"NAX QSA needs a positive multiple of {n_kv_heads} query heads, got {nqh}")
    gqa = nqh // n_kv_heads
    if gqa > MTILE:
        raise ValueError("one token's heads must fit a 16-row NAX tile")
    (out,) = _law_kernel(law)(
        inputs=[
            mx.contiguous(q),
            k,
            v,
            mx.contiguous(ids.astype(mx.uint32)),
            mx.contiguous(counts.astype(mx.uint32)),
            mx.contiguous(n_sel.astype(mx.uint32)),
            mx.contiguous(mx.broadcast_to(q_pos, (batch, length)).astype(mx.int32)),
            mx.contiguous(left_pad.astype(mx.int32)),
            mx.array([scale], dtype=mx.float32),
            mx.array([length, total, u_width], dtype=mx.int32),
        ],
        template=[
            ("T", q.dtype),
            ("D", dim),
            ("NQH", nqh),
            ("NKVH", n_kv_heads),
            ("GQA", gqa),
            ("BS", BLOCK),
        ],
        grid=(128, length, batch * n_kv_heads),
        threadgroup=(128, 1, 1),
        output_shapes=[(batch, nqh, length, dim)],
        output_dtypes=[mx.float32],
    )
    return out
