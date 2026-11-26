# SPDX-License-Identifier: Apache-2.0
# Adapted from jundot/omlx PR 5511 (head 6f82cca242bff6f61b0d05ec3c099350fc0f59b5),
# omlx/patches/inkling2/kernels.py ``gqa_flash_decode_attention`` and its three
# Metal kernels (Apache-2.0). Modified by mlx2; see docs/PROVENANCE.md and
# provenance/omlx-5511-inkling2-splitk.json.
"""GQA-packed split-K flash decoding for short Inkling 2 query blocks.

Opt-in (``inkling2_verify_splitk``), unqualified. Used for 1 <= L < 32 query
rows: DSpark speculative-verify rows (L = 8) and one-token decode (L = 1).

The composed path builds a dense additive mask (bias band, window, padding)
in row chunks and appends a zero K/V sink column, then runs SDPA; every row
chunk re-reads K/V. Here all ``G * L`` query rows that share one KV head are
packed into one tile, so each KV head's K/V is read once for all of them;
keys are split across threadgroups (split-K) and the partial softmax states
are merged by a second kernel. The bias, window, left padding and sink are
computed in-kernel: the sink is the softmax start (m = sink, l = 1, value 0).

Law (the same as ``banded_additive_mask`` plus the sink column):

    score(i, j) = scale * q_i . k_j + bias_i[d]   (bias only for d < E)
    d = offset + i - j; masked when d < 0, j < left_pad, or window > 0 and
    d >= window.

``q`` and ``rel`` arrive already multiplied by the log-scaling tau on global
layers (the caller scales them), exactly as the composed path receives them.

Precision: K/V/Q tiles and the probabilities are staged in FP16 for BF16
inputs (BF16 -> FP16 is exact for |x| in [2^-14, 65504]; Inkling's Q/K are
L2-clipped and V is RMS-normalized, so they are in range) with FP32
accumulation, FP32 online softmax and FP32 partial merges; FP32 inputs stage
in FP32. The reduction order differs from SDPA, so the result is near-exact,
not bitwise. The L == 1 register-streaming kernel stays in FP32 registers.

Changes from omlx: no FP8 E4M3 K/V path; an explicit query ``offset``
(omlx assumes the queries are the last L keys; mlx2 always calls it that
way); left padding clamped to >= 0; FP32 inputs supported (FP32 staging,
scalar loads) for tests; 16-byte K/V loads also require an aligned base
pointer and BF16 (otherwise the in-kernel scalar path), and the L == 1
direct kernel takes 8-byte BF16 loads under the same kind of guard
(``VECTOR_LOADS``); ring buffers are read in place (``ring=``; logical key
order and positions unchanged, so the result is bitwise the result on the
reordered copy).
"""

import mlx.core as mx

_PART_SOURCE = r"""
    constexpr int BK = 16;
    constexpr int NT = NSG * 32;
    constexpr float LOG2E = 1.4426950408889634f;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int sp = threadgroup_position_in_grid.x;
    const int hkv_qt = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;

    const int L = dims[0];
    const int S = dims[1];
    const int NS = dims[2];
    const int OFF = dims[3];
    // Ring read: logical key j lives at physical row (j + RB) mod RW
    // (RB = 0, RW = S for an ordinary chronological K/V).
    const int RB = dims[4];
    const int RW = dims[5];
    const int lp = max(dims[6 + b], 0);

    const int n_rows = G * L;
    const int BQ = NSG * 8;
    const int nqt = (n_rows + BQ - 1) / BQ;
    const int hkv = hkv_qt / nqt;
    const int qt = hkv_qt % nqt;

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const int packed_row = qt * BQ + int(sg) * 8 + fm;
    const bool row_ok = packed_row < n_rows;
    const int g_idx = row_ok ? (packed_row / L) : 0;
    const int q_pos = row_ok ? (packed_row % L) : 0;
    const int h = hkv * G + g_idx;
    const int a_row = OFF + q_pos;
    const float qscale = scale[0] * LOG2E;

    const int a_first = OFF;
    const int a_last = min(OFF + L, S) - 1;
    int lo = WINDOW > 0 ? max(0, a_first - WINDOW + 1) : 0;
    lo = max(lo, lp);
    const int nk = max(0, a_last + 1 - lo);
    const int per_sp = ((nk + NS - 1) / NS + BK - 1) / BK * BK;
    const int k_start = lo + sp * per_sp;
    const int k_end = min(a_last + 1, k_start + per_sp);

    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    threadgroup ST Ks[D * LDK];
    threadgroup ST Vs[BK * LDV];

    simdgroup_matrix<ST, 8, 8> Qf[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        float2 qv = float2(0.0f);
        if (row_ok) {
            const device T* qp = q + b * q_strides[0] + h * q_strides[1]
                + q_pos * q_strides[2] + (db * 8 + fn) * q_strides[3];
            qv = float2(float(qp[0]), float(qp[q_strides[3]]));
        }
        Qf[db].thread_elements()[0] = ST(qv.x);
        Qf[db].thread_elements()[1] = ST(qv.y);
    }
    simdgroup_float8x8 Of[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        Of[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    // One split: the sink is the softmax start here. Several: the merge
    // kernel starts from it and each split starts empty.
    float m = (SINGLE && row_ok) ? float(sink[(b * L + q_pos) * HQ + h]) * LOG2E : -INFINITY;
    float l = SINGLE ? 1.0f : 0.0f;
    const device T* relp = rel + ((size_t)(b * L + q_pos) * HQ + h) * R;

    const device T* kb_ptr = k + b * k_strides[0] + hkv * k_strides[1];
    const device T* vb_ptr = v + b * v_strides[0] + hkv * v_strides[1];
    const bool vec_kv = VEC_OK
        && k_strides[3] == 1 && v_strides[3] == 1
        && (k_strides[2] % 8 == 0) && (v_strides[2] % 8 == 0)
        && ((reinterpret_cast<ulong>(kb_ptr) & 15ul) == 0)
        && ((reinterpret_cast<ulong>(vb_ptr) & 15ul) == 0);

    for (int k0 = k_start; k0 < k_end; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (vec_kv) {
            // 16-byte loads of eight BF16 values; a BF16 is the high half
            // of an FP32, so the widening is a shift (exact).
            for (int c = int(tid); c < BK * D / 8; c += NT) {
                const int r = c / (D / 8);
                const int col = (c % (D / 8)) * 8;
                const int key = k0 + r;
                uint4 kw = uint4(0);
                uint4 vw = uint4(0);
                if (key < k_end) {
                    int row = key + RB;
                    row -= (row >= RW) ? RW : 0;
                    kw = *(const device uint4*)(kb_ptr + row * k_strides[2] + col);
                    vw = *(const device uint4*)(vb_ptr + row * v_strides[2] + col);
                }
                threadgroup ST* kd = Ks + col * LDK + r;
                threadgroup ST* vd = Vs + r * LDV + col;
                for (int j = 0; j < 4; ++j) {
                    kd[(2 * j) * LDK] = ST(as_type<float>(kw[j] << 16));
                    kd[(2 * j + 1) * LDK] = ST(as_type<float>(kw[j] & 0xFFFF0000u));
                    vd[2 * j] = ST(as_type<float>(vw[j] << 16));
                    vd[2 * j + 1] = ST(as_type<float>(vw[j] & 0xFFFF0000u));
                }
            }
        } else {
            for (int c = int(tid); c < BK * D; c += NT) {
                const int r = c / D;
                const int col = c % D;
                const int key = k0 + r;
                float kv = 0.0f;
                float vv = 0.0f;
                if (key < k_end) {
                    int row = key + RB;
                    row -= (row >= RW) ? RW : 0;
                    kv = float(kb_ptr[row * k_strides[2] + col * k_strides[3]]);
                    vv = float(vb_ptr[row * v_strides[2] + col * v_strides[3]]);
                }
                Ks[col * LDK + r] = ST(kv);
                Vs[r * LDV + col] = ST(vv);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (qt * BQ + int(sg) * 8 >= n_rows) {
            continue;
        }

        simdgroup_float8x8 Sf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            Sf[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_matrix<ST, 8, 8> Kt;
                simdgroup_load(Kt, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(Sf[kb], Qf[db], Kt, Sf[kb]);
            }
        }

        float rmax = -INFINITY;
        if (WINDOW == 0 && k0 >= lp && k0 + BK <= k_end && k0 + BK <= a_first - R + 1) {
            // Whole tile visible and past the bias band for every row.
            for (int kb = 0; kb < BK / 8; ++kb) {
                float s0 = row_ok ? Sf[kb].thread_elements()[0] * qscale : -INFINITY;
                float s1 = row_ok ? Sf[kb].thread_elements()[1] * qscale : -INFINITY;
                Sf[kb].thread_elements()[0] = s0;
                Sf[kb].thread_elements()[1] = s1;
                rmax = max(rmax, max(s0, s1));
            }
        } else {
            for (int kb = 0; kb < BK / 8; ++kb) {
                for (int e = 0; e < 2; ++e) {
                    const int key = k0 + kb * 8 + fn + e;
                    const int d = a_row - key;
                    const bool vis = row_ok && key < k_end && d >= 0 && key >= lp
                        && (WINDOW == 0 || d < WINDOW);
                    float s = Sf[kb].thread_elements()[e] * qscale;
                    if (vis) {
                        if (d < R) {
                            s += float(relp[d]) * LOG2E;
                        }
                    } else {
                        s = -INFINITY;
                    }
                    Sf[kb].thread_elements()[e] = s;
                    rmax = max(rmax, s);
                }
            }
        }
        // The four lanes of a fragment row differ in lane bits 0 and 3.
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));
        const float m_new = max(m, rmax);
        const float factor = (m == -INFINITY || m_new == -INFINITY)
            ? 0.0f : fast::exp2(m - m_new);
        float rsum = 0.0f;
        simdgroup_matrix<ST, 8, 8> Pf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float sv = Sf[kb].thread_elements()[e];
                const float p = (sv == -INFINITY) ? 0.0f : fast::exp2(sv - m_new);
                Pf[kb].thread_elements()[e] = ST(p);
                rsum += p;
            }
        }
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 8);
        l = l * factor + rsum;
        m = m_new;
        for (int db = 0; db < D / 8; ++db) {
            Of[db].thread_elements()[0] *= factor;
            Of[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_matrix<ST, 8, 8> Vm;
                simdgroup_load(Vm, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(Of[db], Pf[kb], Vm, Of[db]);
            }
        }
    }

    if (row_ok) {
        if (SINGLE) {
            const float inv = 1.0f / l;
            size_t base_o = (((size_t)b * HQ + h) * L + q_pos) * D;
            for (int db = 0; db < D / 8; ++db) {
                out_o[base_o + db * 8 + fn] = OT(Of[db].thread_elements()[0] * inv);
                out_o[base_o + db * 8 + fn + 1] = OT(Of[db].thread_elements()[1] * inv);
            }
        } else {
            size_t base_ml = (((size_t)b * HQ + h) * L + q_pos) * NS + sp;
            if (fn == 0) {
                out_m[base_ml] = m;
                out_l[base_ml] = l;
            }
            size_t base_o = base_ml * D;
            for (int db = 0; db < D / 8; ++db) {
                out_o[base_o + db * 8 + fn] = OT(Of[db].thread_elements()[0]);
                out_o[base_o + db * 8 + fn + 1] = OT(Of[db].thread_elements()[1]);
            }
        }
    }
"""

# Merge the split partials; the sink starts the softmax (m = sink, l = 1).
_MERGE_SOURCE = r"""
    constexpr float LOG2E = 1.4426950408889634f;
    const uint lane = thread_index_in_simdgroup;
    const int idx = threadgroup_position_in_grid.x;
    const int q_pos = idx % L;
    const int bh = idx / L;
    const int h = bh % HQ;
    const int b = bh / HQ;

    constexpr int D4 = D / 32;
    float m = float(sink[(b * L + q_pos) * HQ + h]) * LOG2E;
    float l = 1.0f;
    float ov[D4];
    for (int i = 0; i < D4; ++i) ov[i] = 0.0f;

    size_t ml_base = (size_t)idx * NS;
    for (int sp = 0; sp < NS; ++sp) {
        float lsp = part_l[ml_base + sp];
        if (lsp == 0.0f) continue;
        float mp = part_m[ml_base + sp];
        float m_new = max(m, mp);
        float a0 = fast::exp2(m - m_new);
        float a1 = fast::exp2(mp - m_new);
        l = l * a0 + lsp * a1;
        m = m_new;
        const device float* pp = part_o + (ml_base + sp) * D;
        for (int i = 0; i < D4; ++i) {
            ov[i] = ov[i] * a0 + pp[int(lane) * D4 + i] * a1;
        }
    }
    float inv = 1.0f / l;
    device T* dst = out + (size_t)idx * D;
    for (int i = 0; i < D4; ++i) {
        dst[int(lane) * D4 + i] = T(ov[i] * inv);
    }
"""

# inkling2_gate_merge: the merge's store with Attention._gate's arithmetic
# (MLX's Sigmoid and Multiply functors, compiled from the installed headers).
_GATED_STORE = r"""
    const size_t row = ((size_t)(b * L + q_pos) * HQ + h) * D;
    for (int i = 0; i < D4; ++i) {
        const float o = float(T(ov[i] * inv));
        const float g = float(gate[row + int(lane) * D4 + i]);
        out[row + int(lane) * D4 + i] = T(Multiply()(o, Sigmoid()(g)));
    }
"""

# L == 1, G <= 6: register streaming, no threadgroup K/V staging and no
# barriers inside the key loop; each simdgroup walks every NSG-th key and
# reads each K/V vector once for all G query heads.
_DIRECT_SOURCE = r"""
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int D4 = D / 32;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int sp = threadgroup_position_in_grid.x;
    const int hkv = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;

    const int S = dims[0];
    const int NS = dims[1];
    const int OFF = dims[2];
    // Ring read, as in the part kernel: logical key j at row (j + RB) mod RW.
    const int RB = dims[3];
    const int RW = dims[4];
    const int lp = max(dims[5 + b], 0);
    const float qscale = scale[0] * LOG2E;

    const int a_row = OFF;
    const int a_last = min(a_row, S - 1);
    int lo = WINDOW > 0 ? max(0, a_row - WINDOW + 1) : 0;
    lo = max(lo, lp);
    const int nk = max(0, a_last + 1 - lo);
    const int per_sp = (nk + NS - 1) / NS;
    const int k_start = lo + sp * per_sp;
    const int k_end = min(a_last + 1, k_start + per_sp);

    float qv[G][D4];
    float ov[G][D4];
    float mv[G];
    float lv[G];
    const device T* relp[G];
    const int d_off = int(lane) * D4;
    for (int g_idx = 0; g_idx < G; ++g_idx) {
        int h = hkv * G + g_idx;
        const device T* qp = q + b * q_strides[0] + h * q_strides[1]
            + d_off * q_strides[3];
        for (int i = 0; i < D4; ++i) {
            qv[g_idx][i] = float(qp[i * q_strides[3]]) * qscale;
            ov[g_idx][i] = 0.0f;
        }
        mv[g_idx] = -INFINITY;
        lv[g_idx] = 0.0f;
        relp[g_idx] = rel + ((size_t)b * HQ + h) * R;
    }

    const device T* kb_ptr = k + b * k_strides[0] + hkv * k_strides[1];
    const device T* vb_ptr = v + b * v_strides[0] + hkv * v_strides[1];
    const size_t ks2 = k_strides[2];
    const size_t vs2 = v_strides[2];
    const size_t ks3 = k_strides[3];
    const size_t vs3 = v_strides[3];
    // VEC (BF16, D4 % 4 == 0): each lane's D4 contiguous values as D4 / 4
    // 8-byte loads when the rows are unit-stride and every row start is
    // 8-byte aligned; the same values (a BF16 is the high half of an FP32,
    // so the widening is a shift, exact) in the same registers, so the
    // arithmetic below is unchanged. Otherwise the scalar loads.
    const bool vec_kv = VEC
        && ks3 == 1 && vs3 == 1 && (ks2 % 4 == 0) && (vs2 % 4 == 0)
        && ((reinterpret_cast<ulong>(kb_ptr) & 7ul) == 0)
        && ((reinterpret_cast<ulong>(vb_ptr) & 7ul) == 0);

    for (int key = k_start + int(sg); key < k_end; key += NSG) {
        float kv[D4];
        float vv[D4];
        int row = key + RB;
        row -= (row >= RW) ? RW : 0;
        const device T* kp = kb_ptr + row * ks2;
        const device T* vp = vb_ptr + row * vs2;
        if (VEC && vec_kv) {
            for (int c = 0; c < D4 / 4; ++c) {
                const uint2 kw = *(const device uint2*)(kp + d_off + 4 * c);
                const uint2 vw = *(const device uint2*)(vp + d_off + 4 * c);
                kv[4 * c] = as_type<float>(kw.x << 16);
                kv[4 * c + 1] = as_type<float>(kw.x & 0xFFFF0000u);
                kv[4 * c + 2] = as_type<float>(kw.y << 16);
                kv[4 * c + 3] = as_type<float>(kw.y & 0xFFFF0000u);
                vv[4 * c] = as_type<float>(vw.x << 16);
                vv[4 * c + 1] = as_type<float>(vw.x & 0xFFFF0000u);
                vv[4 * c + 2] = as_type<float>(vw.y << 16);
                vv[4 * c + 3] = as_type<float>(vw.y & 0xFFFF0000u);
            }
        } else {
            for (int i = 0; i < D4; ++i) {
                kv[i] = float(kp[(d_off + i) * ks3]);
                vv[i] = float(vp[(d_off + i) * vs3]);
            }
        }
        const int d = a_row - key;
        const bool has_rel = d < R;
        for (int g_idx = 0; g_idx < G; ++g_idx) {
            float dot = 0.0f;
            for (int i = 0; i < D4; ++i) dot += kv[i] * qv[g_idx][i];
            dot = simd_sum(dot);
            if (has_rel) dot += float(relp[g_idx][d]) * LOG2E;
            float m_new = max(mv[g_idx], dot);
            float f_old = (mv[g_idx] == -INFINITY) ? 0.0f : fast::exp2(mv[g_idx] - m_new);
            float p = fast::exp2(dot - m_new);
            lv[g_idx] = lv[g_idx] * f_old + p;
            mv[g_idx] = m_new;
            for (int i = 0; i < D4; ++i) {
                ov[g_idx][i] = ov[g_idx][i] * f_old + p * vv[i];
            }
        }
    }

    threadgroup float tg_m[NSG][G];
    threadgroup float tg_l[NSG][G];
    threadgroup float tg_o[NSG][G][D];
    for (int g_idx = 0; g_idx < G; ++g_idx) {
        if (lane == 0) {
            tg_m[sg][g_idx] = mv[g_idx];
            tg_l[sg][g_idx] = lv[g_idx];
        }
        for (int i = 0; i < D4; ++i) {
            tg_o[sg][g_idx][d_off + i] = ov[g_idx][i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        for (int g_idx = 0; g_idx < G; ++g_idx) {
            int h = hkv * G + g_idx;
            float m = -INFINITY;
            float l = 0.0f;
            float oo[D4];
            for (int i = 0; i < D4; ++i) oo[i] = 0.0f;
            for (int s = 0; s < NSG; ++s) {
                float ls = tg_l[s][g_idx];
                if (ls == 0.0f) continue;
                float ms = tg_m[s][g_idx];
                float mn = max(m, ms);
                float a0 = (m == -INFINITY) ? 0.0f : fast::exp2(m - mn);
                float a1 = fast::exp2(ms - mn);
                l = l * a0 + ls * a1;
                m = mn;
                for (int i = 0; i < D4; ++i) {
                    oo[i] = oo[i] * a0 + tg_o[s][g_idx][d_off + i] * a1;
                }
            }
            size_t base_ml = ((size_t)b * HQ + h) * NS + sp;
            if (lane == 0) {
                out_m[base_ml] = m;
                out_l[base_ml] = l;
            }
            size_t base_o = base_ml * D;
            for (int i = 0; i < D4; ++i) {
                out_o[base_o + d_off + i] = oo[i];
            }
        }
    }
"""

MAX_ROWS = 32  # engaged for 1 <= L < MAX_ROWS
# Sliding rings (RotatingKVCache, BatchRotatingKVCache) are read in place by
# the kernels (``ring=``) instead of reordered into a chronological copy
# first, which wrote and re-read the whole ring on every decode step once it
# had wrapped (2026-11-22). Module switch for the A/B: False restores the copy.
# DEFAULT ON (Pierre 2026-11-23), Inkling-2-Mini@440956c8 at 2946cb9fd:
# tests/test_inkling2_bitwise_io_metal.py bitwise on Metal (536/536 ring kernel
# cases and every model-level ring case, no declines; run w1122-2946cb9fd-
# metal-io-r2), and the in-process A/B (w1122-2946cb9fd-ab-io-ring-r2) decoded
# bitwise with decode +3.7% at 1K, +3.5% at 16K, +5.4% at 128K (2 blocks per
# cell, no thermal hold); prefill +12% at 1K, flat at 16K, -4% at 128K (one
# sample each).
RING_DIRECT = True
# The L == 1 direct kernel loads each lane's K/V values with 8-byte vector
# loads for BF16 when D / 32 is a multiple of 4 (D = 128: one uint2 per lane
# per row, was four 2-byte loads), unit stride and 8-byte aligned rows; the
# values and the arithmetic order are unchanged (2026-11-22). Module switch
# for the A/B: False restores the scalar loads. Bitwise on Metal (metal-io-r2),
# but kept OFF: alone it decoded 4-5% slower (Inkling-2-Mini@440956c8,
# 2026-11-22 A/B), and with ring_direct and sink_slot_spare (ab-io-all-r2) it
# added nothing over ring_direct alone (+3.8/+2.7/+0.4% at 1K/16K/128K vs
# +3.7/+3.5/+5.4%).
VECTOR_LOADS = False


def direct_vector_loads(dtype, dim):
    """The direct kernel's ``VEC`` template value for these K/V (the kernel
    still checks strides and alignment and takes the scalar loads otherwise)."""
    return int(VECTOR_LOADS and dtype == mx.bfloat16 and (dim // 32) % 4 == 0)
_DIRECT_MAX_GROUP = 6
_DIRECT_MIN_KEYS = 2048
_kernels = {}


def _kernel(name, inputs, outputs, source):
    kernel = _kernels.get(name)
    if kernel is None:
        kernel = _kernels[name] = mx.fast.metal_kernel(
            name=f"inkling2_splitk_{name}", input_names=inputs, output_names=outputs,
            source=source, ensure_row_contiguous=False,
        )
    return kernel


def decline_reason(q, k, v, *, n_heads, n_kv, head_dim, device=None):
    """None when ``gqa_splitk_attention`` can run these arrays, else a reason.

    ``device`` defaults to the current default device (tests pass ``mx.gpu``
    to check the other gates on a CPU host)."""
    if not 1 <= q.shape[2] < MAX_ROWS:
        return "rows"
    if (mx.default_device() if device is None else device) != mx.gpu:
        return "cpu"
    if not (isinstance(k, mx.array) and isinstance(v, mx.array)):
        return "cache_format"
    if q.dtype not in (mx.bfloat16, mx.float32) or k.dtype != q.dtype or v.dtype != q.dtype:
        return "dtype"
    if head_dim % 32 or head_dim > 128 or n_heads % n_kv:
        return "geometry"
    return None


# Module switch (default off, A/B only; oMLX #5620's mechanism, Apache-2.0,
# from its description): one-token direct calls split the keys in fixed
# 256-key blocks instead of capping the split count at 32, so long contexts
# get more threadgroups (32K keys: 128 splits of 256 instead of 32 of 1024).
# The merge loops over any number of partials. Near-exact: the order of the
# key reduction changes with the split boundaries.
SPLIT_FIXED256 = False
_FIXED256_MAX_SPLITS = 1024


def _splits(n_keys, base_tgs, *, direct):
    # omlx's split-count heuristics, unchanged.
    if direct and SPLIT_FIXED256:
        return 1 if n_keys <= 256 else min(_FIXED256_MAX_SPLITS, (n_keys + 255) // 256)
    if direct:
        ns = 1 if n_keys <= 256 else min(32, max(1, (n_keys + 255) // 256))
        if n_keys >= 2048 and ns < 16 and base_tgs <= 16:
            ns = 16
        if n_keys >= 8192 and ns < 32 and base_tgs <= 16:
            ns = 32
        return ns
    ns = (1 if n_keys <= 128 else
          min(32, max(1, min((n_keys + 63) // 64, (64 + base_tgs - 1) // base_tgs))))
    if n_keys >= 512 and ns < 8 and base_tgs <= 16:
        ns = 8
    if n_keys >= 2048 and ns < 16 and base_tgs <= 16:
        ns = 16
    if n_keys >= 8192 and ns < 32 and base_tgs <= 16:
        ns = 32
    return ns


def _ring_args(k, ring):
    """``(base, size, width)``: logical key j at physical row (j + base) % width."""
    width = k.shape[2]
    if ring is None:
        return 0, width, width
    base, size = int(ring[0]), int(ring[1])
    if not (0 <= base < width and 0 < size <= width):
        raise ValueError(f"ring {ring} does not fit a {width}-row buffer")
    return base, size, width


def ring_gather(a, ring):
    """The chronological K/V a ring read sees: ``a``'s rows ``(j + base) % width``
    for j < size (the reference for ``ring``; the kernel reads in place)."""
    base, size, width = _ring_args(a, ring)
    return mx.take(a, (mx.arange(size, dtype=mx.int32) + base) % width, axis=2)


def gqa_splitk_attention(q, k, v, rel, sinks, *, offset, left_padding, window, scale,
                         ring=None):
    """Short-block attention with Inkling's relative bias, window and sink.

    ``partials`` then ``merge`` (see ``partials`` for the arguments).
    Returns [B, H, L, D] in q's dtype. GPU only; check ``decline_reason``
    first."""
    parts = partials(q, k, v, rel, sinks, offset=offset, left_padding=left_padding,
                     window=window, scale=scale, ring=ring)
    if parts[0] == "out":
        return parts[1]
    return merge(q, *parts[1:])


def merge(q, po, pm, pl, ns, sink):
    """The split partials merged with the sink: [B, H, L, D] in q's dtype."""
    b, hq, length, dim = q.shape
    kernel = _kernel("merge", ["part_o", "part_m", "part_l", "sink"], ["out"], _MERGE_SOURCE)
    return kernel(
        inputs=[po, pm, pl, sink],
        template=[("T", q.dtype), ("D", dim), ("HQ", hq), ("L", length), ("NS", ns)],
        grid=(b * hq * length * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(b, hq, length, dim)],
        output_dtypes=[q.dtype],
    )[0]


_gated_kernel = None
_PLAIN_STORE = """    device T* dst = out + (size_t)idx * D;
    for (int i = 0; i < D4; ++i) {
        dst[int(lane) * D4 + i] = T(ov[i] * inv);
    }
"""


def merge_gated(q, po, pm, pl, ns, sink, gate):
    """``inkling2_gate_merge``: ``merge`` with the output gate applied, in
    the gate's layout ([B, L, H * D]): each value is the merge's rounded
    output times MLX's own ``Sigmoid`` of the FP32 gate, rounded once more,
    as ``Attention._gate`` computes it after the merge. GPU only."""
    global _gated_kernel
    from . import library_kernels

    header, why = library_kernels.library("elementwise")
    if header is None:
        raise RuntimeError(why)
    if _gated_kernel is None:
        if _MERGE_SOURCE.count(_PLAIN_STORE) != 1:
            raise RuntimeError("split-K merge store changed")
        _gated_kernel = mx.fast.metal_kernel(
            name="inkling2_splitk_merge_gated",
            input_names=["part_o", "part_m", "part_l", "sink", "gate"], output_names=["out"],
            header=header, source=_MERGE_SOURCE.replace(_PLAIN_STORE, _GATED_STORE),
            ensure_row_contiguous=False)
    b, hq, length, dim = q.shape
    return _gated_kernel(
        inputs=[po, pm, pl, sink, mx.contiguous(gate)],
        template=[("T", q.dtype), ("D", dim), ("HQ", hq), ("L", length), ("NS", ns)],
        grid=(b * hq * length * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(b, length, hq * dim)],
        output_dtypes=[q.dtype],
    )[0]


def partials(q, k, v, rel, sinks, *, offset, left_padding, window, scale, ring=None):
    """``("out", out)`` when one split writes the result directly, else
    ``("split", po, pm, pl, ns, sink)`` for ``merge``.

    Same arguments as ``flash_relative_attention``: q [B, H, L, D]; k, v
    [B, Hk, S, D] (any strides); rel [B, L, H, E], the per-query bias at
    distances below E; sinks [B, L, H]; queries at key positions
    ``offset + i``. GPU only; check ``decline_reason`` first.

    ``ring=(base, size)`` reads a ring buffer in place: k, v are the
    physical [B, Hk, W, D] buffers and logical key j (0 <= j < size, in
    chronological order) is row ``(j + base) % W``. Everything else (offset,
    left padding, window, bias distances, splits) is in logical positions,
    and the kernels walk the logical keys in the same order, so the result
    is bitwise the result on ``ring_gather(k, ring)`` (the reordered copy).
    """
    b, hq, length, dim = q.shape
    hkv = k.shape[1]
    ring_base, size, ring_width = _ring_args(k, ring)
    if v.shape[2] != k.shape[2]:
        raise ValueError("K and V buffers differ in rows")
    g = hq // hkv
    extent = rel.shape[-1]
    window = int(window)
    offset = int(offset)
    pad = (mx.zeros((b,), mx.int32) if left_padding is None
           else mx.maximum(left_padding, 0).astype(mx.int32))
    sc = mx.array([float(scale)], dtype=mx.float32)
    sink = mx.contiguous(sinks.astype(mx.float32))
    rel = mx.contiguous(rel.astype(q.dtype))
    n_keys = min(size, offset + length)
    if window:
        n_keys = min(n_keys, window + length - 1)
    if length == 1 and g <= _DIRECT_MAX_GROUP and n_keys >= _DIRECT_MIN_KEYS:
        direct = _kernel("direct", ["q", "k", "v", "rel", "dims", "scale"],
                         ["out_o", "out_m", "out_l"], _DIRECT_SOURCE)
        nsg = 4
        ns = _splits(n_keys, b * hkv, direct=True)
        dims = mx.concatenate(
            [mx.array([size, ns, offset, ring_base, ring_width], dtype=mx.int32), pad])
        po, pm, pl = direct(
            inputs=[q, k, v, rel, dims, sc],
            template=[("T", q.dtype), ("D", dim), ("HQ", hq), ("G", g), ("R", extent),
                      ("WINDOW", window), ("NSG", nsg),
                      ("VEC", direct_vector_loads(k.dtype, dim))],
            grid=(ns * nsg * 32, hkv, b),
            threadgroup=(nsg * 32, 1, 1),
            output_shapes=[(b, hq, 1, ns, dim), (b, hq, 1, ns), (b, hq, 1, ns)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
        return "split", po, pm, pl, ns, sink

    part = _kernel("part", ["q", "k", "v", "rel", "sink", "dims", "scale"],
                   ["out_o", "out_m", "out_l"], _PART_SOURCE)
    n_rows = g * length
    nsg = min(4, max(1, (n_rows + 7) // 8))
    nqt = (n_rows + nsg * 8 - 1) // (nsg * 8)
    ns = _splits(n_keys, b * hkv * nqt, direct=False)
    dims = mx.concatenate(
        [mx.array([length, size, ns, offset, ring_base, ring_width], dtype=mx.int32), pad])
    template = [
        ("T", q.dtype), ("ST", mx.float32 if q.dtype == mx.float32 else mx.float16),
        ("OT", q.dtype if ns == 1 else mx.float32), ("D", dim), ("HQ", hq), ("G", g),
        ("R", extent), ("WINDOW", window), ("NSG", nsg), ("SINGLE", int(ns == 1)),
        ("VEC_OK", int(q.dtype == mx.bfloat16)),
    ]
    nt = nsg * 32
    if ns == 1:
        return "out", part(
            inputs=[q, k, v, rel, sink, dims, sc], template=template,
            grid=(nt, hkv * nqt, b), threadgroup=(nt, 1, 1),
            output_shapes=[(b, hq, length, dim), (1,), (1,)],
            output_dtypes=[q.dtype, mx.float32, mx.float32],
        )[0]
    po, pm, pl = part(
        inputs=[q, k, v, rel, sink, dims, sc], template=template,
        grid=(ns * nt, hkv * nqt, b), threadgroup=(nt, 1, 1),
        output_shapes=[(b, hq, length, ns, dim), (b, hq, length, ns), (b, hq, length, ns)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return "split", po, pm, pl, ns, sink


def reference_attention(q, k, v, rel, sinks, *, offset, left_padding, window, scale,
                        ring=None):
    """The composed path's law, same arguments as ``gqa_splitk_attention``.

    A dense additive mask built exactly as ``banded_additive_mask`` builds it
    from bias rows (``rel`` = r @ rel_proj), a zero K/V sink column carrying
    each query's sink logit, and one SDPA: what ``Attention.__call__`` runs
    when the option is off (one mask chunk). Used by the tests and the A/B.
    A ``ring`` is read as its reordered copy (``ring_gather``), which is what
    the composed path attends to.
    """
    if ring is not None:
        k, v = ring_gather(k, ring), ring_gather(v, ring)
    b, h, length, d = q.shape
    size, extent = k.shape[2], rel.shape[-1]
    dtype = rel.dtype
    dist = (mx.arange(length) + int(offset))[:, None] - mx.arange(size)[None, :]
    gidx = mx.broadcast_to(mx.clip(dist, 0, extent - 1)[None, None], (b, h, length, size))
    pb = mx.take_along_axis(rel.transpose(0, 2, 1, 3), gidx, axis=-1)
    pb = mx.where((dist >= extent)[None, None], mx.array(0.0, dtype), pb)
    masked = dist < 0
    if window:
        masked = masked | (dist >= int(window))
    mask = mx.where(masked[None, None], mx.array(-1e30, dtype), pb)
    if left_padding is not None:
        valid = mx.arange(size)[None, :] >= left_padding[:, None]
        mask = mx.where(valid[:, None, None, :], mask, mx.array(-1e30, dtype))
    mask = mask.astype(dtype)
    zeros = mx.zeros((b, k.shape[1], 1, d), k.dtype)
    k = mx.concatenate([k, zeros], axis=2)
    v = mx.concatenate([v, zeros], axis=2)
    mask = mx.concatenate([mask, sinks.transpose(0, 2, 1)[..., None].astype(dtype)], axis=-1)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
