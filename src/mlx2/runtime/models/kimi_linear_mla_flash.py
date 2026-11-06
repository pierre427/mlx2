# SPDX-License-Identifier: Apache-2.0
"""Kimi MLA NAX flash kernels; unqualified, opt-in through ``auto`` only.

Adapted from the oMLX PR4808 NAX attention port in
``deepseek_v41_omlx_nax_attention.py`` (Apache-2.0; see
provenance/omlx-4808-attention.NOTICE and provenance/kimi-mla-flash.json):
its 16x32x16 ``matmul2d`` fragment layout, xor-1/xor-8 row reductions and
online-softmax structure. Geometry, masking, operand sources and the split
partial/combine law are mlx2's.

``expanded``: qk 192 (128 per-head NoPE + 64 shared key) / v 128 prompt
attention over the ``kv_b_proj`` re-expansion, read in its native
``[B, S, H, 128 + 128]`` layout (no transpose, concat or padding copy), with a
bottom-right causal offset for a chunk after a cached prefix.

``absorbed``: qk 576 (512 absorbed latent + 64 shared) / v 512, one latent KV
row per token shared by every head, read from the latent cache directly.

Both use BF16/FP16 operands, FP32 scores, softmax and accumulation. A row
with no visible key yields zeros (the latent law). Neither is exact parity
with the expanded ordinary path: accumulation order differs.
"""
from functools import cache

import mlx.core as mx

HEADS, NOPE, SHARED, VDIM, LATENT = 32, 128, 64, 128, 512
_LOG2E = 1.4426950408889634

_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
// Constant-trip fragment loops must unroll: a register array indexed by a
// loop variable otherwise spills to stack.
#define KUNROLL _Pragma("clang loop unroll(full)")
"""

# One threadgroup: 4 simdgroups x 16 query rows of one (batch, head). Every
# simdgroup runs its own key range; no threadgroup memory or barriers.
_EXPANDED = r"""
    constexpr int DN = 128;
    constexpr int DS = 64;
    constexpr int DQ = DN + DS;
    constexpr int DKV = DN + 128;
    constexpr int BK = 32;
    const int TQ = params[0];
    const int S = params[1];
    const int OFF = params[2];
    const int CAUSAL = params[3];
    const float scale2 = scalep[0];
    const int qb = int(threadgroup_position_in_grid.x);
    const int hh = int(threadgroup_position_in_grid.y);
    const int bb = int(threadgroup_position_in_grid.z);
    const int sg = int(simdgroup_index_in_threadgroup);
    const ushort lane = ushort(thread_index_in_simdgroup);
    const int row0 = qb * 64 + sg * 16;
    if (row0 >= TQ) {
        return;
    }
    const short qid = short(lane >> 2);
    const short fm = short((qid & 4) | ((lane >> 1) & 3));
    const short fn = short(((qid & 2) | (lane & 1)) * 4);
    const int ra = row0 + fm;
    const int rb = row0 + fm + 8;
    const device T* q0 = q + ((ulong(bb) * TQ + min(ra, TQ - 1)) * H + hh) * DQ + fn;
    const device T* q1 = q + ((ulong(bb) * TQ + min(rb, TQ - 1)) * H + hh) * DQ + fn;
    // HM: kv is head-major [B, H, S, 256]; else token-major [B, S, H, 256].
    constexpr ulong KVS = HM ? ulong(DKV) : ulong(H) * DKV;
    const device T* kvh = HM ? kv + (ulong(bb) * H + hh) * ulong(S) * DKV
                             : kv + ulong(bb) * ulong(S) * KVS + ulong(hh) * DKV;
    const device T* ksb = ks + ulong(bb) * ulong(params[4]) * DS;
    const int last = min(row0 + 15, TQ - 1);
    const int klim = CAUSAL ? min(S, OFF + last + 1) : S;
    // Keys after OFF + row0 are hidden from at least one row of the group.
    const int kfree = CAUSAL ? min(S, OFF + row0 + 1) : S;
    const int nkb = (klim + BK - 1) / BK;

    constexpr auto qk_desc = matmul2d_descriptor(
        16, 32, 16, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroup> qk_op;
    constexpr auto pv_desc = matmul2d_descriptor(
        16, 32, 16, false, false, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroup> pv_op;

    auto pa = pv_op.template get_left_input_cooperative_tensor<float, T, float>();
    auto pb = pv_op.template get_right_input_cooperative_tensor<float, T, float>();
    auto o0 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o1 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o2 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o3 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    KUNROLL
    for (short e = 0; e < 16; ++e) {
        o0[e] = 0.0f;
        o1[e] = 0.0f;
        o2[e] = 0.0f;
        o3[e] = 0.0f;
    }
    float m_run[2] = {-INFINITY, -INFINITY};
    float l_run[2] = {0.0f, 0.0f};

    for (int kb = 0; kb < nkb; ++kb) {
        const int k0 = kb * BK;
        auto qa = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
        auto kt = qk_op.template get_right_input_cooperative_tensor<T, T, float>();
        auto sc = qk_op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(qa)>, metal::remove_addrspace_t<decltype(kt)>, float>();
        KUNROLL
        for (short e = 0; e < 16; ++e) {
            sc[e] = 0.0f;
        }
        const device T* kr[2][2];
        const device T* sr[2][2];
        KUNROLL
        for (short tn = 0; tn < 2; ++tn) {
            KUNROLL
            for (short i = 0; i < 2; ++i) {
                const int key = min(k0 + tn * 16 + fm + i * 8, klim - 1);
                kr[tn][i] = kvh + ulong(key) * KVS + fn;
                sr[tn][i] = ksb + ulong(key) * DS + fn;
            }
        }
        _Pragma("clang loop unroll_count(4)")
        for (short kk = 0; kk < DN; kk += 16) {
            KUNROLL
            for (short j = 0; j < 4; ++j) {
                qa[j] = q0[kk + j];
                qa[4 + j] = q1[kk + j];
            }
            KUNROLL
            for (short tn = 0; tn < 2; ++tn) {
                KUNROLL
                for (short i = 0; i < 2; ++i) {
                    KUNROLL
                    for (short j = 0; j < 4; ++j) {
                        kt[tn * 8 + i * 4 + j] = kr[tn][i][kk + j];
                    }
                }
            }
            qk_op.run(qa, kt, sc);
        }
        KUNROLL
        for (short kk = 0; kk < DS; kk += 16) {
            KUNROLL
            for (short j = 0; j < 4; ++j) {
                qa[j] = q0[DN + kk + j];
                qa[4 + j] = q1[DN + kk + j];
            }
            KUNROLL
            for (short tn = 0; tn < 2; ++tn) {
                KUNROLL
                for (short i = 0; i < 2; ++i) {
                    KUNROLL
                    for (short j = 0; j < 4; ++j) {
                        kt[tn * 8 + i * 4 + j] = sr[tn][i][kk + j];
                    }
                }
            }
            qk_op.run(qa, kt, sc);
        }
        const bool edge = k0 + BK > kfree;
        float s[16];
        float mx_i[2] = {-INFINITY, -INFINITY};
        KUNROLL
        for (short tn = 0; tn < 2; ++tn) {
            KUNROLL
            for (short i = 0; i < 2; ++i) {
                const int r = i ? rb : ra;
                KUNROLL
                for (short j = 0; j < 4; ++j) {
                    const short e = tn * 8 + i * 4 + j;
                    const int key = k0 + tn * 16 + fn + j;
                    const bool ok = !edge || (key < S && (!CAUSAL || key <= OFF + r));
                    s[e] = ok ? float(sc[e]) * scale2 : -INFINITY;
                    mx_i[i] = max(mx_i[i], s[e]);
                }
            }
        }
        float factor[2];
        float mnew[2];
        KUNROLL
        for (short i = 0; i < 2; ++i) {
            float m = mx_i[i];
            m = max(m, simd_shuffle_xor(m, ushort(1)));
            m = max(m, simd_shuffle_xor(m, ushort(8)));
            mnew[i] = max(m_run[i], m);
            factor[i] = m_run[i] == -INFINITY ? 0.0f : fast::exp2(m_run[i] - mnew[i]);
            m_run[i] = mnew[i];
        }
        float rs[2] = {0.0f, 0.0f};
        KUNROLL
        for (short e = 0; e < 16; ++e) {
            const short i = (e >> 2) & 1;
            const float p = s[e] == -INFINITY ? 0.0f : fast::exp2(s[e] - mnew[i]);
            s[e] = p;
            rs[i] += p;
        }
        KUNROLL
        for (short i = 0; i < 2; ++i) {
            l_run[i] = l_run[i] * factor[i] + rs[i];
        }
        if (!simd_all(factor[0] == 1.0f && factor[1] == 1.0f)) {
            KUNROLL
            for (short e = 0; e < 16; ++e) {
                const float f = factor[(e >> 2) & 1];
                o0[e] *= f;
                o1[e] *= f;
                o2[e] *= f;
                o3[e] *= f;
            }
        }
        KUNROLL
        for (short ks = 0; ks < 2; ++ks) {
            KUNROLL
            for (short e = 0; e < 8; ++e) {
                pa[e] = s[ks * 8 + e];
            }
            // Rows past this group's causal limit load the last visible
            // key instead: P is 0 there, and 0 * (finite) keeps unseen cache
            // contents (even NaN) out of the output.
            const int va = min(k0 + ks * 16 + fm, klim - 1);
            const int vb = min(k0 + ks * 16 + fm + 8, klim - 1);
            const device T* v0 = kvh + ulong(va) * KVS + DN + fn;
            const device T* v1 = kvh + ulong(vb) * KVS + DN + fn;
            KUNROLL
            for (short np = 0; np < 4; ++np) {
                KUNROLL
                for (short tn = 0; tn < 2; ++tn) {
                    KUNROLL
                    for (short j = 0; j < 4; ++j) {
                        pb[tn * 8 + j] = v0[np * 32 + tn * 16 + j];
                        pb[tn * 8 + 4 + j] = v1[np * 32 + tn * 16 + j];
                    }
                }
                if (np == 0) {
                    pv_op.run(pa, pb, o0);
                } else if (np == 1) {
                    pv_op.run(pa, pb, o1);
                } else if (np == 2) {
                    pv_op.run(pa, pb, o2);
                } else {
                    pv_op.run(pa, pb, o3);
                }
            }
        }
    }

    KUNROLL
    for (short i = 0; i < 2; ++i) {
        float l = l_run[i];
        l += simd_shuffle_xor(l, ushort(1));
        l += simd_shuffle_xor(l, ushort(8));
        const int r = i ? rb : ra;
        if (r >= TQ) {
            continue;
        }
        // No visible key: zeros, as in the latent law.
        const float inv = l == 0.0f ? 0.0f : 1.0f / l;
        device OT* orow = out + ((ulong(bb) * TQ + r) * H + hh) * 128 + fn;
        KUNROLL
        for (short tn = 0; tn < 2; ++tn) {
            KUNROLL
            for (short j = 0; j < 4; ++j) {
                const short e = tn * 8 + i * 4 + j;
                orow[tn * 16 + j] = OT(o0[e] * inv);
                orow[32 + tn * 16 + j] = OT(o1[e] * inv);
                orow[64 + tn * 16 + j] = OT(o2[e] * inv);
                orow[96 + tn * 16 + j] = OT(o3[e] * inv);
            }
        }
        // Natural log-sum-exp of the visible scores (-inf with none), for
        // merging key windows.
        if (fn == 0) {
            lse[(ulong(bb) * TQ + r) * H + hh] =
                l == 0.0f ? -INFINITY : (m_run[i] + log2(l)) * M_LN2_F;
        }
    }
"""


@cache
def _expanded_kernel():
    return mx.fast.metal_kernel(name="mlx2_kimi_mla_flash192_v1",
        input_names=["q", "kv", "ks", "params", "scalep"], output_names=["out", "lse"],
        header=_HEADER, source=_EXPANDED, ensure_row_contiguous=True)


# Law v2 of flash192 (``expanded``): the same grid, masks, online softmax,
# FP32 P and output/lse contract as v1 (_EXPANDED), with the kernel tuning of
# the QSA NAX law v2 (qwen4_qsa_nax, after jundot/omlx #5084 at
# b0541976f5e5031c8ecd4c728f5a5810b6af4fdf, Apache-2.0; see
# provenance/kimi-mla-flash-law-v2.json):
#   * Q, K and V rows load 8 contiguous channels at a time (``fcol = 2*fn``):
#     Q and K share one channel permutation inside each 32-channel slice, so
#     scores are unchanged; V's permutation is undone at the output store;
#   * QK and PV run as 16x32x32 ops (K-width 32): 6 QK ops per 32-key block
#     instead of 12, one PV op per 32 output channels instead of two.
# Bases that are not 16-byte aligned take an in-kernel scalar path with the
# same arithmetic.
_HEADER_V2 = _HEADER + r"""
template <typename T>
inline metal::vec<T, 8> kimi_load8(const device T* p, bool vec) {
    if (vec) {
        return *(const device metal::vec<T, 8>*)p;
    }
    metal::vec<T, 8> r;
    for (short e = 0; e < 8; e++) r[e] = p[e];
    return r;
}
"""

_EXPANDED_V2 = r"""
    constexpr int DN = 128;
    constexpr int DS = 64;
    constexpr int DQ = DN + DS;
    constexpr int DKV = DN + 128;
    constexpr int BK = 32;
    const int TQ = params[0];
    const int S = params[1];
    const int OFF = params[2];
    const int CAUSAL = params[3];
    const float scale2 = scalep[0];
    const int qb = int(threadgroup_position_in_grid.x);
    const int hh = int(threadgroup_position_in_grid.y);
    const int bb = int(threadgroup_position_in_grid.z);
    const int sg = int(simdgroup_index_in_threadgroup);
    const ushort lane = ushort(thread_index_in_simdgroup);
    const int row0 = qb * 64 + sg * 16;
    if (row0 >= TQ) {
        return;
    }
    const short qid = short(lane >> 2);
    const short fm = short((qid & 4) | ((lane >> 1) & 3));
    const short fn = short(((qid & 2) | (lane & 1)) * 4);
    const short fcol = short(2 * fn);   // 8 contiguous channels per fragment pair
    const int ra = row0 + fm;
    const int rb = row0 + fm + 8;
    const bool qv = ((reinterpret_cast<ulong>(q) & 15ul) == 0);
    const bool kvv = ((reinterpret_cast<ulong>(kv) & 15ul) == 0);
    const bool ksv = ((reinterpret_cast<ulong>(ks) & 15ul) == 0);
    const device T* q0 = q + ((ulong(bb) * TQ + min(ra, TQ - 1)) * H + hh) * DQ + fcol;
    const device T* q1 = q + ((ulong(bb) * TQ + min(rb, TQ - 1)) * H + hh) * DQ + fcol;
    constexpr ulong KVS = HM ? ulong(DKV) : ulong(H) * DKV;
    const device T* kvh = HM ? kv + (ulong(bb) * H + hh) * ulong(S) * DKV
                             : kv + ulong(bb) * ulong(S) * KVS + ulong(hh) * DKV;
    const device T* ksb = ks + ulong(bb) * ulong(params[4]) * DS;
    const int last = min(row0 + 15, TQ - 1);
    const int klim = CAUSAL ? min(S, OFF + last + 1) : S;
    const int kfree = CAUSAL ? min(S, OFF + row0 + 1) : S;
    const int nkb = (klim + BK - 1) / BK;

    constexpr auto qk_desc = matmul2d_descriptor(
        16, 32, 32, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroup> qk_op;
    constexpr auto pv_desc = matmul2d_descriptor(
        16, 32, 32, false, false, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroup> pv_op;

    auto pa = pv_op.template get_left_input_cooperative_tensor<float, T, float>();
    auto pb = pv_op.template get_right_input_cooperative_tensor<float, T, float>();
    auto o0 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o1 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o2 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o3 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    KUNROLL
    for (short e = 0; e < 16; ++e) {
        o0[e] = 0.0f;
        o1[e] = 0.0f;
        o2[e] = 0.0f;
        o3[e] = 0.0f;
    }
    float m_run[2] = {-INFINITY, -INFINITY};
    float l_run[2] = {0.0f, 0.0f};

    for (int kb = 0; kb < nkb; ++kb) {
        const int k0 = kb * BK;
        auto qa = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
        auto kt = qk_op.template get_right_input_cooperative_tensor<T, T, float>();
        auto sc = qk_op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(qa)>, metal::remove_addrspace_t<decltype(kt)>, float>();
        KUNROLL
        for (short e = 0; e < 16; ++e) {
            sc[e] = 0.0f;
        }
        // Key rows fm, fm+8, fm+16, fm+24 of this block (rr = tn*2 + i).
        const device T* kr[4];
        const device T* sr[4];
        KUNROLL
        for (short rr = 0; rr < 4; ++rr) {
            const int key = min(k0 + fm + rr * 8, klim - 1);
            kr[rr] = kvh + ulong(key) * KVS + fcol;
            sr[rr] = ksb + ulong(key) * DS + fcol;
        }
        KUNROLL
        for (short kk = 0; kk < DQ; kk += 32) {
            const metal::vec<T, 8> a0 = kimi_load8(q0 + kk, qv);
            const metal::vec<T, 8> a1 = kimi_load8(q1 + kk, qv);
            metal::vec<T, 8> kv8[4];
            KUNROLL
            for (short rr = 0; rr < 4; ++rr) {
                kv8[rr] = kk < DN ? kimi_load8(kr[rr] + kk, kvv) : kimi_load8(sr[rr] + (kk - DN), ksv);
            }
            KUNROLL
            for (short j = 0; j < 4; ++j) {
                qa[j] = a0[j];
                qa[4 + j] = a1[j];
                qa[8 + j] = a0[4 + j];
                qa[12 + j] = a1[4 + j];
                kt[j] = kv8[0][j];
                kt[4 + j] = kv8[1][j];
                kt[8 + j] = kv8[2][j];
                kt[12 + j] = kv8[3][j];
                kt[16 + j] = kv8[0][4 + j];
                kt[20 + j] = kv8[1][4 + j];
                kt[24 + j] = kv8[2][4 + j];
                kt[28 + j] = kv8[3][4 + j];
            }
            qk_op.run(qa, kt, sc);
        }
        const bool edge = k0 + BK > kfree;
        float s[16];
        float mx_i[2] = {-INFINITY, -INFINITY};
        KUNROLL
        for (short tn = 0; tn < 2; ++tn) {
            KUNROLL
            for (short i = 0; i < 2; ++i) {
                const int r = i ? rb : ra;
                KUNROLL
                for (short j = 0; j < 4; ++j) {
                    const short e = tn * 8 + i * 4 + j;
                    const int key = k0 + tn * 16 + fn + j;
                    const bool ok = !edge || (key < S && (!CAUSAL || key <= OFF + r));
                    s[e] = ok ? float(sc[e]) * scale2 : -INFINITY;
                    mx_i[i] = max(mx_i[i], s[e]);
                }
            }
        }
        float factor[2];
        float mnew[2];
        KUNROLL
        for (short i = 0; i < 2; ++i) {
            float m = mx_i[i];
            m = max(m, simd_shuffle_xor(m, ushort(1)));
            m = max(m, simd_shuffle_xor(m, ushort(8)));
            mnew[i] = max(m_run[i], m);
            factor[i] = m_run[i] == -INFINITY ? 0.0f : fast::exp2(m_run[i] - mnew[i]);
            m_run[i] = mnew[i];
        }
        float rs[2] = {0.0f, 0.0f};
        KUNROLL
        for (short e = 0; e < 16; ++e) {
            const short i = (e >> 2) & 1;
            const float p = s[e] == -INFINITY ? 0.0f : fast::exp2(s[e] - mnew[i]);
            s[e] = p;
            rs[i] += p;
        }
        KUNROLL
        for (short i = 0; i < 2; ++i) {
            l_run[i] = l_run[i] * factor[i] + rs[i];
        }
        if (!simd_all(factor[0] == 1.0f && factor[1] == 1.0f)) {
            KUNROLL
            for (short e = 0; e < 16; ++e) {
                const float f = factor[(e >> 2) & 1];
                o0[e] *= f;
                o1[e] *= f;
                o2[e] *= f;
                o3[e] *= f;
            }
        }
        // One 16x32x32 PV op per 32 output channels: P covers the block's 32
        // keys in the score layout; rows past this group's causal limit load
        // the last visible key (P is 0 there).
        KUNROLL
        for (short e = 0; e < 16; ++e) {
            pa[e] = s[e];
        }
        const device T* vr[4];
        KUNROLL
        for (short rr = 0; rr < 4; ++rr) {
            const int key = min(k0 + fm + rr * 8, klim - 1);
            vr[rr] = kvh + ulong(key) * KVS + DN + fcol;
        }
        KUNROLL
        for (short np = 0; np < 4; ++np) {
            metal::vec<T, 8> v8[4];
            KUNROLL
            for (short rr = 0; rr < 4; ++rr) {
                v8[rr] = kimi_load8(vr[rr] + np * 32, kvv);
            }
            KUNROLL
            for (short j = 0; j < 4; ++j) {
                pb[j] = v8[0][j];
                pb[4 + j] = v8[1][j];
                pb[8 + j] = v8[0][4 + j];
                pb[12 + j] = v8[1][4 + j];
                pb[16 + j] = v8[2][j];
                pb[20 + j] = v8[3][j];
                pb[24 + j] = v8[2][4 + j];
                pb[28 + j] = v8[3][4 + j];
            }
            if (np == 0) {
                pv_op.run(pa, pb, o0);
            } else if (np == 1) {
                pv_op.run(pa, pb, o1);
            } else if (np == 2) {
                pv_op.run(pa, pb, o2);
            } else {
                pv_op.run(pa, pb, o3);
            }
        }
    }

    KUNROLL
    for (short i = 0; i < 2; ++i) {
        float l = l_run[i];
        l += simd_shuffle_xor(l, ushort(1));
        l += simd_shuffle_xor(l, ushort(8));
        const int r = i ? rb : ra;
        if (r >= TQ) {
            continue;
        }
        const float inv = l == 0.0f ? 0.0f : 1.0f / l;
        // Output channel of fragment element (f, j) is fcol + f*4 + j.
        device OT* orow = out + ((ulong(bb) * TQ + r) * H + hh) * 128 + fcol;
        KUNROLL
        for (short f = 0; f < 2; ++f) {
            KUNROLL
            for (short j = 0; j < 4; ++j) {
                const short e = f * 8 + i * 4 + j;
                orow[f * 4 + j] = OT(o0[e] * inv);
                orow[32 + f * 4 + j] = OT(o1[e] * inv);
                orow[64 + f * 4 + j] = OT(o2[e] * inv);
                orow[96 + f * 4 + j] = OT(o3[e] * inv);
            }
        }
        if (fn == 0) {
            lse[(ulong(bb) * TQ + r) * H + hh] =
                l == 0.0f ? -INFINITY : (m_run[i] + log2(l)) * M_LN2_F;
        }
    }
"""


@cache
def _expanded_kernel_v2():
    return mx.fast.metal_kernel(name="mlx2_kimi_mla_flash192_v2",
        input_names=["q", "kv", "ks", "params", "scalep"], output_names=["out", "lse"],
        header=_HEADER_V2, source=_EXPANDED_V2, ensure_row_contiguous=True)


# flash192 kernel laws.  "v1" is the default (the only law with end-to-end
# evidence); "v2" is opt-in until it passes the model test flow.
EXPANDED_LAWS = {"v1": "kimi-flash192-v1", "v2": "kimi-flash192-v2"}
_EXPANDED_KERNELS = {"v1": _expanded_kernel, "v2": _expanded_kernel_v2}


def _offset(mask, rows, width, override=None):
    """Bottom-right causal offset for "causal"; None means all visible.

    ``override`` is the offset of a key window inside a longer prefix (rows
    may then see none of its keys)."""
    if mask is None:
        return 0, False
    if isinstance(mask, str) and mask == "causal" and (override is not None or width >= rows):
        return (width - rows if override is None else int(override)), True
    raise ValueError("Kimi MLA flash admits only mask None or bottom-right 'causal'")


def expanded_geometry(q, kv, shared, mask, heads=HEADS, offset=None):
    """Raise unless the expanded kernel's static contract holds.

    kv is token-major ``[B, S, H*256]`` (the kv_b_proj output) or head-major
    ``[B, H, S, 256]``; each head row is 128 NoPE key then 128 value channels.
    """
    if q.dtype not in (mx.bfloat16, mx.float16) or kv.dtype != q.dtype or shared.dtype != q.dtype:
        raise ValueError("Kimi MLA flash192 needs matching BF16/FP16 operands")
    if q.ndim != 3 or kv.ndim not in (3, 4) or shared.ndim != 4:
        raise ValueError("Kimi MLA flash192 needs q[B,T,H*192], kv[B,S,H*256]|[B,H,S,256], shared[B,1,S,64]")
    b, t = q.shape[0], q.shape[1]
    width = kv.shape[1] if kv.ndim == 3 else kv.shape[2]
    layout = (b, width, heads*(NOPE+VDIM)) if kv.ndim == 3 else (b, heads, width, NOPE+VDIM)
    if (q.shape[2] != heads*(NOPE+SHARED) or kv.shape != layout or shared.shape[:2] != (b, 1)
            or shared.shape[3] != SHARED or shared.shape[2] < width or t < 1 or width < 1):
        raise ValueError("Kimi MLA flash192 geometry")
    return (b, t, width, *_offset(mask, t, width, offset))


def expanded(q, kv, shared, mask, *, scale, heads=HEADS, offset=None, return_lse=False,
             out_dtype=None, law="v1"):
    """O[B, T, H*128] for q[B,T,H*192] over kv (token- or head-major) and
    shared[B,1,S,64] -- or the shared cache buffer [B,1,capacity,64] whose
    first S tokens are the prefix (read in place, no B > 1 slice copy).

    ``offset`` is the causal offset of this key window (a prefix block) or a
    hidden-row validation; ``return_lse`` also returns the natural
    log-sum-exp [B, T, H] for merging key windows; ``out_dtype`` FP32 keeps
    a window's output unrounded.
    """
    b, t, width, offset, causal = expanded_geometry(q, kv, shared, mask, heads, offset)
    out_dtype = q.dtype if out_dtype is None else out_dtype
    if law not in _EXPANDED_KERNELS:
        raise ValueError(f"Kimi flash192 law must be one of {sorted(_EXPANDED_KERNELS)}")
    out, lse = _EXPANDED_KERNELS[law]()(
        inputs=[q, kv, shared.reshape(b, shared.shape[2], SHARED),
                mx.array([t, width, offset, int(causal), shared.shape[2]], mx.int32),
                mx.array([scale*_LOG2E], mx.float32)],
        template=[("T", q.dtype), ("H", heads), ("HM", int(kv.ndim == 4)), ("OT", out_dtype)],
        grid=(((t+63)//64)*128, heads, b), threadgroup=(128, 1, 1),
        output_shapes=[(b, t, heads*VDIM), (b, t, heads)], output_dtypes=[out_dtype, mx.float32])
    return (out, lse) if return_lse else out


def expanded_windows(q, latent, shared, kv_proj, mask, *, scale, window, heads=HEADS, law="v1"):
    """Flash192 over the prefix in key windows of ``window`` tokens: each is
    re-expanded alone (transient bounded by one window) and the windows are
    merged by log-sum-exp in FP32. latent[B,1,S,512], shared[B,1,S,64]."""
    width, rows = latent.shape[2], q.shape[1]
    _, causal = _offset(mask, rows, width)
    natural = width-rows if causal else 0
    total, parts = None, []
    for k0 in range(0, width, window):
        k1 = min(width, k0+window)
        if causal and k0 > natural+rows-1:
            break
        kv = kv_proj(latent[:, 0, k0:k1])
        out, lse = expanded(q, kv, shared[:, :, k0:k1], mask, scale=scale, heads=heads,
                            offset=natural-k0 if causal else None,
                            return_lse=True, out_dtype=mx.float32, law=law)
        parts.append((out, lse))
    best = parts[0][1]
    for _, lse in parts[1:]:
        best = mx.maximum(best, lse)
    best = mx.where(mx.isfinite(best), best, 0)
    acc = None
    for out, lse in parts:
        w = mx.exp(lse-best)
        term = out.reshape(*out.shape[:2], heads, VDIM)*w[..., None]
        acc, total = (term, w) if acc is None else (acc+term, total+w)
    acc = acc/mx.maximum(total, 1e-30)[..., None]
    return acc.reshape(*acc.shape[:2], heads*VDIM).astype(q.dtype)


def expanded_reference(q, kv, shared, mask, *, scale, heads=HEADS, offset=None,
                       return_lse=False, out_dtype=None, law=None):
    """FP32 ops law of ``expanded`` (validation only); hidden rows are zeros.
    ``law`` is accepted and ignored: every kernel law targets this one."""
    b, t, width, offset, causal = expanded_geometry(q, kv, shared, mask, heads, offset)
    qf = q.astype(mx.float32).reshape(b, t, heads, NOPE+SHARED).transpose(0, 2, 1, 3)
    kvf = kv.astype(mx.float32)
    if kv.ndim == 3:
        kvf = kvf.reshape(b, width, heads, NOPE+VDIM).transpose(0, 2, 1, 3)
    keys = mx.concatenate([kvf[..., :NOPE], mx.broadcast_to(shared[:, :, :width].astype(mx.float32),
                                                          (b, heads, width, SHARED))], -1)
    score = (qf @ keys.swapaxes(-1, -2)) * scale
    if causal:
        visible = mx.arange(offset, offset+t)[:, None] >= mx.arange(width)[None, :]
        score = mx.where(visible, score, -float("inf"))
    out = (_safe_softmax(score) @ kvf[..., NOPE:]).transpose(0, 2, 1, 3).reshape(b, t, heads*VDIM)
    if out_dtype is not None:
        out = out.astype(out_dtype)
    if not return_lse:
        return out
    top = score.max(-1, keepdims=True)
    safe = mx.where(mx.isfinite(top), top, 0)
    lse = mx.log(mx.exp(score-safe).sum(-1, keepdims=True)) + safe
    return out, lse[..., 0].transpose(0, 2, 1)


def _safe_softmax(score):
    top = score.max(-1, keepdims=True)
    prob = mx.exp(score - mx.where(mx.isfinite(top), top, 0))
    return prob / mx.maximum(prob.sum(-1, keepdims=True), 1e-30)


_AVAILABLE = {}


def _device_is_m5():
    try:
        return mx.metal.is_available() and "M5" in str(mx.device_info().get("device_name", ""))
    except (AttributeError, RuntimeError, TypeError):
        return False


def available(kind, dtype=mx.bfloat16):
    """Memoized: GPU is M5 (NAX validated) and a bounded numerical probe of
    the kernel matches its FP32 ops law. Any refusal or mismatch disables
    the kernel for the process; the caller falls back."""
    key = (kind, dtype)
    if key in _AVAILABLE:
        return _AVAILABLE[key]
    ok = False
    if mx.default_device() == mx.gpu and _device_is_m5():
        try:
            ok = _probe(kind, dtype)
        except Exception as exc:  # compile failure / refusal: decline cleanly
            from .served_exp import is_device_fault
            if is_device_fault(exc):
                raise
            ok = False
    _AVAILABLE[key] = ok
    return ok


def _rel(ref, got):
    ref, got = ref.astype(mx.float32), got.astype(mx.float32)
    num = mx.sqrt(mx.sum((got-ref)**2))
    den = mx.maximum(mx.sqrt(mx.sum(ref**2)), 1e-20)
    return float(num/den), bool(mx.all(mx.isfinite(got)))


def _probe(kind, dtype):
    key = mx.random.key(20261102)
    k1, k2, k3, k4 = mx.random.split(key, 4)
    rows, width = 40, 100
    if kind in ("expanded", "expanded_v2"):
        law = "v2" if kind == "expanded_v2" else "v1"
        q = (mx.random.normal((1, rows, HEADS*(NOPE+SHARED)), key=k1)*0.5).astype(dtype)
        kv = mx.random.normal((1, width, HEADS*(NOPE+VDIM)), key=k2).astype(dtype)
        sh = mx.random.normal((1, 1, width, SHARED), key=k3).astype(dtype)
        kvh = mx.contiguous(kv.reshape(1, width, HEADS, NOPE+VDIM).transpose(0, 2, 1, 3))
        got = mx.concatenate([expanded(q, kv, sh, "causal", scale=(NOPE+SHARED)**-0.5, law=law),
                              expanded(q, kvh, sh, "causal", scale=(NOPE+SHARED)**-0.5, law=law)])
        ref = expanded_reference(q, kv, sh, "causal", scale=(NOPE+SHARED)**-0.5)
        ref = mx.concatenate([ref, ref])
    elif kind == "absorbed":
        q = (mx.random.normal((1, HEADS, rows, LATENT), key=k1)*0.05).astype(dtype)
        qs = (mx.random.normal((1, HEADS, rows, SHARED), key=k4)*0.5).astype(dtype)
        lat = mx.random.normal((1, 1, width, LATENT), key=k2).astype(dtype)
        sh = mx.random.normal((1, 1, width, SHARED), key=k3).astype(dtype)
        got = absorbed(q, qs, lat, sh, "causal", scale=(NOPE+SHARED)**-0.5)
        ref = absorbed_reference(q, qs, lat, sh, "causal", scale=(NOPE+SHARED)**-0.5)
    elif kind == "absorbed_padded":
        # Left-padded batch lanes over two key splits, in-place capacity
        # buffers, a start inside a tile and a lane whose early rows see no
        # key. Its own result: a failure leaves the unpadded route enabled.
        rows, width, cap = 8, 2300, 2560
        q = (mx.random.normal((3, HEADS, rows, LATENT), key=k1)*0.05).astype(dtype)
        qs = (mx.random.normal((3, HEADS, rows, SHARED), key=k4)*0.5).astype(dtype)
        lat = mx.random.normal((3, 1, cap, LATENT), key=k2).astype(dtype)
        sh = mx.random.normal((3, 1, cap, SHARED), key=k3).astype(dtype)
        starts = mx.array([0, 2049, width-3], mx.int32)
        got = absorbed(q, qs, lat, sh, "causal", scale=(NOPE+SHARED)**-0.5, starts=starts,
                       split=2048, width=width)
        ref = absorbed_reference(q, qs, lat, sh, "causal", scale=(NOPE+SHARED)**-0.5,
                                 starts=starts, width=width)
    else:
        raise ValueError(kind)
    rel, finite = _rel(ref, got)
    return finite and rel < 2e-2


# ---------------------------------------------------------------------------
# Absorbed latent flash: qk 576 / v 512, one latent row per token for all heads.
#
# One threadgroup = one (batch, query row, key split): 8 simdgroups = 2 head
# groups x 4 quarters. QK: quarter j4 scores 32 of the tile's 128 keys for 16
# heads; PV: quarter j4 accumulates 128 of the 512 latent output channels over
# the whole tile (P exchanged through threadgroup memory), as in the source.
# ---------------------------------------------------------------------------
_ABSORBED = r"""
    constexpr int D = 512;
    constexpr int DS = 64;
    constexpr int BK = 128;
    const int TQ = params[0];
    const int S = params[1];
    const int OFF = params[2];
    const int CAUSAL = params[3];
    const int SPLIT = params[4];
    const int NSPLIT = params[5];
    const float scale2 = scalep[0];
    const int qi = int(threadgroup_position_in_grid.x);
    const int sp = int(threadgroup_position_in_grid.y);
    const int bb = int(threadgroup_position_in_grid.z);
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int hg = int(sg) / 4;
    const int j4 = int(sg) % 4;
    const int head0 = hg * 16;
    const short qid = short(lane >> 2);
    const short fm = short((qid & 4) | ((lane >> 1) & 3));
    const short fn = short(((qid & 2) | (lane & 1)) * 4);
    const bool row_writer = (lane & 9u) == 0u;

    threadgroup float red_max[2][4][16];
    threadgroup float red_sum[2][4][16];
    threadgroup float p_tile[2][BK / 16][32 * 8];

    const int klim = CAUSAL ? min(S, OFF + qi + 1) : S;
    // Keys before starts[bb] are this lane's left padding (a padded batch
    // cache): the visible keys stay one contiguous range [start, klim).
    const int kstart = min(max(starts[bb], 0), klim);
    const int kbeg = min(max(sp * SPLIT, kstart), klim);
    const int kend = max(kbeg, min(sp * SPLIT + SPLIT, klim));

    constexpr auto qk_desc = matmul2d_descriptor(
        16, 32, 16, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroup> qk_op;
    constexpr auto pv_desc = matmul2d_descriptor(
        16, 32, 16, false, false, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroup> pv_op;

    auto pa = pv_op.template get_left_input_cooperative_tensor<float, T, float>();
    auto pb = pv_op.template get_right_input_cooperative_tensor<float, T, float>();
    auto o0 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o1 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o2 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o3 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    KUNROLL
    for (short e = 0; e < 16; ++e) {
        o0[e] = 0.0f;
        o1[e] = 0.0f;
        o2[e] = 0.0f;
        o3[e] = 0.0f;
    }
    float m_run[2] = {-INFINITY, -INFINITY};
    float l_run[2] = {0.0f, 0.0f};

    // q[B, H, TQ, 512], qs[B, H, TQ, 64]; latent[B, S, 512], shared[B, S, 64].
    const device T* qr0 = q + ((ulong(bb) * H + head0 + fm) * TQ + qi) * D + fn;
    const device T* qr1 = q + ((ulong(bb) * H + head0 + fm + 8) * TQ + qi) * D + fn;
    const device T* qs0 = qs + ((ulong(bb) * H + head0 + fm) * TQ + qi) * DS + fn;
    const device T* qs1 = qs + ((ulong(bb) * H + head0 + fm + 8) * TQ + qi) * DS + fn;
    const device T* lat = latent + ulong(bb) * ulong(params[6]) * D;
    const device T* shb = shared + ulong(bb) * ulong(params[6]) * DS;

    for (int t0 = kbeg; t0 < kend; t0 += BK) {
        const int tile = min(BK, kend - t0);
        auto qa = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
        auto kb = qk_op.template get_right_input_cooperative_tensor<T, T, float>();
        auto sc = qk_op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(qa)>, metal::remove_addrspace_t<decltype(kb)>, float>();
        KUNROLL
        for (short e = 0; e < 16; ++e) {
            sc[e] = 0.0f;
        }
        if (j4 * 32 < tile) {
            const device T* kr[2][2];
            const device T* sr[2][2];
            KUNROLL
            for (short tn = 0; tn < 2; ++tn) {
                KUNROLL
                for (short i = 0; i < 2; ++i) {
                    const int key = t0 + min(j4 * 32 + tn * 16 + fm + i * 8, tile - 1);
                    kr[tn][i] = lat + ulong(key) * D + fn;
                    sr[tn][i] = shb + ulong(key) * DS + fn;
                }
            }
            _Pragma("clang loop unroll_count(4)")
            for (short kk = 0; kk < D; kk += 16) {
                KUNROLL
                for (short j = 0; j < 4; ++j) {
                    qa[j] = qr0[kk + j];
                    qa[4 + j] = qr1[kk + j];
                }
                KUNROLL
                for (short tn = 0; tn < 2; ++tn) {
                    KUNROLL
                    for (short i = 0; i < 2; ++i) {
                        KUNROLL
                        for (short j = 0; j < 4; ++j) {
                            kb[tn * 8 + i * 4 + j] = kr[tn][i][kk + j];
                        }
                    }
                }
                qk_op.run(qa, kb, sc);
            }
            KUNROLL
            for (short kk = 0; kk < DS; kk += 16) {
                KUNROLL
                for (short j = 0; j < 4; ++j) {
                    qa[j] = qs0[kk + j];
                    qa[4 + j] = qs1[kk + j];
                }
                KUNROLL
                for (short tn = 0; tn < 2; ++tn) {
                    KUNROLL
                    for (short i = 0; i < 2; ++i) {
                        KUNROLL
                        for (short j = 0; j < 4; ++j) {
                            kb[tn * 8 + i * 4 + j] = sr[tn][i][kk + j];
                        }
                    }
                }
                qk_op.run(qa, kb, sc);
            }
        }
        // Every key of [kbeg, kend) is visible to this row; only the tile
        // tail is out of range.
        float s[16];
        float pmx[2] = {-INFINITY, -INFINITY};
        KUNROLL
        for (short tn = 0; tn < 2; ++tn) {
            KUNROLL
            for (short i = 0; i < 2; ++i) {
                KUNROLL
                for (short j = 0; j < 4; ++j) {
                    const short e = tn * 8 + i * 4 + j;
                    const bool ok = j4 * 32 + tn * 16 + fn + j < tile;
                    s[e] = ok ? float(sc[e]) * scale2 : -INFINITY;
                    pmx[i] = max(pmx[i], s[e]);
                }
            }
        }
        KUNROLL
        for (short i = 0; i < 2; ++i) {
            pmx[i] = max(pmx[i], simd_shuffle_xor(pmx[i], ushort(1)));
            pmx[i] = max(pmx[i], simd_shuffle_xor(pmx[i], ushort(8)));
        }
        if (row_writer) {
            red_max[hg][j4][fm] = pmx[0];
            red_max[hg][j4][fm + 8] = pmx[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float factor[2];
        float mnew[2];
        KUNROLL
        for (short i = 0; i < 2; ++i) {
            const int r = fm + i * 8;
            const float m = max(max(red_max[hg][0][r], red_max[hg][1][r]),
                                max(red_max[hg][2][r], red_max[hg][3][r]));
            mnew[i] = max(m_run[i], m);
            factor[i] = m_run[i] == -INFINITY ? 0.0f : fast::exp2(m_run[i] - mnew[i]);
            m_run[i] = mnew[i];
        }
        float rs[2] = {0.0f, 0.0f};
        KUNROLL
        for (short tn = 0; tn < 2; ++tn) {
            vec<float, 8> pv;
            KUNROLL
            for (short i = 0; i < 2; ++i) {
                KUNROLL
                for (short j = 0; j < 4; ++j) {
                    const short e = tn * 8 + i * 4 + j;
                    const float p = s[e] == -INFINITY ? 0.0f : fast::exp2(s[e] - mnew[i]);
                    pv[i * 4 + j] = p;
                    rs[i] += p;
                }
            }
            *(threadgroup vec<float, 8>*)(&p_tile[hg][j4 * 2 + tn][lane * 8]) = pv;
        }
        KUNROLL
        for (short i = 0; i < 2; ++i) {
            rs[i] += simd_shuffle_xor(rs[i], ushort(1));
            rs[i] += simd_shuffle_xor(rs[i], ushort(8));
        }
        if (row_writer) {
            red_sum[hg][j4][fm] = rs[0];
            red_sum[hg][j4][fm + 8] = rs[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!simd_all(factor[0] == 1.0f && factor[1] == 1.0f)) {
            KUNROLL
            for (short e = 0; e < 16; ++e) {
                const float f = factor[(e >> 2) & 1];
                o0[e] *= f;
                o1[e] *= f;
                o2[e] *= f;
                o3[e] *= f;
            }
        }
        KUNROLL
        for (short i = 0; i < 2; ++i) {
            const int r = fm + i * 8;
            const float tsum = ((red_sum[hg][0][r] + red_sum[hg][1][r]) + red_sum[hg][2][r]) + red_sum[hg][3][r];
            l_run[i] = l_run[i] * factor[i] + tsum;
        }
        const int n_ks = (tile + 15) / 16;
        for (short ks = 0; ks < n_ks; ++ks) {
            const vec<float, 8> pv = *(const threadgroup vec<float, 8>*)(&p_tile[hg][ks][lane * 8]);
            KUNROLL
            for (short e = 0; e < 8; ++e) {
                pa[e] = pv[e];
            }
            const int k0 = t0 + min(ks * 16 + fm, tile - 1);
            const int k1 = t0 + min(ks * 16 + fm + 8, tile - 1);
            const device T* v0 = lat + ulong(k0) * D + j4 * 128 + fn;
            const device T* v1 = lat + ulong(k1) * D + j4 * 128 + fn;
            KUNROLL
            for (short np = 0; np < 4; ++np) {
                KUNROLL
                for (short tn = 0; tn < 2; ++tn) {
                    KUNROLL
                    for (short j = 0; j < 4; ++j) {
                        pb[tn * 8 + j] = v0[np * 32 + tn * 16 + j];
                        pb[tn * 8 + 4 + j] = v1[np * 32 + tn * 16 + j];
                    }
                }
                if (np == 0) {
                    pv_op.run(pa, pb, o0);
                } else if (np == 1) {
                    pv_op.run(pa, pb, o1);
                } else if (np == 2) {
                    pv_op.run(pa, pb, o2);
                } else {
                    pv_op.run(pa, pb, o3);
                }
            }
        }
    }

    // Partials per (b, row, split, head): m (log2 domain), l, unnormalized o.
    KUNROLL
    for (short i = 0; i < 2; ++i) {
        const int head = head0 + fm + i * 8;
        const ulong slot = ((ulong(bb) * TQ + qi) * NSPLIT + sp) * H + head;
        if (j4 == 0 && (lane & 9u) == 0u) {
            m_out[slot] = m_run[i];
            l_out[slot] = l_run[i];
        }
        device float* orow = o_out + slot * D + j4 * 128 + fn;
        KUNROLL
        for (short tn = 0; tn < 2; ++tn) {
            KUNROLL
            for (short j = 0; j < 4; ++j) {
                const short e = tn * 8 + i * 4 + j;
                orow[tn * 16 + j] = o0[e];
                orow[32 + tn * 16 + j] = o1[e];
                orow[64 + tn * 16 + j] = o2[e];
                orow[96 + tn * 16 + j] = o3[e];
            }
        }
    }
"""

# Research variant "reuse" (ForgeKernel review question 1): each simdgroup
# (head group hg, channel quarter j4) loads its 32-key x 128-channel latent
# fragments ONCE per step and consumes them both as K (partial QK over its
# quarter, plus a 16-channel quarter of the shared key) and as V (its 128
# output channels). A 16x16 NAX fragment has the same lane layout as a
# transposed right operand and as a plain one, so no threadgroup staging of
# the latent is needed; only partial scores are exchanged (16 KiB, two
# barriers per 32 keys). QRES keeps the query fragments resident.
# NOT HELPFUL, left off (review-r3, 2026-11-02): 1.2-2.1x slower than "tiled"
# on every decode/verify shape (B1/B4, 4K-128K, 1-128 rows). "tiled" B1 decode
# at 128K already runs at the cache stream floor (0.41 ms vs a 0.41 ms plain
# reduction over the same 151 MB), so there is no traffic left to remove; the
# extra barriers and 64 fragment registers per lane cost occupancy instead.
_ABSORBED_REUSE = r"""
    constexpr int D = 512;
    constexpr int DS = 64;
    constexpr int BK = 32;
    const int TQ = params[0];
    const int S = params[1];
    const int OFF = params[2];
    const int CAUSAL = params[3];
    const int SPLIT = params[4];
    const int NSPLIT = params[5];
    const float scale2 = scalep[0];
    const int qi = int(threadgroup_position_in_grid.x);
    const int sp = int(threadgroup_position_in_grid.y);
    const int bb = int(threadgroup_position_in_grid.z);
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int hg = int(sg) / 4;
    const int j4 = int(sg) % 4;
    const int head0 = hg * 16;
    const short qid = short(lane >> 2);
    const short fm = short((qid & 4) | ((lane >> 1) & 3));
    const short fn = short(((qid & 2) | (lane & 1)) * 4);

    threadgroup float s_x[2][4][32 * 16];

    const int klim = CAUSAL ? min(S, OFF + qi + 1) : S;
    // Keys before starts[bb] are this lane's left padding (a padded batch
    // cache): the visible keys stay one contiguous range [start, klim).
    const int kstart = min(max(starts[bb], 0), klim);
    const int kbeg = min(max(sp * SPLIT, kstart), klim);
    const int kend = max(kbeg, min(sp * SPLIT + SPLIT, klim));

    constexpr auto qk_desc = matmul2d_descriptor(
        16, 32, 16, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroup> qk_op;
    constexpr auto pv_desc = matmul2d_descriptor(
        16, 32, 16, false, false, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroup> pv_op;

    auto pa = pv_op.template get_left_input_cooperative_tensor<float, T, float>();
    auto pb = pv_op.template get_right_input_cooperative_tensor<float, T, float>();
    auto o0 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o1 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o2 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o3 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    KUNROLL
    for (short e = 0; e < 16; ++e) {
        o0[e] = 0.0f;
        o1[e] = 0.0f;
        o2[e] = 0.0f;
        o3[e] = 0.0f;
    }
    float m_run[2] = {-INFINITY, -INFINITY};
    float l_run[2] = {0.0f, 0.0f};

    const device T* qr0 = q + ((ulong(bb) * H + head0 + fm) * TQ + qi) * D + j4 * 128 + fn;
    const device T* qr1 = q + ((ulong(bb) * H + head0 + fm + 8) * TQ + qi) * D + j4 * 128 + fn;
    const device T* qs0 = qs + ((ulong(bb) * H + head0 + fm) * TQ + qi) * DS + j4 * 16 + fn;
    const device T* qs1 = qs + ((ulong(bb) * H + head0 + fm + 8) * TQ + qi) * DS + j4 * 16 + fn;
    const device T* lat = latent + ulong(bb) * ulong(params[6]) * D + j4 * 128 + fn;
    const device T* shb = shared + ulong(bb) * ulong(params[6]) * DS + j4 * 16 + fn;

    vec<T, 8> qf[QRES ? 9 : 1];
    if (QRES) {
        KUNROLL
        for (short c = 0; c < 8; ++c) {
            KUNROLL
            for (short j = 0; j < 4; ++j) {
                qf[c][j] = qr0[c * 16 + j];
                qf[c][4 + j] = qr1[c * 16 + j];
            }
        }
        KUNROLL
        for (short j = 0; j < 4; ++j) {
            qf[8][j] = qs0[j];
            qf[8][4 + j] = qs1[j];
        }
    }

    for (int t0 = kbeg; t0 < kend; t0 += BK) {
        const int tile = min(BK, kend - t0);
        // Latent fragments of this quarter: key block kb (rows fm, fm + 8),
        // channel chunk c (columns fn .. fn + 3). Tail keys clamp to the
        // last visible key of the tile.
        vec<T, 8> L[2][8];
        vec<T, 8> SH[2];
        KUNROLL
        for (short kb = 0; kb < 2; ++kb) {
            const ulong k0 = ulong(t0 + min(kb * 16 + fm, tile - 1));
            const ulong k1 = ulong(t0 + min(kb * 16 + fm + 8, tile - 1));
            KUNROLL
            for (short c = 0; c < 8; ++c) {
                KUNROLL
                for (short j = 0; j < 4; ++j) {
                    L[kb][c][j] = lat[k0 * D + c * 16 + j];
                    L[kb][c][4 + j] = lat[k1 * D + c * 16 + j];
                }
            }
            KUNROLL
            for (short j = 0; j < 4; ++j) {
                SH[kb][j] = shb[k0 * DS + j];
                SH[kb][4 + j] = shb[k1 * DS + j];
            }
        }
        auto qa = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
        auto kt = qk_op.template get_right_input_cooperative_tensor<T, T, float>();
        auto sc = qk_op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(qa)>, metal::remove_addrspace_t<decltype(kt)>, float>();
        KUNROLL
        for (short e = 0; e < 16; ++e) {
            sc[e] = 0.0f;
        }
        KUNROLL
        for (short c = 0; c < 9; ++c) {
            KUNROLL
            for (short e = 0; e < 4; ++e) {
                if (QRES) {
                    qa[e] = qf[c][e];
                    qa[4 + e] = qf[c][4 + e];
                } else if (c < 8) {
                    qa[e] = qr0[c * 16 + e];
                    qa[4 + e] = qr1[c * 16 + e];
                } else {
                    qa[e] = qs0[e];
                    qa[4 + e] = qs1[e];
                }
            }
            KUNROLL
            for (short e = 0; e < 8; ++e) {
                kt[e] = c < 8 ? L[0][c][e] : SH[0][e];
                kt[8 + e] = c < 8 ? L[1][c][e] : SH[1][e];
            }
            qk_op.run(qa, kt, sc);
        }
        // Sum the four channel-quarter partial scores (fixed order).
        threadgroup float* mine = s_x[hg][j4];
        KUNROLL
        for (short e = 0; e < 16; ++e) {
            mine[lane * 16 + e] = sc[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s[16];
        float mx_i[2] = {-INFINITY, -INFINITY};
        KUNROLL
        for (short e = 0; e < 16; ++e) {
            const float v = ((s_x[hg][0][lane * 16 + e] + s_x[hg][1][lane * 16 + e])
                             + s_x[hg][2][lane * 16 + e]) + s_x[hg][3][lane * 16 + e];
            const short tn = e >> 3;
            const short j = e & 3;
            const bool ok = tn * 16 + fn + j < tile;
            s[e] = ok ? v * scale2 : -INFINITY;
            mx_i[(e >> 2) & 1] = max(mx_i[(e >> 2) & 1], s[e]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float factor[2];
        float mnew[2];
        KUNROLL
        for (short i = 0; i < 2; ++i) {
            float m = mx_i[i];
            m = max(m, simd_shuffle_xor(m, ushort(1)));
            m = max(m, simd_shuffle_xor(m, ushort(8)));
            mnew[i] = max(m_run[i], m);
            factor[i] = m_run[i] == -INFINITY ? 0.0f : fast::exp2(m_run[i] - mnew[i]);
            m_run[i] = mnew[i];
        }
        float rs[2] = {0.0f, 0.0f};
        KUNROLL
        for (short e = 0; e < 16; ++e) {
            const short i = (e >> 2) & 1;
            const float p = s[e] == -INFINITY ? 0.0f : fast::exp2(s[e] - mnew[i]);
            s[e] = p;
            rs[i] += p;
        }
        KUNROLL
        for (short i = 0; i < 2; ++i) {
            l_run[i] = l_run[i] * factor[i] + rs[i];
        }
        if (!simd_all(factor[0] == 1.0f && factor[1] == 1.0f)) {
            KUNROLL
            for (short e = 0; e < 16; ++e) {
                const float f = factor[(e >> 2) & 1];
                o0[e] *= f;
                o1[e] *= f;
                o2[e] *= f;
                o3[e] *= f;
            }
        }
        KUNROLL
        for (short ks = 0; ks < 2; ++ks) {
            KUNROLL
            for (short e = 0; e < 8; ++e) {
                pa[e] = s[ks * 8 + e];
            }
            KUNROLL
            for (short np = 0; np < 4; ++np) {
                KUNROLL
                for (short e = 0; e < 8; ++e) {
                    pb[e] = L[ks][np * 2][e];
                    pb[8 + e] = L[ks][np * 2 + 1][e];
                }
                if (np == 0) {
                    pv_op.run(pa, pb, o0);
                } else if (np == 1) {
                    pv_op.run(pa, pb, o1);
                } else if (np == 2) {
                    pv_op.run(pa, pb, o2);
                } else {
                    pv_op.run(pa, pb, o3);
                }
            }
        }
    }

    KUNROLL
    for (short i = 0; i < 2; ++i) {
        float l = l_run[i];
        l += simd_shuffle_xor(l, ushort(1));
        l += simd_shuffle_xor(l, ushort(8));
        const int head = head0 + fm + i * 8;
        const ulong slot = ((ulong(bb) * TQ + qi) * NSPLIT + sp) * H + head;
        if (j4 == 0 && (lane & 9u) == 0u) {
            m_out[slot] = m_run[i];
            l_out[slot] = l;
        }
        device float* orow = o_out + slot * D + j4 * 128 + fn;
        KUNROLL
        for (short tn = 0; tn < 2; ++tn) {
            KUNROLL
            for (short j = 0; j < 4; ++j) {
                const short e = tn * 8 + i * 4 + j;
                orow[tn * 16 + j] = o0[e];
                orow[32 + tn * 16 + j] = o1[e];
                orow[64 + tn * 16 + j] = o2[e];
                orow[96 + tn * 16 + j] = o3[e];
            }
        }
    }
"""

# One thread per output channel of one (b, row, head): combine split partials.
_COMBINE = r"""
    constexpr int D = 512;
    const int TQ = params[0];
    const int NSPLIT = params[5];
    const int c = int(thread_position_in_grid.x);
    const int head = int(thread_position_in_grid.y);
    const int br = int(thread_position_in_grid.z);   // b * TQ + row
    const int bb = br / TQ;
    const int qi = br % TQ;
    float m = -INFINITY;
    for (int sp = 0; sp < NSPLIT; ++sp) {
        m = max(m, m_in[(ulong(br) * NSPLIT + sp) * H + head]);
    }
    float l = 0.0f;
    float acc = 0.0f;
    if (m != -INFINITY) {
        for (int sp = 0; sp < NSPLIT; ++sp) {
            const ulong slot = (ulong(br) * NSPLIT + sp) * H + head;
            const float ms = m_in[slot];
            const float w = ms == -INFINITY ? 0.0f : fast::exp2(ms - m);
            l += l_in[slot] * w;
            acc += o_in[slot * D + c] * w;
        }
    }
    const float inv = l == 0.0f ? 0.0f : 1.0f / l;
    out[((ulong(bb) * H + head) * TQ + qi) * D + c] = T(acc * inv);
"""


# Absorbed partial-kernel variants: "tiled" (128-key tiles, quarters split
# keys for QK and channels for PV) and the research "reuse" / "reuse_q".
ABSORBED_VARIANTS = ("tiled", "reuse", "reuse_q")  # tiled is the only one wired


@cache
def _absorbed_kernels(variant="tiled"):
    source = _ABSORBED if variant == "tiled" else _ABSORBED_REUSE
    partial = mx.fast.metal_kernel(name=f"mlx2_kimi_mla_latent_flash_{variant}_v1",
        input_names=["q", "qs", "latent", "shared", "params", "scalep", "starts"],
        output_names=["m_out", "l_out", "o_out"], header=_HEADER, source=source,
        ensure_row_contiguous=True)
    combine = mx.fast.metal_kernel(name="mlx2_kimi_mla_latent_combine_v1",
        input_names=["m_in", "l_in", "o_in", "params"], output_names=["out"],
        source=_COMBINE, ensure_row_contiguous=True)
    return partial, combine


def absorbed_geometry(q, qs, latent, shared, mask, heads=HEADS, width=None):
    if (q.dtype not in (mx.bfloat16, mx.float16)
            or any(z.dtype != q.dtype for z in (qs, latent, shared))):
        raise ValueError("Kimi latent flash needs matching BF16/FP16 operands")
    if q.ndim != 4 or qs.ndim != 4 or latent.ndim != 4 or shared.ndim != 4:
        raise ValueError("Kimi latent flash needs q[B,H,T,512], qs[B,H,T,64], latent/shared[B,1,S,*]")
    b, h, t, _ = q.shape
    cap = latent.shape[2]
    width = cap if width is None else int(width)
    # The partial kernel's grid has no head axis: one threadgroup covers all
    # HEADS heads of a row, so any other head count reads or writes out of
    # bounds (``heads`` reaches the combine kernel only).
    if (h != heads or heads != HEADS or q.shape[-1] != LATENT or qs.shape != (b, h, t, SHARED)
            or latent.shape != (b, 1, cap, LATENT) or shared.shape != (b, 1, cap, SHARED)
            or t < 1 or width < 1 or width > cap):
        raise ValueError("Kimi latent flash geometry")
    return (b, t, width, *_offset(mask, t, width))


def split_plan(batch, rows, width, *, target=256, minimum=2048):
    """Key split per threadgroup: enough (row, split) threadgroups to fill
    the GPU, never fewer than ``minimum`` keys (a multiple of the 128 tile)."""
    want = max(1, -(-target // max(1, batch*rows)))
    split = max(minimum, -(-width // want))
    split = -(-split // 128) * 128
    return split, -(-width // split)


def _starts(starts, b):
    """Per-lane first visible key (int32 [B]); None means no left padding."""
    if starts is None:
        return mx.zeros((b,), mx.int32)
    if starts.shape != (b,):
        raise ValueError("Kimi latent flash starts must be one int per batch row")
    return starts.astype(mx.int32)


def absorbed(q, qs, latent, shared, mask, *, scale, heads=HEADS, split=None, offset=None,
             variant="tiled", stage="all", width=None, starts=None):
    """Latent-space O[B, H, T, 512] (before the V un-absorption).

    ``width`` names the visible prefix when latent/shared are the cache
    buffers [B,1,capacity,*] (read in place: a B > 1 slice is not row
    contiguous and would otherwise be copied every call).

    ``offset`` overrides the causal offset (validation of hidden rows only).
    ``starts`` [B] hides each lane's leading keys (a left-padded batch cache:
    key k is visible to lane b iff starts[b] <= k, within the causal limit).
    ``stage="partial"`` returns the split partials (m, l, o) without the
    combine, for timing the two stages apart.
    """
    b, t, width, natural, causal = absorbed_geometry(q, qs, latent, shared, mask, heads, width)
    offset = natural if offset is None else int(offset)
    if split is None:
        split, count = split_plan(b, t, width)
    else:
        split = -(-int(split) // 128) * 128
        count = -(-width // split)
    cap = latent.shape[2]
    params = mx.array([t, width, offset, int(causal), split, count, cap], mx.int32)
    partial, _ = _absorbed_kernels(variant)
    template = [("T", q.dtype), ("H", heads)]
    if variant != "tiled":
        template.append(("QRES", int(variant == "reuse_q")))
    m, l, o = partial(
        inputs=[q, qs, latent.reshape(b, cap, LATENT), shared.reshape(b, cap, SHARED),
                params, mx.array([scale*_LOG2E], mx.float32), _starts(starts, b)],
        template=template,
        grid=(t*256, count, b), threadgroup=(256, 1, 1),
        output_shapes=[(b*t*count*heads,), (b*t*count*heads,), (b*t*count*heads, LATENT)],
        output_dtypes=[mx.float32]*3)
    if stage == "partial":
        return m, l, o
    return combine_partials(m, l, o, params, b, t, heads, q.dtype)


def combine_partials(m, l, o, params, b, t, heads, dtype):
    _, combine = _absorbed_kernels("tiled")
    return combine(inputs=[m, l, o, params], template=[("T", dtype), ("H", heads)],
        grid=(LATENT, heads, b*t), threadgroup=(256, 1, 1),
        output_shapes=[(b, heads, t, LATENT)], output_dtypes=[dtype])[0]


def absorbed_params(q, latent, mask, *, split=None, offset=None):
    """The params vector ``absorbed`` builds (for timing combine alone)."""
    b, _, t, _ = q.shape
    width = latent.shape[2]
    _, causal = _offset(mask, t, width)
    natural = width-t if causal else 0
    if split is None:
        split, count = split_plan(b, t, width)
    else:
        split = -(-int(split) // 128) * 128
        count = -(-width // split)
    return mx.array([t, width, natural if offset is None else offset, int(causal), split, count,
                     width], mx.int32), count


def absorbed_reference(q, qs, latent, shared, mask, *, scale, heads=HEADS, offset=None, width=None,
                       starts=None):
    """FP32 ops law of ``absorbed`` (validation only); hidden rows are zeros."""
    b, t, width, natural, causal = absorbed_geometry(q, qs, latent, shared, mask, heads, width)
    offset = natural if offset is None else int(offset)
    latent, shared = latent[:, :, :width], shared[:, :, :width]
    lat = latent.astype(mx.float32)
    score = (q.astype(mx.float32) @ lat.swapaxes(-1, -2)
             + qs.astype(mx.float32) @ shared.astype(mx.float32).swapaxes(-1, -2)) * scale
    if causal:
        visible = mx.arange(offset, offset+t)[:, None] >= mx.arange(width)[None, :]
        score = mx.where(visible, score, -float("inf"))
    if starts is not None:
        lead = mx.arange(width)[None, :] >= _starts(starts, b)[:, None]
        score = mx.where(lead[:, None, None, :], score, -float("inf"))
    return _safe_softmax(score) @ lat
