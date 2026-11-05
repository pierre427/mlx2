# SPDX-License-Identifier: MIT
"""Inkling 16-row NAX prefill tiles with inline relative-position attention.

Adapted from internal qwen4_qsa_nax.py at f9acea104b9e306e4a944c70e1d6c2997898e0cd
(MIT, upstream mining chain in provenance/flashnext.json). Removes sparse block
selection; packs four query positions and their four GQA heads, with row-specific
relative bias, causal/window/padding visibility and explicit source binding.
"""

import mlx.core as mx

HEADER = "#include <metal_stdlib>\n#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\n"
SOURCE = r"""
    constexpr int NSG = 4;
    constexpr int DS  = D / NSG;
    constexpr int NKS = DS / 16;
    constexpr int NNT = DS / 32;

    const int L=q_shape[2], TOT=k_shape[2];
    constexpr int QT=16/GQA;
    const ushort sg   = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const uint tile = threadgroup_position_in_grid.y;
    const uint  bkv   = threadgroup_position_in_grid.z;   // b * NKVH + hkv
    const uint  b     = bkv / NKVH;
    const uint  hkv   = bkv % NKVH;

    const short qid = lane >> 2;
    const short fm0 = ((qid & 4) | ((lane >> 1) & 3));    // row = query head
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;

    threadgroup float sS[NSG * 32 * 16];

    const size_t k_head =
        (size_t)b * k_strides[0] + (size_t)hkv * k_strides[1];
    const size_t v_head =
        (size_t)b * v_strides[0] + (size_t)hkv * v_strides[1];

    const int lpad=left_pad[b];
    int head_of[2], tok_of[2];
    bool row_live[2];
    for (short rr=0; rr<2; ++rr) {
        int r=fm0+rr*8;
        head_of[rr]=int(hkv)*GQA+r%GQA;
        tok_of[rr]=int(tile)*QT+r/GQA;
        row_live[rr]=tok_of[rr]<L;
    }
    typedef metal::vec<T, 8> frag_t;
    frag_t qf[NKS];
    for (short ks = 0; ks < NKS; ks++) {
        for (short i = 0; i < 8; i++) {
            short rr = (i >> 2);
            int c = sg * DS + ks * 16 + fn0 + (i % 4);
            qf[ks][i] = row_live[rr]
                ? q[(size_t)b*q_strides[0]+head_of[rr]*q_strides[1]+tok_of[rr]*q_strides[2]+c*q_strides[3]]
                : T(0);
        }
    }

    float of[NNT][16];
    for (short t = 0; t < NNT; t++)
        for (short i = 0; i < 16; i++) of[t][i] = 0.0f;
    float rmax_r[2] = {-INFINITY, -INFINITY};
    float rsum_r[2] = {0.0f, 0.0f};

    constexpr auto desc_qk = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto desc_pv = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    const float qk_scale = meta[1];

    const int first_pos=int(meta[0])+int(tile)*QT;
    const int begin=(WIN>0 && first_pos>=lpad) ? max(0,first_pos-WIN+1)/32*32 : 0;
    const int end=first_pos<lpad ? TOT : min(TOT,int(meta[0])+(int(tile)+1)*QT);
    for (int u0=begin; u0<end; u0+=32) {
        uint physical_rows[4];
        for (short qq=0; qq<4; ++qq)
            physical_rows[qq]=uint(min(u0+fm0+qq*8,TOT-1));
        {
            mpp::tensor_ops::matmul2d<desc_qk, metal::execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<T, T, float>();
            auto cb = op.get_right_input_cooperative_tensor<T, T, float>();
            auto cc = op.get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(ca)>,
                metal::remove_addrspace_t<decltype(cb)>, float>();
            for (short i = 0; i < 16; i++) cc[i] = 0.0f;
            for (short ks = 0; ks < NKS; ks++) {
                for (short i = 0; i < 8; i++) ca[i] = qf[ks][i];
                for (short i = 0; i < 8; i++) {
                    int c = sg * DS + ks * 16 + fn0 + (i % 4);
                    const uint physical = physical_rows[(i >> 2)];
                    const uint physical_2 = physical_rows[2 + (i >> 2)];
                    cb[i] = k[
                        k_head + (size_t)physical * k_strides[2]
                        + (size_t)c * k_strides[3]
                    ];
                    cb[8 + i] = k[
                        k_head + (size_t)physical_2 * k_strides[2]
                        + (size_t)c * k_strides[3]
                    ];
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

        // Each matrix row is a (query position, GQA head) pair.
        for (short f=0; f<2; ++f) {
            for (short rr=0; rr<2; ++rr) {
                const int row=tok_of[rr], head=head_of[rr];
                const int pos=int(meta[0])+row;
                const bool padded=pos<lpad;
                for (short co=0; co<4; ++co) {
                    const int col=u0+f*16+fn0+co, dist=pos-col;
                    const short idx=f*8+rr*4+co;
                    bool visible=row_live[rr] && col<TOT && (padded ||
                        (dist>=0 && (WIN==0 || dist<WIN) && col>=lpad));
                    float bias=0.0f;
                    if (visible && !padded && dist<EXT) {
                        for (int d=0; d<R; ++d)
                            bias+=float(rel[(size_t)b*rel_strides[0]+row*rel_strides[1]
                                +head*rel_strides[2]+d*rel_strides[3]])
                                *float(proj[d*proj_strides[0]+dist*proj_strides[1]]);
                    }
                    const float temperature=row_live[rr] ?
                        float(tau[(size_t)b*tau_strides[0]+row*tau_strides[1]]) : 1.0f;
                    s[idx]=visible ? (padded ? 0.0f : s[idx]+float(T(float(T(bias))*temperature))) : -INFINITY;
                }
            }
        }
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

        float alpha[2], p[16], newmax[2], newsum[2];
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
            newmax[rr] = mn;
            newsum[rr] = rsum_r[rr] * alpha[rr] + sm;
        }
        for (short rr = 0; rr < 2; rr++) {
            rmax_r[rr] = newmax[rr];
            rsum_r[rr] = newsum[rr];
        }

        for (short t = 0; t < NNT; t++) {
            for (short f = 0; f < 2; f++)
                for (short rr = 0; rr < 2; rr++)
                    for (short co = 0; co < 4; co++)
                        of[t][f * 8 + rr * 4 + co] *= alpha[rr];

            int nbase = sg * DS + t * 32;
            mpp::tensor_ops::matmul2d<desc_pv, metal::execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<T, T, float>();
            auto cb = op.get_right_input_cooperative_tensor<T, T, float>();
            auto cc = op.get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(ca)>,
                metal::remove_addrspace_t<decltype(cb)>, float>();
            for (short i = 0; i < 16; i++) cc[i] = of[t][i];
            for (short ks = 0; ks < 2; ks++) {
                for (short i = 0; i < 8; i++) ca[i] = (T)p[ks * 8 + i];
                for (short i = 0; i < 8; i++) {
                    uint physical = physical_rows[ks * 2 + (i >> 2)];
                    int co = nbase + fn0 + (i % 4);
                    cb[i] = v[
                        v_head + (size_t)physical * v_strides[2]
                        + (size_t)co * v_strides[3]
                    ];
                    cb[8 + i] = v[
                        v_head + (size_t)physical * v_strides[2]
                        + (size_t)(co + 16) * v_strides[3]
                    ];
                }
                op.run(ca, cb, cc);
            }
            for (short i = 0; i < 16; i++) of[t][i] = cc[i];
        }
    }

    for (short t = 0; t < NNT; t++) {
        for (short f = 0; f < 2; f++) {
            for (short rr = 0; rr < 2; rr++) {
                if (!row_live[rr]) continue;
                float inv = (rsum_r[rr] > 0.0f) ? (1.0f / rsum_r[rr]) : 0.0f;
                for (short co = 0; co < 4; co++) {
                    int c = sg * DS + t * 32 + f * 16 + fn0 + co;
                    out[((size_t)(b * NQH + head_of[rr]) * L + tok_of[rr]) * D + c] =
                        T(of[t][f * 8 + rr * 4 + co] * inv);
                }
            }
        }
    }
"""

_KERNEL = mx.fast.metal_kernel(
    name="mlx2_inkling_relative_attention_nax",
    input_names=["q", "k", "v", "rel", "proj", "tau", "left_pad", "meta"],
    output_names=["out"],
    header=HEADER,
    source=SOURCE,
    ensure_row_contiguous=False,
)


def relative_attention_nax(
    q, k, v, rel, proj, *, offset, scale, window=0, left_padding=None, tau=None
):
    b, h, length, dim = q.shape
    hk = k.shape[1]
    if dim != 128 or h // hk != 4 or q.dtype not in (mx.bfloat16, mx.float16):
        raise ValueError("Inkling NAX attention requires GQA4, D128 and 16-bit inputs")
    tau = (
        mx.ones((b, length), q.dtype)
        if tau is None
        else mx.broadcast_to(tau, (b, length))
    )
    pad = (
        mx.zeros((b,), mx.int32)
        if left_padding is None
        else left_padding.astype(mx.int32)
    )
    return _KERNEL(
        inputs=[q, k, v, rel, proj, tau, pad, mx.array([offset, scale], mx.float32)],
        template=[
            ("T", q.dtype),
            ("D", dim),
            ("NQH", h),
            ("NKVH", hk),
            ("GQA", h // hk),
            ("R", rel.shape[3]),
            ("EXT", proj.shape[1]),
            ("WIN", window),
        ],
        grid=(128, (length + 3) // 4, b * hk),
        threadgroup=(128, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]
