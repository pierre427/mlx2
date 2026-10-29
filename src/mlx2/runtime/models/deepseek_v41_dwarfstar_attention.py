# SPDX-License-Identifier: MIT
"""Unwired original DwarfStar DS4.1 attention templates, minimal layout shim.

Mined MIT code retains provenance/dwarfstar-flash-attention.NOTICE.txt.
Original template arithmetic is unchanged; callable declarations differ.
F16 staging/fast-math/normalization differ from Source's current reference;
this is an explicit numerical-profile component, not serving selection.
"""
from functools import lru_cache
from types import SimpleNamespace

HEADER = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))
#define N_SIMDWIDTH 32
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for(x)
#define PAD2(x,n) (((x)+(n)-1)&~((n)-1))
#define OP_FLASH_ATTN_EXT_VEC_NQPSG 1
#define OP_FLASH_ATTN_EXT_VEC_NCPSG 32
#define FC_flash_attn_ext_has_mask true
#define FC_flash_attn_ext_has_sinks true
#define FC_flash_attn_ext_has_bias false
#define FC_flash_attn_ext_has_scap false
#define FC_flash_attn_ext_has_kvpad false
#define FC_flash_attn_ext_bc_mask true
#define FC_flash_attn_ext_ns10 512
#define FC_flash_attn_ext_ns20 512
#define FC_flash_attn_ext_vec_has_mask true
#define FC_flash_attn_ext_vec_has_sinks true
#define FC_flash_attn_ext_vec_has_bias false
#define FC_flash_attn_ext_vec_has_scap false
#define FC_flash_attn_ext_vec_has_kvpad false
#define FC_flash_attn_ext_vec_shared_kvpad false
#define FC_flash_attn_ext_vec_ns10 512
#define FC_flash_attn_ext_vec_ns20 512
#define FC_flash_attn_ext_vec_nsg NSG_PARAM
#define FC_flash_attn_ext_vec_nwg NWG_PARAM

namespace mlx2_dwarfstar {
template <typename type4>
void dequantize_f16_t4(device const half4 * src, short il, thread type4 & reg) {
    reg = (type4)(*(src));
}

struct ds4_metal_args_flash_attn_ext {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
};

struct ds4_metal_args_flash_attn_ext_vec {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
};

struct ds4_metal_args_flash_attn_ext_vec_reduce {
    int32_t nrows;
};

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}
template<
    typename q_t,
    typename q4_t,
    typename q8x8_t,
    typename k_t,
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,
    typename qk8x8_t,
    typename s_t,
    typename s2_t,
    typename s8x8_t,
    typename o_t,
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t,
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t,
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,
    short DV,
    short Q,
    short C,
    short NSG>
void kernel_flash_attn_ext_impl(
        const device ds4_metal_args_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
    const ushort iq3 = tgpig[2];
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0]*Q;

#define NS10 (FC_flash_attn_ext_ns10)
#define NS20 (FC_flash_attn_ext_ns20)

    constexpr short KV   = 8;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;
    constexpr short DV4  = DV/4;
    constexpr short DV16 = DV/16;

    constexpr short PV   = PAD2(DV, 64);
    constexpr short PV4  = PV/4;
    constexpr short PV8  = PV/8;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NQ  = Q/NSG;
    constexpr short SH  = 2*C;

    constexpr short TS = 2*SH;
    constexpr short T  = DK + 2*PV;

    threadgroup q_t  * sq  = (threadgroup q_t  *) (shmem_f16 + 0*T);
    threadgroup q4_t * sq4 = (threadgroup q4_t *) (shmem_f16 + 0*T);
    threadgroup o_t  * so  = (threadgroup o_t  *) (shmem_f16 + 0*T + Q*DK);
    threadgroup o4_t * so4 = (threadgroup o4_t *) (shmem_f16 + 0*T + Q*DK);
    threadgroup s_t  * ss  = (threadgroup s_t  *) (shmem_f16 + Q*T);
    threadgroup s2_t * ss2 = (threadgroup s2_t *) (shmem_f16 + Q*T);

    threadgroup k_t    * sk    = (threadgroup k_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);
    threadgroup k4x4_t * sk4x4 = (threadgroup k4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);

    threadgroup v_t    * sv    = (threadgroup v_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);
    threadgroup v4x4_t * sv4x4 = (threadgroup v4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);

    threadgroup half2 * sm2 = (threadgroup half2 *) (shmem_f16 + Q*T + 2*C);

    device const half2 * pm2[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        pm2[jj] = (device const half2 *) ((device const char *) mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    {
        const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
        const int32_t nblk0 = ((args.ne11 + C - 1)/C);

        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;
    }

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        device const float4 * q4 = (device const float4 *) ((device const char *) q + j*args.nb01);

        for (short i = tiisg; i < DK4; i += NW) {
            if (iq1 + j < args.ne01) {
                sq4[j*DK4 + i] = (q4_t) q4[i];
            } else {
                sq4[j*DK4 + i] = 0;
            }
        }
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        for (short i = tiisg; i < DV4; i += NW) {
            so4[j*PV4 + i] = 0;
        }

        for (short i = tiisg; i < SH; i += NW) {
            ss[j*SH + i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ] = { [0 ... NQ-1] = 0.0f };

    {
        float M[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };

        float slope = 1.0f;

        if (FC_flash_attn_ext_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        for (int ic0 = 0; ; ++ic0) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            if (FC_flash_attn_ext_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_has_mask) {
                    threadgroup half * sm = (threadgroup half *) (sm2);

                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        for (short i = tiisg; i < C; i += NW) {
                            if (ic + i >= args.ne11) {
                                sm[2*j*SH + i] = -MAXHALF;
                            }
                        }
                    }
                } else {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        pm2[jj] = (device const half2 *) ((device const half *) mask +
                                (iq1 + j)*C +
                                (iq2%args.ne32)*(C*args.ne31) +
                                (iq3%args.ne33)*(C*args.ne31*args.ne32));
                    }
                }

                ic = 0;
            }

            char blk_cur = 1;

            if (FC_flash_attn_ext_has_mask) {
                blk_cur = blk[ic0];

                if (blk_cur == 0) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }

                    continue;
                }

                if (blk_cur == 1) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        if (FC_flash_attn_ext_bc_mask) {
                            sm2[j*SH + tiisg] = (iq1 + j) < args.ne31 ? pm2[jj][tiisg] : half2(-MAXHALF, -MAXHALF);
                        } else {
                            sm2[j*SH + tiisg] = pm2[jj][tiisg];
                        }

                        pm2[jj] += NW;
                    }
                } else if (blk_cur == 2) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }
                }
            }

            if (is_same<kd4x4_t, k4x4_t>::value) {
                device      const k_t * pk = (device const k_t *) (k + ic*args.nb11);
                threadgroup const q_t * pq = sq;
                threadgroup       s_t * ps = ss;

                pk += sgitg*(8*NS10);
                ps += sgitg*(8*1);

                static_assert((C/8) % NSG == 0, "");

                constexpr short NC = (C/8)/NSG;

                FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    if (DK % 16 != 0) {
                        k8x8_t mk;
                        q8x8_t mq;

                        FOR_UNROLL (short i = 0; i < DK8; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mk, pk + 8*i, NS10, 0, true);
                            simdgroup_load(mq, pq + 8*i, DK);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                        }
                    } else {
                        k8x8_t mk[2];
                        q8x8_t mq[2];

                        #pragma unroll (MIN(DK8/2, 4*NSG))
                        for (short i = 0; i < DK8/2; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mq[0], pq + 0*8 + 16*i, DK);
                            simdgroup_load(mq[1], pq + 1*8 + 16*i, DK);

                            simdgroup_load(mk[0], pk + 0*8 + 16*i, NS10, 0, true);
                            simdgroup_load(mk[1], pk + 1*8 + 16*i, NS10, 0, true);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                            simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                        }
                    }

                    simdgroup_store(mqk, ps, SH, 0, false);

                    pk += 8*(NSG*NS10);
                    ps += 8*(NSG);
                }
            } else {
                for (short ccc = 0; ccc < (C/8)/NSG; ++ccc) {
                    const short cc = ccc*NSG + sgitg;

                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    for (short ii = 0; ii < DK16; ii += 4) {
                        device const kd4x4_t * pk4x4 = (device const kd4x4_t *) (k + ((ic + 8*cc + ty)*args.nb11));

                        if (DK16%4 == 0) {
                            {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            FOR_UNROLL (short k = 0; k < 4; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        } else {
                            if (ii + tx < DK16) {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            for (short k = 0; k < 4 && ii + k < DK16; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        }
                    }

                    simdgroup_store(mqk, ss + 8*cc, SH, 0, false);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];

                float2 s2 = ss2[j*SH/2 + tiisg]*args.scale;

                if (FC_flash_attn_ext_has_scap) {
                    s2 = args.logit_softcap*precise::tanh(s2);
                }

                if (blk_cur != 2) {
                    if (FC_flash_attn_ext_has_bias) {
                        s2 += s2_t(sm2[j*SH + tiisg])*slope;
                    } else {
                        s2 += s2_t(sm2[j*SH + tiisg]);
                    }
                }

                M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));

                const float  ms  = exp(m  - M[jj]);
                const float2 vs2 = exp(s2 - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);

                ss2[j*SH/2 + tiisg] = vs2;

                if (DV4 % NW == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                        const short i = ii*NW + tiisg;

                        so4[j*PV4 + i] *= ms;
                    }
                } else {
                    for (short i = tiisg; i < DV4; i += NW) {
                        so4[j*PV4 + i] *= ms;
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            {
                if (is_same<vd4x4_t, v4x4_t>::value) {
                    static_assert(PV8 % NSG == 0, "");

                    constexpr short NO = PV8/NSG;

                    o8x8_t lo[NO];

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_load(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }

                    {
                        device const v_t * pv = (device const v_t *) (v + ic*args.nb21);

                        pv += 8*sgitg;

                        if (DV <= 64) {
                            FOR_UNROLL (short cc = 0; cc < C/8; ++cc) {
                                s8x8_t vs;
                                simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[2];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG, NS20, 0, false);

                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs, mv[0], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs, mv[1], lo[2*ii + 1]);
                                }

                                pv  += 8*NS20;
                            }
                        } else {
                            constexpr short NC = (C/8)/2;

                            FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                                s8x8_t vs[2];

                                simdgroup_load(vs[0], ss + 16*cc + 0, SH, 0, false);
                                simdgroup_load(vs[1], ss + 16*cc + 8, SH, 0, false);

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[4];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[2], pv + 0*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[3], pv + 8*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);

                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[0], mv[0], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[0], mv[1], lo[2*ii + 1]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[1], mv[2], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[1], mv[3], lo[2*ii + 1]);
                                }

                                pv  += 2*8*NS20;
                            }
                        }
                    }

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_store(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }
                } else {
                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    for (short cc = 0; cc < C/8; ++cc) {
                        s8x8_t vs;
                        simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                        for (short ii = 4*sgitg; ii < DV16; ii += 4*NSG) {
                            device const vd4x4_t * pv4x4 = (device const vd4x4_t *) (v + ((ic + 8*cc + ty)*args.nb21));

                            if (DV16%4 == 0) {
                                {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                FOR_UNROLL (short k = 0; k < 4; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            } else {
                                if (ii + tx < DV16) {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                for (short k = 0; k < 4 && ii + k < DV16; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            }
                        }
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (FC_flash_attn_ext_has_sinks) {
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];
                const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;

                M[jj] = simd_max(max(M[jj], s));

                const float ms = exp(m - M[jj]);
                const float vs = exp(s - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs);

                for (short i = tiisg; i < DV4; i += NW) {
                    so4[j*PV4 + i] *= ms;
                }
            }
        }
    }

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        if (iq1 + j >= args.ne01) {
            break;
        }

        device float4 * dst4 = (device float4 *) dst + ((uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)(iq1 + j)*args.ne1)*DV4;

        const float scale = S[jj] == 0.0 ? 0.0f : 1.0f/S[jj];

        if (DV4 % NW == 0) {
            FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                const short i = ii*NW + tiisg;

                dst4[i] = (float4) so4[j*PV4 + i]*scale;
            }
        } else {
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[i] = (float4) so4[j*PV4 + i]*scale;
            }
        }
    }

#undef NS10
#undef NS20
}

template<
    typename q4_t,
    typename k4_t,
    typename v4_t,
    typename qk_t,
    typename s_t,
    typename s4_t,
    typename o4_t,
    typename kd4_t,
    short nl_k,
    void (*deq_k_t4)(device const kd4_t *, short, thread k4_t &),
    typename vd4_t,
    short nl_v,
    void (*deq_v_t4)(device const vd4_t *, short, thread v4_t &),
    short DK,
    short DV,
    short NE = 4,
    short Q  = OP_FLASH_ATTN_EXT_VEC_NQPSG,
    short C  = OP_FLASH_ATTN_EXT_VEC_NCPSG,
    short NSG_PARAM = 1, short NWG_PARAM = 32>
void dwarfstar_flash_vec_impl(
        const device ds4_metal_args_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        threadgroup  half * shmem_f16 ,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");

#define NWG  (FC_flash_attn_ext_vec_nwg)
#define NSG  (FC_flash_attn_ext_vec_nsg)
#define NS10 (FC_flash_attn_ext_vec_ns10)
#define NS20 (FC_flash_attn_ext_vec_ns20)

    const short iwg = tgpig[2]%NWG;

    const ushort iq3 = tgpig[2]/NWG;
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0];

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;
    constexpr short PK  = PAD2(DK, 128);
    constexpr short PK4 = PK/4;
    constexpr short PV  = PAD2(DV, 128);
    constexpr short PV4 = PV/4;
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NL  = NW/NE;
    constexpr short SH  = 4*C;

    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

    threadgroup q4_t  * sq4 = (threadgroup q4_t  *) (shmem_f16 +                      0*PK);
    threadgroup s_t   * ss  = (threadgroup s_t   *) (shmem_f16 +   sgitg*SH       + NSG*PK);
    threadgroup s4_t  * ss4 = (threadgroup s4_t  *) (shmem_f16 +   sgitg*SH       + NSG*PK);
    threadgroup half  * sm  = (threadgroup half  *) (shmem_f16 +   sgitg*SH + 2*C + NSG*PK);
    threadgroup o4_t  * so4 = (threadgroup o4_t  *) (shmem_f16 + 2*sgitg*PV       + NSG*PK + NSG*SH);

    so4 += tiisg;

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    device const float4 * q4 = (device const float4 *) ((device const char *) q);

    if (iq1 < args.ne01) {
        for (short i = tiisg; i < PK4; i += NW) {
            if (i < DK4) {
                sq4[i] = (q4_t) q4[i];
            } else {
                sq4[i] = (q4_t) 0.0f;
            }
        }
    }

    for (short i = 0; i < DV4/NL; ++i) {
        so4[i*NL] = (o4_t) 0.0f;
    }

    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = (s4_t) 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S = 0.0f;
        float M = -FLT_MAX/2;

        const short tx = tiisg%NL;
        const short ty = tiisg/NL;

        device const half * pm = (device const half *) (mask + iq1*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);

        float slope = 1.0f;

        if (FC_flash_attn_ext_vec_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                if (FC_flash_attn_ext_vec_shared_kvpad) {
                    v = k;
                    mask = k + args.nb11*C*args.ne_12_2*args.ne_12_3 +
                        args.nb21*C*args.ne_12_2*args.ne_12_3;
                } else {
                    v = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                    mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;
                }

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_vec_has_mask) {
                    if (ic + tiisg >= args.ne11) {
                        sm[tiisg] = -MAXHALF;
                    }
                } else {
                    pm = (device const half *) (mask) +
                        iq1*C +
                        (iq2%args.ne32)*(C*args.ne31) +
                        (iq3%args.ne33)*(C*args.ne31*args.ne32);
                }

                ic = 0;
            }

            if (FC_flash_attn_ext_vec_has_mask) {
                sm[tiisg] = pm[ic + tiisg];
            }

            if (simd_max(sm[tiisg]) <= -MAXHALF) {
                continue;
            }

            {
                device      const k4_t * pk4 = (device const k4_t *) (k + ic*args.nb11);
                threadgroup const q4_t * pq4 = sq4;

                pk4 += ty*NS10/4 + tx;
                pq4 += tx;

                qk_t mqk[C/NE] = { [ 0 ... C/NE - 1] = 0.0f };

                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                    if (is_same<kd4_t, k4_t>::value) {
                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            mqk[cc] += dot((float4) pk4[cc*NE*NS10/4 +  ii*NL], (float4) pq4[ii*NL]);
                        }
                    } else {
                        device const kd4_t * pk = (device const kd4_t *) (k + ((ic + NE*cc + ty)*args.nb11));

                        k4_t mk;

                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            deq_k_t4(pk + i/nl_k, i%nl_k, mk);

                            mqk[cc] += dot((float4) mk, (float4) sq4[i]);
                        }
                    }

                    if (NE == 1) {
                        mqk[cc] = simd_sum(mqk[cc]);
                    } else {
                        if (NE <= 1) {
                            mqk[cc] += simd_shuffle_down(mqk[cc], 16);
                        }
                        if (NE <= 2) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  8);
                        }
                        if (NE <= 4) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  4);
                        }
                        if (NE <= 8) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  2);
                        }
                        if (NE <= 16) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  1);
                        }

                        mqk[cc] = simd_shuffle(mqk[cc], NL*ty);
                    }
                }

                if (FC_flash_attn_ext_vec_has_mask &&
                   !FC_flash_attn_ext_vec_has_scap &&
                   !FC_flash_attn_ext_vec_has_bias) {
                    ss[NE*tx + ty] = fma(mqk[tx], args.scale, (qk_t) sm[NE*tx + ty]);
                } else {
                    mqk[tx] *= args.scale;

                    if (FC_flash_attn_ext_vec_has_scap) {
                        mqk[tx] = args.logit_softcap*precise::tanh(mqk[tx]);
                    }

                    if (FC_flash_attn_ext_vec_has_bias) {
                        mqk[tx] += (qk_t) sm[NE*tx + ty]*slope;
                    } else {
                        mqk[tx] += (qk_t) sm[NE*tx + ty];
                    }

                    ss[NE*tx + ty] = mqk[tx];
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            {
                const float m = M;
                const float s = ss[tiisg];

                M = simd_max(max(M, s));

                const float ms = exp(m - M);
                const float vs = exp(s - M);

                S = S*ms + simd_sum(vs);

                ss[tiisg] = vs;

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[ii*NL] *= ms;
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            {
                o4_t lo[DV4/NL];
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    lo[ii] = 0.0f;
                }

                if (is_same<vd4_t, v4_t>::value) {
                    device const v4_t * pv4 = (device const v4_t *) (v + ic*args.nb21);

                    pv4 += ty*NS20/4 + tx;

                    const auto sst = ss + ty;

                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            lo[ii] += o4_t(float4(pv4[cc*NE*NS20/4 + ii*NL])*float4(sst[cc*NE]));
                        }
                    }
                } else {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        device const vd4_t * pv4 = (device const vd4_t *) (v + ((ic + NE*cc + ty)*args.nb21));

                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            v4_t mv;
                            deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);

                            lo[ii] += o4_t(float4(mv)*float4(ss[NE*cc + ty]));
                        }
                    }
                }

                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    if (NE > 1) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0], 16);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1], 16);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2], 16);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3], 16);
                    }

                    if (NE > 2) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  8);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  8);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  8);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  8);
                    }

                    if (NE > 4) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  4);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  4);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  4);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  4);
                    }

                    if (NE > 8) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  2);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  2);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  2);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  2);
                    }

                    if (NE > 16) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  1);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  1);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  1);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  1);
                    }
                }

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[ii*NL] += lo[ii];
                    }
                }
            }
        }

        if (FC_flash_attn_ext_vec_has_sinks && sgitg == 0 && iwg == 0) {
            const float m = M;
            const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;

            M = simd_max(max(M, s));

            const float ms = exp(m - M);
            const float vs = exp(s - M);

            S = S*ms + simd_sum(vs);

            if ((DV4/NL % NW == 0) || ty == 0) {
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    so4[ii*NL] *= ms;
                }
            }
        }

        if (tiisg == 0) {
            ss[0] = (s_t) S;
            ss[1] = (s_t) M;
        }
    }

    so4 -= tiisg;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short r = NSG/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            const float S0 = ss[           0];
            const float S1 = ss[r*(SH/2) + 0];

            const float M0 = ss[           1];
            const float M1 = ss[r*(SH/2) + 1];

            const float M = max(M0, M1);

            const float ms0 = exp(M0 - M);
            const float ms1 = exp(M1 - M);

            const float S = S0*ms0 + S1*ms1;

            if (tiisg == 0) {
                ss[0] = S;
                ss[1] = M;
            }

            for (short i = tiisg; i < DV4; i += NW) {
                so4[i] = so4[i]*ms0 + so4[i + r*PV4]*ms1;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0) {
        const int64_t nrows = args.ne3*args.ne2*args.ne1;
        const int64_t rid   = iq3*args.ne2*args.ne1 + iq2 + iq1*args.ne1;

        device float4 * dst4 = (device float4 *) dst;
        device float  * dst1 = (device float  *) dst + nrows*DV*NWG;

        const float S = NWG == 1 ? (ss[0] == 0.0f ? 0.0f : 1.0f/ss[0]) : 1.0f;

        for (short i = tiisg; i < DV4; i += NW) {
            dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[i]*S;
        }

        if (NWG > 1) {
            if (tiisg == 0) {
                dst1[rid*(2*NWG) + 2*iwg + 0] = ss[0];
                dst1[rid*(2*NWG) + 2*iwg + 1] = ss[1];
            }
        }
    }

#undef NWG
#undef NSG
#undef NS10
#undef NS20
}

static __attribute__((noinline)) void ds4_flash_attn_vec_reduce_row(
        constant ds4_metal_args_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        uint   tgpig,
        ushort tiisg,
        ushort sgitg,
        short  NWG_,
        short  DV_) {
    const uint64_t rid = tgpig;

    const short iwg = tiisg;

    device const float  * ss    = (device const float  *) htmp + (uint64_t)args.nrows*DV_*NWG_;

    float S = ss[rid*(2*NWG_) + 2*iwg + 0];
    float M = ss[rid*(2*NWG_) + 2*iwg + 1];

    const float m  = simd_max(M);
    const float ms = exp(M - m);

    S = simd_sum(S*ms);
    S = S == 0.0f ? 0.0f : 1.0f/S;

    const short DV4 = DV_/4;

    device const float4 * htmp4 = (device const float4 *) htmp + rid*DV4*NWG_;
    device       float4 * dst4  = (device       float4 *) dst  + rid*DV4;

    for (short i = sgitg; i < DV4; i += NWG_) {
        const float4 v = simd_sum(htmp4[i*NWG_ + iwg]*ms);

        if (iwg == 0) {
            dst4[i] = v*S;
        }
    }
}

}

"""

ARG_FIELDS = (('ne01', '<i4'), ('ne02', '<i4'), ('ne03', '<i4'), ('nb01', '<u8'), ('nb02', '<u8'), ('nb03', '<u8'), ('ne11', '<i4'), ('ne_12_2', '<i4'), ('ne_12_3', '<i4'), ('ns10', '<i4'), ('nb11', '<u8'), ('nb12', '<u8'), ('nb13', '<u8'), ('ns20', '<i4'), ('nb21', '<u8'), ('nb22', '<u8'), ('nb23', '<u8'), ('ne31', '<i4'), ('ne32', '<i4'), ('ne33', '<i4'), ('nb31', '<u8'), ('nb32', '<u8'), ('nb33', '<u8'), ('ne1', '<i4'), ('ne2', '<i4'), ('ne3', '<i4'), ('scale', '<f4'), ('max_bias', '<f4'), ('m0', '<f4'), ('m1', '<f4'), ('n_head_log2', '<i4'), ('logit_softcap', '<f4'))


def plan(length,local_rows,pool_rows):
    if any(type(x)is not int for x in (length,local_rows,pool_rows))or length not in (1,99):
        raise ValueError('DwarfStar component admits actual M1/M99 only')
    if not 1<=local_rows<=128 or not 1<=pool_rows<=512:
        raise ValueError('DwarfStar local/global bounds')
    vector=length==1;block=32 if vector else 64
    keys=local_rows+pool_rows;padded=(keys+block-1)//block*block
    nsg=1 if vector else 8;nwg=32 if vector else 1
    shared=(nsg*(512+4*32+2*512)*2)if vector else(8*(512+2*512+4*64)*2)
    groups=(length,64,nwg)if vector else((length+7)//8,64,1)
    tg=(32,nsg,1)
    return dict(numerical_profile='dwarfstar_f16_flash_attention_v1',length=length,local_rows=local_rows,pool_rows=pool_rows,
        query_heads=64,KV_heads=1,dim=512,key_count=keys,padded_keys=padded,key_block=block,
        mode='vector32_reduce'if vector else'nonvec_Q8_C64',nsg=nsg,nwg=nwg,threadgroup=tg,threadgroups=groups,
        grid=tuple(a*b for a,b in zip(groups,tg)),threadgroup_bytes=shared,
        partial_floats=length*64*(512*nwg+2*nwg)if vector else 0,
        reduce_threadgroup=(1024,1,1)if vector else None,reduce_grid=(length*64*1024,1,1)if vector else None,
        output_shape=(1,length,64,512),staged_KV_bytes=padded*512*2,mask_bytes=length*padded*2,
        shared_physical_KV_bank=True,original_weight_quantization_unchanged=True,cache_layout_unchanged=True,
        numerical_scope='OriginalDwarfStarF16QKV/fastmath/postdotscale/mask+sinks; notSourcebitexact',
        traffic_scope='Stagingpayloads/logicalbuffers only; not measuredDRAM')


def args_payload(np,p):
    dtype=np.dtype(list(ARG_FIELDS),align=True);row=np.zeros(1,dtype=dtype)
    length,n=p['length'],p['padded_keys'];f32row=512*4;f16row=512*2
    values=dict(ne01=length,ne02=64,ne03=1,nb01=f32row,nb02=length*f32row,nb03=length*64*f32row,
        ne11=n,ne_12_2=1,ne_12_3=1,ns10=512,nb11=f16row,nb12=n*f16row,nb13=n*f16row,
        ns20=512,nb21=f16row,nb22=n*f16row,nb23=n*f16row,ne31=length,ne32=1,ne33=1,
        nb31=n*2,nb32=length*n*2,nb33=length*n*2,ne1=64,ne2=length,ne3=1,
        scale=np.float32(1)/np.sqrt(np.float32(512)),max_bias=0,m0=0,m1=0,n_head_log2=0,logit_softcap=0)
    for key,value in values.items():row[key]=value
    return row.view(np.uint8).copy()


def resolved_prepare_mode(mode,length):
    if type(mode)is not str or mode not in ('reference','fused_decode'):
        raise ValueError('Unknown DwarfStar preparation mode')
    return 'fused_decode'if mode=='fused_decode'and length==1 else'reference'


def prepare(q,local,local_mask,pooled,indices,valid,sinks,*,trusted_all_retained=False,prepare_mode='reference'):
    import mlx.core as mx
    import numpy as np
    resolved_prepare_mode(prepare_mode,1) # Refuse malformed policy before work.
    if trusted_all_retained is not True:raise ValueError('DwarfStar component needs verified chronological allretained IDs')
    if q.ndim!=4 or q.shape[:2]!=(1,64)or q.shape[-1]!=512:
        raise ValueError('DwarfStar requires B1/H64/D512')
    if local.ndim!=4 or local.shape[:2]!=(1,1)or local.shape[-1]!=512 or pooled.ndim!=3 or pooled.shape[0]!=1 or pooled.shape[-1]!=512:
        raise ValueError('DwarfStar sharedKV geometry')
    p=plan(q.shape[2],local.shape[2],pooled.shape[1])
    if (indices.shape!=(1,p['length'],p['pool_rows'])or valid.shape!=indices.shape
            or local_mask.shape!=(1,1,p['length'],p['local_rows'])or sinks.shape!=(64,)
            or indices.dtype not in (mx.int32,mx.uint32)
            or valid.dtype!=mx.bool_ or local_mask.dtype!=mx.bool_
            or any(v.dtype!=mx.float32 for v in (q,local,pooled,sinks))):
        raise ValueError('DwarfStar mask/dtype geometry')
    if mx.default_device()!=mx.gpu or not mx.metal.is_available():
        raise ValueError('DwarfStar adaptation requires caller-owned Metal GPU')
    # Same chronological F16 staging boundary as DwarfStar's host. The padded
    # bank replaces its separate last-block pad allocation, not its arithmetic.
    actual_mode=resolved_prepare_mode(prepare_mode,p['length']);fused=actual_mode=='fused_decode'
    if fused:
        from .deepseek_v41_dwarfstar_staging import stage
        kv,mask,blocks=stage(local,pooled,local_mask,valid)
    else:
        kv=mx.concatenate([local[0,0],pooled[0],mx.zeros((p['padded_keys']-p['key_count'],512),mx.float32)],0).astype(mx.float16)
        visible=mx.concatenate([local_mask[0,0],valid[0],mx.zeros((p['length'],p['padded_keys']-p['key_count']),mx.bool_)],1)
        mask=mx.where(visible,mx.array(0,mx.float16),mx.array(-65504,mx.float16))
        if p['mode'].startswith('nonvec'):
            padded_q=(p['length']+7)//8*8
            v=mx.concatenate([visible,mx.zeros((padded_q-p['length'],p['padded_keys']),mx.bool_)],0)
            v=v.reshape(padded_q//8,8,p['padded_keys']//64,64)
            any_live=mx.any(v,axis=(1,3));all_live=mx.all(v,axis=(1,3))
            blocks=mx.where(all_live,2,mx.where(any_live,1,0)).astype(mx.int8)
        else:blocks=mx.zeros((1,),mx.int8)
    params_host=args_payload(np,p);params=mx.array(params_host,mx.uint8)
    return SimpleNamespace(q=q,kv=kv,mask=mask,blocks=blocks,sinks=sinks,params=params,params_host=params_host,plan=p,
        requested_prepare_mode=prepare_mode,prepare_mode=actual_mode)


def kernel_source(kind):
    if kind=='nonvec':
        return r'''
threadgroup half scratch[14336];
mlx2_dwarfstar::kernel_flash_attn_ext_impl<
half,half4,simdgroup_half8x8,half,half4x4,simdgroup_half8x8,
half,half4x4,simdgroup_half8x8,float,simdgroup_float8x8,
float,float2,simdgroup_float8x8,float,float4,simdgroup_float8x8,
half4x4,1,mlx2_dwarfstar::dequantize_f16,half4x4,1,mlx2_dwarfstar::dequantize_f16,
512,512,8,64,8>(
*((const device mlx2_dwarfstar::ds4_metal_args_flash_attn_ext*)params),
(const device char*)q,(const device char*)kv,(const device char*)kv,
(const device char*)mask,(const device char*)sinks,(const device char*)kv,
(const device char*)blocks,(device char*)out,scratch,threadgroup_position_in_grid,
thread_index_in_simdgroup,simdgroup_index_in_threadgroup);
'''
    if kind=='vector':
        return r'''
threadgroup half scratch[1664];
mlx2_dwarfstar::dwarfstar_flash_vec_impl<
half4,half4,half4,float,float,float4,float4,
half4,1,mlx2_dwarfstar::dequantize_f16_t4,half4,1,mlx2_dwarfstar::dequantize_f16_t4,
512,512,1,1,32,1,32>(
*((const device mlx2_dwarfstar::ds4_metal_args_flash_attn_ext_vec*)params),
(const device char*)q,(const device char*)kv,(const device char*)kv,
(const device char*)mask,(const device char*)sinks,(const device char*)kv,
(device char*)out,scratch,threadgroup_position_in_grid,
thread_index_in_simdgroup,simdgroup_index_in_threadgroup);
'''
    if kind=='reduce':
        return r'''
mlx2_dwarfstar::ds4_flash_attn_vec_reduce_row(
*((constant mlx2_dwarfstar::ds4_metal_args_flash_attn_ext_vec_reduce*)params),
(const device char*)partials,(device char*)out,
threadgroup_position_in_grid.x,thread_index_in_simdgroup,simdgroup_index_in_threadgroup,32,512);
'''
    raise ValueError('Unknown DwarfStar stage')


@lru_cache(maxsize=3)
def _kernel(kind):
    import mlx.core as mx
    names=['params','partials']if kind=='reduce'else['params','q','kv','mask','sinks','blocks']
    return mx.fast.metal_kernel(name='mlx2_dwarfstar_attention_'+kind+'_v1',input_names=names,output_names=['out'],
        header=HEADER,source=kernel_source(kind),ensure_row_contiguous=True,compile_options={'math_mode':'fast'})


def attention(prepared):
    import mlx.core as mx
    if mx.default_device()!=mx.gpu:raise ValueError('DwarfStar attention requires caller-owned Metal GPU')
    p=prepared.plan;kind='vector'if p['mode'].startswith('vector')else'nonvec'
    if p['threadgroup_bytes']>32768:raise ValueError('DwarfStar source exceeds reviewed sharedmemory bound')
    output=_kernel(kind)(inputs=[prepared.params,prepared.q,prepared.kv,prepared.mask,prepared.sinks,prepared.blocks],
        grid=p['grid'],threadgroup=p['threadgroup'],output_shapes=[(p['partial_floats'],)if kind=='vector'else p['output_shape']],output_dtypes=[mx.float32])[0]
    if kind=='vector':
        params=mx.array([p['length']*64],mx.int32)
        output=_kernel('reduce')(inputs=[params,output],grid=p['reduce_grid'],threadgroup=p['reduce_threadgroup'],
            output_shapes=[p['output_shape']],output_dtypes=[mx.float32])[0]
    return output.transpose(0,2,1,3)


def compilation_source(kind):
    if kind not in ('nonvec','vector','reduce'):raise ValueError('Unknown offlineDwarfStar stage')
    # MLX0.32.4 lifts this one-element integer input to constant address space.
    # Match the actual generated signature, not a fictitious device-pointer ABI.
    if kind=='reduce':args='const constant int32_t* params [[buffer(0)]],const device float* partials [[buffer(1)]],device float* out [[buffer(2)]],'
    else:args='const device uchar* params [[buffer(0)]],const device float* q [[buffer(1)]],const device half* kv [[buffer(2)]],const device half* mask [[buffer(3)]],const device float* sinks [[buffer(4)]],const device char* blocks [[buffer(5)]],device float* out [[buffer(6)]],'
    return HEADER+'\nkernel void dwarfstar_probe('+args+'uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],ushort thread_index_in_simdgroup [[thread_index_in_simdgroup]],ushort simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]){\n'+kernel_source(kind)+'\n}\n'
