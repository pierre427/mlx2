# SPDX-License-Identifier: Apache-2.0
"""Unwired Source CSA2 attention numerical profile; caller owns GPU/state.

Adapted oMLX PR4808, Apache-2.0; see provenance/omlx-4808-attention.NOTICE.
BF16 operand staging, NAX products and online softmax differ from Source F32
arithmetic. No Source policy/default/cache integration is performed here.
"""
from dataclasses import dataclass
from functools import lru_cache

PROFILE='source_csa2_omlx_nax_bf16_online_v1'
_UPSTREAM_SHA256='7fa54dee05e31338a756820fd97817d0f78c773b0134766b76c44db95c21685a'
_HEADER=r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""
_SOURCE=r"""
    constexpr int D = 512;
    constexpr int BK = 128;
    const int NS = params[0];
    const int NP = params[1];
    const int L = params[2];
    const int K = params[3];
    const int W = NS;
    const int SLOTS = NS + K;
    const float scale = scalep[0];
    const int qi = int(threadgroup_position_in_grid.x);
    const int hh = int(threadgroup_position_in_grid.y);
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = sg * 32 + lane;
    const int hg = int(sg) / 4;
    const int j4 = int(sg) % 4;   // QK + softmax: slot quarter; PV: dim quarter
    const int head0 = hh * 32 + hg * 16;

    // Source owns visibility and CSA2 row selection; no ratio4 inference here.
    const short qid = short(lane >> 2);
    const short fm = short((qid & 4) | ((lane >> 1) & 3));
    const short fn = short(((qid & 2) | (lane & 1)) * 4);
    // Lanes with fn == 0: one per row (fm) of a fragment after the
    // xor-1 / xor-8 row reductions.
    const bool row_writer = (lane & 9u) == 0u;

    threadgroup int sel[2][BK];
    threadgroup int live[2][BK / 32];
    threadgroup float red_max[2][4][16];
    threadgroup float red_sum[2][4][16];
    // P pieces per (head group, 16-slot step, lane): the 8 values of the
    // lane's fragment slots (rows fm, fm + 8 x slots fn .. fn + 3).
    threadgroup half p_hi[2][BK / 16][32 * 8];
    threadgroup half p_lo[2][BK / 16][32 * 8];

    constexpr auto qk_desc = matmul2d_descriptor(
        16, 32, 16, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroup> qk_op;
    constexpr auto pv_desc = matmul2d_descriptor(
        16, 32, 16, false, false, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroup> pv_op;

    auto pa = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
    auto pm = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
    auto pb = pv_op.template get_right_input_cooperative_tensor<half, T, float>();
    auto o0 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o1 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o2 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o3 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    for (short e = 0; e < 16; ++e) {
        o0[e] = 0.0f;
        o1[e] = 0.0f;
        o2[e] = 0.0f;
        o3[e] = 0.0f;
    }
    float m_run[2] = {-INFINITY, -INFINITY};
    float l_run[2] = {0.0f, 0.0f};

    const device T* qr0 = q + (ulong(head0 + fm) * L + qi) * D + fn;
    const device T* qr1 = q + (ulong(head0 + fm + 8) * L + qi) * D + fn;

    // Slot s of the query: window row s for s < W, then top-k entry s - W.
    auto stage = [&](int t, int b) {
        if (tid < uint(BK)) {
            const int slot = t * BK + int(tid);
            int row = -1;
            if (slot < NS) {
                row = local_mask[ulong(qi) * NS + slot] ? slot : -1;
            } else if (slot < SLOTS) {
                const ulong pos = ulong(qi) * K + slot - NS;
                const uint idx = topk[pos];
                row = selected_valid[pos] && idx < uint(NP) ? int(idx) : -1;
            }
            sel[b][tid] = row;
            const bool any_live = simd_any(row >= 0);
            if (lane == 0) {
                live[b][sg] = any_live ? 1 : 0;
            }
        }
    };

    const int n_tiles = (SLOTS + BK - 1) / BK;
    stage(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int t = 0; t < n_tiles; ++t) {
        const int buf = t & 1;
        const int tile_slots = min(BK, SLOTS - t * BK);
        // A tile without a visible slot leaves the running state unchanged
        // (uniform across the threadgroup).
        if ((live[buf][0] | live[buf][1] | live[buf][2] | live[buf][3]) == 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (t + 1 < n_tiles) {
                stage(t + 1, buf ^ 1);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        // ---- S = Q K^T for slot quarter j4 (16 heads x 32 slots).
        auto qa = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
        auto kb = qk_op.template get_right_input_cooperative_tensor<T, T, float>();
        auto sc = qk_op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(qa)>, metal::remove_addrspace_t<decltype(kb)>, float>();
        for (short e = 0; e < 16; ++e) {
            sc[e] = 0.0f;
        }
        if (j4 * 32 < tile_slots) {
            const device T* kr[2][2];
            for (short tn = 0; tn < 2; ++tn) {
                for (short i = 0; i < 2; ++i) {
                    const int k = j4 * 32 + tn * 16 + fm + i * 8;
                    const int row = sel[buf][k];
                    const device T* src = t * BK + k < W ? kv : pooled;
                    kr[tn][i] = src + ulong(max(row, 0)) * D + fn;
                }
            }
            _Pragma("clang loop unroll_count(4)")
            for (short kk = 0; kk < D; kk += 16) {
                for (short j = 0; j < 4; ++j) {
                    qa[j] = qr0[kk + j];
                    qa[4 + j] = qr1[kk + j];
                }
                for (short tn = 0; tn < 2; ++tn) {
                    for (short i = 0; i < 2; ++i) {
                        for (short j = 0; j < 4; ++j) {
                            kb[tn * 8 + i * 4 + j] = kr[tn][i][kk + j];
                        }
                    }
                }
                qk_op.run(qa, kb, sc);
            }
        }
        bool ok[2][4];
        for (short tn = 0; tn < 2; ++tn) {
            for (short j = 0; j < 4; ++j) {
                ok[tn][j] = sel[buf][j4 * 32 + tn * 16 + fn + j] >= 0;
            }
        }
        // Partial row max of the raw scores over the quarter (scale > 0 and
        // rounding is monotonic: max(s) * c == max(s * c) bitwise).
        float pmx[2];
        for (short i = 0; i < 2; ++i) {
            float m = -INFINITY;
            for (short tn = 0; tn < 2; ++tn) {
                for (short j = 0; j < 4; ++j) {
                    m = ok[tn][j] ? max(m, float(sc[tn * 8 + i * 4 + j])) : m;
                }
            }
            m = max(m, simd_shuffle_xor(m, ushort(1)));
            m = max(m, simd_shuffle_xor(m, ushort(8)));
            pmx[i] = m;
        }
        if (row_writer) {
            red_max[hg][j4][fm] = pmx[0];
            red_max[hg][j4][fm + 8] = pmx[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t + 1 < n_tiles) {
            stage(t + 1, buf ^ 1);
        }

        // ---- Tile row max (the same value in the 4 simdgroups of the group).
        // The tile has a visible slot, so it is finite.
        float factor[2];
        float mnew[2];
        for (short i = 0; i < 2; ++i) {
            const int r = fm + i * 8;
            const float m = max(max(red_max[hg][0][r], red_max[hg][1][r]),
                                max(red_max[hg][2][r], red_max[hg][3][r]));
            mnew[i] = max(m_run[i], m * scale);
            factor[i] = m_run[i] == -INFINITY ? 0.0f : exp(m_run[i] - mnew[i]);
            m_run[i] = mnew[i];
        }
        // ---- P of this quarter, once: exp, fp16 hi + lo pieces, row sums.
        float rs[2] = {0.0f, 0.0f};
        for (short tn = 0; tn < 2; ++tn) {
            vec<half, 8> h, lo;
            for (short i = 0; i < 2; ++i) {
                for (short j = 0; j < 4; ++j) {
                    const float e = ok[tn][j]
                        ? exp(float(sc[tn * 8 + i * 4 + j]) * scale - mnew[i])
                        : 0.0f;
                    const half hi = half(e);
                    h[i * 4 + j] = hi;
                    lo[i * 4 + j] = half(e - float(hi));
                    rs[i] += e;
                }
            }
            const int ks = j4 * 2 + tn;
            *(threadgroup vec<half, 8>*)(&p_hi[hg][ks][lane * 8]) = h;
            *(threadgroup vec<half, 8>*)(&p_lo[hg][ks][lane * 8]) = lo;
        }
        for (short i = 0; i < 2; ++i) {
            rs[i] += simd_shuffle_xor(rs[i], ushort(1));
            rs[i] += simd_shuffle_xor(rs[i], ushort(8));
        }
        if (row_writer) {
            red_sum[hg][j4][fm] = rs[0];
            red_sum[hg][j4][fm + 8] = rs[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- O = O * factor + P V for dim quarter j4 over the whole tile.
        // (factor == 1 exactly for every row of the simdgroup: O * 1 == O.)
        if (!simd_all(factor[0] == 1.0f && factor[1] == 1.0f)) {
            for (short e = 0; e < 16; ++e) {
                const float f = factor[(e >> 2) & 1];
                o0[e] *= f;
                o1[e] *= f;
                o2[e] *= f;
                o3[e] *= f;
            }
        }
        for (short i = 0; i < 2; ++i) {
            const int r = fm + i * 8;
            const float tsum = ((red_sum[hg][0][r] + red_sum[hg][1][r]) + red_sum[hg][2][r]) + red_sum[hg][3][r];
            l_run[i] = l_run[i] * factor[i] + tsum;
        }
        const int n_ks = (tile_slots + 15) / 16;
        for (short ks = 0; ks < n_ks; ++ks) {
            const vec<half, 8> h = *(const threadgroup vec<half, 8>*)(&p_hi[hg][ks][lane * 8]);
            const vec<half, 8> lo = *(const threadgroup vec<half, 8>*)(&p_lo[hg][ks][lane * 8]);
            for (short e = 0; e < 8; ++e) {
                pa[e] = h[e];
                pm[e] = lo[e];
            }
            const int k0 = ks * 16 + fm;
            const int k1 = ks * 16 + fm + 8;
            const device T* v0 = (t * BK + k0 < W ? kv : pooled)
                + ulong(max(sel[buf][k0], 0)) * D + j4 * 128 + fn;
            const device T* v1 = (t * BK + k1 < W ? kv : pooled)
                + ulong(max(sel[buf][k1], 0)) * D + j4 * 128 + fn;
            for (short np = 0; np < 4; ++np) {
                for (short tn = 0; tn < 2; ++tn) {
                    for (short j = 0; j < 4; ++j) {
                        pb[tn * 8 + j] = v0[np * 32 + tn * 16 + j];
                        pb[tn * 8 + 4 + j] = v1[np * 32 + tn * 16 + j];
                    }
                }
                if (np == 0) {
                    pv_op.run(pa, pb, o0);
                    pv_op.run(pm, pb, o0);
                } else if (np == 1) {
                    pv_op.run(pa, pb, o1);
                    pv_op.run(pm, pb, o1);
                } else if (np == 2) {
                    pv_op.run(pa, pb, o2);
                    pv_op.run(pm, pb, o2);
                } else {
                    pv_op.run(pa, pb, o3);
                    pv_op.run(pm, pb, o3);
                }
            }
        }
    }

    for (short i = 0; i < 2; ++i) {
        const int head = head0 + fm + i * 8;
        // With no visible slot m_run is -inf: the sink term is +inf and the
        // row is zero, as in the native kernel.
        const float denom = l_run[i] + exp(float(sinks[head]) - m_run[i]);
        const float inv = denom == 0.0f ? 0.0f : 1.0f / denom;
        device float* orow = out + (ulong(head) * L + qi) * D + j4 * 128 + fn;
        for (short tn = 0; tn < 2; ++tn) {
            for (short j = 0; j < 4; ++j) {
                const short e = tn * 8 + i * 4 + j;
                orow[tn * 16 + j] = o0[e] * inv;
                orow[32 + tn * 16 + j] = o1[e] * inv;
                orow[64 + tn * 16 + j] = o2[e] * inv;
                orow[96 + tn * 16 + j] = o3[e] * inv;
            }
        }
    }
"""


def plan(length,local_count,pool_count,selected_count,*,numerical_profile):
    if numerical_profile!=PROFILE:
        raise ValueError('Explicit Source oMLX attention numerical profile required')
    if (any(type(v)is not int for v in (length,local_count,pool_count,selected_count))
            or not 16<=length<=128 or not 1<=local_count<=255
            or not 1<=pool_count<=4096 or not 1<=selected_count<=min(512,pool_count)):
        raise ValueError('Source NAX attention admits bounded PP16..128, H64/D512 only')
    slots=local_count+selected_count
    return dict(length=length,heads=64,dim=512,local_count=local_count,pool_count=pool_count,
        selected_count=selected_count,slot_tiles=(slots+127)//128,
        grid=(length*256,2,1),threadgroup=(256,1,1),
        logical_staging_bytes=2*(64*length*512+local_count*512+pool_count*512)+5*length*selected_count+length*local_count,
        output_bytes=4*64*length*512,threadgroup_bytes=2*128*4+2*4*4+2*4*16*4*2+2*2*8*32*8*2,
        profile=PROFILE,weight_requantization=False,persistent_cache_mutation=False)


@dataclass(frozen=True)
class Prepared:
    inputs:tuple
    geometry:dict
    numerical_profile:str


def prepare(q,local,local_mask,pooled,indices,valid,sinks,*,numerical_profile):
    """Build BF16 contiguous staging, never evaluate/read back caller arrays.

    The bounded replay harness verifies finite inputs and BF16 roundtrip loss.
    Explicit Source masks select slots; no ratio/position/indexer assumptions.
    Original q/cache inputs are retained by the caller and never modified.
    """
    import mlx.core as mx
    if (q.ndim!=4 or q.shape[:2]!=(1,64)or q.shape[-1]!=512 or q.dtype!=mx.float32
            or local.ndim!=4 or local.shape[:2]!=(1,1)or local.shape[-1]!=512 or local.dtype!=mx.float32
            or pooled.ndim!=3 or pooled.shape[0]!=1 or pooled.shape[-1]!=512 or pooled.dtype!=mx.float32
            or indices.ndim!=3 or indices.dtype not in (mx.uint32,mx.int32)):
        raise ValueError('Source NAX attention requires F32 B1 H64 D512 geometry')
    p=plan(q.shape[2],local.shape[2],pooled.shape[1],indices.shape[-1],numerical_profile=numerical_profile)
    if (indices.shape!=(1,p['length'],p['selected_count'])or valid.shape!=indices.shape or valid.dtype!=mx.bool_
            or local_mask.shape!=(1,1,p['length'],p['local_count'])or local_mask.dtype!=mx.bool_
            or sinks.shape!=(64,)or sinks.dtype!=mx.float32):
        raise ValueError('Source NAX attention mask/index/sink geometry')
    inputs=(mx.contiguous(q[0].astype(mx.bfloat16)),mx.contiguous(local[0,0].astype(mx.bfloat16)),
        mx.contiguous(pooled[0].astype(mx.bfloat16)),mx.contiguous(indices[0].astype(mx.uint32)),
        mx.contiguous(local_mask[0,0]),mx.contiguous(valid[0]),mx.contiguous(sinks),
        mx.array([p['local_count'],p['pool_count'],p['length'],p['selected_count']],mx.int32),
        mx.array([512**-.5],mx.float32))
    return Prepared(inputs,p,PROFILE)


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx
    return mx.fast.metal_kernel(name='mlx2_source_omlx_nax_attention',
        input_names=['q','kv','pooled','topk','local_mask','selected_valid','sinks','params','scalep'],
        output_names=['out'],header=_HEADER,source=_SOURCE,ensure_row_contiguous=True)


def prepared_attention(prepared,*,numerical_profile):
    import mlx.core as mx
    from ..gguf.source_dense_legacy import _device_supported
    if numerical_profile!=PROFILE or not isinstance(prepared,Prepared)or prepared.numerical_profile!=PROFILE:
        raise ValueError('Source NAX attention prepared/profile mismatch')
    if not _device_supported():raise ValueError('Source NAX attention requires caller-owned M5 Max TF32-off GPU')
    p=prepared.geometry
    result=_kernel()(inputs=list(prepared.inputs),template=[('T',mx.bfloat16)],
        grid=p['grid'],threadgroup=p['threadgroup'],output_shapes=[(64,p['length'],512)],output_dtypes=[mx.float32])[0]
    return result[None]


def attention(q,local,local_mask,pooled,indices,valid,sinks,*,numerical_profile):
    return prepared_attention(prepare(q,local,local_mask,pooled,indices,valid,sinks,
        numerical_profile=numerical_profile),numerical_profile=numerical_profile)


def compilation_source():
    buffers=['const device bfloat* q','const device bfloat* kv','const device bfloat* pooled',
        'const device uint* topk','const device bool* local_mask','const device bool* selected_valid',
        'const device float* sinks','const device int* params','const device float* scalep','device float* out']
    args=',\n'.join(f'{v} [[buffer({i})]]'for i,v in enumerate(buffers))
    return ('#include "mlx/backend/metal/kernels/utils.h"\n'+_HEADER+'\nusing T=bfloat;\n'
        +'kernel void source_attention_probe('+args+''',
uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
'''+_SOURCE+'\n}\n')
