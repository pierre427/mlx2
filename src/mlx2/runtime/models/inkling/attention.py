"""Inkling grouped-query attention with inline learned relative bias.

Internal mining: qsdpa_decode_metal.py's FP split-sequence online softmax,
mlx2 b33f9710e7ff0e820047927e5f7c02311ea35848, Apache-2.0. Query tiling,
relative bias, causal/window/padding semantics and runtime merge are original.
No score/bias matrix is materialized. Explicit candidate, reference retained.
"""

from __future__ import annotations

import mlx.core as mx

_SOURCE = r"""
    constexpr int D = 128, NS = 4, J = G * QT, EPL = 4;
    threadgroup float qs[J * D];
    threadgroup float rs[J * R];
    threadgroup float ps[NS * J * 32];
    const int lane = thread_index_in_simdgroup;
    const int sg = simdgroup_index_in_threadgroup;
    const int flat = sg * 32 + lane;
    const int hk = threadgroup_position_in_grid.y % k_shape[1];
    const int b = threadgroup_position_in_grid.y / k_shape[1];
    const int qt = threadgroup_position_in_grid.z % ((q_shape[2] + QT - 1) / QT);
    const int block = threadgroup_position_in_grid.z / ((q_shape[2] + QT - 1) / QT);
    const int blocks = int(meta[2]);
    const int N = k_shape[2], L = q_shape[2], HQ = q_shape[1];
    for (int idx = flat; idx < J * D; idx += NS * 32) {
        int j = idx / D, d = idx % D, row = qt * QT + j / G;
        qs[idx] = row < L ? float(q[b*q_strides[0] + (hk*G+j%G)*q_strides[1]
            + row*q_strides[2] + d*q_strides[3]]) : 0.0f;
    }
    for (int idx = flat; idx < J * R; idx += NS * 32) {
        int j = idx / R, d = idx % R, row = qt * QT + j / G;
        rs[idx] = row < L ? float(rel[b*rel_strides[0] + row*rel_strides[1]
            + (hk*G+j%G)*rel_strides[2] + d*rel_strides[3]]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int chunk = (N + blocks - 1) / blocks;
    const int sub = (chunk + NS - 1) / NS;
    const int s0 = block * chunk + sg * sub;
    const int s1 = min(N, min((block + 1) * chunk, s0 + sub));
    float m[J], l[J], o[J][EPL];
    for (int j=0; j<J; ++j) {
        m[j]=-INFINITY; l[j]=0.0f;
        for (int d=0; d<EPL; ++d) o[j][d]=0.0f;
    }
    threadgroup float* myp = ps + sg * J * 32;
    for (int tile=s0; tile<s1; tile+=32) {
        const int t = tile + lane;
        float score[J];
        for (int j=0; j<J; ++j) score[j]=0.0f;
        if (t<s1) {
            for (int d=0; d<D; d+=4) {
                const size_t base = b*k_strides[0] + hk*k_strides[1]
                    + t*k_strides[2] + d*k_strides[3];
                const float4 kv = float4(float(k[base]), float(k[base+k_strides[3]]),
                    float(k[base+2*k_strides[3]]), float(k[base+3*k_strides[3]]));
                for (int j=0; j<J; ++j) {
                    const threadgroup float* qj=qs+j*D+d;
                    score[j]+=qj[0]*kv.x+qj[1]*kv.y+qj[2]*kv.z+qj[3]*kv.w;
                }
            }
        }
        for (int j=0; j<J; ++j) {
            const int row=qt*QT+j/G, pos=int(meta[0])+row, dist=pos-t;
            // A fully left-padded query has the reference's uniform finite
            // -1e30 mask. Its output is discarded by the recurrent mask.
            const bool padded=pos<int(pad[b]);
            const bool visible=t<s1 && row<L && (padded ||
                (dist>=0 && (WIN==0 || dist<WIN) && t>=int(pad[b])));
            float bias=0.0f;
            if (visible && !padded && dist<EXT) {
                for (int d=0; d<R; ++d)
                    bias+=rs[j*R+d]*float(proj[d*proj_strides[0]+dist*proj_strides[1]]);
            }
            // Preserve the released mask's low-precision materialization and
            // the separate low-precision logarithmic-temperature multiply.
            T rounded_bias=T(bias);
            const float temperature=row<L ? float(tau[b*tau_strides[0]+row*tau_strides[1]]) : 1.0f;
            float sc=padded ? 0.0f : score[j]*float(meta[1])+float(T(float(rounded_bias)*temperature));
            const float top=simd_max(visible ? sc : -INFINITY);
            const float nm=max(m[j],top);
            const float f=m[j]==-INFINITY ? 0.0f : fast::exp(m[j]-nm);
            const float probability=visible ? fast::exp(sc-nm) : 0.0f;
            l[j]=l[j]*f+simd_sum(probability); m[j]=nm;
            for (int d=0; d<EPL; ++d) o[j][d]*=f;
            myp[j*32+lane]=probability;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int u=0; u<min(32,s1-tile); ++u) {
            const size_t base=b*v_strides[0]+hk*v_strides[1]
                +(tile+u)*v_strides[2]+lane*EPL*v_strides[3];
            float vv[EPL];
            for (int d=0; d<EPL; ++d) vv[d]=float(v[base+d*v_strides[3]]);
            for (int j=0; j<J; ++j)
                for (int d=0; d<EPL; ++d) o[j][d]+=myp[j*32+u]*vv[d];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int P=blocks*NS, part=block*NS+sg;
    for (int j=0; j<J; ++j) {
        const int row=qt*QT+j/G;
        if (row>=L) continue;
        const size_t dst=((b*HQ+hk*G+j%G)*L+row)*P+part;
        const float inv=l[j]>0.0f ? 1.0f/l[j] : 0.0f;
        for (int d=0; d<EPL; ++d) po[dst*D+lane*EPL+d]=o[j][d]*inv;
        if (lane==0) {pl[dst]=l[j]; pm[dst]=l[j]>0.0f ? m[j] : -INFINITY;}
    }
"""

_MERGE = r"""
    const int lane=thread_index_in_simdgroup;
    const size_t row=threadgroup_position_in_grid.x;
    const int P=pl_shape[3];
    threadgroup float weights[256];
    float maximum=-INFINITY;
    for (int p=lane; p<P; p+=32) maximum=max(maximum,pm[row*P+p]);
    maximum=simd_max(maximum);
    float total=0.0f;
    for (int p=lane; p<P; p+=32) {
        float weight=pl[row*P+p]>0.0f ? pl[row*P+p]*fast::exp(pm[row*P+p]-maximum) : 0.0f;
        weights[p]=weight; total+=weight;
    }
    total=simd_sum(total);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int d=lane; d<128; d+=32) {
        float value=0.0f;
        for (int p=0; p<P; ++p) value+=weights[p]*po[(row*P+p)*128+d];
        out[row*128+d]=T(total>0.0f ? value/total : 0.0f);
    }
"""

_KERNEL = mx.fast.metal_kernel(
    name="mlx2_inkling_relative_attention",
    input_names=["q", "k", "v", "rel", "proj", "tau", "pad", "meta"],
    output_names=["po", "pl", "pm"],
    source=_SOURCE,
    ensure_row_contiguous=False,
)
_MERGE_KERNEL = mx.fast.metal_kernel(
    name="mlx2_inkling_relative_attention_merge",
    input_names=["po", "pl", "pm"],
    output_names=["out"],
    source=_MERGE,
)


def supported(q, k, v, rel, proj):
    if mx.default_device() != mx.gpu or q.ndim != 4 or k.ndim != 4:
        return False
    b, h, length, d = q.shape
    return (
        b > 0
        and length > 0
        and d == 128
        and k.shape == v.shape
        and k.shape[0] == b
        and k.shape[2] > 0
        and k.shape[3] == d
        and h % k.shape[1] == 0
        and 1 <= h // k.shape[1] <= 4
        and rel.shape[:3] == (b, length, h)
        and 1 <= rel.shape[3] <= 16
        and proj.shape[0] == rel.shape[3]
        and q.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and all(a.dtype == q.dtype for a in (k, v, rel, proj))
    )


def relative_attention(
    q, k, v, rel, proj, *, offset, scale, window=0, left_padding=None, tau=None
):
    """All inputs use explicit strides; K/V are chronological committed views.

    q already includes the reference's rounded temperature multiplication.
    tau applies that same multiplication to the rounded learned bias.
    """
    if not supported(q, k, v, rel, proj):
        raise ValueError("unsupported Inkling fused attention geometry/dtype/device")
    b, h, length, _ = q.shape
    qt = 1 if length == 1 else 4
    tiles = (length + qt - 1) // qt
    blocks = min(
        64, max(1, (128 + b * k.shape[1] * tiles - 1) // (b * k.shape[1] * tiles))
    )
    blocks = min(blocks, max(1, (k.shape[2] + 127) // 128))
    parts = blocks * 4
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
    po, pl, pm = _KERNEL(
        inputs=[
            q,
            k,
            v,
            rel,
            proj,
            tau,
            pad,
            mx.array([offset, scale, blocks], mx.float32),
        ],
        template=[
            ("T", q.dtype),
            ("G", h // k.shape[1]),
            ("QT", qt),
            ("R", rel.shape[3]),
            ("EXT", proj.shape[1]),
            ("WIN", window),
        ],
        grid=(128, b * k.shape[1], blocks * tiles),
        threadgroup=(128, 1, 1),
        output_shapes=[
            (b, h, length, parts, 128),
            (b, h, length, parts),
            (b, h, length, parts),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return _MERGE_KERNEL(
        inputs=[po, pl, pm],
        template=[("T", q.dtype)],
        grid=(32 * b * h * length, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]
