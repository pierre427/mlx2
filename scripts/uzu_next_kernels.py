"""Experimental Uzu mechanisms; no production dispatch or state publication.

Original MLX/Metal translation; see provenance/uzu-next-candidates-20261027.json.
"""

from functools import lru_cache

import mlx.core as mx


def prefix_splits(keys, rows, blocks):
    chunk = -(-keys // blocks)
    # Unmasked tiles must be complete, including each split's last tile.
    return min(blocks, (keys - rows) // chunk) if chunk % 32 == 0 else 0


@lru_cache(None)
def uniform_attention_kernel():
    from mlx2.runtime.models import qsdpa_verify_metal as base

    start = base._SOURCE.index("    for (int tile = kstart; tile < kend; tile += TN)")
    end = base._SOURCE.index("    #undef LOAD_K", start)
    loop = base._SOURCE[start:end]
    # Uniform for the entire group; a complete prefix split cannot encounter
    # either a causal tail or a partial key tile. No extra dispatch or concat.
    source = (
        base._SOURCE[:start]
        + "if (kend <= N-L && chunk % TN == 0) {\n"
        + loop.replace("t0 + u < my_limit", "srow < nf")
        + "} else {\n"
        + loop
        + "}\n"
        + base._SOURCE[end:]
    )
    return mx.fast.metal_kernel(
        name="uzu_research_attention_uniform_prefix",
        input_names=["q", "kw", "ks", "kb", "vw", "vs", "vb", "scale", "lpad"],
        output_names=["part_o", "part_l", "part_m"],
        source=source,
        ensure_row_contiguous=False,
    )


def uniform_prefix_attention(
    q, qk, qv, *, scale, group_size=64, key_bits=8, value_bits=8, blocks=None
):
    from mlx2.runtime.models import qsdpa_verify_metal as base

    b, h, rows, d = q.shape
    hkv, keys = qk[0].shape[1:3]
    groups, per = base.fused_groups(rows, h // hkv)
    blocks = blocks or base.verify_blocks(keys, b, hkv, groups)
    po, pl, pm = uniform_attention_kernel()(
        inputs=[q, *qk, *qv, mx.array([scale], mx.float32), mx.zeros((b,), mx.int32)],
        template=[
            ("T", q.dtype),
            ("D", d),
            ("G", h // hkv),
            ("L", rows),
            ("KB", key_bits),
            ("VB", value_bits),
            ("GS", group_size),
            ("FRB", per),
            ("FG", groups),
            ("NS", base.SIMDGROUPS),
        ],
        grid=(32 * base.SIMDGROUPS, b * hkv * groups, blocks),
        threadgroup=(32 * base.SIMDGROUPS, 1, 1),
        output_shapes=[
            (b, h, rows, blocks, d),
            (b, h, rows, blocks),
            (b, h, rows, blocks),
        ],
        output_dtypes=[mx.float32] * 3,
    )
    return base.merge_partials(po, pl, pm).astype(q.dtype)


@lru_cache(None)
def attention_kernel(free):
    from mlx2.runtime.models import qsdpa_verify_metal as base

    source = base._SOURCE.replace(
        "const int block = int(threadgroup_position_in_grid.z);",
        "const int local_block = int(threadgroup_position_in_grid.z);\n"
        "const int block = local_block + BASE;",
    )
    source = source.replace(
        "const int blocks = int(threadgroups_per_grid.z);", "const int blocks = TOTAL;"
    )
    source = source.replace(
        "const int P = blocks;", "const int P = int(threadgroups_per_grid.z);"
    )
    source = source.replace("* P + block)", "* P + local_block)")
    if free:
        source = source.replace("t0 + u < my_limit", "srow < nf")
    return mx.fast.metal_kernel(
        name="uzu_research_attention_prefix_" + str(int(free)),
        input_names=["q", "kw", "ks", "kb", "vw", "vs", "vb", "scale", "lpad"],
        output_names=["part_o", "part_l", "part_m"],
        source=source,
        ensure_row_contiguous=False,
    )


def prefix_attention(
    q, qk, qv, *, scale, group_size=64, key_bits=8, value_bits=8, blocks=None
):
    from mlx2.runtime.models import qsdpa_verify_metal as base

    b, h, rows, d = q.shape
    if b != 1:
        raise ValueError(
            "research prefix split currently requires B1 without left padding"
        )
    hkv, keys = qk[0].shape[1:3]
    groups, per = base.fused_groups(rows, h // hkv)
    blocks = blocks or base.verify_blocks(keys, b, hkv, groups)
    count = prefix_splits(keys, rows, blocks)
    if not count:
        return base.gqa_quantized_verify_attention(
            q,
            qk,
            qv,
            scale=scale,
            group_size=group_size,
            key_bits=key_bits,
            value_bits=value_bits,
            blocks=blocks,
        )
    parts = []
    for start, size, free in ((0, count, True), (count, blocks - count, False)):
        if not size:
            continue
        parts.append(
            attention_kernel(free)(
                inputs=[
                    q,
                    *qk,
                    *qv,
                    mx.array([scale], mx.float32),
                    mx.zeros((b,), mx.int32),
                ],
                template=[
                    ("T", q.dtype),
                    ("D", d),
                    ("G", h // hkv),
                    ("L", rows),
                    ("KB", key_bits),
                    ("VB", value_bits),
                    ("GS", group_size),
                    ("FRB", per),
                    ("FG", groups),
                    ("NS", base.SIMDGROUPS),
                    ("BASE", start),
                    ("TOTAL", blocks),
                ],
                grid=(32 * base.SIMDGROUPS, b * hkv * groups, size),
                threadgroup=(32 * base.SIMDGROUPS, 1, 1),
                output_shapes=[
                    (b, h, rows, size, d),
                    (b, h, rows, size),
                    (b, h, rows, size),
                ],
                output_dtypes=[mx.float32] * 3,
            )
        )
    po, pl, pm = [
        mx.concatenate([p[i] for p in parts], axis=-2 if i == 0 else -1)
        for i in range(3)
    ]
    return base.merge_partials(po, pl, pm).astype(q.dtype)


# One group per value head. Gram builders overlap the owners' H0 projections;
# inverse columns use forward substitution, then owners use matrix readout.
# No recurrent state is returned: accepted paths retain the serial replay.
MATRIX_GDN = r"""
constexpr int NB = NT / 16;
const int lane = thread_index_in_simdgroup;
const int sg = simdgroup_index_in_threadgroup;
const int tid = sg * 32 + lane;
const int hv = threadgroup_position_in_grid.z;
const int hk = hv / (HV/HK);
const int fm = ((lane/4)&4) + ((lane/2)%4);
const int fn = (((lane/4)&2)*2) + (lane%2)*2;
threadgroup float a[NT*NT], qkd[NT*NT], inverse[NT*NT], initial[NT];
if (tid < NT) {
    float z = 1.0f;
    for (int p=tid<N ? tid : -1; p>=0; p=parents[p]) z *= gates[p*HV+hv];
    initial[tid] = tid < N ? z : 0.0f;
}
simdgroup_float8x8 kh[2][NT/8], qh[2][NT/8], vf[2][NT/8];
if (sg < NB) {
    for (int rb=sg*2; rb<sg*2+2; rb++) {
        for (int cb=0; cb<NT/8; cb++) {
            auto kk = make_filled_simdgroup_matrix<float,8,8>(0.0f);
            auto qk = make_filled_simdgroup_matrix<float,8,8>(0.0f);
            for (int step=0; step<16; step++) {
                simdgroup_float8x8 kl, ql, kr;
                for (int e=0; e<2; e++) {
                    int ri=rb*8+fm, cj=cb*8+fn+e;
                    kl.thread_elements()[e] = ri<N ? float(k[(ri*HK+hk)*128+step*8+fn+e]) : 0.0f;
                    ql.thread_elements()[e] = ri<N ? float(q[(ri*HK+hk)*128+step*8+fn+e]) : 0.0f;
                    kr.thread_elements()[e] = cj<N ? float(k[(cj*HK+hk)*128+step*8+fm]) : 0.0f;
                }
                simdgroup_multiply_accumulate(kk, kl, kr, kk);
                simdgroup_multiply_accumulate(qk, ql, kr, qk);
            }
            for (int e=0; e<2; e++) {
                int ri=rb*8+fm, cj=cb*8+fn+e;
                int p=ri<N ? ri : -1; float decay=1.0f;
                while (p>cj && p>=0) { decay *= gates[p*HV+hv]; p=parents[p]; }
                bool ancestor = ri<N && cj<N && p==cj;
                a[ri*NT+cj] = ancestor && ri!=cj ? float(beta[ri*HV+hv])*decay*kk.thread_elements()[e] : 0.0f;
                qkd[ri*NT+cj] = ancestor ? decay*qk.thread_elements()[e] : 0.0f;
            }
        }
    }
} else {
    const int dv=(sg-NB)*16;
    for (int rb=0; rb<2; rb++) for (int cb=0; cb<NT/8; cb++) {
        kh[rb][cb] = make_filled_simdgroup_matrix<float,8,8>(0.0f);
        qh[rb][cb] = make_filled_simdgroup_matrix<float,8,8>(0.0f);
        for (int e=0; e<2; e++) {
            int token=cb*8+fn+e;
            vf[rb][cb].thread_elements()[e] = token<N ? float(v[(token*HV+hv)*128+dv+rb*8+fm]) : 0.0f;
        }
        for (int step=0; step<16; step++) {
            simdgroup_float8x8 s, kr, qr;
            for (int e=0; e<2; e++) {
                int token=cb*8+fn+e;
                s.thread_elements()[e] = state[(hv*128+dv+rb*8+fm)*128+step*8+fn+e];
                kr.thread_elements()[e] = token<N ? float(k[(token*HK+hk)*128+step*8+fm]) : 0.0f;
                qr.thread_elements()[e] = token<N ? float(q[(token*HK+hk)*128+step*8+fm]) : 0.0f;
            }
            simdgroup_multiply_accumulate(kh[rb][cb], s, kr, kh[rb][cb]);
            simdgroup_multiply_accumulate(qh[rb][cb], s, qr, qh[rb][cb]);
        }
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg==0 && lane<NT) {
    float column[NT];
    for (int i=0; i<NT; i++) {
        float val=i==lane ? 1.0f : 0.0f;
        for (int j=0; j<i; j++) val -= a[i*NT+j]*column[j];
        column[i]=val;
        inverse[i*NT+lane]=val;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg>=NB) {
    simdgroup_float8x8 u[2][NT/8];
    for (int rb=0; rb<2; rb++) for (int cb=0; cb<NT/8; cb++) {
        for (int e=0; e<2; e++) {
            int token=cb*8+fn+e;
            kh[rb][cb].thread_elements()[e] = token<N ? float(beta[token*HV+hv])*(vf[rb][cb].thread_elements()[e] - initial[token]*kh[rb][cb].thread_elements()[e]) : 0.0f;
            qh[rb][cb].thread_elements()[e] *= initial[token];
        }
    }
    for (int rb=0; rb<2; rb++) for (int cb=0; cb<NT/8; cb++) {
        u[rb][cb] = make_filled_simdgroup_matrix<float,8,8>(0.0f);
        for (int kb=0; kb<NT/8; kb++) {
            simdgroup_float8x8 inv;
            for (int e=0; e<2; e++) inv.thread_elements()[e] = inverse[(cb*8+fn+e)*NT+kb*8+fm];
            simdgroup_multiply_accumulate(u[rb][cb], kh[rb][kb], inv, u[rb][cb]);
        }
    }
    for (int rb=0; rb<2; rb++) for (int cb=0; cb<NT/8; cb++) {
        for (int kb=0; kb<NT/8; kb++) {
            simdgroup_float8x8 w;
            for (int e=0; e<2; e++) w.thread_elements()[e] = qkd[(cb*8+fn+e)*NT+kb*8+fm];
            simdgroup_multiply_accumulate(qh[rb][cb], u[rb][kb], w, qh[rb][cb]);
        }
        for (int e=0; e<2; e++) {
            int token=cb*8+fn+e;
            if (token<N) out[(token*HV+hv)*128+(sg-NB)*16+rb*8+fm] = T(qh[rb][cb].thread_elements()[e]);
        }
    }
}
"""


@lru_cache(None)
def matrix_kernel():
    return mx.fast.metal_kernel(
        name="uzu_research_gdn_matrix",
        source=MATRIX_GDN,
        input_names=["q", "k", "v", "gates", "beta", "state", "parents"],
        output_names=["out"],
    )


def matrix_tree(q, k, v, g, beta, state, parents, *, parent_tensor=None):
    n, hk, dk = q.shape[1:]
    hv, dv = v.shape[-2:]
    if not (1 <= n <= 32 and dk == dv == 128 and state.dtype == mx.float32):
        raise ValueError(
            "research matrix GDN requires <=32 rows, DK=DV=128, FP32 state"
        )
    if len(parents) != n or any(p < -1 or p >= i for i, p in enumerate(parents)):
        raise ValueError("parents must precede children")
    nt = 16 if n <= 16 else 32
    # Padded ancestry is never dereferenced; explicit roots guard every loop.
    padded = (
        parent_tensor
        if parent_tensor is not None
        else mx.array(list(parents) + [-1] * (nt - n), mx.int32)
    )
    return matrix_kernel()(
        inputs=[q, k, v, g, beta, state, padded],
        template=[("T", q.dtype), ("N", n), ("NT", nt), ("HK", hk), ("HV", hv)],
        grid=(32 * (8 + nt // 16), 1, hv),
        threadgroup=(32 * (8 + nt // 16), 1, 1),
        output_shapes=[v.shape],
        output_dtypes=[q.dtype],
    )[0]
