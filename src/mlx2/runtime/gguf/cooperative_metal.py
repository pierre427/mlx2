"""Default-off packed block-reuse GEMV and expert candidates.

Mined SIMD chunking/codebook staging from pinned antirez/ds4 (MIT), transcribed
for MLX-owned buffers and bounded grids. These retain FP32 products/accumulation
but change reduction order; they require numerical and model-level probes.
Ordinary packed_metal remains an independent selectable reference.
"""

import math
from functools import lru_cache

from .packed_metal import HEADER, PackedMatrix
from .reader import GGUFError

BLOCK_HEADER = (
    HEADER
    + r"""
inline float iq2_dot32(device const uchar *row, uint chunk, thread const float *a,
                      threadgroup const ulong *grids, threadgroup const uchar *sign_table) {
    device const uchar *block=row+(chunk/8)*66;
    device const uchar *p=block+2+(chunk%8)*8;
    uint aux=gguf_u32(p+4);
    float d=gguf_half(block)*(0.5f+float(aux>>28))*0.25f;
    float sum=0.0f;
    #pragma unroll
    for(uint j=0;j<4;j++) {
        ulong grid=grids[p[j]];
        uchar signs=sign_table[(aux>>(7*j))&127];
        #pragma unroll
        for(uint k=0;k<8;k++) {
            float q=float((grid>>(8*k))&255)*((signs&(1<<k)) ? -1.0f : 1.0f);
            sum+=a[8*j+k]*q;
        }
    }
    return d*sum;
}
inline float q8_dot32(device const uchar *row, uint chunk, thread const float *a) {
    device const uchar *p=row+chunk*34;
    float sum=0.0f;
    #pragma unroll
    for(uint k=0;k<32;k++) sum+=a[k]*float(as_type<char>(p[2+k]));
    return gguf_half(p)*sum;
}
inline float q2_dot8(device const uchar *row, uint block, uint lane, thread const float *a) {
    device const uchar *p=row+block*84;
    uint group=lane/2, base=16+32*(group/8)+16*(group&1)+8*(lane&1);
    uint shift=((group/2)&3)*2, sc=p[group];
    float dot=0.0f, sum=0.0f;
    #pragma unroll
    for(uint i=0;i<8;i++) {
        dot+=a[i]*float((uint(p[base+i])>>shift)&3);
        sum+=a[i];
    }
    return gguf_half(p+80)*float(sc&15)*dot-gguf_half(p+82)*float(sc>>4)*sum;
}
"""
)

STAGE = r"""
const uint tid=thread_index_in_threadgroup;
const uint lane=thread_index_in_simdgroup;
const uint sg=simdgroup_index_in_threadgroup;
threadgroup ulong grids[256];
threadgroup uchar signs[128];
for(uint i=tid;i<256;i+=128) grids[i]=iq2_grid[i];
if(tid<128) signs[tid]=uchar(tid|((popcount(tid)&1)<<7));
threadgroup_barrier(mem_flags::mem_threadgroup);
"""

GATE_UP = (
    STAGE
    + r"""
const uint slot=threadgroup_position_in_grid.z, token=slot/TOPK;
const uint row0=(threadgroup_position_in_grid.x*4+sg)*ROWS;
const uint expert=ids[slot];
if(expert>=EXPERTS) {
    if(lane==0) for(uint r=0;r<ROWS && row0+r<M;r++) mid[size_t(slot)*M+row0+r]=NAN;
    return;
}
float sums_g[ROWS]={0.0f},sums_u[ROWS]={0.0f};
for(uint chunk=lane;chunk<K/32;chunk+=32) {
    float a[32];
    #pragma unroll
    for(uint i=0;i<32;i++) a[i]=x[size_t(token)*K+chunk*32+i];
    #pragma unroll
    for(uint r=0;r<ROWS;r++) {
        if(row0+r<M) {
            size_t offset=(size_t(expert)*M+row0+r)*ROW_BYTES;
            sums_g[r]+=iq2_dot32(gate+offset,chunk,a,grids,signs);
            sums_u[r]+=iq2_dot32(up+offset,chunk,a,grids,signs);
        }
    }
}
#pragma unroll
for(uint r=0;r<ROWS;r++) {
    float g=simd_sum(sums_g[r]),u=simd_sum(sums_u[r]);
    if(lane==0 && row0+r<M) {
        g=min(g,10.0f);u=clamp(u,-10.0f,10.0f);
        mid[size_t(slot)*M+row0+r]=(g/(1.0f+exp(-g)))*u;
    }
}
"""
)

DOWN = r"""
const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
const uint slot=threadgroup_position_in_grid.z;
const uint row0=(threadgroup_position_in_grid.x*4+sg)*ROWS;
const uint expert=ids[slot];
if(expert>=EXPERTS) {
    if(lane==0) for(uint r=0;r<ROWS && row0+r<N;r++) out[size_t(slot)*N+row0+r]=NAN;
    return;
}
float sums[ROWS]={0.0f};
for(uint block=0;block<M/256;block++) {
    float a[8];
    #pragma unroll
    for(uint i=0;i<8;i++) a[i]=mid[size_t(slot)*M+block*256+lane*8+i];
    #pragma unroll
    for(uint r=0;r<ROWS;r++) {
        if(row0+r<N) sums[r]+=q2_dot8(w+(size_t(expert)*N+row0+r)*ROW_BYTES,block,lane,a);
    }
}
#pragma unroll
for(uint r=0;r<ROWS;r++) {
    float sum=simd_sum(sums[r]);
    if(lane==0 && row0+r<N) out[size_t(slot)*N+row0+r]=sum;
}
"""

COMBINE = r"""
uint row=thread_position_in_grid.x,token=thread_position_in_grid.y;
if(row>=N) return;
float sum=0.0f;
for(uint slot=0;slot<TOPK;slot++) {
    uint idx=token*TOPK+slot;
    sum+=scores[idx]*expert_out[size_t(idx)*N+row];
}
y[size_t(token)*N+row]=sum;
"""

DENSE = (
    STAGE
    + r"""
const uint token=threadgroup_position_in_grid.z;
const uint row0=(threadgroup_position_in_grid.x*4+sg)*ROWS;
float sums[ROWS]={0.0f};
for(uint chunk=lane;chunk<K/32;chunk+=32) {
    float a[32];
    #pragma unroll
    for(uint i=0;i<32;i++) a[i]=x[size_t(token)*K+chunk*32+i];
    #pragma unroll
    for(uint r=0;r<ROWS;r++) {
        if(row0+r<N) {
            device const uchar *p=w+size_t(row0+r)*ROW_BYTES;
            sums[r]+=FORMAT==8 ? q8_dot32(p,chunk,a) : iq2_dot32(p,chunk,a,grids,signs);
        }
    }
}
#pragma unroll
for(uint r=0;r<ROWS;r++) {
    float sum=simd_sum(sums[r]);
    if(lane==0 && row0+r<N) y[size_t(token)*N+row0+r]=sum;
}
"""
)


GROUPED = DENSE.replace(
    "const uint token=threadgroup_position_in_grid.z;",
    "const uint token=threadgroup_position_in_grid.z;\nconst uint group=token%GROUPS;",
).replace(
    "w+size_t(row0+r)*ROW_BYTES",
    "w+(size_t(group)*N+row0+r)*ROW_BYTES",
)


@lru_cache(maxsize=5)
def _kernel(kind):
    import mlx.core as mx

    args = {
        "gate_up": (["x", "ids", "gate", "up"], ["mid"], GATE_UP),
        "down": (["mid", "ids", "w"], ["out"], DOWN),
        "combine": (["expert_out", "scores"], ["y"], COMBINE),
        "dense": (["x", "w"], ["y"], DENSE),
        "grouped": (["x", "w"], ["y"], GROUPED),
    }
    inputs, outputs, source = args[kind]
    return mx.fast.metal_kernel(
        name=f"mlx2_gguf_cooperative_{kind}_v1",
        input_names=inputs,
        output_names=outputs,
        header=BLOCK_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


def cooperative_dense(x, weight: PackedMatrix, *, rows=4):
    import mlx.core as mx

    if (
        rows not in (1, 2, 4)
        or weight.ggml_type not in (8, 16)
        or len(weight.shape) != 2
    ):
        raise GGUFError(
            "cooperative dense supports Q8_0/IQ2_XXS rank2 and rows1/2/4 only"
        )
    n, k = weight.shape
    if x.ndim != 2 or x.shape[-1] != k:
        raise GGUFError("cooperative dense needs matching rank2 input")
    b = x.shape[0]
    template = [
        ("K", k),
        ("N", n),
        ("ROW_BYTES", weight.row_bytes),
        ("FORMAT", weight.ggml_type),
        ("ROWS", rows),
    ]
    return _kernel("dense")(
        inputs=[x.astype(mx.float32), weight.data],
        template=template,
        grid=(((n + 4 * rows - 1) // (4 * rows)) * 128, 1, b),
        threadgroup=(128, 1, 1),
        output_shapes=[(b, n)],
        output_dtypes=[mx.float32],
    )[0]


def cooperative_grouped(x, weight: PackedMatrix, *, rows=4):
    """Reuse packed blocks across rows without merging independent groups."""
    import mlx.core as mx

    if (
        rows not in (1, 2, 4)
        or weight.ggml_type not in (8, 16)
        or len(weight.shape) != 3
        or x.ndim < 2
    ):
        raise GGUFError("cooperative grouped needs Q8/IQ2 rank3 and rows1/2/4")
    groups, n, k = weight.shape
    if x.shape[-2:] != (groups, k):
        raise GGUFError("cooperative grouped input axes mismatch")
    leading = x.shape[:-2]
    b = math.prod(leading)
    template = [
        ("K", k),
        ("N", n),
        ("GROUPS", groups),
        ("ROW_BYTES", weight.row_bytes),
        ("FORMAT", weight.ggml_type),
        ("ROWS", rows),
    ]
    y = _kernel("grouped")(
        inputs=[x.reshape(b * groups, k).astype(mx.float32), weight.data],
        template=template,
        grid=(((n + 4 * rows - 1) // (4 * rows)) * 128, 1, b * groups),
        threadgroup=(128, 1, 1),
        output_shapes=[(b * groups, n)],
        output_dtypes=[mx.float32],
    )[0]
    return y.reshape(*leading, groups, n)


def cooperative_moe(
    x, ids, scores, gate: PackedMatrix, up: PackedMatrix, down: PackedMatrix, *, rows=4
):
    import mlx.core as mx

    if (
        rows not in (1, 2, 4)
        or gate.ggml_type != 16
        or up.ggml_type != 16
        or down.ggml_type != 10
    ):
        raise GGUFError(
            "cooperative experts require IQ2_XXS gate/up + Q2_K down and rows1/2/4"
        )
    if len(gate.shape) != 3 or gate.shape != up.shape:
        raise GGUFError("matching rank3 gate/up required")
    experts, m, k = gate.shape
    if down.shape != (experts, k, m) or x.ndim != 2 or x.shape[-1] != k:
        raise GGUFError("cooperative expert/input axes mismatch")
    if (
        ids.ndim != 2
        or ids.dtype != mx.uint32
        or scores.shape != ids.shape
        or ids.shape[0] != x.shape[0]
    ):
        raise GGUFError("matching [tokens,topk] uint32 IDs and scores required")
    b, topk = ids.shape
    if not 1 <= topk <= experts:
        raise GGUFError("invalid topk")
    common = [
        ("K", k),
        ("M", m),
        ("N", k),
        ("TOPK", topk),
        ("EXPERTS", experts),
        ("ROWS", rows),
    ]
    mid = _kernel("gate_up")(
        inputs=[x.astype(mx.float32), ids, gate.data, up.data],
        template=common + [("ROW_BYTES", gate.row_bytes)],
        grid=(((m + 4 * rows - 1) // (4 * rows)) * 128, 1, b * topk),
        threadgroup=(128, 1, 1),
        output_shapes=[(b * topk, m)],
        output_dtypes=[mx.float32],
    )[0]
    expert_out = _kernel("down")(
        inputs=[mid, ids, down.data],
        template=common + [("ROW_BYTES", down.row_bytes)],
        grid=(((k + 4 * rows - 1) // (4 * rows)) * 128, 1, b * topk),
        threadgroup=(128, 1, 1),
        output_shapes=[(b * topk, k)],
        output_dtypes=[mx.float32],
    )[0]
    return _kernel("combine")(
        inputs=[expert_out, scores.astype(mx.float32)],
        template=common,
        grid=(k, b, 1),
        threadgroup=(min(k, 256), 1, 1),
        output_shapes=[(b, k)],
        output_dtypes=[mx.float32],
    )[0]
