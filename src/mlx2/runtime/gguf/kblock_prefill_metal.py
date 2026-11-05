"""K-block NAX prefill tiles: wider row tiles and chunk-decoded weight tiles.

nax_metal.NAX_SOURCE decodes each threadgroup's 32x64 weight tile element by
element (gguf_value) and reuses it for only 16 prompt rows, so a 512-row chunk
decodes every weight 32 times and the K-block decode ALU, not the MMA, bounds
prefill (~3-4 TF/s of ~61 TF/s on the M5 Max). This kernel keeps that tile's
per-output law -- half(x) and half(weight value) operands, one 16x32x64 MMA per
64-wide K step, FP32 sums in K order -- and changes only how the operands are
staged:

* MT row tiles (16*MT prompt rows) share one decoded weight tile;
* each lane fills two 32-value chunks of the weight tile, decoding the block
  scales once per chunk, with the element expressions of gguf_k_value.

Each output should therefore equal nax_prefill's bit for bit (checked by the
microbench), while the decode work per output falls by MT and per element.
Same FP16-operand law as nax_fp16; explicit, unqualified.
"""

import math
from functools import lru_cache

from .kblock_metal import KBLOCK_HEADER
from .packed_metal import PackedMatrix
from .reader import GGUFError

HEADER = (
    KBLOCK_HEADER
    + r"""
// The 32 values of one K-block chunk, each as gguf_k_value computes it, to half.
// Vectorized within registers: four consecutive elements of a chunk sit in four
// contiguous bytes, so one 32-bit word yields four codes with one shift/mask, a
// uchar4 -> float4 convert, one float4 scale and one half4 store. Each value is
// still the scalar expression ((d*scale)*q, minus dmin*min for Q4_K/Q5_K), so
// the halves written should be bit-identical to the scalar decoder's.
inline uint kb_u32(device const uchar *p) {  // 2-byte-aligned 4-byte read
    device const ushort *q=(device const ushort*)p;
    return uint(q[0])|(uint(q[1])<<16);
}
// Two-stage form of kblock_values32 for software pipelining: kb_load issues
// every device read of one chunk into registers; kb_decode turns them into the
// same half values kblock_values32 writes (identical expressions).
struct KRaw { uint a[8]; uint b[8]; uint s0, s1, s2; };
inline KRaw kb_load(device const uchar *row, uint chunk, uint format) {
    KRaw r;
    const uint c=chunk%8;
    if(format==11) {
        device const uchar *p=row+(chunk/8)*110;
        device const uchar *qs=p+32+(c/4)*32;
        #pragma unroll
        for(uint w=0;w<8;w++) { r.a[w]=kb_u32(qs+4*w); r.b[w]=kb_u32(p+4*w); }
        const uint g0=2*c, g1=2*c+1;
        r.s0=uint(p[96+g0%8])|(uint(p[96+g1%8])<<8)|(uint(p[104+g0%4])<<16)|(uint(p[104+g1%4])<<24);
        r.s1=uint(p[108])|(uint(p[109])<<8);
        r.s2=0;
        return r;
    }
    if(format==12 || format==13) {
        device const uchar *p=row+(chunk/8)*(format==12 ? 144 : 176);
        device const uint *qs=(device const uint*)(p+(format==12 ? 16 : 48)+(c/2)*32);
        device const uint *qh=(device const uint*)(p+16);
        #pragma unroll
        for(uint w=0;w<8;w++) { r.a[w]=qs[w]; r.b[w]=format==13 ? qh[w] : 0u; }
        r.s0=uint(p[4+c%4])|(uint(p[8+c%4])<<8)|(uint(p[12+c%4])<<16);
        r.s1=*((device const uint*)p);  // d (low half) and dmin (high half)
        r.s2=0;
        return r;
    }
    device const uchar *p=row+(chunk/8)*210;
    device const uchar *ql=p+(c/4)*64+(c%2)*32;
    device const uchar *qh=p+128+(c/4)*32;
    #pragma unroll
    for(uint w=0;w<8;w++) { r.a[w]=kb_u32(ql+4*w); r.b[w]=kb_u32(qh+4*w); }
    r.s0=uint(p[192+2*c])|(uint(p[193+2*c])<<8);
    r.s1=uint(p[208])|(uint(p[209])<<8);
    r.s2=0;
    return r;
}
inline float kb_half_bits(uint bits) { return float(as_type<half>(ushort(bits))); }
inline void kb_decode(thread const KRaw &r, uint chunk, threadgroup half *out, uint format) {
    threadgroup half4 *o4=(threadgroup half4*)out;
    const uint c=chunk%8;
    if(format==11) {
        const float d=kb_half_bits(r.s1);
        float ds[2];
        #pragma unroll
        for(uint h=0;h<2;h++) {
            const uint g=2*c+h;
            const uint lo=((r.s0>>(8*h))>>(4*(g/8)))&15;
            const uint hi=((r.s0>>(16+8*h))>>(2*(g/4)))&3;
            ds[h]=d*float(int(lo|(hi<<4))-32);
        }
        const uint shift=2*(c%4);
        #pragma unroll
        for(uint w=0;w<8;w++) {
            const uint code=((r.a[w]>>shift)&0x03030303u)|(((r.b[w]>>c)&0x01010101u)<<2);
            o4[w]=half4(ds[w/4]*(float4(as_type<uchar4>(code))-4.0f));
        }
        return;
    }
    if(format==12 || format==13) {
        const uint sc=r.s0&255, mn=(r.s0>>8)&255, aux=(r.s0>>16)&255;
        const uint scale=c<4 ? sc&63 : (aux&15)|((sc>>2)&48);
        const uint minimum=c<4 ? mn&63 : (aux>>4)|((mn>>2)&48);
        const float ds=kb_half_bits(r.s1)*float(scale), dm=kb_half_bits(r.s1>>16)*float(minimum);
        const uint shift=4*(c%2);
        #pragma unroll
        for(uint w=0;w<8;w++) {
            uint code=(r.a[w]>>shift)&0x0F0F0F0Fu;
            if(format==13) code|=((r.b[w]>>c)&0x01010101u)<<4;
            o4[w]=half4(ds*float4(as_type<uchar4>(code))-dm);
        }
        return;
    }
    const float d=kb_half_bits(r.s1);
    const float ds[2]={d*float(int(as_type<char>(uchar(r.s0&255)))), d*float(int(as_type<char>(uchar((r.s0>>8)&255))))};
    const uint lshift=4*((c/2)%2), hshift=2*(c%4);
    #pragma unroll
    for(uint w=0;w<8;w++) {
        const uint code=((r.a[w]>>lshift)&0x0F0F0F0Fu)|(((r.b[w]>>hshift)&0x03030303u)<<4);
        o4[w]=half4(ds[w/4]*(float4(as_type<uchar4>(code))-32.0f));
    }
}
inline void kblock_values32(device const uchar *row, uint chunk, threadgroup half *out, uint format) {
    threadgroup half4 *o4=(threadgroup half4*)out;
    if(format==2) {
        device const uchar *p=row+chunk*18;
        const float d=gguf_k_half(p);
        #pragma unroll
        for(uint w=0;w<4;w++) {
            const uint v=kb_u32(p+2+4*w);
            o4[w]=half4(d*(float4(as_type<uchar4>(v&0x0F0F0F0Fu))-8.0f));
            o4[w+4]=half4(d*(float4(as_type<uchar4>((v>>4)&0x0F0F0F0Fu))-8.0f));
        }
        return;
    }
    const uint c=chunk%8;
    if(format==11) {
        device const uchar *p=row+(chunk/8)*110;
        const float d=gguf_k_half(p+108);
        float ds[2];
        #pragma unroll
        for(uint h=0;h<2;h++) {
            const uint g=2*c+h;
            const uint lo=(p[96+g%8]>>(4*(g/8)))&15;
            const uint hi=(p[104+g%4]>>(2*(g/4)))&3;
            ds[h]=d*float(int(lo|(hi<<4))-32);
        }
        device const uchar *qs=p+32+(c/4)*32;
        const uint shift=2*(c%4);
        #pragma unroll
        for(uint w=0;w<8;w++) {
            // code = q2 | (h << 2) = q + 4, with q = q2 - 4*(1-h)
            const uint code=((kb_u32(qs+4*w)>>shift)&0x03030303u)|(((kb_u32(p+4*w)>>c)&0x01010101u)<<2);
            o4[w]=half4(ds[w/4]*(float4(as_type<uchar4>(code))-4.0f));
        }
        return;
    }
    if(format==12 || format==13) {
        device const uchar *p=row+(chunk/8)*(format==12 ? 144 : 176);
        const uint sc=p[4+c%4], mn=p[8+c%4], aux=p[12+c%4];
        const uint scale=c<4 ? sc&63 : (aux&15)|((sc>>2)&48);
        const uint minimum=c<4 ? mn&63 : (aux>>4)|((mn>>2)&48);
        const float ds=gguf_k_half(p)*float(scale), dm=gguf_k_half(p+2)*float(minimum);
        device const uint *qs=(device const uint*)(p+(format==12 ? 16 : 48)+(c/2)*32);
        device const uint *qh=(device const uint*)(p+16);
        const uint shift=4*(c%2);
        #pragma unroll
        for(uint w=0;w<8;w++) {
            uint code=(qs[w]>>shift)&0x0F0F0F0Fu;
            if(format==13) code|=((qh[w]>>c)&0x01010101u)<<4;
            o4[w]=half4(ds*float4(as_type<uchar4>(code))-dm);
        }
        return;
    }
    device const uchar *p=row+(chunk/8)*210;
    const float d=gguf_k_half(p+208);
    const float ds[2]={d*float(int(as_type<char>(p[192+2*c]))), d*float(int(as_type<char>(p[193+2*c])))};
    device const uchar *ql=p+(c/4)*64+(c%2)*32;
    device const uchar *qh=p+128+(c/4)*32;
    const uint lshift=4*((c/2)%2), hshift=2*(c%4);
    #pragma unroll
    for(uint w=0;w<8;w++) {
        const uint code=((kb_u32(ql+4*w)>>lshift)&0x0F0F0F0Fu)|(((kb_u32(qh+4*w)>>hshift)&0x03030303u)<<4);
        o4[w]=half4(ds[w/4]*(float4(as_type<uchar4>(code))-32.0f));
    }
}
"""
    + "\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n"
)

SOURCE = r"""
const ushort lane=thread_index_in_simdgroup;
const ushort sg=simdgroup_index_in_threadgroup;
const ushort tid=thread_index_in_threadgroup;
const int rowbase=threadgroup_position_in_grid.y*16*MT;
const int colbase=(threadgroup_position_in_grid.x*4+sg)*32;
const short qid=lane>>2;
const short fm=(qid&4)|((lane>>1)&3);
const short fn=((qid&2)|(lane&1))*4;
threadgroup half xa[MT*16*64];
threadgroup half weights[4*32*64];
threadgroup half* wb=weights+sg*32*64;
constexpr auto desc=matmul2d_descriptor(16,32,64,false,true,false,
    matmul2d_descriptor::mode::multiply);
matmul2d<desc,execution_simdgroup> op;
float sums[MT][16];
#pragma clang loop unroll(full)
for(int m=0;m<MT;m++)
#pragma clang loop unroll(full)
    for(int j=0;j<16;j++) sums[m][j]=0.0f;
for(int kb=0;kb<K;kb+=64) {
    for(int i=tid;i<MT*16*64;i+=128) {
        const int row=rowbase+i/64,k=kb+i%64;
        xa[i]=row<TOKENS ? half(x[ulong(row)*K+k]) : half(0.0f);  // x may already be half
    }
    // 32 weight rows x two 32-value chunks: two chunks per lane.
    // DIAG (timing diagnostics only, output meaningless): 1 skips the weight
    // decode after the first K step, 2 skips the MMA.
    if(DIAG!=1 || kb==0)
    for(int t=0;t<2;t++) {
        const int item=lane+32*t, r=item/2, part=item%2;
        const int row=colbase+r;
        threadgroup half *dst=wb+r*64+part*32;
        if(row<N) kblock_values32(w+ulong(row)*ROW_BYTES,uint(kb/32+part),dst,FORMAT);
        else for(int j=0;j<32;j++) dst[j]=half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tensor<threadgroup half,dextents<int32_t,2>,tensor_inline> b(wb,dextents<int32_t,2>(64,32));
#pragma clang loop unroll(full)
    for(int m=0;m<MT && DIAG!=2;m++) {
        tensor<threadgroup half,dextents<int32_t,2>,tensor_inline> a(xa+m*16*64,dextents<int32_t,2>(64,16));
        auto product=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
        op.run(a,b,product);
#pragma clang loop unroll(full)
        for(int j=0;j<16;j++) sums[m][j]+=product[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
#pragma clang loop unroll(full)
for(int m=0;m<MT;m++)
#pragma clang loop unroll(full)
for(int f=0;f<2;f++) {
#pragma clang loop unroll(full)
    for(int r=0;r<2;r++) {
        const int row=rowbase+m*16+fm+r*8,col=colbase+f*16+fn;
        if(row<TOKENS && col<N) {
#pragma clang loop unroll(full)
            for(int j=0;j<4;j++) y[ulong(row)*N+col+j]=sums[m][f*8+r*4+j];
        }
    }
}
"""


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="mlx2_gguf_kblock_nax_prefill_v1",
        input_names=["x", "w"],
        output_names=["y"],
        header=HEADER,
        source=SOURCE,
        ensure_row_contiguous=True,
    )


def kblock_nax_prefill(x, weight: PackedMatrix, *, mt=4, half_x=True, diag=0):
    """nax_prefill's FP16-operand law with MT-row-tile reuse and chunk-decoded tiles."""
    import mlx.core as mx

    if mx.default_device() != mx.gpu or "Apple M5" not in mx.device_info().get("device_name", ""):
        raise GGUFError("K-block NAX prefill requires an owned Apple M5 GPU window")
    if (x.ndim != 2 or len(weight.shape) != 2 or x.shape[-1] != weight.shape[-1]
            or weight.shape[0] % 32 or x.shape[-1] % 256 or mt not in (1, 2, 4)
            or weight.ggml_type not in (2, 11, 12, 13, 14)):
        raise GGUFError("K-block NAX prefill needs block-aligned rank2 K-block weights and MT 1/2/4")
    m, k = x.shape
    n = weight.shape[0]
    return _kernel()(
        # half(x) once (round to nearest even, as the kernel's half() does):
        # the tile reads half the activation bytes and skips per-tile casts.
        inputs=[x.astype(mx.float16 if half_x else mx.float32), weight.data],
        template=[("TOKENS", m), ("K", k), ("N", n), ("ROW_BYTES", weight.row_bytes),
                  ("FORMAT", weight.ggml_type), ("MT", mt), ("DIAG", diag)],
        grid=(((n + 127) // 128) * 128, (m + 16 * mt - 1) // (16 * mt), 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.float32],
    )[0]


# Tile v2 (MPP programming-guide M5 starting point): a 64x64 threadgroup tile
# as one execution_simdgroups<4> op, activations read in place from a half
# device tensor, K-block weight tiles chunk-decoded into a double-buffered
# threadgroup tile (one barrier per 64-wide K step), and accumulation with
# mode::multiply_accumulate into a persistent destination cooperative tensor.
# The accumulation form differs from nax_prefill's "multiply then +=", so v2
# is a candidate law with its own classification, not a bitwise replacement.
SOURCE_V2 = r"""
constexpr int TM = 64, TN = 64, BK = 64;
const int tid = int(thread_index_in_threadgroup);
const int row0 = int(threadgroup_position_in_grid.y) * TM;
const int col0 = int(threadgroup_position_in_grid.x) * TN;
constexpr auto desc = matmul2d_descriptor(TM, TN, BK, false, true, false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroups<4>> op;
tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(
    (device half*)x + int64_t(row0) * K, dextents<int32_t, 2>(K, TOKENS - row0));
threadgroup half wt[2][TN * BK];
auto a0 = tA.slice(0, 0);
tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> b0(wt[0], dextents<int32_t, 2>(BK, TN));
auto c = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
for (uint16_t j = 0; j < c.get_capacity(); ++j)
    if (c.is_valid_element(j)) c[j] = 0.0f;
// One 32-value chunk per thread per K step: weight row tid/2, half tid%2.
const int wr = tid / 2, part = tid % 2, wrow = col0 + wr;
device const uchar *wbase = w + ulong(wrow < N ? wrow : 0) * ROW_BYTES;
// PREFETCH: the raw block bytes of step k+2 are loaded into registers while
// step k's MMA runs and decoded at step k+1 (same values as kblock_values32).
// Default off: bit-identical but no faster here (i1-pf-dense), slightly
// slower in the MoE gate/up tile from register pressure (i2b-pf-moe).
constexpr bool PF = PREFETCH && FORMAT != 2;
KRaw raw;
if (wrow < N) kblock_values32(wbase, uint(part), wt[0] + wr * BK + part * 32, FORMAT);
else for (int j = 0; j < 32; j++) wt[0][wr * BK + part * 32 + j] = half(0.0f);
if (PF && wrow < N && BK < K) raw = kb_load(wbase, uint(BK / 32 + part), FORMAT);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int kb = 0, s = 0; kb < K; kb += BK, s ^= 1) {
    if (kb + BK < K) {
        threadgroup half *dst = wt[s ^ 1] + wr * BK + part * 32;
        if (wrow < N) {
            if (PF) {
                kb_decode(raw, uint((kb + BK) / 32 + part), dst, FORMAT);
                if (kb + 2 * BK < K) raw = kb_load(wbase, uint((kb + 2 * BK) / 32 + part), FORMAT);
            } else kblock_values32(wbase, uint((kb + BK) / 32 + part), dst, FORMAT);
        }
        else for (int j = 0; j < 32; j++) dst[j] = half(0.0f);
    }
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> b(wt[s], dextents<int32_t, 2>(BK, TN));
    auto a = tA.slice(kb, 0);
    op.run(a, b, c);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint16_t j = 0; j < c.get_capacity(); ++j) {
    if (c.is_valid_element(j)) {
        const auto index = c.get_multidimensional_index(j);
        const int col = col0 + index[0], row = row0 + index[1];
        if (col < N && row < TOKENS) y[ulong(row) * N + col] = c[j];
    }
}
"""


@lru_cache(maxsize=1)
def _kernel_v2():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="mlx2_gguf_kblock_nax_prefill_v2",
        input_names=["x", "w"],
        output_names=["y"],
        header=HEADER,
        source=SOURCE_V2,
        ensure_row_contiguous=True,
    )


def kblock_nax_prefill_v2(x, weight: PackedMatrix, *, prefetch=0):
    """Guide-shaped 64x64 K-block NAX tile; a candidate FP16-operand law."""
    import mlx.core as mx

    if mx.default_device() != mx.gpu or "Apple M5" not in mx.device_info().get("device_name", ""):
        raise GGUFError("K-block NAX prefill requires an owned Apple M5 GPU window")
    if (x.ndim != 2 or len(weight.shape) != 2 or x.shape[-1] != weight.shape[-1]
            or x.shape[-1] % 64 or weight.ggml_type not in (2, 11, 12, 13, 14)):
        raise GGUFError("K-block NAX prefill v2 needs rank2 K-block weights with K % 64 == 0")
    m, k = x.shape
    n = weight.shape[0]
    return _kernel_v2()(
        inputs=[x.astype(mx.float16), weight.data],
        template=[("TOKENS", m), ("K", k), ("N", n), ("ROW_BYTES", weight.row_bytes),
                  ("FORMAT", weight.ggml_type), ("PREFETCH", prefetch)],
        grid=(((n + 63) // 64) * 128, (m + 63) // 64, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.float32],
    )[0]


# Decode-once dense prefill (research). The v2 tile decodes every weight once
# per 64-row tile (16x for a 1K chunk). Here one pass writes each weight's
# kblock_values32 halves to a device scratch, and the same 64x64x64
# multiply_accumulate op reads B from device memory instead of threadgroup
# memory: identical operand values and op, so the output should equal v2's.
DECODE_FP16 = r"""
const uint tid = thread_index_in_threadgroup;
const uint gid = thread_position_in_grid.x;
threadgroup half buf[128 * 32];
const uint row = gid / (K / 32), chunk = gid % (K / 32);
if (row < N) {
    kblock_values32(w + ulong(row) * ROW_BYTES, chunk, buf + tid * 32, FORMAT);
    device half4 *o = (device half4 *)(out + ulong(row) * K + chunk * 32);
    threadgroup const half4 *v = (threadgroup const half4 *)(buf + tid * 32);
    for (int i = 0; i < 8; i++) o[i] = v[i];
}
"""

# Every element value of a K-block weight, exactly as the reference per-row
# kernel reads it (gguf_value), to float32: one thread per four consecutive
# values of a row. A one-off per prompt chunk for the projections that have no
# fast tile (grouped MLA k_b/v_b, KDA f_b/g_b at K = 128), whose matmul then
# runs on MLX's dense GEMM over the same element values.
DECODE_F32 = r"""
const uint gid = thread_position_in_grid.x;
const uint per_row = K / 4;
const uint row = gid / per_row, col = (gid % per_row) * 4;
if (row < N) {
    device const uchar *w_row = w + ulong(row) * ROW_BYTES;
    float4 v;
    v.x = gguf_value(w_row, col, FORMAT);
    v.y = gguf_value(w_row, col + 1, FORMAT);
    v.z = gguf_value(w_row, col + 2, FORMAT);
    v.w = gguf_value(w_row, col + 3, FORMAT);
    *((device float4 *)(out + ulong(row) * K + col)) = v;
}
"""

SOURCE_V2D = r"""
const int row0 = int(threadgroup_position_in_grid.y) * TM;
const int col0 = int(threadgroup_position_in_grid.x) * TN;
constexpr auto desc = matmul2d_descriptor(TM, TN, BK, false, true, false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroups<SG>> op;
tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(
    (device half*)x + int64_t(row0) * K, dextents<int32_t, 2>(K, TOKENS - row0));
tensor<device half, dextents<int32_t, 2>, tensor_inline> tB(
    (device half*)wd + int64_t(col0) * K, dextents<int32_t, 2>(K, N - col0));
auto a0 = tA.slice(0, 0);
auto b0 = tB.slice(0, 0);
auto c = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
for (uint16_t j = 0; j < c.get_capacity(); ++j)
    if (c.is_valid_element(j)) c[j] = 0.0f;
for (int kb = 0; kb < K; kb += BK) {
    auto a = tA.slice(kb, 0);
    auto b = tB.slice(kb, 0);
    op.run(a, b, c);
}
for (uint16_t j = 0; j < c.get_capacity(); ++j) {
    if (c.is_valid_element(j)) {
        const auto index = c.get_multidimensional_index(j);
        const int col = col0 + index[0], row = row0 + index[1];
        if (col < N && row < TOKENS) y[ulong(row) * N + col] = c[j];
    }
}
"""


@lru_cache(maxsize=1)
def _kernel_decode_fp16():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="mlx2_gguf_kblock_decode_fp16", input_names=["w"], output_names=["out"],
        header=HEADER, source=DECODE_FP16, ensure_row_contiguous=True)


@lru_cache(maxsize=1)
def _kernel_v2d():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="mlx2_gguf_kblock_nax_prefill_v2d", input_names=["x", "wd"], output_names=["y"],
        header=HEADER, source=SOURCE_V2D, ensure_row_contiguous=True)


@lru_cache(maxsize=1)
def _kernel_decode_f32():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="mlx2_gguf_kblock_decode_f32", input_names=["w"], output_names=["out"],
        header=HEADER, source=DECODE_F32, ensure_row_contiguous=True)


DECODED_F32_FORMATS = (2, 11, 12, 13, 14)


def kblock_decode_f32(weight: PackedMatrix):
    """Every value of a K-block weight as gguf_value computes it, float32, in
    the weight's own shape: rank2 (N, K) or grouped (G, N, K)."""
    import mlx.core as mx

    shape = tuple(weight.shape)
    k = shape[-1]
    n = math.prod(shape[:-1])
    block = 32 if weight.ggml_type == 2 else 256
    if len(shape) not in (2, 3) or weight.ggml_type not in DECODED_F32_FORMATS or k % block:
        raise GGUFError("K-block f32 decode needs rank2/3 K-block weights of whole blocks")
    threads = n * (k // 4)
    out = _kernel_decode_f32()(
        inputs=[weight.data],
        template=[("K", k), ("N", n), ("ROW_BYTES", weight.row_bytes), ("FORMAT", weight.ggml_type)],
        grid=(((threads + 255) // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(n, k)], output_dtypes=[mx.float32])[0]
    return out.reshape(*shape)


def kblock_decode_fp16(weight: PackedMatrix):
    """Every value of a K-block weight as kblock_values32 computes it, half, in
    the weight's own shape: rank2 (N, K) or grouped (G, N, K)."""
    import mlx.core as mx

    shape = tuple(weight.shape)
    if len(shape) not in (2, 3) or shape[-1] % 64 or weight.ggml_type not in (2, 11, 12, 13, 14):
        raise GGUFError("K-block fp16 decode needs rank2/3 K-block weights with K % 64 == 0")
    k = shape[-1]
    n = math.prod(shape[:-1])
    threads = n * (k // 32)
    out = _kernel_decode_fp16()(
        inputs=[weight.data],
        template=[("K", k), ("N", n), ("ROW_BYTES", weight.row_bytes), ("FORMAT", weight.ggml_type)],
        grid=(((threads + 127) // 128) * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(n, k)], output_dtypes=[mx.float16])[0]
    return out.reshape(*shape)


def kblock_nax_prefill_v2d(x, weight: PackedMatrix, *, decoded=None, tm=64, tn=64, sg=4):
    """Decode-once v2: device fp16 scratch, then the v2 op with B read in place."""
    import mlx.core as mx

    if mx.default_device() != mx.gpu or "Apple M5" not in mx.device_info().get("device_name", ""):
        raise GGUFError("K-block NAX prefill requires an owned Apple M5 GPU window")
    if x.ndim != 2 or len(weight.shape) != 2 or x.shape[-1] != weight.shape[-1]:
        raise GGUFError("K-block NAX prefill v2d needs rank2 operands")
    wd = kblock_decode_fp16(weight) if decoded is None else decoded
    m, k = x.shape
    n = weight.shape[0]
    return _kernel_v2d()(
        inputs=[x.astype(mx.float16), wd],
        template=[("TOKENS", m), ("K", k), ("N", n), ("TM", tm), ("TN", tn), ("BK", 64), ("SG", sg)],
        grid=(((n + tn - 1) // tn) * 32 * sg, (m + tm - 1) // tm, 1),
        threadgroup=(32 * sg, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.float32],
    )[0]


# Segmented routed-expert prefill, tile v2. The GPU plan (segmented_prefill.
# build_plan, 32-row tiles) groups assignments by expert; each threadgroup
# owns one (expert, <=32 rows) tile and 64 output columns as one
# execution_simdgroups<4> 32x64x64 op with multiply_accumulate. Gate/up gather
# their activation rows through the plan order into threadgroup memory and
# decode the gate and up weight tiles each K step; down reads the sorted mid
# rows in place. Same FP16-operand law as segmented_nax_fp16 with the v2
# accumulation form (classified separately).
SEG_GATE_UP_V2 = r"""
constexpr int TM = 32, TN = 64, BK = 64;
const int tid = int(thread_index_in_threadgroup);
const uint tile = threadgroup_position_in_grid.y;
if (tile >= tile_count[0]) return;
const uint4 d = *((device const uint4*)tiles + tile);
const uint rowbase = d.x, expert = d.y, valid = d.z;
const int col0 = int(threadgroup_position_in_grid.x) * TN;
constexpr auto desc = matmul2d_descriptor(TM, TN, BK, false, true, false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroups<4>> op;
threadgroup half xa[TM * BK];
threadgroup half wg[TN * BK];
threadgroup half wu[TN * BK];
tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> a(xa, dextents<int32_t, 2>(BK, TM));
tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> bg(wg, dextents<int32_t, 2>(BK, TN));
tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> bu(wu, dextents<int32_t, 2>(BK, TN));
auto cg = op.template get_destination_cooperative_tensor<decltype(a), decltype(bg), float>();
auto cu = op.template get_destination_cooperative_tensor<decltype(a), decltype(bu), float>();
for (uint16_t j = 0; j < cg.get_capacity(); ++j)
    if (cg.is_valid_element(j)) { cg[j] = 0.0f; cu[j] = 0.0f; }
const int wr = tid / 2, part = tid % 2, wrow = col0 + wr;
const size_t woff = (size_t(expert) * N + size_t(wrow < N ? wrow : 0)) * ROW_BYTES;
for (int kb = 0; kb < K; kb += BK) {
    // DIAG (timing only): 1 skips weight decode, 2 skips the MMAs, 3 skips the gather.
    if (DIAG != 3 || kb == 0)
    for (int i = tid; i < TM * BK; i += 128) {
        const uint r = uint(i / BK); const int k = kb + i % BK;
        const uint token = r < valid ? order[rowbase + r] / TOPK : 0;
        xa[i] = r < valid ? half(x[ulong(token) * K + k]) : half(0.0f);
    }
    threadgroup half *dg = wg + wr * BK + part * 32, *du = wu + wr * BK + part * 32;
    if (DIAG == 1 && kb > 0) {}
    else if (wrow < N) {
        kblock_values32(gate + woff, uint(kb / 32 + part), dg, FORMAT);
        kblock_values32(up + woff, uint(kb / 32 + part), du, FORMAT);
    } else for (int j = 0; j < 32; j++) { dg[j] = half(0.0f); du[j] = half(0.0f); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (DIAG != 2) {
        op.run(a, bg, cg);
        op.run(a, bu, cu);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint16_t j = 0; j < cg.get_capacity(); ++j) {
    if (cg.is_valid_element(j)) {
        const auto index = cg.get_multidimensional_index(j);
        const int col = col0 + index[0]; const uint row = uint(index[1]);
        if (col < N && row < valid) {
            const float g = min(cg[j], 10.0f);
            const float u = clamp(cu[j], -10.0f, 10.0f);
            y[ulong(rowbase + row) * N + col] = (g / (1.0f + metal::precise::exp(-g))) * u;
        }
    }
}
"""

SEG_DOWN_V2 = r"""
constexpr int TM = TMT, TN = 64, BK = 64;
const int tid = int(thread_index_in_threadgroup);
const uint tile = threadgroup_position_in_grid.y;
if (tile >= tile_count[0]) return;
const uint4 d = *((device const uint4*)tiles + tile);
const uint rowbase = d.x, expert = d.y, valid = d.z;
const int col0 = int(threadgroup_position_in_grid.x) * TN;
constexpr auto desc = matmul2d_descriptor(TM, TN, BK, false, true, false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroups<4>> op;
tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(
    (device half*)x + int64_t(rowbase) * K, dextents<int32_t, 2>(K, int(valid)));
threadgroup half wt[2][TN * BK];
auto a0 = tA.slice(0, 0);
tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> b0(wt[0], dextents<int32_t, 2>(BK, TN));
auto c = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
for (uint16_t j = 0; j < c.get_capacity(); ++j)
    if (c.is_valid_element(j)) c[j] = 0.0f;
const int wr = tid / 2, part = tid % 2, wrow = col0 + wr;
const size_t woff = (size_t(expert) * N + size_t(wrow < N ? wrow : 0)) * ROW_BYTES;
if (wrow < N) kblock_values32(w + woff, uint(part), wt[0] + wr * BK + part * 32, FORMAT);
else for (int j = 0; j < 32; j++) wt[0][wr * BK + part * 32 + j] = half(0.0f);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int kb = 0, s = 0; kb < K; kb += BK, s ^= 1) {
    if (kb + BK < K) {
        threadgroup half *dst = wt[s ^ 1] + wr * BK + part * 32;
        if (wrow < N) kblock_values32(w + woff, uint((kb + BK) / 32 + part), dst, FORMAT);
        else for (int j = 0; j < 32; j++) dst[j] = half(0.0f);
    }
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> b(wt[s], dextents<int32_t, 2>(BK, TN));
    auto a = tA.slice(kb, 0);
    op.run(a, b, c);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint16_t j = 0; j < c.get_capacity(); ++j) {
    if (c.is_valid_element(j)) {
        const auto index = c.get_multidimensional_index(j);
        const int col = col0 + index[0]; const uint row = uint(index[1]);
        if (col < N && row < valid) y[ulong(rowbase + row) * N + col] = c[j];
    }
}
"""


@lru_cache(maxsize=2)
def _segmented_kernel_v2(kind):
    import mlx.core as mx

    sources = {
        "gate_up": (["x", "order", "tiles", "tile_count", "gate", "up"], SEG_GATE_UP_V2),
        "down": (["x", "tiles", "tile_count", "w"], SEG_DOWN_V2),
    }
    inputs, source = sources[kind]
    return mx.fast.metal_kernel(
        name="mlx2_gguf_kblock_segmented_" + kind + "_v2",
        input_names=inputs, output_names=["y"], header=HEADER, source=source,
        ensure_row_contiguous=True,
    )


def kblock_segmented_moe_prefill_v2(x, ids, scores, gate, up, down, *, max_scratch_bytes=512 << 20, diag=0):
    """Expert-grouped K-block prefill with the v2 tile; segmented plan and combine."""
    import mlx.core as mx

    from . import segmented_prefill as seg

    if mx.default_device() != mx.gpu or "Apple M5" not in mx.device_info().get("device_name", ""):
        raise GGUFError("K-block segmented prefill requires an owned Apple M5 GPU window")
    experts, hidden, k = seg._validate(x, ids, scores, gate, up, down, max_scratch_bytes)
    if gate.ggml_type not in (11, 12, 13, 14) or down.ggml_type not in (11, 12, 13, 14):
        raise GGUFError("K-block segmented prefill v2 covers Q3_K/Q4_K/Q5_K/Q6_K experts")
    if ids.dtype != mx.uint32:
        raise GGUFError("segmented expert IDs must be uint32")
    plan = seg.build_plan(ids, experts, 32)
    mid = _segmented_kernel_v2("gate_up")(
        inputs=[x.astype(mx.float16), plan.order, plan.tiles, plan.tile_count, gate.data, up.data],
        template=[("TOPK", ids.shape[1]), ("K", k), ("N", hidden), ("ROW_BYTES", gate.row_bytes),
                  ("FORMAT", gate.ggml_type), ("DIAG", diag)],
        grid=(((hidden + 63) // 64) * 128, plan.max_tiles, 1), threadgroup=(128, 1, 1),
        output_shapes=[(plan.assignment_rows, hidden)], output_dtypes=[mx.float32],
    )[0]
    projected = _segmented_kernel_v2("down")(
        # half(mid) once, the rounding segmented down applies per tile.
        inputs=[mid.astype(mx.float16), plan.tiles, plan.tile_count, down.data],
        template=[("K", hidden), ("N", k), ("ROW_BYTES", down.row_bytes), ("FORMAT", down.ggml_type),
                  ("TMT", 32)],
        grid=(((k + 63) // 64) * 128, plan.max_tiles, 1), threadgroup=(128, 1, 1),
        output_shapes=[(plan.assignment_rows, k)], output_dtypes=[mx.float32],
    )[0]
    return seg._kernel("combine")(
        inputs=[projected, plan.inverse, scores.astype(mx.float32).reshape(-1), plan.status],
        template=[("TOKENS", x.shape[0]), ("TOPK", ids.shape[1]), ("N", k)],
        grid=(((k + 255) // 256) * 256, x.shape[0], 1), threadgroup=(256, 1, 1),
        output_shapes=[(x.shape[0], k)], output_dtypes=[mx.float32],
    )[0]


def segmented_gate_up_v2_only(x, ids, gate, up, *, diag=0):
    """Timing hook: the v2 gate/up launch alone (plan included)."""
    import mlx.core as mx

    from . import segmented_prefill as seg

    experts, hidden, k = gate.shape
    plan = seg.build_plan(ids, experts, 32)
    return _segmented_kernel_v2("gate_up")(
        inputs=[x.astype(mx.float16), plan.order, plan.tiles, plan.tile_count, gate.data, up.data],
        template=[("TOPK", ids.shape[1]), ("K", k), ("N", hidden), ("ROW_BYTES", gate.row_bytes),
                  ("FORMAT", gate.ggml_type), ("DIAG", diag)],
        grid=(((hidden + 63) // 64) * 128, plan.max_tiles, 1), threadgroup=(128, 1, 1),
        output_shapes=[(plan.assignment_rows, hidden)], output_dtypes=[mx.float32],
    )[0]


# Gate/up v3: activation rows pre-gathered in plan order (one MLX take per
# layer), read in place as a half device tensor like the down kernel; gate and
# up weight tiles double-buffered with one barrier per K step. Same per-output
# arithmetic as v2 (the activation values and MMA chain are unchanged).
SEG_GATE_UP_V3 = r"""
constexpr int TM = TMT, TN = TNT, BK = 64;
const int tid = int(thread_index_in_threadgroup);
const uint tile = threadgroup_position_in_grid.y;
if (tile >= tile_count[0]) return;
const uint4 d = *((device const uint4*)tiles + tile);
const uint rowbase = d.x, expert = d.y, valid = d.z;
const int col0 = int(threadgroup_position_in_grid.x) * TN;
constexpr auto desc = matmul2d_descriptor(TM, TN, BK, false, true, false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroups<4>> op;
tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(
    (device half*)xs + int64_t(rowbase) * K, dextents<int32_t, 2>(K, int(valid)));
threadgroup half wg[2][TN * BK];
threadgroup half wu[2][TN * BK];
auto a0 = tA.slice(0, 0);
tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> b0(wg[0], dextents<int32_t, 2>(BK, TN));
auto cg = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
auto cu = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
for (uint16_t j = 0; j < cg.get_capacity(); ++j)
    if (cg.is_valid_element(j)) { cg[j] = 0.0f; cu[j] = 0.0f; }
// TN=64: each thread decodes one gate and one up chunk per K step (row tid/2).
// TN=32: threads 0-63 decode gate chunks and 64-127 up chunks, one each.
const bool split = TN == 32;
const int lt = split ? tid % 64 : tid;
const int wr = lt / 2, part = lt % 2, wrow = col0 + wr;
const bool do_gate = !split || tid < 64, do_up = !split || tid >= 64;
const size_t woff = (size_t(expert) * N + size_t(wrow < N ? wrow : 0)) * ROW_BYTES;
if (wrow < N) {
    if (do_gate) kblock_values32(gate + woff, uint(part), wg[0] + wr * BK + part * 32, FORMAT);
    if (do_up) kblock_values32(up + woff, uint(part), wu[0] + wr * BK + part * 32, FORMAT);
} else for (int j = 0; j < 32; j++) {
    if (do_gate) wg[0][wr * BK + part * 32 + j] = half(0.0f);
    if (do_up) wu[0][wr * BK + part * 32 + j] = half(0.0f);
}
// PREFETCH: raw bytes of step k+2 load into registers during step k's MMAs.
constexpr bool PF = PREFETCH && !split;
KRaw rg, ru;
if (PF && wrow < N && BK < K) {
    rg = kb_load(gate + woff, uint(BK / 32 + part), FORMAT);
    ru = kb_load(up + woff, uint(BK / 32 + part), FORMAT);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int kb = 0, s = 0; kb < K; kb += BK, s ^= 1) {
    // DIAG (timing only): 1 stops decoding after the first K step (the MMAs
    // keep consuming the stale tiles, so nothing is dead-code eliminated).
    if (kb + BK < K && DIAG != 1) {
        threadgroup half *dg = wg[s ^ 1] + wr * BK + part * 32, *du = wu[s ^ 1] + wr * BK + part * 32;
        if (wrow < N && PF) {
            kb_decode(rg, uint((kb + BK) / 32 + part), dg, FORMAT);
            kb_decode(ru, uint((kb + BK) / 32 + part), du, FORMAT);
            if (kb + 2 * BK < K) {
                rg = kb_load(gate + woff, uint((kb + 2 * BK) / 32 + part), FORMAT);
                ru = kb_load(up + woff, uint((kb + 2 * BK) / 32 + part), FORMAT);
            }
        } else if (wrow < N) {
            if (do_gate) kblock_values32(gate + woff, uint((kb + BK) / 32 + part), dg, FORMAT);
            if (do_up) kblock_values32(up + woff, uint((kb + BK) / 32 + part), du, FORMAT);
        } else for (int j = 0; j < 32; j++) {
            if (do_gate) dg[j] = half(0.0f);
            if (do_up) du[j] = half(0.0f);
        }
    }
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> bg(wg[s], dextents<int32_t, 2>(BK, TN));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> bu(wu[s], dextents<int32_t, 2>(BK, TN));
    auto a = tA.slice(kb, 0);
    op.run(a, bg, cg);
    op.run(a, bu, cu);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint16_t j = 0; j < cg.get_capacity(); ++j) {
    if (cg.is_valid_element(j)) {
        const auto index = cg.get_multidimensional_index(j);
        const int col = col0 + index[0]; const uint row = uint(index[1]);
        if (col < N && row < valid) {
            const float g = min(cg[j], 10.0f);
            const float u = clamp(cu[j], -10.0f, 10.0f);
            y[ulong(rowbase + row) * N + col] = (g / (1.0f + metal::precise::exp(-g))) * u;
        }
    }
}
"""


@lru_cache(maxsize=1)
def _segmented_gate_up_v3():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="mlx2_gguf_kblock_segmented_gate_up_v3",
        input_names=["xs", "tiles", "tile_count", "gate", "up"], output_names=["y"],
        header=HEADER, source=SEG_GATE_UP_V3, ensure_row_contiguous=True,
    )


def kblock_segmented_moe_prefill_v3(x, ids, scores, gate, up, down, *, max_scratch_bytes=512 << 20, tm=32,
                                    stages=None, gate_up_tn=64, diag=0, prefetch=0):
    """v2 with pre-gathered activations and double-buffered gate/up weight tiles."""
    import mlx.core as mx

    from . import segmented_prefill as seg

    if mx.default_device() != mx.gpu or "Apple M5" not in mx.device_info().get("device_name", ""):
        raise GGUFError("K-block segmented prefill requires an owned Apple M5 GPU window")
    experts, hidden, k = seg._validate(x, ids, scores, gate, up, down, max_scratch_bytes)
    if gate.ggml_type not in (11, 12, 13, 14) or down.ggml_type not in (11, 12, 13, 14):
        raise GGUFError("K-block segmented prefill v3 covers Q3_K/Q4_K/Q5_K/Q6_K experts")
    if ids.dtype != mx.uint32:
        raise GGUFError("segmented expert IDs must be uint32")
    if tm not in (16, 32, 64):
        raise GGUFError("segmented v3 tile rows must be 16, 32 or 64")
    plan = seg.build_plan(ids, experts, tm)
    # Activation row of every assignment, in plan (expert-sorted) order.
    xs = mx.take(x.astype(mx.float16), plan.order // ids.shape[1], axis=0)
    if stages is not None:
        stages["plan"], stages["xs"] = plan, xs
    mid = _segmented_gate_up_v3()(
        inputs=[xs, plan.tiles, plan.tile_count, gate.data, up.data],
        template=[("K", k), ("N", hidden), ("ROW_BYTES", gate.row_bytes), ("FORMAT", gate.ggml_type),
                  ("TMT", tm), ("TNT", gate_up_tn), ("DIAG", diag), ("PREFETCH", prefetch)],
        grid=(((hidden + gate_up_tn - 1) // gate_up_tn) * 128, plan.max_tiles, 1), threadgroup=(128, 1, 1),
        output_shapes=[(plan.assignment_rows, hidden)], output_dtypes=[mx.float32],
    )[0]
    if stages is not None:
        stages["mid"] = mid
    projected = _segmented_kernel_v2("down")(
        inputs=[mid.astype(mx.float16), plan.tiles, plan.tile_count, down.data],
        template=[("K", hidden), ("N", k), ("ROW_BYTES", down.row_bytes), ("FORMAT", down.ggml_type),
                  ("TMT", tm)],
        grid=(((k + 63) // 64) * 128, plan.max_tiles, 1), threadgroup=(128, 1, 1),
        output_shapes=[(plan.assignment_rows, k)], output_dtypes=[mx.float32],
    )[0]
    return seg._kernel("combine")(
        inputs=[projected, plan.inverse, scores.astype(mx.float32).reshape(-1), plan.status],
        template=[("TOKENS", x.shape[0]), ("TOPK", ids.shape[1]), ("N", k)],
        grid=(((k + 255) // 256) * 256, x.shape[0], 1), threadgroup=(256, 1, 1),
        output_shapes=[(x.shape[0], k)], output_dtypes=[mx.float32],
    )[0]


# v4: decode sharing across row sub-tiles. The plan uses 64-row tiles per
# expert; each threadgroup decodes a weight tile once per K step and runs two
# 32-row sub-tile MMAs against it (the second only when the tile has more
# than 32 rows, a threadgroup-uniform branch). Every output keeps v3 TM=32's
# 32x64x64 multiply_accumulate chain, so outputs should equal v3's bit for bit
# while popular experts decode their weights half as often.
SEG_GATE_UP_V4 = r"""
constexpr int TM = 32, TN = 64, BK = 64;
const int tid = int(thread_index_in_threadgroup);
const uint tile = threadgroup_position_in_grid.y;
if (tile >= tile_count[0]) return;
const uint4 d = *((device const uint4*)tiles + tile);
const uint rowbase = d.x, expert = d.y, valid = d.z;
const bool second = valid > uint(TM);
const int col0 = int(threadgroup_position_in_grid.x) * TN;
constexpr auto desc = matmul2d_descriptor(TM, TN, BK, false, true, false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroups<4>> op;
tensor<device half, dextents<int32_t, 2>, tensor_inline> tA0(
    (device half*)xs + int64_t(rowbase) * K, dextents<int32_t, 2>(K, int(min(valid, uint(TM)))));
tensor<device half, dextents<int32_t, 2>, tensor_inline> tA1(
    (device half*)xs + int64_t(rowbase + TM) * K, dextents<int32_t, 2>(K, int(second ? valid - TM : 1u)));
threadgroup half wg[2][TN * BK];
threadgroup half wu[2][TN * BK];
auto a00 = tA0.slice(0, 0);
tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> b0(wg[0], dextents<int32_t, 2>(BK, TN));
auto cg0 = op.template get_destination_cooperative_tensor<decltype(a00), decltype(b0), float>();
auto cu0 = op.template get_destination_cooperative_tensor<decltype(a00), decltype(b0), float>();
auto cg1 = op.template get_destination_cooperative_tensor<decltype(a00), decltype(b0), float>();
auto cu1 = op.template get_destination_cooperative_tensor<decltype(a00), decltype(b0), float>();
for (uint16_t j = 0; j < cg0.get_capacity(); ++j)
    if (cg0.is_valid_element(j)) { cg0[j] = 0.0f; cu0[j] = 0.0f; cg1[j] = 0.0f; cu1[j] = 0.0f; }
const int wr = tid / 2, part = tid % 2, wrow = col0 + wr;
const size_t woff = (size_t(expert) * N + size_t(wrow < N ? wrow : 0)) * ROW_BYTES;
if (wrow < N) {
    kblock_values32(gate + woff, uint(part), wg[0] + wr * BK + part * 32, FORMAT);
    kblock_values32(up + woff, uint(part), wu[0] + wr * BK + part * 32, FORMAT);
} else for (int j = 0; j < 32; j++) { wg[0][wr * BK + part * 32 + j] = half(0.0f); wu[0][wr * BK + part * 32 + j] = half(0.0f); }
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int kb = 0, s = 0; kb < K; kb += BK, s ^= 1) {
    if (kb + BK < K) {
        threadgroup half *dg = wg[s ^ 1] + wr * BK + part * 32, *du = wu[s ^ 1] + wr * BK + part * 32;
        if (wrow < N) {
            kblock_values32(gate + woff, uint((kb + BK) / 32 + part), dg, FORMAT);
            kblock_values32(up + woff, uint((kb + BK) / 32 + part), du, FORMAT);
        } else for (int j = 0; j < 32; j++) { dg[j] = half(0.0f); du[j] = half(0.0f); }
    }
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> bg(wg[s], dextents<int32_t, 2>(BK, TN));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> bu(wu[s], dextents<int32_t, 2>(BK, TN));
    auto a0 = tA0.slice(kb, 0);
    op.run(a0, bg, cg0);
    op.run(a0, bu, cu0);
    if (second) {
        auto a1 = tA1.slice(kb, 0);
        op.run(a1, bg, cg1);
        op.run(a1, bu, cu1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint16_t j = 0; j < cg0.get_capacity(); ++j) {
    if (cg0.is_valid_element(j)) {
        const auto index = cg0.get_multidimensional_index(j);
        const int col = col0 + index[0]; const uint row = uint(index[1]);
        if (col < N && row < min(valid, uint(TM))) {
            const float g = min(cg0[j], 10.0f);
            const float u = clamp(cu0[j], -10.0f, 10.0f);
            y[ulong(rowbase + row) * N + col] = (g / (1.0f + metal::precise::exp(-g))) * u;
        }
        if (second && col < N && row + TM < valid) {
            const float g = min(cg1[j], 10.0f);
            const float u = clamp(cu1[j], -10.0f, 10.0f);
            y[ulong(rowbase + TM + row) * N + col] = (g / (1.0f + metal::precise::exp(-g))) * u;
        }
    }
}
"""

SEG_DOWN_V4 = r"""
constexpr int TM = 32, TN = 64, BK = 64;
const int tid = int(thread_index_in_threadgroup);
const uint tile = threadgroup_position_in_grid.y;
if (tile >= tile_count[0]) return;
const uint4 d = *((device const uint4*)tiles + tile);
const uint rowbase = d.x, expert = d.y, valid = d.z;
const bool second = valid > uint(TM);
const int col0 = int(threadgroup_position_in_grid.x) * TN;
constexpr auto desc = matmul2d_descriptor(TM, TN, BK, false, true, false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroups<4>> op;
tensor<device half, dextents<int32_t, 2>, tensor_inline> tA0(
    (device half*)x + int64_t(rowbase) * K, dextents<int32_t, 2>(K, int(min(valid, uint(TM)))));
tensor<device half, dextents<int32_t, 2>, tensor_inline> tA1(
    (device half*)x + int64_t(rowbase + TM) * K, dextents<int32_t, 2>(K, int(second ? valid - TM : 1u)));
threadgroup half wt[2][TN * BK];
auto a00 = tA0.slice(0, 0);
tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> b0(wt[0], dextents<int32_t, 2>(BK, TN));
auto c0 = op.template get_destination_cooperative_tensor<decltype(a00), decltype(b0), float>();
auto c1 = op.template get_destination_cooperative_tensor<decltype(a00), decltype(b0), float>();
for (uint16_t j = 0; j < c0.get_capacity(); ++j)
    if (c0.is_valid_element(j)) { c0[j] = 0.0f; c1[j] = 0.0f; }
const int wr = tid / 2, part = tid % 2, wrow = col0 + wr;
const size_t woff = (size_t(expert) * N + size_t(wrow < N ? wrow : 0)) * ROW_BYTES;
if (wrow < N) kblock_values32(w + woff, uint(part), wt[0] + wr * BK + part * 32, FORMAT);
else for (int j = 0; j < 32; j++) wt[0][wr * BK + part * 32 + j] = half(0.0f);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int kb = 0, s = 0; kb < K; kb += BK, s ^= 1) {
    if (kb + BK < K) {
        threadgroup half *dst = wt[s ^ 1] + wr * BK + part * 32;
        if (wrow < N) kblock_values32(w + woff, uint((kb + BK) / 32 + part), dst, FORMAT);
        else for (int j = 0; j < 32; j++) dst[j] = half(0.0f);
    }
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> b(wt[s], dextents<int32_t, 2>(BK, TN));
    auto a0 = tA0.slice(kb, 0);
    op.run(a0, b, c0);
    if (second) {
        auto a1 = tA1.slice(kb, 0);
        op.run(a1, b, c1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint16_t j = 0; j < c0.get_capacity(); ++j) {
    if (c0.is_valid_element(j)) {
        const auto index = c0.get_multidimensional_index(j);
        const int col = col0 + index[0]; const uint row = uint(index[1]);
        if (col < N && row < min(valid, uint(TM))) y[ulong(rowbase + row) * N + col] = c0[j];
        if (second && col < N && row + TM < valid) y[ulong(rowbase + TM + row) * N + col] = c1[j];
    }
}
"""


@lru_cache(maxsize=2)
def _segmented_kernel_v4(kind):
    import mlx.core as mx

    sources = {
        "gate_up": (["xs", "tiles", "tile_count", "gate", "up"], SEG_GATE_UP_V4),
        "down": (["x", "tiles", "tile_count", "w"], SEG_DOWN_V4),
    }
    inputs, source = sources[kind]
    return mx.fast.metal_kernel(
        name="mlx2_gguf_kblock_segmented_" + kind + "_v4",
        input_names=inputs, output_names=["y"], header=HEADER, source=source,
        ensure_row_contiguous=True,
    )


def kblock_segmented_moe_prefill_v4(x, ids, scores, gate, up, down, *, max_scratch_bytes=512 << 20,
                                    share_down=True):
    """v3 with weight-tile decode shared by two 32-row sub-tiles of one expert."""
    import mlx.core as mx

    from . import segmented_prefill as seg

    if mx.default_device() != mx.gpu or "Apple M5" not in mx.device_info().get("device_name", ""):
        raise GGUFError("K-block segmented prefill requires an owned Apple M5 GPU window")
    experts, hidden, k = seg._validate(x, ids, scores, gate, up, down, max_scratch_bytes)
    if gate.ggml_type not in (11, 12, 13, 14) or down.ggml_type not in (11, 12, 13, 14):
        raise GGUFError("K-block segmented prefill v4 covers Q3_K/Q4_K/Q5_K/Q6_K experts")
    if ids.dtype != mx.uint32:
        raise GGUFError("segmented expert IDs must be uint32")
    plan = seg.build_plan(ids, experts, 64)
    xs = mx.take(x.astype(mx.float16), plan.order // ids.shape[1], axis=0)
    mid = _segmented_kernel_v4("gate_up")(
        inputs=[xs, plan.tiles, plan.tile_count, gate.data, up.data],
        template=[("K", k), ("N", hidden), ("ROW_BYTES", gate.row_bytes), ("FORMAT", gate.ggml_type)],
        grid=(((hidden + 63) // 64) * 128, plan.max_tiles, 1), threadgroup=(128, 1, 1),
        output_shapes=[(plan.assignment_rows, hidden)], output_dtypes=[mx.float32],
    )[0]
    if share_down:
        projected = _segmented_kernel_v4("down")(
            inputs=[mid.astype(mx.float16), plan.tiles, plan.tile_count, down.data],
            template=[("K", hidden), ("N", k), ("ROW_BYTES", down.row_bytes), ("FORMAT", down.ggml_type)],
            grid=(((k + 63) // 64) * 128, plan.max_tiles, 1), threadgroup=(128, 1, 1),
            output_shapes=[(plan.assignment_rows, k)], output_dtypes=[mx.float32],
        )[0]
    else:
        plan32 = seg.build_plan(ids, experts, 32)
        projected = _segmented_kernel_v2("down")(
            inputs=[mid.astype(mx.float16), plan32.tiles, plan32.tile_count, down.data],
            template=[("K", hidden), ("N", k), ("ROW_BYTES", down.row_bytes), ("FORMAT", down.ggml_type),
                      ("TMT", 32)],
            grid=(((k + 63) // 64) * 128, plan32.max_tiles, 1), threadgroup=(128, 1, 1),
            output_shapes=[(plan32.assignment_rows, k)], output_dtypes=[mx.float32],
        )[0]
    return seg._kernel("combine")(
        inputs=[projected, plan.inverse, scores.astype(mx.float32).reshape(-1), plan.status],
        template=[("TOKENS", x.shape[0]), ("TOPK", ids.shape[1]), ("N", k)],
        grid=(((k + 255) // 256) * 256, x.shape[0], 1), threadgroup=(256, 1, 1),
        output_shapes=[(x.shape[0], k)], output_dtypes=[mx.float32],
    )[0]


# W8A8 routed gate/up for Q3_K experts (Kapitsa #311's integer-operand pattern).
# A Q3_K weight is d * (sc*q) with sc*q in [-124, 128]; the negation
# -(sc*q) in [-128, 124] always fits int8, so the weight operand is exact and
# the per-256 super-block fold multiplies by -d. Activations are per-row
# symmetric int8 (int8_prefill._quantize_rows): the only approximation, so this
# is an APPROXIMATE law. int8 x int8 -> int32 multiply_accumulate over the four
# 64-wide K steps of a super-block, then FP32 c += (-d[col]) * acc per block,
# times the row scale in the epilogue.
W8A8_HEADER = HEADER + r"""
inline void q3k_negints32(device const uchar *row, uint chunk, threadgroup char *out) {
    device const uchar *p=row+(chunk/8)*110;
    const uint c=chunk%8;
    int sc[2];
    #pragma unroll
    for(uint h=0;h<2;h++) {
        const uint g=2*c+h;
        const uint lo=(p[96+g%8]>>(4*(g/8)))&15;
        const uint hi=(p[104+g%4]>>(2*(g/4)))&3;
        sc[h]=int(lo|(hi<<4))-32;
    }
    device const ushort *qs=(device const ushort*)(p+32+(c/4)*32);
    device const ushort *hm=(device const ushort*)p;
    const uint shift=2*(c%4);
    #pragma unroll
    for(uint w=0;w<16;w++) {
        const uint qw=qs[w], hw=hm[w];
        #pragma unroll
        for(uint e=0;e<2;e++) {
            const uint j=2*w+e;
            const int q=int((qw>>(8*e+shift))&3)-(((hw>>(8*e+c))&1) ? 0 : 4);
            out[j]=char(-(sc[j/16]*q));
        }
    }
}
"""

SEG_GATE_UP_W8A8 = r"""
constexpr int TM = 32, TN = 64, BK = 64;
const int tid = int(thread_index_in_threadgroup);
const uint tile = threadgroup_position_in_grid.y;
if (tile >= tile_count[0]) return;
const uint4 d4 = *((device const uint4*)tiles + tile);
const uint rowbase = d4.x, expert = d4.y, valid = d4.z;
const int col0 = int(threadgroup_position_in_grid.x) * TN;
constexpr auto desc = matmul2d_descriptor(TM, TN, BK, false, true, false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroups<4>> op;
tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tA(
    (device int8_t*)xq + int64_t(rowbase) * K, dextents<int32_t, 2>(K, int(valid)));
threadgroup char wg[2][TN * BK];
threadgroup char wu[2][TN * BK];
threadgroup float dg[2][TN], du[2][TN];
auto a0 = tA.slice(0, 0);
tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> bg0((threadgroup int8_t*)wg[0], dextents<int32_t, 2>(BK, TN));
auto ig = op.template get_destination_cooperative_tensor<decltype(a0), decltype(bg0), int32_t>();
auto iu = op.template get_destination_cooperative_tensor<decltype(a0), decltype(bg0), int32_t>();
// 32x64 outputs over 128 threads: 16 per thread; static size keeps the folds in registers.
constexpr int CAP = TM * TN / 128;
float fg[CAP], fu[CAP];
#pragma unroll
for (int j = 0; j < CAP; ++j) { fg[j] = 0.0f; fu[j] = 0.0f; if (ig.is_valid_element(j)) { ig[j] = 0; iu[j] = 0; } }
const int wr = tid / 2, part = tid % 2, wrow = col0 + wr;
const size_t woff = (size_t(expert) * N + size_t(wrow < N ? wrow : 0)) * ROW_BYTES;
// Tiles and the super-block scale d are double-buffered: step k+1 decodes
// while step k multiplies; one barrier per K step.
auto stage = [&](int kb, int slot) {
    const uint chunk = uint(kb / 32 + part);
    if (wrow < N) {
        q3k_negints32(gate + woff, chunk, wg[slot] + wr * BK + part * 32);
        q3k_negints32(up + woff, chunk, wu[slot] + wr * BK + part * 32);
        if (part == 0) {
            dg[slot][wr] = gguf_k_half(gate + woff + (chunk / 8) * 110 + 108);
            du[slot][wr] = gguf_k_half(up + woff + (chunk / 8) * 110 + 108);
        }
    } else {
        for (int j = 0; j < 32; j++) { wg[slot][wr * BK + part * 32 + j] = 0; wu[slot][wr * BK + part * 32 + j] = 0; }
        if (part == 0) { dg[slot][wr] = 0.0f; du[slot][wr] = 0.0f; }
    }
};
stage(0, 0);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int kb = 0, s = 0; kb < K; kb += BK, s ^= 1) {
    if (kb + BK < K) stage(kb + BK, s ^ 1);
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> bg((threadgroup int8_t*)wg[s], dextents<int32_t, 2>(BK, TN));
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> bu((threadgroup int8_t*)wu[s], dextents<int32_t, 2>(BK, TN));
    auto a = tA.slice(kb, 0);
    op.run(a, bg, ig);
    op.run(a, bu, iu);
    if ((kb + BK) % 256 == 0) {
        // Super-block fold: weights were stored negated, so multiply by -d.
#pragma unroll
        for (int j = 0; j < CAP; ++j) {
            if (ig.is_valid_element(j)) {
                const int c = ig.get_multidimensional_index(j)[0];
                fg[j] += -dg[s][c] * float(ig[j]);
                fu[j] += -du[s][c] * float(iu[j]);
                ig[j] = 0; iu[j] = 0;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
#pragma unroll
for (int j = 0; j < CAP; ++j) {
    if (ig.is_valid_element(j)) {
        const auto index = ig.get_multidimensional_index(j);
        const int col = col0 + index[0]; const uint row = uint(index[1]);
        if (col < N && row < valid) {
            const float s = xs[rowbase + row];
            const float g = min(fg[j] * s, 10.0f);
            const float u = clamp(fu[j] * s, -10.0f, 10.0f);
            y[ulong(rowbase + row) * N + col] = (g / (1.0f + metal::precise::exp(-g))) * u;
        }
    }
}
"""


@lru_cache(maxsize=1)
def _segmented_gate_up_w8a8():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="mlx2_gguf_kblock_segmented_gate_up_w8a8_v2",
        input_names=["xq", "xs", "tiles", "tile_count", "gate", "up"], output_names=["y"],
        header=W8A8_HEADER, source=SEG_GATE_UP_W8A8, ensure_row_contiguous=True,
    )


def kblock_segmented_moe_prefill_w8a8(x, ids, scores, gate, up, down, *, max_scratch_bytes=512 << 20):
    """W8A8 Q3_K gate/up (exact int8 weights, per-row int8 activations) + v3 down."""
    import mlx.core as mx

    from ..int8_prefill import _quantize_rows
    from . import segmented_prefill as seg

    if mx.default_device() != mx.gpu or "Apple M5" not in mx.device_info().get("device_name", ""):
        raise GGUFError("K-block segmented prefill requires an owned Apple M5 GPU window")
    experts, hidden, k = seg._validate(x, ids, scores, gate, up, down, max_scratch_bytes)
    if gate.ggml_type != 11 or down.ggml_type not in (11, 12, 13, 14) or k % 256:
        raise GGUFError("W8A8 gate/up covers Q3_K gate/up experts with K a multiple of 256")
    if ids.dtype != mx.uint32:
        raise GGUFError("segmented expert IDs must be uint32")
    plan = seg.build_plan(ids, experts, 32)
    xq, xsc = _quantize_rows(x.astype(mx.float32))
    rows = plan.order // ids.shape[1]
    mid = _segmented_gate_up_w8a8()(
        inputs=[mx.take(xq, rows, axis=0), mx.take(xsc, rows, axis=0), plan.tiles, plan.tile_count,
                gate.data, up.data],
        template=[("K", k), ("N", hidden), ("ROW_BYTES", gate.row_bytes)],
        grid=(((hidden + 63) // 64) * 128, plan.max_tiles, 1), threadgroup=(128, 1, 1),
        output_shapes=[(plan.assignment_rows, hidden)], output_dtypes=[mx.float32],
    )[0]
    projected = _segmented_kernel_v2("down")(
        inputs=[mid.astype(mx.float16), plan.tiles, plan.tile_count, down.data],
        template=[("K", hidden), ("N", k), ("ROW_BYTES", down.row_bytes), ("FORMAT", down.ggml_type),
                  ("TMT", 32)],
        grid=(((k + 63) // 64) * 128, plan.max_tiles, 1), threadgroup=(128, 1, 1),
        output_shapes=[(plan.assignment_rows, k)], output_dtypes=[mx.float32],
    )[0]
    return seg._kernel("combine")(
        inputs=[projected, plan.inverse, scores.astype(mx.float32).reshape(-1), plan.status],
        template=[("TOKENS", x.shape[0]), ("TOPK", ids.shape[1]), ("N", k)],
        grid=(((k + 255) // 256) * 256, x.shape[0], 1), threadgroup=(256, 1, 1),
        output_shapes=[(x.shape[0], k)], output_dtypes=[mx.float32],
    )[0]


# Register-fed K-block prefill tile (lane matmul's right-input fragment map,
# runtime/lane/matmul.py): each lane decodes its runs of four consecutive k
# per (weight row, 16-wide K chunk) straight into the right input cooperative
# tensor -- no threadgroup staging, no barriers. One execution_simdgroup op per
# simdgroup: 32 rows x 32 weight rows x 64 K, multiply_accumulate. Operand
# values are the same half(value) as tile v2; only the MMA tiling differs, so
# it is classified against fp64 like v2.
KREG_HEADER = HEADER + r"""
inline ushort2 kreg_u2(device const uchar *p) {  // two aligned ushorts = 4 bytes
    device const ushort *q=(device const ushort*)p;
    return ushort2(q[0], q[1]);
}
// Four consecutive values k0..k0+3 (k0 % 4 == 0) of one K-block row, each as
// gguf_k_value computes it, cast to half.
inline half4 kblock_values4(device const uchar *row, uint k0, uint format) {
    half4 out;
    if(format==2) {
        device const uchar *p=row+(k0/32)*18;
        const uint i=k0%32;
        const float d=gguf_k_half(p);
        const ushort2 b=kreg_u2(p+2+i%16);
        const uint word=uint(b.x)|(uint(b.y)<<16), shift=4*(i/16);
        for(uint e=0;e<4;e++) out[e]=half(d*float(int((word>>(8*e+shift))&15)-8));
        return out;
    }
    const uint i=k0%256;
    if(format==11) {
        device const uchar *p=row+(k0/256)*110;
        const uint g=i/16;
        const uint lo=(p[96+g%8]>>(4*(g/8)))&15;
        const uint hi=(p[104+g%4]>>(2*(g/4)))&3;
        const float ds=gguf_k_half(p+108)*float(int(lo|(hi<<4))-32);
        const ushort2 qv=kreg_u2(p+32+(i/128)*32+i%32), hv=kreg_u2(p+i%32);
        const uint qw=uint(qv.x)|(uint(qv.y)<<16), hw=uint(hv.x)|(uint(hv.y)<<16);
        const uint shift=2*((i/32)%4), hb=i/32;
        for(uint e=0;e<4;e++) {
            const float q=float((qw>>(8*e+shift))&3)+(((hw>>(8*e+hb))&1) ? 0.0f : -4.0f);
            out[e]=half(ds*q);
        }
        return out;
    }
    if(format==12 || format==13) {
        device const uchar *p=row+(k0/256)*(format==12 ? 144 : 176);
        const uint c=i/32;
        const uint sc=p[4+c%4], mn=p[8+c%4], aux=p[12+c%4];
        const uint scale=c<4 ? sc&63 : (aux&15)|((sc>>2)&48);
        const uint minimum=c<4 ? mn&63 : (aux>>4)|((mn>>2)&48);
        const float ds=gguf_k_half(p)*float(scale), dm=gguf_k_half(p+2)*float(minimum);
        const uint qw=*((device const uint*)(p+(format==12 ? 16 : 48)+(i/64)*32+i%32));
        const uint hw=format==13 ? *((device const uint*)(p+16+i%32)) : 0u;
        const uint shift=4*(c%2);
        for(uint e=0;e<4;e++) {
            uint q=(qw>>(8*e+shift))&15;
            if(format==13) q|=((hw>>(8*e+c))&1)<<4;
            out[e]=half(ds*float(q)-dm);
        }
        return out;
    }
    device const uchar *p=row+(k0/256)*210;
    const float ds=gguf_k_half(p+208)*float(int(as_type<char>(p[192+i/16])));
    const ushort2 lv=kreg_u2(p+(i/128)*64+i%64), hv=kreg_u2(p+128+(i/128)*32+i%32);
    const uint lw=uint(lv.x)|(uint(lv.y)<<16), hw=uint(hv.x)|(uint(hv.y)<<16);
    const uint lshift=4*((i/64)%2), hshift=2*((i/32)%4);
    for(uint e=0;e<4;e++) {
        const uint lo=(lw>>(8*e+lshift))&15, hi=(hw>>(8*e+hshift))&3;
        out[e]=half(ds*float(int(lo|(hi<<4))-32));
    }
    return out;
}
"""

KREG_SOURCE = r"""
constexpr int NT = 32, GS = 64, SGS = 4;
const ushort lane = thread_index_in_simdgroup;
const ushort sg = simdgroup_index_in_threadgroup;
const int rb = int(threadgroup_position_in_grid.y) * 16 * TMR;
const int n0 = (int(threadgroup_position_in_grid.x) * SGS + int(sg)) * NT;
constexpr auto desc = matmul2d_descriptor(16 * TMR, NT, GS, false, true, false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroup> op;
tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)x + int64_t(rb) * K,
    dextents<int32_t, 2>(K, TOKENS - rb));
const int bqi = (lane & 1) | (((lane >> 3) & 1) << 1);
const int bnb = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
device const uchar *wrow[4];
bool wok[4];
for (int j = 0; j < 4; ++j) {
    const int col = n0 + bnb + 8 * j;
    wok[j] = col < N;
    wrow[j] = w + ulong(wok[j] ? col : 0) * ROW_BYTES;
}
auto a0 = tA.slice(0, 0);
auto cb0 = op.template get_right_input_cooperative_tensor<half, half, float>();
auto c = op.template get_destination_cooperative_tensor<decltype(a0), metal::remove_addrspace_t<decltype(cb0)>, float>();
for (uint16_t j = 0; j < c.get_capacity(); ++j) if (c.is_valid_element(j)) c[j] = 0.0f;
for (int g = 0; g < K / GS; ++g) {
    auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
    for (int j = 0; j < 4; ++j) {
        for (int kb = 0; kb < 4; ++kb) {
            const half4 v = wok[j] ? kblock_values4(wrow[j], uint(g * GS + kb * 16 + 4 * bqi), FORMAT) : half4(0.0h);
            cb[kb * 16 + j * 4 + 0] = v.x; cb[kb * 16 + j * 4 + 1] = v.y;
            cb[kb * 16 + j * 4 + 2] = v.z; cb[kb * 16 + j * 4 + 3] = v.w;
        }
    }
    auto a = tA.slice(g * GS, 0);
    op.run(a, cb, c);
}
for (uint16_t j = 0; j < c.get_capacity(); ++j) {
    if (c.is_valid_element(j)) {
        const auto index = c.get_multidimensional_index(j);
        const int col = n0 + index[0], row = rb + index[1];
        if (col < N && row < TOKENS) y[ulong(row) * N + col] = c[j];
    }
}
"""


@lru_cache(maxsize=1)
def _kreg_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="mlx2_gguf_kblock_reg_prefill_v1", input_names=["x", "w"], output_names=["y"],
        header=KREG_HEADER, source=KREG_SOURCE, ensure_row_contiguous=True,
    )


def kblock_reg_prefill(x, weight: PackedMatrix, *, tmr=2):
    """Register-fed K-block prefill tile (no threadgroup staging); candidate law."""
    import mlx.core as mx

    if mx.default_device() != mx.gpu or "Apple M5" not in mx.device_info().get("device_name", ""):
        raise GGUFError("K-block register prefill requires an owned Apple M5 GPU window")
    if (x.ndim != 2 or len(weight.shape) != 2 or x.shape[-1] != weight.shape[-1]
            or x.shape[-1] % 64 or weight.ggml_type not in (2, 11, 12, 13, 14) or tmr not in (1, 2, 4)):
        raise GGUFError("K-block register prefill needs rank2 K-block weights with K % 64 == 0")
    m, k = x.shape
    n = weight.shape[0]
    return _kreg_kernel()(
        inputs=[x.astype(mx.float16), weight.data],
        template=[("TOKENS", m), ("K", k), ("N", n), ("ROW_BYTES", weight.row_bytes),
                  ("FORMAT", weight.ggml_type), ("TMR", tmr)],
        grid=(((n + 127) // 128) * 128, (m + 16 * tmr - 1) // (16 * tmr), 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.float32],
    )[0]
