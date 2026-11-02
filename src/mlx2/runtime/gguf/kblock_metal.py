"""Default-off K-block row kernels: dense/grouped GEMV and routed experts.

The packed reference kernels (packed_metal.py) give one simdgroup per output
row and decode every weight element through ``gguf_k_value``, re-reading its
block scales each time. These kernels keep the cooperative layout of
cooperative_metal.py (four simdgroups x ROWS output rows per threadgroup, one
32-value chunk per lane step, the chunk's activations held in registers and
reused across rows and across gate/up) and add chunk decoders for the GGUF
K-block types Q4_0, Q3_K, Q4_K, Q5_K and Q6_K, which decode a chunk's scales
once. Each weight value is the same float expression as ``gguf_k_value``
((d*scale)*q, minus dmin*min for Q4_K/Q5_K); products and accumulation stay
FP32. Only the summation order differs from the reference, so this is a
distinct numerical law that needs its own oracle. Routed experts keep the
reference's slot-ordered combine.
"""

import math
from functools import lru_cache

from .k_blocks import SIZES as K_FORMATS
from .packed_metal import HEADER, PackedMatrix
from .reader import GGUFError

KBLOCK_HEADER = (
    HEADER
    + r"""
// One 32-value chunk of a K-block row: sum_i a[i] * value(chunk*32+i).
inline float kblock_dot32(device const uchar *row, uint chunk, thread const float *a, uint format) {
    float sum=0.0f;
    if(format==2) {  // Q4_0: one 18-byte block per chunk
        device const uchar *p=row+chunk*18;
        const float d=gguf_k_half(p);
        #pragma unroll
        for(uint i=0;i<32;i++) {
            const float v=d*float(int((p[2+i%16]>>(4*(i/16)))&15)-8);
            sum+=a[i]*v;
        }
        return sum;
    }
    const uint c=chunk%8;
    if(format==11) {  // Q3_K, 110 bytes per 256
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
        for(uint j=0;j<32;j++) {
            int q=int((qs[j]>>shift)&3)-4*(1-int((p[j]>>c)&1));
            sum+=a[j]*(ds[j/16]*float(q));
        }
        return sum;
    }
    if(format==12 || format==13) {  // Q4_K 144 / Q5_K 176 bytes per 256
        device const uchar *p=row+(chunk/8)*(format==12 ? 144 : 176);
        const uint sc=p[4+c%4], mn=p[8+c%4], aux=p[12+c%4];
        const uint scale=c<4 ? sc&63 : (aux&15)|((sc>>2)&48);
        const uint minimum=c<4 ? mn&63 : (aux>>4)|((mn>>2)&48);
        const float ds=gguf_k_half(p)*float(scale), dm=gguf_k_half(p+2)*float(minimum);
        device const uchar *qs=p+(format==12 ? 16 : 48)+(c/2)*32;
        const uint shift=4*(c%2);
        #pragma unroll
        for(uint j=0;j<32;j++) {
            uint q=(qs[j]>>shift)&15;
            if(format==13) q|=((p[16+j]>>c)&1)<<4;
            sum+=a[j]*(ds*float(q)-dm);
        }
        return sum;
    }
    // Q6_K, 210 bytes per 256
    device const uchar *p=row+(chunk/8)*210;
    const float d=gguf_k_half(p+208);
    const float ds[2]={d*float(int(as_type<char>(p[192+2*c]))), d*float(int(as_type<char>(p[193+2*c])))};
    device const uchar *ql=p+(c/4)*64+(c%2)*32;
    device const uchar *qh=p+128+(c/4)*32;
    const uint lshift=4*((c/2)%2), hshift=2*(c%4);
    #pragma unroll
    for(uint j=0;j<32;j++) {
        const uint lo=(ql[j]>>lshift)&15, hi=(qh[j]>>hshift)&3;
        sum+=a[j]*(ds[j/16]*float(int(lo|(hi<<4))-32));
    }
    return sum;
}

// Same element values as kblock_dot32 with loads widened to what the block
// stride guarantees: Q4_K/Q5_K blocks (144/176 B) stay 4-byte aligned, Q3_K,
// Q6_K and Q4_0 blocks (110/210/18 B) only 2-byte aligned, alternating between
// blocks, so they use ushort words.
inline float kblock_dot32_v2(device const uchar *row, uint chunk, thread const float *a, uint format) {
    float sum=0.0f;
    if(format==111) {
        // Research container "Q3N": Q3_K repacked losslessly as q+4 nibbles.
        // Per 256-value block, 148 bytes: 8 chunks x 16 nibble bytes (element j
        // of a chunk in byte j%16, nibble j/16), 16 int8 scales sc-32, fp16 d,
        // 2 pad bytes. Element value d*sc*q and accumulation order equal Q3_K's
        // v1 path, so outputs should match it bit for bit.
        const uint c=chunk%8;
        device const uchar *p=row+(chunk/8)*148;
        const float d=gguf_k_half(p+144);
        const float ds[2]={d*float(int(as_type<char>(p[128+2*c]))), d*float(int(as_type<char>(p[129+2*c])))};
        device const uint *nb=(device const uint*)(p+16*c);
        #pragma unroll
        for(uint w=0;w<4;w++) {
            const uint v=nb[w];
            #pragma unroll
            for(uint e=0;e<4;e++) {
                const uint j=4*w+e;
                sum+=a[j]*(ds[0]*(float((v>>(8*e))&15u)-4.0f));
            }
        }
        #pragma unroll
        for(uint w=0;w<4;w++) {
            const uint v=nb[w];
            #pragma unroll
            for(uint e=0;e<4;e++) {
                const uint j=16+4*w+e;
                sum+=a[j]*(ds[1]*(float((v>>(8*e+4))&15u)-4.0f));
            }
        }
        return sum;
    }
    if(format==2) {
        device const uchar *p=row+chunk*18;
        const float d=gguf_k_half(p);
        device const ushort *q=(device const ushort*)(p+2);
        #pragma unroll
        for(uint w=0;w<8;w++) {
            const uint b=q[w];
            #pragma unroll
            for(uint e=0;e<2;e++) {
                const uint byte=(b>>(8*e))&255, i=2*w+e;
                sum+=a[i]*(d*float(int(byte&15)-8));
                sum+=a[i+16]*(d*float(int(byte>>4)-8));
            }
        }
        return sum;
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
        device const ushort *qs=(device const ushort*)(p+32+(c/4)*32);
        device const ushort *hm=(device const ushort*)p;
        const uint shift=2*(c%4);
        #pragma unroll
        for(uint w=0;w<16;w++) {
            const uint qw=qs[w], hw=hm[w];
            #pragma unroll
            for(uint e=0;e<2;e++) {
                const uint j=2*w+e;
                // float(q2)+{0,-4}: equal to float(q2-4*(1-h)) for these
                // small integers, with fewer integer ops (same element value).
                const float q=float((qw>>(8*e+shift))&3)+(((hw>>(8*e+c))&1) ? 0.0f : -4.0f);
                sum+=a[j]*(ds[j/16]*q);
            }
        }
        return sum;
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
            const uint qw=qs[w], hw=format==13 ? qh[w] : 0u;
            #pragma unroll
            for(uint e=0;e<4;e++) {
                uint q=(qw>>(8*e+shift))&15;
                if(format==13) q|=((hw>>(8*e+c))&1)<<4;
                sum+=a[4*w+e]*(ds*float(q)-dm);
            }
        }
        return sum;
    }
    device const uchar *p=row+(chunk/8)*210;
    const float d=gguf_k_half(p+208);
    const float ds[2]={d*float(int(as_type<char>(p[192+2*c]))), d*float(int(as_type<char>(p[193+2*c])))};
    device const ushort *ql=(device const ushort*)(p+(c/4)*64+(c%2)*32);
    device const ushort *qh=(device const ushort*)(p+128+(c/4)*32);
    const uint lshift=4*((c/2)%2), hshift=2*(c%4);
    #pragma unroll
    for(uint w=0;w<16;w++) {
        const uint lw=ql[w], hw=qh[w];
        #pragma unroll
        for(uint e=0;e<2;e++) {
            const uint j=2*w+e;
            const uint lo=(lw>>(8*e+lshift))&15, hi=(hw>>(8*e+hshift))&3;
            sum+=a[j]*(ds[j/16]*float(int(lo|(hi<<4))-32));
        }
    }
    return sum;
}

// Token-shared form of kblock_dot32_v2 for the K-quant formats: a chunk's 32
// weight values are decoded once into registers (the same value expressions),
// then every token t forms part[t]=sum_j a_t[j]*v[j] from 0 in ascending j,
// which is kblock_dot32_v2's per-chunk partial for that token alone.
// xc points at token0's 32 activations of this chunk; token rows are K apart.
inline void kblock_values32_f(device const uchar *row, uint chunk, thread float *v, uint format) {
    const uint c=chunk%8;
    if(format==11) {
        device const uchar *p=row+(chunk/8)*110;
        const float d=gguf_k_half(p+108);
        float ds[2];
        for(uint h=0;h<2;h++) {
            const uint g=2*c+h;
            const uint lo=(p[96+g%8]>>(4*(g/8)))&15;
            const uint hi=(p[104+g%4]>>(2*(g/4)))&3;
            ds[h]=d*float(int(lo|(hi<<4))-32);
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
                const float q=float((qw>>(8*e+shift))&3)+(((hw>>(8*e+c))&1) ? 0.0f : -4.0f);
                v[j]=ds[j/16]*q;
            }
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
            const uint qw=qs[w], hw=format==13 ? qh[w] : 0u;
            #pragma unroll
            for(uint e=0;e<4;e++) {
                uint q=(qw>>(8*e+shift))&15;
                if(format==13) q|=((hw>>(8*e+c))&1)<<4;
                v[4*w+e]=ds*float(q)-dm;
            }
        }
        return;
    }
    device const uchar *p=row+(chunk/8)*210;
    const float d=gguf_k_half(p+208);
    const float ds[2]={d*float(int(as_type<char>(p[192+2*c]))), d*float(int(as_type<char>(p[193+2*c])))};
    device const ushort *ql=(device const ushort*)(p+(c/4)*64+(c%2)*32);
    device const ushort *qh=(device const ushort*)(p+128+(c/4)*32);
    const uint lshift=4*((c/2)%2), hshift=2*(c%4);
    #pragma unroll
    for(uint w=0;w<16;w++) {
        const uint lw=ql[w], hw=qh[w];
        #pragma unroll
        for(uint e=0;e<2;e++) {
            const uint j=2*w+e;
            const uint lo=(lw>>(8*e+lshift))&15, hi=(hw>>(8*e+hshift))&3;
            v[j]=ds[j/16]*float(int(lo|(hi<<4))-32);
        }
    }
}
template<uint T>
inline void kblock_dot32_v2_tok(device const uchar *row, uint chunk, device const float *xc, uint K, uint ntok, thread float *part, uint format) {
    float v[32];
    kblock_values32_f(row, chunk, v, format);
    for(uint t=0;t<T;t++) {
        float sum=0.0f;
        if(t<ntok) {
            device const float4 *a4=(device const float4*)(xc+size_t(t)*K);
            #pragma unroll
            for(uint w=0;w<8;w++) {
                const float4 a=a4[w];
                sum+=a.x*v[4*w]; sum+=a.y*v[4*w+1]; sum+=a.z*v[4*w+2]; sum+=a.w*v[4*w+3];
            }
        }
        part[t]=sum;
    }
}

// Per-(row, chunk) header for the token-shared tile: the block scales and the
// code pointers of one 32-value chunk, decoded once; kb_vals4 then yields
// values 4w..4w+3 with kblock_dot32_v2's exact element expressions.
struct KHdr { float d0, d1, dm; device const uchar *p0; device const uchar *p1; uint s0, s1; };
inline KHdr kb_hdr(device const uchar *row, uint chunk, uint format) {
    KHdr h;
    const uint c=chunk%8;
    if(format==11) {
        device const uchar *p=row+(chunk/8)*110;
        const float d=gguf_k_half(p+108);
        float ds[2];
        for(uint i=0;i<2;i++) {
            const uint g=2*c+i;
            const uint lo=(p[96+g%8]>>(4*(g/8)))&15;
            const uint hi=(p[104+g%4]>>(2*(g/4)))&3;
            ds[i]=d*float(int(lo|(hi<<4))-32);
        }
        h.d0=ds[0]; h.d1=ds[1]; h.dm=0.0f; h.p0=p+32+(c/4)*32; h.p1=p; h.s0=2*(c%4); h.s1=c;
        return h;
    }
    if(format==12 || format==13) {
        device const uchar *p=row+(chunk/8)*(format==12 ? 144 : 176);
        const uint sc=p[4+c%4], mn=p[8+c%4], aux=p[12+c%4];
        const uint scale=c<4 ? sc&63 : (aux&15)|((sc>>2)&48);
        const uint minimum=c<4 ? mn&63 : (aux>>4)|((mn>>2)&48);
        h.d0=gguf_k_half(p)*float(scale); h.d1=0.0f; h.dm=gguf_k_half(p+2)*float(minimum);
        h.p0=p+(format==12 ? 16 : 48)+(c/2)*32; h.p1=p+16; h.s0=4*(c%2); h.s1=c;
        return h;
    }
    device const uchar *p=row+(chunk/8)*210;
    const float d=gguf_k_half(p+208);
    h.d0=d*float(int(as_type<char>(p[192+2*c]))); h.d1=d*float(int(as_type<char>(p[193+2*c]))); h.dm=0.0f;
    h.p0=p+(c/4)*64+(c%2)*32; h.p1=p+128+(c/4)*32; h.s0=4*((c/2)%2); h.s1=2*(c%4);
    return h;
}
inline void kb_vals4(thread const KHdr &h, uint w, thread float *v, uint format) {
    if(format==11) {
        device const ushort *qs=(device const ushort*)h.p0;
        device const ushort *hm=(device const ushort*)h.p1;
        #pragma unroll
        for(uint e2=0;e2<2;e2++) {
            const uint qw=qs[2*w+e2], hw=hm[2*w+e2];
            #pragma unroll
            for(uint e=0;e<2;e++) {
                const uint j=4*w+2*e2+e;
                const float q=float((qw>>(8*e+h.s0))&3)+(((hw>>(8*e+h.s1))&1) ? 0.0f : -4.0f);
                v[2*e2+e]=(j<16 ? h.d0 : h.d1)*q;
            }
        }
        return;
    }
    if(format==12 || format==13) {
        const uint qw=((device const uint*)h.p0)[w];
        const uint hw=format==13 ? ((device const uint*)h.p1)[w] : 0u;
        #pragma unroll
        for(uint e=0;e<4;e++) {
            uint q=(qw>>(8*e+h.s0))&15;
            if(format==13) q|=((hw>>(8*e+h.s1))&1)<<4;
            v[e]=h.d0*float(q)-h.dm;
        }
        return;
    }
    device const ushort *ql=(device const ushort*)h.p0;
    device const ushort *qh=(device const ushort*)h.p1;
    #pragma unroll
    for(uint e2=0;e2<2;e2++) {
        const uint lw=ql[2*w+e2], hw=qh[2*w+e2];
        #pragma unroll
        for(uint e=0;e<2;e++) {
            const uint j=4*w+2*e2+e;
            const uint lo=(lw>>(8*e+h.s0))&15, hi=(hw>>(8*e+h.s1))&3;
            v[2*e2+e]=(j<16 ? h.d0 : h.d1)*float(int(lo|(hi<<4))-32);
        }
    }
}

// Factored-scale variant (llama.cpp style): accumulate sum(a*q) and the
// matching activation sums per scale group, then apply d*scale once. Same
// loads as v2; fewer ALU ops per element; a further FP32 reassociation of the
// per-element (d*scale)*q law, so it is a separate VEC=2 law.
inline float kblock_dot32_v3(device const uchar *row, uint chunk, thread const float *a, uint format) {
    if(format==2) {
        device const uchar *p=row+chunk*18;
        device const ushort *q=(device const ushort*)(p+2);
        float acc=0.0f, asum=0.0f;
        #pragma unroll
        for(uint w=0;w<8;w++) {
            const uint b=q[w];
            #pragma unroll
            for(uint e=0;e<2;e++) {
                const uint byte=(b>>(8*e))&255, i=2*w+e;
                acc+=a[i]*float(byte&15)+a[i+16]*float(byte>>4);
                asum+=a[i]+a[i+16];
            }
        }
        return gguf_k_half(p)*(acc-8.0f*asum);
    }
    const uint c=chunk%8;
    if(format==11) {
        device const uchar *p=row+(chunk/8)*110;
        const float d=gguf_k_half(p+108);
        device const ushort *qs=(device const ushort*)(p+32+(c/4)*32);
        device const ushort *hm=(device const ushort*)p;
        const uint shift=2*(c%4);
        float sum=0.0f;
        #pragma unroll
        for(uint h=0;h<2;h++) {
            const uint g=2*c+h;
            const uint lo=(p[96+g%8]>>(4*(g/8)))&15;
            const uint hi=(p[104+g%4]>>(2*(g/4)))&3;
            float acc=0.0f, low=0.0f;
            #pragma unroll
            for(uint w=8*h;w<8*h+8;w++) {
                const uint qw=qs[w]>>shift, hw=hm[w]>>c;
                #pragma unroll
                for(uint e=0;e<2;e++) {
                    const uint j=2*w+e;
                    acc+=a[j]*float((qw>>(8*e))&3);
                    low+=((hw>>(8*e))&1) ? 0.0f : a[j];
                }
            }
            sum+=(d*float(int(lo|(hi<<4))-32))*(acc-4.0f*low);
        }
        return sum;
    }
    if(format==12 || format==13) {
        device const uchar *p=row+(chunk/8)*(format==12 ? 144 : 176);
        const uint sc=p[4+c%4], mn=p[8+c%4], aux=p[12+c%4];
        const uint scale=c<4 ? sc&63 : (aux&15)|((sc>>2)&48);
        const uint minimum=c<4 ? mn&63 : (aux>>4)|((mn>>2)&48);
        device const uint *qs=(device const uint*)(p+(format==12 ? 16 : 48)+(c/2)*32);
        device const uint *qh=(device const uint*)(p+16);
        const uint shift=4*(c%2);
        float acc=0.0f, asum=0.0f;
        #pragma unroll
        for(uint w=0;w<8;w++) {
            const uint qw=qs[w]>>shift, hw=format==13 ? qh[w]>>c : 0u;
            #pragma unroll
            for(uint e=0;e<4;e++) {
                uint q=(qw>>(8*e))&15;
                if(format==13) q|=((hw>>(8*e))&1)<<4;
                acc+=a[4*w+e]*float(q);
                asum+=a[4*w+e];
            }
        }
        return (gguf_k_half(p)*float(scale))*acc-(gguf_k_half(p+2)*float(minimum))*asum;
    }
    device const uchar *p=row+(chunk/8)*210;
    const float d=gguf_k_half(p+208);
    device const ushort *ql=(device const ushort*)(p+(c/4)*64+(c%2)*32);
    device const ushort *qh=(device const ushort*)(p+128+(c/4)*32);
    const uint lshift=4*((c/2)%2), hshift=2*(c%4);
    float sum=0.0f;
    #pragma unroll
    for(uint h=0;h<2;h++) {
        float acc=0.0f, asum=0.0f;
        #pragma unroll
        for(uint w=8*h;w<8*h+8;w++) {
            const uint lw=ql[w]>>lshift, hw=qh[w]>>hshift;
            #pragma unroll
            for(uint e=0;e<2;e++) {
                const uint j=2*w+e;
                acc+=a[j]*float(((lw>>(8*e))&15)|(((hw>>(8*e))&3)<<4));
                asum+=a[j];
            }
        }
        sum+=(d*float(int(as_type<char>(p[192+2*c+h]))))*(acc-32.0f*asum);
    }
    return sum;
}

// Word-wide v1: four consecutive chunk elements share four contiguous bytes, so
// one 32-bit shift/mask yields four codes, one float4 convert/offset/scale
// yields the four element values (each equal to v1's ds*q), and the products
// are accumulated one element at a time in v1's order. Same law as VEC=1.
inline uint kd_u32(device const uchar *p) {
    device const ushort *q=(device const ushort*)p;
    return uint(q[0])|(uint(q[1])<<16);
}
inline float kblock_dot32_v4(device const uchar *row, uint chunk, thread const float *a, uint format) {
    if(format==2) return kblock_dot32_v2(row, chunk, a, format);  // Q4_0 keeps its own law
    float sum=0.0f;
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
            const uint code=((kd_u32(qs+4*w)>>shift)&0x03030303u)|(((kd_u32(p+4*w)>>c)&0x01010101u)<<2);
            const float4 v=ds[w/4]*(float4(as_type<uchar4>(code))-4.0f);
            sum+=a[4*w]*v.x; sum+=a[4*w+1]*v.y; sum+=a[4*w+2]*v.z; sum+=a[4*w+3]*v.w;
        }
        return sum;
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
            const float4 v=ds*float4(as_type<uchar4>(code))-dm;
            sum+=a[4*w]*v.x; sum+=a[4*w+1]*v.y; sum+=a[4*w+2]*v.z; sum+=a[4*w+3]*v.w;
        }
        return sum;
    }
    device const uchar *p=row+(chunk/8)*210;
    const float d=gguf_k_half(p+208);
    const float ds[2]={d*float(int(as_type<char>(p[192+2*c]))), d*float(int(as_type<char>(p[193+2*c])))};
    device const uchar *ql=p+(c/4)*64+(c%2)*32;
    device const uchar *qh=p+128+(c/4)*32;
    const uint lshift=4*((c/2)%2), hshift=2*(c%4);
    #pragma unroll
    for(uint w=0;w<8;w++) {
        const uint code=((kd_u32(ql+4*w)>>lshift)&0x0F0F0F0Fu)|(((kd_u32(qh+4*w)>>hshift)&0x03030303u)<<4);
        const float4 v=ds[w/4]*(float4(as_type<uchar4>(code))-32.0f);
        sum+=a[4*w]*v.x; sum+=a[4*w+1]*v.y; sum+=a[4*w+2]*v.z; sum+=a[4*w+3]*v.w;
    }
    return sum;
}
// VEC=4 (glm53_kblock_element=fma_v1): kblock_dot32_v2 with the Q3_K and
// Q6_K element values formed by one fused multiply-add instead of an integer
// or float offset followed by a multiply. For a finite fp16 d = H*2^e
// (|H| <= 2047), ds = d*scale and the codes give integer products below 2^24
// (Q3_K: scale in [-32,31], code < 4; Q6_K: int8 scale, code < 64:
// 2047*128*63 < 2^24), so ds*code, the offset -4*ds / -32*ds and their sum are
// all exactly representable: fma(code, ds, offset) rounds nothing and equals
// v2's ds*q. A chunk whose d is not finite takes v2 itself (Inf*code-Inf would
// be NaN where v2 gives +-Inf).
// The one difference is the sign of a zero element (v2: ds*(+0) = -0 when
// ds < 0; here +0); it only reaches a sum through sum + a*(+-0), and the
// running sum starts at +0 and is never -0 (x + -x rounds to +0), so every
// partial is the same bits. Products and accumulation order are v2's.
inline float kblock_dot32_v5(device const uchar *row, uint chunk, thread const float *a, uint format) {
    if(format!=11 && format!=14) return kblock_dot32_v2(row, chunk, a, format);
    float sum=0.0f;
    const uint c=chunk%8;
    if(format==11) {
        device const uchar *p=row+(chunk/8)*110;
        const float d=gguf_k_half(p+108);
        if(!isfinite(d)) return kblock_dot32_v2(row, chunk, a, format);
        float ds[2], off[2];
        #pragma unroll
        for(uint h=0;h<2;h++) {
            const uint g=2*c+h;
            const uint lo=(p[96+g%8]>>(4*(g/8)))&15;
            const uint hi=(p[104+g%4]>>(2*(g/4)))&3;
            ds[h]=d*float(int(lo|(hi<<4))-32);
            off[h]=-4.0f*ds[h];
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
                const float v=fma(float((qw>>(8*e+shift))&3), ds[j/16], ((hw>>(8*e+c))&1) ? 0.0f : off[j/16]);
                sum+=a[j]*v;
            }
        }
        return sum;
    }
    device const uchar *p=row+(chunk/8)*210;
    const float d=gguf_k_half(p+208);
    if(!isfinite(d)) return kblock_dot32_v2(row, chunk, a, format);
    const float ds[2]={d*float(int(as_type<char>(p[192+2*c]))), d*float(int(as_type<char>(p[193+2*c])))};
    const float off[2]={-32.0f*ds[0], -32.0f*ds[1]};
    device const ushort *ql=(device const ushort*)(p+(c/4)*64+(c%2)*32);
    device const ushort *qh=(device const ushort*)(p+128+(c/4)*32);
    const uint lshift=4*((c/2)%2), hshift=2*(c%4);
    #pragma unroll
    for(uint w=0;w<16;w++) {
        const uint lw=ql[w], hw=qh[w];
        #pragma unroll
        for(uint e=0;e<2;e++) {
            const uint j=2*w+e;
            const uint lo=(lw>>(8*e+lshift))&15, hi=(hw>>(8*e+hshift))&3;
            sum+=a[j]*fma(float(lo|(hi<<4)), ds[j/16], off[j/16]);
        }
    }
    return sum;
}
#define KDOT(row, chunk, a, format) (VEC == 4 ? kblock_dot32_v5(row, chunk, a, format) : VEC == 3 ? kblock_dot32_v4(row, chunk, a, format) : VEC == 2 ? kblock_dot32_v3(row, chunk, a, format) : VEC ? kblock_dot32_v2(row, chunk, a, format) : kblock_dot32(row, chunk, a, format))

// Load-ahead (AHEAD=2, glm53_row_kernel=kblock_rows_ahead2): kblock_dot32_v2
// split into a load phase and an arithmetic phase. ka_load issues every device
// read one chunk needs into registers and does no arithmetic on them, so the
// loads of the next chunk can be in flight while ka_dot runs the current one.
// ka_dot is kblock_dot32_v2's element and accumulation expressions applied to
// the same bytes (the scale and fp16 bits are the bytes gguf_k_half reads;
// little-endian ushort/uint words of the same addresses), so its result is the
// same float.
// One struct per format holding exactly the raw words one chunk reads, so the
// two in-flight sets stay in registers (a union of every format's fields was
// spilled to thread memory: 452 -> 72 GB/s on the routed MoE, 1ee18335e).
template<int F> struct KA { };
template<> struct KA<2> { ushort q[8]; ushort d; };
template<> struct KA<11> { ushort qs[16]; ushort hm[16]; ushort lo, hi, d; };
template<> struct KA<12> { uint qs[8]; uchar sc, mn, aux; ushort d, dm; };
template<> struct KA<13> { uint qs[8]; uint qh[8]; uchar sc, mn, aux; ushort d, dm; };
template<> struct KA<14> { ushort ql[16]; ushort qh[16]; ushort sc; ushort d; };
inline float ka_half(ushort bits) { return float(as_type<half>(bits)); }
// Formats without a load-ahead body (e.g. the Q3N research container) never
// take the AHEAD branch (the host keeps AHEAD=1); these keep it compilable.
template<int F> inline KA<F> ka_load(device const uchar *row, uint chunk) { return KA<F>(); }
template<int F> inline float ka_dot(thread const KA<F> &r, uint chunk, thread const float *a) { return 0.0f; }
template<> inline KA<2> ka_load<2>(device const uchar *row, uint chunk) {
    KA<2> r;
    device const uchar *p=row+chunk*18;
    device const ushort *q=(device const ushort*)(p+2);
    #pragma unroll
    for(uint w=0;w<8;w++) r.q[w]=q[w];
    r.d=((device const ushort*)p)[0];
    return r;
}
template<> inline float ka_dot<2>(thread const KA<2> &r, uint chunk, thread const float *a) {
    float sum=0.0f;
    const float d=ka_half(r.d);
    #pragma unroll
    for(uint w=0;w<8;w++) {
        const uint b=r.q[w];
        #pragma unroll
        for(uint e=0;e<2;e++) {
            const uint byte=(b>>(8*e))&255, i=2*w+e;
            sum+=a[i]*(d*float(int(byte&15)-8));
            sum+=a[i+16]*(d*float(int(byte>>4)-8));
        }
    }
    return sum;
}
template<> inline KA<11> ka_load<11>(device const uchar *row, uint chunk) {
    KA<11> r;
    const uint c=chunk%8;
    device const uchar *p=row+(chunk/8)*110;
    device const ushort *qs=(device const ushort*)(p+32+(c/4)*32);
    device const ushort *hm=(device const ushort*)p;
    #pragma unroll
    for(uint w=0;w<16;w++) { r.qs[w]=qs[w]; r.hm[w]=hm[w]; }
    r.lo=*(device const ushort*)(p+96+(2*c)%8);
    r.hi=*(device const ushort*)(p+104+(2*c)%4);
    r.d=*(device const ushort*)(p+108);
    return r;
}
template<> inline float ka_dot<11>(thread const KA<11> &r, uint chunk, thread const float *a) {
    float sum=0.0f;
    const uint c=chunk%8;
    const float d=ka_half(r.d);
    float ds[2];
    #pragma unroll
    for(uint h=0;h<2;h++) {
        const uint g=2*c+h;
        const uint lo=((uint(r.lo)>>(8*h))>>(4*(g/8)))&15;
        const uint hi=((uint(r.hi)>>(8*h))>>(2*(g/4)))&3;
        ds[h]=d*float(int(lo|(hi<<4))-32);
    }
    const uint shift=2*(c%4);
    #pragma unroll
    for(uint w=0;w<16;w++) {
        const uint qw=r.qs[w], hw=r.hm[w];
        #pragma unroll
        for(uint e=0;e<2;e++) {
            const uint j=2*w+e;
            const float q=float((qw>>(8*e+shift))&3)+(((hw>>(8*e+c))&1) ? 0.0f : -4.0f);
            sum+=a[j]*(ds[j/16]*q);
        }
    }
    return sum;
}
template<int F> inline KA<F> ka_load_k45(device const uchar *row, uint chunk) {
    KA<F> r;
    const uint c=chunk%8;
    device const uchar *p=row+(chunk/8)*(F==12 ? 144 : 176);
    r.sc=p[4+c%4]; r.mn=p[8+c%4]; r.aux=p[12+c%4];
    r.d=((device const ushort*)p)[0]; r.dm=((device const ushort*)p)[1];
    device const uint *qs=(device const uint*)(p+(F==12 ? 16 : 48)+(c/2)*32);
    #pragma unroll
    for(uint w=0;w<8;w++) r.qs[w]=qs[w];
    return r;
}
template<> inline KA<12> ka_load<12>(device const uchar *row, uint chunk) { return ka_load_k45<12>(row, chunk); }
template<> inline KA<13> ka_load<13>(device const uchar *row, uint chunk) {
    KA<13> r=ka_load_k45<13>(row, chunk);
    device const uint *qh=(device const uint*)(row+(chunk/8)*176+16);
    #pragma unroll
    for(uint w=0;w<8;w++) r.qh[w]=qh[w];
    return r;
}
template<int F> inline float ka_dot_k45(thread const KA<F> &r, uint chunk, thread const float *a, thread const uint *qh) {
    float sum=0.0f;
    const uint c=chunk%8;
    const uint sc=r.sc, mn=r.mn, aux=r.aux;
    const uint scale=c<4 ? sc&63 : (aux&15)|((sc>>2)&48);
    const uint minimum=c<4 ? mn&63 : (aux>>4)|((mn>>2)&48);
    const float ds=ka_half(r.d)*float(scale), dm=ka_half(r.dm)*float(minimum);
    const uint shift=4*(c%2);
    #pragma unroll
    for(uint w=0;w<8;w++) {
        const uint qw=r.qs[w], hw=F==13 ? qh[w] : 0u;
        #pragma unroll
        for(uint e=0;e<4;e++) {
            uint q=(qw>>(8*e+shift))&15;
            if(F==13) q|=((hw>>(8*e+c))&1)<<4;
            sum+=a[4*w+e]*(ds*float(q)-dm);
        }
    }
    return sum;
}
template<> inline float ka_dot<12>(thread const KA<12> &r, uint chunk, thread const float *a) {
    return ka_dot_k45<12>(r, chunk, a, r.qs);  // qh unused for Q4_K
}
template<> inline float ka_dot<13>(thread const KA<13> &r, uint chunk, thread const float *a) {
    return ka_dot_k45<13>(r, chunk, a, r.qh);
}
template<> inline KA<14> ka_load<14>(device const uchar *row, uint chunk) {
    KA<14> r;
    const uint c=chunk%8;
    device const uchar *p=row+(chunk/8)*210;
    device const ushort *ql=(device const ushort*)(p+(c/4)*64+(c%2)*32);
    device const ushort *qh=(device const ushort*)(p+128+(c/4)*32);
    #pragma unroll
    for(uint w=0;w<16;w++) { r.ql[w]=ql[w]; r.qh[w]=qh[w]; }
    r.sc=*(device const ushort*)(p+192+2*c);
    r.d=*(device const ushort*)(p+208);
    return r;
}
template<> inline float ka_dot<14>(thread const KA<14> &r, uint chunk, thread const float *a) {
    float sum=0.0f;
    const uint c=chunk%8;
    const float d=ka_half(r.d);
    const float ds[2]={d*float(int(as_type<char>(uchar(r.sc&255)))), d*float(int(as_type<char>(uchar(r.sc>>8))))};
    const uint lshift=4*((c/2)%2), hshift=2*(c%4);
    #pragma unroll
    for(uint w=0;w<16;w++) {
        const uint lw=r.ql[w], hw=r.qh[w];
        #pragma unroll
        for(uint e=0;e<2;e++) {
            const uint j=2*w+e;
            const uint lo=(lw>>(8*e+lshift))&15, hi=(hw>>(8*e+hshift))&3;
            sum+=a[j]*(ds[j/16]*float(int(lo|(hi<<4))-32));
        }
    }
    return sum;
}
"""
)


def _ahead_loop(units, nch, src, xoff, guard, fmt="FORMAT"):
    """AHEAD=2 chunk loop over ``units`` = [(tag, sums, row pointer expr)].

    Lane ``lane`` owns chunks lane, lane+32, ... in ascending order (the
    kblock_dot32 law). The (chunk, row, unit) work items of a lane are walked
    in that order with a two-slot register ring: the raw words of item i+1 are
    loaded (ka_load) before item i's arithmetic (ka_dot), so one item's loads
    are always in flight while the previous item computes, at the cost of one
    extra raw set of registers. (A whole chunk step ahead held ROWS x units
    raw sets and was 4x slower from register pressure, 4ede902f7.) The body
    covers two chunks so the item count per step is even and every ring slot
    index is static after unrolling. Each ``sums[r]`` still adds its chunk
    partials in ascending chunk order, so the result equals the AHEAD=1 loop's
    bit for bit. Row pointers clamp to the last valid row and chunk indices to
    the last chunk, so a padded row or the final look-ahead re-reads in range.
    """
    count = len(units)
    decl = "".join(f"    device const uchar *wp_{t}[ROWS];\n" for t, _, _ in units)
    ptrs = "".join(f"wp_{t}[r]={ptr};" for t, _, ptr in units)

    def step(s):
        cs = "chunk" if s == 0 else "c1"
        body = []
        for m, (t, sums, _) in enumerate(units):
            item = f"({s}*ROWS*{count}+r*{count}+{m})"
            if m + 1 < count:
                nxt_ptr, nxt_chunk = f"wp_{units[m + 1][0]}[r]", cs
            else:
                nxt_ptr = f"wp_{units[0][0]}[r+1<ROWS ? r+1 : 0]"
                nxt_chunk = f"(r+1<ROWS ? {cs} : {cs}+32)"
            body.append(
                f"                ring[({item}+1)%2]=ka_load<{fmt}>({nxt_ptr},min({nxt_chunk},NCH-1));\n"
                f"                if({guard}) {sums}[r]+=ka_dot<{fmt}>(ring[{item}%2],{cs},a);\n")
        return (
            "        {\n"
            "            float a[32];\n"
            "            #pragma unroll\n"
            f"            for(uint i=0;i<32;i++) a[i]={src}[{xoff}+{cs}*32+i];\n"
            "            #pragma unroll\n"
            "            for(uint r=0;r<ROWS;r++) {\n"
            + "".join(body)
            + "            }\n"
            "        }\n")

    return (
        "{\n"
        f"    const uint NCH=uint({nch});\n"
        + decl
        + "    #pragma unroll\n"
        f"    for(uint r=0;r<ROWS;r++) {{ {ptrs} }}\n"
        f"    KA<{fmt}> ring[2];\n"
        f"    ring[0]=ka_load<{fmt}>(wp_{units[0][0]}[0],min(lane,NCH-1));\n"
        "    for(uint chunk=lane;chunk<NCH;chunk+=64) {\n"
        "        const uint c1=chunk+32;\n"
        + step(0)
        + "        if(c1>=NCH) break;\n"
        + step(1)
        + "    }\n"
        "}\n")


_LOAD = r"""
float a[32];
#pragma unroll
for(uint i=0;i<32;i++) a[i]=SRC[XOFF+chunk*32+i];
"""

DENSE = (
    r"""
const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
const uint token=threadgroup_position_in_grid.z*TOK;
const uint row0=(threadgroup_position_in_grid.x*4+sg)*ROWS;
float sums[ROWS]={0.0f};
if(!EXACT && TOK>1) {
    // Chunk law shared across TOK tokens: each weight chunk decoded once per
    // row; every (row, token) output keeps the one-token chunk partials and
    // their lane order, so it equals the TOK=1 kernel bit for bit.
    const uint ntok=min(uint(TOK),uint(TOKENS-token));
    float tsums[ROWS][TOK];
    #pragma unroll
    for(uint r=0;r<ROWS;r++)
        #pragma unroll
        for(uint t=0;t<TOK;t++) tsums[r][t]=0.0f;
    for(uint chunk=lane;chunk<K/32;chunk+=32) {
        // Decode 4 values of every weight row, then apply each token's float4
        // once to all rows; each (row, token) partial runs from 0 in ascending j.
        KHdr hdr[ROWS];
        #pragma unroll
        for(uint r=0;r<ROWS;r++) hdr[r]=kb_hdr(w+(WBASE+(row0+r<N ? row0+r : row0))*ROW_BYTES,chunk,FORMAT);
        float part[ROWS][TOK];
        #pragma unroll
        for(uint r=0;r<ROWS;r++)
            #pragma unroll
            for(uint t=0;t<TOK;t++) part[r][t]=0.0f;
        device const float4 *xc=(device const float4*)(x+size_t(token)*K+chunk*32);
        for(uint q=0;q<8;q++) {
            float v[ROWS][4];
            #pragma unroll
            for(uint r=0;r<ROWS;r++) kb_vals4(hdr[r],q,v[r],FORMAT);
            #pragma unroll
            for(uint t=0;t<TOK;t++) {
                if(t<ntok) {
                    const float4 a=xc[size_t(t)*(K/4)+q];
                    #pragma unroll
                    for(uint r=0;r<ROWS;r++) {
                        part[r][t]+=a.x*v[r][0]; part[r][t]+=a.y*v[r][1];
                        part[r][t]+=a.z*v[r][2]; part[r][t]+=a.w*v[r][3];
                    }
                }
            }
        }
        #pragma unroll
        for(uint r=0;r<ROWS;r++)
            #pragma unroll
            for(uint t=0;t<TOK;t++) tsums[r][t]+=part[r][t];
    }
    #pragma unroll
    for(uint r=0;r<ROWS;r++)
        #pragma unroll
        for(uint t=0;t<TOK;t++) {
            const float sum=simd_sum(tsums[r][t]);
            if(lane==0 && row0+r<N && token+t<TOKENS) y[size_t(token+t)*N+row0+r]=sum;
        }
    return;
}
if(EXACT && TOK>1) {
    // TOK tokens share each decoded weight value; every (row, token) output
    // keeps packed_metal's k sequence and x*value products bit for bit.
    float tsums[ROWS][TOK];
    #pragma unroll
    for(uint r=0;r<ROWS;r++)
        #pragma unroll
        for(uint t=0;t<TOK;t++) tsums[r][t]=0.0f;
    for(uint k=lane;k<K;k+=32) {
        float xv[TOK];
        #pragma unroll
        for(uint t=0;t<TOK;t++) xv[t]=token+t<TOKENS ? x[size_t(token+t)*K+k] : 0.0f;
        #pragma unroll
        for(uint r=0;r<ROWS;r++) {
            if(row0+r<N) {
                const float v=gguf_k_value(w+(WBASE+row0+r)*ROW_BYTES,k,FORMAT);
                #pragma unroll
                for(uint t=0;t<TOK;t++) tsums[r][t]+=xv[t]*v;
            }
        }
    }
    #pragma unroll
    for(uint r=0;r<ROWS;r++)
        #pragma unroll
        for(uint t=0;t<TOK;t++) {
            const float sum=simd_sum(tsums[r][t]);
            if(lane==0 && row0+r<N && token+t<TOKENS) y[size_t(token+t)*N+row0+r]=sum;
        }
    return;
}
if(EXACT) {
    // packed_metal DENSE's own law: k=lane,lane+32,..., gguf_k_value per element.
    for(uint k=lane;k<K;k+=32) {
        const float xv=x[size_t(token)*K+k];
        #pragma unroll
        for(uint r=0;r<ROWS;r++)
            if(row0+r<N) sums[r]+=xv*gguf_k_value(w+(WBASE+row0+r)*ROW_BYTES,k,FORMAT);
    }
} else if(AHEAD>1)
"""
    + _ahead_loop([("w", "sums", "w+(WBASE+min(row0+r,uint(N-1)))*ROW_BYTES")], "K/32", "x", "size_t(token)*K", "row0+r<N")
    + r"""else
for(uint chunk=lane;chunk<K/32;chunk+=32) {
"""
    + _LOAD.replace("SRC", "x").replace("XOFF", "size_t(token)*K")
    + r"""
    #pragma unroll
    for(uint r=0;r<ROWS;r++)
        if(row0+r<N) sums[r]+=KDOT(w+(WBASE+row0+r)*ROW_BYTES,chunk,a,FORMAT);
}
#pragma unroll
for(uint r=0;r<ROWS;r++) {
    const float sum=simd_sum(sums[r]);
    if(lane==0 && row0+r<N) y[size_t(token)*N+row0+r]=sum;
}
"""
)

GATE_UP = (
    r"""
const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
const uint slot=threadgroup_position_in_grid.z, token=slot/TOPK;
const uint row0=(threadgroup_position_in_grid.x*4+sg)*ROWS;
const uint expert=ids[slot];
if(expert>=EXPERTS) {
    if(lane==0) for(uint r=0;r<ROWS && row0+r<M;r++) mid[size_t(slot)*M+row0+r]=NAN;
    return;
}
float sums_g[ROWS]={0.0f},sums_u[ROWS]={0.0f};
if(EXACT) {
    for(uint k=lane;k<K;k+=32) {
        const float xv=x[size_t(token)*K+k];
        #pragma unroll
        for(uint r=0;r<ROWS;r++) {
            if(row0+r<M) {
                const size_t offset=(size_t(expert)*M+row0+r)*ROW_BYTES;
                sums_g[r]+=xv*gguf_k_value(gate+offset,k,FORMAT);
                sums_u[r]+=xv*gguf_k_value(up+offset,k,FORMAT);
            }
        }
    }
} else if(AHEAD>1)
"""
    + _ahead_loop([("g", "sums_g", "gate+(size_t(expert)*M+min(row0+r,uint(M-1)))*ROW_BYTES"),
                   ("u", "sums_u", "up+(size_t(expert)*M+min(row0+r,uint(M-1)))*ROW_BYTES")],
                  "K/32", "x", "size_t(token)*K", "row0+r<M")
    + r"""else
for(uint chunk=lane;chunk<K/32;chunk+=32) {
"""
    + _LOAD.replace("SRC", "x").replace("XOFF", "size_t(token)*K")
    + r"""
    #pragma unroll
    for(uint r=0;r<ROWS;r++) {
        if(row0+r<M) {
            const size_t offset=(size_t(expert)*M+row0+r)*ROW_BYTES;
            sums_g[r]+=KDOT(gate+offset,chunk,a,FORMAT);
            sums_u[r]+=KDOT(up+offset,chunk,a,FORMAT);
        }
    }
}
#pragma unroll
for(uint r=0;r<ROWS;r++) {
    float g=simd_sum(sums_g[r]),u=simd_sum(sums_u[r]);
    if(lane==0 && row0+r<M) {
        // packed_metal GATE_UP's activation law, unchanged.
        g=min(g,10.0f);u=clamp(u,-10.0f,10.0f);
        mid[size_t(slot)*M+row0+r]=(g/(1.0f+exp(-g)))*u;
    }
}
"""
)

DOWN = (
    r"""
const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
const uint slot=threadgroup_position_in_grid.z;
const uint row0=(threadgroup_position_in_grid.x*4+sg)*ROWS;
const uint expert=ids[slot];
if(expert>=EXPERTS) {
    if(lane==0) for(uint r=0;r<ROWS && row0+r<N;r++) out[size_t(slot)*N+row0+r]=NAN;
    return;
}
float sums[ROWS]={0.0f};
if(EXACT) {
    for(uint k=lane;k<M;k+=32) {
        const float xv=mid[size_t(slot)*M+k];
        #pragma unroll
        for(uint r=0;r<ROWS;r++)
            if(row0+r<N) sums[r]+=xv*gguf_k_value(w+(size_t(expert)*N+row0+r)*ROW_BYTES,k,FORMAT);
    }
} else if(AHEAD>1)
"""
    + _ahead_loop([("d", "sums", "w+(size_t(expert)*N+min(row0+r,uint(N-1)))*ROW_BYTES")],
                  "M/32", "mid", "size_t(slot)*M", "row0+r<N")
    + r"""else
for(uint chunk=lane;chunk<M/32;chunk+=32) {
"""
    + _LOAD.replace("SRC", "mid").replace("XOFF", "size_t(slot)*M")
    + r"""
    #pragma unroll
    for(uint r=0;r<ROWS;r++)
        if(row0+r<N) sums[r]+=KDOT(w+(size_t(expert)*N+row0+r)*ROW_BYTES,chunk,a,FORMAT);
}
#pragma unroll
for(uint r=0;r<ROWS;r++) {
    const float sum=simd_sum(sums[r]);
    if(lane==0 && row0+r<N) out[size_t(slot)*N+row0+r]=sum;
}
"""
)

# Expert-major routed experts for multi-row decode/verify (chunk law only).
# Slots are sorted by expert; threadgroup z owns G consecutive sorted slots and
# walks their same-expert runs, decoding each weight chunk once per run and
# applying it to every slot of the run. Each slot's partials run from 0 in
# ascending j, chunk partials add in lane order and simd_sum reduces, so every
# mid/out row equals GATE_UP/DOWN's for that slot alone.
_UNION_HEAD = r"""
const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
const uint row0=(threadgroup_position_in_grid.x*4+sg)*ROWS;
// One threadgroup per sorted position; only the first slot of each run of up
// to G same-expert slots works, so distinct experts run in parallel.
const uint p=threadgroup_position_in_grid.z;
const uint expert0=ids[order[p]];
uint first=p;
while(first>0 && ids[order[first-1]]==expert0) first--;
if((p-first)%G) return;
uint last=p+1;
while(last<SLOTS && last<p+G && ids[order[last]]==expert0) last++;
for(uint i=p;i<last;) {
    const uint expert=expert0;
    const uint j=last;
    const uint n=j-i;
    uint slot[G];
    for(uint t=0;t<G;t++) slot[t]=order[t<n ? i+t : i];
"""
_UNION_GATE_UP = _UNION_HEAD + r"""
    if(expert>=EXPERTS) {
        if(lane==0) for(uint t=0;t<n;t++) for(uint r=0;r<ROWS && row0+r<M;r++) mid[size_t(slot[t])*M+row0+r]=NAN;
        i=j; continue;
    }
    float sg_[ROWS][G], su_[ROWS][G];
    for(uint r=0;r<ROWS;r++) for(uint t=0;t<G;t++) { sg_[r][t]=0.0f; su_[r][t]=0.0f; }
    for(uint chunk=lane;chunk<K/32;chunk+=32) {
        KHdr hg[ROWS], hu[ROWS];
        #pragma unroll
        for(uint r=0;r<ROWS;r++) {
            const size_t offset=(size_t(expert)*M+(row0+r<M ? row0+r : row0))*ROW_BYTES;
            hg[r]=kb_hdr(gate+offset,chunk,FORMAT); hu[r]=kb_hdr(up+offset,chunk,FORMAT);
        }
        float pg[ROWS][G], pu[ROWS][G];
        for(uint r=0;r<ROWS;r++) for(uint t=0;t<G;t++) { pg[r][t]=0.0f; pu[r][t]=0.0f; }
        for(uint q=0;q<8;q++) {
            float vg[ROWS][4], vu[ROWS][4];
            #pragma unroll
            for(uint r=0;r<ROWS;r++) { kb_vals4(hg[r],q,vg[r],FORMAT); kb_vals4(hu[r],q,vu[r],FORMAT); }
            for(uint t=0;t<G;t++) {
                if(t<n) {
                    const float4 a=((device const float4*)(x+size_t(slot[t]/TOPK)*K+chunk*32))[q];
                    #pragma unroll
                    for(uint r=0;r<ROWS;r++) {
                        pg[r][t]+=a.x*vg[r][0]; pg[r][t]+=a.y*vg[r][1]; pg[r][t]+=a.z*vg[r][2]; pg[r][t]+=a.w*vg[r][3];
                        pu[r][t]+=a.x*vu[r][0]; pu[r][t]+=a.y*vu[r][1]; pu[r][t]+=a.z*vu[r][2]; pu[r][t]+=a.w*vu[r][3];
                    }
                }
            }
        }
        for(uint r=0;r<ROWS;r++) for(uint t=0;t<G;t++) { sg_[r][t]+=pg[r][t]; su_[r][t]+=pu[r][t]; }
    }
    for(uint r=0;r<ROWS;r++) {
        for(uint t=0;t<G;t++) {
            float g=simd_sum(sg_[r][t]),u=simd_sum(su_[r][t]);
            if(lane==0 && t<n && row0+r<M) {
                g=min(g,10.0f);u=clamp(u,-10.0f,10.0f);
                mid[size_t(slot[t])*M+row0+r]=(g/(1.0f+exp(-g)))*u;
            }
        }
    }
    i=j;
}
"""
_UNION_DOWN = _UNION_HEAD + r"""
    if(expert>=EXPERTS) {
        if(lane==0) for(uint t=0;t<n;t++) for(uint r=0;r<ROWS && row0+r<N;r++) out[size_t(slot[t])*N+row0+r]=NAN;
        i=j; continue;
    }
    float ss[ROWS][G];
    for(uint r=0;r<ROWS;r++) for(uint t=0;t<G;t++) ss[r][t]=0.0f;
    for(uint chunk=lane;chunk<M/32;chunk+=32) {
        KHdr hw[ROWS];
        #pragma unroll
        for(uint r=0;r<ROWS;r++) hw[r]=kb_hdr(w+(size_t(expert)*N+(row0+r<N ? row0+r : row0))*ROW_BYTES,chunk,FORMAT);
        float pp[ROWS][G];
        for(uint r=0;r<ROWS;r++) for(uint t=0;t<G;t++) pp[r][t]=0.0f;
        for(uint q=0;q<8;q++) {
            float v[ROWS][4];
            #pragma unroll
            for(uint r=0;r<ROWS;r++) kb_vals4(hw[r],q,v[r],FORMAT);
            for(uint t=0;t<G;t++) {
                if(t<n) {
                    const float4 a=((device const float4*)(mid+size_t(slot[t])*M+chunk*32))[q];
                    #pragma unroll
                    for(uint r=0;r<ROWS;r++) { pp[r][t]+=a.x*v[r][0]; pp[r][t]+=a.y*v[r][1]; pp[r][t]+=a.z*v[r][2]; pp[r][t]+=a.w*v[r][3]; }
                }
            }
        }
        for(uint r=0;r<ROWS;r++) for(uint t=0;t<G;t++) ss[r][t]+=pp[r][t];
    }
    for(uint r=0;r<ROWS;r++)
        for(uint t=0;t<G;t++) {
            const float sum=simd_sum(ss[r][t]);
            if(lane==0 && t<n && row0+r<N) out[size_t(slot[t])*N+row0+r]=sum;
        }
    i=j;
}
"""

# The reference DOWN kernel's per-token combine: total += score * sum, slots in order.
COMBINE = r"""
const uint row=thread_position_in_grid.x,token=thread_position_in_grid.y;
if(row>=N) return;
float total=0.0f;
for(uint slot=0;slot<TOPK;slot++) {
    const uint index=token*TOPK+slot;
    total+=scores[index]*expert_out[size_t(index)*N+row];
}
y[size_t(token)*N+row]=total;
"""


# Shared-expert fold: GLM's shared expert has the routed experts' formats,
# shapes and clipped SwiGLU, so it runs as slot TOPK of each token in the same
# launches (z = tokens x (TOPK+1)). Its activation becomes the routed kernel's
# g/(1+exp(-g)) instead of MLX nn.silu -- a distinct FP32 law -- and the combine
# adds it after the routed slots, matching the reference routed + shared order.
_FOLD_IDS = (
    "const uint k_slot=slot%(TOPK+1), fold_token=slot/(TOPK+1);\n"
    "const bool is_shared=k_slot==TOPK;\n"
    "const uint expert=is_shared ? 0u : ids[fold_token*TOPK+k_slot];"
)
GATE_UP_SHARED = GATE_UP.replace(
    "const uint slot=threadgroup_position_in_grid.z, token=slot/TOPK;",
    "const uint slot=threadgroup_position_in_grid.z, token=slot/(TOPK+1);",
).replace("const uint expert=ids[slot];", _FOLD_IDS).replace(
    "KDOT(gate+offset", "KDOT((is_shared ? gate_sh : gate)+offset"
).replace("KDOT(up+offset", "KDOT((is_shared ? up_sh : up)+offset").replace(
    "gguf_k_value(gate+offset", "gguf_k_value((is_shared ? gate_sh : gate)+offset"
).replace("gguf_k_value(up+offset", "gguf_k_value((is_shared ? up_sh : up)+offset").replace(
    "wp_g[r]=gate+", "wp_g[r]=(is_shared ? gate_sh : gate)+").replace(
    "wp_u[r]=up+", "wp_u[r]=(is_shared ? up_sh : up)+")
DOWN_SHARED = DOWN.replace("const uint expert=ids[slot];", _FOLD_IDS).replace(
    "KDOT(w+(size_t(expert)", "KDOT((is_shared ? w_sh : w)+(size_t(expert)"
).replace("gguf_k_value(w+(size_t(expert)", "gguf_k_value((is_shared ? w_sh : w)+(size_t(expert)").replace(
    "wp_d[r]=w+", "wp_d[r]=(is_shared ? w_sh : w)+")
COMBINE_SHARED = r"""
const uint row=thread_position_in_grid.x,token=thread_position_in_grid.y;
if(row>=N) return;
float total=0.0f;
for(uint slot=0;slot<TOPK;slot++)
    total+=scores[token*TOPK+slot]*expert_out[size_t(token*(TOPK+1)+slot)*N+row];
y[size_t(token)*N+row]=total+expert_out[size_t(token*(TOPK+1)+TOPK)*N+row];
"""


@lru_cache(maxsize=16)
def _kernel(kind):
    import mlx.core as mx

    args = {
        "dense": (["x", "w"], ["y"], DENSE.replace("WBASE", "size_t(0)")),
        "grouped": (["x", "w"], ["y"], DENSE.replace("WBASE", "size_t(token%GROUPS)*N")),
        "gate_up": (["x", "ids", "gate", "up"], ["mid"], GATE_UP),
        "down": (["mid", "ids", "w"], ["out"], DOWN),
        "combine": (["expert_out", "scores"], ["y"], COMBINE),
        "gate_up_shared": (["x", "ids", "gate", "up", "gate_sh", "up_sh"], ["mid"], GATE_UP_SHARED),
        "down_shared": (["mid", "ids", "w", "w_sh"], ["out"], DOWN_SHARED),
        "combine_shared": (["expert_out", "scores"], ["y"], COMBINE_SHARED),
        "gate_up_union": (["x", "ids", "order", "gate", "up"], ["mid"], _UNION_GATE_UP),
        "down_union": (["mid", "ids", "order", "w"], ["out"], _UNION_DOWN),
    }
    inputs, outputs, source = args[kind]
    return mx.fast.metal_kernel(
        name=f"mlx2_gguf_kblock_{kind}_v1",
        input_names=inputs,
        output_names=outputs,
        header=KBLOCK_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


def exact_law(ggml_type):
    """Formats whose chunk-major reduction is not closer to fp64 than stock.

    Classification m6b-classify (exactness-tolerance classes): chunk-major
    sums were 0.47-0.61x stock's fp64 RMS error on Q3_K/Q4_K/Q5_K/Q6_K (class 2
    candidate) but 1.39x on Q4_0, where K=128 rows leave four lanes summing 32
    values serially. Q4_0 therefore keeps packed_metal's exact per-row law.
    """
    return ggml_type == 2


# Research-only lossless Q3_K container (see kblock_dot32_v2): 256 values per
# 148-byte block. Never produced by an artifact loader.
Q3N_FORMAT, Q3N_BLOCK_BYTES = 111, 148


@lru_cache(maxsize=256)
def kblock_admits(ggml_type, k):
    """K-block rows whose input width is whole chunks of their block size."""
    if ggml_type == Q3N_FORMAT:
        return k % 256 == 0
    block = K_FORMATS.get(ggml_type, (0,))[0]
    return block > 0 and k % max(block, 32) == 0


AHEAD_VALUES = (1, 2)
_AHEAD_FORMATS = (2, 11, 12, 13, 14)


def _ahead(ahead, vec, exact, ggml_type=None):
    """Effective AHEAD template value of one chunk-law launch.

    ``ahead=2`` is the load-ahead form of the VEC=1 chunk law (ka_load/ka_dot),
    bitwise equal to ahead=1 and slower on the M5 Max in every form measured
    (see GGUFWeights ``kblock_rows_ahead2``); it stays a default-off research
    value;
    the exact per-element law and the research containers have no load-ahead
    body and keep AHEAD=1 (their own law, unchanged). Any other VEC with
    ``ahead=2`` is refused rather than silently run under another decoder.
    """
    if ahead not in AHEAD_VALUES:
        raise GGUFError(f"K-block ahead must be one of {AHEAD_VALUES}")
    if ahead == 1 or exact or (ggml_type is not None and ggml_type not in _AHEAD_FORMATS):
        return 1
    if vec != 1:
        raise GGUFError("K-block load-ahead is the VEC=1 chunk law only")
    return ahead


def _grid(n, rows):
    return ((n + 4 * rows - 1) // (4 * rows)) * 128


@lru_cache(maxsize=1024)
def _dense_launch(shape, ggml_type, row_bytes, rows, vec, exact, tok, b, ahead=1):
    """(template, grid, output_shapes) of one kblock_dense launch; pure in its
    integer arguments, so the per-step host cost is one cache lookup."""
    n, k = shape
    is_exact = int(bool(exact) or exact_law(ggml_type))
    # Exact law: tokens share by default (its per-token order is fixed). The
    # chunk law shares only when asked (tok>1, VEC=1, K-quant formats), with
    # each token's chunk partials and lane order equal to the TOK=1 kernel.
    tok = (8 if b >= 8 else 1) if tok is None and is_exact else (tok or 1)
    if tok > 1 and not is_exact and (vec != 1 or ggml_type not in (11, 12, 13, 14)):
        raise GGUFError("K-block chunk-law token sharing needs VEC=1 and a K-quant format")
    # The token-shared chunk body (TOK>1) has no load-ahead form: AHEAD=1.
    ahead = _ahead(ahead, vec, is_exact, ggml_type) if tok == 1 else _ahead(ahead, vec, 1)
    return (
        [("K", k), ("N", n), ("ROW_BYTES", row_bytes),
         ("FORMAT", ggml_type), ("ROWS", rows), ("VEC", vec),
         ("EXACT", is_exact), ("TOK", tok), ("TOKENS", b), ("AHEAD", ahead)],
        (_grid(n, rows), 1, (b + tok - 1) // tok),
        [(b, n)],
    )


def kblock_dense(x, weight: PackedMatrix, *, rows=4, vec=1, exact=0, tok=None, ahead=1):
    import mlx.core as mx

    shape = weight.shape
    if len(shape) != 2 or not kblock_admits(weight.ggml_type, shape[1]):
        raise GGUFError("K-block dense needs a rank2 K-block weight of whole chunks")
    xs = x.shape
    if len(xs) != 2 or xs[-1] != shape[1] or rows not in (1, 2, 4, 8):
        raise GGUFError("K-block dense needs matching rank2 input and rows 1/2/4/8")
    template, grid, output_shapes = _dense_launch(
        shape, weight.ggml_type, weight.row_bytes, rows, vec, exact, tok, xs[0], ahead)
    return _kernel("dense")(
        inputs=[x.astype(mx.float32), weight.data],
        template=template, grid=grid, threadgroup=(128, 1, 1),
        output_shapes=output_shapes, output_dtypes=[mx.float32],
    )[0]


def kblock_moe_union(x, ids, scores, gate: PackedMatrix, up: PackedMatrix, down: PackedMatrix, *, rows=4, group=2,
                     ahead=1):
    """kblock_moe for several rows with each routed expert decoded once per run
    of sorted slots (chunk law; K-quant experts). Every slot's mid/out row and
    the slot-ordered combine equal kblock_moe's. ``ahead`` is validated and
    accepted for policy symmetry; the union body has no load-ahead form, so it
    always runs AHEAD=1 (the same law either way)."""
    import mlx.core as mx

    _ahead(ahead, 1, 1)
    if (len(gate.shape) != 3 or gate.shape != up.shape or gate.ggml_type != up.ggml_type
            or gate.ggml_type not in (11, 12, 13, 14) or down.ggml_type not in (11, 12, 13, 14)):
        raise GGUFError("expert union needs matching K-quant gate/up and a K-quant down")
    experts, m, k = gate.shape
    if (down.shape != (experts, k, m) or x.ndim != 2 or x.shape[-1] != k
            or not kblock_admits(gate.ggml_type, k) or not kblock_admits(down.ggml_type, m)):
        raise GGUFError("K-block expert/input axes or formats mismatch")
    if ids.ndim != 2 or ids.dtype != mx.uint32 or scores.shape != ids.shape or ids.shape[0] != x.shape[0]:
        raise GGUFError("matching [tokens,topk] uint32 IDs and scores required")
    b, topk = ids.shape
    slots = b * topk
    flat = ids.reshape(-1)
    order = mx.argsort(flat).astype(mx.uint32)
    common = [("K", k), ("M", m), ("N", k), ("TOPK", topk), ("EXPERTS", experts), ("ROWS", rows),
              ("VEC", 1), ("EXACT", 0), ("G", group), ("SLOTS", slots)]
    zs = slots
    mid = _kernel("gate_up_union")(
        inputs=[x.astype(mx.float32), flat, order, gate.data, up.data],
        template=common + [("ROW_BYTES", gate.row_bytes), ("FORMAT", gate.ggml_type)],
        grid=(_grid(m, rows), 1, zs), threadgroup=(128, 1, 1),
        output_shapes=[(slots, m)], output_dtypes=[mx.float32],
    )[0]
    expert_out = _kernel("down_union")(
        inputs=[mid, flat, order, down.data],
        template=common + [("ROW_BYTES", down.row_bytes), ("FORMAT", down.ggml_type)],
        grid=(_grid(k, rows), 1, zs), threadgroup=(128, 1, 1),
        output_shapes=[(slots, k)], output_dtypes=[mx.float32],
    )[0]
    return _kernel("combine")(
        inputs=[expert_out, scores.astype(mx.float32)],
        template=common,
        grid=(k, b, 1), threadgroup=(min(k, 256), 1, 1),
        output_shapes=[(b, k)], output_dtypes=[mx.float32],
    )[0]


def kblock_grouped(x, weight: PackedMatrix, *, rows=4, vec=1, exact=0, ahead=1):
    import mlx.core as mx

    if len(weight.shape) != 3 or not kblock_admits(weight.ggml_type, weight.shape[2]):
        raise GGUFError("K-block grouped needs a rank3 K-block weight of whole chunks")
    groups, n, k = weight.shape
    if x.ndim < 2 or x.shape[-2:] != (groups, k):
        raise GGUFError("K-block grouped input axes mismatch")
    leading = x.shape[:-2]
    b = math.prod(leading)
    y = _kernel("grouped")(
        inputs=[x.reshape(b * groups, k).astype(mx.float32), weight.data],
        template=[("K", k), ("N", n), ("GROUPS", groups), ("ROW_BYTES", weight.row_bytes),
                  ("FORMAT", weight.ggml_type), ("ROWS", rows), ("VEC", vec),
                  ("EXACT", int(bool(exact) or exact_law(weight.ggml_type))), ("TOK", 1),
                  ("TOKENS", b * groups),
                  ("AHEAD", _ahead(ahead, vec, bool(exact) or exact_law(weight.ggml_type), weight.ggml_type))],
        grid=(_grid(n, rows), 1, b * groups), threadgroup=(128, 1, 1),
        output_shapes=[(b * groups, n)], output_dtypes=[mx.float32],
    )[0]
    return y.reshape(*leading, groups, n)


def kblock_moe(x, ids, scores, gate: PackedMatrix, up: PackedMatrix, down: PackedMatrix, *, rows=4, vec=1, exact=0,
               ahead=1):
    """Routed experts with the reference activation and slot-ordered combine."""
    import mlx.core as mx

    if len(gate.shape) != 3 or gate.shape != up.shape or gate.ggml_type != up.ggml_type:
        raise GGUFError("matching K-block rank3 gate/up required")
    experts, m, k = gate.shape
    if (down.shape != (experts, k, m) or x.ndim != 2 or x.shape[-1] != k
            or not kblock_admits(gate.ggml_type, k) or not kblock_admits(down.ggml_type, m)):
        raise GGUFError("K-block expert/input axes or formats mismatch")
    if ids.ndim != 2 or ids.dtype != mx.uint32 or scores.shape != ids.shape or ids.shape[0] != x.shape[0]:
        raise GGUFError("matching [tokens,topk] uint32 IDs and scores required")
    b, topk = ids.shape
    common = [("K", k), ("M", m), ("N", k), ("TOPK", topk), ("EXPERTS", experts), ("ROWS", rows), ("VEC", vec), ("EXACT", exact)]
    gate_ahead = _ahead(ahead, vec, exact, gate.ggml_type)
    down_ahead = _ahead(ahead, vec, exact, down.ggml_type)
    mid = _kernel("gate_up")(
        inputs=[x.astype(mx.float32), ids, gate.data, up.data],
        template=common + [("ROW_BYTES", gate.row_bytes), ("FORMAT", gate.ggml_type), ("AHEAD", gate_ahead)],
        grid=(_grid(m, rows), 1, b * topk), threadgroup=(128, 1, 1),
        output_shapes=[(b * topk, m)], output_dtypes=[mx.float32],
    )[0]
    expert_out = _kernel("down")(
        inputs=[mid, ids, down.data],
        template=common + [("ROW_BYTES", down.row_bytes), ("FORMAT", down.ggml_type), ("AHEAD", down_ahead)],
        grid=(_grid(k, rows), 1, b * topk), threadgroup=(128, 1, 1),
        output_shapes=[(b * topk, k)], output_dtypes=[mx.float32],
    )[0]
    return _kernel("combine")(
        inputs=[expert_out, scores.astype(mx.float32)],
        template=common,
        grid=(k, b, 1), threadgroup=(min(k, 256), 1, 1),
        output_shapes=[(b, k)], output_dtypes=[mx.float32],
    )[0]


def _moe_shared_static(gate, up, down, gate_sh, up_sh, down_sh):
    return (gate.shape, up.shape, down.shape, gate_sh.shape, up_sh.shape, down_sh.shape,
            gate.ggml_type, up.ggml_type, down.ggml_type, gate_sh.ggml_type, up_sh.ggml_type,
            down_sh.ggml_type, gate.row_bytes, down.row_bytes, gate_sh.row_bytes, down_sh.row_bytes)


@lru_cache(maxsize=256)
def _moe_shared_launch(static, x_shape, ids_shape, scores_shape, ids_u32, rows, vec, exact, ahead=1):
    """Validated (templates, grids, output shapes) of kblock_moe_shared's three
    launches; pure in the static weight geometry and the input shapes."""
    (gate_shape, up_shape, down_shape, gate_sh_shape, up_sh_shape, down_sh_shape,
     gate_t, up_t, down_t, gate_sh_t, up_sh_t, down_sh_t,
     gate_rb, down_rb, gate_sh_rb, down_sh_rb) = static
    if len(gate_shape) != 3 or gate_shape != up_shape or gate_t != up_t:
        raise GGUFError("matching K-block rank3 gate/up required")
    experts, m, k = gate_shape
    if (down_shape != (experts, k, m) or len(x_shape) != 2 or x_shape[-1] != k
            or not kblock_admits(gate_t, k) or not kblock_admits(down_t, m)):
        raise GGUFError("K-block expert/input axes or formats mismatch")
    if (gate_sh_shape != (m, k) or up_sh_shape != (m, k) or down_sh_shape != (k, m)
            or {gate_sh_t, up_sh_t} != {gate_t}
            or down_sh_t != down_t
            or gate_sh_rb != gate_rb or down_sh_rb != down_rb):
        raise GGUFError("shared expert must match the routed expert formats and shapes")
    if len(ids_shape) != 2 or not ids_u32 or scores_shape != ids_shape or ids_shape[0] != x_shape[0]:
        raise GGUFError("matching [tokens,topk] uint32 IDs and scores required")
    b, topk = ids_shape
    slots = b * (topk + 1)
    common = [("K", k), ("M", m), ("N", k), ("TOPK", topk), ("EXPERTS", experts), ("ROWS", rows), ("VEC", vec), ("EXACT", exact)]
    return (
        (common + [("ROW_BYTES", gate_rb), ("FORMAT", gate_t), ("AHEAD", _ahead(ahead, vec, exact, gate_t))],
         (_grid(m, rows), 1, slots), [(slots, m)]),
        (common + [("ROW_BYTES", down_rb), ("FORMAT", down_t), ("AHEAD", _ahead(ahead, vec, exact, down_t))],
         (_grid(k, rows), 1, slots), [(slots, k)]),
        (common, (k, b, 1), (min(k, 256), 1, 1), [(b, k)]),
    )


def kblock_moe_shared(x, ids, scores, gate, up, down, gate_sh, up_sh, down_sh, *, rows=4, vec=1, exact=0, ahead=1):
    """Routed experts plus a format-identical shared expert in the same launches."""
    import mlx.core as mx

    first, second, third = _moe_shared_launch(
        _moe_shared_static(gate, up, down, gate_sh, up_sh, down_sh),
        x.shape, ids.shape, scores.shape, ids.dtype == mx.uint32, rows, vec, exact, ahead)
    mid = _kernel("gate_up_shared")(
        inputs=[x.astype(mx.float32), ids, gate.data, up.data, gate_sh.data, up_sh.data],
        template=first[0], grid=first[1], threadgroup=(128, 1, 1),
        output_shapes=first[2], output_dtypes=[mx.float32],
    )[0]
    expert_out = _kernel("down_shared")(
        inputs=[mid, ids, down.data, down_sh.data],
        template=second[0], grid=second[1], threadgroup=(128, 1, 1),
        output_shapes=second[2], output_dtypes=[mx.float32],
    )[0]
    return _kernel("combine_shared")(
        inputs=[expert_out, scores.astype(mx.float32)],
        template=third[0], grid=third[1], threadgroup=third[2],
        output_shapes=third[3], output_dtypes=[mx.float32],
    )[0]


def _multi_source(count):
    """Several K-block projections in one launch, one concatenated output.

    Rows are numbered across the tensors (R0=0 ... R<count>); each simdgroup's
    ROWS-block lies inside one tensor (row counts are multiples of ROWS), reads
    that tensor's own input and format, and runs kblock_dense's exact per-row
    reduction, so every output equals the separate kblock_dense call bit for bit.
    """
    head = r"""
const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
const uint token=threadgroup_position_in_grid.z;
const uint row0=(threadgroup_position_in_grid.x*4+sg)*ROWS;
"""
    body = r"""
    {
        const uint local=row0-R@;
        float sums[ROWS]={0.0f};
        if(E@) {
            for(uint k=lane;k<K@;k+=32) {
                const float xv=x@[size_t(token)*K@+k];
                #pragma unroll
                for(uint r=0;r<ROWS;r++)
                    if(local+r<N@) sums[r]+=xv*gguf_k_value(w@+size_t(local+r)*RB@,k,F@);
            }
        } else if(AHEAD>1)
""" + _ahead_loop([("m", "sums", "w@+size_t(min(local+r,uint(N@-1)))*RB@")], "K@/32", "x@",
                  "size_t(token)*K@", "local+r<N@", fmt="F@") + r"""        else
        for(uint chunk=lane;chunk<K@/32;chunk+=32) {
            float a[32];
            #pragma unroll
            for(uint i=0;i<32;i++) a[i]=x@[size_t(token)*K@+chunk*32+i];
            #pragma unroll
            for(uint r=0;r<ROWS;r++)
                if(local+r<N@) sums[r]+=KDOT(w@+size_t(local+r)*RB@,chunk,a,F@);
        }
        #pragma unroll
        for(uint r=0;r<ROWS;r++) {
            const float sum=simd_sum(sums[r]);
            if(lane==0 && local+r<N@) y[size_t(token)*RTOTAL+R@+local+r]=sum;
        }
        return;
    }
"""
    source = head
    for t in range(count):
        source += f"if(row0<R{t + 1})" + body.replace("@", str(t))
    return source


@lru_cache(maxsize=4)
def _multi_kernel(count):
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name=f"mlx2_gguf_kblock_multi{count}_v1",
        input_names=[f"x{t}" for t in range(count)] + [f"w{t}" for t in range(count)],
        output_names=["y"],
        header=KBLOCK_HEADER,
        source=_multi_source(count),
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=512)
def _multi_launch(static, x_shapes, rows, vec, exact, ahead=1):
    """Validated (template, grid, output_shapes) of one kblock_multi launch;
    ``static`` is each weight's (shape, ggml_type, row_bytes)."""
    if not 1 <= len(static) <= 6 or len(x_shapes) != len(static):
        raise GGUFError("K-block multi projection takes 1-6 matching inputs and weights")
    b = x_shapes[0][0]
    # One AHEAD per launch; exact-law tensors (E@) take their own branch first,
    # and a launch holding a format without a load-ahead body keeps AHEAD=1.
    formats = {t for _, t, _ in static if not (exact or exact_law(t))}
    template, start = [("ROWS", rows), ("VEC", vec), ("EXACT", exact),
                       ("AHEAD", _ahead(ahead, vec, exact) if formats <= set(_AHEAD_FORMATS) else _ahead(ahead, vec, 1))], 0
    for t, (xs, (shape, ggml_type, row_bytes)) in enumerate(zip(x_shapes, static)):
        n, k = shape if len(shape) == 2 else (0, 0)
        if (len(shape) != 2 or not kblock_admits(ggml_type, k) or n % rows
                or xs != (b, k)):
            raise GGUFError("K-block multi projection needs rank2 K-block weights with ROWS-aligned rows")
        template += [(f"R{t}", start), (f"N{t}", n), (f"K{t}", k), (f"F{t}", ggml_type),
                     (f"RB{t}", row_bytes), (f"E{t}", int(bool(exact) or exact_law(ggml_type)))]
        start += n
    template += [(f"R{len(static)}", start), ("RTOTAL", start)]
    return template, (_grid(start, rows), 1, b), [(b, start)]


def kblock_multi(inputs, weights, *, rows=4, vec=1, exact=0, ahead=1):
    """Concatenated [b, sum(N)] output of several rank2 K-block projections."""
    import mlx.core as mx

    template, grid, output_shapes = _multi_launch(
        tuple((w.shape, w.ggml_type, w.row_bytes) for w in weights),
        tuple(x.shape for x in inputs), rows, vec, exact, ahead)
    return _multi_kernel(len(weights))(
        inputs=[x.astype(mx.float32) for x in inputs] + [w.data for w in weights],
        template=template, grid=grid, threadgroup=(128, 1, 1),
        output_shapes=output_shapes, output_dtypes=[mx.float32],
    )[0]
