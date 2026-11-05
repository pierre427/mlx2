# SPDX-License-Identifier: Apache-2.0
"""Opt-in M5 native FP multi-row attention. See provenance/apple-four-gaps.json.

16 fused query/head rows share each KV tile. Tensor QK uses native operands;
PV uses FP16 high/residual probabilities and FP32 accumulators. History splits
merge before the learned sink is added exactly once. Explicit bool masks cover
trees, padding and rotating-cache views; unsupported shapes retain stock SDPA.
This is a distinct numerical profile, default off and unqualified.
"""
from collections import Counter
from functools import lru_cache
import os
import math

import mlx.core as mx

VERSION = "m5-fp-verify-tensor-v1"
# Kept off: the owned Flint 8-row short-context warm A/B was slower than
# stock SDPA. Long-context speed and general model qualification are unproven;
# see docs/research/apple-four-gaps-20261103/README.md. Selection is explicit.
COUNTERS = Counter()


def status():
    return {'implemented':True,'selected':os.environ.get('MLX2_FP_VERIFY_KERNEL')=='1',
            'qualified':False,'observed_host_dispatch':COUNTERS['calls']>0,
            'profile':VERSION,'counts':dict(COUNTERS)}


HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""
SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const int fused0 = int(threadgroup_position_in_grid.x) * 16;
    const int bh = int(threadgroup_position_in_grid.y);
    const int b = bh / HK, hk = bh % HK;
    const int split = int(threadgroup_position_in_grid.z);
    const int chunk = (N + SPLITS - 1) / SPLITS;
    const int start = split * chunk, end = min(N, start + chunk);
    const short qid = short(lane >> 2);
    const short fm = short((qid & 4) | ((lane >> 1) & 3));
    const short fn = short(((qid & 2) | (lane & 1)) * 4);
    constexpr auto qkd = matmul2d_descriptor(16, 32, 16, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto pvd = matmul2d_descriptor(16, 32, 16, false, false, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qkd, execution_simdgroup> qkop;
    matmul2d<pvd, execution_simdgroup> pvop;
    auto qa = qkop.template get_left_input_cooperative_tensor<T,T,float>();
    auto kb = qkop.template get_right_input_cooperative_tensor<T,T,float>();
    auto pa = pvop.template get_left_input_cooperative_tensor<half,T,float>();
    auto pl = pvop.template get_left_input_cooperative_tensor<half,T,float>();
    auto vb = pvop.template get_right_input_cooperative_tensor<half,T,float>();
    using PA = metal::remove_addrspace_t<decltype(pa)>;
    using VB = metal::remove_addrspace_t<decltype(vb)>;
    auto o0 = pvop.template get_destination_cooperative_tensor<PA,VB,float>();
    auto o1 = o0, o2 = o0, o3 = o0;
    for (short e=0;e<16;++e) { o0[e]=0; o1[e]=0; o2[e]=0; o3[e]=0; }
    float m[2]={-INFINITY,-INFINITY}, l[2]={0,0};
    threadgroup half ph[2][256], plo[2][256];
    for (int tile=start;tile<end;tile+=32) {
        auto sc = qkop.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(qa)>,metal::remove_addrspace_t<decltype(kb)>,float>();
        for(short e=0;e<16;++e) sc[e]=0;
        for(short kk=0;kk<128;kk+=16) {
            for(short i=0;i<2;++i) {
                const int f=fused0+fm+i*8, row=f/G, h=hk*G+f%G;
                for(short j=0;j<4;++j)
                    qa[i*4+j] = f<L*G ? q[((b*HQ+h)*L+row)*128+kk+fn+j] : T(0);
            }
            for(short tn=0;tn<2;++tn) for(short i=0;i<2;++i) {
                const int t=tile+tn*16+fm+i*8;
                for(short j=0;j<4;++j)
                    kb[tn*8+i*4+j]=t<end ? k[((b*HK+hk)*N+t)*128+kk+fn+j] : T(0);
            }
            qkop.run(qa,kb,sc);
        }
        float factor[2];
        for(short i=0;i<2;++i) {
            const int f=fused0+fm+i*8, row=f/G, h=hk*G+f%G;
            float mm=-INFINITY;
            bool ok[2][4];
            for(short tn=0;tn<2;++tn) for(short j=0;j<4;++j) {
                const int t=tile+tn*16+fn+j;
                bool live=f<L*G && t<end;
                if(CAUSAL) live=live && t<=N-L+row;
                if(MASKED && live) live=mask[((b*HQ+h)*L+row)*N+t];
                ok[tn][j]=live;
                if(live) mm=max(mm,float(sc[tn*8+i*4+j])*scale[0]);
            }
            mm=max(mm,simd_shuffle_xor(mm,ushort(1)));
            mm=max(mm,simd_shuffle_xor(mm,ushort(8)));
            const float mn=max(m[i],mm);
            factor[i]=m[i]==-INFINITY ? 0.0f : exp(m[i]-mn);
            float sum=0;
            for(short tn=0;tn<2;++tn) for(short j=0;j<4;++j) {
                const float p=ok[tn][j] ? exp(float(sc[tn*8+i*4+j])*scale[0]-mn) : 0;
                const half hi=half(p);
                ph[tn][lane*8+i*4+j]=hi;
                plo[tn][lane*8+i*4+j]=half(p-float(hi));
                sum+=p;
            }
            sum+=simd_shuffle_xor(sum,ushort(1));
            sum+=simd_shuffle_xor(sum,ushort(8));
            l[i]=l[i]*factor[i]+sum; m[i]=mn;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for(short e=0;e<16;++e) {
            const float f=factor[(e>>2)&1];
            o0[e]*=f; o1[e]*=f; o2[e]*=f; o3[e]*=f;
        }
        for(short ks=0;ks<2;++ks) {
            for(short e=0;e<8;++e) {pa[e]=ph[ks][lane*8+e];pl[e]=plo[ks][lane*8+e];}
            for(short np=0;np<4;++np) {
                for(short tn=0;tn<2;++tn) for(short i=0;i<2;++i) {
                    const int t=tile+ks*16+fm+i*8;
                    for(short j=0;j<4;++j) vb[tn*8+i*4+j]=t<end ?
                        v[((b*HK+hk)*N+t)*128+np*32+tn*16+fn+j] : T(0);
                }
                if(np==0) {pvop.run(pa,vb,o0);pvop.run(pl,vb,o0);}
                if(np==1) {pvop.run(pa,vb,o1);pvop.run(pl,vb,o1);}
                if(np==2) {pvop.run(pa,vb,o2);pvop.run(pl,vb,o2);}
                if(np==3) {pvop.run(pa,vb,o3);pvop.run(pl,vb,o3);}
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(short i=0;i<2;++i) {
        const int f=fused0+fm+i*8, row=f/G, h=hk*G+f%G;
        if(f>=L*G) continue;
        const ulong index=((ulong(split)*B+b)*HQ+h)*L+row;
        if((lane&9u)==0) {maxima[index]=m[i]; sums[index]=l[i];}
        for(short tn=0;tn<2;++tn) for(short j=0;j<4;++j) {
            const short e=tn*8+i*4+j;
            partial[index*128+tn*16+fn+j]=o0[e];
            partial[index*128+32+tn*16+fn+j]=o1[e];
            partial[index*128+64+tn*16+fn+j]=o2[e];
            partial[index*128+96+tn*16+fn+j]=o3[e];
        }
    }
"""


def supported(q, k, v, mask=None, sinks=None):
    if (q.ndim != 4 or k.ndim != 4 or v.shape != k.shape or q.shape[0] != k.shape[0]
            or min(q.shape) < 1 or min(k.shape) < 1
            or q.shape[-1] != 128 or k.shape[-1] != 128 or not 2 <= q.shape[2] <= 16
            or k.shape[2] < q.shape[2] or q.shape[1] % k.shape[1]
            or q.dtype not in (mx.float16, mx.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype):
        return False
    if isinstance(mask, str):
        if mask != "causal": return False
    elif mask is not None:
        if mask.dtype != mx.bool_: return False
        try:
            if mx.broadcast_shapes(mask.shape, (*q.shape[:3], k.shape[2])) != (*q.shape[:3], k.shape[2]): return False
        except ValueError: return False
    return sinks is None or (sinks.shape == (q.shape[1],) and sinks.dtype in (mx.float32,mx.float16,mx.bfloat16))


def use_kernel(q, k, v, mask=None, sinks=None):
    if os.environ.get("MLX2_FP_VERIFY_KERNEL") != "1" or not supported(q,k,v,mask,sinks):
        return False
    from ..gguf.source_dense_legacy import _device_supported
    return _device_supported()


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(name="mlx2_fp_verify_tensor", input_names=["q","k","v","mask","scale"],
        output_names=["partial","maxima","sums"], header=HEADER, source=SOURCE, ensure_row_contiguous=True)


def attention(q,k,v,*,scale,mask=None,sinks=None):
    if not supported(q,k,v,mask,sinks): raise ValueError("unsupported native FP verify geometry/mask")
    if isinstance(scale,bool) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('native FP verify scale must be finite and positive')
    from ..gguf.source_dense_legacy import _device_supported
    if not _device_supported(): raise ValueError("native FP verify requires caller-owned M5 GPU")
    b,h,l,d=q.shape; hk=k.shape[1]; n=k.shape[2]; g=h//hk
    splits=min(32,max(1,(n+2047)//2048))
    explicit=mask is not None and not isinstance(mask,str)
    visibility=mx.contiguous(mx.broadcast_to(mask,(b,h,l,n))) if explicit else mx.zeros((1,),mx.bool_)
    args=[("T",q.dtype),("B",b),("HQ",h),("HK",hk),("L",l),("N",n),("G",g),
          ("SPLITS",splits),("MASKED",explicit),("CAUSAL",isinstance(mask,str))]
    partial,m,s=_kernel()(inputs=[mx.contiguous(q),mx.contiguous(k),mx.contiguous(v),visibility,mx.array([scale],mx.float32)],
        template=args,grid=(((l*g+15)//16)*32,b*hk,splits),threadgroup=(32,1,1),
        output_shapes=[(splits,b,h,l,d),(splits,b,h,l),(splits,b,h,l)],output_dtypes=[mx.float32]*3)
    out=merge(partial,m,s,sinks=sinks)
    COUNTERS["calls"]+=1; COUNTERS["query_rows"]+=b*l
    return out.astype(q.dtype)


def merge(partial,m,s,*,sinks=None):
    """Stable split merge; fully masked sink-free rows emit exact zeros."""
    top=mx.max(m,axis=0)
    if sinks is not None: top=mx.maximum(top,sinks.astype(mx.float32)[None,:,None])
    finite=mx.isfinite(top)
    weights=mx.where(mx.isfinite(m)&finite[None],mx.exp(m-mx.where(finite,top,0)[None]),0)
    denominator=mx.sum(s*weights,axis=0)
    if sinks is not None: denominator+=mx.exp(sinks.astype(mx.float32)[None,:,None]-top)
    denominator=mx.where(denominator>0,denominator,1)
    return mx.sum(partial*weights[...,None],axis=0)/denominator[...,None]
