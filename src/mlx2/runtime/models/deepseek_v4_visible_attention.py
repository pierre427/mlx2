# SPDX-License-Identifier: Apache-2.0
"""Opt-in visible-row FP32 DeepSeek V4 attention; ordinary reference stays default.

Mined mechanism: provenance/deepseek-v4-visible-attention.json. No NxN scores,
no full pooled mask or gathered [heads,queries,pools,dim] activation. Each
SIMDgroup computes one independent query/head with fixed chronological row
order. A synthetic first-use self-check is not model qualification.
"""
from __future__ import annotations

import threading
from functools import lru_cache

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"
SOURCE = r'''
    const uint query_row = threadgroup_position_in_grid.y;
    const uint tid = thread_index_in_threadgroup;
    const uint head = threadgroup_position_in_grid.x * 4 + tid / 32;
    const uint lane = tid % 32;
    if (head >= HEADS) return;
    const int offset=params[0], window=params[1], ratio=params[2];
    const int P=params[3], S=params[4], L=params[5], K=params[6];
    const int absolute=offset+int(query_row);
    const int base=offset+L-S;
    const int first=max(base,absolute-window+1)-base;
    const int last=absolute-base;
    const int visible=ratio==0 ? 0 : min((absolute+1)/ratio,P);
    const float scale=0.04419417382415922f; // 1/sqrt512
    device const float4 *query=(device const float4 *)(q+(ulong(head)*L+query_row)*512);
    const float4 q0=query[lane],q1=query[lane+32],q2=query[lane+64],q3=query[lane+96];
    float maximum=-INFINITY, denominator=0.0f;
    float4 a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;
    const int local_count=last-first+1;
    const int pool_count=K==0 ? visible : K;
    for(int row=0;row<local_count+pool_count;row++) {
        const bool compressed=row>=local_count;
        uint index;
        if (!compressed) index=uint(first+row);
        else {
            const uint pindex=uint(row-local_count);
            index=K==0 ? pindex : topk[ulong(query_row)*K+pindex];
            if(index>=uint(P) || (K>0 && pindex>0 && topk[ulong(query_row)*K+pindex-1]==index)) {
                // Invalid indices are a request/kernel contract failure, never
                // silently substituted with another compressed vector.
                device float4 *output=(device float4 *)(out+(ulong(head)*L+query_row)*512);
                output[lane]=float4(NAN); output[lane+32]=float4(NAN);
                output[lane+64]=float4(NAN); output[lane+96]=float4(NAN);
                return;
            }
            if(index>=uint(visible)) continue;
        }
        device const float4 *values=(device const float4 *)(compressed ? pooled : kv)+ulong(index)*128;
        const float4 v0=values[lane],v1=values[lane+32],v2=values[lane+64],v3=values[lane+96];
        float score=simd_sum(dot(q0,v0)+dot(q1,v1)+dot(q2,v2)+dot(q3,v3))*scale;
        const float next=max(maximum,score);
        const float rescale=maximum==-INFINITY ? 0.0f : exp(maximum-next);
        const float weight=exp(score-next);
        denominator=denominator*rescale+weight;
        a0=a0*rescale+weight*v0; a1=a1*rescale+weight*v1;
        a2=a2*rescale+weight*v2; a3=a3*rescale+weight*v3;
        maximum=next;
    }
    // Merge the sink stably: it contributes denominator mass and zero value.
    const float next=max(maximum,sinks[head]);
    const float rescale=maximum==-INFINITY ? 0.0f : exp(maximum-next);
    denominator=denominator*rescale+exp(sinks[head]-next);
    const float multiplier=rescale/denominator;
    device float4 *output=(device float4 *)(out+(ulong(head)*L+query_row)*512);
    output[lane]=a0*multiplier; output[lane+32]=a1*multiplier;
    output[lane+64]=a2*multiplier; output[lane+96]=a3*multiplier;
'''

_LOCK=threading.RLock()
_READY=False
_FAILURE=None
_SCHEDULED_CALLS=0


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx
    return mx.fast.metal_kernel(name="mlx2_deepseek_v4_visible_fp32_v1",
        input_names=["q","kv","pooled","topk","sinks","params"],
        output_names=["out"],header=HEADER,source=SOURCE,
        ensure_row_contiguous=True)


def validate_geometry(q, local, pooled, indices, sinks, offset, window, ratio):
    if type(offset) is not int or offset<0 or type(window) is not int or window<1 or type(ratio) is not int or ratio not in (0,4,128):
        raise ValueError("invalid visible-attention position/window/compression")
    if q.ndim!=4 or q.shape[0]!=1 or q.shape[-1]!=512 or q.shape[1]<1 or q.shape[2]<1:
        raise ValueError("visible attention requires nonempty B1/H/L/D512")
    if local.ndim!=4 or local.shape[:2]!=(1,1) or local.shape[-1]!=512:
        raise ValueError("visible attention requires one shared local KV tensor")
    length=q.shape[2]; rows=local.shape[2]
    if window > 2**31 - 1 or offset + length > 2**31 - 1:
        raise ValueError("attention logical positions exceed int32 kernel contract")
    if rows<length or rows>offset+length or offset+length-rows>max(0,offset-window+1):
        raise ValueError("local band does not contain every query's causal window")
    if sinks.shape!=(q.shape[1],):
        raise ValueError("visible attention sink geometry mismatch")
    if pooled is not None and (pooled.ndim!=3 or pooled.shape[0]!=1 or pooled.shape[-1]!=512):
        raise ValueError("visible attention pooled geometry mismatch")
    count=0 if pooled is None else pooled.shape[1]
    if count and ratio==0:
        raise ValueError("local-only layer cannot attend compressed rows")
    if ratio and count!=(offset+length)//ratio:
        raise ValueError("compressed pool count differs from completed windows")
    if indices is not None and (ratio!=4 or not count or indices.ndim!=3
            or indices.shape[:2]!=(1,length) or not 1<=indices.shape[2]<=min(512,count)):
        raise ValueError("invalid ratio4 indexer top-k geometry")


def reference_visible_attention(q, local, pooled, indices, sinks, *, offset, window, ratio):
    """Independent NumPy float64 dense-softmax oracle, never an inference backend."""
    import numpy as np
    validate_geometry(q,local,pooled,indices,sinks,offset,window,ratio)
    if indices is not None:
        if not np.issubdtype(indices.dtype,np.integer) or np.any(indices<0) or np.any(indices>=pooled.shape[1]):
            raise ValueError("invalid pooled row index")
        if any(len(np.unique(row))!=len(row) for row in indices[0]):
            raise ValueError("indexer rows must contain unique pool indices")
    result=np.zeros_like(q,dtype=np.float64)
    base=offset+q.shape[2]-local.shape[2]
    for t in range(q.shape[2]):
        absolute=offset+t
        rows=local[0,0,max(base,absolute-window+1)-base:absolute-base+1]
        visible=0 if not ratio else min((absolute+1)//ratio,0 if pooled is None else pooled.shape[1])
        if pooled is not None and visible:
            chosen=np.arange(visible) if indices is None else np.sort(indices[0,t])
            if np.any(chosen<0) or np.any(chosen>=pooled.shape[1]):
                raise ValueError("invalid pooled row index")
            chosen=chosen[chosen<visible]
            rows=np.concatenate((rows,pooled[0,chosen]),axis=0)
        for head in range(q.shape[1]):
            scores=q[0,head,t].astype(np.float64)@rows.astype(np.float64).T/np.sqrt(512)
            maximum=max(float(np.max(scores)),float(sinks[head]))
            weights=np.exp(scores-maximum)
            result[0,head,t]=weights@rows/(weights.sum()+np.exp(sinks[head]-maximum))
    return result.astype(np.float32)


def _dispatch(q,local,pooled,indices,sinks,*,offset,window,ratio):
    import mlx.core as mx
    length=q.shape[2]; heads=q.shape[1]; count=0 if pooled is None else pooled.shape[1]
    if pooled is None:
        pooled=mx.zeros((1,1,512),dtype=mx.float32)
    width=0 if indices is None else indices.shape[-1]
    if indices is None:
        indices=mx.zeros((1,1,1),dtype=mx.uint32)
    else:
        indices=mx.sort(indices.astype(mx.uint32),axis=-1)
    params=mx.array([offset,window,ratio,count,local.shape[2],length,width],dtype=mx.int32)
    output=_kernel()(inputs=[q[0],local[0,0],pooled[0],indices[0],sinks,params],
        template=[("HEADS",heads)], grid=(((heads+3)//4)*128,length,1),
        threadgroup=(128,1,1), output_shapes=[(heads,length,512)],output_dtypes=[mx.float32])[0]
    return output[None]


def self_check():
    """Run once under the caller's GPU ownership, before target request mutation."""
    global _READY,_FAILURE
    import mlx.core as mx
    import numpy as np
    if mx.default_device()!=mx.gpu or not mx.metal.is_available():
        raise RuntimeError("visible-attention candidate requires owned Metal GPU execution")
    with _LOCK:
        if _FAILURE is not None:
            raise RuntimeError("visible-attention candidate self-check previously failed") from _FAILURE
        if _READY:
            return
        try:
            rng=np.random.default_rng(260424)
            q=rng.normal(0,.2,(1,64,3,512)).astype(np.float32)
            local=rng.normal(0,.2,(1,1,6,512)).astype(np.float32)
            pooled=rng.normal(0,.2,(1,4,512)).astype(np.float32)
            sinks=rng.normal(0,.3,(64,)).astype(np.float32)
            indices=np.array([[[3,0,2],[1,3,0],[2,1,3]]],np.uint32)
            for ratio,offset,pool,top in [(0,9,None,None),(4,13,pooled,indices),(128,509,pooled,None)]:
                # count must match completed-window geometry in every probe.
                if ratio==128:
                    pool=pool[:,:4]
                expected=reference_visible_attention(q,local,pool,top,sinks,offset=offset,window=4,ratio=ratio)
                out=_dispatch(mx.array(q),mx.array(local),None if pool is None else mx.array(pool),
                              None if top is None else mx.array(top),mx.array(sinks),offset=offset,window=4,ratio=ratio)
                mx.eval(out)
                np.testing.assert_allclose(np.array(out),expected,rtol=3e-5,atol=3e-6)
            _READY=True
        except Exception as error:
            _FAILURE=error
            raise RuntimeError("visible-attention candidate failed synthetic numerical self-check") from error


def visible_attention(q,local,pooled,indices,sinks,*,offset,window,ratio):
    global _SCHEDULED_CALLS
    import mlx.core as mx
    validate_geometry(q,local,pooled,indices,sinks,offset,window,ratio)
    if any(x is not None and x.dtype!=mx.float32 for x in (q,local,pooled,sinks)):
        raise ValueError("visible-attention profile requires physical FP32 inputs")
    if indices is not None and indices.dtype not in (mx.int32,mx.uint32):
        raise ValueError("visible-attention indexer IDs must be 32-bit integers")
    self_check()
    out=_dispatch(q,local,pooled,indices,sinks,offset=offset,window=window,ratio=ratio)
    _SCHEDULED_CALLS+=1
    return out


def route_status():
    return {"policy":"visible_rows_fp32","qualified":False,
            "synthetic_self_check_passed":_READY,"failed":_FAILURE is not None,
            "scheduled_target_calls":_SCHEDULED_CALLS,
            "arithmetic":"FP32-SIMDgroup-fixed-chronological-online-softmax"}


def metal_compile_source():
    """Standalone CPU toolchain wrapper, no Metal device dispatch."""
    return HEADER+"\nconstant uint HEADS=64;\nkernel void cpu_compile_probe("+ \
        "device const float* q [[buffer(0)]],device const float* kv [[buffer(1)]],"+ \
        "device const float* pooled [[buffer(2)]],device const uint* topk [[buffer(3)]],"+ \
        "device const float* sinks [[buffer(4)]],device const int* params [[buffer(5)]],"+ \
        "device float* out [[buffer(6)]],uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],"+ \
        "uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]) {\n"+SOURCE+"\n}\n"
