"""Opt-in fixed HC4 FP32 Metal primitives; ordinary graph remains reference.

Sinkhorn schedule adapted from pinned DwarfStar/DS4 (MIT); see provenance.
Single-thread coefficients preserve fixed row reductions independent of batch.
No automatic route selection and no GPU work on import.
"""
from functools import lru_cache
import math

SPLIT_SOURCE = r'''
const uint row = thread_position_in_grid.x;
if(row >= ROWS) return;
const float e=epsilon[0];
float c[16];
for(uint i=0;i<4;i++) {
  const float pre=mixes[row*24+i]*scale[0]+base[i];
  const float post=mixes[row*24+4+i]*scale[1]+base[4+i];
  out[row*24+i]=1.0f/(1.0f+exp(-pre))+e;
  out[row*24+4+i]=2.0f/(1.0f+exp(-post));
}
for(uint r=0;r<4;r++) {
  float maximum=-INFINITY;
  for(uint s=0;s<4;s++) { c[r*4+s]=mixes[row*24+8+r*4+s]*scale[2]+base[8+r*4+s]; maximum=max(maximum,c[r*4+s]); }
  float sum=0.0f;
  for(uint s=0;s<4;s++) { c[r*4+s]=exp(c[r*4+s]-maximum); sum+=c[r*4+s]; }
  for(uint s=0;s<4;s++) c[r*4+s]=c[r*4+s]/sum+e;
}
for(uint s=0;s<4;s++) {
  float sum=0.0f;
  for(uint r=0;r<4;r++) sum+=c[r*4+s];
  for(uint r=0;r<4;r++) c[r*4+s]/=(sum+e);
}
for(uint it=1;it<ITERS;it++) {
  for(uint r=0;r<4;r++) {
    float sum=0.0f;
    for(uint s=0;s<4;s++) sum+=c[r*4+s];
    for(uint s=0;s<4;s++) c[r*4+s]/=(sum+e);
  }
  for(uint s=0;s<4;s++) {
    float sum=0.0f;
    for(uint r=0;r<4;r++) sum+=c[r*4+s];
    for(uint r=0;r<4;r++) c[r*4+s]/=(sum+e);
  }
}
for(uint i=0;i<16;i++) out[row*24+8+i]=c[i];
'''
SUM_SOURCE = r'''
const uint index=thread_position_in_grid.x;
if(index>=ROWS*DIM) return;
const uint row=index/DIM, d=index%DIM;
float sum=0.0f;
for(uint h=0;h<4;h++) sum+=value[(row*4+h)*DIM+d]*weights[row*4+h];
out[index]=sum;
'''
EXPAND_SOURCE = r'''
const uint index=thread_position_in_grid.x;
if(index>=ROWS*4*DIM) return;
const uint row=index/(4*DIM), h=(index/DIM)%4, d=index%DIM;
float sum=branch[row*DIM+d]*post[row*4+h];
for(uint s=0;s<4;s++) sum+=residual[(row*4+s)*DIM+d]*comb[row*16+s*4+h];
out[index]=sum;
'''
HEAD_SOURCE = r'''
const uint index=thread_position_in_grid.x;
if(index>=ROWS*DIM) return;
const uint row=index/DIM,d=index%DIM;
float sum=0.0f;
for(uint h=0;h<4;h++) {
  const float z=mixes[row*4+h]*scale[0]+base[h];
  const float w=1.0f/(1.0f+exp(-z))+epsilon[0];
  sum+=value[(row*4+h)*DIM+d]*w;
}
out[index]=sum;
'''
SOURCES = {
    'split': (SPLIT_SOURCE, ['mixes','scale','base','epsilon']),
    'sum': (SUM_SOURCE, ['value','weights']),
    'expand': (EXPAND_SOURCE, ['branch','residual','post','comb']),
    'head': (HEAD_SOURCE, ['value','mixes','scale','base','epsilon']),
}


_SCHEDULED = {kind: 0 for kind in SOURCES}


def route_status():
    """Lazy graph scheduling counts; evaluation is owned by the caller."""
    return {"policy": "fused_fp32", "scheduled_calls": dict(_SCHEDULED),
            "scheduled_total": sum(_SCHEDULED.values()),
            "observation": "scheduled only; counters do not establish GPU completion"}


@lru_cache(maxsize=4)
def _kernel(kind):
    import mlx.core as mx
    source, names = SOURCES[kind]
    kernel = mx.fast.metal_kernel(name='mlx2_deepseek_v4_hc_'+kind+'_v1',
        input_names=names, output_names=['out'], source=source,
        ensure_row_contiguous=True)
    def schedule(**kwargs):
        result = kernel(**kwargs)
        _SCHEDULED[kind] += 1
        return result
    return schedule


def _mx():
    import mlx.core as mx
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise ValueError('fused HC requires an owned Metal GPU execution window')
    return mx


def _epsilon(value):
    if not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('HC epsilon must be positive finite')


def split_sinkhorn(mixes, scale, base, *, iterations=20, eps=1e-6):
    _epsilon(eps)
    if mixes.shape[-1] != 24 or scale.shape != (3,) or base.shape != (24,) or type(iterations) is not int or not 1<=iterations<=100:
        raise ValueError('fused Sinkhorn requires HC4 coefficient geometry')
    mx=_mx(); rows=math.prod(mixes.shape[:-1])
    if not rows: raise ValueError('fused HC requires nonempty rows')
    out=_kernel('split')(inputs=[mixes.astype(mx.float32),scale.astype(mx.float32),base.astype(mx.float32),mx.array([eps],mx.float32)],
        template=[('ROWS',rows),('ITERS',iterations)], grid=(rows,1,1),threadgroup=(min(32,rows),1,1),
        output_shapes=[mixes.shape],output_dtypes=[mx.float32])[0]
    return out[...,:4],out[...,4:8],out[...,8:].reshape(*mixes.shape[:-1],4,4)


def weighted_sum(value, weights):
    if value.shape[-2] != 4 or weights.shape != value.shape[:-1]:
        raise ValueError('fused HC weighted sum geometry mismatch')
    mx=_mx(); rows=math.prod(value.shape[:-2]); dim=value.shape[-1]
    return _kernel('sum')(inputs=[value.astype(mx.float32),weights.astype(mx.float32)],
        template=[('ROWS',rows),('DIM',dim)], grid=(rows*dim,1,1),threadgroup=(256,1,1),
        output_shapes=[(*value.shape[:-2],dim)],output_dtypes=[mx.float32])[0]


def expand(branch, residual, post, comb):
    shape=residual.shape
    if shape[-2]!=4 or branch.shape!=(*shape[:-2],shape[-1]) or post.shape!=shape[:-1] or comb.shape!=(*shape[:-2],4,4):
        raise ValueError('fused HC expansion geometry mismatch')
    mx=_mx(); rows=math.prod(shape[:-2]); dim=shape[-1]
    return _kernel('expand')(inputs=[x.astype(mx.float32) for x in (branch,residual,post,comb)],
        template=[('ROWS',rows),('DIM',dim)], grid=(rows*4*dim,1,1),threadgroup=(256,1,1),
        output_shapes=[shape],output_dtypes=[mx.float32])[0]


def head_reduce(value, mixes, scale, base, *, eps=1e-6):
    _epsilon(eps)
    if value.shape[-2]!=4 or mixes.shape!=value.shape[:-1] or scale.shape!=(1,) or base.shape!=(4,):
        raise ValueError('fused HC head geometry mismatch')
    mx=_mx(); rows=math.prod(value.shape[:-2]); dim=value.shape[-1]
    return _kernel('head')(inputs=[value.astype(mx.float32),mixes.astype(mx.float32),scale.astype(mx.float32),base.astype(mx.float32),mx.array([eps],mx.float32)],
        template=[('ROWS',rows),('DIM',dim)], grid=(rows*dim,1,1),threadgroup=(256,1,1),
        output_shapes=[(*value.shape[:-2],dim)],output_dtypes=[mx.float32])[0]


def reference_split(mixes, scale, base, *, iterations=20, eps=1e-6):
    """Independent high precision CPU equation oracle, not an execution route."""
    import numpy as np
    z=np.asarray(mixes,dtype=np.float64)
    s=np.asarray(scale,dtype=np.float64); b=np.asarray(base,dtype=np.float64)
    sigmoid=lambda x: np.exp(-np.logaddexp(0,-x))
    pre=sigmoid(z[...,:4]*s[0]+b[:4])+eps
    post=2*sigmoid(z[...,4:8]*s[1]+b[4:8])
    c=(z[...,8:]*s[2]+b[8:]).reshape(*z.shape[:-1],4,4)
    c=np.exp(c-c.max(axis=-1,keepdims=True));c=c/c.sum(axis=-1,keepdims=True)+eps
    c/=c.sum(axis=-2,keepdims=True)+eps
    for _ in range(iterations-1):
        c/=c.sum(axis=-1,keepdims=True)+eps
        c/=c.sum(axis=-2,keepdims=True)+eps
    return pre,post,c


def compilation_source():
    """Standalone representative specializations for CPU-side Metal compiler."""
    text='#include <metal_stdlib>\nusing namespace metal;\n'
    for kind,(source,names) in SOURCES.items():
        args=[f'device const float *{name} [[buffer({i})]]' for i,name in enumerate(names)]
        args += [f'device float *out [[buffer({len(names)})]]','uint3 thread_position_in_grid [[thread_position_in_grid]]']
        text+='kernel void check_'+kind+'('+','.join(args)+'){\nconstexpr uint ROWS=4,DIM=64,ITERS=20;\n'+source+'\n}\n'
    return text
