# SPDX-License-Identifier: MIT
"""Exact pointwise boundaries; retain stock reductions, softplus and top-k."""

from functools import lru_cache

HEADER = r"""
#pragma clang fp contract(off)
inline float bf16_value(float x) {
    uint bits=as_type<uint>(x);
    return as_type<float>((bits+0x7fffu+((bits>>16)&1u))&0xffff0000u);
}
"""


@lru_cache(maxsize=4)
def kernel(kind):
    import mlx.core as mx

    names, outs, source = {
        "round": (["x"],["y"],"uint i=thread_position_in_grid.x;if(i<SIZE)y[i]=bf16_value(x[i]);"),
        "rotary": (["x","cosines","sines"],["y"],r"""
uint i=thread_position_in_grid.x;if(i>=SIZE)return;
uint k=i%DIM, pos=(i/DIM)%LENGTH;
float result=x[i];
if(k>=DIM-ROPE) {
    uint j=k-(DIM-ROPE), pair=j/2, base=i-(j%2);
    float a=x[base],b=x[base+1],c=cosines[pos*(ROPE/2)+pair],s=sines[pos*(ROPE/2)+pair];
    result=(j%2)==0 ? a*c-b*s : a*s+b*c;
}
y[i]=bf16_value(result);
"""),
        "router_scores": (["softplus","correction"],["scores","rank"],r"""
uint i=thread_position_in_grid.x;if(i>=SIZE)return;
float score=metal::precise::sqrt(softplus[i]);scores[i]=score;rank[i]=-(score+correction[i%EXPERTS]);
"""),
        "router_weights": (["weights","sums"],["out"],r"""
uint i=thread_position_in_grid.x;if(i>=SIZE)return;
float value=weights[i];if(NORMALIZE)value=value/(sums[i/TOPK]+1e-20f);
out[i]=value*as_type<float>(uint(SCALE_BITS));
""")
    }[kind]
    return mx.fast.metal_kernel(name=f"mlx2_v41_pointwise_{kind}_v1",input_names=names,output_names=outs,
                               header=HEADER,source=source,ensure_row_contiguous=True)


def rounded(x):
    import mlx.core as mx

    return kernel("round")(inputs=[x.astype(mx.float32)],template=[("SIZE",x.size)],
                           grid=(((x.size+255)//256)*256,1,1),threadgroup=(256,1,1),
                           output_shapes=[x.shape],output_dtypes=[mx.float32])[0]


def _checked_phase(x, rope, phase):
    if (not isinstance(phase, tuple) or len(phase) != 2
            or any(value.shape != (x.shape[-2], rope.dims // 2)
                   or value.dtype != x.dtype for value in phase)):
        raise ValueError("V4.1 precomputed rotary phase geometry/dtype mismatch")
    return phase


def rotate(x, rope, offset=0, *, positions=None, inverse=False, phase=None):
    import mlx.core as mx

    if phase is None:
        pos=mx.arange(offset,offset+x.shape[-2]) if positions is None else mx.array(positions)
        angles=pos.astype(mx.float32)[:,None]*rope.inv[None]
        if inverse:angles=-angles
        cosines,sines=mx.cos(angles).astype(x.dtype),mx.sin(angles).astype(x.dtype)
    else:
        cosines,sines=_checked_phase(x,rope,phase)
    return kernel("rotary")(inputs=[x,cosines,sines],
                            template=[("SIZE",x.size),("DIM",x.shape[-1]),("LENGTH",x.shape[-2]),("ROPE",rope.dims)],
                            grid=(((x.size+255)//256)*256,1,1),threadgroup=(256,1,1),
                            output_shapes=[x.shape],output_dtypes=[mx.float32])[0]


def router(logits, *, top_k, scale, correction, normalize=True):
    import mlx.core as mx
    import mlx.nn as nn
    import struct

    scores,rank=kernel("router_scores")(
        inputs=[nn.softplus(logits.astype(mx.float32)),correction],
        template=[("SIZE",logits.size),("EXPERTS",logits.shape[-1])],
        grid=(((logits.size+255)//256)*256,1,1),threadgroup=(256,1,1),
        output_shapes=[logits.shape,logits.shape],output_dtypes=[mx.float32,mx.float32])
    indices=mx.argpartition(rank,kth=top_k-1,axis=-1)[...,:top_k]
    weights=mx.take_along_axis(scores,indices,axis=-1)
    sums=mx.sum(weights,axis=-1,keepdims=True)
    out=kernel("router_weights")(inputs=[weights,sums],
        template=[("SIZE",weights.size),("TOPK",top_k),("SCALE_BITS",struct.unpack("I",struct.pack("f",scale))[0]),("NORMALIZE",normalize)],
        grid=(((weights.size+255)//256)*256,1,1),threadgroup=(256,1,1),
        output_shapes=[weights.shape],output_dtypes=[mx.float32])[0]
    return indices.astype(mx.uint32),out


@lru_cache(maxsize=1)
def rotate_qat_kernel():
    import mlx.core as mx
    from .deepseek_v41_quant import FUSED_HEADER,FUSED_SOURCE

    expression=r"""
uint coordinate=i%DIM, pos=(i/DIM)%LENGTH;
float rotated=value[i];
if(coordinate>=DIM-ROPE) {
    uint j=coordinate-(DIM-ROPE), pair=j/2, base=i-(j%2);
    float a=value[base],b=value[base+1],c=cosines[pos*(ROPE/2)+pair],s=sines[pos*(ROPE/2)+pair];
    rotated=(j%2)==0 ? a*c-b*s : a*s+b*c;
}
const float v=bf16_value(rotated);
"""
    source=FUSED_SOURCE.replace("const float v=value[i];",expression).replace("out[i]=quantized*scale;","out[i]=bf16_value(quantized*scale);")
    return mx.fast.metal_kernel(name="mlx2_v41_rotary_qat_bf16_v1",input_names=["value","inverse","cosines","sines"],
                               output_names=["out"],header=FUSED_HEADER+HEADER,source=source,ensure_row_contiguous=True)


def rotate_qat(x,rope,kind,offset=0,*,positions=None,inverse=False,phase=None):
    import mlx.core as mx
    from .deepseek_v41_quant import _FUSED_KINDS

    if phase is None:
        pos=mx.arange(offset,offset+x.shape[-2]) if positions is None else mx.array(positions)
        angles=pos.astype(mx.float32)[:,None]*rope.inv[None]
        if inverse:angles=-angles
        phase=mx.cos(angles).astype(x.dtype),mx.sin(angles).astype(x.dtype)
    else:
        phase=_checked_phase(x,rope,phase)
    code=_FUSED_KINDS[kind]
    return rotate_qat_kernel()(inputs=[x,mx.array([1/(448 if code==0 else 6)],mx.float32),*phase],
        template=[("ELEMENTS",x.size),("BLOCK",16 if code==2 else 32),("KIND",code),
                  ("DIM",x.shape[-1]),("LENGTH",x.shape[-2]),("ROPE",rope.dims)],
        grid=(x.size,1,1),threadgroup=(128,1,1),output_shapes=[x.shape],output_dtypes=[mx.float32])[0]


@lru_cache(maxsize=2)
def expand_kernel(joined):
    import mlx.core as mx
    from .deepseek_v4_hc import EXPAND_SOURCE

    source=EXPAND_SOURCE.replace("out[index]=sum;","out[index]=bf16_value(sum);")
    names=["branch","residual","post","comb"]
    if joined:
        names.append("shared")
        source=source.replace("branch[row*DIM+d]*post[row*4+h]","bf16_value(branch[row*DIM+d]+shared[row*DIM+d])*post[row*4+h]")
    # The existing HC kernel allows contraction. Preserve that exact kernel
    # law; only the mandatory BF16 store and optional rounded branch add move.
    return mx.fast.metal_kernel(name=f"mlx2_v41_hc_store_{int(joined)}_v1",input_names=names,output_names=["out"],
                               header=HEADER.replace("#pragma clang fp contract(off)",""),source=source,ensure_row_contiguous=True)


def expand(branch,residual,post,comb,shared=None):
    import mlx.core as mx

    values=[branch,residual,post,comb]+([] if shared is None else [shared])
    return expand_kernel(shared is not None)(inputs=values,template=[("ROWS",branch.size//branch.shape[-1]),("DIM",branch.shape[-1])],
        grid=(residual.size,1,1),threadgroup=(256,1,1),output_shapes=[residual.shape],output_dtypes=[mx.float32])[0]


@lru_cache(maxsize=1)
def epilogue_check():
    import mlx.core as mx
    import numpy as np
    from .deepseek_v4 import Rotary
    from .deepseek_v4_hc import expand as reference_expand
    from .deepseek_v41_quant import fused_quantize

    rng=np.random.default_rng(411108)
    for rows in (1,6,10):
        for kind,dim in (("fp8",512),("index_fp4",128),("compressed_fp4",512)):
            x=mx.array(rng.normal(size=(1,2,rows,dim)).astype(np.float32));rope=Rotary(64,10000)
            for inverse in (False,True):
                expected=fused_quantize(rope(x,127,inverse=inverse).astype(mx.bfloat16).astype(mx.float32),kind).astype(mx.bfloat16).astype(mx.float32)
                actual=rotate_qat(x,rope,kind,127,inverse=inverse);mx.eval(expected,actual)
                if np.array(expected).tobytes()!=np.array(actual).tobytes():raise ValueError(f"V4.1 fused rotary/QAT gate failed {kind}")
        h=mx.array(rng.normal(size=(1,rows,4,512)).astype(np.float32))
        b=mx.array(rng.normal(size=(1,rows,512)).astype(np.float32));s=b*.5
        post=mx.array(rng.normal(size=(1,rows,4)).astype(np.float32));comb=mx.array(rng.normal(size=(1,rows,4,4)).astype(np.float32))
        for shared in (None,s):
            branch=b if shared is None else (b+shared).astype(mx.bfloat16).astype(mx.float32)
            expected=reference_expand(branch,h,post,comb).astype(mx.bfloat16).astype(mx.float32)
            actual=expand(b,h,post,comb,shared);mx.eval(expected,actual)
            if np.array(expected).tobytes()!=np.array(actual).tobytes():raise ValueError("V4.1 HC store gate failed")


@lru_cache(maxsize=1)
def self_check():
    import mlx.core as mx
    import numpy as np
    from .deepseek_v4 import Rotary, route_experts

    rng=np.random.default_rng(41107)
    values=rng.normal(size=(1,4,6,128)).astype(np.float32)
    values.ravel()[:4]=[0.,-0.,2**-125,-2**-125]
    x=mx.array(values);rope=Rotary(64,10000)
    pairs=[(rounded(x),x.astype(mx.bfloat16).astype(mx.float32))]
    for inverse in (False,True):
        pairs.append((rotate(x,rope,127,inverse=inverse),rope(x,127,inverse=inverse).astype(mx.bfloat16).astype(mx.float32)))
    for rows in (1,6,10):
        logits=mx.array(rng.normal(size=(rows,384)).astype(np.float32));correction=mx.array(rng.normal(size=384).astype(np.float32))
        for normalize in (False,True):
            kw=dict(top_k=6,scale=1.5,correction=correction,normalize=normalize)
            pairs.extend(zip(router(logits,**kw),route_experts(logits,**kw)))
    for index,(actual,expected) in enumerate(pairs):
        mx.eval(actual,expected)
        if np.array(actual).tobytes()!=np.array(expected).tobytes():raise ValueError(f"V4.1 pointwise exact gate failed at pair {index}")
