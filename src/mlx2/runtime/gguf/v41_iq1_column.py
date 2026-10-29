# SPDX-License-Identifier: Apache-2.0
"""Experimental F32 column GEMV over exact IQ1_S T64 expert banks.

Adjacent SIMD lanes own adjacent output rows. A lane decodes each32-value
group to F32 registers, prefetches the next payload, then accumulates its
products in element order before adding that group to the output sum. This
changes accumulation order from row-exact and has a distinct numeric profile.
No provider selects it. See deepseek-v41-iq1-column-kernels provenance.
"""

from functools import lru_cache

from . import v41_iq1_tiled as tiled
from .reader import GGUFError

REVISION = "mlx2.v41-iq1-t64-column-f32-group32-runtime-capacity.v2"
NUMERIC_PROFILE = "native_iq1_s_tiled_column_f32"
missing_routes = tiled.missing_routes

HEADER = tiled.HEADER+r"""
#pragma clang fp reassociate(off)
template<typename QS,typename QH>
inline uint2 v41_column_payload(QS qs,QH qh,ulong group) {
    const ulong low=group*4,high=group*2;
    return uint2(uint(qs[low])|(uint(qs[low+1])<<8)|(uint(qs[low+2])<<16)|(uint(qs[low+3])<<24),
                 uint(qh[high])|(uint(qh[high+1])<<8));
}
template<typename D>
inline float v41_column_scale(D d,ulong block) {
    const ushort bits=ushort(d[block*2])|(ushort(d[block*2+1])<<8);
    return float(as_type<half>(bits));
}
inline void v41_column_decode(uint2 payload,float d,thread float4 (&v)[8]) {
    const uint high=payload.y;
    const float scale=d*float(2*((high>>12)&7)+1);
    const float delta=(high&0x8000) ? -0.125f : 0.125f;
#pragma clang loop unroll(full)
    for(uint j=0;j<4;j++) {
        const uint index=((payload.x>>(8*j))&255)|(((high>>(3*j))&7)<<8);
        const uint code=v41_iq1_nibbles[index];
        const int4 lo=int4(int(code&15),int((code>>8)&15),int((code>>16)&15),int((code>>24)&15))-1;
        const int4 hi=int4(int((code>>4)&15),int((code>>12)&15),int((code>>20)&15),int((code>>28)&15))-1;
        v[2*j]=scale*(float4(lo)+delta);
        v[2*j+1]=scale*(float4(hi)+delta);
    }
}
"""

PREFIX = r"""
const uint row=thread_position_in_grid.x,first=threadgroup_position_in_grid.y;
if(row>=N) return;
const uint expert=ids[first];
if(expert>=EXPERTS) { y[ulong(first)*N+row]=0.0f; return; }
const uint slot=slot_map[expert];
if(slot>=capacity) { y[ulong(first)*N+row]=0.0f; return; }
const uint groups=K/32;
const ulong group_base=ulong(slot)*N*groups+ulong(row/64)*groups*64+row%64;
const ulong block_base=ulong(slot)*N*(K/256)+ulong(row/64)*(K/256)*64+row%64;
"""


def kernel_spec(kind):
    if kind not in ("gate", "down"):
        raise GGUFError("unknown IQ1 column projection")
    names = ["x", "ids", "slot_map"]
    if kind == "gate":
        names += ["scores", "gqs", "gqh", "gd", "uqs", "uqh", "ud"]
        source = PREFIX+r"""
const uint token=first/TOPK;
uint2 gp=v41_column_payload(gqs,gqh,group_base),up=v41_column_payload(uqs,uqh,group_base);
float gs=v41_column_scale(gd,block_base),us=v41_column_scale(ud,block_base);
float gsum=0.0f,usum=0.0f;
for(uint group=0;group<groups;group++) {
    float4 gv[8],uv[8];
    v41_column_decode(gp,gs,gv);
    v41_column_decode(up,us,uv);
    if(group+1<groups) {
        gp=v41_column_payload(gqs,gqh,group_base+ulong(group+1)*64);
        up=v41_column_payload(uqs,uqh,group_base+ulong(group+1)*64);
        if((group+1)%8==0) {
            gs=v41_column_scale(gd,block_base+ulong((group+1)/8)*64);
            us=v41_column_scale(ud,block_base+ulong((group+1)/8)*64);
        }
    }
    float sg=0.0f,su=0.0f;
#pragma clang loop unroll(full)
    for(uint i=0;i<8;i++) {
        const ulong at=ulong(token)*K+group*32+i*4;
        const float4 a=float4(x[at],x[at+1],x[at+2],x[at+3]);
        const float4 g=gv[i],u=uv[i];
        sg=sg+g.x*a.x; su=su+u.x*a.x;
        sg=sg+g.y*a.y; su=su+u.y*a.y;
        sg=sg+g.z*a.z; su=su+u.z*a.z;
        sg=sg+g.w*a.w; su=su+u.w*a.w;
    }
    gsum=gsum+sg; usum=usum+su;
}
gsum=min(v41_bf16(gsum),10.0f); usum=clamp(v41_bf16(usum),-10.0f,10.0f);
y[ulong(first)*N+row]=v41_bf16((gsum/(1.0f+exp(-gsum)))*usum*scores[first]);
"""
    else:
        names += ["wqs", "wqh", "wd"]
        source = PREFIX+r"""
uint2 payload=v41_column_payload(wqs,wqh,group_base);
float scale=v41_column_scale(wd,block_base);
float total=0.0f;
for(uint group=0;group<groups;group++) {
    float4 weights[8];
    v41_column_decode(payload,scale,weights);
    if(group+1<groups) {
        payload=v41_column_payload(wqs,wqh,group_base+ulong(group+1)*64);
        if((group+1)%8==0) scale=v41_column_scale(wd,block_base+ulong((group+1)/8)*64);
    }
    float sum=0.0f;
#pragma clang loop unroll(full)
    for(uint i=0;i<8;i++) {
        const ulong at=ulong(first)*K+group*32+i*4;
        const float4 a=float4(x[at],x[at+1],x[at+2],x[at+3]),w=weights[i];
        sum=sum+w.x*a.x;
        sum=sum+w.y*a.y;
        sum=sum+w.z*a.z;
        sum=sum+w.w*a.w;
    }
    total=total+sum;
}
y[ulong(first)*N+row]=v41_bf16(total);
"""
    plane = "gqs" if kind == "gate" else "wqs"
    source = f"const uint capacity=uint({plane}_shape[0]/(ulong(N)*K/8));\n"+source
    return names, source


@lru_cache(maxsize=2)
def kernel(kind):
    import mlx.core as mx
    names, source = kernel_spec(kind)
    return mx.fast.metal_kernel(name="mlx2_v41_iq1_column_"+kind+"_v2",
        input_names=names, output_names=["y"], source=source,
        header=HEADER, ensure_row_contiguous=True)


def moe(x, ids, scores, planes, slot_map, *, capacity, hidden_size, intermediate_size,
        prefill=False, max_scratch_bytes=512 << 20, counters=None):
    """Experimental F32 column reduction; no host routing or weight expansion.

    Missing-route output is provisional exactly as in the tiled contract. The
    owner must evaluate missing_routes and discard/retry before publication.
    Prefill selection is unsupported; multi-row ordinary inputs retain this
    same column law and do not silently dispatch the FP16 NAX implementation.
    """
    import mlx.core as mx
    if prefill is not False:
        raise GGUFError("IQ1 column candidate does not implement selected prefill")
    tiled._validate(x, ids, scores, planes, slot_map, capacity,
                    hidden_size, intermediate_size, False, max_scratch_bytes)
    if counters is not None:
        counters["column_f32_calls"] = counters.get("column_f32_calls", 0)+1
    if capacity == 0:
        return mx.zeros(x.shape, mx.float32)
    routes = ids.size
    mid = kernel("gate")(inputs=[x.astype(mx.float32), ids, slot_map, scores.reshape(-1).astype(mx.float32), *planes[:6]],
        template=[("EXPERTS", slot_map.size), ("TOPK", ids.shape[1]),
                  ("K", hidden_size), ("N", intermediate_size)],
        grid=(((intermediate_size+127)//128)*128, routes, 1), threadgroup=(128, 1, 1),
        output_shapes=[(routes, intermediate_size)], output_dtypes=[mx.float32])[0]
    projected = kernel("down")(inputs=[mid, ids, slot_map, *planes[6:]],
        template=[("EXPERTS", slot_map.size), ("K", intermediate_size), ("N", hidden_size)],
        grid=(((hidden_size+127)//128)*128, routes, 1), threadgroup=(128, 1, 1),
        output_shapes=[(routes, hidden_size)], output_dtypes=[mx.float32])[0]
    return tiled.kernel("exact_combine")(inputs=[projected],
        template=[("N", hidden_size), ("TOPK", ids.shape[1])],
        grid=(((hidden_size+255)//256)*256, ids.shape[0], 1), threadgroup=(256, 1, 1),
        output_shapes=[x.shape], output_dtypes=[mx.float32])[0]


def metal_compile_source(kind, *, constant_inputs=False):
    """Offline compiler source without any GPU initialization/execution."""
    names, source = kernel_spec(kind)
    bindings = []
    for i, name in enumerate(names):
        dtype = "uint" if name in ("ids", "slot_map") else "float" if name in ("x", "scores") else "uchar"
        space = "constant" if constant_inputs else "device"
        bindings.append(f"{space} const {dtype}* {name} [[buffer({i})]]")
    for name in names:
        if name+"_shape" in source:
            bindings.append(f"constant int* {name}_shape [[buffer({len(bindings)})]]")
    bindings += [f"device float* y [[buffer({len(bindings)})]]",
                 "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
                 "uint3 thread_position_in_grid [[thread_position_in_grid]]"]
    constants = "constant uint K=5120,N=2304,TOPK=6,EXPERTS=384;\n"
    return "#include <metal_stdlib>\nusing namespace metal;\n"+HEADER+constants+"kernel void probe("+",".join(bindings)+") {\n"+source+"\n}\n"
