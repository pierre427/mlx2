# SPDX-License-Identifier: Apache-2.0
"""Expert-major packed GGUF NAX prefill candidate; no host expert-count sync.

Mechanisms/provenance: deepseek-v4-segmented-prefill.json. Sorted assignments
retain original token/slot order within each expert; one GPU prepass emits
compact16-row tiles. Only bounded activation/partial-output arrays exist:
expert weights remain native packed GGML throughout. This is a distinct
FP16-operand arithmetic profile, default-off and unqualified.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from contextlib import contextmanager
from contextvars import ContextVar
import math

from .reader import GGUFError
from .nax_metal import NAX_HEADER

POLICY='gguf-segmented-nax-fp16-v1'
TILE_ROWS=16
_PREFILL_REQUEST=ContextVar('mlx2_deepseek_segmented_prefill',default=None)


@contextmanager
def expert_prefill_context(policy, *, tokens):
    """Scope distinct expert arithmetic to one explicit prompt forward only.

    Verification, decode and normal model calls never inherit the candidate
    merely because their sequence contains multiple rows. Width binding also
    prevents a nested operation from treating another row geometry as prefill.
    """
    if policy not in ('reference','segmented_nax_fp16'):
        raise GGUFError('unknown expert prefill arithmetic policy')
    if type(tokens) is not int or tokens<=0:
        raise GGUFError('expert prefill context requires positive token count')
    token=_PREFILL_REQUEST.set((policy,tokens))
    try:
        yield
    finally:
        _PREFILL_REQUEST.reset(token)


def segmented_prefill_active(tokens):
    """Factory gate: exact scoped prompt width and at least one full tile."""
    return (type(tokens) is int and tokens>=TILE_ROWS
            and _PREFILL_REQUEST.get()==('segmented_nax_fp16',tokens))

SCAN_SOURCE=r'''
const uint tid=thread_index_in_threadgroup;
threadgroup atomic_uint invalid;
threadgroup uint scan[256];
if(tid==0) atomic_store_explicit(&invalid,0u,memory_order_relaxed);
for(uint i=tid;i<MAX_TILES;i+=256) *((device uint4*)tiles+i)=uint4(0);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint row=tid;row<ROWS;row+=256) {
    if(original_ids[row]>=EXPERTS) atomic_store_explicit(&invalid,1u,memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(atomic_load_explicit(&invalid,memory_order_relaxed)!=0u) {
    if(tid==0) { tile_count[0]=0; status[0]=1; }
    return;
}
uint first=0,last=0;
if(tid<EXPERTS) {
    uint lo=0,hi=ROWS;
    while(lo<hi) { uint mid=(lo+hi)/2; if(sorted_ids[mid]<tid) lo=mid+1; else hi=mid; }
    first=lo; hi=ROWS;
    while(lo<hi) { uint mid=(lo+hi)/2; if(sorted_ids[mid]<=tid) lo=mid+1; else hi=mid; }
    last=lo;
}
const uint count=(last-first+15)/16;
scan[tid]=count;
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint stride=1;stride<256;stride*=2) {
    const uint add=tid>=stride ? scan[tid-stride] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scan[tid]+=add;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
const uint total=scan[255];
if(total>MAX_TILES) {
    if(tid==0) { tile_count[0]=0; status[0]=2; }
    return;
}
const uint start=scan[tid]-count;
for(uint tile=0;tile<count;tile++) {
    const uint row=first+tile*16;
    *((device uint4*)tiles+start+tile)=uint4(row,tid,min(16u,last-row),0);
}
if(tid==0) { tile_count[0]=total; status[0]=0; }
'''

COMMON_PREFIX=r'''
const ushort lane=thread_index_in_simdgroup;
const ushort sg=simdgroup_index_in_threadgroup;
const ushort tid=thread_index_in_threadgroup;
const uint tile=threadgroup_position_in_grid.y;
if(tile>=tile_count[0]) return;
const uint4 desc=*((device const uint4*)tiles+tile);
const uint rowbase=desc.x,expert=desc.y,valid_rows=desc.z;
const int colbase=(threadgroup_position_in_grid.x*4+sg)*32;
const short qid=lane>>2;
const short fm=(qid&4)|((lane>>1)&3);
const short fn=((qid&2)|(lane&1))*4;
threadgroup half xa[16*64];
threadgroup half weights[4*32*64];
threadgroup half* wb=weights+sg*32*64;
constexpr auto desc_mma=matmul2d_descriptor(16,32,64,false,true,false,
    matmul2d_descriptor::mode::multiply);
matmul2d<desc_mma,execution_simdgroup> op;
'''


def _project_source(*,gate_up):
    accumulator='''float sums[16];
#pragma clang loop unroll(full)
for(int j=0;j<16;j++) sums[j]=0.0f;
'''
    if gate_up:
        accumulator+='''float up_sums[16];
#pragma clang loop unroll(full)
for(int j=0;j<16;j++) up_sums[j]=0.0f;
'''
    source=COMMON_PREFIX+accumulator+r'''
for(int kb=0;kb<K;kb+=64) {
    for(int i=tid;i<16*64;i+=128) {
        const uint row=i/64; const int k=kb+i%64;
'''
    source+=('''        const uint token=row<valid_rows ? order[rowbase+row]/TOPK : 0;
        xa[i]=row<valid_rows ? half(x[ulong(token)*K+k]) : half(0.0f);
''' if gate_up else '''        xa[i]=row<valid_rows ? half(x[ulong(rowbase+row)*K+k]) : half(0.0f);
''')
    source+=r'''
    }
    for(int i=lane;i<32*64;i+=32) {
        const int row=colbase+i/64,k=kb+i%64;
        wb[i]=row<N ? half(gguf_value(w+(ulong(expert)*N+row)*ROW_BYTES,k,FORMAT)) : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tensor<threadgroup half,dextents<int32_t,2>,tensor_inline> a(xa,dextents<int32_t,2>(64,16));
    tensor<threadgroup half,dextents<int32_t,2>,tensor_inline> b(wb,dextents<int32_t,2>(64,32));
    auto product=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    op.run(a,b,product);
#pragma clang loop unroll(full)
    for(int j=0;j<16;j++) sums[j]+=product[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
'''
    if gate_up:
        source+=r'''
    // Reuse the18KiB staging allocation, retaining FP32 gate accumulators.
    for(int i=lane;i<32*64;i+=32) {
        const int row=colbase+i/64,k=kb+i%64;
        wb[i]=row<N ? half(gguf_value(up+(ulong(expert)*N+row)*ROW_BYTES,k,FORMAT)) : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    op.run(a,b,product);
#pragma clang loop unroll(full)
    for(int j=0;j<16;j++) up_sums[j]+=product[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
'''
    source+=r'''
}
#pragma clang loop unroll(full)
for(int f=0;f<2;f++) {
#pragma clang loop unroll(full)
    for(int r=0;r<2;r++) {
        const uint row=fm+r*8; const int col=colbase+f*16+fn;
        if(row<valid_rows && col<N) {
#pragma clang loop unroll(full)
            for(int j=0;j<4;j++) {
'''
    source+=('''                const int index=f*8+r*4+j;
                const float gate=min(sums[index],10.0f);
                const float u=clamp(up_sums[index],-10.0f,10.0f);
                y[ulong(rowbase+row)*N+col+j]=(gate/(1.0f+metal::precise::exp(-gate)))*u;
''' if gate_up else '''                y[ulong(rowbase+row)*N+col+j]=sums[f*8+r*4+j];
''')
    return source+'            }\n        }\n    }\n}\n'


GATE_UP_SOURCE=_project_source(gate_up=True)
DOWN_SOURCE=_project_source(gate_up=False)
COMBINE_SOURCE=r'''
const uint column=thread_position_in_grid.x;
const uint token=thread_position_in_grid.y;
if(column>=N || token>=TOKENS) return;
if(status[0]!=0) { out[ulong(token)*N+column]=NAN; return; }
float total=0.0f;
for(uint slot=0;slot<TOPK;slot++) {
    const uint original=token*TOPK+slot;
    total+=scores[original]*rows[ulong(inverse[original])*N+column];
}
out[ulong(token)*N+column]=total;
'''


@lru_cache(maxsize=4)
def _kernel(kind):
    import mlx.core as mx
    sources={
        'scan':(['sorted_ids','original_ids'],['tiles','tile_count','status'],SCAN_SOURCE),
        'gate_up':(['x','order','tiles','tile_count','w','up'],['y'],GATE_UP_SOURCE),
        'down':(['x','tiles','tile_count','w'],['y'],DOWN_SOURCE),
        'combine':(['rows','inverse','scores','status'],['out'],COMBINE_SOURCE),
    }
    ins,outs,source=sources[kind]
    return mx.fast.metal_kernel(name='mlx2_gguf_segmented_nax_'+kind+'_v1',
        input_names=ins,output_names=outs,header=NAX_HEADER,source=source,
        ensure_row_contiguous=True)


@dataclass(frozen=True)
class SegmentedPlan:
    order:object
    inverse:object
    sorted_ids:object
    tiles:object
    tile_count:object
    status:object
    assignment_rows:int
    experts:int
    topk:int
    max_tiles:int


def plan_capacity(rows,experts):
    if type(rows) is not int or type(experts) is not int or not 1<=experts<=256 or rows<1 or rows*experts>=2**32:
        raise GGUFError('segmented plan requires positive rows,<=256experts and unique32bit composite keys')
    return min(rows,(rows+15)//16+experts-1)


def reference_plan(ids,experts):
    """Independent CPU descriptor oracle, not a model execution backend."""
    import numpy as np
    if ids.ndim!=2 or not np.issubdtype(ids.dtype,np.integer) or ids.size<1:
        raise GGUFError('CPU plan requires nonempty integer[T,topk]IDs')
    if np.any(ids<0) or np.any(ids>=experts):
        raise GGUFError('CPU plan IDs outside expert table')
    capacity=plan_capacity(ids.size,experts)
    flat=ids.reshape(-1)
    order=np.argsort(flat,kind='stable').astype(np.uint32)
    inverse=np.argsort(order).astype(np.uint32)
    sorted_ids=flat[order].astype(np.uint32)
    tiles=[]
    for expert in range(experts):
        assigned=np.flatnonzero(sorted_ids==expert)
        if not assigned.size: continue
        for start in range(0,len(assigned),16):
            tiles.append((int(assigned[start]),expert,min(16,len(assigned)-start),0))
    if len(tiles)>capacity: raise GGUFError('CPU tile bound violated')
    return {'order':order,'inverse':inverse,'sorted_ids':sorted_ids,
            'tiles':np.array(tiles,np.uint32).reshape(-1,4),'max_tiles':capacity}


def build_plan(ids,experts):
    import mlx.core as mx
    if mx.default_device()!=mx.gpu or not mx.metal.is_available():
        raise GGUFError('segmented GPU plan requires an owned Metal execution window')
    if ids.ndim!=2 or ids.dtype!=mx.uint32:
        raise GGUFError('segmented GPU plan requires uint32[T,topk]IDs')
    rows=ids.size; capacity=plan_capacity(rows,experts)
    flat=ids.reshape(-1)
    # Unique composite keys provide stable expert order even when argsort's
    # equal-key stability is unspecified. No device-to-host count transfer.
    keys=flat*mx.array(rows,dtype=mx.uint32)+mx.arange(rows,dtype=mx.uint32)
    order=mx.argsort(keys).astype(mx.uint32)
    inverse=mx.argsort(order).astype(mx.uint32)
    sorted_ids=flat[order]
    tiles,count,status=_kernel('scan')(inputs=[sorted_ids,flat],
        template=[('ROWS',rows),('EXPERTS',experts),('MAX_TILES',capacity)],
        grid=(256,1,1),threadgroup=(256,1,1),
        output_shapes=[(capacity,4),(1,),(1,)],output_dtypes=[mx.uint32]*3)
    return SegmentedPlan(order,inverse,sorted_ids,tiles,count,status,rows,experts,ids.shape[1],capacity)


def scratch_bytes(tokens,topk,hidden,output,experts=256):
    rows=tokens*topk
    tiles=plan_capacity(rows,experts)
    return rows*(hidden+output)*4 + rows*6*4 + tiles*4*4 + 8


def _validate(x,ids,scores,gate,up,down,max_scratch_bytes):
    if x.ndim!=2 or ids.ndim!=2 or ids.shape!=scores.shape or ids.shape[0]!=x.shape[0] or not ids.shape[1]:
        raise GGUFError('segmented prefill requires matching x[T,K],ids/scores[T,topk]')
    if len(gate.shape)!=3 or gate.shape!=up.shape:
        raise GGUFError('segmented gate/up need matching expert-major3D weights')
    experts,hidden,k=gate.shape
    if down.shape!=(experts,k,hidden) or x.shape[1]!=k or ids.shape[1]>experts:
        raise GGUFError('segmented down/input/expert geometry mismatch')
    if gate.ggml_type!=up.ggml_type or any(w.ggml_type not in (8,10,16) for w in (gate,up,down)):
        raise GGUFError('segmented native GGUF formats unsupported')
    if k%256 or hidden%256:
        raise GGUFError('segmented GGUF rows must align256coordinates')
    required=scratch_bytes(x.shape[0],ids.shape[1],hidden,k,experts)
    if type(max_scratch_bytes) is not int or max_scratch_bytes<required:
        raise GGUFError(f'segmented prefill scratch{required} exceeds cap{max_scratch_bytes}')
    return experts,hidden,k


def segmented_moe_prefill(x,ids,scores,gate,up,down,*,max_scratch_bytes=512<<20):
    """Explicit M5 NAX candidate; no automatic fallback or serving selection."""
    import mlx.core as mx
    if mx.default_device()!=mx.gpu or not mx.metal.is_available() or 'Apple M5' not in mx.device_info().get('device_name',''):
        raise GGUFError('segmented GGUF NAX requires owned AppleM5 GPU execution')
    experts,hidden,k=_validate(x,ids,scores,gate,up,down,max_scratch_bytes)
    if ids.dtype!=mx.uint32:
        raise GGUFError('segmented expert IDs must beuint32')
    plan=build_plan(ids,experts)
    common=[('TOPK',ids.shape[1]),('K',k),('N',hidden),('ROW_BYTES',gate.row_bytes),('FORMAT',gate.ggml_type)]
    mid=_kernel('gate_up')(inputs=[x.astype(mx.float32),plan.order,plan.tiles,plan.tile_count,gate.data,up.data],
        template=common,grid=(((hidden+127)//128)*128,plan.max_tiles,1),threadgroup=(128,1,1),
        output_shapes=[(plan.assignment_rows,hidden)],output_dtypes=[mx.float32])[0]
    projected=_kernel('down')(inputs=[mid,plan.tiles,plan.tile_count,down.data],
        template=[('K',hidden),('N',k),('ROW_BYTES',down.row_bytes),('FORMAT',down.ggml_type)],
        grid=(((k+127)//128)*128,plan.max_tiles,1),threadgroup=(128,1,1),
        output_shapes=[(plan.assignment_rows,k)],output_dtypes=[mx.float32])[0]
    return _kernel('combine')(inputs=[projected,plan.inverse,scores.astype(mx.float32).reshape(-1),plan.status],
        template=[('TOKENS',x.shape[0]),('TOPK',ids.shape[1]),('N',k)],
        grid=(((k+255)//256)*256,x.shape[0],1),threadgroup=(256,1,1),
        output_shapes=[(x.shape[0],k)],output_dtypes=[mx.float32])[0]


def metal_compile_source(kind):
    """Standalone Metal4 CPU compiler source; never dispatches a GPU."""
    constants='\nconstant uint ROWS=192,EXPERTS=6,MAX_TILES=17,TOPK=6,K=256,N=256,FORMAT=16,ROW_BYTES=66,TOKENS=32;\n'
    bindings={
        'scan':'device const uint* sorted_ids [[buffer(0)]],device const uint* original_ids [[buffer(1)]],device uint* tiles [[buffer(2)]],device uint* tile_count [[buffer(3)]],device uint* status [[buffer(4)]],',
        'gate_up':'device const float* x [[buffer(0)]],device const uint* order [[buffer(1)]],device const uint* tiles [[buffer(2)]],device const uint* tile_count [[buffer(3)]],device const uchar* w [[buffer(4)]],device const uchar* up [[buffer(5)]],device float* y [[buffer(6)]],',
        'down':'device const float* x [[buffer(0)]],device const uint* tiles [[buffer(1)]],device const uint* tile_count [[buffer(2)]],device const uchar* w [[buffer(3)]],device float* y [[buffer(4)]],',
        'combine':'device const float* rows [[buffer(0)]],device const uint* inverse [[buffer(1)]],device const float* scores [[buffer(2)]],device const uint* status [[buffer(3)]],device float* out [[buffer(4)]],',
    }
    attrs='uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],uint3 thread_position_in_grid [[thread_position_in_grid]],uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]'
    source={'scan':SCAN_SOURCE,'gate_up':GATE_UP_SOURCE,'down':DOWN_SOURCE,'combine':COMBINE_SOURCE}[kind]
    return '#include <metal_stdlib>\nusing namespace metal;\n'+NAX_HEADER+constants+'kernel void compile_'+kind+'('+bindings[kind]+attrs+') {\n'+source+'\n}\n'
