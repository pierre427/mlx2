# SPDX-License-Identifier: MIT
"""Bounded original Source split-K F32 K8 projection component candidate.

The installed legacy arithmetic and contiguous activation loader are retained
verbatim. There is no NAX substitution, BF16 operand cast, or partial rounding.
Opt-in Source adapter policies remain unqualified; recorded GPU parity is
bounded to its source/profile probes. See provenance in
deepseek-v41-source-prefill-legacy-tiles.json.
"""
from functools import lru_cache
import hashlib

K,N=5120,2304
DEFAULT_TILE=(32,32)
HASHES={
 'fp_quantized.h':'fb4bf0919642717c5e36cfa7be0628e651844586cae1b405c27b2057a17dd1c8',
 'fp4.h':'e3159945d7e5ae24f7a1ca9c4875b2a5d15cfb6952405df24a0df1bb2622a93a',
 'fp8.h':'9a9d77a6295f66b4ada4311c3b855b6ecb7a2b38f28623109e3c61d2e17e81ae',
 'steel/defines.h':'b03cea6a7d5cfe814e6838d8536e2aacdcafd2744cb408e954fda336ce21759a',
 'steel/gemm/mma.h':'9b37709de40c3acaa5411875e52c8d26e6de093261741f671fbdab9c39a852b7',
 'steel/gemm/loader.h':'c1b82153670b1e18371a6e88fdbcf440af692957775c2f837700315ab8a5112e',
 'steel/gemm/transforms.h':'73487e29beab5af9478bc931c49b5e8d66937c2fa2651499867f47ac5529afc2',
 'steel/utils.h':'d4c36298145d1c5617a94f07dbfabf6ac932afe2fd7ea0c9291f1fb385c5404a',
 'steel/utils/integral_constant.h':'886eccaac4f8f00eab5a10551fdb5315df7ab7462aba13d69bb95765c1a4735e',
 'steel/utils/type_traits.h':'4766c5a3809d3cf7740e63acc5703ef76b6b4e85d8baf16602c9d517a2fd426c',
 'steel/gemm/kernels/steel_gemm_splitk.h':'2ada2a7ea548f6c4b0e7962b4a2f163e8a212620ecc46e078a06647df12b0635',
}


def _tile(tile):
    if (type(tile)is not tuple or len(tile)!=2 or any(type(v)is not int for v in tile)
            or tile not in ((32,32),(16,64))):
        raise ValueError('Source legacy tile admits only(32,32) or(16,64)')
    return tile


def plan(rows,*,tile=DEFAULT_TILE):
    if type(rows) is not int or not 13<=rows<=96:
        raise ValueError('Source legacy tile component admits M13..96 only')
    p=5 if rows<=32 else 2
    bm,bn=_tile(tile);groups=(N//bn,(rows+bm-1)//bm,p)
    return dict(rows=rows,partitions=p,k_partition=K//p,tile=(bm,32,bn),tile_config=tile,
        warp_shape=(2,2),k_step=8,threadgroup=(32,2,2),
        grid=tuple(a*b for a,b in zip(groups,(32,2,2))),
        threadgroups=groups,
        partial_shape=(p,rows,N),partial_bytes=p*rows*N*4,
        raw_shape=(rows,N),scratch_bytes=(p+1)*rows*N*4,
        threadgroup_bytes=(bm+bn)*36*4,input_bindings=3,
        input_dtype='float32',threadgroup_dtype='float32',partial_dtype='float32',
        global_weight_repack_bytes=0,profile='experimental-unselected-legacy-law')


@lru_cache(maxsize=1)
def _installed():
    from .source_dense import _installed as installed
    root,headers=installed()
    for name,wanted in HASHES.items():
        if hashlib.sha256((headers/name).read_bytes()).hexdigest()!=wanted:
            raise RuntimeError('Source legacy tile installed helper changed; re-audit required')
    return root,headers


@lru_cache(maxsize=1)
def _header():
    _,headers=_installed();seen=set()
    def expand(name):
        if name in seen:return ''
        if name not in HASHES:
            raise RuntimeError('Source legacy tile unpinned header dependency: '+name)
        seen.add(name);result=[]
        for line in (headers/name).read_text().splitlines():
            trimmed=line.strip()
            prefix='#include "mlx/backend/metal/kernels/'
            if trimmed.startswith(prefix) and trimmed.endswith('"'):
                result.append(expand(trimmed[len(prefix):-1]))
            elif trimmed!='#pragma once':result.append(line)
        return '\n'.join(result)
    deps='\n'.join(expand(s)for s in ('steel/gemm/mma.h','steel/gemm/loader.h','fp4.h','fp8.h'))
    text=(headers/'fp_quantized.h').read_text()
    prefix=text[text.index('using namespace metal;'):text.index('template <typename T, int group_size, int bits, int D>')]
    begin=text.index('template <\n    typename T,\n    const int group_size,\n    const int bits,\n    const bool aligned_N,\n    typename U = T,')
    end=text.index('\ntemplate <\n    typename T,\n    int group_size,\n    int bits,\n    bool has_global_scale = false,\n    int BM = 32,',begin)
    helper=text[begin:end]
    assert helper.count('METAL_FUNC void fp_qmm_t_impl(')==1
    # No substitutions in the loader or projection helper: even constant
    # reference arguments and thread-position attributes remain stock.
    return ('// Derived installed MLX helpers, Apple2024-2026, MIT.\n'
        '// License: provenance/source-mxfp4-installed-MLX-LICENSE.txt.\n'
        +deps+'\nnamespace mlx2_source_legacy_tiles {\n'+prefix+helper+'\n}\n')


def kernel_spec(rows,*,tile=DEFAULT_TILE):
    p=plan(rows,tile=tile);bm,bn=tile
    constants=f'''\nnamespace mlx2_source_legacy_tiles {{
constant int SRC_K=5120,SRC_N=2304,SRC_M={rows},SRC_KPART={p['k_partition']};
}}\n'''
    source=f'''
using namespace mlx2_source_legacy_tiles;
const uint3 tid=threadgroup_position_in_grid;
const int k_start=tid.z*SRC_KPART;
const device uint32_t* w=(const device uint32_t*)(backing+layout_offsets[0]+k_start/2);
const device uchar* scales=backing+layout_offsets[1]+k_start/32;
const device float* selected_x=x+k_start;
device float* selected_y=partials+tid.z*{rows}*2304;
threadgroup float Xs[{bm}*36];
threadgroup float Ws[{bn}*36];
fp_qmm_t_impl<float,32,4,true,float,false,{bm},32,{bn}>(
 w,scales,nullptr,selected_x,selected_y,Xs,Ws,SRC_K,SRC_N,SRC_M,SRC_KPART,
 tid,thread_index_in_threadgroup,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
'''
    return dict(p,header=_header()+constants,source=source,
        input_names=['x','backing','layout_offsets'],output_names=['partials'])


def reduction_spec():
    # Derived gemm_splitk_accum: its scalar ordered loop is preserved.
    # A separate safe compile prevents reassociation after wrapper inlining.
    return dict(input_names=['partials','metadata'],output_names=['raw'],
        header='#include <metal_stdlib>\nusing namespace metal;\n',
        source='''
const uint column=thread_position_in_grid.x,row=thread_position_in_grid.y;
const int partitions=metadata[0],stride=metadata[1],ldd=metadata[2];
const device float* C_split=partials+column+row*size_t(ldd);
size_t offset=0;
float out=0;
#pragma clang loop unroll(disable)
for(int i=0;i<partitions;i++) {
 out+=C_split[offset];
 offset+=stride;
}
raw[column+row*size_t(ldd)]=out;
''',compile_options={'math_mode':'safe'})


@lru_cache(maxsize=168)
def _kernel(rows,tile=DEFAULT_TILE):
    import mlx.core as mx
    s=kernel_spec(rows,tile=tile)
    name=f'mlx2_source_legacy_splitk_r{rows}'
    if tile!=DEFAULT_TILE:name+=f'_m{tile[0]}n{tile[1]}'
    return mx.fast.metal_kernel(name=name,
        input_names=s['input_names'],output_names=s['output_names'],
        header=s['header'],source=s['source'],ensure_row_contiguous=True)


@lru_cache(maxsize=1)
def _reduction_kernel():
    import mlx.core as mx
    s=reduction_spec()
    return mx.fast.metal_kernel(name='mlx2_source_legacy_splitk_ordered_accum',
        input_names=s['input_names'],output_names=s['output_names'],
        header=s['header'],source=s['source'],compile_options=s['compile_options'],
        ensure_row_contiguous=True)


def prepare_inputs(x,expert,projection='gate',*,tile=DEFAULT_TILE):
    import mlx.core as mx
    from ...adapters.deepseek_v41_source_buffers import PackedExpert,ExpertLayout
    if projection not in ('gate','up'):
        raise ValueError('Source legacy tile component admits gate/up only')
    if (not isinstance(expert,PackedExpert) or not isinstance(expert.layout,ExpertLayout)
            or expert.layout.kind!='original'):
        raise ValueError('Source legacy tile component needs one original PackedExpert')
    if x.ndim!=2 or x.shape[-1]!=K or x.dtype!=mx.float32:
        raise ValueError('Source legacy tile component requires selected F32[M,5120]')
    p=plan(x.shape[0],tile=tile)
    if (expert.backing.ndim!=1 or expert.backing.dtype!=mx.uint8
            or expert.backing.shape!=(expert.layout.nbytes,)):
        raise ValueError('Source legacy tile immutable backing geometry mismatch')
    q,s=(expert.layout.tensor(projection+'.'+f)for f in ('qweight','scales'))
    if (q.shape!=(N,K//8) or q.dtype!='<u4' or s.shape!=(N,K//32) or s.dtype!='|u1'
            or expert[projection+'.qweight'].shape!=q.shape
            or expert[projection+'.qweight'].dtype!=mx.uint32
            or expert[projection+'.scales'].shape!=s.shape
            or expert[projection+'.scales'].dtype!=mx.uint8):
        raise ValueError('Source legacy tile packed projection geometry mismatch')
    return p,(q.offset,s.offset)


def project(x,expert,*,projection='gate',return_partials=False,tile=DEFAULT_TILE):
    """One contiguous selected projection; caller retains immutable backing.

    Diagnostic tuple is (BF16-rounded F32,raw F32,F32partitions).
    Recorded GPU probes cover bounded raw/BF16 outputs; candidate partials
    are compared between implementations, not read from stock internals.
    """
    import mlx.core as mx
    if type(return_partials) is not bool:
        raise ValueError('Source legacy tile diagnostic flag must be boolean')
    p,offsets=prepare_inputs(x,expert,projection,tile=tile)
    from .source_dense_legacy import _device_supported
    if not _device_supported():
        raise ValueError('Source legacy tile needs caller-owned M5 Max TF32-off GPU')
    kernel=_kernel(p['rows'])if tile==DEFAULT_TILE else _kernel(p['rows'],tile)
    partials=kernel(inputs=[x,expert.backing,mx.array(offsets,mx.uint32)],
        grid=p['grid'],threadgroup=p['threadgroup'],output_shapes=[p['partial_shape']],
        output_dtypes=[mx.float32])[0]
    metadata=mx.array([p['partitions'],p['rows']*N,N],mx.int32)
    raw=_reduction_kernel()(inputs=[partials,metadata],grid=(N,p['rows'],1),
        threadgroup=(256,1,1),output_shapes=[p['raw_shape']],output_dtypes=[mx.float32])[0]
    rounded=raw.astype(mx.bfloat16).astype(mx.float32)
    return (rounded,raw,partials) if return_partials else rounded


def compilation_source(rows,*,tile=DEFAULT_TILE):
    s=kernel_spec(rows,tile=tile)
    return ('#include "mlx/backend/metal/kernels/utils.h"\n'+s['header']+'''
kernel void legacy_tile_probe(const device float* x [[buffer(0)]],
const device uchar* backing [[buffer(1)]],const device uint* layout_offsets [[buffer(2)]],
device float* partials [[buffer(3)]],
uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
'''+s['source']+'\n}\n')


def batch_plan(row_counts,*,tile=DEFAULT_TILE):
    """Compact whole-expert arena; bucket by both stock P and row tiles."""
    counts=tuple(row_counts)
    if not 1<=len(counts)<=15:
        raise ValueError('Source legacy batch admits1..15 whole experts')
    plans=[plan(m,tile=tile)for m in counts];bm,bn=tile
    bands={(p['partitions'],(m+31)//32)for p,m in zip(plans,counts)}
    if len(bands)!=1:
        raise ValueError('Source legacy batch requires homogeneous(P,ceilM32)')
    partitions,_=next(iter(bands));tiles=max(p['threadgroups'][1]for p in plans);offsets=[0]
    for count in counts:offsets.append(offsets[-1]+count)
    total=offsets[-1];groups=(N//bn,tiles,partitions*len(counts))
    return dict(row_counts=counts,row_offsets=tuple(offsets),rows=total,
        experts=len(counts),partitions=partitions,k_partition=K//partitions,m_tiles=tiles,
        tile_config=tile,tile=(bm,32,bn),
        partial_shape=(partitions,total,N),raw_shape=(total,N),partition_stride=total*N,
        partial_bytes=partitions*total*N*4,scratch_bytes=(partitions+1)*total*N*4,
        gathered_input_bytes=total*K*4,threadgroup=(32,2,2),threadgroups=groups,
        grid=tuple(a*b for a,b in zip(groups,(32,2,2))),threadgroup_bytes=(bm+bn)*36*4,
        input_bindings=len(counts)+2,global_weight_repack_bytes=0,
        profile='experimental-unselected-legacy-law-batch')


@lru_cache(maxsize=1)
def _batch_header():
    header=_header();old='const constant int& M,'
    if header.count(old)!=1:
        raise RuntimeError('Source legacy batch M forwarding anchor changed')
    # Argument forwarding only. The entire stock helper body, including
    # loaders, K8 MMA sequence, masks and stores, remains byte-identical.
    return header.replace(old,'const int M,')


def batch_kernel_spec(partitions,m_tiles,expert_count,*,tile=DEFAULT_TILE):
    bm,bn=_tile(tile)
    bands=((5,1),(2,2),(2,3))if tile==DEFAULT_TILE else ((5,1),(5,2),(2,3),(2,4),(2,5),(2,6))
    if (type(partitions)is not int or type(m_tiles)is not int or type(expert_count)is not int
            or (partitions,m_tiles)not in bands
            or not 1<=expert_count<=15):
        raise ValueError('Source legacy batch unsupported compiled bucket')
    kpart=K//partitions
    constants=f'''\nnamespace mlx2_source_legacy_tiles {{
constant int SRC_K=5120,SRC_N=2304,SRC_KPART={kpart};
}}\n'''
    cases='\n'.join(f'case {i}: backing=expert{i};break;'for i in range(expert_count))
    skip=f'if(physical.y*{bm}>=original_m)return;\n'if tile!=DEFAULT_TILE else ''
    source=f'''
using namespace mlx2_source_legacy_tiles;
const uint3 physical=threadgroup_position_in_grid;
const uint slot=physical.z/{partitions},part=physical.z%{partitions};
const uint total_rows=descriptor[0];
const uint row_offset=descriptor[1+slot*4],original_m=descriptor[2+slot*4];
{skip}const uint qoffset=descriptor[3+slot*4],soffset=descriptor[4+slot*4];
const int k_start=part*SRC_KPART;
const device uchar* backing=nullptr;
switch(slot) {{
{cases}
}}
const device uint32_t* w=(const device uint32_t*)(backing+qoffset+k_start/2);
const device uchar* scales=backing+soffset+k_start/32;
const device float* selected_x=gathered_x+row_offset*5120+k_start;
device float* selected_y=partials+part*total_rows*2304+row_offset*2304;
threadgroup float Xs[{bm}*36];
threadgroup float Ws[{bn}*36];
fp_qmm_t_impl<float,32,4,true,float,false,{bm},32,{bn}>(
 w,scales,nullptr,selected_x,selected_y,Xs,Ws,SRC_K,SRC_N,original_m,SRC_KPART,
 uint3(physical.xy,part),thread_index_in_threadgroup,
 simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
'''
    return dict(header=_batch_header()+constants,source=source,
        input_names=['gathered_x','descriptor']+[f'expert{i}'for i in range(expert_count)],
        output_names=['partials'],partitions=partitions,m_tiles=m_tiles,experts=expert_count,
        total_buffer_bindings=expert_count+3)


@lru_cache(maxsize=135)
def _batch_kernel(partitions,m_tiles,expert_count,tile=DEFAULT_TILE):
    import mlx.core as mx
    s=batch_kernel_spec(partitions,m_tiles,expert_count,tile=tile)
    name=f'mlx2_source_legacy_batch_p{partitions}_t{m_tiles}_e{expert_count}'
    if tile!=DEFAULT_TILE:name+=f'_m{tile[0]}n{tile[1]}'
    return mx.fast.metal_kernel(name=name,
        input_names=s['input_names'],output_names=s['output_names'],
        header=s['header'],source=s['source'],ensure_row_contiguous=True)


def prepare_batch_inputs(gathered_x,experts,row_counts,projection='gate',*,tile=DEFAULT_TILE):
    import mlx.core as mx
    import numpy as np
    p=batch_plan(row_counts,tile=tile);experts=tuple(experts)
    if (len(experts)!=p['experts'] or gathered_x.ndim!=2
            or gathered_x.shape!=(p['rows'],K) or gathered_x.dtype!=mx.float32):
        raise ValueError('Source legacy batch gathered/expert geometry mismatch')
    descriptor=[p['rows']]
    for i,(expert,count)in enumerate(zip(experts,p['row_counts'])):
        # Geometry only, with no device views/copies/readbacks for validation.
        class Shape:
            ndim=2;shape=(count,K);dtype=mx.float32
        _,offsets=prepare_inputs(Shape(),expert,projection,tile=tile)
        descriptor.extend((p['row_offsets'][i],count,*offsets))
    descriptor=np.asarray(descriptor,np.uint32);descriptor.flags.writeable=False
    return p,descriptor,experts


def batch_project(gathered_x,experts,row_counts,*,projection='gate',return_partials=False,tile=DEFAULT_TILE):
    """Original whole-expert projection batch; immutable backings stay owned.

    Input rows are already contiguous expert segments in caller/planner order.
    Returns (rounded,raw,partials) diagnostically. Bounded parity receipts are
    source/profile-bound and do not qualify the route.
    """
    import mlx.core as mx
    if type(return_partials)is not bool:
        raise ValueError('Source legacy batch diagnostic flag must be boolean')
    p,descriptor,experts=prepare_batch_inputs(gathered_x,experts,row_counts,projection,tile=tile)
    from .source_dense_legacy import _device_supported
    if not _device_supported():
        raise ValueError('Source legacy batch needs caller-owned M5 Max TF32-off GPU')
    args=(p['partitions'],p['m_tiles'],p['experts'])
    kernel=_batch_kernel(*args)if tile==DEFAULT_TILE else _batch_kernel(*args,tile)
    partials=kernel(
        inputs=[gathered_x,mx.array(descriptor,mx.uint32)]+[e.backing for e in experts],
        grid=p['grid'],threadgroup=p['threadgroup'],output_shapes=[p['partial_shape']],
        output_dtypes=[mx.float32])[0]
    metadata=mx.array([p['partitions'],p['partition_stride'],N],mx.int32)
    raw=_reduction_kernel()(inputs=[partials,metadata],grid=(N,p['rows'],1),
        threadgroup=(256,1,1),output_shapes=[p['raw_shape']],output_dtypes=[mx.float32])[0]
    rounded=raw.astype(mx.bfloat16).astype(mx.float32)
    return (rounded,raw,partials)if return_partials else rounded


def batch_compilation_source(partitions,m_tiles,expert_count,*,tile=DEFAULT_TILE):
    s=batch_kernel_spec(partitions,m_tiles,expert_count,tile=tile)
    buffers=['const device float* gathered_x','const device uint* descriptor']+[
        f'const device uchar* expert{i}'for i in range(expert_count)]+['device float* partials']
    declarations=',\n'.join(f'{d} [[buffer({i})]]'for i,d in enumerate(buffers))
    return ('#include "mlx/backend/metal/kernels/utils.h"\n'+s['header']
        +'\nkernel void legacy_batch_probe('+declarations+''',
uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
'''+s['source']+'\n}\n')


def middle_consumer_spec(exp_spelling):
    """Two ordered F32 reducers plus the existing Source GU epilogue.

    Compile mode remains the existing epilogue's mode. Reassociation is
    disabled locally for the add-only reducer instead of changing exp/math
    lowering for the served sigmoid helper.
    """
    from .source_gate_up_fusion import _EPILOGUE,EXP_SPELLINGS
    if exp_spelling not in EXP_SPELLINGS:
        raise ValueError('Source legacy middle unknown served sigmoid spelling')
    ordered=r'''
inline float2 source_legacy_ordered_pair(
 const device float* gate,const device float* up,
 const int partitions,const int stride) {
 #pragma clang fp reassociate(off)
 #pragma clang fp contract(off)
 float g=0,u=0;
 size_t offset=0;
 #pragma clang loop unroll(disable)
 for(int p=0;p<partitions;p++) {
  g+=gate[offset];
  u+=up[offset];
  offset+=stride;
 }
 return float2(g,u);
}
'''
    return dict(header='#include <metal_stdlib>\nusing namespace metal;\n'
                +_EPILOGUE.replace('__EXP__',exp_spelling)+ordered,
        source=r'''
const uint i=thread_position_in_grid.x;
const int partitions=metadata[0],stride=metadata[1];
const float2 reduced=source_legacy_ordered_pair(gate_partials+i,up_partials+i,partitions,stride);
const float value=source_gu_weighted(reduced.x,reduced.y,scores[slots[i/2304]]);
float maximum=abs(value);
for(uint step=16;step>0;step/=2) maximum=max(maximum,simd_shuffle_xor(maximum,step));
out[i]=source_gu_qat(value,maximum,inverse[0]);
''',input_names=['gate_partials','up_partials','scores','slots','inverse','metadata'],
        output_names=['out'],threadgroup=(128,1,1),total_buffer_bindings=7,
        compile_options=None)


@lru_cache(maxsize=2)
def _middle_consumer_kernel(exp_spelling):
    import mlx.core as mx
    s=middle_consumer_spec(exp_spelling)
    return mx.fast.metal_kernel(name='mlx2_source_legacy_reduce_middle_'+
        ('precise'if 'precise'in exp_spelling else'fast'),
        input_names=s['input_names'],output_names=s['output_names'],
        header=s['header'],source=s['source'],ensure_row_contiguous=True)


def prepare_middle_inputs(gathered_x,experts,row_counts,scores,slots,*,tile=DEFAULT_TILE):
    import mlx.core as mx
    experts=tuple(experts)
    gate,gate_descriptor,retained=prepare_batch_inputs(gathered_x,experts,row_counts,'gate',tile=tile)
    up,up_descriptor,_=prepare_batch_inputs(gathered_x,experts,row_counts,'up',tile=tile)
    if (scores.ndim!=1 or not 1<=scores.shape[0]<=6144 or scores.dtype!=mx.float32
            or slots.ndim!=1 or slots.shape!=(gate['rows'],) or slots.dtype!=mx.uint32):
        raise ValueError('Source legacy middle requires flatF32 scores and trustedU32 occurrence slots')
    if gate['partial_shape']!=up['partial_shape']:
        raise ValueError('Source legacy middle gate/up partition ownership mismatch')
    return gate,gate_descriptor,up_descriptor,retained


def batch_middle(gathered_x,experts,row_counts,scores,slots,*,exp_spelling,tile=DEFAULT_TILE):
    """F32 gathered rows→only final FP8-QAT F32 middle, no GU intermediates.

    Caller proves valid score slots and served exp spelling, retains immutable
    backings/producers, and owns GPU completion. There are two independent
    partial kernels followed by one consumer; no cross-threadgroup barrier.
    """
    import mlx.core as mx
    # Validate spelling before dispatch or optional served-function probes.
    middle_consumer_spec(exp_spelling)
    p,gate_descriptor,up_descriptor,experts=prepare_middle_inputs(
        gathered_x,experts,row_counts,scores,slots,tile=tile)
    from .source_dense_legacy import _device_supported
    if not _device_supported():
        raise ValueError('Source legacy middle needs caller-owned M5 Max TF32-off GPU')
    args=(p['partitions'],p['m_tiles'],p['experts'])
    producer=_batch_kernel(*args)if tile==DEFAULT_TILE else _batch_kernel(*args,tile)
    def produce(descriptor):
        return producer(inputs=[gathered_x,mx.array(descriptor,mx.uint32)]+[e.backing for e in experts],
            grid=p['grid'],threadgroup=p['threadgroup'],output_shapes=[p['partial_shape']],
            output_dtypes=[mx.float32])[0]
    gate_partials,up_partials=produce(gate_descriptor),produce(up_descriptor)
    return _middle_consumer_kernel(exp_spelling)(
        inputs=[gate_partials,up_partials,scores,slots,mx.array([1/448],mx.float32),
                mx.array([p['partitions'],p['partition_stride']],mx.int32)],
        grid=(p['rows']*N,1,1),threadgroup=(128,1,1),
        output_shapes=[p['raw_shape']],output_dtypes=[mx.float32])[0]


def middle_compilation_source(exp_spelling):
    s=middle_consumer_spec(exp_spelling)
    return s['header']+'''
kernel void legacy_middle_probe(const device float* gate_partials [[buffer(0)]],
const device float* up_partials [[buffer(1)]],const device float* scores [[buffer(2)]],
const device uint* slots [[buffer(3)]],const device float* inverse [[buffer(4)]],
const device int* metadata [[buffer(5)]],device float* out [[buffer(6)]],
uint3 thread_position_in_grid [[thread_position_in_grid]]) {
'''+s['source']+'\n}\n'
