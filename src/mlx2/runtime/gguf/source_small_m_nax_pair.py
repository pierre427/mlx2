# SPDX-License-Identifier: Apache-2.0
"""Unwired original M8..12 mapped paired NAX experiment.

Each SIMD group loads one BF16 activation fragment and uses it for independent
F32 gate/up accumulators. P1/P5 are distinct from the ordinary small-M QMV law.
Caller retains original expert backing leases through completion. No GPU on
import, adapter selection, or weight repacking. See matching provenance.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

K,N=5120,2304
PROFILES={f'small_m_nax_p{p}_ordered_f32_v1':p for p in (1,5)}
PINS={'source_prefill_nax.py':'29c1f7ec6f21228d180d42b02111e3385a8ee41b18da073dcc15e6bf680a3416',
      'source_prefill_nax_middle.py':'f25f35372f219c436e61076210d5b24b980d0ed935a3bb4514f6887cdb694518'}


def plan(row_counts,*,numerical_profile):
    counts=tuple(row_counts)
    if numerical_profile not in PROFILES:raise ValueError('Explicit small-M numerical profile required')
    if not 1<=len(counts)<=15 or any(type(m)is not int or not 8<=m<=12 for m in counts):
        raise ValueError('Small-M original rows8..12, experts1..15 required')
    p=PROFILES[numerical_profile];rows=sum(counts);offsets=[0]
    for m in counts:offsets.append(offsets[-1]+m)
    return dict(numerical_profile=numerical_profile,partitions=p,k_partition=K//p,experts=len(counts),
        row_counts=counts,row_offsets=tuple(offsets),rows=rows,partial_shape=(p,rows,N),output_shape=(rows,N),
        grid=(N,1,p*len(counts)),threadgroup=(64,1,1),threadgroups=(36,1,p*len(counts)),
        active_fragment_rows=16,nominal_fragment_utilization=rows/(16*len(counts)),
        threadgroup_bytes=2*64*72*2,paired_partial_bytes=2*p*rows*N*4,
        explicit_activation_gather_bytes=0,coefficient_repack_bytes=0,
        accumulator_elements_per_thread=32,activation_fragment_loads_per_substep_per_simdgroup=1,
        input_bindings=len(counts)+3,total_buffer_bindings=len(counts)+5)


def validate_row_map(row_counts,row_indices,source_rows):
    """Validate CPU-generated indices before upload; never read GPU indices back."""
    import numpy as np
    counts=tuple(row_counts)
    if (type(source_rows)is not int or not 1<=source_rows<=1024
            or not isinstance(row_indices,np.ndarray)or row_indices.dtype!=np.uint32
            or row_indices.shape!=(sum(counts),)or not row_indices.flags.c_contiguous
            or np.any(row_indices>=source_rows)):
        raise ValueError('Small-M mapped CPU U32 indices out of bounds/shape')
    return row_indices


@lru_cache(maxsize=1)
def verify_sources():
    root=Path(__file__).resolve().parent
    for name,expected in PINS.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Small-M pinned source changed: '+name)
    from .source_prefill_nax_profile import verify_sources as verify
    verify()


def producer_spec(partitions,expert_count,*,projection='paired'):
    if type(partitions)is not int or partitions not in(1,5)or type(expert_count)is not int or not 1<=expert_count<=15:
        raise ValueError('Small-M producer bucket invalid')
    if projection not in('paired','gate','up'):raise ValueError('Small-M projection invalid')
    from .source_prefill_nax import _header
    cases='\n'.join(f'case {i}: backing=expert{i};break;'for i in range(expert_count))
    # Single projection mode is an independent same-profile diagnostic; paired
    # mode uses the SAME Arow in two MMAs, not a doubled virtual-N grid.
    paired=projection=='paired';select=0 if projection!='up'else 2
    setup_up='''Mxfp4Q<32> uq{backing+descriptor[base+5]}; uq.advance(column*160+kbegin/32);
TileLoader<Mxfp4Q<32>,G,false> ul(backing+descriptor[base+4]+column*2560+kbegin/2,
 backing+descriptor[base+4]+column*2560+kbegin/2,5120,uq,uq,sg*32+lane);
threadgroup bfloat UW[64*72];
NAXTile<float,1,2> U; U.clear();'''if paired else''
    load_up='ul.fetch(k);ul.store(UW);'if paired else''
    bt_up='NAXTile<bfloat,2,2> UB; UB.template load<bfloat,72,1>(UW+sg*32*72+kk1);'if paired else''
    mma_up='''BaseNAXFrag::mma(U.frag_at(0,0),U.frag_at(0,1),Arow.frag_at(0,kk),
 metal::bool_constant<false>{},UB.frag_at(0,kk),UB.frag_at(1,kk),metal::bool_constant<true>{});'''if paired else''
    store_up='U.store_safe(up_partials+dst,2304,short2(32,m));'if paired else''
    output='gate_partials'if paired else'partials'
    source=f'''
using namespace omlx_gqmm;
using G=Geo<32,64>;
const uint slot=threadgroup_position_in_grid.z/{partitions};
const uint part=threadgroup_position_in_grid.z%{partitions};
const uint base=1+slot*6,total=descriptor[0];
const uint row_base=descriptor[base],m=descriptor[base+1];
const uint column=threadgroup_position_in_grid.x*64;
const uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
const uint kbegin=part*{5120//partitions};
const device uchar* backing=nullptr;
switch(slot){{{cases}}}
Mxfp4Q<32> gq{{backing+descriptor[base+3+{select}]}};gq.advance(column*160+kbegin/32);
TileLoader<Mxfp4Q<32>,G,false> gl(backing+descriptor[base+2+{select}]+column*2560+kbegin/2,
 backing+descriptor[base+2+{select}]+column*2560+kbegin/2,5120,gq,gq,sg*32+lane);
threadgroup bfloat GW[64*72];
NAXTile<float,1,2> D; D.clear();
{setup_up}
uint a_off[kTM][2];map_rows(a_off,row_map,row_base,m,0,5120);
const short2 sc=BaseNAXFrag::get_coord();
for(int k=0;k<{5120//partitions//64};k++){{
 threadgroup_barrier(mem_flags::mem_threadgroup);
 gl.fetch(k);gl.store(GW);{load_up}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 STEEL_PRAGMA_NO_UNROLL
 for(short kk1=0;kk1<64;kk1+=32){{
  NAXTile<bfloat,1,2> Arow;
  load_a_map<bfloat,1>(Arow,x+kbegin+k*64+kk1,a_off,0);
  STEEL_PRAGMA_UNROLL
  for(short h=0;h<2;h++){{
   if(h*8+sc.y>=m){{
    STEEL_PRAGMA_UNROLL
    for(short kk=0;kk<2;kk++){{
     STEEL_PRAGMA_UNROLL
     for(short c=0;c<4;c++)Arow.frag_at(0,kk)[h*4+c]=bfloat(0);
    }}
   }}
  }}
  NAXTile<bfloat,2,2> B;B.template load<bfloat,72,1>(GW+sg*32*72+kk1);
  {bt_up}
  STEEL_PRAGMA_UNROLL
  for(short kk=0;kk<2;kk++){{
   BaseNAXFrag::mma(D.frag_at(0,0),D.frag_at(0,1),Arow.frag_at(0,kk),
    metal::bool_constant<false>{{}},B.frag_at(0,kk),B.frag_at(1,kk),metal::bool_constant<true>{{}});
   {mma_up}
  }}
 }}
}}
const uint dst=part*total*2304+row_base*2304+column+sg*32;
D.store_safe({output}+dst,2304,short2(32,m));
{store_up}
'''
    return dict(header=_header(),source=source,input_names=['x','row_map','descriptor']+[f'expert{i}'for i in range(expert_count)],
                output_names=['gate_partials','up_partials']if paired else['partials'])


@lru_cache(maxsize=90)
def _producer(partitions,expert_count,projection):
    import mlx.core as mx
    return mx.fast.metal_kernel(name=f'mlx2_small_m_pair_p{partitions}_e{expert_count}_{projection}_v1',
        ensure_row_contiguous=True,**producer_spec(partitions,expert_count,projection=projection))


def prepare(x,experts,row_counts,row_indices,*,numerical_profile):
    import mlx.core as mx
    from ...adapters.deepseek_v41_source_buffers import PackedExpert,ExpertLayout
    from .source_gate_up_fusion import _batch_layout
    from .source_dense_legacy import _device_supported
    p=plan(row_counts,numerical_profile=numerical_profile);experts=tuple(experts)
    if (len(experts)!=p['experts']or x.ndim!=2 or x.shape[1]!=K or x.dtype!=mx.float32):
        raise ValueError('Small-M original backing/input shape mismatch')
    validate_row_map(p['row_counts'],row_indices,x.shape[0]);descriptor=[p['rows']]
    for i,e in enumerate(experts):
        if(not isinstance(e,PackedExpert)or not isinstance(e.layout,ExpertLayout)or e.layout.kind!='original'
                or e.backing.dtype!=mx.uint8 or e.backing.shape!=(e.layout.nbytes,)):
            raise ValueError('Small-M requires one packed ORIGINAL allocation perexpert')
        kind,gq,gs,uq,us=_batch_layout(e.layout)
        descriptor.extend([p['row_offsets'][i],p['row_counts'][i],gq,gs,uq,us])
    if not _device_supported():raise ValueError('Small-M requires caller-owned M5 Max TF32-off GPU')
    verify_sources()
    return p,[x,mx.array(row_indices,mx.uint32),mx.array(descriptor,mx.uint32)]+[e.backing for e in experts]


def produce(prepared,*,projection='paired'):
    import mlx.core as mx
    p,inputs=prepared;s=producer_spec(p['partitions'],p['experts'],projection=projection)
    values=_producer(p['partitions'],p['experts'],projection)(inputs=inputs,grid=p['grid'],threadgroup=p['threadgroup'],
        output_shapes=[p['partial_shape']]*len(s['output_names']),output_dtypes=[mx.float32]*len(s['output_names']))
    return tuple(values)if projection=='paired'else values[0]


def reduce_middle(gate,up,scores,slots,*,exp_spelling,return_reduced=False):
    import mlx.core as mx
    from .source_prefill_nax_middle import _consumer,consumer_spec,_score_geometry
    from .source_dense_legacy import _device_supported
    consumer_spec(exp_spelling,return_reduced=return_reduced)
    if(gate.ndim!=3 or gate.shape!=up.shape or gate.shape[0]not in(1,5)or not 1<=gate.shape[1]<=180
            or gate.shape[2]!=N or gate.dtype!=mx.float32 or up.dtype!=mx.float32):
        raise ValueError('Small-M partial geometry mismatch')
    p,rows,n=gate.shape;_score_geometry(rows,scores,slots)
    if not _device_supported():raise ValueError('Small-M requires caller-owned M5 Max TF32-off GPU')
    verify_sources();count=3 if return_reduced else 1
    outputs=_consumer(exp_spelling,return_reduced)(inputs=[gate,up,scores,slots,mx.array([1/448],mx.float32),mx.array([p,rows*n],mx.int32)],
        grid=(rows*n,1,1),threadgroup=(128,1,1),output_shapes=[(rows,n)]*count,output_dtypes=[mx.float32]*count)
    return tuple(outputs)if return_reduced else outputs[0]


def middle(x,experts,row_counts,row_indices,scores,slots,*,numerical_profile,exp_spelling,return_partials=False):
    if type(return_partials)is not bool:raise ValueError('Small-M diagnostic flag must be boolean')
    from .source_prefill_nax_middle import _score_geometry,consumer_spec
    p=plan(row_counts,numerical_profile=numerical_profile);_score_geometry(p['rows'],scores,slots);consumer_spec(exp_spelling)
    prepared=prepare(x,experts,row_counts,row_indices,numerical_profile=numerical_profile)
    gate,up=produce(prepared);out=reduce_middle(gate,up,scores,slots,exp_spelling=exp_spelling)
    return (out,gate,up)if return_partials else out


def compilation_source(partitions,expert_count,*,projection='paired'):
    s=producer_spec(partitions,expert_count,projection=projection)
    args=['const device float* x','const device uint* row_map','const device uint* descriptor']+[f'const device uchar* expert{i}'for i in range(expert_count)]
    args+=['device float* '+name for name in s['output_names']]
    declarations=',\n'.join(f'{a} [[buffer({i})]]'for i,a in enumerate(args))
    return '#include "mlx/backend/metal/kernels/utils.h"\n'+s['header']+'\nkernel void small_m_pair_probe('+declarations+''',
uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {\n'''+s['source']+'\n}\n'
