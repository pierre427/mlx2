# SPDX-License-Identifier: Apache-2.0
"""Original-MXFP4 NAX numerical profile with an explicit different dot law.

The NAX K16 inner product is not the ordinary Source legacy K8 reference.
Only its outer split-K reduction is preserved: ascending F32 additions from+0.
An explicit opt-in adapter policy may use bounded batches; reference is default.
See the associated provenance records for distinct single/batch evidence.
"""
from functools import lru_cache
from pathlib import Path
import hashlib

PROFILE='mxfp4_nax_k16_ordered_f32_v1'
PINS={
    'source_prefill_nax.py':'29c1f7ec6f21228d180d42b02111e3385a8ee41b18da073dcc15e6bf680a3416',
    'source_prefill_legacy_tiles.py':'004b4f1a8980661dce95f646329fc64c28b8aa7512ca864790b7622b4fa4bb16',
}


@lru_cache(maxsize=1)
def verify_sources():
    root=Path(__file__).resolve().parent
    for name,expected in PINS.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Source NAX numerical-profile dependency drift: '+name)
    from .source_prefill_legacy_tiles import _installed
    _installed()


def plan(rows,*,numerical_profile):
    if numerical_profile!=PROFILE:
        raise ValueError('Explicit Source NAX numerical profile required')
    from .source_prefill_nax import plan as producer_plan
    p=producer_plan(rows)
    return dict(p,numerical_profile=PROFILE,outer_reduction='ascending F32 +0; safe compile',
                inner_reduction='native BF16 NAX K16, F32 accumulator; different from legacy K8',
                raw_shape=(rows,2304))


def project(x,expert,*,projection='gate',return_partials=False,numerical_profile):
    """Selected contiguous F32 rows; returns rounded or (rounded,raw,partials).

    Caller owns the GPU lease and original packed expert. Prepared inputs must
    be finite Source FP8-QAT values on the BF16 grid. This API does not read back
    their values. The bounded replay harness verifies those content contracts.
    No partial is BF16-rounded and no generic tensor sum is invoked.
    """
    import mlx.core as mx
    p=plan(x.shape[0],numerical_profile=numerical_profile)
    if type(return_partials) is not bool:
        raise ValueError('Source NAX diagnostic flag must be boolean')
    from .source_prefill_nax import prepare_inputs,_kernel
    from .source_prefill_legacy_tiles import _reduction_kernel
    from .source_dense_legacy import _device_supported
    rows=mx.arange(p['rows'],dtype=mx.uint32)
    _,offsets=prepare_inputs(x,expert,rows,projection)
    if not _device_supported():
        raise ValueError('Source NAX profile requires caller-owned M5 Max TF32-off GPU')
    verify_sources()
    partials=_kernel(p['rows'])(inputs=[x,rows,expert.backing,mx.array(offsets,mx.uint32)],
        grid=p['grid'],threadgroup=p['threadgroup'],output_shapes=[p['partial_shape']],
        output_dtypes=[mx.float32])[0]
    metadata=mx.array([p['partitions'],p['rows']*2304,2304],mx.int32)
    raw=_reduction_kernel()(inputs=[partials,metadata],grid=(2304,p['rows'],1),
        threadgroup=(256,1,1),output_shapes=[p['raw_shape']],output_dtypes=[mx.float32])[0]
    rounded=raw.astype(mx.bfloat16).astype(mx.float32)
    return (rounded,raw,partials) if return_partials else rounded


def batch_plan(row_counts,*,numerical_profile):
    """CPU geometry: preserve whole expert M/P; never dispatch from sum(M)."""
    if numerical_profile!=PROFILE:
        raise ValueError('Explicit Source NAX numerical profile required')
    from .source_prefill_legacy_tiles import batch_plan as legacy_plan
    p=legacy_plan(row_counts)
    p.update(numerical_profile=PROFILE,threadgroup=(64,1,1),
             threadgroups=(36,p['m_tiles'],p['partitions']*p['experts']),
             tile_config=(32,64),tile=(32,64,64),
             grid=(2304,p['m_tiles'],p['partitions']*p['experts']),
             threadgroup_bytes=64*(64+8)*2,input_bindings=p['experts']+3,
             profile='opt-in-native-K16-ordered-F32-different-dot-law')
    return p


def batch_kernel_spec(partitions,m_tiles,expert_count):
    """Uniform backing choice; each threadgroup owns one expert/tile/part."""
    if (type(partitions)is not int or type(m_tiles)is not int or type(expert_count)is not int
            or (partitions,m_tiles)not in ((5,1),(2,2),(2,3))or not 1<=expert_count<=15):
        raise ValueError('Source NAX batch unsupported compiled bucket')
    from .source_prefill_nax import _header
    kpart=5120//partitions
    cases='\n'.join(f'case {i}: backing=expert{i};break;'for i in range(expert_count))
    source=f'''
const uint3 physical=threadgroup_position_in_grid;
const uint slot=physical.z/{partitions},part=physical.z%{partitions};
const uint total_rows=descriptor[0];
const uint row_offset=descriptor[1+slot*4],original_m=descriptor[2+slot*4];
const uint qoffset=descriptor[3+slot*4],soffset=descriptor[4+slot*4];
const uint first=physical.y*32;
const uint4 desc=uint4(row_offset+first,0,min(uint(32),original_m-first),0);
const uint column=physical.x*64;
const device uchar* backing=nullptr;
switch(slot) {{
{cases}
}}
const device uchar* w=backing+qoffset;
omlx_gqmm::Mxfp4Q<32> q{{backing+soffset}};
threadgroup bfloat Ws[64*(64+8)];
omlx_gqmm::source_prefill_partition_seg<bfloat,omlx_gqmm::Mxfp4Q<32>,
    omlx_gqmm::Geo<32,64>,true,true,0,true,false>(
    gathered_x,rows_map,w,q,desc,column,partials+part*total_rows*2304,
    2304,5120,part*{kpart},{kpart},Ws,
    simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
'''
    return dict(header=_header(),source=source,
                input_names=['gathered_x','rows_map','descriptor']+[f'expert{i}'for i in range(expert_count)],
                output_names=['partials'],total_buffer_bindings=expert_count+4)


@lru_cache(maxsize=45)
def _batch_kernel(partitions,m_tiles,expert_count):
    import mlx.core as mx
    s=batch_kernel_spec(partitions,m_tiles,expert_count)
    return mx.fast.metal_kernel(name=f'mlx2_source_nax_batch_p{partitions}_t{m_tiles}_e{expert_count}',
        input_names=s['input_names'],output_names=s['output_names'],
        header=s['header'],source=s['source'],ensure_row_contiguous=True)


def prepare_batch_inputs(gathered_x,experts,row_counts,projection='gate',*,numerical_profile):
    from .source_prefill_legacy_tiles import prepare_batch_inputs as legacy_inputs
    counts=tuple(row_counts)
    p=batch_plan(counts,numerical_profile=numerical_profile)
    _,descriptor,experts=legacy_inputs(gathered_x,experts,counts,projection)
    return p,descriptor,experts


def batch_project(gathered_x,experts,row_counts,*,projection='gate',return_partials=False,numerical_profile):
    """At most15 original backing objects, one contiguous activation gather.

    Candidate dot arithmetic is explicitly profile-bound. The caller retains
    expert leases until completion. No coefficient repack or host readback.
    Raw output remains F32; separately rounded output uses the existing owner
    BF16 boundary, then the adapter applies its existing scored FP8 middle.
    """
    import mlx.core as mx
    from .source_prefill_legacy_tiles import _reduction_kernel
    from .source_dense_legacy import _device_supported
    if type(return_partials)is not bool:
        raise ValueError('Source NAX batch diagnostic flag must be boolean')
    p,descriptor,experts=prepare_batch_inputs(gathered_x,experts,row_counts,projection,numerical_profile=numerical_profile)
    if not _device_supported():
        raise ValueError('Source NAX batch requires caller-owned M5 Max TF32-off GPU')
    verify_sources()
    partials=_batch_kernel(p['partitions'],p['m_tiles'],p['experts'])(
        inputs=[gathered_x,mx.arange(p['rows'],dtype=mx.uint32),mx.array(descriptor,mx.uint32)]+[e.backing for e in experts],
        grid=p['grid'],threadgroup=p['threadgroup'],output_shapes=[p['partial_shape']],output_dtypes=[mx.float32])[0]
    metadata=mx.array([p['partitions'],p['partition_stride'],2304],mx.int32)
    raw=_reduction_kernel()(inputs=[partials,metadata],grid=(2304,p['rows'],1),threadgroup=(256,1,1),
        output_shapes=[p['raw_shape']],output_dtypes=[mx.float32])[0]
    rounded=raw.astype(mx.bfloat16).astype(mx.float32)
    return (rounded,raw,partials)if return_partials else rounded


def batch_compilation_source(partitions,m_tiles,expert_count):
    s=batch_kernel_spec(partitions,m_tiles,expert_count)
    buffers=['const device float* gathered_x','const device uint* rows_map','const device uint* descriptor']+[
        f'const device uchar* expert{i}'for i in range(expert_count)]+['device float* partials']
    declarations=',\n'.join(f'{d} [[buffer({i})]]'for i,d in enumerate(buffers))
    return ('#include "mlx/backend/metal/kernels/utils.h"\n'+s['header']
        +'\nkernel void nax_batch_probe('+declarations+''',
uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
'''+s['source']+'\n}\n')
