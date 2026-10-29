# SPDX-License-Identifier: MIT
"""Unwired paired ordered-reduction/Source-middle consumer for NAX partials.

Two existing NAX producers plus one consumer. No producer arithmetic, installed
profile or adapter selection changes. Provenance: deepseek-v41-source-nax-middle.
"""
from functools import lru_cache
from pathlib import Path
import hashlib

PINS={
 'source_prefill_nax_profile.py':'f01a8095b15aa804bdd0bed88578139599a7ded2a988a09839935ca1bb44779b',
 'source_prefill_legacy_tiles.py':'004b4f1a8980661dce95f646329fc64c28b8aa7512ca864790b7622b4fa4bb16',
 'source_gate_up_fusion.py':'4b57e3191bf2fef8558d709e1114d965916bb75131fa57e968586e52edec3501',
 'source_middle_fusion.py':'743b00fcf9af4b9e294d6b5b6482b687e0efcfe363d326bf614ad0f00104f9fd',
}


@lru_cache(maxsize=1)
def verify_sources():
    root=Path(__file__).resolve().parent
    for name,expected in PINS.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Unwired NAX middle dependency drift: '+name)
    from .source_prefill_nax_profile import verify_sources as verify_producer
    verify_producer()


def consumer_spec(exp_spelling,*,return_reduced=False):
    from .source_prefill_legacy_tiles import middle_consumer_spec
    if type(return_reduced)is not bool:raise ValueError('NAX middle diagnostic flag must be boolean')
    s=middle_consumer_spec(exp_spelling)
    assert s['header'].count('float g=0,u=0;')==1
    # Retain the epilogue's compilation mode. Volatile loads/stores prevent
    # fast-math constant folding of the +0 seed or fusing multiple additions.
    s['header']=s['header'].replace('float g=0,u=0;','volatile float g=0.0f,u=0.0f;')
    if return_reduced:
        needle='const float value=source_gu_weighted'
        s['source']=s['source'].replace(needle,'gate_raw[i]=reduced.x; up_raw[i]=reduced.y;\n'+needle)
        s['output_names']=['out','gate_raw','up_raw'];s['total_buffer_bindings']=9
    return s


@lru_cache(maxsize=4)
def _consumer(exp_spelling,return_reduced=False):
    import mlx.core as mx
    s=consumer_spec(exp_spelling,return_reduced=return_reduced)
    return mx.fast.metal_kernel(name='mlx2_source_nax_ordered_middle_'+('precise'if'precise'in exp_spelling else'fast')+('_raw'if return_reduced else''),
        input_names=s['input_names'],output_names=s['output_names'],header=s['header'],source=s['source'],ensure_row_contiguous=True)


def _score_geometry(rows,scores,slots):
    import mlx.core as mx
    if (scores.ndim!=1 or not 1<=scores.shape[0]<=6144 or scores.dtype!=mx.float32
            or slots.ndim!=1 or slots.shape!=(rows,) or slots.dtype!=mx.uint32):
        raise ValueError('NAX middle requires flat F32 scores/trusted U32 occurrence slots')


def reduce_middle(gate,up,scores,slots,*,exp_spelling,return_reduced=False):
    import mlx.core as mx
    from .source_dense_legacy import _device_supported
    consumer_spec(exp_spelling,return_reduced=return_reduced)
    if (gate.ndim!=3 or gate.shape!=up.shape or gate.shape[0]not in (2,5)
            or not 1<=gate.shape[1]<=1440 or gate.shape[2]!=2304
            or gate.dtype!=mx.float32 or up.dtype!=mx.float32):
        raise ValueError('NAX middle partial geometry/dtype mismatch')
    p,rows,n=gate.shape;_score_geometry(rows,scores,slots)
    if not _device_supported():raise ValueError('NAX middle requires caller-owned M5 Max TF32-off GPU')
    verify_sources();count=3 if return_reduced else 1
    result=_consumer(exp_spelling,return_reduced)(inputs=[gate,up,scores,slots,mx.array([1/448],mx.float32),mx.array([p,rows*n],mx.int32)],
        grid=(rows*n,1,1),threadgroup=(128,1,1),output_shapes=[(rows,n)]*count,output_dtypes=[mx.float32]*count)
    return tuple(result)if return_reduced else result[0]


def batch_middle(x,experts,row_counts,scores,slots,*,exp_spelling,numerical_profile,return_partials=False):
    """Borrow original expert backings; caller retains leases through completion."""
    import mlx.core as mx
    from . import source_prefill_nax_profile as native
    from .source_dense_legacy import _device_supported
    if type(return_partials)is not bool:raise ValueError('NAX middle diagnostic flag must be boolean')
    consumer_spec(exp_spelling)
    counts=tuple(row_counts);experts=tuple(experts)
    p,g,experts=native.prepare_batch_inputs(x,experts,counts,'gate',numerical_profile=numerical_profile)
    uplan,u,_=native.prepare_batch_inputs(x,experts,counts,'up',numerical_profile=numerical_profile)
    if p['partial_shape']!=uplan['partial_shape']:raise ValueError('NAX GU partial ownership mismatch')
    _score_geometry(p['rows'],scores,slots)
    if not _device_supported():raise ValueError('NAX middle requires caller-owned M5 Max TF32-off GPU')
    verify_sources();producer=native._batch_kernel(p['partitions'],p['m_tiles'],p['experts'])
    rowmap=mx.arange(p['rows'],dtype=mx.uint32)
    def produce(descriptor):
        return producer(inputs=[x,rowmap,mx.array(descriptor,mx.uint32)]+[e.backing for e in experts],
            grid=p['grid'],threadgroup=p['threadgroup'],output_shapes=[p['partial_shape']],output_dtypes=[mx.float32])[0]
    gate,up=produce(g),produce(u)
    out=reduce_middle(gate,up,scores,slots,exp_spelling=exp_spelling)
    return (out,gate,up)if return_partials else out


def compilation_source(exp_spelling,*,return_reduced=False):
    s=consumer_spec(exp_spelling,return_reduced=return_reduced)
    buffers=['const device float* gate_partials','const device float* up_partials','const device float* scores',
             'const device uint* slots','const device float* inverse','const device int* metadata','device float* out']
    if return_reduced:buffers+=['device float* gate_raw','device float* up_raw']
    declarations=',\n'.join(f'{d} [[buffer({i})]]'for i,d in enumerate(buffers))
    return s['header']+'\nkernel void nax_middle_probe('+declarations+',\nuint3 thread_position_in_grid [[thread_position_in_grid]]) {\n'+s['source']+'\n}\n'
