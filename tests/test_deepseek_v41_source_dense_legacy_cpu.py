"""CPU geometry and raw-law source checks, not Metal numerical evidence."""
from collections import Counter
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime.gguf import source_dense_legacy as dense


def array(shape,dtype):
    return SimpleNamespace(shape=shape,ndim=len(shape),dtype=dtype)


def inputs(rows):
    return array((1,rows,8,4096),mx.float32),array((8192,4096),mx.bfloat16)


@pytest.mark.parametrize('rows',(2,3,7,15,16,17,31,32,33,63,64,65,99,100,1024))
def test_plan_has_one_unsplit_k_chain_and_no_global_cast(rows):
    p=dense.plan(rows)
    assert p['k_step']==8 and p['k_steps']==512 and p['split_k']==1
    assert p['output_shape']==(1,rows,8,1024)
    assert p['grid']==(16*128,(rows+15)//16,8)
    assert p['global_f32_weight_copy_bytes']==p['threadgroup_scratch_bytes']==0
    assert p['coefficient_bytes']==67108864
    assert dense.supports(*inputs(rows),check_device=False)


@pytest.mark.parametrize('tile_m,tile_n,simdgroups',((8,32,2),(16,64,4),(32,128,4),(16,128,8)))
@pytest.mark.parametrize('live_rows',(2,7,16,31))
def test_lane_fragment_ownership_masks_tails_exactly(tile_m,tile_n,simdgroups,live_rows):
    p=dense.plan(live_rows,tile_m=tile_m,tile_n=tile_n,simdgroups=simdgroups)
    for m_base in range(0,live_rows,tile_m):
        owners=Counter()
        for sg in range(simdgroups):
            n_base=sg*8*p['n_fragments_per_simd']
            for lane in range(32):
                qid=lane//4
                fm=(qid&4)+(lane//2)%4
                fn=(qid&2)*2+(lane%2)*2
                for mf in range(p['m_fragments']):
                    row=m_base+mf*8+fm
                    for nf in range(p['n_fragments_per_simd']):
                        col=n_base+nf*8+fn
                        if row<live_rows:
                            owners[(row,col)]+=1
                            owners[(row,col+1)]+=1
        assert owners==Counter((row,col) for row in range(m_base,min(m_base+tile_m,live_rows)) for col in range(tile_n))


@pytest.mark.parametrize('rows,options',[(1,{}),(0,{}),(1025,{}),(True,{}),
    (7,{'tile_m':7}),(7,{'tile_n':48}),(7,{'simdgroups':1}),
    (7,{'tile_n':32,'simdgroups':8}),(7,{'tile_m':16.0})])
def test_unsupported_shapes_or_vector_reference_refused(rows,options):
    with pytest.raises(ValueError,match='Source legacy WoA'):
        dense.plan(rows,**options)


@pytest.mark.parametrize('change',('dtype','rank','batch','groups','width','weight_dtype','weight_shape'))
def test_bad_geometry_never_touches_device(change,monkeypatch):
    x,w=inputs(7)
    if change=='dtype':x.dtype=mx.bfloat16
    elif change=='rank':x.ndim=3
    elif change=='batch':x.shape=(2,7,8,4096)
    elif change=='groups':x.shape=(1,7,7,4096)
    elif change=='width':x.shape=(1,7,8,4095)
    elif change=='weight_dtype':w.dtype=mx.float32
    elif change=='weight_shape':w.shape=(8191,4096)
    monkeypatch.setattr(dense,'_device_supported',lambda:pytest.fail('bad shape touched GPU'))
    assert not dense.supports(x,w)
    with pytest.raises(ValueError,match='geometry'):
        dense.project(x,w)


def test_process_tf32_flag_is_an_explicit_admission_boundary(monkeypatch):
    from mlx2.runtime.gguf import source_dense
    from mlx2.process_env import TF32_ENV
    monkeypatch.setattr(source_dense,'device_supported',lambda:True)
    monkeypatch.setenv(TF32_ENV,'0')
    assert dense._device_supported()
    monkeypatch.setenv(TF32_ENV,'1')
    assert not dense._device_supported()


@pytest.mark.parametrize('rows',(2,7,99,100))
def test_generated_source_is_strict_f32_k8_with_register_bfloat_loads(rows):
    p=dense.kernel_spec(rows)
    assert p['input_names']==['x','weight'] and p['output_names']==['out']
    assert 'simdgroup_matrix<float,8,8> acc[MF][NF]' in p['source']
    assert 'k0<4096;k0+=8' in p['source']
    assert 'float(weight[' in p['source']
    assert 'simdgroup_multiply_accumulate(acc[m][serp],a[m],b[serp],acc[m][serp])' in p['source']
    assert 'row<M ? x[' in p['source'] and 'if(row<M)' in p['source']
    assert 'mpp::' not in p['source'] and 'NAX' not in p['header']
    assert 'threadgroup float' not in p['source']
    assert 'simdgroup_barrier(mem_flags::mem_none)' in p['source']


def test_dispatch_binds_original_bfloat_coefficients_without_copy(monkeypatch):
    x,w=inputs(99);seen=[];value=object()
    def kernel(**kwargs):
        seen.append(kwargs)
        return [value]
    monkeypatch.setattr(dense,'_device_supported',lambda:True)
    monkeypatch.setattr(dense,'_kernel',lambda *args:kernel)
    assert dense.project(x,w) is value
    assert seen[0]['inputs'][0] is x and seen[0]['inputs'][1] is w
    assert seen[0]['output_shapes']==[(1,99,8,1024)]
    assert seen[0]['output_dtypes']==[mx.float32]
