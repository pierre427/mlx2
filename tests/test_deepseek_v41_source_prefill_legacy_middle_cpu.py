"""Ordered Source middle consumer structure/ownership; CPU and offline only."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess
import numpy as np
import mlx.core as mx
import pytest
from mlx2.runtime.gguf import source_prefill_legacy_tiles as tile
from mlx2.runtime.gguf import source_gate_up_fusion as gu
from mlx2.adapters.deepseek_v41_source_buffers import PackedExpert,canonical_layout


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    def forbidden(*a,**kw):raise AssertionError('CPU middle test attempted GPU')
    monkeypatch.setattr(mx.fast,'metal_kernel',forbidden)
    yield
    mx.set_default_device(old)


def packed():
    layout=canonical_layout('original');types={'<u4':mx.uint32,'|u1':mx.uint8}
    fields={t.name:SimpleNamespace(shape=t.shape,dtype=types[t.dtype])for t in layout.tensors}
    return PackedExpert(SimpleNamespace(ndim=1,shape=(layout.nbytes,),dtype=mx.uint8),layout,fields)


@pytest.mark.parametrize('spelling',gu.EXP_SPELLINGS)
def test_exact_existing_epilogue_text_and_two_ordered_f32_chains(spelling):
    s=tile.middle_consumer_spec(spelling)
    assert gu._EPILOGUE.replace('__EXP__',spelling)in s['header']
    assert '#pragma clang fp reassociate(off)'in s['header']
    assert 'float g=0,u=0;'in s['header']
    assert 'g+=gate[offset];'in s['header'] and 'u+=up[offset];'in s['header']
    assert s['header'].index('g+=gate[offset];')<s['header'].index('offset+=stride;')
    assert '#pragma clang loop unroll(disable)'in s['header']
    assert s['compile_options']is None  # Existing epilogue compile mode retained.
    assert 'scores[slots[i/2304]]'in s['source']
    assert 'simd_shuffle_xor(maximum,step)'in s['source']
    assert s['output_names']==['out'] and s['total_buffer_bindings']==7
    assert 'threadgroup_barrier'not in s['source']
    assert s['threadgroup']==(128,1,1)


@pytest.mark.parametrize('spelling',['exp','metal::fast::exp','metal::native::exp',None,False])
def test_unknown_served_spelling_refused(spelling):
    with pytest.raises(ValueError):tile.middle_consumer_spec(spelling)


@pytest.mark.parametrize('counts',[(13,32),(33,48,64),(65,96),(96,)*15])
def test_partial_bounds_and_matching_gate_up_descriptor(counts):
    e=[packed()for _ in counts]
    x=SimpleNamespace(ndim=2,shape=(sum(counts),5120),dtype=mx.float32)
    scores=SimpleNamespace(ndim=1,shape=(6144,),dtype=mx.float32)
    slots=SimpleNamespace(ndim=1,shape=(sum(counts),),dtype=mx.uint32)
    p,g,u,retained=tile.prepare_middle_inputs(x,e,counts,scores,slots)
    assert p['partial_bytes']*2<=53_084_160
    assert p['rows']*2304%128==0  # Full SIMD32 groups, no cross-row maxima.
    assert g[0]==u[0]==sum(counts)
    for i,expert in enumerate(e):
        assert retained[i]is expert
        assert tuple(g[1+4*i:3+4*i])==tuple(u[1+4*i:3+4*i])
        assert g[3+4*i]==expert.layout.tensor('gate.qweight').offset
        assert u[3+4*i]==expert.layout.tensor('up.qweight').offset


@pytest.mark.parametrize('bad',['scoresndim','scoresdtype','scoresempty','scorestoo_many','slotsndim','slotsdtype','slotssize'])
def test_scores_slots_structural_refusal_before_dispatch(bad):
    e=[packed()];x=SimpleNamespace(ndim=2,shape=(13,5120),dtype=mx.float32)
    scores=SimpleNamespace(ndim=1,shape=(99,),dtype=mx.float32)
    slots=SimpleNamespace(ndim=1,shape=(13,),dtype=mx.uint32)
    if bad=='scoresndim':scores.ndim=2
    elif bad=='scoresdtype':scores.dtype=mx.float16
    elif bad=='scoresempty':scores.shape=(0,)
    elif bad=='scorestoo_many':scores.shape=(6145,)
    elif bad=='slotsndim':slots.ndim=2
    elif bad=='slotsdtype':slots.dtype=mx.int32
    else:slots.shape=(12,)
    with pytest.raises(ValueError):tile.prepare_middle_inputs(x,e,(13,),scores,slots)


@pytest.mark.parametrize('geometry',[(13,32),(33,64),(65,96)])
@pytest.mark.parametrize('candidate_tile',[(32,32),(16,64)])
def test_two_partial_producers_one_consumer_no_gu_or_score_materialization(geometry,candidate_tile,monkeypatch):
    counts=geometry;experts=[packed(),packed()];p=tile.batch_plan(counts,tile=candidate_tile)
    x=mx.zeros((sum(counts),5120));scores=mx.arange(97,dtype=mx.float32)
    slots=mx.array((np.arange(sum(counts),dtype=np.uint32)*7)%97)
    gp=mx.zeros(p['partial_shape']);up=mx.ones(p['partial_shape']);output=mx.full(p['raw_shape'],.125)
    calls=[];seen={}
    def producer(**kw):calls.append(kw);return[gp if len(calls)==1 else up]
    def select(*args):seen['producer_args']=args;return producer
    def consumer(**kw):seen['consumer']=kw;return[output]
    def forbidden(*a,**kw):raise AssertionError('Fusion materialized redundant GU/score data')
    monkeypatch.setattr(tile,'_batch_kernel',select)
    monkeypatch.setattr(tile,'_middle_consumer_kernel',lambda spelling:consumer)
    monkeypatch.setattr(tile,'_reduction_kernel',forbidden)
    monkeypatch.setattr(tile,'batch_project',forbidden)
    monkeypatch.setattr(mx,'concatenate',forbidden)
    monkeypatch.setattr(mx,'take',forbidden)
    from mlx2.runtime.gguf import source_dense_legacy
    monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:True)
    result=tile.batch_middle(x,experts,counts,scores,slots,exp_spelling='metal::exp',tile=candidate_tile)
    assert result is output and len(calls)==2
    assert all(call['inputs'][0]is x for call in calls)
    assert all(backing is expert.backing for call in calls for backing,expert in zip(call['inputs'][2:],experts))
    c=seen['consumer'];assert c['inputs'][0]is gp and c['inputs'][1]is up
    assert c['inputs'][2]is scores and c['inputs'][3]is slots
    assert np.array_equal(np.asarray(c['inputs'][5]),[p['partitions'],p['partition_stride']])
    assert np.asarray(c['inputs'][4])[0]==np.float32(1/448)
    assert c['grid']==(sum(counts)*2304,1,1)and c['threadgroup']==(128,1,1)
    assert c['output_shapes']==[p['raw_shape']]and c['output_dtypes']==[mx.float32]


@pytest.mark.parametrize('stage',['gate','up','consumer'])
def test_failure_leaves_caller_backings_owned_without_unrelated_dispatch(stage,monkeypatch):
    e=packed();x=mx.zeros((13,5120));scores=mx.ones((20,));slots=mx.arange(13,dtype=mx.uint32)
    partials=mx.zeros((5,13,2304));calls=[]
    def producer(**kw):
        calls.append(kw)
        if stage==('gate'if len(calls)==1 else'up'):raise RuntimeError('injected '+stage)
        return[partials]
    def consumer(**kw):
        assert len(calls)==2
        raise RuntimeError('injected consumer')
    monkeypatch.setattr(tile,'_batch_kernel',lambda *a:producer)
    monkeypatch.setattr(tile,'_middle_consumer_kernel',lambda spelling:consumer)
    from mlx2.runtime.gguf import source_dense_legacy
    monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:True)
    with pytest.raises(RuntimeError,match='injected'):
        tile.batch_middle(x,[e],(13,),scores,slots,exp_spelling='metal::exp')
    assert len(calls)==(1 if stage=='gate' else 2)
    assert all(c['inputs'][2]is e.backing for c in calls)


def test_cpu_refused_before_partial_production():
    with pytest.raises(ValueError,match='caller-owned'):
        tile.batch_middle(mx.zeros((13,5120)),[packed()],(13,),mx.ones((13,)),
            mx.arange(13,dtype=mx.uint32),exp_spelling='metal::exp')


@pytest.mark.parametrize('spelling',gu.EXP_SPELLINGS)
def test_offline_consumer_compiles_without_gpu(spelling,tmp_path):
    if not shutil.which('xcrun'):pytest.skip('Metal compiler unavailable')
    source=tmp_path/'middle.metal';source.write_text(tile.middle_compilation_source(spelling))
    result=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.2','-c',str(source),'-o',str(tmp_path/'middle.air')],capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
