"""Small-M paired NAX plans/ownership/source and offline Metal; no GPU."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
import mlx.core as mx
from mlx2.runtime.gguf import source_small_m_nax_pair as m
from mlx2.runtime.gguf import source_dense_legacy
from test_deepseek_v41_source_prefill_legacy_middle_cpu import packed

@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:pytest.fail('GPU forbidden'))
    yield
    mx.set_default_device(old)

@pytest.mark.parametrize('p',[1,5])
@pytest.mark.parametrize('counts',[(8,),(10,),(12,),(8,10,12),(8,10,12)*5])
def test_plan(p,counts):
    z=m.plan(counts,numerical_profile=f'small_m_nax_p{p}_ordered_f32_v1')
    assert z['partial_shape']==(p,sum(counts),2304)
    assert z['threadgroups']==(36,1,p*len(counts))and z['threadgroup']==(64,1,1)
    assert z['threadgroup_bytes']==18432 and z['total_buffer_bindings']<=20
    assert z['paired_partial_bytes']==2*p*sum(counts)*2304*4
    assert z['explicit_activation_gather_bytes']==0 and z['coefficient_repack_bytes']==0

@pytest.mark.parametrize('counts,profile',[((7,),'small_m_nax_p1_ordered_f32_v1'),((13,),'small_m_nax_p5_ordered_f32_v1'),((True,),'small_m_nax_p1_ordered_f32_v1'),((8,)*16,'small_m_nax_p1_ordered_f32_v1'),((),'small_m_nax_p1_ordered_f32_v1'),((8,),'reference')])
def test_plan_rejects(counts,profile):
    with pytest.raises(ValueError):m.plan(counts,numerical_profile=profile)

@pytest.mark.parametrize('bad',['dtype','length','range','noncontiguous','source'])
def test_cpu_map_rejected(bad):
    a=np.arange(8,dtype=np.uint32);source=99
    if bad=='dtype':a=a.astype(np.int32)
    elif bad=='length':a=a[:7]
    elif bad=='range':a[-1]=99
    elif bad=='noncontiguous':a=np.arange(16,dtype=np.uint32)[::2]
    else:source=0
    with pytest.raises(ValueError):m.validate_row_map((8,),a,source)

@pytest.mark.parametrize('p',[1,5])
def test_pair_reuses_same_A_fragment_and_independent_accumulators(p):
    s=m.producer_spec(p,3)['source']
    assert s.count('load_a_map<bfloat,1>')==1 and s.count('BaseNAXFrag::mma(')==2
    assert 'D.frag_at(0,0)'in s and'U.frag_at(0,0)'in s
    assert s.count('Arow.frag_at(0,kk),')==2 and'virtual'not in s
    assert 'GW[64*72]'in s and'UW[64*72]'in s
    assert 'part*total*2304+row_base*2304+column+sg*32'in s
    for projection in ('gate','up'):
        q=m.producer_spec(p,3,projection=projection)['source'];assert q.count('BaseNAXFrag::mma(')==1


def test_prepare_keeps_backings_and_original_mapped_input(monkeypatch):
    monkeypatch.setattr(m,'verify_sources',lambda:None);monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:True)
    x=mx.zeros((99,5120));e=[packed(),packed()];rows=np.arange(18,dtype=np.uint32)[::-1].copy()
    p,ins=m.prepare(x,e,(8,10),rows,numerical_profile='small_m_nax_p1_ordered_f32_v1')
    assert ins[0]is x and ins[3]is e[0].backing and ins[4]is e[1].backing
    assert np.array_equal(np.asarray(ins[1]),rows)
    d=np.asarray(ins[2]);assert d[0]==18 and tuple(d[1:3])==(0,8)and tuple(d[7:9])==(8,10)


def test_one_producer_one_consumer_and_no_weight_or_input_concat(monkeypatch):
    monkeypatch.setattr(m,'verify_sources',lambda:None);monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:True)
    monkeypatch.setattr(mx,'concatenate',lambda *a,**k:pytest.fail('No coefficient/activation concatenation'))
    calls=[];gp=mx.zeros((5,8,2304));up=mx.ones((5,8,2304));sentinel=object()
    monkeypatch.setattr(m,'_producer',lambda *a:(lambda **kw:(calls.append(kw)or[gp,up])))
    monkeypatch.setattr(m,'reduce_middle',lambda g,u,*a,**k:sentinel if g is gp and u is up else None)
    result=m.middle(mx.zeros((99,5120)),[packed()],(8,),np.arange(8,dtype=np.uint32),mx.ones((8,)),mx.arange(8,dtype=mx.uint32),numerical_profile='small_m_nax_p5_ordered_f32_v1',exp_spelling='metal::exp',return_partials=True)
    assert result==(sentinel,gp,up)and len(calls)==1

@pytest.mark.parametrize('p',[1,5])
def test_reducer_metadata_keeps_pluszero_seed_and_owning_epilogue(monkeypatch,p):
    from mlx2.runtime.gguf import source_prefill_nax_middle as f
    from mlx2.runtime.gguf import source_gate_up_fusion as gu
    monkeypatch.setattr(m,'verify_sources',lambda:None);monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:True)
    calls=[];value=object();monkeypatch.setattr(f,'_consumer',lambda *a:(lambda **kw:(calls.append(kw)or[value])))
    g=mx.zeros((p,8,2304));assert m.reduce_middle(g,g,mx.ones((8,)),mx.arange(8,dtype=mx.uint32),exp_spelling='metal::exp')is value
    assert np.array_equal(np.asarray(calls[0]['inputs'][-1]),[p,8*2304])
    s=f.consumer_spec('metal::exp');assert 'volatile float g=0.0f,u=0.0f;'in s['header']
    assert gu._EPILOGUE.replace('__EXP__','metal::exp')in s['header']


def test_no_unowned_launch_and_pins():
    with pytest.raises(ValueError,match='caller-owned'):
        m.prepare(mx.zeros((99,5120)),[packed()],(8,),np.arange(8,dtype=np.uint32),numerical_profile='small_m_nax_p1_ordered_f32_v1')
    m.verify_sources()

@pytest.mark.parametrize('p,e,projection',[(1,1,'paired'),(5,15,'paired'),(1,3,'gate'),(5,3,'up')])
def test_offline_metal4_true16row_fragment(p,e,projection,tmp_path):
    if not shutil.which('xcrun'):pytest.skip('Metal compiler unavailable')
    path=tmp_path/'small.metal';path.write_text(m.compilation_source(p,e,projection=projection))
    cmd=['xcrun','-sdk','macosx','metal','-std=metal4.0','-I',str(Path(mx.__file__).parent/'include'),'-c',str(path),'-o',str(tmp_path/'small.air')]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=45);assert r.returncode==0,r.stdout+r.stderr
