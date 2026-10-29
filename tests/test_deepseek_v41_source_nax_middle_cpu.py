"""Unwired NAX producer reuse and ordered middle consumer; CPU/offline only."""
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
import pytest
from mlx2.runtime.gguf import source_prefill_nax_middle as fused
from mlx2.runtime.gguf import source_prefill_nax_profile as native
from mlx2.runtime.gguf import source_prefill_legacy_tiles as legacy
from mlx2.runtime.gguf import source_gate_up_fusion as gu
from mlx2.runtime.gguf import source_dense_legacy
from test_deepseek_v41_source_prefill_legacy_middle_cpu import packed


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:pytest.fail('GPU forbidden'))
    yield
    mx.set_default_device(old)


@pytest.mark.parametrize('spelling',gu.EXP_SPELLINGS)
@pytest.mark.parametrize('diagnostic',[False,True])
def test_epilogue_and_flags_unchanged_ordered_seed_volatile(spelling,diagnostic):
    s=fused.consumer_spec(spelling,return_reduced=diagnostic)
    assert gu._EPILOGUE.replace('__EXP__',spelling)in s['header']
    assert 'volatile float g=0.0f,u=0.0f;'in s['header']
    assert '#pragma clang fp reassociate(off)'in s['header'] and '#pragma clang fp contract(off)'in s['header']
    assert 'g+=gate[offset];'in s['header']and'u+=up[offset];'in s['header']
    assert 'offset+=stride;'in s['header']and'#pragma clang loop unroll(disable)'in s['header']
    assert s['compile_options']is None
    assert len(s['output_names'])==(3 if diagnostic else 1)
    assert 'scores[slots[i/2304]]'in s['source']


@pytest.mark.parametrize('counts',[(13,16,32),(33,34,64),(65,96),(96,)*15])
def test_exact_two_existing_producers_one_consumer_no_extra_reducers(counts,monkeypatch):
    p=native.batch_plan(counts,numerical_profile=native.PROFILE)
    experts=[packed()for _ in counts];x=mx.zeros((sum(counts),5120));scores=mx.ones((sum(counts),));slots=mx.arange(sum(counts),dtype=mx.uint32)
    gp=mx.zeros(p['partial_shape']);up=mx.ones(p['partial_shape']);out=mx.zeros(p['raw_shape']);calls=[];seen=[]
    def producer(**kw):calls.append(kw);return[gp if len(calls)==1 else up]
    monkeypatch.setattr(native,'_batch_kernel',lambda *a:producer)
    monkeypatch.setattr(fused,'verify_sources',lambda:None)
    monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:True)
    def consumer(*a,**kw):seen.append((a,kw));return out
    monkeypatch.setattr(fused,'reduce_middle',consumer)
    monkeypatch.setattr(native,'batch_project',lambda *a,**k:pytest.fail('extra reducer'))
    monkeypatch.setattr(legacy,'_reduction_kernel',lambda *a:pytest.fail('extra reducer'))
    monkeypatch.setattr(mx,'concatenate',lambda *a,**k:pytest.fail('weight repack'))
    result=fused.batch_middle(x,experts,counts,scores,slots,exp_spelling='metal::exp',numerical_profile=native.PROFILE,return_partials=True)
    assert result==(out,gp,up)and len(calls)==2 and len(seen)==1
    for c in calls:
        assert c['inputs'][0]is x and c['grid']==p['grid']and c['threadgroup']==p['threadgroup']
        assert all(a is e.backing for a,e in zip(c['inputs'][3:],experts))
        assert np.array_equal(np.asarray(c['inputs'][1]),np.arange(sum(counts),dtype=np.uint32))
    assert seen[0][0][:2]==(gp,up)and seen[0][0][2]is scores and seen[0][0][3]is slots


@pytest.mark.parametrize('bad',['shape','partitions','dtype','slots','scores','spelling','diagnostic'])
def test_invalid_partial_contract_before_dispatch(monkeypatch,bad):
    g=mx.zeros((2,3,2304));u=g;s=mx.ones((3,));ids=mx.arange(3,dtype=mx.uint32);kw={'exp_spelling':'metal::exp'}
    if bad=='shape':u=mx.zeros((2,2,2304))
    elif bad=='partitions':g=u=mx.zeros((3,3,2304))
    elif bad=='dtype':g=g.astype(mx.bfloat16)
    elif bad=='slots':ids=ids.astype(mx.int32)
    elif bad=='scores':s=mx.ones((0,))
    elif bad=='spelling':kw['exp_spelling']='exp'
    else:kw['return_reduced']='yes'
    with pytest.raises(ValueError):fused.reduce_middle(g,u,s,ids,**kw)


def test_metadata_and_output_ownership(monkeypatch):
    g=mx.zeros((5,8,2304));u=mx.ones((5,8,2304));s=mx.ones((10,));ids=mx.arange(8,dtype=mx.uint32);sentinel=object();seen=[]
    monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:True);monkeypatch.setattr(fused,'verify_sources',lambda:None)
    monkeypatch.setattr(fused,'_consumer',lambda *a:(lambda **kw:(seen.append(kw)or[sentinel])))
    assert fused.reduce_middle(g,u,s,ids,exp_spelling='metal::exp')is sentinel
    kw=seen[0];assert kw['inputs'][:4]==[g,u,s,ids]
    assert np.array_equal(np.asarray(kw['inputs'][5]),[5,8*2304])
    assert kw['grid']==(8*2304,1,1) and kw['output_shapes']==[(8,2304)]


def test_unowned_CPU_rejected_and_pins_unchanged():
    with pytest.raises(ValueError,match='caller-owned'):
        fused.reduce_middle(mx.zeros((2,1,2304)),mx.zeros((2,1,2304)),mx.ones((1,)),mx.zeros((1,),mx.uint32),exp_spelling='metal::exp')
    fused.verify_sources()


@pytest.mark.parametrize('spelling',gu.EXP_SPELLINGS)
@pytest.mark.parametrize('diagnostic',[False,True])
def test_offline_default_mode_consumer_compiles(spelling,diagnostic,tmp_path):
    if not shutil.which('xcrun'):pytest.skip('Metal compiler unavailable')
    p=tmp_path/'consumer.metal';p.write_text(fused.compilation_source(spelling,return_reduced=diagnostic))
    result=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.2','-c',str(p),'-o',str(tmp_path/'consumer.air')],capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
