"""Attention geometry/partition/sink/source/offline tests, no GPU execution."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import subprocess
import shutil
import numpy as np
import mlx.core as mx
import pytest
from mlx2.runtime.models import deepseek_v41_head_group_attention as p


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    def forbidden(*a,**kw):raise AssertionError('CPU test requested Metal execution')
    monkeypatch.setattr(mx.fast,'metal_kernel',forbidden)
    yield
    mx.set_default_device(old)


@pytest.fixture
def helper(monkeypatch):
    directory=Path(__file__).resolve().parents[1]/'scripts/research';monkeypatch.syspath_prepend(str(directory))
    path=directory/'ds41_head_group_components.py';spec=importlib.util.spec_from_file_location('headgroup_components_cpu',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.mark.parametrize('length',[1,99])
@pytest.mark.parametrize('head_group,splits',p.VARIANTS)
def test_grid_and_disjoint_KV_partition_ownership(length,head_group,splits):
    d=p.plan(length,101,50,50,head_group=head_group,kv_splits=splits)
    assert d['threadgroup_bytes']==16448<32768
    assert d['threadgroup']==(32*head_group,1,1)
    assert d['threadgroups']==(64//head_group,length,splits)
    assert sorted(g*head_group+s for g in range(64//head_group)for s in range(head_group))==list(range(64))
    total=151;owned=[i for part in range(splits)for i in range(total*part//splits,total*(part+1)//splits)]
    assert owned==list(range(total))
    assert d['logical_all_slot_KV_load_bytes']==(64//head_group)*length*151*512*4
    assert d['logical_Q_load_bytes']==64*length*512*4*splits
    assert d['sink_contributions_per_head_query']==1
    assert d['partial_bytes']<=4*64*99*514*4


@pytest.mark.parametrize('kwargs',[dict(length=2),dict(length=True),dict(local_rows=0),dict(local_rows=129),
    dict(pool_rows=0),dict(pool_rows=513),dict(selected_rows=0),dict(selected_rows=513),dict(head_group=2),dict(kv_splits=2),dict(head_group=1,kv_splits=4)])
def test_unreviewed_geometry_refuses(kwargs):
    args=dict(length=1,local_rows=101,pool_rows=50,selected_rows=50);args.update(kwargs)
    with pytest.raises(ValueError):p.plan(**args)


def fixture():
    rng=np.random.default_rng(841)
    return [rng.normal(0,.2,(1,64,1,512)).astype(np.float32),rng.normal(0,.2,(1,1,4,512)).astype(np.float32),
        np.array([[[[True,False,True,False]]]]),rng.normal(0,.2,(1,4,512)).astype(np.float32),
        np.array([[[3,0,2,1]]],np.int32),np.array([[[False,True,False,False]]]),rng.normal(0,.3,64).astype(np.float32)]


def split_reference(arrays):
    q,local,lm,pool,ids,valid,sinks=arrays;values=np.concatenate((local[0,0],pool[0,ids[0,0]]),0).astype(np.float64)
    mask=np.concatenate((lm[0,0,0],valid[0,0]));query=(q[0,:,0]*np.float32(512**-.5)).astype(np.float64)
    scores=query@values.T;partials=[]
    for s in range(4):
        subset=np.arange(8*s//4,8*(s+1)//4);subset=subset[mask[subset]]
        if not len(subset):partials.append((np.full(64,-np.inf),np.zeros(64),np.zeros((64,512))));continue
        selected=scores[:,subset];m=selected.max(-1);w=np.exp(selected-m[:,None])
        partials.append((m,w.sum(-1),w@values[subset]))
    maximum=np.maximum(np.maximum.reduce([a for a,b,c in partials]),sinks)
    denominator=np.exp(sinks-maximum);numerator=np.zeros((64,512))
    for m,l,a in partials:
        alpha=np.exp(m-maximum);denominator+=l*alpha;numerator+=a*alpha[:,None]
    return(numerator/denominator[:,None]).astype(np.float32),partials


def test_empty_chunks_stable_merge_matches_independent_dense_oracle():
    arrays=fixture();actual,parts=split_reference(arrays)
    assert any(np.all(l==0)for m,l,a in parts)
    np.testing.assert_allclose(actual,p.reference_numpy(*arrays)[0,:,0],rtol=1e-6,atol=1e-7)


def test_sink_mass_once_and_all_masked_reference():
    arrays=fixture();arrays[0].fill(0);arrays[1].fill(1);arrays[2].fill(False);arrays[2][...,0]=True
    arrays[5].fill(False);arrays[6].fill(0)
    output,parts=split_reference(arrays)
    assert np.all(output==np.float32(.5)) # one key mass1 plus one sink mass1
    arrays[2].fill(False);arrays[6]=np.linspace(-1000,1000,64,dtype=np.float32)
    assert np.isfinite(p.reference_numpy(*arrays)).all()and not p.reference_numpy(*arrays).any()


def test_duplicate_selected_slots_are_not_silently_deduplicated():
    arrays=fixture();arrays[0].fill(0);arrays[2].fill(False);arrays[3].fill(1);arrays[4].fill(0)
    arrays[5].fill(False);arrays[5][...,:2]=True;arrays[6].fill(0)
    np.testing.assert_array_equal(p.reference_numpy(*arrays),np.full(arrays[0].shape,np.float32(2/3)))
    arrays[4][...,0]=-1
    with pytest.raises(ValueError,match='invalid'):p.reference_numpy(*arrays)


def test_prescale_mask_and_sink_source_contracts():
    assert '*0.04419417382415922f' in p.PRODUCER
    assert 'score=simd_sum(score)'in p.PRODUCER and 'qv[j]*tile' in p.PRODUCER
    assert 'sinks['not in p.PRODUCER[:p.PRODUCER.index('if(SPLITS==1)')]
    assert 'float denominator=exp(sinks[head]-maximum)'in p.MERGE
    assert 'index<0 || index>=long(POOL_ROWS)'in p.PRODUCER
    assert '#pragma clang fp contract(off)'in p.HEADER
    assert 'NAN' in p.PRODUCER and 'if(denominator'not in p.PRODUCER


def test_geometry_no_gpu_and_immutable_backings(monkeypatch):
    arrays=fixture();values=[SimpleNamespace(ndim=a.ndim,shape=a.shape,dtype=mx.float32)for a in arrays]
    values[2].dtype=values[5].dtype=mx.bool_;values[4].dtype=mx.int32
    assert p.geometry(*values)['head_group']==4
    with pytest.raises(ValueError,match='caller-owned'):p.attention(*values)
    values[1].shape=(1,4,4,512)
    with pytest.raises(ValueError,match='single local'):p.geometry(*values)


@pytest.mark.parametrize('h,s,index_type',[(h,s,t)for h,s in p.VARIANTS for t in ('int','uint')])
def test_offline_producer_compile(h,s,index_type,tmp_path):
    if not shutil.which('xcrun'):pytest.skip('Metal compiler not installed')
    path=tmp_path/'headgroup.metal';path.write_text(p.compilation_source(h,s,index_type=index_type))
    run=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.2','-c',str(path),'-o',str(tmp_path/'headgroup.air')],capture_output=True,text=True,timeout=45)
    assert run.returncode==0,run.stdout+run.stderr


def test_offline_merge_compile(tmp_path):
    if not shutil.which('xcrun'):pytest.skip('Metal compiler not installed')
    path=tmp_path/'merge.metal';path.write_text(p.compilation_source(4,4,kind='merge'))
    run=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.2','-c',str(path),'-o',str(tmp_path/'merge.air')],capture_output=True,text=True,timeout=45)
    assert run.returncode==0,run.stdout+run.stderr


def test_helper_synthetic_stability_and_no_gpu(helper,tmp_path):
    for name,args in helper.synthetic_stability(np):
        output=p.reference_numpy(*args);assert np.isfinite(output).all()
        if name.startswith('all_masked'):assert not output.any()
    with pytest.raises(RuntimeError,match='caller-owned'):
        helper.run_components(SimpleNamespace(default_device=lambda:'cpu',gpu='gpu'),np,tmp_path,tmp_path,{},None,None)
    source=Path(helper.__file__).read_text()
    assert source.index("section['required_reference_controls_exact']=True")<source.index('for name,host in synthetic_stability(np):')
    assert 'quality_verdict=\'not evaluated\''in source


def test_metrics_signed_zero_finite_and_nonfinite_diagnostics(helper):
    a=np.zeros((1,1,1,2),np.float32);b=a.copy();b[...,0]=-0.0
    d=helper.numerical_metrics(np,a,b);assert d['all_finite']and d['changed_words']==1 and d['rmse']==0
    b[...,1]=np.nan;assert not helper.numerical_metrics(np,a,b)['all_finite']


def test_followup_only_changes_timing_subset_not_numerical_inventory(helper):
    assert helper.component_options({})==(3,['h1_s1','h4_s1','h4_s4'])
    assert helper.component_options({'head_group_options':dict(repeats=7,timing_variants=['h4_s1','h4_s4'])})==(7,['h4_s1','h4_s4'])
    source=Path(helper.__file__).read_text()
    assert 'for h,s in candidate.VARIANTS:'in source
    assert "row['head_group_only_raw_consistency']"in source
    assert 'if admitted and name in timing_variants:'in source


@pytest.mark.parametrize('options',[None,{'unknown':True},{'repeats':True},{'repeats':2},{'repeats':8},
    {'timing_variants':[]},{'timing_variants':['h8_s1']},{'timing_variants':['h4_s1','h4_s1']},
    {'timing_variants':[[]]},{'timing_variants':'h4_s1'}])
def test_bad_followup_options_refuse(helper,options):
    with pytest.raises(ValueError):helper.component_options({'head_group_options':options})
