"""Bounded layout/boundary/source/offline checks; no Metal execution."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import shutil
import subprocess
import numpy as np
import mlx.core as mx
import pytest
from mlx2.runtime.models import deepseek_v41_output_chain as chain


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    previous=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('GPU kernel requested')))
    yield
    mx.set_default_device(previous)


@pytest.fixture
def helper():
    path=Path(__file__).resolve().parents[1]/'scripts/research/ds41_output_chain_components.py'
    spec=importlib.util.spec_from_file_location('source_output_chain_cpu',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.mark.parametrize('rows',[1,99])
def test_direct_output_permutation_bijection_and_group32_alignment(rows,helper):
    p=chain.layout(rows);i=np.arange(p['elements'],dtype=np.uint32)
    d=i%512;pos=(i//512)%rows;head=i//(512*rows)
    target=((pos*8+head//8)*8+head%8)*512+d
    assert np.array_equal(np.sort(target),i)
    assert np.array_equal(target.reshape(-1,32)%32,np.broadcast_to(np.arange(32),(len(i)//32,32)))
    values=i.reshape(p['input_shape']);packed=np.empty(len(i),dtype=np.uint32);packed[target]=i
    assert np.array_equal(helper.grouped_layout(values).ravel(),packed)
    assert p['prepared_wo_b_shape']==(1,rows,8192)


@pytest.mark.parametrize('rows',[0,2,7,100,True,-1])
def test_non_capture_geometry_refuses(rows):
    with pytest.raises(ValueError):chain.layout(rows)


def test_owner_boundaries_and_qat_body_are_preserved():
    from mlx2.runtime.models.deepseek_v41_quant import FUSED_SOURCE
    a=chain.source('wo_a')['source'];b=chain.source('wo_b')['source']
    assert 'float rotated=bf16_value(value[i]);'in a
    assert 'float a=bf16_value(value[base]),b=bf16_value(value[base+1]);'in a
    assert 'const float v=bf16_value(rotated);'in a
    assert 'out[destination]=bf16_value(quantized*scale);'in a
    assert b.replace('const float v=bf16_value(value[i]);','const float v=value[i];')==FUSED_SOURCE
    assert 'out[i]=quantized*scale;'in b and 'out[i]=bf16_value'not in b
    assert '#pragma clang fp contract(off)'in chain.source('wo_a')['header']


def test_unknown_stage_refuses():
    with pytest.raises(ValueError):chain.source('combined')


def test_no_gpu_gate_before_kernel_and_input_identity(monkeypatch):
    x=SimpleNamespace(ndim=4,dtype=mx.float32,shape=(1,64,1,512),size=32768)
    rope=SimpleNamespace(dims=64)
    with pytest.raises(ValueError,match='caller-owned'):chain.prepare_wo_a(x,rope,100)
    y=SimpleNamespace(ndim=4,dtype=mx.float32,shape=(1,1,8,1024),size=8192)
    with pytest.raises(ValueError,match='caller-owned'):chain.prepare_wo_b(y)
    monkeypatch.setattr(chain,'_gpu',lambda:None);seen={};output=object()
    def kernel(**kw):seen.update(kw);return[output]
    monkeypatch.setattr(chain,'_kernel',lambda kind:kernel)
    assert chain.prepare_wo_b(y)is output and seen['inputs'][0]is y
    assert seen['output_shapes']==[(1,1,8192)]


@pytest.mark.parametrize('kind',['wo_a','wo_b'])
def test_offline_metal_compile_only(kind,tmp_path):
    if not shutil.which('xcrun'):pytest.skip('OfflineMetal compiler missing')
    path=tmp_path/'output_chain.metal';path.write_text(chain.compilation_source(kind))
    result=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.2','-c',str(path),'-o',str(tmp_path/'output_chain.air')],capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stdout+result.stderr


def test_capture_inventory_phase_frontier_gates(helper):
    cases=[(dict(layer=l,phase=p,length=99 if p=='pp99'else 1,frontier=0 if p=='pp99'else 100),{})
        for l in (2,20,21)for p in ('pp99','decode1')]
    helper.case_inventory(cases)
    cases[-1][0]['frontier']=99
    with pytest.raises(ValueError,match='phase/frontier'):helper.case_inventory(cases)
    with pytest.raises(ValueError,match='six'):helper.case_inventory(cases[:-1])


@pytest.mark.parametrize('fence_fails',[False,True])
def test_failed_dequant_retains_coefficients_until_fence(helper,fence_fails):
    q,s,w=object(),object(),object();seen=[]
    def fail(*a):seen.append(('eval',a));raise RuntimeError('dequant failed')
    def fence():
        seen.append(('fence',q,s))
        if fence_fails:raise RuntimeError('fence failed')
    fake=SimpleNamespace(dequantize=lambda *a,**kw:w,eval=fail,synchronize=fence,bfloat16='bf16')
    with pytest.raises(RuntimeError,match='dequant failed'):helper._dequant_wo_a(fake,q,s)
    assert seen[-1]==('fence',q,s)
    assert bool(helper._DEQUANT_QUARANTINE)==fence_fails
    if fence_fails:assert helper._DEQUANT_QUARANTINE[0]==(q,s,w)


def test_component_scope_and_no_gpu_gate(helper,tmp_path):
    with pytest.raises(RuntimeError,match='caller-owned'):
        helper.run_components(SimpleNamespace(default_device=lambda:'cpu',gpu='gpu'),np,tmp_path,tmp_path,{},None,None)
    text=Path(helper.__file__).read_text()
    assert 'SourceV41Weights'not in text and 'load_dense'not in text
    assert "mx.einsum('btgi,goi->btgo',x.astype(mx.float32),f32_grouped_weight)"in text
    assert "mx.quantized_matmul(value,q,scales=s,mode='mxfp8')"in text
    assert "if check['exact']:"in text and "trajectory_scope='derived chain"in text


def test_followup_subset_default_is_source_bound(helper):
    repeats,variants=helper.component_options({})
    assert repeats==3 and len(variants)==3
    assert helper.component_options({'output_chain_options':dict(repeats=7,variants=['packed_inverse_QAT'])})==(7,[('packed_inverse_QAT',True,False)])


@pytest.mark.parametrize('options',[None,{'unknown':True},{'repeats':True},{'repeats':0},{'repeats':16},
    {'variants':[]},{'variants':['bad']},{'variants':['packed_inverse_QAT','packed_inverse_QAT']},{'variants':[[]]}])
def test_bad_subset_refuses_before_capture(helper,options):
    with pytest.raises(ValueError):helper.component_options({'output_chain_options':options})


@pytest.mark.parametrize('rows,policy,accepted,expect_direct',[(1,'woa_f32',True,True),
    (1,'direct_f32',True,True),(1,'woa_f32',False,False),(1,'reference',True,False),(99,'woa_f32',True,False)])
def test_captured_grouped_policy_and_direct_refusal_match_provider(helper,rows,policy,accepted,expect_direct):
    value=SimpleNamespace(shape=(1,rows,8,4096));value.astype=lambda dtype:value
    bf16,f32,direct,fallback=object(),object(),object(),object();seen=[]
    def project(x,w,kind):
        seen.append((x,w,kind));return direct if accepted else None
    fake=SimpleNamespace(float32='f32',einsum=lambda eq,x,w:fallback)
    assert helper.owning_grouped(fake,value,bf16,f32,policy,project)is(direct if expect_direct else fallback)
    assert bool(seen)==(rows==1 and policy in ('woa_f32','direct_f32'))
    if seen:assert seen[0]==(value,bf16,'wo_a')
