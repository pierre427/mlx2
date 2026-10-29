"""CPU/mock and offline Metal compilation only; no model or GPU execution."""
from pathlib import Path
import importlib.util
import hashlib
import json
import shutil
import subprocess
import sys
import types
import numpy as np
import mlx.core as mx
import pytest
from mlx2.runtime.gguf import source_prefill_nax_profile as p
from mlx2.runtime.gguf import source_prefill_nax as producer
from mlx2.runtime.gguf import source_prefill_legacy_tiles as reducer
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=load('nax_profile_cpu_fixture',ROOT/'tests/test_deepseek_v41_source_prefill_nax_cpu.py')
h=load('nax_profile_component_fixture',ROOT/'scripts/research/ds41_prefill_nax_profile_components.py')
@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
 old=mx.default_device();mx.set_default_device(mx.cpu)
 monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:(_ for _ in ()).throw(AssertionError('GPU forbidden')))
 yield
 mx.set_default_device(old)
@pytest.mark.parametrize('m',[13,16,32,33,34,64,96])
def test_explicit_profile_preserves_partition_domain(m):
 d=p.plan(m,numerical_profile=p.PROFILE);assert d['partitions']==(5 if m<=32 else 2)
 assert d['raw_shape']==(m,2304) and d['numerical_profile']==p.PROFILE
@pytest.mark.parametrize('profile',[None,'reference','bf16_p1','legacy',True])
def test_wrong_profile_rejected(profile):
 with pytest.raises(ValueError):p.plan(32,numerical_profile=profile)
@pytest.mark.parametrize('m',[1,12,97,0,True])
def test_unsupported_rows_rejected(m):
 with pytest.raises(ValueError):p.plan(m,numerical_profile=p.PROFILE)
def test_no_implicit_profile():
 with pytest.raises(TypeError):p.plan(32)
def test_local_pins_and_outer_source_contract():
 p.verify_sources()
 assert 'float out=0;'in reducer.reduction_spec()['source']
 assert 'out+=C_split[offset];'in reducer.reduction_spec()['source']
 assert reducer.reduction_spec()['compile_options']=={'math_mode':'safe'}
 text=Path(p.__file__).read_text();assert 'mx.sum('not in text and 'producer.project('not in text
@pytest.mark.parametrize('midpoint',[True,False])
def test_project_uses_partial_producer_ordered_reducer_only(monkeypatch,midpoint):
 n=32;vals=np.zeros((5,n,2304),np.float32)
 if midpoint:vals[0]=1+2**-8;vals[1]=-1;expected=np.float32(2**-8)
 else:vals[0]=1e20;vals[1]=-1e20;vals[2]=3;vals[3]=4;vals[4]=5;expected=np.float32(12)
 parts=mx.array(vals);seen={}
 def make(**kw):seen['producer']=kw;return[parts]
 def reduce(**kw):
  seen['reducer']=kw;return[mx.array(h.ordered_partials(np,np.asarray(kw['inputs'][0])))]
 monkeypatch.setattr(producer,'_kernel',lambda rows:make);monkeypatch.setattr(reducer,'_reduction_kernel',lambda:reduce)
 from mlx2.runtime.gguf import source_dense_legacy
 monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:True);monkeypatch.setattr(p,'verify_sources',lambda:None)
 monkeypatch.setattr(mx,'sum',lambda *a,**k:(_ for _ in ()).throw(AssertionError('generic sum forbidden')))
 x=mx.zeros((n,5120));e=old.packed();rounded,raw,actual=p.project(x,e,numerical_profile=p.PROFILE,return_partials=True)
 assert actual is parts and np.all(np.asarray(raw)==expected)
 assert np.array_equal(np.asarray(rounded),np.asarray(raw))
 assert seen['producer']['inputs'][0]is x and seen['producer']['inputs'][2]is e.backing
 assert np.array_equal(np.asarray(seen['producer']['inputs'][1]),np.arange(n,dtype=np.uint32))
 assert np.array_equal(np.asarray(seen['reducer']['inputs'][1]),[5,n*2304,2304])
def test_device_refusal_before_kernel():
 with pytest.raises(ValueError,match='caller-owned'):p.project(mx.zeros((32,5120)),old.packed(),numerical_profile=p.PROFILE)
@pytest.mark.parametrize('m',[32,34])
def test_offline_metal4_producer_and_safe_reducer(m,tmp_path):
 if not shutil.which('xcrun'):pytest.skip('compiler unavailable')
 p.verify_sources();s=producer.kernel_spec(m)
 source='#include "mlx/backend/metal/kernels/utils.h"\n'+s['header']+'''
kernel void probe(const device float* x [[buffer(0)]],const device uint* rows_map [[buffer(1)]],
const device uchar* backing [[buffer(2)]],const device uint* layout_offsets [[buffer(3)]],
device float* partials [[buffer(4)]],uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
'''+s['source']+'\n}\n'
 path=tmp_path/'producer.metal';path.write_text(source)
 args=['xcrun','-sdk','macosx','metal','-std=metal4.0','-I',str(Path(mx.__file__).parent/'include'),'-c',str(path),'-o',str(tmp_path/'p.air')]
 run=subprocess.run(args,capture_output=True,text=True,timeout=45);assert run.returncode==0,run.stderr
 r=reducer.reduction_spec();source=r['header']+'''kernel void reduce(const device float* partials [[buffer(0)]],const device int* metadata [[buffer(1)]],device float* raw [[buffer(2)]],uint3 thread_position_in_grid [[thread_position_in_grid]]) {'''+r['source']+'}\n'
 path=tmp_path/'reducer.metal';path.write_text(source)
 run=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-fno-fast-math','-c',str(path),'-o',str(tmp_path/'r.air')],capture_output=True,text=True,timeout=45);assert run.returncode==0,run.stderr

def test_ordered_cpu_cancellation_counterexample():
 v=np.array([1e20,-1e20,3,4,5],np.float32).reshape(5,1,1)
 assert h.ordered_partials(np,v)[0,0]==12

@pytest.mark.parametrize('kind',['exact','different','bad_outer','nonfinite','stock_drift','input_drift'])
def test_component_separates_numerics_from_engineering(tmp_path,monkeypatch,kind):
 monkeypatch.syspath_prepend(str(ROOT/'scripts/research'));helper=types.ModuleType('ds41_prefill_legacy_gu_components')
 cases=[]
 for expert,m,parts in [(38,34,2),(50,32,5)]:
  for proj in ('gate','up'):
   x=np.zeros((m,5120),np.float32)
   if kind=='input_drift':x[0,0]=1.00001
   z=np.zeros((m,2304),np.float32)
   cases.append((dict(layer=0,expert=expert,projection=proj,rows=m,partitions=parts),dict(activation=x,reference_preround=z,reference_boundary=z)))
 helper.capture_module=lambda path:types.SimpleNamespace(load_capture=lambda path:({'checkpoint':'fixture','source_revision':'fixture','source_snapshot_sha256':'capture'},cases));helper.check_coefficients=lambda *a:None;monkeypatch.setitem(sys.modules,'ds41_prefill_legacy_gu_components',helper)
 fake=types.SimpleNamespace(gpu='GPU',default_device=lambda:'GPU',float32=np.float32,bfloat16=np.float32,array=np.array,eval=lambda *a:None,synchronize=lambda:None,get_peak_memory=lambda:123,
 quantized_matmul=lambda x,*a,**kw:np.full((len(x),2304),1 if kind=='stock_drift'else 0,np.float32))
 def candidate(x,e,*,projection,return_partials,numerical_profile):
  assert numerical_profile==p.PROFILE
  n=len(x);parts=np.zeros((5 if n==32 else 2,n,2304),np.float32)
  value=0 if kind=='exact'else np.nan if kind=='nonfinite'else 1
  parts[0]=value;raw=np.full((n,2304),value,np.float32)
  if kind=='bad_outer':parts[0]=2
  return (raw,raw,parts)if return_partials else raw
 timed=[]
 def timing(*a,**k):timed.append(True);return{'synthetic_timing_only':True}
 capture=tmp_path/'capture.json';capture.write_text('{}');report={'checkpoint':'fixture','source_revision':'fixture','source_snapshot_sha256':'component'}
 kwargs=dict(candidate=candidate,profile_timing=timing)
 def run():return h.run_nax_profile(fake,np,capture,lambda *a:{'gate.qweight':None,'gate.scales':None,'up.qweight':None,'up.scales':None},tmp_path,report,None,lambda p,d:p.write_text(json.dumps(d)),**kwargs)
 if kind in ('stock_drift','input_drift'):
  with pytest.raises(ValueError):run()
  assert not timed;return
 r=run();assert r['harness_completed'] and r['required_reference_controls_exact']
 assert r['candidate_raw_and_boundary_exact']is(kind=='exact')
 assert len(timed)==(4 if kind in ('exact','different')else 0)
 assert r['ordered_reduction_exact']is(kind not in ('bad_outer','nonfinite'))
 assert not r['model_quality_assessed'] and not r['candidate_used_by_model']

def test_profile_timer_admits_finite_different_outputs():
 fake=types.SimpleNamespace(synchronize=lambda:None,eval=lambda *a:None)
 result=h.time_different_profile(fake,np,{'reference':lambda:np.zeros((2,2),np.float32),p.PROFILE:lambda:np.ones((2,2),np.float32)})
 assert all(len(r['samples'])==3 for r in result.values())
 assert result['reference']['output_sha256']!=result[p.PROFILE]['output_sha256']
@pytest.mark.parametrize('kind',['nonfinite','unstable'])
def test_profile_timer_rejects_invalid_or_unstable_execution(kind):
 fake=types.SimpleNamespace(synchronize=lambda:None,eval=lambda *a:None);count=[0]
 def fn():
  count[0]+=1;return np.full((2,2),np.nan if kind=='nonfinite'else count[0],np.float32)
 with pytest.raises(ValueError):h.time_different_profile(fake,np,{'reference':fn})
def test_candidate_only_explicit_source_adapter_route():
 for path in (ROOT/'src/mlx2/adapters').glob('*.py'):
  if path.name!='deepseek_v41_source_weights.py':
   assert 'source_prefill_nax_profile'not in path.read_text()
 from mlx2.adapters.deepseek_v41_source import source_policy
 assert source_policy().get('deepseek_source_prefill_wide_math','reference')=='reference'
