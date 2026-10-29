"""Synthetic CPU oracles and caller return-contract; no model/GPU execution."""
from pathlib import Path
import importlib.util
import types
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture
def helper(monkeypatch):
 monkeypatch.setenv('VECLIB_MAXIMUM_THREADS','1');monkeypatch.setenv('OPENBLAS_NUM_THREADS','1')
 monkeypatch.syspath_prepend(str(ROOT/'scripts/research'))
 s=importlib.util.spec_from_file_location('omlx_attention_fixture',ROOT/'scripts/research/ds41_omlx_nax_attention_components.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def data(length=2,heads=2,dim=4):
 return dict(q=np.arange(heads*length*dim,dtype=np.float32).reshape(1,heads,length,dim)/32,
   local=np.arange(3*dim,dtype=np.float32).reshape(1,1,3,dim)/16,local_mask=np.ones((1,1,length,3),bool),
   pooled=np.arange(2*dim,dtype=np.float32).reshape(1,2,dim)/8,indices=np.ones((1,length,2),np.uint32),
   valid=np.ones((1,length,2),bool),sinks=np.array([.1]*heads,np.float32))

def test_fp64_original_operand_reference_preserves_masks_duplicates_and_sink(helper):
 a=data();a['local_mask'][0,0,0]=[True,False,True];a['valid'][0,1]=[False,True]
 truth=helper.fp64_reference(np,a);expected=np.zeros_like(truth)
 for h in range(2):
  for t in range(2):
   rows=[a['local'][0,0,j].astype(np.float64)for j in range(3)if a['local_mask'][0,0,t,j]]+[a['pooled'][0,a['indices'][0,t,j]].astype(np.float64)for j in range(2)if a['valid'][0,t,j]]
   scores=[sum(float(a['q'][0,h,t,k])*.5*row[k]for k in range(4))for row in rows];maximum=max(scores+[float(a['sinks'][h])]);weights=np.exp(np.array(scores)-maximum)
   expected[0,h,t]=sum(w*row for w,row in zip(weights,rows))/(sum(weights)+np.exp(float(a['sinks'][h])-maximum))
 np.testing.assert_allclose(truth,expected,rtol=1e-14,atol=1e-14)
 assert truth.dtype==np.float64

def test_all_masked_reference_is_finite_zero(helper):
 a=data();a['local_mask'][:]=False;a['valid'][:]=False;a['sinks'][:]=1000
 assert np.array_equal(helper.fp64_reference(np,a),np.zeros(a['q'].shape,np.float64))

@pytest.mark.parametrize('bad',['negative','oob','oversize'])
def test_reference_rejects_unsafe_selection(helper,bad):
 a=data()
 if bad=='negative':a['indices']=np.full(a['indices'].shape,-1,np.int32)
 if bad=='oob':a['indices'][:]=2
 if bad=='oversize':a['q']=np.zeros((1,2,129,4),np.float32)
 with pytest.raises(ValueError):helper.fp64_reference(np,a)

@pytest.mark.parametrize('kind',['exact','different_profile','nonfinite','stock_drift'])
def test_six_capture_helper_contract_and_optional_numerical_verdicts(tmp_path,monkeypatch,helper,kind):
 import ds41_attention_components as capture
 from mlx2.runtime.models import deepseek_v41
 cases=[]
 for phase,length in [('pp99',16),('decode1',1)]:
  for layer in (2,20,21):
   a=data(length,64,512);a['q'][:]=0;a['local'][:]=0;a['pooled'][:]=0
   a.update(reference_raw=np.zeros((1,64,length,512),np.float32),reference_bf16=np.zeros((1,64,length,512),np.float32))
   cases.append((dict(layer=layer,phase=phase,length=length,frontier=0 if length>1 else 100),a))
 p=tmp_path/'capture.json';p.write_text('{}')
 monkeypatch.setattr(capture,'capture_module',lambda path:(types.SimpleNamespace(load_capture=lambda path:({'checkpoint':'fixture','source_revision':'cpu','source_snapshot_sha256':'capture'},cases)),p))
 monkeypatch.setattr(capture,'restored_inputs',lambda mx,np,c,a:([a[k]for k in ('q','local','local_mask','pooled','indices','valid','sinks')],None,{'CPU_fixture':True}))
 monkeypatch.setattr(deepseek_v41,'selected_attention',lambda *args,**kw:np.full(args[0].shape,1 if kind=='stock_drift'else 0,np.float32))
 fake=types.SimpleNamespace(gpu='gpu',default_device=lambda:'gpu',bfloat16=np.float32,float32=np.float32,eval=lambda *a:None,synchronize=lambda:None,get_peak_memory=lambda:123)
 def prepare(q,local,mask,pool,ids,valid,sinks,**kw):return types.SimpleNamespace(inputs=(q[0],local[0,0],pool[0],ids[0],mask[0,0],valid[0],sinks),geometry={'CPU_fixture':True})
 def core(p,**kw):return np.full((1,)+p.inputs[0].shape,np.nan if kind=='nonfinite'else 1 if kind=='different_profile'else 0,np.float32)
 candidate=types.SimpleNamespace(prepare=prepare,prepared_attention=core,attention=lambda *args,**kw:core(prepare(*args,**kw)))
 timed=[]
 def timer(*args,**kw):timed.append(True);return{'CPU_fixture':True}
 report={'checkpoint':'fixture','source_revision':'cpu','source_snapshot_sha256':'component'}
 def run():return helper.run_components(fake,np,p,tmp_path,report,None,lambda *a:None,candidate=candidate,timer=timer)
 if kind=='stock_drift':
  with pytest.raises(ValueError,match='Required'):run()
  assert not report[helper.KEY]['complete'];return
 r=run();assert r['complete']and r['harness_completed']and r['required_reference_controls_exact']and isinstance(r['verdicts'],dict)
 assert len(r['cases'])==6 and sum(c['candidate_dispatched']for c in r['cases'])==3
 assert r['verdicts']['all3_PP_raw_exact']is(kind=='exact')
 assert r['verdicts']['all3_PP_operator_finite']is(kind!='nonfinite')
 assert len(timed)==(0 if kind=='nonfinite'else 3)
 assert not r['model_quality_assessed']and not r['candidate_used_by_model']


def test_fp64_fallback_requires_controller_thread_environment(helper,monkeypatch):
 import sys
 monkeypatch.setitem(sys.modules,'threadpoolctl',None)
 monkeypatch.delenv('VECLIB_MAXIMUM_THREADS',raising=False)
 with pytest.raises(ValueError,match='controller BLAS'):helper.fp64_reference(np,data())

@pytest.mark.parametrize('repeats',[3,7])
def test_peer_timer_supports_bounded_repeat_contract(helper,repeats):
 fake=types.SimpleNamespace(synchronize=lambda:None,eval=lambda *a:None)
 r=helper.time_different_profile(fake,np,{'reference':lambda:np.ones((2,),np.float32),'candidate':lambda:np.zeros((2,),np.float32)},repeats=repeats)
 assert all(len(v['samples'])==repeats for v in r.values())

@pytest.mark.parametrize('repeats',[True,1,5,8,None])
def test_peer_timer_rejects_other_counts(helper,repeats):
 with pytest.raises(ValueError):helper.time_different_profile(None,np,{},repeats=repeats)
