"""CPU fixtures for native batch evidence gates; no actual model/GPU execution."""
import importlib.util
from pathlib import Path
import sys
import types
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture
def helper(monkeypatch):
 monkeypatch.syspath_prepend(str(ROOT/'scripts/research'))
 path=ROOT/'scripts/research/ds41_prefill_nax_batch_components.py';s=importlib.util.spec_from_file_location('native_batch_fixture',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_case_inventory_preserves_original_bands_and_binding_bound(helper):
 p=helper.case_plans();assert len(p)==9
 assert {v['experts']for v in p}=={1,3,15}
 for v in p:
  assert len({(5 if n<=32 else 2,(n+31)//32)for n in v['row_counts']})==1
  assert v['experts']+4<=19

@pytest.mark.parametrize('kind',['exact','different_profile','bad_partial','bad_raw','nonfinite','stock_drift','complete_contract','fused_contract','fused_bad_middle','fused_bad_partial','fused_bad_adversarial'])
def test_batch_must_match_single_not_stock_to_admit_profile_timing(tmp_path,monkeypatch,helper,kind):
 captured=[]
 for eid,m,p in [(38,34,2),(50,32,5)]:
  for proj in ('gate','up'):
   z=np.zeros((m,2304),np.float32);captured.append((dict(layer=0,expert=eid,rows=m,partitions=p,projection=proj),dict(activation=np.zeros((m,5120),np.float32),reference_preround=z,reference_boundary=z)))
 mod=types.ModuleType('ds41_prefill_legacy_gu_components');mod.capture_module=lambda path:types.SimpleNamespace(load_capture=lambda path:({'checkpoint':'fixture','source_revision':'cpu','source_snapshot_sha256':'capture'},captured));mod.check_coefficients=lambda *a:None;monkeypatch.setitem(sys.modules,'ds41_prefill_legacy_gu_components',mod)
 ids=[38,50]+list(range(13));monkeypatch.setattr(helper,'original_ids',lambda *a:(ids,'hot'))
 q=np.zeros((2304,640),np.uint32);s=np.zeros((2304,160),np.uint8);bank={k+'.'+f:v for k in ('gate','up')for f,v in [('qweight',q),('scales',s)]}
 fake=types.SimpleNamespace(gpu='gpu',default_device=lambda:'gpu',float32=np.float32,uint32=np.uint32,bfloat16=np.float32,array=np.array,concatenate=np.concatenate,eval=lambda *a:None,get_peak_memory=lambda:123,
 quantized_matmul=lambda x,*a,**kw:np.full((len(x),2304),1 if kind=='stock_drift'else 0,np.float32))
 def single(x,e,*,return_partials=False,**kw):
  m=len(x);raw=np.full((m,2304),1 if kind=='different_profile'else 0,np.float32);parts=np.zeros((5 if m<=32 else 2,m,2304),np.float32);parts[0]=raw
  return(raw,raw,parts)if return_partials else raw
 def batch(x,es,counts,*,return_partials=False,**kw):
  m=len(x);raw=np.full((m,2304),1 if kind=='different_profile'else 0,np.float32);parts=np.zeros((5,m,2304),np.float32);parts[0]=raw
  if kind=='bad_partial':parts[0,0,0]+=1
  if kind=='bad_raw':raw[0,0]+=1
  if kind=='nonfinite':raw[0,0]=np.nan
  return(raw,raw,parts)if return_partials else raw
 timed=[]
 def timing(*args,**kwargs):timed.append(set(args[2]));return{'CPU_fixture_only':True}
 p=tmp_path/'capture.json';p.write_text('{}');report={'checkpoint':'fixture','source_revision':'cpu','source_snapshot_sha256':'component'}
 fused=kind.startswith('fused_')
 if fused:
  report['nax_batch_options']={'fused_middle':True}
  from mlx2.runtime.gguf import source_middle_fusion
  monkeypatch.setattr(source_middle_fusion,'exp_spelling',lambda:'metal::exp')
  monkeypatch.setattr(helper,'adversarial_checks',lambda *a:[dict(all_exact=kind!='fused_bad_adversarial')]*2)
 def fused_fn(x,es,counts,scores,slots,*,return_partials=False,**kw):
  g,_,gp=batch(x,es,counts,return_partials=True);u,_,up=batch(x,es,counts,return_partials=True)
  out=(g+u)*scores[slots,None]
  if kind=='fused_bad_middle':out[0,0]+=1
  if kind=='fused_bad_partial':gp[0,0,0]+=1
  return(out,gp,up)if return_partials else out
 complete=kind in ('complete_contract','fused_contract')
 def run():return helper.run_nax_batch(fake,np,p,lambda *a:bank,tmp_path,report,None,lambda *a:None,single=single,batch=batch,middle_fn=lambda g,u,s,l:(g+u)*s[l,None],profile_timing=timing,plans=[dict(row_counts=(13,16,32),experts=3,partitions=5,row_tiles=1)]*(9 if complete else 1),fused_fn=fused_fn)
 if kind=='stock_drift':
  with pytest.raises(ValueError,match='Mandatory'):run()
  assert not timed;return
 d=run();assert d['harness_completed']and d['required_reference_controls_exact']
 assert d['all_batch_addressing_exact']==(kind in ('exact','different_profile','complete_contract','fused_contract'))
 assert bool(timed)==(kind in ('exact','different_profile','complete_contract','fused_contract'))
 assert not d['candidate_used_by_model']and not d['model_quality_assessed']
 assert d['complete']is complete
 if kind=='complete_contract':
  assert len(d['cases'])==9 and len(d['reference_controls'])==4
  assert d['complete']and d['harness_completed']and d['required_reference_controls_exact']and d['all_batch_addressing_exact']and d['all_finite']
 if kind=='different_profile':assert not d['cases'][0]['checks']['gate_stock_vs_native_raw']['exact']
 if fused:
  assert d['all_fused_middle_exact']is(kind=='fused_contract')
  if timed:assert all('fused_middle'in names for names in timed)

@pytest.mark.parametrize('value',[{'fused_middle':1},{'fused_middle':None},{'other':True},[]])
def test_unknown_fusion_options_refuse(helper,value):
 with pytest.raises(ValueError):helper.fused_option({'nax_batch_options':value})

def test_adversarial_signedzero_cancellation_round_clamp_inventory(helper):
 for p in (2,5):
  g,u,s,ids=helper.adversarial_partials(np,p)
  assert g.shape==u.shape==(p,8,2304)and g.dtype==np.float32
  assert np.isfinite(g).all()and np.isfinite(u).all()
  assert set(ids.tolist())==set(range(8))and np.signbit(s[0])and s[0]==0
  raw=helper.ordered_partials(np,g)
  assert raw[0,2]==0 and not np.signbit(raw[0,2])
  assert np.any(g==np.float32(1.00390625))and np.any(g==np.float32(10.03125))
 assert not helper.fused_option({})
