"""Native numerical-profile batch geometry/ownership, CPU and offline only."""
import hashlib
from pathlib import Path
import subprocess
import shutil
import numpy as np
import mlx.core as mx
import pytest
from mlx2.runtime.gguf import source_prefill_nax_profile as p,source_prefill_legacy_tiles as legacy
from mlx2.adapters import deepseek_v41_source_weights as sw
from mlx2.adapters.deepseek_v41_source import source_policy,DeepSeekV41SourceAdapter
from test_deepseek_v41_source_prefill_nax_cpu import packed
from test_deepseek_v41_source_prefill_wide_integration_cpu import case,run,expected

@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
 old=mx.default_device();mx.set_default_device(mx.cpu)
 monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:(_ for _ in ()).throw(AssertionError('GPU forbidden')))
 yield
 mx.set_default_device(old)

@pytest.mark.parametrize('counts',[(13,16,32),(33,34,64),(65,96),(13,)*15])
def test_batch_address_coverage_preserves_original_M_partition_and_tail(counts):
 d=p.batch_plan(counts,numerical_profile=p.PROFILE);seen=set()
 for part in range(d['partitions']):
  for slot,m in enumerate(counts):
   start=d['row_offsets'][slot]
   for tile in range(d['m_tiles']):
    first=tile*32;assert first<m
    for row in range(first,min(first+32,m)):
     key=(part,start+row);assert key not in seen;seen.add(key)
 assert len(seen)==d['partitions']*sum(counts)
 assert d['partial_shape']==(d['partitions'],sum(counts),2304)
 assert d['grid']==(2304,d['m_tiles'],d['partitions']*len(counts))
 assert d['input_bindings']+1<=19 and d['global_weight_repack_bytes']==0
 assert d['threadgroup']==(64,1,1)

@pytest.mark.parametrize('counts',[(),(13,)*16,(12,),(97,),(32,33),(64,65),(True,),('13',)])
def test_bad_counts_fail_closed(counts):
 with pytest.raises(ValueError):p.batch_plan(counts,numerical_profile=p.PROFILE)

def test_geometry_explicit_profile_and_original_backing_descriptor():
 e=packed();x=mx.zeros((45,5120));d,meta,es=p.prepare_batch_inputs(x,[e,e],(13,32),numerical_profile=p.PROFILE)
 assert es==(e,e) and not meta.flags.writeable
 assert meta.tolist()==[45,0,13,e.layout.tensor('gate.qweight').offset,e.layout.tensor('gate.scales').offset,13,32,e.layout.tensor('gate.qweight').offset,e.layout.tensor('gate.scales').offset]
 for bad in (None,'reference',True):
  with pytest.raises(ValueError):p.batch_plan((13,),numerical_profile=bad)
 with pytest.raises(ValueError):p.prepare_batch_inputs(x,[e],(13,32),numerical_profile=p.PROFILE)
 with pytest.raises(ValueError):p.prepare_batch_inputs(x,[e,e],(13,32),projection='down',numerical_profile=p.PROFILE)
 with pytest.raises(ValueError):p.prepare_batch_inputs(x.astype(mx.bfloat16),[e,e],(13,32),numerical_profile=p.PROFILE)


def test_native_batch_calls_one_producer_and_ordered_reducer_borrowing_backings(monkeypatch):
 from mlx2.runtime.gguf import source_dense_legacy
 monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:True);monkeypatch.setattr(p,'verify_sources',lambda:None)
 e=packed();x=mx.zeros((45,5120));seen={};parts=mx.zeros((5,45,2304))
 def make(**kw):seen['producer']=kw;return[parts]
 def reduce(**kw):seen['reduce']=kw;return[mx.ones((45,2304))]
 monkeypatch.setattr(p,'_batch_kernel',lambda *a:make);monkeypatch.setattr(legacy,'_reduction_kernel',lambda:reduce)
 rounded,raw,partial=p.batch_project(x,[e,e],(13,32),numerical_profile=p.PROFILE,return_partials=True)
 args=seen['producer']['inputs'];assert args[0]is x and args[3]is e.backing and args[4]is e.backing
 np.testing.assert_array_equal(np.asarray(args[1]),np.arange(45,dtype=np.uint32))
 assert partial is parts and seen['reduce']['inputs'][0]is parts
 np.testing.assert_array_equal(np.asarray(seen['reduce']['inputs'][1]),[5,45*2304,2304])
 assert np.array_equal(np.asarray(rounded),np.asarray(raw))

@pytest.mark.parametrize('counts',[(13,),(33,64),(65,96)])
def test_cpu_device_refuses_before_kernel(counts):
 with pytest.raises(ValueError,match='caller-owned'):
  p.batch_project(mx.zeros((sum(counts),5120)),[packed()]*len(counts),counts,numerical_profile=p.PROFILE)

@pytest.mark.parametrize('owned',[True,False])
@pytest.mark.parametrize('down',[True,False])
def test_native_adapter_reuses_worklist_rows_and_preserves_Q1_down_owners(monkeypatch,owned,down):
 c=case(monkeypatch,owned=owned,batch_down=down);c.w.source_prefill_wide_math='nax_k16_ordered_f32_v1'
 old=legacy.batch_project;calls=[]
 def native(*args,numerical_profile,**kwargs):
  assert numerical_profile==p.PROFILE;calls.append(kwargs['projection']);return old(*args,**kwargs)
 monkeypatch.setattr(p,'batch_project',native);monkeypatch.setattr(sw,'_verify_nax_prefill_source',lambda:None)
 assert run(c).tobytes()==expected(c).tobytes()
 assert calls==['gate','up']*3
 assert c.w.scheduled_expert_calls['official_source_prefill_nax_k16_jobs']==5
 assert c.w.scheduled_expert_calls['official_source_prefill_nax_k16_projection_batches']==6
 assert [(e,proj,n)for e,proj,n in c.calls['projections']if proj!='down']==[(6,'gate',97),(6,'up',97)]
 if owned:assert not c.store._leased and not c.store._active
 else:assert c.store._leased and c.store._active

@pytest.mark.parametrize('projection',['gate','up'])
def test_native_failure_preserves_ordinary_fence_cleanup(monkeypatch,projection):
 c=case(monkeypatch);c.w.source_prefill_wide_math='nax_k16_ordered_f32_v1';old=legacy.batch_project
 monkeypatch.setattr(sw,'_verify_nax_prefill_source',lambda:None)
 def fn(*a,**k):
  k.pop('numerical_profile');assert c.store._leased
  if k['projection']==projection:raise RuntimeError('native failure')
  return old(*a,**k)
 monkeypatch.setattr(p,'batch_project',fn)
 with pytest.raises(RuntimeError,match='native failure'):run(c)
 assert not c.store._leased and not c.store._active and len(c.fences)==1


def test_native_policy_identity_default_dependencies_and_source_pins():
 key='deepseek_source_prefill_wide_math';value='nax_k16_ordered_f32_v1'
 base={'deepseek_source_expert_math':'routed_fused_fp32','deepseek_source_prefill_batch':'gate_up'}
 assert source_policy(base).get(key,'reference')=='reference'
 assert source_policy({**base,key:value})[key]==value
 assert key in DeepSeekV41SourceAdapter.adapter_owned_default_policy_keys
 for opts in ({key:value},{**base,key:value,'deepseek_source_prefill_wide_dtype':'bf16_p1'}):
  with pytest.raises(ValueError):source_policy(opts)
 sw._verify_nax_prefill_source()
 root=Path(p.__file__).parent
 assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in sw.NAX_PREFILL_SOURCE_PINS.items())

@pytest.mark.parametrize('parts,tiles,count',[(5,1,1),(2,2,15),(2,3,3)])
def test_batch_offline_metal4_only(parts,tiles,count,tmp_path):
 if not shutil.which('xcrun'):pytest.skip('compiler unavailable')
 path=tmp_path/'batch.metal';path.write_text(p.batch_compilation_source(parts,tiles,count))
 result=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-I',str(Path(mx.__file__).parent/'include'),'-c',str(path),'-o',str(tmp_path/'batch.air')],capture_output=True,text=True,timeout=45)
 assert result.returncode==0,result.stderr


def test_native_receipt_and_policy_digest_are_distinct(monkeypatch):
 import json
 from mlx2.adapters.deepseek_v41_mlx_weights import NativeV41Weights
 c=case(monkeypatch);c.w.source_prefill_wide_math='nax_k16_ordered_f32_v1'
 monkeypatch.setattr(NativeV41Weights,'dense_receipt',lambda self:{})
 receipt=c.w.dense_receipt()
 assert receipt['source_prefill_wide_math']=='nax_k16_ordered_f32_v1'
 assert 'distinct unqualified numerical profile'in receipt['source_routed_dot_contract']
 base={'deepseek_source_expert_math':'routed_fused_fp32','deepseek_source_prefill_batch':'gate_up'}
 digest=lambda v:hashlib.sha256(json.dumps(source_policy(v),sort_keys=True).encode()).hexdigest()
 assert digest(base)!=digest({**base,'deepseek_source_prefill_wide_math':'nax_k16_ordered_f32_v1'})


def test_native_pin_drift_fails_before_dispatch(monkeypatch):
 sw._verify_nax_prefill_source.cache_clear()
 with monkeypatch.context()as m:
  m.setitem(sw.NAX_PREFILL_SOURCE_PINS,'source_prefill_nax_profile.py','0'*64)
  with pytest.raises(RuntimeError,match='re-audit'):sw._verify_nax_prefill_source()
 sw._verify_nax_prefill_source.cache_clear();sw._verify_nax_prefill_source()
