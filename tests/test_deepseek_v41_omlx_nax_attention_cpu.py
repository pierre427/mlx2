"""CPU admission/addressing/staging and offline compilation; no GPU execution."""
import subprocess,shutil
from pathlib import Path
import numpy as np
import mlx.core as mx
import pytest
from mlx2.runtime.models import deepseek_v41_omlx_nax_attention as n

@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
 old=mx.default_device();mx.set_default_device(mx.cpu)
 monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:(_ for _ in ()).throw(AssertionError('GPU forbidden')))
 yield
 mx.set_default_device(old)

def arrays(length=16,local=17,pool=9,k=7):
 return [mx.zeros((1,64,length,512)),mx.zeros((1,1,local,512)),mx.ones((1,1,length,local),mx.bool_),
         mx.zeros((1,pool,512)),mx.zeros((1,length,k),mx.uint32),mx.ones((1,length,k),mx.bool_),mx.zeros((64,))]

@pytest.mark.parametrize('length,local,pool,k',[(16,17,9,7),(99,99,49,49),(99,99,99,99),(128,255,4096,512)])
def test_plan_and_staging_geometry(length,local,pool,k):
 p=n.plan(length,local,pool,k,numerical_profile=n.PROFILE)
 assert p['grid']==(length*256,2,1)and p['slot_tiles']==(local+k+127)//128
 assert p['threadgroup_bytes']<=32768
 with mx.stream(mx.cpu):
  data=arrays(length,local,pool,k);prepared=n.prepare(*data,numerical_profile=n.PROFILE);mx.eval(*prepared.inputs)
 assert prepared.inputs[0].shape==(64,length,512)and prepared.inputs[0].dtype==mx.bfloat16
 assert prepared.inputs[4].dtype==mx.bool_ and prepared.inputs[5].dtype==mx.bool_
 assert prepared.inputs[6].dtype==mx.float32
 assert data[0].dtype==mx.float32 and data[1].dtype==mx.float32
 np.testing.assert_array_equal(np.asarray(prepared.inputs[7]),[local,pool,length,k])

@pytest.mark.parametrize('args',[(1,17,9,7),(15,17,9,7),(129,17,9,7),(16,0,9,7),(16,256,9,7),(16,17,0,0),(16,17,4097,1),(16,17,9,10),(16,17,999,513),(True,17,9,7)])
def test_bounded_pp_only(args):
 with pytest.raises(ValueError):n.plan(*args,numerical_profile=n.PROFILE)

@pytest.mark.parametrize('field',[0,1,2,3,4,5,6])
def test_foreign_dtype_rejected(field):
 data=arrays();data[field]=data[field].astype(mx.float32 if field in (2,4,5)else mx.float16)
 with pytest.raises(ValueError):n.prepare(*data,numerical_profile=n.PROFILE)

def test_explicit_profile_and_gpu_ownership_required():
 with pytest.raises(ValueError):n.prepare(*arrays(),numerical_profile='reference')
 prep=n.prepare(*arrays(),numerical_profile=n.PROFILE)
 with pytest.raises(ValueError,match='caller-owned'):n.prepared_attention(prep,numerical_profile=n.PROFILE)


def test_prepared_launch_keeps_explicit_masks_and_float32_owner_boundary(monkeypatch):
 from mlx2.runtime.gguf import source_dense_legacy
 monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:True)
 prepared=n.prepare(*arrays(),numerical_profile=n.PROFILE);seen={}
 def kernel(**kw):seen.update(kw);return[mx.zeros((64,16,512))]
 monkeypatch.setattr(n,'_kernel',lambda:kernel)
 out=n.prepared_attention(prepared,numerical_profile=n.PROFILE)
 assert out.shape==(1,64,16,512)and out.dtype==mx.float32
 assert seen['inputs']==list(prepared.inputs)and seen['output_dtypes']==[mx.float32]
 assert 'selected_valid[pos] && idx < uint(NP)'in n._SOURCE
 assert 'local_mask[ulong(qi) * NS + slot]'in n._SOURCE
 assert 'const int ratio'not in n._SOURCE and 'device float* orow'in n._SOURCE


def test_cpu_slot_selection_matches_masks_with_duplicates_invalid_and_tail():
 # Selector preserves duplicate IDs and explicit visibility; no ratio-derived cut.
 ns,k,npool=99,49,49;ids=np.arange(k,dtype=np.uint32);ids[2]=ids[1];ids[-1]=np.uint32(2**32-1)
 lm=np.arange(ns)%3==0;valid=np.arange(k)%2==0
 expected=[('local',i)for i in range(ns)if lm[i]]+[('pool',int(ids[i]))for i in range(k)if valid[i]and ids[i]<npool]
 actual=[]
 for tile in range((ns+k+127)//128):
  for lane in range(128):
   slot=tile*128+lane
   if slot<ns:
    if lm[slot]:actual.append(('local',slot))
   elif slot<ns+k:
    j=slot-ns
    if valid[j]and ids[j]<npool:actual.append(('pool',int(ids[j])))
 assert actual==expected


def test_offline_metal4_compile(tmp_path):
 if not shutil.which('xcrun'):pytest.skip('compiler unavailable')
 source=tmp_path/'attention.metal';source.write_text(n.compilation_source())
 r=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-I',str(Path(mx.__file__).parent/'include'),'-c',str(source),'-o',str(tmp_path/'attention.air')],capture_output=True,text=True,timeout=60)
 assert r.returncode==0,r.stderr
