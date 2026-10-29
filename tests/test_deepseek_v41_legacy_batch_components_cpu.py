"""CPU derived-case planning and proof-flow fixtures; no GPU/model execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('batch_components_cpu',ROOT/'scripts/research/ds41_prefill_legacy_batch_components.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_derived_matrix_and_partial_axes():
    plans=m.case_plans();assert len(plans)==28
    assert {p['expert_count']for p in plans}=={1,3,15}
    for p in plans:
        assert p['partial_shape']==(p['partitions'],sum(p['row_counts']),2304)
        assert p['row_offsets'][-1]==p['input_shape'][0]
        assert all((x+31)//32==p['row_tiles']for x in p['row_counts'])
        assert all((5 if x<=32 else 2)==p['partitions']for x in p['row_counts'])
    p=m.case_plans()[-1];assert p['input_shape'][0]>96 and p['partitions']==2

def test_crop_repeat_preserves_exact_rows():
    x=np.arange(34*5120,dtype=np.float32).reshape(34,5120)
    for rows in (13,16,32,33,34,64,65,96):
        y=m.derived_rows(np,x,rows,2)
        assert y.shape==(rows,5120) and y.flags.c_contiguous
        assert y.tobytes()==x[(np.arange(rows)+2)%34].tobytes()
    with pytest.raises(ValueError):m.derived_rows(np,np.zeros(x.shape,np.float16),13)

def test_hotmap_identity_and_distinct_experts(tmp_path):
    hot={'source':'fixture','hot_budget_bytes':60_000_000_000,'experts':[[0,i]for i in range(25)]+[[1,15]]}
    hp=tmp_path/'hot.json';hp.write_text(json.dumps(hot));binding=hashlib.sha256(hp.read_bytes()).hexdigest()
    (tmp_path/'policy.json').write_text(json.dumps({'deepseek_source_experts':{'hotset_path':str(hp)}}))
    (tmp_path/'results.json').write_text(json.dumps({'passed':True,'cleanup_errors':[],'state_discarded':True,'expert_initial':{'hot_map_binding':binding,'source_binding':'fixture'}}))
    cases=[({'layer':0,'expert':e},None)for e in (15,20)]
    ids,sha=m.original_ids(tmp_path/'capture.json',cases)
    assert ids[:2]==[15,20] and len(ids)==len(set(ids))==15 and sha==binding
    hp.write_text('{}')
    with pytest.raises(ValueError,match='changed'):m.original_ids(tmp_path/'capture.json',cases)

@pytest.mark.parametrize('exact',[True,False])
def test_batch_raw_boundary_partial_gates_control_timing(tmp_path,monkeypatch,exact):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/research'))
    helper=types.ModuleType('ds41_prefill_legacy_gu_components')
    source=np.zeros((13,5120),np.float32)
    cases=[({'layer':0,'expert':e,'partitions':p,'projection':proj}, {'activation':source})for e,p in [(0,5),(1,2)]for proj in ('gate','up')]
    helper.capture_module=lambda p:types.SimpleNamespace(load_capture=lambda path:({'checkpoint':'synthetic','source_revision':'fixture','source_snapshot_sha256':'capture'},cases))
    helper.check_coefficients=lambda *a:None
    monkeypatch.setitem(sys.modules,'ds41_prefill_legacy_gu_components',helper)
    monkeypatch.setattr(m,'original_ids',lambda *a:(list(range(15)),'fixture-hotmap'))
    plan=next(p for p in m.case_plans()if p['row_counts']==(13,16,32)and p['projection']=='gate')
    monkeypatch.setattr(m,'case_plans',lambda:[plan]*28)
    capture=tmp_path/'capture.json';capture.write_text('{}')
    mx=types.SimpleNamespace(gpu='fixture_gpu',default_device=lambda:'fixture_gpu',float32=np.float32,bfloat16=np.float32,
        array=np.array,concatenate=np.concatenate,eval=lambda *a:None,synchronize=lambda:None,get_peak_memory=lambda:0,
        quantized_matmul=lambda x,q,**kw:np.zeros((len(x),2304),np.float32))
    q=np.zeros((2304,640),np.uint32);sc=np.zeros((2304,160),np.uint8)
    def load(layer,expert):return {p+'.qweight':q for p in ('gate','up')}|{p+'.scales':sc for p in ('gate','up')}
    def single(x,e,*,projection,return_partials=False):
        y=np.zeros((len(x),2304),np.float32);parts=np.zeros((5,len(x),2304),np.float32)
        return (y,y,parts)if return_partials else y
    def batch(x,e,counts,*,projection,return_partials=False):
        y=np.zeros((len(x),2304),np.float32);parts=np.full((5,len(x),2304),0 if exact else 1,np.float32)
        return (y,y,parts)if return_partials else y
    timed=[]
    def compare(mx,np,functions,repeats):timed.append(repeats);return {'synthetic_only':True}
    report={'checkpoint':'synthetic','source_revision':'fixture','source_snapshot_sha256':'component'}
    r=m.run_legacy_batch(mx,np,capture,load,tmp_path,report,compare,lambda p,d:p.write_text(json.dumps(d)),single=single,batch=batch)
    assert r['all_raw_boundary_partial_exact'] is exact
    assert len(timed)==(28 if exact else 0)
    assert all(c['checks']['batch_raw_vs_stock']['exact']for c in r['cases'])
    assert all(c['checks']['batch_partials_vs_single']['exact']is exact for c in r['cases'])
