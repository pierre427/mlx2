"""CPU guards for borrowed-owner factorial entry points, no GPU or weights."""
from types import SimpleNamespace as NS
import numpy as np
import pytest

import ds41_synthetic_original_down_concat as subject
import ds41_synthetic_down_stage_components as base
from mlx2.runtime.gguf import source_down_worklist as down
from mlx2.adapters import deepseek_v41_source_math as assembly
from mlx2.adapters.deepseek_v41_source_prefill import plan_jobs


def test_plan_is_ten_bounded_original_cases():
    p=subject.plan()
    assert p['case_count']==len(p['profiles'])*len(p['variants'])==10
    assert p['conservative_operand_and_scratch_bound_bytes']<p['array_budget_bytes']==1<<30
    assert p['resident_expert_payload_bytes']==15*18_800_640
    assert [p['local_rows']for p in p['profiles']]==[2,6]


@pytest.mark.parametrize('variant,traversal',(('flat_worklist_concat','flat'),('stock_worklist_concat','stock')))
@pytest.mark.parametrize('rows',(2,6))
def test_actual_prepare_borrows_GU_owner_and_restores_owned_views_before_concat(monkeypatch,variant,traversal,rows):
    values,ids=base._routes(np,np.random.default_rng(41),(rows,)*15)
    kinds={i:'original'for i in ids};p=plan_jobs(values,kinds);bank={i:object()for i in ids}
    class Array:
        def __init__(self,shape,owner=None,span=None):self.shape=shape;self.owner=owner;self.span=span
        def __getitem__(self,key):
            assert isinstance(key,slice)and key.step is None
            lo=0 if key.start is None else key.start;hi=self.shape[0]if key.stop is None else key.stop
            return Array((hi-lo,self.shape[1]),self,(lo,hi))
    producer_owner=Array((rows*15,2304));middle=Array((rows*15,2304))
    descriptor=tuple((p.jobs[i].expert_id,p.jobs[i].count)for i in p.batches[0].job_indices)
    producer=((descriptor,producer_owner),);owned=Array((rows*15,5120));calls=[]
    def project(x,experts,counts,*,traversal):
        calls.append((x,experts,counts,traversal));return owned
    def combine(parts,v,order):
        assert v is values and np.array_equal(order,np.argsort(values.ravel(),kind='stable'))
        assert all(part.owner is owned and part.span==(j*rows,(j+1)*rows)for j,part in enumerate(parts))
        return 'ordered-output'
    monkeypatch.setattr(down,'pp_down',project);monkeypatch.setattr(assembly,'ordered_combine',combine)
    # No tensor operation may occur except existing DOWN and existing combine.
    result=subject._stage(NS(),np,values,kinds,middle,bank,producer,variant)
    assert result=='ordered-output'and len(calls)==1
    x,experts,counts,selected=calls[0]
    assert x is producer_owner and selected==traversal and counts==(rows,)*15
    assert experts==tuple(bank[j.expert_id]for j in p.jobs)


@pytest.mark.parametrize('variant,delegated,traversal',(
    ('per_expert_stock_concat','per_expert_stock_concat',None),
    ('flat_worklist_mapped','direct_worklist_mapped','flat'),
    ('stock_worklist_mapped','direct_worklist_mapped','stock')))
def test_controls_delegate_to_existing_actual_stage_utility(monkeypatch,variant,delegated,traversal):
    calls=[];monkeypatch.setattr(base,'_stage',lambda *a,**k:calls.append((a,k)))
    fields=[object()for _ in range(7)]
    subject._stage(*fields,variant)
    args,kwargs=calls[0]
    assert args[:6]==tuple(fields[:6])and args[6]==delegated and args[7]is fields[6]
    assert kwargs==({}if traversal is None else{'traversal':traversal})


def test_bad_variant_fails_closed():
    with pytest.raises(ValueError,match='Unknown original'):subject._stage(None,np,None,None,None,None,None,'other')


def test_run_clocks_fresh_calls_rotates_and_shares_exact_same_profile_owners(monkeypatch):
    mx=NS(gpu='gpu',default_device=lambda:'gpu',array=np.array,concatenate=np.concatenate,
          eval=lambda *a:None,isfinite=np.isfinite,all=np.all,get_peak_memory=lambda:100)
    monkeypatch.setattr(base,'_expert',lambda *a:object())
    calls=[]
    def stage(mx,np,values,kinds,middle,bank,producer,variant):
        calls.append((values.shape[0],variant,id(producer),tuple(map(id,bank.values()))))
        return np.zeros((values.shape[0],5120),np.float32)
    monkeypatch.setattr(subject,'_stage',stage)
    live=[];report=subject.run(mx,lambda:live.append(True))
    assert len(calls)==80 and len(live)==87
    assert report['case_count']==10 and report['all_control_output_bits_exact']
    assert all(len(c['samples'])==7 for c in report['cases'])
    assert report['rotated_order'][0][0]=='original15_R2/per_expert_stock_concat'
    assert report['rotated_order'][1][0]=='original15_R2/flat_worklist_mapped'
    for token_rows in (5,15):
        same=[c for c in calls if c[0]==token_rows]
        assert len({c[2]for c in same})==1 and len({c[3]for c in same})==1
    assert report['plan']['actual_weights_loaded']is False
