"""Evidence contract and leaf timing CPU fixtures; no model/GPU execution."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

@pytest.fixture
def helper(monkeypatch):
    root=Path(__file__).resolve().parents[1];monkeypatch.syspath_prepend(str(root/'scripts/research'))
    spec=importlib.util.spec_from_file_location('small_m_helper_fixture',root/'scripts/research/ds41_small_m_nax_components.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_case_geometry_and_profile_inventory(helper):
    p=helper.case_plans();assert len(p)==10
    assert {x['partitions']for x in p}=={1,5}
    for part in(1,5):
        cases=[x for x in p if x['partitions']==part];assert [len(x['row_counts'])for x in cases]==[1,1,1,3,15]
        assert all(all(8<=n<=12 for n in x['row_counts'])for x in cases)
        assert all(f'_p{part}_'in x['numerical_profile']for x in cases)


def test_ordered_zero_seed_and_cancellation(helper):
    a=np.zeros((5,1,4),np.float32);a[:,0,0]=[2**24,1,-2**24,0,0];a[:,0,1]=[-0.]*5
    a[0,0,2]=np.float32(1.00390625);a[0,0,3]=10.03125
    out=helper.ordered(np,a);assert out[0,0]==0 and not np.signbit(out[0,1]);assert out[0,2]==a[0,0,2]
    one=helper.ordered(np,a[:1]);assert one[0,0]==2**24 and not np.signbit(one[0,1])

@pytest.mark.parametrize('failure',['partial','ordered','middle','nonfinite'])
def test_engineering_failure_suppresses_timing_admission(helper,failure):
    checks={'paired_partial':{'exact':True,'finite':True},'ordered_raw':{'exact':True,'finite':True},'same_profile_middle':{'exact':True,'finite':True},'stock_vs_native_middle':{'exact':False,'finite':True}}
    if failure=='partial':checks['paired_partial']['exact']=False
    elif failure=='ordered':checks['ordered_raw']['exact']=False
    elif failure=='middle':checks['same_profile_middle']['exact']=False
    else:checks['stock_vs_native_middle']['finite']=False
    assert not helper.admitted(checks)


def test_different_profile_is_not_engineering_failure(helper):
    assert helper.admitted({'paired_partial':{'exact':True,'finite':True},'stock_vs_native_middle':{'exact':False,'finite':True}})


def test_controller_return_contract_complete_only_full_inventory(helper):
    c={'checks':{'raw':{'exact':True,'finite':True}}};section={'reference_controls':[c]*4,'cases':[{**z,'engineering_exact':True,'finite':True} for z in helper.case_plans()],'adversarials':[{'exact':True}]*2}
    r=helper.finish(section);assert r['complete']and r['harness_completed']and r['required_reference_controls_exact']and all(r['verdicts'].values())
    section['cases']=section['cases'][:9];r=helper.finish(section);assert not r['complete']and not r['verdicts']['all_engineering_exact']
    section['reference_controls']=[]
    with pytest.raises(ValueError,match='Mandatory'):helper.finish(section)


def test_leaf_timer_no_artificial_concat_and_distinct_profile_allowed(helper,monkeypatch):
    calls=[];fake=SimpleNamespace(synchronize=lambda:None,eval=lambda *a:calls.append(len(a)))
    rng=np.random.default_rng(4102612);monkeypatch.setattr(np.random,'default_rng',lambda *a:rng)
    monkeypatch.setattr(np,'concatenate',lambda *a,**k:pytest.fail('No timed output assembly'))
    r=helper.timed_leaves(fake,np,{'baseline':lambda:[np.zeros((8,2)),np.zeros((12,2))],'candidate':lambda:np.ones((20,2))})
    assert r['baseline']['output_leaves']==2 and r['candidate']['output_leaves']==1
    assert len(r['baseline']['samples'])==3 and set(calls)=={1,2}


def test_unstable_output_refuses(helper):
    state=[0]
    def unstable():state[0]+=1;return np.array([state[0]],np.float32)
    with pytest.raises(ValueError,match='unstable'):helper.timed_leaves(SimpleNamespace(synchronize=lambda:None,eval=lambda *a:None),np,{'bad':unstable})


def test_extended_inventory_explicit_and_complete(helper):
    options={'all_single_m':True};plans=helper.case_plans(options)
    assert len(plans)==14
    for p in (1,5):
        assert [c['row_counts'] for c in plans if c['partitions']==p][:5]==[(8,),(9,),(10,),(11,),(12,)]
    section={'small_m_options':options,'reference_controls':[{'checks':{'raw':{'exact':True,'finite':True}}}]*4,
        'cases':[{**z,'engineering_exact':True,'finite':True} for z in plans],'adversarials':[{'exact':True}]*2}
    assert helper.finish(section)['complete']
    section['cases'][1]=section['cases'][0]
    assert not helper.finish(section)['complete']  # correct count alone is insufficient
    section['cases']=[{**z,'engineering_exact':True,'finite':True} for z in helper.case_plans()]
    assert not helper.finish(section)['complete']
    section['reference_controls'][0]={'checks':{'raw':{'exact':False,'finite':True}}}
    with pytest.raises(ValueError,match='Mandatory'):helper.finish(section)

@pytest.mark.parametrize('options',[{'all_single_m':1},{'all_single_m':None},{'other':True},[]])
def test_options_fail_closed(helper,options):
    with pytest.raises(ValueError,match='explicit boolean'):helper.case_plans(options)
