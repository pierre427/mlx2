"""synthetic@cpu-candidate-probe-validator; no model/GPU execution."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('ds41_candidate_pruning_probe',ROOT/'scripts/research/ds41_candidate_pruning_probe.py')
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)


def config():
    return SimpleNamespace(candidate_topk_blocks=2048,candidate_block_size=8,index_topk=512,
        index_n_heads=32,index_head_dim=128,num_attention_heads=64,head_dim=512,
        index_source_layer_ids=(2,8,14,20,24,28,32,36),candidate_source_layer_id=20,
        compress_ratios=(0,0)+(2,)*18+(1,)*20)


def test_plan_crosses_actual_capacity_inside_fixed_memory_envelope():
    p=m.plan(config())
    assert p['pool_rows']==16385 and p['candidate_capacity']==16384 and p['layer']==24
    assert p['component_allocation_envelope_bytes']<=192<<20
    assert p['APCv2'].startswith('not_applicable')and not p['qualification']
    with pytest.raises(ValueError,match='beyond actual candidate capacity'):m.plan(config(),pool_rows=16384)
    with pytest.raises(ValueError,match='bounded block'):m.plan(config(),pool_rows=20000)


def fixture():
    return {'indices':np.array([[[1,2]]],np.uint32),'valid':np.array([[[True,False]]]),
            'selected_scores':np.array([[[3.,-np.inf]]],np.float32),
            'owner_output':np.array([[[[.5,1.]]]],np.float32)}


def test_strict_selected_score_and_owner_oracle_does_not_promote_near_matches():
    a=fixture();b=copy.deepcopy(a)
    assert m.exact_oracle(a,b)['passed']
    b['owner_output'][0,0,0,0]=np.nextafter(np.float32(.5),np.float32(1.))
    v=m.exact_oracle(a,b);assert not v['passed']and not v['gates']['owner_output_exact']
    b=copy.deepcopy(a);b['selected_scores'][0,0,0]=np.nextafter(np.float32(3.),np.float32(4.))
    assert not m.exact_oracle(a,b)['passed']


def test_nonfinite_scores_are_allowed_only_where_selection_is_invalid():
    a=fixture();b=copy.deepcopy(a)
    a['valid'][:]=True;b['valid'][:]=True
    assert not m.exact_oracle(a,b)['gates']['selected_scores_finite_where_valid']


def test_proxy_isolates_policy_counters_without_mutating_loaded_model():
    base=SimpleNamespace(candidate_score_policy='reference',scheduled_kv_calls={'prior':4},linear=object())
    proxy=m._ModelPolicy(base,'candidate_only');proxy.scheduled_kv_calls['candidate_only_unscored_rows']=1
    assert base.candidate_score_policy=='reference'and base.scheduled_kv_calls=={'prior':4}
    assert proxy.linear is base.linear


def test_actual_selector_crosses_capacity_on_cpu_with_stubbed_projections(monkeypatch):
    # Real Attention.select and stable_pool_topk, synthetic projection operands;
    # this independently checks threshold/selection semantics without a model.
    import mlx.core as mx
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Metal forbidden')))
    try:
        from mlx2.runtime.models.deepseek_v41 import Attention
        c=config();n=16385
        keys=mx.broadcast_to(mx.arange(n,dtype=mx.float32)[None,:,None]/n,(1,n,128))
        source=SimpleNamespace(index_keys=keys,_pool_count=n)
        prior=np.ones((1,1,n),bool);prior[...,:8]=False
        results={}
        for policy in ('reference','candidate_only'):
            selector=object.__new__(Attention);selector.c=c;selector.layer=24;selector.ratio=1
            selector.model=SimpleNamespace(index_policy='reference',candidate_score_policy=policy,scheduled_kv_calls={})
            selector.linear=lambda x,name:mx.ones((1,1,4096 if name=='indexer.attn_q_b'else 32),mx.float32)
            selector.rotate_quantized=lambda value,*a,**kw:value
            shared={'candidates':mx.array(prior)}
            # Preserve the legacy stale bound-method trap in the fixture; the
            # real owning-class dispatch must still use the isolated policy.
            selector.select=selector.select
            selector=copy.copy(selector)
            selector.model=SimpleNamespace(index_policy='reference',candidate_score_policy=policy,scheduled_kv_calls={})
            m._dispatch_select(selector,mx.zeros((1,1,5120)),mx.zeros((1,1,1280)),source,n-1,shared)
            mx.eval(shared['indices'],shared['selected_mask'])
            results[policy]=(np.asarray(shared['indices']),np.asarray(shared['selected_mask']),selector.model.scheduled_kv_calls)
        np.testing.assert_array_equal(results['reference'][0],results['candidate_only'][0])
        np.testing.assert_array_equal(results['reference'][1],results['candidate_only'][1])
        assert results['candidate_only'][2]['candidate_only_score_rows']==16384
        assert results['candidate_only'][2]['candidate_only_unscored_rows']==1
    finally:mx.set_default_device(old)


def test_capture_hook_restores_class_lookup_without_leaving_bound_instance_method():
    class Selector:
        def __init__(self):self.model=SimpleNamespace(candidate_score_policy='reference')
        def select(self):return self.model.candidate_score_policy
    attention=Selector()
    assert 'select'not in vars(attention)
    with m._temporary_attribute(attention,'select',lambda:'capture'):
        assert attention.select()=='capture'
    assert 'select'not in vars(attention)
    candidate=copy.copy(attention);candidate.model=m._ModelPolicy(attention.model,'candidate_only')
    assert candidate.select()=='candidate_only'


def test_dispatch_bypasses_legacy_bound_method_copy_trap_without_weakening_policy():
    class Selector:
        def __init__(self):self.model=SimpleNamespace(candidate_score_policy='reference')
        def select(self):
            self.model.scheduled_kv_calls['called']=self.model.candidate_score_policy
            return self.model.candidate_score_policy
    attention=Selector();attention.model.scheduled_kv_calls={}
    # Reproduce r2: an old capture hook restored the resolved bound method.
    attention.select=attention.select
    candidate=copy.copy(attention);candidate.model=m._ModelPolicy(attention.model,'candidate_only')
    assert candidate.select()=='reference'
    assert not candidate.model.scheduled_kv_calls
    assert m._dispatch_select(candidate)=='candidate_only'
    assert candidate.model.scheduled_kv_calls=={'called':'candidate_only'}
    assert attention.model.candidate_score_policy=='reference'
