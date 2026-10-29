# SPDX-License-Identifier: MIT
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('ds41_advanced_options',Path(__file__).resolve().parents[1]/'scripts/research/ds41_advanced_options.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_advisory_helper_import_has_no_ownership_or_model_work():
    assert m.OWNERS[0].name=='owner.json'
    with pytest.raises(RuntimeError):m._live(lambda:False)
    with pytest.raises(RuntimeError):m._live(None)


def test_each_resident_variant_changes_one_session_option():
    variants=m.resident_variants()
    assert variants[0]=={'id':'control'}
    assert len(variants)==9
    assert all(len(v)==2 for v in variants[1:])
    assert m.resident_variants(prefill=True)[-1]['prefill_task_policy']=='persistent_8_2_v1'


def test_evidence_refuses_overwrite_and_keeps_valid_json(tmp_path):
    path=tmp_path/'receipt.json';m._write(path,{'passed':False})
    with pytest.raises(FileExistsError):m._write(path,{'passed':True})
    assert json.loads(path.read_text())=={'passed':False}


def test_canonical_profile_gate_precedes_device_import(tmp_path):
    from types import SimpleNamespace
    adapter=SimpleNamespace(kernel_policy={'deepseek_source_prefill_partition':'adaptive_v1'})
    with pytest.raises(ValueError,match='canonical_m1'):
        with m.canonical_prefix(adapter,tmp_path,lambda:True,[1,2]):pass


def test_actual_prefixed_expert_counters_fail_closed_and_detect_work():
    before={key:10 for key in m.EXPERT_COUNTER_KEYS}
    after=before.copy()
    assert all(m.expert_io_report(before,after)[name]is True for name in ('zero_reads','zero_uploads','zero_misses','zero_evictions'))
    after['v41_expert_bytes_read']+=4096
    assert m.expert_io_report(before,after)['zero_reads']is False
    after['v41_expert_uploaded_experts']+=1
    assert m.expert_io_report(before,after)['zero_uploads']is False
    with pytest.raises(ValueError):m.expert_io_report({'bytes_read':0,'uploaded_experts':0},{'bytes_read':0,'uploaded_experts':0})
    for invalid in (None,False,0.0,-1,9):
        broken=after.copy();broken['v41_expert_evictions']=invalid
        with pytest.raises(ValueError):m.expert_io_report(before,broken)
    missing=after.copy();del missing['v41_expert_cache_misses']
    with pytest.raises(ValueError):m.expert_io_report(before,missing)


def test_resident_telemetry_requires_actual_provider_and_session_counters():
    from types import SimpleNamespace as N
    raw={key:0 for key in m.EXPERT_COUNTER_KEYS}
    adapter=N(weights=N(expert_stream=N(counters=lambda:raw),source_route_host_materializations=4,
        source_resident_bank_session=N(stats={'ordinary_replays':0})),engram_tables={0:N(stats={'bytes_read':16})})
    observed=m.resident_counter_snapshot(adapter)
    assert observed['engram_bytes_read']==16 and observed['host_route_materializations']==4
    del adapter.weights.source_resident_bank_session.stats['ordinary_replays']
    with pytest.raises(ValueError):m.resident_counter_snapshot(adapter)


def test_publication_comparison_excludes_only_validated_slack_and_keeps_live_bytes_exact(monkeypatch):
    import numpy as np
    from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache
    row=DeepSeekV41LayerCache(2,xp=np,head_dim=4,index_head_dim=2,identity='synthetic@publication-extent',source=True,history=False)
    row.offset=10;row.keys=np.zeros((1,1,10,4),np.float32)
    row.buf_kv=np.zeros((1,0,4),np.float32);row.buf_gate=row.buf_kv.copy()
    row._pooled=np.zeros((1,256,4),np.float32);row._index_keys=np.zeros((1,256,2),np.float32)
    row._pool_count=5;row._capacity=256
    row.validate_state()
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts/research'))
    clone=m.publication_snapshot([row])[0]
    comparison=m.publication_cache_compare([row],[clone],np)
    assert comparison['equal']is True and comparison['raw_capacity'][0]['reference_capacity']==256
    assert comparison['raw_capacity'][0]['candidate_capacity']==5
    assert clone._capacity==clone._pool_count==5 and clone.meta_state==row.meta_state
    np.testing.assert_array_equal(clone.pooled,row.pooled)
    assert clone.pooled.tobytes()==row.pooled.tobytes()
    clone._pooled[0,0,0]=1
    assert clone.pooled.tobytes()!=row.pooled.tobytes()
    assert m.publication_cache_compare([row],[clone],np)['equal']is False
    row._capacity=255
    with pytest.raises(ValueError,match='layout'):m.publication_snapshot([row])
    row._capacity=256;row._pool_count=4
    with pytest.raises(ValueError):m.publication_snapshot([row])


@pytest.mark.parametrize('hybrid',[False,True])
def test_real_cohort_callback_owns_lifecycle_and_omits_an_observed_expert(monkeypatch,hybrid):
    from contextlib import nullcontext
    from types import SimpleNamespace as N
    from mlx2.adapters import deepseek_v41_source_prefill_bank as builder
    observed={(0,1),(1,2),(2,3)}
    store=N(_ready=lambda:None,_active=False,_leased={},_pending_uploads={},_pending_host={},
        pool=N(counters=lambda:{'weight_read_inflight':0}),_cache={k:object()for k in observed},
        _resident_bank=None,_resident_bank_pinned_keys=frozenset(),cache_budget_bytes=75_000_000_000)
    store._adaptive_evict_key=lambda key:store._cache.pop(key)
    store.window=lambda:nullcontext()
    store.acquire=lambda prefix,expert:store._cache.setdefault((int(prefix.split('.')[1]),expert),object())
    weights=N(expert_stream=store,source_resident_gather_policy='reference',source_resident_bank_session=None)
    model=N(weights=weights);adapter=N(model=model)
    built=[]
    def build(mx,current,keys,**kwargs):
        assert current is store
        bank=N(keys=frozenset(keys),assert_current=lambda:True,receipt=lambda:{'active_leases':0,'published':True})
        def close():store._resident_bank=None
        bank.close=close;store._resident_bank=bank;built.append(bank)
        return bank
    monkeypatch.setattr(builder,'build_prefill_bank',build)
    full,miss=m.ast_resident_callbacks(adapter,lambda:True,observed,hybrid=hybrid)
    for callback,missing in ((full,False),(miss,True)):
        with callback(model,None,None,None,lambda:True)as receipt:
            assert receipt['whole_cohort']is not missing
            assert len(built[-1].keys)==len(observed)-int(missing)
            if missing:assert tuple(receipt['genuine_omitted_expert'])in observed-built[-1].keys
            if hybrid:assert model._source_ast_bank_manager.bank is built[-1]
            else:
                assert weights.source_resident_bank_session.bank is built[-1]
                assert receipt['continuation_cache_fork']=='descriptor_cow_v1'
        assert store._resident_bank is None and store._resident_bank_pinned_keys==frozenset()
        assert weights.source_resident_bank_session is None
        assert getattr(model,'_source_ast_bank_manager',None)is None


def test_ast_proof_descriptor_fork_preserves_prebank_physical_capacity(monkeypatch):
    import mlx.core as mx
    from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache
    from mlx2.adapters.deepseek_v41_source_bank_session import _fork
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw:pytest.fail('GPU forbidden'))
    clones=[]
    try:
        row=DeepSeekV41LayerCache(2,xp=mx,head_dim=4,index_head_dim=2,identity='synthetic@proof-capacity',source=True,history=False)
        row.offset=10;row.keys=mx.zeros((1,1,10,4),mx.float32)
        row.buf_kv=mx.zeros((1,0,4),mx.float32);row.buf_gate=mx.zeros((1,0,4),mx.float32)
        row._pooled=mx.zeros((1,256,4),mx.float32);row._index_keys=mx.zeros((1,256,2),mx.float32)
        row._pool_count=5;row._capacity=256;row.validate_state()
        physical=_fork([row],'physical_copy_v1');clones.extend(physical)
        descriptors=_fork([row],'descriptor_cow_v1');clones.extend(descriptors)
        mx.eval([s.state for s in [row,*clones]])
        assert physical[0]._capacity==5 and physical[0]._pool_count==5
        assert descriptors[0]._capacity==256 and descriptors[0]._pool_count==5
        assert descriptors[0].meta_state==row.meta_state
        for left,right in zip(row.state,descriptors[0].state):
            if left is not None:assert bool(mx.array_equal(left,right).item())
        row.close()
    finally:
        for clone in clones:clone.close()
        mx.set_default_device(old)


@pytest.mark.parametrize('failure',[False,True])
def test_resident_ast_serving_selects_prepared_bank_only_inside_provider(failure):
    from types import SimpleNamespace as N
    bank=N(assert_current=lambda:True)
    session=N(bank=bank,active=False,replaying=False,poisoned=False)
    weights=N(source_resident_bank_session=session,source_resident_gather_policy='resident_bank_v1',
        expert_stream=N(_resident_bank=bank))
    model=N(weights=weights);adapter=N(model=model,weights=weights);calls=[]
    def execute(value):
        assert weights.source_resident_bank_session is session
        assert weights.source_resident_gather_policy=='resident_bank_v1'
        calls.append(value)
        if failure:raise RuntimeError('synthetic genuine target failure')
        return 'actual target result'
    provider=N(model=model,expert_law='source_ast_resident_rows_v1',forward_law='fully_batched_m1_v1',forward=execute)
    with m.resident_ast_serving_scope(adapter,session,lambda:True)as install:
        # Empty-prefix/body-only ordinary calls must never advertise a B1 bank.
        assert weights.source_resident_bank_session is None
        assert weights.source_resident_gather_policy=='reference'
        install(provider)
        if failure:
            with pytest.raises(RuntimeError):provider.forward('real prepared tile')
        else:assert provider.forward('real prepared tile')=='actual target result'
        assert weights.source_resident_bank_session is None and weights.source_resident_gather_policy=='reference'
    assert calls==['real prepared tile']and provider.forward is execute
    assert weights.source_resident_bank_session is session and weights.source_resident_gather_policy=='resident_bank_v1'


def test_incoming_scheduler_tree_gets_fresh_ordinary_census_and_real_bank_growth(monkeypatch):
    import mlx.core as mx
    from contextlib import contextmanager,nullcontext
    from types import SimpleNamespace as N
    from mlx2.adapters import deepseek_v41_source_bank as bank_module
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts/research'))
    import ds41_ast_target_components as proof
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw:pytest.fail('GPU forbidden'))
    store=N(_ready=lambda:None,_active=False,_leased={},_pending_uploads={},_pending_host={},
        pool=N(counters=lambda:{'weight_read_inflight':0}),_cache={(0,0):object()},
        _resident_bank_pinned_keys=frozenset({(0,0)}),cache_budget_bytes=75_000_000_000)
    store._adaptive_evict_key=lambda key:store._cache.pop(key)
    store.acquire=lambda prefix,expert:store._cache.setdefault((0,expert),object())
    store.window=lambda:nullcontext()
    class Bank:
        def __init__(self,keys,generation):self.keys=frozenset(keys);self.generation=generation;self.closed=False
        def assert_current(self):
            assert not self.closed and store._resident_bank is self
        def receipt(self):return {'generation':self.generation,'keys':sorted(self.keys)}
        @contextmanager
        def replacement(self,keys):
            assert self.keys.issubset(keys);store._resident_bank=None
            yield
            assert store._resident_bank.keys.issuperset(self.keys);self.closed=True
    bank=Bank({(0,0)},1);store._resident_bank=bank;session=N(bank=bank)
    weights=N(expert_stream=store,source_resident_bank_session=None,source_resident_gather_policy='reference')
    cache=[N(offset=264,state=[mx.array([0])])]
    model=N(weights=weights)
    def forward(inputs,private,layers):
        token=int(inputs[0,0].item());store.acquire('blk.0',token%5)
        private[0].offset+=1
        return mx.zeros((1,1,19)),mx.zeros((1,1,24))
    model.forward_with_taps=forward;adapter=N(model=model,weights=weights)
    monkeypatch.setattr(proof,'_proof_snapshot',lambda rows:[N(offset=r.offset,state=[mx.array([0])],close=lambda:None)for r in rows])
    def build(xp,current,keys,**kwargs):
        assert current is store
        fresh=Bank(keys,kwargs['generation']);store._resident_bank=fresh;return fresh
    monkeypatch.setattr(bank_module,'build_bank',build)
    from mlx2.adapters.deepseek_v41_ast_target import PARENTS
    try:
        report=m.incoming_ast_bank_prepare(adapter,session,lambda:True,model,tuple(range(1,17)),PARENTS,cache,(2,3,4))
        assert report['ordinary_control_steps']==18 and report['missing_experts']==4
        assert session.bank is store._resident_bank and session.bank.generation==2
        assert session.bank.keys==frozenset((0,i)for i in range(5))
        assert report['numerical_output_substituted']is False and report['oracle_admission_granted']is False
        assert cache[0].offset==264 and bank.closed
        assert store._resident_bank_pinned_keys==session.bank.keys
    finally:mx.set_default_device(old)


def test_m16_boundary_preserves_actual_production_values_and_hook_cleanup(monkeypatch):
    import sys
    from types import SimpleNamespace as N
    import mlx.core as mx
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts/research'))
    old=mx.default_device();mx.set_default_device(mx.cpu)
    weights=N(expert_math_policy='routed_fused_fp32',source_prefill_batch_policy='reference',scheduled_expert_calls={})
    outputs=[]
    def production(x,ids,scores,prefix,**kwargs):
        key='native_m16'if x.shape[0]==16 else'ordinary_m1'
        weights.scheduled_expert_calls[key]=weights.scheduled_expert_calls.get(key,0)+1
        result=x+(1 if x.shape[0]==16 else 0);outputs.append(result);return result
    weights.moe=production;model=N(weights=weights)
    x=mx.arange(32,dtype=mx.float32).reshape(16,2);ids=mx.zeros((16,1),mx.uint32);scores=mx.ones((16,1))
    try:
        with m.ordinary_m16_boundary(model,lambda:True)as report:
            result=weights.moe(x,ids,scores,'blk.14')
            assert result is outputs[0]
            for ordinal in range(18):weights.moe(x[12:13],ids[12:13],scores[12:13],'blk.14')
        assert weights.moe is production
        assert report['extra_m1_calls']==1 and report['ordinary_calls']==18
        assert report['same_input_m16_vs_m1']['equal']is False
        assert all(report['ordinary_node12'][k]['equal']for k in ('x','ids','scores','output_vs_same_input_m1'))
        assert report['ordinary_node12']['output_vs_candidate_m16']['equal']is False
        assert report['candidate_observed_calls']=={'native_m16':1}
        assert report['component_observed_calls']=={'ordinary_m1':1}
        with pytest.raises(RuntimeError):
            with m.ordinary_m16_boundary(model,lambda:True):raise RuntimeError('injected')
        assert weights.moe is production
    finally:mx.set_default_device(old)


def test_m16_boundary_layers_validate_before_wrapping_and_summarize(monkeypatch):
    import mlx.core as mx
    from types import SimpleNamespace as N
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts/research'))
    old=mx.default_device();mx.set_default_device(mx.cpu)
    w=N(expert_math_policy='routed_fused_fp32',source_prefill_batch_policy='reference',scheduled_expert_calls={})
    def production(x,ids,scores,prefix):return x+(1 if x.shape[0]==16 and prefix=='blk.1' else 0)
    w.moe=production;model=N(weights=w)
    try:
        with pytest.raises(ValueError):
            with m.ordinary_m16_boundaries(model,lambda:True,[14,14]):pass
        assert w.moe is production
        x=mx.zeros((16,2));ids=mx.zeros((16,1),mx.uint32);scores=mx.ones((16,1))
        with m.ordinary_m16_boundaries(model,lambda:True,[0,1])as report:
            for layer in (0,1):w.moe(x,ids,scores,f'blk.{layer}')
            for _ in range(18):
                for layer in (0,1):w.moe(x[12:13],ids[12:13],scores[12:13],f'blk.{layer}')
        assert w.moe is production
        assert report['first_same_input_m16_m1_difference']==1
        assert report['first_ordinary_input_difference']is None
        assert all(v['extra_m1_calls']==1 for v in report['layers'])
    finally:mx.set_default_device(old)
