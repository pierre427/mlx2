"""Synthetic packed/route contracts only; no checkpoint or GPU execution."""
from dataclasses import replace
import inspect
import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_source_prefill_bank import build_prefill_bank
from mlx2.adapters.deepseek_v41_source_bank import plan_bank, _QUARANTINE
from mlx2.runtime.gguf import source_resident_prefill as pp
from test_deepseek_v41_source_bank_cpu import CPU, expert, store


def test_incremental_peak_is_one_bounded_retirement_not_full_duplicate():
    values=[((0,i),expert(seed=i))for i in range(4)]
    s=store(*values);size=values[0][1].allocation_nbytes
    p=plan_bank(s,tuple(s._cache),max_bank_bytes=size,construction='incremental')
    assert p.peak_bound_bytes==s._cache_bytes+2*size+3*p.metadata_bytes
    assert p.peak_bound_bytes<2*s._cache_bytes+size+3*p.metadata_bytes
    b=build_prefill_bank(CPU(),s,tuple(s._cache),ownercheck=lambda:True,
        max_bank_bytes=size,budget_bytes=p.peak_bound_bytes,staging_bytes=0)
    assert b.plan.construction=='incremental'and len(b.banks)==4
    for key,old in values:
        for name in old:np.testing.assert_array_equal(s._cache[key][name],old[name])
    assert s._cache_bytes==4*size and s._resident_bank_preparing is None
    b.close();assert s._cache_bytes==0


def test_completed_partial_failure_evacuates_retired_whole_cohort_only():
    old=[((0,i),expert(seed=i))for i in range(3)];s=store(*old);mx=CPU();cat=mx.concatenate
    def fail_second(values,axis):
        if mx.cats==1:raise MemoryError('synthetic second-bank failure')
        return cat(values,axis)
    mx.concatenate=fail_second
    with pytest.raises(MemoryError,match='second-bank'):
        build_prefill_bank(mx,s,tuple(s._cache),ownercheck=lambda:True,max_bank_bytes=old[0][1].allocation_nbytes)
    assert set(s._cache)=={(0,1),(0,2)}
    assert s._cache[(0,1)]is old[1][1] and s._cache[(0,2)]is old[2][1]
    assert s._cache_bytes==2*old[0][1].allocation_nbytes
    assert not getattr(s,'_resident_bank',None) and s._resident_bank_preparing is None
    assert not s._resident_bank_pinned_keys


def test_failed_completion_quarantines_partial_new_old_and_pending_owners():
    old=[((0,i),expert(seed=i))for i in range(2)];s=store(*old);mx=CPU();cat=mx.concatenate
    def fail_second(values,axis):
        if mx.cats:raise MemoryError('allocation')
        return cat(values,axis)
    mx.concatenate=fail_second
    def fence():
        s.fences+=1
        if s.fences>1:raise RuntimeError('uncertain GPU drain')
    s._fence=fence;start=len(_QUARANTINE)
    try:
        with pytest.raises(RuntimeError,match='uncertain GPU drain'):
            build_prefill_bank(mx,s,tuple(s._cache),ownercheck=lambda:True,max_bank_bytes=old[0][1].allocation_nbytes)
        assert len(_QUARANTINE)==start+1 and s._resident_bank_preparing
        assert s._resident_bank_pinned_keys==frozenset({(0,0)})
        assert 'uncertain GPU drain'in s._error
    finally:del _QUARANTINE[start:]


def test_pp_metadata_bounds_and_explicit_nonpromoted_wide_occupancy():
    for rows in (2,4,8,16,32,128):
        p=pp.plan(rows)
        assert p['max_tasks']==rows*6 and p['grouped_weight_reuse']
        assert p['original_occupancy_limit']==12 and p['host_route_readbacks']==0
        assert p['metadata_bytes']<100_000
    for rows in (1,129,True):
        with pytest.raises(ValueError):pp.plan(rows)
    key=((1,10_000_000,4_700_160,(0,1,0,2,3,0,4,5,0)),)
    for rows in (13,16,32,128):
        route=pp.route_spec(key,rows,0,7)
        assert 'count>12u' in route['source'] and '128u'in route['source']
        tasks=pp.task_spec(rows)
        assert 'if(bad||count==0)continue' in tasks['source']
        assert f'row<{rows}'in tasks['source']


def test_bank_prefill_has_no_host_route_readback_and_final_sum_is_ordered():
    source=inspect.getsource(pp.bank_prefill)
    assert not any(s in source for s in ('np.asarray','tolist(','.item(','.numpy('))
    combine=pp.combine_spec(16)['source']
    assert 'volatile float total' in combine and 'unroll(disable)'in combine
    assert 'ids[row*6+slots[j-1]]>ids[row*6+a]'in combine


def test_source_generation_keeps_exact_row_bands_and_finite_bindings(monkeypatch):
    import mlx.core as mx
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:pytest.fail('CPU test attempted Metal construction'))
    key=((1,4_700_160,4_700_160,(0,1,0,2,3,0,4,5,0)),
         (2,1_658_880,1_658_880,(0,1,2,3,4,5,6,7,8)))
    gu=pp.gu_spec(key);down=pp.down_spec(key)
    assert all('case '+str(r)+': {' in gu['source']for r in range(1,13))
    assert 'case 13:'not in gu['source']
    assert 'fp_qmv_wide_to_shared_impl' in down['header']
    assert all('fp_qmv_wide_to_shared_impl<float,32,4,'+str(v)+',16>'in down['source']for v in range(1,6))
    assert 'scores[ordered[start+vec0+v]]'in gu['source']
    assert 'output[(start+local)*5120' in down['source']
    assert len(gu['input_names'])+len(gu['output_names'])==10


def test_prefill_session_selection_is_explicit_fresh_or_resumed_and_bounded():
    from types import SimpleNamespace
    from mlx2.adapters.deepseek_v41_source_prefill_session import PreparedPrefillBankSession
    s=object.__new__(PreparedPrefillBankSession);s.active=s.replaying=False
    for offset in (0,99):
        for rows in (1,2,16,128):assert s.selected(SimpleNamespace(shape=(1,rows)),[SimpleNamespace(offset=offset)])
    assert not s.selected(SimpleNamespace(shape=(1,129)),[SimpleNamespace(offset=0)])
    assert not s.selected(SimpleNamespace(shape=(2,16)),[SimpleNamespace(offset=0)])
    s.active=True;assert not s.selected(SimpleNamespace(shape=(1,16)),[SimpleNamespace(offset=0)])


def test_installed_wide_splitk_partition_formula_and_signature_only_changes(monkeypatch):
    import mlx.core as mx
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:pytest.fail('GPU forbidden'))
    from mlx2.runtime.gguf import source_resident_prefill_wide as wide
    expected={13:(5,3),16:(5,3),32:(5,3),33:(2,1),64:(2,1),96:(2,1),97:(1,1),128:(1,1)}
    for m,p in expected.items():assert (wide.split_partitions(m,2304,5120),wide.split_partitions(m,5120,2304))==p
    key=((1,4_700_160,4_700_160,(0,1,0,2,3,0,4,5,0)),)
    for down in (False,True):
        spec=wide.projection_spec(key,32,down=down)
        assert 'const constant int&'not in spec['header']
        assert 'fp_qmm_t_impl<float,32,4,true,float,false,32,32,32>'in spec['source']
    for f in (wide.middle_spec(32,'metal::exp'),wide.down_reduce_spec(32)):
        assert 'if(expert>=384u)'in f['source']


def test_wide_component_uses_actual_middle_public_signature():
    import ast
    from pathlib import Path
    from mlx2.runtime.gguf.source_middle_fusion import middle
    assert tuple(inspect.signature(middle).parameters)==('gate','up','scores','slots')
    path=Path(__file__).resolve().parents[1]/'scripts/research/ds41_resident_prefill_components.py'
    tree=ast.parse(path.read_text())
    calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='separate_middle']
    assert len(calls)==1 and not calls[0].keywords
    assert 'if exp_spelling()!=spelling'in path.read_text()


def test_nonroute_kernel_identity_reuses_emitted_source_across_layers(monkeypatch):
    import mlx.core as mx
    from types import SimpleNamespace
    from mlx2.runtime.gguf import source_resident_prefill_wide as wide
    made=[]
    def factory(**spec):
        value=SimpleNamespace(spec=spec);made.append(value);return value
    monkeypatch.setattr(mx.fast,'metal_kernel',factory)
    key=((1,4_700_160,4_700_160,(0,1,0,2,3,0,4,5,0)),)
    try:
        for module,stages in ((pp,('tasks','gu','down','combine')),
                              (wide,('tasks','gather','gu_wide','down_wide','middle','down_reduce'))):
            module._compiled_kernel.cache_clear()
            for stage in stages:
                a=module._kernel(stage,key,16,2,17,'metal::exp')
                b=module._kernel(stage,key,16,39,72,'metal::exp')
                assert a is b
                assert a.spec['input_names']and a.spec['output_names']
            route_a=module._kernel('route',key,16,2,17,'metal::exp')
            route_b=module._kernel('route',key,16,39,72,'metal::exp')
            assert route_a is not route_b
            assert '2u*384u' in route_a.spec['source'] and '17u'in route_a.spec['source']
            assert '39u*384u'in route_b.spec['source']and '72u'in route_b.spec['source']
        assert len(made)==14 # ten shared stages + four actual route identities
    finally:
        pp._compiled_kernel.cache_clear();wide._compiled_kernel.cache_clear()


def test_prefill_session_wide_selection_and_identity_are_explicit(monkeypatch):
    from types import SimpleNamespace
    from mlx2.adapters.deepseek_v41_source_prefill_session import PreparedPrefillBankSession,implementation_identity
    from mlx2.runtime.gguf import source_resident_prefill_wide as wide
    from mlx2.adapters.deepseek_v41_source_bank_session import PreparedBankSession
    monkeypatch.setattr(PreparedBankSession,'__init__',lambda *a,**k:None)
    seen=[]
    def consumer(*args,**kwargs):
        seen.append(kwargs);return SimpleNamespace(status='status',roots=('root',),output='output')
    monkeypatch.setattr(wide,'bank_prefill_wide',consumer)
    s=PreparedPrefillBankSession(SimpleNamespace(source_resident_prefill_policy='expert_major_wide_v1'))
    s.active=True;s.lease=SimpleNamespace(generation=7);s.statuses=[];s.roots=[]
    assert s.moe(SimpleNamespace(shape=(99,5120)),'ids','scores','blk.3')=='output'
    assert seen[0]['layer']==3 and s.statuses==['status']and s.roots==['root']
    with pytest.raises(ValueError):PreparedPrefillBankSession(SimpleNamespace(source_resident_prefill_policy='reference'))
    assert len(implementation_identity())==64


def test_model_gate_raw_cache_comparison_and_fixed_shape_contract():
    from importlib.util import spec_from_file_location,module_from_spec
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    path=Path(__file__).resolve().parents[1]/'scripts/research/ds41_resident_prefill_model_gate.py'
    spec=spec_from_file_location('_pp_model_gate',path);gate=module_from_spec(spec);spec.loader.exec_module(gate)
    assert gate.plan(99)['max_bank_bytes']==3<<30 and gate.plan(128)['APCv2_publication']is False
    with pytest.raises(ValueError):gate.plan(100)
    mx=SimpleNamespace(uint8=np.uint8)
    a=SimpleNamespace(meta_state=('revision','identity','99'),_capacity=2,_pool_count=1)
    b=SimpleNamespace(meta_state=a.meta_state,_capacity=2,_pool_count=1)
    for name in gate.PLANES:
        value=np.array([0.0,-0.0],np.float32);setattr(a,name,value);setattr(b,name,value.copy())
    assert gate._cache_compare([a],[b],mx)['equal']
    b.keys[1]=0.0
    assert not gate._cache_compare([a],[b],mx)['equal'] # signed-zero bytes matter


def test_model_gate_expert_only_park_requires_activation_before_acquire():
    from importlib.util import spec_from_file_location,module_from_spec
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/research/ds41_resident_prefill_model_gate.py'
    spec=spec_from_file_location('_pp_model_park_gate',path);gate=module_from_spec(spec);spec.loader.exec_module(gate)
    calls=[]
    class StrictStore:
        _parked=False
        _cache={'hello':'coefficient'}
        def park(self):
            assert not self._parked;calls.append('park');self._cache={};self._parked=True
        def activate(self):
            assert self._parked and not self._cache;calls.append('activate');self._parked=False
        def acquire(self):
            if self._parked:raise RuntimeError('native store is parked')
            calls.append('acquire')
    store=StrictStore()
    def idlecheck(value):assert value is store and not value._parked;calls.append('idle')
    def live():calls.append('live');return True
    gate._park_experts(store,live,idlecheck=idlecheck);store.acquire()
    assert calls==['idle','park','live','activate','live','acquire']


def test_model_gate_hybrid_binding_uses_source_q1_hotmap_and_default_json_separators():
    from importlib.util import spec_from_file_location,module_from_spec
    from pathlib import Path
    from types import SimpleNamespace
    path=Path(__file__).resolve().parents[1]/'scripts/research/ds41_resident_prefill_model_gate.py'
    spec=spec_from_file_location('_pp_model_identity_gate',path);gate=module_from_spec(spec);spec.loader.exec_module(gate)
    store=SimpleNamespace(binding='c30fd25e3cc79ba7044686fa3f9a7a1c0f514f77ee6293d2f5f62bc409e02ddb',
        q1_binding='09508a4061700fbfc06f37f024e4dbee027e7bd90bd1fe6ecc38ec8b3fc27289',
        hot_binding='371051887455ccaf61f60c8e3fb74f41c7f53a2f389c88c581991007ff4024e8')
    adapter=SimpleNamespace(weights=SimpleNamespace(expert_stream=store),identity={'fingerprint':store.binding})
    assert gate._checkpoint(adapter)==gate.CHECKPOINT
    store.hot_binding='0'*64
    with pytest.raises(ValueError,match='hybrid checkpoint'):gate._checkpoint(adapter)
    adapter.identity['fingerprint']='927325be8884e00a7bf60b9aa7d540b167a5efec569b9c7c30b1d34c2793d000'
    with pytest.raises(ValueError,match='original-source identity'):gate._checkpoint(adapter)


def test_persistent_task_wrapper_preserves_entire_body_and_adds_final_scratch_barrier():
    from mlx2.runtime.gguf import source_resident_prefill_tasks as tasks,source_resident_prefill_wide as wide
    key=((1,4_700_160,4_700_160,(0,1,0,2,3,0,4,5,0)),)
    for stage in ('gu','down','gu_wide','down_wide'):
        is_wide=stage.endswith('_wide');stride=8 if stage.startswith('gu')else 2
        original=(wide.projection_spec(key,128,down=stage=='down_wide')if is_wide else
                  pp.gu_spec(key)if stage=='gu'else pp.down_spec(key))
        actual=tasks.wide_persistent_spec(stage,key,128)if is_wide else tasks.persistent_spec(stage,key)
        variable='job'if is_wide else'task'
        limit=('counts[2]'if stage=='down_wide'else'counts[1]')if is_wide else'task_count[0]'
        prefix=f'const uint {variable}=threadgroup_position_in_grid.z;if({variable}>={limit})return;'
        wrapper=f'for(uint {variable}=threadgroup_position_in_grid.z;{variable}<{limit};{variable}+={stride}u){{'
        assert actual['source']==original['source'].replace(prefix,wrapper)+'\nthreadgroup_barrier(mem_flags::mem_threadgroup);\n}\n'
        assert actual['header']==original['header']
        assert actual['input_names']==original['input_names']and actual['output_names']==original['output_names']
        assert tasks.task_grid('persistent_8_2_v1',stage,768)==stride
        assert tasks.task_grid('padded_v1',stage,768)==768
    with pytest.raises(ValueError):tasks.validate_policy('automatic')


def test_persistent_kernel_cache_is_layer_independent_and_no_host_count_readback(monkeypatch):
    import mlx.core as mx
    from mlx2.runtime.gguf import source_resident_prefill_tasks as tasks
    made=[]
    def factory(**spec):made.append(spec);return object()
    monkeypatch.setattr(mx.fast,'metal_kernel',factory)
    key=((1,4_700_160,4_700_160,(0,1,0,2,3,0,4,5,0)),)
    try:
        tasks.kernel.cache_clear()
        for stage in ('gu','down','gu_wide','down_wide'):
            assert tasks.kernel(stage,key,128,'metal::exp')is tasks.kernel(stage,key,128,'metal::exp')
        assert len(made)==4 and all(v['ensure_row_contiguous']is False for v in made)
        source=inspect.getsource(tasks)
        assert not any(s in source for s in ('np.asarray','tolist(','.item(','.numpy('))
    finally:tasks.kernel.cache_clear()


def test_persistent_stripes_visit_every_gpu_task_once_including_empty_and_tails():
    for stride in (2,8):
        for count in (0,1,2,7,8,9,13,64,128,594,768):
            visits=[task for stripe in range(stride)for task in range(stripe,count,stride)]
            assert sorted(visits)==list(range(count))and len(visits)==len(set(visits))


def test_warm_resident_abba_order_and_per_arm_route_io_swap_contracts():
    from importlib.util import spec_from_file_location,module_from_spec
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/research/ds41_resident_prefill_model_gate.py'
    spec=spec_from_file_location('_pp_warm_gate',path);gate=module_from_spec(spec);spec.loader.exec_module(gate)
    expected=[('padded_v1',v)for v in ('ordinary','padded_v1','padded_v1','ordinary')]
    expected += [('persistent_8_2_v1',v)for v in ('ordinary','persistent_8_2_v1','persistent_8_2_v1','ordinary')]
    assert gate._warm_order(2)==expected*2 and gate._warm_order(0)==[]
    with pytest.raises(ValueError):gate._warm_order(4)
    before=dict(Swapins=3,Swapouts=5,raw_swapusage='used=1M',thermal_ok=True,foreign_model_processes=[])
    delta=dict(expert_bytes_read=0,expert_uploads=0,engram_bytes_read=0,ordinary_replays=0,host_route_materializations=40)
    ordinary=gate._warm_gates(delta,before,before,ordinary=True)
    assert ordinary['zero_expert_and_engram_io']and ordinary['required_host_route_contract']and ordinary['swap_stable']
    assert not gate._warm_gates(delta,before,before,ordinary=False)['required_host_route_contract']
    delta['host_route_materializations']=0;delta['expert_bytes_read']=1
    assert not gate._warm_gates(delta,before,before,ordinary=False)['zero_expert_and_engram_io']
    assert not gate._warm_gates({},before,dict(before,Swapins=4),ordinary=False)['swap_stable']
    assert ordinary['rate_eligible']is False
