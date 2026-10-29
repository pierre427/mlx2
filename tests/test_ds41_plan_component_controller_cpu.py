"""Mocked controller lifecycle only: no GPU, actual telemetry, locks or CPG."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_actual_head_verdict_keeps_checkpoint_and_distinct_profile_separate(controller):
    c=controller;plan={'checkpoint':'actual-source@'+'c'*64,'cells':['one','two']}
    assert c.component_checkpoint('head',plan,'a'*64)==plan['checkpoint']
    result=dict(source_and_artifact_unchanged=True,cleanup='synchronized_and_released',
        cells=[dict(cell=cell,raw_samples=[{}]*14,
                    comparison={'finite':True,'classification':'distinct_numerical_profile'})
               for cell in plan['cells']])
    assert c.suite_verdict('head',result,{'suite_inventory':plan})==(True,True)
    result['cells'].pop()
    assert c.suite_verdict('head',result,{'suite_inventory':plan})==(False,False)


@pytest.fixture
def controller(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/research/probe_ds41_plan_components.py'
    loader=importlib.util.spec_from_file_location('ds41_plan_component_controller_test',path)
    c=importlib.util.module_from_spec(loader);loader.loader.exec_module(c)
    root=tmp_path/'root';(root/'src/mlx2').mkdir(parents=True)
    (root/'src/mlx2/test_source.py').write_text('value=1')
    (root/'scripts/research').mkdir(parents=True)
    (root/'scripts/run_with_gpu_locks.py').write_text('wrapper=1')
    (root/'scripts/research/probe_ds41_plan_components.py').write_text('controller=1')
    monkeypatch.setattr(c,'ROOT',root)
    monkeypatch.setattr(c.subprocess,'check_output',lambda *a,**kw:'b'*40)
    monkeypatch.setattr(c,'inventory',lambda name:{'case_count':1}if name=='attention'else[{'case_id':'q1'}])
    return c


def prepared(c,tmp_path,suite='attention',*,purpose='performance'):
    out=tmp_path/'unit';spec=c.prepare(out,suite,purpose=purpose)
    return out,spec


def fake_deps(c,out,spec):
    events=[];t=[1000.];samples=[]
    def wall():return t[0]
    def sleep(seconds):t[0]+=seconds
    def sample(profile):
        t[0]+=.01
        result=dict(time_unix=t[0],temperature_sampled_at_unix=t[0],thermal_profile=profile,
            thermal_ok=True,thermal=dict(thermal_state=0,die_max_c=45.,hid_battery_temperature_c=30.,
                pmset_no_thermal_warning=True,pmset_no_performance_warning=True,pmset_no_cpu_power_warning=True),
            foreign_model_processes=[],Swapins=0,Swapouts=0,swap_used_bytes=0,
            numerics_environment={'MLX_ENABLE_TF32':'0'})
        samples.append(result);return result
    def failures(thermal,profile):
        return {'above':thermal['die_max_c']>60 or thermal['hid_battery_temperature_c']>40,
                'warnings':not all(thermal[k]is True for k in ('pmset_no_thermal_warning','pmset_no_performance_warning','pmset_no_cpu_power_warning'))}
    def gates(before,after,now):
        all_rows=before+after
        return {'three':len(before)==len(after)==3,
            'hid_three_sample_stability':all(max(row['thermal'][k]for row in rows)-min(row['thermal'][k]for row in rows)<=.5
                         for rows in (before,after)for k in ('die_max_c','hid_battery_temperature_c')),
            'swap_stable':all(all(row[k]==all_rows[0][k]for row in all_rows)for k in ('Swapins','Swapouts','swap_used_bytes')),
            'environment_stable':all(row['numerics_environment']==all_rows[0]['numerics_environment']for row in all_rows),
            'fresh':0<=now-after[-1]['temperature_sampled_at_unix']<=10.}
    def atomic(path,value):path.write_text(json.dumps(value))
    class MX:
        gpu='mock-gpu'
        def set_default_device(self,device):events.append('gpu_selected')
        def set_memory_limit(self,n):assert n==6<<30;events.append('memory_limit')
        def set_cache_limit(self,n):assert n==128<<20
        def synchronize(self):events.append('synchronize')
        def get_peak_memory(self):return 1<<20
        def clear_cache(self):events.append('clear_cache')
    mx=MX()
    def ownership(spec):events.append('ownership');return {'desktop_lease':{'cpg_session':'test-session'}}
    def host(minimum):events.append(('host',minimum));return SimpleNamespace(available_bytes=100<<30)
    def suite_run(name,mx,guard):
        assert guard()is True;sleep(1.1);assert guard()is True;events.append('suite')
        return {'complete':True,'cases':[{}],'numerical_validity_passed':True}
    class Lease:
        def __init__(self,*args,**kwargs):
            self.owner={'cpg_session':'test-session'};self.path=out/'desktop-lease.json';self.success=False
        def __enter__(self):events.append('lease_enter');return self
        def __exit__(self,*args):events.append('lease_exit');self.owner.update(release_task={},complete_worker={})
    class Child:
        pid=123
        stopped=False
        def stop(self,timeout):self.stopped=True;events.append('child_stop')
        def live(self):return []if self.stopped else[123]
    child=Child()
    def spawn(command,**kwargs):
        assert kwargs['start_new_session']is True
        assert kwargs['env']['MLX_ENABLE_TF32']=='0'
        events.append('spawn');return child
    def monitor(child,lease,path,queue_ready):
        assert queue_ready is False;events.append('monitor')
        atomic(out/'paired-locks.json',dict(status='command_completed',finished_at=t[0],
            owner={'session':'test-session','label':out.name}))
        atomic(out/'results.json',dict(passed=True,methodological_passed=True,numerical_passed=True,
            source_unchanged=True,purpose=spec['purpose'],**{k:spec[k]for k in ('checkpoint','suite','source_revision','source_sha256')}))
        return 0
    return SimpleNamespace(host=host,prelease=lambda before,waiters,**kw:(_ for _ in ()).throw(RuntimeError('waiter'))if waiters else None,
        monitor=monitor,sample=sample,failures=failures,waiters=lambda own:[],gates=gates,atomic=atomic,
        ownership=ownership,lease=Lease,spawn=spawn,mlx=lambda:mx,suite_run=suite_run,
        sleep=sleep,clock=wall,wall=wall,events=events,samples=samples,mx=mx,child=child)


def authorize_unit(c,out,spec,monkeypatch):
    c.write_new_json(out/'run-attempt.json',{'source_sha256':spec['source_sha256'],'run_gpu':True,'purpose':spec['purpose']})
    monkeypatch.setenv('MLX_ENABLE_TF32','0')


def test_prepare_freezes_without_dependency_hardware_or_lease_calls(controller,tmp_path,monkeypatch):
    c=controller
    monkeypatch.setattr(c,'dependencies',lambda:(_ for _ in ()).throw(AssertionError('execution dependencies forbidden')))
    out,spec=prepared(c,tmp_path)
    assert c.validate_spec(out)==spec
    assert spec['gpu_authorization']=='pending_user_approval'and spec['thermal_policy']['required_samples']==3
    assert spec['executed']is False and not spec['rate_eligible']
    with pytest.raises(FileExistsError):c.prepare(out,'attention')


@pytest.mark.parametrize('entry',['unit','run_prepared'])
def test_execution_requires_explicit_flag_before_any_execution_dependencies(controller,tmp_path,monkeypatch,entry):
    c=controller
    monkeypatch.setattr(c,'dependencies',lambda:(_ for _ in ()).throw(AssertionError('hardware forbidden')))
    with pytest.raises(PermissionError):getattr(c,entry)(tmp_path/'missing')


@pytest.mark.parametrize('field,value',[('memory_limit_bytes',7<<30),('thermal_profile','waived'),
    ('host_live_reserve_bytes',1),('thermal_policy',{}),('checkpoint','other@'+'a'*64)])
def test_modified_spec_cannot_relax_owned_bounds(controller,tmp_path,field,value):
    c=controller;out,spec=prepared(c,tmp_path);spec[field]=value
    (out/'spec.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError):c.validate_spec(out)


def test_frozen_source_drift_or_extra_import_input_refuses(controller,tmp_path):
    c=controller;out,spec=prepared(c,tmp_path)
    (out/'source/src/mlx2/unsealed.py').write_text('unexpected')
    with pytest.raises(ValueError,match='inventory'):c.validate_spec(out)


def test_unit_checks_ownership_and_initial_thermals_before_gpu_selection(controller,tmp_path,monkeypatch):
    c=controller;out,spec=prepared(c,tmp_path);d=fake_deps(c,out,spec);authorize_unit(c,out,spec,monkeypatch)
    assert c.unit(out,run_gpu=True,d=d)==0
    assert d.events.index('ownership')<d.events.index('gpu_selected')<d.events.index('suite')
    result=json.loads((out/'results.json').read_text())
    assert result['passed']and result['peak_mlx_bytes']==1<<20
    assert result['qualification']is False and result['performance_qualified']is False
    assert d.events[-1]=='clear_cache'


def test_hot_or_unstable_initial_samples_refuse_without_selecting_gpu(controller,tmp_path,monkeypatch):
    c=controller;out,spec=prepared(c,tmp_path);d=fake_deps(c,out,spec);authorize_unit(c,out,spec,monkeypatch)
    original=d.sample
    def hot(profile):
        row=original(profile);row['thermal']['die_max_c']=65.;return row
    d.sample=hot
    assert c.unit(out,run_gpu=True,d=d)==1
    assert 'gpu_selected'not in d.events
    assert (out/'error.json').exists()


def test_live_guard_caches_at_most_one_second_and_does_not_mix_old_brackets(controller,tmp_path):
    c=controller;out,spec=prepared(c,tmp_path);d=fake_deps(c,out,spec)
    before=c.three_samples(d);guard=c.LiveGuard(d,spec,out,before)
    assert guard()is True;count=len(d.samples)
    assert guard()is True and len(d.samples)==count
    d.sleep(12.);assert guard()is True and len(d.samples)==count+1
    assert d.events.count('ownership')==2
    original=d.sample
    def swapped(profile):row=original(profile);row['Swapins']=1;return row
    d.sample=swapped;d.sleep(1.1)
    with pytest.raises(RuntimeError,match='swap'):guard()


def test_partial_suite_error_is_atomic_and_cleans_device_owners(controller,tmp_path,monkeypatch):
    c=controller;out,spec=prepared(c,tmp_path);d=fake_deps(c,out,spec);authorize_unit(c,out,spec,monkeypatch)
    class Partial(RuntimeError):partial_result={'complete':False,'cases':[{'completed':'first'}]}
    def failed(*args):raise Partial('injected interruption')
    d.suite_run=failed
    assert c.unit(out,run_gpu=True,d=d)==1
    assert json.loads((out/'partial-component-result.json').read_text())==Partial.partial_result
    assert d.events[-1]=='clear_cache'


def test_parent_monitor_failure_stops_group_before_lease_release(controller,tmp_path):
    c=controller;out,spec=prepared(c,tmp_path);d=fake_deps(c,out,spec)
    def failed(*args,**kwargs):raise TimeoutError('injected monitor deadline')
    d.monitor=failed
    assert c.run_prepared(out,run_gpu=True,d=d)==1
    assert d.events.index('child_stop')<d.events.index('lease_exit')
    assert json.loads((out/'release-verified.json').read_text())['owned_processes_absent']


def test_parent_checks_bound_child_report_and_preserves_one_attempt(controller,tmp_path):
    c=controller;out,spec=prepared(c,tmp_path);d=fake_deps(c,out,spec)
    assert c.run_prepared(out,run_gpu=True,d=d)==0
    assert d.events.index('child_stop')<d.events.index('lease_exit')
    with pytest.raises(FileExistsError):c.run_prepared(out,run_gpu=True,d=d)


def test_final_source_failure_still_writes_release_receipt(controller,tmp_path):
    c=controller;out,spec=prepared(c,tmp_path);d=fake_deps(c,out,spec);original=d.monitor
    def drift(*args,**kwargs):
        result=original(*args,**kwargs);(out/'source/src/mlx2/test_source.py').write_text('changed');return result
    d.monitor=drift
    assert c.run_prepared(out,run_gpu=True,d=d)==1
    release=json.loads((out/'release-verified.json').read_text())
    assert release['source_unchanged']is False and release['source_error']


def test_q1_tuple_inventory_is_canonical_and_finite_mismatch_is_not_quality_failure(controller):
    c=controller;expected=[{'case_id':'q1','scopes':['first','second']}]
    row={'case':{'case_id':'q1','scopes':('first','second')},
        'comparisons':{scope:{'finite':True,'raw_bits_exact':False,'classification':'distinct_numerical_profile'}
            for scope in ('fresh_copy_cast_consumer','prepared_kernel_lower_bound')}}
    assert c.suite_verdict('q1',{'case_count':1,'cases':[row]},{'suite_inventory':expected})==(True,True)


@pytest.mark.parametrize('purpose',['functional','performance'])
def test_sealed_purpose_distinguishes_functional_noise_from_performance_admission(controller,tmp_path,monkeypatch,purpose):
    c=controller;out,spec=prepared(c,tmp_path,purpose=purpose);d=fake_deps(c,out,spec)
    authorize_unit(c,out,spec,monkeypatch);original=d.sample
    def noisy(profile):
        row=original(profile);index=len(d.samples)
        row['thermal']['die_max_c']=45.+(.6 if index%3==2 else 0.)
        row['Swapins']=index//3
        return row
    d.sample=noisy
    result=c.unit(out,run_gpu=True,d=d)
    report=json.loads((out/'results.json').read_text())
    if purpose=='performance':
        assert result==1 and 'gpu_selected'not in d.events
        assert report['admission_gates']['hid_three_sample_stability']is False
    else:
        assert result==0 and report['passed']and report['methodological_passed']and report['numerical_passed']
        assert report['performance_gates']['hid_three_sample_stability']is False
        assert report['performance_gates']['swap_stable']is False
        assert 'purpose_functional'in report['performance_ineligible_reasons']
        assert report['component_timing_eligible']is False and report['rate_eligible']is False
        assert report['performance_qualified']is False and report['timing_scope']=='diagnostic_only'
        assert report['live_performance_ineligible_events']
        raw=json.loads((out/'component-result.json').read_text())
        assert raw['controller_purpose']=='functional'and raw['controller_timing_scope']=='diagnostic_only'


@pytest.mark.parametrize('failure',['ceiling','ownership'])
def test_functional_purpose_still_refuses_hard_admission_failure(controller,tmp_path,monkeypatch,failure):
    c=controller;out,spec=prepared(c,tmp_path,purpose='functional');d=fake_deps(c,out,spec)
    authorize_unit(c,out,spec,monkeypatch)
    if failure=='ceiling':
        original=d.sample
        def hot(profile):row=original(profile);row['thermal']['die_max_c']=61.;return row
        d.sample=hot
    else:
        def lost(spec):raise RuntimeError('ownership lost')
        d.ownership=lost
    assert c.unit(out,run_gpu=True,d=d)==1
    assert 'gpu_selected'not in d.events


def test_functional_live_environment_change_still_refuses(controller,tmp_path):
    c=controller;out,spec=prepared(c,tmp_path,purpose='functional');d=fake_deps(c,out,spec)
    before=c.three_samples(d,'functional');guard=c.LiveGuard(d,spec,out,before);original=d.sample
    def changed(profile):row=original(profile);row['numerics_environment']={'MLX_ENABLE_TF32':'1'};return row
    d.sample=changed
    with pytest.raises(RuntimeError,match='environment'):guard()


def test_execution_cannot_override_sealed_purpose(controller,tmp_path):
    c=controller;out,spec=prepared(c,tmp_path,purpose='functional')
    assert c.validate_spec(out)['purpose']=='functional'
    with pytest.raises(SystemExit):c.main(['run','--output',str(out),'--run-gpu','--purpose','performance'])
    with pytest.raises(ValueError):c.prepare(tmp_path/'other','attention',purpose='unknown')


def test_persistent_prefill_suite_keeps_actual_checkpoint_and_source_closure():
    path=Path(__file__).resolve().parents[1]/'scripts/research/probe_ds41_plan_components.py'
    loader=importlib.util.spec_from_file_location('ds41_persistent_controller_inventory',path)
    c=importlib.util.module_from_spec(loader);loader.loader.exec_module(c)
    plan=c.inventory('prefill_persistent')
    assert c.MEMORY_LIMIT==6<<30 and plan['task_policy']=='persistent_8_2_v1'
    assert len(plan['valid_cases'])==14 and len(plan['refusal_cases'])==4
    assert c.component_checkpoint('prefill_persistent',plan,'synthetic')==plan['checkpoint']
    result=dict(passed=True,case_count=18,checkpoint=plan['checkpoint'],
        cases=[dict(label=label,passed=True)for label in plan['valid_cases']],
        refusals=[dict(label=label,passed=True)for label in plan['refusal_cases']])
    assert c.suite_verdict('prefill_persistent',result,dict(suite_inventory=plan,checkpoint=plan['checkpoint']))==(True,True)
    paths=c.runtime_source_paths(c.ROOT)
    assert 'src/mlx2/runtime/gguf/source_resident_prefill_tasks.py'in paths
    assert 'scripts/research/ds41_resident_prefill_persistent_components.py'in paths
