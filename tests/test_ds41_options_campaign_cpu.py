# SPDX-License-Identifier: MIT
"""CPU receipt regressions; no model, MLX or GPU execution."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('ds41_options_campaign', Path(__file__).resolve().parents[1]/'scripts/research/ds41_options_campaign.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(mode='exact'):
    arm = dict(id='test', apc_mode=mode, policy={}, expert_math='reference')
    if mode == 'disabled':
        arm['apc_reason'] = 'Adaptive partition identity disables publication'
    source = dict(checkpoint='model@'+'a'*64, source_revision='b'*40, extra_source_policy={}, source_expert_math='reference', source_snapshot_sha256='c'*64)
    result = dict(status='completed', passed=True, checkpoint=source['checkpoint'],
        source={'commit': source['source_revision'], 'source_snapshot_sha256': source['source_snapshot_sha256']},
        ordinary_reference={'passed': True}, requests={})
    for label in ['cold', 'warm', 'apc_warm']:
        result['requests'][label] = dict(response=dict(text='1\n2\n3\n4\n5\n',
            receipt=dict(route='ordinary', route_receipt={'selected': True}, prompt_tokens=100,
                         completion_tokens=32, cached_tokens=99 if label=='apc_warm' and mode=='exact' else 0,
                         cache='apcv2', elapsed_seconds=4, ttft_seconds=1),
            token_trace={'token_ids': list(range(32)), 'execution_widths': [1]*32}))
    return source, result, dict(owners_absent=True, source_unchanged=True), arm


def test_exact_and_disabled_profiles_have_different_cache_contracts():
    for mode in ('exact', 'disabled'):
        values=fixture(mode)
        assert m.assess_data(*values)['passed']
    source,result,release,arm=fixture()
    result['requests']['apc_warm']['response']['receipt']['cached_tokens']=0
    assert 'apc_warm_cache_frontier' in m.assess_data(source,result,release,arm)['failed_gates']


def test_raw_completion_without_cleanup_cannot_resume():
    source,result,release,arm=fixture()
    release['owners_absent']=False
    assert not m.assess_data(source,result,release,arm)['passed']
    release['owners_absent']=True
    result['requests']['warm']['response']['token_trace']['token_ids']=[100]*32
    assert 'ordinary_repeat_token_parity' in m.assess_data(source,result,release,arm)['failed_gates']


def test_counter_scheduling_scope_and_complete_plan():
    row=dict(status_before={'adapter':{'scheduled_dense_calls':{'fused':2}}},
             status_after={'adapter':{'scheduled_dense_calls':{'fused':5}}})
    assert m.observations(row)=={'adapter.scheduled_dense_calls.fused':3}
    assert all(s['max_tokens']==32 and s['mode']=='control' for s in m.plan())
    assert [s['clear_apc'] for s in m.plan()]==[True,True,False]


def test_manifest_requires_explicit_cache_profile_and_rejects_false_coherence():
    with pytest.raises(ValueError):m.validate_arm(dict(id='test'))
    with pytest.raises(ValueError):m.validate_arm(dict(id='test',apc_mode='disabled'))
    assert m.counting('1\n2\n3\n4\n5')
    assert not m.counting('1\n2\n4\n5\n6')
    assert not m.counting('Some text 1 2 3 4 5')


def test_owned_release_verdict_is_stronger_than_global_absence():
    source,result,release,arm=fixture()
    release.update(owners_absent=False,owned_release_verified=True)
    assert m.assess_data(source,result,release,arm)['gates']['ownership_released']
    release.update(owners_absent=True,owned_release_verified=False)
    assert not m.assess_data(source,result,release,arm)['gates']['ownership_released']


def test_stale_failed_summary_recovers_only_from_passing_current_raw(monkeypatch,tmp_path):
    source,result,release,arm=fixture()
    unit=tmp_path/'unit';unit.mkdir()
    (unit/'release-supplement.json').write_text('{"proof":"fixture"}')
    check=m.assess_data(source,result,release,arm)
    old={'passed':False,'failed_gates':['ownership_released']}
    prior={'unit':str(unit),'arm':arm,'status':'failed','assessment':old,'error':{'message':'old summary'}}
    monkeypatch.setattr(m,'assess_unit',lambda *args:check)
    assert m.reassess_prior(prior,arm)is True
    assert prior['status']=='passed'and prior['assessment']==check
    assert prior['audit_history'][0]['previous_assessment']==old
    assert prior['audit_history'][0]['previous_status']=='failed'
    assert prior['reassessment']['supplemental_release']is True
    assert 'release-supplement.json'in prior['reassessment']['evidence_sha256']
    assert prior['reassessment']['executed_gpu']is False
    assert 'error'not in prior


@pytest.mark.parametrize('outcome',['missing','malformed','failed'])
def test_incomplete_or_invalid_failed_raw_unit_requires_new_attempt(monkeypatch,tmp_path,outcome):
    *_,arm=fixture();prior={'unit':str(tmp_path),'arm':arm,'status':'failed','assessment':{'passed':False}}
    def assess(*args):
        if outcome=='missing':raise FileNotFoundError('missing')
        if outcome=='malformed':raise ValueError('bad supplement')
        return {'passed':False,'failed_gates':['ownership_released']}
    monkeypatch.setattr(m,'assess_unit',assess)
    assert m.reassess_prior(prior,arm)is False
    assert prior['status']=='failed'and 'reassessment'not in prior


def test_previously_passing_raw_receipt_remains_fail_closed(monkeypatch,tmp_path):
    *_,arm=fixture();prior={'unit':str(tmp_path),'arm':arm,'status':'passed','assessment':{'passed':True}}
    monkeypatch.setattr(m,'assess_unit',lambda *args:{'passed':False})
    with pytest.raises(ValueError,match='no longer passes'):m.reassess_prior(prior,arm)
    def missing(*args):raise FileNotFoundError('missing')
    monkeypatch.setattr(m,'assess_unit',missing)
    with pytest.raises(ValueError,match='cannot be revalidated'):m.reassess_prior(prior,arm)


def test_resume_never_launches_gpu_for_a_raw_passing_stale_summary(monkeypatch,tmp_path):
    source,result,release,arm=fixture();check=m.assess_data(source,result,release,arm)
    manifest={'arms':[arm]};output=tmp_path/'campaign';output.mkdir()
    report={'manifest_sha256':m.digest(manifest),'arms':{'test':{'unit':str(tmp_path/'unit'),
        'arm':arm,'status':'failed','assessment':{'passed':False}}}}
    m.atomic(output/'campaign.json',report)
    monkeypatch.setattr(m,'assess_unit',lambda *args:check)
    monkeypatch.setattr(m.subprocess,'Popen',lambda *args,**kwargs:pytest.fail('resume dispatched GPU'))
    report=m.run_campaign(tmp_path,manifest,output,python='unused')
    assert report['arms']['test']['status']=='passed'
    assert m.read(output/'campaign.json')['arms']['test']['audit_history'][0]['previous_status']=='failed'


def test_cpu_reconcile_does_not_rewrite_unit_evidence(monkeypatch,tmp_path):
    source,result,release,arm=fixture();check=m.assess_data(source,result,release,arm)
    unit=tmp_path/'unit';unit.mkdir();raw=unit/'results.json';raw.write_text('original raw receipt')
    path=tmp_path/'campaign.json';m.atomic(path,{'arms':{'test':{'unit':str(unit),'arm':arm,
        'status':'failed','assessment':{'passed':False}}}})
    monkeypatch.setattr(m,'assess_unit',lambda *args:check)
    monkeypatch.setattr(m.subprocess,'Popen',lambda *args,**kwargs:pytest.fail('CPU reconcile dispatched GPU'))
    assert m.reconcile_prior_summary(path,arm)['passed']
    assert raw.read_text()=='original raw receipt'


def test_residency_arms_bind_policy_and_keep_dynamic_parity_informational():
    source,result,release,arm=fixture()
    arm['expert_residency']={'residency':'split_v1','high_fraction':0.7,'pre_evict_slots':2}
    source['expert_policy']={'mode':'hot_original_q1','residency':'split_v1','high_fraction':0.7,'pre_evict_slots':2}
    assert m.assess_data(source,result,release,arm)['gates']['expert_residency_bound']
    source['expert_policy']['high_fraction']=0.6
    assert 'expert_residency_bound' in m.assess_data(source,result,release,arm)['failed_gates']
    dynamic=dict(arm,expert_residency={'residency':'dynamic_precision_v1'})
    with pytest.raises(ValueError,match='approximate'):
        m.validate_arm(dynamic)
    source,result,release,arm=fixture('disabled')
    arm.update(expert_residency={'residency':'dynamic_precision_v1'},token_parity='informational')
    source['expert_policy']={'residency':'dynamic_precision_v1'}
    result['requests']['warm']['response']['token_trace']['token_ids']=list(range(1,33))
    verdict=m.assess_data(source,result,release,arm)
    assert verdict['passed'] and verdict['informational_token_parity']['ordinary_repeat_token_parity'] is False
    with pytest.raises(ValueError,match='absolute'):
        m.validate_arm(dict(arm,hotset='relative.json'))


def test_expected_apc_cached_and_advanced_case_binding():
    validate_arm = m.validate_arm
    grid = dict(id='g', apc_mode='exact', expected_apc_cached=3072)
    assert validate_arm(grid)['expected_apc_cached'] == 3072
    with pytest.raises(ValueError):
        validate_arm(dict(id='g', apc_mode='disabled', apc_reason='x', expected_apc_cached=3072))
    case = {'draft_model': 'embedded_dspark', 'num_draft': 2}
    assert validate_arm(dict(id='d', apc_mode='disabled', apc_reason='x', advanced_case='draft_ab_serial_d2', policy=case))
    with pytest.raises(ValueError):
        validate_arm(dict(id='d', apc_mode='disabled', apc_reason='x', advanced_case='draft_ab_serial_d2', policy={}))
