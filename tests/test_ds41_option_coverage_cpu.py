"""synthetic@cpu-coverage-validator; no model/GPU work."""
from pathlib import Path
import importlib.util
import json

ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('ds41_option_coverage',ROOT/'scripts/research/ds41_option_coverage.py')
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)


def case(option,value,counters,*,observer=True,selected=True):
    arm={'option':option,'value':value,'policy':{option:value}}
    data={'complete':True,'output_cells':3,'counters':counters,'observer':{}if observer else None,
          'last_status':{'execution':{'execution_policy':{option:value if selected else 'reference'}}}}
    if observer:data['observer']={'calls_after_successful_return':counters}
    return m.assess(arm,data)


def test_candidate_scoring_is_distinct_from_pruning_coverage():
    r=case('deepseek_candidate_score','candidate_only',{'candidate_only_score_rows':100,'candidate_only_unscored_rows':0})
    assert r['classification']=='observed_scheduled_partial_scope'and r['requeue_required']
    assert '16384'in r['requeue_instruction']
    r=case('deepseek_candidate_score','candidate_only',{'candidate_only_score_rows':100,'candidate_only_unscored_rows':7})
    assert r['classification']=='observed_scheduled'and not r['requeue_required']


def test_ced_calls_without_skipped_rows_need_distinct_large_forward():
    r=case('deepseek_ced','dependency_exact',{'ced_prefill_calls':4,'ced_skipped_layer_rows':0})
    assert r['requeue_required']and 'prefill-step256'in r['requeue_instruction']


def test_adaptive_completion_eligibility_does_not_prove_actual_bounded_epochs():
    r=case('deepseek_source_completion','bounded_2',{'source_adaptive_bounded_completion_executions':100})
    assert r['classification']=='selected_unobserved'and r['missing_counter_groups']
    r=case('deepseek_source_completion','bounded_2',{'completion.submitted_epochs':80,'completion.retired_epochs':80})
    assert r['classification']=='observed_scheduled'


def test_generic_policy_needs_actual_helper_observer_not_selection():
    r=case('deepseek_qat','fused_fp32',{},observer=False)
    assert r['classification']=='selected_uninstrumented'
    r=case('deepseek_qat','fused_fp32',{'model.qat.fused_fp32':40})
    assert r['classification']=='observed_scheduled'
    assert not r['scheduling_is_GPU_completion_proof']and not r['qualification']


def test_fallback_only_and_wrong_selection_are_not_positive_coverage():
    r=case('deepseek_source_q_b_prefill','tile64x32_raw',{'source_q_b_prefill_fallback_shape':40})
    assert r['classification']=='fallback_only_or_partial_target_coverage'
    r=case('deepseek_source_q_b_prefill','tile64x32_raw',{'source_q_b_prefill_tile64x32_raw':40},selected=False)
    assert r['classification']=='selection_mismatch'and r['requeue_required']


def test_phase_and_status_counters_are_not_double_counted(tmp_path):
    spec={'checkpoint':'synthetic@cpu-coverage-validator','source_revision':'a'*40,'source_snapshot_sha256':'b'*64}
    results={'passed':True,'requests':{'cold':{'status_before':{'execution':{'scheduled_kv_calls':{'native_packed_local_rows':1}}},
        'status_after':{'execution':{'scheduled_kv_calls':{'native_packed_local_rows':5}}}}}}
    (tmp_path/'spec.json').write_text(json.dumps(spec));(tmp_path/'results.json').write_text(json.dumps(results))
    rows=[{'label':'cold','input_rows':99,'kv_scheduled_delta':{'native_packed_local_rows':4}},
          {'label':'uncompleted','input_rows':1,'kv_scheduled_delta':{'native_packed_local_rows':1000}}]
    (tmp_path/'model-phases.jsonl').write_text('\n'.join(json.dumps(r)for r in rows)+'\n{"partial"')
    loaded=m.load_unit(tmp_path)
    assert loaded['counters']['native_packed_local_rows']==4
    assert loaded['partial_phase_lines_ignored']==1 and loaded['input_rows']==[99]


def test_outer_status_positive_is_not_shadowed_by_inner_phase_zero(tmp_path):
    spec={'checkpoint':'synthetic@cpu-coverage-validator','source_revision':'a'*40,'source_snapshot_sha256':'b'*64}
    result={'passed':True,'requests':{'cold':{'status_before':{'execution':{'scheduled_kv_calls':{'source_canonical_m1_prefill_tokens':0}}},
        'status_after':{'execution':{'scheduled_kv_calls':{'source_canonical_m1_prefill_tokens':99}}}}}}
    (tmp_path/'spec.json').write_text(json.dumps(spec));(tmp_path/'results.json').write_text(json.dumps(result))
    (tmp_path/'model-phases.jsonl').write_text(json.dumps({'label':'cold','input_rows':1,'kv_scheduled_delta':{'source_canonical_m1_prefill_tokens':0}})+'\n')
    assert m.load_unit(tmp_path)['counters']['source_canonical_m1_prefill_tokens']==99
