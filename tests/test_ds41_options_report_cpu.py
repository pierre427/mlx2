"""Report joins preserve contract scope, negative attempts and timing bounds."""
import copy
import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/research'))
import ds41_options_report as report


def response():
    return {'text':'\n'.join(map(str,range(1,17)))+'\n','finish_reason':'length',
        'token_trace':{'token_ids':list(range(32)),'execution_widths':[1]*32},
        'receipt':{'prompt_tokens':100,'completion_tokens':32,'cached_tokens':0,'cache':'apcv2','route':'ordinary',
            'route_receipt':'unqualified','ttft_seconds':2.,'elapsed_seconds':10.}}


def make_unit(tmp_path,*,math='reference',observed=True):
    unit=tmp_path/'ds41-options-test-r1';unit.mkdir(parents=True);(unit/'source').mkdir()
    spec={'checkpoint':'test@'+'a'*64,'source_revision':'b'*40,'source_snapshot_sha256':'c'*64,
        'source_expert_math':math,'extra_source_policy':{'deepseek_source_dense_qat':'fused_fp32'},
        'prompt_tokens':100,'output_tokens':32,'prefill_step':128,'context_cap':16384}
    result={'checkpoint':spec['checkpoint'],'source':{'commit':spec['source_revision'],'source_snapshot_sha256':spec['source_snapshot_sha256']},
        'status':'completed','passed':True,'ordinary_reference':{'passed':True},'requests':{}}
    execution={'source_expert_math':'reference','execution_policy':spec['extra_source_policy']}
    for label in ('cold','warm','apc_warm'):
        before={'artifact':'a'*64,'execution':{**execution,'scheduled_dense_calls':{'source_dense_qat_fused':0}}}
        after={'artifact':'a'*64,'execution':{**execution,'scheduled_dense_calls':{'source_dense_qat_fused':1 if observed else 0}}}
        result['requests'][label]={'response':response(),'status_before':before,'status_after':after}
    for path,value in [('spec.json',spec),('results.json',result),('release-verified.json',{'owners_absent':True,'source_unchanged':True}),
        ('source/desktop-source.json',{'source_revision':spec['source_revision'],'source_sha256':spec['source_snapshot_sha256']})]:
        (unit/path).write_text(json.dumps(value))
    return unit


def requirement(option='deepseek_source_dense_qat',value='fused_fp32'):
    return {'id':'fixture','option':option,'value':value,'lane':'ordinary_text','policy':{option:value},'coverage_trigger':'actual dispatch','resources':[]}


def test_timing_uses_terminal_n_minus_one_and_never_incomplete_rows():
    value=response();rate=report.terminal_rates(value,completed=True)
    assert rate['decode_tokens_per_second']==31/8 and rate['prompt_tokens_per_TTFT_second']==50
    assert report.terminal_rates(value,completed=False)is None
    value['token_trace']['token_ids'].pop()
    assert report.terminal_rates(value,completed=True)is None


@pytest.mark.parametrize('ttft,elapsed',[(0,10),(float('nan'),10),(4,2)])
def test_invalid_timing_never_becomes_a_performance_number(ttft,elapsed):
    value=response();value['receipt'].update(ttft_seconds=ttft,elapsed_seconds=elapsed)
    assert report.terminal_rates(value,completed=True)is None


def test_actual_reference_math_diagnostic_binds_omitted_parser_key(tmp_path):
    unit=make_unit(tmp_path)
    row,rates=report.summarize_unit(unit)
    assert row['status']=='passed'and row['binding_gates']['policy_and_math_observed']
    assert len(rates)==3 and all(rate['qualified']is False for rate in rates)
    ledger=report.coverage([requirement()],[row],{})
    assert ledger[0]['coverage']=='positive'and ledger[0]['observed_used']


def test_mismatched_math_is_contract_failure_not_performance(tmp_path):
    unit=make_unit(tmp_path,math='routed_fused_fp32')
    row,rates=report.summarize_unit(unit)
    assert row['functional_passed']and row['status']=='contract_failure'
    assert not row['contract_passed']and not rates


def test_selected_without_observed_dispatch_is_partial(tmp_path):
    unit=make_unit(tmp_path,observed=False);row,_=report.summarize_unit(unit)
    ledger=report.coverage([requirement()],[row],{})
    assert ledger[0]['tested_successfully']and ledger[0]['coverage']=='partial'
    missing=report.coverage([requirement('deepseek_source_native_mx','original_pp_half_v1')],[row],{})
    assert missing[0]['coverage']=='missing'


def test_generic_fused_counter_does_not_prove_unrelated_fusion(tmp_path):
    row,_=report.summarize_unit(make_unit(tmp_path));row['policy']['deepseek_hc_epilogue']='fused'
    ledger=report.coverage([requirement('deepseek_hc_epilogue','fused')],[row],{})
    assert ledger[0]['coverage']=='partial'and not ledger[0]['observed_used']


def test_incomplete_unit_is_retained_without_crash_or_rates(tmp_path):
    unit=tmp_path/'ds41-options-incomplete';unit.mkdir()
    (unit/'spec.json').write_text('{"checkpoint":"incomplete"}')
    row,rates=report.summarize_unit(unit)
    assert row['status']=='setup_or_incomplete'and row['read_errors']and not rates


def test_failed_attempt_keeps_option_evidence_but_cannot_promote(tmp_path):
    unit=make_unit(tmp_path);value=json.loads((unit/'results.json').read_text());value.update(passed=False,status='failed')
    (unit/'results.json').write_text(json.dumps(value));row,rates=report.summarize_unit(unit)
    ledger=report.coverage([requirement()],[row],{})
    assert not rates and ledger[0]['coverage']=='partial'
    assert ledger[0]['all_attempt_units']==[unit.name]and not ledger[0]['positive_units']


def test_runtime_artifact_mismatch_does_not_inherit_spec_checkpoint(tmp_path):
    unit=make_unit(tmp_path);value=json.loads((unit/'results.json').read_text())
    for row in value['requests'].values():
        row['status_before']['artifact']=row['status_after']['artifact']='f'*64
    (unit/'results.json').write_text(json.dumps(value));summary,rates=report.summarize_unit(unit)
    assert not summary['binding_gates']['observed_artifact_matches']and not rates


def test_policy_and_math_must_match_in_one_observed_profile(tmp_path):
    unit=make_unit(tmp_path,math='routed_fused_fp32');value=json.loads((unit/'results.json').read_text())
    for row in value['requests'].values():
        row['status_before']['execution']['source_expert_math']='routed_fused_fp32'
        row['status_before']['execution']['execution_policy']['deepseek_source_dense_qat']='reference'
        row['status_after']['execution']['source_expert_math']='reference'
    (unit/'results.json').write_text(json.dumps(value));summary,rates=report.summarize_unit(unit)
    assert not summary['binding_gates']['policy_and_math_observed']and not rates


def _retarget(unit,option,value):
    spec=json.loads((unit/'spec.json').read_text());spec['extra_source_policy']={option:value};(unit/'spec.json').write_text(json.dumps(spec))
    result=json.loads((unit/'results.json').read_text())
    for request in result['requests'].values():
        for key in ('status_before','status_after'):request[key]['execution']['execution_policy']={option:value}
    (unit/'results.json').write_text(json.dumps(result));return result


def test_actual_helper_observer_fixes_alias_gap_without_promoting_unrelated_fusion(tmp_path):
    unit=make_unit(tmp_path);_retarget(unit,'deepseek_qat','fused_fp32')
    (unit/'policy-observer.json').write_text(json.dumps({'shutdown_snapshot':True,'calls_after_successful_return':{'model.qat.fused_fp32':19}}))
    row,_=report.summarize_unit(unit)
    assert report.coverage([requirement('deepseek_qat','fused_fp32')],[row],{})[0]['coverage']=='positive'
    row['observed_policies'].append({'deepseek_hc_epilogue':'fused'})
    assert report.coverage([requirement('deepseek_hc_epilogue','fused')],[row],{})[0]['coverage']=='partial'


def test_engram_format_and_actual_access_are_required_together(tmp_path):
    unit=make_unit(tmp_path);result=_retarget(unit,'deepseek_engram_cache','bf16_lut')
    for request in result['requests'].values():
        request['status_after']['execution']['engram']={'1':{'cache_format':'bf16_lut','requested_rows':24},'14':{'cache_format':'bf16_lut','requested_rows':24}}
    (unit/'results.json').write_text(json.dumps(result));row,_=report.summarize_unit(unit)
    assert report.coverage([requirement('deepseek_engram_cache','bf16_lut')],[row],{})[0]['coverage']=='positive'
    row['option_coverage_evidence']['last_status']['execution']['engram']['14']['requested_rows']=0
    assert report.coverage([requirement('deepseek_engram_cache','bf16_lut')],[row],{})[0]['coverage']=='partial'


def test_completion_advice_counter_cannot_substitute_for_actual_epochs(tmp_path):
    unit=make_unit(tmp_path);result=_retarget(unit,'deepseek_source_completion','bounded_2')
    for request in result['requests'].values():
        request['status_before']['execution']['scheduled_expert_calls']={'source_adaptive_bounded_completion_executions':0}
        request['status_after']['execution']['scheduled_expert_calls']={'source_adaptive_bounded_completion_executions':40}
    (unit/'results.json').write_text(json.dumps(result));row,_=report.summarize_unit(unit)
    assert report.coverage([requirement('deepseek_source_completion','bounded_2')],[row],{})[0]['coverage']=='partial'


def test_candidate_component_needs_real_branch_counters_not_just_output_parity():
    advanced={'schema':'mlx2.ds41-candidate-pruning-component.v1','passed':True,'source_unchanged':True,
        'cases':[{'passed':True,'oracle':{'passed':True},'gates':{'source_owners_unchanged':True},
                  'arms':{'candidate_only':{'selected_policy':'candidate_only','scheduled_kv_calls':{}}}}]}
    assert not report.candidate_component_observed(advanced)
    advanced['cases'][0]['arms']['candidate_only']['scheduled_kv_calls']['candidate_only_unscored_rows']=1
    assert report.candidate_component_observed(advanced)


def test_descendant_component_requires_paired_exact_nodes_and_actual_reuse():
    def arm(policy):
        return dict(policy=policy,all_nodes_exact=True,zero_io_host_routes_or_replay=True,
                    preparation=dict(policy=policy,builds=42,hits=150))
    a=dict(schema='mlx2.ds41-ast-batched-model-probe.v1',passed=True,completed=True,
        paired_kv_preparation=dict(all_raw_exact=True,all_no_io=True,source_model_identity_unchanged=True,
            arms=[arm('reference'),arm('descendant_reuse_v1')]))
    assert report.descendant_component_observed(a)
    a['paired_kv_preparation']['arms'][1]['preparation']['hits']=0
    assert not report.descendant_component_observed(a)
    a['paired_kv_preparation']['arms'][1]['preparation']['hits']=150
    a['paired_kv_preparation']['arms'][0]['all_nodes_exact']=False
    assert not report.descendant_component_observed(a)


def test_descendant_selection_and_passed_dspark_action_do_not_prove_reuse():
    row=dict(action='dspark_descendant',status='passed',advanced=dict(observed_used=True))
    assert not report._observed(requirement('deepseek_source_ast_kv_preparation','descendant_reuse_v1'),row)
    row['advanced']['descendant_preparation_component_observed']=True
    assert report._observed(requirement('deepseek_source_ast_kv_preparation','descendant_reuse_v1'),row)


def _native_component_fixture():
    spec=dict(checkpoint='test@fingerprint',source_revision='a'*40,source_snapshot_sha256='b'*64)
    receipt=dict(forward_law='fully_batched_v1',numerical_profile='source_ast_native_batched_segments_v1',
        tensor_tile_rows=16,attention_rows_per_call=16,expert_projection_rows_per_layer=16,
        layer_traversals=1,physical_target_invocations=1,attention_calls=40,grouped_moe_calls=40,
        shared_expert_calls=40,head_calls=1,serial_whole_model_calls=0,
        materialized_node_frontiers_during_forward=0,apc_publication=False,apcv2_lineage_bound=False,
        parents=[-1,0,1,2,3,4,0,6,7,8,9,0,11,12,13,14])
    equal=lambda:dict(equal=True)
    nodes=[dict(node=i,exact=True,cache=equal(),features=equal(),logits=equal())for i in range(16)]
    cases=[dict(label=label,passed=True,finite=True,status=[0]*16,
        raw={k:dict(equal=True,shape=shape)for k,shape in {'down':[96,5120],'middle':[96,2304],'output':[16,5120]}.items()})
        for label in ('original_shared16','q1_shared16','mixed_shared16','globally_distinct96')]
    controls=[dict(label=label,passed=True,unaffected_rows_raw=equal())for label in ('missing_cohort','invalid_id','duplicate_id')]
    a=dict(**spec,schema='mlx2.ds41-ast-batched-model-probe.v1',passed=True,completed=True,
        forward_law='fully_batched_v1',target_profile='source_ast_native_batched_segments_v1',
        APCv2_publication=False,exact_oracle_admitted=False,
        candidate=dict(all_nodes_exact=True,zero_io_host_routes_or_replay=True,receipt=receipt,nodes=nodes),
        moe_components=dict(**spec,schema='mlx2.ds41-ast-batched-moe-components.v1',passed=True,cases=cases,controls=controls),
        private_selected_state=dict(APCv2_publication=False,prefix_unchanged=equal(),cache=equal()),
        continuation=dict(ordinary_both_arms=True,prefix_unchanged=equal(),cache=equal(),logits=equal()))
    return a,spec


def test_native_component_requires_bound_shape_model_and_negative_controls():
    import copy
    a,spec=_native_component_fixture()
    assert report.native_batched_component_observed(a,spec)
    changes=[lambda a:a.update(source_snapshot_sha256='c'*64),
        lambda a:a['candidate']['receipt'].update(tensor_tile_rows=15),
        lambda a:a['candidate']['receipt'].update(serial_whole_model_calls=16),
        lambda a:a['candidate']['nodes'][7]['cache'].update(equal=False),
        lambda a:a['moe_components']['cases'][0]['raw']['output'].update(shape=[15,5120]),
        lambda a:a['moe_components']['controls'][0].update(passed=False),
        lambda a:a['continuation']['logits'].update(equal=False),
        lambda a:a.update(APCv2_publication=True),lambda a:a.update(exact_oracle_admitted=True)]
    for change in changes:
        bad=copy.deepcopy(a);change(bad)
        assert not report.native_batched_component_observed(bad,spec)
    assert not report.native_batched_component_observed(dict(passed=True,completed=True,forward_law='fully_batched_v1'),spec)


def test_passed_native_action_is_not_enough_for_fully_batched_coverage():
    row=dict(action='dspark_native_batched',status='passed',advanced=dict(observed_used=True))
    req=requirement('deepseek_source_dspark_target_forward','fully_batched_v1')
    assert not report._observed(req,row)
    row['advanced']['native_batched_component_observed']=True
    assert report._observed(req,row)
