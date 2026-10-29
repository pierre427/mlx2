# SPDX-License-Identifier: MIT
"""CPU-only DS4.1 receipt aggregation; never dispatches a model or GPU job.

Retains every attempt and separates selected policy, observed scheduling,
functional outcome and evidence contract. No result is promoted to formal
qualification or a controlled performance claim by this reporting tool.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).resolve().parent))
from mlx2.desktop_campaign_release import release_evidence
from ds41_options_campaign import assess_unit,observations
from ds41_option_catalog import catalog
from ds41_option_coverage import load_unit as load_coverage_unit, assess as assess_coverage

SCHEMA='mlx2.ds41-options-report.v1'
SETTINGS=('prompt_tokens','output_tokens','context_cap','prefill_step','apc_bytes','persist','expert_cache_bytes')
REPLACEMENTS={
 'ds41-options-baseline-20261027-r1':('ds41-options-baseline-20261027-r2','Harness max_tokens contract repair'),
 'ds41-options-vision-red-20261027-r1':('ds41-options-vision-red-20261027-r2','Actual image assessment replaces prior failed attempt; raw verdict retained'),
 'ds41-options-physical-b2-20261027-r1':('ds41-options-physical-b2-20261027-r2','Source snapshot validation repair'),
 'ds41-options-resident-gather-20261027-r1':('ds41-options-resident-gather-20261027-r2','Admitted BF16 WoA prerequisite'),
 'ds41-options-resident-prefill-20261027-r1':('ds41-options-resident-prefill-20261027-r3','Raw capacity/slack comparison was methodological; maintained live-state publication comparison used later'),
 'ds41-options-resident-prefill-20261027-r2':('ds41-options-resident-prefill-20261027-r3','99-row fallback negative replaced by12-row supported occupancy; different profile retained'),
 'ds41-options-dspark-ast-20261027-r1':('ds41-options-dspark-ast-20261027-r4','Admitted long-prefix fixture replaces unsupported index geometry'),
 'ds41-options-dspark-ast-20261027-r2':('ds41-options-dspark-ast-20261027-r4','Private raw gate failure retained; later BF16 boundary repair'),
 'ds41-options-dspark-ast-20261027-r3':('ds41-options-dspark-ast-20261027-r4','Real private-forward BF16 boundary defect repaired; raw exact gate and matched-serving proof rerun'),
}


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _finite(x):return type(x)in(int,float)and math.isfinite(x)
def _invalid_constant(value):raise ValueError('Nonfinite JSON constant: '+value)
def _read(path,errors):
    try:
        value=json.loads(Path(path).read_text(),parse_constant=_invalid_constant)
        if type(value)is not dict:raise ValueError('receipt must be an object')
        return value
    except (OSError,ValueError)as error:
        errors.append({'path':str(path),'error':str(error)});return {}


def _atomic(path,value):
    path=Path(path)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False)as stream:
        name=stream.name;json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(name,path)


def _pairs(value,path=''):
    if type(value)is dict:
        for key,item in value.items():
            here=path+'.'+key if path else key
            yield here,key,item
            if type(item)in(dict,list):yield from _pairs(item,here)
    elif type(value)is list:
        for i,item in enumerate(value):
            if type(item)in(dict,list):yield from _pairs(item,path+'.'+str(i))


def terminal_rates(response,*,completed):
    """Derive only terminal server timing; TTFT is not pure prompt compute."""
    receipt=response.get('receipt',{});n=receipt.get('completion_tokens');p=receipt.get('prompt_tokens')
    elapsed=receipt.get('elapsed_seconds');ttft=receipt.get('ttft_seconds');cached=receipt.get('cached_tokens')
    trace=response.get('token_trace',receipt.get('token_trace',{}));ids=trace.get('token_ids',response.get('token_ids'))
    valid=(completed is True and bool(response.get('finish_reason'))and type(n)is int and n>0
        and type(p)is int and p>0 and type(cached)is int and 0<=cached<=p
        and type(ids)is list and len(ids)==n and _finite(elapsed)and _finite(ttft)and 0<ttft<=elapsed
        and bool(receipt.get('route_receipt')))
    if not valid:return None
    delta=elapsed-ttft
    return dict(prompt_tokens=p,output_tokens=n,cached_tokens=cached,TTFT_seconds=ttft,
        elapsed_seconds=elapsed,prompt_tokens_per_TTFT_second=p/ttft,
        decode_tokens_per_second=(n-1)/delta if n>1 and delta>0 else None,
        HTTP_wall_seconds=response.get('HTTP_seconds',response.get('client_elapsed_seconds'))if _finite(response.get('HTTP_seconds',response.get('client_elapsed_seconds')))else None,
        route=receipt.get('route'),profile=receipt.get('profile'),
        timing_scope='single terminal observation; prompt/TTFT includes queue, host and first-token overhead; decode=(N-1)/(elapsed-TTFT), including terminal work')


def _responses(value,path='result'):
    if type(value)is dict:
        if type(value.get('receipt'))is dict and type(value.get('text'))is str:
            yield path,value
            return
        for key,item in value.items():
            if key not in ('status_before','status_after','status_ready','environment_before','environment_after','events','services'):
                yield from _responses(item,path+'.'+key)
    elif type(value)is list:
        for i,item in enumerate(value):yield from _responses(item,path+'.'+str(i))


def source_binding(spec,result,descriptor):
    source=result.get('source',{});checkpoint=spec.get('checkpoint');revision=spec.get('source_revision');snapshot=spec.get('source_snapshot_sha256')
    return dict(checkpoint_named=type(checkpoint)is str and bool(re.fullmatch('.+@[0-9a-f]{64}',checkpoint)),
        source_revision_named=type(revision)is str and bool(re.fullmatch('[0-9a-f]{40}',revision)),
        source_snapshot_named=type(snapshot)is str and bool(re.fullmatch('[0-9a-f]{64}',snapshot)),
        checkpoint_matches=result.get('checkpoint')==checkpoint,
        source_revision_matches=source.get('commit',result.get('source_revision'))==revision,
        source_snapshot_matches=source.get('source_snapshot_sha256',result.get('source_snapshot_sha256'))==snapshot,
        frozen_descriptor_matches=descriptor.get('source_revision')==revision and descriptor.get('source_sha256',descriptor.get('source_snapshot_sha256'))==snapshot)


def candidate_component_observed(advanced):
    if (advanced.get('schema')!='mlx2.ds41-candidate-pruning-component.v1'
            or advanced.get('passed')is not True or advanced.get('source_unchanged')is not True):return False
    cases=advanced.get('cases',[])
    return bool(cases)and all(case.get('passed')is True and case.get('oracle',{}).get('passed')is True
        and case.get('gates',{}).get('source_owners_unchanged')is True
        and case.get('arms',{}).get('candidate_only',{}).get('selected_policy')=='candidate_only'
        and case.get('arms',{}).get('candidate_only',{}).get('scheduled_kv_calls',{}).get('candidate_only_unscored_rows',0)>0
        for case in cases)


def descendant_component_observed(advanced):
    paired=advanced.get('paired_kv_preparation',{})
    if (advanced.get('schema')!='mlx2.ds41-ast-batched-model-probe.v1'
            or advanced.get('passed')is not True or advanced.get('completed')is not True
            or paired.get('all_raw_exact')is not True or paired.get('all_no_io')is not True
            or paired.get('source_model_identity_unchanged')is not True):return False
    arms={arm.get('policy'):arm for arm in paired.get('arms',[])}
    if set(arms)!={'reference','descendant_reuse_v1'}:return False
    candidate=arms['descendant_reuse_v1'];preparation=candidate.get('preparation',{})
    return (all(a.get('all_nodes_exact')is True and a.get('zero_io_host_routes_or_replay')is True for a in arms.values())
        and preparation.get('policy')=='descendant_reuse_v1'
        and preparation.get('builds',0)>0 and preparation.get('hits',0)>0)


def native_batched_component_observed(advanced,spec):
    """Private native-M16 functional proof; never an ordinary/APCv2 oracle."""
    if (advanced.get('schema')!='mlx2.ds41-ast-batched-model-probe.v1'
            or advanced.get('passed')is not True or advanced.get('completed')is not True
            or advanced.get('forward_law')!='fully_batched_v1'
            or advanced.get('target_profile')!='source_ast_native_batched_segments_v1'
            or advanced.get('APCv2_publication')is not False
            or advanced.get('exact_oracle_admitted')is not False):return False
    for key in ('checkpoint','source_revision','source_snapshot_sha256'):
        if not spec.get(key)or advanced.get(key)!=spec[key]:return False
    candidate=advanced.get('candidate',{});receipt=candidate.get('receipt',{})
    expected=dict(forward_law='fully_batched_v1',numerical_profile='source_ast_native_batched_segments_v1',
        tensor_tile_rows=16,attention_rows_per_call=16,expert_projection_rows_per_layer=16,
        layer_traversals=1,physical_target_invocations=1,attention_calls=40,
        grouped_moe_calls=40,shared_expert_calls=40,head_calls=1,serial_whole_model_calls=0,
        materialized_node_frontiers_during_forward=0,apc_publication=False,apcv2_lineage_bound=False)
    if any(receipt.get(k)!=v for k,v in expected.items()):return False
    if receipt.get('parents')!=[-1,0,1,2,3,4,0,6,7,8,9,0,11,12,13,14]:return False
    nodes=candidate.get('nodes',[])
    if (candidate.get('all_nodes_exact')is not True or candidate.get('zero_io_host_routes_or_replay')is not True
            or len(nodes)!=16 or {n.get('node')for n in nodes}!=set(range(16))
            or not all(n.get('exact')is True and all(n.get(k,{}).get('equal')is True for k in ('cache','features','logits'))for n in nodes)):return False
    component=advanced.get('moe_components',{})
    if component.get('schema')!='mlx2.ds41-ast-batched-moe-components.v1'or component.get('passed')is not True:return False
    if any(component.get(k)!=spec[k]for k in ('checkpoint','source_revision','source_snapshot_sha256')):return False
    cases=component.get('cases',[]);controls=component.get('controls',[])
    if {c.get('label')for c in cases}!={'original_shared16','q1_shared16','mixed_shared16','globally_distinct96'}:return False
    shapes={'down':[96,5120],'middle':[96,2304],'output':[16,5120]}
    if not all(c.get('passed')is True and c.get('finite')is True and c.get('status')==[0]*16
        and all(c.get('raw',{}).get(k,{}).get('equal')is True and c['raw'][k].get('shape')==shape for k,shape in shapes.items())for c in cases):return False
    if {c.get('label')for c in controls}!={'missing_cohort','invalid_id','duplicate_id'}or not all(c.get('passed')is True and c.get('unaffected_rows_raw',{}).get('equal')is True for c in controls):return False
    private=advanced.get('private_selected_state',{});continuation=advanced.get('continuation',{})
    return (private.get('APCv2_publication')is False and private.get('prefix_unchanged',{}).get('equal')is True
        and private.get('cache',{}).get('equal')is True and continuation.get('ordinary_both_arms')is True
        and continuation.get('prefix_unchanged',{}).get('equal')is True
        and all(continuation.get(k,{}).get('equal')is True for k in ('cache','logits')))


def summarize_unit(unit,arm=None):
    errors=[];spec=_read(unit/'spec.json',errors);result=_read(unit/'results.json',errors)
    descriptor=_read(unit/'source/desktop-source.json',errors)
    try:release=release_evidence(unit)
    except (OSError,ValueError,TypeError,KeyError,AttributeError)as error:
        release={};errors.append({'path':str(unit/'release-verified.json'),'error':str(error)})
    binding=source_binding(spec,result,descriptor)
    actual=[];artifacts=[]
    def take(execution,settings=None):
        p=execution.get('execution_policy')
        if type(p)is dict:
            p=dict(p)
            for key,value in (settings or {}).get('execution_policy',{}).items():p.setdefault(key,value)
            # Old parser receipts omit an implicit reference math key; the
            # separate adapter diagnostic still records its actual selection.
            if type(execution.get('source_expert_math'))is str:
                p['deepseek_source_expert_math']=execution['source_expert_math']
            # The parser removes the generic draft selector from kernel
            # policy; its instantiated drafter identity remains authoritative.
            speculation=execution.get('speculation')or{}
            if type(speculation)is dict and speculation.get('kind')=='adaptive_copy_chain'and execution.get('route')=='external_draft':
                p['draft_model']='prompt_lookup'
            actual.append(p)
        if execution.get('effective_artifact'):artifacts.append(execution['effective_artifact'])
    for response_row in result.get('requests',{}).values():
        if type(response_row)is dict:
            for key in ('status_after','status_before'):
                status=response_row.get(key,{})
                if status.get('artifact'):artifacts.append(status['artifact'])
                take(status.get('execution',{}),status.get('settings',{}))
    advanced=result.get('advanced',{})
    if type(advanced)is dict:
        for key in ('status_after','status_before'):
            status=advanced.get(key,{})
            if status.get('artifact'):artifacts.append(status['artifact'])
            take(status.get('execution',{}),status.get('settings',{}))
    if not actual:
        status=result.get('status_ready',{})
        if status.get('artifact'):artifacts.append(status['artifact'])
        take(status.get('execution',{}),status.get('settings',{}))
    control=result.get('ordinary_reference',{})
    if not actual:take(control.get('diagnostics',{}))
    expected={**spec.get('extra_source_policy',{}),'deepseek_source_expert_math':spec.get('source_expert_math','reference')}
    if spec.get('implicit_defaults'):expected.update(spec.get('expected_defaults',{}))
    policy_bound=bool(actual)and any(all(p.get(key)==value for key,value in expected.items())for p in actual)
    binding['policy_and_math_observed']=policy_bound
    binding['observed_artifact_matches']=bool(artifacts)and any(value==str(spec.get('checkpoint','')).rsplit('@',1)[-1]for value in artifacts)
    release_ok=release.get('owned_release_verified',release.get('owners_absent'))is True
    binding.update(ownership_released=release_ok,source_unchanged=release.get('source_unchanged')is True)
    assessment=None
    advanced_lane=bool(result.get('action')or spec.get('advanced_case')or advanced)
    if not advanced_lane and result.get('requests'):
        if arm is None:
            arm=dict(id='report',apc_mode='exact'if expected.get('deepseek_source_prefill_partition')=='canonical_m1_v1'else'disabled',
                apc_reason='Adaptive partitions explicitly disable prefix reuse',policy=spec.get('extra_source_policy',{}),expert_math=spec.get('source_expert_math','reference'))
        try:assessment=assess_unit(unit,arm)
        except (OSError,ValueError,TypeError,KeyError,AttributeError)as error:errors.append({'path':str(unit),'error':'assessment: '+str(error)})
    functional=True if result.get('passed')is True else None
    contract=all(binding.values())and(assessment is None or not any(not value for key,value in assessment.get('gates',{}).items()if 'functional'not in key))
    if assessment is not None:
        quality=[value for key,value in assessment.get('gates',{}).items()if 'functional'in key]
        functional=all(quality)if quality else None
    advanced_gates=advanced.get('assessment',{}).get('gates',{})if type(advanced)is dict else{}
    if type(advanced_gates)is dict and advanced_gates:
        quality=[value for key,value in advanced_gates.items()if key.endswith('_quality')or'functional'in key]
        if quality:functional=all(quality)
        contract=contract and all(value is True for key,value in advanced_gates.items()if not key.endswith('_quality')and'functional'not in key)
    completed=result.get('status')in('completed','complete')
    if not result:status='setup_or_incomplete'
    elif not completed and result.get('status')not in('failed','error'):status='running'
    elif completed and not release:status='awaiting_release_evidence'
    elif completed and functional is True and contract and result.get('passed')is True:status='passed'
    elif functional is False:status='functional_failure'
    elif completed and not contract:status='contract_failure'
    elif control.get('passed')is not True:status='setup_failure'
    elif 'unsupported'in str(result.get('error',{})).lower():status='unsupported_or_fixture_refusal'
    else:status='feature_or_coverage_failure'
    if unit.name=='ds41-options-baseline-20261027-r1':status='methodological_failure'
    if unit.name=='ds41-options-resident-prefill-20261027-r1':status='methodological_failure'
    if unit.name=='ds41-options-physical-b2-20261027-r1':status='methodological_failure'
    if unit.name=='ds41-options-resident-prefill-20261027-r2':status='negative_fallback_coverage'
    if unit.name in ('ds41-options-dspark-ast-20261027-r2','ds41-options-dspark-ast-20261027-r3'):status='feature_correctness_failure'
    counters={}
    for row in result.get('requests',{}).values():
        if type(row)is dict:
            for key,value in observations(row).items():counters[key]=counters.get(key,0)+value
    coverage_data=load_coverage_unit(unit)
    last_execution=coverage_data.get('last_status',{}).get('execution',{})
    coverage_data['last_status']={'execution':{key:last_execution.get(key,{})for key in ('execution_policy','engram')}}
    observer=coverage_data.get('observer')
    if observer:coverage_data['observer']={key:observer.get(key)for key in ('shutdown_snapshot','calls_after_successful_return')}
    coverage_data.pop('source',None)
    component_candidate=candidate_component_observed(advanced)if type(advanced)is dict else False
    component_descendant=descendant_component_observed(advanced)if type(advanced)is dict else False
    component_native=native_batched_component_observed(advanced,spec)if type(advanced)is dict else False
    policies=[]
    for p in actual:
        selected={key:value for key,value in p.items()if key.startswith('deepseek_')and type(value)in(str,int,bool)}
        if selected not in policies:policies.append(selected)
    if type(advanced)is dict and advanced.get('observed_used')is True:
        derived={}
        if advanced.get('expert_law'):derived['deepseek_source_dspark_target_expert_law']=advanced['expert_law']
        if advanced.get('forward_law'):derived['deepseek_source_dspark_target_forward']=advanced['forward_law']
        if any(tile.get('observed_used')is True for tile in advanced.get('ast_tree_tiles',[])):derived['deepseek_source_dspark_ast']='tile3x5_v1'
        if derived:policies.append(derived)
    if component_candidate:policies.append({'deepseek_candidate_score':'candidate_only'})
    if component_descendant:policies.append({'deepseek_source_ast_kv_preparation':'descendant_reuse_v1'})
    if component_native:policies.append({'deepseek_source_dspark_target_forward':'fully_batched_v1'})
    advanced_summary={key:value for key,value in advanced.items()if type(value)in(str,int,float,bool)and key not in('oracle_path',)}if type(advanced)is dict else{}
    if type(advanced)is dict:
        advanced_summary['assessment']=advanced.get('assessment')
        advanced_summary['candidate_pruning_component_observed']=component_candidate
        advanced_summary['descendant_preparation_component_observed']=component_descendant
        advanced_summary['native_batched_component_observed']=component_native
        advanced_summary['cases']=[{key:value for key,value in case.items()if key in('variant','gates','passed','counters')}for case in advanced.get('cases',[])if type(case)is dict]
    row=dict(unit=str(unit),name=unit.name,status=status,completed=completed,functional_passed=functional,
        contract_passed=contract,binding_gates=binding,checkpoint=spec.get('checkpoint'),source_revision=spec.get('source_revision'),
        source_snapshot_sha256=spec.get('source_snapshot_sha256'),policy=expected,observed_policies=policies,
        settings={key:spec.get(key)for key in SETTINGS},expert_policy=spec.get('expert_policy'),
        lane='performance_ladder'if spec.get('ladder_enabled')else'advanced'if advanced_lane else'ordinary_100_32'if spec.get('prompt_tokens')==100 and spec.get('output_tokens')==32 else'ordinary_other_shape',
        action=result.get('action',spec.get('advanced_case')),advanced=advanced_summary,positive_scheduling_counters={k:v for k,v in counters.items()if v>0},
        ordinary_reference_passed=control.get('passed')is True,error=result.get('error'),read_errors=errors,
        failed_gates=(assessment.get('failed_gates',[])if assessment else[k for k,v in binding.items()if not v])+['advanced.'+key for key,value in advanced_gates.items()if value is not True],
        resource_proofs={key:{name:value for name,value in result.get(key,{}).items()if type(value)in(str,bool,int,float)or name in ('counter_delta','counter_keys')}for key in ('canonical_prefix','pinned_expert_cache','implicit_defaults','heldout_quality')if type(result.get(key))is dict},
        option_coverage_evidence=coverage_data,
        implemented=True,selected=bool(actual),observed_used=bool(counters)or bool(component_candidate)or(bool(advanced_summary.get('observed_used'))),observed_used_scope='Unit model execution; individual option dispatch is classified separately in option_ledger',qualified=False,performance_qualified=False,
        release_supplement=release.get('release_supplement'),raw_evidence_sha256={p.name:_sha(p)for p in (unit/'spec.json',unit/'results.json',unit/'release-verified.json')if p.is_file()})
    rates=[]
    for path,response in _responses(result):
        rate=terminal_rates(response,completed=completed and functional and contract)
        if rate:rates.append(dict(unit=str(unit),response_path=path,checkpoint=row['checkpoint'],source_revision=row['source_revision'],source_snapshot_sha256=row['source_snapshot_sha256'],lane=row['lane'],policy=expected,settings=row['settings'],qualified=False,performance_qualified=False,**rate))
    for cell_path in sorted((unit/'ladder').glob('context-*/result.json')):
        cell=_read(cell_path,errors)
        for path,response in _responses(cell,'ladder.'+cell_path.parent.name):
            rate=terminal_rates(response,completed=completed and functional and contract and cell.get('status')=='complete')
            if rate:rates.append(dict(unit=str(unit),response_path=path,checkpoint=row['checkpoint'],source_revision=row['source_revision'],source_snapshot_sha256=row['source_snapshot_sha256'],lane='performance_ladder',policy=expected,settings=row['settings'],qualified=False,performance_qualified=False,**rate))
    return row,rates


def _observed(requirement,unit):
    key,value=requirement['option'],requirement['value'];action=unit.get('action')or'';advanced=unit.get('advanced',{})
    if key in('deepseek_vision','deepseek_physical_batch','deepseek_verify_commit'):
        return type(advanced.get('assessment'))is dict and advanced['assessment'].get('observed_used')is True
    if key=='deepseek_source_ast_kv_preparation':
        return advanced.get('descendant_preparation_component_observed')is True
    if key=='deepseek_source_dspark_target_forward'and value=='fully_batched_v1':
        return advanced.get('native_batched_component_observed')is True
    if key.startswith('deepseek_source_dspark_'):
        return advanced.get('observed_used')is True or(action.startswith('dspark_')and unit['status']=='passed')
    if key in('deepseek_source_resident_gather','deepseek_source_resident_prefill'):
        return action.startswith('resident_')and bool(advanced.get('cases'))and all(case.get('passed')is True for case in advanced['cases'])
    if key=='deepseek_candidate_score'and advanced.get('candidate_pruning_component_observed')is True:
        return True
    # The shared strict detector joins model phases, completed status deltas,
    # actual custom-helper observations and Engram format/access metadata.
    # It requires the target branch, not a substring or eligibility counter.
    result=assess_coverage(requirement,unit.get('option_coverage_evidence',{}))
    return result['classification']in('observed_scheduled','observed_format_access')


def coverage(requirements,units,selected_defaults):
    rows=[]
    for requirement in requirements:
        key,value=requirement['option'],requirement['value']
        candidates=[u for u in units if any(p.get(key)==value for p in u['observed_policies'])or u['policy'].get(key)==value]
        passed=[u for u in candidates if u['status']=='passed']
        observed=[u for u in passed if _observed(requirement,u)]
        rows.append(dict(id=requirement['id'],option=key,value=value,lane=requirement['lane'],
            prerequisites={k:v for k,v in requirement['policy'].items()if k!=key},coverage_trigger=requirement['coverage_trigger'],
            resources=requirement['resources'],coverage='positive'if observed else'partial'if candidates else'missing',
            tested_successfully=bool(passed),observed_used=bool(observed),selected_default=selected_defaults.get(key)==value,
            qualified=False,observation_scopes=[dict(unit=u['name'],scope=('checkpoint_seeded_synthetic_component; no full long-context model claim'if u.get('advanced',{}).get('candidate_pruning_component_observed')else 'private_nativeM16_component_model; no ordinary oracle or APCv2 qualification'if u.get('advanced',{}).get('native_batched_component_observed')else 'paired16row_component; no admitted serving oracle'if u.get('advanced',{}).get('descendant_preparation_component_observed')else u.get('lane')))for u in observed],positive_units=[u['name']for u in observed],passing_selected_units=[u['name']for u in passed],
            all_attempt_units=[u['name']for u in candidates],limits='One-option/declared-combination bounded evidence only; missing triggers, fallback paths and untested shapes remain outside coverage'))
    return rows


def aggregate(root):
    root=Path(root).resolve();run_root=root/'qualification/runs';units=[];rates=[];campaigns=[];arm_by_unit={};errors=[]
    for path in sorted(run_root.glob('ds41-options*/campaign.json')):
        value=_read(path,errors);entries=value.get('arms',value.get('cases',{}));summary=[]
        if type(entries)is dict:
            for key,entry in entries.items():
                if type(entry)is not dict:continue
                assessment=entry.get('assessment')if type(entry.get('assessment'))is dict else{}
                summary.append(dict(id=key,unit=entry.get('unit'),status=entry.get('status'),passed=assessment.get('passed',entry.get('passed'))))
                if entry.get('unit')and type(entry.get('arm'))is dict:arm_by_unit[str(Path(entry['unit']).resolve())]=entry['arm']
        campaigns.append(dict(path=str(path),sha256=_sha(path),entries=summary,summary_is_not_raw_evidence=True))
    for unit in sorted(run_root.glob('ds41-options*')):
        if not unit.is_dir()or not(unit/'spec.json').exists():continue
        try:
            row,rr=summarize_unit(unit,arm_by_unit.get(str(unit.resolve())));units.append(row);rates+=rr
        except Exception as error:
            errors.append({'path':str(unit),'error':str(error)});units.append(dict(name=unit.name,unit=str(unit),status='unreadable_or_incomplete',policy={},observed_policies=[],qualified=False,performance_qualified=False))
    by_name={u['name']:u for u in units};relations=[]
    for old,(new,reason)in REPLACEMENTS.items():
        if old in by_name:
            relation=dict(prior=old,replacement=new,replacement_present=new in by_name,replacement_passed=by_name.get(new,{}).get('status')=='passed',reason=reason)
            if new in by_name:
                relation['same_declared_profile']=all(by_name[old].get(k)==by_name[new].get(k)for k in('checkpoint','policy','settings'))
            relations.append(relation)
    canonical_prior='ds41-options-source-prefill-partition--canonical-m1-v1-080ff6c0-r1'
    if canonical_prior in by_name:
        for new,row in by_name.items():
            if row['status']=='passed'and(new.startswith('ds41-options-canonical-requeue-')or new.startswith('ds41-options-combined-canonical-')):
                relations.append(dict(prior=canonical_prior,replacement=new,replacement_present=True,replacement_passed=True,
                    same_declared_profile=all(by_name[canonical_prior].get(k)==row.get(k)for k in('checkpoint','policy','settings')),
                    reason='Canonical preparation retry/combined canonical evidence; math and settings remain separately bound'))
    cat=catalog()
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter
    defaults=DeepSeekV41SourceAdapter.default_route_execution_policy.get('ordinary',{})
    ledger=coverage(cat['arms'],units,defaults)
    profile_coverage={lane:dict(unit_count=sum(u.get('lane')==lane for u in units),status_counts=dict(Counter(u['status']for u in units if u.get('lane')==lane)),profiles=[{key:u.get(key)for key in ('name','checkpoint','source_revision','source_snapshot_sha256','settings','policy','action','status')}for u in units if u.get('lane')==lane])for lane in sorted({u.get('lane','unknown')for u in units})}
    return dict(schema=SCHEMA,generated_at_unix=time.time(),root=str(root),reporter_sha256=_sha(__file__),catalog_source_root=str(ROOT),catalog_source_revision=cat['source_revision'],catalog_owning_source_sha256=cat['owning_source_sha256'],profile_coverage=profile_coverage,qualification=False,performance_qualified=False,
        preflight_executed=False,scope='Current local DS4.1 options receipts; all attempts preserved; no all-combinations or long-context qualification claim',
        status_counts=dict(Counter(u['status']for u in units)),unit_count=len(units),units=units,performance_rows=rates,
        catalog_requirement_count=cat['arm_count'],option_count=cat['option_count'],option_coverage_counts=dict(Counter(r['coverage']for r in ledger)),
        option_ledger=ledger,source_choices=cat['choices'],serving_tunables=cat['serving_tunables'],
        selected_defaults=defaults,default_scope='Guarded ordinary Source Hot60GB/Q1g64 adaptive B1 profile; explicit compatible operator choices win; unqualified',
        campaign_summaries=campaigns,replacements=relations,read_errors=errors,
        limits=['Single repetition without thermal controls; recorded rates are observational and may include paging/host/queue effects.',
            'Positive scheduling counters prove dispatch scheduling, not GPU timestamp completion. Selected-without-observed is partial.',
            'Different numerical policies, source snapshots, prefill partitions, precision maps and shapes are distinct profiles.',
            'Advanced image, batch, resident and AST cells do not replace ordinary100-prompt/32-output coverage.',
            'Formal qualification, all combinations and long-context performance are not established.'])


def _tsv(path,rows,columns):
    with Path(path).open('w',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,delimiter='\t',extrasaction='ignore');writer.writeheader()
        for row in rows:writer.writerow({key:json.dumps(value,sort_keys=True,separators=(',',':'))if type(value)in(dict,list)else value for key,value in row.items()})


def write_report(root,out,*,replace=False):
    out=Path(out)
    if out.exists()and not replace:raise FileExistsError('Use a fresh report directory or explicit --replace for an atomic snapshot')
    out.mkdir(parents=True,exist_ok=True)
    report=aggregate(root)
    for name,rows,cols in [
        ('units.tsv',report['units'],['name','status','lane','action','checkpoint','source_revision','source_snapshot_sha256','functional_passed','contract_passed','selected','observed_used','qualified','policy','settings','failed_gates','unit']),
        ('options.tsv',report['option_ledger'],['id','option','value','lane','coverage','tested_successfully','observed_used','selected_default','qualified','prerequisites','coverage_trigger','positive_units','passing_selected_units','all_attempt_units']),
        ('performance.tsv',report['performance_rows'],['unit','response_path','lane','checkpoint','source_revision','source_snapshot_sha256','prompt_tokens','output_tokens','cached_tokens','TTFT_seconds','elapsed_seconds','prompt_tokens_per_TTFT_second','decode_tokens_per_second','HTTP_wall_seconds','route','profile','policy','settings','qualified','performance_qualified'])]:
        temporary=out/('.'+name+'.tmp');_tsv(temporary,rows,cols);os.replace(temporary,out/name)
    report['output_files_sha256']={name:_sha(out/name)for name in ('units.tsv','options.tsv','performance.tsv')}
    _atomic(out/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--replace',action='store_true');args=parser.parse_args()
    report=write_report(args.root,args.out,replace=args.replace)
    print(json.dumps({key:report[key]for key in('unit_count','status_counts','catalog_requirement_count','option_coverage_counts')}))
