# SPDX-License-Identifier: MIT
"""CPU-only Source option branch coverage from actual campaign receipts.

Scheduling counters never prove GPU completion. A completed response is separate
whole-request execution evidence. Selected policies and fallback-only paths are
reported separately; no qualification/default decision is made here.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]

# Each inner tuple is an OR group; every group must have a positive counter.
RULES={
 'deepseek_kv_storage': [('native_packed_local_rows',)],
 'deepseek_candidate_score':[('candidate_only_score_rows',)],
 'deepseek_ced':[('ced_prefill_calls',)],
 'deepseek_hc':[('hc.split_sinkhorn',),('hc.expand','fused.expand')],
 'deepseek_qat':[('model.qat.fused_fp32',)],
 'deepseek_pointwise':[('fused.rounded',),('fused.rotate',),('fused.router',)],
 'deepseek_attention_epilogue':[('fused.rotate_qat',)],
 'deepseek_hc_epilogue':[('fused.expand',)],
 'deepseek_source_prefill_partition':[('source_canonical_m1_prefill_tokens',)],
 'deepseek_source_submission':[('source_shared_first_submissions',)],
 'deepseek_source_row_packet':[('source_row_packet_uploads',)],
 'deepseek_source_frame_reuse':[('official_source_frame_route_order_reused',),('official_source_frame_route_histogram_reused',),('official_source_frame_combine_offsets_reused',)],
 'deepseek_source_native_mx':[('official_source_native_mx_gate_up_jobs',),('official_source_native_mx_down',)],
 'deepseek_source_ffn_input':[('source_ffn_input_qat_materialized',),('source_ffn_input_qat_calls_avoided',)],
 'deepseek_source_woa_pp':[('source_woa_native_pp',)],
 'deepseek_source_fp8_pp':[('source_fp8_native_pp_attn_q_b',),('source_fp8_native_pp_attn_output_b',)],
 'deepseek_source_hc_boundary':[('source_hc_boundary_expand_collapse',)],
 'deepseek_source_q_b_prefill':[('source_q_b_prefill_tile64x32_raw',)],
 'deepseek_source_qakv':[('source_qakv_qat_reuse_pairs',)],
 'deepseek_source_prefill_wide_dtype':[('official_source_prefill_bf16_p1_gate_up','official_source_prefill_bf16_p1_down')],
 'deepseek_source_dense_qat':[('source_dense_qat_fused',)],
 'deepseek_source_hc_coefficients':[('source_fused_fp32',)],
 'deepseek_index_policy':[('all_retained_index',)],
 'deepseek_selected_kv_policy':[('decode_gather_build',),('decode_gather_hit',)],
}
GENERIC_OBSERVER_KEYS={'deepseek_hc','deepseek_qat','deepseek_pointwise','deepseek_attention_epilogue','deepseek_hc_epilogue'}
DISTINCT={
 'deepseek_candidate_score':'For pruning benefit require candidate_only_unscored_rows>0: candidate capacity is2048*8=16384 pooled index rows. The100-token test can schedule candidate scoring without pruning any row.',
 'deepseek_ced':'Require actual per-forward PP rows>128 and ced_skipped_layer_rows>0. Increasing total context with prefill-step128 still cannot exercise pruning; request prefill-step256+ and verify actual rows.',
 'deepseek_index_policy':'Short<=512 pool bypass and long>512 pool selector fallback need separate receipts; short bypass is a real observed branch, not broad selector qualification.',
 'deepseek_source_prefill_wide_dtype':'Original local expert GU needs M>=97; DOWN needs M>=33. A100-token prompt may cover only one boundary or neither; record both counts separately.',
 'deepseek_source_prefill_wide_math':'Original local expert M13..96 plus actual planned tile batches required; total prompt length is not local expert occupancy.',
 'deepseek_source_native_mx':'Original local M16..99 within globalPP2..99 plus current coefficient attestation; Q1/small-M/decode fallback does not cover native MX.',
 'deepseek_source_fp8_pp':'Exact PP99 Q_b and WoB; canonicalM1 and APC hits cannot exercise this PP branch.',
 'deepseek_source_woa_pp':'Exact PP99 WoA; canonicalM1 and APC hits cannot exercise this PP branch.',
 'deepseek_source_q_b_prefill':'Exact PP99 Q_b raw tile; canonicalM1 and APC hits cannot exercise it.',
 'deepseek_physical_batch':'Separate simultaneous distinct B2/B4 cohort; require token trace widths and jointFFN/live counters, not selected width.',
 'deepseek_vision':'Separate real image request and consumed image-prefill receipt; text output does not exercise vision.',
 'deepseek_verify_commit':'Separate prompt-lookup accepted/rejected proposals and target continuation; ordinary generation does not cover commit.',
}


def rule(option,value):
    if option=='deepseek_source_expert_math':
        if value=='hoisted_fused_fp32':return [('official_mxfp4_act32_hoisted_fused',)]
        if value=='paired_q1_fused_fp32':return [('official_q1_paired_gate_up',)]
        if value=='paired_fused_fp32':return [('official_q1_paired_gate_up',),('official_mxfp4_paired_gate_up',)]
        if value=='epilogue_fused_fp32':return [('official_source_fused_middle',)]
        if value=='routed_fused_fp32':return [('official_source_gate_up_middle_batch_fused','official_source_gate_up_middle_fused'),('official_source_down_combine_fused',)]
        return [('official_mxfp4_act32_'+value,)]
    if option=='deepseek_source_prefill_batch':
        return [('official_source_prefill_gu_batches',)]+([('official_source_prefill_down_batches',)]if value=='gate_up_down'else[])
    if option=='deepseek_source_prefill_wide_math':
        keys=[('official_source_prefill_wide_gu_batches',)]
        if value.endswith('_16x64'):keys.append(('official_source_prefill_wide_16x64_batches',))
        if '_fused'in value:keys.append(('official_source_prefill_wide_fused_middle_batches',))
        if value=='nax_k16_ordered_f32_v1':keys.append(('official_source_prefill_nax_k16_jobs',))
        return keys
    if option=='deepseek_source_prefill_down':
        return [('official_source_prefill_q1_m1_owner_store',)]if value=='q1_m1_owner_store'else[('official_source_prefill_q1_tiny_down_jobs',)]
    if option=='deepseek_source_attention_math':
        return [({'all_retained_broadcast':'source_all_retained_broadcast_attention','omlx_nax_bf16_online_v1':'source_omlx_nax_attention','source_short_decode_joint_f32_v1':'source_short_attention_joint','source_short_decode_grouped_f32_v1':'source_short_attention_native'}[value],)]
    if option=='deepseek_source_bf16_loads':
        return [('source_bf16_grouped_direct_f32','source_bf16_grouped_legacy_f32')]+([('source_bf16_head_direct_f32',)]if value=='direct_f32'else[])
    if option=='deepseek_source_completion':return [('completion.submitted_epochs',),('completion.retired_epochs',)]
    return RULES.get(option)


def _read(path,*,optional=False):
    if optional and not path.exists():return None,None
    raw=path.read_bytes();return json.loads(raw),hashlib.sha256(raw).hexdigest()


def _status_counters(status):
    execution=status.get('execution',{});result={}
    for group in ('scheduled_expert_calls','scheduled_dense_calls','scheduled_grouped_calls','scheduled_kv_calls','scheduled_batch_calls','source_hc_boundary_counters'):
        for key,value in execution.get(group,{}).items():
            if type(value)in(int,float):result[key]=value
    return result


def load_unit(path):
    path=Path(path);spec,specsha=_read(path/'spec.json',optional=True);results,resultsha=_read(path/'results.json',optional=True)
    if results is None:return {'complete':False,'counters':{},'source':spec,'evidence':{'spec.json':specsha}}
    counters=Counter();statusc=Counter();rows=[];completed=set(results.get('requests',{}))
    phasepath=path/'model-phases.jsonl';phasesha=None;partial_lines=0
    if phasepath.exists():
        raw=phasepath.read_bytes();phasesha=hashlib.sha256(raw).hexdigest()
        for line in raw.splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:partial_lines+=1;continue
            if r.get('label')not in completed:continue
            rows.append(r.get('input_rows'))
            for group in ('expert_math_scheduled_delta','dense_math_scheduled_delta','grouped_math_scheduled_delta','kv_scheduled_delta','hc_math_scheduled_delta'):
                counters.update({k:v for k,v in r.get(group,{}).items()if type(v)in(int,float)})
            for key,value in (r.get('completion')or{}).get('stats',{}).items():
                if type(value)in(int,float):counters['completion.'+key]=max(counters['completion.'+key],value)
    last=results.get('status_ready',{})
    for request in results.get('requests',{}).values():
        before=_status_counters(request.get('status_before',{}));after=_status_counters(request.get('status_after',{}));last=request.get('status_after',{})
        statusc.update({key:value-before.get(key,0)for key,value in after.items()})
    for key,value in statusc.items():
        # A completed outer operation may increment after the inner-forward
        # phase snapshot (canonical prefill is one example). Use the larger
        # exact observation, never sum duplicate phase/status views or let a
        # present zero hide a positive completed-request delta.
        counters[key]=max(counters.get(key,0),value)
    observer,observersha=_read(path/'policy-observer.json',optional=True)
    if observer:
        counters.update(observer.get('calls_after_successful_return',{}))
    completion=last.get('execution',{}).get('dense_storage',{}).get('source_completion_receipt')or{}
    for key,value in completion.get('stats',{}).items():
        if type(value)in(int,float):counters['completion.'+key]=max(counters['completion.'+key],value)
    return {'complete':results.get('passed')is True,'output_cells':len(completed),'counters':dict(counters),
        'input_rows':sorted({r for r in rows if type(r)is int}),'last_status':last,'source':spec,'observer':observer,
        'evidence':{'spec.json':specsha,'results.json':resultsha,'model-phases.jsonl':phasesha,'policy-observer.json':observersha},
        'partial_phase_lines_ignored':partial_lines,'error':results.get('error')}


def assess(arm,data):
    option,value=arm['option'],arm['value'];c=data.get('counters',{});groups=rule(option,value)
    positive={key:amount for key,amount in c.items()if amount>0}
    matches={str(i):{key:positive[key]for key in group if key in positive}for i,group in enumerate(groups or[])}
    missing=[list(group)for group in groups or[]if not any(key in positive for key in group)]
    fallback_prefixes={
        'deepseek_source_q_b_prefill':('source_q_b_prefill_',),
        'deepseek_source_fp8_pp':('source_fp8_native_pp_',),
        'deepseek_source_woa_pp':('source_woa_native_pp_',),
        'deepseek_source_native_mx':('official_source_native_mx_fallback_',),
        'deepseek_source_attention_math':('source_short_attention_','source_omlx_nax_attention_'),
        'deepseek_source_ffn_input':('source_ffn_input_',),
        'deepseek_source_hc_boundary':('source_hc_boundary_reference',),
        'deepseek_source_bf16_loads':('source_bf16_grouped_fallback','source_bf16_head_fallback'),
    }.get(option,())
    fallback={key:value for key,value in positive.items()if any(key.startswith(prefix)for prefix in fallback_prefixes)
              and any(word in key for word in ('fallback','reference_geometry','reference_device','boundary_reference'))}
    actual_policy=data.get('last_status',{}).get('execution',{}).get('execution_policy',{})
    selected=actual_policy.get(option)==value
    selection_known=bool(actual_policy)
    special=[]
    if option=='deepseek_candidate_score'and c.get('candidate_only_unscored_rows',0)<=0:special.append(DISTINCT[option])
    if option=='deepseek_ced'and c.get('ced_skipped_layer_rows',0)<=0:special.append(DISTINCT[option])
    if selection_known and not selected:classification='selection_mismatch'
    elif not data.get('output_cells'):classification='selected_no_completed_output' if selected else 'configured_no_completed_output'
    elif groups and not missing:classification='observed_scheduled_partial_scope'if special else'observed_scheduled'
    elif option=='deepseek_engram_cache':
        tables=data.get('last_status',{}).get('execution',{}).get('engram',{})
        observed=bool(tables)and all(t.get('cache_format')==value and t.get('requested_rows',0)>0 for t in tables.values())
        classification='observed_format_access'if observed else'selected_unobserved'
    elif groups and option in GENERIC_OBSERVER_KEYS and not data.get('observer'):classification='selected_uninstrumented'
    elif groups and fallback:classification='fallback_only_or_partial_target_coverage'
    elif groups:classification='selected_unobserved'
    else:classification='needs_distinct_profile_probe'
    if not data.get('complete')and classification.startswith('observed'):classification+='__unit_incomplete'
    needs=bool(missing or special or groups is None or not data.get('complete') or selection_known and not selected)
    return {'option':option,'value':value,'classification':classification,'configured_policy':arm.get('policy'),'selected':selected,
        'observed_policy':actual_policy,
        'whole_request_completed':data.get('complete',False),'completed_output_cells':data.get('output_cells',0),
        'observed_matching_counters':matches,'missing_counter_groups':missing,'fallback_counters':fallback,
        'observed_forward_rows':data.get('input_rows',[]),'scope_gaps':special,
        'requeue_required':needs,'requeue_instruction':('Install host policy observer and repeat ordinary probe.'if classification=='selected_uninstrumented'else
            ' '.join(special)or DISTINCT.get(option,arm.get('coverage_trigger','Require explicit actual-route/shape evidence.'))),
        'evidence':data.get('evidence',{}),'checkpoint':(data.get('source')or{}).get('checkpoint'),
        'source_revision':(data.get('source')or{}).get('source_revision'),
        'source_snapshot_sha256':(data.get('source')or{}).get('source_snapshot_sha256'),
        'observer_shutdown_snapshot':bool((data.get('observer')or{}).get('shutdown_snapshot')),
        'scheduling_is_GPU_completion_proof':False,'qualification':False}


def campaign_report(path):
    campaign,campaignsha=_read(Path(path));reports=[]
    for name,item in campaign.get('arms',{}).items():
        row=assess(item['arm'],load_unit(item['unit']));row.update(id=name,unit=item['unit'],campaign_status=item.get('status'));reports.append(row)
    sys.path.insert(0,str(Path(__file__).parent))
    from ds41_option_catalog import catalog
    attempted={(row['option'],row['value'])for row in reports}
    requirements=[{'option':arm['option'],'value':arm['value'],'lane':arm['lane'],
        'required_counter_groups':rule(arm['option'],arm['value']),
        'coverage_trigger':arm['coverage_trigger'],
        'distinct_probe_requirement':DISTINCT.get(arm['option']),
        'attempted':(arm['option'],arm['value'])in attempted}for arm in catalog()['arms']]
    return {'schema':'mlx2.ds41-option-coverage.v1','campaign':str(Path(path).resolve()),'campaign_sha256':campaignsha,
        'scope':'Current observed attempts only; no qualification/default authority','arms':reports,
        'counts':dict(Counter(r['classification']for r in reports)),'requeue':[r['id']for r in reports if r['requeue_required']],
        'catalog_requirements':requirements,
        'unattempted_catalog_arms':[{'option':row['option'],'value':row['value']}for row in requirements if not row['attempted']]}


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('usage: ds41_option_coverage.py CAMPAIGN_JSON')
    print(json.dumps(campaign_report(sys.argv[1]),indent=2,sort_keys=True))
