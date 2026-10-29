# SPDX-License-Identifier: MIT
"""One-repetition ordinary DS4.1 ladder on a caller-owned loopback server.

This module neither loads a model nor owns a GPU/server. The caller supplies
live ownership guards and process cleanup. Temperatures and paging are recorded
as observations, never admission gates. No formal qualification is performed.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from mlx2.desktop_campaign_worker import atomic
from mlx2.runtime.models.deepseek_v41_encoding import encode_messages
from deepseek_v41_context_ladder import http_json, normalize_code, stream
from bench_deepseek_v41_1k import metrics

CONTEXTS=(128,512,1024,2048,4096)
OUTPUT_RESERVE=64


def exact_request(tokenizer,context,kind):
    if type(context)is not int or context<=OUTPUT_RESERVE or kind not in ('identifier','counting'):
        raise ValueError('unsupported exact context/request kind')
    target=context-OUTPUT_RESERVE
    code='D41-'+hashlib.sha256(str(context).encode()).hexdigest()[:8].upper()
    prefix=('Assignment code: '+code+'\n')if kind=='identifier'else'Background notes:\n'
    suffix=('\nReturn only the assignment code, with exact case and characters.'if kind=='identifier'else
            '\nCount upward from 1 using numerals, one number per line. Begin immediately and continue without explanation.')
    sentence='The archive preserves stable records and revision-bound state.\n'
    def build(repetitions,padding=0):
        request={'messages':[{'role':'user','content':prefix+sentence*repetitions+' x'*padding+suffix}],
                 'enable_thinking':False}
        rendered=encode_messages(request['messages'],thinking_mode='chat')
        return request,rendered,tokenizer.encode(rendered,add_special_tokens=False).ids
    if len(build(0)[2])>target:raise ValueError('prompt template exceeds exact token target')
    low,high=0,target
    while low<high:
        middle=(low+high+1)//2
        if len(build(middle)[2])<=target:low=middle
        else:high=middle-1
    for repetitions in range(low,max(-1,low-3),-1):
        for padding in range(128):
            request,rendered,tokens=build(repetitions,padding)
            if len(tokens)==target:
                return dict(context_budget=context,prompt_tokens=target,output_reserve=OUTPUT_RESERVE,
                    kind=kind,output_tokens=32 if kind=='counting' else 64,expected_code=code if kind=='identifier'else None,
                    request=request,rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                    token_ids_sha256=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest())
    raise ValueError('could not construct exact DS4.1 ladder prompt length')


def prepare(model,output,*,contexts=CONTEXTS):
    """Prepare CPU/tokenizer-only requests in a fresh evidence directory."""
    from tokenizers import Tokenizer
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    tokenizer=Tokenizer.from_file(str(Path(model)/'tokenizer.json'))
    rows=[]
    for context in contexts:
        for kind in ('identifier','counting'):
            request=exact_request(tokenizer,context,kind);name=f'request-{context}-{kind}.json'
            atomic(output/name,request)
            rows.append({key:value for key,value in request.items()if key!='request'}|{'path':name})
    atomic(output/'requests.json',rows)
    return rows


def _finite(value):
    return type(value)in(int,float)and math.isfinite(value)


def timing(response,metric_delta):
    receipt=response['receipt'];n=receipt.get('completion_tokens')
    prompt=receipt.get('prompt_tokens');cached=receipt.get('cached_tokens')
    ttft=receipt.get('ttft_seconds');elapsed=receipt.get('elapsed_seconds')
    decode=elapsed-ttft if _finite(elapsed)and _finite(ttft)else None
    prefill=metric_delta.get('mlx2_request_prefill_time_seconds_sum')
    prefill_count=metric_delta.get('mlx2_request_prefill_time_seconds_count')
    fresh=prompt-cached if type(prompt)is int and type(cached)is int else None
    return dict(actual_prompt_tokens=prompt,cached_tokens=cached,uncached_prompt_tokens=fresh,
        output_tokens=n,TTFT_seconds=ttft,server_elapsed_seconds=elapsed,
        HTTP_wall_seconds=response.get('HTTP_seconds'),
        prompt_tokens_per_TTFT_second=prompt/ttft if type(prompt)is int and _finite(ttft)and ttft>0 else None,
        prefill_metric_seconds=prefill if prefill_count==1 else None,
        uncached_prompt_tokens_per_prefill_metric_second=fresh/prefill if type(fresh)is int and fresh>0 and prefill_count==1 and _finite(prefill)and prefill>0 else None,
        decode_receipt_seconds=decode,decode_intervals=max(0,n-1)if type(n)is int else None,
        decode_tokens_per_second=(n-1)/decode if type(n)is int and n>=2 and _finite(decode)and decode>0 else None,
        inter_token_metric_seconds=metric_delta.get('mlx2_inter_token_latency_seconds_sum'),
        inter_token_metric_intervals=metric_delta.get('mlx2_inter_token_latency_seconds_count'))


def expert_delta(before,after):
    """Read the expert provider counters only, never APC/PLE or preheat copies."""
    def counters(status):
        value=status.get('execution',{}).get('expert_stream',{}).get('counters',{})
        return {key:item for key,item in value.items()if _finite(item)}
    a,b=counters(before),counters(after)
    return {key:value-a.get(key,0)for key,value in b.items()}


def assess(response,request,*,expected_cached_tokens):
    receipt=response['receipt'];ids=response.get('token_ids',[])
    if request['kind']=='identifier':
        functional=normalize_code(response['text'])==request['expected_code']
    else:
        numbers=re.findall(r'\d+',response['text'])
        functional=len(numbers)>=4 and list(map(int,numbers))==list(range(1,len(numbers)+1))
    widths=(response.get('token_trace')or{}).get('execution_widths',[])
    gates=dict(functional=functional,prompt_length=receipt.get('prompt_tokens')==request['prompt_tokens'],
        cached_tokens=receipt.get('cached_tokens')==expected_cached_tokens,
        APCv2_engine=receipt.get('cache')=='apcv2',ordinary_route=receipt.get('route')=='ordinary'and bool(receipt.get('route_receipt')),
        width_one=bool(widths)and set(widths)=={1},token_trace=bool(ids)and len(ids)==receipt.get('completion_tokens'),
        output_bound=type(receipt.get('completion_tokens'))is int and 0<receipt['completion_tokens']<=request['output_tokens'])
    if request['kind']=='counting':gates['requested_32_outputs']=receipt.get('completion_tokens')==32
    contract=all(value for key,value in gates.items()if key!='functional')
    return dict(passed=functional and contract,functional_passed=functional,contract_passed=contract,gates=gates)


def _divergence(a,b):
    for i,(x,y)in enumerate(zip(a,b)):
        if x!=y:return dict(zero_based_position=i,shared_prefix=a[:i],cold_token=x,warm_token=y)
    if len(a)!=len(b):
        i=min(len(a),len(b));return dict(zero_based_position=i,shared_prefix=a[:i],
            cold_token=a[i]if i<len(a)else None,warm_token=b[i]if i<len(b)else None)
    return None


def run(spec,model_id,*,guard,environment,phase,admin_token,base='http://127.0.0.1:8285'):
    """Run prepared requests; caller retains server/process/lease ownership."""
    out=Path(spec['out']);requests=json.loads((out/'requests.json').read_text())
    if (out/'ladder.json').exists()or any(out.glob('context-*')):
        raise ValueError('refusing to overwrite a started performance ladder')
    policy=spec.get('execution_policy',spec.get('extra_source_policy',{}))
    apc_enabled=spec.get('ladder_apc_enabled',policy.get('deepseek_source_prefill_partition','adaptive_v1')=='canonical_m1_v1')
    if type(apc_enabled)is not bool:raise ValueError('ladder_apc_enabled must be boolean')
    report=dict(schema='mlx2.ds41-options-ladder.v1',checkpoint=spec['checkpoint'],source_revision=spec['source_revision'],
        source_snapshot_sha256=spec['source_snapshot_sha256'],execution_policy=policy,
        repetitions=1,thermal_controlled=False,qualification=False,performance_qualified=False,
        APCv2_reuse_expected=apc_enabled,context_semantics='prompt=context-64;64 output reserve',
        timing_definitions={'TTFT':'server request start to first token; includes queue/host/prefill/first-token overhead',
            'prefill_metric':'server execution-lane prefill timer; host/runtime and first-token overhead, not pure GPU throughput',
            'decode':'(output_tokens-1)/(server_elapsed-TTFT); terminal receipt work included',
            'observations':'single repetition; expert cache carries across requests/rungs; thermal, swap and host observations do not gate'},
        caller_process_cleanup=True,cells=[],status='running')
    atomic(out/'ladder.json',report)
    try:
        guard();report['server_before']=http_json(base,'/v1/status');report['environment_before']=environment()
        report['observed_execution_policy']=report['server_before'].get('execution',{}).get('execution_policy')
        report['observed_settings']=report['server_before'].get('settings')
        atomic(out/'ladder.json',report)
        for info in requests:
            guard();request=json.loads((out/info['path']).read_text())
            directory=out/f"context-{info['context_budget']}-{info['kind']}";directory.mkdir(exist_ok=False)
            cell={key:value for key,value in request.items()if key!='request'}
            cell.update(repetitions=1,thermal_controlled=False,APCv2_reuse_expected=apc_enabled,status='running',arms={})
            atomic(directory/'result.json',cell)
            clear=http_json(base,'/v1/admin/apc/clear',{'reason':'ds41-options-ladder-cold'},timeout=60,admin_token=admin_token)
            cell['APC_clear']=clear
            body={**request['request'],'model':model_id,'temperature':0,'max_tokens':request['output_tokens'],
                'stream':True,'return_token_trace':True,'return_progress':True,'logprobs':False,
                'stream_options':{'include_usage':True}}
            if info['kind']=='counting':body['min_tokens']=32
            for arm in ('cold','warm'):
                guard();phase('ladder_arm_started',context_budget=info['context_budget'],kind=info['kind'],arm=arm)
                before=http_json(base,'/v1/status');raw,bm=metrics(base);(directory/f'{arm}-metrics-before.prom').write_text(raw)
                env_before=environment();response=stream(base,body,directory/f'{arm}.sse',timeout=3600)
                env_after=environment();after=http_json(base,'/v1/status');raw,am=metrics(base);(directory/f'{arm}-metrics-after.prom').write_text(raw)
                delta={key:value-bm.get(key,0)for key,value in am.items()}
                row=dict(response=response,status_before=before,status_after=after,environment_before=env_before,
                    environment_after=env_after,metrics_delta=delta,expert_counter_delta=expert_delta(before,after),
                    timing=timing(response,delta),
                    expert_metric_delta={key:value for key,value in delta.items()if 'expert' in key or 'weight_read' in key},
                    expert_counters_observed=bool(expert_delta(before,after)),
                    memory_observations={which:{key:status.get(key)for key in ('metal_active_bytes','metal_peak_bytes','process_physical_footprint_bytes','host_memory_available_bytes','host_memory_pressure_level')}for which,status in (('before',before),('after',after))},
                    cache_arm='APCv2_prefix_replay'if arm=='warm'and apc_enabled else'ordinary_expert_cache_warm'if arm=='warm'else'APCv2_cold_expert_cache_carried')
                row['assessment']=assess(response,request,expected_cached_tokens=request['prompt_tokens']-1 if arm=='warm'and apc_enabled else 0)
                cell['arms'][arm]=row;atomic(directory/'result.json',cell)
                phase('ladder_arm_complete',context_budget=info['context_budget'],kind=info['kind'],arm=arm,
                    text=response['text'],timing=row['timing'],assessment=row['assessment'])
            cold,warm=(cell['arms'][arm]['response']for arm in ('cold','warm'))
            divergent=_divergence(cold['token_ids'],warm['token_ids'])
            cell['first_divergence']=divergent;cell['token_parity']=divergent is None
            cell['passed']=all(row['assessment']['passed']for row in cell['arms'].values())and(divergent is None or not apc_enabled)
            cell['status']='complete';atomic(directory/'result.json',cell)
            report['cells'].append(dict(context_budget=info['context_budget'],kind=info['kind'],path=str(directory.relative_to(out)/'result.json'),
                passed=cell['passed'],token_parity=cell['token_parity'],first_divergence=divergent,
                arms={arm:{'timing':row['timing'],'assessment':row['assessment'],'expert_counter_delta':row['expert_counter_delta']}for arm,row in cell['arms'].items()}))
            atomic(out/'ladder.json',report)
            if any(not row['assessment']['contract_passed']for row in cell['arms'].values())or(apc_enabled and divergent is not None):
                report['stopped_reason']='contract failure or exact APCv2 replay divergence; preserve first differing token before repair'
                break
        report.update(complete=len(report['cells'])==len(requests),status='complete',environment_after=environment(),server_after=http_json(base,'/v1/status'))
        report['passed']=report['complete']and all(cell['passed']for cell in report['cells'])
        atomic(out/'ladder.json',report)
        return report
    except BaseException as error:
        report.update(status='failed',complete=False,passed=False,error={'type':type(error).__name__,'message':str(error)})
        atomic(out/'ladder.json',report)
        raise


run_ladder=run
