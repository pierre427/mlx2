# SPDX-License-Identifier: MIT
"""Caller-owned actual Source modality/batch/draft probes; CPU-only planning.

``cases`` prints plans. ``run`` sends requests to ONE already-running loopback
server whose caller owns the ordinary reference, source freeze, CPG/paired-lock
window and finally process-group cleanup. This module never acquires ownership
or starts a model/server. Importing and case planning never import MLX.
"""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import threading
import time
import unicodedata
import zlib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).parent))


def _png():
    """Deterministic newly created solid-red raster test fixture, no image input."""
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    width=height=96
    rows=(b'\x00'+bytes((255,0,0))*width)*height
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(rows,9))+chunk(b'IEND',b'')


def cases():
    result=[dict(id=f'physical_b{w}',policy={'deepseek_physical_batch':f'b{w}'},width=w,route='ordinary',kind='batch')for w in (2,4)]
    result.append(dict(id='vision_red',policy={'deepseek_vision':'enabled'},width=1,route='ordinary',kind='vision'))
    for depth in range(1,6):
        result.append(dict(id=f'embedded_dspark_d{depth}',policy={'draft_model':'embedded_dspark','num_draft':depth},width=1,route='external_draft',kind='draft'))
        for commit in ('replay','local_frontier'):
            result.append(dict(id=f'prompt_lookup_{commit}_d{depth}',policy={'draft_model':'prompt_lookup','num_draft':depth,'deepseek_verify_commit':commit},width=1,route='external_draft',kind='draft'))
    # Timed DSpark A/B: the same prompts on the ordinary route, serial verify
    # and the scheduled lockstep chain; each prompt is primed once, then timed.
    result.append(dict(id='draft_ab_ordinary',policy={},width=1,route='ordinary',kind='draft_ab_baseline'))
    for depth in (2,3):
        result.append(dict(id=f'draft_ab_serial_d{depth}',policy={'draft_model':'embedded_dspark','num_draft':depth},
                           width=1,route='external_draft',kind='draft_ab'))
        result.append(dict(id=f'draft_ab_chain_d{depth}',policy={'draft_model':'embedded_dspark','num_draft':depth,
            'deepseek_verify_commit':'local_frontier','deepseek_verify_rows':'lockstep_multirow_v1',
            'deepseek_dspark_verify_schedule':'survival_v1'},width=1,route='external_draft',kind='draft_ab'))
    return result


def case_by_id(name):
    return next((row for row in cases()if row['id']==name),None)or (_ for _ in ()).throw(ValueError('unknown advanced case: '+str(name)))


def _body(text,max_tokens=64):
    return {'messages':[{'role':'user','content':text}],'enable_thinking':False,'temperature':0,'max_tokens':max_tokens,
            'stream':True,'return_token_trace':True,'return_progress':True,'stream_options':{'include_usage':True}}


def request_plan(case):
    if case['kind']=='batch':
        return [dict(label=f'lane-{i}',body=_body(f'Return only the exact identifier DS41-LANE-{i}-7F29. Do not add explanation or punctuation.'),
                     oracle={'kind':'exact','expected':f'DS41-LANE-{i}-7F29'})for i in range(case['width'])]
    if case['kind']=='vision':
        png=_png();body=_body('What is the dominant color of the image? Reply with only one lowercase color word.',32)
        body['messages'][0]['content']=[{'type':'text','text':body['messages'][0]['content']},
            {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode()}}]
        return [dict(label='image-red',body=body,oracle={'kind':'word','expected':'red'},
                     image={'sha256':hashlib.sha256(png).hexdigest(),'width':96,'height':96,'mime':'image/png','construction':'new deterministic uniform RGB255,0,0 PNG'})]
    if case['kind'] in ('draft_ab','draft_ab_baseline'):
        rows=request_plan(dict(case,kind='draft'))
        return [dict(row,label=row['label']+suffix,body=json.loads(json.dumps(row['body'])))
                for row in rows for suffix in ('-prime','')]
    # The distractor encourages real prompt-copy mismatch, but no specific
    # proposal or rejection is fabricated. Counters determine actual coverage.
    copy=' '.join(['red green blue']*12)
    return [
        dict(label='count',body=_body('Count from 1 to 100 using numerals, one number per line. Begin with 1 immediately and do not add an introduction.',96),oracle={'kind':'count_prefix','count':8}),
        dict(label='copy-correction',body=_body('Reference pattern: '+copy+'\nNow write red green orange repeated eight times, separated by single spaces. Begin immediately; no introduction.',96),oracle={'kind':'prefix','expected':'red green orange red green orange'}),
        dict(label='heldout',body=_body('Reverse the characters of this exact string: A7kP2mQ9. Return only the reversed string.',48),oracle={'kind':'exact','expected':'9Qm2Pk7A'}),
    ]


def oracle_pass(text,oracle):
    value=text.strip()
    if oracle['kind']=='exact':
        normalized=''.join('-' if unicodedata.category(c)=='Pd' else c for c in value)
        return normalized==oracle['expected']
    if oracle['kind']=='word':return re.sub(r'[.!]$','',value.lower())==oracle['expected']
    if oracle['kind']=='prefix':return ' '.join(value.split()).startswith(oracle['expected'])
    if oracle['kind']=='count_prefix':return [int(x)for x in re.findall(r'\d+',value)][:oracle['count']]==list(range(1,oracle['count']+1))
    raise ValueError('unknown oracle')


def validate_server(case,status,*,fingerprint,source_snapshot,policy=None,runtime_build=None):
    if status.get('artifact')!=fingerprint:raise ValueError('server effective checkpoint mismatch')
    # Desktop source_manifest hashes a per-file manifest across packaging/
    # harness inputs. serving.runtime_identity hashes runtime paths+bytes and
    # resource metadata. Keep both exact bindings, never compare their hashes.
    if runtime_build is None:
        from mlx2.serving import runtime_identity
        runtime_build=runtime_identity()
    if status.get('build_runtime')!=runtime_build:raise ValueError('server runtime build identity mismatch')
    runtime=status.get('runtime');scope=status.get('runtime_source_scope')or{}
    if scope.get('mode')=='full_source':
        expected_runtime=dict(runtime_build)
        if isinstance(runtime,dict)and 'source_scope'in runtime:raise ValueError('full-source runtime has contradictory route scope')
    elif scope.get('mode')=='route_source':
        if not isinstance(runtime,dict)or not isinstance(runtime.get('source_scope'),dict):raise ValueError('missing route runtime source scope')
        from mlx2.route_identity import recompute_runtime_identity
        expected_runtime=recompute_runtime_identity(runtime,build=runtime_build,root=ROOT/'src/mlx2')
    else:raise ValueError('unknown runtime source scope')
    if runtime!=expected_runtime:raise ValueError('server served runtime identity mismatch')
    settings=status.get('settings',{})
    if settings.get('route')!=case['route']:raise ValueError('server route mismatch')
    if settings.get('max_lanes')!=case['width']or settings.get('max_inflight',0)<case['width']:raise ValueError('server lane/inflight shape mismatch')
    execution=status.get('execution',{});actual=execution.get('execution_policy',{})
    from mlx2.adapters.deepseek_v41_source import source_policy
    from mlx2.serving import SERVER_OWNED_EXECUTION_POLICY_KEYS
    normalized=source_policy({key:value for key,value in (policy or case['policy']).items() if key not in SERVER_OWNED_EXECUTION_POLICY_KEYS})
    for key,value in normalized.items():
        if key not in {'moe_expert_streaming','host_memory_signals'}and actual.get(key)!=value:
            raise ValueError('server policy mismatch: '+key)
    if case['kind']=='batch'and execution.get('physical_width')!=case['width']:raise ValueError('batch adapter physical width mismatch')
    if case['kind']=='vision'and not execution.get('vision_admission'):raise ValueError('vision weights were not admitted/loaded')
    if case['policy'].get('draft_model')=='embedded_dspark'and 'selected'not in str(execution.get('native_dspark','')):
        raise ValueError('embedded DSpark auxiliary was not loaded/selected')
    return {'desktop_source_manifest_sha256':source_snapshot,'runtime_build':runtime_build,
            'served_runtime':expected_runtime,'runtime_source_scope':scope,
            'binding_rule':'independently verified desktop manifest and owning runtime/route identity algorithms'}


def assess(case,rows,status):
    gates={};proposed=accepted=rounds=0;last_round_rejection_evidence=[]
    for row in rows:
        label=row['label'];response=row['response'];receipt=response['receipt']
        gates[label+'_quality']=oracle_pass(response['text'],row['oracle'])
        gates[label+'_route']=receipt.get('route')==case['route']and bool(receipt.get('route_receipt'))
        gates[label+'_cache']=receipt.get('cache')=='apcv2'and receipt.get('cached_tokens')==0
        trace=response['token_trace'];gates[label+'_trace']=bool(trace['token_ids'])
        if case['kind']=='batch':
            gates[label+'_execution_width']=case['width']in trace['execution_widths']
            gates[label+'_ordinary_width']=case['width']in trace['ordinary_execution_widths']
        if case['kind']=='vision':
            media=receipt.get('media_decode')or{}
            native=row.get('native_media_observation')or{}
            consumed=(media.get('media_prefill')=='isolated_prefill_forward'and 'image_inputs'in media.get('media_prefill_inputs',[]))
            native_used=(native.get('schema')=='mlx2.ds41-native-media-observation.v1'
                and native.get('encoder_calls_delta',0)>0 and native.get('image_prefill_returns_delta',0)>0
                and native.get('images_encoded_delta',0)>0 and native.get('source_binding_matched') is True
                and native.get('prefill_target_binding_matched') is True)
            gates[label+'_image_prefill']=consumed or native_used
        if case['kind'] in ('draft','draft_ab'):
            s=receipt.get('speculation')or{}
            expected_kind=('external_deepseek_v41_embedded_dspark_chain' if case['policy'].get('draft_model')=='embedded_dspark' else 'external_prompt_lookup')
            gates[label+'_draft_kind']=s.get('kind')==expected_kind
            p,a,n=s.get('proposed',0),s.get('accepted',0),s.get('external_rounds',0)
            gates[label+'_counter_contract']=all(type(v)is int and v>=0 for v in (p,a,n))and a<=p
            proposed+=p;accepted+=a;rounds+=n
            # A shorter accepted final round can also result from output/EOS
            # truncation. Retain it as evidence, not a proved rejection event.
            if s.get('round_proposed',0)>s.get('round_accepted',0):
                last_round_rejection_evidence.append({'label':label,'raw':s,'classification':'unaccepted_proposals; rejection_vs_output_truncation_unresolved'})
    execution=status.get('execution',{})
    if case['kind']=='batch':
        counters=execution.get('scheduled_batch_calls',{})
        gates['joint_ffn_observed']=counters.get('joint_ffn_calls',0)>0 and counters.get('maximum_joint_ffn_requests',0)>=case['width']
        gates['all_requests_live']=counters.get('maximum_live_requests',0)>=case['width']
    if case['kind'] in ('draft','draft_ab'):gates['actual_draft_proposals']=proposed>0 and rounds>0
    return {'passed':all(gates.values()),'gates':gates,'qualification':False,'observed_used':all(v for k,v in gates.items()if not k.endswith('_quality')),
            'draft_observations':{'proposed':proposed,'accepted':accepted,'unaccepted':proposed-accepted,'external_rounds':rounds,
                'acceptance_observed':accepted>0,'final_round_evidence':last_round_rejection_evidence,
                'rejection_coverage':'not_proven_by_terminal_aggregate; inspect round evidence before promoting rejection coverage'},
            'scope':'bounded actual route/output probes; no replicated performance or formal qualification claim'}


def native_media_observation(before,after,*,source_fingerprint):
    bc=before.get('calls_after_successful_return',{});ac=after.get('calls_after_successful_return',{})
    native=after.get('native_vision')or{}
    return {'schema':'mlx2.ds41-native-media-observation.v1',
        'encoder_calls_delta':ac.get('vision.encoder_calls_completed',0)-bc.get('vision.encoder_calls_completed',0),
        'images_encoded_delta':ac.get('vision.images_encoded',0)-bc.get('vision.images_encoded',0),
        'image_prefill_returns_delta':ac.get('vision.native_image_prefill_returns',0)-bc.get('vision.native_image_prefill_returns',0),
        'source_binding_matched':native.get('source_fingerprint')==source_fingerprint,
        'prefill_target_binding_matched':bool(native.get('target_identity'))and native.get('target_identity')==native.get('prefill_target_identity')and native.get('prefill_returned')is True,
        'native_vision':native,'observer_sequence_before':before.get('sequence'),'observer_sequence_after':after.get('sequence'),
        'contract':'Source ordinary native media; existing encoder merged-embedding completion plus successful image_inputs target prefill; no output-text inference',
        'observation':'host method returns, existing owned completion and CPU bindings; no added synchronization or readback'}


def run_requests(spec,output,*,base='http://127.0.0.1:8285',guard=None,case_id=None):
    """Send to caller's already-owned frozen-source loopback server.

    Caller MUST run ordinary reference before this and stop its complete server
    process group in finally before releasing CPG/paired ownership afterward.
    ``output`` must be new; spec needs out/model/checkpoint/source bindings.
    """
    from urllib.parse import urlparse
    if urlparse(base).hostname not in ('127.0.0.1','localhost','::1'):raise ValueError('loopback URL required')
    from mlx2.desktop_campaign_worker import atomic
    from mlx2.desktop_runner import digest,source_manifest
    from probe_deepseek_v4_packed import verify_ownership
    from deepseek_v41_context_ladder import http_json,stream
    raw_case=case_id or spec.get('advanced_case')or spec.get('advanced')
    case=case_by_id(raw_case['id']if isinstance(raw_case,dict)else raw_case)
    window=Path(spec['out']);output=Path(output)
    if output.exists()and any(output.iterdir()):raise ValueError('refusing nonempty advanced evidence directory')
    output.mkdir(parents=True,exist_ok=True)
    def checked():
        if guard is not None:guard()
        return verify_ownership(window,[Path('/Users/Shared/mlxuag/gpu.lock/owner.json'),Path('/tmp/gpu.lock/owner.json')])
    report={'schema':'mlx2.ds41-modality-options.v1','case':case,'checkpoint':spec['checkpoint'],
            'source_revision':spec['source_revision'],'source_snapshot_sha256':spec['source_snapshot_sha256'],
            'qualification':False,'performance_claim':False,'rows':[],'passed':False,'server_owned_by':'caller; cleanup before CPG release'}
    try:
        report['ownership']=checked()
        if digest(source_manifest(ROOT))!=spec['source_snapshot_sha256']:raise ValueError('probe source differs from sealed snapshot')
        reference_path=window/'ordinary-reference.json'
        reference=json.loads(reference_path.read_text())
        if (reference.get('passed') is not True or reference.get('checkpoint')!=spec['checkpoint']
                or reference.get('source',{}).get('source_snapshot_sha256')!=spec['source_snapshot_sha256']):
            raise ValueError('matching passed ordinary reference required before advanced requests')
        report['ordinary_reference_sha256']=hashlib.sha256(reference_path.read_bytes()).hexdigest()
        fingerprint=spec['checkpoint'].rsplit('@',1)[1]
        from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact
        artifact=inspect_artifact(spec['model'])
        report['base_artifact']=artifact['identity']
        status=http_json(base,'/v1/status');report['status_before']=status
        if status.get('base_artifact',status.get('artifact'))!=artifact['identity']['fingerprint']:
            raise ValueError('live Source artifact differs from loaded server base checkpoint')
        policy=json.loads((window/'policy.json').read_text())if(window/'policy.json').exists()else case['policy']
        report['source_identity_verification']=validate_server(case,status,fingerprint=fingerprint,source_snapshot=spec['source_snapshot_sha256'],policy=policy)
        report['request_plan']=request_plan(case)
        model_id=http_json(base,'/v1/models')['data'][0]['id']
        for row in report['request_plan']:row['body']['model']=model_id
        atomic(output/'results.json',report)
        cohort={'id':'ds41-'+window.name+'-'+case['id'],'size':case['width']}
        barrier=threading.Barrier(case['width'])if case['kind']=='batch'else None
        def one(row):
            checked();body=dict(row['body'])
            observer_path=window/'policy-observer.json'
            observer_before=json.loads(observer_path.read_text())if case['kind']=='vision'and observer_path.exists()else{}
            if barrier is not None:
                body.update(batch_cohort=cohort,skip_writing_prefix_cache=True);barrier.wait(timeout=30)
            result=stream(base,body,output/(row['label']+'.sse'),timeout=900)
            value={'label':row['label'],'oracle':row['oracle'],'response':result}
            if case['kind']=='vision':
                observer_after=json.loads(observer_path.read_text())if observer_path.exists()else{}
                value['native_media_observation']=native_media_observation(observer_before,observer_after,
                    source_fingerprint=artifact['identity']['fingerprint'])
            atomic(output/(row['label']+'.json'),value);return value
        if barrier is not None:
            with ThreadPoolExecutor(max_workers=case['width'])as pool:report['rows']=list(pool.map(one,report['request_plan']))
        else:
            for row in report['request_plan']:
                report['rows'].append(one(row));atomic(output/'results.json',report)
        checked();report['status_after']=http_json(base,'/v1/status')
        report['assessment']=assess(case,report['rows'],report['status_after']);report['passed']=report['assessment']['passed']
        if digest(source_manifest(ROOT))!=spec['source_snapshot_sha256']:raise ValueError('source changed during advanced probe')
    except BaseException as error:
        report['error']={'type':type(error).__name__,'message':str(error)}
        report['passed']=False
        raise
    finally:
        atomic(output/'results.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('cases','run'))
    parser.add_argument('--spec',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--case');parser.add_argument('--base',default='http://127.0.0.1:8285')
    args=parser.parse_args()
    if args.mode=='cases':print(json.dumps(cases(),indent=2));return 0
    if args.spec is None or args.output is None:parser.error('run requires --spec and fresh --output')
    spec=json.loads(args.spec.read_text());result=run_requests(spec,args.output,base=args.base,case_id=args.case)
    print(json.dumps({'passed':result['passed'],'assessment':result.get('assessment')}));return 0 if result['passed']else 1


if __name__=='__main__':raise SystemExit(main())
