#!/usr/bin/env python3
"""One caller-owned exact-artifact HTTP/APCv2 diagnostic, never qualification.

Only loopback servers started here are stopped. The outer campaign owns the
CPG/paired leases, thermal checks and final lease release. Evidence is private,
append-only by fresh directory, and every server process group is retired.
"""
from __future__ import annotations
import argparse,json,os,signal,socket,subprocess,sys,time
from dataclasses import dataclass
from pathlib import Path
import urllib.request,urllib.error
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from probe_deepseek_v4_packed import atomic,source_receipt,verify_ownership
from probe_deepseek_v4_ordinary import load_request,functional_result
REFERENCE={'messages':[{'role':'user','content':'Reply with exactly Hello.'}],'enable_thinking':False}
DEFAULT_REQUEST={'messages':[{'role':'user','content':'Reference notes:\n'+''.join(f'Entry {i}: APCv2 stores exact prefix state; one ordinary request uses one decode lane.\n' for i in range(24))+'\nFinal instruction: Reply with exactly Hello.'}],'enable_thinking':False}
EXTERNAL_KEYS={'draft_model','num_draft','deepseek_verifier','deepseek_draft_confidence'}


def command(model,port,policy,cache,*,external,prefill_step,cache_bytes,persist):
    # The serving contract has one APCv2 backing directory; persistence uses
    # that same path rather than a separate scratch/persist split.
    cache_directory=cache/('persist' if persist else 'scratch')
    result=[sys.executable,'-m','mlx2.server','--model',str(model),'--host','127.0.0.1','--port',str(port),
        '--fabric','off',
        '--max-lanes','1','--max-inflight','1','--max-context','4096','--default-max-tokens','128',
        '--prefill-step',str(prefill_step),'--cache-bytes',str(cache_bytes),'--cache-dir',str(cache_directory),
        '--execution-policy',str(policy),'--gpu-keep-warm-seconds','0','--residency-keep-seconds','0',
        '--external-draft' if external else '--ordinary']
    if persist:result+=['--apc-persist-dir',str(cache_directory),'--apc-persist-on-shutdown','--apc-persist-shutdown-seconds','15']
    return result


def get_json(base,path,timeout=5):
    with urllib.request.urlopen(base+path,timeout=timeout) as response:
        return json.load(response)


def stream_chat(base,body,path,*,timeout):
    request=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),
        headers={'Content-Type':'application/json'},method='POST')
    start=time.perf_counter();tokens=[];text=[];reasoning=[];receipt=None;finish=None;done=False;usage=None;total=0
    try:
        with urllib.request.urlopen(request,timeout=timeout) as response,path.open('wb') as raw:
            for line in response:
                raw.write(line);total+=len(line)
                if total>8<<20:raise ValueError('bounded SSE evidence exceeded8MiB')
                value=line.decode('utf-8').strip()
                if not value.startswith('data:'):continue
                value=value[5:].strip()
                if value=='[DONE]':done=True;break
                event=json.loads(value)
                if event.get('error'):raise RuntimeError('SSE error: '+json.dumps(event['error']))
                if event.get('mlx2') is not None:receipt=event['mlx2']
                if event.get('usage') is not None:usage=event['usage']
                for choice in event.get('choices',[]):
                    for item in (choice.get('logprobs') or {}).get('content',[]):tokens.append(int(item['id']))
                    delta=choice.get('delta') or {};text.append(delta.get('content') or '');reasoning.append(delta.get('reasoning_content') or '')
                    finish=choice.get('finish_reason') or finish
    except urllib.error.HTTPError as error:
        error_body=error.read(1<<20);path.write_bytes(error_body)
        raise RuntimeError(f'HTTP{error.code}; captured response body in {path.name}') from error
    if not done or receipt is None or finish is None:
        raise RuntimeError('incomplete stream, missing DONE or terminal receipt/finish')
    return {'text':''.join(text),'reasoning_content':''.join(reasoning),'token_ids':tokens,'receipt':receipt,
            'finish_reason':finish,'usage':usage,'seconds':time.perf_counter()-start,'raw_sse_path':path.name,
            'timing_scope':'observational HTTP wall latency; not device throughput'}


def expected_external_identity(artifact,policy,source):
    """Derive expected binding from sealed bytes+config+local source, never status."""
    from mlx2.adapters.deepseek_v4_gguf import split_execution_policy
    from mlx2.adapters.deepseek_v4_dspark_artifact import inspect_sidecar
    from mlx2.runtime.models.deepseek_v4 import DeepSeekV4Config
    from mlx2.runtime.models.deepseek_v4_identity import (
        native_target_identity,dspark_target_identity,dspark_execution_binding,dspark_artifact_fingerprint)
    from mlx2.runtime.drafters.deepseek_v4_dspark import DeepSeekV4DSparkConfig,verify_runtime_binding
    kernel,external=split_execution_policy(policy)
    if not external:raise ValueError('external expected identity requires explicit sealed sidecar')
    sidecar=inspect_sidecar(external['draft_model'])
    fields=dict(sidecar['config']);fields['target_layers']=tuple(fields['target_layers'])
    config=DeepSeekV4DSparkConfig(**fields);config.validate_target(DeepSeekV4Config.from_gguf(artifact['metadata']))
    backbone=verify_runtime_binding(config,target_sha256=artifact['identity']['weight_sha256'],
        sidecar_sha256=sidecar['sha256'],target_metadata=artifact['metadata'],
        source_snapshot=source['source_snapshot_sha256'])
    target=native_target_identity(artifact['identity']['fingerprint'],kernel)
    target_policy=dspark_target_identity(target,kernel['deepseek_packed'],kernel['deepseek_embedding'],
        external['deepseek_verifier'],external['deepseek_draft_confidence'])
    execution=dspark_execution_binding(backbone,target_policy,external['deepseek_verifier'])
    return {'fingerprint':dspark_artifact_fingerprint(execution,external['num_draft']),
        'target_fingerprint':artifact['identity']['fingerprint'],'draft_fingerprint':sidecar['sha256'],
        'external_source_snapshot':source['source_snapshot_sha256'],'execution_binding':execution,
        'target_execution_identity':target,'derivation':'sealed target/sidecar metadata + verified local source + normalized explicit policies'}


def identity_assessment(status,*,fingerprint,build_source,external):
    settings=status.get('settings') or {}
    pairs={'artifact':(fingerprint,status.get('artifact')),
        'build_source':(build_source,(status.get('build_runtime') or {}).get('source_sha256')),
        'max_lanes':(1,settings.get('max_lanes')),'max_inflight':(1,settings.get('max_inflight')),
        'route':('external_draft' if external else 'ordinary',settings.get('route'))}
    checks={key:{'expected':expected,'observed':observed,'passed':expected==observed}
        for key,(expected,observed) in pairs.items()}
    return {'passed':all(row['passed'] for row in checks.values()),'checks':checks,
            'mismatched_fields':[key for key,row in checks.items() if not row['passed']]}


def identity_check(status,*,fingerprint,build_source,external):
    assessment=identity_assessment(status,fingerprint=fingerprint,build_source=build_source,external=external)
    if not assessment['passed']:
        raise RuntimeError('server identity mismatch: '+json.dumps(assessment,sort_keys=True))
    return assessment


def record_ready_status(entry,status,result,path,*,fingerprint,build_source,external):
    # Persist raw status BEFORE any guard can raise, including malformed data.
    entry['status_before']=status;atomic(path,result)
    assessment=identity_assessment(status,fingerprint=fingerprint,build_source=build_source,external=external)
    entry['identity_assessment']=assessment;atomic(path,result)
    if not assessment['passed']:
        raise RuntimeError('server identity mismatch: '+json.dumps(assessment,sort_keys=True))
    return assessment


def pair_check(cold,warm,*,external,law,expected):
    failures=[];expected_route='external_draft' if external else 'ordinary';first_difference=None
    for label,row,cached in [('cold',cold,False),('warm',warm,True)]:
        r=row.get('receipt') or {};n=r.get('cached_tokens')
        if r.get('cache')!='apcv2':failures.append(label+': missing APCv2 receipt')
        if type(n) is not int or (n<=0 if cached else n!=0):failures.append(label+': unexpected cached_tokens='+repr(n))
        if r.get('route')!=expected_route or not r.get('route_receipt'):failures.append(label+': wrong/missing terminal route receipt')
        if not row.get('token_ids'):failures.append(label+': missing output token IDs')
        if not functional_result(row.get('text',''),law=law,expected=expected,finite=True)['passed']:failures.append(label+': targeted text oracle failed')
    if cold.get('token_ids')!=warm.get('token_ids'):
        failures.append('cold/warm token IDs differ')
        a,b=cold.get('token_ids') or [],warm.get('token_ids') or []
        at=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
        first_difference={'output_position':at,'shared_prefix_length':at,
            'cold_selected':a[at] if at<len(a) else None,'warm_selected':b[at] if at<len(b) else None,
            'scope':'selected first divergence only; no post-divergence logits interpreted and no equivalence verdict'}
    if cold.get('text')!=warm.get('text') or cold.get('reasoning_content')!=warm.get('reasoning_content'):failures.append('cold/warm returned message differs')
    if (cold.get('receipt') or {}).get('prompt_tokens')!=(warm.get('receipt') or {}).get('prompt_tokens'):failures.append('cold/warm prompt lengths differ')
    return {'passed':not failures,'failures':failures,'first_selected_divergence':first_difference,
            'equivalence':'exact tokens/messages only; mismatches retained, no near-tie classification from downstream logits'}


@dataclass
class OwnedServer:
    process:object
    log:object
    pgid:int
    command:list


def stop_owned(server,*,grace=20):
    """Retire only the new session created by this harness; never global pkill."""
    if not isinstance(server,OwnedServer) or server.pgid!=server.process.pid or server.pgid<=1 or server.pgid==os.getpgrp():
        raise RuntimeError('refusing foreign/unowned process group cleanup')
    receipt={'pid':server.process.pid,'process_group':server.pgid,'signals':[]}
    def send(sig):
        try:os.killpg(server.pgid,sig);return True
        except ProcessLookupError:return False
    def group_alive():
        try:os.killpg(server.pgid,0);return True
        except ProcessLookupError:return False
    try:
        if group_alive():
            if send(signal.SIGTERM):receipt['signals'].append('SIGTERM')
        try:server.process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            if group_alive() and send(signal.SIGKILL):receipt['signals'].append('SIGKILL')
            server.process.wait(timeout=5)
        # Root exit does not establish descendant cleanup.
        if group_alive():
            if send(signal.SIGKILL):receipt['signals'].append('SIGKILL_remaining_group')
        receipt.update(returncode=server.process.returncode,process_group_absent=not group_alive())
        return receipt
    finally:server.log.close()


def start_server(cmd,log_path,*,guard,startup_seconds):
    guard()
    log=log_path.open('wb');process=None
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','MLX_ENABLE_TF32':'0'}
    try:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        server=OwnedServer(process,log,process.pid,cmd)
        if os.getpgid(process.pid)!=process.pid:raise RuntimeError('server did not start its own process group')
        return server
    except BaseException:
        if process is not None:stop_owned(OwnedServer(process,log,process.pid,cmd),grace=3)
        else:log.close()
        raise


def wait_ready(server,base,*,guard,seconds):
    start=time.monotonic()
    while time.monotonic()-start<seconds:
        guard()
        if server.process.poll() is not None:raise RuntimeError('owned server exited before healthy; inspect server log')
        try:
            status=get_json(base,'/v1/status',timeout=2)
            if status.get('healthy'):return status
        except (urllib.error.URLError,TimeoutError,ConnectionError,OSError):pass
        time.sleep(.5)
    raise TimeoutError('owned server startup exceeded bound')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--run-gpu',action='store_true')
    p.add_argument('--ownership-window',type=Path);p.add_argument('--owner-path',type=Path,action='append')
    p.add_argument('--execution-policy',type=json.loads,default={});p.add_argument('--external-draft',action='store_true')
    p.add_argument('--disk-restart',action='store_true');p.add_argument('--request-json',type=Path)
    p.add_argument('--expected-text',default='Hello');p.add_argument('--functional-law',choices=('exact','nonempty_finite'),default='exact')
    p.add_argument('--port',type=int,default=18871);p.add_argument('--max-tokens',type=int,default=16)
    p.add_argument('--prefill-step',type=int,default=128);p.add_argument('--cache-bytes',type=int,default=512<<20)
    p.add_argument('--startup-seconds',type=int,default=240);p.add_argument('--request-seconds',type=int,default=180)
    p.add_argument('--wall-seconds',type=int,default=900);args=p.parse_args(argv)
    if args.output.exists():p.error('refusing existing evidence directory')
    if not args.run_gpu or args.ownership_window is None:p.error('real model HTTP requires explicit GPU and existing ownership')
    if (not 1024<=args.port<=65535 or not 1<=args.max_tokens<=128 or not 16<=args.prefill_step<=512
            or not 64<<20<=args.cache_bytes<=8<<30 or not 30<=args.startup_seconds<=600
            or not 30<=args.request_seconds<=600 or not 60<=args.wall_seconds<=1800):p.error('bounded port/output/prefill/cache/deadlines required')
    if not isinstance(args.execution_policy,dict):p.error('execution-policy must be a JSON object')
    if args.external_draft!=bool(args.execution_policy.get('draft_model')):p.error('external route requires explicit exact draft_model policy; ordinary cannot include one')
    if args.functional_law=='nonempty_finite' and args.request_json is None:p.error('weak oracle requires explicit custom request')
    request,binding=load_request(args.request_json) if args.request_json else (DEFAULT_REQUEST,None)
    owners=args.owner_path or [Path('/Users/Shared/mlxuag/gpu.lock/owner.json'),Path('/tmp/gpu.lock/owner.json')]
    if len(owners)!=2:p.error('exactly two paired owner receipt paths required')
    def guard():return verify_ownership(args.ownership_window,owners)
    ownership=guard();source=source_receipt()
    from mlx2.adapters.deepseek_v4_gguf_artifact import inspect_artifact
    from mlx2.serving import runtime_identity
    artifact=inspect_artifact(args.model_path);build_source=runtime_identity()['source_sha256']
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',args.port))==0:p.error('loopback port already occupied; no foreign process will be stopped')
    args.output.mkdir(parents=True,mode=0o700);base=f'http://127.0.0.1:{args.port}'
    policies={'ordinary':{k:v for k,v in args.execution_policy.items() if k not in EXTERNAL_KEYS},'selected':args.execution_policy}
    external_identity=expected_external_identity(artifact,args.execution_policy,source) if args.external_draft else None
    for name,value in policies.items():atomic(args.output/(name+'-policy.json'),value)
    result={'schema':'mlx2.deepseek-v4-http-apcv2-targeted.v1','status':'running','source':source,
        'checkpoint':args.model_path.name+'@'+artifact['identity']['fingerprint'],'artifact':artifact['identity'],
        'ownership':ownership,'qualification':False,'performance_claim':False,'request':request,'request_binding':binding,
        'expected_external_identity':external_identity,
        'controls':{'external_draft':args.external_draft,'physical_width':1,'max_inflight':1,'max_context':4096,
                    'max_tokens':args.max_tokens,'temperature':0,'cache_bytes':args.cache_bytes,'prefill_step':args.prefill_step,
                    'disk_restart':args.disk_restart,'execution_policy':args.execution_policy},'server_runs':[],'cleanup':[]}
    atomic(args.output/'results.json',result);server=None
    previous=signal.getsignal(signal.SIGALRM)
    previous_stop={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def expired(*_):raise TimeoutError('bounded HTTP unit wall deadline exceeded')
    def interrupted(signum,*_):raise InterruptedError(f'owned HTTP unit interrupted by signal{signum}')
    signal.signal(signal.SIGALRM,expired);signal.alarm(args.wall_seconds)
    for sig in previous_stop:signal.signal(sig,interrupted)
    def launch(label,external,cache):
        nonlocal server
        cache.mkdir(parents=True,exist_ok=True,mode=0o700)
        cmd=command(args.model_path,args.port,args.output/('selected-policy.json' if external else 'ordinary-policy.json'),cache,
            external=external,prefill_step=args.prefill_step,cache_bytes=args.cache_bytes,persist=args.disk_restart)
        entry={'label':label,'command':cmd,'log':label+'.server.log'};result['server_runs'].append(entry);atomic(args.output/'results.json',result)
        server=start_server(cmd,args.output/entry['log'],guard=guard,startup_seconds=args.startup_seconds)
        entry.update(pid=server.process.pid,owned_process_group=server.pgid)
        atomic(args.output/'results.json',result)
        status=wait_ready(server,base,guard=guard,seconds=args.startup_seconds)
        expected=external_identity['fingerprint'] if external else artifact['identity']['fingerprint']
        entry['expected_artifact_identity']=external_identity if external else artifact['identity']
        record_ready_status(entry,status,result,args.output/'results.json',
            fingerprint=expected,build_source=build_source,external=external)
    def retire():
        nonlocal server
        if server is not None:
            cleanup=stop_owned(server);result['cleanup'].append(cleanup);atomic(args.output/'results.json',result)
            if not cleanup['process_group_absent']:raise RuntimeError('owned server group remains; refusing another server')
            server=None
    def generate(label,body):
        guard();value=stream_chat(base,body,args.output/(label+'.sse'),timeout=args.request_seconds)
        result[label]=value;atomic(args.output/'results.json',result);return value
    try:
        cache=args.output/'cache';launch('ordinary-reference',False,cache if not args.external_draft else args.output/'reference-cache')
        reference=generate('ordinary_reference',{**REFERENCE,'temperature':0,'max_tokens':16,'stream':True,'logprobs':True,'skip_writing_prefix_cache':True})
        if reference['text'].strip()!='Hello' or reference['receipt'].get('route')!='ordinary':raise RuntimeError('ordinary functional reference failed')
        if args.external_draft:retire();launch('external-cold-warm',True,cache)
        body={**request,'temperature':0,'max_tokens':args.max_tokens,'stream':True,'logprobs':True}
        cold=generate('cold',body);warm=generate('warm',body)
        result['checks']=pair_check(cold,warm,external=args.external_draft,law=args.functional_law,expected=args.expected_text)
        result['server_runs'][-1]['status_after']=get_json(base,'/v1/status')
        if args.disk_restart:
            retire();launch('disk-restored',args.external_draft,cache)
            restored=generate('disk_restored',body)
            result['disk_checks']=pair_check(cold,restored,external=args.external_draft,law=args.functional_law,expected=args.expected_text)
            result['server_runs'][-1]['status_after']=get_json(base,'/v1/status')
        result['passed']=result['checks']['passed'] and result.get('disk_checks',{'passed':True})['passed']
        result['status']='completed';guard()
    except BaseException as error:
        result.update(status='failed',passed=False,error={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        signal.alarm(0)
        # A second stop request must not interrupt retirement of the child
        # session and release the caller's lease while that model still runs.
        for sig in previous_stop:signal.signal(sig,signal.SIG_IGN)
        try:retire()
        except BaseException as error:result.update(status='failed',passed=False,cleanup_error={'type':type(error).__name__,'message':str(error)})
        if any(not x.get('process_group_absent') for x in result['cleanup']):result.update(status='failed',passed=False)
        atomic(args.output/'results.json',result)
        signal.signal(signal.SIGALRM,previous)
        for sig,handler in previous_stop.items():signal.signal(sig,handler)
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
