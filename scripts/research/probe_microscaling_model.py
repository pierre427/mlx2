#!/usr/bin/env python3
"""One artifact, one owned ordinary/SC-W4A8 model window; never a default change.

Sanity/functional and exploratory ABBA evidence, without thermal holds or
publishable speed claims. The known local Kimi service is restored in finally.
"""
from __future__ import annotations
import argparse
import ast
from collections.abc import Mapping
import gc
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback
import unicodedata
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
from mlx2.desktop_campaign_worker import atomic,check_ownership
from probe_microscaling_components import environment,SERVICE,PLIST


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest():
    files=sorted((ROOT/'src/mlx2').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'scripts/research/probe_microscaling_components.py']
    return {str(p.relative_to(ROOT)):digest(p) for p in files}


def verify_sources(spec):
    for name,value in spec['source_files'].items():
        if digest(ROOT/name)!=value: raise RuntimeError('source changed after sealing: '+name)


def first_divergence(a,b):
    at=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
    if at==len(a)==len(b): return None
    return {'output_position':at,'shared_prefix_length':at,'left':a[at] if at<len(a) else None,
            'right':b[at] if at<len(b) else None}


def normalize_dashes(text):
    return ''.join('-' if unicodedata.category(c)=='Pd' else c for c in text)


def oracle(text,kind,expected):
    text=text.strip()
    if kind=='identifier': return normalize_dashes(text)==expected
    if kind=='json':
        try: return json.loads(text)==expected
        except (ValueError,TypeError): return False
    if kind=='python_add':
        if text.startswith('```'):
            lines=text.splitlines(); text='\n'.join(lines[1:-1]) if lines[-1].strip()=='```' else text
        try:
            tree=ast.parse(text)
            f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='add')
            body=f.body[1:] if f.body and isinstance(f.body[0],ast.Expr) and isinstance(f.body[0].value,ast.Constant) else f.body
            # Only a lone return: earlier statements could rebind a or b.
            if len(body)!=1 or not isinstance(body[0],ast.Return): return False
            ret=body[0].value
            return ([a.arg for a in f.args.args]==['a','b'] and isinstance(ret,ast.BinOp)
                and isinstance(ret.op,ast.Add) and isinstance(ret.left,ast.Name) and ret.left.id=='a'
                and isinstance(ret.right,ast.Name) and ret.right.id=='b')
        except (ValueError,SyntaxError,StopIteration): return False
    raise ValueError('unknown functional oracle')


def make_prompt(tokenizer,target,*,kind='identifier'):
    instructions={
        'identifier':('The shipment identifier is ZX-3187. Reply with the shipment identifier only.','ZX-3187'),
        'json':('Reply with exactly this JSON object, without Markdown: {"ok":true,"count":3}',{'ok':True,'count':3}),
        'python_add':('Write only a Python function named add with parameters a and b that returns a+b. No explanation.',None),
    }
    instruction,expected=instructions[kind]
    def content(n):
        filler='\n'.join(f'Record {i:04d}: The team checked inventory, storage, and delivery schedules.' for i in range(n))
        return 'Archive notes for the current audit.\n'+filler+'\n\nFinal task: '+instruction
    def tokens(text):
        encoded=tokenizer.apply_chat_template([{'role':'user','content':text}],add_generation_prompt=True,
            tokenize=True,enable_thinking=False,return_dict=False)
        # Transformers versions may return a BatchEncoding even for one chat.
        if isinstance(encoded,Mapping): encoded=encoded['input_ids']
        if hasattr(encoded,'tolist'): encoded=encoded.tolist()
        if encoded and isinstance(encoded[0],list):
            if len(encoded)!=1: raise ValueError('expected one rendered chat')
            encoded=encoded[0]
        if not isinstance(encoded,list) or not encoded or any(type(t) is not int for t in encoded):
            raise ValueError('chat template did not produce a flat token-id list')
        return encoded
    low,high=0,max(1,target)
    while low<high:
        mid=(low+high)//2
        if len(tokens(content(mid)))<target: low=mid+1
        else: high=mid
    text=content(low); ids=tokens(text)
    return {'kind':kind,'expected':expected,'messages':[{'role':'user','content':text}],
            'requested_context':target,'rendered_prompt_tokens':len(ids),'rendered_prompt_ids':ids}


def next_row(mx,logits):
    row=logits.reshape(-1,logits.shape[-1])[-1].astype(mx.float32)
    lp=row-mx.logsumexp(row); ids=mx.argsort(lp)[-8:][::-1]
    mx.eval(row,ids,lp)
    return row,{'top': [{'id':int(t),'logprob':float(v)} for t,v in zip(ids.tolist(),lp[ids].tolist())],
                'finite':bool(mx.all(mx.isfinite(row)).item()),'selected':int(mx.argmax(row).item())}


def direct_reference(spec,prompts):
    """Ordinary load smoke and one same-prefix teacher-forced A16/A8 row."""
    import mlx.core as mx
    from mlx2.adapters.registry import inspect_model
    from mlx2.runtime.microscaling.installer import apply_for_adapter,remove
    from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows
    mx.set_default_device(mx.gpu)
    resolution=inspect_model(Path(spec['model'])); adapter=None; handle=None
    result={'kind':'ordinary direct model reference; teacher-forced final prompt position',
            'checkpoint':spec['checkpoint'],'source_revision':spec['source_revision']}
    try:
        adapter=resolution.adapter_type(spec['model'])
        ids=prompts[0]['rendered_prompt_ids']; x=mx.array(ids,dtype=mx.int32)[None]
        with prefill_checkpoint_rows([len(ids)],width=len(ids)):
            ordinary=adapter.model(x,cache=None)
        a,arow=next_row(mx,ordinary); result['ordinary']=arow
        if not arow['finite']: raise RuntimeError('ordinary direct reference is nonfinite')
        if spec['scope']!='off':
            handle=apply_for_adapter(adapter,spec['scope'],max_decode_rows=1)
            with prefill_checkpoint_rows([len(ids)],width=len(ids)):
                candidate=adapter.model(x,cache=None)
            b,brow=next_row(mx,candidate)
            delta=a-b; p=mx.softmax(a)
            kl=mx.sum(p*((a-mx.logsumexp(a))-(b-mx.logsumexp(b))))
            result['candidate']=brow
            result['comparison']={'top1_equal':arow['selected']==brow['selected'],
                'max_abs_logit_difference':float(mx.max(mx.abs(delta)).item()),
                'mean_abs_logit_difference':float(mx.mean(mx.abs(delta)).item()),
                'KL_reference_to_candidate':float(kl.item()),
                'meaning':'approximate activation-law comparison at one shared teacher-forced prompt position'}
            result['microscaling']=handle.status()
            if not brow['finite'] or not result['microscaling']['engaged']:
                raise RuntimeError('direct candidate nonfinite or did not engage')
        result['passed']=True
        return result
    finally:
        if handle is not None: remove(handle)
        if adapter is not None: adapter.close()


def find(document,key):
    if isinstance(document,dict):
        if key in document: return document[key]
        for value in document.values():
            found=find(value,key)
            if found is not None: return found
    elif isinstance(document,list):
        for value in document:
            found=find(value,key)
            if found is not None: return found
    return None


def request_json(base,path,payload=None,*,admin_token=None,timeout=300):
    headers={'Content-Type':'application/json'}
    if admin_token: headers['Authorization']='Bearer '+admin_token
    request=Request(base+path,data=None if payload is None else json.dumps(payload).encode(),headers=headers)
    try:
        with urlopen(request,timeout=timeout) as response: return json.load(response)
    except HTTPError as error:
        body=error.read(1<<20).decode(errors='replace')
        raise RuntimeError(f'HTTP{error.code}: {body}') from error


def chat(base,prompt,path,*,max_tokens=48,window=None):
    from mlx2.generation_trace import reconcile_completion_tokens
    body={'model':prompt['model_id'],'messages':prompt['messages'],'temperature':0,'max_tokens':max_tokens,
          'stream':False,'return_token_trace':True,'enable_thinking':False}
    if window is not None:
        body.update(logprobs=True,top_logprobs=8,logprobs_start=window+1,logprobs_end=window+1)
    started=time.perf_counter(); raw=request_json(base,'/v1/chat/completions',body)
    elapsed=time.perf_counter()-started; atomic(path,raw)
    accounting=reconcile_completion_tokens(raw,maximum=max_tokens)
    receipt=raw['mlx2']; trace=receipt['token_trace']; text=raw['choices'][0]['message'].get('content') or ''
    return {'text':text,'token_ids':trace['token_ids'],'ordinary_execution_widths':trace['ordinary_execution_widths'],
        'receipt':receipt,'usage':raw['usage'],'accounting':accounting,'seconds':elapsed,
        'ttft_seconds':find(receipt,'ttft_seconds'),'oracle_passed':oracle(text,prompt['kind'],prompt['expected']),
        'raw_response_path':path.name,'logprobs':raw['choices'][0].get('logprobs'),
        'finish_reason':raw['choices'][0].get('finish_reason')}


def pair_assessment(cold,warm):
    failures=[]; numerical_failures=[]
    for name,row in [('cold',cold),('warm',warm)]:
        r=row['receipt']; cached=r.get('cached_tokens')
        if r.get('cache')!='apcv2': failures.append(name+': cache receipt is not APCv2')
        if r.get('route')!='ordinary' or not r.get('route_receipt'): failures.append(name+': ordinary route receipt missing')
        if type(cached) is not int or (cached!=0 if name=='cold' else cached<=0):
            failures.append(name+': unexpected cached token count '+repr(cached))
        if any(width not in (None,1) for width in row['ordinary_execution_widths']): failures.append(name+': ordinary execution width differs from one')
    divergence=first_divergence(cold['token_ids'],warm['token_ids'])
    if divergence is not None: numerical_failures.append('width-one cold/warm token mismatch')
    if cold['text']!=warm['text']: numerical_failures.append('cold/warm text differs')
    return {'methodological_passed':not failures,'numerical_passed':not numerical_failures,
            'contract_failures':failures,'numerical_failures':numerical_failures,
            'failures':failures+numerical_failures,'first_divergence':divergence,
            'numerical_classification':'exact_token_parity' if divergence is None else 'width_one_exact_parity_failed',
            'functional_oracles_passed':cold['oracle_passed'] and warm['oracle_passed']}


def _window_table(row):
    content=(row.get('logprobs') or {}).get('content') or []
    if len(content)!=1: return None
    item=content[0]
    return {int(r['id']):float(r['logprob']) for r in [item,*item.get('top_logprobs',[])] if 'id' in r and 'logprob' in r}


def compare_approximate(left,right,windows=None):
    divergence=first_divergence(left['token_ids'],right['token_ids'])
    result={'first_divergence':divergence,'both_functional_oracles_pass':left['oracle_passed'] and right['oracle_passed'],
            'left_functional_passed':left['oracle_passed'],'right_functional_passed':right['oracle_passed'],
            'classification':'exact_token_parity' if divergence is None else 'approximate_token_change_unclassified',
            'scope':'A16 versus A8 approximation; not a width-one APCv2 equivalence verdict'}
    if divergence and windows and divergence['left'] is not None and divergence['right'] is not None:
        at=divergence['output_position']
        prefix_reproduced=all(w['token_ids'][:at+1]==original['token_ids'][:at+1] for w,original in zip(windows,(left,right)))
        result['window_prefix_reproduced']=prefix_reproduced
        if not prefix_reproduced:
            result['classification']='approximate_token_change_window_not_reproduced'
            return result
        a,b=(_window_table(w) for w in windows)
        if a and b and all(t in table for t in (divergence['left'],divergence['right']) for table in (a,b)):
            cross=[max(a.values())-a[divergence['right']],max(b.values())-b[divergence['left']]]
            result['cross_selected_token_gaps_nats']=cross
            if max(cross)<=.5 and result['both_functional_oracles_pass']:
                result['classification']='near_tie_equivalent_approximate_arms'
            else: result['classification']='approximate_high_margin_or_functional_change'
        result['window_scope']='only the already selected first-divergence position; no downstream logits interpreted'
    return result


def one_server(spec,prompts,out,arm,ordinal,window_requests):
    from mlx2.desktop_campaign_process import spawn
    with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    base=f'http://127.0.0.1:{port}'; out.mkdir(exist_ok=False)
    token=secrets.token_hex(24); token_file=out/'admin-token'; token_file.write_text(token); token_file.chmod(0o600)
    command=[sys.executable,'-m','mlx2.server','--model',spec['model'],'--host','127.0.0.1','--port',str(port),
        '--fabric','off','--ordinary','--qualification-mode','--max-lanes','1','--max-inflight','1',
        '--max-context',str(max(p['rendered_prompt_tokens'] for p in prompts)+512),'--default-max-tokens','64',
        '--cache-bytes',str(2<<30),'--cache-dir',str(out/'apcv2'),'--prefill-step',str(spec['prefill_step']),
        '--lane-matmul','off','--int8-prefill','off','--microscaling-prefill',spec['scope'] if arm=='candidate' else 'off',
        '--gpu-keep-warm-seconds','0','--residency-keep-seconds','0','--admin-token-file',str(token_file)]
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    row={'arm':arm,'ordinal':ordinal,'command':command,'cells':[]}; server=None
    log=(out/'server.log').open('w')
    def clear(): return request_json(base,'/v1/admin/apc/clear',{'reason':'bounded-sc-mxfp4-probe'},admin_token=token)
    try:
        server=spawn(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+spec['startup_seconds']
        while True:
            if server.poll() is not None: raise RuntimeError('loopback server exited before readiness')
            try:
                status=request_json(base,'/v1/status',timeout=5)
                if status.get('ready') or status.get('state')=='ready' or status.get('status')=='ready': break
            except (URLError,TimeoutError,ConnectionError): pass
            if time.monotonic()>deadline: raise TimeoutError('loopback startup timed out')
            time.sleep(.2)
        row['status_before']=status
        if status.get('artifact')!=spec['artifact_fingerprint']: raise RuntimeError('server artifact fingerprint differs')
        row['warmups']=[]
        # Compile each measured shape before timing cold APCv2 inputs. Keep
        # startup/compilation evidence separate from steady-state latency.
        for index,prompt in enumerate(prompts):
            if ordinal>=2 and prompt['kind']!='identifier': continue
            clear()
            row['warmups'].append(chat(base,prompt,out/f'warmup-{index}.json',max_tokens=1))
        for index,prompt in enumerate(prompts):
            if ordinal>=2 and prompt['kind']!='identifier': continue
            clear(); cold=chat(base,prompt,out/f'cell-{index}-cold.json'); warm=chat(base,prompt,out/f'cell-{index}-warm.json')
            pair=pair_assessment(cold,warm)
            cell={'prompt_index':index,'requested_context':prompt['requested_context'],'kind':prompt['kind'],
                  'cold':cold,'warm':warm,'pair':pair}
            if pair['first_divergence'] is not None:
                at=pair['first_divergence']['output_position']; clear()
                cell['first_divergence_window']=[chat(base,prompt,out/f'cell-{index}-cold-window.json',window=at),
                    chat(base,prompt,out/f'cell-{index}-warm-window.json',window=at)]
            row['cells'].append(cell); atomic(out/'arm.json',row)
        row['windows']={}
        for index,at in window_requests.items():
            clear(); row['windows'][str(index)]=chat(base,prompts[index],out/f'cross-arm-{index}-window.json',window=at)
        # Worker snapshots publish asynchronously; response receipts are authoritative for engagement.
        row['status_after']=request_json(base,'/v1/status')
        if arm=='candidate':
            receipts=[find(c['cold']['receipt'],'microscaling_prefill') for c in row['cells']]
            row['engaged']=any(r and r.get('engaged') for r in receipts)
            if not row['engaged']: raise RuntimeError('selected candidate did not engage on serving requests')
        else: row['engaged']=False
        row['completed']=True
        return row
    finally:
        if server is not None:
            server.stop(timeout=20); row['server_stopped']=not server.live(); row['server_returncode']=server.returncode
        token_file.unlink(missing_ok=True); log.close(); atomic(out/'arm.json',row)


def unit(spec):
    out=Path(spec['out']); report={**spec,'qualified':False,'passed':False,'arms':[],
        'scope_note':'functional/engagement/width-one APCv2 and exploratory matched ABBA; no performance qualification'}
    try:
        report['ownership']=check_ownership(spec); verify_sources(spec); report['before']=environment()
        if report['before']['foreign_model_processes']: raise RuntimeError('foreign model process; yield')
        from mlx2.adapters.registry import inspect_model
        resolution=inspect_model(Path(spec['model']))
        if resolution.artifact_fingerprint!=spec['artifact_fingerprint']: raise RuntimeError('model identity changed')
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(spec['model'],local_files_only=True)
        prompts=[make_prompt(tokenizer,n) for n in spec['contexts']]
        prompts += [make_prompt(tokenizer,max(512,spec['contexts'][0]),kind=kind) for kind in ('json','python_add')]
        for prompt in prompts: prompt['model_id']=Path(spec['model']).name
        atomic(out/'prompts.json',prompts)
        report['direct_reference']=direct_reference(spec,prompts)
        gc.collect()
        import mlx.core as mx
        mx.synchronize(); mx.clear_cache(); report['direct_cleanup_active_bytes']=mx.get_active_memory()
        cross_windows={int(k):int(v) for k,v in spec.get('capture_windows',{}).items()}; initial={}
        if any(index<0 or index>=len(prompts) or at<0 for index,at in cross_windows.items()):
            raise ValueError('capture window outside prompt/output geometry')
        order=['ordinary','candidate','candidate','ordinary'] if spec['scope']!='off' else ['ordinary']
        for ordinal,arm in enumerate(order):
            verify_sources(spec); check_ownership(spec)
            row=one_server(spec,prompts,out/f'{ordinal}-{arm}',arm,ordinal,cross_windows)
            report['arms'].append(row); atomic(out/'results.json',report)
            if arm not in initial: initial[arm]=row
            if arm=='candidate' and 'ordinary' in initial:
                ordinary={c['prompt_index']:c for c in initial['ordinary']['cells']}
                for cell in row['cells']:
                    index=cell['prompt_index']; diff=first_divergence(ordinary[index]['cold']['token_ids'],cell['cold']['token_ids'])
                    if diff and diff['left'] is not None and diff['right'] is not None: cross_windows[index]=diff['output_position']
        comparisons=[]
        if 'candidate' in initial:
            left={c['prompt_index']:c for c in initial['ordinary']['cells']}; right={c['prompt_index']:c for c in initial['candidate']['cells']}
            # Windows collected only after the first mismatch was localized.
            last_a=report['arms'][-1].get('windows',{}); last_b=report['arms'][2].get('windows',{})
            for index in sorted(set(left)&set(right)):
                windows=[last_a[str(index)],last_b[str(index)]] if str(index) in last_a and str(index) in last_b else None
                comparisons.append({'prompt_index':index,**compare_approximate(left[index]['cold'],right[index]['cold'],windows)})
        report['approximate_comparisons']=comparisons
        method_failures=[]; numerical_failures=[]
        for arm in report['arms']:
            for cell in arm['cells']:
                method_failures.extend(f"{arm['ordinal']}:{cell['prompt_index']}: "+x for x in cell['pair']['contract_failures'])
                numerical_failures.extend(f"{arm['ordinal']}:{cell['prompt_index']}: "+x for x in cell['pair']['numerical_failures'])
        report['methodological_failures']=method_failures
        report['methodological_passed']=not method_failures
        report['cold_warm_numerical_failures']=numerical_failures
        report['cold_warm_numerical_passed']=not numerical_failures
        report['functional_outcomes']=[{'arm':r['arm'],'ordinal':r['ordinal'],'passed':sum(c['cold']['oracle_passed'] for c in r['cells']),'total':len(r['cells'])} for r in report['arms']]
        report['ab_summary']=[]
        for target in spec['contexts']:
            values={}
            for arm in ('ordinary','candidate'):
                values[arm]=[c['cold']['ttft_seconds'] for r in report['arms'] if r['arm']==arm for c in r['cells']
                    if c['kind']=='identifier' and c['requested_context']==target and c['cold']['ttft_seconds'] is not None]
            report['ab_summary'].append({'requested_context':target,'ttft_seconds':values,
                'median_ttft_seconds':{a:statistics.median(v) if v else None for a,v in values.items()},
                'timing_scope':'exploratory HTTP TTFT; no thermal gate or publishable serving speed claim'})
        report['functional_regressions']=[c['prompt_index'] for c in comparisons if c['left_functional_passed'] and not c['right_functional_passed']]
        report['ordinary_functional_passed']=all(c['cold']['oracle_passed'] and c['warm']['oracle_passed'] for r in report['arms'] if r['arm']=='ordinary' for c in r['cells'])
        report['candidate_functional_passed']=all(c['cold']['oracle_passed'] and c['warm']['oracle_passed'] for r in report['arms'] if r['arm']=='candidate' for c in r['cells']) if spec['scope']!='off' else None
        report['passed']=not method_failures and not numerical_failures and report['ordinary_functional_passed'] and report['candidate_functional_passed'] is not False
        verify_sources(spec)
    except BaseException as e:
        report['passed']=False
        report['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
        print(report['error']['traceback'],flush=True)
    finally:
        report['after']=environment(); report['finished_at']=time.time(); atomic(out/'results.json',report)
    return 0 if report['passed'] else 1


def owned(args):
    from mlx2.adapters.registry import inspect_model
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    from mlx2.desktop_campaign_process import spawn
    out=args.output.resolve(); out.mkdir(exist_ok=False)
    # Either resolver shape (Flash-Next returns the identity itself).
    model=args.model.expanduser().resolve(); identity={'fingerprint':inspect_model(model).artifact_fingerprint}
    manifest=source_manifest(); revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    spec={'out':str(out),'job_id':out.name,'model':str(model),'scope':args.scope,'contexts':args.contexts,
        'artifact_fingerprint':identity['fingerprint'],'checkpoint':model.name+'@sha256:'+identity['fingerprint'],
        'source_revision':revision,'source_files':manifest,'startup_seconds':args.startup_seconds,
        'prefill_step':args.prefill_step,
        'capture_windows':dict(args.capture_window),
        'window_seconds':args.window_seconds,'mcp_url':'http://127.0.0.1:8766/mcp','started_at':time.time()}
    atomic(out/'spec.json',spec)
    (out/'source').mkdir()
    dirty=subprocess.check_output(['git','ls-files','-m','-o','--exclude-standard','--','src','scripts'],cwd=ROOT,text=True).splitlines()
    for name in dirty:
        source=ROOT/name
        if source.is_file() and name in manifest:
            target=out/'source'/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(source.read_bytes())
    before=environment(); atomic(out/'prelease-environment.json',before)
    if before['waiters'] or any(p.is_dir() for p in (Path('/tmp/gpu.lock'),Path('/Users/Shared/mlxuag/gpu.lock'))):
        raise RuntimeError('yield to current GPU waiter/owner before pausing service')
    loaded=subprocess.run(['launchctl','print',f'gui/{os.getuid()}/{SERVICE}'],capture_output=True).returncode==0
    service={'label':SERVICE,'loaded_before':loaded,'paused':False,'restored':False}; child=None
    try:
        if loaded:
            subprocess.run(['launchctl','bootout',f'gui/{os.getuid()}/{SERVICE}'],check=True)
            service['paused']=True; atomic(out/'service.json',service)
        with DesktopCPGLease(out,out.name,checkpoint=spec['checkpoint'],source_revision=revision) as lease:
            try:
                command=[sys.executable,str(ROOT/'scripts/run_with_gpu_locks.py'),'--session',lease.owner['cpg_session'],
                    '--label',out.name,'--desktop-cpg-lease',str(lease.path),'--receipt',str(out/'paired-locks.json'),
                    '--',sys.executable,'-u',str(Path(__file__).resolve()),'unit','--output',str(out)]
                child=spawn(command,start_new_session=True); deadline=time.monotonic()+args.window_seconds
                while child.poll() is None:
                    lease.check()
                    if time.monotonic()>deadline: raise TimeoutError('bounded artifact window exceeded')
                    try: child.wait(timeout=1)
                    except subprocess.TimeoutExpired: pass
                lease.success=child.returncode==0
                return child.returncode
            finally:
                if child is not None: child.stop(timeout=20)
    finally:
        if service['paused']:
            r=subprocess.run(['launchctl','bootstrap',f'gui/{os.getuid()}',str(PLIST)],capture_output=True,text=True)
            service.update(restored=r.returncode==0,restore_returncode=r.returncode,restore_stderr=r.stderr)
        atomic(out/'service.json',service)
        if service['paused'] and not service['restored']: raise RuntimeError('Kimi service restoration failed')


if __name__=='__main__':
    def interrupted(signum,frame): raise KeyboardInterrupt(f'signal {signum}')
    for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP): signal.signal(signum,interrupted)
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['owned','unit'])
    p.add_argument('--model',type=Path); p.add_argument('--scope',choices=['off','experts','mlp','all'],default='experts')
    p.add_argument('--output',type=Path,required=True); p.add_argument('--contexts',type=int,nargs='+',default=[512,2048])
    p.add_argument('--prefill-step',type=int,default=512,choices=[512,1024,2048,4096])
    def capture_window(value):
        try:
            index,position=map(int,value.split(':'))
            if index<0 or not 0<=position<256: raise ValueError()
            return index,position
        except ValueError: raise argparse.ArgumentTypeError('expected PROMPT_INDEX:OUTPUT_POSITION (position 0..255)')
    p.add_argument('--capture-window',type=capture_window,action='append',default=[],
                   help='capture one already-localized output position, also across separate artifacts')
    p.add_argument('--startup-seconds',type=int,default=300); p.add_argument('--window-seconds',type=int,default=1200)
    a=p.parse_args()
    if a.mode=='owned' and (a.model is None or any(n<512 or n>16384 for n in a.contexts) or not 60<=a.window_seconds<=3600): p.error('model, bounded contexts 512..16384 and window60..3600 required')
    raise SystemExit(unit(json.loads((a.output/'spec.json').read_text())) if a.mode=='unit' else owned(a))
