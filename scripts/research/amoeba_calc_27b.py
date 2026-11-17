#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
# Scope and terms: provenance/LICENSE-calculator-scope.json; provenance/LICENSE-calculator.txt
"""Bounded, source-bound 27B calculator transfer through ordinary mlx2 serving.

Prepare freezes questions without loading a model. Run requires an external
CPG/paired-lock ownership wrapper. All stock batches finish before calculator
batches; the ordinary model is not modified by neural carve tiles. No tuning
uses the scored set. Logprob IDs enable exact client-side loop alarms; these
instrumented timings are not a performance benchmark.
"""
from __future__ import annotations
import argparse, concurrent.futures, faulthandler, hashlib, json, os
from pathlib import Path
import runpy, signal, subprocess, sys, time, urllib.request
from fractions import Fraction

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/research'),str(ROOT/'scripts'),str(ROOT)]
MODEL=Path.home()/'mlx-models/Qwen3.8-27B-BF16-q8-mlx2'


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)


def prepare(out):
    from amoeba_calc_eval import problems
    out.mkdir(parents=True,exist_ok=False)
    hard=[json.loads(x) for x in (ROOT/'benchmarks/calculator_hard1000/20261117-latex-repair/problems.jsonl').read_text().splitlines()]
    counts={};rows=[]
    for r in hard:
        c=r['category'];counts[c]=counts.get(c,0)+1
        if counts[c]<=4:rows.append(r)
    ordinary=problems(20)
    for suite,indices in [('math',range(0,100,5)),('ood',[c*25+i for c in range(4) for i in range(4)]),('noisy_words',range(8)),('gsm8k',range(20))]:
        group=[r for r in ordinary if r['suite']==suite]
        rows.extend(group[i] for i in indices)
    counting=[("How many r's are in strawberry?",'3'),("Count 's' in 'Mississippi'.",'4'),("Count 'r' in 'stRawbeRry' case-sensitive.",'1'),("Count 'r' in 'stRawbeRry' ignore case.",'3'),('How many items are in []?','0'),('How many items are in [4, 5, 4, 9]?','4'),('What is the sum of [0.1, 0.2, 6.5]?','6.8'),('What is the sum of [123456789123.12345678, 0.00000001]?','123456789123.12345679')]
    for i,(q,a) in enumerate(counting):rows.append(dict(id=f'count-{i}',suite='counting',category='literal_count',question=q,expected=a))
    copy=['Call 555-1234 on 12/25/2024 at 10:30.','Order #4471-889 shipped; version 3.11.2 = current.','The price is $5.99 and the score was 3-2.','x = 12; y = x + 4; print(y)','The year is 1999; room 42 has 8 chairs.','E = mc^2 is a famous equation.','Set k = 7 before starting.','The date is 2026-11-17.','2 + 3 =','3 * 9 = 27','Count \'r\' in \'strawberry\' =','Sum [1, 2, 3] =']
    for i,text in enumerate(copy):rows.append(dict(id=f'copy-{i}',suite='nonmath',category='copy',question='Repeat exactly this text, with no explanation:\n'+text,expected_text=text))
    for i,q in enumerate(['Describe a lighthouse in two sentences.','Explain how a compiler works in two sentences.','Describe a pine forest after rain.','Write a short story about a gardener.']):
        rows.append(dict(id=f'prose-{i}',suite='nonmath',category='prose',question=q))
    assert len(rows)==128 and len({r['id'] for r in rows})==128
    payload=''.join(json.dumps(r)+'\n' for r in rows).encode();(out/'problems.jsonl').write_bytes(payload)
    dump(out/'freeze.json',{'n':128,'sha256':hashlib.sha256(payload).hexdigest(),'selection':'First four per hard1000 category; fixed basic/OOD/noisy/GSM indices; eight literal counting cases; twelve copy negatives; four prose. Reused diagnostic cases, not a new held-out benchmark.','created_at':time.time()})


def train(out,model):
    os.environ['MLX_METAL_FAST_SYNCH']='0'
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    from mlx2.runtime.generation_stream import bind_generation_thread_stream
    bind_generation_thread_stream()
    import amoeba27b_lib as A
    A.footprint_watchdog(80.0,print)
    from research.amoeba.det_backend27 import Backend27
    original=Backend27.forward;calls=[0]
    out.parent.mkdir(parents=True,exist_ok=True)
    stack=(out.parent/'probe-watchdog.txt').open('x')
    faulthandler.enable(file=stack);faulthandler.dump_traceback_later(180,file=stack,exit=True)
    def forward(self,*a,**k):
        result=original(self,*a,**k);calls[0]+=1
        faulthandler.dump_traceback_later(45,file=stack,exit=True)
        if calls[0]%16==0:print('probe forwards',calls[0],flush=True)
        return result
    Backend27.forward=forward
    from mlx2.runtime import amoeba_det_tiles as D
    original_answer=D.calculator_answer;parses=[0]
    def answer(*a,**k):
        result=original_answer(*a,**k);parses[0]+=1
        if parses[0]%128==0:faulthandler.dump_traceback_later(45,file=stack,exit=True)
        return result
    D.calculator_answer=answer
    original_init=Backend27.__init__
    def init(self,*a,**k):
        original_init(self,*a,**k)
        tok=self.host.tok
        prompt=tok.apply_chat_template([{'role':'user','content':'What is 2+2? Reply with the number only.'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tok.encode(prompt,add_special_tokens=False);generated=[]
        eos={tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>')}
        for _ in range(16):
            logits,_,_=self.forward(mx.array([ids]))
            if not bool(mx.all(mx.isfinite(logits[:,-1])).item()):raise RuntimeError('nonfinite ordinary-reference logits')
            token=int(mx.argmax(logits[0,-1]).item());ids.append(token);generated.append(token)
            if token in eos:break
        output=tok.decode(generated)
        dump(out.parent/'ordinary-smoke.json',dict(checkpoint=self.artifact,output=output,token_ids=generated,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),route='ordinary',tiles=False))
        if output.replace('<|im_end|>','').replace('<|endoftext|>','').strip()!='4':raise RuntimeError('ordinary-reference arithmetic smoke failed')
    Backend27.__init__=init
    sys.argv=['amoeba_det_calc_cpu.py','--backend','27b','--model',str(model),'--out',str(out),'--layers','24,36,48,56','--n-pos','400','--n-neg','40','--arms','host,force','--probe-only']
    try:runpy.run_path(str(ROOT/'scripts/amoeba_det_calc_cpu.py'),run_name='__main__')
    finally:faulthandler.cancel_dump_traceback_later();stack.close()


def get(url,timeout=5):
    with urllib.request.urlopen(url,timeout=timeout) as r:return json.load(r)


def chat(base,row,tok,max_tokens=4096):
    from amoeba_calc_defenses import CompletionDefenses
    math_case='expected' in row
    messages=([{'role':'system','content':'Solve the problem. Give an exact answer, with brief working if useful. End with #### followed by the final numeric answer.'}] if math_case else [])+[{'role':'user','content':row['question']}]
    body=dict(messages=messages,temperature=0,max_tokens=max_tokens,enable_thinking=False,reasoning_effort='none',stream=True,logprobs=True,top_logprobs=0,stream_options={'include_usage':True})
    guard=CompletionDefenses();ids=[];parts=[];terminal={};start=time.monotonic();finish=None;last_progress=start
    req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=60) as response:
        for raw in response:
            if time.monotonic()-last_progress>60:raise TimeoutError('no generated token progress for 60 seconds')
            if not raw.startswith(b'data:'):continue
            data=raw[5:].strip()
            if data==b'[DONE]':break
            item=json.loads(data)
            if 'error' in item:raise RuntimeError(str(item['error']))
            choices=item.get('choices') or []
            if not choices:continue
            choice=choices[0];delta=choice.get('delta') or {}
            parts.append(delta.get('content') or '')
            if delta.get('content') or (choice.get('logprobs') or {}).get('content'):last_progress=time.monotonic()
            for lp in (choice.get('logprobs') or {}).get('content',[]):
                ids.append(int(lp['id']))
                text=tok.decode(ids) if len(ids)%16==0 else None
                if guard.feed(ids[-1],text):finish='defense';break
            if finish=='defense':break
            if choice.get('finish_reason') is not None:
                finish=choice['finish_reason'];terminal=item
    text=''.join(parts)
    if not finish:raise RuntimeError('stream ended without a finish reason')
    result=dict(output=text,token_ids=ids,finish_reason=finish,wall_s=time.monotonic()-start,defenses=guard.receipt(),terminal=terminal)
    from amoeba_calc_eval import numeric_answer
    if math_case:result['correct']=finish not in ('length','defense') and numeric_answer(text)==Fraction(row['expected'])
    if 'expected_text' in row:result['correct']=text.strip()==row['expected_text']
    return result


def run(out,corpus,model,manifest,port,width,max_tokens,arms=('stock','calculator'),drafter=None,
        *, controls_only=False, acceptance_questions=0):
    data=(corpus/'problems.jsonl').read_bytes();freeze=json.loads((corpus/'freeze.json').read_text())
    assert hashlib.sha256(data).hexdigest()==freeze['sha256']
    rows=[json.loads(x) for x in data.splitlines()]
    out.mkdir(parents=True,exist_ok=False)
    from mlx2.adapters.registry import inspect_model
    from transformers import AutoTokenizer
    resolution=inspect_model(model);fp=resolution.artifact_fingerprint;checkpoint=model.name+'@'+fp.split(':')[-1][:8]
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    arm_routes = {arm: 'external_draft' if arm.endswith('_dflash') else 'ordinary' for arm in arms}
    metadata=dict(checkpoint=checkpoint,fingerprint=fp,source_revision=rev,corpus=freeze,width=width,max_tokens=max_tokens,arm_order=list(arms),routes=arm_routes,qualified=False,profile='native HTTP serving, logprob-ID diagnostics; no publishable timing claim',metal_fast_synch='0')
    metadata["controls_only"] = controls_only
    metadata["acceptance_questions"] = acceptance_questions
    if 'external_draft' in arm_routes.values():
        from mlx2.adapters.dflash2 import inspect_drafter
        if drafter is None:
            raise ValueError('DFlash baseline requires an explicit local drafter')
        draft_identity = inspect_drafter(drafter, model)
        metadata['drafter'] = {'path': str(drafter), 'fingerprint': draft_identity['fingerprint']}
    dump(out/'metadata.json',metadata)
    results={};base=f'http://127.0.0.1:{port}'
    controls=[dict(id='control-prose',question='Reply with exactly the word PINE.',expected_text='PINE'),dict(id='control-math',question='Calculate (19+8)*7.',expected='189'),dict(id='control-count',question="How many r's are in strawberry?",expected='3'),dict(id='control-fraction',question=r'Calculate \(\frac{3}{8}+0.125\).',expected='0.5')]
    def interrupted(signum,frame):raise InterruptedError(f'research runner signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    for arm in arms:
        ad=out/arm;ad.mkdir();cmd=[sys.executable,'-m','mlx2.server','--model',str(model),'--port',str(port),'--max-lanes',str(width)]
        if arm_routes[arm] == 'external_draft':
            cmd+=['--ast','--ast-drafter',str(drafter)]
        else:cmd+=['--ordinary']
        if arm.startswith('calculator'):
            policy=ad/'execution-policy.json';dump(policy,{'amoeba_det_tiles':str(manifest.resolve())})
            cmd+=['--execution-policy',str(policy.resolve())]
        cmd+=['--drain-on-sigterm','2','--fault-log',str((ad/'fault.log').resolve())]
        dump(ad/'command.json',cmd)
        env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'MLX_METAL_FAST_SYNCH':'0','HF_HUB_OFFLINE':'1'}
        log=(ad/'server.log').open('x');proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+240
            while True:
                if proc.poll() is not None:raise RuntimeError(f'{arm} server exited {proc.returncode}')
                try:
                    if get(base+'/health').get('status')=='ok':break
                except (OSError,ValueError):pass
                if time.monotonic()>deadline:raise TimeoutError('server startup')
                time.sleep(1)
            status = get(base+'/v1/status')
            dump(ad/'status-start.json',status)
            if (status.get('settings') or {}).get('route') != arm_routes[arm]:
                raise RuntimeError(f'{arm} loaded a route other than {arm_routes[arm]}')
            sequential=[chat(base,r,tok,128) for r in controls]
            with concurrent.futures.ThreadPoolExecutor(width) as pool:batched=list(pool.map(lambda r:chat(base,r,tok,128),controls))
            dump(ad/'controls.json',[dict(case=r,sequential=s,batched=b) for r,s,b in zip(controls,sequential,batched)])
            dump(ad/'status-controls.json',get(base+'/v1/status'))
            if arm.startswith('calculator') and any(not x.get('correct',True) for x in sequential+batched):
                raise RuntimeError('calculator functional B1/B4 control failed')
            values=[]
            # Pure copy/prose first, before any scored math, to localize false fires.
            ordered=[r for r in rows if r['suite']=='nonmath']
            if not controls_only:
                ordered += [r for r in rows if r['suite']!='nonmath']
            with concurrent.futures.ThreadPoolExecutor(width) as pool:
                for i in range(0,len(ordered),width):
                    chunk=ordered[i:i+width]
                    dump(ad/'progress.json',dict(completed=i,total=len(ordered),pending=[r['id'] for r in chunk],time=time.time()))
                    answers=list(pool.map(lambda r:chat(base,r,tok,max_tokens),chunk))
                    for r,a in zip(chunk,answers):
                        record=dict(**r,**a);values.append(record)
                        with (ad/'rows.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    print(arm,i+len(chunk),'/',len(ordered),'correct',sum(x.get('correct',False) for x in values),flush=True)
            if acceptance_questions and arm_routes[arm] == 'external_draft':
                subprocess.run(
                    [sys.executable, str(ROOT/'scripts/spec_acceptance_gsm8k.py'),
                     '--base-url', base, '--checkpoint', model.name+'@'+fp,
                     '--output', str(ad/'acceptance'),
                     '--num-questions', str(acceptance_questions),
                     '--format', 'chat-fewshot', '--max-tokens', str(max_tokens),
                     '--request', '{"enable_thinking":false}'],
                    cwd=ROOT, env=env, check=True, timeout=1800,
                )
            dump(ad/'status-end.json',get(base+'/v1/status'))
            dump(ad/'progress.json',dict(completed=len(ordered),total=len(ordered),pending=[],time=time.time()))
            results[arm]={r['id']:r for r in values}
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
            log.close()
    summary={**metadata,'scores':{},'comparisons':{}}
    for arm,values in results.items():
        stats={}
        for r in values.values():
            if 'correct' not in r:continue
            key=r['suite'];st=stats.setdefault(key,dict(n=0,correct=0,truncated=0,defended=0));st['n']+=1;st['correct']+=r['correct'];st['truncated']+=r['finish_reason']=='length';st['defended']+=r['finish_reason']=='defense'
        summary['scores'][arm]=stats
    for calc_arm in (arm for arm in results if arm.startswith('calculator')):
        for baseline in (arm for arm in results if arm.startswith('stock')):
            paired = {'helped': 0, 'hurt': 0, 'prose_token_exact': 0}
            for case_id, stock_row in results[baseline].items():
                calc_row = results[calc_arm].get(case_id)
                if calc_row is None:
                    continue
                if 'expected' in stock_row:
                    a, b = stock_row['correct'], calc_row['correct']
                    paired['helped'] += b and not a
                    paired['hurt'] += a and not b
                elif stock_row['category'] == 'prose':
                    paired['prose_token_exact'] += stock_row['token_ids'] == calc_row['token_ids']
            summary['comparisons'][baseline+' vs '+calc_arm] = paired
    dump(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','train','run']);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--model',type=Path,default=MODEL);ap.add_argument('--corpus',type=Path);ap.add_argument('--manifest',type=Path);ap.add_argument('--port',type=int,default=8391);ap.add_argument('--width',type=int,default=4);ap.add_argument('--max-tokens',type=int,default=4096);ap.add_argument('--arms',default='stock,calculator');ap.add_argument('--drafter',type=Path)
    ap.add_argument('--controls-only', action='store_true', help='Run B1/B4 function controls and nonmath cases')
    ap.add_argument('--acceptance-questions', type=int, default=0, help='Number of standard GSM8K cases per external draft arm')
    args=ap.parse_args()
    if args.phase=='prepare':prepare(args.out)
    elif args.phase=='train':train(args.out,args.model)
    else:
        if not 8391<=args.port<=8395:raise ValueError('development port must be8391-8395')
        arms=args.arms.split(',')
        if any(a not in ('stock','calculator','stock_dflash','calculator_dflash') for a in arms) or len(set(arms))!=len(arms):raise ValueError('invalid arms')
        run(args.out,args.corpus,args.model,args.manifest,args.port,args.width,args.max_tokens,arms,args.drafter,
            controls_only=args.controls_only, acceptance_questions=args.acceptance_questions)

if __name__=='__main__':main()
