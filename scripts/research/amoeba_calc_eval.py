# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
# Scope and earlier grants: provenance/LICENSE-calculator-scope.json
# Terms: provenance/LICENSE-calculator.txt
"""Paired calculator evaluation through stock MLX-LM; research-only, unqualified."""
from __future__ import annotations
import argparse, ast, hashlib, json, os, random, re, subprocess, time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


_DECIMAL = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_BOLD_SPLIT_DECIMAL = re.compile(r"(?<![\w.*])\*\*(\$?[+-]?\d+)\*\*(?=\.\d)")
_ANSWER_NUMBER = re.compile(
    rf"(?<![\w.])(?:"
    r"(?P<sign>[+-]?)\s*\\frac\{(?P<num>[+-]?\d+)\}\{(?P<den>[+-]?\d+)\}"
    rf"|(?P<mantissa>{_DECIMAL})\s*(?:\\times|×|\*)\s*10\s*\^\s*(?:\{{(?P<braced>[+-]?\d+)\}}|(?P<power>[+-]?\d+))"
    rf"|(?P<value>{_DECIMAL})(?:[eE](?P<exponent>[+-]?\d+)|\s*/\s*(?P<divisor>[+-]?\d+))?"
    r")"
)


def numeric_answer(text):
    tail = re.split(r'(?<!#)#{4}(?!#)', text)[-1].replace('−', '-').replace(',', '')
    # Markdown can split a single decimal across an emphasis boundary.
    tail = _BOLD_SPLIT_DECIMAL.sub(r'\1', tail)
    answer = None
    for match in _ANSWER_NUMBER.finditer(tail):
        try:
            if match['num'] is not None:
                value = Fraction(int(match['num']), int(match['den']))
                answer = -value if match['sign'] == '-' else value
            else:
                value = Fraction(match['mantissa'] or match['value'])
                exponent = int(match['braced'] or match['power'] or match['exponent'] or 0)
                divisor = int(match['divisor'] or 1)
                answer = value * Fraction(10) ** exponent / divisor if abs(exponent) <= 100 else None
        except (ValueError, ZeroDivisionError):
            answer = None
    return answer


def expression_value(text):
    """Independent, bounded AST oracle for extraction. No eval or calculator-parser reuse."""
    text = text.strip().strip('`').replace('^', '**').replace('×', '*').replace('÷', '/')
    if len(text) > 200:
        return None
    def visit(n):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return Fraction(ast.get_source_segment(text, n))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -visit(n.operand)
        if isinstance(n, ast.BinOp):
            a, b = visit(n.left), visit(n.right)
            if isinstance(n.op, ast.Add): return a+b
            if isinstance(n.op, ast.Sub): return a-b
            if isinstance(n.op, ast.Mult): return a*b
            if isinstance(n.op, ast.Div): return a/b
            if isinstance(n.op, ast.Pow) and b.denominator == 1 and abs(b) <= 6: return a**int(b)
        raise ValueError('unsupported expression')
    try:
        return visit(ast.parse(text, mode='eval').body)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def problems(gsm_limit):
    rng=random.Random(20261117); rows=[]
    def add(suite, category, question, value, expression=None):
        rows.append(dict(id=f'{suite}-{sum(r["suite"]==suite for r in rows):04d}', suite=suite,
                         category=category, question=question, expected=str(value), expression=expression))
    for cat,n in [('addition',10),('subtraction',10),('multiplication',20),('division',20),('multistep',20),('exponents',20)]:
        for i in range(n):
            a,b,c=rng.randint(21,999),rng.randint(11,99),rng.randint(2,9)
            e={'addition':f'{a} + {b}','subtraction':f'{a+b} - {b}', 'multiplication':f'{a} * {b}',
               'division':f'{a*b} / {b}','multistep':f'({a} + {b}) * {c}', 'exponents':f'{c}^3 + {b}'}[cat]
            add('math',cat,f'Calculate {e}.',expression_value(e),e)
    for i in range(100):
        a,b=rng.randrange(10**11,10**12),rng.randrange(10**11,10**12)
        if i<25: e=f'{a} + {b}';cat='12digit_addition'
        elif i<50: e=f'{a} * {b}';cat='12digit_multiplication'
        elif i<75:
            e=f'{a//100000}.{a%100000:05d} * {b//10000}.{b%10000:04d}';cat='decimal_multiplication'
        else:
            e=f'{a}.{b%10000:04d} / {rng.choice([2,4,5,8,16,25])}';cat='decimal_division'
        add('ood',cat,f'Calculate {e}.',expression_value(e),e)
    for i in range(100):
        a,b,c=rng.randint(20,99),rng.randint(2,12),rng.randint(2,9);distr=rng.randint(1000,9000)
        if i%4==0:
            q=f'Invoice {distr}, dated March 17: a club buys {b} packs of {a} pencils. A nearby shop has 37 pens. How many pencils did the club buy?';e=f'{a} * {b}'
        elif i%4==1:
            q=f'Room {distr} holds {a+b} books. After lending {b} books, the librarian puts up 14 posters. How many books remain in the room?';e=f'{a+b} - {b}'
        elif i%4==2:
            q=f'Bus route {distr} passes 11 stops. A school shares {a*b} stickers equally among {b} children. Each child also owns 6 crayons. How many stickers does each child receive?';e=f'{a*b} / {b}'
        else:
            q=f'Order {distr}: {c} shelves each hold {a} red and {b} blue cups. The shop opens at 9 am and has 24 chairs. How many cups are on the shelves?';e=f'({a} + {b}) * {c}'
        add('noisy_words',['multiply','subtract','divide','multistep'][i%4],q,expression_value(e),e)
    for i,line in enumerate((ROOT/'benchmarks/gsm8k/test.jsonl').read_text().splitlines()[:gsm_limit]):
        r=json.loads(line);add('gsm8k','official_test',r['question'],Fraction(r['answer'].split('####')[-1].strip().replace(',','')))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--phase',choices=['smoke','correctness','perf'],default='smoke');ap.add_argument('--gsm-limit',type=int,default=100)
    ap.add_argument('--corpus',type=Path,help='Frozen JSONL corpus with independent expected answers')
    ap.add_argument('--batch-size',type=int,default=1)
    ap.add_argument('--synchronous-callbacks',action='store_true',help='Drain model forward before calculator callbacks; diagnostic scheduling mode')
    ap.add_argument('--complete-literal-requests',action='store_true',help='Opt-in whole-expression/count request gate with exact terminal answer')
    ap.add_argument('--defenses',action='store_true',help='Enable run-on/spiral detection and bounded final-answer continuation')
    ap.add_argument('--arms',choices=['both','calculator'],default='both',help='Scored correctness arms')
    ap.add_argument('--suite',choices=['all','gsm8k','math','ood','noisy_words'],default='all')
    ap.add_argument('--calculator-backend',choices=['cpu','gpu','gemm'],default='gpu')
    ap.add_argument('--prefilter',choices=['off','gpu'],default='gpu')
    ap.add_argument('--perf-stop-at-eos',action='store_true',help='Measure natural completions under the same 256-token cap')
    ap.add_argument('--case-ids-file',type=Path,help='JSON array of case IDs for a bounded repair probe')
    ap.add_argument('--max-tokens',type=int,default=384);args=ap.parse_args()
    if args.batch_size<1 or (args.batch_size>1 and args.phase!='correctness'):
        raise ValueError('batches require correctness mode')
    if args.defenses and args.max_tokens < 4096:
        raise ValueError('completion defenses require at least 4096 tokens for their default thresholds')
    corpus_data=args.corpus.read_bytes() if args.corpus else None
    frozen_corpus=None
    if args.corpus:
        frozen_corpus=json.loads((args.corpus.parent/'freeze.json').read_text())
        if hashlib.sha256(corpus_data).hexdigest()!=frozen_corpus['dataset_sha256']:
            raise ValueError('frozen corpus SHA-256 mismatch')
    args.out.mkdir(parents=True,exist_ok=False)
    # This offline MLX-LM harness does not inherit the serving thread's stream
    # binding. Fast cross-stream fences can GPU-spin forever after compaction.
    # Pin normal Metal events before mlx2 or any device operation is imported.
    os.environ['MLX_METAL_FAST_SYNCH']='0'
    import mlx.core as mx
    import shutil
    from mlx_lm import load
    from mlx_lm.generate import generate_step, generation_stream
    from mlx2.runtime import amoeba_det_tiles as D
    from mlx2.runtime import amoeba_math_latex, amoeba_calc_gpu, amoeba_calc_trigger
    from mlx2.runtime import amoeba_counting, amoeba_calc_requests, amoeba_calc_resident
    from mlx2.apple_telemetry import TemperatureSampler
    from amoeba_calc_batch import answer_in_progress, calculator_processor
    mx.set_default_device(mx.gpu)
    with mx.stream(generation_stream):
        decode_stream=mx.default_stream(mx.gpu)
    mx.set_default_stream(decode_stream)
    snap=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    # Each run owns an immutable configured artifact; never mutate the input manifest.
    manifest=D.load_manifest(args.manifest);configured=args.out/'tile';configured.mkdir()
    for name in manifest['files']: shutil.copy2(args.manifest.parent/name,configured/name)
    manifest={k:v for k,v in manifest.items() if not k.startswith('_')}
    manifest['probe_source_revision']=manifest.get('mlx2_source_revision')
    manifest['mlx2_source_revision']=rev
    for tile in manifest['tiles']:
        if tile['kind']=='calculator':
            if tile.get('parser') == D.PARSER_V5:
                if args.calculator_backend!='gpu' or args.prefilter!='gpu':
                    raise ValueError('the resident calculator requires the GPU backend and cue filter')
            else:
                tile['arithmetic_backend']=args.calculator_backend;tile['trigger_prefilter']=args.prefilter
                tile['parser']=D.PARSER_V4;tile['context_tokens']=256
            tile['complete_literal_requests']=args.complete_literal_requests
    (configured/'manifest.json').write_text(json.dumps(manifest,indent=2))
    args.manifest=configured/'manifest.json'
    model,tok=load(str(snap));layers=model.model.layers
    original=[l.mlp for l in layers]
    ts=D.install(layers,args.manifest,{'mlp':'mlp','activation_fn':lambda x:x*mx.sigmoid(x)},source_fingerprint='Qwen/Qwen3-0.6B@c1899de2')
    wrapped=[l.mlp for l in layers];temp=TemperatureSampler()
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    meta=dict(checkpoint='Qwen/Qwen3-0.6B@c1899de2',checkpoint_revision=snap.name,source_revision=rev,
              runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),device=mx.device_info(),dtype=str(model.model.embed_tokens.weight.dtype),
              manifest=D.load_manifest(args.manifest),phase=args.phase,qualification='unqualified',gsm_limit=args.gsm_limit,
              case_ids=json.loads(args.case_ids_file.read_text()) if args.case_ids_file else None,
              perf_stop_at_eos=args.perf_stop_at_eos,eval_arms=args.arms,suite=args.suite,
              completion_defenses=args.defenses,batch_size=args.batch_size,max_tokens=args.max_tokens,
              gsm8k_sha256=hashlib.sha256((ROOT/'benchmarks/gsm8k/test.jsonl').read_bytes()).hexdigest(),
              synchronous_callbacks=args.synchronous_callbacks,
              metal_fast_synch=os.environ['MLX_METAL_FAST_SYNCH'],generation_stream_bound=True,
              complete_literal_requests=args.complete_literal_requests,
              calculator_source_files={Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                       for m in (D,amoeba_math_latex,amoeba_calc_gpu,amoeba_calc_trigger,
                                                 amoeba_counting,amoeba_calc_requests,amoeba_calc_resident)},
              resident_header_sha256=hashlib.sha256(amoeba_calc_resident.header().encode()).hexdigest(),
              corpus_sha256=hashlib.sha256(corpus_data).hexdigest() if corpus_data else None,frozen_corpus=frozen_corpus)
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2,default=str))
    def gen(question,arm,max_tokens=args.max_tokens,ignore_eos=False,extraction=False):
        for l,m in zip(layers,wrapped if arm=='calculator' else original): l.mlp=m
        instruction=('Extract the arithmetic expression needed to answer the question. Output ONLY an expression using numbers, +, -, *, /, ^ and parentheses. No prose or answer.' if extraction else 'Solve this problem. End with #### followed by the final numeric answer.')
        prompt=tok.apply_chat_template([{'role':'user','content':instruction+'\n'+question}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tok.encode(prompt,add_special_tokens=False)
        use_defenses=args.defenses and not extraction and max_tokens>=4096
        defense=None;nudge=[];reserve=0
        if use_defenses:
            from amoeba_calc_defenses import CompletionDefenses, NUDGE, phase_budgets
            defense=CompletionDefenses();nudge=tok.encode(NUDGE,add_special_tokens=False)
            initial_limit,reserve=phase_budgets(max_tokens,len(nudge))
        else:initial_limit=max_tokens
        start=time.perf_counter();first=None;new=[];events=[];generated=0
        def stage(prompt_ids,limit,watch=None):
            nonlocal first,generated
            request_text=question if args.complete_literal_requests and not extraction else None
            proc=calculator_processor(ts,tok,len(prompt_ids),request_text) if arm=='calculator' else None
            prefix=mx.array(prompt_ids[:-1]);stage_ids=[]
            def process(tokens,logits):
                proc.bind_row(0)
                # MLX-LM's processor history starts at the final prefill token.
                return proc(mx.concatenate([prefix,tokens]),logits)
            stream=generate_step(mx.array(prompt_ids),model,max_tokens=limit,logits_processors=[process] if proc else None)
            try:
                for token,_ in stream:
                    t=int(token.item()) if hasattr(token,'item') else int(token)
                    if first is None:first=time.perf_counter()
                    stage_ids.append(t);generated+=1
                    if not ignore_eos and t in tok.eos_token_ids:break
                    if watch:
                        text=tok.decode(stage_ids) if len(stage_ids)%16==0 else None
                        if watch.feed(t,text) and not answer_in_progress(proc):break
            finally:stream.close()
            if proc:events.extend(getattr(proc,'events',()))
            return stage_ids
        new=stage(ids,initial_limit,defense)
        natural=bool(new and new[-1] in tok.eos_token_ids)
        continued=False;trigger=None;injected=0
        if defense and not natural:
            trigger=defense.reason or 'budget_soft'
            available=max_tokens-len(new)-len(nudge)
            tail=stage(ids+new+nudge,min(reserve,available))
            new=new+nudge+tail;injected=len(nudge);continued=True
        end=time.perf_counter();s=tok.decode(new)
        guard_receipt=(dict(**defense.receipt(),continuation=continued,trigger=trigger,
                            initial_limit=initial_limit,continuation_limit=reserve,
                            injected_tokens=injected,generated_tokens=generated,total_output_cap=max_tokens)
                       if defense else dict(enabled=False))
        return dict(output=s,token_ids=new,tokens=len(new),wall_s=end-start,decode_s=end-first if first else 0,
                    decode_tps=(generated-1)/(end-first) if first and generated>1 else None,
                    truncated=bool(new and new[-1] not in tok.eos_token_ids and not ignore_eos),events=events,
                    defenses=guard_receipt,receipt=ts.receipt() if arm=='calculator' else None)
    def gen_many(questions,max_tokens=args.max_tokens,arm="calculator"):
        from amoeba_calc_batch import batch_stage
        from amoeba_calc_defenses import NUDGE, phase_budgets
        for l,m in zip(layers,wrapped if arm=="calculator" else original):l.mlp=m
        instruction='Solve this problem. End with #### followed by the final numeric answer.'
        prompts=[tok.encode(tok.apply_chat_template([{'role':'user','content':instruction+'\n'+q}],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False) for q in questions]
        nudge=tok.encode(NUDGE,add_special_tokens=False)
        defended=args.defenses and max_tokens>=4096
        initial,reserve=phase_budgets(max_tokens,len(nudge)) if defended else (max_tokens,0)
        start=time.perf_counter()
        states=batch_stage(model,tok,ts,prompts,[initial]*len(prompts),watch=defended,calculator=arm=="calculator",synchronous_callbacks=args.synchronous_callbacks,
                           request_texts=questions if args.complete_literal_requests else None)
        pending=[i for i,s in enumerate(states) if s['token_ids'][-1] not in tok.eos_token_ids] if defended else []
        if pending:
            tails=batch_stage(model,tok,ts,[prompts[i]+states[i]['token_ids']+nudge for i in pending],
                              [min(reserve,max_tokens-len(states[i]['token_ids'])-len(nudge)) for i in pending],calculator=arm=="calculator",synchronous_callbacks=args.synchronous_callbacks)
            for i,tail in zip(pending,tails):
                state=states[i];state['token_ids']+=nudge+tail['token_ids'];state['events']+=tail['events']
                state['finish_reason']=tail['finish_reason']
                state['injected_tokens']=len(nudge)
        wall=time.perf_counter()-start
        results=[]
        receipt=ts.receipt() if arm=='calculator' else None
        for state in states:
            ids=state['token_ids'];guard=state['defense']
            if len(ids)>max_tokens:raise RuntimeError('batched output exceeded token cap')
            results.append(dict(output=tok.decode(ids),token_ids=ids,tokens=len(ids),wall_s=wall,
                                timing_scope='whole_batch',truncated=ids[-1] not in tok.eos_token_ids,
                                events=state['events'],receipt=receipt,batch_widths=state['widths'],
                                physical_rows=state['rows'],
                                defenses=(dict(**guard.receipt(),continuation=state.get('injected_tokens',0)>0,
                                               trigger=guard.reason or ('budget_soft' if state.get('injected_tokens') else None),
                                               injected_tokens=state.get('injected_tokens',0),total_output_cap=max_tokens)
                                          if guard else dict(enabled=False))))
        return results
    def write(name,row):
        with (args.out/name).open('a') as f:f.write(json.dumps(row,default=str)+'\n')
    # Ordinary reference and an all-closed wrapper probe before any scored rows.
    q='Describe the scent of a pine forest.'
    ref=gen(q,'stock',32);closed=gen(q,'calculator',32)
    write('reference.jsonl',dict(stock=ref,calculator=closed,exact=ref['token_ids']==closed['token_ids']))
    if ref['token_ids']!=closed['token_ids'] or closed['receipt']['counters']['calc']['plans']:
        raise RuntimeError('closed-tile ordinary reference failed')
    if args.phase=='smoke':
        for q in ['Calculate 109 * 71.','Calculate 123456789123 * 987654321987.','Calculate 3^3 + 17.','Calculate 3^3 + 78.','Order 1343: 7 shelves each hold 63 red and 12 blue cups. The shop opens at 9 am and has 24 chairs. How many cups are on the shelves?' ]:
            r={a:gen(q,a) for a in ['stock','calculator']};write('smoke.jsonl',dict(question=q,arms=r));print(q,{a:v['output'] for a,v in r.items()},flush=True)
    elif args.phase=='correctness':
        rows=([json.loads(line) for line in corpus_data.decode().splitlines() if line.strip()] if args.corpus else problems(args.gsm_limit))
        if frozen_corpus and len(rows)!=frozen_corpus['n']:raise ValueError('frozen corpus count mismatch')
        if len({r['id'] for r in rows})!=len(rows):raise ValueError('corpus IDs must be unique')
        for r in rows:Fraction(r['expected'])
        if args.suite!='all':
            rows=[r for r in rows if r['suite']==args.suite]
        if args.case_ids_file:
            wanted=json.loads(args.case_ids_file.read_text())
            if not isinstance(wanted,list) or not wanted or len(set(wanted))!=len(wanted) or not set(wanted)<={r['id'] for r in rows}:
                raise ValueError('case IDs must be a nonempty unique subset of this problem set')
            rows=[r for r in rows if r['id'] in set(wanted)]
        (args.out/'problems.json').write_text(json.dumps(rows,indent=2))
        if args.batch_size>1:
            # A small sequential/batched functional control before any scored retry.
            control=[r for r in problems(0) if r['id'] in ['math-0000','math-0010','math-0020','math-0033','math-0040','math-0060','math-0080','math-0098']]
            sequential=[gen(r['question'],'calculator',128) for r in control]
            batched=[]
            for i in range(0,len(control),args.batch_size):batched+=gen_many([r['question'] for r in control[i:i+args.batch_size]],128)
            for row,seq,bat in zip(control,sequential,batched):
                ok=all(not x['truncated'] and numeric_answer(x['output'])==Fraction(row['expected']) for x in [seq,bat])
                write('batch_reference.jsonl',dict(**row,sequential=seq,batched=bat,correct=ok,token_exact=seq['token_ids']==bat['token_ids']))
                if not ok:raise RuntimeError('batched calculator functional control failed')
            for i in range(0,len(rows),args.batch_size):
                chunk=rows[i:i+args.batch_size]
                order=['calculator'] if args.arms=='calculator' else (['stock','calculator'] if (i//args.batch_size)%2==0 else ['calculator','stock'])
                outputs={}
                for arm in order:
                    progress=dict(batch_index=i//args.batch_size,case_ids=[r['id'] for r in chunk],arm=arm,status='started')
                    (args.out/'progress.tmp').write_text(json.dumps(progress))
                    (args.out/'progress.tmp').replace(args.out/'progress.json')
                    outputs[arm]=gen_many([r['question'] for r in chunk],arm=arm)
                    for row,result in zip(chunk,outputs[arm]):
                        write('arm_rows.jsonl',dict(id=row['id'],batch_index=i//args.batch_size,arm=arm,result=result))
                for j,row in enumerate(chunk):
                    arms={arm:outputs[arm][j] for arm in order}
                    for result in arms.values():result['correct']=not result['truncated'] and numeric_answer(result['output'])==Fraction(row['expected'])
                    write('rows.jsonl',dict(**row,arms=arms,batch_index=i//args.batch_size,arm_order=order))
                print('batch',i+len(chunk),'of',len(rows),flush=True)
        else:
            for i,row in enumerate(rows):
                arms={}
                selected=['calculator'] if args.arms=='calculator' else (['stock','calculator'] if i%2==0 else ['calculator','stock'])
                for a in selected:
                    arms[a]=gen(row['question'],a);arms[a]['correct']=not arms[a]['truncated'] and numeric_answer(arms[a]['output'])==Fraction(row['expected'])
                write('rows.jsonl',{**row,'arms':arms});print(i+1,row['suite'],{a:v['correct'] for a,v in arms.items()},flush=True)
        for row in [r for r in rows if r['suite']=='noisy_words']:
            r=gen(row['question'],'stock',128,extraction=True);r['semantic_correct']=expression_value(r['output'].replace('<|im_end|>','').strip())==Fraction(row['expected'])
            write('extraction.jsonl',{**row,**r})
        for row in [r for r in rows if r['suite']=='ood']:
            ans,info=D.calculator_answer('Calculate '+row['expression']+' =',parser=D.PARSER_V3,arithmetic_backend=args.calculator_backend)
            write('parser_oracle.jsonl',{**row,'answer':ans,'info':info,'correct':numeric_answer(ans or '')==Fraction(row['expected'])})
    else:
        prompts=['Write a vivid description of a pine forest after rain.','Tell a gentle story about a lost kitten finding its way home.','Explain how a gardener can care for roses.','Describe the sights and sounds of a quiet seaside village.']
        # Warm both routes before paired alternation. The default fixes output count;
        # optional natural EOS measures ordinary completions under a common cap.
        for a in ['stock','calculator']:gen(prompts[0],a,64,ignore_eos=not args.perf_stop_at_eos)
        for rep in range(6):
            for pi,q in enumerate(prompts):
                for a in (['stock','calculator'] if (rep+pi)%2==0 else ['calculator','stock']):
                    # Settling is restricted to this explicitly requested performance phase.
                    start=time.time()
                    while (temp.die_summary()['die_max_c'] or 0)>60:
                        if time.time()-start>300:raise RuntimeError('performance thermal admission timed out')
                        time.sleep(5)
                    before=temp.die_summary();r=gen(q,a,256,ignore_eos=not args.perf_stop_at_eos)
                    write('perf.jsonl',dict(rep=rep,prompt=pi,arm=a,before=before,after=temp.die_summary(),**r))
                    print('perf',rep,pi,a,r['decode_tps'],flush=True)
    (args.out/'complete.json').write_text(json.dumps(dict(source_revision=rev,receipt=ts.receipt(),finished=time.time()),indent=2))

if __name__=='__main__':main()
