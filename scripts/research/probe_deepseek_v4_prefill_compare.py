#!/usr/bin/env python3
"""Plan/run bounded same-prompt ordinary versus dense+expert NAX diagnostics.

Each arm is a fresh ordinary-probe process with immutable policy and cache
identity. Different FP16 operand arithmetic remains explicit. No DSpark,
qualification, all-token logit dump or controlled performance claim.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from probe_deepseek_v4_packed import atomic,source_receipt,verify_ownership
from probe_deepseek_v4_ordinary import load_request,load_reference,first_logit_comparison


def policies(base,candidate_prefill="both"):
    from mlx2.adapters.deepseek_v4_gguf import validate_execution_policy
    if candidate_prefill not in ('both','dense','experts'):raise ValueError('unknown candidate prefill selection')
    if not isinstance(base,dict) or set(base)&{'draft_model','draft_models','num_draft','deepseek_verifier','deepseek_draft_confidence'}:
        raise ValueError('prefill comparison excludes all draft/sidecar policies')
    reference=validate_execution_policy(base)
    if reference['deepseek_dense_prefill']!='reference' or reference['deepseek_expert_prefill']!='reference':
        raise ValueError('base policy must keep both prefill candidates reference')
    candidate=dict(reference)
    if candidate_prefill in ('both','dense'):candidate['deepseek_dense_prefill']='nax_fp16'
    if candidate_prefill in ('both','experts'):candidate['deepseek_expert_prefill']='segmented_nax_fp16'
    return reference,candidate


def prepare_request(artifact,path):
    from mlx2.adapters.deepseek_v4_gguf_artifact import REFERENCE_FILE
    from mlx2.adapters.deepseek_v4_gguf_tokenizer import load_tokenizer
    root=Path(artifact['identity']['path'])
    tokenizer,_=load_tokenizer(json.loads((root/REFERENCE_FILE).read_text()),artifact['metadata'])
    def count(request):
        rendered=tokenizer.apply_chat_template(request['messages'],add_generation_prompt=True,tokenize=False,enable_thinking=False)
        return len(tokenizer.encode(rendered,add_special_tokens=False))
    if path is not None:
        request,binding=load_request(path);return request,count(request),binding
    for lines in range(16,65):
        request={'messages':[{'role':'user','content':'Reference notes:\n'+''.join(f'Entry {i}: exact cache state belongs to one immutable prefix; one request uses one logical decode lane.\n' for i in range(lines))+'\nFinal instruction: Reply with exactly Hello.'}],'enable_thinking':False}
        tokens=count(request)
        if 641<=tokens<=2048:return request,tokens,None
    raise ValueError('could not form bounded641..2048-token default request')


def arm_command(model,output,request,policy,step,args,*,position=0,reference=None):
    cmd=[sys.executable,str(ROOT/'scripts/research/probe_deepseek_v4_ordinary.py'),
        '--model-path',str(model),'--output',str(output),'--run-gpu','--ownership-window',str(args.ownership_window),
        '--request-json',str(request),'--expected-text',args.expected_text,'--functional-law',args.functional_law,
        '--max-tokens',str(args.max_tokens),'--prefill-step',str(step),'--execution-policy',json.dumps(policy,sort_keys=True),
        '--memory-limit-gib',str(args.memory_limit_gib),'--capture-output-position',str(position)]
    if reference is not None:cmd+=['--reference-logits',str(reference)]
    for path in args.owner_path or []:cmd+=['--owner-path',str(path)]
    return cmd


def usage_check(row,candidate,candidate_prefill="both"):
    adapter=(row.get('route_after_evaluated_generation') or {}).get('adapter') or {}
    counts={key:int((adapter.get(group) or {}).get(key,0)) for group,key in [
        ('scheduled_dense_calls','nax_prefill'),('scheduled_grouped_calls','nax_grouped_prefill'),
        ('scheduled_expert_calls','segmented_moe_prefill')]}
    if candidate_prefill not in ('both','dense','experts'):raise ValueError('unknown candidate prefill selection')
    expected={key:bool(candidate and ((key=='segmented_moe_prefill' and candidate_prefill in ('both','experts')) or
        (key!='segmented_moe_prefill' and candidate_prefill in ('both','dense')))) for key in counts}
    return {'passed':all((n>0) if expected[key] else n==0 for key,n in counts.items()),
            'expected_scheduled_nonzero':expected,'scheduled_counts_after_forced_evaluation':counts,
            'scope':'Python scheduled path counters after evaluated logits/cache; no hardware trace or qualification claim'}


def first_difference(a,b):
    if a==b:return None
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))


def pair_summary(reference,candidate,ref_root,cand_root,vocab,candidate_prefill="both"):
    a=reference['generation']['token_ids'];b=candidate['generation']['token_ids']
    same_prompt=reference['prompt']['rendered_utf8_sha256']==candidate['prompt']['rendered_utf8_sha256'] and reference['prompt']['tokens']==candidate['prompt']['tokens']
    av,_=load_reference(ref_root/reference['captured_logits']['path'],vocab)
    bv,_=load_reference(cand_root/candidate['captured_logits']['path'],vocab)
    return {'same_prompt':same_prompt,'reference_usage':usage_check(reference,False,candidate_prefill),'candidate_usage':usage_check(candidate,True,candidate_prefill),
        'selected_tokens_exact':a==b,'first_selected_divergence':first_difference(a,b),
        'visible_text_exact':reference['generation']['text']==candidate['generation']['text'],
        'first_position_logits_bitwise_exact':bool(np.array_equal(av,bv)),
        'first_position_vector_comparison':first_logit_comparison(bv,av),
        'candidate_prefill':candidate_prefill,
        'arithmetic':f'{candidate_prefill} candidate uses scoped FP16 NAX operands; other prefill policies remain reference; ordinary is a separate numerical profile',
        'equivalence_verdict':'not_assessed; selected token or tolerance agreement alone does not qualify this route'}


def completed_arm(output,returncode):
    """A completed functional failure is evidence; a crash is not a comparable arm."""
    path=Path(output)/'results.json'
    try:row=json.loads(path.read_text())
    except (OSError,ValueError) as error:
        raise RuntimeError(f'arm exited{returncode} without readable completed receipt: {path}') from error
    if (row.get('status')!='completed' or type(row.get('passed')) is not bool
            or not isinstance(row.get('generation',{}).get('token_ids'),list)
            or not row['generation']['token_ids'] or 'captured_logits' not in row):
        raise RuntimeError(f'arm exited{returncode} with incomplete/crashed receipt: {path}')
    return {**row,'process_exit_code':returncode}


def replay_disposition(position,*,arms_passed,skip,replay_failed):
    if skip:return 'skipped_by_operator'
    if not arms_passed and not replay_failed:return 'pending_explicit_opt_in_functional_failure'
    return 'pending'


def execute(cmd,log,guard,deadline):
    guard();process=None
    with log.open('wb') as stream:
        try:
            process=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')},stdout=stream,stderr=subprocess.STDOUT)
            while process.poll() is None:
                guard()
                if time.monotonic()>deadline:raise TimeoutError('bounded prefill comparison deadline exceeded')
                time.sleep(.5)
            return completed_arm(Path(cmd[cmd.index('--output')+1]),process.returncode)
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--base-policy',type=json.loads,default={})
    p.add_argument('--candidate-prefill',choices=('both','dense','experts'),default='both')
    p.add_argument('--request-json',type=Path);p.add_argument('--expected-text',default='Hello')
    p.add_argument('--functional-law',choices=('exact','nonempty_finite'),default='exact')
    p.add_argument('--steps',type=int,nargs='+',default=[128,512]);p.add_argument('--max-tokens',type=int,default=16)
    p.add_argument('--memory-limit-gib',type=float,default=108);p.add_argument('--wall-seconds',type=int,default=1200)
    p.add_argument('--run-gpu',action='store_true');p.add_argument('--ownership-window',type=Path)
    p.add_argument('--owner-path',type=Path,action='append');p.add_argument('--skip-divergence-replay',action='store_true')
    p.add_argument('--replay-failed-functional',action='store_true',help='explicitly allow extra GPU replay after a completed arm fails its oracle')
    args=p.parse_args(argv)
    if args.output.exists():p.error('refusing existing evidence directory')
    if (not args.steps or len(set(args.steps))!=len(args.steps) or set(args.steps)-{128,512}
            or not 1<=args.max_tokens<=128 or not 80<=args.memory_limit_gib<=112
            or not 120<=args.wall_seconds<=1800):p.error('bounded unique128/512steps, output1..128, memory80..112GiB and wall120..1800s required')
    if args.run_gpu and args.ownership_window is None:p.error('execution requires existing owned GPU window')
    try:reference_policy,candidate_policy=policies(args.base_policy,args.candidate_prefill)
    except ValueError as error:p.error(str(error))
    owners=args.owner_path or [Path('/Users/Shared/mlxuag/gpu.lock/owner.json'),Path('/tmp/gpu.lock/owner.json')]
    if len(owners)!=2:p.error('exactly two paired owner receipts required')
    def guard():return verify_ownership(args.ownership_window,owners) if args.run_gpu else None
    ownership=guard();source=source_receipt()
    from mlx2.adapters.deepseek_v4_gguf_artifact import inspect_artifact
    artifact=inspect_artifact(args.model_path);request,tokens,binding=prepare_request(artifact,args.request_json)
    if not max(args.steps)+1<=tokens<=2048:p.error('request must actually exercise every full prefill chunk and fit2048tokens')
    args.output.mkdir(parents=True);request_path=args.output/'request.json';atomic(request_path,request)
    receipt={'schema':'mlx2.deepseek-v4-prefill-comparison.v1','status':'planned' if not args.run_gpu else 'running',
        'checkpoint':args.model_path.name+'@'+artifact['identity']['fingerprint'],'source':source,'ownership':ownership,
        'qualification':False,'performance_claim':False,'run_gpu':args.run_gpu,'request_tokens':tokens,
        'request_sha256':hashlib.sha256(request_path.read_bytes()).hexdigest(),'input_request_binding':binding,
        'physical_batch_width':1,'sidecar_included':False,'candidate_prefill':args.candidate_prefill,
        'math_policies':{'reference':{k:reference_policy[k] for k in ('deepseek_dense_prefill','deepseek_expert_prefill')},
                         'candidate':{k:candidate_policy[k] for k in ('deepseek_dense_prefill','deepseek_expert_prefill')}},
        'reference_policy':reference_policy,'candidate_policy':candidate_policy,
        'timing_scope':'observational separate model-load arms; compile/thermal/APC warmth not controlled','steps':[]}
    for step in args.steps:
        ref=args.output/f'reference-step{step}';cand=args.output/f'nax-step{step}'
        receipt['steps'].append({'step':step,'reference_command':arm_command(args.model_path,ref,request_path,reference_policy,step,args),
            'candidate_command':arm_command(args.model_path,cand,request_path,candidate_policy,step,args,reference=ref/'first-position-logits.npy')})
    atomic(args.output/'results.json',receipt)
    if not args.run_gpu:return 0
    deadline=time.monotonic()+args.wall_seconds
    try:
        for cell in receipt['steps']:
            step=cell['step'];ref=args.output/f'reference-step{step}';cand=args.output/f'nax-step{step}'
            a=execute(cell['reference_command'],args.output/f'reference-step{step}.log',guard,deadline)
            b=execute(cell['candidate_command'],args.output/f'nax-step{step}.log',guard,deadline)
            cell['arm_exit_codes']={'reference':a['process_exit_code'],'candidate':b['process_exit_code']}
            cell['arm_functional_checks']={'reference':a['passed'],'candidate':b['passed']}
            cell['comparison']=pair_summary(a,b,ref,cand,int(artifact['metadata']['deepseek4.vocab_size']),args.candidate_prefill)
            atomic(args.output/'results.json',receipt)
            position=cell['comparison']['first_selected_divergence']
            if position is not None and position>0:
                disposition=replay_disposition(position,arms_passed=a['passed'] and b['passed'],
                    skip=args.skip_divergence_replay,replay_failed=args.replay_failed_functional)
                cell['divergence_replay']={'output_position':position,'status':disposition,
                    'captured':False,'scope':'first selected divergence known from completed tokens; its logits are not captured yet'}
                if disposition=='pending':
                    replay=[]
                    for name,policy in [('reference',reference_policy),('nax',candidate_policy)]:
                        out=args.output/f'{name}-step{step}-divergence{position}'
                        rp=None if not replay else replay[0][0]/f'output-position-{position}-logits.npy'
                        cmd=arm_command(args.model_path,out,request_path,policy,step,args,position=position,reference=rp)
                        cell['divergence_replay'].setdefault('commands',[]).append(cmd);atomic(args.output/'results.json',receipt)
                        row=execute(cmd,args.output/f'{name}-step{step}-divergence.log',guard,deadline)
                        replay.append((out,row))
                    original_prefix=a['generation']['token_ids'][:position]
                    reproduced=all(row['generation']['token_ids'][:position]==original_prefix for _,row in replay)
                    cell['divergence_replay'].update(status='completed',captured=True,shared_original_prefix_reproduced=reproduced,
                        comparison=replay[1][1]['reference_logits']['comparison'],
                        scope='only original selected divergence row captured; no downstream equivalence inference')
            cell['required_checks_passed']=a['passed'] and b['passed'] and cell['comparison']['same_prompt'] and cell['comparison']['reference_usage']['passed'] and cell['comparison']['candidate_usage']['passed']
            atomic(args.output/'results.json',receipt)
        receipt['all_required_checks_passed']=all(c['required_checks_passed'] for c in receipt['steps'])
        receipt['status']='completed' if receipt['all_required_checks_passed'] else 'failed'
        if not receipt['all_required_checks_passed']:receipt['failure_kind']='completed_arm_checks_failed'
        guard()
    except BaseException as error:
        receipt.update(status='failed',all_required_checks_passed=False,error={'type':type(error).__name__,'message':str(error)});raise
    finally:atomic(args.output/'results.json',receipt)
    return 0 if receipt['all_required_checks_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
