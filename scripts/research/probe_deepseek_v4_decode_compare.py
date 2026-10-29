#!/usr/bin/env python3
"""Controlled B1 ordinary/sequential/causal-chain-slab comparison candidate.

One target+sidecar allocation, fresh caches per arm, no APC reuse and no tree
AST claim. Caller owns the bounded GPU lease. This tool never auto-publishes a
performance claim, even when its comparison gates pass.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from probe_deepseek_v4_packed import atomic,source_receipt,verify_ownership
from probe_deepseek_v4_dspark import load_request
from deepseek_v4_probe_residency import wire_preloaded_weights,restore_probe_wired_limit

ARMS=('ordinary','sequential','causal_slab')
# Each arm occupies each ordinal position once in rotating measured orders.
ORDERS=(('ordinary','sequential','causal_slab'),
        ('causal_slab','ordinary','sequential'),
        ('sequential','causal_slab','ordinary'))
THERMAL_POLICY={'required_thermal_state':0,'require_temperatures':True,
                'max_battery_temperature_c':40.,'max_virtual_temperature_c':45.}
LEGACY_PROFILE='battery_virtual_nominal_v1'
HID_PROFILE='m5_hid_nominal_v1'
HID_THERMAL_POLICY={'required_thermal_state':0,'max_die_temperature_c':60.,
    'max_hid_battery_temperature_c':40.,'max_temperature_delta_c':.5,
    'max_sample_age_seconds':10.,'required_samples':3,
    'source':'mlx2.apple_telemetry.TemperatureSampler.read',
    'references':['scripts/benchmark_ast_repair.py: --start-max-c default60',
                  'scripts/run_qualification_matrix.py:3sample stability0.5C',
                  'qualification/four-model-experiments.json:battery ceiling40C']}


def thermal_failures(sample,profile=LEGACY_PROFILE):
    """Explain the existing project policy without changing its thresholds."""
    missing=[];warnings=[];above=[]
    for key in ('thermal_state','pmset_no_thermal_warning','pmset_no_performance_warning','pmset_no_cpu_power_warning'):
        if key not in sample:missing.append(key)
    for key in ('pmset_no_thermal_warning','pmset_no_performance_warning','pmset_no_cpu_power_warning'):
        if key in sample and sample[key] is not True:warnings.append(key)
    sensors=(('die_max_c',60.),('hid_battery_temperature_c',40.)) if profile==HID_PROFILE else (('battery_temperature_c',40.),('virtual_temperature_c',45.))
    for key,limit in sensors:
        if key not in sample:missing.append(key)
        elif not isinstance(sample[key],(float,int)) or not math.isfinite(sample[key]):missing.append(key)
        elif sample[key]>limit:above.append({'sensor':key,'value':sample[key],'limit':limit})
    return {'missing_sensors_or_fields':missing,'warning_fields':warnings,'above_limits':above,
            'non_nominal_state':sample.get('thermal_state') not in (None,0)}


def foreign_waiters(own_session, directory=Path('/Users/Shared/mlxuag/gpu.lock.waiters')):
    """Read the shared heartbeat ledger; never remove or reorder waiters."""
    result=[]
    for path in directory.glob('*'):
        fields=path.name.rsplit('.',3)
        if len(fields)!=4 or fields[0]==own_session:continue
        try:
            if time.time()-path.stat().st_mtime>300:continue
            pid=int(fields[-2]);os.kill(pid,0)
        except (ValueError,FileNotFoundError,ProcessLookupError):continue
        except PermissionError:pass  # Treat an inaccessible live PID as queued.
        result.append({'name':path.name,'pid':pid})
    return result


def first_divergence(reference,actual):
    for i,(a,b) in enumerate(zip(reference,actual)):
        if a!=b:return {'index':i,'ordinary_token':a,'arm_token':b,'shared_prefix_tokens':i}
    if len(reference)!=len(actual):
        return {'index':min(len(reference),len(actual)),'length_mismatch':True,
                'ordinary_length':len(reference),'arm_length':len(actual)}
    return None


def validate_args(args):
    if not args.run_gpu or args.ownership_window is None:raise ValueError('explicit GPU and ownership window required')
    if args.output.exists():raise ValueError('refusing existing evidence directory')
    if not 64<=args.max_tokens<=128:raise ValueError('output budget64..128 required')
    if not 1<=args.prefill_step<=128:raise ValueError('prefill step1..128 required')
    if not 96<=args.memory_limit_gib<=112:raise ValueError('memory limit96..112GiB required')
    if not 0<=args.seed<2**32:raise ValueError('seed must be uint32')
    if not 60<=args.timeout_seconds<=1800:raise ValueError('bounded timeout60..1800 seconds required')
    if type(args.execution_policy)is not dict:raise ValueError('execution policy must be dictionary')
    if set(args.execution_policy)&{'draft_model','num_draft','deepseek_verifier','deepseek_draft_confidence'}:raise ValueError('comparison owns sidecar/verifier/depth policy')
    from mlx2.adapters.deepseek_v4_gguf import validate_execution_policy
    validate_execution_policy(args.execution_policy)


def sample_environment(profile=LEGACY_PROFILE):
    from run_qualification_matrix import sample_thermal,thermally_stable
    from mlx2.process_env import PROCESS_NUMERICS
    thermal=sample_thermal()
    if profile not in (LEGACY_PROFILE,HID_PROFILE):raise ValueError('unknown thermal profile')
    if profile==HID_PROFILE:
        from mlx2.apple_telemetry import TemperatureSampler
        sampler=TemperatureSampler()
        try:sensors=sampler.read()
        finally:sampler.close()
        thermal['hid_raw_sensors_c']=sensors
        die=[v for k,v in sensors.items() if 'tdie' in k]
        if die:thermal['die_max_c']=max(die)
        if 'gas gauge battery' in sensors:thermal['hid_battery_temperature_c']=sensors['gas gauge battery']
    failures=thermal_failures(thermal,profile)
    thermal_ok=(not any(failures.values())) if profile==HID_PROFILE else thermally_stable(thermal,THERMAL_POLICY)
    temperature_sampled_at=time.time()
    swap=subprocess.check_output(['sysctl','vm.swapusage'],text=True,timeout=5)
    match=re.search(r'used\s*=\s*([0-9.]+)([MGT])',swap)
    if match is None:raise RuntimeError('swap telemetry unavailable')
    vm=subprocess.check_output(['vm_stat'],text=True,timeout=5)
    counters={name:int(re.search(r'^'+name+r':\s*(\d+)',vm,re.M).group(1)) for name in ('Swapins','Swapouts')}
    ps=subprocess.check_output(['ps','-axo','pid=,ppid=,command='],text=True,timeout=5)
    rows=[]
    for line in ps.splitlines():
        fields=line.strip().split(None,2)
        if len(fields)==3:rows.append((int(fields[0]),int(fields[1]),fields[2]))
    ancestors={os.getpid()};parent=os.getppid()
    parents={pid:ppid for pid,ppid,_ in rows}
    while parent and parent not in ancestors:ancestors.add(parent);parent=parents.get(parent,0)
    foreign=[{'pid':pid,'command_sha256':hashlib.sha256(cmd.encode()).hexdigest()}
             for pid,_,cmd in rows if pid not in ancestors and
             re.search(r'(?:mlx2\.server|mlx_lm\.server|llama-server|(?:^|/)ds4(?:\s|$)|probe_deepseek_v4_)',cmd)]
    return {'time_unix':time.time(),'temperature_sampled_at_unix':temperature_sampled_at,'thermal':thermal,'thermal_profile':profile,
            'thermal_ok':thermal_ok,
            'thermal_failures':failures,
            'raw_swapusage':swap,'raw_vm_stat':vm,
            'swap_used_bytes':int(float(match.group(1))*{'M':2**20,'G':2**30,'T':2**40}[match.group(2)]),
            **counters,'foreign_model_processes':foreign,
            'numerics_environment':{key:os.environ.get(key) for key in PROCESS_NUMERICS}}


def gate_samples(before,after,*,now=None):
    samples=[*before,*after]
    result={'thermal':all(x['thermal_ok'] for x in samples),
            'no_foreign_model_processes':all(not x['foreign_model_processes'] for x in samples),
            'environment_stable':all(x['numerics_environment']==samples[0]['numerics_environment'] for x in samples),
            'swap_stable':all(x['Swapins']==samples[0]['Swapins'] and x['Swapouts']==samples[0]['Swapouts']
                              and x['swap_used_bytes']==samples[0]['swap_used_bytes'] for x in samples)}
    profiles={x.get('thermal_profile',LEGACY_PROFILE) for x in samples}
    result['thermal_profile_consistent']=len(profiles)==1
    if HID_PROFILE in profiles:
        result['hid_three_sample_stability']=all(
            len(rows)==3 and all(all(isinstance(x['thermal'].get(k),(int,float)) for k in ('die_max_c','hid_battery_temperature_c')) for x in rows)
            and all(max(x['thermal'][k] for x in rows)-min(x['thermal'][k] for x in rows)<=.5 for k in ('die_max_c','hid_battery_temperature_c'))
            for rows in (before,after))
        # Check each bracket at its endpoint. Historical pre-arm samples are
        # not stale merely because a measured arm legitimately took time.
        now=time.time() if now is None else now
        result['hid_sample_recency']=(0<=now-after[-1].get('temperature_sampled_at_unix',after[-1]['time_unix'])<=10 and
            all(all(0<=rows[-1]['time_unix']-x['time_unix']<=10 for x in rows) for rows in (before,after)))
    return result


def comparison_summary(records,budget):
    reasons=[];comparisons=[]
    environments=[s['numerics_environment'] for r in records for s in r.get('before',[])+r.get('after',[])]
    if environments and any(x!=environments[0] for x in environments):reasons.append('numerics environment changed across arms')
    for repetition in range(3):
        rows={r['arm']:r for r in records if r['repetition']==repetition}
        if set(rows)!=set(ARMS):reasons.append('incomplete repetition');continue
        ordinary=rows['ordinary']
        for arm,row in rows.items():
            if not all(row['gates'].values()):reasons.append(f'{repetition}/{arm}: environment gate')
            if row.get('paired_cache_resumes',0)!=0 or row.get('fresh_cache',True) is not True:
                reasons.append(f'{repetition}/{arm}: warm APC/cache contamination')
            if len(row['tokens'])!=budget:reasons.append(f'{repetition}/{arm}: short output')
            divergence=first_divergence(ordinary['tokens'],row['tokens'])
            row['exact_ordinary_tokens']=divergence is None
            row['first_divergence']=divergence
            if divergence is not None:reasons.append(f'{repetition}/{arm}: token divergence requires bounded replay')
            if arm!='ordinary':comparisons.append({'repetition':repetition,'arm':arm,
                'ordinary_over_arm_seconds':ordinary['decode_seconds']/row['decode_seconds']})
    ordinary_rows=[r for r in records if r['arm']=='ordinary']
    if ordinary_rows and any(r['tokens']!=ordinary_rows[0]['tokens'] for r in ordinary_rows):reasons.append('ordinary repeat token drift')
    eligible=not reasons and len(records)==9
    return {'comparison_eligible':eligible,'gate_failures':sorted(set(reasons)),
            'performance_claim':False,'qualification':False,
            'paired_ratios':comparisons if eligible else None,
            'median_speedup':{arm:statistics.median(x['ordinary_over_arm_seconds'] for x in comparisons if x['arm']==arm)
                              for arm in ARMS[1:]} if eligible else None,
            'median_evaluated_prefill_tokens_per_second':{
                arm:statistics.median(r['prefill_evaluated_tokens_per_second'] for r in records if r['arm']==arm)
                for arm in ARMS} if eligible and all('prefill_evaluated_tokens_per_second' in r for r in records) else None,
            'interpretation':'ordinary control retains full sidecar allocation; not shipping ordinary footprint; causal chain slab is not tree AST'}


def run_comparison(run_arm,sample,guard,publish,*,budget,sleep=time.sleep,
                   publish_admission=lambda event:None,waiters=lambda:[],
                   clock=time.monotonic,settle_seconds=60):
    if not 0<=settle_seconds<=60:raise ValueError('settling bound must be0..60 seconds')
    warmups=[];records=[];settling_used=False
    for arm in ARMS:
        guard();warmups.append(run_arm(arm));publish(warmups,records,None)
    for repetition,order in enumerate(ORDERS):
        for ordinal,arm in enumerate(order):
            guard();before=[]
            event={'repetition':repetition,'ordinal':ordinal,'arm':arm,'attempts':[],'status':'sampling'}
            def collect(kind):
                values=[];attempt={'kind':kind,'samples':values}
                event['attempts'].append(attempt);publish_admission(event)
                for i in range(3):
                    guard()
                    try:values.append(sample())
                    except Exception as error:
                        attempt['sampling_error']={'type':type(error).__name__,'message':str(error)}
                        event['status']='telemetry_error';publish_admission(event);raise
                    publish_admission(event)
                    if i<2:sleep(1)
                attempt['gates']=gate_samples(values,values)
                publish_admission(event)
                return values,attempt['gates']
            before,admission=collect('initial')
            if not all(admission.values()):
                missing=any(x.get('thermal_failures',{}).get('missing_sensors_or_fields') for x in before)
                recoverable=(not missing and admission['no_foreign_model_processes'] and admission['environment_stable'])
                if recoverable and not settling_used and settle_seconds:
                    settling_used=True;settle_start=clock();deadline=settle_start+settle_seconds
                    event['status']='settling';event['settle_limit_seconds']=settle_seconds;publish_admission(event)
                    while clock()+3<deadline:
                        guard();queued=waiters()
                        if queued:
                            event.update(status='yield_foreign_waiter',foreign_waiters=queued);publish_admission(event)
                            raise RuntimeError('yielding bounded settling window to foreign GPU waiter')
                        sleep(min(5,max(0,deadline-clock()-3)))
                        before,admission=collect('settling')
                        if all(admission.values()):break
                        if any(x.get('thermal_failures',{}).get('missing_sensors_or_fields') for x in before):break
                    event['settling_elapsed_seconds']=clock()-settle_start
                event['status']='admitted_after_settling' if all(admission.values()) else 'refused'
                event['missing_thermal_fields']=missing;publish_admission(event)
                if not all(admission.values()):raise RuntimeError('comparison admission gate failed: '+json.dumps(admission))
            else:
                event['status']='admitted';publish_admission(event)
            row=run_arm(arm)
            after=[]
            for i in range(3):
                after.append(sample())
                if i<2:sleep(1)
            row.update(repetition=repetition,ordinal=ordinal,arm=arm,before=before,after=after,gates=gate_samples(before,after))
            records.append(row);guard();publish(warmups,records,None)
    summary=comparison_summary(records,budget);publish(warmups,records,summary)
    return summary


class LoadedArms:
    def __init__(self,adapter,prompt,budget,prefill_step,seed,guard):
        self.adapter,self.model,self.prompt=adapter,adapter.model,prompt
        self.budget,self.prefill_step,self.seed,self.guard=budget,prefill_step,seed,guard
        from mlx2.adapters.deepseek_v4_external import DeepSeekV4DSparkDraftModel
        self.drafts={mode:DeepSeekV4DSparkDraftModel(adapter.draft_model.backbone,verification_mode=mode,confidence_threshold=adapter.draft_model.confidence_threshold).bind(self.model)
                     for mode in ARMS[1:]}

    def __call__(self,arm):
        import mlx.core as mx
        mx.random.seed(self.seed)
        cache=generator=None;counts={'target_forward_calls':0,'target_input_rows':0}
        original=self.model._forward
        original_prefill=self.model.prefill_body
        prefill_calls=[]
        def counted(inputs,*args,**kwargs):
            counts['target_forward_calls']+=1;counts['target_input_rows']+=inputs.shape[1]
            return original(inputs,*args,**kwargs)
        self.model._forward=counted
        def timed_prefill(inputs, cache, capture_layers):
            tic=time.perf_counter()
            output=original_prefill(inputs,cache,capture_layers)
            mx.eval(output,[row.state for row in cache])
            prefill_calls.append({'tokens':inputs.shape[1],'evaluated_seconds':time.perf_counter()-tic})
            return output
        self.model.prefill_body=timed_prefill
        try:
            # Same allocator-cache policy for each fresh state, outside timing.
            mx.clear_cache();self.guard();prefill_start=time.perf_counter()
            tokens=[];round_times=[];stats={};last_receipt=None
            if arm=='ordinary':
                cache=self.model.make_cache()
                for offset in range(0,len(self.prompt)-1,self.prefill_step):
                    self.guard();part=self.prompt[offset:min(offset+self.prefill_step,len(self.prompt)-1)]
                    out=self.model.prefill_body(mx.array([part],mx.uint32),cache,self.drafts['sequential'].config.target_layer_ids)
                    mx.eval(out,[c.state for c in cache])
                prefill_seconds=time.perf_counter()-prefill_start
                counts={key:0 for key in counts};anchor=self.prompt[-1]
                start=time.perf_counter()
                for _ in range(self.budget):
                    tic=time.perf_counter()
                    logits=self.model(mx.array([[anchor]],mx.uint32),cache,logits_to_keep=0)
                    mx.eval(logits,[c.state for c in cache])
                    anchor=int(mx.argmax(logits[0,-1]).item())
                    round_times.append(time.perf_counter()-tic);tokens.append(anchor)
                    if anchor in self.adapter.tokenizer.eos_token_ids:break
                elapsed=time.perf_counter()-start
            else:
                from mlx2.adapters.deepseek_v4_external import DeepSeekV4ExternalBatchGenerator
                generator=DeepSeekV4ExternalBatchGenerator(self.model,draft_model=self.drafts[arm],
                    num_draft=5,completion_batch_size=1,prefill_step_size=self.prefill_step,
                    cost_per_commit_handoff=False,stop_tokens=tuple((int(t),) for t in self.adapter.tokenizer.eos_token_ids),
                    target_artifact_fingerprint=self.adapter.target_identity['fingerprint'])
                generator.insert([self.prompt],max_tokens=[self.budget],sampling_configs=[{'sampling_temp':0.}])
                # Pinned executor prefill primitive creates the exact paired
                # boundary. next() may combine final prefill and decode, which
                # would contaminate separated timing; never call it here.
                for _ in range(len(self.prompt)+2):
                    if all(lane.anchor is not None for lane in generator.lanes.values()):break
                    self.guard()
                    generator._prefill(next(iter(generator.lanes.values())))
                if not generator.lanes or any(lane.anchor is None for lane in generator.lanes.values()):raise RuntimeError('external prefill did not reach exact boundary')
                if generator.scheduler_stats['paired_cache_resumes']!=0:raise RuntimeError('warm APC resume contaminates comparison')
                prefill_seconds=time.perf_counter()-prefill_start;counts={key:0 for key in counts}
                start=time.perf_counter()
                for _ in range(self.budget*3+4):
                    tic=time.perf_counter();_,responses=generator.next()
                    round_times.append(time.perf_counter()-tic)
                    for response in responses:
                        tokens.append(int(response.token));last_receipt=getattr(response,'speculative_receipt',None)
                    if not generator.lanes:break
                elapsed=time.perf_counter()-start
                if generator.lanes:raise RuntimeError('bounded decode polls exhausted')
                stats=dict(generator.scheduler_stats)
            self.guard()
            prefill_evaluated=sum(x['evaluated_seconds'] for x in prefill_calls)
            prefill_tokens=sum(x['tokens'] for x in prefill_calls)
            if prefill_tokens!=len(self.prompt)-1:raise RuntimeError('prefill evaluated token count differs from common prompt prefix')
            return {'arm':arm,'tokens':tokens,'decode_seconds':elapsed,'prefill_seconds':prefill_seconds,
                    'prefill_evaluated_seconds':prefill_evaluated,'prefill_evaluated_tokens':prefill_tokens,
                    'prefill_evaluated_tokens_per_second':prefill_tokens/prefill_evaluated,
                    'prefill_calls':prefill_calls,
                    'prefill_timing_scope':'summed evaluated target body chunks, all but final prompt token; excludes load/checks/receipt IO and draft support projection',
                    'observed_tokens_per_second':len(tokens)/elapsed,'round_seconds':round_times,
                    'scheduled_target_work':dict(counts),'scheduler_stats':stats,'last_speculative_receipt':last_receipt,
                    'accepted_proposals':stats.get('accepted_proposals',0),'proposed_tokens':stats.get('proposed_tokens',0),
                    'paired_cache_resumes':stats.get('paired_cache_resumes',0),'fresh_cache':True,
                    'allocator_active_bytes':mx.get_active_memory(),
                    'timing_scope':'evaluated decode wall including sampling/round/snapshot/replay overhead; no per-token file serialization'}
        finally:
            self.model._forward=original
            self.model.prefill_body=original_prefill
            if generator is not None:generator.close()
            if cache is not None:
                for row in cache:row.close()
            gc.collect()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model-path','sidecar-path','output','request-json'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run-gpu',action='store_true');p.add_argument('--ownership-window',type=Path)
    p.add_argument('--owner-path',type=Path,action='append');p.add_argument('--max-tokens',type=int,default=64)
    p.add_argument('--prefill-step',type=int,default=128);p.add_argument('--seed',type=int,default=20261024)
    p.add_argument('--memory-limit-gib',type=float,default=108);p.add_argument('--timeout-seconds',type=int,default=1800)
    p.add_argument('--execution-policy',type=json.loads,default={})
    p.add_argument('--thermal-profile',choices=(LEGACY_PROFILE,HID_PROFILE),default=LEGACY_PROFILE)
    args=p.parse_args(argv)
    try:validate_args(args)
    except ValueError as error:p.error(str(error))
    owners=args.owner_path or [Path('/Users/Shared/mlxuag/gpu.lock/owner.json'),Path('/tmp/gpu.lock/owner.json')]
    if len(owners)!=2:p.error('paired owner paths required')
    started=time.monotonic()
    def guard():
        if time.monotonic()-started>args.timeout_seconds:raise RuntimeError('bounded comparison deadline exceeded')
        return verify_ownership(args.ownership_window,owners)
    ownership=guard();request,request_binding=load_request(args.request_json)
    receipt={'schema':'mlx2.deepseek-v4-decode-comparison.v1','status':'running','source':source_receipt(),
        'ownership':ownership,'performance_claim':False,'qualification':False,'warmups':[],'records':[],
        'admissions':[],
        'controls':{'output_budget':args.max_tokens,'prefill_step':args.prefill_step,'seed':args.seed,'repetitions':3,
                    'orders':ORDERS,'cache':'fresh per arm; APC disabled','sidecar':'same allocation resident in all arms including ordinary',
                    'thermal_profile':args.thermal_profile,
                    'thermal_policy':HID_THERMAL_POLICY if args.thermal_profile==HID_PROFILE else THERMAL_POLICY,
                    'temperature_recency_scope':'host sample retrieval timestamps; hardware sensor timestamps unavailable',
                    'swap_gate':'zero swapins/swapouts/used-byte change per bracket'}}
    args.output.mkdir(parents=True);atomic(args.output/'results.json',receipt)
    adapter=None
    try:
        import mlx.core as mx
        from mlx2.process_env import require_process_numerics
        from mlx2.adapters.deepseek_v4_gguf import DeepSeekV4GGUFAdapter
        require_process_numerics('DeepSeek controlled decode comparison')
        mx.set_default_device(mx.gpu);mx.set_memory_limit(int(args.memory_limit_gib*(1<<30)));mx.set_cache_limit(256<<20)
        policy={**args.execution_policy,'draft_model':str(args.sidecar_path),'num_draft':5,'deepseek_verifier':'sequential'}
        adapter=DeepSeekV4GGUFAdapter(str(args.model_path),execution_policy=policy)
        wire_preloaded_weights(mx,receipt)
        wiring=receipt['weight_residency']
        receipt['wiring_capacity_gate']=bool(wiring.get('wired') and wiring['active_bytes_at_wire']<=wiring['wired_limit_bytes'])
        if not receipt['wiring_capacity_gate']:raise RuntimeError('preloaded allocations exceed configured residency request or wiring unavailable')
        prompt=list(adapter.prompt_tokens(request))
        if not 2<=len(prompt)<=2048:raise ValueError('bounded prompt2..2048 tokens required')
        receipt.update(checkpoint=args.model_path.name+'@'+adapter.target_identity['fingerprint'],
                       artifact=adapter.target_identity,sidecar=adapter.external_record,execution_policy=adapter.kernel_policy,
                       request_binding=request_binding,prompt_tokens=prompt,target_execution_identity=adapter.model.identity)
        receipt['controls'].update(memory_limit_gib=args.memory_limit_gib,allocator_cache_limit_bytes=256<<20,
            cache_layout=adapter.layout,cache_reservation_bytes=adapter.cache_budget(mtp=False).project(len(prompt)+args.max_tokens),
            prefill_boundary='all but final prompt token, via pinned paired executor prefill primitive for draft arms',
            output_stop_policy='honor EOS; short output makes performance comparison ineligible')
        atomic(args.output/'results.json',receipt)
        def publish(warmups,records,summary):
            receipt.update(warmups=warmups,records=records)
            if summary is not None:receipt['summary']=summary
            atomic(args.output/'results.json',receipt)
        def publish_admission(event):
            key=(event['repetition'],event['ordinal'])
            rows=receipt['admissions']
            if rows and (rows[-1]['repetition'],rows[-1]['ordinal'])==key:rows[-1]=event
            else:rows.append(event)
            atomic(args.output/'results.json',receipt)
        summary=run_comparison(LoadedArms(adapter,prompt,args.max_tokens,args.prefill_step,args.seed,guard),
            lambda:sample_environment(args.thermal_profile),guard,publish,budget=args.max_tokens,publish_admission=publish_admission,
            waiters=lambda:foreign_waiters(ownership['desktop_lease']['cpg_session']))
        receipt.update(status='completed',comparison_eligible=summary['comparison_eligible'])
    except BaseException as error:
        receipt.update(status='failed',error={'type':type(error).__name__,'message':str(error)});raise
    finally:
        if adapter is not None:
            try:adapter.close()
            except Exception as error:receipt.update(status='failed',cleanup_error=str(error))
        gc.collect()
        try:
            mx.clear_cache();restore_probe_wired_limit(mx,receipt)
        except Exception as error:receipt.update(status='failed',residency_cleanup_error=str(error))
        atomic(args.output/'results.json',receipt)
    return 0 if receipt['status']=='completed' and receipt.get('comparison_eligible') else 1


if __name__=='__main__':raise SystemExit(main())
