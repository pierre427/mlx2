"""Real hybrid serving-attachment lifecycle smoke; dry by default, root-run only.

Uses real Job, BatchGenerator, source-bound serving installer and greedy sampler.
This is not HTTP execution, model numeric parity or a performance qualification.
The supervisor imports no MLX and kills the isolated worker at 100s/48GiB RSS.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
MODEL=str(Path.home()/'mlx-models'/'Qwen3.8-27B-MLX-4bit')
MANIFEST='/tmp/mlx2-hybrid27b-artifact-1004.json'
WHEEL=str(Path.home()/'.cache/uv/sdists-v9/path/6b22317da775785a/d5Co9cAdGp8_XULv/mlx-0.32.2.dev20260919+39400a0d4-cp312-cp312-macosx_26_0_arm64.whl')
NATIVE='/tmp/mlx2-native-bf16-storage-build-1004/_paged_kv_native.cpython-312-darwin.so'
NATIVE_SHA='f220d9391ea4c48c56c5fb11c4e0ab37a62298dcc9d9d270a9d0394a894577f1'
MAX_SECONDS=100
MAX_RSS=48<<30
FAILURE_ROOTS=[]


def make_profile(identity,stock_reduction=False,stock_singleton=False):
    if type(stock_reduction) is not bool:raise ValueError("stock reduction must be explicit boolean")
    if type(stock_singleton) is not bool or (stock_singleton and not stock_reduction):
        raise ValueError("stock singleton requires exact boolean and explicit stock reduction")
    flags={'MLX2_PAGED_HYBRID_B2':'1','MLX2_PAGED_Q1_SIMD_TILE':'1',
        'MLX2_PAGED_GROUPED_Q1_WRITE':'1','MLX2_PAGED_PRIVATE_TAIL_REUSE':'1',
        'MLX2_PAGED_Q1_SIMD_STRIPES':'16','MLX2_PAGED_Q1_SPLIT_KV':'0',
        'MLX2_PAGED_Q1_STOCK_REDUCTION':'1' if stock_reduction else '0',
        'MLX2_PAGED_Q1_STOCK_SINGLETON':'1' if stock_singleton else '0',
        **{key:'0' for key in ('MLX2_PAGED_Q1_STOCK_SDPA','MLX2_PAGED_Q1_INLINE_METADATA',
            'MLX2_PAGED_B2_DEFERRED_EVAL','MLX2_PAGED_B2_DEFERRED_WRITE_EVAL',
            'MLX2_PAGED_GROUPED_SAMPLER','MLX2_PAGED_GROUPED_DIRECT_FENCE')}}
    return dict(schema='mlx2.native-hybrid-b2-research-admission.v1',
        profile_id='hybrid27b-serving-lifecycle-short-1004',identity=identity,
        context_bounds={'minimum':32,'maximum':96,'distinct':True},max_tokens=4,
        sampling={'mode':'greedy','processors':False},required_environment=flags,
        qualified=False,price_usable=False,serving_default=False,warm_apcv2=False,
        q1_simd_stripes=16,q1_split_partition=0,storage_dtype='bfloat16',memory_budget_bytes=2<<30,
        stock_reduction=stock_reduction,stock_singleton=stock_singleton)


def plan():
    return dict(schema='mlx2.hybrid-serving-lifecycle-smoke.v1',status='planned',gpu_executed=False,
        qualified=False,performance_qualified=False,numeric_parity='not_tested',
        measurement_scope='real Job and BatchGenerator serving attachment, not HTTP',
        prompts=[32,96],output_caps=[2,4],cancellation_after_lane0_tokens=None,
        alternative='--cancel-after-two uses caps4/4 and cancels lane0 after two actual sampled tokens',
        hard_seconds=MAX_SECONDS,max_rss_bytes=MAX_RSS,native_sha256=NATIVE_SHA,
        required_proofs=['no sampler during attachment','actual greedy sampler calls',
            'native hybrid selected/used receipts','B2 physical read proof','B1 survivor',
            'terminal callbacks and KV/GDN atomic boundary','complete owner/page/charge retirement'])


def save(path,result):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(path)


def run_worker(args,result):
    import hashlib
    from threading import RLock
    from varlen_pack_price_bench import _gpuq_owner
    result['gpuq_owner']=_gpuq_owner()
    if len(args.native_sha256)!=64 or any(c not in '0123456789abcdef' for c in args.native_sha256):
        raise ValueError('explicit native SHA256 required')
    if hashlib.sha256(Path(args.native).read_bytes()).hexdigest()!=args.native_sha256:
        raise RuntimeError('frozen BF16 native binary differs')
    sys.path.insert(0,str(Path(args.native).parent));sys.path.insert(0,str(ROOT/'src'))
    from mlx2.runtime.paged_price_identity import cached_live_price_identity
    identity=cached_live_price_identity(Path(args.artifact_manifest),Path(args.mlx_wheel),Path(args.native),
        adapter_artifact_root=Path(args.model).resolve())
    profile=make_profile(identity,args.stock_reduction,args.stock_singleton)
    result.update(native_sha256=args.native_sha256,native_path=str(Path(args.native).resolve()),
                  stock_reduction=args.stock_reduction,stock_singleton=args.stock_singleton)
    if args.prepare_profile:
        save(args.profile,profile)
        result.update(status='profile_prepared',identity=identity,gpu_executed=False)
        return
    from mlx2.runtime.paged_hybrid_research_profile import load_hybrid_research_profile
    os.environ.update(profile['required_environment'])
    profile=load_hybrid_research_profile(args.profile,live_identity=identity,context_lengths=(32,96),environment=os.environ)
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter,configure_environment
    configure_environment()
    import mlx.core as mx
    from mlx2 import serving
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import make_sampler
    from mlx2.runtime import qwen35_paged_graph_factory as factory
    result.update(identity=identity,gpu_executed=True,numeric_parity='not_tested',
        numerical_policy='loaded BF16 unchanged, stock-reduction flag'+str(int(args.stock_reduction)),events=[],sampler_calls={})
    started=time.perf_counter();adapter=None;batch=None;candidate=None;owners=();uids=();clean=False
    initial_charge=factory._CHARGED
    try:
        adapter=Qwen3827BAdapter(args.model,require_mtp=False)
        result['model_load_seconds']=time.perf_counter()-started
        seed=adapter.tokenizer.encode('Explain database isolation, causality, durability and distributed consistency. ',add_special_tokens=False)
        if not seed: raise RuntimeError('empty tokenizer seed')
        prompts=tuple(tuple((seed*((n+len(seed)-1)//len(seed)))[:n]) for n in (32,96))
        caps=(2,4) if args.unequal_caps else (4,4)
        result.update(output_caps=list(caps),cancellation_after_lane0_tokens=None if args.unequal_caps else 2)
        lock=RLock();greedy=make_sampler(temp=0.0)
        observed={0:0,1:0}
        def sampler(index):
            def invoke(logprobs):
                observed[index]+=1
                token=greedy(logprobs)
                # Observation only: never replace the sampled value.
                if not bool(mx.all(token==mx.argmax(logprobs,axis=-1)).item()):
                    raise RuntimeError('actual sampler differs from greedy reference')
                return token
            return invoke
        batch=BatchGenerator(adapter.model,max_tokens=4,prefill_batch_size=2,
            completion_batch_size=2,prefill_step_size=128,stop_tokens=[])
        uids=tuple(batch.insert([list(p) for p in prompts],max_tokens=list(caps),samplers=[sampler(0),sampler(1)]))
        jobs=[]
        for uid,prompt,cap in zip(uids,prompts,caps):
            request={'temperature':0,'max_tokens':cap,'paged_native_hybrid_b2':True,
                'skip_writing_prefix_cache':True,'batch_cohort':{'id':'hybrid-lifecycle-smoke','size':2},
                'repetition_penalty':1,'presence_penalty':0,'frequency_penalty':0}
            sampling,defaults=serving.resolve_sampling(request,serving.vendor_sampling(adapter),thinking=None)
            jobs.append(serving.Job(request=request,uid=uid,native_b2_prompt=prompt,effective_max_tokens=cap,
                                    effective_sampling=sampling,sampling_defaults=defaults))
        jobs=tuple(jobs)
        result['effective_sampling']=[dict(job.effective_sampling) for job in jobs]
        tick=time.perf_counter()
        with lock:
            attachment=serving.install_explicit_native_hybrid_b2_cohort(batch,adapter,jobs,lifecycle_lock=lock,
                profile_path=str(args.profile),manifest_path=args.artifact_manifest,mlx_wheel_path=args.mlx_wheel)
        result['attach_seconds']=time.perf_counter()-tick;result['attachment_receipts']=attachment
        if any(observed.values()): raise RuntimeError('sampler executed during atomic attachment')
        if set(batch._native_continuations)!=set(uids): raise RuntimeError('both native lanes did not attach')
        candidate=batch._native_continuations[uids[0]].candidate
        owners=tuple(batch._native_continuations[uid].owner for uid in uids)
        arena=candidate.backend.writer.backend
        original_close=arena.close_after_terminal
        def observed_terminal_close():
            # Natural final completion may close the arena inside next().
            # Capture real counters before close, then invoke the exact method.
            if not arena._closed:
                result['physical_before_close']=candidate.backend.profile_counters_snapshot()
            return original_close()
        arena.close_after_terminal=observed_terminal_close
        if candidate is not batch._native_continuations[uids[1]].candidate: raise RuntimeError('cohort has separate graphs')
        if any(r.get('route')!='native_hybrid_paged_b2' or not r.get('selected') or r.get('observed_used') or r.get('qualified') for r in attachment):
            raise RuntimeError('attachment receipt state differs')
        result['physical_before']=candidate.backend.profile_counters_snapshot()
        emitted={uid:[] for uid in uids};cancelled=False;survivor_steps=[]
        for round_index in range(6):
            tick=time.perf_counter()
            with lock:prompt_events,responses=batch.next()
            result['sampler_calls']=dict(observed)
            result['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if result['peak_rss_bytes']>MAX_RSS:raise MemoryError('48 GiB peak RSS ceiling')
            failures=batch.take_lane_failures()
            if failures:result['lane_failures']=failures;raise RuntimeError('real native serving lane failed')
            for response in responses:
                receipt=dict(response.mtp_receipt or {})
                index=uids.index(response.uid);emitted[response.uid].append(int(response.token))
                event={'round':round_index,'lane':index,'token':int(response.token),
                    'finish_reason':response.finish_reason,'execution_width':getattr(response,'execution_width',1),
                    'route_receipt':receipt,'round_seconds':time.perf_counter()-tick}
                result['events'].append(event)
                if receipt.get('route')!='native_hybrid_paged_b2' or not receipt.get('selected') or receipt.get('qualified'):
                    raise RuntimeError('actual serving route receipt differs')
                if len(emitted[response.uid])>1 and not receipt.get('observed_used'):
                    raise RuntimeError('decoded response lacks observed native use')
                if (cancelled or (args.unequal_caps and len(emitted[uids[0]])==2)) and index==1 and len(emitted[response.uid])>2:
                    survivor_steps.append(event)
            if not args.unequal_caps and not cancelled and len(emitted[uids[0]])==2:
                with lock:
                    jobs[0].cancelled.set();batch.remove([uids[0]])
                cancelled=True;result['cancelled_uid']=uids[0]
            save(args.output,result)
            if len(emitted[uids[1]])==4:break
        if (len(emitted[uids[0]]),len(emitted[uids[1]]))!=(2,4):raise RuntimeError('unequal2/4 completion proof differs')
        if observed!={0:2,1:4}:raise RuntimeError('actual sampler invocation count differs')
        if len(survivor_steps)!=2 or any(event['execution_width']!=1 for event in survivor_steps):
            raise RuntimeError('two actual B1 survivor steps required')
        if result['events'][-1]['finish_reason']!='length':raise RuntimeError('survivor did not reach its real token bound')
        result.update(output_token_ids={str(uid):tokens for uid,tokens in emitted.items()},
            sampler_calls=observed,physical_after=(result.get("physical_before_close") or candidate.backend.profile_counters_snapshot()),
            B1_survivor_proved=True,numeric_parity='not_tested',status='lifecycle_passed')
        if args.stock_singleton:
            physical=result['physical_after'];depth=candidate.native_layer_count
            if (physical.get('q1_stock_reduction_dispatches')!=3*depth or
                    physical.get('q1_stock_singleton_dispatches')!=2*depth or
                    physical.get('q1_stripe_dispatches_32')!=3*depth):
                raise RuntimeError('actual serving singleton stock32 physical totals differ')
            for event in survivor_steps:
                receipt=event['route_receipt'];proof=receipt.get('hybrid_graph_proof',{})
                if (receipt.get('stock_singleton_selected') is not True or
                        receipt.get('stock_singleton_observed_used') is not True or
                        receipt.get('q1_simd_stripes')!=32 or proof.get('native_stock_singleton_dispatches')!=depth):
                    raise RuntimeError('serving survivor singleton stock32 receipt differs')
        clean=True
    finally:
        if batch is not None:
            with lock:
                batch.remove(list(uids));batch._reap_native_retiring()
                factory.reap_hybrid_admission_orphans()
                if candidate is not None:
                    candidate._serving_resources.reap()
                    writer=candidate.backend.writer
                    result['cleanup']={'pending_epochs':len(writer.pending_epochs),'pending_ledger':writer.ledger.pending_count,
                        'allocated_pages':writer.pool.allocated_count,'owners_retired':all(owner.fully_retired for owner in owners),
                        'resources_closed':candidate._serving_resources.closed,'global_charge_bytes':factory._CHARGED,
                        'hybrid_admission_orphans':len(factory._ORPHANS)}
                    terminal=(not writer.pending_epochs and not writer.ledger.pending_count and
                        writer.pool.allocated_count==0 and all(owner.fully_retired for owner in owners) and
                        candidate._serving_resources.closed and factory._CHARGED==initial_charge)
                    if not terminal:
                        FAILURE_ROOTS.append((adapter,batch,candidate,owners));raise RuntimeError('native serving resources retained')
                batch.close()
        if adapter is not None and not FAILURE_ROOTS:adapter.close()
        if clean:result['status']='lifecycle_passed'


def supervise(args,result):
    command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker']
    began=time.monotonic();process=subprocess.Popen(command,start_new_session=True)
    try:
        while process.poll() is None:
            if time.monotonic()-began>MAX_SECONDS:raise TimeoutError('100 second worker deadline')
            try:
                rss=int(subprocess.check_output(['ps','-o','rss=','-p',str(process.pid)],text=True).strip() or '0')*1024
            except (subprocess.CalledProcessError,ValueError):rss=0
            if rss>MAX_RSS:raise MemoryError('48 GiB worker RSS ceiling')
            time.sleep(.2)
        return process.returncode
    except BaseException as error:
        os.killpg(process.pid,signal.SIGKILL);process.wait()
        if args.output.is_file():result=json.loads(args.output.read_text())
        result.update(status='failed',error=f'{type(error).__name__}: {error}',worker_killed=True,qualified=False)
        save(args.output,result);return 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=MODEL);parser.add_argument('--artifact-manifest',default=MANIFEST)
    parser.add_argument('--mlx-wheel',default=WHEEL);parser.add_argument('--native',default=NATIVE)
    parser.add_argument('--native-sha256',default=NATIVE_SHA)
    parser.add_argument('--stock-reduction',action='store_true')
    parser.add_argument('--stock-singleton',action='store_true')
    parser.add_argument('--profile',type=Path,default=Path('/tmp/mlx2-hybrid-serving-smoke-profile-1004.json'))
    parser.add_argument('--output',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--prepare-profile',action='store_true');mode.add_argument('--execute',action='store_true')
    scenario=parser.add_mutually_exclusive_group()
    scenario.add_argument('--unequal-caps',action='store_true',default=True)
    scenario.add_argument('--cancel-after-two',action='store_false',dest='unequal_caps')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();result=plan()
    if (args.execute or args.prepare_profile) and not args.worker:return supervise(args,result)
    code=0
    try:
        if args.execute or args.prepare_profile:run_worker(args,result)
    except BaseException as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}',retained_failure_roots=len(FAILURE_ROOTS));code=1
    save(args.output,result);return code

if __name__=='__main__':raise SystemExit(main())
