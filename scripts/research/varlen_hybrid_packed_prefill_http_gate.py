"""Bounded real HTTP cold packed-prefill NAX hybrid27B gate, dry by default and root-run only.

Two actual completions, source-bound admission, native vs ordinary sampled IDs,
B2->B1 lifecycle and final HTTP receipts. No FP16 cast or qualification claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from varlen_hybrid_serving_smoke import MODEL,MANIFEST,WHEEL,save
from varlen_hybrid_packed_prefill_geometry_gate import DEFAULT_NATIVE,DEFAULT_NATIVE_SHA
ROOT=Path(__file__).resolve().parents[2]
NATIVE=DEFAULT_NATIVE
NATIVE_SHA=DEFAULT_NATIVE_SHA
MAX_SECONDS=100;MAX_RSS=48<<30
FAILURE_ROOTS=[]


def preflight():
    return dict(schema='mlx2.hybrid-packed-prefill-http-nax-gate.v1',status='planned',gpu_executed=False,
        qualified=False,price_usable=False,numeric_tensor_parity='not_tested',token_parity='not_tested',
        measurement_scope='real loopback completions HTTP and native vs ordinary greedy output',
        context_tokens=[32,96],max_tokens=[2,4],stock_reduction=True,
        hard_seconds=MAX_SECONDS,max_rss_bytes=MAX_RSS,native_sha256=NATIVE_SHA)


def summarize_http(body,cap,native,stock_singleton=False):
    details=body.get('mlx2',{})
    receipt=details.get('route_receipt',{})
    if native:
        if (not isinstance(receipt,dict) or receipt.get('route')!='native_hybrid_paged_b2' or
                receipt.get('selected') is not True or receipt.get('observed_used') is not True or
                receipt.get('qualified') is not False or receipt.get('price_usable') is not False or
                receipt.get('prefill_mode')!='native_packed_prefill' or receipt.get('prefill_layout')!='real_rows' or
                receipt.get('native_prefill_observed_used') is not True or
                receipt.get('stock_reduction_selected') is not True or
                receipt.get('state_planes')!=['kv','gdn'] or
                len(receipt.get('output_token_ids',()))!=cap):
            raise RuntimeError('final HTTP native hybrid route receipt differs')
        prefill=receipt.get('native_prefill_proof',{})
        if (receipt.get('native_prefill_attention_calls')!=16 or prefill.get('prefill_nax_exact') is not True or
                prefill.get('serving_numerical_reference')!='same_geometry_ordinary_mixed' or
                prefill.get('physical_counters')!={'grouped_multirow_write_count':16,'grouped_multirow_row_count':2048,
                    'prefill_matrix_dispatch_count':16,**{'prefill_nax_'+stage+'_dispatch_count':16 for stage in ('score','softmax','value')}}):
            raise RuntimeError('final HTTP packed prefill NAX source/physical receipt differs')
        expected_singleton=stock_singleton and cap==4
        proof=receipt.get('hybrid_graph_proof',{})
        if (receipt.get('stock_singleton_selected',False) is not stock_singleton or
                receipt.get('stock_singleton_observed_used',False) is not expected_singleton or
                receipt.get('q1_simd_stripes')!=(32 if stock_singleton or cap==2 else 16) or
                proof.get('native_stock_singleton_dispatches',0)!=(16 if expected_singleton else 0)):
            raise RuntimeError('final HTTP singleton stock32 receipt differs')
    elif (isinstance(receipt,dict) and receipt.get('route')=='native_hybrid_paged_b2') or details.get('route')=='native_hybrid_paged_b2':
        raise RuntimeError('ordinary HTTP selected native hybrid')
    if details.get('qualification')!='unqualified':raise RuntimeError('HTTP qualification label differs')
    if body.get('usage',{}).get('completion_tokens')!=cap:raise RuntimeError('actual HTTP output cap differs')
    choices=body.get('choices',[])
    if len(choices)!=1 or choices[0].get('finish_reason')!='length':raise RuntimeError('HTTP completion did not reach length bound')
    return {'id':body.get('id'),'text':choices[0].get('text'),'route_receipt':receipt,'usage':body['usage']}


def validate_physical(snapshot,depth,stock_singleton=False):
    stock_steps=3 if stock_singleton else 1
    if (snapshot.get('q1_stock_reduction_dispatches')!=stock_steps*depth or
            snapshot.get('q1_stock_singleton_dispatches',0)!=(2*depth if stock_singleton else 0) or
            snapshot.get('q1_stripe_dispatches_32')!=stock_steps*depth or
            snapshot.get('q1_stripe_dispatches_16')!=(0 if stock_singleton else 2*depth) or
            snapshot.get('q1_tile_dispatches')!=3*depth):
        raise RuntimeError('HTTP actual B2/B1 stock32 physical proof differs')


def post(base,body):
    request=Request(base+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    try:
        with urlopen(request,timeout=40) as response:return response.status,json.load(response)
    except HTTPError as error:
        with error:return error.code,json.load(error)


def host_diagnostics(engine,phase,captures=()):
    """Host-only evidence; never wait for a lock held by the stalled worker."""
    states={'phase':phase,'monotonic':time.monotonic(),'factory_captures':len(captures)}
    if engine is not None:
        states.update(error=str(engine.error) if engine.error else None,
            worker_alive=engine.thread.is_alive(),ready=engine.ready.is_set(),
            stop_requested=engine.stop_event.is_set(),queued_jobs=getattr(engine,'queued_jobs',None))
        for lock_name,field in (('submission_lock','pending_cohorts'),('lock','jobs')):
            lock=getattr(engine,lock_name)
            acquired=lock.acquire(blocking=False)
            states[lock_name+'_acquired']=acquired
            if acquired:
                try:
                    if field=='pending_cohorts':
                        states[field]=[{'key':list(key),'size':value.get('size'),
                            'job_ids':[job.id for job in value.get('jobs',())]}
                            for key,value in engine.pending_cohorts.items()]
                    else:
                        states[field]=[{'id':job.id,'uid':job.uid,
                            'cancelled':job.cancelled.is_set(),'completion_tokens':job.completion_tokens,
                            'effective_max_tokens':job.effective_max_tokens,
                            'native_attached':getattr(job,'native_paged_receipt',None) is not None}
                            for job in engine.jobs.values()]
                finally:lock.release()
        states['prompt_lock_locked']=engine.prompt_lock.locked()
        states['incoming_qsize']=engine.incoming.qsize()
    frames=sys._current_frames();names={thread.ident:thread.name for thread in threading.enumerate()}
    states['thread_stacks']=[{'ident':ident,'name':names.get(ident),
        'stack':traceback.format_stack(frame,limit=24)} for ident,frame in frames.items()]
    return states


def pair(base,bodies,engine=None):
    workers=ThreadPoolExecutor(max_workers=2)
    try:
        first=workers.submit(post,base,bodies[0])
        if engine is not None:
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                # A wedged submission lock must not wedge the diagnostic caller.
                acquired=engine.submission_lock.acquire(blocking=False)
                if acquired:
                    try:
                        staged=engine.pending_cohorts.get(('default','hybrid-http-gate'))
                        ready=staged is not None and len(staged['jobs'])==1
                    finally:engine.submission_lock.release()
                    if ready:break
                time.sleep(.002)
            else:raise RuntimeError('first HTTP cohort member did not stage')
        second=workers.submit(post,base,bodies[1])
        deadline=time.monotonic()+45
        return tuple(future.result(timeout=max(.001,deadline-time.monotonic())) for future in (first,second))
    finally:
        # HTTP sockets have their own40s timeout; do not add executor joining delay.
        workers.shutdown(wait=False,cancel_futures=True)


def bounded_cleanup(engine,server,server_thread,captures,result):
    cleanup={'server_stopped':True,'engine_closed':engine is None,'retained':False}
    if server is not None:
        stopper=threading.Thread(target=server.shutdown,daemon=True,name='http-gate-server-close')
        stopper.start();stopper.join(timeout=2)
        cleanup['server_stopped']=not stopper.is_alive()
        server.server_close()
    if server_thread is not None:server_thread.join(timeout=1)
    if engine is not None:
        # close() itself may wait30s; bound this runner's cleanup to5s.
        closer=threading.Thread(target=engine.close,daemon=True,name='http-gate-engine-close')
        closer.start();closer.join(timeout=5)
        cleanup['engine_closed']=not closer.is_alive() and not engine.thread.is_alive()
    cleanup['retained']=not cleanup['engine_closed'] or any(
        not candidate._serving_resources.closed for _,candidate,_ in captures)
    if cleanup['retained']:FAILURE_ROOTS.append((engine,captures))
    result['bounded_cleanup']=cleanup
    return cleanup


def exact_prompts(adapter):
    # Text-first, complete lexical words: never truncate a token's byte sequence.
    words=('Explain database isolation causality durability and distributed consistency '
           'transaction ordering replication consensus recovery atomicity availability').split()
    suffixes=('.', '?', ' in detail.', ' now.', ' please.', ' briefly.',
              ' safely.', ' precisely.', ' today.')
    result=[]
    for target in (32,96):
        found=None
        for count in range(1,257):
            prefix=' '.join(words[index % len(words)] for index in range(count))
            for suffix in suffixes:
                text=prefix+suffix
                ids=tuple(adapter.prompt_tokens({'prompt':text}))
                if len(ids)!=target:continue
                decoded=adapter.tokenizer.decode(list(ids))
                if tuple(adapter.prompt_tokens({'prompt':decoded}))!=ids:continue
                found=(text,ids);break
            if found is not None:break
        if found is None:raise RuntimeError(f'no stable text-first {target}-token HTTP prompt')
        result.append(found)
    return tuple(result)


def validate_evaluation_proof(proof,block):
    if type(block) is not int or block not in (1,4,16):
        raise RuntimeError('unsupported HTTP evaluation block')
    if (type(proof.get('prefill_eval_block_size')) is not int or proof.get('prefill_eval_block_size')!=block or
            proof.get('native_reader_simultaneous_scratch_bytes')!=1585152*min(block,16) or
            proof.get('bootstrap_charge_components',{}).get('native_reader_bytes')!=1585152*min(block,16)):
        raise RuntimeError('actual HTTP evaluation block/scratch charge differs from selected profile')


def execute(args,result):
    from varlen_pack_price_bench import _gpuq_owner
    result['gpuq_owner']=_gpuq_owner()
    if hashlib.sha256(Path(args.native).read_bytes()).hexdigest()!=args.native_sha256:
        raise RuntimeError('frozen native hash differs')
    sys.path.insert(0,str(Path(args.native).parent));sys.path.insert(0,str(ROOT/'src'))
    from mlx2.runtime.paged_price_identity import cached_live_price_identity
    identity=cached_live_price_identity(Path(args.artifact_manifest),Path(args.mlx_wheel),Path(args.native),
        adapter_artifact_root=Path(args.model).resolve())
    result.update(identity=identity,native_sha256=args.native_sha256)
    from mlx2.runtime.paged_packed_prefill_serving_profile import make_profile,load_profile
    profile=make_profile(identity,prefill_eval_block_size=args.prefill_eval_block_size)
    if args.prepare_profile:
        save(args.profile,profile);result['status']='profile_prepared';return
    os.environ.update(profile['required_environment'])
    profile=load_profile(args.profile,live_identity=identity,context_lengths=(32,96),environment=os.environ)
    if profile.get('prefill_eval_block_size',1)!=args.prefill_eval_block_size:
        raise RuntimeError('HTTP evaluation block selector differs from pinned profile')
    os.environ.update(MLX2_NATIVE_PACKED_PREFILL_B2_PROFILE=str(args.profile),
        MLX2_NATIVE_PAGED_MANIFEST=args.artifact_manifest,MLX2_NATIVE_PAGED_MLX_WHEEL=args.mlx_wheel)
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter,configure_environment
    configure_environment()
    import mlx.core as mx
    from mlx2 import serving
    from mlx2.server import handler_for
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime import qwen35_paged_graph_factory as factory
    result.update(gpu_executed=True,status='running',sampled_events=[])
    sampled={};effective={};captures=[];final_physical={}
    engine=server=server_thread=None
    watchdog_stop=threading.Event();watchdog=None
    diagnostic_path=args.output.with_name(args.output.name+'.diagnostics.json')
    result['diagnostic_path']=str(diagnostic_path)
    def checkpoint(phase):
        result['phase']=phase;save(args.output,result)
    def watchdog_loop():
        while not watchdog_stop.wait(10):
            try:
                state=host_diagnostics(engine,result.get('phase'),captures)
                save(diagnostic_path,state)
            except BaseException as error:
                save(diagnostic_path,{'diagnostic_error':repr(error),'phase':result.get('phase')})
    original_next=BatchGenerator.next
    from mlx2.runtime import hybrid_packed_prefill as packed_factory
    original_factory=packed_factory.create_cold_packed_hybrid
    initial_charge=factory._CHARGED
    def observed_next(batch,*a,**kw):
        prompts,responses=original_next(batch,*a,**kw)
        for response in responses:
            job=next((job for job in engine.jobs.values() if job.uid==response.uid),None)
            if job is None:raise RuntimeError('actual HTTP sampler response lacks live Job')
            receipt=dict(response.mtp_receipt or {})
            sampled.setdefault(job.id,[]).append(int(response.token))
            effective[job.id]=dict(job.effective_sampling or {})
            result['sampled_events'].append({'id':job.id,'uid':response.uid,'token':int(response.token),
                'execution_width':getattr(response,'execution_width',1),'route_receipt':receipt})
            if (args.cancel_after_two and job.request.get('paged_native_hybrid_packed_prefill') is True and
                    len(job.native_b2_prompt)==32 and len(sampled[job.id])==2):
                job.cancelled.set();result['cancelled_job_id']=job.id
        return prompts,responses
    def captured_factory(*a,**kw):
        checkpoint('native_factory_enter')
        owners,candidate,bootstrap=original_factory(*a,**kw)
        checkpoint('native_factory_return')
        captures.append((owners,candidate,bootstrap))
        arena=candidate.backend.writer.backend;original_close=arena.close_after_terminal
        def observed_close():
            if not arena._closed:
                final_physical.update(candidate.backend.profile_counters_snapshot())
                final_physical['packed_native_counters']={name:int(getattr(arena._native,name)(arena._arena))
                    for name in candidate._packed_prefill_receipt['physical_counters']}
            return original_close()
        arena.close_after_terminal=observed_close
        return owners,candidate,bootstrap
    BatchGenerator.next=observed_next;packed_factory.create_cold_packed_hybrid=captured_factory
    started=time.perf_counter()
    watchdog=threading.Thread(target=watchdog_loop,daemon=True,name='http-gate-watchdog');watchdog.start()
    try:
        checkpoint('engine_load')
        engine=serving.ServingEngine(args.model,adapter_factory=Qwen3827BAdapter,max_lanes=2,
            max_inflight=2,mtp=False,prompt_lookup=False,qualification_mode=False,
            prefill_step=128,max_context=128,cache_bytes=1<<29,batch_cohort_timeout_ms=200)
        if not engine.ready.wait(40) or engine.error:raise RuntimeError(f'HTTP engine load failed: {engine.error}')
        result['load_seconds']=time.perf_counter()-started
        if cached_live_price_identity(Path(args.artifact_manifest),Path(args.mlx_wheel),Path(args.native),
                adapter_artifact_root=Path(engine.adapter.identity['path']).resolve())!=identity:
            raise RuntimeError('loaded HTTP artifact differs')
        prompts=exact_prompts(engine.adapter)
        result['prompt_token_ids']=[list(ids) for _,ids in prompts]
        result['prompt_sha256']=[hashlib.sha256(text.encode()).hexdigest() for text,_ in prompts]
        common=tuple({'model':Path(args.model).name,'prompt':text,'max_tokens':cap,'temperature':0,
            'repetition_penalty':1,'presence_penalty':0,'frequency_penalty':0,
            'skip_writing_prefix_cache':True} for (text,_),cap in zip(prompts,((4,4) if args.cancel_after_two else (2,4))))
        native_bodies=tuple({**body,'paged_native_hybrid_b2':True,'paged_native_hybrid_packed_prefill':True,
            'batch_cohort':{'id':'hybrid-http-gate','size':2}} for body in common)
        server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(engine))
        server.daemon_threads=True;server.block_on_close=False
        server_thread=threading.Thread(target=server.serve_forever,daemon=True);server_thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        checkpoint('native_http_pair')
        tick=time.perf_counter();native=pair(base,native_bodies,engine)
        result['native_http_pair_seconds']=time.perf_counter()-tick
        result['native_http_responses']=[{'status':status,'body':body} for status,body in native];save(args.output,result)
        if not args.cancel_after_two and any(status!=200 for status,_ in native):raise RuntimeError('native hybrid HTTP failed')
        native_summary=([None,summarize_http(native[1][1],4,True,True)] if args.cancel_after_two else
            [summarize_http(body,cap,True,True) for (_,body),cap in zip(native,(2,4))])
        if len(captures)!=1:raise RuntimeError('HTTP cohort did not allocate exactly one native graph')
        owners,candidate,bootstrap=captures[0];writer=candidate.backend.writer
        deadline=time.monotonic()+5
        while not candidate._serving_resources.closed and time.monotonic()<deadline:
            factory.reap_hybrid_admission_orphans();candidate._serving_resources.reap();time.sleep(.005)
        cleanup={'pending_epochs':len(writer.pending_epochs),'pending_ledger':writer.ledger.pending_count,
            'allocated_pages':writer.pool.allocated_count,'owners_retired':all(owner.fully_retired for owner in owners),
            'resources_closed':candidate._serving_resources.closed,'global_charge_bytes':factory._CHARGED}
        result['native_cleanup']=cleanup;result['physical_before_close']=final_physical
        if (cleanup['pending_epochs'] or cleanup['pending_ledger'] or cleanup['allocated_pages'] or
                not cleanup['owners_retired'] or not cleanup['resources_closed'] or factory._CHARGED!=initial_charge):
            raise RuntimeError('HTTP native resources retained')
        validate_physical(final_physical,candidate.native_layer_count,True)
        proof=candidate._packed_prefill_receipt
        expected={'grouped_multirow_write_count':16,'grouped_multirow_row_count':2048,'prefill_matrix_dispatch_count':16,
            **{'prefill_nax_'+stage+'_dispatch_count':16 for stage in ('score','softmax','value')}}
        if proof['physical_counters']!=expected or final_physical.get('packed_native_counters')!=expected:
            raise RuntimeError('actual HTTP packed NAX physical stages/rows differ')
        validate_evaluation_proof(proof,args.prefill_eval_block_size)
        result['packed_prefill_proof']=proof
        native_ids=([sampled.get(result.get('cancelled_job_id')),sampled.get(native[1][1].get('id'))] if args.cancel_after_two else
            [sampled.get(body['id']) for _,body in native])
        if [len(ids or ()) for ids in native_ids]!=[2,4]:raise RuntimeError('actual native sampler output count differs')
        if not args.cancel_after_two and [summary['route_receipt']['output_token_ids'] for summary in native_summary]!=native_ids:
            raise RuntimeError('final HTTP IDs differ from actual sampled IDs')
        survivor=[event for event in result['sampled_events'] if event['id']==native[1][1]['id']][-2:]
        if len(survivor)!=2 or any(event['execution_width']!=1 for event in survivor):raise RuntimeError('HTTP B1 survivor not observed')
        if args.cancel_after_two:
            if result.get('cancelled_job_id') is None:raise RuntimeError('actual cancellation was not requested')
            if native[1][0]!=200:raise RuntimeError('cancelled cohort survivor HTTP failed')
            result.update(status='cancellation_lifecycle_passed',token_parity='not_tested',numeric_tensor_parity='not_tested',
                cancellation_scope='actual engine Job.cancelled after two sampled lane0 tokens; live HTTP survivor completes4',
                native_output_token_ids=native_ids,HTTP_native_receipts=[native_summary[1]['route_receipt']],
                total_seconds=time.perf_counter()-started)
            return
        checkpoint('ordinary_http_pair')
        tick=time.perf_counter();ordinary=pair(base,common)
        result['ordinary_http_pair_seconds']=time.perf_counter()-tick
        result['ordinary_http_responses']=[{'status':status,'body':body} for status,body in ordinary];save(args.output,result)
        if any(status!=200 for status,_ in ordinary):raise RuntimeError('ordinary HTTP reference failed')
        ordinary_summary=[summarize_http(body,cap,False) for (_,body),cap in zip(ordinary,(2,4))]
        ordinary_ids=[sampled.get(body['id']) for _,body in ordinary]
        if [len(ids or ()) for ids in ordinary_ids]!=[2,4]:raise RuntimeError('actual ordinary sampler output count differs')
        native_sampling=[effective.get(body['id']) for _,body in native]
        ordinary_sampling=[effective.get(body['id']) for _,body in ordinary]
        if native_sampling!=ordinary_sampling:raise RuntimeError('native/ordinary effective sampling differs')
        if any(values is None or values.get('temperature')!=0 or values.get('repetition_penalty')!=1 or
               values.get('presence_penalty')!=0 or values.get('frequency_penalty')!=0 for values in native_sampling):
            raise RuntimeError('actual effective greedy controls differ')
        if len(captures)!=1:raise RuntimeError('ordinary default allocated native arena')
        result.update(native_output_token_ids=native_ids,ordinary_output_token_ids=ordinary_ids,
            effective_sampling=native_sampling,HTTP_native_receipts=[s['route_receipt'] for s in native_summary])
        if native_ids!=ordinary_ids or [s['text'] for s in native_summary]!=[s['text'] for s in ordinary_summary]:
            result['token_parity']='failed';raise RuntimeError('HTTP native/ordinary sampled token or text drift')
        result.update(status='passed',token_parity='passed',numeric_tensor_parity='not_tested',
            qualified=False,price_usable=False,total_seconds=time.perf_counter()-started)
    except BaseException:
        result['failure_diagnostics']=host_diagnostics(engine,result.get('phase'),captures)
        save(args.output,result)
        raise
    finally:
        watchdog_stop.set()
        if watchdog is not None:watchdog.join(timeout=1)
        cleanup=bounded_cleanup(engine,server,server_thread,captures,result)
        BatchGenerator.next=original_next;packed_factory.create_cold_packed_hybrid=original_factory
        if not cleanup['engine_closed'] and result.get('status') in ('passed','cancellation_lifecycle_passed'):
            result['status']='failed';raise RuntimeError('serving thread did not close within bounded cleanup')


def supervise(args,result):
    process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],start_new_session=True)
    began=time.monotonic();peak_rss=0
    try:
        while process.poll() is None:
            if time.monotonic()-began>MAX_SECONDS:raise TimeoutError('100 second HTTP worker deadline')
            try:rss=int(subprocess.check_output(['ps','-o','rss=','-p',str(process.pid)],text=True).strip() or '0')*1024
            except (subprocess.CalledProcessError,ValueError):rss=0
            peak_rss=max(peak_rss,rss)
            if rss>MAX_RSS:raise MemoryError('48 GiB HTTP worker RSS ceiling')
            time.sleep(.2)
        if args.output.is_file():
            result=json.loads(args.output.read_text())
            diagnostic_path=args.output.with_name(args.output.name+'.diagnostics.json')
            if diagnostic_path.is_file():result['last_watchdog']=json.loads(diagnostic_path.read_text())
            result.update(supervisor_peak_rss_bytes=peak_rss,supervisor_seconds=time.monotonic()-began)
            save(args.output,result)
        return process.returncode
    except BaseException as error:
        os.killpg(process.pid,signal.SIGKILL);process.wait()
        if args.output.is_file():result=json.loads(args.output.read_text())
        diagnostic_path=args.output.with_name(args.output.name+'.diagnostics.json')
        if diagnostic_path.is_file():result['last_watchdog']=json.loads(diagnostic_path.read_text())
        result.update(status='failed',error=f'{type(error).__name__}: {error}',worker_killed=True,qualified=False)
        save(args.output,result);return 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=MODEL);parser.add_argument('--artifact-manifest',default=MANIFEST)
    parser.add_argument('--mlx-wheel',default=WHEEL);parser.add_argument('--native',default=NATIVE)
    parser.add_argument('--native-sha256',default=NATIVE_SHA)
    parser.add_argument('--profile',type=Path,default=Path('/tmp/mlx2-packed-prefill-http-profile-1004.json'))
    parser.add_argument('--prefill-eval-block-size',type=int,choices=(1,4,16),default=1,
        help='explicit NAX prefill evaluation block; default1 retains eager serving')
    parser.add_argument('--output',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--prepare-profile',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.set_defaults(stock_singleton=True)
    parser.add_argument('--cancel-after-two',action='store_true',help='separate actual HTTP cohort cancellation lifecycle arm, caps4/4')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();result=preflight();result['prefill_eval_block_size']=args.prefill_eval_block_size;result['stock_singleton']=args.stock_singleton;result['cancel_after_two']=args.cancel_after_two
    if (args.prepare_profile or args.execute) and not args.worker:return supervise(args,result)
    code=0
    try:
        if args.prepare_profile or args.execute:execute(args,result)
    except BaseException as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}',retained_failure_roots=len(FAILURE_ROOTS));code=1
    save(args.output,result);return code

if __name__=='__main__':raise SystemExit(main())
