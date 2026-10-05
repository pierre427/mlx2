"""Root-controlled actual400-case B20 suite; terminal-proved lease quantums.

One model/service, 20 distinct requests per domain, original max192/EOS.
Every init/start/resume/cancel/shutdown requires a fresh verified dual lease.
No GPU is used by importing or planning this research driver.
"""
from __future__ import annotations
import argparse,gc,hashlib,json,os,secrets,socket,stat,subprocess,sys,threading,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from spomin_400case_native_suite import ROOT,validate_inputs,domain_rows,request_body,save
from long_b2_cap20_http_smoke import verified_lease
from varlen_hybrid_serving_smoke import MODEL,MANIFEST,WHEEL
from varlen_hybrid_packed_prefill_long_http_gate import bounded_cleanup
sys.path.insert(0,str(ROOT/'src'))
from mlx2.runtime.paged_n20_phase_control import PhaseController
MAX_SECONDS=100;MAX_RSS=48<<30
NATIVE='/tmp/mlx2-n20-q1-b1-stock-build-1004/_paged_kv_native.cpython-312-darwin.so'
NATIVE_SHA='2ddf5b91a88bbf8f451c1eaaae9919bdff1e6effd161262b7df61c4e5d2cf350'
FAILURE_ROOTS=[]

def collect_client_results(futures):
    results=[]
    for future in futures:
        try:results.append(future.result(timeout=1800))
        except BaseException as error:results.append((0,{'client_error':repr(error)}))
    return results

def post_case(request, opener=urlopen):
    try:
        with opener(request,timeout=1800) as response:return response.status,json.load(response)
    except HTTPError as error:
        raw=error.read(1<<20)
        try:body=json.loads(raw)
        except (ValueError,UnicodeDecodeError):body={'raw_error_body':raw.decode('utf-8',errors='replace')}
        return error.code,body
    except BaseException as error:return 0,{'client_error':repr(error)}

def wait_engine_ready(engine, timeout=40):
    """Wake on worker failure rather than waiting for a ready flag never set."""
    deadline=time.monotonic()+timeout
    while True:
        if engine.error:raise RuntimeError('single model engine load failed: '+str(engine.error))
        if engine.ready.is_set():return
        remaining=deadline-time.monotonic()
        if remaining<=0:raise RuntimeError('single model engine load timed out')
        engine.ready.wait(min(.05,remaining))

def plan():
    return dict(schema='mlx2.spomin400-phased-http.v1',status='planned',gpu_executed=False,
        domains=20,cases_per_domain=20,requests_per_arm=400,client_concurrency=20,max_tokens=192,
        native_max_compute_width=20,dynamic_eos_cancel_shrink=True,qualified=False,price_usable=False,
        max_rss_bytes=MAX_RSS,hard_seconds_per_quantum=MAX_SECONDS,
        rate_scope='actual event wall includes lease pauses; owned quantum compute and pauses separately; no raw kernel rate')

def score(row,text):
    from run_spomin_20x20 import concept_hits
    hits,total=concept_hits(text,row['expected_concepts'])
    return dict(sentinel_present=row['sentinel'].lower() in text.lower(),
        needles_present={k:v.lower() in text.lower() for k,v in row['needles'].items()},concept_hits=hits,concept_groups=total)

def summarize(rows,outputs,events,starts,ended,*,owned_starts=None,owned_ended=None):
    result=[]
    for row,(status,body) in zip(rows,outputs):
        case=row['case_id'];usage=body.get('usage',{});choices=body.get('choices',[]);actual=usage.get('completion_tokens');samples=events.get(case,[])
        if status!=200 or type(actual) is not int or not 1<=actual<=192 or len(choices)!=1 or choices[0].get('finish_reason') not in ('stop','length') or (choices[0]['finish_reason']=='length' and actual!=192) or len(samples)!=actual or usage.get('prompt_tokens')!=row['prompt_tokens'] or usage.get('prompt_tokens_details',{}).get('cached_tokens',0)!=0:
            raise RuntimeError('actual case usage/finish/cold/sample count differs: '+case)
        first=samples[0]['monotonic_ns']/1e9;last=samples[-1]['monotonic_ns']/1e9;text=choices[0].get('message',{}).get('content','')
        result.append(dict(case_id=case,domain=row['domain'],body_sha256=row['body_sha256'],text=text,finish_reason=choices[0]['finish_reason'],usage=usage,
            sample_events=samples,output_token_ids=[s['token'] for s in samples],ttft_wall_seconds=first-starts[case],decode_first_to_last_wall_seconds=last-first,
            decode_tokens_per_second=(actual-1)/(last-first) if actual>1 and last>first else None,
            route_receipt=body.get('mlx2',{}).get('route_receipt',{}),**score(row,text)))
    if owned_starts is not None:
        for item in result:
            case=item['case_id'];samples=item['sample_events']
            a=samples[0]['owned_elapsed_seconds'];b=samples[-1]['owned_elapsed_seconds']
            item.update(ttft_owned_elapsed_seconds=a-owned_starts[case],decode_first_to_last_owned_elapsed_seconds=b-a,
                        decode_tokens_per_owned_elapsed_second=(len(samples)-1)/(b-a) if len(samples)>1 and b>a else None)
    first=min(r['sample_events'][0]['monotonic_ns'] for r in result)/1e9;last=max(r['sample_events'][-1]['monotonic_ns'] for r in result)/1e9
    beginning=min(starts.values());prefill=max(r['sample_events'][0]['monotonic_ns'] for r in result)/1e9-beginning
    summary=dict(rows=result,actual_completion_tokens=sum(len(r['sample_events']) for r in result),
        http_domain_wall_seconds=ended-beginning,client_start_spread_seconds=max(starts.values())-beginning,
        prefill_through_all_first_samples_wall_seconds=prefill,prefill_prompt_tokens_per_second=sum(r['prompt_tokens'] for r in rows)/prefill,
        complete_output_tokens_per_second=sum(len(r['sample_events']) for r in result)/(ended-beginning),
        aggregate_decode_tokens_per_second=sum(max(0,len(r['sample_events'])-1) for r in result)/(last-first) if last>first else None,
        observed_compute_widths=sorted({int(event['width']) for r in result for event in r['sample_events']}),
        observed_peak_compute_width=max(int(event['width']) for r in result for event in r['sample_events']),
        all_sentinels=all(r['sentinel_present'] for r in result),all_needles=all(all(r['needles_present'].values()) for r in result),
        concept_hits=sum(r['concept_hits'] for r in result),concept_groups=sum(r['concept_groups'] for r in result))
    if owned_starts is not None:
        origin=min(owned_starts.values());first_owned=min(r['sample_events'][0]['owned_elapsed_seconds'] for r in result)
        last_owned=max(r['sample_events'][-1]['owned_elapsed_seconds'] for r in result)
        prefill_owned=max(r['sample_events'][0]['owned_elapsed_seconds'] for r in result)-origin
        complete_owned=owned_ended-origin
        if prefill_owned<=0 or complete_owned<=0:raise RuntimeError('owned elapsed clock ordering differs')
        summary.update(prefill_through_all_first_samples_owned_elapsed_seconds=prefill_owned,
            prefill_prompt_tokens_per_owned_elapsed_second=sum(r['prompt_tokens'] for r in rows)/prefill_owned,
            aggregate_decode_owned_elapsed_seconds=last_owned-first_owned,
            aggregate_decode_tokens_per_owned_elapsed_second=sum(max(0,len(r['sample_events'])-1) for r in result)/(last_owned-first_owned) if last_owned>first_owned else None,
            complete_owned_elapsed_seconds=complete_owned,complete_output_tokens_per_owned_elapsed_second=sum(len(r['sample_events']) for r in result)/complete_owned,
            owned_rate_scope='host elapsed while lease active, excludes operator pauses; includes CPU and GPU completion; not device-only')
    return summary

class Service:
    def __init__(self,args):
        self.args=args;self.result=plan();self.engine=self.server=self.server_thread=None;self.current=None;self.controller=None
        self.events={};self.starts={};self.captures=[];self.restores=[];self.cells=[];self.seen_leases=set();self.initialized=False;self.closed=False
    def verify(self,owner):
        from varlen_pack_price_bench import _gpuq_owner
        if _gpuq_owner()!=owner:raise RuntimeError('both GPU lock owners changed before private work')
    def init(self):
        if self.initialized:raise RuntimeError('one model init only')
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=self.args.expected_source or subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise RuntimeError('source must be frozen clean')
        if hashlib.sha256(self.args.native.read_bytes()).hexdigest()!=self.args.native_sha256:raise RuntimeError('binary source binding differs')
        self.inputs=validate_inputs(json.loads(self.args.inputs.read_text()))
        sys.path.insert(0,str(self.args.native.parent))
        # These helpers are stdlib-only. Pin the complete numeric environment
        # before adapter/generator/resource imports can snapshot model modules.
        from mlx2.runtime import hybrid_packed_prefill_n as factory
        from mlx2.runtime.paged_price_identity import cached_live_price_identity
        os.environ.update(factory.environment(gdn_eval_wave_max_segments=self.args.gdn_eval_wave_max_segments))
        import _paged_kv_native as native
        native_preflight=factory.preflight_native_inputs(native,self.inputs)
        identity=cached_live_price_identity(self.args.artifact_manifest,self.args.mlx_wheel,self.args.native,adapter_artifact_root=Path(self.args.model).resolve())
        profile=factory.make_profile(identity,self.inputs,gdn_eval_wave_max_segments=self.args.gdn_eval_wave_max_segments);save(self.args.profile,profile)
        os.environ.update(profile['required_environment'])
        os.environ.update(MLX2_NATIVE_PACKED_PREFILL_N20_PROFILE=str(self.args.profile),MLX2_NATIVE_PAGED_MANIFEST=str(self.args.artifact_manifest),MLX2_NATIVE_PAGED_MLX_WHEEL=str(self.args.mlx_wheel))
        from mlx2.adapters.qwen38_27b import Qwen3827BAdapter,configure_environment
        configure_environment()
        import mlx.core as mx
        from mlx2.runtime import qwen35_paged_graph_factory as resources
        from mlx2.runtime.generate import BatchGenerator
        from mlx2 import serving
        from mlx2.server import handler_for,validate_request
        self.validate_http_request=validate_request;self.profile=profile
        self.mx=mx;self.previous_cache_limit=mx.set_cache_limit(0);mx.clear_cache();self.resources=resources;self.initial_charge=resources._CHARGED
        original_next=BatchGenerator.next;original_factory=factory.create_cold_packed_hybrid_n
        def observed_next(batch,*a,**kw):
            active=self.current is not None
            if active:self.controller.enter()
            prompts,responses=original_next(batch,*a,**kw)
            if active:
                mx.synchronize()
                for response in responses:
                    job=next((j for j in self.engine.jobs.values() if j.uid==response.uid),None)
                    if job is None:raise RuntimeError('sample event lacks live case Job')
                    case=job.request.get('native_research_input_id')
                    if case not in self.starts:raise RuntimeError('sample case binding missing')
                    self.events.setdefault(case,[]).append(dict(token=int(response.token),**self.controller.sample_clock(),source_commit=self.args.expected_source,width=getattr(response,'execution_width',1)))
                for capture in self.captures:
                    backend=capture['candidate'].backend;writer=backend.writer
                    if writer.pending_epochs or writer.ledger.pending_count or backend._orphaned_reads:raise RuntimeError('cannot pause nonterminal native decode')
                self.controller.boundary(dict(layer_kind='generator',materialized=True,native_terminals_drained=True,public_state_published=True,real_response_count=len(responses)))
            return prompts,responses
        def captured_factory(*a,**kw):
            owners,candidate,boots=original_factory(*a,**kw);self.captures.append(dict(owners=owners,candidate=candidate));return owners,candidate,boots
        BatchGenerator.next=observed_next;factory.create_cold_packed_hybrid_n=captured_factory
        self.restores=[(BatchGenerator,'next',original_next),(factory,'create_cold_packed_hybrid_n',original_factory)]
        self.engine=serving.ServingEngine(self.args.model,adapter_factory=Qwen3827BAdapter,max_lanes=20,max_inflight=20,mtp=False,prompt_lookup=False,qualification_mode=False,prefill_step=self.args.prefill_step,max_context=8192,cache_bytes=1<<29,batch_cohort_timeout_ms=5000)
        wait_engine_ready(self.engine)
        if cached_live_price_identity(self.args.artifact_manifest,self.args.mlx_wheel,self.args.native,adapter_artifact_root=Path(self.engine.adapter.identity['path']).resolve())!=identity:raise RuntimeError('loaded identity changed')
        def phase_boundary(event):
            telemetry=dict(event,mlx_active_bytes=int(mx.get_active_memory()),mlx_peak_bytes=int(mx.get_peak_memory()),mlx_cache_bytes=int(mx.get_cache_memory()),mlx_cache_limit_bytes=0)
            if telemetry['mlx_active_bytes']+telemetry['mlx_cache_bytes']>MAX_RSS:raise MemoryError('MLX allocator active/cache guard exceeds48GiB')
            self.controller.boundary(telemetry)
        self.engine.adapter._native_n20_phase_boundary=phase_boundary
        self.server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(self.engine));self.server.daemon_threads=True;self.server.block_on_close=False
        self.server_thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.server_thread.start();self.base='http://127.0.0.1:'+str(self.server.server_port)
        self.initialized=True;self.result.update(status='idle',gpu_executed=True,identity=identity,inputs_sha256=self.inputs['inputs_sha256'],input_artifact_file_sha256=hashlib.sha256(self.args.inputs.read_bytes()).hexdigest(),root_prepared_manifest_validated=True,native_preload_capability_proof=native_preflight,model_loads=1,engine_instances=1,gdn_eval_wave_max_segments=self.args.gdn_eval_wave_max_segments,ordinary_control_admission='original_independent_requests',native_admission='declared_atomic20_cohort',prefill_step_override=self.args.prefill_step,prefill_step_selected=self.engine.prefill_step,prefill_step_source=self.engine.prefill_step_source)
        return dict(initialized=True,source=identity['source_commit'])
    def start(self,command,owner):
        if not self.initialized or self.current is not None:raise RuntimeError('one domain at a time after init')
        index=command.get('domain_index');arm=command.get('arm')
        if arm not in ('native','ordinary'):raise ValueError('explicit native/ordinary arm required')
        rows=domain_rows(self.inputs,index)
        if any(c['domain_index']==index and c['arm']==arm and not c['warmup'] for c in self.cells) and not command.get('warmup',False):raise ValueError('duplicate measured actual domain arm')
        bodies={}
        for row in rows:
            body=request_body(row,model=Path(self.args.model).name,native=arm=='native',cohort_id='spomin400-'+str(index)+'-'+arm,inputs_sha256=self.inputs['inputs_sha256'])
            body.update(native_research_input_id=row['case_id'],native_research_inputs_sha256=self.inputs['inputs_sha256'])
            bodies[row['case_id']]=self.validate_http_request(body)
            if self.profile['inputs_sha256']!=body['native_research_inputs_sha256'] or row['case_id'] not in self.profile['source_inputs']:
                raise RuntimeError('preflight actual profile/case binding differs')
        self.current=dict(domain_index=index,arm=arm,warmup=command.get('warmup',False));self.rows=rows;self.events={};self.starts={};self.owned_starts={};self.outputs=None;self.domain_error=None
        self.controller=PhaseController(self.verify);self.controller.grant(owner,command.get('phases',4))
        barrier=threading.Barrier(21);pool=ThreadPoolExecutor(max_workers=20);self.pool=pool
        def one(row):
            barrier.wait(timeout=30);clock=self.controller.sample_clock();self.starts[row['case_id']]=clock['monotonic_ns']/1e9;self.owned_starts[row['case_id']]=clock['owned_elapsed_seconds']
            body=bodies[row['case_id']]
            request=Request(self.base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            return post_case(request)
        futures=[pool.submit(one,row) for row in rows];self.domain_futures=futures;barrier.wait(timeout=30)
        def completion():
            try:
                self.outputs=collect_client_results(futures)
                failures=[dict(case_id=row['case_id'],status=status,error_body=body) for row,(status,body) in zip(rows,self.outputs) if status!=200]
                if failures:
                    self.result['client_failures']=failures
                    self.domain_error=json.dumps(failures)
            except BaseException as error:self.domain_error=repr(error)
            finally:
                # HTTP completion may arrive while the generator waits at a
                # fully evaluated boundary. Root must resume/cancel to unload.
                clock=self.controller.sample_clock();self.domain_http_ended=clock['monotonic_ns']/1e9;self.domain_owned_ended=clock['owned_elapsed_seconds'];self.controller.wake()
        self.completion_thread=threading.Thread(target=completion,daemon=True);self.completion_thread.start()
        return self.progress()
    def progress(self):
        state=self.controller.wait_quantum(90,completion=lambda:self.outputs is not None or self.domain_error is not None)
        if self.domain_error:raise RuntimeError(self.domain_error)
        if self.outputs is not None:
            # All responses have left product state. No new graph work is
            # launched during the following host retirement proof.
            cell=summarize(self.rows,self.outputs,self.events,self.starts,self.domain_http_ended,owned_starts=self.owned_starts,owned_ended=self.domain_owned_ended)
            if cell['client_start_spread_seconds']>.25:raise RuntimeError('original B20 start spread bound failed')
            if self.current['arm']=='native':
                if len(self.captures)!=1 or any(r['route_receipt'].get('route')!='native_hybrid_packed_n20_research' or r['route_receipt'].get('prefill_cohort_width')!=20 for r in cell['rows']):raise RuntimeError('genuine twenty lane native admission not observed')
                for capture in self.captures:
                    candidate=capture['candidate'];writer=candidate.backend.writer;candidate._serving_resources.reap()
                    if writer.pending_epochs or writer.ledger.pending_count or writer.pool.allocated_count or not candidate._serving_resources.closed:raise RuntimeError('domain native owner/resource retirement incomplete')
            if self.resources._CHARGED!=self.initial_charge:raise RuntimeError('domain memory charge not restored')
            idle=dict(jobs=len(self.engine.jobs),pending_cohorts=len(self.engine.pending_cohorts),queued_jobs=self.engine.queued_jobs,incoming=self.engine.incoming.qsize())
            if any(idle.values()):raise RuntimeError('HTTP completion did not retire product queue/slots')
            self.controller.seal_completed(self.controller.owner,dict(layer_kind='domain_complete',materialized=True,native_terminals_drained=True,public_state_published=True,product_idle=True))
            state=self.controller.wait_quantum(1)
            # Complete request wall contains deliberate lease gaps. Do not
            # relabel owned phase time as HTTP/decode wall.
            cell.update(self.current,owned_quantum_compute_seconds=state['owned_compute_seconds'],paused_wait_seconds=state['paused_wait_seconds'],phases=list(self.controller.events),idle_queue_proof=dict(native_charge_restored=True,**idle))
            self.cells.append(cell);self.result.update(status='idle',cells=self.cells)
            self.captures.clear();self.current=None;self.controller.release_completed(self.controller.owner);self.pool.shutdown(wait=False);gc.collect();self.mx.clear_cache()
            return dict(domain_complete=True,cell=cell)
        return dict(domain_complete=False,**state)
    def resume(self,command,owner):
        if self.current is None:raise RuntimeError('no private domain pending')
        self.controller.grant(owner,command.get('phases',4));return self.progress()
    def seal_unstarted_failure(self,owner):
        """Seal failed HTTP work only after product AND native retirement proof."""
        if self.controller.paused:return
        self.verify(owner)
        if (not self.domain_error or self.outputs is None or
                not all(f.done() for f in self.domain_futures)):
            raise RuntimeError('failed active command lacks settled-client retirement proof')
        with self.engine.lock:
            idle=(not self.engine.jobs and not self.engine.pending_cohorts and not self.engine.queued_jobs and self.engine.incoming.empty())
            if not idle:raise RuntimeError('failed command still owns product admission state')
        # A failed post-bootstrap cohort can be retired truthfully; captured
        # roots alone do not imply ambiguity. Never close a live owner here.
        for capture in self.captures:
            if type(capture) is not dict or 'candidate' not in capture or 'owners' not in capture:
                raise RuntimeError('failed command capture lacks native ownership proof')
            candidate=capture['candidate'];writer=candidate.backend.writer
            resources=candidate._serving_resources
            if not resources.reap() or not resources.closed:
                raise RuntimeError('failed command native resource retirement incomplete')
            if (writer.pending_epochs or writer.ledger.pending_count or
                    candidate.backend._orphaned_reads or writer.pool.allocated_count or
                    any(not owner.fully_retired for owner in capture['owners'])):
                raise RuntimeError('failed command retains native owners or terminals')
        if self.resources._CHARGED!=self.initial_charge:
            raise RuntimeError('failed command still owns native charge')
        label='failed_domain_retirement' if self.captures or self.events else 'preadmission_failure'
        self.controller.seal_completed(owner,dict(layer_kind=label,materialized=True,native_terminals_drained=True,public_state_published=bool(self.captures),product_idle=True))
        self.result['failed_command_retirement_proof']=dict(product_idle=True,all_clients_settled=True,native_captures=len(self.captures),native_charge_restored=True,native_cleanup_proved=True)

    def cancel(self,owner):
        if self.current is None:raise RuntimeError('no pending domain')
        self.seal_unstarted_failure(owner)
        self.controller.cancel(owner)
        for job in tuple(self.engine.jobs.values()):job.cancelled.set()
        self.completion_thread.join(5)
        return dict(cancel_requested=True,shutdown_required=True,private_roots_retained=True)
    def shutdown(self,owner):
        if self.current is not None and not self.controller.aborted:
            self.seal_unstarted_failure(owner)
            self.controller.cancel(owner)
            for job in tuple(self.engine.jobs.values()):job.cancelled.set()
        captures=tuple((c['owners'],c['candidate'],None) for c in self.captures)
        cleanup=bounded_cleanup(self.engine,self.server,self.server_thread,captures,self.result)
        if hasattr(self,'resources'):
            deadline=time.monotonic()+5
            while self.resources._CHARGED!=self.initial_charge and time.monotonic()<deadline:
                self.resources.reap_hybrid_admission_orphans();time.sleep(.005)
            cleanup['native_charge_restored']=self.resources._CHARGED==self.initial_charge
            cleanup['retained']=cleanup['retained'] or not cleanup['native_charge_restored']
        for obj,name,value in reversed(self.restores):setattr(obj,name,value)
        if cleanup['retained'] or not cleanup['engine_closed']:FAILURE_ROOTS.append(self);raise RuntimeError('shutdown retained private native roots')
        try:
            measured=[c for c in self.cells if not c['warmup']];pairs=[]
            for index in range(20):
                pair={c['arm']:c for c in measured if c['domain_index']==index}
                if len(pair)!=2:continue
                left=pair['native']['rows'];right=pair['ordinary']['rows']
                if any(a['case_id']!=b['case_id'] or a['output_token_ids']!=b['output_token_ids'] or a['text']!=b['text'] or a['finish_reason']!=b['finish_reason'] for a,b in zip(left,right)):raise RuntimeError('actual matched400-case token/text/finish parity failed')
                pairs.append(dict(domain_index=index,exact_tokens=True))
            complete=len(measured)==40 and len(pairs)==20
            self.result.update(status='passed' if complete else 'incomplete',pairs=pairs,actual_requests_per_arm={arm:sum(len(c['rows']) for c in measured if c['arm']==arm) for arm in ('native','ordinary')},cleanup=cleanup)
            return dict(complete=complete,cleanup=cleanup)
        finally:
            if hasattr(self,'mx') and hasattr(self,'previous_cache_limit'):self.mx.set_cache_limit(self.previous_cache_limit)
            self.closed=True

def serve(args):
    if args.channel.exists() or args.auth_token_file.exists():raise RuntimeError('fresh private local paths required')
    token=secrets.token_hex(32);fd=os.open(args.auth_token_file,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as handle:handle.write(token)
    listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.bind(str(args.channel));os.chmod(args.channel,0o600);listener.listen(1)
    service=Service(args);save(args.output,service.result)
    try:
        while not service.closed:
            connection,_=listener.accept()
            with connection:
                connection.settimeout(5);raw=b''
                while b'\n' not in raw:
                    part=connection.recv(4096)
                    if not part or len(raw)+len(part)>16384:raise RuntimeError('bounded local command framing')
                    raw+=part
                command=json.loads(raw.split(b'\n')[0]);stop=threading.Event();response=dict(status='failed')
                try:
                    if command.get('auth')!=token:raise RuntimeError('private channel authentication failed')
                    owner=verified_lease(command);key=(owner['session'],owner['lease_id'])
                    if key in service.seen_leases:raise RuntimeError('fresh lease required for every command')
                    service.seen_leases.add(key);tick=time.monotonic();peak=[0]
                    def watchdog():
                        while not stop.wait(.2):
                            rss=int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())],text=True).strip() or '0')*1024;peak[0]=max(peak[0],rss)
                            if rss>MAX_RSS or time.monotonic()-tick>MAX_SECONDS:
                                service.result.update(status='worker_killed',external_rss_bytes=peak[0],hard_guard=True);save(args.output,service.result);os._exit(124)
                    monitor=threading.Thread(target=watchdog,daemon=True);monitor.start();action=command['action']
                    if action=='init':data=service.init()
                    elif action=='start':data=service.start(command,owner)
                    elif action=='resume':data=service.resume(command,owner)
                    elif action=='cancel':data=service.cancel(owner)
                    elif action=='shutdown':data=service.shutdown(owner)
                    else:raise ValueError('unknown command')
                    service.verify(owner)
                    response=dict(status='passed',action=action,data=data,gpuq_owner=owner,command_seconds=time.monotonic()-tick,external_peak_rss_bytes=peak[0])
                except BaseException as error:
                    response.update(error=repr(error),traceback=traceback.format_exc());service.result.update(status='failed');FAILURE_ROOTS.append(service)
                    if command.get('action')=='init':
                        # No requests exist yet; retire the partial engine and
                        # restore patched methods/cache policy under this lease.
                        try:
                            service.verify(owner)
                            response['initialization_cleanup']=service.shutdown(owner)
                        except BaseException as cleanup_error:
                            response['initialization_cleanup_error']=repr(cleanup_error)
                        service.result['status']='failed'
                    if service.current is not None and not service.controller.paused and not service.controller.aborted:
                        try:
                            service.seal_unstarted_failure(owner)
                        except BaseException as quiescence_error:
                            # Never return control (and an apparently releasable
                            # lease) while an unproved graph may still execute.
                            # Supervisor process retirement is labelled separately
                            # from matching native terminal/resource cleanup.
                            response['quiescence_error']=repr(quiescence_error)
                            service.result.update(status='worker_killed',failed_command=response,process_retirement_required=True,native_cleanup_proved=False)
                            save(args.output,service.result);os._exit(124)
                    response['terminal_proved_pause']=service.current is None or service.controller.paused or service.controller.aborted
                    # Keep the channel open: an owned shutdown is necessary to
                    # retire private roots; timeout watchdog remains bounded.
                finally:
                    stop.set();service.result.setdefault('command_receipts',[]).append(response);save(args.output,service.result)
                connection.sendall(json.dumps(response).encode()+b'\n')
    finally:listener.close();args.channel.unlink(missing_ok=True);args.auth_token_file.unlink(missing_ok=True)

def client(args):
    if args.channel.is_symlink() or args.auth_token_file.is_symlink() or stat.S_IMODE(args.auth_token_file.stat().st_mode)!=0o600 or args.auth_token_file.stat().st_uid!=os.getuid() or not stat.S_ISSOCK(args.channel.stat().st_mode):raise RuntimeError('private local channel/token required')
    command=dict(auth=args.auth_token_file.read_text(),action=args.command,session=args.session,lease=args.lease,domain_index=args.domain_index,arm=args.arm,phases=args.phases,warmup=args.warmup)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as conn:
        conn.settimeout(110);conn.connect(str(args.channel));conn.sendall(json.dumps(command).encode()+b'\n');raw=b''
        while b'\n' not in raw:
            part=conn.recv(65536)
            if not part or len(raw)+len(part)>32<<20:raise RuntimeError('bounded command response framing')
            raw+=part
    response=json.loads(raw.split(b'\n')[0]);print(json.dumps(response));return int(response['status']!='passed')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=MODEL);parser.add_argument('--artifact-manifest',type=Path,default=Path(MANIFEST));parser.add_argument('--mlx-wheel',type=Path,default=Path(WHEEL));parser.add_argument('--native',type=Path,default=Path(NATIVE));parser.add_argument('--native-sha256',default=NATIVE_SHA)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--profile',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--channel',type=Path,required=True);parser.add_argument('--auth-token-file',type=Path,required=True)
    parser.add_argument('--prefill-step',type=int,default=None,help='explicit ordinary override; omitted preserves production adapter/engine default')
    parser.add_argument('--gdn-eval-wave-max-segments',type=int,choices=(1,2,4),default=1)
    parser.add_argument('--serve-channel',action='store_true');parser.add_argument('--expected-source');parser.add_argument('--command',choices=('init','start','resume','cancel','shutdown'));parser.add_argument('--domain-index',type=int);parser.add_argument('--arm',choices=('native','ordinary'));parser.add_argument('--phases',type=int,default=4);parser.add_argument('--warmup',action='store_true');parser.add_argument('--session',default=os.environ.get('GPUQ_SESSION'));parser.add_argument('--lease',default=os.environ.get('GPUQ_LEASE'))
    args=parser.parse_args()
    if args.command:return client(args)
    if args.serve_channel:
        if not args.expected_source:parser.error('frozen source required')
        serve(args);return 0
    save(args.output,plan());return 0
if __name__=='__main__':raise SystemExit(main())
