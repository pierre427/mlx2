"""Root-driven persistent-service, cold B2 cohorts; twenty measured requests per arm.

This is NOT the maintained400-case Spomin20×20 suite.
Dry/import-safe. Each init/warmup/cohort/shutdown command requires a fresh dual
GPUQ lease. One model/service stays idle between commands; no batch20 claim.
"""
from __future__ import annotations
import argparse,gc,hashlib,json,os,secrets,signal,socket,stat,subprocess,sys,threading,time
from pathlib import Path
from http.server import ThreadingHTTPServer
from varlen_hybrid_serving_smoke import MODEL,MANIFEST,WHEEL,save
from varlen_packed_long_contract import COUNTS,PROMPT_SHA,exact_prompts,physical_prefill,validate_q1
from varlen_hybrid_packed_prefill_long_http_gate import pair,bounded_cleanup,host_diagnostics
ROOT=Path(__file__).resolve().parents[2]
NATIVE='/tmp/mlx2-long-page-load-integrated-build-1004/_paged_kv_native.cpython-312-darwin.so'
NATIVE_SHA='d619ff84878e199e08fb5740184719c9bd9c79c43804064dd143bd9954df31ed'
CAP=20;COHORTS=10;MAX_SECONDS=100;MAX_RSS=48<<30
FAILURE_ROOTS=[]


def plan():
    return {'schema':'mlx2.long-b2-cap20-http-smoke.v1','status':'planned','gpu_executed':False,
        'requests_per_measured_arm':20,'output_cap_per_request':20,'expected_output_tokens_per_arm':400,
        'physical_cohort_width':2,'queue_concurrency':2,'cohorts_per_arm':10,'warmup_cohorts_per_arm':1,
        'context_tokens':list(COUNTS),'prompt_receipt_sha256':PROMPT_SHA,'native_sha256':NATIVE_SHA,
        'one_persistent_service':True,'one_model_load':True,'fresh_request_caches':True,
        'hard_seconds_per_command':MAX_SECONDS,'max_rss_bytes':MAX_RSS,'qualified':False,'price_usable':False,
        'numeric_tensor_parity':'not_tested','scope':'cold HTTP end-to-end paired requests; first/last actual sampled event clocks'}


def verified_lease(command):
    session=command.get('session');lease=command.get('lease')
    if not isinstance(session,str) or not session or not isinstance(lease,str) or not lease:
        raise ValueError('fresh command session/lease required')
    previous=(os.environ.get('GPUQ_SESSION'),os.environ.get('GPUQ_LEASE'))
    os.environ.update(GPUQ_SESSION=session,GPUQ_LEASE=lease)
    try:
        from varlen_pack_price_bench import _gpuq_owner
        return _gpuq_owner()
    except BaseException:
        for key,value in zip(('GPUQ_SESSION','GPUQ_LEASE'),previous):
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        raise


def rates(events,began,ended):
    if len(events)!=2 or any(len(lane)!=CAP for lane in events):raise RuntimeError('actual20 token events per lane required')
    if any(any(b['monotonic_ns']<=a['monotonic_ns'] for a,b in zip(lane,lane[1:])) for lane in events):
        raise RuntimeError('nonmonotonic sample clocks')
    first=[lane[0]['monotonic_ns']/1e9 for lane in events];last=[lane[-1]['monotonic_ns']/1e9 for lane in events]
    prefill=max(first)-began;decode=max(last)-min(first)
    if prefill<=0 or decode<=0:raise RuntimeError('invalid measured event interval')
    return {'http_pair_seconds':ended-began,'ttft_seconds':[t-began for t in first],
        'prefill_through_both_first_samples_seconds':prefill,'prefill_through_first_sample_prompt_tokens_per_second':sum(COUNTS)/prefill,
        'decode_first_to_last_seconds':decode,'aggregate_decode_tokens_per_second':2*(CAP-1)/decode,
        'per_request_decode_tokens_per_second':[(CAP-1)/(b-a) for a,b in zip(first,last)],
        'complete_output_tokens_per_second':2*CAP/(ended-began),'actual_completion_tokens':2*CAP,
        'decode_rate_numerator':2*(CAP-1),'decode_scope':'actual first-to-last sample events, excluding each first token'}


def validate_response(status,body,native):
    if status!=200:raise RuntimeError('actual HTTP response status differs: '+str(status))
    usage=body.get('usage',{});details=body.get('mlx2',{});receipt=details.get('route_receipt',{})
    if (usage.get('completion_tokens')!=CAP or usage.get('prompt_tokens_details',{}).get('cached_tokens',0)!=0 or
        details.get('qualification')!='unqualified' or len(body.get('choices',()))!=1 or
        body['choices'][0].get('finish_reason')!='length'):
        raise RuntimeError('actual cap20/cold cache/finish/qualification differs')
    if native:
        if (receipt.get('route')!='native_hybrid_paged_b2' or receipt.get('selected') is not True or
            receipt.get('observed_used') is not True or receipt.get('qualified') is not False or receipt.get('price_usable') is not False or
            receipt.get('prefill_mode')!='native_packed_prefill' or receipt.get('native_prefill_observed_used') is not True or
            receipt.get('state_planes')!=['kv','gdn'] or len(receipt.get('output_token_ids',()))!=CAP):
            raise RuntimeError('native cap20 final route receipt differs')
        proof=receipt.get('native_prefill_proof',{})
        if (proof.get('prefill_long_nax') is not True or proof.get('physical_counters')!=physical_prefill() or
            proof.get('serving_numerical_reference')!='same_geometry_ordinary_mixed'):
            raise RuntimeError('native long physical prefill proof differs')
        validate_q1(receipt.get('hybrid_graph_proof',{}),2)
    elif receipt.get('route')=='native_hybrid_paged_b2' or details.get('route')=='native_hybrid_paged_b2':
        raise RuntimeError('ordinary control selected native')
    return {'id':body['id'],'text':body['choices'][0]['text'],'usage':usage,'route_receipt':receipt}


class Sequence:
    def __init__(self):self.initialized=False;self.warm=False;self.next=0;self.closed=False;self.leases=set()
    def accept(self,command,owner):
        if self.closed:raise ValueError('service closed')
        key=(owner['session'],owner['lease_id'])
        if key in self.leases:raise ValueError('each command requires a fresh lease')
        action=command.get('action')
        if action=='init':
            if self.initialized:raise ValueError('duplicate init')
            self.initialized=True
        elif action=='warmup':
            if not self.initialized or self.warm or self.next:raise ValueError('one initial warmup required')
            self.warm=True
        elif action=='cohort':
            index=command.get('cohort_index')
            if not self.warm or type(index) is not int or index!=self.next or index>=COHORTS:
                raise ValueError('ordered cohort index0..9 required after warmup')
            self.next+=1
        elif action=='shutdown':
            if not self.initialized:raise ValueError('service not initialized')
            self.closed=True
        else:raise ValueError('unknown command')
        self.leases.add(key)


class Service:
    def __init__(self,args):
        self.args=args;self.result=plan();self.sequence=Sequence();self.engine=self.server=self.server_thread=None
        self.captures=[];self.events={};self.effective={};self.cells=[];self.restores=[];self.current='idle'
    def init(self):
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=self.args.expected_source or subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):
            raise RuntimeError('frozen source binding differs')
        if hashlib.sha256(Path(self.args.native).read_bytes()).hexdigest()!=self.args.native_sha256:raise RuntimeError('native binary identity differs')
        sys.path.insert(0,str(Path(self.args.native).parent));sys.path.insert(0,str(ROOT/'src'))
        from mlx2.adapters.qwen38_27b import Qwen3827BAdapter,configure_environment
        configure_environment()
        import mlx.core as mx
        from mlx2.runtime.paged_price_identity import cached_live_price_identity
        from mlx2.runtime.paged_packed_prefill_serving_profile import make_profile,load_profile
        identity=cached_live_price_identity(self.args.artifact_manifest,self.args.mlx_wheel,self.args.native,adapter_artifact_root=Path(self.args.model).resolve())
        profile=make_profile(identity,long_fused=True,research_output_cap20=True)
        save(self.args.profile,profile);os.environ.update(profile['required_environment'])
        load_profile(self.args.profile,live_identity=identity,context_lengths=COUNTS,environment=os.environ)
        os.environ.update(MLX2_NATIVE_PACKED_PREFILL_B2_PROFILE=str(self.args.profile),MLX2_NATIVE_PAGED_MANIFEST=self.args.artifact_manifest,MLX2_NATIVE_PAGED_MLX_WHEEL=self.args.mlx_wheel)
        from mlx2 import serving
        from mlx2.server import handler_for
        from mlx2.runtime.generate import BatchGenerator
        from mlx2.runtime import hybrid_packed_prefill as packed, qwen35_paged_graph_factory as resources
        self.mx=mx;self.resources=resources;self.initial_charge=resources._CHARGED
        original_next=BatchGenerator.next;original_factory=packed.create_cold_packed_hybrid
        def observed_next(batch,*a,**kw):
            prompts,responses=original_next(batch,*a,**kw)
            for response in responses:
                job=next((j for j in self.engine.jobs.values() if j.uid==response.uid),None)
                if job is None:raise RuntimeError('actual sampler has no live Job')
                self.events.setdefault(job.id,[]).append({'token':int(response.token),'monotonic_ns':time.monotonic_ns(),
                    'width':getattr(response,'execution_width',1)})
                self.effective[job.id]=dict(job.effective_sampling or {})
            return prompts,responses
        def captured_factory(*a,**kw):
            owners,candidate,bootstrap=original_factory(*a,**kw)
            capture={'owners':owners,'candidate':candidate,'bootstrap':bootstrap,'physical':{}}
            self.captures.append(capture)
            arena=candidate.backend.writer.backend;original_close=arena.close_after_terminal
            def close():
                if not arena._closed:capture['physical'].update(candidate.backend.profile_counters_snapshot())
                return original_close()
            arena.close_after_terminal=close
            return owners,candidate,bootstrap
        BatchGenerator.next=observed_next;packed.create_cold_packed_hybrid=captured_factory
        self.restores=[(BatchGenerator,'next',original_next),(packed,'create_cold_packed_hybrid',original_factory)]
        began=time.monotonic();self.engine=serving.ServingEngine(self.args.model,adapter_factory=Qwen3827BAdapter,max_lanes=2,max_inflight=2,
            mtp=False,prompt_lookup=False,qualification_mode=False,prefill_step=8192,max_context=8192,cache_bytes=1<<29,batch_cohort_timeout_ms=200)
        if not self.engine.ready.wait(40) or self.engine.error:raise RuntimeError('persistent engine load failed: '+str(self.engine.error))
        if cached_live_price_identity(self.args.artifact_manifest,self.args.mlx_wheel,self.args.native,adapter_artifact_root=Path(self.engine.adapter.identity['path']).resolve())!=identity:
            raise RuntimeError('loaded adapter identity differs')
        self.prompts=exact_prompts(self.engine.adapter)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(self.engine));self.server.daemon_threads=True;self.server.block_on_close=False
        self.server_thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.server_thread.start()
        self.base=f'http://127.0.0.1:{self.server.server_port}'
        self.result.update(gpu_executed=True,status='initialized',identity=identity,load_seconds=time.monotonic()-began,
            prompt_token_ids=[list(ids) for _,ids in self.prompts],prompt_text_sha256=[hashlib.sha256(text.encode()).hexdigest() for text,_ in self.prompts],
            model_loads=1,engine_instances=1,persistent_service_port=self.server.server_port)
        return {'initialized':True,'source':identity['source_commit']}
    def arm(self,native,index):
        self.current='native' if native else 'ordinary';self.events={};self.effective={};before=len(self.captures)
        common=[{'model':Path(self.args.model).name,'prompt':text,'max_tokens':CAP,'temperature':0,
            'repetition_penalty':1,'presence_penalty':0,'frequency_penalty':0,'skip_writing_prefix_cache':True,
            'batch_cohort':{'id':'hybrid-http-gate','size':2}} for text,_ in self.prompts]
        if native:
            for body in common:body.update(paged_native_hybrid_b2=True,paged_native_hybrid_packed_prefill=True,paged_native_long_cap20_research=True)
        began=time.monotonic();responses=pair(self.base,tuple(common),self.engine);ended=time.monotonic()
        summaries=[validate_response(status,body,native) for status,body in responses]
        event_rows=[self.events.get(body['id'],[]) for _,body in responses]
        if any(any(event['width']!=2 for event in lane[1:]) for lane in event_rows):raise RuntimeError('actual decode cohort width differs from2')
        sampling=[self.effective.get(body['id']) for _,body in responses]
        if any(not item or item.get('temperature')!=0 or item.get('repetition_penalty')!=1 or item.get('presence_penalty')!=0 or item.get('frequency_penalty')!=0 for item in sampling):
            raise RuntimeError('effective greedy sampling differs')
        result={'arm':self.current,'index':index,'summaries':summaries,'events':event_rows,'effective_sampling':sampling,**rates(event_rows,began,ended)}
        if native:
            if len(self.captures)!=before+1:raise RuntimeError('one physical native cohort required')
            capture=self.captures[-1];candidate=capture['candidate'];writer=candidate.backend.writer
            deadline=time.monotonic()+5
            while not candidate._serving_resources.closed and time.monotonic()<deadline:
                self.resources.reap_hybrid_admission_orphans();candidate._serving_resources.reap();time.sleep(.005)
            clean={'pending_epochs':len(writer.pending_epochs),'pending_ledger':writer.ledger.pending_count,'pages':writer.pool.allocated_count,
                'owners_retired':all(o.fully_retired for o in capture['owners']),'resources_closed':candidate._serving_resources.closed,'charge':self.resources._CHARGED}
            if clean['pending_epochs'] or clean['pending_ledger'] or clean['pages'] or not clean['owners_retired'] or not clean['resources_closed'] or clean['charge']!=self.initial_charge:
                raise RuntimeError('cohort terminal/charge cleanup failed')
            physical=capture['physical'];expected=(CAP-1)*16
            if physical.get('q1_stock_long_partial_dispatches')!=expected or physical.get('q1_stock_long_reduce_dispatches')!=expected:
                raise RuntimeError('nineteen actual B2 nativeQ1 dispatches not proven')
            result.update(cleanup=clean,physical_counters=physical,prefill_model_seconds=candidate._packed_prefill_receipt.get('model_prefill_seconds'))
            self.captures.clear();capture=candidate=writer=None
        elif len(self.captures)!=before:raise RuntimeError('ordinary control allocated native arena')
        result['token_ids']=[[event['token'] for event in lane] for lane in event_rows]
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            with self.engine.lock:
                live=bool(self.engine.jobs)
            with self.engine.submission_lock:
                staged=bool(self.engine.pending_cohorts)
            if not live and not staged and self.engine.incoming.empty() and self.engine.queued_jobs==0:break
            time.sleep(.005)
        else:raise RuntimeError('service is not quiescent before lease release')
        result['idle_queue_proof']={'jobs':0,'pending_cohorts':0,'queued_jobs':0,'incoming':0}
        self.events={};self.effective={};self.mx.synchronize();gc.collect();self.mx.clear_cache()
        result['allocator_after_cleanup']={'active_bytes':int(self.mx.get_active_memory()),'cache_bytes':int(self.mx.get_cache_memory())}
        return result
    def cohort(self,index):
        order=[True,False] if index<0 or index%2 else [False,True]
        rows=[self.arm(native,index) for native in order];native=next(r for r in rows if r['arm']=='native');ordinary=next(r for r in rows if r['arm']=='ordinary')
        if native['token_ids']!=ordinary['token_ids'] or [s['text'] for s in native['summaries']]!=[s['text'] for s in ordinary['summaries']] or native['effective_sampling']!=ordinary['effective_sampling']:
            raise RuntimeError('actual identical-input native/ordinary token/text/sampling parity failed')
        cell={'index':index,'warmup':index<0,'order':[r['arm'] for r in rows],'native':native,'ordinary':ordinary,'exact_token_parity':True}
        self.cells.append(cell);self.current='idle';self.result.update(status='idle',cells=self.cells,completed_measured_cohorts=sum(not c['warmup'] for c in self.cells))
        return {'cohort_index':index,'exact_token_parity':True,'order':cell['order']}
    def shutdown(self):
        captures=[(c['owners'],c['candidate'],c['bootstrap']) for c in self.captures]
        cleanup=bounded_cleanup(self.engine,self.server,self.server_thread,captures,self.result)
        for obj,name,original in reversed(self.restores):setattr(obj,name,original)
        if cleanup['retained'] or not cleanup['engine_closed']:raise RuntimeError('persistent service shutdown retained roots')
        self.engine=self.server=self.server_thread=None;self.captures.clear();gc.collect()
        measured=[c for c in self.cells if not c['warmup']]
        if len(measured)==COHORTS:
            totals={}
            for arm in ('native','ordinary'):
                rows=[c[arm] for c in measured];wall=sum(r['http_pair_seconds'] for r in rows);prefill=sum(r['prefill_through_both_first_samples_seconds'] for r in rows);decode=sum(r['decode_first_to_last_seconds'] for r in rows)
                totals[arm]={'actual_requests':20,'actual_completion_tokens':sum(r['actual_completion_tokens'] for r in rows),
                    'summed_http_pair_seconds':wall,'complete_output_tokens_per_second':400/wall,
                    'prefill_prompt_tokens_per_second':sum(COUNTS)*COHORTS/prefill,'aggregate_decode_tokens_per_second':380/decode,
                    'rate_scope':'sum of ten B2 intervals; excludes idle lease gaps/load/cleanup; full-service wall separately recorded'}
            self.result.update(status='passed',token_parity='passed',totals=totals)
        else:self.result.update(status='incomplete',token_parity='partial')
        return cleanup


def serve(args):
    token_path=args.auth_token_file
    if token_path.exists() or args.channel.exists():raise RuntimeError('fresh local channel/token paths required')
    token=secrets.token_hex(32);fd=os.open(token_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:f.write(token)
    listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.bind(str(args.channel));os.chmod(args.channel,0o600);listener.listen(1)
    service=Service(args);started=time.monotonic();save(args.output,service.result)
    try:
        while not service.sequence.closed:
            connection,_=listener.accept()
            with connection:
                connection.settimeout(5);raw=b''
                while b'\n' not in raw:
                    chunk=connection.recv(4096)
                    if not chunk or len(raw)+len(chunk)>16384:raise RuntimeError('bounded command framing failed')
                    raw+=chunk
                command=json.loads(raw.split(b'\n')[0]);response={'status':'failed'};stop=threading.Event();watchdog=None
                try:
                    if command.get('auth')!=token:raise ValueError('local command authentication failed')
                    owner=verified_lease(command);service.sequence.accept(command,owner)
                    action=command['action'];tick=time.monotonic();peak=[0]
                    def monitor():
                        while not stop.wait(.2):
                            rss=int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())],text=True).strip() or '0')*1024;peak[0]=max(peak[0],rss)
                            if rss>MAX_RSS or time.monotonic()-tick>MAX_SECONDS:
                                service.result.update(status='worker_killed',deadline_or_rss=True,peak_rss_bytes=peak[0]);save(args.output,service.result);os._exit(124)
                    watchdog=threading.Thread(target=monitor,daemon=True);watchdog.start()
                    if action=='init':data=service.init()
                    elif action in ('warmup','cohort'):data=service.cohort(-1 if action=='warmup' else command['cohort_index'])
                    else:data=service.shutdown()
                    # Lease must remain owned until command terminal success and cleanup.
                    if verified_lease(command)!=owner:raise RuntimeError('lease ownership changed during command')
                    response={'status':'passed','action':action,'data':data,'gpuq_owner':owner,'command_seconds':time.monotonic()-tick,'peak_rss_bytes':peak[0]}
                except BaseException as error:
                    response.update(error=repr(error));service.result.update(status='failed',failure_diagnostics=host_diagnostics(service.engine,service.current))
                    FAILURE_ROOTS.append(service);service.sequence.closed=True
                finally:
                    stop.set()
                    if watchdog:watchdog.join(timeout=1)
                    service.result.setdefault('command_receipts',[]).append(response);service.result['service_wall_seconds']=time.monotonic()-started;save(args.output,service.result)
                connection.sendall(json.dumps(response).encode()+b'\n')
    finally:
        listener.close();args.channel.unlink(missing_ok=True);token_path.unlink(missing_ok=True)


def client(args):
    mode=stat.S_IMODE(args.auth_token_file.stat().st_mode)
    if mode!=0o600 or args.auth_token_file.stat().st_uid!=os.getuid() or args.channel.stat().st_uid!=os.getuid() or not stat.S_ISSOCK(args.channel.stat().st_mode) or args.auth_token_file.is_symlink() or args.channel.is_symlink():raise RuntimeError('private local channel/token required')
    command={'auth':args.auth_token_file.read_text(),'action':args.command,'session':args.session,'lease':args.lease,'cohort_index':args.cohort_index}
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
        connection.settimeout(MAX_SECONDS+10);connection.connect(str(args.channel));connection.sendall(json.dumps(command).encode()+b'\n');raw=b''
        while b'\n' not in raw:
            chunk=connection.recv(65536)
            if not chunk or len(raw)+len(chunk)>4<<20:raise RuntimeError('command response framing failed')
            raw+=chunk
    response=json.loads(raw.split(b'\n')[0]);print(json.dumps(response));return 0 if response['status']=='passed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=MODEL);parser.add_argument('--artifact-manifest',default=MANIFEST);parser.add_argument('--mlx-wheel',default=WHEEL)
    parser.add_argument('--native',default=NATIVE);parser.add_argument('--native-sha256',default=NATIVE_SHA)
    parser.add_argument('--profile',type=Path,default=Path('/tmp/domain20-profile.json'));parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--channel',type=Path,default=Path('/tmp/domain20-channel.sock'));parser.add_argument('--auth-token-file',type=Path,default=Path('/tmp/domain20-auth.token'))
    parser.add_argument('--serve-channel',action='store_true');parser.add_argument('--command',choices=('init','warmup','cohort','shutdown'))
    parser.add_argument('--expected-source');parser.add_argument('--session',default=os.environ.get('GPUQ_SESSION'));parser.add_argument('--lease',default=os.environ.get('GPUQ_LEASE'));parser.add_argument('--cohort-index',type=int)
    args=parser.parse_args()
    if args.command:return client(args)
    if args.serve_channel:
        if not args.expected_source:parser.error('persistent worker requires frozen --expected-source')
        serve(args);return 0
    save(args.output,plan());return 0

if __name__=='__main__':raise SystemExit(main())
