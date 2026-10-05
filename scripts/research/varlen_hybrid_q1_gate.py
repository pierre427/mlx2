"""Bounded explicit hybrid Q1 parity gate; never establishes serving qualification."""
from __future__ import annotations
import argparse, hashlib, json, math, os, resource, signal, subprocess, time
from pathlib import Path
from contextlib import nullcontext
ROOT = Path(__file__).resolve().parents[2]
RETAINED_FAILURES = []

class RetainedStaging:
    """Conservative gate budget retained through complete owner teardown."""
    def __init__(self, size, budget):
        if budget['charged'] + size > budget['capacity']:
            raise MemoryError('hybrid staging budget exhausted')
        budget['charged'] += size
        self.bytes, self.budget = size, budget
        self.roots, self.completed = (), False
    def release(self):
        # Keep the entire charge while recurrent checkpoints remain live.
        self.completed = True
    def retain_failure_roots(self, roots):
        self.roots = roots
    def close_after_teardown(self):
        self.roots = ()
        self.budget['charged'] -= self.bytes


def validate_stock_long_launch(args, result, environment, *, source_clean):
    """CPU-only frozen source/binary/selector contract before model loading."""
    if type(getattr(args, 'q1_deferred', False)) is not bool:
        raise ValueError('hybrid deferred Q1 diagnostic selector must be boolean')
    if type(getattr(args, 'q1_deferred_writes_only', False)) is not bool or (
            getattr(args, 'q1_deferred_writes_only', False) and getattr(args, 'q1_deferred', False)):
        raise ValueError('hybrid Q1 deferral modes are mutually exclusive boolean selectors')
    inline = getattr(args, 'stock_long_inline_metadata', False)
    if inline and not args.stock_long:
        raise ValueError('inline metadata requires explicit stock-long selection')
    if environment.get('MLX2_PAGED_Q1_STOCK_LONG_INLINE_METADATA', '0') != ('1' if inline else '0'):
        raise ValueError('inline metadata environment differs from explicit selector')
    if not args.stock_long and not getattr(args,'q1_joined_recurrent',False) and not getattr(args,'q1_direct_grouped_fence',False) and not getattr(args,'q1_deferred',False) and not getattr(args,'q1_deferred_writes_only',False): return
    if args.stock_long and (args.split_partition != 0 or environment.get('MLX2_PAGED_Q1_SPLIT_KV','0') != '0' or
            environment.get('MLX2_PAGED_Q1_STOCK_LONG','0') != '1' or
            environment.get('MLX_SDPA_BLOCKS','0') != '0'):
        raise ValueError('stock-long gate requires explicit flag1 and no split/block overrides')
    if (not source_clean or type(args.source_commit) is not str or
            len(args.source_commit) != 40 or args.source_commit != result['source_commit'] or
            type(args.native_sha256) is not str or len(args.native_sha256) != 64 or
            any(c not in '0123456789abcdef' for c in args.native_sha256)):
        raise ValueError('hybrid diagnostic gate requires clean pinned source and native SHA256')


def execute(args, result):
    from varlen_pack_price_bench import _gpuq_owner
    result['gpuq_owner'] = _gpuq_owner()
    from mlx2.adapters.qwen38_27b import configure_environment
    configure_environment()
    import mlx.core as mx
    import _paged_kv_native
    actual_native_sha256 = hashlib.sha256(Path(_paged_kv_native.__file__).read_bytes()).hexdigest()
    result['native_sha256'] = actual_native_sha256
    if args.native_sha256 is not None and actual_native_sha256 != args.native_sha256:
        raise ValueError('loaded native binary differs from frozen gate SHA256')
    if args.stock_long and any(not callable(getattr(_paged_kv_native,name,None)) for name in
            ('q1_stock_long_partial_dispatch_count','q1_stock_long_reduce_dispatch_count')):
        raise ValueError('native stock-long physical-counter ABI unavailable')
    if args.stock_long_inline_metadata and not callable(getattr(_paged_kv_native,'q1_stock_long_metadata_dispatch_count',None)):
        raise ValueError('native stock-long inline metadata physical-counter ABI unavailable')
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    from mlx2.adapters.qwen35_paged_candidate import Qwen35PagedCandidate, clone_recurrent_caches
    from mlx2.runtime.paged_attention_plan import PAGE_SIZE
    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
    from mlx2.runtime.paged_kv_write import NativeWriteBackend, PagedKVWriteOwner
    from mlx2.runtime.qwen3_paged_native_backend import NativeQwen3PagedBackend
    from mlx2.runtime.paged_gdn_checkpoint import GDNBoundaryCheckpoint
    from mlx2.runtime.paged_native_atomic_owner import NativeAtomicRequestOwner
    from mlx2.runtime.paged_native_retirement import reap_native_request_owner
    from mlx2.runtime.paged_request_transaction import CandidateRequest
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('hybrid gate deadline')))
    signal.alarm(100)
    adapter = Qwen3827BAdapter(args.model, require_mtp=False)
    model = getattr(adapter.model, 'language_model', adapter.model)
    revision = adapter.identity['fingerprint']
    result.update(model_identity=adapter.identity, native_sha256=hashlib.sha256(Path(_paged_kv_native.__file__).read_bytes()).hexdigest(), device=dict(mx.device_info()))
    contexts=tuple(int(value) for value in args.contexts.split(','))
    if args.prompt_receipt:
        pinned=Path(args.prompt_receipt);saved=json.loads(pinned.read_text())
        if (Path(saved['model_root']).resolve()!=Path(args.model).resolve() or
                saved['config_sha256']!=hashlib.sha256((Path(args.model)/'config.json').read_bytes()).hexdigest()):
            raise ValueError('long prompt receipt model identity differs')
        prompts=tuple(tuple(entry['token_ids']) for entry in saved['prompts'][:2])
        contexts=tuple(len(tokens) for tokens in prompts)
        result['prompt_source']={'path':str(pinned.resolve()),'sha256':hashlib.sha256(pinned.read_bytes()).hexdigest(),'kind':'full-domain-software-architecture-27b-baseline'}
    else:
        seed=adapter.tokenizer.encode('Explain database isolation, causality, durability and distributed consistency. ',add_special_tokens=False)
        prompts=tuple(tuple((seed*((length+len(seed)-1)//len(seed)))[:length]) for length in contexts)
        result['prompt_source']={'kind':'short-structural-repeated-domain-seed'}
    if len(contexts)!=2 or len(prompts)!=2 or any(length<1 or length+args.steps>8192 for length in contexts):
        raise ValueError('two bounded prompt lengths plus decode must fit8192')
    if max(contexts)+args.steps>128 and args.split_partition==0 and not args.stock_long:
        raise ValueError('long hybrid gate requires an explicit split partition')
    if args.stock_long and (max(contexts)+1 <= 1024 or
            model.args.num_attention_heads // model.args.num_key_value_heads <= 4 or
            not str(mx.device_info().get('architecture','')).endswith('s')):
        raise ValueError('stock-long gate requires long1025..8192, GQA abovefour and architecture s')
    result['stock_long'] = args.stock_long
    result['stock_long_inline_metadata'] = args.stock_long_inline_metadata
    result['q1_all_valid_recurrent'] = args.q1_all_valid_recurrent
    result['q1_joined_recurrent'] = args.q1_joined_recurrent
    result['q1_direct_grouped_fence'] = args.q1_direct_grouped_fence
    result['q1_deferred'] = args.q1_deferred
    result['q1_deferred_writes_only'] = args.q1_deferred_writes_only
    result['frozen_native_sha256'] = args.native_sha256
    result['context_tokens']=contexts
    result['numeric_envelope']={'logit_atol':args.logit_atol,'state_atol':args.state_atol,'rtol':0.0,'qualification':False}
    reference_caches = tuple(tuple(model.make_cache()) for _ in prompts)
    initial = []
    start=time.perf_counter()
    for tokens,caches in zip(prompts,reference_caches):
        hidden=model.model(mx.array([list(tokens)]),cache=list(caches))
        logits=model.logits(hidden[:,-1:,:]);mx.eval(logits,*[leaf for c in caches for leaf in c.state])
        initial.append(logits[0,0])
    result['ordinary_prefill_seconds']=time.perf_counter()-start
    storage_dtype = str(model.compute_dtype).removeprefix('mlx.core.') if args.kv_precision=='same' else 'float16'
    profile=TokenKVProfile(model.args.num_key_value_heads,256,storage_dtype)
    depth=sum(not layer.is_linear for layer in model.model.layers)
    capacity=sum(depth*((len(tokens)+args.steps+PAGE_SIZE-1)//PAGE_SIZE+2) for tokens in prompts)
    pool=PagedKVPool(capacity)
    native=NativeWriteBackend(capacity*profile.page_bytes,mx.default_stream(mx.gpu),permit_candidate=True,storage_dtype=storage_dtype)
    writer=PagedKVWriteOwner(pool,native,page_bytes=profile.page_bytes,permit_candidate=True)
    backend=NativeQwen3PagedBackend(writer,permit_candidate=True,profile_host=True)
    candidate=Qwen35PagedCandidate(adapter.model,backend,q1_stripes=16,kv_precision=args.kv_precision,q1_split_partition=args.split_partition,stock_long=args.stock_long,stock_long_inline_metadata=args.stock_long_inline_metadata,q1_all_valid_recurrent=args.q1_all_valid_recurrent,q1_joined_recurrent=args.q1_joined_recurrent,q1_direct_grouped_fence=args.q1_direct_grouped_fence,q1_deferred=args.q1_deferred,q1_deferred_writes_only=args.q1_deferred_writes_only)
    budget={'charged':0,'capacity':(4<<30) if max(contexts)>128 or args.q1_joined_recurrent else (512<<20)};reservations=[];owners=[];branches=[];prepared=[];unattached_layers=[]
    result['bootstrap_receipts']=[];result['steps']=[]
    try:
        from varlen_hybrid_ordinary_b2_reference import OrdinaryHybridB2Reference, compare_recurrent_slots
        reference=OrdinaryHybridB2Reference.from_row_caches(model,reference_caches,full_attention_layers=candidate.layer_map.full_attention,recurrent_layers=candidate.layer_map.recurrent,expected_offsets=contexts,mx=mx)
        def reserve(size):
            item=RetainedStaging(size,budget);reservations.append(item);return item
        bootstrap_start=time.perf_counter()
        for index,tokens in enumerate(prompts):
            layers=tuple(PagedKVTokenOwner(writer,profile,permit_candidate=True) for _ in range(depth))
            unattached_layers.extend(layers)
            boot=candidate.bootstrap_ordinary(tokens,layers,reserve_staging=reserve,permit_candidate=True)
            if writer.pending_epochs or writer.ledger.pending_count or any(layer.offset!=len(tokens) for layer in layers):
                raise RuntimeError('hybrid bootstrap is not terminal')
            checkpoint=GDNBoundaryCheckpoint(revision,index,len(tokens),0,boot.recurrent_caches)
            owners.append(NativeAtomicRequestOwner(revision,layers,{'gdn':(checkpoint,)},supported_planes=('kv','gdn'),enabled=True,reuse_private_tail=True,checkpoint_planes=('gdn',),recurrent_clone=clone_recurrent_caches,lane_id=index))
            unattached_layers.clear()
            boot_error=float(mx.max(mx.abs(boot.logits[0].astype(mx.float32)-initial[index].astype(mx.float32))).item())
            if not math.isfinite(boot_error) or boot_error>args.logit_atol or int(mx.argmax(boot.logits[0]).item())!=int(mx.argmax(initial[index]).item()):
                raise RuntimeError('hybrid bootstrap logit/token parity failed')
            result['bootstrap_receipts'].append({**boot.receipt,'ordinary_logit_max_abs_error':boot_error})
        result['candidate_bootstrap_seconds']=time.perf_counter()-bootstrap_start
        result['ordinary_prompt_tokens_per_second']=sum(contexts)/result['ordinary_prefill_seconds']
        result['candidate_bootstrap_prompt_tokens_per_second']=sum(contexts)/result['candidate_bootstrap_seconds']
        result['retained_staging_charge_bytes']=budget['charged']
        next_tokens=tuple(int(mx.argmax(logits).item()) for logits in initial)
        candidate_seconds=ordinary_seconds=0.0
        result['native_host_profile_before_decode']=backend.profile_counters_snapshot()
        for step in range(args.steps):
            branches=[owner.begin(CandidateRequest(i,revision,1,('kv','gdn'))) for i,owner in enumerate(owners)]
            probe=None
            if args.fa_probe:
                from varlen_hybrid_fa_boundary_probe import FABoundaryProbe
                from mlx2.runtime.models import qwen38_27b as attention_module
                probe=FABoundaryProbe();candidate._fa_boundary_probe=probe.candidate_callback
            start=time.perf_counter()
            packed,receipt=candidate.forward_staged(tuple(candidate.packed_lane((token,),branch) for token,branch in zip(next_tokens,branches)),tuple(branches),permit_candidate=True,reserve_scratch=reserve)
            mx.eval(packed);candidate_step_seconds=time.perf_counter()-start;candidate_seconds+=candidate_step_seconds
            start=time.perf_counter()
            with probe.capture_ordinary(attention_module) if probe else nullcontext():
                ordinary_logits=reference.forward_one(next_tokens)
            ordinary_step_seconds=time.perf_counter()-start;ordinary_seconds+=ordinary_step_seconds
            if probe:result.setdefault("fa_boundary_probes",[]).append(probe.compare(tuple(branches),attention_module=attention_module,mx=mx))
            ordinary=tuple(ordinary_logits[i] for i in range(2))
            errors=[float(mx.max(mx.abs(packed[i].astype(mx.float32)-value.astype(mx.float32))).item()) for i,value in enumerate(ordinary)]
            greedy=[int(mx.argmax(packed[i]).item())==int(mx.argmax(value).item()) for i,value in enumerate(ordinary)]
            state_comparison=compare_recurrent_slots(tuple(branch.recurrent_caches for branch in branches),reference,atol=args.state_atol,rtol=0.0)
            recurrent=[max(slot['max_abs'] for slot in state_comparison['slots'] if slot['lane']==i) for i in range(2)]
            result['native_host_profile']=backend.profile_counters_snapshot()
            result.update(candidate_q1_seconds=candidate_seconds,ordinary_b2_q1_seconds=ordinary_seconds,candidate_diagnostic_aggregate_tps=2*(step+1)/candidate_seconds,ordinary_b2_diagnostic_aggregate_tps=2*(step+1)/ordinary_seconds)
            result['steps'].append({'step':step,'candidate_forward_seconds':candidate_step_seconds,'ordinary_forward_seconds':ordinary_step_seconds,'max_logit_abs_error':errors,'greedy_equal':greedy,'max_recurrent_abs_error':recurrent,'recurrent_comparison':state_comparison,'physical':receipt})
            if not all(greedy) or any(not math.isfinite(x) for x in errors+recurrent) or max(errors)>args.logit_atol or max(recurrent)>args.state_atol:
                raise RuntimeError('hybrid candidate numeric/token parity gate failed')
            prepared=[]
            for branch in branches:prepared.append(branch.prepare(1))
            for state in prepared:state.publish()
            branches=[];prepared=[]
            for owner in owners:owner.reap_retired()
            next_tokens=tuple(int(mx.argmax(value).item()) for value in ordinary)
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>48*1024**3:raise MemoryError('48 GiB gate RSS bound')
        result['native_host_profile']=backend.profile_counters_snapshot()
        result.update(candidate_q1_seconds=candidate_seconds,ordinary_b2_q1_seconds=ordinary_seconds,candidate_aggregate_tps=2*args.steps/candidate_seconds,ordinary_b2_aggregate_tps=2*args.steps/ordinary_seconds,qualification=False)
    finally:
        for state in prepared:state.rollback()
        for branch in branches:branch.rollback()
        for owner in owners:owner.close();reap_native_request_owner(owner,writer,backend)
        backend.drain_failed_read_events()
        if writer.pending_epochs:writer.poll_completions()
        if writer.poisoned and not writer.pending_epochs and not writer.ledger.pending_count:
            writer.teardown_failed_arena()
            candidate.release_failure_roots_after_teardown()
        unattached_retired = not writer.pending_epochs and not writer.ledger.pending_count and not backend._orphaned_reads and (not writer.poisoned or writer.failed_arena_torn_down)
        if unattached_retired:
            for layer in unattached_layers:layer.close()
        result['unattached_layers_retired']=unattached_retired
        result['owners_retired']=all(owner.fully_retired for owner in owners)
        result['pending_epochs']=len(writer.pending_epochs)
        result['pending_ledger']=writer.ledger.pending_count
        clean = result['owners_retired'] and not writer.pending_epochs and not writer.ledger.pending_count and not backend._orphaned_reads and unattached_retired
        if clean:
            if args.q1_joined_recurrent and candidate._joined_charge is not None:
                candidate.release_joined_after_retirement(tuple(owners))
            for item in reservations:item.close_after_teardown()
        result['remaining_staging_charge_bytes']=budget['charged']
        if clean:adapter.close()
        else:RETAINED_FAILURES.append((adapter,candidate,owners,unattached_layers,reservations,backend))
        signal.alarm(0)
        if not clean:raise RuntimeError('hybrid gate retained unresolved terminal leases')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--output',required=True);parser.add_argument('--steps',type=int,default=8);parser.add_argument('--logit-atol',type=float,default=.25);parser.add_argument('--state-atol',type=float,default=.001);parser.add_argument('--kv-precision',choices=['same','float16_candidate'],default='same');parser.add_argument('--fa-probe',action='store_true');parser.add_argument('--contexts',default='32,96');parser.add_argument('--prompt-receipt');parser.add_argument('--split-partition',type=int,choices=[0,128,256],default=0);parser.add_argument('--stock-long',action='store_true');parser.add_argument('--stock-long-inline-metadata',action='store_true');parser.add_argument('--q1-all-valid-recurrent',action='store_true');parser.add_argument('--q1-joined-recurrent',action='store_true');parser.add_argument('--q1-direct-grouped-fence',action='store_true');parser.add_argument('--q1-deferred',action='store_true');parser.add_argument('--q1-deferred-writes-only',action='store_true');parser.add_argument('--native-sha256');parser.add_argument('--source-commit')
    args=parser.parse_args()
    if args.q1_joined_recurrent and not args.q1_all_valid_recurrent:raise ValueError('joined recurrent requires explicit all-valid Q1 flag')
    if args.stock_long:os.environ['MLX2_PAGED_Q1_STOCK_LONG']='1'
    if args.stock_long_inline_metadata:os.environ['MLX2_PAGED_Q1_STOCK_LONG_INLINE_METADATA']='1'
    if not 1<=args.steps<=16:raise ValueError('bounded steps required')
    result={'schema':'mlx2.hybrid-q1-short-parity.v1','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'status':'running','qualified':False}
    source_clean=not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()
    try:
        validate_stock_long_launch(args,result,os.environ,source_clean=source_clean)
        execute(args,result);result['status']='passed'
    except BaseException as error:result.update(status='failed',error=repr(error));raise
    finally:Path(args.output).write_text(json.dumps(result,indent=2,default=str))
