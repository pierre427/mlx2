"""Source-bound distinct N20 factory/admission; no B2 scope/default widening."""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
SCHEMA='mlx2.hybrid-packed-prefill-n20-research.v1'
ROUTE='native_hybrid_packed_n20_research'
MAX_RSS=48<<30;BUDGET=40<<30
COUNTERS=('grouped_n20_write_count','grouped_n20_row_count','prefill_long_n20_dispatch_count',
          'q1_stock_long_n20_partial_dispatch_count','q1_stock_long_n20_reduce_dispatch_count')
CAP={'version':1,'storage_dtype':'bfloat16','head_dim':256,'query_heads':24,'kv_heads':4,
     'max_spans':20,'max_total_rows':163840,'max_page_ids':2560,'min_prefill_count':256,'max_prefill_count':8192,
     'max_causal_end':8192,'origin_zero':True,'window_zero':True,'architecture':'s','prefill_score_scratch_bytes':0,
     'max_q1_scratch_bytes':67108864,'write_dispatches':1,'prefill_read_dispatches':1,'q1_read_dispatches':2,
     'qualified':False,'selector':'MLX2_PAGED_PACKED_N20','b1_survivor_scalar_dispatches':1,'b1_stock_long_selector':'MLX2_PAGED_Q1_STOCK_LONG_N20_SINGLETON','b1_stock_long_dispatches':2}


def token_sha(tokens):return hashlib.sha256(json.dumps(list(tokens),separators=(',',':')).encode()).hexdigest()
def wave_selection(value):
    if type(value) is not int or value not in (1,2,4):
        raise ValueError('explicit GDN evaluation wave must be 1, 2 or 4')
    return value

def mlp_selection(value):
    if value not in ('staged_qmm','single_eval_qmm','staged_bf16','single_eval_bf16',
                     'tiled_q4_swiglu','packed_gate_up_qmm'):
        raise ValueError('explicit N20 MLP experiment required')
    return value


def environment(gdn_eval_wave_max_segments=1):
    from .hybrid_packed_prefill_long import long_environment
    wave=wave_selection(gdn_eval_wave_max_segments)
    result={**long_environment(),'MLX2_PAGED_PACKED_N20':'1','MLX2_PAGED_Q1_STOCK_LONG_N20_SINGLETON':'1'}
    if wave>1:
        result.update(MLX2_PAGED_GDN_EVAL_WAVE_MAX_SEGMENTS=str(wave),MLX_GDN_PACKED='1',MLX_GDN_CORE='0')
    return result


def require_n_capabilities(native,backend_type,counts,*,q1=False):
    if type(counts) is not tuple or not 1<=len(counts)<=20 or any(type(n) is not int or not (1025 if q1 else 256)<=n<=8192 for n in counts) or sum(counts)>163840:
        raise ValueError('bounded genuineN20 geometry required')
    names=('packed_n20_capability','grouped_multirow_write_n20','q1_scalar_dispatch_count','q1_stock_long_n20_singleton_partial_dispatch_count','q1_stock_long_n20_singleton_reduce_dispatch_count',*COUNTERS)
    if any(not callable(getattr(native,name,None)) for name in names):raise ValueError('nativeN20 full capability/counter ABI missing')
    cap=native.packed_n20_capability()
    if type(cap) is not dict or any(type(cap.get(k)) is not type(v) or cap.get(k)!=v for k,v in CAP.items()):raise ValueError('nativeN20 exact capability differs')
    if backend_type is not None and any(not callable(getattr(backend_type,name,None)) for name in ('append_packed_multirow_n20','append_staged_grouped_q1_n20','prepare_read_n20')):
        raise ValueError('nativeN20 host lifecycle capability missing')
    return cap


def preflight_native_inputs(native, inputs):
    """Check real raw ABI and largest source-bound plane before loading weights."""
    from .paged_native_arena_geometry import require_arena_storage
    proofs=[]
    for domain in inputs['domain_order']:
        counts=tuple(row['prompt_tokens'] for row in inputs['rows'] if row['domain']==domain)
        if len(counts)!=20:raise ValueError('complete actual domain required for native preflight')
        require_n_capabilities(native,None,counts)
        # Pinned 16 FA layers, KV4, D256, BF16, page64 and two private tail pages.
        pages=sum(16*((count+192+63)//64+2) for count in counts)
        proofs.append(require_arena_storage(native,pages*4*256*64*2))
    if len(proofs)!=20:raise ValueError('all20 domains required for native preflight')
    return max(proofs,key=lambda proof:proof['plane_bytes'])


def make_profile(identity,inputs,*,gdn_eval_wave_max_segments=1,
                 gdn_eval_wave_expanded_charge=False,
                 mlp_materialization_mode='staged_qmm'):
    from .paged_pack_price import _identity
    _identity(identity)
    if inputs.get('case_count')!=400 or inputs.get('client_concurrency')!=20 or inputs.get('generation_max_tokens')!=192:raise ValueError('actual400case sourceinput scope required')
    wave=wave_selection(gdn_eval_wave_max_segments);mlp=mlp_selection(mlp_materialization_mode)
    if type(gdn_eval_wave_expanded_charge) is not bool or gdn_eval_wave_expanded_charge and wave==1:
        raise ValueError('expanded GDN charge requires a grouped wave')
    result={'schema':SCHEMA,'profile_id':'spomin400-nativeN20-candidate-v1','identity':dict(identity),'required_environment':environment(wave),
        'qualified':False,'price_usable':False,'serving_default':False,'memory_budget_bytes':BUDGET,'max_rss_bytes':MAX_RSS,
        'process_headroom_bytes':1<<30,'storage_dtype':'bfloat16','max_tokens':192,'cohort_size':20,'max_active_lanes':20,
        'prefill_layer_lifetime':True,'prefill_eval_block_size':1,'prefill_phase_protocol':'terminal-proved-layer-boundaries',
        'corpus_sha256':inputs['corpus_sha256'],'inputs_sha256':inputs['inputs_sha256'],
        'source_inputs':{r['case_id']:{'domain':r['domain'],'prompt_tokens':r['prompt_tokens'],'token_ids_sha256':token_sha(r['prompt_token_ids'])} for r in inputs['rows']}}
    if wave>1:result['gdn_eval_wave_max_segments']=wave
    if gdn_eval_wave_expanded_charge:result['gdn_eval_wave_expanded_charge']=True
    if mlp!='staged_qmm':result['mlp_materialization_mode']=mlp
    return result


def validate_profile(profile,*,live_identity,source_input_ids,counts,tokens,environment_values):
    from .paged_pack_price import _identity
    from .hybrid_packed_prefill_long import long_environment
    fields={'schema','profile_id','identity','required_environment','qualified','price_usable','serving_default','memory_budget_bytes','max_rss_bytes',
        'process_headroom_bytes','storage_dtype','max_tokens','cohort_size','max_active_lanes','prefill_layer_lifetime','prefill_eval_block_size','prefill_phase_protocol',
        'corpus_sha256','inputs_sha256','source_inputs'}
    wave=wave_selection(profile.get('gdn_eval_wave_max_segments',1)) if type(profile) is dict else 1
    mlp=mlp_selection(profile.get('mlp_materialization_mode','staged_qmm')) if type(profile) is dict else 'staged_qmm'
    expanded=profile.get('gdn_eval_wave_expanded_charge',False) if type(profile) is dict else False
    if type(expanded) is not bool or expanded and wave==1:
        raise ValueError('expanded GDN charge requires a grouped wave')
    if type(profile) is not dict or set(profile)-{'gdn_eval_wave_max_segments','gdn_eval_wave_expanded_charge','mlp_materialization_mode'}!=fields or profile['schema']!=SCHEMA or _identity(profile['identity'])!=_identity(live_identity) or profile['profile_id']!='spomin400-nativeN20-candidate-v1' or any(profile[k] is not False for k in ('qualified','price_usable','serving_default')) or profile['memory_budget_bytes']!=BUDGET or profile['max_rss_bytes']!=MAX_RSS or profile['process_headroom_bytes']!=1<<30 or profile['storage_dtype']!='bfloat16' or profile['max_tokens']!=192 or profile['cohort_size']!=20 or profile['max_active_lanes']!=20 or profile['prefill_layer_lifetime'] is not True or profile['prefill_eval_block_size']!=1 or profile['prefill_phase_protocol']!='terminal-proved-layer-boundaries':
        raise ValueError('N20 source-bound profile scope differs')
    if profile['required_environment']!=environment(wave) or any(environment_values.get(k,'0')!=v for k,v in environment(wave).items()) or (wave==1 and environment_values.get('MLX2_PAGED_GDN_EVAL_WAVE_MAX_SEGMENTS','1')!='1'):raise ValueError('N20 environment differs')
    if type(source_input_ids) is not tuple or len(source_input_ids)!=len(counts) or len(set(source_input_ids))!=len(counts) or len(tokens)!=len(counts) or not 1<=len(counts)<=20:
        raise ValueError('unique actualN source inputs required')
    from .packed_n20_corpus_contract import CORPUS_SHA
    if profile['corpus_sha256']!=CORPUS_SHA or type(profile['source_inputs']) is not dict or len(profile['source_inputs'])!=400:raise ValueError('actual frozen400case corpus required')
    domains=set()
    for key,count,row in zip(source_input_ids,counts,tokens):
        expected=profile['source_inputs'].get(key)
        if type(expected) is not dict or set(expected)!={'domain','prompt_tokens','token_ids_sha256'} or expected['prompt_tokens']!=count or expected['token_ids_sha256']!=token_sha(row) or not 1025<=count<=8192-192:raise ValueError('N20 exact source input tokens differ')
        domains.add(expected['domain'])
    if len(domains)!=1:raise ValueError('one actualdomain per closedN20 cohort required')
    return profile


def load_profile(path,**kwargs):return validate_profile(json.loads(Path(path).read_text()),**kwargs)


def bootstrap_attribution(candidate):
    proof=candidate._packed_prefill_receipt;depth=candidate.native_layer_count;counts=tuple(proof.get('segment_lengths',()))
    expected={COUNTERS[0]:depth,COUNTERS[1]:sum(counts)*depth,COUNTERS[2]:depth}
    if not 1<=len(counts)<=20 or proof.get('physical_counters')!=expected or proof.get('terminal_read_count')!=depth or proof.get('bootstrap_generation')!=0 or proof.get('real_projection_rows')!=sum(counts) or proof.get('prefill_layer_lifetime') is not True or proof.get('public_state_published',False) is not False:raise ValueError('N20 bootstrap physical/lifetime proof differs')
    validate_gdn_wave_attribution(candidate,proof)
    if proof.get('mlp_materialization_mode','staged_qmm')!=getattr(candidate,'_prefill_mlp_materialization_mode','staged_qmm'):
        raise ValueError('selected MLP materialization proof differs')
    if proof.get('mlp_loaded_geometry')!=getattr(candidate,'_prefill_mlp_geometry',None):
        raise ValueError('selected loaded MLP geometry proof differs')
    if proof.get('gdn_evaluation_wave_expanded_charge',False)!=getattr(candidate,'_prefill_gdn_eval_wave_expanded_charge',False):
        raise ValueError('selected expanded GDN charge proof differs')
    arena=candidate.backend.writer.backend
    for name,value in expected.items():
        if int(getattr(arena._native,name)(arena._arena))<value:raise ValueError('N20 live bootstrap counter proof differs')
    if candidate.backend.terminal_successes<depth or candidate.backend.writer.pending_epochs or candidate.backend.writer.ledger.pending_count:raise ValueError('N20 bootstrap terminals pending')
    return {'prefill_mode':'native_packed_prefill','prefill_layout':'real_rows','prefill_cohort_width':len(counts),'bootstrap_generation':0,
        'native_prefill_observed_used':True,'native_prefill_attention_calls':depth,'native_prefill_proof':dict(proof),'state_planes':['kv','gdn'],
        'gdn_eval_wave_max_segments_selected':getattr(candidate,'_prefill_gdn_eval_wave_max_segments',1),
        'gdn_evaluation_waves_observed_used':getattr(candidate,'_prefill_gdn_eval_wave_max_segments',1)>1,
        'gdn_completed_host_wave_evaluations':proof.get('materialized_stage_counts',{}).get('gdn_segment_wave',0),
        'gdn_eval_wave_expanded_charge_selected':getattr(candidate,'_prefill_gdn_eval_wave_expanded_charge',False),
        'mlp_materialization_mode_selected':getattr(candidate,'_prefill_mlp_materialization_mode','staged_qmm'),
        'mlp_loaded_geometry':getattr(candidate,'_prefill_mlp_geometry',None)}



def validate_gdn_wave_attribution(candidate,proof):
    wave=wave_selection(getattr(candidate,'_prefill_gdn_eval_wave_max_segments',1))
    if wave==1:
        if proof.get('gdn_evaluation_wave_max_segments',1)!=1 or proof.get('gdn_evaluation_wave_plan') is not None:
            raise ValueError('unselected grouped GDN receipt')
        return
    plan=getattr(candidate,'_prefill_gdn_wave_plan',None)
    layers=len(candidate.layer_map.recurrent)
    counts=tuple(proof.get('segment_lengths',()))
    if (plan is None or plan.get('max_segments')!=wave or plan.get('segment_lengths')!=counts or
            proof.get('gdn_evaluation_wave_max_segments')!=wave or proof.get('gdn_evaluation_wave_plan')!=plan or
            proof.get('gdn_wave_proof_kind')!='completed_host_evaluations_not_device_dispatches' or
            proof.get('gdn_segment_core_calls')!=layers*len(counts) or
            proof.get('materialized_stage_counts',{}).get('gdn_segment_wave')!=layers*len(plan['groups'])):
        raise ValueError('selected GDN wave actual evaluation/geometry proof differs')


def model_memory_bound(candidate,counts,caps,profile,mx):
    from .hybrid_packed_prefill import n20_stagewise_charge_components
    from mlx.utils import tree_flatten
    candidate._prefill_gdn_eval_wave_max_segments=wave_selection(profile.get('gdn_eval_wave_max_segments',1))
    candidate._prefill_gdn_eval_wave_expanded_charge=profile.get('gdn_eval_wave_expanded_charge',False)
    candidate._prefill_mlp_materialization_mode=mlp_selection(profile.get('mlp_materialization_mode','staged_qmm'))
    if candidate._prefill_mlp_materialization_mode in ('tiled_q4_swiglu','packed_gate_up_qmm'):
        from .dense_mlp_geometry import infer_uniform_dense_glu_geometry
        candidate._prefill_mlp_geometry=infer_uniform_dense_glu_geometry(
            (layer.mlp for layer in candidate.trunk.layers),
            candidate._prefill_mlp_semantics)
    for layer in candidate.trunk.layers:
        layer.mlp._prefill_materialization_mode=candidate._prefill_mlp_materialization_mode
        if candidate._prefill_mlp_materialization_mode in ('tiled_q4_swiglu','packed_gate_up_qmm'):
            layer.mlp._prefill_mlp_semantics=candidate._prefill_mlp_semantics
            layer.mlp._prefill_mlp_geometry=candidate._prefill_mlp_geometry['geometry']
    components=n20_stagewise_charge_components(candidate,counts,caps)
    weight_bytes=sum(int(value.nbytes) for _,value in tree_flatten(candidate.model.parameters()))
    from .packed_prefill_lifetime import gdn_evaluation_wave_plan
    wave_bound=(components['layer_activation_bytes']+components.get('gdn_grouped_wave_extra_bytes',0))
    candidate._prefill_gdn_wave_plan=(gdn_evaluation_wave_plan(counts,wave_bound,candidate._prefill_gdn_eval_wave_max_segments)
        if candidate._prefill_gdn_eval_wave_max_segments>1 else None)
    total=sum(components.values());process_bound=total+weight_bytes+profile['process_headroom_bytes']
    if total>profile['memory_budget_bytes'] or process_bound>profile['max_rss_bytes'] or int(mx.device_info().get('memory_size',0))<64<<30:
        raise MemoryError('N20 full conservative stage/model/RSS bound refuses before allocation')
    return components,weight_bytes,total,process_bound


def estimate_cold_cohort_memory_n(adapter,requests,*,profile_path,manifest_path,mlx_wheel_path,permit_candidate=False):
    """Adapter-owned bound for the complete closed cohort, before allocation."""
    if permit_candidate is not True or type(requests) is not tuple or not 1<=len(requests)<=20:
        raise ValueError('explicit complete N20 memory cohort required')
    from ..adapters.qwen35_paged_n_candidate import (
        Qwen35PagedNCandidate, configure_native_ragged_prompt_lookup)
    from .qwen3_paged_native_backend import NativeQwen3PagedBackend
    from .paged_price_identity import cached_live_price_identity
    import _paged_kv_native as native
    import mlx.core as mx
    candidate=Qwen35PagedNCandidate(adapter.model,None)
    from .dense_mlp_geometry import validate_semantics
    candidate._prefill_mlp_semantics=validate_semantics(
        getattr(type(adapter),'packed_prefill_dense_mlp_semantics',None))
    configure_native_ragged_prompt_lookup(candidate)
    ids=tuple(r[0] for r in requests);tokens=tuple(r[1] for r in requests);counts=tuple(map(len,tokens));caps=tuple(r[2] for r in requests)
    if len(set(ids))!=len(ids) or any(type(r) is not tuple or len(r)!=3 or type(r[1]) is not tuple or any(type(t) is not int or not 0<=t<candidate.args.vocab_size for t in r[1]) or type(r[2]) is not int or not 1<=r[2]<=192 or len(r[1])+r[2]>8192 for r in requests):
        raise ValueError('exact N20 token/cap memory contract differs')
    identity=cached_live_price_identity(Path(manifest_path),Path(mlx_wheel_path),Path(native.__file__).resolve(),adapter_artifact_root=Path(adapter.identity['path']).resolve())
    profile=load_profile(profile_path,live_identity=identity,source_input_ids=ids,counts=counts,tokens=tokens,environment_values=os.environ)
    require_n_capabilities(native,NativeQwen3PagedBackend,counts)
    if candidate.native_dtype_preflight()!='bfloat16' or not str(mx.device_info().get('architecture','')).endswith('s'):raise ValueError('actual BF16/s memory route required')
    components,weights,total,process=model_memory_bound(candidate,counts,caps,profile,mx)
    from .paged_native_arena_geometry import require_arena_storage
    require_arena_storage(native,components['arena_bytes']//2)
    return dict(schema='mlx2.adapter-shared-cohort-memory.v1',route=ROUTE,source_identity=identity,
        source_input_ids=ids,token_ids_sha256=tuple(map(token_sha,tokens)),output_caps=caps,
        runtime_bytes=total,components=components,loaded_parameter_bytes=weights,process_headroom_bytes=profile['process_headroom_bytes'],
        process_bound_bytes=process,max_process_bytes=profile['max_rss_bytes'],profile_id=profile['profile_id'],qualified=False)

def create_cold_packed_hybrid_n(adapter,requests,*,profile,live_identity,source_input_ids,permit_candidate=False,cancelled=lambda:False,phase_boundary=None):
    if permit_candidate is not True:raise ValueError('N20 candidate disabled')
    from ..adapters.qwen35_paged_n_candidate import (
        Qwen35PagedNCandidate, configure_native_ragged_prompt_lookup)
    from ..adapters.qwen35_paged_candidate import HybridBootstrap,clone_recurrent_caches
    from .qwen3_paged_native_backend import NativeQwen3PagedBackend
    candidate=Qwen35PagedNCandidate(adapter.model,None)
    from .dense_mlp_geometry import validate_semantics
    candidate._prefill_mlp_semantics=validate_semantics(
        getattr(type(adapter),'packed_prefill_dense_mlp_semantics',None))
    configure_native_ragged_prompt_lookup(candidate)
    if type(requests) is not tuple or not 1<=len(requests)<=20 or any(type(r) is not tuple or len(r)!=4 for r in requests):raise ValueError('actualN cold lane requests required')
    counts=tuple(len(r[2]) for r in requests);caps=tuple(r[3] for r in requests);uids=tuple(r[0] for r in requests)
    if len(set(uids))!=len(uids) or any(type(uid) is not int or uid<0 for uid in uids) or any(r[1]!=adapter.identity['fingerprint'] or type(r[2]) is not tuple or any(type(t) is not int or not 0<=t<candidate.args.vocab_size for t in r[2]) or type(r[3]) is not int or not 1<=r[3]<=192 or len(r[2])+r[3]>8192 for r in requests):raise ValueError('N20 lane revision/token/cap bounds differ')
    validate_profile(profile,live_identity=live_identity,source_input_ids=source_input_ids,counts=counts,tokens=tuple(r[2] for r in requests),environment_values=os.environ)
    import _paged_kv_native as native
    import mlx.core as mx
    require_n_capabilities(native,NativeQwen3PagedBackend,counts)
    if candidate.native_dtype_preflight()!='bfloat16' or not str(mx.device_info().get('architecture','')).endswith('s'):raise ValueError('N20 requires actualsameBF16/s architecture')
    from .hybrid_packed_prefill import n20_stagewise_charge_components,run_packed_math
    if any(not callable(getattr(layer.mlp,'materialized',None)) or (layer.is_linear and not callable(getattr(layer.linear_attn,'mixed_materialized',None))) for layer in candidate.trunk.layers):raise ValueError('source-bound stagewise model capabilities missing before allocation')
    components,weight_bytes,total,process_bound=model_memory_bound(candidate,counts,caps,profile,mx)
    if candidate._prefill_mlp_materialization_mode=='packed_gate_up_qmm':
        from .models.tensorfold_prefill import PackedProjectionGroup
        for layer in candidate.trunk.layers:
            layer.mlp._prefill_mlp_group=PackedProjectionGroup(
                (layer.mlp.gate_proj,layer.mlp.up_proj))
        candidate._prefill_mlp_packed_group_layers=len(candidate.trunk.layers)
    from .paged_native_arena_geometry import require_arena_storage
    arena_storage_proof=require_arena_storage(native,components['arena_bytes']//2)
    if not callable(phase_boundary):raise ValueError('N20 root-owned bounded phase protocol required')
    def bounded_phase(event):
        active=int(mx.get_active_memory());cache=int(mx.get_cache_memory())
        if active+cache+components['arena_bytes']>profile['max_rss_bytes']:raise MemoryError('MLX allocator plus declared external native arena exceeds48GiB')
        phase_boundary({**event,'loaded_parameter_bytes':weight_bytes,'declared_arena_bytes':components['arena_bytes'],'modeled_process_bound_bytes':process_bound,'mlx_active_bytes':active,'mlx_cache_bytes':cache,'mlx_peak_bytes':int(mx.get_peak_memory()),'memory_scope':'MLX tracked allocator plus charged external native arena; process RSS separately'})
    candidate._prefill_phase_boundary=bounded_phase;candidate._prefill_generation_caps=caps;candidate._prefill_eval_block_size=1
    from . import qwen35_paged_graph_factory as R
    from .paged_kv_pool import PagedKVPool
    from .paged_kv_token import PagedKVTokenOwner,TokenKVProfile
    from .paged_kv_write import NativeWriteBackend,PagedKVWriteOwner
    from .paged_native_atomic_owner import NativeAtomicRequestOwner
    from .paged_gdn_checkpoint import GDNBoundaryCheckpoint
    kv=TokenKVProfile(4,256,'bfloat16');capacity=sum(candidate.native_layer_count*((count+cap+63)//64+2) for count,cap in zip(counts,caps))
    if components['arena_bytes']!=capacity*kv.page_bytes*2:raise ValueError('N20 modeled arena capacity differs')
    R.reap_hybrid_admission_orphans();resources=R.HybridServingResources(total,max(total-components['arena_bytes'],components['decode_scratch_bytes']),profile['memory_budget_bytes'],research_limit_bytes=40<<30)
    began=time.perf_counter()
    try:
        arena=NativeWriteBackend(capacity*kv.page_bytes,mx.default_stream(mx.gpu),permit_candidate=True,storage_dtype='bfloat16');resources._unattached_arena=arena
        writer=PagedKVWriteOwner(PagedKVPool(capacity),arena,page_bytes=kv.page_bytes,permit_candidate=True)
        candidate.backend=NativeQwen3PagedBackend(writer,permit_candidate=True,profile_host=True);resources.candidate=candidate
        candidate._serving_resources=resources;candidate.reserve_serving_scratch=resources.reserve;candidate.reap_serving_resources=resources.reap;candidate._research_staged_graph=True;candidate.bootstrap_generation=0
        candidate._prefill_long_nax=True;candidate._prefill_nax_exact=False
        candidate._serving_prompt_ids_by_uid={r[0]:r[2] for r in requests};candidate._serving_b2=False
        rows=[]
        for _ in requests:
            layers=tuple(PagedKVTokenOwner(writer,kv,permit_candidate=True) for _ in range(candidate.native_layer_count));resources.unattached.extend(layers);rows.append(layers)
        reservation=resources.reserve(total-components['arena_bytes'])
        logits,recurrent,receipt=run_packed_math(candidate,tuple(rows),tuple(r[2] for r in requests),reservation,cancelled=cancelled)
        if cancelled():raise ValueError('N20 cancelled before all-owner construction')
        receipt.update(source_identity=dict(live_identity),bootstrap_charge_components=components,arena_storage_preallocation_proof=arena_storage_proof,loaded_weight_bytes=weight_bytes,conservative_process_bound_bytes=process_bound,public_state_published=False,prefill_layer_lifetime=True,serving_numerical_reference='same_geometry_ordinary_mixed')
        boots=[]
        for index,(r,layers,count) in enumerate(zip(requests,rows,counts)):
            uid,revision,_,_=r;checkpoint=GDNBoundaryCheckpoint(revision,uid,count,0,recurrent[index])
            owner=NativeAtomicRequestOwner(revision,layers,{'gdn':(checkpoint,)},supported_planes=('kv','gdn'),enabled=True,checkpoint_planes=('gdn',),accepted_prefix_checkpoints=True,recurrent_clone=clone_recurrent_caches,lane_id=uid,reuse_private_tail=True)
            resources.owners.append(owner)
            for layer in layers:resources.unattached.remove(layer)
            boots.append(HybridBootstrap(logits[index:index+1],recurrent[index],count,dict(receipt)))
        if cancelled():raise ValueError('N20 cancelled before full handoff')
        receipt['factory_install_seconds']=time.perf_counter()-began
        for boot in boots:boot.receipt.update(receipt)
        candidate._packed_prefill_receipt=receipt;candidate._hybrid_bootstrap_receipts=tuple(b.receipt for b in boots)
        bootstrap_attribution(candidate)
        return tuple(resources.owners),candidate,tuple(boots)
    except BaseException:
        try:resources.abort()
        except BaseException:
            if resources not in R._ORPHANS:R._ORPHANS.append(resources)
        raise
