"""Explicit cold real-row hybrid prefill; no default route or checkpoint widening.

Native matrix reader/grouped multirow writer must be present before allocation.
All tensor imports remain behind explicit execution. Source provenance is in
provenance/hybrid-packed-prefill-orchestrator.json.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

SCHEMA = 'mlx2.hybrid-packed-prefill-research.v1'
PREFILL_ENV = {'MLX2_PAGED_PREFILL_MATRIX': '1',
               'MLX2_PAGED_GROUPED_MULTIROW_WRITE': '1'}


def require_capabilities(native, backend_type, counts=None, *, nax_exact=False):
    if type(nax_exact) is not bool:raise ValueError('exact NAX option must be boolean')
    for name in ('grouped_multirow_write', 'grouped_multirow_write_count',
                 'grouped_multirow_row_count', 'prefill_matrix_capability',
                 'prefill_matrix_dispatch_count'):
        if not callable(getattr(native, name, None)):
            raise ValueError('packed hybrid native capability missing: ' + name)
    if not callable(getattr(backend_type, 'append_packed_multirow', None)):
        raise ValueError('packed hybrid host writer capability missing')
    cap = native.prefill_matrix_capability()
    if (type(cap) is not dict or cap.get('version') != 2 or cap.get('head_dim') != 256 or
        cap.get('storage_dtype') != 'bfloat16' or cap.get('segmented_causal') is not True or
        cap.get('passes') != 2 or cap.get('arithmetic') != 'stock_short_bf16_scores_and_normalized_probabilities' or
        type(cap.get('min_query_count')) is not int or cap['min_query_count'] < 1 or
        cap.get('global_scratch_bytes') != 0 or cap.get('max_causal_end',0) < 129 or
        type(cap.get('max_query_count')) is not int or cap['max_query_count'] < 129 or
        (counts is not None and any(not cap['min_query_count'] <= n <= cap['max_query_count'] for n in counts))):
        raise ValueError('packed hybrid matrix capability geometry differs')
    if nax_exact:
        for name in ('prefill_nax_capability','prefill_nax_score_dispatch_count','prefill_nax_softmax_dispatch_count','prefill_nax_value_dispatch_count'):
            if not callable(getattr(native,name,None)):raise ValueError('exact NAX native capability missing: '+name)
        exact=native.prefill_nax_capability()
        expected={'version':1,'storage_dtype':'bfloat16','head_dim':256,'query_heads':24,'kv_heads':4,
                  'min_query_count':9,'max_query_count':129,'max_causal_end':129,'max_spans':2,
                  'origin_zero':True,'window_zero':True,'architecture':'s','physical_dispatches':3,
                  'scratch_bound_bytes':3195072,'scratch_bytes_per_query_row':24*129*4,'qualified':False}
        if type(exact) is not dict or any(type(exact.get(k)) is not type(v) or exact.get(k)!=v for k,v in expected.items()):
            raise ValueError('exact NAX bounded geometry/charge differs')
        nax_scratch_bytes(counts)
        return exact
    return cap


def evaluation_block_size(value):
    if type(value) is not int or value not in (1,4,16):
        raise ValueError('prefill evaluation block must be1,4or16')
    return value


def nax_scratch_bytes(counts):
    if (type(counts) not in (tuple,list) or not 1<=len(counts)<=2 or
            any(type(n) is not int or not 9<=n<=129 for n in counts)):
        raise ValueError('exact NAX short counts9..129 required')
    return sum(counts)*24*129*4


def validate_requests(requests, revision, vocab_size, *, long_fused=False, research_output_cap20=False):
    if type(research_output_cap20) is not bool or (research_output_cap20 and not long_fused):raise ValueError('cap20 requires explicit long research')
    if (type(requests) is not tuple or len(requests) != 2 or
        any(type(x) is not tuple or len(x) != 4 for x in requests)):
        raise ValueError('two exact cold requests required')
    uids = set()
    for uid, rev, tokens, cap in requests:
        if (type(uid) is not int or uid < 0 or uid in uids or rev != revision or
            type(tokens) is not tuple or not 2 <= len(tokens) <= (8192 if long_fused else 129) or
            any(type(t) is not int or not 0 <= t < vocab_size for t in tokens) or
            type(cap) is not int or not 1 <= cap <= (20 if research_output_cap20 else 4) or
            (long_fused and len(tokens)+cap>8192)):
            raise ValueError('cold lane revision/token/cap geometry differs')
        uids.add(uid)
    return tuple(len(x[2]) for x in requests)


def validate_profile(profile, *, live_identity, counts, environment):
    from .paged_pack_price import _identity
    fields = {'schema','profile_id','identity','storage_dtype','memory_budget_bytes',
              'required_environment','qualified','price_usable','serving_default',
              'context_lengths','q1_simd_stripes','stock_reduction','stock_singleton'}
    if (type(profile) is not dict or not fields<=set(profile)<=fields|{'prefill_nax_exact','prefill_eval_block_size'} or
        type(profile.get('prefill_nax_exact',False)) is not bool or profile['schema'] != SCHEMA or
        not isinstance(profile['profile_id'], str) or not profile['profile_id'] or
        _identity(profile['identity']) != _identity(live_identity) or
        any(profile[k] is not False for k in ('qualified','price_usable','serving_default')) or
        profile['storage_dtype'] != 'bfloat16' or profile['context_lengths'] != list(counts) or
        profile['q1_simd_stripes'] != 16 or type(profile['q1_simd_stripes']) is not int or
        profile['stock_reduction'] is not True or profile['stock_singleton'] is not True or
        type(profile['memory_budget_bytes']) is not int or
        not 0 < profile['memory_budget_bytes'] <= 12 << 30):
        raise ValueError('packed hybrid profile identity/scope differs')
    block_size=evaluation_block_size(profile.get('prefill_eval_block_size',1))
    if block_size!=1 and profile.get('prefill_nax_exact',False) is not True:
        raise ValueError('batched prefill evaluation requires explicit NAX arm')
    expected = packed_environment(profile.get('prefill_nax_exact',False),block_size)
    if profile['required_environment'] != expected or any(environment.get(k,'0') != v for k,v in expected.items()):
        raise ValueError('packed hybrid source environment differs')
    # Only the proven short Q1 handoff, not a new arbitrary-gap continuation.
    if (counts[0] == counts[1] or abs(counts[0]-counts[1]) % 32 or
            any(not 32 <= n <= 124 for n in counts)):
        raise ValueError('packed hybrid short handoff domain differs')
    return profile


def packed_environment(nax_exact=False, eval_block_size=1):
    if type(nax_exact) is not bool:raise ValueError('exact NAX option must be boolean')
    evaluation_block_size(eval_block_size)
    return {**PREFILL_ENV, 'MLX2_PAGED_PREFILL_EVAL_BLOCK_SIZE':str(eval_block_size), 'MLX2_PAGED_PREFILL_NAX_LONG_FUSED':'0', 'MLX2_PAGED_PREFILL_NAX_EXACT':'1' if nax_exact else '0', 'MLX2_PAGED_HYBRID_B2':'1',
        'MLX2_PAGED_Q1_SIMD_TILE':'1','MLX2_PAGED_GROUPED_Q1_WRITE':'1',
        'MLX2_PAGED_PRIVATE_TAIL_REUSE':'1','MLX2_PAGED_Q1_SIMD_STRIPES':'16',
        'MLX2_PAGED_Q1_SPLIT_KV':'0','MLX2_PAGED_Q1_STOCK_REDUCTION':'1',
        'MLX2_PAGED_Q1_STOCK_SINGLETON':'1',
        **{k:'0' for k in ('MLX2_PAGED_Q1_STOCK_LONG','MLX2_PAGED_Q1_STOCK_SDPA',
            'MLX2_PAGED_Q1_INLINE_METADATA','MLX2_PAGED_B2_DEFERRED_EVAL',
            'MLX2_PAGED_B2_DEFERRED_WRITE_EVAL','MLX2_PAGED_GROUPED_SAMPLER',
            'MLX2_PAGED_GROUPED_DIRECT_FENCE')}}


def load_profile(path, *, live_identity, counts, environment):
    profile=json.loads(Path(path).read_text())
    if type(profile) is dict and profile.get('schema')=='mlx2.hybrid-packed-prefill-long-research.v1':
        from .hybrid_packed_prefill_long import validate_long_profile
        return validate_long_profile(profile,live_identity=live_identity,counts=counts,environment=environment)
    return validate_profile(profile, live_identity=live_identity,counts=counts, environment=environment)


def n20_stagewise_charge_components(candidate, counts, caps):
    from .packed_prefill_lifetime import n20_stagewise_charge_components as components
    return components(candidate, counts, caps)


def scratch_components(candidate, counts, *, nax_exact=False, eval_block_size=1, long_fused=False):
    """Conservative four-layer evaluated-block bound, not exact MLX peak pricing.

    Eight copies of all row projection dimensions cover retained activations,
    outputs and GEMM workspace; 64MiB reserves metadata/allocator workspace.
    All final recurrent boundaries plus their successor/container overlap are
    counted separately via three existing full bootstrap bounds per lane.
    """
    if getattr(candidate, '_prefill_packed_n20', False) is True:
        components=n20_stagewise_charge_components(candidate,tuple(counts),candidate._prefill_generation_caps)
        return {k:v for k,v in components.items() if k not in ('arena_bytes','decode_scratch_bytes')}
    a=candidate.args; rows=sum(counts)
    block=evaluation_block_size(eval_block_size)
    simultaneous=min(block,candidate.native_layer_count)
    if len(counts) != 2 or any(type(n) is not int or not 2 <= n <= (8192 if long_fused else 129) for n in counts):
        raise ValueError('bounded two-lane packed scratch required')
    conv=2*a.linear_num_key_heads*a.linear_key_head_dim+a.linear_num_value_heads*a.linear_value_head_dim
    widths=6*a.hidden_size+3*a.intermediate_size+conv+a.linear_num_value_heads*a.linear_value_head_dim+2*a.linear_num_value_heads+(a.num_attention_heads*3+2*a.num_key_value_heads)*256
    return {'retained_projection_bytes':rows*widths*2*8*simultaneous,
            'workspace_bytes':64<<20,
            'recurrent_staging_bytes':sum(3*candidate.bootstrap_staging_bytes(n) for n in counts),
            'native_reader_bytes':nax_scratch_bytes(counts)*simultaneous if nax_exact else 0}


def scratch_bound(candidate, counts, *, nax_exact=False, eval_block_size=1, long_fused=False):
    return sum(scratch_components(candidate,counts,nax_exact=nax_exact,eval_block_size=eval_block_size,long_fused=long_fused).values())


def _runtime():
    import mlx.core as mx
    import _paged_kv_native as native
    from .paged_attention_pack import prepare_staged_token_read
    from .models.precise_ops import gate_sigmoid
    return SimpleNamespace(mx=mx, native=native, prepare_read=prepare_staged_token_read,
                           gate_sigmoid=gate_sigmoid)


def _project_attention(attention, hidden, counts, runtime):
    mx=runtime.mx; total=sum(counts)
    qg=attention.q_proj(hidden).reshape(1,total,attention.num_attention_heads,-1)
    q,gate=mx.split(qg,2,axis=-1)
    q=attention.q_norm(q).transpose(0,2,1,3)
    k=attention.k_norm(attention.k_proj(hidden).reshape(1,total,attention.num_key_value_heads,-1)).transpose(0,2,1,3)
    v=attention.v_proj(hidden).reshape(total,attention.num_key_value_heads,256)
    qs=[];ks=[];start=0
    for count in counts:
        qs.append(attention.rope(q[:,:,start:start+count,:],offset=0).transpose(0,2,1,3).reshape(count,attention.num_attention_heads,256))
        ks.append(attention.rope(k[:,:,start:start+count,:],offset=0).transpose(0,2,1,3).reshape(count,attention.num_key_value_heads,256))
        start+=count
    return mx.concatenate(qs,axis=0),mx.concatenate(ks,axis=0),v,gate.reshape(1,total,-1)


def _require_finite_roots(mx, logits, roots):
    """Validate every materialized output with one scalar-vector evaluation.

    Each original predicate remains present; do not short-circuit later roots.
    The caller retains all roots until this check and terminal proof succeed.
    """
    checks=mx.stack(tuple(mx.all(mx.isfinite(value)) for value in (logits,*roots)))
    mx.eval(checks)
    if not all(checks.tolist()):
        raise ValueError('packed logits/recurrent state nonfinite')


def _counter(runtime, backend, name):
    arena=backend.writer.backend
    return int(getattr(runtime.native,name)(arena._arena))


def run_packed_math(candidate, lane_layers, prompt_ids, reservation, *, cancelled, runtime=None):
    """Both cold lanes private; returns materialized logits/cache boundaries.

    One FA terminal drain per evaluated block bounds lazy graph retention.
    Cancellation is checked only at safe host boundaries; submitted work still
    drains. On failure the caller retains all roots and closes unattached pages.
    """
    runtime=runtime or _runtime();mx=runtime.mx;counts=tuple(map(len,prompt_ids));total=sum(counts)
    packed_n20=getattr(candidate,'_prefill_packed_n20',False)
    layer_lifetime=getattr(candidate,'_prefill_layer_lifetime',False)
    if type(packed_n20) is not bool or type(layer_lifetime) is not bool or packed_n20 != layer_lifetime:
        raise ValueError('N20 requires its explicit materialized lifetime policy')
    exact_nax=getattr(candidate,'_prefill_nax_exact',False)
    long_fused=getattr(candidate,'_prefill_long_nax',False)
    reader_scratch=nax_scratch_bytes(counts) if exact_nax else 0
    block_size=evaluation_block_size(getattr(candidate,'_prefill_eval_block_size',1))
    simultaneous=min(block_size,candidate.native_layer_count)
    if block_size!=1 and exact_nax is not True:raise ValueError('batched prefill evaluation requires NAX arm')
    if packed_n20 and (block_size != 1 or exact_nax or not long_fused or
            not callable(getattr(candidate.backend,'append_packed_multirow_n20',None)) or
            not callable(getattr(candidate.backend,'prepare_read_n20',None))):
        raise ValueError('N20 requires explicit long writer/reader and stagewise capability')
    if type(getattr(reservation,'bytes',None)) is not int or reservation.bytes < scratch_bound(candidate,counts,nax_exact=exact_nax,eval_block_size=block_size,long_fused=long_fused):
        raise MemoryError('complete packed bootstrap reservation required')
    roots=[];uses=[];block_uses=[];block_owners=[];native_roots=[];recurrent=();hidden=None;logits=None;done=False;peak_live=0
    explicit_evals=0;terminal_batches=0;max_simultaneous_reads=0
    host_stage_seconds={}
    def add_host_time(stage,began):
        host_stage_seconds[stage]=host_stage_seconds.get(stage,0.0)+(time.perf_counter()-began)
    stage_owner=None
    if layer_lifetime:
        from .packed_prefill_lifetime import StageMaterialization, gdn_evaluation_wave_plan
        charged=scratch_components(candidate,counts,long_fused=long_fused)
        activation_bound=(charged['layer_activation_bytes']+
                          charged.get('gdn_grouped_wave_extra_bytes',0))
        wave_max=getattr(candidate,'_prefill_gdn_eval_wave_max_segments',1)
        wave_plan=gdn_evaluation_wave_plan(counts,activation_bound,wave_max) if wave_max!=1 else None
        stage_owner=StageMaterialization(mx,activation_bound,gdn_wave_plan=wave_plan)
    phase_boundaries=0
    counters=('grouped_multirow_write_count','grouped_multirow_row_count','prefill_matrix_dispatch_count')
    if packed_n20:
        counters=('grouped_n20_write_count','grouped_n20_row_count','prefill_long_n20_dispatch_count')
    elif long_fused:counters+=('prefill_long_nax_dispatch_count',)
    if exact_nax:counters+=tuple('prefill_nax_'+stage+'_dispatch_count' for stage in ('score','softmax','value'))
    before={k:_counter(runtime,candidate.backend,k) for k in counters}
    try:
        cache_rows=tuple(tuple(candidate.language_model.make_cache()) for _ in prompt_ids)
        recurrent=tuple(tuple(row[i] for i in candidate.layer_map.recurrent) for row in cache_rows)
        for row in recurrent:
            if any(c.cache != [None,None] or c.speculating or c.lengths is not None or c.left_padding is not None for c in row):
                raise ValueError('packed prefill requires empty private unmasked recurrent caches')
        dtype=getattr(mx,candidate.native_dtype_preflight())
        with mx.stream(candidate.backend.writer.backend.stream):
            hidden=candidate.trunk.embed_tokens(mx.array([list(t for lane in prompt_ids for t in lane)]))
            if hidden.shape[:2] != (1,total) or hidden.dtype != dtype:
                raise ValueError('packed embeddings differ from real rows/compute dtype')
            fa=0;gdn=0;start_time=time.perf_counter()
            for index,layer in enumerate(candidate.trunk.layers):
                if cancelled():raise ValueError('packed prefill cancelled before layer submission')
                residual=hidden;normalized=layer.input_layernorm(hidden)
                if layer.is_linear:
                    start=0;parts=[]
                    for count,caches in zip(counts,recurrent):
                        parts.append((1,count,start,caches[gdn],None));start+=count
                    if not callable(getattr(layer.linear_attn,'mixed',None)):
                        raise ValueError('hybrid GDN lacks segmented ordinary mixed math')
                    out=(layer.linear_attn.mixed_materialized(normalized,parts,materialize=stage_owner)
                         if layer_lifetime else layer.linear_attn.mixed(normalized,parts));gdn+=1
                else:
                    attention=layer.self_attn
                    q,k,v,gate=_project_attention(attention,normalized,counts,runtime)
                    if any(x.dtype != dtype for x in (q,k,v)):
                        raise ValueError('packed native projection dtype differs')
                    owners=tuple(layers[fa] for layers in lane_layers)
                    if layer_lifetime:stage_owner('fa_projections',q,k,v,gate)
                    append=(candidate.backend.append_packed_multirow_n20 if packed_n20 else candidate.backend.append_packed_multirow)
                    host_began=time.perf_counter()
                    tickets=append(owners,k,v,counts,permit_candidate=True)
                    add_host_time('attention.writer_submit',host_began)
                    native_roots.append((tickets,q,k,v,owners))
                    prepare=(candidate.backend.prepare_read_n20 if packed_n20 else runtime.prepare_read)
                    host_began=time.perf_counter()
                    use=prepare(owners,counts,query_heads=attention.num_attention_heads,
                        permit_candidate=True,profile_host=candidate.backend.profiling_enabled,
                        **({'long_fused':True} if long_fused and not packed_n20 else {}))
                    add_host_time('attention.read_plan',host_began)
                    uses.append(use);block_uses.append(use);block_owners.extend(owners)
                    max_simultaneous_reads=max(max_simultaneous_reads,len(block_uses))
                    host_began=time.perf_counter()
                    attended=candidate.backend.read_staged(use,q,tickets,scale=attention.scale)
                    add_host_time('attention.read_bind',host_began)
                    native_roots.append(attended)
                    if block_size==1:
                        if stage_owner is not None:stage_owner('attention.read_eval',attended)
                        else:
                            host_began=time.perf_counter();mx.eval(attended)
                            add_host_time('attention.read_eval',host_began)
                        explicit_evals+=1
                    if attended.shape != (total,attention.num_attention_heads,256) or attended.dtype != dtype:
                        raise ValueError('native matrix output shape/dtype differs')
                    if layer_lifetime:
                        host_began=time.perf_counter()
                        proofs=candidate.backend.drain_staged(tuple(owners),(use,))
                        add_host_time('attention.terminal_drain',host_began)
                        if len(proofs)!=1:raise RuntimeError('N20 attention terminal proof absent')
                        terminal_batches+=1;block_uses.clear();block_owners.clear();native_roots.clear();uses.clear()
                    out=attention.o_proj(attended.reshape(1,total,-1)*runtime.gate_sigmoid(gate));fa+=1
                    if layer_lifetime:
                        stage_owner('fa_output',out)
                        del q,k,v,gate,attended,tickets,use,owners,proofs
                hidden=residual+out
                if layer_lifetime:
                    stage_owner('attention_residual',hidden)
                    del residual,normalized,out
                    hidden=hidden+layer.mlp.materialized(layer.post_attention_layernorm(hidden),materialize=stage_owner)
                else:hidden=hidden+layer.mlp(layer.post_attention_layernorm(hidden))
                roots=[leaf for caches in recurrent for c in caches for leaf in c.cache if leaf is not None]
                live=sum(getattr(x,'nbytes',0) for x in ((hidden,*roots) if layer_lifetime else (hidden,residual,normalized,out,*roots)))
                peak_live=max(peak_live,live)
                if live > reservation.bytes:raise MemoryError('packed live tensors exceed reservation')
                if layer_lifetime:
                    host_began=time.perf_counter();mx.eval(hidden,*roots);explicit_evals+=1
                    add_host_time('layer.boundary_eval',host_began)
                    event={'layer_index':index,'next_layer_index':index+1,'layer_kind':'gdn' if layer.is_linear else 'fa',
                        'real_rows':total,'segment_lengths':counts,'materialized':True,'native_terminals_drained':True,
                        'completed_fa_reads':fa,'public_state_published':False}
                    host_began=time.perf_counter()
                    reservation.evaluated_phase(event,writer=candidate.backend.writer,
                        orphaned_reads=candidate.backend._orphaned_reads,
                        callback=getattr(candidate,'_prefill_phase_boundary',None))
                    add_host_time('phase.boundary_and_wait',host_began)
                    phase_boundaries+=1
                    if cancelled():raise ValueError('packed prefill cancelled at materialized phase boundary')
                elif not layer.is_linear and (len(block_uses)>=block_size or fa==candidate.native_layer_count):
                    mx.eval(hidden,*roots);explicit_evals+=1
                    proofs=candidate.backend.drain_staged(tuple(block_owners),tuple(block_uses))
                    if len(proofs)!=len(block_uses):raise RuntimeError('packed evaluated block exact terminal proof absent')
                    terminal_batches+=1;block_uses.clear();block_owners.clear();native_roots.clear()
            final=candidate.trunk.norm(hidden)
            last=[];start=0
            for n in counts:last.append(final[:,start+n-1:start+n,:]);start+=n
            logits=candidate.language_model.logits(mx.concatenate(last,axis=0))[:,0,:]
            host_began=time.perf_counter();mx.eval(logits,*roots);explicit_evals+=1
            add_host_time('head_and_final_state_eval',host_began)
            if block_uses or block_owners or native_roots:raise RuntimeError('packed final graph has undrained native block')
            host_began=time.perf_counter()
            _require_finite_roots(mx,logits,roots)
            add_host_time('finite_validation',host_began)
            writer=candidate.backend.writer
            if writer.pending_epochs or writer.ledger.pending_count or any(o.offset!=n or o._pending or o._stage is not None for layers,n in zip(lane_layers,counts) for o in layers):
                raise RuntimeError('packed cold terminal/offset state incomplete')
            after={k:_counter(runtime,candidate.backend,k) for k in counters}
            delta={k:after[k]-before[k] for k in counters};depth=candidate.native_layer_count
            expected=({'grouped_n20_write_count':depth,'grouped_n20_row_count':depth*total,'prefill_long_n20_dispatch_count':depth}
                if packed_n20 else {'grouped_multirow_write_count':depth,'grouped_multirow_row_count':depth*total,'prefill_matrix_dispatch_count':depth})
            if long_fused and not packed_n20:expected['prefill_long_nax_dispatch_count']=depth
            if exact_nax:expected.update({name:depth for name in counters[3:]})
            if delta != expected:
                raise RuntimeError('packed physical dispatch/row proof differs')
            if (stage_owner is not None and stage_owner.gdn_wave_plan is not None and
                    stage_owner.stages.count('gdn_segment_wave') != gdn*len(stage_owner.gdn_wave_plan['groups'])):
                raise RuntimeError('GDN completed host evaluation wave proof differs')
            done=True
            return logits,recurrent,{'prefill_mode':'packed_hybrid_real_rows','real_projection_rows':total,
                'segment_lengths':counts,'gdn_core_mode':'ordinary_segmented','gdn_segment_core_calls':gdn*len(counts),
                'full_attention_layers':fa,'terminal_read_count':fa,'physical_counters':delta,'model_prefill_seconds':time.perf_counter()-start_time,
                'bootstrap_reserved_bytes':reservation.bytes,'maximum_checked_live_tensor_bytes':peak_live,
                'bootstrap_charge_components':scratch_components(candidate,counts,nax_exact=exact_nax,eval_block_size=block_size,long_fused=long_fused),
                'prefill_long_nax':long_fused,'prefill_nax_exact':exact_nax,'prefill_eval_block_size':block_size,
                'host_evaluation_policy':'materialized_stagewise_per_layer' if layer_lifetime else ('eager_per_fa' if block_size==1 else 'batched_explicit_eval_with_eager_native_async_enqueues'),
                'prefill_packed_n20':packed_n20,'projection_lifetime_policy':'stagewise_terminal_retirement' if layer_lifetime else 'evaluated_fa_blocks',
                'materialized_stage_evaluations':stage_owner.evaluations if stage_owner else 0,
                'materialized_stage_counts':{s:stage_owner.stages.count(s) for s in sorted(set(stage_owner.stages))} if stage_owner else {},
                'diagnostic_stage_seconds':dict(stage_owner.stage_seconds) if stage_owner else {},
                'diagnostic_host_stage_seconds':dict(host_stage_seconds),
                'diagnostic_stage_maximum_evaluated_bytes':dict(stage_owner.stage_maximum_evaluated_bytes) if stage_owner else {},
                'diagnostic_stage_memory_maxima':{s:dict(v) for s,v in stage_owner.stage_memory_maxima.items()} if stage_owner else {},
                'diagnostic_timing_scope':'wall clock around explicit MLX eval and host lifecycle calls; synchronization changes execution and is not performance evidence',
                'gdn_materialized_segment_lengths':counts if stage_owner else (),
                'gdn_evaluation_wave_plan':stage_owner.gdn_wave_plan if stage_owner else None,
                'gdn_evaluation_wave_max_segments':getattr(candidate,'_prefill_gdn_eval_wave_max_segments',1),
                'gdn_evaluation_wave_expanded_charge':getattr(candidate,'_prefill_gdn_eval_wave_expanded_charge',False),
                'gdn_wave_proof_kind':'completed_host_evaluations_not_device_dispatches',
                'mlp_materialization_mode':getattr(candidate,'_prefill_mlp_materialization_mode','staged_qmm'),
                'mlp_loaded_geometry':getattr(candidate,'_prefill_mlp_geometry',None),
                'maximum_materialized_stage_bytes':stage_owner.maximum_evaluated_bytes if stage_owner else 0,
                'evaluated_phase_boundaries':phase_boundaries,
                'explicit_eval_calls':explicit_evals,'terminal_drain_batches':terminal_batches,
                'finite_validation_policy':'all_predicates_one_eval_one_host_vector',
                'finite_validation_predicate_count':1+len(roots),
                'finite_validation_eval_calls':1,'finite_validation_host_extractions':1,
                'maximum_simultaneous_reads':max_simultaneous_reads,
                'native_reader_scratch_bytes':reader_scratch,
                'native_reader_simultaneous_scratch_bytes':reader_scratch*simultaneous,
                'native_reader_scratch_bound_bytes':3195072 if exact_nax else 0,
                'prefill_attention_arithmetic':'nax_fused_stock_long' if long_fused else ('nax_three_stage_stock_short' if exact_nax else 'two_pass_stock_short'),
                'bootstrap_generation':0,'qualified':False,'price_usable':False,'selected':True,'observed_used':True}
    finally:
        if done:reservation.release()
        else:
            reservation.retain_failure_roots((candidate,lane_layers,recurrent,hidden,logits,tuple(roots),tuple(uses),stage_owner.failure_roots if stage_owner else (),tuple(native_roots)))
            # An exception after native enqueue must preserve exact read leases
            # for late callback retirement, even before final evaluation/drain.
            for use in uses:
                if use.state=='submitted':candidate.backend._orphaned_reads[use.lease.epoch]=use


def create_cold_packed_hybrid(adapter, requests, *, profile, live_identity,
                              permit_candidate=False, cancelled=lambda:False):
    if permit_candidate is not True:raise RuntimeError('cold packed hybrid prefill disabled')
    from ..adapters.qwen35_paged_candidate import Qwen35PagedCandidate,HybridBootstrap,clone_recurrent_caches
    from .qwen3_paged_native_backend import NativeQwen3PagedBackend
    long_fused=profile.get('schema')=='mlx2.hybrid-packed-prefill-long-research.v1'
    probe=Qwen35PagedCandidate(adapter.model,None,q1_stripes=16,stock_long=long_fused)
    if probe.bootstrap_generation != 0:raise ValueError('cold hybrid bootstrap generation contract differs')
    counts=validate_requests(requests,adapter.identity['fingerprint'],probe.args.vocab_size,long_fused=long_fused,research_output_cap20=profile.get('research_output_cap20',False))
    if long_fused:
        from .hybrid_packed_prefill_long import validate_long_profile,require_long_capabilities
        validate_long_profile(profile,live_identity=live_identity,counts=counts,environment=os.environ)
    else:validate_profile(profile,live_identity=live_identity,counts=counts,environment=os.environ)
    import _paged_kv_native as native_extension
    exact_nax=profile.get('prefill_nax_exact',False)
    block_size=evaluation_block_size(profile.get('prefill_eval_block_size',1))
    if long_fused:require_long_capabilities(native_extension,NativeQwen3PagedBackend,counts)
    else:require_capabilities(native_extension,NativeQwen3PagedBackend,counts,nax_exact=exact_nax)
    if (exact_nax or long_fused) and (probe.args.num_attention_heads!=24 or probe.args.num_key_value_heads!=4):
        raise ValueError('loaded exact NAX head geometry differs')
    from .paged_hybrid_research_profile import require_hybrid_native_capabilities
    require_hybrid_native_capabilities(profile,native_extension)
    if probe.native_dtype_preflight() != 'bfloat16':raise ValueError('packed hybrid requires same BF16 loaded compute')
    if cancelled():raise ValueError('cold packed cohort cancelled before allocation')
    import mlx.core as mx
    if (exact_nax or long_fused) and not str(mx.device_info().get('architecture','')).endswith('s'):
        raise ValueError('exact NAX requires pinned architecture s before allocation')
    from . import qwen35_paged_graph_factory as resources_module
    from .paged_kv_pool import PagedKVPool
    from .paged_kv_token import PagedKVTokenOwner,TokenKVProfile
    from .paged_kv_write import NativeWriteBackend,PagedKVWriteOwner
    from .paged_native_atomic_owner import NativeAtomicRequestOwner
    from .paged_gdn_checkpoint import GDNBoundaryCheckpoint
    kv=TokenKVProfile(probe.args.num_key_value_heads,256,'bfloat16');depth=probe.native_layer_count
    capacity=sum(depth*((n+cap+63)//64+2) for n,(_,_,_,cap) in zip(counts,requests))
    scratch=scratch_bound(probe,counts,nax_exact=exact_nax,eval_block_size=block_size,long_fused=long_fused);arena_bytes=2*capacity*kv.page_bytes
    resources_module.reap_hybrid_admission_orphans()
    if long_fused:
        from .hybrid_packed_prefill_long import require_host_budget
        require_host_budget(profile,arena_bytes+scratch,mx.device_info())
    decode_scratch=probe.forward_scratch_bytes(tuple(n+cap-2 for n,(_,_,_,cap) in zip(counts,requests))) if long_fused else 0
    resources=resources_module.HybridServingResources(arena_bytes+scratch+decode_scratch,max(scratch,decode_scratch),profile['memory_budget_bytes'],
        **({'research_limit_bytes':40<<30} if long_fused else {}))
    arena=None;candidate=None;began=time.perf_counter()
    try:
        arena=NativeWriteBackend(capacity*kv.page_bytes,mx.default_stream(mx.gpu),permit_candidate=True,storage_dtype='bfloat16')
        resources._unattached_arena=arena
        writer=PagedKVWriteOwner(PagedKVPool(capacity),arena,page_bytes=kv.page_bytes,permit_candidate=True)
        backend=NativeQwen3PagedBackend(writer,permit_candidate=True,profile_host=True)
        candidate=probe;candidate.backend=backend;candidate.bootstrap_generation=0;resources.candidate=candidate
        candidate._prefill_long_nax=long_fused
        candidate._prefill_nax_exact=exact_nax
        candidate._prefill_eval_block_size=block_size
        candidate._research_staged_graph=True;candidate._serving_b2=True
        candidate._serving_stock_reduction=not long_fused;candidate._serving_stock_singleton=not long_fused
        candidate.serving_route='native_hybrid_paged_b2';candidate._b2_profile_id=profile['profile_id'];candidate._serving_resources=resources
        candidate.reserve_serving_scratch=resources.reserve;candidate.reap_serving_resources=resources.reap
        candidate._serving_prompt_ids_by_uid={uid:tokens for uid,_,tokens,_ in requests}
        lane_layers=[]
        for _ in requests:
            row=[]
            for _ in range(depth):
                owner=PagedKVTokenOwner(writer,kv,permit_candidate=True)
                row.append(owner);resources.unattached.append(owner)
            lane_layers.append(tuple(row))
        reservation=resources.reserve(scratch)
        logits,recurrent,receipt=run_packed_math(candidate,tuple(lane_layers),tuple(r[2] for r in requests),
            reservation,cancelled=cancelled)
        if cancelled():raise ValueError('packed cohort cancelled before joint owner construction')
        receipt['source_identity']=dict(live_identity)
        boots=[]
        for lane,((uid,revision,_,_),layers,count) in enumerate(zip(requests,lane_layers,counts)):
            boundary=GDNBoundaryCheckpoint(revision,uid,count,0,recurrent[lane])
            owner=NativeAtomicRequestOwner(revision,layers,{'gdn':(boundary,)},supported_planes=('kv','gdn'),
                enabled=True,checkpoint_planes=('gdn',),recurrent_clone=clone_recurrent_caches,lane_id=uid,reuse_private_tail=True)
            resources.owners.append(owner)
            for layer in layers:resources.unattached.remove(layer)
            boots.append(HybridBootstrap(logits[lane:lane+1],recurrent[lane],count,dict(receipt)))
        if cancelled():raise ValueError('packed cohort cancelled before handoff')
        receipt['factory_install_seconds']=time.perf_counter()-began
        for boot in boots:boot.receipt.update(receipt)
        candidate._hybrid_bootstrap_receipts=tuple(boot.receipt for boot in boots)
        candidate._packed_prefill_receipt=receipt
        return tuple(resources.owners),candidate,tuple(boots)
    except BaseException:
        try:
            resources.abort()
        except BaseException:
            if resources not in resources_module._ORPHANS:resources_module._ORPHANS.append(resources)
        raise
