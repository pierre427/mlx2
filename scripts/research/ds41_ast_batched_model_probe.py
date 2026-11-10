# SPDX-License-Identifier: MIT
"""Owned one-tile actual DSpark/native-M16/resident-expert functional probe.

No generic oracle is admitted. Ordinary native-M drift is classified separately
from the mandatory raw exact batched-M1-versus-resident comparison. Timings are
single diagnostic observations, not controlled performance or qualification.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
from pathlib import Path
from contextlib import contextmanager
import gc
import hashlib
import json
import os
import time

_QUARANTINE=[]


def _write(path,value):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('x')as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def _materialize(result):
    segments=result.transaction.segments
    return [[segment.materialize(node)for segment in segments]for node in range(16)]


def _close(cache):
    if cache is not None:
        for row in cache:row.close()


def _delta(after,before):
    return {k:v-before.get(k,0)for k,v in after.items()if type(v)in(int,float)}



def _align_private_reference(reference,candidate,profile):
    """Align only the intentional private-profile tag; never alter payload."""
    if len(reference)!=len(candidate):raise ValueError('Private selected layer inventory differs')
    changes=[]
    for layer,(left,right)in enumerate(zip(reference,candidate)):
        if getattr(right,'ast_private_profile',None)!=profile:
            raise ValueError('Selected diagnostic cache private profile differs')
        before=getattr(left,'ast_private_profile',None)
        if before not in(None,profile):raise ValueError('Reference carries another private numerical profile')
        left.ast_private_profile=profile
        changes.append(dict(layer=layer,reference_before=before,private_profile=profile,payload_modified=False))
    return changes


def _reuse_components(descriptor,adapter):
    from ds41_resident_prefill_model_gate import _checkpoint
    from mlx2.runtime.gguf.source_ast_batched_moe import implementation_identity
    path=Path(descriptor['path']);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=descriptor['sha256']:raise ValueError('Prior actual MoE component receipt changed')
    value=json.loads(raw)
    helper=Path(__file__).with_name('ds41_ast_batched_moe_components.py')
    if (value.get('schema')!='mlx2.ds41-ast-batched-moe-components.v1'
            or value.get('passed')is not True or value.get('status')!='passed'
            or value.get('checkpoint')!=_checkpoint(adapter)
            or value.get('implementation_sha256')!=implementation_identity()
            or value.get('helper_sha256')!=hashlib.sha256(helper.read_bytes()).hexdigest()):
        raise ValueError('Prior actual MoE component module/artifact identity differs')
    cases=value.get('cases',[]);controls=value.get('controls',[])
    if(set(v.get('label')for v in cases)!={'original_shared16','q1_shared16','mixed_shared16','globally_distinct96'}
       or len(cases)!=4 or any(v.get('passed')is not True or v.get('finite')is not True
            or any(v.get('raw',{}).get(k,{}).get('equal')is not True for k in('middle','down','output'))for v in cases)
       or set(v.get('label')for v in controls)!={'missing_cohort','invalid_id','duplicate_id'}
       or len(controls)!=3 or any(v.get('passed')is not True for v in controls)):
        raise ValueError('Prior actual MoE component parity/refusal gates incomplete')
    return dict(reused=True,path=str(path.resolve()),sha256=descriptor['sha256'],
        checkpoint=value['checkpoint'],implementation_sha256=value['implementation_sha256'],
        helper_sha256=value['helper_sha256'],valid_cases=4,refusal_controls=3,
        original_source_snapshot_sha256=value.get('source_snapshot_sha256'),qualified=False)



def _probe_profile(forward_law,require_ordinary_exact):
    from mlx2.adapters.deepseek_v41_ast_batched_target import PROFILES,M1_FORWARD_LAW
    if forward_law not in PROFILES or type(require_ordinary_exact)is not bool:
        raise ValueError('Invalid batched probe forward law/exact gate')
    if require_ordinary_exact and forward_law!=M1_FORWARD_LAW:
        raise ValueError('Ordinary exact gate requires explicit M1 batch-axis candidate law')
    return PROFILES[forward_law]


def _first_ordinary_divergence(rows):
    for row in rows:
        if row['exact']:continue
        bad_layers=[]
        for layer in row['cache']['layers']:
            planes=[name for name,value in layer['planes'].items()if not value['equal']]
            if planes or not layer['capacity_equal']:
                bad_layers.append(dict(layer=layer['layer'],planes=planes,capacity_equal=layer['capacity_equal']))
        return dict(node=row['node'],logits_exact=row['logits']['equal'],features_exact=row['features']['equal'],
            identity_topology_frontier_exact=row.get('identity_topology_frontier_exact'),
            first_divergent_cache_layer=None if not bad_layers else bad_layers[0],
            private_profile_metadata_is_intentional=True)
    return None



def _prompt_ids(adapter,prefix_tokens=None):
    """Explicit ordinary prefix shape; default short Hello stays byte-for-byte."""
    request={'messages':[{'role':'user','content':'Reply with exactly Hello.'}],'enable_thinking':False}
    encoded=adapter.tokenizer.encode(adapter.render_prompt(request),add_special_tokens=False)
    suffix=list(getattr(encoded,'ids',encoded))
    if prefix_tokens is None:return suffix
    limit=adapter.model.context_limit-6
    if type(prefix_tokens)is not int or not len(suffix)<=prefix_tokens<=limit:
        raise ValueError('Batched probe explicit prefix is outside target ancestry capacity')
    encoded=adapter.tokenizer.encode('A neutral memory test uses numbered lines and ordinary words. ',add_special_tokens=False)
    filler=list(getattr(encoded,'ids',encoded))
    if not filler:raise ValueError('Batched probe filler produced no tokens')
    count=prefix_tokens-len(suffix)
    return(filler*((count+len(filler)-1)//len(filler)))[:count]+suffix


def _setup_pause(live,seconds,log,phase,*,sleep=time.sleep):
    if type(seconds)not in(int,float)or not 0<=seconds<=3:raise ValueError('Setup pacing must be0..3seconds')
    if not seconds:return
    live();started=time.perf_counter();sleep(seconds)
    log.append(dict(phase=phase,fixed_pause_seconds=seconds,actual_seconds=time.perf_counter()-started,inside_resident_timer=False))
    live()


def _prefill_prefix(model,tokens,cache,capture_layers,live,*,chunk_size=None,xp=None,pause_seconds=0,pacing_log=None,sleep=time.sleep):
    """Bound setup row width; every tap keeps its actual absolute position."""
    if xp is None:
        import mlx.core as xp
    if not tokens or not callable(live):raise ValueError('Prefix setup needs tokens and live owner')
    if chunk_size is not None and(type(chunk_size)is not int or not 1<=chunk_size<=1024):
        raise ValueError('Prefix setup chunk must be rows1..1024')
    if chunk_size is None and len(tokens)>1024:
        raise ValueError('Whole prefix setup exceeds1024row advice; use explicit bounded chunks')
    initial=cache[0].offset;parts=[];spans=[];started=time.perf_counter()
    pacing_log=[]if pacing_log is None else pacing_log
    def evaluate(first,last,body):
        if any(row.offset!=initial+first for row in cache):raise ValueError('Prefix cache frontier lost contiguous position')
        live();inputs=xp.array([tokens[first:last]],xp.uint32);begin=time.perf_counter()
        if body:logits=None;features=model.prefill_body(inputs,cache,capture_layers)
        else:logits,features=model.forward_with_taps(inputs,cache,capture_layers)
        xp.eval(logits,features,[row.state for row in cache]);seconds=time.perf_counter()-begin
        if features.shape[:2]!=(1,last-first)or any(row.offset!=initial+last for row in cache):
            raise ValueError('Prefix taps/cache do not cover actual input rows')
        parts.append(features);spans.append(dict(start=initial+first,stop=initial+last,rows=last-first,
            body_only=body,seconds=seconds));return logits
    if chunk_size is None:logits=evaluate(0,len(tokens),False);profile='ordinary_whole_prefix_forward'
    else:
        for first in range(0,len(tokens)-1,chunk_size):
            last=min(len(tokens)-1,first+chunk_size);evaluate(first,last,True)
            _setup_pause(live,pause_seconds,pacing_log,f'prefill_body:{initial+first}-{initial+last}',sleep=sleep)
        logits=evaluate(len(tokens)-1,len(tokens),False);profile='ordinary_body_chunks_then_B1_final'
    features=parts[0]if len(parts)==1 else xp.concatenate(parts,axis=1);xp.eval(features);live()
    return logits,features,dict(profile=profile,chunk_size=chunk_size,spans=spans,
        start_position=initial,end_position=initial+len(tokens),feature_rows=features.shape[1],
        final_forward_rows=spans[-1]['rows'],seconds=time.perf_counter()-started,
        partition_equivalence_to_whole_prefix_claim=False,setup_only=True)


@contextmanager
def _preparation_policy(model,policy):
    if policy not in('reference','descendant_reuse_v1'):
        raise ValueError('Unknown batched KV preparation policy')
    if getattr(model,'_mlx2_ds41_ast_active',None)is not None:
        raise ValueError('Cannot change KV preparation during an active AST transaction')
    missing=object();previous=getattr(model,'source_ast_kv_preparation_policy',missing)
    model.source_ast_kv_preparation_policy=policy
    try:yield
    finally:
        if previous is missing:del model.source_ast_kv_preparation_policy
        else:model.source_ast_kv_preparation_policy=previous


def _preparation_metrics(receipt,policy):
    value=receipt.get('selected_kv_preparation')
    if(not isinstance(value,dict)or value.get('policy')!=policy
            or any(type(value.get(k))is not int or value[k]<0 for k in('builds','hits','built_payload_bytes'))
            or value.get('measured_device_traffic')is not False):
        raise ValueError('Batched KV preparation metrics incomplete or policy differs')
    return dict(value)


def _preparation_order(abba):
    if type(abba)is not bool:raise ValueError('Paired preparation ABBA flag must beboolean')
    return('reference','descendant_reuse_v1','descendant_reuse_v1','reference')if abba else('reference','descendant_reuse_v1')


def _paired_preparation_tiles(adapter,prefix,anchor,paths,discovery,discovery_states,live,*,abba=False):
    """Warm both arms once, then measure one paired sequence; all raw gates each."""
    import mlx.core as mx
    from mlx2.adapters.deepseek_v41_ast_batched_target import execute_batched_tree,RESIDENT_EXPERT_LAW
    from ds41_resident_prefill_model_gate import _array_compare,_cache_compare,_counters
    model=adapter.model;rows=[];warmups=[];order=_preparation_order(abba)
    # One warm per distinct arm BEFORE timed AB/ABBA. Every execute has its own
    # transaction-local preparation dictionary; no cross-tile cache hit claim.
    for policy in('reference','descendant_reuse_v1'):
        warm=None;drained=True
        with _preparation_policy(model,policy):
            try:
                live();drained=False
                warm=execute_batched_tree(model,prefix,anchor,paths,expert_law=RESIDENT_EXPERT_LAW,forward_law='fully_batched_m1_v1')
                mx.eval(warm.logits,warm.features);drained=True
                checks=dict(logits=_array_compare(discovery.logits,warm.logits,mx),features=_array_compare(discovery.features,warm.features,mx))
                metrics=_preparation_metrics(warm.receipt,policy)
                warmups.append(dict(policy=policy,checks=checks,preparation=metrics))
                if not all(v['equal']for v in checks.values()):raise RuntimeError('KV preparation warm differs from ordinary-exact discovery')
            finally:
                if not drained:
                    try:mx.synchronize();drained=True
                    except BaseException:_QUARANTINE.append((adapter,prefix,discovery,discovery_states,warm))
                if drained and warm is not None:warm.transaction.abort()
    for sequence,policy in enumerate(order):
        tile=None;states=None;drained=True
        with _preparation_policy(model,policy):
            try:
                live();before=_counters(adapter);drained=False;started=time.perf_counter()
                tile=execute_batched_tree(model,prefix,anchor,paths,expert_law=RESIDENT_EXPERT_LAW,forward_law='fully_batched_m1_v1')
                mx.eval(tile.logits,tile.features);seconds=time.perf_counter()-started;drained=True;live()
                delta=_delta(_counters(adapter),before)
                zero=all(delta.get(k,0)==0 for k in('expert_bytes_read','expert_uploads','engram_bytes_read','host_route_materializations','ordinary_replays'))
                metrics=_preparation_metrics(tile.receipt,policy)
                drained=False;states=_materialize(tile);mx.eval([[v.state for v in cache]for cache in states]);drained=True
                nodes=[]
                for node in range(16):
                    checks=dict(logits=_array_compare(discovery.logits[:,node:node+1],tile.logits[:,node:node+1],mx),
                        features=_array_compare(discovery.features[:,node:node+1],tile.features[:,node:node+1],mx),
                        cache=_cache_compare(discovery_states[node],states[node],mx))
                    nodes.append(dict(node=node,checks=checks,exact=all(v['equal']for v in checks.values())))
                row=dict(sequence=sequence,policy=policy,seconds=seconds,counter_delta=delta,
                    zero_io_host_routes_or_replay=zero,preparation=metrics,nodes=nodes,all_nodes_exact=all(v['exact']for v in nodes),
                    cross_tile_preparation_reuse=False,rate_eligible=False,scope='one measured tile; sameprefix/paths/bank; allstatechecks outside timer')
                rows.append(row)
                if not(zero and row['all_nodes_exact']):raise RuntimeError('Paired KV preparation raw/noIO gate failed')
            finally:
                if not drained:
                    try:mx.synchronize();drained=True
                    except BaseException:_QUARANTINE.append((adapter,prefix,discovery,discovery_states,tile,states))
                if drained:
                    if tile is not None:tile.transaction.abort()
                    for cache in states or():_close(cache)
    return dict(arms=rows,warmups=warmups,all_raw_exact=all(v['all_nodes_exact']for v in rows),
        all_no_io=all(v['zero_io_host_routes_or_replay']for v in rows),
        reuse_hit_observed=any(v['preparation']['hits']>0 for v in rows if v['policy']=='descendant_reuse_v1'),
        measured_device_traffic=False,controlled_performance=False,source_model_identity_unchanged=True,
        order=list(order),timed_samples_per_arm=2 if abba else 1,independent_ABBA_repetitions=1 if abba else 0)

def run(adapter,unit,check_live,*,prior_moe_components=None,timeline_tiles=0,
        forward_law='fully_batched_v1',require_ordinary_exact=False,prefix_tokens=None,paired_kv_preparation=False,paired_kv_abba=False,prefill_chunk_size=None,setup_pause_seconds=0,save_prefix_checkpoint=False,stage_profile=False,paired_projection=False):
    import mlx.core as mx
    if not callable(check_live):raise ValueError('Batched model probe requires owned live guard')
    if mx.default_device()!=mx.gpu:raise ValueError('Batched model probe requires parent-owned GPU')
    from mlx2.runtime.cow_cache import snapshot_prompt_cache_descriptors
    from mlx2.adapters.deepseek_v41_dspark import load_source_dspark
    from mlx2.adapters.deepseek_v41_ast_batched_target import execute_batched_tree,RESIDENT_EXPERT_LAW
    from mlx2.adapters.deepseek_v41_source_prefill_bank import build_prefill_bank
    from mlx2.adapters.deepseek_v41_source_bank_session import PreparedBankSession
    from ds41_resident_prefill_model_gate import _array_compare,_cache_compare,_park_experts,_checkpoint,_counters
    from ds41_ast_batched_moe_components import run as moe_components
    def live():
        if check_live()is False:raise RuntimeError('Batched model probe ownership lost')
        return True
    _preparation_order(paired_kv_abba)
    if paired_kv_abba and not paired_kv_preparation:raise ValueError('ABBA requires paired KV preparation')
    if type(paired_kv_preparation)is not bool or paired_kv_preparation and not require_ordinary_exact:
        raise ValueError('Paired KV preparation requires explicit ordinary-exact M1 gate')
    if type(save_prefix_checkpoint)is not bool:raise ValueError('Private prefix checkpoint flag must beboolean')
    if type(stage_profile)is not bool or stage_profile and not require_ordinary_exact:
        raise ValueError('Stage profile requires explicit ordinary-exact M1 gate')
    if type(paired_projection)is not bool or paired_projection and not require_ordinary_exact:
        raise ValueError('Paired projection probe requires explicit ordinary-exact M1 gate')
    if type(setup_pause_seconds)not in(int,float)or not 0<=setup_pause_seconds<=3:raise ValueError('Setup pacing must be0..3seconds')
    live();PROFILE=_probe_profile(forward_law,require_ordinary_exact)
    if timeline_tiles and forward_law!='fully_batched_v1':raise ValueError('This timeline helper binds nativeM16 only')
    if type(timeline_tiles)is not int or timeline_tiles not in(0,8,16):raise ValueError('Timeline tile count must be0,8or16')
    unit=Path(unit);path=unit/'ast-batched-model-probe.json'
    if path.exists()or path.with_name(path.name+'.tmp').exists():raise FileExistsError('Batched model evidence exists')
    w=adapter.weights;store=w.expert_stream;model=adapter.model
    if getattr(w,'source_resident_bank_session',None)is not None:raise ValueError('Batched model probe requires idle unattached bank')
    saved=(w.source_resident_gather_policy,w.source_resident_prefill_policy)
    acquire=store.acquire;persist=getattr(store,'persist_atlas',None);prior_pins=getattr(store,'_resident_bank_pinned_keys',frozenset())
    result=dict(schema='mlx2.ds41-ast-batched-model-probe.v1',checkpoint=_checkpoint(adapter),
        target_profile=PROFILE,forward_law=forward_law,require_ordinary_exact=require_ordinary_exact,
        resident_law=RESIDENT_EXPERT_LAW,status='moe_components',
        qualified=False,rate_eligible=False,APCv2_publication=False,exact_oracle_admitted=False,
        whole_model_profile=('M1 batch-axis dense/attention candidate with residentM1 expert arithmetic'if require_ordinary_exact
            else'nativeM16 dense/attention with privateM1 expert arithmetic'),
        ordinary_comparison=('REQUIRED all16nodes rawlogits/features/sixplanes +identity/topology/frontier/capacity; private revision differs intentionally'if require_ordinary_exact
            else'diagnose nativeM profile drift; not an exactness prerequisite for this explicit functional profile'))
    sealed=unit/'spec.json'
    if sealed.exists():
        spec=json.loads(sealed.read_text());result.update(source_revision=spec.get('source_revision'),source_snapshot_sha256=spec.get('source_snapshot_sha256'))
    pacing_log=[];result['setup_pacing']=dict(fixed_seconds=setup_pause_seconds,events=pacing_log,resident_timers_paced=False,ABBA_timers_paced=False)
    _write(path,result)
    prefix=prefix_control=committed=continuation_control=None;head=bank=session=None
    discovery=warm=candidate=None;discovery_states=candidate_states=None;ordinary_states=[];drained=True;pin_active=False
    try:
        result['moe_components']=(_reuse_components(prior_moe_components,adapter)if prior_moe_components is not None
            else moe_components(adapter,unit,live))
        w.source_resident_gather_policy=w.source_resident_prefill_policy='reference'
        store.persist_atlas=lambda:None
        ids=_prompt_ids(adapter,prefix_tokens);live();prefill_started=time.perf_counter()
        binding=dict(schema='mlx2.ds41-ast-private-prefix-binding.v1',checkpoint=result['checkpoint'],
            source_revision=result.get('source_revision'),source_snapshot_sha256=result.get('source_snapshot_sha256'),
            target_identity=model.identity,forward_law=forward_law,numerical_profile=PROFILE,prefix_tokens=len(ids),
            prompt_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),capture_layers=[37,38,39],
            feature_width=3*model.args.hidden_size,vocab_size=model.args.vocab_size,
            prefill_profile='ordinary_body_chunks_then_B1_final'if prefill_chunk_size is not None else'ordinary_whole_prefix_forward',
            prefill_chunk_size=prefill_chunk_size,private_prefix_not_APCv2=True)
        resume_raw=os.environ.get('MLX2_BENCH_DISCOVERY_CHECKPOINT')
        if resume_raw:
            from ds41_ast_private_prefix import load_prefix
            resume=json.loads(resume_raw);drained=False
            prefix,seed_logits,seed_features,prefill_partition=load_prefix(resume['path'],mx,model,binding,live,
                expected_manifest_sha256=resume['manifest_sha256']);drained=True
            result.update(prefix_resumed=True,resume=resume,prefix_restore_seconds=time.perf_counter()-prefill_started,
                original_prefill_seconds=prefill_partition['seconds'],prefill_seconds=0.)
        else:
            prefix=model.make_cache();drained=False
            seed_logits,seed_features,prefill_partition=_prefill_prefix(model,ids,prefix,(37,38,39),live,
                chunk_size=prefill_chunk_size,pause_seconds=setup_pause_seconds,pacing_log=pacing_log)
            drained=True;result.update(prefix_resumed=False,prefill_seconds=time.perf_counter()-prefill_started)
            if save_prefix_checkpoint:
                from ds41_ast_private_prefix import save_prefix
                result['private_prefix_checkpoint']=save_prefix(unit/'private-prefix-checkpoint',mx,prefix,
                    seed_logits,seed_features,binding,prefill_partition,live)
        result.update(prefill_partition=prefill_partition,prefix_prompt_tokens=len(ids),
            prefix_prompt_ids_sha256=binding['prompt_ids_sha256'],
            prefill_scope='explicit ordinary partitioned setup; targettaps positions preserved; separate from residenttile timing')
        _write(path,result)
        anchor=int(mx.argmax(seed_logits[0,-1]).item())
        prefix_control,_,_=snapshot_prompt_cache_descriptors(prefix,require_committed=True)
        head=load_source_dspark(model,expert_cache_bytes=512<<20,check_live=live)
        history=head.make_history();head.record_features(seed_features,0,history)
        if history.next_position!=len(ids):raise ValueError('DSpark history lost target prefix feature positions')
        result['dspark_feature_history']=dict(start_position=history.start_position,end_position=history.next_position,
            target_feature_rows=seed_features.shape[1],record_position=0)
        paths,confidence=head.propose_paths(anchor,history);mx.eval(seed_features);live()
        result.update(status='ordinary_path_controls',prompt_ids=ids,anchor=anchor,paths=[list(p)for p in paths],
            draft_confidence=[list(p)for p in confidence],dspark=head.receipt(),prefix_frontier=prefix[0].offset)
        head.close();head=None;_write(path,result)
        # Native-M arithmetic is distinct. Ordinary path rows classify drift;
        # they do not authorize exact generic AST/APCv2 publication.
        ordinary_logits={};ordinary_features={};ordinary_by_node={}
        for rank,branch in enumerate(paths):
            local,_,_=snapshot_prompt_cache_descriptors(prefix,require_committed=True)
            ordinary_states.append(local)
            for depth,token in enumerate((anchor,*branch)):
                live();drained=False
                logits,features=model.forward_with_taps(mx.array([[token]],mx.uint32),local,(37,38,39))
                mx.eval(logits,features,[r.state for r in local]);drained=True
                _setup_pause(live,setup_pause_seconds,pacing_log,f'ordinary_reference:branch{rank}:depth{depth}')
                node=0 if depth==0 else 1+rank*5+depth-1
                if node not in ordinary_by_node:
                    ordinary_logits[node]=logits;ordinary_features[node]=features
                    ordinary_by_node[node],_,_=snapshot_prompt_cache_descriptors(local,require_committed=True)
                    ordinary_states.append(ordinary_by_node[node])
        result['ordinary_path_controls_completed']=18;result['status']='batched_m1_discovery';_write(path,result)
        seen=set()
        def observed(prefix_name,expert):
            seen.add((int(prefix_name.split('.')[1]),int(expert)))
            return acquire(prefix_name,expert)
        store.acquire=observed;live();drained=False;started=time.perf_counter()
        discovery=execute_batched_tree(model,prefix,anchor,paths,expert_law='source_ast_m1_rows_v1',forward_law=forward_law)
        mx.eval(discovery.logits,discovery.features);drained=True;store.acquire=acquire
        drained=False;discovery_states=_materialize(discovery)
        mx.eval([[r.state for r in rows]for rows in discovery_states]);drained=True;live()
        ordinary=[]
        for node in range(16):
            logits=_array_compare(ordinary_logits[node],discovery.logits[:,node:node+1],mx)
            features=_array_compare(ordinary_features[node],discovery.features[:,node:node+1],mx)
            state=_cache_compare(ordinary_by_node[node],discovery_states[node],mx)
            payload_exact=all(row['capacity_equal']and all(v['equal']for v in row['planes'].values())for row in state['layers'])
            topology_exact=all(tuple(a.meta_state[1:])==tuple(b.meta_state[1:])for a,b in zip(ordinary_by_node[node],discovery_states[node]))
            ordinary.append(dict(node=node,logits=logits,features=features,cache=state,
                expected_private_profile_metadata=True,identity_topology_frontier_exact=topology_exact,
                exact=logits['equal']and features['equal']and payload_exact and topology_exact))
        result.update(discovery_seconds_including_streaming=time.perf_counter()-started,
            discovery_receipt=discovery.receipt,ordinary_nodes=ordinary,
            ordinary_numerical_classification='raw_payload_exact_private_profile_metadata'if all(r['exact']for r in ordinary)else'nativeM_profile_drift',
            observed_experts=[list(k)for k in sorted(seen)],observed_expert_count=len(seen),
            observed_experts_sha256=hashlib.sha256(json.dumps(sorted(seen)).encode()).hexdigest())
        result['ordinary_all16_nodes_raw_exact']=all(row['exact']for row in ordinary)
        result['ordinary_first_divergence']=_first_ordinary_divergence(ordinary)
        discovery.transaction.abort();_write(path,result)
        if require_ordinary_exact and not result['ordinary_all16_nodes_raw_exact']:
            raise RuntimeError('M1 batch-axis candidate differs from ordinary reference; first node/cachelayer retained')
        result['status']='closed_cohort_preload';_write(path,result)
        _park_experts(store,live);gc.collect();mx.clear_cache();live()
        payload=sum(18_800_640 if store.kind(k)=='original'else 6_635_520 for k in seen)
        if payload+18_800_640>store.maximum_cache_bytes:raise ValueError('Actual AST cohort exceeds configured maximum')
        store.set_adaptive_cache_target(payload+18_800_640,working_set_experts=6)
        store._resident_bank_pinned_keys=prior_pins|frozenset(seen);pin_active=True
        table={key:(p,i)for p,values in store.tables.items()for i,key in enumerate(values)if key in seen}
        if set(table)!=seen:raise ValueError('AST cohort is not in registered provider tables')
        for key in sorted(seen):
            live()
            with store.window():store.acquire(*table[key])
        result['preload']=dict(payload_bytes=store._cache_bytes,target=store.adaptive_state,
            actual_expected_payload_bytes=payload,keys=len(store._cache),configured_maximum_bytes=store.maximum_cache_bytes,
            allocator_active_bytes=mx.get_active_memory(),allocator_cached_bytes=mx.get_cache_memory())
        _write(path,result)
        if set(store._cache)!=seen or store._cache_bytes!=payload:raise RuntimeError('Closed AST cohort could not be retained')
        bank=build_prefill_bank(mx,store,tuple(sorted(seen)),ownercheck=live,
            generation=getattr(store,'_resident_bank_generation',0)+1,budget_bytes=75_000_000_000,max_bank_bytes=3<<30)
        pin_active=False;w.source_resident_gather_policy='resident_bank_v1'
        session=PreparedBankSession(w,bank,ownercheck=live);w.source_resident_bank_session=session
        result['bank']=bank.receipt();result['status']='resident_warm';_write(path,result)
        live();drained=False;warm=execute_batched_tree(model,prefix,anchor,paths,expert_law=RESIDENT_EXPERT_LAW,forward_law=forward_law)
        mx.eval(warm.logits,warm.features);drained=True;live()
        result['warm']=dict(receipt=warm.receipt,logits=_array_compare(discovery.logits,warm.logits,mx),
            features=_array_compare(discovery.features,warm.features,mx))
        if not(result['warm']['logits']['equal']and result['warm']['features']['equal']):raise RuntimeError('Resident AST warm tile differs from nativeM16/M1 discovery')
        warm.transaction.abort();warm=None;result['status']='resident_measured';_write(path,result)
        if timeline_tiles:
            from ds41_ast_batched_timeline import run as capture_tiles
            result['timeline']=capture_tiles(adapter,unit,live,prefix,anchor,paths,discovery,discovery_states,tiles=timeline_tiles)
            _write(path,result)
        before=_counters(adapter);live();drained=False;started=time.perf_counter()
        candidate=execute_batched_tree(model,prefix,anchor,paths,expert_law=RESIDENT_EXPERT_LAW,forward_law=forward_law)
        mx.eval(candidate.logits,candidate.features);seconds=time.perf_counter()-started;drained=True;live()
        delta=_delta(_counters(adapter),before)
        zero=all(delta.get(k,0)==0 for k in('expert_bytes_read','expert_uploads','engram_bytes_read','host_route_materializations','ordinary_replays'))
        drained=False;candidate_states=_materialize(candidate)
        mx.eval([[r.state for r in rows]for rows in candidate_states]);drained=True
        comparisons=[]
        for node in range(16):
            logits=_array_compare(discovery.logits[:,node:node+1],candidate.logits[:,node:node+1],mx)
            features=_array_compare(discovery.features[:,node:node+1],candidate.features[:,node:node+1],mx)
            state=_cache_compare(discovery_states[node],candidate_states[node],mx)
            comparisons.append(dict(node=node,logits=logits,features=features,cache=state,
                exact=logits['equal']and features['equal']and state['equal']))
        result['candidate']=dict(seconds=seconds,counter_delta=delta,zero_io_host_routes_or_replay=zero,
            nodes=comparisons,all_nodes_exact=all(v['exact']for v in comparisons),receipt=candidate.receipt,
            rate_eligible=False,scope='one16rowtile after untimed warm; exploratory, not controlled throughput')
        for rows in candidate_states:_close(rows)
        candidate_states=None
        if not(zero and result['candidate']['all_nodes_exact']):raise RuntimeError('Resident AST exact/noIO tile gate failed')
        # Commit remains private to the probe, never an APC/generic exact oracle.
        selected=(0,6,7,8);committed=candidate.transaction.diagnostic_cache(selected)
        candidate.transaction.abort()
        if stage_profile:
            from ds41_ast_exact_stage_profile import run as profile_exact_stages
            result['status']='synchronized_stage_profile';_write(path,result);drained=False
            result['stage_profile']=profile_exact_stages(adapter,unit,live,prefix,anchor,paths,discovery,discovery_states)
            drained=True;_write(path,result)
        if paired_projection:
            from ds41_ast_paired_projection_probe import run as profile_paired_projection
            result['status']='paired_projection_probe';_write(path,result);drained=False
            result['paired_projection']=profile_paired_projection(adapter,unit,live,prefix,anchor,paths,discovery,discovery_states)
            drained=True;_write(path,result)
        if paired_kv_preparation:
            result['paired_kv_preparation']=_paired_preparation_tiles(adapter,prefix,anchor,paths,discovery,discovery_states,live,abba=paired_kv_abba)
            _write(path,result)
        # Flagged private states cannot be cloned through publication/serialization
        # APIs. Both diagnostic materializations are independent owned wrappers.
        continuation_control=discovery_states[8]
        alignment=_align_private_reference(continuation_control,committed,PROFILE)
        result['private_selected_state']=dict(path=list(selected),intentional_profile_metadata_alignment=alignment,
            cache=_cache_compare(committed,continuation_control,mx),
            prefix_unchanged=_cache_compare(prefix,prefix_control,mx),APCv2_publication=False)
        if not(result['private_selected_state']['cache']['equal']and result['private_selected_state']['prefix_unchanged']['equal']):raise RuntimeError('Private AST selected-path/isolation gate failed')
        session.close();session=bank=None;w.source_resident_gather_policy='reference'
        token=paths[1][3];live();drained=False
        a=model(mx.array([[token]],mx.uint32),committed,logits_to_keep=1)
        b=model(mx.array([[token]],mx.uint32),continuation_control,logits_to_keep=1)
        mx.eval(a,b,[r.state for r in committed],[r.state for r in continuation_control]);drained=True;live()
        result['continuation']=dict(token=token,ordinary_both_arms=True,logits=_array_compare(a,b,mx),
            cache=_cache_compare(committed,continuation_control,mx),prefix_unchanged=_cache_compare(prefix,prefix_control,mx))
        if not all(result['continuation'][k]['equal']for k in('logits','cache','prefix_unchanged')):raise RuntimeError('Private AST ordinary continuation/isolation gate failed')
        result.update(status='passed',passed=True,completed=True)
        return result
    except BaseException as error:
        result.update(status='failed',passed=False,error=dict(type=type(error).__name__,message=str(error)))
        error.partial_result=result;raise
    finally:
        store.acquire=acquire;cleanup_error=None
        if not drained:
            try:mx.synchronize();drained=True
            except BaseException as error:cleanup_error=error
        if drained:
            for value in(discovery,warm,candidate):
                if value is not None:
                    try:value.transaction.abort()
                    except BaseException as error:cleanup_error=cleanup_error or error
            if head is not None:
                try:head.close()
                except BaseException as error:cleanup_error=cleanup_error or error
            if session is not None:
                try:session.close();session=bank=None
                except BaseException as error:cleanup_error=cleanup_error or error
            elif bank is not None:
                try:bank.close();bank=None
                except BaseException as error:cleanup_error=cleanup_error or error
            for cache in(prefix,prefix_control,committed,continuation_control,*ordinary_states,*(discovery_states or[]),*(candidate_states or[])):_close(cache)
        else:_QUARANTINE.append((adapter,prefix,prefix_control,committed,continuation_control,ordinary_states,discovery_states,candidate_states,head,bank,session,discovery,warm,candidate))
        if pin_active and drained and not getattr(store,'_resident_bank_preparing',None):store._resident_bank_pinned_keys=prior_pins
        w.source_resident_gather_policy,w.source_resident_prefill_policy=saved
        if persist is not None:store.persist_atlas=persist
        if drained:
            try:store.set_adaptive_cache_target(None,working_set_experts=6)
            except BaseException as error:cleanup_error=cleanup_error or error
        if cleanup_error is not None:result.update(status='cleanup_failed',passed=False,cleanup_error=str(cleanup_error))
        _write(path,result)
        if cleanup_error is not None:raise cleanup_error
