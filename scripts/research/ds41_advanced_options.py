# SPDX-License-Identifier: MIT
"""Caller-owned advanced Source probes; import/preparation never use the GPU.

These functions accept an already-loaded adapter inside an existing bounded
CPG/paired-lock window. They acquire no ownership and run no preflight. Exact
APCv2 state, advisory inventories and actual pinned expert banks stay distinct.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

OWNERS = (Path('/Users/Shared/mlxuag/gpu.lock/owner.json'), Path('/tmp/gpu.lock/owner.json'))


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    if path.exists() or temporary.exists():
        raise FileExistsError('Refusing to overwrite advanced evidence: ' + str(path))
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _bindings(unit):
    spec=json.loads((Path(unit)/'spec.json').read_text())
    return {key:spec[key] for key in ('checkpoint','source_revision','source_snapshot_sha256')}


def owned_guard(unit):
    from probe_deepseek_v4_packed import verify_ownership
    unit = Path(unit)
    def guard():
        verify_ownership(unit, OWNERS)
        return True
    return guard


def _live(guard):
    if not callable(guard) or guard() is not True:
        raise RuntimeError('Advanced probe requires a live caller-owned GPU window')


def _close(cache):
    if cache is not None:
        close = getattr(cache, 'close', None)
        if callable(close):close()
        else:
            for row in cache:row.close()


EXPERT_COUNTER_KEYS=('v41_expert_bytes_read','v41_expert_uploaded_experts',
                     'v41_expert_cache_misses','v41_expert_evictions')


def strict_counter_delta(before, after, keys):
    """Missing, reset, bool, float or unknown telemetry cannot imply zero work."""
    if type(before)is not dict or type(after)is not dict:
        raise ValueError('Counter evidence must contain actual snapshots')
    delta={}
    for key in keys:
        if key not in before or key not in after:
            raise ValueError('Required counter evidence missing: '+key)
        left,right=before[key],after[key]
        if type(left)is not int or type(right)is not int or left<0 or right<left:
            raise ValueError('Required counter is invalid or reset: '+key)
        delta[key]=right-left
    return delta


def expert_io_report(before,after):
    delta=strict_counter_delta(before,after,EXPERT_COUNTER_KEYS)
    return dict(counter_keys=list(EXPERT_COUNTER_KEYS),counter_delta=delta,telemetry_valid=True,
        zero_reads=delta['v41_expert_bytes_read']==0,
        zero_uploads=delta['v41_expert_uploaded_experts']==0,
        zero_misses=delta['v41_expert_cache_misses']==0,
        zero_evictions=delta['v41_expert_evictions']==0)


def resident_counter_snapshot(adapter):
    """Recognize the provider's actual names and require every claimed metric."""
    w=adapter.weights;raw=w.expert_stream.counters()
    strict_counter_delta(raw,raw,EXPERT_COUNTER_KEYS)
    result=dict(expert_bytes_read=raw['v41_expert_bytes_read'],
        expert_uploads=raw['v41_expert_uploaded_experts'],
        expert_misses=raw['v41_expert_cache_misses'],expert_evictions=raw['v41_expert_evictions'])
    result['host_route_materializations']=getattr(w,'source_route_host_materializations',None)
    result['engram_bytes_read']=0
    for table in adapter.engram_tables.values():
        value=table.stats.get('bytes_read')
        if type(value)is not int or value<0:raise ValueError('Missing actual Engram read counter')
        result['engram_bytes_read']+=value
    session=getattr(w,'source_resident_bank_session',None)
    if session is None:raise ValueError('Resident telemetry requires an attached actual session')
    result['ordinary_replays']=session.stats.get('ordinary_replays')
    strict_counter_delta(result,result,tuple(result))
    return result


def resident_variants(*, prefill=False):
    """One-option-at-a-time session settings; defaults form a separate control."""
    from mlx2.adapters.deepseek_v41_source_bank_session import (
        CACHE_FORK_POLICIES, EXPERT_KERNEL_VARIANTS, SUBMISSION_POLICIES, OPERAND_POLICIES)
    variants = [dict(id='control')]
    for name, values in [('cache_fork_policy', CACHE_FORK_POLICIES),
                         ('expert_kernel_variant', EXPERT_KERNEL_VARIANTS),
                         ('submission_policy', SUBMISSION_POLICIES), ('operand_policy', OPERAND_POLICIES)]:
        variants += [dict(id=name+'-'+value, **{name:value}) for value in values[1:]]
    if prefill:
        variants.append(dict(id='prefill_task_policy-persistent_8_2_v1', prefill_task_policy='persistent_8_2_v1'))
    return variants


@contextmanager
def pinned_inventory(adapter, keys, guard, *, prefill=False, max_bank_bytes=3 << 30, consumer=True):
    """Actual provider-owned immutable bank; never an advisory-residency claim.

    Only already-resident observed keys are accepted. Construction charges old
    payload plus one incremental bank and scratch. The supported bank/session
    owns coefficient views and pins until close fences all work.
    """
    _live(guard)
    import mlx.core as mx
    from mlx2.adapters.deepseek_v41_source_prefill_bank import build_prefill_bank
    store = adapter.weight_stream
    selected = tuple(sorted(set(tuple(k) for k in keys)))
    if not selected or any(key not in store._cache for key in selected):
        raise ValueError('Pinning needs a complete, already-loaded actual route inventory')
    bank = session = None
    try:
        bank = build_prefill_bank(mx, store, selected, ownercheck=guard,
            generation=getattr(store, '_resident_bank_generation', 0)+1,
            budget_bytes=store.cache_budget_bytes, max_bank_bytes=max_bank_bytes)
        if consumer:
            session = adapter.prepare_resident_bank_session(bank, ownercheck=guard, prefill=prefill)
        yield session if consumer else bank
    finally:
        if session is not None:
            # Poisoned ownership must remain quarantined for the owning unit.
            if not session.poisoned:session.close()
        elif bank is not None:bank.close()


@contextmanager
def canonical_prefix(adapter, unit, guard, prompt_ids, *, apc_bytes=256 << 20):
    """Store and restore the same canonical numerical identity through APCv2.

    Yields (lookup, report). The caller may pin an observed expert inventory and
    run subsequent compatible tests while this context retains exact APC state.
    No key rewriting or cross-policy state transplantation is supported.
    """
    _live(guard)
    if adapter.kernel_policy.get('deepseek_source_prefill_partition') != 'canonical_m1_v1':
        raise ValueError('Reusable Source prefix requires canonical_m1_v1 loaded identity')
    import mlx.core as mx
    from mlx2.runtime.apc_v2 import APCKey, APCv2
    from mlx2.runtime.checkpoint_resources import publication_hints, restore_resources
    from mlx2.serving import detach_checkpoint_graph
    from mlx2.runtime.attention_topology import apcv2_cache_layout_fingerprint
    ids = list(prompt_ids)
    if len(ids) < 2:raise ValueError('Canonical prefix needs at least two prompt tokens')
    unit = Path(unit)
    cache = adapter.model.make_cache(); apc = lookup = None
    report = dict(schema='mlx2.ds41-options-canonical-prefix.v1', qualified=False,
                  exact_model_identity=adapter.model.identity, resource_inventory_is_pin=False, **_bindings(unit))
    try:
        logits=adapter.model(mx.array([ids[:-1]],mx.uint32),cache,logits_to_keep=1)
        mx.eval(logits,[s.state for s in cache]); _live(guard)
        hints, hint_receipt=publication_hints(adapter,cache,ids[:-1])
        detach_checkpoint_graph(cache)
        layout=apcv2_cache_layout_fingerprint(adapter)
        apc=APCv2(max_size=2,max_bytes=apc_bytes,layout_name=adapter.layout)
        key=APCKey(adapter.identity['fingerprint'],revision=adapter.model.identity,
            adapter=adapter.numerics_revision,tokenizer_fingerprint=adapter.identity['fingerprint'],
            cache_layout_fingerprint=layout)
        capability=apc.store(key,ids[:-1],cache,expert_hints=hints)
        lookup=apc.lookup(key,ids)
        if not lookup.hit or lookup.cached_tokens != len(ids)-1 or lookup.remaining_tokens != ids[-1:]:
            raise RuntimeError('Canonical APCv2 exact-prefix contract failed')
        restored=restore_resources(adapter,lookup,ids)
        report.update(passed=True,prompt_tokens=len(ids),cached_tokens=lookup.cached_tokens,
            stored=capability.stored,hint_publication=hint_receipt,resource_restore=restored,
            exact_numerical_state=True,APCv2_stats=apc.apc_stats)
        _write(unit/'canonical-prefix.json',report)
        yield lookup,report
    finally:
        if lookup is not None:_close(lookup.cache)
        _close(cache)
        if apc is not None:apc.close()


def publication_snapshot(cache):
    """Canonical maintained publication descriptors, preserving every live bit.

    Source's state/from_state contract intentionally excludes allocation slack.
    Validate that raw capacity really is only slack before serializing it; an
    invalid live count, physical layout or incomplete frontier still refuses.
    """
    from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache
    result=[]
    try:
        for row in cache:
            if type(row)is not DeepSeekV41LayerCache or row.pending or row.ast_private_profile is not None:
                raise ValueError('Publication comparison requires exact committed Source cache planes')
            row.validate_state()
            count=row.offset//row.ratio if row.source else 0
            capacity=row._capacity
            if (type(capacity)is not int or capacity<count or row._pool_count!=count):
                raise ValueError('Source raw capacity/live frontier disagrees with publication extent')
            for name,width in (('_pooled',row.head_dim),('_index_keys',row.index_head_dim)):
                plane=getattr(row,name)
                if plane is None:
                    if count or capacity:raise ValueError('Source raw capacity has no physical owner')
                elif tuple(plane.shape)!=(1,capacity,width):
                    raise ValueError('Source raw pool allocation layout disagrees with its capacity')
            result.append(type(row).from_state(row.state,row.meta_state,xp=row.xp))
        return result
    except BaseException:
        _close(result)
        raise


def publication_cache_compare(left,right,xp):
    """Exact normalized physical comparison plus original allocation diagnostics."""
    from ds41_resident_prefill_model_gate import _cache_compare
    a=b=None
    try:
        a=publication_snapshot(left);b=publication_snapshot(right)
        result=_cache_compare(a,b,xp)
        result['comparison_scope']='Maintained canonical from_state(state,meta_state) publication in both arms; six live planes, exact metadata/frontier/layout; allocation slack excluded only after validation'
        result['raw_capacity']=[dict(layer=i,reference_capacity=x._capacity,candidate_capacity=y._capacity,
            reference_live_count=x._pool_count,candidate_live_count=y._pool_count)
            for i,(x,y)in enumerate(zip(left,right))]
        return result
    finally:
        _close(a);_close(b)


def resident_probe(adapter, unit, guard, prompt_ids, *, prefill=False, variants=None):
    """Discover actual ordinary routes, pin that set, and test session options.

    All candidate logits and every cache plane compare to a prebank ordinary
    control. Miss/replay is a recorded coverage failure, not observed bank use.
    This direct probe does not insert APCv2 state or produce qualified rates.
    """
    _live(guard)
    import mlx.core as mx
    from ds41_resident_prefill_model_gate import _array_compare, _cache_compare
    from mlx2.runtime.cow_cache import snapshot_prompt_cache_descriptors
    from mlx2.adapters.deepseek_v41_source_bank import _idle
    unit=Path(unit); path=unit/'resident-options.json'
    if path.exists():raise FileExistsError(path)
    model,w,store=adapter.model,adapter.weights,adapter.weight_stream
    saved=(w.source_resident_gather_policy,w.source_resident_prefill_policy)
    if saved[0]!='resident_bank_v1' or prefill and saved[1] not in ('expert_major_v1','expert_major_wide_v1'):
        raise ValueError('Load the selected resident gather/prefill policy before the probe')
    if w.source_resident_bank_session is not None:raise ValueError('Probe needs an unattached idle resident session')
    ids=list(prompt_ids)
    if not ids or prefill and len(ids)>128:raise ValueError('Bounded PP supports1..128 actual rows')
    caches=[]; report=dict(schema='mlx2.ds41-resident-options.v1',qualified=False,
        APCv2_publication=False,performance_qualified=False,prefill=prefill,prompt_rows=len(ids),cases=[], passed=False, **_bindings(unit))
    acquire=store.acquire
    try:
        w.source_resident_gather_policy=w.source_resident_prefill_policy='reference'
        seed=model.make_cache();caches.append(seed)
        if prefill:inputs=mx.array([ids],mx.uint32)
        else:
            first=model(mx.array([ids],mx.uint32),seed,logits_to_keep=1)
            mx.eval(first,[s.state for s in seed]);inputs=mx.array([[int(mx.argmax(first[0,-1]).item())]],mx.uint32)
        reference,_,_=snapshot_prompt_cache_descriptors(seed,require_committed=True);caches.append(reference)
        seen=set()
        def observe(prefix,expert):
            seen.add((int(prefix.split('.')[1]),int(expert)))
            return acquire(prefix,expert)
        store.acquire=observe
        try:
            expected=model(inputs,reference,logits_to_keep=1);mx.eval(expected,[s.state for s in reference])
        finally:store.acquire=acquire
        _live(guard);_idle(store)
        if not seen.issubset(store._cache):raise RuntimeError('Actual route set could not remain resident; use ordinary test')
        # Retire unrelated completed expert objects before bank construction.
        for key in tuple(store._cache):
            if key not in seen:store._adaptive_evict_key(key)
        w.source_resident_gather_policy,w.source_resident_prefill_policy=saved
        with pinned_inventory(adapter,seen,guard,prefill=prefill) as session:
            report['bank']=session.bank.receipt();report['expert_keys']=sorted(seen)
            for variant in variants or resident_variants(prefill=prefill):
                _live(guard)
                # Reset each setter to the same session defaults before one delta.
                selections=dict(cache_fork_policy='physical_copy_v1',expert_kernel_variant='reference',
                    submission_policy='per_layer_async_v1',operand_policy='full_operands_v1')
                if prefill:selections['prefill_task_policy']='padded_v1'
                selections.update({k:v for k,v in variant.items() if k!='id'})
                for name,value in selections.items():getattr(session,'set_'+name)(value)
                candidate,_,_=snapshot_prompt_cache_descriptors(seed,require_committed=True);caches.append(candidate)
                before=resident_counter_snapshot(adapter)
                output=model(inputs,candidate,logits_to_keep=1);mx.eval(output,[s.state for s in candidate]);_live(guard)
                delta=strict_counter_delta(before,resident_counter_snapshot(adapter),tuple(before))
                arrays=_array_compare(expected,output,mx)
                raw_state=_cache_compare(reference,candidate,mx)
                state=publication_cache_compare(reference,candidate,mx)
                route=session.receipt()
                finite_checks=[mx.all(mx.isfinite(output))]+[mx.all(mx.isfinite(plane))for row in candidate for plane in row.state if plane is not None]
                mx.eval(finite_checks)
                gates=dict(logits_exact=arrays['equal'],state_exact=state['equal'],
                    finite_output_and_state=all(bool(value.item())for value in finite_checks),
                    no_expert_IO=all(delta[k]==0 for k in ('expert_bytes_read','expert_uploads','expert_misses','expert_evictions','ordinary_replays')),
                    no_host_routes=delta['host_route_materializations']==0,
                    committed=session.last_receipt is not None and session.last_receipt.get('status')=='ok')
                report['cases'].append(dict(variant=variant,gates=gates,passed=all(gates.values()),
                    logits=arrays,cache=state,raw_allocation_comparison=raw_state,counters=delta,session=route))
            report['passed']=all(row['passed']for row in report['cases'])
        return report
    finally:
        store.acquire=acquire
        w.source_resident_gather_policy,w.source_resident_prefill_policy=saved
        session=getattr(w,'source_resident_bank_session',None)
        if session is None or not session.poisoned:
            for cache in caches:_close(cache)
        _write(path,report)


def prepare_dspark_oracle(adapter, unit, guard, prompt_ids, *, expert_law='source_ast_m1_rows_v1',
                          forward_law='fully_batched_m1_v1', resident_prepare=None, hybrid_miss_prepare=None):
    """Create fresh actual all16 raw proof; existing oracle validator stays final.

    Resident/hybrid callbacks must install genuine owned cohorts AFTER ordinary
    controls. Missing callbacks refuse. Native-M16 profile drift cannot become
    an exact oracle through this function.
    """
    _live(guard)
    import mlx.core as mx
    import numpy as np
    from mlx2.adapters.deepseek_v41_dspark import load_source_dspark
    from ds41_ast_target_components import run, _first_failed_proof_check
    from ds41_dspark_ast_serving_components import run as serving_run
    unit=Path(unit);spec=json.loads((unit/'spec.json').read_text())
    if forward_law!='fully_batched_m1_v1':raise ValueError('Exact oracle requires the full M1 batch-axis proof')
    if getattr(adapter.model,'_mlx2_source_ast_target_laws',None)!=(expert_law,forward_law):
        raise ValueError('Load matching dspark_prepare/oracle_v1 target laws before proof')
    for name in ('dspark-target-comparison.json','dspark-ast-raw.npz','dspark-ast-functional.json','ast-serving-oracle.json'):
        if (unit/name).exists():raise FileExistsError('Fresh proof output required: '+name)
    cache=adapter.model.make_cache();head=None
    try:
        ids=list(prompt_ids)
        logits,features=adapter.model.forward_with_taps(mx.array([ids],mx.uint32),cache,(37,38,39))
        mx.eval(logits,features,[s.state for s in cache]);_live(guard)
        anchor=int(mx.argmax(logits[0,-1]).item())
        head=load_source_dspark(adapter.model,expert_cache_bytes=512<<20,check_live=guard)
        history=head.make_history();head.record_features(features,0,history)
        paths,confidence=head.propose_paths(anchor,history);head_receipt=head.receipt();head.close();head=None
        observed=set();store=adapter.weight_stream;acquire=store.acquire
        def census(prefix,expert):
            observed.add((int(prefix.split('.')[1]),int(expert)))
            return acquire(prefix,expert)
        if expert_law in ('source_ast_resident_rows_v1','source_ast_hybrid_rows_v1') and resident_prepare is None:
            resident_prepare,missing=ast_resident_callbacks(adapter,guard,observed,hybrid=expert_law=='source_ast_hybrid_rows_v1')
            if expert_law=='source_ast_hybrid_rows_v1':hybrid_miss_prepare=missing
        store.acquire=census
        try:
            comparison=run(adapter.model,cache,guard,anchor=anchor,paths=paths,expert_law=expert_law,
                forward_law=forward_law,resident_prepare=resident_prepare,hybrid_miss_prepare=hybrid_miss_prepare,
                diagnostic_path=unit/'dspark-private-gate.json')
        finally:store.acquire=acquire
        raw=comparison.pop('raw_arrays')
        failure=_first_failed_proof_check(comparison)
        if failure is not None or comparison.get('numerical_classification')!='raw_bits_exact':
            _write(unit/'dspark-proof-failure.json',dict(first_failure=failure,comparison=comparison))
            raise RuntimeError('Actual all16 Source proof did not pass exact publication gates')
        with (unit/'dspark-ast-raw.npz').open('xb') as stream:np.savez(stream,**raw)
        _write(unit/'dspark-target-comparison.json',comparison)
        arrays=[];metadata=[];absence=[]
        def walk(value):
            if isinstance(value,dict):
                if 'raw_bits_equal'in value:arrays.append(value)
                if 'metadata_equal'in value:metadata.append(value['metadata_equal'])
                if 'absence_equal'in value:absence.append(value['absence_equal'])
                for child in value.values():walk(child)
            elif isinstance(value,(list,tuple)):
                for child in value:walk(child)
        walk(comparison)
        record=dict(schema='mlx2.ds41-dspark-ast-owned-functional.v1',stage='completed',passed=True,
            raw_bits_exact=True,numerical_classification='raw_bits_exact',checkpoint=spec['checkpoint'],
            source_revision=spec['source_revision'],source_snapshot_sha256=spec['source_snapshot_sha256'],
            target_receipt=comparison['receipt'],comparison_count=len(arrays),metadata_checks=len(metadata),
            absence_checks=len(absence),raw_arrays_sha256=hashlib.sha256((unit/'dspark-ast-raw.npz').read_bytes()).hexdigest(),
            head=head_receipt,qualified=False,paths=[list(p)for p in paths],draft_confidence=[list(c)for c in confidence])
        _write(unit/'dspark-ast-functional.json',record)
    finally:
        if head is not None:head.close()
        _close(cache)
    # Existing validator rereads source, raw arrays and every proof gate before
    # emitting/admitting the oracle and executing a genuine generic AST request.
    if expert_law=='source_ast_resident_rows_v1':
        with resident_prepare(adapter.model,None,None,None,guard):
            session=adapter.weights.source_resident_bank_session
            with resident_ast_serving_scope(adapter,session,guard,prepare=incoming_ast_bank_prepare)as install:
                return serving_run(adapter,unit,guard,proof_unit=unit,prompt_ids=ids,max_tokens=8,
                    expert_law=expert_law,forward_law=forward_law,resident_provider_scope=install)
    return serving_run(adapter,unit,guard,proof_unit=unit,prompt_ids=ids,max_tokens=8,
        expert_law=expert_law,forward_law=forward_law)


def ast_resident_callbacks(adapter, guard, observed, *, hybrid=False):
    """Build genuine full/missing banks after independent ordinary controls."""
    @contextmanager
    def cohort(model,cache,anchor,paths,check_live,*,missing=False):
        _live(guard)
        if model is not adapter.model:raise ValueError('Foreign AST cohort target')
        import mlx.core as mx
        from mlx2.adapters.deepseek_v41_source_bank import _idle
        from mlx2.adapters.deepseek_v41_source_prefill_bank import build_prefill_bank
        from mlx2.adapters.deepseek_v41_source_bank_session import PreparedBankSession
        from mlx2.adapters.deepseek_v41_ast_bank_manager import SourceASTBankManager
        w=model.weights;store=w.expert_stream;_idle(store)
        actual=frozenset(observed)
        if not actual:raise ValueError('Fresh independent ordinary route census is empty')
        omitted=min(actual)if missing else None
        keys=actual-({omitted}if omitted is not None else set())
        if not keys:raise ValueError('Missing-cohort proof requires a real nonempty bank')
        old_policy=w.source_resident_gather_policy
        if old_policy!='reference' or w.source_resident_bank_session is not None:
            raise ValueError('AST cohort must follow ordinary ownership')
        if getattr(model,'_source_ast_bank_manager',None)is not None or getattr(w,'_source_ast_bank_manager',None)is not None:
            raise ValueError('AST callback cannot replace a live manager')
        old_pins=getattr(store,'_resident_bank_pinned_keys',frozenset());bank=session=manager=None
        try:
            for key in tuple(store._cache):
                if key not in keys:store._adaptive_evict_key(key)
            store._resident_bank_pinned_keys=old_pins|keys
            for layer,expert in sorted(keys):
                _live(guard)
                with store.window():store.acquire('blk.'+str(layer),expert)
            if not keys.issubset(store._cache):raise RuntimeError('Actual AST cohort could not remain resident')
            bank=build_prefill_bank(mx,store,tuple(sorted(keys)),ownercheck=guard,
                generation=getattr(store,'_resident_bank_generation',0)+1,
                budget_bytes=store.cache_budget_bytes,max_bank_bytes=3<<30)
            if hybrid:
                manager=SourceASTBankManager(model,ownercheck=guard);manager.bank=bank
                model._source_ast_bank_manager=manager;w._source_ast_bank_manager=manager
            else:
                w.source_resident_gather_policy='resident_bank_v1'
                session=PreparedBankSession(w,bank,ownercheck=guard);w.source_resident_bank_session=session
                # Proof compares full physical capacity against prebank ordinary
                # controls; use the maintained descriptor-preserving fork law.
                session.set_cache_fork_policy('descriptor_cow_v1')
            yield dict(whole_cohort=not missing,genuine_omitted_expert=None if omitted is None else list(omitted),
                ordinary_observed_experts=len(actual),bank_experts=len(keys),bank=bank.receipt(),
                continuation_cache_fork=None if session is None else session.cache_fork_policy,
                observation='Fresh actual ordinary controls; no synthetic IDs or LUT edits')
        finally:
            if manager is not None:
                manager.close()
                if getattr(model,'_source_ast_bank_manager',None)is manager:del model._source_ast_bank_manager
                if getattr(w,'_source_ast_bank_manager',None)is manager:del w._source_ast_bank_manager
                bank=None
            if session is not None:session.close();bank=None
            if bank is not None:bank.close()
            w.source_resident_gather_policy=old_policy
            if getattr(store,'_resident_bank',None)is None and not getattr(store,'_resident_bank_preparing',None):
                store._resident_bank_pinned_keys=old_pins
    def full(*args):return cohort(*args,missing=False)
    def miss(*args):return cohort(*args,missing=True)
    return full,miss


def native_batched_probe(adapter,unit,guard,*,descendant=False):
    """Existing complete native-M16/resident model gate; no generic oracle."""
    from ds41_ast_batched_model_probe import run
    return run(adapter,unit,guard,forward_law='fully_batched_m1_v1'if descendant else'fully_batched_v1',
        require_ordinary_exact=descendant,paired_kv_preparation=descendant,prefix_tokens=258 if descendant else None)


@contextmanager
def ordinary_m16_boundary(model,guard,*,layer=14,node=12):
    """Observe one actual M16/M1 boundary; return production values unchanged.

    The extra M1 call is an untimed component diagnostic in the producing
    Source expert window. It cannot grant an exact AST oracle or publication.
    """
    import mlx.core as mx
    from ds41_resident_prefill_model_gate import _array_compare
    weights=model.weights;original=weights.moe
    report=dict(layer=layer,node=node,extra_component_oracles=True,
        production_outputs_unchanged=True,oracle_admission_granted=False,
        extra_m1_calls=0,ordinary_calls=0)
    captured={}
    def compare(a,b):
        raw=_array_compare(a,b,mx)
        if not raw['equal']:
            from ds41_ast_target_components import compare_arrays
            raw['difference']=compare_arrays(a,b,atol=0.,rtol=0.)
        return raw
    def hook(x,ids,scores,prefix,**kwargs):
        if prefix!=f'blk.{layer}':return original(x,ids,scores,prefix,**kwargs)
        if x.shape[0]==16:
            if captured:raise ValueError('Boundary diagnostic expected one target M16 tile')
            before=dict(weights.scheduled_expert_calls)
            output=original(x,ids,scores,prefix,**kwargs)
            mx.eval(x,ids,scores,output);_live(guard)
            after=dict(weights.scheduled_expert_calls)
            # Preserve the actual producing values before issuing the extra
            # ordinary component call; this is not a replacement target arm.
            captured.update(x=x[node:node+1],ids=ids[node:node+1],scores=scores[node:node+1],output=output[node:node+1])
            one=original(captured['x'],captured['ids'],captured['scores'],prefix,**kwargs)
            mx.eval(one);_live(guard);report['extra_m1_calls']+=1
            final=dict(weights.scheduled_expert_calls)
            report['candidate_observed_calls']={k:v-before.get(k,0)for k,v in after.items()if v!=before.get(k,0)}
            report['component_observed_calls']={k:v-after.get(k,0)for k,v in final.items()if v!=after.get(k,0)}
            captured['same_input_m1']=one
            report['same_input_m16_vs_m1']=compare(captured['output'],one)
            report['actual_shape_boundary']=dict(candidate_rows=16,component_rows=1,
                source_expert_math=weights.expert_math_policy,
                source_prefill_batch=getattr(weights,'source_prefill_batch_policy','reference'),
                candidate_fuse_down_shape_eligible=False,component_fuse_down_shape_eligible=True)
            return output
        output=original(x,ids,scores,prefix,**kwargs)
        if captured and x.shape[0]==1:
            ordinal=report['ordinary_calls'];report['ordinary_calls']+=1
            if ordinal==14:
                mx.eval(x,ids,scores,output)
                report['ordinary_node12']=dict(x=compare(captured['x'],x),ids=compare(captured['ids'],ids),
                    scores=compare(captured['scores'],scores),output_vs_same_input_m1=compare(captured['same_input_m1'],output),
                    output_vs_candidate_m16=compare(captured['output'],output))
        return output
    weights.moe=hook
    try:yield report
    finally:weights.moe=original


@contextmanager
def ordinary_m16_boundaries(model,guard,layers):
    from contextlib import ExitStack
    layers=tuple(layers)
    if not layers or len(set(layers))!=len(layers)or any(type(i)is not int or not 0<=i<=14 for i in layers):
        raise ValueError('M16 boundary layers must be unique within0..14')
    report=dict(layers=[],extra_component_oracles=True,production_outputs_unchanged=True,oracle_admission_granted=False)
    with ExitStack()as stack:
        report['layers']=[stack.enter_context(ordinary_m16_boundary(model,guard,layer=i))for i in layers]
        yield report
    ordered=sorted(report['layers'],key=lambda r:r['layer'])
    report['first_same_input_m16_m1_difference']=next((r['layer']for r in ordered if not r.get('same_input_m16_vs_m1',{}).get('equal',False)),None)
    report['first_ordinary_input_difference']=next((r['layer']for r in ordered if not r.get('ordinary_node12',{}).get('x',{}).get('equal',False)),None)


def ordinary_m16_probe(adapter,unit,guard,prompt_ids):
    """Actual ordinary-M16 target tile; current exact serving validator excludes it."""
    _live(guard)
    import mlx.core as mx
    import numpy as np
    from mlx2.adapters.deepseek_v41_dspark import load_source_dspark
    from ds41_ast_target_components import run,_first_failed_proof_check
    model=adapter.model;cache=model.make_cache();head=None
    report=dict(schema='mlx2.ds41-ordinary-m16-probe.v1',**_bindings(unit),passed=False,
        qualified=False,generic_exact_oracle_supported=False,APCv2_publication=False)
    try:
        logits,features=model.forward_with_taps(mx.array([list(prompt_ids)],mx.uint32),cache,(37,38,39))
        mx.eval(logits,features,[s.state for s in cache]);_live(guard)
        head=load_source_dspark(model,expert_cache_bytes=512<<20,check_live=guard)
        history=head.make_history();head.record_features(features,0,history)
        paths,_=head.propose_paths(int(mx.argmax(logits[0,-1]).item()),history)
        report['head']=head.receipt();head.close();head=None
        from contextlib import nullcontext
        diagnostic=os.environ.get('MLX2_ADVANCED_M16_BOUNDARY')=='1'
        layers=tuple(int(v)for v in os.environ.get('MLX2_ADVANCED_M16_BOUNDARY_LAYERS','14').split(','))
        scope=ordinary_m16_boundaries(model,guard,layers)if diagnostic else nullcontext(None)
        with scope as boundary:
            comparison=run(model,cache,guard,anchor=int(mx.argmax(logits[0,-1]).item()),paths=paths,
                expert_law='ordinary_m16_v1',forward_law='reference')
        if boundary is not None:report['numeric_boundary_diagnostic']=boundary
        raw=comparison.pop('raw_arrays')
        with (Path(unit)/'ordinary-m16-raw.npz').open('xb')as stream:np.savez(stream,**raw)
        report.update(comparison=comparison,first_failed_check=_first_failed_proof_check(comparison),
            passed=_first_failed_proof_check(comparison)is None,
            serving_limit='Existing generic exact validator supports M1/resident/hybrid laws only; no oracle fabricated')
        return report
    finally:
        if head is not None:head.close()
        _close(cache);_write(Path(unit)/'ordinary-m16-probe.json',report)


@contextmanager
def resident_ast_serving_scope(adapter,session,guard,*,prepare=None):
    """Retain a real bank while selecting it only for the owned AST tile.

    A decode-only B1 session cannot own an empty prefix or body-only prefill.
    Those use ordinary demand loading over the same immutable pinned bank.
    Only the exact target provider invocation borrows the prepared session.
    """
    _live(guard)
    w=adapter.weights
    if (w.source_resident_bank_session is not session or w.source_resident_gather_policy!='resident_bank_v1'
            or session.active or session.replaying or session.poisoned):
        raise ValueError('Resident AST serving needs its actual idle prepared session')
    bank=session.bank
    if getattr(w.expert_stream,'_resident_bank',None)is not bank:
        raise ValueError('Resident AST serving bank/provider ownership differs')
    bank.assert_current();installed=[];preparations=[]
    w.source_resident_bank_session=None;w.source_resident_gather_policy='reference'
    def install(provider):
        if (provider.model is not adapter.model or provider.expert_law!='source_ast_resident_rows_v1'
                or provider.forward_law!='fully_batched_m1_v1'):
            raise ValueError('Resident AST serving scope cannot wrap a foreign target law')
        if installed:raise ValueError('Resident AST serving provider already installed')
        original=provider.forward
        def forward(*args,**kwargs):
            _live(guard);session.bank.assert_current()
            if (session.active or session.replaying or session.poisoned
                    or w.source_resident_bank_session is not None or w.source_resident_gather_policy!='reference'):
                raise ValueError('Resident AST tile cannot enter a foreign or active resource scope')
            if prepare is not None:
                try:preparations.append(prepare(adapter,session,guard,*args[:5]))
                except BaseException as error:
                    preparations.append(getattr(error,'resource_preparation',dict(passed=False,error=str(error))))
                    raise
            w.source_resident_bank_session=session;w.source_resident_gather_policy='resident_bank_v1'
            try:return original(*args,**kwargs)
            finally:
                w.source_resident_bank_session=None;w.source_resident_gather_policy='reference'
        provider.forward=forward;installed.append((provider,original))
        return preparations
    try:yield install
    finally:
        for provider,original in installed:provider.forward=original
        # The outer genuine-cohort callback retains/then closes this session.
        w.source_resident_bank_session=session;w.source_resident_gather_policy='resident_bank_v1'


_INCOMING_CENSUS_QUARANTINE=[]


def incoming_ast_bank_prepare(adapter,session,guard,model,inputs,parents,cache,layers):
    """Preheat the actual incoming scheduler tree, then grow its real bank.

    This untimed resource preparation uses independent ordinary continuations
    from the exact incoming frontier. It never substitutes their output for
    the subsequent AST tile or grants numerical publication admission.
    """
    _live(guard)
    if model is not adapter.model:raise ValueError('Foreign incoming AST target')
    import mlx.core as mx
    from mlx2.adapters.deepseek_v41_ast_target import _canonical_rows
    from mlx2.adapters.deepseek_v41_source_bank import build_bank,_idle
    from ds41_ast_target_components import _proof_snapshot
    w=adapter.weights;store=w.expert_stream;bank=session.bank
    if w.source_resident_bank_session is not None or w.source_resident_gather_policy!='reference':
        raise ValueError('Incoming AST census requires ordinary idle resource selection')
    _idle(store);bank.assert_current()
    canonical=_canonical_rows(parents);tokens=tuple(int(inputs[i])for i in canonical)
    if len(tokens)!=16:raise ValueError('Incoming AST census requires exactly16 actual tokens')
    observed=set();acquire=store.acquire;copies=[];drained=True
    report=dict(schema='mlx2.ds41-incoming-ast-resources.v1',base_offset=cache[0].offset,
        canonical_inputs=list(tokens),old_bank_generation=bank.generation,ordinary_control_steps=0,
        numerical_output_substituted=False,oracle_admission_granted=False,passed=False)
    def census(prefix,expert):
        observed.add((int(prefix.split('.')[1]),int(expert)))
        return acquire(prefix,expert)
    prior_pins=getattr(store,'_resident_bank_pinned_keys',frozenset())
    try:
        store.acquire=census
        try:
            for branch in range(3):
                private=_proof_snapshot(cache);copies.append(private)
                for token in (tokens[0],*tokens[1+branch*5:6+branch*5]):
                    _live(guard)
                    drained=False
                    logits,features=model.forward_with_taps(mx.array([[token]],mx.uint32),private,layers)
                    mx.eval(logits,features,[row.state for row in private]);drained=True
                    report['ordinary_control_steps']+=1
                _close(private);copies.pop()
        finally:store.acquire=acquire
        if not observed:raise ValueError('Incoming AST ordinary census observed no expert routes')
        needed=frozenset(observed);missing=needed-bank.keys;keys=bank.keys|needed
        report.update(needed_experts=len(needed),missing_experts=len(missing),missing_keys=[list(k)for k in sorted(missing)])
        if missing:
            _idle(store)
            for key in tuple(store._cache):
                if key not in keys:store._adaptive_evict_key(key)
            store._resident_bank_pinned_keys=prior_pins|keys
            for layer,expert in sorted(keys):
                _live(guard)
                with store.window():store.acquire('blk.'+str(layer),expert)
            with bank.replacement(tuple(sorted(keys))):
                fresh=build_bank(mx,store,tuple(sorted(keys)),ownercheck=guard,
                    generation=bank.generation+1,budget_bytes=store.cache_budget_bytes,max_bank_bytes=3<<30,staging_bytes=0)
            session.bank=fresh;bank=fresh
        bank.assert_current()
        report.update(passed=True,bank_generation=bank.generation,bank=bank.receipt(),
            coverage='Actual incoming scheduler tree ordinary expert census; immutable precision map unchanged')
        return report
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error));error.resource_preparation=report
        if not drained:
            try:mx.synchronize();drained=True
            except BaseException as fence_error:
                _INCOMING_CENSUS_QUARANTINE.append((adapter,session,tuple(copies),error))
                copies=[];error.add_note('Incoming census completion uncertain; owners quarantined: '+repr(fence_error))
        raise
    finally:
        store.acquire=acquire
        for private in copies:_close(private)
        if (getattr(store,'_resident_bank',None)is session.bank and not getattr(store,'_resident_bank_preparing',None)
                and not getattr(store,'_resident_bank_replacing',None)and not getattr(store,'_error',None)):
            store._resident_bank_pinned_keys=session.bank.keys
