# SPDX-License-Identifier: MIT
"""Optional Source expert-cache inventory for APCv2 resource preparation.

Metadata is advisory, non-owning and separate from the numerical KV snapshot.
No router IDs, tensors or GPU values are read back by this producer/validator.
"""
import hashlib
import json
from pathlib import Path
import threading

ABI='mlx2.source-original-q1-packed-bank.v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()


def implementation_identity():
    names=('deepseek_v41_source_checkpoint_resources.py','deepseek_v41_source_buffers.py',
           'deepseek_v41_source_weights.py','deepseek_v41_source_lfu.py','deepseek_v41_mlx_weights.py')
    return _digest([(name,hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest())for name in names])


def binding(adapter):
    model=adapter.model;weights=model.weights;store=weights.expert_stream
    manager=getattr(model,'_source_ast_bank_manager',None)
    if manager is not None:return tuple(sorted(manager.hint_binding().items()))
    return tuple(sorted(dict(checkpoint=weights.artifact['identity']['fingerprint'],
        source_revision=_digest((model.identity,implementation_identity())),
        quantization=_digest((weights.expert_math_policy,getattr(store,'q1_binding',None),getattr(store,'hot_binding',None))),
        layout=_digest((store.binding,getattr(store,'q1_binding',None),getattr(store,'hot_binding',None))),
        resource_abi=ABI).items()))


def _frontier(adapter,cache):
    from ..runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache
    rows=tuple(cache)
    expected=adapter.model.args.num_hidden_layers
    if not rows or len(rows)!=expected:return None
    if any(type(row)is not DeepSeekV41LayerCache or row.identity!=adapter.model.identity
           or getattr(row,'_closed',False)or row.pending or row.ast_private_profile is not None for row in rows):return None
    positions={row.offset for row in rows}
    if len(positions)!=1:return None
    position=next(iter(positions))
    if type(position)is not int or position<0:return None
    # meta_state contains only existing CPU identity/topology/frontier fields.
    # validate_state() reads the Engram tail GPU values and is intentionally
    # not an optional resource-metadata operation.
    return position,tuple(tuple(row.meta_state)for row in rows)


def _snapshot_revision(metadata,covered,token_digest,resource_binding):
    origin=tuple(tuple(row[:-1])+(str(covered),)for row in metadata)
    return _digest(('source-expert-inventory-snapshot-v1',origin,token_digest,resource_binding))


def _registry(store):
    # Paths never enter the metadata. Resolve every pair through THIS loaded
    # artifact's registry, including its immutable original/Q1 hot map.
    result={}
    for prefix,keys in store.tables.items():
        for index,key in enumerate(keys):result[key]=(prefix,index)
    return result


def _geometry(store):
    rows=[]
    for layer in range(len(store.tables)):
        keys=store.tables.get('blk.'+str(layer))
        if keys is None:return None
        rows.append(len(keys))
    return tuple(rows)


def _cancelled(adapter,manager,callback):
    check=callback if callback is not None else getattr(manager,'cancelled',None)
    if check is None:check=getattr(adapter,'_checkpoint_resource_cancelled',None)
    return bool(check())if callable(check)else False


def _idle(adapter,store,manager):
    model=adapter.model;weights=model.weights
    if (getattr(model,'_weight_executions',0)or getattr(model,'_mlx2_ds41_ast_active',None)is not None
            or getattr(model,'_mlx2_ds41_ast_poisoned',False)):return 'target_active_or_poisoned'
    if manager is not None:
        if manager.active or manager.poisoned or manager.closed:return 'bank_manager_active_or_unavailable'
        if manager.cancelled():return 'cancelled_resource_owner'
        manager.check_live()
    session=getattr(weights,'source_resident_bank_session',None)
    if session is not None and any(getattr(session,key,False)for key in ('active','replaying','poisoned')):
        return 'bank_session_active_or_unavailable'
    if (getattr(store,'_active',False)or getattr(store,'_leased',{})or getattr(store,'_pending_uploads',{})
            or getattr(store,'_pending_host',{})or getattr(store,'_completion_collector',None)is not None
            or getattr(store,'_resident_bank_preparing',None)or getattr(store,'_resident_bank_replacing',None)):
        return 'expert_resources_active'
    if getattr(store,'_closed',False)or getattr(store,'_parked',False)or getattr(store,'_error',None):return 'expert_provider_unavailable'
    owner=getattr(store,'_owner',None)
    if owner is not None and owner!=threading.get_ident():return 'foreign_evaluation_owner'
    pool=getattr(store,'pool',None)
    if pool is not None:
        if getattr(pool,'failure',None)is not None:return 'expert_reader_failed'
        if pool.counters().get('weight_read_inflight',0):return 'expert_reader_active'
    return None


def snapshot(adapter,prompt_cache,tokens):
    from ..runtime.expert_residency_hints import ExpertResidencyHints,MAX_HINT_EXPERTS,prefix_token_digest
    frontier=_frontier(adapter,prompt_cache)
    if frontier is None:return None
    covered,metadata=frontier;values=tuple(tokens)
    if len(values)!=covered:return None
    try:digest=prefix_token_digest(values)
    except ValueError:return None
    bound=binding(adapter)
    store=adapter.model.weights.expert_stream;manager=getattr(adapter.model,'_source_ast_bank_manager',None)
    with store._lock:
        if _idle(adapter,store,manager)is not None:return None
        registry=_registry(store);keys=[key for key in store._cache if key in registry]
        index=getattr(store,'index',None);counts=getattr(index,'counts',{});recent=getattr(index,'last',{})
        keys.sort(key=lambda key:(-counts.get(key,0),-recent.get(key,0),key))
        selected=tuple(sorted(keys[:MAX_HINT_EXPERTS]));truncated=len(keys)>len(selected)
        return ExpertResidencyHints(**dict(bound),snapshot_revision=_snapshot_revision(metadata,covered,digest,bound),
            token_digest=digest,covered_tokens=covered,layer_expert_counts=_geometry(store),experts=selected,
            phase='resident_snapshot',coverage='advisory',truncated=truncated)


def _available(callback):
    if callback is None:
        from ..memory import host_available_bytes
        value=host_available_bytes(host_signals=True)
    else:value=callback()
    return value if type(value)is int and value>=0 else None


def _safe_allocation_decline(store,error,prior_error):
    """Native window marks inactive ONLY after its finally fence succeeded.

    A completed optional allocation failure can decline without poisoning lazy
    loading. Unknown/fence/read-integrity errors retain all existing safeguards.
    """
    if (prior_error is not None or getattr(store,'_active',True)or getattr(store,'_leased',{})
            or getattr(store,'_pending_uploads',{})or getattr(store,'_pending_host',{})
            or getattr(store,'_error',None)!=str(error)
            or getattr(getattr(store,'pool',None),'failure',None)is not None):return False
    total=0
    for value in store._cache.values():
        amount=getattr(value,'allocation_nbytes',None)
        if amount is None and type(value)is dict:
            amounts=[getattr(array,'nbytes',None)for array in value.values()]
            if any(type(size)is not int or size<0 for size in amounts):return False
            amount=sum(amounts)
        if type(amount)is not int or amount<0:return False
        total+=amount
    if total!=store._cache_bytes:return False
    store._error=prior_error
    return True


def restore(adapter,hints,prompt_cache,*,available_bytes=None,cancelled=None):
    from ..runtime.expert_residency_hints import ExpertResidencyHints
    from .deepseek_v41_source_buffers import ORIGINAL_BYTES,Q1_BYTES
    report=dict(schema='mlx2.source-checkpoint-resources.v1',hint_hit=False,resource_resident=0,prefetched=0,
        prefetched_bytes=0,requested=0,truncated=False,resource_lease=False,reservation_bytes=0,
        bank_constructed=False,KV_validity_authority=False,
        residency_tier='Application expert-cache presence; physical RAM residency and OS paging unmeasured')
    def decline(reason):report['reason']=reason;return report
    if type(hints)is not ExpertResidencyHints:return decline('absent_or_foreign_hint')
    if hints.phase!='resident_snapshot'or hints.coverage!='advisory':return decline('unsupported_observation_phase')
    frontier=_frontier(adapter,prompt_cache)
    if frontier is None:return decline('cache_frontier_or_model_mismatch')
    covered,metadata=frontier;bound=binding(adapter);restored=hints.for_restore(covered,bound)
    if restored is None or hints.layer_expert_counts!=_geometry(adapter.model.weights.expert_stream):return decline('stale_binding_or_geometry')
    if hints.snapshot_revision!=_snapshot_revision(metadata,hints.covered_tokens,hints.token_digest,bound):return decline('advisory_snapshot_revision_mismatch')
    store=adapter.model.weights.expert_stream;manager=getattr(adapter.model,'_source_ast_bank_manager',None)
    registry=_registry(store)
    if any(key not in registry for key in restored.experts):return decline('unknown_current_artifact_expert')
    report.update(hint_hit=True,requested=len(restored.experts),truncated=restored.truncated,
        observation_phase=restored.phase,coverage=restored.coverage,covered_tokens=covered)
    with store._lock:
        report['resource_resident']=sum(key in store._cache for key in restored.experts)
        reason=_idle(adapter,store,manager)
        if reason is not None:return decline(reason)
        if _cancelled(adapter,manager,cancelled):return decline('cancelled_before_resource_prepare')
        if getattr(store,'_resident_bank',None)is not None:return decline('existing_pinned_bank_preserved')
        staging=getattr(store,'_adaptive_staging',store.minimum_cache_bytes)
        working=store.minimum_cache_bytes*max(1,getattr(store,'_adaptive_working_set',1))
        # Match adaptive_residency_update's live target (available minus
        # headroom); otherwise an advisory prefetch evicts resident experts.
        headroom=max(getattr(store,'_adaptive_headroom',0)or 0,getattr(store,'_adaptive_protected_bytes',0)or 0)
        for key in restored.experts:
            if key in store._cache:continue
            if _cancelled(adapter,manager,cancelled):report['reason']='cancelled_resource_prepare';break
            reason=_idle(adapter,store,manager)
            if reason is not None:report['reason']=reason;break
            available=_available(available_bytes)
            kind=store.kind(key)if callable(getattr(store,'kind',None))else'original'
            if kind not in ('original','q1'):report['reason']='unsupported_current_expert_format';break
            size=Q1_BYTES if kind=='q1'else ORIGINAL_BYTES
            hard_room=store.cache_budget_bytes-store._cache_bytes-staging-working
            if available is None or available<size+staging+headroom or hard_room<size:
                report['reason']='actual_availability_or_operator_working_headroom';break
            prior_error=getattr(store,'_error',None)
            try:
                with store.window():store.acquire(*registry[key])
            except MemoryError as error:
                if _safe_allocation_decline(store,error,prior_error):
                    report['reason']='completed_optional_allocation_declined';break
                raise
            if key in store._cache:
                report['prefetched']+=1;report['prefetched_bytes']+=size
        report['resource_resident']=sum(key in store._cache for key in restored.experts)
    report.setdefault('reason','bounded_prefetch_complete'if report['prefetched']else'current_inventory_already_resident')
    report['precision_unchanged']=True
    return report
