# SPDX-License-Identifier: MIT
"""Diagnostic-only M1 KV ablations; no work on import or serving route.

Caller owns GPU, one private actual decode cache and <=32 fixed-frame replays.
No output/quality/parity claim. Suppressed cache is not publishable/reusable by
APCv2; caller discards it after diagnostics. This helper never times or evals.
Provenance: provenance/deepseek-v41-decode-ablation.json.
"""
from contextlib import contextmanager
import threading

MODES=('baseline','freeze_kv_path','skip_local_owned_copy')
_LOCK=threading.Lock()
FIELDS=('offset','keys','buf_kv','buf_gate','_pooled','_index_keys','_pool_count','_capacity','hash_tail','_in_forward')


def describe(mode):
    if type(mode)is not str or mode not in MODES:raise ValueError('Unknown decode KV diagnostic mode')
    return dict(schema='mlx2.ds41-decode-kv-ablation.v1',mode=mode,diagnostic_only=True,
        rate_eligible=False,qualification=False,output_quality_eligible=False,APCv2_publication=False,
        caller_must_discard_private_cache=True,lazy_unused_input_producers_may_be_pruned=True,
        scope=('Normal KV path with scoped counters'if mode=='baseline'else
            'Frozen valid history/offset: skip local concat/owned copy/retention and complete source publish (including producer math); not pure write cost'if mode=='freeze_kv_path'else
            'Ordinary local concat/current-token band/offset; retain final band view instead of owned local copy; source publication/pools remain normal'))


def _cache_metadata(model,cache):
    if (type(cache)not in(list,tuple)or len(cache)!=40 or len({id(s)for s in cache})!=40
            or len(getattr(model,'layers',()))!=40):raise ValueError('Decode KV diagnostic requires one40-layer private cache')
    offsets={s.offset for s in cache}
    if len(offsets)!=1 or type(next(iter(offsets)))is not int or not 1<=next(iter(offsets))<16384:
        raise ValueError('Decode KV diagnostic requires nonempty bounded frozen frontier')
    frontier=next(iter(offsets))
    for state in cache:
        if (getattr(state,'_closed',True)or state.head_dim!=512 or state.window_size!=128
                or state.keys is None or state.keys.shape!=(1,1,min(frontier,128),512)):
            raise ValueError('Decode KV diagnostic cache shape/lifetime mismatch')
    return frontier


def clone_cache_physical(model,cache):
    """Private physical-state clone, explicitly materialized OUTSIDE timers.

    Copy complete backing capacities, not `.state`/cropped pool properties.
    This deliberately uses real copies so any indexed cache update in a replay
    cannot mutate the seed or the caller's actual ordinary frontier.
    """
    from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache
    _cache_metadata(model,cache)
    if any(type(state)is not DeepSeekV41LayerCache for state in cache):
        raise ValueError('Decode clone requires actual Source cache objects')
    clones=[];arrays=[]
    for state in cache:
        xp=state.xp
        target=type(state)(state.ratio,xp=xp,window_size=state.window_size,
            head_dim=state.head_dim,index_head_dim=state.index_head_dim,
            identity=state.identity,source=state.source,history=state.history,
            compressed_vocab=state.compressed_vocab)
        for name in ('keys','buf_kv','buf_gate','_pooled','_index_keys','hash_tail'):
            value=getattr(state,name)
            owned=None if value is None else xp.asarray(value,copy=True)
            setattr(target,name,owned)
            if owned is not None:arrays.append(owned)
        target.offset=state.offset
        target._pool_count=state._pool_count
        target._capacity=state._capacity
        target._in_forward=False
        clones.append(target)
    # MLX copies are lazy; returning them unevaluated would move reset cost
    # into the timed replay. No scalar predicates or CPU value readbacks.
    xp=cache[0].xp
    if xp.__name__=='mlx.core':xp.eval(*arrays)
    return clones


@contextmanager
def instrument_kv(model,cache,*,mode='baseline',input_rows=1,maximum_local_calls=40):
    """One actual M1 forward, preferably replayed from a fixed seed externally.

    Frozen mode leaves offset and KV/pool arrays untouched, so post-forward
    shape validation remains safe; attention sees stale history on purpose.
    Live ordinary generator/frontier contracts must not be used for this arm.
    Root instrumentation owns timers and whole-module bypasses independently.
    """
    receipt=describe(mode)
    if type(input_rows)is not int or input_rows!=1 or maximum_local_calls!=40:
        raise ValueError('Decode KV ablation requires one M1 forward/40 local calls')
    frontier=_cache_metadata(model,cache)
    if not _LOCK.acquire(blocking=False):raise RuntimeError('Nested/concurrent decode KV ablation refused')
    patches=[];thread=threading.get_ident();registered={id(state):state for state in cache}
    saved={id(state):{name:getattr(state,name)for name in FIELDS}for state in cache}
    local_seen=set();receipt.update(frontier=frontier,local_calls=0,publish_calls=0,pool_append_calls=0,
        local_owned_copies_skipped=0,local_updates_skipped=0,source_publish_paths_skipped=0,
        pool_appends_skipped=0,foreign_cache_calls_unmodified=0,completed=False)
    def patch(owner,name,value):
        own=name in vars(owner);old=getattr(owner,name);patches.append((owner,name,old,own));setattr(owner,name,value)
    def ours(state):return threading.get_ident()==thread and registered.get(id(state))is state
    try:
        from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache
        from mlx2.runtime.models.deepseek_v41 import Attention
        if any(type(state)is not DeepSeekV41LayerCache for state in cache):raise ValueError('Decode KV ablation Source cache type')
        local=DeepSeekV41LayerCache.update_local;append=DeepSeekV41LayerCache.append_pools
        publish=Attention.publish;snapshot=DeepSeekV41LayerCache.snapshot
        def update(state,keys):
            if not ours(state):return local(state,keys)
            if (id(state)in local_seen or keys.shape!=(1,1,1,512)):
                raise ValueError('Decode KV ablation repeated state or non-M1 update')
            local_seen.add(id(state));receipt['local_calls']+=1
            if mode=='freeze_kv_path':
                receipt['local_updates_skipped']+=1
                return state.keys
            if mode=='skip_local_owned_copy':
                prior=state.keys[:,:,-127:,:]
                band=state.xp.concatenate((prior,keys),axis=2)
                state.keys=band[:,:,-128:,:]
                state.offset+=1;receipt['local_owned_copies_skipped']+=1
                return band
            return local(state,keys)
        def source_publish(obj,x,state,start):
            if obj.model is not model or not ours(state):return publish(obj,x,state,start)
            receipt['publish_calls']+=1
            if mode=='freeze_kv_path':
                receipt['source_publish_paths_skipped']+=1
                return None
            return publish(obj,x,state,start)
        def pools(state,pooled,index_keys):
            if not ours(state):return append(state,pooled,index_keys)
            receipt['pool_append_calls']+=1
            if mode=='freeze_kv_path':
                receipt['pool_appends_skipped']+=1
                return None
            return append(state,pooled,index_keys)
        def no_publication(state,*args,**kwargs):
            if ours(state)and mode!='baseline':raise RuntimeError('Diagnostic ablation cache must not be published')
            return snapshot(state,*args,**kwargs)
        patch(DeepSeekV41LayerCache,'update_local',update)
        patch(DeepSeekV41LayerCache,'append_pools',pools)
        patch(DeepSeekV41LayerCache,'snapshot',no_publication)
        patch(Attention,'publish',source_publish)
        yield receipt
        receipt['completed']=True
    finally:
        # Freeze mode changes no cache planes; only model Engram hash-tail and
        # forward flags may have been replaced independently of these hooks.
        if mode=='freeze_kv_path':
            for state in cache:
                for name,value in saved[id(state)].items():setattr(state,name,value)
        for owner,name,old,own in reversed(patches):
            if own:setattr(owner,name,old)
            else:delattr(owner,name)
        receipt['offsets_after']=[state.offset for state in cache]
        _LOCK.release()
