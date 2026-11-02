# SPDX-License-Identifier: MIT
"""Explicit per-expert LFU profiles, with immutable precision selection per run.

Only routed experts participate. See provenance/deepseek-v41-source-q1-lfu.json.
"""
from collections import Counter, deque
from contextlib import contextmanager
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .deepseek_v41_source_artifact import checked_file, read_tensors
from .deepseek_v41_source_weights import SourceExpertStore, weight_plan, expert_names, PROJECTIONS
from .deepseek_v41_mlx_weights import _stamp, _verify_file, read_native_file
from .deepseek_v41_source_buffers import read_original_packed, borrow_q1_packed, materialize
from . import deepseek_v41_source_residency as residency
from . import deepseek_v41_source_prefill_residency as prefill_residency

REVISION = 'source-expert-lfu-token-frequency-v1'
Q1_REVISION = 'source-q1-affine-g64-lloyd4-f16-v1'
ORIGINAL_BYTES = 18800640
Q1_BYTES = 6635520

def parse_policy(value, *, adaptive=True):
    """Validate the default adaptive profile or an explicit fixed-pool transition."""
    if type(adaptive)is not bool:raise ValueError('Source expert residency mode must be boolean')
    if not isinstance(value,dict) or set(value)-{'mode','budget_bytes','hot_bytes','q1_root','hotset_path','atlas_path','preheat'}-residency.RESIDENCY_KEYS:
        raise ValueError('invalid Source expert LFU policy')
    result=dict(value)
    if 'atlas_path' in result and (not isinstance(result['atlas_path'],str) or not Path(result['atlas_path']).is_absolute()):
        raise ValueError('expert atlas path must be absolute')
    if 'preheat' in result and (type(result['preheat']) is not bool or 'atlas_path' not in result):
        raise ValueError('preheat requires a persistent atlas')
    if result.get('mode') not in ('original_lfu','hot_original_q1'):
        raise ValueError('Source expert mode must be original_lfu or hot_original_q1')
    if type(result.get('budget_bytes')) is not int or not ORIGINAL_BYTES*2 <= result['budget_bytes'] <= residency.LIVE_MAXIMUM_CLAMP_BYTES:
        raise ValueError('Source expert maximum must fit one expert plus staging and be at most90GB')
    if adaptive and result['mode']=='hot_original_q1' and 'residency' not in result:
        # 2026-10-28 checkpoint A/B: split kind pools keep the whole hot map
        # resident and are token-identical to the shared pool (+13% warm
        # decode at 75 GB). Explicit shared_v1 keeps the historical pool.
        result['residency']=residency.RESIDENCY_SPLIT
    residency.validate(result,adaptive=adaptive,original_bytes=ORIGINAL_BYTES)
    if result['mode']=='hot_original_q1':
        # Adaptive storage shares one cache ceiling; hot_bytes binds the
        # immutable precision map and need not fit as a resident kind pool.
        # Keep the historical precision-map bound and the stricter fixed-pool
        # rule for an explicit transition back to legacy residency.
        hot_limit=90_000_000_000 if adaptive else result['budget_bytes']
        if type(result.get('hot_bytes')) is not int or not ORIGINAL_BYTES<=result['hot_bytes']<hot_limit-ORIGINAL_BYTES:
            raise ValueError('invalid original-hot/Q1 precision-map budget' if adaptive else 'invalid original-hot/Q1 fixed-pool budget split')
        for key in ('q1_root','hotset_path'):
            if not isinstance(result.get(key),str) or not Path(result[key]).is_absolute():
                raise ValueError('hybrid expert policy requires absolute Q1 and hotset paths')
    elif set(result)-{'mode','budget_bytes','atlas_path','preheat','cache_maximum','live_reserve_bytes'}:
        raise ValueError('original-only LFU cannot accept Q1 options')
    return result

def _host_available():
    from ..memory import host_available_bytes
    return int(host_available_bytes(host_signals=True))

def source_admission(artifact,request,policy=None,*,available_bytes=None):
    """Source demand loading validates contracts, not a fictitious full cache.

    Dense/cache estimates remain advisory. A low-memory host may attempt its
    imminent expert/staging allocation and swap; actual allocation/I/O failures
    remain visible. Other model adapters retain their original admission path.
    """
    from ..runtime.streamed_load import WeightStreamRequest
    if not isinstance(request,WeightStreamRequest):raise ValueError('Source requires a validated WeightStreamRequest')
    if (request.mode!='moe_experts' or request.mtp_resident or request.collector is not None
            or request.prefill_admission!='lru' or request.router_hint!='off' or request.expert_overlap):
        raise ValueError('Source demand experts require synchronous ordinary streaming')
    live=None
    if policy is not None:
        policy=parse_policy(policy);budget=policy['budget_bytes']
        if request.budget_mode=='explicit' and request.budget_bytes!=budget:
            raise ValueError('LFU execution policy and requested expert maximum disagree')
        if policy.get('cache_maximum','explicit')=='live_v1':
            # budget_bytes is the operator clamp; the maximum is what live
            # memory admits after dense/KV/workspace headroom at load.
            clamp=budget
            budget,live=residency.derive_live_maximum(clamp,weight_plan(artifact,clamp),
                reserve_bytes=residency.settings(policy)['live_reserve_bytes'],
                available_bytes=available_bytes if available_bytes is not None else _host_available,
                minimum=2*ORIGINAL_BYTES)
    else:budget=residency.AUTO_MAXIMUM_BYTES if request.budget_mode=='auto' else request.budget_bytes
    limit=residency.LIVE_MAXIMUM_CLAMP_BYTES if live is not None else residency.EXPLICIT_MAXIMUM_BYTES
    if type(budget)is not int or not 2*ORIGINAL_BYTES<=budget<=limit:
        raise ValueError('Source expert maximum must fit one expert plus staging and be at most90GB')
    result=weight_plan(artifact,budget)
    if live is not None:result['expert_cache_maximum']=live
    result['estimated_full_reservation_bytes']=result['total_reservation_bytes']
    progress=result['minimum_expert_cache_bytes']+ORIGINAL_BYTES
    result.update(expert_active_bound_bytes=budget,expert_staging_bound_bytes=ORIGINAL_BYTES,
        expert_cache_bytes=budget,configured_expert_cache_ceiling_bytes=budget,
        total_reservation_bytes=progress,imminent_working_reserve_bytes=progress,
        headroom_bytes=0,metadata_reserve_bytes=0,ple_cache_bytes=16<<20,
        admission_limit_bytes=request.admission_limit_bytes,required_bytes=progress,
        ceiling_source=request.budget_mode,phase='pre_materialization',
        memory_admission_policy='attempt_after_reclaim',adaptive_residency=True,
        estimates_are_advisory=True,full_cache_preallocation=False)
    return result


def admission(artifact,request,policy):
    return source_admission(artifact,request,policy)

class LFUIndex:
    """Token-selection frequency; LRU ties; stale heap entries cannot evict leases."""
    def __init__(self):
        self.counts=Counter();self.last={};self.clock=0;self.heap=[]
    def observe(self,layer,ids):
        for expert,count in Counter(int(v) for v in np.asarray(ids).ravel()).items():
            self.counts[(layer,expert)]+=count
    def touch(self,key):
        self.clock+=1;self.last[key]=self.clock
        heapq.heappush(self.heap,(self.counts[key],self.clock,key))
    def victim(self,resident,leased,eligible):
        held=[]
        try:
            while self.heap:
                count,last,key=heapq.heappop(self.heap)
                if key not in resident or last!=self.last.get(key):continue
                if count!=self.counts[key]:
                    heapq.heappush(self.heap,(self.counts[key],last,key));continue
                if key in leased or not eligible(key):held.append((count,last,key));continue
                return key
            raise MemoryError('expert budget exhausted by protected/active objects')
        finally:
            for item in held:heapq.heappush(self.heap,item)

def read_atlas(path,source):
    if not path.exists():return {'schema':'mlx2.expert-frequency-atlas.v1','source':source,'clock':0,'counts':[]}
    value=json.loads(path.read_text())
    # Keep malformed advisory metadata on the ValueError path understood by
    # _persist_atlas_advisory. Missing fields must not raise KeyError after a
    # forward has already advanced its state or while closing its weights.
    if (not isinstance(value,dict) or value.get('schema')!='mlx2.expert-frequency-atlas.v1'
            or value.get('source')!=source or type(value.get('clock')) is not int
            or value['clock']<0 or not isinstance(value.get('counts'),list)):
        raise ValueError('expert atlas source/schema mismatch')
    seen=set()
    for row in value['counts']:
        if (not isinstance(row,dict)
                or any(type(row.get(name)) is not int
                       for name in ('layer','expert','token_selections','last_access'))):
            raise ValueError('invalid per-expert frequency atlas record')
        key=(row['layer'],row['expert'])
        if key in seen or not 0<=key[0]<40 or not 0<=key[1]<384 or row['token_selections']<0 or row['last_access']<0:
            raise ValueError('invalid per-expert frequency atlas record')
        seen.add(key)
    return value

def load_q1_manifest(root,source):
    root=Path(root).resolve();manifest=json.loads(checked_file(root,'manifest.json').read_text())
    digest=manifest.pop('fingerprint')
    if hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=digest:
        raise ValueError('Q1 manifest fingerprint mismatch')
    if manifest['source']!=source or manifest['quantization']!=Q1_REVISION or len(manifest['experts'])!=15360 or manifest.get('dense_modified') is not False:
        raise ValueError('Q1 bank is not the admitted Source expert-only derivative')
    records={}
    for record in manifest['experts']:
        key=(record['layer'],record['expert'])
        if key in records or not 0<=key[0]<40 or not 0<=key[1]<384 or record['payload_bytes']!=Q1_BYTES:
            raise ValueError('invalid Q1 expert inventory')
        path=checked_file(root,record['file']);records[key]={**record,'stat':_stamp(path.stat())}
        if records[key]['stat']['size']!=record['size']:raise ValueError('Q1 size changed')
    return root,records,digest

def q1_read(root,record):
    path=_verify_file(root,record['file'],record)
    with path.open('rb') as f:
        length=int.from_bytes(f.read(8),'little')
        if not 2<=length<=1<<20:raise ValueError('Q1 header bound')
        header=json.loads(f.read(length));header.pop('__metadata__',None)
    result=read_native_file(root,record['file'],{**record,'header_bytes':length+8,'tensors':header})
    expected={p+'.'+f for p in PROJECTIONS for f in ('qweight','scales','biases')}
    if set(result)!=expected or sum(v.nbytes for v in result.values())!=Q1_BYTES:raise ValueError('Q1 tensor inventory mismatch')
    for name,v in result.items():
        n,k=(5120,2304) if name.startswith('down.') else (2304,5120)
        shape=(n,k//(32 if name.endswith('qweight') else 64))
        dtype=np.dtype('<u4' if name.endswith('qweight') else '<f2')
        if v.shape!=shape or v.dtype!=dtype:raise ValueError('Q1 tensor geometry mismatch')
    return result

class LFUSourceExpertStore(SourceExpertStore):
    def __init__(self,artifact,bounds,policy,*,read_workers=2,packed_buffers=False):
        if type(packed_buffers) is not bool:raise ValueError('packed_buffers must be boolean')
        super().__init__(artifact,bounds,read_workers=read_workers)
        try:
            self.packed_buffers=packed_buffers
            self.policy=policy;self.index=LFUIndex();self.hot=frozenset();self.q1_root=None;self.q1_records={};self.q1_binding=None
            self.payload_ceiling=self.cache_budget_bytes-ORIGINAL_BYTES
            self.pool_bytes={'original':0,'q1':0};self.peak_payload_bytes=0
            self.stats.update(original_hits=0,original_misses=0,q1_hits=0,q1_misses=0,original_bytes_read=0,q1_bytes_read=0,q1_token_selections=0,original_token_selections=0)
            if packed_buffers:
                self.stats.update(packed_backing_allocations=0,packed_typed_views=0,packed_upload_bytes=0,packed_peak_retained_host_bytes=0)
            self.hot_binding=None;self.hot_map_provenance=None
            if policy['mode']=='hot_original_q1':
                self.q1_root,self.q1_records,self.q1_binding=load_q1_manifest(policy['q1_root'],self.binding)
                raw=Path(policy['hotset_path']).read_bytes();hot=json.loads(raw)
                if hot['source']!=self.binding or hot['original_expert_bytes']!=ORIGINAL_BYTES or hot['hot_budget_bytes']!=policy['hot_bytes']:
                    raise ValueError('hot expert map does not bind this Source/profile')
                self.hot=frozenset(tuple(k) for k in hot['experts'])
                if len(self.hot)!=len(hot['experts']) or any(k not in self.q1_records for k in self.hot) or len(self.hot)*ORIGINAL_BYTES>policy['hot_bytes']:
                    raise ValueError('hot map geometry/budget violation')
                self.hot_binding=hashlib.sha256(raw).hexdigest()
                from .deepseek_v41_source_defaults import HOT_MAP_SHA256
                from .deepseek_v41_source_hotmap import provenance
                self.hot_map_provenance=provenance(raw,hot,measured_sha256=HOT_MAP_SHA256,
                    source=self.binding,hot_budget_bytes=policy['hot_bytes'])
            self.atlas_path=Path(policy['atlas_path']) if policy.get('atlas_path') else None
            if self.atlas_path:
                atlas=read_atlas(self.atlas_path,self.binding);self.index.clock=atlas['clock']
                for row in atlas['counts']:
                    key=(row['layer'],row['expert']);self.index.counts[key]=row['token_selections'];self.index.last[key]=row['last_access']
            self._atlas_saved=self.index.counts.copy();self._atlas_saved_at=time.monotonic()
            self._init_residency()
            self.preheat_receipt={'requested':bool(policy.get('preheat',self.atlas_path is not None)),'performed':False}
            self.plan.transient_bound_bytes=0
            self.plan.allocation='global_byte_lfu'
            self.configure_adaptive_residency()
        except BaseException:
            self.pool.close()
            raise
    def kind(self,key):
        if self.q1_root is None:return 'original'
        # dynamic_precision_v1: the served representation, fixed between
        # cycle boundaries. Exact modes: the frozen hot map.
        high=getattr(self,'_dyn_high',None)
        return 'original' if key in (self.hot if high is None else high) else 'q1'
    # ---- Settable residency (provenance/deepseek-v41-source-expert-residency.json).
    def _init_residency(self):
        self.residency=residency.settings(self.policy)
        mode=self._residency_mode=self.residency['residency']
        self._cycle_pending=False;self._cycle_index=0;self._cycle_touched=set()
        self._miss_ewma={'original':0.0,'q1':0.0};self._miss_snapshot={'original':0,'q1':0}
        self._pre_evict_credit={'original':0,'q1':0};self._last_cycle=None
        self._high_fraction=self._high_fraction_source=None;self._pre_evict_slots=0
        self._init_prefill_residency()
        if mode==residency.RESIDENCY_SHARED:return
        self._high_fraction,self._high_fraction_source=residency.resolve_high_fraction(
            self.residency['high_fraction'],hot_objects=len(self.hot),original_bytes=ORIGINAL_BYTES,
            payload_maximum=self.maximum_cache_bytes-ORIGINAL_BYTES)
        self._pre_evict_slots=self.residency['pre_evict_slots']
        self.stats.update(pre_evicted_objects=0,pre_evicted_bytes=0,pre_evict_pressure_events=0,
            pre_evict_memory_pressure_events=0,sync_evictions=0,sync_evictions_original=0,sync_evictions_q1=0,
            sync_evictions_avoided=0,split_soft_overcommits=0,residency_cycles=0)
        if mode!=residency.RESIDENCY_DYNAMIC:return
        self._dyn_high=set();self._dyn_scores={};self._cycle_counts=Counter()
        self._cycle_mix={'original':0,'q1':0};self._mix_history=deque(maxlen=64)
        self._promotions={};self._promotion_pool=None;self._promotions_disabled=False
        self.stats.update(promotions=0,promotion_bytes=0,promotion_upload_ns=0,promotion_reads_issued=0,
            promotion_read_failures=0,promotions_dropped=0,demotions=0,demotion_bytes=0,
            dynamic_mxfp4_selections=0,dynamic_q1_selections=0)
        # Seed decayed usage from the persisted atlas at its per-token rate,
        # scaled to the decay equilibrium. Advisory: it ranks, never serves.
        total=sum(self.index.counts.values())
        top_k=getattr(self.plan,'top_k',None) or 6
        tokens=total/(max(1,len(getattr(self,'tables',{})) or 40)*top_k)
        if tokens>0:
            scale=1.0/(tokens*(1.0-self.residency['promote_decay']))
            self._dyn_scores={key:(count*scale,0) for key,count in self.index.counts.items() if count>0}
    def _split_active(self):
        return getattr(self,'_high_fraction',None) is not None and getattr(self,'adaptive_residency_enabled',False)
    def _dynamic(self):
        return getattr(self,'_residency_mode',residency.RESIDENCY_SHARED)==residency.RESIDENCY_DYNAMIC
    def pool_targets(self):
        """Live per-kind payload targets: high_fraction of the adaptive target."""
        payload=getattr(self,'_adaptive_payload_ceiling',None)
        if payload is None:payload=self.payload_ceiling
        payload=max(0,min(payload,self.maximum_cache_bytes-ORIGINAL_BYTES))
        high=int(payload*self._high_fraction)
        return {'original':high,'q1':payload-high}
    def _victim_of_kind(self,kind,protected):
        pinned=getattr(self,'_resident_bank_pinned_keys',frozenset())
        shield=self._active_shield()
        try:return self.index.victim(self._cache,protected,lambda k:k not in pinned and k not in shield and self.kind(k)==kind)
        except MemoryError:return None
    def _evict_resident(self,key):
        if self._dynamic() and key in self._dyn_high:return self._demote(key)
        return self._adaptive_evict_key(key)
    def _demote(self,key):
        # Evict while the key still reads as original, then serve Q1 next.
        amount=self._adaptive_evict_key(key);self._dyn_high.discard(key)
        self.stats['demotions']+=1;self.stats['demotion_bytes']+=amount
        return amount
    def _score(self,key):
        value,stamp=self._dyn_scores.get(key,(0.0,self._cycle_index))
        return value*self.residency['promote_decay']**(self._cycle_index-stamp)
    def _coldest_high(self,protected):
        pinned=getattr(self,'_resident_bank_pinned_keys',frozenset())
        best=None
        for key in self._dyn_high:
            if key in protected or key in pinned or key not in self._cache:continue
            rank=(self._score(key),self.index.last.get(key,0),key)
            if best is None or rank<best:best=rank
        return None if best is None else best[2]
    def _expert_read_job(self,key):
        """The exact host read acquire performs for key; shared by early reads."""
        if self.kind(key)=='q1':return (q1_read,self.q1_root,self.q1_records[key])
        if getattr(self,'packed_buffers',False):return (read_original_packed,self.artifact,*key)
        return (read_tensors,self.artifact,expert_names(*key))
    def configure_adaptive_residency(self,enabled=True,**options):
        if enabled is False:
            # Refuse before changing any target, pool, or precision owner.
            parse_policy(self.policy,adaptive=False)
        # overlap_v1 early reads hold bounded host slabs before their upload.
        options.setdefault('staging_bytes',ORIGINAL_BYTES+self.overlap_host_reserve_bytes()+self.promotion_host_reserve_bytes()+self.readahead_host_reserve_bytes())
        options.setdefault('staging_within_cache',True)
        result=super().configure_adaptive_residency(enabled,**options)
        if enabled is False:
            # _evict_for wrote the adaptive payload target here; fixed pools
            # are bounded by the configured budget again.
            self.payload_ceiling=self.cache_budget_bytes-ORIGINAL_BYTES
        return result
    def promotion_host_reserve_bytes(self):
        return self.residency['promote_slots']*ORIGINAL_BYTES if self._dynamic() else 0
    def _adaptive_victim(self):
        # Live-memory pressure eviction is never shielded by protect_decode_v1.
        shielded=getattr(self,'_shield_on',False);self._shield_on=False
        try:return self._adaptive_victim_unshielded()
        finally:self._shield_on=shielded
    def _adaptive_victim_unshielded(self):
        protected={**self._leased,**self._pending_uploads}
        pinned=getattr(self,'_resident_bank_pinned_keys',frozenset())
        if self._split_active():
            # Only a pool above its own reservation yields; the high pool of
            # dynamic precision changes only at cycle boundaries.
            targets=self.pool_targets()
            for kind in sorted(targets,key=lambda k:targets[k]-self.pool_bytes[k]):
                if self.pool_bytes[kind]<=targets[kind] or kind=='original' and self._dynamic():continue
                victim=self._victim_of_kind(kind,protected)
                if victim is not None:return victim
            return None
        try:return self.index.victim(self._cache,protected,lambda key:key not in pinned)
        except MemoryError:return None
    def _adaptive_evict_key(self,key):
        if key in getattr(self,'_resident_bank_pinned_keys',frozenset()):
            raise ValueError('Cannot individually evict an attached resident bank allocation')
        kind=self.kind(key);amount=super()._adaptive_evict_key(key)
        self.pool_bytes[kind]-=amount
        return amount
    def observe_routes(self,prefix,ids,*,histogram=None):
        layer=int(prefix.split('.')[1])
        if histogram is None:
            counts=Counter(int(v) for v in np.asarray(ids).ravel()).items()
        else:
            # Borrow the frame planner's run counts; no GPU readback or
            # second route histogram. Dynamic counts never outlive the call.
            experts,amounts=map(np.asarray,histogram)
            if (experts.ndim!=1 or amounts.shape!=experts.shape
                    or experts.dtype.kind not in 'iu' or amounts.dtype.kind not in 'iu'
                    or not experts.size or np.any(experts>=384)
                    or np.any(experts<0) or np.any(amounts<=0)
                    or np.any(experts[1:]<=experts[:-1])
                    or int(amounts.sum())!=np.asarray(ids).size):
                raise ValueError('Source frame route histogram geometry mismatch')
            counts=zip(map(int,experts),map(int,amounts))
        dynamic=self._dynamic()
        for expert,count in counts:
            key=(layer,expert);kind=self.kind(key)
            self.index.counts[key]+=count
            self.stats[kind+'_token_selections']+=count
            if dynamic:
                self._cycle_counts[key]+=count;self._cycle_mix[kind]+=count
                self.stats['dynamic_mxfp4_selections' if kind=='original' else 'dynamic_q1_selections']+=count
    def _evict_for_split(self,kind,size):
        """Synchronous miss eviction within the incoming kind's own pool.

        The other pool yields only bytes above its own reservation. Active
        leases/pending uploads may exceed a soft target; the configured
        maximum stays a hard bound.
        """
        self.adaptive_residency_update()
        self.payload_ceiling=self._adaptive_payload_ceiling
        targets=self.pool_targets();hard=self.maximum_cache_bytes-ORIGINAL_BYTES
        other='q1' if kind=='original' else 'original'
        dynamic=self._dynamic();protected={**self._leased,**self._pending_uploads}
        evicted=0
        for _ in range(len(self._cache)+1):
            own=self.pool_bytes[kind]+size>targets[kind]
            soft=self._cache_bytes+size>targets['original']+targets['q1']
            total=self._cache_bytes+size>hard
            if not (own or soft or total):break
            victim=None
            if (own or total) and not (dynamic and kind=='original'):
                victim=self._victim_of_kind(kind,protected)
            if victim is None and (soft or total) and self.pool_bytes[other]>targets[other] and not (dynamic and other=='original'):
                victim=self._victim_of_kind(other,protected)
            if victim is None:
                if total:raise MemoryError('Configured expert maximum exhausted by protected/active objects; use a sequential completed-expert window')
                self.stats['split_soft_overcommits']+=1;break
            victim_kind=self.kind(victim)
            self._evict_resident(victim);evicted+=1
            self.stats['sync_evictions']+=1;self.stats['sync_evictions_'+victim_kind]+=1
        else:raise RuntimeError('Adaptive expert eviction made no progress')
        if not evicted and self._pre_evict_credit[kind]>=size:
            # This miss found a slot that pre-eviction freed at a boundary.
            self._pre_evict_credit[kind]-=size;self.stats['sync_evictions_avoided']+=1
    def _evict_for(self,key,size):
        kind=self.kind(key)
        if getattr(self,'adaptive_residency_enabled',False) and self._split_active():
            return self._evict_for_split(kind,size)
        if getattr(self,'adaptive_residency_enabled',False):
            self.adaptive_residency_update(incoming_bytes=size)
            self.payload_ceiling=self._adaptive_payload_ceiling
            maximum=self.maximum_cache_bytes-ORIGINAL_BYTES
            # The hot map fixes expert precision, not a prepaid cache pool.
            # Adaptive original/Q1 payloads share the measured total capacity.
            # A soft live-memory target can be exceeded by immutable active
            # leases, but the operator's configured maximum remains bounded.
            for _ in range(len(self._cache)+1):
                if self._cache_bytes+size<=maximum:return
                shield=self._active_shield()
                try:victim=self.index.victim(self._cache,{**self._leased,**self._pending_uploads},
                    (lambda k:k not in getattr(self,'_resident_bank_pinned_keys',frozenset()) and k not in shield))
                except MemoryError:raise MemoryError('Configured expert maximum exhausted by protected/active objects; use a sequential completed-expert window')from None
                self._adaptive_evict_key(victim)
            raise RuntimeError('Adaptive expert eviction made no progress')
        limit=self.payload_ceiling if self.q1_root is None else (self.policy['hot_bytes'] if kind=='original' else self.policy['budget_bytes']-self.policy['hot_bytes']-ORIGINAL_BYTES)
        while self.pool_bytes[kind]+size>limit or self._cache_bytes+size>self.payload_ceiling:
            victim=self.index.victim(self._cache,self._leased,lambda k:k not in getattr(self,'_resident_bank_pinned_keys',frozenset()) and self.kind(k)==kind and (self.q1_root is None or kind=='q1'))
            arrays=self._cache.pop(victim);amount=sum(v.nbytes for v in arrays.values());self._cache_bytes-=amount;self.pool_bytes[kind]-=amount;self.stats['evictions']+=1
    def _upload_expert(self,key,kind,size,raw):
        """Upload one host expert; pending owners survive a failed evaluation."""
        import mlx.core as mx
        if getattr(self,'packed_buffers',False) and kind=='q1':raw=borrow_q1_packed(raw)
        self._pending_host[key]=raw;arrays=self._pending_uploads[key]={}
        if getattr(self,'packed_buffers',False):
            if raw.host_allocation_nbytes>ORIGINAL_BYTES:raise ValueError('packed expert host allocation exceeds staging reserve')
            self.stats['packed_peak_retained_host_bytes']=max(self.stats['packed_peak_retained_host_bytes'],raw.host_allocation_nbytes)
            stream=self._upload_stream()
            arrays=self._pending_uploads[key]=materialize(raw) if stream is None else materialize(raw,stream=stream)
            # Mapping subclasses are not guaranteed evaluator pytrees.
            # The pending host and full backing survive eval failures.
            mx.eval(arrays.backing,*arrays.values())
            if self.expert_overlap_enabled():
                self.stats['upload_offqueue_evals']+=1
            observer=getattr(self,'_packed_original_loaded',None) if kind=='original' else None
            if observer is not None:
                observer(key,raw,arrays)
            if arrays.allocation_nbytes!=size:raise ValueError('packed expert backing exceeded admitted bytes')
            self.stats['packed_backing_allocations']+=1
            self.stats['packed_typed_views']+=len(arrays)
            self.stats['packed_upload_bytes']+=arrays.allocation_nbytes
        elif kind=='q1':
            arrays.update({name:mx.array(value) for name,value in raw.items()})
        else:
            for name,part in PROJECTIONS.items():
                prefix_name=f'layers.{key[0]}.ffn.experts.{key[1]}.{part}.'
                arrays[name+'.qweight']=mx.array(raw[prefix_name+'weight'].view('<u4'))
                arrays[name+'.scales']=mx.array(raw[prefix_name+'scale'])
        if not getattr(self,'packed_buffers',False):mx.eval(arrays)
        if sum(v.nbytes for v in arrays.values())!=size:raise ValueError('LFU upload exceeded admitted bytes')
        self._pending_host.pop(key)
        return arrays
    def _insert_resident(self,key,kind,size,arrays):
        self._cache[key]=arrays;self._cache_bytes+=size;self.pool_bytes[kind]+=size
        self.peak_payload_bytes=max(self.peak_payload_bytes,self._cache_bytes)
    def acquire(self,prefix,expert):
        import mlx.core as mx
        with self._lock:
            self._ready()
            if not self._active or prefix not in self.tables or type(expert) is not int or not 0<=expert<384:
                raise ValueError('LFU expert acquisition outside ownership')
            key=self.tables[prefix][expert];kind=self.kind(key)
            touched=getattr(self,'_cycle_touched',None)
            if touched is not None:touched.add(key)
            if kind=='original':
                # Resolve every tensor on every acquire, then check each owning
                # shard once. Fresh verification precedes BOTH hit branches;
                # no seal result is retained across experts or windows.
                # See provenance/deepseek-v41-source-binding-checks.json.
                files=dict.fromkeys(self.artifact['tensors'][name]['file'] for name in expert_names(*key))
                for file in files:
                    checked_file(self.root,file,self.artifact['seals'][file])
            else:_verify_file(self.q1_root,self.q1_records[key]['file'],self.q1_records[key])
            if key in self._leased:
                collector=getattr(self,'_completion_collector',None)
                if collector is not None:collector(key)
                return self._leased[key]
            transient=getattr(self,'_prefill_transient',None)
            if transient and key[0]!=self._prefill_transient_layer:
                self._clear_prefill_transient();transient=None
            if key in self._cache:
                arrays=self._cache[key];self.stats['cache_hits']+=1;self.stats[kind+'_hits']+=1
                shield=getattr(self,'_prefill_shield',None)
                if shield is not None and key in shield:self.stats['prefill_shield_retained_hits']+=1
            elif transient and key in transient:
                # protect_decode_v1: a same-layer stream-through object; the
                # identical bytes an ordinary miss would upload again.
                arrays=transient[key];self.stats['cache_hits']+=1;self.stats[kind+'_hits']+=1
                self.stats['prefill_transient_hits']+=1
            else:
                size=Q1_BYTES if kind=='q1' else ORIGINAL_BYTES;insert=self._admit_miss(key,size)
                # _read_expert owns demand_wait_ns; overlap_v1 may consume an
                # early read of these exact bytes instead of reading now.
                raw=self._read_expert(key,*self._expert_read_job(key))
                arrays=self._upload_expert(key,kind,size,raw);del raw
                if insert:self._insert_resident(key,kind,size,arrays)
                else:self._hold_prefill_transient(key,size,arrays)
                self.stats['cache_misses']+=1;self.stats[kind+'_misses']+=1;self.stats['uploaded_experts']+=1;self.stats['upload_batches']+=1
                read_size=self.q1_records[key]['size'] if kind=='q1' else size
                self.stats['bytes_read']+=read_size;self.stats[kind+'_bytes_read']+=read_size
            self.index.touch(key)
            if len(self.index.heap)>max(1024,4*len(self._cache)):
                self.index.heap=[(self.index.counts[k],self.index.last[k],k) for k in self._cache]
                heapq.heapify(self.index.heap)
            self._leased[key]=arrays;self._pending_uploads.pop(key,None)
            collector=getattr(self,'_completion_collector',None)
            if collector is not None:collector(key)
            return arrays
    # ---- Opt-in prefill_residency=protect_decode_v1
    # (provenance/deepseek-v41-source-prefill-residency.json).
    def _init_prefill_residency(self):
        self._prefill_residency=self.residency.get('prefill_residency',prefill_residency.EVICT)
        self._prefill_shield=None;self._shield_on=False
        self._prefill_transient={};self._prefill_transient_bytes=0;self._prefill_transient_layer=None
        if self._prefill_residency==prefill_residency.PROTECT:
            for name in prefill_residency.COUNTERS:self.stats.setdefault(name,0)
    def _active_shield(self):
        shield=getattr(self,'_prefill_shield',None)
        return shield if shield is not None and getattr(self,'_shield_on',False) else frozenset()
    def observe_moe_call(self,prefix,rows):
        """Weights hook before a MoE call's acquires: prefill shield and read-ahead."""
        if type(rows) is not int or rows<1 or prefix not in self.tables:
            raise ValueError('Source MoE call observation outside admitted layers')
        with self._lock:
            if getattr(self,'_prefill_residency',prefill_residency.EVICT)==prefill_residency.PROTECT:
                if rows>=prefill_residency.PROTECT_MIN_ROWS:
                    if self._prefill_shield is None:
                        self._prefill_shield=frozenset(self._cache)
                        self.stats['prefill_shield_snapshots']+=1
                        self.stats['prefill_shield_snapshot_objects']+=len(self._prefill_shield)
                elif self._prefill_shield is not None:
                    self._release_prefill_shield()
            readahead=getattr(self,'_prefill_readahead',None)
            if readahead is not None:readahead.observe_call(prefix,rows)
    def _release_prefill_shield(self):
        self._prefill_shield=None;self.stats['prefill_shield_releases']+=1
        self._clear_prefill_transient()
    def _clear_prefill_transient(self):
        self._prefill_transient={};self._prefill_transient_bytes=0;self._prefill_transient_layer=None
    def _admit_miss(self,key,size):
        """Evict for a miss; False means serve it as a bounded transient."""
        if getattr(self,'_prefill_shield',None) is None or not getattr(self,'adaptive_residency_enabled',False):
            self._evict_for(key,size);return True
        before=self.stats.get('split_soft_overcommits',0)
        self._shield_on=True
        try:
            self._evict_for(key,size);blocked=self.stats.get('split_soft_overcommits',0)!=before
        except MemoryError:
            blocked=True
        finally:
            self._shield_on=False
        if not blocked:return True
        # Only shielded objects could make room. A shielded soft overcommit
        # is not the ordinary lease overcommit; do not report it as one.
        if 'split_soft_overcommits' in self.stats:self.stats['split_soft_overcommits']=before
        self.stats['prefill_shield_blocks']+=1
        hard=self.maximum_cache_bytes-ORIGINAL_BYTES
        if (self._prefill_transient_bytes+size<=prefill_residency.TRANSIENT_MAX_BYTES
                and self._cache_bytes+self._prefill_transient_bytes+size<=hard):
            return False
        # Transients would exceed their bound: the shield yields.
        evicted=self.stats['evictions'];resident=self._cache_bytes
        self._evict_for(key,size)
        self.stats['prefill_shield_yields']+=max(1,self.stats['evictions']-evicted)
        self.stats['prefill_shield_yield_bytes']+=max(0,resident-self._cache_bytes)
        return True
    def _hold_prefill_transient(self,key,size,arrays):
        if self._prefill_transient_layer!=key[0]:self._clear_prefill_transient()
        self._prefill_transient_layer=key[0]
        self._prefill_transient[key]=arrays;self._prefill_transient_bytes+=size
        self.stats['prefill_transient_uploads']+=1;self.stats['prefill_transient_bytes']+=size
        self.stats['prefill_transient_peak_bytes']=max(self.stats['prefill_transient_peak_bytes'],self._prefill_transient_bytes)
    # ---- Opt-in deepseek_source_prefill_readahead=layer_ahead_v1
    # (provenance/deepseek-v41-source-prefill-readahead.json).
    def enable_prefill_readahead(self,*,pool=None):
        from .deepseek_v41_source_prefill_readahead import LayerReadAhead
        with self._lock:
            self._ready()
            if self._active or getattr(self,'_prefill_readahead',None) is not None:
                raise ValueError('Source prefill read-ahead must be enabled once on an idle provider')
            if self._dynamic():
                raise ValueError('Source prefill read-ahead cannot combine with dynamic precision')
            if self.expert_overlap_enabled():
                raise ValueError('Source prefill read-ahead cannot combine with overlap_v1')
            self._prefill_readahead=LayerReadAhead(self,pool=pool)
            if getattr(self,'_adaptive_residency_enabled',False):
                self._adaptive_staging=max(self._adaptive_staging,
                    ORIGINAL_BYTES+self.overlap_host_reserve_bytes()+self.promotion_host_reserve_bytes()+self.readahead_host_reserve_bytes())
            return self._prefill_readahead.receipt()
    def readahead_host_reserve_bytes(self):
        readahead=getattr(self,'_prefill_readahead',None)
        return 0 if readahead is None else readahead.max_bytes
    def _read_expert(self,key,function,*args):
        readahead=getattr(self,'_prefill_readahead',None)
        if readahead is not None and readahead.reads:
            started=time.perf_counter_ns()
            raw=readahead.take(key,function)
            if raw is not None:
                self.stats['demand_wait_ns']+=time.perf_counter_ns()-started
                return raw
        return super()._read_expert(key,function,*args)
    def apc_revision(self):
        value=REVISION+':'+str(self.q1_binding)+':'+str(self.hot_binding)
        # Exact split pools and a live maximum change residency, not numerics.
        if self._dynamic():value+=':residency='+residency.RESIDENCY_DYNAMIC
        return value
    def identity_hot_binding(self):
        """Hot-map digest bound into the effective artifact identity.

        Under dynamic precision the map only seeds residency; the served
        precision depends on cache history, so its identity is salted.
        """
        if not self._dynamic() or self.hot_binding is None:return self.hot_binding
        return hashlib.sha256(('mlx2.source-expert-residency.'+residency.RESIDENCY_DYNAMIC+':'+self.hot_binding).encode()).hexdigest()
    # ---- Cycle boundary: after a forward's provider window has closed.
    @contextmanager
    def window(self):
        with super().window():
            yield
        if getattr(self,'_cycle_pending',False) and not self._active:
            self._run_cycle_boundary()
    def _boundary_protected(self):
        hard=set(self._leased)|set(self._pending_uploads)|set(getattr(self,'_resident_bank_pinned_keys',()))
        soft=set(hard)|set(self._cycle_touched)|set(getattr(self,'_promotions',()))
        overlap=getattr(self,'_overlap',None)
        if overlap is not None:
            soft|=set(overlap['hints'])
            ticket=overlap['ticket']
            if ticket is not None:soft|=set(ticket.demand)|set(ticket.queued)
        return hard,soft
    def _run_cycle_boundary(self):
        """Promotions, demotions and bounded pre-eviction between forwards."""
        with self._lock:
            if (self._active or self._closed or self._parked or self._error
                    or getattr(self,'_residency_mode',residency.RESIDENCY_SHARED)==residency.RESIDENCY_SHARED):
                return None
            self._cycle_pending=False;self._cycle_index+=1;self.stats['residency_cycles']+=1
            misses={}
            for kind in ('original','q1'):
                misses[kind]=self.stats[kind+'_misses']-self._miss_snapshot[kind]
                self._miss_snapshot[kind]=self.stats[kind+'_misses']
                self._miss_ewma[kind]=0.5*self._miss_ewma[kind]+0.5*misses[kind]
            before={k:self.stats[k] for k in ('pre_evicted_objects','pre_evicted_bytes')}
            self.adaptive_residency_update()
            hard,soft=self._boundary_protected()
            dynamic=self._dynamic_boundary(hard,soft) if self._dynamic() else None
            targets=self.pool_targets()
            for kind in targets:
                self._pre_evict_credit[kind]=max(0,min(self._pre_evict_credit[kind],targets[kind]-self.pool_bytes[kind]))
            self._pre_evict(soft)
            self._last_cycle=dict(cycle=self._cycle_index,misses=misses,
                expected_misses={k:round(v,3) for k,v in self._miss_ewma.items()},
                pool_targets=self.pool_targets(),pool_bytes=dict(self.pool_bytes),
                pre_evicted_objects=self.stats['pre_evicted_objects']-before['pre_evicted_objects'],
                pre_evicted_bytes=self.stats['pre_evicted_bytes']-before['pre_evicted_bytes'],
                dynamic=dynamic)
            self._cycle_touched=set()
            return dict(self._last_cycle)
    def _pre_evict(self,protected):
        """Free up to pre_evict_slots coldest unprotected objects per pressured pool."""
        slots=self._pre_evict_slots
        if not slots:return 0
        targets=self.pool_targets();last=getattr(self,'_adaptive_last',{})
        sizes={'original':ORIGINAL_BYTES,'q1':Q1_BYTES}
        # Round half up: a decayed EWMA tail must not keep a quiet pool pressured.
        expected={k:int(self._miss_ewma[k]+0.5)*sizes[k] for k in sizes}
        headroom=max(getattr(self,'_adaptive_headroom',0),getattr(self,'_adaptive_protected_bytes',0))
        memory=(last.get('availability_known') is True
                and last.get('measured_available_bytes',0)<headroom+sum(expected.values()))
        if memory:self.stats['pre_evict_memory_pressure_events']+=1
        freed=0
        for kind in ('q1','original'):
            need=self.pool_bytes[kind]+expected[kind]-targets[kind]
            if need<=0 and not memory:continue
            self.stats['pre_evict_pressure_events']+=1
            count=slots if memory else min(slots,-(-need//sizes[kind]))
            for _ in range(count):
                victim=self._victim_of_kind(kind,protected)
                if victim is None:break
                amount=self._evict_resident(victim);freed+=amount
                self.stats['pre_evicted_objects']+=1;self.stats['pre_evicted_bytes']+=amount
                self._pre_evict_credit[kind]+=amount
        return freed
    def _original_read_job(self,key):
        if getattr(self,'packed_buffers',False):return (read_original_packed,self.artifact,*key)
        return (read_tensors,self.artifact,expert_names(*key))
    def _promotion_reader(self):
        if self._promotion_pool is None:
            from ..runtime.io_ownership import ReadPool,storage_owner
            self._promotion_pool=ReadPool(2,'v41-source-expert-promotion',timeout=60,
                max_pending=self.residency['promote_slots'],
                owner=storage_owner('v41-source-expert-promotion',self.root))
        return self._promotion_pool
    def _install_promotion(self,key,raw,soft):
        """Upload a background-read MXFP4 copy at a boundary; False if stale."""
        size=ORIGINAL_BYTES;hysteresis=self.residency['promote_hysteresis']
        if key in self._dyn_high or key in self._leased or key in self._pending_uploads:return False
        for file in dict.fromkeys(self.artifact['tensors'][name]['file'] for name in expert_names(*key)):
            checked_file(self.root,file,self.artifact['seals'][file])
        score=self._score(key);targets=self.pool_targets()
        hard=self.maximum_cache_bytes-ORIGINAL_BYTES
        q1_copy=Q1_BYTES if key in self._cache else 0
        # Room in the high pool comes only from colder high entries.
        displaced=[];room=self.pool_bytes['original']
        protected=soft|{key}
        while room+size>targets['original']:
            victim=self._coldest_high(protected|set(displaced))
            if victim is None or self._score(victim)*hysteresis>=score:
                self.stats['promotions_dropped']+=1;return False
            displaced.append(victim);room-=ORIGINAL_BYTES
        projected=self._cache_bytes-ORIGINAL_BYTES*len(displaced)-q1_copy+size
        if projected>hard:
            self.stats['promotions_dropped']+=1;return False
        for victim in displaced:self._demote(victim)
        if q1_copy:self._adaptive_evict_key(key)
        started=time.perf_counter_ns()
        try:
            arrays=self._upload_expert(key,'original',size,raw)
        except BaseException as error:
            # A failed upload poisons the provider, as in a window.
            self._error=str(error)
            raise
        finally:
            self.stats['promotion_upload_ns']+=time.perf_counter_ns()-started
        self._dyn_high.add(key);self._insert_resident(key,'original',size,arrays)
        self._pending_uploads.pop(key,None);self.index.touch(key)
        self.stats['promotions']+=1;self.stats['promotion_bytes']+=size
        return True
    def _dynamic_boundary(self,hard,soft):
        index=self._cycle_index
        for key,count in self._cycle_counts.items():
            self._dyn_scores[key]=(self._score(key)+count,index)
        observed=list(self._cycle_counts);self._cycle_counts.clear()
        mix=dict(self._cycle_mix);self._cycle_mix={'original':0,'q1':0}
        before={k:self.stats[k] for k in ('promotions','demotions','promotions_dropped','promotion_bytes')}
        for key,future in list(self._promotions.items()):
            if not future.done():continue
            del self._promotions[key]
            try:raw=future.result()
            except Exception:
                # Optional reads never poison; promotions stop for this load.
                self.stats['promotion_read_failures']+=1;self._promotions_disabled=True;continue
            self._install_promotion(key,raw,soft);del raw
        targets=self.pool_targets()
        # Demote under pressure: the high target tracks live memory.
        while self.pool_bytes['original']>targets['original']:
            victim=self._coldest_high(hard)
            if victim is None:break
            self._demote(victim)
        if not self._promotions_disabled:
            hysteresis=self.residency['promote_hysteresis'];floor=float(self.residency['promote_min_score'])
            if self.pool_bytes['original']+ORIGINAL_BYTES>targets['original']:
                coldest=self._coldest_high(soft)
                floor=max(floor,math.inf if coldest is None else self._score(coldest)*hysteresis)
            candidates=sorted((k for k in observed if k not in self._dyn_high and k not in self._promotions
                               and self._score(k)>=floor),key=lambda k:(-self._score(k),k))
            from ..runtime.io_ownership import ReadQueueFull
            for key in candidates[:max(0,self.residency['promote_slots']-len(self._promotions))]:
                try:future=self._promotion_reader().submit(*self._original_read_job(key))
                except ReadQueueFull:break
                self._promotions[key]=future;self.stats['promotion_reads_issued']+=1
        row=dict(cycle=index,mxfp4_selections=mix['original'],q1_selections=mix['q1'],
                 **{k:self.stats[k]-before[k] for k in before},promotions_inflight=len(self._promotions))
        self._mix_history.append(row)
        return row
    def _drain_promotions(self,timeout=60.0):
        promotions=getattr(self,'_promotions',None)
        if not promotions:return
        for future in tuple(promotions.values()):
            if future.cancel():continue
            try:future.result(timeout=timeout)
            except Exception:pass
        promotions.clear()
    def residency_receipt(self):
        mode=getattr(self,'_residency_mode',residency.RESIDENCY_SHARED)
        bounds=getattr(self,'bounds',None)
        result=dict(mode=mode,settings=dict(getattr(self,'residency',{})),
            cache_maximum=(bounds.get('expert_cache_maximum') if isinstance(bounds,dict) else None)
                or {'policy':'explicit','maximum_bytes':getattr(self,'maximum_cache_bytes',None)},
            qualification='unqualified')
        result['prefill_residency']=prefill_residency.receipt(self)
        if mode==residency.RESIDENCY_SHARED:
            result.update(pools='shared_total_adaptive',precision='frozen hot map')
            return result
        result.update(pools='split_kind_pools_adaptive',high_fraction=self._high_fraction,
            high_fraction_source=self._high_fraction_source,pool_targets=self.pool_targets(),
            pool_bytes=dict(self.pool_bytes),pre_evict_slots=self._pre_evict_slots,
            pre_evict_credit=dict(self._pre_evict_credit),last_cycle=self._last_cycle,
            counters={k:self.stats[k] for k in ('pre_evicted_objects','pre_evicted_bytes','pre_evict_pressure_events',
                'pre_evict_memory_pressure_events','sync_evictions','sync_evictions_original','sync_evictions_q1',
                'sync_evictions_avoided','split_soft_overcommits','residency_cycles')},
            precision='frozen hot map' if mode==residency.RESIDENCY_SPLIT else
                'dynamic per cycle boundary; approximate; depends on cache history')
        if mode==residency.RESIDENCY_DYNAMIC:
            result['dynamic']=dict(high_objects=len(self._dyn_high),promotions_inflight=len(self._promotions),
                promotions_disabled=self._promotions_disabled,history=list(self._mix_history),
                counters={k:self.stats[k] for k in ('promotions','promotion_bytes','promotion_upload_ns',
                    'promotion_reads_issued','promotion_read_failures','promotions_dropped','demotions',
                    'demotion_bytes','dynamic_mxfp4_selections','dynamic_q1_selections')},
                apcv2='disabled: '+residency.DYNAMIC_APC_REASON)
        return result
    def _packed_backing_totals(self):
        # Packed construction starts empty and only admits one backing per key.
        # Cache counters therefore cover resident objects without a full walk.
        # Failed uploads can retain another object at the same key; count that
        # distinct owner, and deduplicate owners shared by leases/pending uploads.
        count,size=len(self._cache),self._cache_bytes
        extra=set()
        for owners in (self._leased,self._pending_uploads):
            for key,value in owners.items():
                if self._cache.get(key) is value or not hasattr(value,'backing'):continue
                identity=id(value)
                if identity not in extra:
                    extra.add(identity);count+=1;size+=value.allocation_nbytes
        return count,size
    def _transient_unleased_bytes(self):
        transient=getattr(self,'_prefill_transient',None) or {}
        return sum(getattr(v,'allocation_nbytes',None) or sum(a.nbytes for a in v.values())
                   for k,v in transient.items() if self._leased.get(k) is not v)
    def live_bytes(self):
        if not getattr(self,'packed_buffers',False):return super().live_bytes()+self._transient_unleased_bytes()
        return self._packed_backing_totals()[1]+self._transient_unleased_bytes()+sum(len(t._cache)*t.row_bytes for t in self.ple_tables.values())
    resident_bytes=live_bytes
    def receipt(self):
        result=super().receipt()
        result.update(allocation='LFU by token-expert selections, LRU ties',policy=self.policy,quantization=Q1_REVISION if self.q1_root else 'original',q1_bank_binding=self.q1_binding,hot_map_binding=self.hot_binding,hot_expert_objects=len(self.hot),pool_payload_bytes=self.pool_bytes.copy(),peak_expert_payload_bytes=self.peak_payload_bytes,expert_total_payload_and_host_staging_ceiling=self.cache_budget_bytes,host_staging_reserve_bytes=ORIGINAL_BYTES,precision_selection='fixed per expert for profile lifetime',expert_requantization=self.q1_root is not None)
        result['pool_capacity_policy']='shared_total_adaptive'if self.adaptive_residency_enabled else'legacy_fixed_kind_pools'
        result['hot_budget_role']='immutable_precision_selection_only'if self.adaptive_residency_enabled else'precision_selection_and_kind_pool_limit'
        result['expert_residency']=self.residency_receipt()
        readahead=getattr(self,'_prefill_readahead',None)
        if readahead is not None:result['prefill_readahead']=readahead.receipt()
        result['hot_map_provenance']=getattr(self,'hot_map_provenance',None)
        if self._split_active():result['pool_capacity_policy']='split_kind_pools_adaptive'
        if self._dynamic():
            result.update(hot_budget_role='dynamic_precision_seed_only',
                precision_selection='dynamic per cycle boundary; fixed within each forward; approximate')
        result['plan']['allocation']='global_byte_lfu'
        result['numeric_profile']='source_hot_original_cold_q1_g64' if self.q1_root else 'official_mxfp4_fp8'
        if self._dynamic():result['numeric_profile']='source_dynamic_precision_mxfp4_q1g64_v1'
        result.update(atlas_path=str(self.atlas_path) if self.atlas_path else None,atlas_known_experts=len(self.index.counts),atlas_persist_error=getattr(self,'atlas_persist_error',None),preheat=self.preheat_receipt)
        if getattr(self,'packed_buffers',False):
            backing_count,backing_bytes=self._packed_backing_totals()
            result.update(packed_buffers=True,packed_buffer_layout='source-expert-single-backing-v1',
                          packed_live_backing_allocations=backing_count,
                          packed_live_backing_payload_bytes=backing_bytes,
                          pending_host_source_bytes=sum(v.host_allocation_nbytes for v in self._pending_host.values()),
                          packed_allocation_counter_scope='successful scheduled backing uploads; physical Metal allocations require separate observation')
        return result
    def frequency_snapshot(self):
        return {'source':self.binding,'original_expert_bytes':ORIGINAL_BYTES,'counts':[{'layer':k[0],'expert':k[1],'token_selections':count,'last_access':self.index.last.get(k,0)} for k,count in self.index.counts.items()]}
    def persist_atlas(self):
        if self.atlas_path is None:return
        delta=self.index.counts-self._atlas_saved
        if not delta:return
        import fcntl
        path=self.atlas_path;path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix(path.suffix+'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            prior=read_atlas(path,self.binding);rows={(r['layer'],r['expert']):dict(r) for r in prior['counts']}
            for key,count in delta.items():
                row=rows.setdefault(key,{'layer':key[0],'expert':key[1],'token_selections':0,'last_access':0})
                row['token_selections']+=count;row['last_access']=max(row['last_access'],self.index.last.get(key,0))
            value={'schema':'mlx2.expert-frequency-atlas.v1','source':self.binding,'clock':max(prior['clock'],self.index.clock),'counts':list(rows.values())}
            temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
            # The flock owns this per-process name: a leftover from an earlier
            # failed write is ours and must not wedge every later persist.
            temporary.unlink(missing_ok=True)
            try:
                with temporary.open('x') as out:json.dump(value,out);out.write('\n');out.flush();os.fsync(out.fileno())
                os.replace(temporary,path)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            self._atlas_saved=self.index.counts.copy();self._atlas_saved_at=time.monotonic()
    def _persist_atlas_advisory(self):
        # The atlas is advisory warmup evidence. A write failure must not fail
        # a forward whose state already advanced, nor quarantine the provider
        # at close; it is retained in the receipt and retried later.
        try:self.persist_atlas()
        except (OSError,ValueError,TypeError) as error:
            self._atlas_saved_at=time.monotonic()
            self.stats['atlas_persist_failures']=self.stats.get('atlas_persist_failures',0)+1
            self.atlas_persist_error={'type':type(error).__name__,'message':str(error)}
    def end_context(self):
        super().end_context()
        readahead=getattr(self,'_prefill_readahead',None)
        if readahead is not None:
            with self._lock:readahead.end_wave()
        if getattr(self,'_residency_mode',residency.RESIDENCY_SHARED)!=residency.RESIDENCY_SHARED:
            # The forward is complete. A bounded-completion window may still
            # own leases; then the boundary runs when that window closes.
            self._cycle_pending=True
            if not self._active:self._run_cycle_boundary()
        if time.monotonic()-self._atlas_saved_at>=5:self._persist_atlas_advisory()
    def preheat_selection(self):
        ranked=sorted(self.files,key=lambda k:(-self.index.counts[k],-self.index.last.get(k,0),k[1],k[0]))
        if self.q1_root is None:
            selected=ranked[:self.payload_ceiling//ORIGINAL_BYTES]
        else:
            hot=[k for k in ranked if k in self.hot]
            cold=[k for k in ranked if k not in self.hot][:(self.policy['budget_bytes']-self.policy['hot_bytes']-ORIGINAL_BYTES)//Q1_BYTES]
            selected=hot+cold
        return selected
    def _adaptive_preheat(self):
        # Explicit warmup is bounded by observed routes, not the bank maximum.
        # Every upload uses its ordinary completion window. No allocation is
        # pinned afterward: APCv2 resource hints remain advisory, exact KV state
        # and the immutable original/Q1 assignment remain unchanged.
        if self.policy.get('preheat') is not True:
            self.preheat_receipt.update(performed=False,
                reason='adaptive demand residency; warmup requires explicit preheat=true')
            return self.preheat_receipt
        start=time.perf_counter();before=self.counters()
        selected=sorted((key for key in self.files if self.index.counts[key]>0),
            key=lambda key:(-self.index.counts[key],-self.index.last.get(key,0),key[1],key[0]))
        self.preheat_receipt.update(performed=False,scope='observed atlas experts; advisory cache warmth',
            requested_objects=len(selected),loaded_objects=0,resource_lease=False,
            unseen_fill_objects=0,precision_unchanged=True)
        if not selected:
            self.preheat_receipt.update(reason='no_observed_atlas_experts',objects=0,
                deferred_objects=0,seconds=time.perf_counter()-start)
            return self.preheat_receipt
        with self._lock:
            self._ready()
            if self._active or self._leased or self._pending_uploads or self._pending_host:
                raise ValueError('Adaptive preheat requires an idle expert provider')
        loaded=0;reason='observed_inventory_already_resident'
        split=self._split_active()
        if self._dynamic():
            # Seed the high pool with the hot map's most observed experts that
            # fit its reservation; they are served as MXFP4 once resident.
            room=self.pool_targets()['original']-self.pool_bytes['original']
            for key in selected:
                if key in self.hot and key not in self._cache and room>=ORIGINAL_BYTES:
                    self._dyn_high.add(key);room-=ORIGINAL_BYTES
        staging=self._adaptive_staging
        working=self.minimum_cache_bytes*self._adaptive_working_set
        headroom=max(9<<30,self._adaptive_headroom,self._adaptive_protected_bytes)
        try:
            for layer,expert in selected:
                key=(layer,expert)
                if key in self._cache:continue
                callback=self._adaptive_available
                try:
                    if callback is None:
                        from ..runtime.os_memory import host_memory_snapshot
                        host=host_memory_snapshot();available=None if host is None else host.available_bytes
                    else:available=callback()
                except Exception:
                    available=None
                size=Q1_BYTES if self.kind(key)=='q1' else ORIGINAL_BYTES
                if split and self.pool_bytes[self.kind(key)]+size>self.pool_targets()[self.kind(key)]:
                    continue  # this kind's reservation is full; never churn it
                if type(available)is not int or available<0:
                    reason='live_availability_unknown';break
                if available<size+staging+working+headroom:
                    reason='live_availability_or_working_headroom';break
                if self._cache_bytes+size+staging+working>self.cache_budget_bytes:
                    reason='configured_maximum_or_working_headroom';break
                with self.window():self.acquire(f'blk.{layer}',expert)
                loaded+=int(key in self._cache)
                reason='bounded_observed_preheat_complete'
        except BaseException as error:
            reason='preheat_failed'
            self.preheat_receipt['error']={'type':type(error).__name__,'message':str(error)}
            raise
        finally:
            if self._dynamic():
                # Invariant: a high key is resident as original.
                self._dyn_high&=set(self._cache)
            if hasattr(self,'_cycle_touched'):self._cycle_touched=set()
            resident=[key for key in selected if key in self._cache]
            self.preheat_receipt.update(performed=bool(resident) and reason!='preheat_failed',reason=reason,
                seconds=time.perf_counter()-start,objects=len(resident),loaded_objects=loaded,
                deferred_objects=len(selected)-len(resident),
                original_objects=sum(self.kind(key)=='original' for key in resident),
                q1_objects=sum(self.kind(key)=='q1' for key in resident),
                payload_bytes=self._cache_bytes,
                counter_delta={key:value-before.get(key,0) for key,value in self.counters().items()})
        return self.preheat_receipt
    def preheat(self):
        if not self.preheat_receipt['requested']:return self.preheat_receipt
        if self.adaptive_residency_enabled:return self._adaptive_preheat()
        start=time.perf_counter();before=self.counters();selected=self.preheat_selection()
        from ..runtime.os_memory import host_memory_snapshot
        host=host_memory_snapshot()
        needed=sum(Q1_BYTES if self.kind(k)=='q1' else ORIGINAL_BYTES for k in selected)
        if host is None or host.available_bytes < needed-self._cache_bytes+(9<<30):
            raise MemoryError('live host memory cannot preheat requested experts while preserving host/lane headroom')
        # Residency is a hint only; preheating never increments actual call counts.
        with self.window():
            for layer,expert in selected:self.acquire(f'blk.{layer}',expert)
        self.preheat_receipt.update(performed=True,seconds=time.perf_counter()-start,objects=len(selected),original_objects=sum(self.kind(k)=='original' for k in selected),q1_objects=sum(self.kind(k)=='q1' for k in selected),unseen_fill_objects=sum(self.index.counts[k]==0 for k in selected),payload_bytes=self._cache_bytes,counter_delta={k:v-before.get(k,0) for k,v in self.counters().items()})
        return self.preheat_receipt
    def park(self):
        self._drain_promotions()
        readahead=getattr(self,'_prefill_readahead',None)
        if readahead is not None:readahead.drain()
        if getattr(self,'_prefill_shield',None) is not None:self._release_prefill_shield()
        self._clear_prefill_transient()
        super().park();self.pool_bytes={'original':0,'q1':0}
        if self._dynamic():self._dyn_high=set()
        if hasattr(self,'_pre_evict_credit'):self._pre_evict_credit={'original':0,'q1':0}
        return self.receipt()
    def resize_cache_budget(self,budget):
        if self.adaptive_residency_enabled:return self.set_adaptive_cache_target(budget)['effective_cache_target_bytes']
        raise ValueError('LFU experimental profile uses fixed precision-bound budgets')
    def close(self):
        try:self._persist_atlas_advisory()
        finally:
            try:
                self._drain_promotions(timeout=5.0)
                if getattr(self,'_promotion_pool',None) is not None:self._promotion_pool.close()
                readahead=getattr(self,'_prefill_readahead',None)
                if readahead is not None:readahead.close()
            finally:
                self._clear_prefill_transient();self._prefill_shield=None
                super().close()
                self.pool_bytes={'original':0,'q1':0}
                if self._dynamic():self._dyn_high=set()
