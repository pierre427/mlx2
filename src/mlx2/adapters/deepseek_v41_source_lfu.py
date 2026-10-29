# SPDX-License-Identifier: MIT
"""Explicit per-expert LFU profiles, with immutable precision selection per run.

Only routed experts participate. See provenance/deepseek-v41-source-q1-lfu.json.
"""
from collections import Counter
import hashlib
import heapq
import json
import os
from pathlib import Path
import time

import numpy as np

from .deepseek_v41_source_artifact import checked_file, read_tensors
from .deepseek_v41_source_weights import SourceExpertStore, weight_plan, expert_names, PROJECTIONS
from .deepseek_v41_mlx_weights import _stamp, _verify_file, read_native_file
from .deepseek_v41_source_buffers import read_original_packed, borrow_q1_packed, materialize

REVISION = 'source-expert-lfu-token-frequency-v1'
Q1_REVISION = 'source-q1-affine-g64-lloyd4-f16-v1'
ORIGINAL_BYTES = 18800640
Q1_BYTES = 6635520

def parse_policy(value):
    if not isinstance(value,dict) or set(value)-{'mode','budget_bytes','hot_bytes','q1_root','hotset_path','atlas_path','preheat'}:
        raise ValueError('invalid Source expert LFU policy')
    result=dict(value)
    if 'atlas_path' in result and (not isinstance(result['atlas_path'],str) or not Path(result['atlas_path']).is_absolute()):
        raise ValueError('expert atlas path must be absolute')
    if 'preheat' in result and (type(result['preheat']) is not bool or 'atlas_path' not in result):
        raise ValueError('preheat requires a persistent atlas')
    if result.get('mode') not in ('original_lfu','hot_original_q1'):
        raise ValueError('Source expert mode must be original_lfu or hot_original_q1')
    if type(result.get('budget_bytes')) is not int or not ORIGINAL_BYTES*384 <= result['budget_bytes'] <= 90_000_000_000:
        raise ValueError('explicit Source expert total budget must fit one layer and be at most90GB')
    if result['mode']=='hot_original_q1':
        if type(result.get('hot_bytes')) is not int or not ORIGINAL_BYTES<=result['hot_bytes']<result['budget_bytes']-ORIGINAL_BYTES:
            raise ValueError('invalid original-hot/Q1 budget split')
        for key in ('q1_root','hotset_path'):
            if not isinstance(result.get(key),str) or not Path(result[key]).is_absolute():
                raise ValueError('hybrid expert policy requires absolute Q1 and hotset paths')
    elif set(result)-{'mode','budget_bytes','atlas_path','preheat'}:
        raise ValueError('original-only LFU cannot accept Q1 options')
    return result

def admission(artifact,request,policy):
    from .deepseek_v41_mlx import admission_plan
    from dataclasses import replace
    # Preserve the existing request-contract validation at a smaller bound, then
    # size the explicitly requested Source-only profile without a legacy70GiB cap.
    admission_plan(artifact,replace(request,budget_bytes=32<<30,budget_mode='explicit'),planner=weight_plan)
    if request.budget_mode!='explicit' or request.budget_bytes!=policy['budget_bytes']:
        raise ValueError('LFU execution policy and requested expert budget disagree')
    result=weight_plan(artifact,policy['budget_bytes'])
    # Leased experts remain inside the same hard ceiling; eviction skips leases.
    # One original host read is reserved inside the total expert budget too.
    result.update(expert_active_bound_bytes=policy['budget_bytes'],expert_staging_bound_bytes=0,
                  expert_cache_bytes=policy['budget_bytes'])
    result['total_reservation_bytes']=sum(result[k] for k in ('dense_payload_bytes','dense_prepared_bound_bytes','dense_staging_bound_bytes','expert_cache_bytes'))
    result.update(headroom_bytes=max(5<<30,request.headroom_bytes),metadata_reserve_bytes=256<<20,
                  ple_cache_bytes=16<<20,admission_limit_bytes=request.admission_limit_bytes,
                  ceiling_source='explicit',phase='pre_materialization')
    result['required_bytes']=result['total_reservation_bytes']+result['headroom_bytes']+result['metadata_reserve_bytes']+result['ple_cache_bytes']
    if result['required_bytes']>request.admission_limit_bytes:
        raise MemoryError('explicit Source LFU profile exceeds measured process admission')
    return result

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
    if value.get('schema')!='mlx2.expert-frequency-atlas.v1' or value.get('source')!=source or type(value.get('clock')) is not int or value['clock']<0:
        raise ValueError('expert atlas source/schema mismatch')
    seen=set()
    for row in value['counts']:
        key=(row['layer'],row['expert'])
        if key in seen or any(type(x) is not int for x in (*key,row['token_selections'],row['last_access'])) or not 0<=key[0]<40 or not 0<=key[1]<384 or row['token_selections']<0 or row['last_access']<0:
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
            self.hot_binding=None
            if policy['mode']=='hot_original_q1':
                self.q1_root,self.q1_records,self.q1_binding=load_q1_manifest(policy['q1_root'],self.binding)
                raw=Path(policy['hotset_path']).read_bytes();hot=json.loads(raw)
                if hot['source']!=self.binding or hot['original_expert_bytes']!=ORIGINAL_BYTES or hot['hot_budget_bytes']!=policy['hot_bytes']:
                    raise ValueError('hot expert map does not bind this Source/profile')
                self.hot=frozenset(tuple(k) for k in hot['experts'])
                if len(self.hot)!=len(hot['experts']) or any(k not in self.q1_records for k in self.hot) or len(self.hot)*ORIGINAL_BYTES>policy['hot_bytes']:
                    raise ValueError('hot map geometry/budget violation')
                self.hot_binding=hashlib.sha256(raw).hexdigest()
            self.atlas_path=Path(policy['atlas_path']) if policy.get('atlas_path') else None
            if self.atlas_path:
                atlas=read_atlas(self.atlas_path,self.binding);self.index.clock=atlas['clock']
                for row in atlas['counts']:
                    key=(row['layer'],row['expert']);self.index.counts[key]=row['token_selections'];self.index.last[key]=row['last_access']
            self._atlas_saved=self.index.counts.copy();self._atlas_saved_at=time.monotonic()
            self.preheat_receipt={'requested':bool(policy.get('preheat',self.atlas_path is not None)),'performed':False}
            self.plan.transient_bound_bytes=0
            self.plan.allocation='global_byte_lfu'
        except BaseException:
            self.pool.close()
            raise
    def kind(self,key):
        return 'q1' if self.q1_root is not None and key not in self.hot else 'original'
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
        for expert,count in counts:
            self.index.counts[(layer,expert)]+=count
            self.stats[self.kind((layer,expert))+'_token_selections']+=count
    def _evict_for(self,key,size):
        kind=self.kind(key)
        limit=self.payload_ceiling if self.q1_root is None else (self.policy['hot_bytes'] if kind=='original' else self.policy['budget_bytes']-self.policy['hot_bytes']-ORIGINAL_BYTES)
        while self.pool_bytes[kind]+size>limit or self._cache_bytes+size>self.payload_ceiling:
            victim=self.index.victim(self._cache,self._leased,lambda k:self.kind(k)==kind and (self.q1_root is None or kind=='q1'))
            arrays=self._cache.pop(victim);amount=sum(v.nbytes for v in arrays.values());self._cache_bytes-=amount;self.pool_bytes[kind]-=amount;self.stats['evictions']+=1
    def acquire(self,prefix,expert):
        import mlx.core as mx
        with self._lock:
            self._ready()
            if not self._active or prefix not in self.tables or type(expert) is not int or not 0<=expert<384:
                raise ValueError('LFU expert acquisition outside ownership')
            key=self.tables[prefix][expert];kind=self.kind(key)
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
            if key in self._cache:
                arrays=self._cache[key];self.stats['cache_hits']+=1;self.stats[kind+'_hits']+=1
            else:
                size=Q1_BYTES if kind=='q1' else ORIGINAL_BYTES;self._evict_for(key,size)
                start=time.perf_counter_ns()
                try:
                    if getattr(self,'packed_buffers',False):
                        raw=(borrow_q1_packed(self.pool.call(q1_read,self.q1_root,self.q1_records[key]))
                             if kind=='q1' else self.pool.call(read_original_packed,self.artifact,*key))
                    else:
                        raw=self.pool.call(q1_read,self.q1_root,self.q1_records[key]) if kind=='q1' else self.pool.call(read_tensors,self.artifact,expert_names(*key))
                finally:self.stats['demand_wait_ns']+=time.perf_counter_ns()-start
                self._pending_host[key]=raw;arrays=self._pending_uploads[key]={}
                if getattr(self,'packed_buffers',False):
                    if raw.host_allocation_nbytes>ORIGINAL_BYTES:raise ValueError('packed expert host allocation exceeds staging reserve')
                    self.stats['packed_peak_retained_host_bytes']=max(self.stats['packed_peak_retained_host_bytes'],raw.host_allocation_nbytes)
                    arrays=self._pending_uploads[key]=materialize(raw)
                    # Mapping subclasses are not guaranteed evaluator pytrees.
                    # The pending host and full backing survive eval failures.
                    mx.eval(arrays.backing,*arrays.values())
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
                self._pending_host.pop(key);del raw
                self._cache[key]=arrays;self._cache_bytes+=size;self.pool_bytes[kind]+=size
                self.peak_payload_bytes=max(self.peak_payload_bytes,self._cache_bytes)
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
    def apc_revision(self):
        return REVISION+':'+str(self.q1_binding)+':'+str(self.hot_binding)
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
    def live_bytes(self):
        if not getattr(self,'packed_buffers',False):return super().live_bytes()
        return self._packed_backing_totals()[1]+sum(len(t._cache)*t.row_bytes for t in self.ple_tables.values())
    resident_bytes=live_bytes
    def receipt(self):
        result=super().receipt()
        result.update(allocation='LFU by token-expert selections, LRU ties',policy=self.policy,quantization=Q1_REVISION if self.q1_root else 'original',q1_bank_binding=self.q1_binding,hot_map_binding=self.hot_binding,hot_expert_objects=len(self.hot),pool_payload_bytes=self.pool_bytes.copy(),peak_expert_payload_bytes=self.peak_payload_bytes,expert_total_payload_and_host_staging_ceiling=self.cache_budget_bytes,host_staging_reserve_bytes=ORIGINAL_BYTES,precision_selection='fixed per expert for profile lifetime',expert_requantization=self.q1_root is not None)
        result['plan']['allocation']='global_byte_lfu'
        result['numeric_profile']='source_hot_original_cold_q1_g64' if self.q1_root else 'official_mxfp4_fp8'
        result.update(atlas_path=str(self.atlas_path) if self.atlas_path else None,atlas_known_experts=len(self.index.counts),preheat=self.preheat_receipt)
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
            with temporary.open('x') as out:json.dump(value,out);out.write('\n');out.flush();os.fsync(out.fileno())
            os.replace(temporary,path);self._atlas_saved=self.index.counts.copy();self._atlas_saved_at=time.monotonic()
    def end_context(self):
        if time.monotonic()-self._atlas_saved_at>=5:self.persist_atlas()
    def preheat_selection(self):
        ranked=sorted(self.files,key=lambda k:(-self.index.counts[k],-self.index.last.get(k,0),k[1],k[0]))
        if self.q1_root is None:
            selected=ranked[:self.payload_ceiling//ORIGINAL_BYTES]
        else:
            hot=[k for k in ranked if k in self.hot]
            cold=[k for k in ranked if k not in self.hot][:(self.policy['budget_bytes']-self.policy['hot_bytes']-ORIGINAL_BYTES)//Q1_BYTES]
            selected=hot+cold
        return selected
    def preheat(self):
        if not self.preheat_receipt['requested']:return self.preheat_receipt
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
        super().park();self.pool_bytes={'original':0,'q1':0};return self.receipt()
    def resize_cache_budget(self,budget):
        raise ValueError('LFU experimental profile uses fixed precision-bound budgets')
    def close(self):
        try:self.persist_atlas()
        finally:
            super().close()
            self.pool_bytes={'original':0,'q1':0}
