# SPDX-License-Identifier: MIT
"""Bounded lossless bank preparation for a large already-resident PP set.

Preparation is an exclusive, owned idle operation. No forward or cache eviction
may interleave with it. Publication occurs only after the final LUT completes.
A completed failure drops only the retired cohorts, leaving ordinary cache misses
reloadable. Failed completion retains every allocation and poisons the provider.
Provenance: deepseek-v41-source-resident-prefill.json.
"""
from collections import OrderedDict
import numpy as np

from .deepseek_v41_source_bank import (
    BANK_ANCHOR_WORDS, KIND_CODES, LUT_SHAPE, PublishedSourceBank,
    _QUARANTINE, _alias, _binding, _idle, _live, plan_bank,
)
from .deepseek_v41_source_buffers import PackedExpert


def _one(mx, store, slices, size, pending):
    originals={s.key:store._cache[s.key]for s in slices}
    pending.append(originals)
    dtype={'<u4':mx.uint32,'|u1':mx.uint8,'<f2':mx.float16}
    mx.eval(*(v for e in originals.values()for v in (e.backing,*e.values())))
    for s in slices:
        e=originals[s.key]
        if e.backing.dtype!=mx.uint8 or set(e)!={t.name for t in s.layout.tensors}:
            raise ValueError('PP bank input backing/inventory mismatch')
        for t in s.layout.tensors:
            v=e[t.name]
            if tuple(v.shape)!=t.shape or v.dtype!=dtype[t.dtype]:
                raise ValueError('PP bank typed input geometry mismatch')
            _alias(e.backing,v,byte_offset=t.offset,byte_length=t.nbytes)
    inputs=[originals[s.key].backing.view(mx.uint32)for s in slices];pending.extend(inputs)
    full=mx.concatenate(inputs,axis=0);pending.append(full)
    if full.shape!=(size//4,)or full.dtype!=mx.uint32:raise ValueError('PP bank geometry mismatch')
    anchor=full[:BANK_ANCHOR_WORDS];pending.append(anchor);rebound={}
    for s in slices:
        layout=s.layout
        # Never form a >2GiB byte dimension: slice the small expert in words.
        backing=full[s.offset_words:s.offset_words+layout.nbytes//4].view(mx.uint8)
        tensors={t.name:backing[t.offset:t.offset+t.nbytes].view(dtype[t.dtype]).reshape(t.shape)
                 for t in layout.tensors}
        e=PackedExpert(backing,layout,tensors);rebound[s.key]=e
        pending.extend((backing,*tensors.values()))
    mx.eval(full,anchor,*(v for e in rebound.values()for v in (e.backing,*e.values())))
    _alias(full,anchor,byte_length=4*BANK_ANCHOR_WORDS)
    for s in slices:
        e=rebound[s.key];base=s.offset_words*4
        _alias(full,e.backing,byte_offset=base,byte_length=s.layout.nbytes)
        for t in s.layout.tensors:_alias(full,e[t.name],byte_offset=base+t.offset,byte_length=t.nbytes)
    return full,anchor,rebound


def build_prefill_bank(mx,store,keys,*,ownercheck,generation=1,
                       budget_bytes=75_000_000_000,max_bank_bytes=3<<30,
                       staging_bytes=18_800_640):
    """Retire one completed old cohort at a time, never duplicate the PP set.

    The explicit worst-case envelope is old resident cache + two largest bank
    allocations (new bank and concatenation scratch) + metadata + staging.
    An inadmissible explicit candidate refuses; normal model loading/reference
    streaming is unaffected and must remain available to its caller.
    """
    _live(ownercheck)
    with store._lock:
        _idle(store)
        if getattr(store,'_resident_bank',None)is not None or getattr(store,'_resident_bank_preparing',None):
            raise ValueError('PP bank preparation cannot replace a live bank/preparation')
        if generation<=getattr(store,'_resident_bank_generation',0):raise ValueError('PP bank generation must increase')
        store._fence()
        plan=plan_bank(store,keys,generation=generation,budget_bytes=budget_bytes,
            max_bank_bytes=max_bank_bytes,staging_bytes=staging_bytes,construction='incremental')
        banks=[];anchors=[];rebound={};pending=[]
        store._resident_bank_preparing=dict(generation=generation,keys=plan.keys,construction='incremental')
        original_pins=getattr(store,'_resident_bank_pinned_keys',frozenset())
        try:
            for index,size in enumerate(plan.bank_nbytes):
                _live(ownercheck)
                if _binding(store)!=plan.binding:raise ValueError('PP bank source/profile changed')
                rows=tuple(s for s in plan.slices if s.bank_index==index)
                full,anchor,local=_one(mx,store,rows,size,pending)
                # Every alias and its completion is proved before retirement.
                # No other evaluation owner can observe this private stage.
                for key,value in local.items():store._cache[key]=value
                rebound.update(local);banks.append(full);anchors.append(anchor)
                store._resident_bank_pinned_keys=original_pins|frozenset(rebound)
                pending.clear()  # old inputs have completed; cache now owns views
            lut_cpu=np.zeros(LUT_SHAPE,np.uint32);pending.append(lut_cpu)
            for s in plan.slices:lut_cpu[s.key]=(KIND_CODES[s.layout.kind],s.bank_index,s.offset_words,generation)
            lut=mx.array(lut_cpu);pending.append(lut);mx.eval(lut)
            if lut.shape!=LUT_SHAPE or lut.dtype!=mx.uint32:raise ValueError('PP bank LUT geometry mismatch')
            _live(ownercheck)
            if _binding(store)!=plan.binding or any(store._cache.get(k)is not v for k,v in rebound.items()):
                raise ValueError('PP bank staged cache ownership/profile changed')
            published=PublishedSourceBank(store,plan,(tuple(banks),tuple(anchors),lut),rebound,ownercheck)
            store._resident_bank_generation=generation;store._resident_bank_pinned_keys=plan.keys
            store._resident_bank=published;store._resident_bank_preparing=None
            return published
        except BaseException:
            try:store._fence()
            except BaseException as error:
                store._error='PP bank preparation completion failed: '+str(error)
                _QUARANTINE.append((store,tuple(banks),tuple(anchors),rebound,tuple(pending)))
                # Keep marker/pins: no forward or teardown can claim it retired.
                raise
            # Completed retired cohorts are evicted as WHOLE allocations.
            # Their checkpoint data remains reloadable by the ordinary provider.
            for key,value in rebound.items():
                if store._cache.get(key)is not value:continue
                del store._cache[key];store._cache_bytes-=value.allocation_nbytes
                if hasattr(store,'pool_bytes'):store.pool_bytes[value.layout.kind]-=value.allocation_nbytes
            store._resident_bank_pinned_keys=original_pins
            store._resident_bank_preparing=None
            raise
