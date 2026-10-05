"""Direct NumPy source contracts; no MLX/native runtime imports or GPU."""
from contextlib import nullcontext
from collections import deque, namedtuple
import ast
import importlib.abc
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

if __name__ != '__main__': raise unittest.SkipTest('run directly: source contracts block runtime imports')
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'mlx' or name.startswith('mlx.') or name == '_paged_kv_native':
            raise RuntimeError('MLX/native runtime import prohibited')
sys.meta_path.insert(0, NoRuntime())
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('hybrid_candidate',ROOT/'src/mlx2/adapters/qwen35_paged_candidate.py')
M=importlib.util.module_from_spec(spec); sys.modules[spec.name]=M; spec.loader.exec_module(M)

class Projection:
    """Quantized module double; calling it is supported, casting weights is not."""
    def __init__(self,width): self.width=width; self.calls=0; self.weight=NS(dtype='uint32')
    def __call__(self,x):
        self.calls+=1
        seed=x.astype(np.float32).mean(axis=-1,keepdims=True)
        return (np.sin(seed*.1+np.arange(self.width,dtype=np.float32)*.007)*.12).astype(np.float16)
    def astype(self,*args): raise AssertionError('quantized module cast')

def norm(x): return (x.astype(np.float32)/np.sqrt(np.mean(x.astype(np.float32)**2,axis=-1,keepdims=True)+1e-4)).astype(x.dtype)
def rope(x,offset):
    out=x.copy(); theta=(np.arange(32,dtype=np.float32)+1)*offset*.0003
    a=x[...,:32].astype(np.float32); b=x[...,32:64].astype(np.float32)
    out[...,:32]=(a*np.cos(theta)-b*np.sin(theta)).astype(x.dtype)
    out[...,32:64]=(a*np.sin(theta)+b*np.cos(theta)).astype(x.dtype)
    return out

def sdpa(q,k,v):
    outs=[]
    for head in range(q.shape[0]):
        kv_head=head//(q.shape[0]//k.shape[1])
        score=(k[:,kv_head].astype(np.float32)@q[head].astype(np.float32))/math.sqrt(256)
        weight=np.exp(score-score.max()); weight/=weight.sum()
        outs.append(weight@v[:,kv_head].astype(np.float32))
    return np.asarray(outs,dtype=np.float16)

class Attention:
    num_attention_heads=16; num_key_value_heads=4; head_dim=256; scale=256**-.5
    def __init__(self):
        self.q_proj=Projection(16*512); self.k_proj=Projection(4*256); self.v_proj=Projection(4*256)
        self.q_norm=self.k_norm=norm; self.rope=rope
        self.o_proj=lambda x: np.repeat(x.astype(np.float32).mean(axis=-1,keepdims=True),8,axis=-1).astype(x.dtype)*np.float16(.02)
    def reference(self,x,owner,offset):
        # Ordinary single-row attention specification, separate per-head Q/gate extraction.
        qg=self.q_proj(x).reshape(16,512); q=norm(qg[:,:256]); gate=qg[:,256:].reshape(1,1,-1)
        k=norm(self.k_proj(x).reshape(4,256)); v=self.v_proj(x).reshape(4,256)
        q=rope(q[None,:,None,:],offset)[0,:,0,:]; k=rope(k[None,:,None,:],offset)[0,:,0,:]
        k_all=np.concatenate((owner.keys,k[None]),axis=0); v_all=np.concatenate((owner.values,v[None]),axis=0)
        out=sdpa(q,k_all,v_all).reshape(1,1,-1)
        return self.o_proj(out/(1+np.exp(-gate.astype(np.float32))).astype(np.float16))

class Cache:
    def __init__(self,seed=0):
        self.cache=[np.full((1,3,2),seed,np.float16),np.full((1,1,2,2),seed,np.float32)]
        self.speculating=False; self.left_padding=self.lengths=None; self._rollbacks=deque(); self._checkpoints=[]
    def __getitem__(self,i): return self.cache[i]
    def __setitem__(self,i,v): self.cache[i]=v
class View:
    def __init__(self,rows):
        self.rows=rows; self.speculating=True; self.lengths=self.left_padding=None
        self.cache=[np.concatenate([c.cache[i] for c in rows]) for i in (0,1)]
    def prepare(self,lengths):
        assert lengths in ([1],[1,1]); self.lengths=np.asarray(lengths)
    def finalize(self): self.lengths=self.left_padding=None
    def __getitem__(self,i): return self.cache[i]
    def __setitem__(self,i,v):
        self.cache[i]=v
        for r,c in enumerate(self.rows): c[i]=v[r:r+1].copy()
class JoinedView(View):
    def __init__(self,rows,joined):
        self.rows=rows; self.speculating=True; self.lengths=self.left_padding=None
        self.cache=list(joined)
class GDN:
    fused_gdn_enabled=True
    def __call__(self,x,mask,cache):
        assert not cache.speculating and self.fused_gdn_enabled
        delta=x.astype(np.float32).mean(axis=(1,2))
        cache[0]=(cache[0]+delta[:,None,None]*.01).astype(np.float16)
        cache[1]=cache[1]+delta[:,None,None,None]*.01
        return (x.astype(np.float32)*.002+cache[1].mean(axis=(1,2,3))[:,None,None]*.001).astype(x.dtype)
class Layer:
    def __init__(self,index):
        self.is_linear=(index+1)%4!=0; self.input_layernorm=self.post_attention_layernorm=norm
        if self.is_linear: self.linear_attn=GDN()
        else: self.self_attn=Attention()
        self.mlp=lambda x:(x*.001).astype(x.dtype)
class KV:
    def __init__(self): self.offset=0; self.keys=self.values=None
    def keys_and_values(self): return self.keys[:,:,:self.offset],self.values[:,:,:self.offset]
class Trunk:
    pipeline_size=1; pipeline_rank=0; eager_dispatch_stride=4; eager_dispatch_max_rows=64
    def __init__(self):
        self._test_config=NS(num_hidden_layers=32,hidden_size=2560,num_attention_heads=16,num_key_value_heads=4,
                     linear_num_value_heads=32,num_experts=0,full_attention_interval=4,head_dim=256,
                     linear_num_key_heads=16,linear_key_head_dim=128,linear_value_head_dim=128,
                     linear_conv_kernel_dim=4,vocab_size=1000)
        self.layers=[Layer(i) for i in range(32)]; self.norm=norm
    def embed_tokens(self,tokens): return (np.asarray(tokens)[...,None]*.001+np.arange(8)*.003).astype(np.float16)
    def __call__(self,tokens,cache):
        # Prefill double deliberately allocates 256-token KV buffers, including nonlogical padding.
        length=tokens.shape[1]
        for layer,c in zip(self.layers,cache):
            if layer.is_linear:
                c.cache=[np.full((1,3,2),.2,np.float16),np.full((1,1,2,2),.3,np.float32)]
            else:
                c.keys=np.full((1,4,256,256),99,np.float16); c.values=c.keys.copy(); c.offset=length
                c.keys[:,:,:length]=.2; c.values[:,:,:length]=.3
        return self.embed_tokens(tokens)
class Language:
    def __init__(self): self.model=Trunk(); self.args=self.model._test_config; self.compute_dtype="float16"
    def logits(self,hidden): return hidden
    def make_cache(self): return [Cache() if l.is_linear else KV() for l in self.model.layers]
class Use:
    def __init__(self,owners,epoch): self.owners=owners; self.state='prepared'; self.lease=NS(epoch=epoch)
    def abort_before_submit(self): self.state='closed'
class Branch:
    def __init__(self,layers,caches,events,*,lane_id=0,generation=0,revision='test-revision',owner=None):
        self.layers=layers; self.recurrent_caches=caches; self._closed=False; self._owner=owner or object()
        checkpoint=NS(revision=revision,lane_id=lane_id,generation=generation,offset=layers[0].offset)
        self._request=NS(proposed_rows=1,planes=('kv','gdn'),revision=revision,lane_id=lane_id)
        self._origin=NS(layers=(NS(offset=layers[0].offset),),revision=revision,generation=generation,
                        companions=(('gdn',(checkpoint,)),))
        self.events=events; self.staged=False
    def prove_staged_layer_read(self,index,lane,proof):
        assert proof.state=='closed'; self.events.append(('proof',index,lane))
    def stage_recurrent_boundary(self,caches,offset):
        assert caches is self.recurrent_caches; assert offset==self._origin.layers[0].offset+1
        self.events.append(('stage',offset)); self.staged=True
class Backend:
    profiling_enabled=False
    def __init__(self,events):
        self.events=events; self.tile=self.grouped=self.partial=self.reduce=self.scalar=self.stock=self.long_partial=self.long_reduce=self.long_inline=0; self.cow=False; self.bad_scalar=False; self.direct_grouped_fence_reads=0; self.stripes={8:0,16:0,32:0}; self._failed=False; self._orphaned_reads={}; self.fail_read=False; self.bad_tile=False
        arena=NS(stream=object(),q1_tile_dispatch_count=lambda:self.tile,
                 q1_stripe_dispatch_count=lambda s:self.stripes.get(s,0),grouped_q1_write_count=lambda:self.grouped,
                 write_dispatch_count=lambda:self.scalar,q1_stock_reduction_dispatch_count=lambda:self.stock,
                 q1_stock_long_partial_dispatch_count=lambda:self.long_partial,
                 q1_stock_long_reduce_dispatch_count=lambda:self.long_reduce,
                 q1_stock_long_metadata_dispatch_count=lambda:self.long_inline,
                 _native=NS(q1_stock_long_metadata_dispatch_count=lambda _:self.long_inline),
                 q1_split_partial_dispatch_count=lambda:self.partial,
                 q1_split_reduce_dispatch_count=lambda:self.reduce)
        self.deferred_counts=dict(grouped_write_async_evals=0,staged_read_async_evals=0,
            deferred_q1_write_roots=0,deferred_q1_read_roots=0,
            deferred_q1_final_evals=0,deferred_q1_failure_flushes=0)
        self.deferred_roots=[];self.abort_calls=0;self.retain_ambiguous=False
        arena.defer_staged_q1_eval=False;arena.defer_staged_q1_writes=False
        def begin_deferred_q1(*,writes_only=False):
            assert not arena.defer_staged_q1_eval and not self.deferred_roots
            arena.defer_staged_q1_eval=not writes_only;arena.defer_staged_q1_writes=True
        def end_deferred_q1(*,retain_roots=False):
            arena.defer_staged_q1_eval=False;arena.defer_staged_q1_writes=False
            if not retain_roots:self.deferred_roots.clear()
        arena.begin_deferred_q1=begin_deferred_q1;arena.end_deferred_q1=end_deferred_q1
        arena.deferred_q1_roots=lambda:tuple(self.deferred_roots)
        arena.eval_submission_snapshot=lambda:{name:getattr(arena,name) for name in self.deferred_counts}
        for name in self.deferred_counts:
            setattr(arena,name,0)
        self.writer=NS(backend=arena,poisoned=False,pending_epochs={},ledger=NS(pending_count=0),failed_arena_torn_down=False)
    def abort_deferred_q1(self,owners,uses):
        self.abort_calls+=1;self.writer.backend.deferred_q1_failure_flushes+=1
        for use in uses:use.state='closed'
        self._failed=True;self.writer.poisoned=True
    def retain_deferred_q1_roots(self,uses):
        return self.retain_ambiguous or any(use.state in ('prepared','submitted') for use in uses)
    def append_staged(self,owners,keys,values,counts):
        assert counts in ((1,),(1,1))
        if len(owners)==2:
            self.grouped+=1
            if self.writer.backend.defer_staged_q1_writes:
                self.writer.backend.deferred_q1_write_roots+=1
                self.deferred_roots.append(('write',self.grouped))
        else: self.scalar+=len(owners[0].planned_spans(1,staged=True))-(1 if self.bad_scalar else 0)
        for r,o in enumerate(owners):
            o.keys=np.concatenate((o.keys,keys[r:r+1]),axis=0); o.values=np.concatenate((o.values,values[r:r+1]),axis=0); o.offset+=1
        return tuple(NS(dependency=object(),grouped_q1=len(owners)==2) for _ in range((1+int(self.cow)) if len(owners)==2 else 4+int(self.cow)))
    def read_staged(self,use,queries,tickets,scale):
        use.state='submitted'
        if self.writer.backend.defer_staged_q1_eval or self.writer.backend.defer_staged_q1_writes:
            self.writer.backend.deferred_q1_read_roots+=1
            self.deferred_roots.append(('read',use.lease.epoch))
            if not self.writer.backend.defer_staged_q1_eval:
                self.writer.backend.staged_read_async_evals+=1
        if getattr(self,'direct_grouped_fence',False) and len(tickets)==1 and tickets[0].grouped_q1:
            self.direct_grouped_fence_reads+=1
        if self.fail_read: raise RuntimeError('ambiguous native read')
        if len(use.owners)==2 and max(o.offset for o in use.owners)>1024 and os.environ.get('MLX2_PAGED_Q1_STOCK_LONG')=='1':
            self.long_partial+=1;self.long_reduce+=1
            if os.environ.get('MLX2_PAGED_Q1_STOCK_LONG_INLINE_METADATA')=='1': self.long_inline+=1
        elif len(use.owners)==2 and max(o.offset for o in use.owners)>128: self.partial+=1; self.reduce+=1
        elif max(o.offset for o in use.owners)<=128:
            self.tile+=0 if self.bad_tile else 1
            stock=len(use.owners)==2 and os.environ.get("MLX2_PAGED_Q1_STOCK_REDUCTION")=="1"
            if stock:self.stock+=1
            self.stripes[32 if stock else int(os.environ['MLX2_PAGED_Q1_SIMD_STRIPES'])] += 0 if self.bad_tile else 1
        return np.asarray([sdpa(q,o.keys,o.values) for q,o in zip(queries,use.owners)])
    def drain_staged(self,owners,uses):
        self.events.append('drain')
        for u in uses: u.state='closed'
        return uses
    def append_completed(self,owners,keys,values,counts):
        o=owners[0]; self.events.append(('import',counts[0])); assert keys.shape[0]==counts[0]
        assert np.all(keys==np.float16(.2))  # No allocation-tail sentinel imported.
        o.keys=keys.copy(); o.values=values.copy(); o.offset+=counts[0]

class Contracts(unittest.TestCase):
    def test_direct_grouped_fence_selection_and_physical_count(self):
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_direct_grouped_fence=True,_runtime_factory=lambda:self.runtime)
        _,branches,lanes=self.lanes()
        _,receipt=self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertTrue(receipt['q1_direct_grouped_fence_selected'])
        self.assertEqual(receipt['direct_grouped_fence_reads'],8)
        self.assertEqual(receipt['grouped_q1_writes'],8)
    def test_direct_grouped_fence_singleton_and_extra_copy_dependency_fallback(self):
        for singleton in (False,True):
            self.setUp();self.backend.cow=True
            self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_direct_grouped_fence=True,_runtime_factory=lambda:self.runtime)
            _,branches,lanes=self.lanes()
            if singleton:lanes,branches=lanes[:1],branches[:1]
            _,receipt=self.candidate.forward_staged(lanes,branches,permit_candidate=True)
            self.assertEqual(receipt['direct_grouped_fence_reads'],0)
    def test_direct_grouped_fence_requires_boolean_and_stable_backend_selection(self):
        with self.assertRaisesRegex(TypeError,'must be boolean'):
            M.Qwen35PagedCandidate(self.lang,self.backend,q1_direct_grouped_fence=1)
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_direct_grouped_fence=True,_runtime_factory=lambda:self.runtime)
        self.backend.direct_grouped_fence=False
        _,branches,lanes=self.lanes()
        with self.assertRaisesRegex(ValueError,'selection differs'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertEqual(self.backend.grouped,0)
    def test_missing_direct_fence_physical_count_blocks_checkpoint_staging(self):
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_direct_grouped_fence=True,_runtime_factory=lambda:self.runtime)
        original=self.backend.read_staged
        def missing_counter(*args,**kwargs):
            result=original(*args,**kwargs);self.backend.direct_grouped_fence_reads=0;return result
        self.backend.read_staged=missing_counter
        _,branches,lanes=self.lanes()
        with self.assertRaisesRegex(RuntimeError,'fence read proof differs'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertFalse(any(isinstance(event,tuple) and event[0]=='stage' for event in self.events))
        self.assertTrue(self.candidate._failure_roots)
    def setUp(self):
        self.events=[]; self.lang=Language(); self.backend=Backend(self.events); self.fail_eval=False
        def eval_(*roots):
            self.events.append(('eval',len(roots)))
            if self.fail_eval: raise RuntimeError('ambiguous eval')
        self.mx=NS(array=np.asarray,float16=np.dtype(np.float16),float32=np.dtype(np.float32),bfloat16=np.dtype(np.float32),all=np.all,isfinite=np.isfinite,split=np.split,concatenate=np.concatenate,
                   device_info=lambda:{'architecture':'applegpu_g16s'},stream=lambda s:nullcontext(),eval=eval_,async_eval=lambda x:self.events.append('eager'))
        epoch=[0]
        def prepare(owners,counts,**kwargs): epoch[0]+=1; return Use(owners,epoch[0])
        self.runtime=NS(mx=self.mx,gate_sigmoid=lambda x:(1/(1+np.exp(-x.astype(np.float32)))).astype(x.dtype),batch_cache=View,joined_batch_cache=JoinedView,array_cache_type=Cache,prepare_read=prepare)
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,_runtime_factory=lambda:self.runtime)
        self.env=patch.dict(os.environ,{'MLX2_PAGED_Q1_SIMD_TILE':'1','MLX2_PAGED_Q1_SIMD_STRIPES':'16','MLX2_PAGED_GROUPED_Q1_WRITE':'1','MLX2_PAGED_Q1_STOCK_SDPA':'0','MLX2_PAGED_Q1_SPLIT_KV':'0','MLX2_PAGED_Q1_STOCK_REDUCTION':'0','MLX2_PAGED_Q1_STOCK_LONG':'0','MLX_SDPA_BLOCKS':'0'})
        self.env.start(); self.addCleanup(self.env.stop)
    def test_real_joined_view_factory_skips_base_refresh_concatenations(self):
        calls=[]
        class RefreshingBase:
            def __init__(self,rows):
                self.rows=rows;self.cache=[None,None];self._refresh_state()
            def _refresh_state(self):
                self.cache=[np.concatenate([row[slot] for row in self.rows],axis=0)
                            for slot in (0,1)]
                calls.extend((0,1))
        rows=((np.ones((1,3,2)),np.ones((1,1,2,2))),)*2
        expected=(np.concatenate((rows[0][0],rows[1][0])),
                  np.concatenate((rows[0][1],rows[1][1])))
        joined=M._joined_view_factory(RefreshingBase)(rows,expected)
        self.assertEqual(calls,[])
        self.assertIs(joined.cache[0],expected[0]);self.assertIs(joined.cache[1],expected[1])
        RefreshingBase(rows)
        self.assertEqual(calls,[0,1])
    def owner(self,offset):
        rng=np.random.default_rng(offset)
        return NS(writer=self.backend.writer,offset=offset,profile=NS(dtype='float16',head_dim=256,kv_heads=4),
                  keys=rng.normal(0,.1,(offset,4,256)).astype(np.float16),
                  values=rng.normal(0,.1,(offset,4,256)).astype(np.float16),_pending=[],
                  planned_spans=lambda count,staged:tuple(range(4)))
    def lanes(self):
        originals=tuple(tuple(Cache(.1+r*.03) for _ in range(24)) for r in range(2))
        branches=tuple(Branch(tuple(self.owner(offset) for _ in range(8)),M.clone_recurrent_caches(originals[r]),self.events,lane_id=r) for r,offset in enumerate((32,96)))
        lanes=tuple(M.HybridPackedLane((13+r,),b.layers,b.recurrent_caches) for r,b in enumerate(branches))
        return originals,branches,lanes
    def test_full_hybrid_math_matches_independent_single_lane_reference(self):
        originals,branches,lanes=self.lanes(); expected=[]; expected_caches=[]
        for r,lane in enumerate(lanes):
            hidden=self.lang.model.embed_tokens(np.asarray([[lane.token_ids[0]]]))
            caches=M.clone_recurrent_caches(originals[r]); fa=gdn=0
            for layer in self.lang.model.layers:
                mixed=norm(hidden)
                if layer.is_linear: attended=layer.linear_attn(mixed,None,caches[gdn]); gdn+=1
                else: attended=layer.self_attn.reference(mixed,lane.layers[fa],branches[r]._origin.layers[0].offset); fa+=1
                hidden=hidden+attended; hidden=hidden+layer.mlp(norm(hidden))
            expected.append(norm(hidden)); expected_caches.append(caches)
        logits,receipt=self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        np.testing.assert_allclose(logits,np.concatenate(expected)[:,0,:],rtol=.002,atol=.002)
        for r,lane in enumerate(lanes):
            for actual,reference,original in zip(lane.recurrent_caches,expected_caches[r],originals[r]):
                np.testing.assert_allclose(actual[1],reference[1],rtol=.002,atol=.002)
                self.assertTrue(np.all(original[1]==np.float32(.1+r*.03)))
        self.assertEqual(receipt['native_tile_dispatches'],8); self.assertEqual(receipt['grouped_q1_writes'],8)
        eval_index=next(i for i,e in enumerate(self.events) if isinstance(e,tuple) and e[0]=='eval')
        self.assertEqual(self.events[eval_index],('eval',97))
        self.assertLess(eval_index,self.events.index('drain')); self.assertTrue(all(b.staged for b in branches))
        self.assertEqual(self.events.count('eager'),8)
    def test_all_valid_recurrent_q1_skips_only_metadata_and_preserves_math(self):
        counts={'prepare':0,'finalize':0}
        class TrackedView(View):
            def prepare(self,lengths):
                counts['prepare']+=1; super().prepare(lengths)
            def finalize(self):
                counts['finalize']+=1; super().finalize()
        self.runtime.batch_cache=TrackedView
        _,branches,lanes=self.lanes()
        baseline,base_receipt=self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        baseline_states=tuple(tuple(slot.copy() for cache in lane.recurrent_caches for slot in cache.cache) for lane in lanes)
        self.assertEqual(counts,{'prepare':24,'finalize':24})
        self.assertFalse(base_receipt['q1_all_valid_recurrent_selected'])
        self.assertEqual(base_receipt['q1_all_valid_recurrent_layers'],0)
        self.events=[]; self.backend=Backend(self.events)
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_all_valid_recurrent=True,_runtime_factory=lambda:self.runtime)
        _,branches,lanes=self.lanes()
        actual,receipt=self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        np.testing.assert_array_equal(actual,baseline)
        for lane,reference in zip(lanes,baseline_states):
            for actual_slot,reference_slot in zip((slot for cache in lane.recurrent_caches for slot in cache.cache),reference):
                np.testing.assert_array_equal(actual_slot,reference_slot)
        self.assertEqual(counts,{'prepare':24,'finalize':24})
        self.assertTrue(receipt['q1_all_valid_recurrent_selected'])
        self.assertEqual(receipt['q1_all_valid_recurrent_layers'],24)
        self.assertEqual(receipt['native_tile_dispatches'],8)
        self.assertEqual(receipt['grouped_q1_writes'],8)
        self.assertTrue(all(branch.staged for branch in branches))
    def test_all_valid_recurrent_requires_plain_initialized_private_rows(self):
        with self.assertRaisesRegex(TypeError,'boolean'):
            M.Qwen35PagedCandidate(self.lang,self.backend,q1_all_valid_recurrent=1)
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_all_valid_recurrent=True,_runtime_factory=lambda:self.runtime)
        _,branches,lanes=self.lanes()
        self.runtime.array_cache_type=None
        with self.assertRaisesRegex(ValueError,'ArraysCache type'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.runtime.array_cache_type=Cache
        class ForeignCache(Cache): pass
        # Rebuild the immutable branch tuple with a subclass row; it must fail before writes.
        first=(ForeignCache(),)+branches[0].recurrent_caches[1:]
        replacement=Branch(branches[0].layers,first,self.events)
        lane=M.HybridPackedLane(lanes[0].token_ids,replacement.layers,replacement.recurrent_caches)
        with self.assertRaisesRegex(ValueError,'private initialized B1 recurrent cache'):
            self.candidate.forward_staged((lane,lanes[1]),(replacement,branches[1]),permit_candidate=True)
        self.assertEqual(self.backend.grouped,0)
    def _joined_setup(self):
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_all_valid_recurrent=True,
            q1_joined_recurrent=True,_runtime_factory=lambda:self.runtime)
        charges=[]
        def reserve(size):
            charge=NS(bytes=size,roots=(),released=False)
            charge.release=lambda:setattr(charge,'released',True)
            charge.retain_failure_roots=lambda roots:setattr(charge,'roots',roots)
            charges.append(charge);return charge
        _,branches,lanes=self.lanes()
        _,receipt=self.candidate.forward_staged(lanes,branches,permit_candidate=True,reserve_scratch=reserve)
        self.assertEqual(receipt['q1_joined_reused_slots'],0)
        self.assertEqual(receipt['q1_joined_fallback_materializations'],48)
        self.assertEqual(len(charges),1)
        self.assertEqual(len(self.candidate._joined_bindings),24)
        return branches,charges,reserve
    def _next_joined_lanes(self,prior,*,generation=1,reverse=False,copy_leaf=False,new_owner=False,revision='test-revision'):
        order=(1,0) if reverse else (0,1)
        branches=[]
        for lane_id in order:
            caches=M.clone_recurrent_caches(prior[lane_id].recurrent_caches)
            if copy_leaf and lane_id==0:
                caches[0].cache[0]=caches[0].cache[0].copy()
            offset=(32,96)[lane_id]+generation
            branches.append(Branch(tuple(self.owner(offset) for _ in range(8)),caches,self.events,
                                   lane_id=lane_id,generation=generation,revision=revision,
                                   owner=(None if new_owner and lane_id==0 else prior[lane_id]._owner)))
        branches=tuple(branches)
        lanes=tuple(self.candidate.packed_lane((13+b._request.lane_id,),b) for b in branches)
        return branches,lanes
    def test_joined_recurrent_exact_next_generation_skips_real_concat_and_matches_default(self):
        prior,charges,reserve=self._joined_setup()
        second,lanes=self._next_joined_lanes(prior)
        baseline_branches,baseline_lanes=self._next_joined_lanes(prior)
        baseline_candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_all_valid_recurrent=True,
                                                   _runtime_factory=lambda:self.runtime)
        baseline,baseline_receipt=baseline_candidate.forward_staged(baseline_lanes,baseline_branches,permit_candidate=True)
        actual,receipt=self.candidate.forward_staged(lanes,second,permit_candidate=True,reserve_scratch=reserve)
        np.testing.assert_array_equal(actual,baseline)
        for actual_branch,reference_branch in zip(second,baseline_branches):
            for actual_cache,reference_cache in zip(actual_branch.recurrent_caches,reference_branch.recurrent_caches):
                for actual_leaf,reference_leaf in zip(actual_cache.cache,reference_cache.cache):
                    np.testing.assert_array_equal(actual_leaf,reference_leaf)
        self.assertEqual(receipt['q1_joined_reused_slots'],48)
        self.assertEqual(receipt['q1_joined_fallback_materializations'],0)
        self.assertEqual(baseline_receipt['q1_joined_reused_slots'],0)
        self.assertEqual(len(charges),1)
        self.assertTrue(all(b.staged for b in second))
        with self.assertRaisesRegex(RuntimeError,'retirement'):
            self.candidate.release_joined_after_retirement((NS(fully_retired=False),))
        self.candidate.release_joined_after_retirement((NS(fully_retired=True),))
        self.assertFalse(self.candidate._joined_bindings)
        self.assertIsNone(self.candidate._joined_charge)
    def test_joined_recurrent_stale_leaf_membership_rollback_and_b1_fall_back(self):
        for shape in ('leaf','membership','rollback','owner','revision','checkpoint','b1'):
            self.setUp()
            prior,charges,reserve=self._joined_setup()
            branches,lanes=self._next_joined_lanes(prior,generation=0 if shape=='rollback' else 1,
                reverse=shape=='membership',copy_leaf=shape=='leaf',new_owner=shape=='owner',
                revision='other-revision' if shape=='revision' else 'test-revision')
            if shape=='checkpoint': branches[0]._origin.companions=()
            if shape=='b1': branches,lanes=(branches[0],),(lanes[0],)
            _,receipt=self.candidate.forward_staged(lanes,branches,permit_candidate=True,reserve_scratch=reserve)
            self.assertEqual(receipt['q1_joined_reused_slots'],0)
            self.assertEqual(receipt['q1_joined_fallback_materializations'],48 if shape!='b1' else 48)
            self.assertEqual(len(charges),1)
            if shape=='b1': self.assertFalse(self.candidate._joined_bindings)
    def test_joined_recurrent_requires_charge_before_native_write(self):
        with self.assertRaisesRegex(ValueError,'all-valid'):
            M.Qwen35PagedCandidate(self.lang,self.backend,q1_joined_recurrent=True)
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_all_valid_recurrent=True,
            q1_joined_recurrent=True,_runtime_factory=lambda:self.runtime)
        _,branches,lanes=self.lanes()
        with self.assertRaisesRegex(ValueError,'charged reservation'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertEqual(self.backend.grouped,0)
    def test_joined_recurrent_charge_cannot_cover_next_shape_falls_back(self):
        prior,charges,reserve=self._joined_setup()
        charges[0].bytes=0
        branches,lanes=self._next_joined_lanes(prior)
        _,receipt=self.candidate.forward_staged(lanes,branches,permit_candidate=True,reserve_scratch=reserve)
        self.assertEqual(receipt['q1_joined_reused_slots'],0)
        self.assertEqual(receipt['q1_joined_fallback_materializations'],48)
        self.assertFalse(self.candidate._joined_bindings)
        self.assertEqual(len(charges),1)
    def test_singleton_survivor_uses_scalar_same_math_and_private_boundary(self):
        originals,branches,lanes=self.lanes(); lane=lanes[1]; branch=branches[1]
        hidden=self.lang.model.embed_tokens(np.asarray([[lane.token_ids[0]]]))
        caches=M.clone_recurrent_caches(originals[1]); fa=gdn=0
        for layer in self.lang.model.layers:
            mixed=norm(hidden)
            if layer.is_linear: attended=layer.linear_attn(mixed,None,caches[gdn]); gdn+=1
            else: attended=layer.self_attn.reference(mixed,lane.layers[fa],96); fa+=1
            hidden=hidden+attended; hidden=hidden+layer.mlp(norm(hidden))
        logits,receipt=self.candidate.forward_staged((lane,),(branch,),permit_candidate=True)
        np.testing.assert_allclose(logits,norm(hidden)[:,0,:],rtol=.002,atol=.002)
        self.assertEqual(receipt['packed_lanes'],1); self.assertEqual(receipt['native_tile_dispatches'],8)
        self.assertEqual(receipt['grouped_q1_writes'],0); self.assertEqual(receipt['scalar_native_writes'],32)
        self.assertTrue(branch.staged); self.assertFalse(branches[0].staged)
        for actual,reference,original in zip(lane.recurrent_caches,caches,originals[1]):
            np.testing.assert_allclose(actual[1],reference[1],rtol=.002,atol=.002)
            self.assertTrue(np.all(original[1]==np.float32(.13)))
    def test_singleton_cow_dependencies_are_not_counted_as_head_writes(self):
        _,branches,lanes=self.lanes();self.backend.cow=True
        _,receipt=self.candidate.forward_staged((lanes[1],),(branches[1],),permit_candidate=True)
        self.assertEqual(receipt['scalar_native_writes'],32)
        self.assertEqual(receipt['native_write_dependency_count'],40)
        self.assertEqual(receipt['native_cow_copy_dependencies'],8)
    def test_missing_singleton_head_write_refuses_before_boundary_stage(self):
        _,branches,lanes=self.lanes();self.backend.bad_scalar=True
        with self.assertRaisesRegex(RuntimeError,'write dispatch proof'):
            self.candidate.forward_staged((lanes[1],),(branches[1],),permit_candidate=True)
        self.assertFalse(branches[1].staged)
    def test_stock_reduction_requires_actual32_and_specialized_counter(self):
        _,branches,lanes=self.lanes();os.environ['MLX2_PAGED_Q1_STOCK_REDUCTION']='1'
        _,receipt=self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertEqual(receipt['q1_simd_stripes'],32)
        self.assertEqual(receipt['native_stock_reduction_dispatches'],8)
    def test_stock_reduction_missing_specialized_counter_refuses_publication(self):
        _,branches,lanes=self.lanes();os.environ['MLX2_PAGED_Q1_STOCK_REDUCTION']='1'
        self.backend.writer.backend.q1_stock_reduction_dispatch_count=lambda:0
        with self.assertRaisesRegex(RuntimeError,'stock-reduction dispatch proof'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertFalse(any(branch.staged for branch in branches))
    def test_long_singleton_stock_flag_uses_scalar_without_split_scratch(self):
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_split_partition=128,_runtime_factory=lambda:self.runtime)
        os.environ['MLX2_PAGED_Q1_SPLIT_KV']='128';os.environ['MLX2_PAGED_Q1_STOCK_REDUCTION']='1'
        _,branches,lanes=self.lanes()
        branch=Branch(tuple(self.owner(129) for _ in range(8)),branches[1].recurrent_caches,self.events)
        lane=self.candidate.packed_lane(lanes[1].token_ids,branch)
        _,receipt=self.candidate.forward_staged((lane,),(branch,),permit_candidate=True)
        self.assertEqual(receipt['native_tile_dispatches'],0)
        self.assertEqual(receipt['native_stock_reduction_dispatches'],0)
        self.assertEqual(receipt['native_split_partial_dispatches'],0)
        self.assertEqual(receipt['native_split_scratch_reserved_bytes'],0)
        self.assertEqual(receipt['scalar_native_writes'],32)
    def test_serving_stock_environment_drift_refuses_before_native_mutation(self):
        _,branches,lanes=self.lanes();self.candidate._serving_stock_reduction=True
        with self.assertRaisesRegex(ValueError,'source-bound profile'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertEqual(self.backend.grouped,0);self.assertFalse(any(branch.staged for branch in branches))
    def test_stock_flag_survivor_uses_requested_short_geometry(self):
        _,branches,lanes=self.lanes();os.environ['MLX2_PAGED_Q1_STOCK_REDUCTION']='1'
        _,receipt=self.candidate.forward_staged((lanes[1],),(branches[1],),permit_candidate=True)
        self.assertEqual(receipt['q1_simd_stripes'],16)
        self.assertEqual(receipt['native_stock_reduction_dispatches'],0)
        self.assertEqual(receipt['native_tile_dispatches'],8)
    def test_boundary_probe_is_opt_in_and_receives_actual_roots(self):
        _,branches,lanes=self.lanes(); events=[]
        self.candidate._fa_boundary_probe=events.append
        self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertEqual([event['layer_index'] for event in events],[3,7,11,15,19,23,27,31])
        self.assertEqual(events[0]['offsets'],(32,96))
        self.assertEqual(events[0]['queries'].shape,(2,16,256))
        self.assertEqual(events[0]['native_attention'].shape,(2,16,256))
    def test_ambiguous_read_quarantines_roots_without_state_staging(self):
        originals,branches,lanes=self.lanes(); self.backend.fail_read=True
        with self.assertRaisesRegex(RuntimeError,'ambiguous native read'): self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertFalse(any(b.staged for b in branches)); self.assertTrue(self.backend.writer.poisoned)
        self.assertEqual(len(self.backend._orphaned_reads),1); self.assertEqual(len(self.candidate._failure_roots),1)
        with self.assertRaisesRegex(RuntimeError,'teardown'): self.candidate.release_failure_roots_after_teardown()
        self.backend._orphaned_reads.clear(); self.backend.writer.failed_arena_torn_down=True
        self.candidate.release_failure_roots_after_teardown(); self.assertEqual(self.candidate._failure_roots,[])
    def test_final_eval_failure_never_drains_or_stages_recurrent_boundary(self):
        _,branches,lanes=self.lanes(); self.fail_eval=True
        with self.assertRaisesRegex(RuntimeError,'ambiguous eval'): self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertNotIn('drain',self.events); self.assertFalse(any(b.staged for b in branches)); self.assertEqual(len(self.backend._orphaned_reads),8)
    def test_deferred_b2_roots_final_eval_terminal_and_counter_proof(self):
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_deferred=True,_runtime_factory=lambda:self.runtime)
        _,branches,lanes=self.lanes()
        _,receipt=self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertEqual(receipt['q1_deferred_eval_delta'],{
            'grouped_write_async_evals':0,'staged_read_async_evals':0,
            'deferred_q1_write_roots':8,'deferred_q1_read_roots':8,
            'deferred_q1_final_evals':1,'deferred_q1_failure_flushes':0})
        self.assertTrue(receipt['q1_deferred_active'])
        self.assertEqual(self.backend.abort_calls,0)
        self.assertFalse(self.backend.deferred_roots)
        self.assertTrue(all(branch.staged for branch in branches))
        self.assertTrue(any(event[0]=='eval' and event[1]>=16 for event in self.events if isinstance(event,tuple)))
    def test_deferred_failure_flushes_and_retains_roots_until_teardown(self):
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_deferred=True,_runtime_factory=lambda:self.runtime)
        _,branches,lanes=self.lanes();self.fail_eval=True
        with self.assertRaisesRegex(RuntimeError,'ambiguous eval'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertEqual(self.backend.abort_calls,1)
        self.assertEqual(self.backend.writer.backend.deferred_q1_failure_flushes,1)
        self.assertFalse(any(branch.staged for branch in branches))
        self.assertTrue(self.backend.writer.poisoned)
        self.assertEqual(len(self.candidate._failure_roots[-1][-1]),16)
        with self.assertRaisesRegex(RuntimeError,'teardown'):
            self.candidate.release_failure_roots_after_teardown()
        self.backend.writer.failed_arena_torn_down=True
        self.candidate.release_failure_roots_after_teardown()
    def test_deferred_ambiguous_flush_keeps_submitted_read_lease_and_backend_roots(self):
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_deferred=True,_runtime_factory=lambda:self.runtime)
        _,branches,lanes=self.lanes();self.fail_eval=True
        def ambiguous(*args): raise RuntimeError('ambiguous flush')
        self.backend.abort_deferred_q1=ambiguous
        with self.assertRaisesRegex(RuntimeError,'ambiguous eval'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertEqual(len(self.backend._orphaned_reads),8)
        self.assertEqual(len(self.backend.deferred_roots),16)
        self.assertFalse(any(branch.staged for branch in branches))
        self.assertTrue(self.backend.writer.poisoned)
    def test_deferred_b1_survivor_uses_eager_path_without_deferred_roots(self):
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_deferred=True,_runtime_factory=lambda:self.runtime)
        _,branches,lanes=self.lanes()
        _,receipt=self.candidate.forward_staged(lanes[:1],branches[:1],permit_candidate=True)
        self.assertFalse(receipt['q1_deferred_active'])
        self.assertEqual(receipt['q1_deferred_eval_delta'],{})
        self.assertFalse(self.backend.deferred_roots)
    def test_write_only_deferred_b2_preserves_eager_reads_and_terminal_proof(self):
        self.backend.profiling_enabled=True;self.backend.host_profile_ns={'graph_eval':0}
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_deferred_writes_only=True,
            _runtime_factory=lambda:self.runtime)
        _,branches,lanes=self.lanes()
        _,receipt=self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertTrue(receipt['q1_deferred_writes_only_selected'])
        self.assertEqual(receipt['q1_deferred_eval_delta'],{
            'grouped_write_async_evals':0,'staged_read_async_evals':8,
            'deferred_q1_write_roots':8,'deferred_q1_read_roots':8,
            'deferred_q1_final_evals':1,'deferred_q1_failure_flushes':0})
        self.assertFalse(self.backend.deferred_roots)
        self.assertTrue(all(branch.staged for branch in branches))
        self.assertGreater(self.backend.host_profile_ns['graph_eval'],0)
        self.assertGreater(self.backend.host_profile_ns['graph_eval_process_cpu'],0)
    def test_write_only_deferred_ambiguous_read_retains_roots_and_blocks_stage(self):
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_deferred_writes_only=True,
            _runtime_factory=lambda:self.runtime)
        _,branches,lanes=self.lanes();self.backend.fail_read=True
        self.backend.abort_deferred_q1=lambda *args:(_ for _ in ()).throw(RuntimeError('late terminal'))
        with self.assertRaisesRegex(RuntimeError,'ambiguous native read'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertTrue(self.backend.deferred_roots)
        self.assertTrue(self.backend._orphaned_reads)
        self.assertFalse(any(branch.staged for branch in branches))
    def test_write_only_deferred_final_eval_failure_flushes_before_publication(self):
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_deferred_writes_only=True,
            _runtime_factory=lambda:self.runtime)
        _,branches,lanes=self.lanes();self.fail_eval=True
        with self.assertRaisesRegex(RuntimeError,'ambiguous eval'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertEqual(self.backend.abort_calls,1)
        self.assertEqual(len(self.candidate._failure_roots[-1][-1]),16)
        self.assertTrue(self.backend.writer.poisoned)
        self.assertFalse(any(branch.staged for branch in branches))
    def test_deferred_modes_are_exclusive(self):
        with self.assertRaisesRegex(ValueError,'exclusive grouped-write'):
            M.Qwen35PagedCandidate(self.lang,self.backend,q1_deferred=True,
                q1_deferred_writes_only=True)
    def test_deferred_missing_lifecycle_refuses_before_first_native_write(self):
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_deferred=True,_runtime_factory=lambda:self.runtime)
        _,branches,lanes=self.lanes()
        self.backend.writer.backend.begin_deferred_q1=None
        with self.assertRaisesRegex(ValueError,'lifecycle capability'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertEqual(self.backend.grouped,0)
        self.assertFalse(any(branch.staged for branch in branches))
    def test_deferred_missing_read_root_blocks_recurrent_stage(self):
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_deferred=True,_runtime_factory=lambda:self.runtime)
        _,branches,lanes=self.lanes()
        original=self.backend.read_staged
        def missing_counter(*args,**kwargs):
            value=original(*args,**kwargs)
            self.backend.writer.backend.deferred_q1_read_roots=0
            return value
        self.backend.read_staged=missing_counter
        with self.assertRaisesRegex(RuntimeError,'submission/root proof'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertFalse(any(branch.staged for branch in branches))
        self.assertEqual(self.backend.abort_calls,1)
    def test_scalar_fallback_fails_physical_proof(self):
        _,branches,lanes=self.lanes(); self.backend.bad_tile=True
        with self.assertRaisesRegex(RuntimeError,'tile dispatch proof'): self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertFalse(any(b.staged for b in branches))
    def test_short_context_refusal_precedes_any_native_mutation(self):
        _,branches,lanes=self.lanes(); branches[1]._origin.layers[0].offset=128
        with self.assertRaisesRegex(ValueError,'visible context'): self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertEqual(self.backend.grouped,0)
    def test_clone_metadata_is_private_and_tensor_roots_are_shared(self):
        c=Cache(); c._checkpoints=[[(32,list(c.cache))]]; clone=M.clone_recurrent_caches((c,))[0]
        self.assertIsNot(clone,c); self.assertIsNot(clone.cache,c.cache); self.assertIs(clone[1],c[1])
        self.assertIsNot(clone._checkpoints[0],c._checkpoints[0]); clone._checkpoints[0].append(('new',[]))
        self.assertEqual(len(c._checkpoints[0]),1)
    def test_bootstrap_import_uses_logical_kv_with_charged_rounded_allocation(self):
        layers=tuple(self.owner(0) for _ in range(8)); reservations=[]
        def reserve(size):
            value=NS(bytes=size,released=False,failures=[])
            value.release=lambda:setattr(value,'released',True)
            value.retain_failure_roots=lambda roots:value.failures.append(roots)
            reservations.append(value); return value
        result=self.candidate.bootstrap_ordinary(tuple(range(32)),layers,reserve_staging=reserve,permit_candidate=True)
        self.assertTrue(reservations[0].released); self.assertEqual(len(result.recurrent_caches),24)
        self.assertEqual([o.offset for o in layers],[32]*8)
        self.assertEqual(result.receipt['ordinary_kv_bytes'],8*2*4*256*256*2)
        self.assertEqual(result.receipt['packed_import_bytes'],8*2*4*32*256*2)
        self.assertEqual(self.events.count(('import',32)),8)
    def test_bootstrap_eval_failure_keeps_staging_charge_and_roots(self):
        self.fail_eval=True; reservation=NS(bytes=self.candidate.bootstrap_staging_bytes(32),released=False,roots=None)
        reservation.release=lambda:setattr(reservation,'released',True)
        reservation.retain_failure_roots=lambda roots:setattr(reservation,'roots',roots)
        with self.assertRaisesRegex(RuntimeError,'ambiguous eval'): self.candidate.bootstrap_ordinary(tuple(range(32)),tuple(self.owner(0) for _ in range(8)),reserve_staging=lambda size:reservation,permit_candidate=True)
        self.assertFalse(reservation.released); self.assertIsNotNone(reservation.roots)
    def test_mixed_recurrent_dtype_refuses_before_native_mutation(self):
        _,branches,lanes=self.lanes()
        lanes[1].recurrent_caches[0][1]=lanes[1].recurrent_caches[0][1].astype(np.float16)
        with self.assertRaisesRegex(ValueError,'geometry/dtype'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertEqual(self.backend.grouped,0)
    def test_insufficient_reservation_is_retained_without_prefill(self):
        reservation=NS(bytes=1,roots=None,release=lambda:None)
        reservation.retain_failure_roots=lambda roots:setattr(reservation,'roots',roots)
        with self.assertRaisesRegex(ValueError,'reservation'):
            self.candidate.bootstrap_ordinary(tuple(range(32)),tuple(self.owner(0) for _ in range(8)),reserve_staging=lambda size:reservation,permit_candidate=True)
        self.assertIsNotNone(reservation.roots); self.assertEqual(self.events,[])
    def test_same_dtype_preflight_refuses_bfloat16_before_mutation(self):
        self.lang.compute_dtype='bfloat16'; _,branches,lanes=self.lanes()
        with self.assertRaisesRegex(ValueError,'storage dtype'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertEqual(self.backend.grouped,0)
    def test_explicit_conversion_checks_are_materialized_before_state_staging(self):
        self.candidate.kv_precision='float16_candidate'; _,branches,lanes=self.lanes()
        logits,receipt=self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertEqual(receipt['representability_checks'],24)
        self.assertEqual(receipt['kv_precision_policy'],'float16_candidate')
        self.assertIn(('eval',121),self.events); self.assertTrue(all(b.staged for b in branches))
    def test_unrepresentable_conversion_is_rejected_before_publication(self):
        self.candidate.kv_precision='float16_candidate'
        converted,checks=self.candidate._native_qkv((np.asarray([1e8],np.float32),),self.mx)
        with self.assertRaisesRegex(ValueError,'unrepresentable'):
            self.candidate._require_representable(checks)
        self.assertEqual(converted[0].dtype,np.float16)
    def test_nonfinite_conversion_is_rejected(self):
        self.candidate.kv_precision='float16_candidate'
        _,checks=self.candidate._native_qkv((np.asarray([np.nan],np.float32),),self.mx)
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            self.candidate._require_representable(checks)
    def test_bf16_policy_admits_only_explicit_boundary_conversion(self):
        self.lang.compute_dtype='bfloat16'; self.candidate.kv_precision='float16_candidate'
        self.assertEqual(self.candidate.native_dtype_preflight(),'bfloat16')
        self.candidate.kv_precision='same'
        self.assertEqual(self.candidate.native_dtype_preflight(),'bfloat16')
    def test_retention_callback_failure_does_not_lose_staging_roots(self):
        self.fail_eval=True
        reservation=NS(bytes=self.candidate.bootstrap_staging_bytes(32),release=lambda:None,
            retain_failure_roots=lambda roots:(_ for _ in ()).throw(RuntimeError('retention callback fault')))
        with self.assertRaisesRegex(RuntimeError,'ambiguous eval'):
            self.candidate.bootstrap_ordinary(tuple(range(32)),tuple(self.owner(0) for _ in range(8)),reserve_staging=lambda size:reservation,permit_candidate=True)
        self.assertIs(self.candidate._bootstrap_failure_roots[0][0],reservation)
    def test_release_failure_retains_completed_bootstrap_graph(self):
        reservation=NS(bytes=self.candidate.bootstrap_staging_bytes(32),
            release=lambda:(_ for _ in ()).throw(RuntimeError('release callback fault')),retain_failure_roots=lambda roots:None)
        with self.assertRaisesRegex(RuntimeError,'release callback fault'):
            self.candidate.bootstrap_ordinary(tuple(range(32)),tuple(self.owner(0) for _ in range(8)),reserve_staging=lambda size:reservation,permit_candidate=True)
        self.assertIs(self.candidate._bootstrap_failure_roots[0][0],reservation)
    def test_real_trunk_has_no_args_contract(self):
        self.assertFalse(hasattr(self.lang.model,"args"))
        self.assertIs(self.candidate.args,self.lang.args)
    def test_exact_bf16_boundary_preserves_array_identity(self):
        self.lang.compute_dtype='bfloat16'; self.candidate.kv_precision='same'
        plane=np.asarray([.125],np.float32)  # BF16 dtype metadata proxy; no actual BF16 runtime.
        native,checks=self.candidate._native_qkv((plane,),self.mx)
        self.assertIs(native[0],plane); self.assertEqual(checks,())
    def test_long_q1_requires_split_dispatches_and_charged_scratch(self):
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,q1_split_partition=128,_runtime_factory=lambda:self.runtime)
        os.environ['MLX2_PAGED_Q1_SPLIT_KV']='128'
        _,branches,lanes=self.lanes()
        branches=list(branches); branches[1]=Branch(tuple(self.owner(129) for _ in range(8)),branches[1].recurrent_caches,self.events)
        branches=tuple(branches); lanes=tuple(self.candidate.packed_lane(l.token_ids,b) for l,b in zip(lanes,branches))
        with self.assertRaisesRegex(ValueError,'scratch reservation'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        reservation=NS(bytes=self.candidate.forward_scratch_bytes((32,129)),released=False,retain_failure_roots=lambda r:None)
        reservation.release=lambda:setattr(reservation,'released',True)
        _,receipt=self.candidate.forward_staged(lanes,branches,permit_candidate=True,reserve_scratch=lambda size:reservation)
        self.assertEqual(receipt['native_tile_dispatches'],0); self.assertEqual(receipt['native_split_partial_dispatches'],8)
        self.assertEqual(receipt['native_split_reduce_dispatches'],8); self.assertTrue(reservation.released)
    def long_stock(self):
        args=self.lang.args;args.num_hidden_layers=64;args.hidden_size=5120
        args.num_attention_heads=24;args.linear_num_value_heads=48
        self.lang.model.layers=[Layer(index) for index in range(64)]
        for layer in self.lang.model.layers:
            if not layer.is_linear:
                layer.self_attn.num_attention_heads=24
                layer.self_attn.q_proj=Projection(24*512)
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,stock_long=True,_runtime_factory=lambda:self.runtime)
        os.environ['MLX2_PAGED_Q1_STOCK_LONG']='1'
        branches=tuple(Branch(tuple(self.owner(offset) for _ in range(16)),tuple(Cache(.1) for _ in range(48)),self.events)
                       for offset in (1024,1026))
        lanes=tuple(self.candidate.packed_lane((13+index,),branch) for index,branch in enumerate(branches))
        return branches,lanes
    def test_stock_long128_charged_shape_and_exact_physical_counters(self):
        branches,lanes=self.long_stock()
        expected=16*2*24*128*(256+2)*4
        self.assertEqual(self.candidate.forward_scratch_bytes((1024,1026)),expected)
        reservation=NS(bytes=expected,released=False,retain_failure_roots=lambda roots:None)
        reservation.release=lambda:setattr(reservation,'released',True)
        _,receipt=self.candidate.forward_staged(lanes,branches,permit_candidate=True,reserve_scratch=lambda size:reservation)
        self.assertEqual(receipt['native_stock_long_partial_dispatches'],16)
        self.assertEqual(receipt['native_stock_long_reduce_dispatches'],16)
        self.assertEqual(receipt['native_split_partial_dispatches'],0)
        self.assertEqual(receipt['native_split_reduce_dispatches'],0)
        self.assertEqual(receipt['native_tile_dispatches'],0)
        self.assertEqual(receipt['native_partial_numerator_rounding'],'float16')
        self.assertTrue(reservation.released);self.assertTrue(all(branch.staged for branch in branches))
    def test_stock_long_inline_exact_counter_and_prewrite_guards(self):
        branches,lanes=self.long_stock()
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,stock_long=True,
            stock_long_inline_metadata=True,_runtime_factory=lambda:self.runtime)
        expected=self.candidate.forward_scratch_bytes((1024,1026))
        reservation=NS(bytes=expected,release=lambda:None,retain_failure_roots=lambda roots:None)
        with self.assertRaisesRegex(ValueError,'flag differs'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True,reserve_scratch=lambda size:reservation)
        self.assertEqual(self.backend.grouped,0)
        os.environ['MLX2_PAGED_Q1_STOCK_LONG_INLINE_METADATA']='1'
        self.backend.writer.backend._native=NS()
        with self.assertRaisesRegex(ValueError,'physical-counter ABI'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True,reserve_scratch=lambda size:reservation)
        self.assertEqual(self.backend.grouped,0)
        self.backend.writer.backend._native=NS(q1_stock_long_metadata_dispatch_count=lambda _:self.backend.long_inline)
        _,receipt=self.candidate.forward_staged(lanes,branches,permit_candidate=True,reserve_scratch=lambda size:reservation)
        self.assertEqual(receipt['native_stock_long_metadata_dispatches'],16)
        self.assertTrue(receipt['stock_long_inline_metadata_selected'])
        self.assertTrue(all(branch.staged for branch in branches))
    def test_stock_long_inline_missing_physical_count_refuses_gdn_stage(self):
        branches,lanes=self.long_stock()
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,stock_long=True,
            stock_long_inline_metadata=True,_runtime_factory=lambda:self.runtime)
        os.environ['MLX2_PAGED_Q1_STOCK_LONG_INLINE_METADATA']='1'
        self.backend.writer.backend.q1_stock_long_metadata_dispatch_count=lambda:0
        expected=self.candidate.forward_scratch_bytes((1024,1026))
        reservation=NS(bytes=expected,release=lambda:None,retain_failure_roots=lambda roots:None)
        with self.assertRaisesRegex(RuntimeError,'inline metadata dispatch proof'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True,reserve_scratch=lambda size:reservation)
        self.assertFalse(any(branch.staged for branch in branches))
    def test_stock_long_inline_rejects_short_or_singleton_before_write(self):
        branches,lanes=self.long_stock()
        self.candidate=M.Qwen35PagedCandidate(self.lang,self.backend,stock_long=True,
            stock_long_inline_metadata=True,_runtime_factory=lambda:self.runtime)
        os.environ['MLX2_PAGED_Q1_STOCK_LONG_INLINE_METADATA']='1'
        with self.assertRaisesRegex(ValueError,'eligible long B2'):
            self.candidate.forward_staged(lanes[:1],branches[:1],permit_candidate=True)
        self.assertEqual(self.backend.grouped,0)
        for branch in branches:
            for owner in branch.layers: owner.offset=63
            branch._origin.layers=(NS(offset=63),)
        with self.assertRaisesRegex(ValueError,'eligible long B2'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertEqual(self.backend.grouped,0)
    def test_stock_long_missing_counter_refuses_before_write_or_scratch(self):
        branches,lanes=self.long_stock()
        self.backend.writer.backend.q1_stock_long_reduce_dispatch_count=None
        with self.assertRaisesRegex(ValueError,'partial/reduce capability'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertEqual(self.backend.grouped,0);self.assertFalse(any(branch.staged for branch in branches))
    def test_stock_long_architecture_and_block_override_fail_before_write(self):
        branches,lanes=self.long_stock();self.mx.device_info=lambda:{'architecture':'applegpu_g16x'}
        with self.assertRaisesRegex(ValueError,'architecture s'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.mx.device_info=lambda:{'architecture':'applegpu_g16s'};os.environ['MLX_SDPA_BLOCKS']='64'
        with self.assertRaisesRegex(ValueError,'block overrides'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertEqual(self.backend.grouped,0)
    def test_stock_long_rejects_four_way_gqa_and_old_split(self):
        candidate=M.Qwen35PagedCandidate(self.lang,self.backend,stock_long=True,_runtime_factory=lambda:self.runtime)
        self.assertEqual(candidate.max_visible_tokens,8192)
        with self.assertRaisesRegex(ValueError,'excludes old split'):
            M.Qwen35PagedCandidate(self.lang,self.backend,stock_long=True,q1_split_partition=128)
        branches,lanes=self.long_stock();self.lang.args.num_attention_heads=16
        with self.assertRaisesRegex(ValueError,'GQA above four'):
            self.candidate.forward_staged(lanes,branches,permit_candidate=True)
        self.assertEqual(self.backend.grouped,0)
    def test_topology_maps_and_admission(self):
        mapping=M.hybrid_layer_map(self.lang.model,self.lang.args); self.assertEqual(mapping.full_attention,tuple(range(3,32,4)))
        self.lang.args.full_attention_interval=3
        with self.assertRaisesRegex(ValueError,'topology'): M.hybrid_layer_map(self.lang.model,self.lang.args)


class ActualHeadSpanContracts(unittest.TestCase):
    def test_shared_tail_q1_submits_four_head_writes_plus_one_ordered_copy(self):
        tree=ast.parse((ROOT/'src/mlx2/runtime/paged_kv_token.py').read_text())
        methods=[node for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=='PagedKVTokenOwner'
                 for node in cls.body if isinstance(node,ast.FunctionDef) and node.name in ('planned_spans','append_staged')]
        span=namedtuple('TokenWriteSpan','source_token_offset token_count kv_head block_index within_page_offset byte_count')
        scope={'TokenWriteSpan':span,'PAGE_SIZE':64,'U32_MAX':2**32-1,
               '_positive':lambda *args:None,'WriteTicket':type('WriteTicket',(),{}),'CopyTicket':type('CopyTicket',(),{})}
        exec(compile(ast.Module(body=methods,type_ignores=[]),'paged_kv_token.py','exec'),scope)
        events=[];copy_dependency=object();source=object();destination=object()
        def copy_page(*args,**kwargs):
            events.append(('copy',kwargs['byte_count']));return NS(dependency=copy_dependency)
        def depend_source(chunk,dependency):
            self.assertIs(dependency,copy_dependency);events.append(('dependency',chunk.nbytes));return chunk
        def write(handle,**kwargs):
            events.append(('write',kwargs['within_page_offset'],kwargs['byte_count']))
            return NS(dependency=object())
        backend=NS(copy_page=lambda *args:None,depend_source=depend_source,
                   validate_sources=lambda key,value:key.nbytes==value.nbytes)
        writer=NS(pool=NS(references=lambda handle:2,free_count=10),backend=backend,
                  submit_copy=copy_page,submit_write=write)
        stage=NS(source=source,destination=destination,handles=(object(),destination))
        sequence=NS(kv_end=96,handles=(object(),source),first_block=0,stage_append=lambda *args,**kwargs:stage)
        owner=NS(_ready=lambda:None,sequence=sequence,_reuse_shared_tail_refs=1,writer=writer,
                 profile=NS(kv_heads=4,head_token_bytes=512,page_bytes=4*64*512))
        owner.planned_spans=lambda count,staged=False:scope['planned_spans'](owner,count,staged=staged)
        spans=owner.planned_spans(1,staged=True)
        self.assertEqual([item.kv_head for item in spans],[0,1,2,3])
        chunks=tuple(NS(nbytes=item.byte_count) for item in spans)
        tickets=scope['append_staged'](owner,chunks,chunks,token_count=1)
        self.assertEqual(len(spans),4);self.assertEqual(len(tickets),5)
        self.assertIs(tickets[0].dependency,copy_dependency)
        self.assertEqual(events[0],('copy',131072))
        self.assertEqual([item[0] for item in events],['copy']+['dependency']*8+['write']*4)
        self.assertEqual([item[1] for item in events if item[0]=='write'],[16384,49152,81920,114688])
        self.assertIs(owner._pending,tickets)

if __name__=='__main__': unittest.main()
