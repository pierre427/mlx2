"""Direct NumPy control-flow contracts; prohibits MLX/native runtime imports."""
import ast
import copy
import builtins
from unittest.mock import patch
from contextlib import nullcontext
import importlib.abc
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import numpy as np

if __name__ != '__main__':raise unittest.SkipTest('run directly without MLX')
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native':raise RuntimeError('runtime forbidden')
sys.meta_path.insert(0,NoRuntime())
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('mlx2.runtime.hybrid_packed_prefill',ROOT/'src/mlx2/runtime/hybrid_packed_prefill.py')
M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M)
sys.path.insert(0,str(ROOT/'src'))
M.__package__='mlx2.runtime'
from mlx2.runtime.packed_prefill_receipt import bootstrap_prefill_attribution
from mlx2.runtime.paged_pack_price import IDENTITY_FIELDS

def identity():
    value={k:'a'*64 for k in IDENTITY_FIELDS}
    value.update(host='cpu',hardware='cpu',source_commit='b'*40,mlx_wheel_version='pinned')
    return value

class Reservation:
    def __init__(self):self.bytes=1<<30;self.completed=False;self.roots=()
    def release(self):self.completed=True
    def retain_failure_roots(self,roots):self.roots=roots
class Cache:
    def __init__(self):self.cache=[None,None];self.speculating=False;self.lengths=self.left_padding=None
class Proj:
    def __init__(self,width):self.width=width;self.calls=[]
    def __call__(self,x):
        self.calls.append(tuple(x.shape));return np.broadcast_to(x.mean(-1,keepdims=True),(*x.shape[:-1],self.width)).copy()
class GDN:
    def __init__(self):self.parts=[];self.calls=0
    def mixed(self,x,parts):
        self.calls+=1;self.parts.append(tuple((r,n,s,m) for r,n,s,c,m in parts))
        for r,n,start,cache,mask in parts:
            assert r==1 and mask is None
            cache.cache=[x[:,start+n-1:start+n].copy(),x[:,start:start+n].sum(axis=1,keepdims=True)]
        return x*.01
class Attention:
    num_attention_heads=2;num_key_value_heads=1;scale=256**-.5
    def __init__(self):
        self.q_proj=Proj(1024);self.k_proj=Proj(256);self.v_proj=Proj(256);self.o_proj=Proj(4)
        self.q_norm=self.k_norm=lambda x:x
        self.rope_calls=[]
    def rope(self,x,offset):self.rope_calls.append((tuple(x.shape),offset));return x
class Backend:
    profiling_enabled=True
    def __init__(self):
        self.writer=NS(backend=NS(_arena=self,stream=None),pending_epochs=(),ledger=NS(pending_count=0))
        self._orphaned_reads={};self.writes=0;self.rows=0;self.reads=0;self.fail_drain=False;self.bad_offset=False
    def append_packed_multirow(self,owners,k,v,counts,permit_candidate):
        self.writes+=1;self.rows+=sum(counts);self.owners=owners;self.counts=counts
        assert k.shape==v.shape==(sum(counts),1,256)
        return (NS(dependency=1,packed_multirow=True),)
    def read_staged(self,use,q,tickets,scale):
        self.reads+=1;use.state='submitted';return q*.02
    def drain_staged(self,owners,uses):
        if self.fail_drain:raise RuntimeError('injected terminal failure')
        for owner,n in zip(owners,self.counts):owner.offset=n-1 if self.bad_offset else n
        for u in uses:u.state='closed'
        return (NS(),)


def setup():
    b=Backend();g=GDN();a=Attention()
    identity=lambda x:x
    layers=(NS(is_linear=True,input_layernorm=identity,post_attention_layernorm=identity,linear_attn=g,mlp=lambda x:x*.03),
            NS(is_linear=False,input_layernorm=identity,post_attention_layernorm=identity,self_attn=a,mlp=lambda x:x*.03))
    args=NS(hidden_size=4,intermediate_size=8,linear_num_key_heads=1,linear_key_head_dim=2,
            linear_num_value_heads=1,linear_value_head_dim=2,num_attention_heads=2,num_key_value_heads=1)
    c=NS(args=args,backend=b,layer_map=NS(recurrent=(0,),full_attention=(1,)),native_layer_count=1)
    c.trunk=NS(layers=layers,embed_tokens=lambda t:np.repeat(t[...,None].astype(np.float16),4,axis=-1),norm=identity)
    c.language_model=NS(make_cache=lambda:[Cache(),None],logits=lambda x:x)
    c.native_dtype_preflight=lambda:'bfloat16';c.bootstrap_staging_bytes=lambda n:0
    mx=NS(bfloat16=np.dtype('float16'),array=np.asarray,split=lambda x,n,axis:np.split(x,n,axis),
          concatenate=np.concatenate,stream=lambda s:nullcontext(),eval=lambda *x:None,
          all=np.all,isfinite=np.isfinite,stack=np.stack)
    native=NS(grouped_multirow_write_count=lambda arena:arena.writes,
              grouped_multirow_row_count=lambda arena:arena.rows,
              prefill_matrix_dispatch_count=lambda arena:arena.reads)
    rt=NS(mx=mx,native=native,gate_sigmoid=lambda x:1/(1+np.exp(-x)),
          prepare_read=lambda *args,**kw:NS(state='prepared',lease=NS(epoch=9)))
    owners=tuple((NS(offset=0,_pending=(),_stage=None),) for _ in range(2))
    return c,owners,rt,a,g

class Tests(unittest.TestCase):
    def run_math(self,c,owners,rt,res,**kw):
        return M.run_packed_math(c,owners,((1,2),(3,4,5)),res,runtime=rt,cancelled=kw.get('cancelled',lambda:False))
    def test_actual_shared_projections_and_segment_states(self):
        c,o,rt,a,g=setup();r=Reservation();logits,caches,receipt=self.run_math(c,o,rt,r)
        self.assertEqual(logits.shape,(2,4));self.assertTrue(r.completed)
        self.assertEqual(a.q_proj.calls,[(1,5,4)]);self.assertEqual(a.k_proj.calls,[(1,5,4)])
        self.assertEqual(a.o_proj.calls,[(1,5,512)])
        self.assertEqual(g.parts,[((1,2,0,None),(1,3,2,None))])
        self.assertNotEqual(float(caches[0][0].cache[1].sum()),float(caches[1][0].cache[1].sum()))
        self.assertEqual(receipt['physical_counters'],{'grouped_multirow_write_count':1,'grouped_multirow_row_count':5,'prefill_matrix_dispatch_count':1})
        self.assertTrue(all(offset==0 for shape,offset in a.rope_calls))
    def test_terminal_failure_retains_exact_read_and_tensor_roots(self):
        c,o,rt,a,g=setup();c.backend.fail_drain=True;r=Reservation()
        with self.assertRaisesRegex(RuntimeError,'terminal'):self.run_math(c,o,rt,r)
        self.assertFalse(r.completed);self.assertTrue(r.roots)
        self.assertEqual(list(c.backend._orphaned_reads),[9])
    def test_offset_mismatch_blocks_handoff(self):
        c,o,rt,a,g=setup();c.backend.bad_offset=True;r=Reservation()
        with self.assertRaisesRegex(RuntimeError,'offset'):self.run_math(c,o,rt,r)
        self.assertFalse(r.completed);self.assertTrue(r.roots)
    def test_cancel_before_submission_has_no_native_use(self):
        c,o,rt,a,g=setup();r=Reservation()
        with self.assertRaisesRegex(ValueError,'cancelled'):self.run_math(c,o,rt,r,cancelled=lambda:True)
        self.assertEqual(c.backend.writes,0);self.assertEqual(c.backend.reads,0)
    def test_masked_cache_refused_before_native_use(self):
        c,o,rt,a,g=setup();r=Reservation()
        def cache():
            v=Cache();v.lengths=(2,);return [v,None]
        c.language_model.make_cache=cache
        with self.assertRaisesRegex(ValueError,'unmasked'):self.run_math(c,o,rt,r)
        self.assertEqual(c.backend.writes,0);self.assertTrue(r.roots)
    def test_cancel_at_second_layer_retains_partial_state_without_write(self):
        c,o,rt,a,g=setup();r=Reservation();checks=iter((False,True))
        with self.assertRaisesRegex(ValueError,'cancelled'):self.run_math(c,o,rt,r,cancelled=lambda:next(checks))
        self.assertEqual(g.calls,1);self.assertEqual(c.backend.writes,0);self.assertTrue(r.roots)
    def test_charge_failure_precedes_tensor_or_native_use(self):
        c,o,rt,a,g=setup();r=Reservation();r.bytes=0
        with self.assertRaises(MemoryError):self.run_math(c,o,rt,r)
        self.assertEqual(c.backend.writes,0);self.assertEqual(g.calls,0)
    def test_physical_counter_mismatch_retains_roots(self):
        c,o,rt,a,g=setup();rt.native.prefill_matrix_dispatch_count=lambda a:0;r=Reservation()
        with self.assertRaisesRegex(RuntimeError,'physical'):self.run_math(c,o,rt,r)
        self.assertFalse(r.completed);self.assertTrue(r.roots)
    def test_missing_capability_fails_before_query(self):
        with self.assertRaisesRegex(ValueError,'missing'):M.require_capabilities(NS(),Backend)
    def test_capability_geometry_rejected(self):
        names=('grouped_multirow_write','grouped_multirow_write_count','grouped_multirow_row_count','prefill_matrix_dispatch_count')
        n=NS(**{k:lambda:None for k in names},prefill_matrix_capability=lambda:{'head_dim':128})
        with self.assertRaisesRegex(ValueError,'geometry'):M.require_capabilities(n,Backend)
    def test_request_revision_empty_bool_and_duplicate_refused(self):
        valid=((0,'r',(1,2),2),(1,'r',(3,4),4))
        self.assertEqual(M.validate_requests(valid,'r',100),(2,2))
        for x in (((0,'x',(1,2),2),valid[1]),(valid[0],valid[0]),((0,'r',(True,2),2),valid[1]),((0,'r',(),2),valid[1])):
            with self.assertRaises(ValueError):M.validate_requests(x,'r',100)
    def test_profile_identity_environment_and_unqualified_boundaries(self):
        ident=identity()
        p={'schema':M.SCHEMA,'profile_id':'test','identity':ident,'storage_dtype':'bfloat16',
            'memory_budget_bytes':1<<30,'required_environment':M.packed_environment(),
            'qualified':False,'price_usable':False,'serving_default':False,
            'context_lengths':[32,96],'q1_simd_stripes':16,'stock_reduction':True,'stock_singleton':True}
        self.assertIs(M.validate_profile(p,live_identity=ident,counts=(32,96),environment=M.packed_environment()),p)
        for key,value in (('qualified',True),('storage_dtype','float16'),('context_lengths',[32,97]),('q1_simd_stripes',True)):
            bad=copy.deepcopy(p);bad[key]=value
            with self.assertRaises(ValueError):M.validate_profile(bad,live_identity=ident,counts=(32,96),environment=M.packed_environment())
        drift=copy.deepcopy(ident);drift['source_commit']='c'*40
        with self.assertRaises(ValueError):M.validate_profile(p,live_identity=drift,counts=(32,96),environment=M.packed_environment())
        env=M.packed_environment();env['MLX2_PAGED_PREFILL_MATRIX']='0'
        with self.assertRaises(ValueError):M.validate_profile(p,live_identity=ident,counts=(32,96),environment=env)
    def test_imported_attribution_retained_when_no_packed_proof(self):
        self.assertEqual(bootstrap_prefill_attribution(NS())['prefill_mode'],'ordinary_completed_import')
    def packed_candidate(self):
        proof={'prefill_mode':'packed_hybrid_real_rows','bootstrap_generation':0,'segment_lengths':(32,96),
            'real_projection_rows':128,'full_attention_layers':16,'terminal_read_count':16,
            'qualified':False,'price_usable':False,'selected':True,'observed_used':True,
            'physical_counters':{'grouped_multirow_write_count':16,'grouped_multirow_row_count':2048,'prefill_matrix_dispatch_count':16},
            'source_identity':identity()}
        arena=NS(_arena=object(),_native=NS(**{k:(lambda arena,n=n:n) for k,n in proof['physical_counters'].items()}))
        return NS(_packed_prefill_receipt=proof,native_layer_count=16,bootstrap_generation=0,
            backend=NS(read_submissions=16,terminal_successes=16,writer=NS(backend=arena)))
    def test_packed_attribution_requires_actual_live_counts(self):
        c=self.packed_candidate();p=bootstrap_prefill_attribution(c)
        self.assertEqual(p['prefill_mode'],'native_packed_prefill');self.assertEqual(p['prefill_layout'],'real_rows')
        self.assertTrue(p['native_prefill_observed_used'])
        c.backend.writer.backend._native.prefill_matrix_dispatch_count=lambda arena:15
        with self.assertRaises(ValueError):bootstrap_prefill_attribution(c)
    def test_invalid_packed_attribution_refused(self):
        for key,value in (('real_projection_rows',192),('terminal_read_count',0),('qualified',True),('bootstrap_generation',1)):
            c=self.packed_candidate();c._packed_prefill_receipt[key]=value
            with self.assertRaises(ValueError):bootstrap_prefill_attribution(c)
        c=self.packed_candidate();c.backend.terminal_successes=0
        with self.assertRaises(ValueError):bootstrap_prefill_attribution(c)
    def factory_fixture(self, fail_second=False, cancel_after_math=False, nax_exact=False, long_fused=False, host_memory=128<<30, long_cap_missing=False):
        # Execute the actual new factory body with host-only allocation/math
        # seams. Actual HybridServingResources accounts and retires the pages.
        from mlx2.runtime import qwen35_paged_graph_factory as R
        from mlx2.runtime.paged_gdn_checkpoint import GDNBoundaryCheckpoint
        base_charge=R._CHARGED;pool=NS(allocated_count=0,retire=lambda epoch:None)
        state={'constructed':0,'closed':0,'math_done':False}
        class TokenOwner:
            def __init__(self,writer,kv,permit_candidate):
                self.writer=writer;self.offset=0;self.closed=False;pool.allocated_count+=1
            def close(self):
                if not self.closed:self.closed=True;pool.allocated_count-=1;state['closed']+=1
        class PublicOwner:
            def __init__(self,revision,layers,companions,**kwargs):
                state['constructed']+=1
                if fail_second and state['constructed']==2:raise RuntimeError('second owner construction failed')
                self.layers=layers;self.fully_retired=False
            def close(self):
                for layer in self.layers:layer.close()
                self.fully_retired=True
            def reap_retired(self):pass
            def reap_quarantine(self):pass
        arena=NS(_closed=False,close_after_terminal=lambda:setattr(arena,'_closed',True),stream=None)
        def writer_init(pool_arg,backend,**kwargs):
            return NS(pool=pool,backend=backend,pending_epochs=(),ledger=NS(pending_count=0,completed_epoch=0),poisoned=False)
        class B:
            append_packed_multirow=lambda *a,**kw:None
            def __init__(self,writer,**kw):
                self.writer=writer;self._orphaned_reads={}
            def drain_failed_read_events(self):pass
        args=NS(vocab_size=100,hidden_size=4,intermediate_size=8,linear_num_key_heads=1,linear_key_head_dim=2,
                linear_num_value_heads=1,linear_value_head_dim=2,num_attention_heads=2,num_key_value_heads=1)
        candidate=NS(args=args,bootstrap_generation=0,native_layer_count=2,layer_map=NS(full_attention=(1,3)),
            native_dtype_preflight=lambda:'bfloat16',bootstrap_staging_bytes=lambda n:0,
            _failure_roots=[],_bootstrap_failure_roots=[])
        cap={'version':2,'min_query_count':9,'max_query_count':1023,'passes':2,'arithmetic':'stock_short_bf16_scores_and_normalized_probabilities','head_dim':256,'storage_dtype':'bfloat16','segmented_causal':True,'global_scratch_bytes':0,'max_causal_end':8192}
        raw=NS(prefill_matrix_capability=lambda:cap,**{k:lambda *a:0 for k in
            ('grouped_multirow_write','grouped_multirow_write_count','grouped_multirow_row_count',
             'prefill_matrix_dispatch_count','q1_stock_reduction_dispatch_count','q1_stock_singleton_dispatch_count')})
        if nax_exact:
            args.num_attention_heads=24;args.num_key_value_heads=4
            raw.prefill_nax_capability=lambda:{'version':1,'storage_dtype':'bfloat16','head_dim':256,'query_heads':24,'kv_heads':4,'min_query_count':9,'max_query_count':129,'max_causal_end':129,'max_spans':2,'origin_zero':True,'window_zero':True,'architecture':'s','architecture':'s','physical_dispatches':3,'scratch_bound_bytes':3195072,'scratch_bytes_per_query_row':24*129*4,'qualified':False}
            for stage in ('score','softmax','value'):setattr(raw,'prefill_nax_'+stage+'_dispatch_count',lambda a:0)
        if long_fused:
            from mlx2.runtime.hybrid_packed_prefill_long import make_long_profile
            args.num_attention_heads=24;args.num_key_value_heads=4
            candidate.forward_scratch_bytes=lambda offsets:1024
            raw.prefill_long_nax_capability=lambda:{'version':1,'storage_dtype':'bfloat16','head_dim':256,'query_heads':24,'kv_heads':4,
                'max_spans':2,'min_query_count':256,'max_query_count':8192,'max_causal_end':8192,'origin_zero':True,'window_zero':True,'architecture':'s',
                'scratch_bytes':0,'physical_dispatches':1,'qualified':False}
            for name in ('prefill_long_nax_dispatch_count','q1_stock_long_partial_dispatch_count','q1_stock_long_reduce_dispatch_count'):setattr(raw,name,lambda a:0)
            if long_cap_missing:del raw.prefill_long_nax_dispatch_count
        mx=NS(default_stream=lambda d:None,gpu=object(),device_info=lambda:{'architecture':'applegpu_g17s','memory_size':host_memory})
        class Profile:
            page_bytes=64
            def __init__(self,*a):pass
        def math(candidate,layers,prompts,reservation,**kwargs):
            for owners,tokens in zip(layers,prompts):
                for owner in owners:owner.offset=len(tokens)
            state['math_done']=True;reservation.release()
            return np.zeros((2,4)),((Cache(),),(Cache(),)),{'selected':True}
        maps={'adapters.qwen35_paged_candidate':NS(Qwen35PagedCandidate=lambda *a,**kw:candidate,
                HybridBootstrap=lambda logits,caches,n,receipt:NS(logits=logits,recurrent_caches=caches,offset=n,receipt=receipt),clone_recurrent_caches=lambda c:c),
            'qwen3_paged_native_backend':NS(NativeQwen3PagedBackend=B),
            '_paged_kv_native':raw,'mlx.core':NS(core=mx),'':NS(qwen35_paged_graph_factory=R),
            'paged_hybrid_research_profile':NS(require_hybrid_native_capabilities=lambda *a:None),
            'paged_kv_pool':NS(PagedKVPool=lambda n:pool),
            'paged_kv_token':NS(PagedKVTokenOwner=TokenOwner,TokenKVProfile=Profile),
            'paged_kv_write':NS(NativeWriteBackend=lambda *a,**kw:arena,PagedKVWriteOwner=writer_init),
            'paged_native_atomic_owner':NS(NativeAtomicRequestOwner=PublicOwner),
            'paged_gdn_checkpoint':NS(GDNBoundaryCheckpoint=GDNBoundaryCheckpoint)}
        def imports(name,globals=None,locals=None,fromlist=(),level=0):
            if name in maps:return maps[name]
            return builtins.__import__(name,globals,locals,fromlist,level)
        tree=ast.parse((ROOT/'src/mlx2/runtime/hybrid_packed_prefill.py').read_text())
        fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='create_cold_packed_hybrid')
        ns=dict(M.__dict__);ns['__builtins__']={**vars(builtins),'__import__':imports};ns['run_packed_math']=math
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_cold_factory','exec'),ns)
        ident=identity();profile={'schema':M.SCHEMA,'profile_id':'test','identity':ident,'storage_dtype':'bfloat16',
            'memory_budget_bytes':1<<30,'required_environment':M.packed_environment(),'qualified':False,
            'price_usable':False,'serving_default':False,'context_lengths':[32,96],'q1_simd_stripes':16,
            'stock_reduction':True,'stock_singleton':True}
        if nax_exact:
            profile['prefill_nax_exact']=True;profile['required_environment']=M.packed_environment(True)
        request=((0,'revision',tuple(range(32)),2),(1,'revision',tuple(range(96)),4))
        if long_fused:
            profile=make_long_profile(ident)
            request=((0,'revision',tuple(i%100 for i in range(6950)),2),(1,'revision',tuple(i%100 for i in range(6929)),4))
        def call():
            with patch.dict(M.os.environ,profile['required_environment']):
                return ns['create_cold_packed_hybrid'](NS(model=object(),identity={'fingerprint':'revision'}),request,
                    profile=profile,live_identity=ident,permit_candidate=True,
                    cancelled=lambda:cancel_after_math and state['math_done'])
        return call,state,R,base_charge,pool
    def test_long_factory_capability_and_host_budget_fail_before_allocation(self):
        for kw in ({'long_cap_missing':True},{'host_memory':48<<30}):
            call,state,R,charge,pool=self.factory_fixture(long_fused=True,**kw)
            with self.subTest(kw=kw),self.assertRaises((ValueError,MemoryError)):call()
            self.assertEqual(state['constructed'],0);self.assertFalse(state['math_done'])
            self.assertEqual(R._CHARGED,charge);self.assertEqual(pool.allocated_count,0)
    def test_long_factory_whole_offsets_charge_and_failure_retirement(self):
        call,state,R,charge,pool=self.factory_fixture(long_fused=True)
        owners,candidate,boots=call()
        self.assertEqual([b.offset for b in boots],[6950,6929]);self.assertTrue(candidate._prefill_long_nax)
        self.assertFalse(candidate._serving_stock_reduction);self.assertFalse(candidate._serving_stock_singleton)
        self.assertGreater(candidate._serving_resources.charge,1024)
        candidate._serving_resources.abort();self.assertEqual(R._CHARGED,charge);self.assertEqual(pool.allocated_count,0)
        call,state,R,charge,pool=self.factory_fixture(long_fused=True,fail_second=True)
        with self.assertRaisesRegex(RuntimeError,'second owner'):call()
        self.assertEqual(R._CHARGED,charge);self.assertEqual(pool.allocated_count,0)

    def test_real_factory_second_owner_failure_retires_both_lanes_and_charge(self):
        call,state,R,charge,pool=self.factory_fixture(fail_second=True)
        with self.assertRaisesRegex(RuntimeError,'second owner'):call()
        self.assertEqual(pool.allocated_count,0);self.assertEqual(state['closed'],4)
        self.assertEqual(R._CHARGED,charge)
    def test_real_factory_cancel_after_math_never_constructs_owner(self):
        call,state,R,charge,pool=self.factory_fixture(cancel_after_math=True)
        with self.assertRaisesRegex(ValueError,'before joint owner'):call()
        self.assertEqual(state['constructed'],0);self.assertEqual(pool.allocated_count,0)
        self.assertEqual(R._CHARGED,charge)
    def block_fixture(self, block=4, fail_eval=False, fail_terminal=False):
        c,o,rt,a,g=setup();c._prefill_nax_exact=True;c._prefill_eval_block_size=block
        c.args.num_attention_heads=24;c.args.num_key_value_heads=4
        a.num_attention_heads=24;a.num_key_value_heads=4
        a.q_proj=Proj(24*512);a.k_proj=Proj(4*256);a.v_proj=Proj(4*256)
        base=c.trunk.layers;c.trunk.layers=tuple(copy.copy(x) for _ in range(4) for x in base)
        c.native_layer_count=4;c.layer_map=NS(recurrent=(0,2,4,6),full_attention=(1,3,5,7))
        c.language_model.make_cache=lambda:[v for _ in range(4) for v in (Cache(),None)]
        owners=tuple(tuple(NS(offset=0,_pending=(),_stage=None) for _ in range(4)) for _ in range(2))
        submitted=[];drains=[];evaluated=[];epochs=iter(range(10,20))
        def write(own,k,v,counts,permit_candidate):
            c.backend.writes+=1;c.backend.rows+=sum(counts);submitted.append((own,counts))
            return (NS(dependency=object(),packed_multirow=True),)
        def drain(own,uses):
            drains.append((tuple(own),tuple(uses)))
            if fail_terminal:raise RuntimeError('injected batched terminal ambiguity')
            for row,counts in submitted:
                for owner,n in zip(row,counts):owner.offset=n
            for use in uses:use.state='closed'
            return tuple(NS() for _ in uses)
        def evaluate(*arrays):
            evaluated.append(arrays)
            if fail_eval:raise RuntimeError('injected block evaluation ambiguity')
        c.backend.append_packed_multirow=write;c.backend.drain_staged=drain;rt.mx.eval=evaluate
        rt.prepare_read=lambda *args,**kw:NS(state='prepared',lease=NS(epoch=next(epochs)))
        for stage in ('score','softmax','value'):setattr(rt.native,'prefill_nax_'+stage+'_dispatch_count',lambda backend:backend.reads)
        return c,owners,rt,submitted,drains,evaluated

    def test_block_option_is_strict_and_requires_explicit_nax_profile(self):
        for bad in (True,'4',0,2,32,None):
            with self.assertRaises(ValueError):M.evaluation_block_size(bad)
        ident=identity()
        p={'schema':M.SCHEMA,'profile_id':'blocked','identity':ident,'storage_dtype':'bfloat16',
           'memory_budget_bytes':8<<30,'required_environment':M.packed_environment(True,4),
           'qualified':False,'price_usable':False,'serving_default':False,'context_lengths':[32,96],
           'q1_simd_stripes':16,'stock_reduction':True,'stock_singleton':True,
           'prefill_nax_exact':True,'prefill_eval_block_size':4}
        self.assertIs(M.validate_profile(p,live_identity=ident,counts=(32,96),environment=M.packed_environment(True,4)),p)
        p['prefill_nax_exact']=False
        with self.assertRaisesRegex(ValueError,'requires explicit NAX'):M.validate_profile(p,live_identity=ident,counts=(32,96),environment=M.packed_environment(True,4))

    def test_four_layer_batch_has_one_terminal_boundary_and_precharged_overlap(self):
        c,o,rt,submitted,drains,evals=self.block_fixture();r=Reservation();r.bytes=1<<31
        logits,states,proof=M.run_packed_math(c,o,(tuple(range(32)),tuple(range(96))),r,runtime=rt,cancelled=lambda:False)
        self.assertEqual(len(drains),1);self.assertEqual(len(drains[0][0]),8);self.assertEqual(len(drains[0][1]),4)
        self.assertEqual(proof['explicit_eval_calls'],2);self.assertEqual(proof['terminal_drain_batches'],1)
        self.assertEqual(proof['maximum_simultaneous_reads'],4)
        self.assertEqual(proof['native_reader_simultaneous_scratch_bytes'],4*128*24*129*4)
        self.assertTrue(r.completed);self.assertEqual(len(evals),3) # model/block, final logits, batched finite flags.
        eager=M.scratch_bound(c,(32,96),nax_exact=True,eval_block_size=1)
        batch=M.scratch_bound(c,(32,96),nax_exact=True,eval_block_size=4)
        self.assertGreater(batch,eager+3*128*24*129*4)
        self.assertTrue(all(v==32 for v in [x.offset for x in o[0]]));self.assertTrue(all(x.offset==96 for x in o[1]))

    def test_eager_reference_keeps_per_layer_materialization_and_drains(self):
        c,o,rt,submitted,drains,evals=self.block_fixture(block=1);r=Reservation();r.bytes=1<<31
        _,_,proof=M.run_packed_math(c,o,(tuple(range(32)),tuple(range(96))),r,runtime=rt,cancelled=lambda:False)
        self.assertEqual(len(drains),4);self.assertEqual(proof['explicit_eval_calls'],9)
        self.assertEqual(proof['maximum_simultaneous_reads'],1)
        self.assertEqual(proof['host_evaluation_policy'],'eager_per_fa')

    def test_block_eval_and_terminal_failures_retain_every_submitted_root(self):
        for eval_failure,terminal_failure in ((True,False),(False,True)):
            c,o,rt,submitted,drains,evals=self.block_fixture(fail_eval=eval_failure,fail_terminal=terminal_failure)
            r=Reservation();r.bytes=1<<31
            with self.assertRaisesRegex(RuntimeError,'ambiguity'):
                M.run_packed_math(c,o,(tuple(range(32)),tuple(range(96))),r,runtime=rt,cancelled=lambda:False)
            self.assertFalse(r.completed);self.assertEqual(len(c.backend._orphaned_reads),4)
            self.assertEqual(len(r.roots[-1]),8) # QKV/tickets/owners + attention for eachpendingFA.
            self.assertTrue(all(x.offset==0 for row in o for x in row))

    def test_final_finite_failure_retains_completed_state_without_public_handoff(self):
        for bad_predicate in (0,1,16):
            with self.subTest(bad_predicate=bad_predicate):
                c,o,rt,submitted,drains,evals=self.block_fixture();r=Reservation();r.bytes=1<<31
                predicates=[]
                def isfinite(value):
                    index=len(predicates);predicates.append(value)
                    return np.zeros_like(value,dtype=bool) if index==bad_predicate else np.isfinite(value)
                rt.mx.isfinite=isfinite
                with self.assertRaisesRegex(ValueError,'nonfinite'):
                    M.run_packed_math(c,o,(tuple(range(32)),tuple(range(96))),r,runtime=rt,cancelled=lambda:False)
                self.assertEqual(len(predicates),17);self.assertEqual(len(evals),3)
                self.assertFalse(r.completed);self.assertTrue(r.roots)
                self.assertEqual(len(drains),1) # matched known-success terminal proof is retained.
                self.assertTrue(all(x.offset==n for row,n in zip(o,(32,96)) for x in row))

    def test_cancel_inside_pending_block_keeps_three_reads_and_no_handoff(self):
        c,o,rt,submitted,drains,evals=self.block_fixture();r=Reservation();r.bytes=1<<31
        calls=iter((False,)*6+(True,))
        with self.assertRaisesRegex(ValueError,'cancelled'):
            M.run_packed_math(c,o,(tuple(range(32)),tuple(range(96))),r,runtime=rt,cancelled=lambda:next(calls))
        self.assertEqual(len(submitted),3);self.assertEqual(len(drains),0);self.assertEqual(len(evals),0)
        self.assertFalse(r.completed);self.assertEqual(len(c.backend._orphaned_reads),3)
        self.assertEqual(len(r.roots[-1]),6)

    def test_exact_nax_factory_charges_and_keeps_source_arm_explicit(self):
        call,state,R,charge,pool=self.factory_fixture(nax_exact=True)
        owners,candidate,boots=call()
        self.assertTrue(candidate._prefill_nax_exact)
        expected=M.scratch_bound(candidate,(32,96),nax_exact=True)
        self.assertEqual(candidate._serving_resources.transient_bound,expected)
        self.assertEqual(expected-M.scratch_bound(candidate,(32,96)),128*24*129*4)
        self.assertEqual(R._CHARGED-charge,candidate._serving_resources.charge)
        for owner in owners:owner.close()
        self.assertTrue(candidate._serving_resources.reap());self.assertEqual(R._CHARGED,charge)

    def test_exact_nax_boolean_environment_and_bounds_fail_closed(self):
        for bad in (1,'1',None):
            with self.assertRaises(ValueError):M.packed_environment(bad)
        for counts in ((8,96),(32,130),(32,96,32),(True,96)):
            with self.assertRaises(ValueError):M.nax_scratch_bytes(counts)
        self.assertEqual(M.nax_scratch_bytes((129,129)),3195072)
        self.assertEqual(M.packed_environment()['MLX2_PAGED_PREFILL_NAX_EXACT'],'0')
        self.assertEqual(M.packed_environment(True)['MLX2_PAGED_PREFILL_NAX_EXACT'],'1')

    def test_exact_nax_math_proves_three_stages_and_retains_on_partial_counter(self):
        for broken in (False,True):
            c,o,rt,a,g=setup();c._prefill_nax_exact=True
            c.args.num_attention_heads=24;c.args.num_key_value_heads=4
            a.num_attention_heads=24;a.num_key_value_heads=4
            a.q_proj=Proj(24*512);a.k_proj=Proj(4*256);a.v_proj=Proj(4*256)
            def write(owners,k,v,counts,permit_candidate):
                c.backend.writes+=1;c.backend.rows+=sum(counts);c.backend.counts=counts
                self.assertEqual(k.shape,(128,4,256));self.assertEqual(v.shape,k.shape)
                return (NS(dependency=1,packed_multirow=True),)
            c.backend.append_packed_multirow=write
            for stage in ('score','softmax','value'):
                setattr(rt.native,'prefill_nax_'+stage+'_dispatch_count',lambda backend:backend.reads)
            if broken:rt.native.prefill_nax_value_dispatch_count=lambda backend:0
            r=Reservation();ids=(tuple(range(32)),tuple(range(96)))
            if broken:
                with self.assertRaisesRegex(RuntimeError,'physical'):M.run_packed_math(c,o,ids,r,runtime=rt,cancelled=lambda:False)
                self.assertFalse(r.completed);self.assertTrue(r.roots)
            else:
                logits,caches,proof=M.run_packed_math(c,o,ids,r,runtime=rt,cancelled=lambda:False)
                self.assertTrue(r.completed);self.assertTrue(proof['prefill_nax_exact'])
                self.assertEqual(proof['native_reader_scratch_bytes'],128*24*129*4)
                for stage in ('score','softmax','value'):self.assertEqual(proof['physical_counters']['prefill_nax_'+stage+'_dispatch_count'],1)

    def test_old_zero_scratch_raw_abi_works_but_explicit_nax_refuses(self):
        cap={'version':2,'head_dim':256,'storage_dtype':'bfloat16','segmented_causal':True,
             'passes':2,'arithmetic':'stock_short_bf16_scores_and_normalized_probabilities',
             'min_query_count':9,'max_query_count':1023,'max_causal_end':8192,'global_scratch_bytes':0}
        raw=NS(prefill_matrix_capability=lambda:cap,**{k:lambda *a:0 for k in
            ('grouped_multirow_write','grouped_multirow_write_count','grouped_multirow_row_count','prefill_matrix_dispatch_count')})
        self.assertIs(M.require_capabilities(raw,Backend,(32,96)),cap)
        with self.assertRaisesRegex(ValueError,'NAX native capability missing'):
            M.require_capabilities(raw,Backend,(32,96),nax_exact=True)

    def test_exact_nax_reservation_failure_precedes_all_math(self):
        c,o,rt,a,g=setup();c._prefill_nax_exact=True;r=Reservation()
        r.bytes=M.scratch_bound(c,(32,96),nax_exact=True)-1
        with self.assertRaises(MemoryError):M.run_packed_math(c,o,(tuple(range(32)),tuple(range(96))),r,runtime=rt,cancelled=lambda:False)
        self.assertEqual(c.backend.writes,0);self.assertEqual(g.calls,0)

    def test_exact_nax_attribution_requires_all_three_live_stages_and_charge(self):
        c=self.packed_candidate();p=c._packed_prefill_receipt
        p.update(prefill_nax_exact=True,native_reader_scratch_bytes=128*24*129*4,
                 native_reader_scratch_bound_bytes=3195072,prefill_attention_arithmetic='nax_three_stage_stock_short')
        for stage in ('score','softmax','value'):
            name='prefill_nax_'+stage+'_dispatch_count';p['physical_counters'][name]=16
            setattr(c.backend.writer.backend._native,name,lambda arena:16)
        self.assertEqual(bootstrap_prefill_attribution(c)['prefill_mode'],'native_packed_prefill')
        c.backend.writer.backend._native.prefill_nax_value_dispatch_count=lambda arena:15
        with self.assertRaises(ValueError):bootstrap_prefill_attribution(c)
        c.backend.writer.backend._native.prefill_nax_value_dispatch_count=lambda arena:16
        p['native_reader_scratch_bytes']=0
        with self.assertRaises(ValueError):bootstrap_prefill_attribution(c)

    def test_unattached_arena_close_failure_retains_charge_then_reaps(self):
        from mlx2.runtime import qwen35_paged_graph_factory as R
        charge=R._CHARGED;resources=R.HybridServingResources(4096,2048,8192)
        arena=NS(_closed=False)
        def fail():raise RuntimeError('ambiguous arena close')
        arena.close_after_terminal=fail;resources._unattached_arena=arena
        with self.assertRaisesRegex(RuntimeError,'ambiguous'):resources.reap()
        self.assertEqual(R._CHARGED-charge,4096);self.assertFalse(resources.closed)
        arena.close_after_terminal=lambda:setattr(arena,'_closed',True)
        self.assertTrue(resources.reap());self.assertEqual(R._CHARGED,charge)

    def test_factory_proof_precedes_initial_owner_constructor(self):
        tree=ast.parse((ROOT/'src/mlx2/runtime/hybrid_packed_prefill.py').read_text())
        fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='create_cold_packed_hybrid')
        calls={}
        for n in ast.walk(fn):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):calls.setdefault(n.func.id,[]).append(n.lineno)
        self.assertLess(min(calls['require_capabilities']),min(calls['NativeWriteBackend']))
        self.assertLess(min(calls['run_packed_math']),min(calls['NativeAtomicRequestOwner']))

class N20StagewiseTests(unittest.TestCase):
    def selected(self):
        c,o,rt,a,g=setup()
        c._prefill_packed_n20=c._prefill_layer_lifetime=c._prefill_long_nax=True
        c._prefill_nax_exact=False;c._prefill_generation_caps=(2,4)
        c.backend.append_packed_multirow_n20=c.backend.append_packed_multirow
        c.backend.prepare_read_n20=rt.prepare_read
        rt.native.grouped_n20_write_count=lambda arena:arena.writes
        rt.native.grouped_n20_row_count=lambda arena:arena.rows
        rt.native.prefill_long_n20_dispatch_count=lambda arena:arena.reads
        def mixed(x,parts,*,materialize):
            result=g.mixed(x,parts);materialize('gdn_fake',result);return result
        g.mixed_materialized=mixed
        for layer in c.trunk.layers:
            ordinary=layer.mlp
            def materialized(x,*,materialize,ordinary=ordinary):
                result=ordinary(x);materialize('mlp_fake',result);return result
            layer.mlp=NS(materialized=materialized)
        r=Reservation();r.events=[]
        def phase(event,*,writer,orphaned_reads,callback):
            self.assertFalse(writer.pending_epochs);self.assertEqual(writer.ledger.pending_count,0)
            self.assertFalse(orphaned_reads);self.assertFalse(event['public_state_published'])
            r.events.append(event)
            if callback:callback(event)
        r.evaluated_phase=phase
        return c,o,rt,r
    def run_selected(self,c,o,rt,r,**kw):
        # Separate real-geometry byte tests cover admission. This tiny fixture
        # executes the actual N control path without allocating model tensors.
        with patch.object(M,'n20_stagewise_charge_components',return_value={'arena_bytes':0,'decode_scratch_bytes':0,'layer_activation_bytes':1<<28}):
            return M.run_packed_math(c,o,((1,2),(3,4,5)),r,runtime=rt,cancelled=kw.get('cancelled',lambda:False))
    def test_n_actual_path_preserves_ordinary_math_and_terminal_counts(self):
        c,o,rt,a,g=setup();reference=M.run_packed_math(c,o,((1,2),(3,4,5)),Reservation(),runtime=rt,cancelled=lambda:False)
        c,o,rt,r=self.selected();logits,caches,receipt=self.run_selected(c,o,rt,r)
        np.testing.assert_array_equal(logits,reference[0])
        for a,b in zip(caches,reference[1]):
            for x,y in zip(a[0].cache,b[0].cache):np.testing.assert_array_equal(x,y)
        self.assertEqual(receipt['physical_counters'],{'grouped_n20_write_count':1,'grouped_n20_row_count':5,'prefill_long_n20_dispatch_count':1})
        self.assertEqual(receipt['evaluated_phase_boundaries'],2);self.assertEqual(receipt['maximum_simultaneous_reads'],1)
        self.assertTrue(r.completed);self.assertEqual(c.backend._orphaned_reads,{})
    def test_n_cancel_at_first_phase_retains_private_state_and_no_writer(self):
        c,o,rt,r=self.selected();cancelled=[False]
        c._prefill_phase_boundary=lambda event:cancelled.__setitem__(0,True)
        with self.assertRaisesRegex(ValueError,'phase boundary'):self.run_selected(c,o,rt,r,cancelled=lambda:cancelled[0])
        self.assertFalse(r.completed);self.assertTrue(r.roots);self.assertEqual(c.backend.writes,0)
        self.assertEqual(len(r.events),1)
    def test_n_terminal_failure_retains_native_roots_and_exact_read_lease(self):
        c,o,rt,r=self.selected();c.backend.fail_drain=True
        with self.assertRaisesRegex(RuntimeError,'terminal'):self.run_selected(c,o,rt,r)
        self.assertFalse(r.completed);self.assertTrue(r.roots[-1]);self.assertEqual(tuple(c.backend._orphaned_reads),(9,))
        self.assertEqual(len(r.events),1)
    def test_n_eval_failure_retains_materialization_roots_before_native_submit(self):
        c,o,rt,r=self.selected()
        def fail(*v):raise RuntimeError('GPU completion failure fixture')
        rt.mx.eval=fail
        with self.assertRaisesRegex(RuntimeError,'completion failure'):self.run_selected(c,o,rt,r)
        self.assertFalse(r.completed);self.assertTrue(r.roots[-2]);self.assertEqual(c.backend.writes,0)
    def test_n_missing_host_capability_or_mismatched_lifetime_refuses_before_math(self):
        c,o,rt,r=self.selected();c.backend.prepare_read_n20=None
        with self.assertRaisesRegex(ValueError,'capability'):self.run_selected(c,o,rt,r)
        self.assertEqual(c.backend.writes,0)
        c,o,rt,r=self.selected();c._prefill_layer_lifetime=False
        with self.assertRaisesRegex(ValueError,'lifetime'):self.run_selected(c,o,rt,r)

if __name__=='__main__':unittest.main()
