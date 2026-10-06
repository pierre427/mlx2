"""Long-profile/capability/accounting/proof contracts without MLX/native imports."""
import copy,importlib.abc,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1]
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native':raise RuntimeError('runtime forbidden')
sys.meta_path.insert(0,Guard());sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts/research'))
from mlx2.runtime import hybrid_packed_prefill as S,hybrid_packed_prefill_long as L,paged_packed_prefill_serving_profile as P,qwen35_paged_graph_factory as R
from mlx2.runtime.paged_pack_price import IDENTITY_FIELDS
from varlen_packed_long_contract import prompt_ids,physical_prefill,validate_q1
from varlen_hybrid_packed_prefill_long_http_gate import validate_evaluation_proof,summarize_http
I={k:'a'*64 for k in IDENTITY_FIELDS};I.update(host='cpu',hardware='cpu',source_commit='b'*40,mlx_wheel_version='pinned')
class Tests(unittest.TestCase):
    def load(self,p,counts=L.COUNTS,env=None):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'p.json';path.write_text(json.dumps(p))
            return P.load_profile(path,live_identity=I,context_lengths=counts,environment=env or p['required_environment'])
    def test_long_serving_factory_translation_is_separate_exact_domain(self):
        p=P.make_profile(I,long_fused=True);self.assertEqual(p['memory_budget_bytes'],40<<30)
        for counts in (L.COUNTS,tuple(reversed(L.COUNTS))):
            factory=P.factory_profile(self.load(p,counts),counts)
            L.validate_long_profile(factory,live_identity=I,counts=counts,environment=L.long_environment())
        for counts in ((32,96),(6950,6928),(8192,8191)):
            with self.assertRaises(ValueError):self.load(p,counts)
        with self.assertRaises(ValueError):S.validate_profile(P.factory_profile(p,L.COUNTS),live_identity=I,counts=L.COUNTS,environment=L.long_environment())
    def test_actual_generated_model_profile_loads_production_long_schema(self):
        from varlen_hybrid_packed_prefill_long_geometry_gate import make_profile
        generated=make_profile(I)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model-profile.json';path.write_text(json.dumps(generated))
            loaded=S.load_profile(path,live_identity=I,counts=L.COUNTS,environment=L.long_environment())
            self.assertEqual(loaded,generated)
            with self.assertRaises(ValueError):S.load_profile(path,live_identity=I,counts=(32,96),environment=L.long_environment())
            env=dict(L.long_environment());env['MLX2_PAGED_PREFILL_NAX_LONG_FUSED']='0'
            with self.assertRaises(ValueError):S.load_profile(path,live_identity=I,counts=L.COUNTS,environment=env)

    def test_long_source_env_budget_and_block_fail_closed(self):
        p=P.make_profile(I,long_fused=True)
        for key,value in (('qualified',True),('prefill_nax_exact',True),('memory_budget_bytes',12<<30),('prefill_eval_block_size',16)):
            bad=copy.deepcopy(p);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.load(bad)
        for key in ('MLX2_PAGED_PREFILL_NAX_LONG_FUSED','MLX2_PAGED_Q1_STOCK_LONG'):
            env=dict(L.long_environment());env[key]='0'
            with self.assertRaises(ValueError):self.load(p,env=env)
        bad=copy.deepcopy(p);bad['identity']['source_commit']='c'*40
        with self.assertRaises(ValueError):self.load(bad)
        with self.assertRaises(ValueError):P.make_profile(I,long_fused=True,prefill_eval_block_size=4)
    def test_long_capability_rejects_missing_abi_and_nonzero_score_scratch(self):
        cap={'version':1,'storage_dtype':'bfloat16','head_dim':256,'query_heads':24,'kv_heads':4,'max_spans':2,
            'min_query_count':256,'max_query_count':8192,'max_causal_end':8192,'origin_zero':True,'window_zero':True,'architecture':'s',
            'scratch_bytes':0,'physical_dispatches':1,'qualified':False}
        names=('grouped_multirow_write','grouped_multirow_write_count','grouped_multirow_row_count','prefill_matrix_dispatch_count',
            'prefill_long_nax_dispatch_count','q1_stock_long_partial_dispatch_count','q1_stock_long_reduce_dispatch_count')
        raw=NS(prefill_long_nax_capability=lambda:cap,**{n:lambda a:0 for n in names});backend=NS(append_packed_multirow=lambda:None)
        L.require_long_capabilities(raw,backend,L.COUNTS)
        for name in names:
            value=getattr(raw,name);delattr(raw,name)
            with self.assertRaises(ValueError):L.require_long_capabilities(raw,backend,L.COUNTS)
            setattr(raw,name,value)
        cap['scratch_bytes']=1
        with self.assertRaises(ValueError):L.require_long_capabilities(raw,backend,L.COUNTS)
    def test_whole_projection_charge_is_not_hidden_in_short_budget(self):
        config=json.loads(Path('~/mlx-models/Qwen3.8-27B-MLX-4bit/config.json').read_text())['text_config']
        candidate=NS(args=NS(**config),native_layer_count=16,bootstrap_staging_bytes=lambda n:0)
        charge=S.scratch_bound(candidate,L.COUNTS,long_fused=True)
        self.assertGreater(charge,24<<30);self.assertLess(charge,40<<30)
        with self.assertRaises(ValueError):S.scratch_bound(candidate,L.COUNTS)
        L.require_host_budget(P.make_profile(I,long_fused=True),charge,{'memory_size':128<<30})
        for info in ({},{'memory_size':48<<30},{'memory_size':True}):
            with self.assertRaises(MemoryError):L.require_host_budget(P.make_profile(I,long_fused=True),charge,info)
    def test_long_global_charge_is_explicit_and_short_default_is_preserved(self):
        start=R._CHARGED
        try:
            with self.assertRaises(MemoryError):R.HybridServingResources(20<<30,1,40<<30)
            resources=R.HybridServingResources(20<<30,1,40<<30,research_limit_bytes=40<<30)
            with self.assertRaises(MemoryError):R.HybridServingResources(1,1,12<<30)
            resources.abort();self.assertEqual(R._CHARGED,start)
        finally:
            if R._CHARGED!=start:resources.abort()
    def test_actual_long_plan_accepts_whole_rows_preserves_page_bounds(self):
        from mlx2.runtime.paged_attention_plan import PagedAttentionPlan,SequenceSpan,PageHandle
        counts=L.COUNTS;spans=[];table=[];begin=0
        for count in counts:
            pages=(count+63)//64;offset=len(table)
            table.extend(PageHandle(i,1) for i in range(offset,offset+pages))
            spans.append(SequenceSpan(begin,count,0,count,0,0,offset,pages,1));begin+=count
        args=dict(spans=tuple(spans),page_table=tuple(table),total_rows=begin,query_heads=24,kv_heads=4,head_dim=256,
            dtype='bfloat16',pool_capacity=len(table),live_generations={h.page_id:1 for h in table})
        with self.assertRaisesRegex(ValueError,'capacity'):PagedAttentionPlan(**args)
        plan=PagedAttentionPlan(**args,profile='prefill_long_nax_v1',max_work_items=2*8192*24,max_scratch_bytes=0)
        self.assertEqual(plan.spans[0].visible_bounds(6949),(0,6950));self.assertEqual(plan.spans[1].visible_bounds(6928),(0,6929))
        bad=dict(args);bad['page_table']=bad['page_table'][:-1]
        with self.assertRaises(ValueError):PagedAttentionPlan(**bad,profile='prefill_long_nax_v1',max_work_items=2*8192*24,max_scratch_bytes=0)
        for key,value in (('dtype','float16'),('query_heads',12),('kv_heads',8)):
            bad={**args,key:value}
            with self.assertRaises(ValueError):PagedAttentionPlan(**bad,profile='prefill_long_nax_v1',max_work_items=2*8192*24,max_scratch_bytes=0)

    def test_pinned_whole_prompts_and_zero_score_scratch_proof(self):
        self.assertEqual(tuple(map(len,prompt_ids())),L.COUNTS)
        self.assertEqual(physical_prefill()['grouped_multirow_row_count'],222064)
        proof={'prefill_eval_block_size':1,'native_reader_scratch_bytes':0,'native_reader_simultaneous_scratch_bytes':0,
            'bootstrap_charge_components':{'native_reader_bytes':0}}
        validate_evaluation_proof(proof,1)
        bad=copy.deepcopy(proof);bad['native_reader_scratch_bytes']=1
        with self.assertRaises(RuntimeError):validate_evaluation_proof(bad,1)
    def test_actual_http_summary_requires_long_counter_receipt_and_unqualified_label(self):
        proof={'stock_long_selected':True,'native_tile_dispatches':0,'native_stock_reduction_dispatches':0,
            'native_split_partial_dispatches':0,'native_stock_long_partial_dispatches':16,
            'native_stock_long_reduce_dispatches':16,'grouped_q1_writes':16}
        for cap in (2,4):
            if cap==4:proof.update(native_stock_long_partial_dispatches=0,native_stock_long_reduce_dispatches=0,
                grouped_q1_writes=0,scalar_native_writes=64,expected_scalar_write_spans=64)
            receipt={'route':'native_hybrid_paged_b2','selected':True,'observed_used':True,'qualified':False,'price_usable':False,
                'prefill_mode':'native_packed_prefill','prefill_layout':'real_rows','native_prefill_observed_used':True,
                'stock_reduction_selected':False,'state_planes':['kv','gdn'],'output_token_ids':list(range(cap)),
                'native_prefill_attention_calls':16,'native_prefill_proof':{'prefill_long_nax':True,'prefill_nax_exact':False,
                    'serving_numerical_reference':'same_geometry_ordinary_mixed','physical_counters':physical_prefill()},
                'hybrid_graph_proof':dict(proof)}
            body={'id':'cpu','mlx2':{'qualification':'unqualified','route_receipt':receipt},
                'usage':{'completion_tokens':cap},'choices':[{'text':'cpu','finish_reason':'length'}]}
            summarize_http(body,cap,True)
            receipt['native_prefill_proof']['physical_counters']['prefill_long_nax_dispatch_count']=0
            with self.assertRaises(RuntimeError):summarize_http(body,cap,True)

    def test_q1_long_and_scalar_survivor_proof_distinguish_physical_routes(self):
        p={'stock_long_selected':True,'native_tile_dispatches':0,'native_stock_reduction_dispatches':0,
            'native_split_partial_dispatches':0,'native_stock_long_partial_dispatches':16,
            'native_stock_long_reduce_dispatches':16,'grouped_q1_writes':16}
        validate_q1(p,2)
        p.update(native_stock_long_partial_dispatches=0,native_stock_long_reduce_dispatches=0,grouped_q1_writes=0,
            scalar_native_writes=64,expected_scalar_write_spans=64)
        validate_q1(p,1)
        p['scalar_native_writes']=63
        with self.assertRaises(RuntimeError):validate_q1(p,1)
if __name__=='__main__':unittest.main()
