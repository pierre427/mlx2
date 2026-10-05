"""CPU fail-closed final packed HTTP receipt and physical proof checks."""
import argparse,ast,copy,importlib.abc,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native':raise RuntimeError('GPU import forbidden')
sys.meta_path.insert(0,Guard());sys.path.insert(0,str(ROOT/'scripts/research'))
import varlen_hybrid_packed_prefill_http_gate as G
class Tests(unittest.TestCase):
    def body(self,cap):
        physical={'grouped_multirow_write_count':16,'grouped_multirow_row_count':2048,'prefill_matrix_dispatch_count':16,
            **{'prefill_nax_'+stage+'_dispatch_count':16 for stage in ('score','softmax','value')}}
        return {'id':'cpu','usage':{'completion_tokens':cap},'choices':[{'text':'cpu','finish_reason':'length'}],
            'mlx2':{'qualification':'unqualified','route_receipt':{'route':'native_hybrid_paged_b2','selected':True,
            'observed_used':True,'qualified':False,'price_usable':False,'prefill_mode':'native_packed_prefill',
            'prefill_layout':'real_rows','native_prefill_observed_used':True,'native_prefill_attention_calls':16,
            'native_prefill_proof':{'prefill_nax_exact':True,'serving_numerical_reference':'same_geometry_ordinary_mixed','physical_counters':physical},
            'stock_reduction_selected':True,'stock_singleton_selected':True,'stock_singleton_observed_used':cap==4,
            'state_planes':['kv','gdn'],'output_token_ids':list(range(cap)),'q1_simd_stripes':32,
            'hybrid_graph_proof':{'native_stock_singleton_dispatches':16 if cap==4 else 0}}}}
    def test_b2_and_survivor_actual_prefill_and_singleton_proof(self):
        for cap in (2,4):self.assertEqual(G.summarize_http(self.body(cap),cap,True,True)['usage']['completion_tokens'],cap)
    def test_refuses_import_mislabel_missing_stage_and_promoted_qualification(self):
        for field,value in (('prefill_mode','ordinary_completed_import'),('native_prefill_attention_calls',0),('qualified',True)):
            body=self.body(4);body['mlx2']['route_receipt'][field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):G.summarize_http(body,4,True,True)
        body=self.body(4);body['mlx2']['route_receipt']['native_prefill_proof']['physical_counters']['prefill_nax_value_dispatch_count']=0
        with self.assertRaises(RuntimeError):G.summarize_http(body,4,True,True)
    def test_real_http_cli_defaults_and_explicit_choices(self):
        main=next(n for n in ast.parse(Path(G.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
        nodes=[]
        for n in main.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in n.targets):break
            nodes.append(n)
        env={'argparse':argparse,'Path':Path,'__doc__':'CPU CLI','MODEL':'model','MANIFEST':'manifest','WHEEL':'wheel','NATIVE':'native','NATIVE_SHA':'sha'}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'real HTTP parser','exec'),env)
        parser=env['parser']
        self.assertEqual(parser.parse_args(['--output','/tmp/cpu.json']).prefill_eval_block_size,1)
        for block in (1,4,16):
            self.assertEqual(parser.parse_args(['--output','/tmp/cpu.json','--prefill-eval-block-size',str(block)]).prefill_eval_block_size,block)
        with self.assertRaises(SystemExit):parser.parse_args(['--output','/tmp/cpu.json','--prefill-eval-block-size','2'])

    def test_selected_block_requires_actual_retained_reader_charge(self):
        for block in (1,4,16):
            proof={'prefill_eval_block_size':block,'native_reader_simultaneous_scratch_bytes':1585152*block,
                'bootstrap_charge_components':{'native_reader_bytes':1585152*block}}
            G.validate_evaluation_proof(proof,block)
            for key in ('prefill_eval_block_size','native_reader_simultaneous_scratch_bytes','bootstrap_charge_components'):
                bad=copy.deepcopy(proof);bad.pop(key)
                with self.subTest(block=block,key=key),self.assertRaises(RuntimeError):G.validate_evaluation_proof(bad,block)
            with self.assertRaises(RuntimeError):G.validate_evaluation_proof(proof,2)

    def test_q1_total_requires_actual_singleton_stock32(self):
        snapshot={'q1_stock_reduction_dispatches':48,'q1_stock_singleton_dispatches':32,'q1_stripe_dispatches_32':48,'q1_stripe_dispatches_16':0,'q1_tile_dispatches':48}
        G.validate_physical(snapshot,16,True)
        for key in snapshot:
            broken=copy.deepcopy(snapshot);broken[key]+=1
            with self.subTest(key=key),self.assertRaises(RuntimeError):G.validate_physical(broken,16,True)
if __name__=='__main__':unittest.main()
