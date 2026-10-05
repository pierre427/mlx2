"""Source-only controlled primitive harness checks, no MLX/native imports."""
import importlib.abc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import zipfile
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native':raise RuntimeError('runtime forbidden')
sys.meta_path.insert(0,Guard())
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('harness',ROOT/'scripts/research/varlen_n20_attention_controlled_perf.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

def document():
    domains=['domain'+str(i) for i in range(20)]
    rows=[{'domain':d,'case_id':d+':'+str(i),'prompt_tokens':256+i,'prompt_token_ids':[0]*(256+i)} for d in domains for i in range(20)]
    return {'schema':'mlx2.spomin-400case-native-inputs.v1','case_count':400,'cases_per_domain':20,'client_concurrency':20,
            'domain_order':domains,'rows':rows,'inputs_sha256':'b'*64}

def retired_result():
    return {'status':'passed','warmup':False,'all_pages_retired':True,'index':1,'timings':dict(
        native_host_plan_seconds=.01,native_bind_seconds=.02,native_eval_seconds=.03,native_terminal_drain_seconds=.04,
        stock_graph_bind_seconds=.02,stock_eval_seconds=.08)}

class Tests(unittest.TestCase):
    def test_actual_domain20_geometry_and_token_count_binding(self):
        d=document();plan=M.geometry(d,'domain0');self.assertEqual(len(plan['counts']),20)
        self.assertEqual(plan['native_arena_kv_plane_bytes'],2*plan['capacity_pages']*4*64*256*2)
        self.assertEqual(plan['score_scratch_bytes'],0)
        d['rows'][0]['prompt_tokens']+=1
        with self.assertRaises(ValueError):M.geometry(d,'domain0')
    def test_missing_duplicate_or_out_of_scope_domains_refused(self):
        d=document();d['rows'][0]['case_id']=d['rows'][1]['case_id']
        with self.assertRaises(ValueError):M.geometry(d,'domain0')
        with self.assertRaises(ValueError):M.geometry(document(),'unknown')
        d=document();d['client_concurrency']=2
        with self.assertRaises(ValueError):M.geometry(d,'domain0')
    def test_clean_exact_source_and_dirty_or_head_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);subprocess.run(['git','init','-q',str(root)],check=True)
            (root/'x').write_text('pinned');subprocess.run(['git','add','x'],cwd=root,check=True)
            subprocess.run(['git','-c','user.name=CPU','-c','user.email=cpu@example.invalid','commit','-qm','pin'],cwd=root,check=True)
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
            self.assertEqual(M.source_check(head,root),head)
            with self.assertRaises(RuntimeError):M.source_check('0'*40,root)
            (root/'x').write_text('drift')
            with self.assertRaises(RuntimeError):M.source_check(head,root)
    def test_native_wheel_and_input_raw_hash_fail_closed_and_preflight_no_gpu(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(M,'source_check',return_value='a'*40):
            root=Path(tmp);native=root/'native.so';native.write_bytes(b'native');wheel=root/'wheel.whl'
            with zipfile.ZipFile(wheel,'w') as z:z.writestr('mlx/core.cpython-312-darwin.so',b'core');z.writestr('mlx/lib/libmlx.dylib',b'lib')
            inputs=root/'inputs.json';inputs.write_text(json.dumps(document()))
            args=['a'*40,native,M.sha(native),wheel,M.sha(wheel),inputs,M.sha(inputs),'domain0']
            result=M.preflight(*args);self.assertFalse(result['gpu_executed']);self.assertFalse(result['qualified'])
            self.assertEqual(result['prepared_inputs_declared_payload_sha256'],'b'*64)
            self.assertNotEqual(result['prepared_inputs_raw_sha256'],'b'*64)
            self.assertEqual(result['schedule'],[['stock','native'],['native','stock'],['stock','native'],['native','stock']])
            for index in (2,4,6):
                bad=list(args);bad[index]='0'*64
                with self.assertRaises(RuntimeError):M.preflight(*bad)
    def test_physical_terminal_exact_count_and_every_owner_offset(self):
        counts=(7000,)*20;use=NS(lease=NS(epoch=9),terminal_succeeded=True)
        owners=tuple(NS(offset=n) for n in counts);writer=NS(poisoned=False,pending_epochs=(),ledger=NS(pending_count=0))
        result={};M.terminal_proof(result,use,(NS(event=(9,True)),),writer,owners,counts,(1,140000,1))
        self.assertTrue(result['read_terminal_success'])
        with self.assertRaises(RuntimeError):M.terminal_proof({},use,(NS(event=(9,True)),),writer,owners,counts,(0,140000,1))
        owners[3].offset-=1
        with self.assertRaises(RuntimeError):M.terminal_proof({},use,(NS(event=(9,True)),),writer,owners,counts,(1,140000,1))
    def test_full_retirement_and_failure_root_retention(self):
        pool=NS(capacity=20,free_count=0);calls=[]
        def close_owner():pool.free_count+=1
        owners=tuple(NS(close=close_owner,poll_completions=lambda:None) for _ in range(20))
        arena=NS(_closed=False)
        def close_arena():arena._closed=True
        arena.close_after_terminal=close_arena
        writer=NS(poisoned=False,pending_epochs=(),ledger=NS(pending_count=0))
        backend=NS(_orphaned_reads={});mx=NS(synchronize=lambda s:calls.append(s));result={}
        M.retire(mx,'stream',arena,writer,backend,owners,NS(state='closed'),pool,('root',),result)
        self.assertTrue(result['all_pages_retired']);self.assertEqual(result['pending_leases'],0)
        writer.poisoned=True;result={};before=len(M.FAILURE_ROOTS)
        with self.assertRaises(RuntimeError):M.retire(mx,'stream',arena,writer,backend,owners,None,pool,('exactroot',),result)
        self.assertEqual(M.FAILURE_ROOTS[-1],('exactroot',));self.assertEqual(len(M.FAILURE_ROOTS),before+1)
        self.assertTrue(result['failure_roots_retained'])
    def test_summary_keeps_all_three_cells_and_outlier(self):
        pairs=[dict(retired_result(),warmup=True,index=0)]+[dict(retired_result(),index=i) for i in range(1,4)]
        pairs[3]['timings']=dict(pairs[3]['timings'],native_eval_seconds=1.03)
        summary=M.summarize(pairs);self.assertEqual(len(summary['paired_cells']),3)
        self.assertAlmostEqual(summary['mean_native_attention_host_complete_seconds'],(.1+.1+1.1)/3)
        self.assertEqual(summary['median_paired_stock_over_native_ratio'],1)
        pairs[3]['all_pages_retired']=False
        with self.assertRaises(RuntimeError):M.summarize(pairs)
    def test_memory_and_deadline_bounds_report_allocator_separately(self):
        mx=NS(get_active_memory=lambda:123,get_peak_memory=lambda:456,get_cache_memory=lambda:78)
        with patch.object(M,'rss_bytes',return_value=100),patch.object(M.time,'monotonic',return_value=1):
            result=M.bounds(mx,0);self.assertEqual(result['mlx_active_bytes'],123);self.assertEqual(result['mlx_cache_bytes'],78)
        with patch.object(M,'rss_bytes',return_value=M.MAX_BYTES+1):
            with self.assertRaises(MemoryError):M.bounds(mx,0)
        with patch.object(M,'rss_bytes',return_value=0),patch.object(M.time,'monotonic',return_value=101):
            with self.assertRaises(TimeoutError):M.bounds(mx,0)

if __name__=='__main__':unittest.main()
