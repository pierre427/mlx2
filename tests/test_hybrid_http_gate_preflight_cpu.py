"""Stdlib CPU preflight; no model or device runtime imports."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/research/varlen_hybrid_http_gate.py'
sys.path.insert(0,str(SCRIPT.parent))
spec=importlib.util.spec_from_file_location('hybrid_http_gate',SCRIPT)
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)

class HTTPGateCPU(unittest.TestCase):
    def body(self,native=True):
        receipt={'route':'native_hybrid_paged_b2','selected':True,'observed_used':True,
            'qualified':False,'price_usable':False,'prefill_mode':'ordinary_completed_import',
            'native_prefill_observed_used':False,'stock_reduction_selected':True,
            'state_planes':['kv','gdn'],'q1_simd_stripes':32,'output_token_ids':[1,2]} if native else 'unqualified'
        return {'id':'actual','mlx2':{'qualification':'unqualified','route_receipt':receipt},
            'usage':{'completion_tokens':2},'choices':[{'text':'two','finish_reason':'length'}]}
    def test_final_receipt_requires_observed_use_and_output_bound(self):
        body=self.body();self.assertEqual(gate.summarize_http(body,2,True)['id'],'actual')
        for key,value in [('observed_used',False),('stock_reduction_selected',False),('qualified',True),('output_token_ids',[1])]:
            changed=self.body();changed['mlx2']['route_receipt'][key]=value
            with self.assertRaises(RuntimeError):gate.summarize_http(changed,2,True)
        changed=self.body();changed['choices'][0]['finish_reason']='stop'
        with self.assertRaises(RuntimeError):gate.summarize_http(changed,2,True)
    def test_stock_singleton_final_survivor_receipt_is_selected_and_observed(self):
        b2=self.body();r=b2['mlx2']['route_receipt'];r.update(stock_singleton_selected=True,
            stock_singleton_observed_used=False,hybrid_graph_proof={'native_stock_singleton_dispatches':0})
        gate.summarize_http(b2,2,True,True)
        b1=self.body();b1['usage']['completion_tokens']=4;r=b1['mlx2']['route_receipt']
        r.update(output_token_ids=[1,2,3,4],stock_singleton_selected=True,
            stock_singleton_observed_used=True,hybrid_graph_proof={'native_stock_singleton_dispatches':16})
        gate.summarize_http(b1,4,True,True)
        for key,value in (('stock_singleton_selected',False),('stock_singleton_observed_used',False),('q1_simd_stripes',16)):
            saved=r[key];r[key]=value
            with self.assertRaises(RuntimeError):gate.summarize_http(b1,4,True,True)
            r[key]=saved
        r['hybrid_graph_proof']['native_stock_singleton_dispatches']=0
        with self.assertRaises(RuntimeError):gate.summarize_http(b1,4,True,True)
    def test_stock0_legacy_survivor_and_actual_counter_totals(self):
        body=self.body();body['usage']['completion_tokens']=4
        body['mlx2']['route_receipt'].update(output_token_ids=[1,2,3,4],q1_simd_stripes=16)
        gate.summarize_http(body,4,True,False)
        legacy={'q1_stock_reduction_dispatches':16,'q1_stock_singleton_dispatches':0,
            'q1_stripe_dispatches_32':16,'q1_stripe_dispatches_16':32,'q1_tile_dispatches':48}
        gate.validate_physical(legacy,16,False)
        current={**legacy,'q1_stock_reduction_dispatches':48,'q1_stock_singleton_dispatches':32,
            'q1_stripe_dispatches_32':48,'q1_stripe_dispatches_16':0}
        gate.validate_physical(current,16,True)
        with self.assertRaises(RuntimeError):gate.validate_physical(legacy,16,True)
        with self.assertRaises(RuntimeError):gate.validate_physical(current,16,False)
        for key in ('q1_stock_singleton_dispatches','q1_stripe_dispatches_32'):
            with self.assertRaises(RuntimeError):gate.validate_physical({**current,key:0},16,True)
    def test_ordinary_string_route_supported_but_native_rejected(self):
        self.assertEqual(gate.summarize_http(self.body(False),2,False)['text'],'two')
        with self.assertRaises(RuntimeError):gate.summarize_http(self.body(),2,False)
    def test_diagnostics_do_not_wait_for_held_lifecycle_lock(self):
        lock=threading.Lock();lock.acquire()
        submission=threading.Lock();submission.acquire()
        engine=SimpleNamespace(error=None,thread=SimpleNamespace(is_alive=lambda:True),
            ready=threading.Event(),stop_event=threading.Event(),queued_jobs=2,
            lock=lock,submission_lock=submission,prompt_lock=threading.Lock(),
            incoming=SimpleNamespace(qsize=lambda:0))
        try:state=gate.host_diagnostics(engine,'native_factory_return',[object()])
        finally:lock.release();submission.release()
        self.assertFalse(state['lock_acquired']);self.assertFalse(state['submission_lock_acquired'])
        self.assertEqual(state['factory_captures'],1);self.assertTrue(state['thread_stacks'])
        json.dumps(state)
    def test_cleanup_bounds_engine_join_and_retains_stalled_roots(self):
        result={};engine=SimpleNamespace(close=lambda:None,thread=SimpleNamespace(is_alive=lambda:True))
        closer=SimpleNamespace(start=lambda:None,join=lambda timeout:None,is_alive=lambda:True)
        before=len(gate.FAILURE_ROOTS)
        with patch.object(gate.threading,'Thread',return_value=closer):
            cleanup=gate.bounded_cleanup(engine,None,None,[],result)
        self.assertFalse(cleanup['engine_closed']);self.assertTrue(cleanup['retained'])
        self.assertEqual(len(gate.FAILURE_ROOTS),before+1)
        del gate.FAILURE_ROOTS[before:]
    def test_pair_exception_does_not_join_executor(self):
        future=SimpleNamespace(result=lambda timeout:(_ for _ in ()).throw(TimeoutError('socket')))
        executor=SimpleNamespace(submit=lambda *args:future,shutdown=lambda **kwargs:None)
        with patch.object(gate,'ThreadPoolExecutor',return_value=executor),patch.object(executor,'shutdown') as shutdown:
            with self.assertRaises(TimeoutError):gate.pair('http://unused',({},{}))
            shutdown.assert_called_once_with(wait=False,cancel_futures=True)
    def test_text_first_fixture_real_local_tokenizer_cpu(self):
        # Optional artifact verification: stdlib preflight remains usable elsewhere.
        mlx_modules_before = {
            name for name in sys.modules if name == 'mlx' or name.startswith('mlx.')
        }
        model=Path('~/mlx-models/Qwen3.8-27B-MLX-4bit')
        if not model.is_dir():self.skipTest('local tokenizer artifact absent')
        try:import tokenizers
        except ImportError:self.skipTest('CPU tokenizers package absent')
        tokenizer=tokenizers.Tokenizer.from_file(str(model/'tokenizer.json'))
        adapter=SimpleNamespace(tokenizer=SimpleNamespace(decode=tokenizer.decode),
            prompt_tokens=lambda request:tokenizer.encode(request['prompt'],add_special_tokens=False).ids)
        prompts=gate.exact_prompts(adapter)
        self.assertEqual([len(ids) for _,ids in prompts],[32,96])
        for text,ids in prompts:
            self.assertTrue(text.startswith('Explain database isolation'))
            self.assertEqual(tuple(adapter.prompt_tokens({'prompt':text})),ids)
            self.assertEqual(tuple(adapter.prompt_tokens({'prompt':tokenizer.decode(list(ids))})),ids)
        mlx_modules_after = {
            name for name in sys.modules if name == 'mlx' or name.startswith('mlx.')
        }
        self.assertEqual(mlx_modules_after, mlx_modules_before)
    def test_dry_cli_forbids_runtime_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'out.json'
            code='''import runpy,sys
class Block:
 def find_spec(self,name,*args):
  if name.startswith(('mlx','_paged_kv_native')):raise RuntimeError('runtime forbidden')
sys.meta_path.insert(0,Block());sys.path.insert(0,sys.argv[1]);sys.argv=[sys.argv[2],'--output',sys.argv[3]]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
            done=subprocess.run([sys.executable,'-c',code,str(SCRIPT.parent),str(SCRIPT),str(out)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            result=json.loads(out.read_text());self.assertFalse(result['gpu_executed'])
            self.assertEqual(result['hard_seconds'],100);self.assertEqual(result['max_rss_bytes'],48<<30)
            self.assertEqual(result['numeric_tensor_parity'],'not_tested')
    def supervisor_failure(self,clock,rss):
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(output=Path(directory)/'out.json')
            proc=SimpleNamespace(pid=12345,poll=lambda:None,wait=lambda:0)
            with patch.object(gate.subprocess,'Popen',return_value=proc) as spawn,patch.object(gate.time,'monotonic',side_effect=clock),patch.object(gate.subprocess,'check_output',return_value=str(rss)),patch.object(gate.os,'killpg') as kill:
                self.assertEqual(gate.supervise(args,gate.preflight()),1)
                kill.assert_called_once_with(proc.pid,gate.signal.SIGKILL)
                self.assertTrue(spawn.call_args.kwargs['start_new_session'])
                self.assertTrue(json.loads(args.output.read_text())['worker_killed'])
    def test_deadline_kills_entire_worker_group(self):self.supervisor_failure([0,101],0)
    def test_rss_ceiling_kills_entire_worker_group(self):self.supervisor_failure([0,1],(48<<30)//1024+1)
    def test_source_uses_real_http_sampler_and_three_factory_returns(self):
        source=SCRIPT.read_text();tree=ast.parse(source)
        self.assertFalse(any('mlx' in ast.unparse(n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))))
        for text in ('handler_for(engine)',"base+'/v1/completions'",'owners,candidate,bootstrap=original_factory',
            'return owners,candidate,bootstrap','int(response.token)',"range(1,257)",'len(ids)!=target',
            "native_ids!=ordinary_ids",'all(owner.fully_retired for owner in owners)',
            'return original_close()',"q1_stock_reduction_dispatches",'execution_width'):
            self.assertIn(text,source)
        self.assertNotIn('.astype(mx.float16)',source)
        self.assertNotIn('response.token =',source)
        self.assertIs(gate.make_profile({'source_commit':'frozen'},True)['stock_reduction'],True)

if __name__=='__main__':unittest.main()
