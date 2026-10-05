"""Stdlib-only smoke preflight. Never import MLX, load weights or run GPU."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/research/varlen_hybrid_serving_smoke.py'
spec=importlib.util.spec_from_file_location('hybrid_smoke',SCRIPT)
smoke=importlib.util.module_from_spec(spec);spec.loader.exec_module(smoke)

class SmokeCPU(unittest.TestCase):
    def test_profile_binds_exact_short_lifecycle_scope(self):
        identity={'source_commit':'frozen'};p=smoke.make_profile(identity)
        self.assertIs(p['identity'],identity)
        self.assertEqual(p['context_bounds'],{'minimum':32,'maximum':96,'distinct':True})
        self.assertLessEqual(p['context_bounds']['maximum']+p['max_tokens']-1,128)
        self.assertEqual(p['max_tokens'],4);self.assertEqual(p['storage_dtype'],'bfloat16')
        for name in ('qualified','price_usable','serving_default','warm_apcv2'):self.assertIs(p[name],False)
        self.assertEqual(p['required_environment']['MLX2_PAGED_Q1_SPLIT_KV'],'0')
        self.assertEqual(p['required_environment']['MLX2_PAGED_GROUPED_SAMPLER'],'0')
        self.assertIs(p['stock_reduction'],False)
        self.assertEqual(p['required_environment']['MLX2_PAGED_Q1_STOCK_REDUCTION'],'0')
        stock=smoke.make_profile(identity,True)
        self.assertIs(stock['stock_reduction'],True)
        self.assertEqual(stock['required_environment']['MLX2_PAGED_Q1_STOCK_REDUCTION'],'1')
        with self.assertRaises(ValueError):smoke.make_profile(identity,'true')
    def test_singleton_profile_is_explicit_boolean_default_off_and_requires_stock(self):
        ordinary=smoke.make_profile({'source_commit':'frozen'},True)
        self.assertIs(ordinary['stock_singleton'],False)
        self.assertEqual(ordinary['required_environment']['MLX2_PAGED_Q1_STOCK_SINGLETON'],'0')
        selected=smoke.make_profile({'source_commit':'frozen'},True,True)
        self.assertIs(selected['stock_singleton'],True)
        self.assertEqual(selected['required_environment']['MLX2_PAGED_Q1_STOCK_SINGLETON'],'1')
        for value in (1,'true'):
            with self.assertRaises(ValueError):smoke.make_profile({},True,value)
        with self.assertRaises(ValueError):smoke.make_profile({},False,True)
    def test_dry_cli_with_all_runtime_imports_forbidden(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'out.json'
            code='''import runpy,sys
class Block:
 def find_spec(self,name,*args):
  if name.startswith(('mlx','_paged_kv_native')):raise RuntimeError('runtime import forbidden')
sys.meta_path.insert(0,Block());sys.argv=[sys.argv[1],'--output',sys.argv[2]]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
            done=subprocess.run([sys.executable,'-c',code,str(SCRIPT),str(out)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            result=json.loads(out.read_text());self.assertFalse(result['gpu_executed'])
            self.assertEqual(result['hard_seconds'],100);self.assertEqual(result['max_rss_bytes'],48<<30)
            self.assertEqual(result['numeric_parity'],'not_tested')
    def supervise_failure(self,clock,rss):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'out.json';args=SimpleNamespace(output=output)
            proc=SimpleNamespace(pid=12345,poll=lambda:None,wait=lambda:0)
            with patch.object(smoke.subprocess,'Popen',return_value=proc) as spawn,patch.object(smoke.time,'monotonic',side_effect=clock),patch.object(smoke.subprocess,'check_output',return_value=str(rss)),patch.object(smoke.os,'killpg') as kill:
                self.assertEqual(smoke.supervise(args,smoke.plan()),1)
                kill.assert_called_once_with(proc.pid,smoke.signal.SIGKILL)
                self.assertTrue(spawn.call_args.kwargs['start_new_session'])
                self.assertEqual(json.loads(output.read_text())['status'],'failed')
    def test_supervisor_kills_at_whole_worker_deadline(self):self.supervise_failure([0,101],0)
    def test_supervisor_kills_at_resident_ceiling(self):self.supervise_failure([0,1],(48<<30)//1024+1)
    def test_source_uses_real_jobs_installer_sampler_and_retirement(self):
        source=SCRIPT.read_text();tree=ast.parse(source)
        self.assertFalse(any('mlx' in ast.unparse(n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))))
        for call in ('serving.Job(', 'serving.resolve_sampling(', 'make_sampler(temp=0.0)',
            'BatchGenerator(adapter.model', 'serving.install_explicit_native_hybrid_b2_cohort(',
            "jobs[0].cancelled.set()",'batch.remove([uids[0]])','batch.take_lane_failures()',
            'factory.reap_hybrid_admission_orphans()', 'candidate._serving_resources.reap()',
            'all(owner.fully_retired for owner in owners)',
            "result['physical_before_close']=candidate.backend.profile_counters_snapshot()",
            'return original_close()'):
            self.assertIn(call,source)
        self.assertIn("if any(observed.values())",source)
        self.assertIn("if observed!={0:2,1:4}",source)
        self.assertNotIn('.astype(mx.float16)',source)
        self.assertNotIn('response.token =',source)

if __name__=='__main__':unittest.main()
