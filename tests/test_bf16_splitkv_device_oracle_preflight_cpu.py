"""CPU preflight only: the oracle runtime never imports or executes MLX here."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/research/varlen_bf16_splitkv_device_oracle.py'
spec=importlib.util.spec_from_file_location('bf16_oracle',SCRIPT)
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)

class OraclePreflightCPU(unittest.TestCase):
    def test_case_coverage_and_scratch_budget(self):
        plans=oracle.preflight()['cases']
        self.assertEqual(len(plans),8)
        self.assertEqual({p['partition'] for p in plans},{128,256})
        self.assertEqual({n for p in plans for n in p['contexts']},{63,127,128,129,4096,8192})
        for p in plans:
            partitions=(max(p['contexts'])+p['partition']-1)//p['partition'] if p['split'] else 0
            self.assertEqual(p['scratch_bytes'],2*24*partitions*258*4)
            self.assertEqual((p['expected_partial'],p['expected_reduce'],p['expected_tile']),
                             (int(p['split']),int(p['split']),int(not p['split'])))
            # BF16 arena planes plus random/reference tensors fit the 4GiB ceiling.
            self.assertLess(2*p['plane_bytes']+p['scratch_bytes']+sum(p['contexts'])*4*256*8,256<<20)
    def test_plan_rejects_unbounded_or_wrong_inputs(self):
        for contexts,partition in [((31,129),128),((63,8193),256),((63,),128),([63,129],128),((63,129),64),((True,129),128)]:
            with self.assertRaises(ValueError):oracle.case_plan(contexts,partition)
    def test_dry_cli_with_runtime_imports_forbidden(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'receipt.json'
            code='''import runpy,sys
class Block:
 def find_spec(self,name,*args):
  if name.startswith(('mlx','_paged_kv_native')): raise RuntimeError('runtime import forbidden')
sys.meta_path.insert(0,Block())
sys.argv=[sys.argv[1],'--output',sys.argv[2]]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
            result=subprocess.run([sys.executable,'-c',code,str(SCRIPT),str(path)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            data=json.loads(path.read_text());self.assertFalse(data['gpu_executed']);self.assertEqual(data['status'],'planned')
    def test_wall_and_rss_bounds(self):
        with patch.object(oracle.time,'monotonic',return_value=61):
            with self.assertRaises(TimeoutError):oracle.check_bounds(0)
        class Usage:ru_maxrss=oracle.MAX_RSS+1
        with patch.object(oracle.resource,'getrusage',return_value=Usage()):
            with self.assertRaises(MemoryError):oracle.check_bounds(oracle.time.monotonic())
    def test_runtime_contract_order_and_unpadded_reference(self):
        tree=ast.parse(SCRIPT.read_text())
        top_imports=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('mlx' in ast.unparse(node) for node in top_imports))
        source=SCRIPT.read_text()
        positions=[source.index(text) for text in ('backend.append_completed(', 'backend.append_staged(',
                    'backend.read_staged(', 'backend.drain_staged(', 'branch.prove_staged_layer_read(',
                    'branch.prepare(1)', 'state.publish()', 'owner.snapshot()', 'arena.diagnostic_read(')]
        self.assertEqual(positions,sorted(positions))
        self.assertIn("keys[i].transpose(1, 0, 2)[None, :, :, :]",source)
        self.assertNotIn('mx.pad(',source)
        self.assertIn('native.storage_dtype(arena._arena)',source)
        self.assertIn('arena.close_after_terminal()',source)
        self.assertIn('if pool.free_count != pool.capacity',source)
        self.assertIn('FAILURE_ROOTS.append(',source)
        self.assertIn("output.dtype != mx.bfloat16",source)
        self.assertIn("BINARY.read_bytes()).hexdigest() != BINARY_SHA",source)

if __name__=='__main__':unittest.main()
