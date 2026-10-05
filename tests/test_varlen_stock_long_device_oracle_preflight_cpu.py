"""Five CPU-only checks for the frozen stock-long device oracle."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/research/varlen_stock_long_device_oracle.py'
spec = importlib.util.spec_from_file_location('stock_long_oracle', SCRIPT)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class StockLongOraclePreflight(unittest.TestCase):
    def test_bounded_dense_cases_and_scratch(self):
        plans = oracle.preflight()['cases']
        self.assertEqual([p['contexts'] for p in plans], [[1024, 1025], [4096, 8192], [6951, 6930]])
        for plan in plans:
            self.assertEqual(plan['blocks'], 128)
            self.assertEqual(plan['scratch_bytes'], 2 * 24 * 128 * 258 * 4)
            self.assertEqual(plan['virtual_left_padding'],
                             [max(plan['contexts']) - n for n in plan['contexts']])
            self.assertLess(2 * plan['plane_bytes'] + plan['scratch_bytes'], 256 << 20)

    def test_plan_rejects_unbounded_or_wrong_inputs(self):
        for contexts in ((1024, 1024), (8193, 1), (0, 6951), (1024,), [1024, 1025], (True, 1025)):
            with self.assertRaises(ValueError):
                oracle.case_plan(contexts)

    def test_dry_cli_does_not_import_or_execute_mlx(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt = Path(folder) / 'receipt.json'
            code = '''import runpy,sys
class Block:
 def find_spec(self,name,*args):
  if name.startswith(('mlx','_paged_kv_native')): raise RuntimeError('runtime import forbidden')
sys.meta_path.insert(0,Block())
sys.argv=[sys.argv[1],'--output',sys.argv[2]]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
            result = subprocess.run([sys.executable, '-c', code, str(SCRIPT), str(receipt)],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(receipt.read_text())
            self.assertEqual(data['status'], 'planned')
            self.assertFalse(data['gpu_executed'])

    def test_deadline_and_rss_guards(self):
        with patch.object(oracle.time, 'monotonic', return_value=61):
            with self.assertRaises(TimeoutError): oracle.check_bounds(0)
        class Usage: ru_maxrss = oracle.MAX_RSS + 1
        with patch.object(oracle.resource, 'getrusage', return_value=Usage()):
            with self.assertRaises(MemoryError): oracle.check_bounds(oracle.time.monotonic())

    def test_dense_mask_and_native_proof_contract(self):
        source = SCRIPT.read_text()
        top = [n for n in ast.parse(source).body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any('mlx' in ast.unparse(n) for n in top))
        self.assertIn("mask=visible_mask", source)
        self.assertIn("mx.zeros((dense_length - n, 4, 256), dtype=mx.bfloat16)", source)
        self.assertNotIn("for i in range(2)], axis=0).reshape", source)
        self.assertIn("native.q1_stock_long_partial_dispatch_count(arena._arena)", source)
        self.assertIn("native.q1_stock_long_reduce_dispatch_count(arena._arena)", source)
        self.assertIn("raw_bf16_payloads_equal", source)
        self.assertIn("exact_elements_by_lane", source)
        self.assertIn("arena.close_after_terminal()", source)
        self.assertIn("FAILURE_ROOTS.append(", source)
        self.assertIn("BINARY.read_bytes()).hexdigest() != BINARY_SHA", source)


if __name__ == '__main__': unittest.main()
