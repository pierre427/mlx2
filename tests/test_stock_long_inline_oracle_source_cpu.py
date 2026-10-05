"""Import-safe strict geometry and physical-proof contracts for the inline oracle."""
import ast
import importlib.abc
import importlib.util
from pathlib import Path
import sys
import unittest

if __name__ != '__main__': raise unittest.SkipTest('run directly without runtime imports')
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'mlx' or name.startswith('mlx.') or name == '_paged_kv_native':
            raise RuntimeError('runtime import prohibited')
sys.meta_path.insert(0, NoRuntime())
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'scripts/research/varlen_stock_long_inline_device_oracle.py'
spec=importlib.util.spec_from_file_location('inline_oracle', PATH)
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)

class Contracts(unittest.TestCase):
    def test_plan_is_bounded_dense_left_padded_bf16(self):
        receipt=oracle.preflight()
        self.assertFalse(receipt['gpu_executed'])
        self.assertFalse(receipt['qualified'])
        self.assertEqual(receipt['dtype'],'bfloat16')
        self.assertEqual([tuple(x['contexts']) for x in receipt['cases']],list(oracle.CONTEXTS))
        for case in receipt['cases']:
            self.assertEqual(case['virtual_left_padding'],[case['dense_length']-x for x in case['contexts']])
            self.assertEqual(case['expected_inline_metadata'],1)
            self.assertEqual((case['expected_partial'],case['expected_reduce']),(1,1))
            self.assertLessEqual(case['scratch_bytes'],oracle.MAX_RSS)
        for bad in ((1024,1024),(0,8192),(8192,8193),(1,)):
            with self.assertRaises(ValueError):oracle.case_plan(bad)
    def test_execute_requires_frozen_source_native_and_actual_raw_proof(self):
        text=PATH.read_text()
        tree=ast.parse(text)
        self.assertIn('git',text)
        self.assertIn('source worktree is dirty',text)
        self.assertIn('frozen native binary hash differs',text)
        self.assertIn('native inline metadata counter ABI unavailable',text)
        self.assertIn('inline_after - inline_before != 1',text)
        self.assertIn('raw_bf16_output_equal',text)
        self.assertIn('owner.fully_retired',text)
        self.assertTrue(any(isinstance(n,ast.FunctionDef) and n.name=='execute_case' for n in tree.body))

unittest.main()
