"""CPU finite-publication gate: one eval/extraction, every predicate retained."""
import importlib.abc
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native':raise RuntimeError('runtime forbidden')
sys.meta_path.insert(0,NoRuntime())
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cold_finite',ROOT/'src/mlx2/runtime/hybrid_packed_prefill.py')
M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M)

class Vector:
    def __init__(self,mx,values):self.mx=mx;self.values=values
    def tolist(self):
        self.mx.extractions+=1
        if self.mx.fail_extract:raise RuntimeError('host extraction failed')
        return self.values.tolist()
class MX:
    def __init__(self):self.predicates=0;self.evaluations=0;self.extractions=0;self.fail_eval=False;self.fail_extract=False
    def isfinite(self,value):self.predicates+=1;return np.isfinite(value)
    all=staticmethod(np.all)
    def stack(self,values):return Vector(self,np.stack(values))
    def eval(self,value):
        self.evaluations+=1
        if self.fail_eval:raise RuntimeError('evaluation failed')

class FiniteTests(unittest.TestCase):
    def state(self):return np.ones((2,8),dtype=np.float32),[np.ones((1,4),dtype=np.float32) for _ in range(192)]
    def test_all_193_predicates_one_eval_one_extraction(self):
        logits,roots=self.state();mx=MX();M._require_finite_roots(mx,logits,roots)
        self.assertEqual((mx.predicates,mx.evaluations,mx.extractions),(193,1,1))
    def test_logits_early_late_state_rejected_without_short_circuit(self):
        for index in (-1,0,191):
            for bad in (np.nan,np.inf,-np.inf):
                with self.subTest(index=index,bad=bad):
                    logits,roots=self.state();target=logits if index==-1 else roots[index];target.flat[0]=bad;mx=MX()
                    with self.assertRaisesRegex(ValueError,'nonfinite'):M._require_finite_roots(mx,logits,roots)
                    self.assertEqual((mx.predicates,mx.evaluations,mx.extractions),(193,1,1))
    def test_eval_or_host_failure_propagates_with_roots_untouched(self):
        for failure in ('fail_eval','fail_extract'):
            logits,roots=self.state();before=[x.copy() for x in roots];mx=MX();setattr(mx,failure,True)
            with self.assertRaises(RuntimeError):M._require_finite_roots(mx,logits,roots)
            for actual,expected in zip(roots,before):np.testing.assert_array_equal(actual,expected)
            self.assertEqual(mx.evaluations,1)
    def test_validator_remains_before_terminal_offset_and_public_attachment(self):
        source=(ROOT/'src/mlx2/runtime/hybrid_packed_prefill.py').read_text()
        finite=source.index('_require_finite_roots(mx,logits,roots)')
        self.assertLess(finite,source.index("writer=candidate.backend.writer",finite))
        self.assertLess(finite,source.index('GDNBoundaryCheckpoint(revision,uid,count,0'))
        self.assertIn('reservation.retain_failure_roots',source)

if __name__=='__main__':unittest.main()
