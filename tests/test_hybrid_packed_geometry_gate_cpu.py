"""CPU-only geometry control seams; all tensor/runtime imports forbidden."""
import importlib.abc,importlib.util,sys,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='mlx' or fullname.startswith('mlx.') or fullname=='_paged_kv_native':raise RuntimeError('GPU import forbidden')
sys.meta_path.insert(0,Guard());sys.path.insert(0,str(ROOT/'scripts/research'))
spec=importlib.util.spec_from_file_location('geometry_gate',ROOT/'scripts/research/varlen_hybrid_packed_prefill_geometry_gate.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
class Array:
    def __init__(self,shape):self.shape=shape
    def __getitem__(self,key):
        key=key if isinstance(key,tuple) else (key,)
        shape=[]
        for dim,part in zip(self.shape,key):
            if isinstance(part,slice):shape.append(len(range(*part.indices(dim))))
        return Array(tuple(shape)+self.shape[len(key):])
class MX:
    def __init__(self):self.evaluated=None
    def array(self,rows):return Array((len(rows),len(rows[0])))
    def concatenate(self,arrays,axis):
        shape=list(arrays[0].shape);shape[axis]=sum(x.shape[axis] for x in arrays);return Array(tuple(shape))
    def eval(self,*arrays):self.evaluated=arrays
class Model:
    def __init__(self):self.rows=[];self.final_shape=None
    def make_cache(self):
        row=[SimpleNamespace(state=[Array((1,3,4)),Array((1,2,4,4))])];self.rows.append(row);return row
    def mixed_forward(self,segments):
        self.segments=segments;return [Array((1,tokens.shape[1],4)) for tokens,caches in segments]
    def logits(self,hidden):self.final_shape=hidden.shape;return Array((2,1,8))
class Tests(unittest.TestCase):
    def test_independent_cold_caches_and_exact_final_projection_shape(self):
        model=Model();mx=MX();ids=(tuple(range(32)),tuple(range(96)))
        rows,logits=gate.stock_mixed_reference(model,ids,mx)
        self.assertEqual([s[0].shape for s in model.segments],[(1,32),(1,96)])
        self.assertIsNot(rows[0][0],rows[1][0]);self.assertEqual(model.final_shape,(2,1,4))
        self.assertEqual([x.shape for x in logits],[(8,),(8,)])
        self.assertEqual(len(mx.evaluated),5)
        self.assertIs(model.segments[0][1][0],rows[0][0])
    def test_geometry_mismatch_fails_before_final_projection(self):
        model=Model();model.mixed_forward=lambda segments:[Array((1,32,4)),Array((1,95,4))]
        with self.assertRaisesRegex(ValueError,'geometry differs'):gate.stock_mixed_reference(model,(tuple(range(32)),tuple(range(96))),MX())
        self.assertIsNone(model.final_shape)
    def test_distinct_labels_and_no_qualification_promotion(self):
        plan=gate.plan();self.assertFalse(plan['qualified']);self.assertFalse(plan['serving_selected'])
        self.assertIn('serial exact arm retained separately',plan['reference'])
        source=Path(gate.__file__).read_text()
        self.assertIn("result['serial_exact_parity']",source)
        self.assertIn("r['finite'] and r['exact']",source)
        self.assertNotIn('allclose(',source)
if __name__=='__main__':unittest.main()
