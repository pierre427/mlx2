"""Actual400 corpus arena capacities checked before native allocation/charge."""
import ast,importlib.abc,json,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native':raise RuntimeError('runtime import forbidden')
sys.meta_path.insert(0,Guard())
from mlx2.runtime.paged_native_arena_geometry import require_arena_storage,MAX_HOST_PLANE_BYTES,SHAPE_DIM_MAX
CAP=dict(version=1,layout='contiguous_uint8_2d_large',large_plane_threshold_bytes=SHAPE_DIM_MAX,large_plane_alignment_bytes=4096,max_plane_bytes=MAX_HOST_PLANE_BYTES,exact_byte_allocation=True)
def native(cap=CAP):return NS(arena_storage_capability=lambda:dict(cap))
class Tests(unittest.TestCase):
 def test_actual_gpu_provider_receipt_and_cpp_schema(self):
  receipt=json.loads(Path('/tmp/mlx2-n20-large-plane-463e-device.json').read_text())
  cap=receipt['capability']
  self.assertEqual(cap,CAP)
  proof=require_arena_storage(native(cap),receipt['plane_bytes'])
  self.assertEqual(proof['allocation_bytes'],receipt['total_arena_bytes'])
  source=(ROOT/'native/paged_kv/binding.cpp').read_text()
  provider=source.split('module.def("arena_storage_capability"',1)[1].split('module.def("create_arena"',1)[0]
  import re
  self.assertEqual(set(re.findall(r'result\["([^"\n]+)"\]',provider)),set(cap))
  self.assertIn('result["exact_byte_allocation"] = true;',provider)
 def test_preload_real_raw_contract_precedes_model_imports(self):
  from mlx2.runtime import hybrid_packed_prefill_n as factory
  data=json.loads(Path('/tmp/mlx2-spomin400-nativeN-inputs.json').read_text())
  actual=json.loads(Path('/tmp/mlx2-n20-large-plane-463e-device.json').read_text())['capability']
  raw=native(actual);raw.packed_n20_capability=lambda:dict(factory.CAP)
  for name in ('grouped_multirow_write_n20','q1_scalar_dispatch_count','q1_stock_long_n20_singleton_partial_dispatch_count','q1_stock_long_n20_singleton_reduce_dispatch_count',*factory.COUNTERS):setattr(raw,name,lambda:0)
  proof=factory.preflight_native_inputs(raw,data)
  self.assertGreaterEqual(proof['plane_bytes'],4_802_478_080)
  del raw.grouped_multirow_write_n20
  with self.assertRaisesRegex(ValueError,'ABI missing'):factory.preflight_native_inputs(raw,data)
  tree=ast.parse((ROOT/'scripts/research/spomin_400case_phased_http_bench.py').read_text())
  init=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='init')
  check=next(n for n in ast.walk(init) if isinstance(n,ast.Call) and ast.unparse(n.func)=='factory.preflight_native_inputs')
  imports=[n for n in ast.walk(init) if isinstance(n,ast.ImportFrom) and n.module=='mlx2.adapters.qwen38_27b']
  self.assertTrue(imports and all(n.lineno>check.lineno for n in imports))
 def test_all_actual_domains_exact_plane_allocation_no_padding(self):
  data=json.loads(Path('/tmp/mlx2-spomin400-nativeN-inputs.json').read_text())
  for domain in data['domain_order']:
   counts=[r['prompt_tokens'] for r in data['rows'] if r['domain']==domain]
   pages=sum(16*((n+192+63)//64+2) for n in counts);plane=pages*4*256*64*2
   p=require_arena_storage(native(),plane)
   self.assertEqual(p['plane_shape'][0]*p['plane_shape'][1],plane);self.assertEqual(p['allocation_bytes'],2*plane);self.assertEqual(p['padding_bytes'],0)
  counts=[r['prompt_tokens'] for r in data['rows'][:20]]
  self.assertEqual(sum(16*((n+192+63)//64+2) for n in counts)*131072,4_802_478_080)
 def test_old_native_or_small_device_refuses_before_charge(self):
  with self.assertRaisesRegex(ValueError,'ABI missing'):require_arena_storage(NS(),4_802_478_080)
  with self.assertRaisesRegex(MemoryError,'Metal buffer'):require_arena_storage(native({**CAP,'max_plane_bytes':1<<30}),4_802_478_080)
 def test_small_exact_bytes_and_large_alignment_fail_closed(self):
  for count in (1,4097,SHAPE_DIM_MAX):self.assertEqual(require_arena_storage(native(),count)['plane_shape'],(count,))
  with self.assertRaisesRegex(ValueError,'alignment'):require_arena_storage(native(),SHAPE_DIM_MAX+2)
  for count in (True,0,MAX_HOST_PLANE_BYTES+4096):
   with self.assertRaises(ValueError):require_arena_storage(native(),count)
 def test_capability_drift_refused(self):
  for edit in ({'version':True},{'exact_byte_allocation':False},{'exact_byte_allocation':1},{'layout':'padded'}, {'large_plane_alignment_bytes':2048},{'max_plane_bytes':MAX_HOST_PLANE_BYTES+4096}):
   with self.assertRaisesRegex(ValueError,'contract differs'):require_arena_storage(native({**CAP,**edit}),4_802_478_080)
 def test_estimator_and_factory_check_actual_size_before_resource_constructor(self):
  tree=ast.parse((ROOT/'src/mlx2/runtime/hybrid_packed_prefill_n.py').read_text())
  for name in ('estimate_cold_cohort_memory_n','create_cold_packed_hybrid_n'):
   fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
   guard=next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='require_arena_storage')
   self.assertEqual(ast.unparse(guard.args[1]),"components['arena_bytes'] // 2")
   charge=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and ast.unparse(n.func) in ('R.HybridServingResources','NativeWriteBackend')]
   self.assertTrue(all(n.lineno>guard.lineno for n in charge))
if __name__=='__main__':unittest.main()
