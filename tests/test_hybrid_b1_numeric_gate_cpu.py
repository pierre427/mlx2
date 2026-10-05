"""CPU author checks only: not model, device or numeric-route qualification."""
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
SCRIPT=ROOT/'scripts/research/varlen_hybrid_b1_numeric_gate.py'
sys.path.insert(0,str(SCRIPT.parent))
spec=importlib.util.spec_from_file_location('hybrid_b1_gate',SCRIPT)
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)

class B1GateCPU(unittest.TestCase):
    def test_default_dry_imports_no_runtime_and_labels_correctly(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'out.json'
            code='''import runpy,sys
class Block:
 def find_spec(self,name,*args):
  if name.startswith(('mlx','_paged_kv_native')):raise RuntimeError('runtime import forbidden')
sys.meta_path.insert(0,Block());sys.path.insert(0,sys.argv[1]);sys.argv=[sys.argv[2],'--output',sys.argv[3]]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
            done=subprocess.run([sys.executable,'-c',code,str(SCRIPT.parent),str(SCRIPT),str(out)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            result=json.loads(out.read_text());self.assertFalse(result['gpu_executed'])
            self.assertFalse(result['qualified']);self.assertFalse(result['serving_selected'])
            self.assertEqual(result['widths'],[2,1,1]);self.assertEqual(result['numeric_parity'],'not_tested')
    def test_actual_stock32_domain_accepts_only_aligned_short_gap(self):
        # Extract the pure source guard without importing MLX or the runtime package.
        source=ROOT/'src/mlx2/runtime/paged_hybrid_research_profile.py'
        tree=ast.parse(source.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_hybrid_stock_pair')
        namespace={};exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
        validate=namespace['validate_hybrid_stock_pair'];profile={'stock_reduction':True,'q1_split_partition':0,'q1_simd_stripes':16,'max_tokens':4}
        for lengths in ((32,64),(32,96),(33,65),(61,125)):self.assertTrue(validate(profile,lengths))
        for lengths in ((32,63),(63,64),(64,65),(127,128),(128,129)):
            with self.assertRaises(ValueError):validate(profile,lengths)
    def test_physical_B1_requires_stock0_and_actual_scalar_spans(self):
        receipt={'packed_lanes':1,'native_tile_dispatches':16,'native_stock_reduction_dispatches':0,
            'q1_simd_stripes':16,'grouped_q1_writes':0,'scalar_native_writes':64,'expected_scalar_write_spans':64,
            'native_split_partial_dispatches':0,'native_split_reduce_dispatches':0}
        gate.validate_physical(receipt,1,16)
        for key,value in (('native_stock_reduction_dispatches',16),('scalar_native_writes',16),('q1_simd_stripes',32)):
            changed={**receipt,key:value}
            with self.assertRaises(RuntimeError):gate.validate_physical(changed,1,16)
    def test_full_tensor_metrics_detect_drift_dtype_and_nonfinite(self):
        try:import numpy as np
        except ImportError:self.skipTest('CPU numpy absent')
        mx=SimpleNamespace(float32=np.float32,all=np.all,isfinite=np.isfinite,abs=np.abs,max=np.max,maximum=np.maximum,mean=np.mean)
        values=np.array([[1.,2.,0.]],dtype=np.float32)
        metrics=gate.tensor_metrics(values,values.copy(),mx);self.assertTrue(metrics['exact']);self.assertEqual(metrics['exact_fraction'],1)
        changed=values.copy();changed[0,1]+=.125
        self.assertFalse(gate.tensor_metrics(changed,values,mx)['exact'])
        changed[0,1]=np.inf;self.assertFalse(gate.tensor_metrics(changed,values,mx)['finite'])
        with self.assertRaises(RuntimeError):gate.tensor_metrics(values,values.astype(np.float64),mx)
    def supervisor_failure(self,clock,rss):
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(output=Path(directory)/'out.json');proc=SimpleNamespace(pid=12345,poll=lambda:None,wait=lambda:0)
            with patch.object(gate.subprocess,'Popen',return_value=proc) as spawn,patch.object(gate.time,'monotonic',side_effect=clock),patch.object(gate.subprocess,'check_output',return_value=str(rss)),patch.object(gate.os,'killpg') as kill:
                self.assertEqual(gate.supervise(args,gate.plan()),1)
                kill.assert_called_once_with(proc.pid,gate.signal.SIGKILL)
                self.assertTrue(spawn.call_args.kwargs['start_new_session'])
    def test_supervisor_hard_deadline(self):self.supervisor_failure([0,101],0)
    def test_supervisor_hard_resident_limit(self):self.supervisor_failure([0,1],(48<<30)//1024+1)
    def test_source_preserves_real_factory_input_publication_and_retirement(self):
        source=SCRIPT.read_text();ast.parse(source)
        for fragment in ('owners,candidate,bootstrap=factory.create_shared_hybrid_graph_pack(',
            'reference.forward_one(input_ids)','cache.filter([1])','owners[0].fully_retired',
            'next_ids=(next_ids[1],)','for state in prepared:state.publish()',
            'candidate._serving_resources.reap()','factory._CHARGED==initial_charge',
            "result['numeric_parity']='passed'",'compare_slots(tuple(branches)',
            'model.model(mx.array([[input_ids[0]]]),cache=reference_caches)'):
            self.assertIn(fragment,source)
        self.assertNotIn('.astype(mx.float16)',source)
        self.assertNotIn('logit_atol',source)

if __name__=='__main__':unittest.main()
