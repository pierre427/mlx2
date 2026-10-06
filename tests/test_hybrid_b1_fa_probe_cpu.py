"""CPU FA probe wiring/oracle, no device or model execution."""
import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/research/varlen_hybrid_b1_fa_boundary_probe.py'
sys.path.insert(0,str(SCRIPT.parent))
spec=importlib.util.spec_from_file_location('b1_fa_probe',SCRIPT)
probe_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe_module)

class B1FAProbeCPU(unittest.TestCase):
    def test_actual_capture_restores_entrypoint_on_missing_calls(self):
        original=lambda *args,**kwargs:None
        module=SimpleNamespace(scaled_dot_product_attention=original)
        probe=probe_module.B1FABoundaryProbe()
        with self.assertRaises(RuntimeError):
            with probe.capture_ordinary(module):pass
        self.assertIs(module.scaled_dot_product_attention,original)
    def test_B1_callback_rejects_B2_and_duplicates(self):
        probe=probe_module.B1FABoundaryProbe()
        event={'layer_index':3,'fa_index':0,'offsets':(1,2),'queries':None,'keys':None,
            'values':None,'native_attention':None,'hidden_dtype':'bfloat16'}
        with self.assertRaises(ValueError):probe.candidate_callback(event)
        event['offsets']=(1,);probe.candidate_callback(event)
        with self.assertRaises(ValueError):probe.candidate_callback(event)
    def test_same_QKV_oracle_separates_attention_arithmetic(self):
        mlx_modules_before = {
            name for name in sys.modules if name == 'mlx' or name.startswith('mlx.')
        }
        try:import numpy as np
        except ImportError:self.skipTest('CPU numpy absent')
        mx=SimpleNamespace(float32=np.float32,bool_=np.bool_,all=np.all,isfinite=np.isfinite,
            abs=np.abs,max=np.max,maximum=np.maximum,mean=np.mean,sum=np.sum)
        query=np.ones((1,2,1,4),dtype=np.float32);keys=np.ones((1,2,2,4),dtype=np.float32)
        values=keys.copy();cache=SimpleNamespace(offset=np.array([2]),left_padding=np.array([0]))
        original=lambda q,k,v,**kwargs:q.copy()
        module=SimpleNamespace(scaled_dot_product_attention=original)
        probe=probe_module.B1FABoundaryProbe()
        with probe.capture_ordinary(module):
            for _ in range(16):module.scaled_dot_product_attention(query,keys,values,cache=cache,scale=.25,mask=None)
        self.assertIs(module.scaled_dot_product_attention,original)
        for layer in probe_module.LAYERS:
            probe.candidate_callback({'layer_index':layer,'fa_index':layer//4,'offsets':(1,),
                'queries':query[:,:,0,:],'keys':keys[:,:,-1,:],'values':values[:,:,-1,:],
                'native_attention':query[:,:,0,:]+np.float32(.125),'hidden_dtype':'float32'})
        branch=SimpleNamespace(layers=[object() for _ in range(16)])
        with patch.object(probe_module,'_export_logical_native_kv',return_value=(keys[0],values[0])):
            result=probe.compare((branch,),mx=mx)
        self.assertEqual(result['actual_ordinary_calls'],16)
        for row in result['details']:
            self.assertTrue(row['same_QKV_exact']);self.assertTrue(row['same_QKV_native_vs_stock_difference'])
            self.assertEqual(row['metrics']['native_vs_same_QKV_stock_replay']['max_abs'],.125)
            self.assertTrue(row['metrics']['stock_replay_vs_actual_ordinary_attention']['exact'])
        mlx_modules_after = {
            name for name in sys.modules if name == 'mlx' or name.startswith('mlx.')
        }
        self.assertEqual(mlx_modules_after, mlx_modules_before)
    def test_runner_baseline_last_token_projection_and_optional_probe(self):
        source=(ROOT/'scripts/research/varlen_hybrid_b1_numeric_gate.py').read_text()
        self.assertIn('logits=model.logits(hidden[:,-1:,:])[:,-1,:]',source)
        self.assertIn('last-token-projection bootstrap baseline is not exact',source)
        self.assertIn('if args.fa_probe and width==1:',source)
        self.assertIn('probe.capture_ordinary(attention_module)',source)
        self.assertIn('probe.compare(tuple(branches),mx=mx)',source)
        self.assertIn('candidate._fa_boundary_probe=None',source)
        self.assertNotIn('logit_atol',source)

if __name__=='__main__':unittest.main()
