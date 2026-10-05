"""Pure planner and actual source-method wave contracts, no MLX/native runtime."""
import ast
import importlib.abc
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import sys
import numpy as np
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native':raise RuntimeError('runtime forbidden')
sys.meta_path.insert(0,Guard())
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('life_tests',ROOT/'tests/test_packed_prefill_lifetime_cpu.py')
T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T);L=T.L
guard_spec=importlib.util.spec_from_file_location('mlx2.runtime.models.import_env',ROOT/'src/mlx2/runtime/models/import_env.py')
IMPORT=importlib.util.module_from_spec(guard_spec);guard_spec.loader.exec_module(IMPORT)
COUNTS=(6982,6961,6940,6966,6982,7009,6977,6954,6964,6974,6973,6993,6975,6974,6974,7010,6977,6989,6980,6977)
BOUND=sum(COUNTS)*124928
ENV={'MLX_GDN_PACKED':'1','MLX_GDN_CORE':'0','MLX2_PAGED_GDN_EVAL_WAVE_MAX_SEGMENTS':'4'}

def configured_candidate(wave=4):
    c=T.candidate();c._prefill_gdn_eval_wave_max_segments=wave
    for layer in c.trunk.layers:
        layer.linear_attn.conv1d=NS(groups=10240,stride=1,dilation=1,padding=0,weight=NS(shape=(10240,4,1),dtype='bfloat16'))
        layer.linear_attn.norm=NS(weight=NS(dtype='bfloat16'))
    return c

def model_run(plan=None,fail_core=None,fail_eval=False):
    selected=T.method('src/mlx2/runtime/models/qwen3_5.py','GatedDeltaNet','mixed_materialized',{'mx':NS(concatenate=np.concatenate)})
    x=np.arange(sum(COUNTS)*3,dtype=np.float32).reshape(1,sum(COUNTS),3)/1000000
    calls=[];evaluations=[];caches=[NS(cache=[None,None],speculating=False) for _ in COUNTS]
    def core(q,z,b,a,mask,c,*,dtype):
        if fail_core is not None and len(calls)==fail_core:raise ValueError('core graph construction failure')
        calls.append((q.shape,z.shape,b.shape,a.shape,dtype,mask));c.cache=[q[:,-1:].copy(),q.sum(axis=1)]
        return q+z+b+a
    def evaluate(*v):
        evaluations.append(v)
        if fail_eval and len(evaluations)==2:raise ValueError('wave evaluation failure')
    owner=L.StageMaterialization(NS(eval=evaluate),BOUND,gdn_wave_plan=plan)
    module=NS(sharding_group=None,_input_projections=lambda x:(x*2,x*3,x*4,x*5),_recurrent_core=core,out_proj=lambda x:x*6)
    parts=[];start=0
    for n,cache in zip(COUNTS,caches):parts.append((1,n,start,cache,None));start+=n
    try:return selected(module,x,parts,materialize=owner),caches,calls,owner
    except BaseException as exc:return exc,caches,calls,owner

class Tests(unittest.TestCase):
    def test_four_waves_fit_exact_existing_charge_two_waves_have_more_margin(self):
        p=L.gdn_evaluation_wave_plan(COUNTS,BOUND,4)
        self.assertEqual(p['groups'],tuple(tuple(range(i,i+4)) for i in range(0,20,4)))
        self.assertLessEqual(p['grouped_peak_bytes'],BOUND)
        self.assertGreater(BOUND-p['grouped_peak_bytes'],234<<20)
        p2=L.gdn_evaluation_wave_plan(COUNTS,BOUND,2)
        self.assertEqual(len(p2['groups']),10);self.assertLess(p2['grouped_peak_bytes'],p['grouped_peak_bytes'])
    def test_excess_width_bool_and_insufficient_charge_refused(self):
        for width in (0,3,5,True):
            with self.assertRaises(ValueError):L.gdn_evaluation_wave_plan(COUNTS,BOUND,width)
        with self.assertRaises(MemoryError):L.gdn_evaluation_wave_plan(COUNTS,1,4)
        with self.assertRaises(MemoryError):L.gdn_evaluation_wave_plan(COUNTS[:2],sum(COUNTS[:2])*124928,2)
        # Fifth-segment coexistence exceeds the unchanged charge, even though
        # exposing a new width5 option would reduce more host calls.
        fifth=78016*sum(COUNTS)+222720*sum(COUNTS[:5])+6496256*5
        self.assertGreater(fifth,BOUND)
    def test_plan_covers_ragged_remainder_without_reordering(self):
        counts=COUNTS[:19];p=L.gdn_evaluation_wave_plan(counts,sum(counts)*124928,4)
        self.assertEqual(tuple(i for group in p['groups'] for i in group),tuple(range(19)))
        self.assertTrue(all(len(g)<=4 for g in p['groups']));self.assertLessEqual(p['grouped_peak_bytes'],p['activation_bound_bytes'])
    def test_same_projection_core_shapes_state_outputs_and_segment_order(self):
        serial=model_run();p=L.gdn_evaluation_wave_plan(COUNTS,BOUND,4);grouped=model_run(p)
        self.assertNotIsInstance(grouped[0],BaseException);np.testing.assert_array_equal(serial[0],grouped[0])
        self.assertEqual(serial[2],grouped[2]);self.assertEqual(len(grouped[2]),20)
        for a,b in zip(serial[1],grouped[1]):
            for x,y in zip(a.cache,b.cache):np.testing.assert_array_equal(x,y)
        self.assertEqual(serial[3].stages.count('gdn_segment'),20)
        self.assertEqual(grouped[3].stages.count('gdn_segment_wave'),5)
        self.assertEqual(grouped[3].failure_roots,())
    def test_pending_core_error_retains_prior_graphs_and_projection_roots(self):
        result,caches,calls,owner=model_run(L.gdn_evaluation_wave_plan(COUNTS,BOUND,4),fail_core=1)
        self.assertIsInstance(result,ValueError);self.assertEqual(len(calls),1)
        self.assertGreaterEqual(len(owner.failure_roots),7);self.assertEqual(owner.evaluations,1)
        self.assertTrue(any(v is caches[0].cache[1] for v in owner.failure_roots))
    def test_wave_eval_failure_retains_all_wave_output_and_state_roots(self):
        result,caches,calls,owner=model_run(L.gdn_evaluation_wave_plan(COUNTS,BOUND,4),fail_eval=True)
        self.assertIsInstance(result,ValueError);self.assertEqual(len(calls),4)
        self.assertEqual(len(owner.failure_roots),12);self.assertEqual(owner.evaluations,1)
        self.assertTrue(all(any(v is c.cache[1] for v in owner.failure_roots) for c in caches[:4]))
    def test_source_charge_unchanged_and_policy_refusal_precedes_allocation(self):
        original=L.n20_stagewise_charge_components(T.candidate(),COUNTS,(192,)*20)
        with patch.dict('os.environ',ENV,clear=True),patch.dict(sys.modules,{'mlx2.runtime.models.import_env':IMPORT}):
            self.assertEqual(original,L.n20_stagewise_charge_components(configured_candidate(),COUNTS,(192,)*20))
            c=configured_candidate();c.trunk.layers[0].linear_attn.conv1d.groups=1
            with self.assertRaises(ValueError):L.n20_stagewise_charge_components(c,COUNTS,(192,)*20)
        with patch.dict('os.environ',dict(ENV,MLX_GDN_CORE='1'),clear=True):
            with self.assertRaises(ValueError):L.n20_stagewise_charge_components(configured_candidate(),COUNTS,(192,)*20)
        with patch.dict('os.environ',dict(ENV,MLX2_PAGED_GDN_EVAL_WAVE_MAX_SEGMENTS='2'),clear=True):
            with self.assertRaises(ValueError):L.n20_stagewise_charge_components(configured_candidate(),COUNTS,(192,)*20)
    def test_late_import_environment_drift_refused_before_allocation(self):
        module='mlx2.runtime.models.gated_delta'
        IMPORT.snapshot(module,{'MLX_GDN_PACKED':'0','MLX_GDN_CORE':'0'})
        try:
            with patch.dict('os.environ',ENV,clear=True),patch.dict(sys.modules,{'mlx2.runtime.models.import_env':IMPORT,module:NS()}):
                with self.assertRaises(IMPORT.ImportOrderError):
                    L.n20_stagewise_charge_components(configured_candidate(),COUNTS,(192,)*20)
        finally:IMPORT._SNAPSHOTS.clear()
    def test_mask_and_segment_order_refused_before_any_projection(self):
        selected=T.method('src/mlx2/runtime/models/qwen3_5.py','GatedDeltaNet','mixed_materialized',{'mx':NS(concatenate=np.concatenate)})
        plan=L.gdn_evaluation_wave_plan(COUNTS,BOUND,4)
        owner=L.StageMaterialization(NS(eval=lambda *v:None),BOUND,gdn_wave_plan=plan)
        module=NS(sharding_group=None)
        parts=[];start=0
        for count in COUNTS:
            parts.append((1,count,start,NS(cache=[None,None],speculating=False),None));start+=count
        parts[0]=(*parts[0][:4],np.ones(COUNTS[0],dtype=bool))
        with self.assertRaisesRegex(ValueError,'order/mask'):selected(module,np.ones((1,1,3)),parts,materialize=owner)
        parts[0]=(*parts[0][:4],None);parts[0],parts[1]=parts[1],parts[0]
        with self.assertRaisesRegex(ValueError,'order/mask'):selected(module,np.ones((1,1,3)),parts,materialize=owner)
    def test_materializer_refuses_forged_memory_or_segment_plan(self):
        p=L.gdn_evaluation_wave_plan(COUNTS,BOUND,4)
        for bad in (dict(p,grouped_peak_bytes=0),dict(p,groups=((0,1),)),dict(p,activation_bound_bytes=BOUND+1)):
            with self.assertRaises(ValueError):L.StageMaterialization(NS(eval=lambda *v:None),BOUND,gdn_wave_plan=bad)

if __name__=='__main__':unittest.main()
