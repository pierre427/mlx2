"""Source-bound grouped-GDN policy reaches allocation and actual receipt gates."""
import runpy,sys,unittest,copy
from pathlib import Path
from types import SimpleNamespace as NS,ModuleType
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
A=runpy.run_path(str(ROOT/'tests/test_packed_n20_admission_cpu.py'),run_name='wave_admission_helpers')
W=runpy.run_path(str(ROOT/'tests/test_packed_gdn_evaluation_waves_cpu.py'),run_name='wave_policy_helpers')
N=A['N'];L=W['L'];IMPORT=W['IMPORT']

class WaveProfileTests(unittest.TestCase):
 def test_optional_selection_and_legacy_profile_environment(self):
  p=A['profile']();self.assertNotIn('gdn_eval_wave_max_segments',p)
  self.assertNotIn('MLX2_PAGED_GDN_EVAL_WAVE_MAX_SEGMENTS',N.environment())
  for value in (True,0,3,5,'4',4.0,None):
   with self.subTest(value=value),self.assertRaises(ValueError):N.environment(value)
  for wave in (2,4):
   env=N.environment(wave)
   self.assertEqual(env['MLX2_PAGED_GDN_EVAL_WAVE_MAX_SEGMENTS'],str(wave))
   self.assertEqual((env['MLX_GDN_PACKED'],env['MLX_GDN_CORE']),('1','0'))
   self.assertEqual({k:v for k,v in env.items() if k not in ('MLX2_PAGED_GDN_EVAL_WAVE_MAX_SEGMENTS','MLX_GDN_PACKED','MLX_GDN_CORE')},{k:v for k,v in N.environment().items() if k not in ('MLX_GDN_PACKED','MLX_GDN_CORE')})

 def test_profile_selector_env_and_source_identity_fail_closed(self):
  base=A['profile']()
  for wave in (1,2,4):
   p={**base,'gdn_eval_wave_max_segments':wave,'required_environment':N.environment(wave)}
   args=dict(live_identity=A['identity'](),source_input_ids=('case0','case1'),counts=(1025,1025),tokens=((0,)*1025,(1,)*1025),environment_values=N.environment(wave))
   self.assertIs(N.validate_profile(p,**args),p)
   for bad in ({**p,'gdn_eval_wave_max_segments':True},{**p,'extra':1},{**p,'identity':{**p['identity'],'source_commit':'c'*40}}):
    with self.assertRaises(ValueError):N.validate_profile(bad,**args)
   with self.assertRaises(ValueError):N.validate_profile(p,**{**args,'environment_values':{**N.environment(wave),'MLX2_PAGED_GDN_EVAL_WAVE_MAX_SEGMENTS':'4' if wave!=4 else '2'}})

 def measured_candidate(self):
  c=W['configured_candidate']();c.model=NS(parameters=lambda:[NS(nbytes=15132802048)])
  return c

 def test_complete_charge_policy_preallocation_and_actual_weight_bound(self):
  counts=(7000,)*20;caps=(192,)*20;c=self.measured_candidate()
  module=ModuleType('mlx.utils');module.tree_flatten=lambda values:list(enumerate(values))
  profile={**A['profile'](),'gdn_eval_wave_max_segments':4}
  with patch.dict('os.environ',N.environment(4),clear=True),patch.dict(sys.modules,{'mlx.utils':module,'mlx2.runtime.models.import_env':IMPORT}):
   baseline=L.n20_stagewise_charge_components(W['T'].candidate(),counts,caps)
   components,weights,total,process=N.model_memory_bound(c,counts,caps,profile,NS(device_info=lambda:{'memory_size':64<<30}))
   self.assertEqual(components,baseline);self.assertEqual(weights,15132802048)
   self.assertEqual(total,sum(components.values()));self.assertEqual(process,total+weights+(1<<30))
   self.assertEqual(c._prefill_gdn_eval_wave_max_segments,4)
   self.assertEqual(c._prefill_gdn_wave_plan,L.gdn_evaluation_wave_plan(counts,components['layer_activation_bytes'],4))
   with self.assertRaises(MemoryError):N.model_memory_bound(c,(6982,6961),(2,2),profile,NS(device_info=lambda:{'memory_size':64<<30}))
   with self.assertRaises(MemoryError):N.model_memory_bound(c,counts,caps,{**profile,'max_rss_bytes':1},NS(device_info=lambda:{'memory_size':64<<30}))

 def test_actual_completed_wave_and_all_segments_proof_before_attribution(self):
  counts=(7000,)*20;c=self.measured_candidate();components=L.n20_stagewise_charge_components(W['T'].candidate(),counts,(192,)*20)
  plan=L.gdn_evaluation_wave_plan(counts,components['layer_activation_bytes'],4)
  c._prefill_gdn_wave_plan=plan
  proof=dict(segment_lengths=counts,gdn_evaluation_wave_max_segments=4,gdn_evaluation_wave_plan=plan,gdn_wave_proof_kind='completed_host_evaluations_not_device_dispatches',gdn_segment_core_calls=960,materialized_stage_counts={'gdn_segment_wave':240})
  N.validate_gdn_wave_attribution(c,proof)
  for bad in ({**proof,'gdn_evaluation_wave_max_segments':1},{**proof,'gdn_segment_core_calls':959},{**proof,'gdn_wave_proof_kind':'device_dispatches'}, {**proof,'materialized_stage_counts':{'gdn_segment_wave':239}},{**proof,'gdn_evaluation_wave_plan':{**plan,'grouped_peak_bytes':0}}):
   with self.assertRaises(ValueError):N.validate_gdn_wave_attribution(c,bad)
  with self.assertRaises(ValueError):N.validate_gdn_wave_attribution(NS(),proof)

 def test_profile_generation_selects_explicitly_without_qualification(self):
  rows=[dict(case_id='case'+str(i),domain='one',prompt_tokens=1025,prompt_token_ids=[i]*1025) for i in range(400)]
  inputs=dict(case_count=400,client_concurrency=20,generation_max_tokens=192,corpus_sha256=A['profile']()['corpus_sha256'],inputs_sha256='c'*64,rows=rows)
  p=N.make_profile(A['identity'](),inputs,gdn_eval_wave_max_segments=4)
  self.assertEqual(p['gdn_eval_wave_max_segments'],4);self.assertEqual(p['required_environment'],N.environment(4))
  self.assertFalse(p['qualified']);self.assertFalse(p['serving_default']);self.assertFalse(p['price_usable'])

 def test_mlp_experiment_is_profile_bound_and_sets_every_layer(self):
  rows=[dict(case_id='case'+str(i),domain='one',prompt_tokens=1025,prompt_token_ids=[i]*1025) for i in range(400)]
  inputs=dict(case_count=400,client_concurrency=20,generation_max_tokens=192,corpus_sha256=A['profile']()['corpus_sha256'],inputs_sha256='c'*64,rows=rows)
  args=dict(live_identity=A['identity'](),source_input_ids=('case0','case1'),counts=(1025,1025),tokens=((0,)*1025,(1,)*1025),environment_values=N.environment())
  for mode in ('staged_qmm','single_eval_qmm','staged_bf16','single_eval_bf16','tiled_q4_swiglu','packed_gate_up_qmm'):
   p=N.make_profile(A['identity'](),inputs,mlp_materialization_mode=mode)
   if mode=='staged_qmm':self.assertNotIn('mlp_materialization_mode',p)
   else:self.assertEqual(p['mlp_materialization_mode'],mode)
   self.assertIs(N.validate_profile(p,**args),p)
  for bad in (None,True,'fast','bf16'):
   with self.assertRaises(ValueError):N.make_profile(A['identity'](),inputs,mlp_materialization_mode=bad)
  c=self.measured_candidate();module=ModuleType('mlx.utils');module.tree_flatten=lambda values:list(enumerate(values))
  p={**A['profile'](),'mlp_materialization_mode':'single_eval_bf16'}
  with patch.dict(sys.modules,{'mlx.utils':module}):
   components,_,_,_=N.model_memory_bound(c,(7000,)*3,(4,)*3,p,NS(device_info=lambda:{'memory_size':64<<30}))
  self.assertEqual(c._prefill_mlp_materialization_mode,'single_eval_bf16')
  self.assertTrue(all(layer.mlp._prefill_materialization_mode=='single_eval_bf16' for layer in c.trunk.layers))
  self.assertIn('mlp_dequantized_projection_bytes',components)
  c=self.measured_candidate();c._prefill_mlp_semantics={'semantic':'declared'}
  geometry={'schema':'mlx2.loaded-dense-glu-cohort.v1','layer_count':64,
            'geometry':{'sha256':'a'*64}}
  p={**A['profile'](),'mlp_materialization_mode':'tiled_q4_swiglu'}
  geometry_module=ModuleType('mlx2.runtime.dense_mlp_geometry')
  geometry_module.infer_uniform_dense_glu_geometry=lambda layers,semantics:geometry
  with patch.dict(sys.modules,{'mlx.utils':module,'mlx2.runtime.dense_mlp_geometry':geometry_module}):
   components,_,_,_=N.model_memory_bound(c,(7000,)*3,(4,)*3,p,NS(device_info=lambda:{'memory_size':64<<30}))
  self.assertNotIn('mlp_dequantized_projection_bytes',components)
  self.assertEqual(c._prefill_mlp_geometry,geometry)
  self.assertTrue(all(layer.mlp._prefill_mlp_geometry==geometry['geometry'] for layer in c.trunk.layers))

 def test_expanded_wave_charge_is_explicit_profile_policy(self):
  rows=[dict(case_id='case'+str(i),domain='one',prompt_tokens=1025,prompt_token_ids=[i]*1025) for i in range(400)]
  inputs=dict(case_count=400,client_concurrency=20,generation_max_tokens=192,corpus_sha256=A['profile']()['corpus_sha256'],inputs_sha256='c'*64,rows=rows)
  with self.assertRaises(ValueError):N.make_profile(A['identity'](),inputs,gdn_eval_wave_expanded_charge=True)
  p=N.make_profile(A['identity'](),inputs,gdn_eval_wave_max_segments=4,gdn_eval_wave_expanded_charge=True)
  self.assertIs(p['gdn_eval_wave_expanded_charge'],True)
  args=dict(live_identity=A['identity'](),source_input_ids=('case0','case1'),counts=(1025,1025),tokens=((0,)*1025,(1,)*1025),environment_values=N.environment(4))
  self.assertIs(N.validate_profile(p,**args),p)
  with self.assertRaises(ValueError):N.validate_profile(
   {**p,'gdn_eval_wave_max_segments':1,'required_environment':N.environment()},
   **{**args,'environment_values':N.environment()})

if __name__=='__main__':unittest.main()
