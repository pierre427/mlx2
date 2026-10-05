"""Real phase controller accounting and source-bound event rates; no MLX."""
import ast,importlib.abc,sys,threading,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native':raise RuntimeError('runtime import forbidden')
sys.meta_path.insert(0,Guard());sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts/research'))
from mlx2.runtime import paged_n20_phase_control as C
import spomin_400case_phased_http_bench as B
class Tests(unittest.TestCase):
 def test_actual_pause_gap_and_fresh_resume_excluded_from_clock(self):
  tick=[1000.];fake=NS(monotonic=lambda:tick[0],monotonic_ns=lambda:int(tick[0]*1e9))
  c=C.PhaseController(lambda owner:None)
  with patch.object(C,'time',fake):
   c.grant({'session':'s','lease_id':'first'},1);tick[0]=1004.
   worker=threading.Thread(target=lambda:c.boundary(dict(materialized=True,native_terminals_drained=True,public_state_published=False)))
   worker.start()
   with c.condition:self.assertTrue(c.condition.wait_for(lambda:c.paused,1))
   tick[0]=1104.;paused=c.sample_clock();self.assertEqual(paused['owned_elapsed_seconds'],4.)
   self.assertTrue(paused['paused']);self.assertEqual(paused['lease_owner']['lease_id'],'first')
   c.grant({'session':'s','lease_id':'second'},1);worker.join(1);self.assertFalse(worker.is_alive())
   tick[0]=1106.;live=c.sample_clock();self.assertEqual(live['owned_elapsed_seconds'],6.)
   self.assertFalse(live['paused']);self.assertEqual(live['lease_owner']['lease_id'],'second')
   c.finish();tick[0]=2000.;self.assertEqual(c.sample_clock()['owned_elapsed_seconds'],6.)
 def test_event_wall_and_owned_rates_are_both_retained(self):
  row=dict(case_id='x',domain='d',body_sha256='a',prompt_tokens=7000,sentinel='audit',needles={},expected_concepts=[])
  body=dict(usage=dict(completion_tokens=2,prompt_tokens=7000),choices=[dict(finish_reason='stop',message=dict(content='audit'))])
  events={'x':[dict(token=1,monotonic_ns=102_000_000_000,owned_elapsed_seconds=2.,width=20),dict(token=2,monotonic_ns=203_000_000_000,owned_elapsed_seconds=3.,width=1)]}
  result=B.summarize([row],[(200,body)],events,{'x':100.},204.,owned_starts={'x':0.},owned_ended=4.)
  self.assertEqual(result['rows'][0]['decode_first_to_last_wall_seconds'],101.)
  self.assertEqual(result['rows'][0]['decode_first_to_last_owned_elapsed_seconds'],1.)
  self.assertEqual(result['aggregate_decode_tokens_per_owned_elapsed_second'],1.)
  self.assertEqual(result['aggregate_decode_tokens_per_second'],1/101.)
  self.assertEqual(result['prefill_prompt_tokens_per_owned_elapsed_second'],3500.)
  self.assertEqual(result['complete_owned_elapsed_seconds'],4.)
  self.assertEqual(result['http_domain_wall_seconds'],104.)
  self.assertEqual(result['rows'][0]['sample_events'],events['x'])
  self.assertIn('not device-only',result['owned_rate_scope'])
 def test_driver_ordinary_default_and_actual_samples_use_controller_clock(self):
  tree=ast.parse((ROOT/'scripts/research/spomin_400case_phased_http_bench.py').read_text())
  calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
  engine=next(n for n in calls if ast.unparse(n.func)=='serving.ServingEngine')
  self.assertEqual(ast.unparse(next(k.value for k in engine.keywords if k.arg=='prefill_step')),'self.args.prefill_step')
  option=next(n for n in calls if ast.unparse(n.func)=='parser.add_argument' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='--prefill-step')
  self.assertIsNone(next(k.value.value for k in option.keywords if k.arg=='default'))
  self.assertGreaterEqual(sum(ast.unparse(n.func)=='self.controller.sample_clock' for n in calls),3)
  source=ast.unparse(tree)
  self.assertIn('source_commit=self.args.expected_source',source)
  self.assertIn('prefill_step_selected=self.engine.prefill_step',source)
if __name__=='__main__':unittest.main()
