"""Original independent ordinary admission and explicit fixed wave selection."""
import argparse,ast,importlib.abc,json,sys,time,unittest
from collections import defaultdict
from pathlib import Path
from threading import RLock
from types import SimpleNamespace as NS
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/research')]
INPUTS=Path('/tmp/mlx2-spomin400-nativeN-inputs.json')
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native':raise RuntimeError('runtime import forbidden')
sys.meta_path.insert(0,Guard())
import spomin_400case_native_suite as S
import spomin_400case_phased_http_bench as B
class Tests(unittest.TestCase):
 @pytest.mark.skipif(not INPUTS.is_file(),reason='requires retained spomin400 native input evidence')
 def test_original400_body_fields_preserved_ordinary_has_no_atomic_cohort(self):
  data=S.validate_inputs(json.loads(Path('/tmp/mlx2-spomin400-nativeN-inputs.json').read_text()))
  for row in data['rows']:
   for native in (False,True):
    b=S.request_body(row,model='m',native=native,cohort_id='domain',inputs_sha256=data['inputs_sha256'])
    for key,value in row['body'].items():self.assertEqual(b[key],value)
    self.assertEqual('batch_cohort' in b,native);self.assertIs(b['skip_writing_prefix_cache'],True)
 def test_actual_engine_publication_keeps20_ordinary_requests_independent(self):
  tree=ast.parse((ROOT/'src/mlx2/serving.py').read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_publish_job')
  env={'time':time,'Overloaded':RuntimeError};exec(compile(ast.Module(body=[node],type_ignores=[]),'actual engine publication','exec'),env)
  calls=[];engine=NS(max_lanes=20,pending_cohorts={},jobs={},lock=RLock(),counts=defaultdict(int),_publish_jobs=lambda jobs,**kw:calls.append((tuple(jobs),kw)))
  row={'body':{'messages':[{'role':'user','content':'actual'}],'max_tokens':192,'temperature':0,'enable_thinking':False},'case_id':'actual'}
  for uid in range(20):env['_publish_job'](engine,NS(request=S.request_body(row,model='m',native=False,cohort_id='x'),tenant_id='tenant',id=str(uid)))
  self.assertEqual([len(jobs) for jobs,_ in calls],[1]*20);self.assertFalse(engine.pending_cohorts)
  calls.clear()
  for uid in range(20):env['_publish_job'](engine,NS(request=S.request_body(row,model='m',native=True,cohort_id='x',inputs_sha256='a'*64),tenant_id='tenant',id=str(uid)))
  self.assertEqual([len(jobs) for jobs,_ in calls],[20]);self.assertFalse(engine.pending_cohorts)
 def test_wave_cli_defaults1_and_explicit_choices_only(self):
  tree=ast.parse((ROOT/'scripts/research/spomin_400case_phased_http_bench.py').read_text())
  call=next(n for n in ast.walk(tree) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and n.value.args and isinstance(n.value.args[0],ast.Constant) and n.value.args[0].value=='--gdn-eval-wave-max-segments')
  parser=argparse.ArgumentParser();exec(compile(ast.Module(body=[call],type_ignores=[]),'actual wave CLI','exec'),{'parser':parser})
  self.assertEqual(parser.parse_args([]).gdn_eval_wave_max_segments,1)
  for wave in (1,2,4):self.assertEqual(parser.parse_args(['--gdn-eval-wave-max-segments',str(wave)]).gdn_eval_wave_max_segments,wave)
  init=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='init')
  for name in ('factory.environment','factory.make_profile'):
   calls=[n for n in ast.walk(init) if isinstance(n,ast.Call) and ast.unparse(n.func)==name]
   self.assertEqual(len(calls),1);self.assertEqual(ast.unparse(calls[0].keywords[0].value),'self.args.gdn_eval_wave_max_segments')
 def test_event_metrics_report_actual_widths_not_client_concurrency(self):
  row=dict(case_id='x',domain='d',body_sha256='h',prompt_tokens=20,sentinel='ok',needles={},expected_concepts=[])
  body={'usage':{'completion_tokens':2,'prompt_tokens':20},'choices':[{'finish_reason':'stop','message':{'content':'ok'}}]}
  events={'x':[{'token':1,'monotonic_ns':2_000_000_000,'width':3},{'token':2,'monotonic_ns':3_000_000_000,'width':1}]}
  summary=B.summarize([row],[(200,body)],events,{'x':1.},4.)
  self.assertEqual(summary['observed_compute_widths'],[1,3]);self.assertEqual(summary['observed_peak_compute_width'],3)
if __name__=='__main__':unittest.main()
