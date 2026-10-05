"""Exercise actual server parser against frozen400 source requests; CPU only."""
import ast,importlib.abc,io,json,math,sys,threading,unittest
from collections.abc import Mapping
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from urllib.error import HTTPError
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/research')]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native':raise RuntimeError('runtime import forbidden')
sys.meta_path.insert(0,Guard())
import spomin_400case_native_suite as S
import spomin_400case_phased_http_bench as B
from mlx2.runtime.paged_n20_phase_control import PhaseController

def parser():
 tree=ast.parse((ROOT/'src/mlx2/server.py').read_text())
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('validate_request','normalize_client_options')]
 compat=ast.parse((ROOT/'src/mlx2/openai_compat.py').read_text());nodes += [n for n in compat.body if isinstance(n,ast.FunctionDef) and n.name=='normalize_tool_choice']
 env={'__package__':'mlx2','math':math,'Mapping':Mapping,'MAX_TOP_LOGPROBS':20,'MAX_OUTPUT_TOKENS':2_097_152}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual server validator','exec'),env)
 return env['validate_request']
class Tests(unittest.TestCase):
 def setUp(self):
  self.validate=parser();self.inputs=S.validate_inputs(json.loads(Path('/tmp/mlx2-spomin400-nativeN-inputs.json').read_text()));self.rows=S.domain_rows(self.inputs,0)
 def body(self,row,native=True):
  body=S.request_body(row,model='actualmodel',native=native,cohort_id='actual-domain0',inputs_sha256=self.inputs['inputs_sha256'])
  body.update(native_research_input_id=row['case_id'],native_research_inputs_sha256=self.inputs['inputs_sha256']);return body
 def test_all20_real_native_and_ordinary_bodies_reach_actual_parser(self):
  for row in self.rows:
   for native in (True,False):
    b=self.body(row,native);self.assertEqual(self.validate(b),b)
 def test_optin_metadata_and_geometry_fail_closed(self):
  for edit in ({'native_research_input_id':''},{'native_research_inputs_sha256':'bad'},{'native_research_input_id':3},{'paged_native_packed_n20_research':'1'},{'batch_cohort':{'id':'x','size':21}},{'max_tokens':193},{'enable_thinking':True},{'skip_writing_prefix_cache':False},{'paged_native_hybrid_b2':True}):
   with self.subTest(edit=edit),self.assertRaises(ValueError):self.validate({**self.body(self.rows[0]),**edit})
 def test_http_error_body_and_all_client_exceptions_collected(self):
  def opener(*a,**kw):raise HTTPError('http://localhost',400,'Bad Request',{},io.BytesIO(b'{"error":"unsupported field"}'))
  self.assertEqual(B.post_case(object(),opener),(400,{'error':'unsupported field'}))
  fs=[]
  for i in range(20):
   f=Future();f.set_exception(RuntimeError('case'+str(i))) if i%2 else f.set_result((400,{'error':'case'+str(i)}));fs.append(f)
  results=B.collect_client_results(fs);self.assertEqual(len(results),20);self.assertTrue(all(status in (0,400) for status,_ in results))
 def test_bad_request_preflight_never_grants_or_launches_http(self):
  service=B.Service(NS(model='model'));service.initialized=True;service.inputs=self.inputs
  service.validate_http_request=lambda body:(_ for _ in ()).throw(ValueError('explicit source contract failed'))
  with self.assertRaisesRegex(ValueError,'explicit source contract'):service.start({'domain_index':0,'arm':'native'},dict(session='s',lease_id='l'))
  self.assertIsNone(service.current);self.assertIsNone(service.controller)
 def test_ambiguous_failed_command_cannot_return_live_lease(self):
  tree=ast.parse((ROOT/'scripts/research/spomin_400case_phased_http_bench.py').read_text())
  serve=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='serve')
  text=ast.unparse(serve)
  self.assertIn('service.seal_unstarted_failure(owner)',text)
  self.assertIn('native_cleanup_proved=False',text)
  self.assertIn('os._exit(124)',text)
  self.assertLess(text.index('service.seal_unstarted_failure(owner)'),text.index('connection.sendall'))
 def service(self):
  s=B.Service(NS());owner={'session':'root','lease_id':'fresh'};s.verify=lambda o:None;s.controller=PhaseController(s.verify);s.controller.grant(owner,4)
  s.domain_error='HTTP400';s.outputs=[(400,{})]*20;s.domain_futures=[]
  for _ in range(20):f=Future();f.set_result((400,{}));s.domain_futures.append(f)
  s.engine=NS(lock=threading.RLock(),jobs={},pending_cohorts={},queued_jobs=0,incoming=NS(empty=lambda:True));s.resources=NS(_CHARGED=0);s.initial_charge=0
  return s,owner
 def test_failed_all20_preadmission_seals_before_cancel(self):
  s,o=self.service();s.seal_unstarted_failure(o);self.assertTrue(s.controller.paused);s.controller.cancel(o);self.assertTrue(s.controller.aborted)
 def test_all20_http400_owned_shutdown_retires_unentered_domain(self):
  s,o=self.service();s.current={'arm':'native'}
  with patch.object(B,'bounded_cleanup',return_value={'retained':False,'engine_closed':True}):
   result=s.shutdown(o)
  self.assertFalse(result['complete']);self.assertTrue(s.closed);self.assertTrue(s.controller.aborted)
  self.assertTrue(result['cleanup']['native_charge_restored'])
 def test_partial_admission_or_unsettled_future_never_seals(self):
  for edit in ('jobs','capture','future','charge'):
   s,o=self.service()
   if edit=='jobs':s.engine.jobs[1]=object()
   if edit=='capture':s.captures.append(object())
   if edit=='future':s.domain_futures[0]=Future()
   if edit=='charge':s.resources._CHARGED=1
   with self.subTest(edit=edit),self.assertRaises(RuntimeError):s.seal_unstarted_failure(o)
   self.assertFalse(s.controller.paused)
if __name__=='__main__':unittest.main()
