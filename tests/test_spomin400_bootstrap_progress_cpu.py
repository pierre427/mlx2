"""Real Job/serving stall and grant laws plus known failed-native retirement."""
import ast,sys,threading,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from concurrent.futures import Future
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_spomin400_sampling_contract_cpu import source_functions,Job,ROOT
import spomin_400case_phased_http_bench as B
from mlx2.runtime.paged_n20_phase_control import PhaseController
import time
ENV=source_functions('src/mlx2/serving.py',{'record_native_bootstrap_progress','unmaterialized_lane_bytes'},{'time':time})
mark=ENV['record_native_bootstrap_progress'];grants=ENV['unmaterialized_lane_bytes']
TREE=ast.parse((ROOT/'src/mlx2/serving.py').read_text())
STALL=next(n for n in ast.walk(TREE) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='stalled' for t in n.targets) and isinstance(n.value,ast.ListComp))
class Tests(unittest.TestCase):
 def jobs(self):
  jobs=[Job({'max_tokens':192}) for _ in range(20)]
  for uid,j in enumerate(jobs):j.uid=uid;j.last_progress=100.;j.admission_reserved_gib=1.5
  return jobs
 def stalled(self,jobs,now):
  env=dict(active={j.uid:j for j in jobs},now=now,stall_seconds=30.)
  exec(compile(ast.Module(body=[STALL],type_ignores=[]),'actual product watchdog','exec'),env)
  return env['stalled']
 def test_actual_job_long_bootstrap_stall_reset_keeps_all_reserved_bytes(self):
  jobs=self.jobs();before=grants(jobs)
  self.assertEqual(len(self.stalled(jobs,500.)),20)
  mark(tuple(jobs),threading.RLock(),clock=lambda:500.)
  self.assertEqual(self.stalled(jobs,501.),[]);self.assertEqual(grants(jobs),before)
  # Normal decode no-progress watchdog still fires; no timeout waiver.
  self.assertEqual(len(self.stalled(jobs,531.)),20)
  # Actual first-sample reservation release, then EOS shrink and full retirement.
  release=next(n for n in ast.walk(TREE) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and ast.unparse(t)=='job.admission_reserved_gib' for t in n.targets) and isinstance(n.value,ast.Constant) and n.value.value==0.0)
  for job in jobs:exec(compile(ast.Module(body=[release],type_ignores=[]),'actual first sample grant release','exec'),{'job':job})
  self.assertEqual(grants(jobs),0);self.assertEqual(grants(jobs[1:]),0);self.assertEqual(grants([]),0)
 def test_cancelled_last_member_never_publishes_partial_progress(self):
  jobs=self.jobs();jobs[-1].cancelled.set()
  with self.assertRaises(ValueError):mark(tuple(jobs),threading.RLock(),clock=lambda:500.)
  self.assertTrue(all(j.last_progress==100. for j in jobs))
 def test_progress_only_after_successful_atomic_attachment(self):
  fn=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='install_explicit_native_hybrid_n_cohort')
  calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call)]
  attach=next(n for n in calls if ast.unparse(n.func)=='batch.install_native_hybrid_n_cohort')
  advance=next(n for n in calls if ast.unparse(n.func)=='record_native_bootstrap_progress')
  self.assertGreater(advance.lineno,attach.lineno)
  # It remains within guarded try, so failed constructors never execute it.
  guard=next(n for n in fn.body if isinstance(n,ast.Try))
  self.assertIn(advance,list(ast.walk(ast.Module(body=guard.body,type_ignores=[]))))
 def failed(self):
  s=B.Service(NS());owner={'session':'s','lease_id':'fresh'};s.verify=lambda o:None;s.controller=PhaseController(s.verify);s.controller.grant(owner,4)
  s.domain_error='postbootstrap429';s.outputs=[(429,{})]*20;s.domain_futures=[]
  for _ in range(20):f=Future();f.set_result((429,{}));s.domain_futures.append(f)
  s.engine=NS(lock=threading.RLock(),jobs={},pending_cohorts={},queued_jobs=0,incoming=NS(empty=lambda:True));s.resources=NS(_CHARGED=0);s.initial_charge=0
  owners=tuple(NS(fully_retired=True) for _ in range(20));writer=NS(pending_epochs=(),ledger=NS(pending_count=0),pool=NS(allocated_count=0))
  resources=NS(closed=True,reap=lambda:True)
  candidate=NS(backend=NS(writer=writer,_orphaned_reads=[]),_serving_resources=resources)
  s.captures=[dict(candidate=candidate,owners=owners)];return s,owner,candidate,owners
 def test_known_terminal_postbootstrap_failure_seals_then_owned_shutdown(self):
  s,o,c,owners=self.failed();s.current={'arm':'native'};s.seal_unstarted_failure(o)
  self.assertTrue(s.controller.paused);self.assertTrue(s.result['failed_command_retirement_proof']['native_cleanup_proved'])
  self.assertEqual(s.controller.events[-1]['layer_kind'],'failed_domain_retirement')
  with patch.object(B,'bounded_cleanup',return_value={'retained':False,'engine_closed':True}):s.shutdown(o)
  self.assertTrue(s.closed)
 def test_postbootstrap_pending_owner_charge_or_jobs_never_seals(self):
  for edit in ('epoch','read','owner','charge','job','open'):
   s,o,c,owners=self.failed()
   if edit=='epoch':c.backend.writer.pending_epochs=(1,)
   if edit=='read':c.backend._orphaned_reads=[object()]
   if edit=='owner':owners[-1].fully_retired=False
   if edit=='charge':s.resources._CHARGED=1
   if edit=='job':s.engine.jobs[1]=object()
   if edit=='open':c._serving_resources.closed=False
   with self.subTest(edit=edit),self.assertRaises(RuntimeError):s.seal_unstarted_failure(o)
   self.assertFalse(s.controller.paused)
if __name__=='__main__':unittest.main()
