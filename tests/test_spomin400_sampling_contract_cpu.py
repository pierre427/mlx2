"""Real request/vendor/Job/processor/sample contracts without tensor imports."""
import ast,importlib.abc,json,queue,sys,threading,time,uuid,unittest
from dataclasses import dataclass,field
from collections import deque
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/research')]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native':raise RuntimeError('runtime import forbidden')
sys.meta_path.insert(0,Guard())
from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
from mlx2.sampling_defaults import resolve_sampling,vendor_sampling
from mlx2.runtime.paged_n20_request import validate_ready_job,validate_ordinary_processors
import spomin_400case_native_suite as S

def source_functions(path,names,env):
 tree=ast.parse((ROOT/path).read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])),str(path),'exec'),env)
 return env
ENV=source_functions('src/mlx2/serving.py',{'Job'},{'__name__':__name__,'dataclass':dataclass,'field':field,'queue':queue,'uuid':uuid,'time':time,'threading':threading})
Job=ENV['Job']
P=source_functions('src/mlx2/runtime/sample_utils.py',{'_probe_safe','_generated_window','make_presence_penalty'},{'__name__':'mlx2.runtime.sample_utils'})
make_presence=P['make_presence_penalty']
tree=ast.parse((ROOT/'src/mlx2/runtime/paged_native_continuation.py').read_text());stage=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_stage_sample')
mx=NS(array=lambda x,dtype:np.array(x,dtype=dtype),uint32=np.uint32,float32=np.float32,logsumexp=lambda x,axis,keepdims:np.log(np.exp(x).sum(axis=axis,keepdims=keepdims)))
E={'mx':mx};exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),stage],type_ignores=[])),'actual native sampler','exec'),E)
# Actual ordinary EOS trie and public Response record used by native sampling.
T=source_functions('src/mlx2/runtime/generate.py',{'_build_trie','_step_trie','StopSequenceMatcher'},{'deque':deque})
gtree=ast.parse((ROOT/'src/mlx2/runtime/generate.py').read_text())
gen=next(n for n in gtree.body if isinstance(n,ast.ClassDef) and n.name=='GenerationBatch')
response=next(n for n in gen.body if isinstance(n,ast.ClassDef) and n.name=='Response')
R={'__name__':__name__,'dataclass':dataclass}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),response],type_ignores=[])),'actual Response','exec'),R)
reply=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_response_from_sample')
RE={'StopSequenceMatcher':T['StopSequenceMatcher'],'GenerationBatch':NS(Response=R['Response']),'_invalid_output_reason':lambda *a:None}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),reply],type_ignores=[])),'actual native Response construction','exec'),RE)
class Tests(unittest.TestCase):
 def setUp(self):
  inputs=S.validate_inputs(json.loads(Path('/tmp/mlx2-spomin400-nativeN-inputs.json').read_text()));self.rows=S.domain_rows(inputs,0);self.inputs=inputs
 def job(self,row):
  body=S.request_body(row,model='real',native=True,cohort_id='actual0',inputs_sha256=self.inputs['inputs_sha256'])
  job=Job(body);job.uid=0;job.native_b2_prompt=tuple(row['prompt_token_ids']);job.effective_max_tokens=192
  job.effective_sampling,job.sampling_defaults=resolve_sampling(body,vendor_sampling(Qwen3827BAdapter),thinking=False)
  return job
 def test_actual_unmodified_twenty_jobs_resolve_vendor_presence_and_are_admitted(self):
  for row in self.rows:
   job=self.job(row);self.assertNotIn('presence_penalty',job.request);self.assertEqual(job.effective_sampling['presence_penalty'],1.5)
   self.assertEqual(job.sampling_defaults['profile'],'instruct');self.assertIs(validate_ready_job(job,20),job.effective_sampling)
   p=make_presence(1.5,0,generation_start=len(job.native_b2_prompt));self.assertIs(validate_ordinary_processors([p],job.effective_sampling,len(job.native_b2_prompt))[0],p)
 def test_named_actual_job_refusals(self):
  for name,change in [('effective_greedy',lambda j:j.effective_sampling.update(temperature=.5)),('neutral_frequency',lambda j:j.effective_sampling.update(frequency_penalty=1)),('not_cancelled',lambda j:j.cancelled.set()),('zero_cached_tokens',lambda j:setattr(j,'native_b2_cached_tokens',1)),('no_thinking_guard',lambda j:setattr(j,'thinking_guard',object()))]:
   job=self.job(self.rows[0]);change(job)
   with self.subTest(name=name),self.assertRaisesRegex(ValueError,name):validate_ready_job(job,20)
 def test_processor_shape_history_origin_and_binding_refuse_drift(self):
  job=self.job(self.rows[0]);length=len(job.native_b2_prompt)
  for ps in ([],[make_presence(1.5,0,generation_start=length-1)],[make_presence(1.,0,generation_start=length)],[lambda *a:None]):
   with self.assertRaisesRegex(ValueError,'queued-processor refusal'):validate_ordinary_processors(ps,job.effective_sampling,length)
 def test_actual_response_eos_usage_and_generator_width_shrink(self):
  from mlx2.runtime import paged_native_graph_n as graph
  matcher=T['StopSequenceMatcher']([[9]])
  candidate=NS(_serving_n20=True,serving_route='native_hybrid_packed_n20_research')
  lanes={}
  for uid in range(20):
   lane=NS(uid=uid,candidate=candidate,count=0,maximum=3 if uid==19 else 2,matcher=matcher,matcher_state=matcher.make_state(),tokens=[1]*1025,_pending_token=None,research_only=False,terminal_successes=16,native_read_calls=16,revision='r',apcv2_restored_tokens=0,price_provenance=None,lane_rng=None)
   lanes[uid]=lane
  # Execute the ACTUAL N selection/retirement arm from BatchGenerator.next.
  batchcls=next(n for n in gtree.body if isinstance(n,ast.ClassDef) and n.name=='BatchGenerator')
  nxt=next(n for n in batchcls.body if isinstance(n,ast.FunctionDef) and n.name=='next')
  arm=next(n for n in ast.walk(nxt) if isinstance(n,ast.If) and "_serving_n20" in ast.unparse(n.test))
  candidate_assign=next(n for n in ast.walk(nxt) if isinstance(n,ast.Assign) and
                        any(isinstance(t,ast.Name) and t.id=='candidate' for t in n.targets))
  def run(group):
   out=[]
   for lane in group:
    token=9 if lane.uid==0 else 3
    out.append(graph.decorate(RE['_response_from_sample'](lane,0,len(lane.tokens),np.array([token]),np.zeros((1,10))),candidate,len(group)))
   return tuple(out)
  retired=[];batch=NS(_native_continuations=dict(lanes),_retire_native_continuation=lambda lane:retired.append(lane.uid),_native_lane_failures=[])
  seen=[]
  with patch.object(graph,'run_native_graph_n',run),patch.object(graph,'bootstrap_attribution',return_value={'prefill_cohort_width':20}):
   for expected in (20,19,1):
    result=([],[]);processed=set();items=tuple(batch._native_continuations.items())
    for uid,lane in items:
     if uid in processed:continue
     # The source arm's continue needs its original for-loop wrapper.
     wrapper=ast.For(target=ast.Tuple(elts=[ast.Name(id='uid',ctx=ast.Store()),ast.Name(id='continuation',ctx=ast.Store())],ctx=ast.Store()),iter=ast.Name(id='items',ctx=ast.Load()),body=[ast.If(test=ast.Compare(left=ast.Name(id='uid',ctx=ast.Load()),ops=[ast.In()],comparators=[ast.Name(id='processed_native',ctx=ast.Load())]),body=[ast.Continue()],orelse=[]),candidate_assign,arm],orelse=[])
     env={'__package__':'mlx2.runtime','self':batch,'items':items,'native_items':items,'processed_native':processed,'result':result}
     exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),'actual BatchGenerator N arm','exec'),env)
     break
    self.assertEqual(len(result[1]),expected);self.assertTrue(all(r.execution_width==expected and r.mtp_receipt['route']=='native_hybrid_packed_n20_research' for r in result[1]))
    seen.extend(result[1])
  self.assertEqual(seen[0].finish_reason,'stop');self.assertEqual(lanes[0].count,1)
  self.assertEqual(lanes[19].count,3);self.assertFalse(batch._native_continuations);self.assertEqual(len(set(retired)),20)
  self.assertEqual(sum(l.count for l in lanes.values()),40) # actual EOS/length events, never20*192
 def test_real_native_sample_applies_same_existing_ordinary_generated_only_penalty(self):
  for width in (1,2,19,20):
   for lane in range(width):
    prompt=[2,2,3];generated=[] if lane==0 else [0,0,1]
    p=make_presence(1.5,0,generation_start=len(prompt));logits=np.array([2.,1.7,1.6,1.3],dtype=np.float32)
    ordinary=p(np.array(prompt+generated,dtype=np.uint32),logits[None].copy())
    obj=NS(tokens=prompt+generated,processors=[p],sampler=lambda x:np.argmax(x,axis=-1))
    token,logprobs=E['_stage_sample'](obj,logits.copy())
    self.assertEqual(int(token[0]),int(np.argmax(ordinary[0])))
    np.testing.assert_allclose(logprobs,ordinary-np.log(np.exp(ordinary).sum(axis=-1,keepdims=True)))
    self.assertEqual(float(ordinary[0,2]),float(logits[2])) # prompt token never penalized
if __name__=='__main__':unittest.main()
