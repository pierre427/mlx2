"""Bounded root lease handoff and real host N graph failure/shrink contracts."""
import sys,threading,time
from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import nullcontext
from unittest.mock import patch
import unittest
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mlx2.runtime.paged_n20_phase_control import PhaseController
from mlx2.runtime import paged_native_graph_n as G
class Tests(unittest.TestCase):
    def test_real_prompt_lookup_source_builds_the_scheduler_token_row(self):
        c=NS(_native_ragged_prompt_lookup=({
            'sources':['prompt_lookup'],'limit':1,'ngram_min':3,'ngram_max':3},2))
        lane=NS(tokens=[1,2,3,4,1,2],_pending_token=3,
                maximum=10,count=0,processors=[])
        rows,receipts=G._prompt_lookup_token_rows((lane,),c)
        self.assertEqual(rows,((3,4,1),))
        self.assertEqual(receipts[0]['source'],'prompt_lookup')
        self.assertEqual(receipts[0]['proposal_distribution'],
                         'deterministic_point_mass')
        self.assertEqual((receipts[0]['ngram_min'],receipts[0]['ngram_max']),(3,3))

    def test_private_evaluated_boundary_pauses_until_fresh_lease(self):
        seen=[];p=PhaseController(lambda owner:seen.append(owner['lease_id']));p.grant({'session':'s','lease_id':'1'},1)
        done=[]
        worker=threading.Thread(target=lambda:(p.boundary(dict(materialized=True,native_terminals_drained=True,public_state_published=False)),done.append(1),p.finish()))
        worker.start();proof=p.wait_quantum(1)
        self.assertTrue(proof['paused']);self.assertFalse(done)
        with self.assertRaises(RuntimeError):p.grant({'session':'s','lease_id':'1'},1)
        p.grant({'session':'s','lease_id':'2'},1);worker.join(1)
        self.assertEqual(done,[1]);self.assertTrue(p.finished)
    def test_completed_private_gate_releases_blocked_generator_without_graph(self):
        p=PhaseController(lambda owner:None);p.grant({'session':'s','lease_id':'1'},1);done=[]
        t=threading.Thread(target=lambda:(p.boundary(dict(materialized=True,native_terminals_drained=True,public_state_published=True)),done.append(1)))
        t.start();p.wait_quantum(1);p.release_completed({'session':'s','lease_id':'1'});t.join(1)
        self.assertEqual(done,[1]);self.assertTrue(p.finished)
    def test_completion_wakes_quantum_with_unused_budget_and_no_next_step(self):
        p=PhaseController(lambda owner:None);owner={'session':'s','lease_id':'1'};p.grant(owner,4);complete=[]
        def http_completion():
            time.sleep(.01);complete.append(1);p.wake()
        t=threading.Thread(target=http_completion);t.start()
        state=p.wait_quantum(1,completion=lambda:bool(complete));self.assertFalse(state['paused'])
        p.seal_completed(owner,dict(materialized=True,native_terminals_drained=True,product_idle=True));self.assertTrue(p.paused)
        p.release_completed(owner);t.join(1)
    def test_completion_refuses_pending_product_or_ambiguous_terminal(self):
        p=PhaseController(lambda owner:None);owner={'session':'s','lease_id':'1'};p.grant(owner,4)
        with self.assertRaises(RuntimeError):p.seal_completed(owner,dict(materialized=True,native_terminals_drained=True,product_idle=False))
        self.assertFalse(p.paused)
    def test_ambiguous_terminal_never_yields(self):
        p=PhaseController(lambda owner:None);p.grant({'session':'s','lease_id':'1'},1)
        with self.assertRaises(RuntimeError):p.boundary(dict(materialized=True,native_terminals_drained=False,public_state_published=False))
        self.assertFalse(p.paused);self.assertEqual(p.events,[])
    def test_cancel_only_terminal_paused_private_roots(self):
        p=PhaseController(lambda owner:None)
        errors=[]
        def run():
            try:p.enter()
            except RuntimeError:errors.append('cancel')
        t=threading.Thread(target=run);t.start();p.cancel({'session':'s','lease_id':'cancel'});t.join(1);self.assertEqual(errors,['cancel'])
    def test_prime_twenty_then_singleton_never_advances_finished_peer(self):
        c=NS(_serving_n20=True)
        calls=[]
        def lane(i):
            response=NS(mtp_receipt={});return NS(candidate=c,closed=False,research_only=False,owner=NS(supported_planes=('kv','gdn')),_first_logits=object(),_pending_token=None,next=lambda:(calls.append(i) or response))
        lanes=tuple(lane(i) for i in range(20))
        with patch.object(G,'bootstrap_attribution',return_value={'prefill_cohort_width':20}):
            rows=G.run_native_graph_n(lanes);self.assertTrue(all(r.execution_width==20 for r in rows))
            rows=G.run_native_graph_n((lanes[-1],));self.assertEqual(rows[0].execution_width,1)
        self.assertEqual(calls,list(range(20))+[19])
    def test_prepare_failure_rolls_all_branches_without_publication(self):
        events=[];backend=NS(read_submissions=0,terminal_successes=0,staged_read_spans=[])
        c=NS(_serving_n20=True,native_layer_count=1,backend=backend,reserve_serving_scratch=lambda n:None,packed_lane=lambda t,b:(t,b))
        def forward(lanes,branches,**kw):
            backend.read_submissions+=1;backend.terminal_successes+=1;backend.staged_read_spans.append(3);return [object()]*3,{'packed_lanes':3}
        c.forward_staged=forward;lanes=[]
        for i in range(3):
            state=NS(publish=lambda:events.append('publish'),rollback=lambda:events.append('state_rollback'))
            def prepare(n,i=i,state=state):
                if i==1:raise RuntimeError('middle prepare')
                return state
            branch=NS(prepare=prepare,rollback=lambda:events.append('branch_rollback'))
            owner=NS(supported_planes=('kv','gdn'),_reuse_private_tail=True,snapshot=lambda:nullcontext(NS(revision='r',offset=1,layer_owners=[object()],generation=0)),begin=lambda req,branch=branch:branch)
            lanes.append(NS(uid=i,revision='r',tokens=[1],candidate=c,closed=False,research_only=False,owner=owner,_first_logits=None,_pending_token=2))
        with self.assertRaisesRegex(RuntimeError,'middle prepare'):G.run_native_graph_n(tuple(lanes))
        self.assertNotIn('publish',events);self.assertEqual(events.count('branch_rollback'),3);self.assertEqual(events.count('state_rollback'),1)
    def test_q1_uses_shared_layout_and_labels_ordinary_publication(self):
        events=[];backend=NS(read_submissions=0,terminal_successes=0,staged_read_spans=[])
        c=NS(_serving_n20=True,native_layer_count=1,backend=backend,
             reserve_serving_scratch=lambda n:None,
             packed_lane=lambda tokens,branch:(tokens,branch))
        def forward(lanes,branches,**kw):
            backend.read_submissions+=1;backend.terminal_successes+=1
            backend.staged_read_spans.append(2)
            return [object(),object()],{'packed_lanes':2}
        c.forward_staged=forward;lanes=[]
        for i in range(2):
            state=NS(publish=lambda:events.append('publish'),rollback=lambda:events.append('state_rollback'))
            branch=NS(prepare=lambda n,state=state:state,rollback=lambda:events.append('branch_rollback'))
            owner=NS(supported_planes=('kv','gdn'),_reuse_private_tail=True,
                     snapshot=lambda:nullcontext(NS(revision='r',offset=1,layer_owners=[object()],generation=7)),
                     begin=lambda req,branch=branch:branch,reap_retired=lambda:None)
            response=NS(mtp_receipt={})
            lanes.append(NS(uid=i,revision='r',tokens=[1],candidate=c,closed=False,
                            research_only=False,owner=owner,_first_logits=None,
                            _pending_token=2,native_read_calls=0,terminal_successes=0,
                            _next_with_reader=lambda *a,response=response,**k:response))
        with patch.object(G,'bootstrap_attribution',return_value={'prefill_cohort_width':2}), \
             patch.object(G,'publish_native_cohort',side_effect=lambda states:[s.publish() for s in states]):
            rows=G.run_native_graph_n(tuple(lanes))
        receipt=rows[0].mtp_receipt['ragged_verify_layout']
        self.assertEqual(receipt['query_lengths'],[1,1])
        self.assertEqual(receipt['backend_mode'],'flattened')
        self.assertEqual(receipt['execution'],'ordinary_continuation_query')
        self.assertFalse(rows[0].mtp_receipt['speculative_verification'])
        self.assertEqual(events,['publish','publish'])
    def test_k_positive_runs_shrinking_online_cohort_and_publishes_prefixes(self):
        touched=[];events=[]
        backend=NS(read_submissions=0,terminal_successes=0,staged_read_spans=[])
        c=NS(_serving_n20=True,native_layer_count=1,backend=backend,
             reserve_serving_scratch=lambda n:None,
             packed_lane=lambda tokens,branch:(tokens,branch))
        def forward(layout,packed,branches,*,decide_next,**kw):
            logits=mx.zeros((2,32));backend.read_submissions+=1
            backend.terminal_successes+=1;backend.staged_read_spans.append(2)
            self.assertEqual(decide_next((0,1),0,logits),(True,False))
            logits2=mx.zeros((1,32));backend.read_submissions+=1
            backend.terminal_successes+=1;backend.staged_read_spans.append(1)
            self.assertEqual(decide_next((0,),1,logits2),(False,))
            return (logits,logits2),{
                'executed_query_lengths':[2,1],'round_widths':[2,1],
                'physical_counters':{}}
        c.forward_staged_ragged=forward;lanes=[]
        sampled={1:[11,12],2:[7]}
        for uid in (1,2):
            state=NS(publish=lambda uid=uid:events.append(('publish',uid)),
                     rollback=lambda:events.append(('state_rollback',uid)))
            branch=NS(prepare=lambda n,state=state:state,
                      rollback=lambda:events.append(('branch_rollback',uid)))
            owner=NS(supported_planes=('kv','gdn'),_reuse_private_tail=True,
                     snapshot=lambda uid=uid:(touched.append(('snapshot',uid)) or
                         nullcontext(NS(revision='r',offset=1,
                                        layer_owners=[object()],generation=4))),
                     begin=lambda req,branch=branch,uid=uid:(
                         touched.append(('begin',uid,req.proposed_rows)) or branch),
                     reap_retired=lambda uid=uid:events.append(('reap',uid)))
            matcher=__import__('mlx2.runtime.generate',fromlist=['StopSequenceMatcher']).StopSequenceMatcher()
            def stage(logits,context,uid=uid):
                token=sampled[uid].pop(0)
                return mx.array([token]),mx.zeros((1,32))
            def response(_generation,_offset,_sampled,_logprobs,uid=uid):
                return NS(uid=uid,mtp_receipt={},finish_reason=None)
            lanes.append(NS(uid=uid,revision='r',tokens=[0],candidate=c,
                closed=False,research_only=False,owner=owner,_first_logits=None,
                _pending_token=10 if uid==1 else 20,native_read_calls=0,
                terminal_successes=0,maximum=10,count=0,processors=[],
                matcher=matcher,matcher_state=matcher.make_state(),
                _stage_sample_for_context=stage,_response_from_sample=response))
        with patch.object(G,'bootstrap_attribution',return_value={}), \
             patch.object(G,'publish_native_cohort',side_effect=lambda states:[s.publish() for s in states]):
            rows=G.run_native_ragged_verify(tuple(lanes),((10,11),(20,)))
        self.assertEqual(len(rows),3)
        self.assertEqual(lanes[0].tokens,[0,10,11]);self.assertEqual(lanes[1].tokens,[0,20])
        self.assertEqual(touched,[('snapshot',1),('snapshot',2),('begin',1,2),('begin',2,1)])
        self.assertEqual([r.execution_width for r in rows],[2,2,1])
        self.assertTrue(all(r.mtp_receipt['speculative_verification'] for r in rows))
        self.assertEqual(events[:2],[('publish',1),('publish',2)])
if __name__=='__main__':unittest.main()
