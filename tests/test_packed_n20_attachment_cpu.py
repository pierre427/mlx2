"""Actual generator N attachment body, middle/last failure and cancellation."""
import ast,sys,unittest
from typing import Optional
from pathlib import Path
from collections import deque
from contextlib import nullcontext
from threading import RLock
from types import SimpleNamespace as NS,ModuleType
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mlx2.runtime.paged_native_contract import supports_native_checkpoint_candidate
class Owner:
    supported_planes=('kv','gdn')
    def __init__(self,uid):self._lane_id=uid;self.public=NS(generation=0,offset=1025,revision='r',layer_owners=(object(),)*16,companions=(('gdn',(NS(lane_id=uid,revision='r',offset=1025,generation=0),)),))
    def snapshot(self):return nullcontext(self.public)
class Logits:
    shape=(1,10)
    def __getitem__(self,k):return NS(shape=(10,))
class Continuation:
    calls=0;fail_at=None
    def __init__(self,**kw):
        index=self.calls;Continuation.calls+=1
        if index==self.fail_at:raise RuntimeError('constructor refusal')
        self.__dict__.update(kw)
def module(name,**values):m=ModuleType(name);m.__dict__.update(values);return m
def ordinary_presence(penalty,origin):
    tree=ast.parse((ROOT/'src/mlx2/runtime/sample_utils.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_probe_safe','_generated_window','make_presence_penalty')]
    env={'__name__':'mlx2.runtime.sample_utils','Optional':Optional}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual ordinary presence factory','exec'),env)
    return env['make_presence_penalty'](penalty,0,generation_start=origin)
class Tests(unittest.TestCase):
    def setUp(self):
        Continuation.calls=0;Continuation.fail_at=None;model=object()
        self.owners=tuple(Owner(i) for i in range(20));self.boot=tuple(NS(offset=1025,logits=Logits()) for _ in self.owners)
        queue=deque((i,[[1]*1025],192,object(),[],None,[],object(),None,None) for i in range(20))
        self.batch=NS(model=model,self_mtp=None,_unprocessed_sequences=queue,_native_continuations={},_native_lane_rngs={i:object() for i in range(20)},_native_prompt_responses=[],sampler=lambda x:x)
        self.batch._find_uids=lambda uids:{uid:(0,index) for index,row in enumerate(self.batch._unprocessed_sequences) for uid in uids if row[0]==uid}
        self.c=NS(model=model,native_layer_count=16,logical_layer_count=64,state_planes=('kv','gdn'),bootstrap_generation=0,supports_singleton=True,owns_physical_dispatch_proof=True,packed_lane=lambda *a:None,forward_staged=lambda *a:None,_serving_prompt_ids_by_uid={i:(1,)*1025 for i in range(20)},_serving_resources=NS(charge=1),backend=NS(writer=NS(poisoned=False,pending_epochs=(),ledger=NS(pending_count=0))))
        tree=ast.parse((ROOT/'src/mlx2/runtime/generate.py').read_text());node=next(f for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=='BatchGenerator' for f in cls.body if isinstance(f,ast.FunctionDef) and f.name=='install_native_hybrid_n_cohort')
        env={'__package__':'mlx2.runtime','deque':deque,'PromptProcessingBatch':NS(Response=lambda *a:a)};exec(compile(ast.Module(body=[node],type_ignores=[]),'real N attachment','exec'),env);self.attach=env[node.name]
        modules={
            'mlx2.runtime.paged_native_contract':module('contract',supports_native_checkpoint_candidate=supports_native_checkpoint_candidate),
            'mlx2.runtime.paged_native_continuation':module('continuation',NativeQwen3Continuation=Continuation),
            'mlx2.runtime.paged_native_atomic_owner':module('owner',NativeAtomicRequestOwner=Owner),
            'mlx2.runtime.packed_prefill_receipt':module('receipt',bootstrap_prefill_attribution=lambda c:dict(prefill_mode='native_packed_prefill',native_prefill_attention_calls=16))}
        p=patch.dict(sys.modules,modules);p.start();self.addCleanup(p.stop)
    def call(self,**kw):return self.attach(self.batch,self.owners,self.c,self.boot,RLock(),permit_native=True,**kw)
    def test_twenty_generation_zero_owners_attach_atomically(self):
        receipts=self.call();self.assertEqual(len(self.batch._native_continuations),20);self.assertFalse(self.batch._unprocessed_sequences)
        self.assertTrue(all(r['route']=='native_hybrid_packed_n20_research' and r['bootstrap_generation']==0 and r['observed_used'] is False for r in receipts))
    def test_last_constructor_failure_does_not_publish_nineteen_host_lanes(self):
        old=self.batch._unprocessed_sequences;Continuation.fail_at=19
        with self.assertRaises(RuntimeError):self.call()
        self.assertIs(self.batch._unprocessed_sequences,old);self.assertEqual(self.batch._native_continuations,{});self.assertEqual(len(self.batch._native_lane_rngs),20)
    def test_cancel_after_all_constructors_preserves_queue(self):
        calls=[];old=self.batch._unprocessed_sequences
        with self.assertRaises(ValueError):self.call(cancelled=lambda:(calls.append(1) or len(calls)==2))
        self.assertIs(self.batch._unprocessed_sequences,old);self.assertEqual(self.batch._native_continuations,{})
    def test_resolved_presence_processor_identity_survives_twenty_atomic_attachments(self):
        processors={i:ordinary_presence(1.5,1025) for i in range(20)}
        self.c._serving_sampling_by_uid={i:{'presence_penalty':1.5} for i in range(20)}
        self.batch._unprocessed_sequences=deque(tuple([*row[:6],[processors[row[0]]],*row[7:]]) for row in self.batch._unprocessed_sequences)
        receipts=self.call()
        for uid,lane in self.batch._native_continuations.items():self.assertIs(lane.processors[0],processors[uid])
        self.assertTrue(all(r['ordinary_processor_count']==1 for r in receipts))
    def test_bad_last_processor_preserves_entire_queued_cohort(self):
        self.c._serving_sampling_by_uid={i:{'presence_penalty':1.5} for i in range(20)}
        self.batch._unprocessed_sequences=deque(tuple([*row[:6],[ordinary_presence(1.5,1024 if row[0]==19 else 1025)],*row[7:]]) for row in self.batch._unprocessed_sequences)
        old=self.batch._unprocessed_sequences
        with self.assertRaisesRegex(ValueError,'ordinary_presence_contract'):self.call()
        self.assertIs(self.batch._unprocessed_sequences,old);self.assertEqual(self.batch._native_continuations,{})
    def test_pending_bootstrap_terminal_cannot_attach(self):
        self.c.backend.writer.ledger.pending_count=1
        with self.assertRaises(ValueError):self.call()
        self.assertEqual(Continuation.calls,0)
if __name__=='__main__':unittest.main()
