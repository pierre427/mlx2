"""CPU fault proofs of atomic attachment and retained native memory charge.

Run directly. Runtime imports are forbidden; production method bodies are
extracted with AST and supplied host-only owner/queue doubles.
"""
import ast
import hashlib
import math
from collections.abc import Mapping
import json
import tempfile
from collections import deque
from contextlib import nullcontext
import importlib.abc
import importlib.util
from pathlib import Path
import sys
from threading import RLock, Lock, Event
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch

if __name__ != '__main__': raise unittest.SkipTest('run directly; no runtime imports')
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native':
            raise RuntimeError('MLX/native import prohibited')
sys.meta_path.insert(0,NoRuntime())
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

F=load_file('hybrid_factory','src/mlx2/runtime/qwen35_paged_graph_factory.py')
F.__package__='mlx2.runtime'

def fake_module(name,**values):
    module=ModuleType(name);module.__dict__.update(values);sys.modules[name]=module

class Candidate: pass
class Owner:
    supported_planes=('kv','gdn')
    def __init__(self,uid,offset):
        self._lane_id=uid
        boundary=NS(lane_id=uid,offset=offset,revision='rev',generation=0)
        self.public=NS(generation=0,offset=offset,revision='rev',
                       companions=(('gdn',(boundary,)),),layer_owners=(object(),)*8)
    def snapshot(self): return nullcontext(self.public)
class Continuation:
    made=[]; fail_at=None
    def __init__(self,**kwargs):
        if len(self.made)==self.fail_at: raise RuntimeError('second constructor refused')
        self.__dict__.update(kwargs);self.made.append(self)

fake_module('mlx2.adapters.qwen35_paged_candidate',Qwen35PagedCandidate=Candidate)
contract=load_file('native_contract_source','src/mlx2/runtime/paged_native_contract.py')
fake_module('mlx2.runtime.paged_native_contract',supports_native_checkpoint_candidate=contract.supports_native_checkpoint_candidate)
fake_module('mlx2.runtime.paged_native_atomic_owner',NativeAtomicRequestOwner=Owner)
fake_module('mlx2.runtime.paged_native_continuation',NativeQwen3Continuation=Continuation)
fake_module('mlx2.runtime.paged_native_retirement',reap_native_request_owner=lambda *args:None)
source=ast.parse((ROOT/'src/mlx2/runtime/generate.py').read_text())
method=next(node for cls in source.body if isinstance(cls,ast.ClassDef) and cls.name=='BatchGenerator'
            for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='install_native_hybrid_cohort')
namespace={'__package__':'mlx2.runtime','deque':deque,
           'PromptProcessingBatch':NS(Response=lambda *args:args)}
exec(compile(ast.Module(body=[method],type_ignores=[]),'generate.py','exec'),namespace)
attach=namespace['install_native_hybrid_cohort']

class Logits:
    shape=(1,1024)
    def __getitem__(self,key): return NS(shape=(1024,))

class AtomicAttachment(unittest.TestCase):
    def setUp(self):
        Continuation.made=[];Continuation.fail_at=None
        model=object()
        queue=deque([(uid,[[*range(length)]],32,object(),[],None,[],object(),None,None)
                     for uid,length in ((7,32),(8,96))])
        self.batch=NS(model=model,self_mtp=None,_unprocessed_sequences=queue,
            _native_continuations={},_native_lane_rngs={7:object(),8:object()},
            _native_prompt_responses=[],sampler=lambda x:x)
        self.batch._find_uids=lambda uids:{uid:(0,index) for index,item in enumerate(self.batch._unprocessed_sequences)
                                          for uid in uids if item[0]==uid}
        self.owners=(Owner(7,32),Owner(8,96))
        self.candidate=Candidate();self.candidate.model=model;self.candidate.native_layer_count=8
        self.candidate.state_planes=('kv','gdn');self.candidate.bootstrap_generation=0
        self.candidate.supports_singleton=True;self.candidate.owns_physical_dispatch_proof=True
        self.candidate.packed_lane=lambda *args:None;self.candidate.forward_staged=lambda *args:None
        self.candidate._serving_prompt_ids_by_uid={7:tuple(range(32)),8:tuple(range(96))}
        self.candidate.logical_layer_count=32;self.candidate._serving_resources=NS(charge=12345)
        self.candidate.backend=NS(writer=NS(poisoned=False,pending_epochs={},ledger=NS(pending_count=0)))
        self.boot=(NS(offset=32,logits=Logits()),NS(offset=96,logits=Logits()))
    def call(self,**kwargs):
        return attach(self.batch,self.owners,self.candidate,self.boot,RLock(),permit_native=True,**kwargs)
    def unchanged(self,queue,rngs):
        self.assertIs(self.batch._unprocessed_sequences,queue)
        self.assertIs(self.batch._native_lane_rngs,rngs)
        self.assertEqual(self.batch._native_continuations,{})
        self.assertEqual(self.batch._native_prompt_responses,[])
    def test_generation_zero_import_attaches_both_without_sampler_or_native_read_claim(self):
        receipts=self.call()
        self.assertEqual(set(self.batch._native_continuations),{7,8})
        self.assertEqual(len(self.batch._unprocessed_sequences),0)
        self.assertEqual(self.batch._native_lane_rngs,{})
        self.assertEqual(len(self.batch._native_prompt_responses),2)
        self.assertTrue(all(row['bootstrap_generation']==0 and row['prefill_native_attention_calls']==0
                            and row['observed_used'] is False for row in receipts))
    def test_second_constructor_failure_keeps_both_queued_and_rngs(self):
        queue=self.batch._unprocessed_sequences;rngs=self.batch._native_lane_rngs
        Continuation.fail_at=1
        with self.assertRaisesRegex(RuntimeError,'second constructor'):self.call()
        self.unchanged(queue,rngs)
    def test_cancellation_at_commit_keeps_both_queued(self):
        calls=[];queue=self.batch._unprocessed_sequences;rngs=self.batch._native_lane_rngs
        def cancelled(): calls.append(1);return len(calls)==2
        with self.assertRaisesRegex(ValueError,'cancelled'):self.call(cancelled=cancelled)
        self.unchanged(queue,rngs)
    def test_submitted_bootstrap_cannot_attach(self):
        queue=self.batch._unprocessed_sequences;rngs=self.batch._native_lane_rngs
        self.candidate.backend.writer.pending_epochs={1:object()}
        with self.assertRaisesRegex(ValueError,'not terminal'):self.call()
        self.unchanged(queue,rngs)
    def test_same_length_changed_prompt_ids_refuse_before_commit(self):
        queue=self.batch._unprocessed_sequences;rngs=self.batch._native_lane_rngs
        self.candidate._serving_prompt_ids_by_uid[8]=(99,)*96
        with self.assertRaisesRegex(ValueError,'pristine full'):self.call()
        self.unchanged(queue,rngs)
    def test_wrong_lane_checkpoint_refuses_before_commit(self):
        queue=self.batch._unprocessed_sequences;rngs=self.batch._native_lane_rngs
        self.owners[1].public.companions[0][1][0].lane_id=7
        with self.assertRaisesRegex(ValueError,'boundary drifted'):self.call()
        self.unchanged(queue,rngs)

class ServingInstallerLock(unittest.TestCase):
    """Exercise the real outer installer and real generator attachment together."""
    setUp = AtomicAttachment.setUp
    def invoke_installer(self, *, cancel_after_bootstrap=False):
        tree=ast.parse((ROOT/'src/mlx2/serving.py').read_text())
        functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and
                   node.name in ('install_explicit_native_hybrid_b2_cohort','_hybrid_output_caps_match')]
        env={'__package__':'mlx2','os':NS(environ={})}
        exec(compile(ast.Module(body=functions,type_ignores=[]),'serving.py','exec'),env)
        # A genuine nonreentrant Lock with a bounded acquisition turns the old
        # nested-lock deadlock into a deterministic CPU test failure.
        raw=Lock();counts=NS(enters=0)
        class BoundedLock:
            def __enter__(self):
                if not raw.acquire(timeout=.05):raise AssertionError('nested lifecycle lock acquisition')
                counts.enters+=1;return self
            def __exit__(self,*args):raw.release()
        lock=BoundedLock()
        jobs=tuple(NS(uid=uid,tenant_id='default',request={
            'paged_native_hybrid_b2':True,'skip_writing_prefix_cache':True,
            'batch_cohort':{'id':'http','size':2},'temperature':0},
            effective_sampling={'temperature':0,'repetition_penalty':1,'presence_penalty':0,'frequency_penalty':0},
            effective_max_tokens=cap,preempted=False,cancelled=Event(),native_b2_cached_tokens=0,
            native_b2_prompt=tuple(range(length)),thinking_guard=None,structured=None)
            for uid,length,cap in ((7,32,2),(8,96,4)))
        self.batch.install_native_hybrid_cohort=lambda owners,candidate,boot,lock,**kw:attach(self.batch,owners,candidate,boot,lock,**kw)
        aborts=[];self.candidate._serving_resources.abort=lambda:aborts.append('abort')
        def factory(*args,**kwargs):
            self.assertFalse(raw.locked(),'private bootstrap must not hold publication lock')
            if cancel_after_bootstrap:jobs[0].cancelled.set()
            return self.owners,self.candidate,self.boot
        from mlx2.adapters.native_hybrid import HybridNativeCohort
        adapter=NS(native_cohort_backend=HybridNativeCohort,identity={'path':'/artifact','fingerprint':'rev'},create_native_paged_hybrid_b2=factory)
        identity=ModuleType('mlx2.runtime.paged_price_identity');identity.cached_live_price_identity=lambda *args,**kw:{}
        profile=ModuleType('mlx2.runtime.paged_hybrid_research_profile')
        profile.load_hybrid_research_profile=lambda *args,**kw:{'max_tokens':4,'profile_id':'test','storage_dtype':'bfloat16','q1_simd_stripes':16}
        native=ModuleType('_paged_kv_native');native.__file__='/native.so'
        with patch.dict(sys.modules,{'_paged_kv_native':native,'mlx2.runtime.paged_price_identity':identity,
                                    'mlx2.runtime.paged_hybrid_research_profile':profile}):
            try:
                result=env['install_explicit_native_hybrid_b2_cohort'](self.batch,adapter,jobs,lifecycle_lock=lock,
                    profile_path='/profile',manifest_path='/manifest',mlx_wheel_path='/wheel')
            except BaseException:
                self.assertEqual(aborts,['abort']);raise
        return result,counts
    def test_real_installer_attaches_with_one_nonreentrant_publication_lock(self):
        receipts,counts=self.invoke_installer()
        self.assertEqual(counts.enters,1)
        self.assertEqual(set(self.batch._native_continuations),{7,8})
        self.assertEqual(len(receipts),2)
    def test_cancelled_private_bootstrap_aborts_before_attachment(self):
        with self.assertRaisesRegex(ValueError,'cancelled before attachment'):
            self.invoke_installer(cancel_after_bootstrap=True)
        self.assertEqual(self.batch._native_continuations,{})
        self.assertEqual(len(self.batch._unprocessed_sequences),2)

class BudgetLifetime(unittest.TestCase):
    def setUp(self):
        self.initial=F._CHARGED;self.events=[]
        self.resources=F.HybridServingResources(4096,2048,8192)
        self.addCleanup(self.cleanup)
        writer=NS(pending_epochs={},ledger=NS(pending_count=0,completed_epoch=0),poisoned=False,
                  pool=NS(allocated_count=0,retire=lambda epoch:None))
        writer.poll_completions=lambda:None
        writer.teardown_failed_arena=lambda:self.events.append('teardown')
        writer.backend=NS(close_after_terminal=lambda:self.events.append('close_arena'))
        backend=NS(writer=writer,_orphaned_reads={},drain_failed_read_events=lambda:None)
        self.candidate=NS(backend=backend,_failure_roots=[],_bootstrap_failure_roots=[])
        self.resources.candidate=self.candidate
    def cleanup(self):
        self.resources.owners=[];self.resources.unattached=[]
        writer=self.candidate.backend.writer;writer.pending_epochs={};writer.ledger.pending_count=0
        writer.pool.allocated_count=0;self.candidate.backend._orphaned_reads={}
        self.resources.reap();self.assertEqual(F._CHARGED,self.initial)
    def test_live_owner_retains_complete_aggregate_charge(self):
        self.resources.owners=[NS(fully_retired=False)]
        self.assertFalse(self.resources.reap());self.assertEqual(F._CHARGED,self.initial+4096)
        self.assertNotIn('close_arena',self.events)
        self.resources.owners[0].fully_retired=True
        self.assertTrue(self.resources.reap());self.assertEqual(self.events,['close_arena'])
    def test_ambiguous_write_retains_unattached_pages_and_charge(self):
        writer=self.candidate.backend.writer;writer.poisoned=True;writer.pending_epochs={1:object()}
        self.resources.unattached=[NS(close=lambda:self.events.append('close_layer'))]
        self.assertFalse(self.resources.reap());self.assertEqual(self.events,[])
        self.assertEqual(F._CHARGED,self.initial+4096)
        writer.pending_epochs={}
        self.assertTrue(self.resources.reap());self.assertEqual(self.events,['teardown','close_layer','close_arena'])
    def test_ambiguous_read_retains_unattached_pages(self):
        self.candidate.backend._orphaned_reads={1:object()}
        self.resources.unattached=[NS(close=lambda:self.events.append('close_layer'))]
        self.assertFalse(self.resources.reap());self.assertEqual(self.events,[])
    def test_regular_idle_reaper_releases_late_completed_bootstrap_without_admission(self):
        tree=ast.parse((ROOT/'src/mlx2/serving.py').read_text())
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and
                      node.name=='_reap_native_admission_orphans')
        fake_module('mlx2.runtime.qwen35_paged_graph_factory',reap_hybrid_admission_orphans=F.reap_hybrid_admission_orphans)
        env={'__package__':'mlx2','_NATIVE_ADMISSION_ORPHANS':[]}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'serving.py','exec'),env)
        writer=self.candidate.backend.writer;writer.pending_epochs={1:object()}
        self.resources.unattached=[NS(close=lambda:self.events.append('close_layer'))]
        F._ORPHANS.append(self.resources)
        self.addCleanup(lambda:F._ORPHANS.remove(self.resources) if self.resources in F._ORPHANS else None)
        env['_reap_native_admission_orphans']()
        self.assertIn(self.resources,F._ORPHANS);self.assertEqual(self.events,[])
        writer.pending_epochs={}
        env['_reap_native_admission_orphans']()
        self.assertNotIn(self.resources,F._ORPHANS)
        self.assertTrue(self.resources.closed);self.assertEqual(F._CHARGED,self.initial)
        self.assertEqual(self.events,['close_layer','close_arena'])
    def test_failure_reservation_release_does_not_uncharge_session(self):
        reservation=self.resources.reserve(1024);reservation.retain_failure_roots((object(),))
        self.assertEqual(F._CHARGED,self.initial+4096);self.assertEqual(len(self.resources.failure_reservations),1)
        with self.assertRaises(MemoryError):self.resources.reserve(4096)
        self.assertTrue(self.resources.reap());self.assertEqual(F._CHARGED,self.initial)

class ServingCompletionContracts(unittest.TestCase):
    def setUp(self):
        self.source=ast.parse((ROOT/'src/mlx2/serving.py').read_text())
    def test_grouped_receipt_uses_hybrid_bootstrap_and_actual_stock_geometry(self):
        tree=ast.parse((ROOT/'src/mlx2/runtime/paged_native_graph_group.py').read_text())
        block=next(node for node in ast.walk(tree) if isinstance(node,ast.If) and
            isinstance(node.test,ast.Name) and node.test.id=='serving' and
            'admission_profile' in ast.unparse(node))
        for hybrid in (True,False):
            candidate=NS(_b2_profile_id='test',_b2_q1_stripes=4)
            if hybrid:candidate.bootstrap_generation=0;candidate.owns_physical_dispatch_proof=True
            response=NS(mtp_receipt={'prefill_mode':'ordinary_completed_import' if hybrid else 'serial_native',
                                    'native_prefill_observed_used':False,'native_prefill_attention_calls':0})
            env={'serving':True,'candidate':candidate,'response':response,'physical':None,
                 'combined_counts':None,'grouped_sampling':False,'backend':NS(),
                 'graph_receipt':{'q1_simd_stripes':32,'native_stock_reduction_dispatches':16},
                 '__name__':'mlx2.runtime.paged_native_graph_group','__package__':'mlx2.runtime'}
            exec(compile(ast.Module(body=[block],type_ignores=[]),'grouped receipt','exec'),env)
            self.assertEqual(response.mtp_receipt['prefill_mode'],'ordinary_completed_import' if hybrid else 'serial_native')
            self.assertEqual(response.mtp_receipt['q1_simd_stripes'],32 if hybrid else 4)
            self.assertFalse(response.mtp_receipt['native_prefill_observed_used'])
            self.assertEqual(response.mtp_receipt['native_prefill_attention_calls'],0)
    def test_singleton_receipt_uses_completed_singleton_physical_proof(self):
        tree=ast.parse((ROOT/'src/mlx2/runtime/paged_native_continuation.py').read_text())
        block=next(node for node in ast.walk(tree) if isinstance(node,ast.If) and
            'bootstrap_generation' in ast.unparse(node.test) and 'survivor_q1_stripes' in ast.unparse(node))
        proof={'q1_simd_stripes':16,'native_stock_reduction_dispatches':0,'packed_lanes':1}
        lane=NS(candidate=NS(bootstrap_generation=0,native_layer_count=16,logical_layer_count=64,
                           q1_stripes=16,_serving_stock_reduction=True),
                _last_native_graph_proof=proof,research_only=False,terminal_successes=16)
        receipt={};exec(compile(ast.Module(body=[block],type_ignores=[]),'singleton receipt','exec'),{'self':lane,'receipt':receipt,
            '__name__':'mlx2.runtime.paged_native_continuation','__package__':'mlx2.runtime'})
        self.assertEqual(receipt['prefill_mode'],'ordinary_completed_import')
        self.assertEqual(receipt['q1_simd_stripes'],16)
        self.assertEqual(receipt['hybrid_graph_proof']['native_stock_reduction_dispatches'],0)
        self.assertFalse(receipt['native_prefill_observed_used'])
        self.assertTrue(receipt['observed_used'])
    def test_opt_in_singleton_receipt_preserves_actual_stock32_proof(self):
        tree=ast.parse((ROOT/'src/mlx2/runtime/paged_native_continuation.py').read_text())
        block=next(node for node in ast.walk(tree) if isinstance(node,ast.If) and
            'bootstrap_generation' in ast.unparse(node.test) and 'survivor_q1_stripes' in ast.unparse(node))
        proof={'q1_simd_stripes':32,'native_stock_reduction_dispatches':16,
            'native_stock_singleton_dispatches':16,'packed_lanes':1}
        lane=NS(candidate=NS(bootstrap_generation=0,native_layer_count=16,logical_layer_count=64,
            q1_stripes=16,_serving_stock_reduction=True,_serving_stock_singleton=True),
            _last_native_graph_proof=proof,research_only=False,terminal_successes=16)
        receipt={};exec(compile(ast.Module(body=[block],type_ignores=[]),'actual stock singleton receipt','exec'),{'self':lane,'receipt':receipt,
            '__name__':'mlx2.runtime.paged_native_continuation','__package__':'mlx2.runtime'})
        self.assertEqual(receipt['q1_simd_stripes'],32);self.assertEqual(receipt['survivor_q1_stripes'],32)
        self.assertEqual(receipt['survivor_fallback_q1_stripes'],16)
        self.assertTrue(receipt['stock_singleton_selected']);self.assertTrue(receipt['stock_singleton_observed_used'])
        self.assertEqual(receipt['hybrid_graph_proof']['native_stock_singleton_dispatches'],16)
    def test_hybrid_final_route_and_output_ids_preserve_used_receipt(self):
        original={'route':'native_hybrid_paged_b2','selected':True,'observed_used':True,'qualified':False}
        env={'response':NS(mtp_receipt=original),'job':NS(receipt_token_ids=[13,14,15]),
             'self':NS(snapshot={'settings':{'route':'ordinary'}},route_selection_source='default'),
             'route_receipt':{'route':'ordinary'}}
        selected=None
        for node in ast.walk(self.source):
            for field in ('body','orelse','finalbody'):
                body=getattr(node,field,None)
                if not isinstance(body,list):continue
                for index,item in enumerate(body):
                    if (isinstance(item,ast.Assign) and isinstance(item.targets[0],ast.Name) and
                        item.targets[0].id=='native_route' and isinstance(item.value,ast.IfExp)):
                        selected=[item,body[index+1]]
        self.assertIsNotNone(selected)
        exec(compile(ast.Module(body=selected,type_ignores=[]),'serving.py','exec'),env)
        self.assertEqual(env['native_route']['output_token_ids'],[13,14,15])
        self.assertTrue(env['native_route']['observed_used']);self.assertFalse(env['native_route']['qualified'])
        receipt=next(node for node in ast.walk(self.source) if isinstance(node,ast.Dict) and
            all(name in [key.value for key in node.keys if isinstance(key,ast.Constant)]
                for name in ('route','route_receipt','route_selection_source','request_controls')))
        selected_keys=[];selected_values=[]
        for key,value in zip(receipt.keys,receipt.values):
            if isinstance(key,ast.Constant) and key.value in ('route','route_receipt','route_selection_source'):
                selected_keys.append(key);selected_values.append(value)
        result=eval(compile(ast.fix_missing_locations(ast.Expression(ast.Dict(keys=selected_keys,values=selected_values))),'serving.py','eval'),env)
        self.assertEqual(result['route'],'native_hybrid_paged_b2')
        self.assertEqual(result['route_selection_source'],'explicit_request')
        self.assertIs(result['route_receipt'],env['native_route'])
    def test_hybrid_completion_uses_native_apc_skip_counter(self):
        branch=next(node for node in ast.walk(self.source) if isinstance(node,ast.If) and
            any(isinstance(item,ast.AugAssign) and isinstance(item.target,ast.Subscript) and
                isinstance(item.target.slice,ast.Constant) and item.target.slice.value=='apcv2_store_skipped_native'
                for item in node.body))
        response=NS(finish_reason='length',mtp_receipt={'route':'native_hybrid_paged_b2'})
        self.assertTrue(eval(compile(ast.Expression(branch.test),'serving.py','eval'),{'response':response}))
    def test_unequal_output_caps_fit_pinned_upper_bound(self):
        function=next(node for node in self.source.body if isinstance(node,ast.FunctionDef) and
                      node.name=='_hybrid_output_caps_match')
        env={};exec(compile(ast.Module(body=[function],type_ignores=[]),'serving.py','exec'),env)
        check=env['_hybrid_output_caps_match']
        self.assertTrue(check((NS(effective_max_tokens=2),NS(effective_max_tokens=4)),{'max_tokens':4}))
        self.assertFalse(check((NS(effective_max_tokens=2),NS(effective_max_tokens=5)),{'max_tokens':4}))
        self.assertFalse(check((NS(effective_max_tokens=True),NS(effective_max_tokens=4)),{'max_tokens':4}))

class ProfileBinding(unittest.TestCase):
    def setUp(self):
        pack=ast.parse((ROOT/'src/mlx2/runtime/paged_pack_price.py').read_text())
        nodes=[node for node in pack.body if (isinstance(node,ast.FunctionDef) and node.name in ('_sha','_identity')) or
            (isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='IDENTITY_FIELDS' for target in node.targets))]
        env={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'identity.py','exec'),env)
        fake_module('mlx2.runtime.paged_pack_price',_identity=env['_identity'])
        self.profile=load_file('mlx2.runtime.paged_hybrid_research_profile','src/mlx2/runtime/paged_hybrid_research_profile.py')
        self.identity={key:('a'*40 if key=='source_commit' else 'a'*64 if key.endswith('sha256') else 'fixed')
                       for key in env['IDENTITY_FIELDS']}
        self.environment={'MLX2_PAGED_HYBRID_B2':'1','MLX2_PAGED_Q1_SIMD_TILE':'1',
            'MLX2_PAGED_GROUPED_Q1_WRITE':'1','MLX2_PAGED_PRIVATE_TAIL_REUSE':'1',
            'MLX2_PAGED_Q1_SIMD_STRIPES':'16','MLX2_PAGED_Q1_SPLIT_KV':'0',
            **{key:'0' for key in ('MLX2_PAGED_Q1_STOCK_SDPA','MLX2_PAGED_Q1_STOCK_REDUCTION','MLX2_PAGED_Q1_INLINE_METADATA',
                'MLX2_PAGED_B2_DEFERRED_EVAL','MLX2_PAGED_B2_DEFERRED_WRITE_EVAL',
                'MLX2_PAGED_GROUPED_SAMPLER','MLX2_PAGED_GROUPED_DIRECT_FENCE')}}
        self.data={'schema':self.profile.SCHEMA,'profile_id':'exact','identity':self.identity,
            'context_bounds':{'minimum':32,'maximum':97,'distinct':True},'max_tokens':32,
            'sampling':{'mode':'greedy','processors':False},'required_environment':dict(self.environment),
            'qualified':False,'price_usable':False,'serving_default':False,'warm_apcv2':False,
            'q1_simd_stripes':16,'q1_split_partition':0,'storage_dtype':'bfloat16','memory_budget_bytes':512<<20}
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'profile.json'
    def load(self):
        self.path.write_text(json.dumps(self.data))
        return self.profile.load_hybrid_research_profile(self.path,live_identity=self.identity,
            context_lengths=(32,96),environment=self.environment)
    def test_exact_scope_permits_only_default_off_same_dtype_complete_request(self):
        self.assertEqual(self.load()['storage_dtype'],'bfloat16')
    def stock(self):
        self.data['stock_reduction']=True
        self.environment['MLX2_PAGED_Q1_STOCK_REDUCTION']='1'
        self.data['required_environment']['MLX2_PAGED_Q1_STOCK_REDUCTION']='1'
    def test_explicit_false_preserves_legacy_stock_zero_contract(self):
        self.data['stock_reduction']=False
        self.assertIs(self.load()['stock_reduction'],False)
    def test_stock32_true_admits_aligned_short_pair_with_survivor_geometry(self):
        self.stock()
        self.assertIs(self.load()['stock_reduction'],True)
        self.assertEqual(self.load()['q1_simd_stripes'],16)
    def test_stock32_selector_requires_exact_boolean(self):
        self.data['stock_reduction']=1
        with self.assertRaisesRegex(ValueError,'exact boolean'):self.load()
    def test_stock32_requested_flag_must_match_profile(self):
        self.data['stock_reduction']=True
        with self.assertRaisesRegex(ValueError,'environment'):self.load()
    def test_stock32_rejects_arbitrary_ragged_padding_before_bootstrap_or_charge(self):
        self.stock();self.path.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError,'aligned ragged'):
            self.profile.load_hybrid_research_profile(self.path,live_identity=self.identity,
                context_lengths=(32,65),environment=self.environment)
        F.__package__='mlx2.runtime'
        initial=F._CHARGED
        requests=((1,'rev',tuple(range(32)),4),(2,'rev',tuple(range(65)),4))
        # Adapter has no model: refusal must precede model/native imports.
        with self.assertRaisesRegex(ValueError,'aligned ragged'):
            F.create_shared_hybrid_graph_pack(NS(identity={'fingerprint':'rev'}),requests,
                profile=self.data,permit_candidate=True)
        self.assertEqual(F._CHARGED,initial)
    def test_stock32_long_split_is_not_admitted(self):
        self.stock();self.data['q1_split_partition']=128
        self.environment['MLX2_PAGED_Q1_SPLIT_KV']='128'
        self.data['required_environment']['MLX2_PAGED_Q1_SPLIT_KV']='128'
        with self.assertRaisesRegex(ValueError,'short aligned'):self.load()
    def test_stock32_native_counter_capability_is_required(self):
        self.stock()
        with self.assertRaisesRegex(ValueError,'physical counter'):
            self.profile.require_hybrid_native_capabilities(self.data,NS())
        with self.assertRaisesRegex(ValueError,'physical counter'):
            self.profile.require_hybrid_native_capabilities(self.data,NS(q1_stock_reduction_dispatch_count=1))
        self.profile.require_hybrid_native_capabilities(self.data,NS(q1_stock_reduction_dispatch_count=lambda arena:0))
        self.data['stock_reduction']=False
        self.profile.require_hybrid_native_capabilities(self.data,NS())
    def singleton(self):
        self.stock();self.data['stock_singleton']=True
        self.environment['MLX2_PAGED_Q1_STOCK_SINGLETON']='1'
        self.data['required_environment']['MLX2_PAGED_Q1_STOCK_SINGLETON']='1'
    def test_explicit_singleton_true_is_source_environment_bound(self):
        self.singleton();self.assertIs(self.load()['stock_singleton'],True)
        self.environment['MLX2_PAGED_Q1_STOCK_SINGLETON']='0'
        with self.assertRaisesRegex(ValueError,'singleton stock environment'):self.load()
    def test_legacy_profile_refuses_unbound_singleton_flag(self):
        self.stock();self.environment['MLX2_PAGED_Q1_STOCK_SINGLETON']='1'
        with self.assertRaisesRegex(ValueError,'singleton stock environment'):self.load()
    def test_singleton_requires_stock_and_exact_boolean(self):
        self.singleton();self.data['stock_reduction']=False
        with self.assertRaisesRegex(ValueError,'stock_singleton'):self.load()
        self.singleton();self.data['stock_singleton']=1
        with self.assertRaisesRegex(ValueError,'stock_singleton'):self.load()
    def test_singleton_native_capability_required_before_bootstrap(self):
        self.singleton()
        with self.assertRaisesRegex(ValueError,'singleton stock32 physical counter'):
            self.profile.require_hybrid_native_capabilities(self.data,NS(q1_stock_reduction_dispatch_count=lambda _:0))
        self.profile.require_hybrid_native_capabilities(self.data,NS(q1_stock_reduction_dispatch_count=lambda _:0,
            q1_stock_singleton_dispatch_count=lambda _:0))
    def test_long_stock_is_not_serving_admitted(self):
        self.environment['MLX2_PAGED_Q1_STOCK_LONG']='1'
        with self.assertRaisesRegex(ValueError,'serving admission is not enabled'):self.load()
    def test_source_drift_fails_before_factory(self):
        self.data['identity']=dict(self.identity,source_commit='b'*40)
        with self.assertRaisesRegex(ValueError,'identity'):self.load()
    def test_undeclared_physical_flag_fails(self):
        self.environment['MLX2_PAGED_Q1_INLINE_METADATA']='1'
        with self.assertRaisesRegex(ValueError,'environment'):self.load()
    def test_last_decode_context_must_fit_native_bound(self):
        self.data['context_bounds']['maximum']=98
        with self.assertRaisesRegex(ValueError,'context bound'):self.load()
    def test_profile_cannot_claim_qualification_or_price(self):
        self.data['qualified']=True
        with self.assertRaisesRegex(ValueError,'scope'):self.load()

class ShardedArtifactBinding(unittest.TestCase):
    def test_every_indexed_shard_is_required_and_hash_bound(self):
        identity=load_file('price_identity_source','src/mlx2/runtime/paged_price_identity.py')
        with tempfile.TemporaryDirectory() as directory:
            root=(Path(directory)/'model').resolve();root.mkdir()
            bodies={'config.json':b'{}','tokenizer.json':b'{}',
                'model-00001-of-00002.safetensors':b'first','model-00002-of-00002.safetensors':b'second',
                'model.safetensors.index.json':json.dumps({'weight_map':{'a':'model-00001-of-00002.safetensors',
                    'b':'model-00002-of-00002.safetensors'}}).encode()}
            for name,body in bodies.items():(root/name).write_bytes(body)
            data={'root':str(root),'files':{name:hashlib.sha256(body).hexdigest() for name,body in bodies.items()}}
            manifest=Path(directory)/'manifest.json';manifest.write_text(json.dumps(data))
            self.assertEqual(identity._artifact_sha256(manifest,root),identity._sha256(manifest))
            saved=data['files'].pop('model-00002-of-00002.safetensors');manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError,'every indexed'):identity._artifact_sha256(manifest,root)
            data['files']['model-00002-of-00002.safetensors']=saved;manifest.write_text(json.dumps(data))
            (root/'model-00002-of-00002.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'bytes differ'):identity._artifact_sha256(manifest,root)

class HttpCapabilityValidation(unittest.TestCase):
    def setUp(self):
        server=ast.parse((ROOT/'src/mlx2/server.py').read_text())
        nodes=[node for node in server.body if isinstance(node,ast.FunctionDef) and
            node.name in ('validate_request','normalize_client_options')]
        compat=ast.parse((ROOT/'src/mlx2/openai_compat.py').read_text())
        nodes += [node for node in compat.body if isinstance(node,ast.FunctionDef) and node.name in
            ('normalize_tool_choice','drop_null_fields','flatten_text_parts','flatten_text_messages')]
        environment={'__package__':'mlx2','__name__':'mlx2.server','math':math,'Mapping':Mapping,'MAX_TOP_LOGPROBS':20,'MAX_OUTPUT_TOKENS':2_097_152}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'server.py','exec'),environment)
        self.validate=environment['validate_request']
        self.body={'messages':[{'role':'user','content':'hello'}],'temperature':0,'max_tokens':32,
            'skip_writing_prefix_cache':True,'paged_native_hybrid_b2':True,'batch_cohort':{'id':'test','size':2}}
    def test_explicit_hybrid_flag_survives_normalization(self):
        self.assertIs(self.validate(self.body)['paged_native_hybrid_b2'],True)
    def test_string_flag_cannot_select_capability(self):
        self.body['paged_native_hybrid_b2']='1'
        with self.assertRaisesRegex(ValueError,'must be boolean'):self.validate(self.body)
    def test_native_capabilities_are_exclusive(self):
        self.body['paged_native_qwen3_b2']=True
        with self.assertRaisesRegex(ValueError,'mutually exclusive'):self.validate(self.body)
    def test_no_write_and_exact_cohort_required(self):
        self.body['skip_writing_prefix_cache']=False
        with self.assertRaisesRegex(ValueError,'cold two-member'):self.validate(self.body)

if __name__=='__main__':unittest.main()
