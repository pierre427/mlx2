"""Real profile/installer/atomic-attach bodies, guarded CPU fault regressions."""
import ast,copy,importlib.abc,json,os,sys,tempfile,unittest
from pathlib import Path
from threading import Event,Lock
from types import SimpleNamespace as NS,ModuleType
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native':raise RuntimeError('GPU/native import forbidden')
sys.meta_path.insert(0,Guard());sys.path.insert(0,str(ROOT/'src'))
from mlx2.runtime import paged_packed_prefill_serving_profile as P
from mlx2.runtime.packed_prefill_receipt import bootstrap_prefill_attribution
IDENTITY={'host':'cpu','hardware':'cpu','artifact_sha256':'a'*64,'source_commit':'b'*40,
          'source_tree_sha256':'c'*64,'mlx_wheel_version':'pinned','mlx_wheel_sha256':'d'*64,'kernel_sha256':'e'*64}
# Reuse existing host queue/owner doubles; retain production extracted methods.
path=ROOT/'tests/test_hybrid_serving_lifecycle_source_cpu.py';tree=ast.parse(path.read_text())
nodes=[n for n in tree.body if not (isinstance(n,ast.If) and '__name__' in ast.unparse(n.test))]
fixture={'__file__':str(path),'__name__':'cpu_fixture'};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),fixture)

class Profiles(unittest.TestCase):
    def load(self,data,counts=(32,96),environment=None):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'profile.json';p.write_text(json.dumps(data))
            return P.load_profile(p,live_identity=IDENTITY,context_lengths=counts,environment=environment or data['required_environment'])
    def test_absent_false_and_explicit_request_bounds(self):
        self.assertFalse(P.validate_request({}));self.assertFalse(P.validate_request({'paged_native_hybrid_packed_prefill':False}))
        body={'paged_native_hybrid_packed_prefill':True,'paged_native_hybrid_b2':True,'skip_writing_prefix_cache':True,'temperature':0,'max_tokens':4}
        self.assertTrue(P.validate_request(body))
        for key,value in (('max_tokens',True),('max_tokens',5),('temperature',1),('paged_native_hybrid_b2',False),('skip_writing_prefix_cache',False)):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):P.validate_request({**body,key:value})
    def test_profile_accepts_only_proven_pair_in_either_order(self):
        profile=P.make_profile(IDENTITY)
        for counts in ((32,96),(96,32)):
            result=self.load(profile,counts);translated=P.factory_profile(result,counts)
            self.assertEqual(translated['context_lengths'],list(counts));self.assertTrue(translated['prefill_nax_exact'])
            self.assertNotIn('max_tokens',translated);self.assertNotIn('numerical_reference',translated)
        for counts in ((32,64),(63,129),(6950,6929),(32,32),(True,96)):
            with self.subTest(counts=counts),self.assertRaises(ValueError):self.load(profile,counts)
    def test_no_scope_identity_env_or_qualification_widening(self):
        base=P.make_profile(IDENTITY)
        for key,value in (('qualified',True),('price_usable',True),('serving_default',True),('warm_apcv2',True),('prefill_nax_exact',False),('max_tokens',True),('numerical_reference','serial_exact'),('context_lengths',[32,124])):
            profile=copy.deepcopy(base);profile[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.load(profile)
        profile=copy.deepcopy(base);profile['identity']['kernel_sha256']='f'*64
        with self.assertRaises(ValueError):self.load(profile)
        env=dict(base['required_environment']);env['MLX2_PAGED_PREFILL_NAX_EXACT']='0'
        with self.assertRaises(ValueError):self.load(base,environment=env)
    def test_explicit_blocks_bind_factory_environment_and_budget(self):
        for block in (1,4,16):
            profile=P.make_profile(IDENTITY,prefill_eval_block_size=block)
            translated=P.factory_profile(self.load(profile), (32,96))
            self.assertEqual(translated.get('prefill_eval_block_size',1),block)
            self.assertEqual(translated['required_environment']['MLX2_PAGED_PREFILL_EVAL_BLOCK_SIZE'],str(block))
            self.assertEqual(translated['memory_budget_bytes'],12<<30)
            broken=copy.deepcopy(profile);broken['required_environment']['MLX2_PAGED_PREFILL_EVAL_BLOCK_SIZE']='2'
            with self.assertRaises(ValueError):self.load(broken)
        for bad in (True,False,0,2,8,17,'16',16.0,None):
            with self.subTest(bad=bad),self.assertRaises(ValueError):P.make_profile(IDENTITY,prefill_eval_block_size=bad)
            broken=P.make_profile(IDENTITY);broken['prefill_eval_block_size']=bad
            with self.assertRaises(ValueError):self.load(broken)
        for bad in (True,0,-1,(12<<30)+1):
            broken=P.make_profile(IDENTITY,prefill_eval_block_size=16);broken['memory_budget_bytes']=bad
            with self.assertRaises(ValueError):self.load(broken)

    def test_cli_startup_checks_declared_environment_without_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'profile.json';profile=P.make_profile(IDENTITY);path.write_text(json.dumps(profile))
            self.assertEqual(P.startup_environment(path),profile['required_environment'])
            profile['required_environment']['MLX2_PAGED_Q1_STOCK_LONG']='1';path.write_text(json.dumps(profile))
            with self.assertRaises(ValueError):P.startup_environment(path)

class PackedAttachment(unittest.TestCase):
    call=fixture['AtomicAttachment'].call;unchanged=fixture['AtomicAttachment'].unchanged
    def setUp(self):
        fixture['AtomicAttachment'].setUp(self)
        depth=8;physical={'grouped_multirow_write_count':depth,'grouped_multirow_row_count':depth*128,'prefill_matrix_dispatch_count':depth,
            **{'prefill_nax_'+stage+'_dispatch_count':depth for stage in ('score','softmax','value')}}
        self.candidate._packed_prefill_receipt={'prefill_mode':'packed_hybrid_real_rows','bootstrap_generation':0,
            'segment_lengths':(32,96),'real_projection_rows':128,'full_attention_layers':depth,'terminal_read_count':depth,
            'qualified':False,'price_usable':False,'selected':True,'observed_used':True,'physical_counters':physical,
            'source_identity':IDENTITY,'prefill_nax_exact':True,'native_reader_scratch_bytes':1585152,
            'native_reader_scratch_bound_bytes':3195072,'prefill_attention_arithmetic':'nax_three_stage_stock_short'}
        raw=NS(**{name:(lambda arena,value=value:value) for name,value in physical.items()})
        self.candidate.backend.read_submissions=depth;self.candidate.backend.terminal_successes=depth
        self.candidate.backend.writer.backend=NS(_native=raw,_arena=object())
    def test_actual_atomic_attach_attributes_native_prefill_and_preserves_rng(self):
        receipts=self.call();self.assertEqual(set(self.batch._native_continuations),{7,8})
        for r in receipts:
            self.assertEqual(r['prefill_mode'],'native_packed_prefill');self.assertTrue(r['native_prefill_observed_used'])
            self.assertEqual(r['native_prefill_attention_calls'],8);self.assertEqual(r['prefill_native_attention_calls'],8)
            self.assertFalse(r['qualified']);self.assertFalse(r['observed_used'])
    def test_invalid_physical_proof_keeps_both_queued_before_continuation(self):
        queue=self.batch._unprocessed_sequences;rngs=self.batch._native_lane_rngs
        self.candidate._packed_prefill_receipt['physical_counters']['prefill_nax_score_dispatch_count']=0
        with self.assertRaisesRegex(ValueError,'proof'):self.call()
        self.unchanged(queue,rngs);self.assertEqual(fixture['Continuation'].made,[])
    def _invoke_installer(self,*,cancel=False,abort_error=False,remove_error=False):
        functions=[n for n in ast.parse((ROOT/'src/mlx2/serving.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('install_explicit_native_hybrid_b2_cohort','_hybrid_output_caps_match')]
        env={'__package__':'mlx2','os':NS(environ=P.make_profile(IDENTITY)['required_environment'])}
        exec(compile(ast.Module(body=functions,type_ignores=[]),'serving.py','exec'),env)
        jobs=tuple(NS(uid=uid,tenant_id='tenant',request={'paged_native_hybrid_b2':True,'paged_native_hybrid_packed_prefill':True,
            'max_tokens':cap,'skip_writing_prefix_cache':True,'batch_cohort':{'id':'cpu','size':2},'temperature':0},
            effective_sampling={'temperature':0,'repetition_penalty':1,'presence_penalty':0,'frequency_penalty':0},
            effective_max_tokens=cap,preempted=False,cancelled=Event(),native_b2_cached_tokens=0,native_b2_prompt=tuple(range(length)),
            thinking_guard=None,structured=None) for uid,length,cap in ((7,32,2),(8,96,4)))
        def attach(owners,candidate,boots,lock,**kw):
            result=fixture['attach'](self.batch,owners,candidate,boots,lock,**kw)
            if remove_error:jobs[0].cancelled.set()
            return result
        self.batch.install_native_hybrid_cohort=attach
        if remove_error:
            def remove(uids):raise RuntimeError('injected removal failure')
            self.batch.remove=remove
        called=[];self.install_events=called
        self.candidate._serving_resources.retirement_failures=[]
        def abort():
            called.append('abort')
            if abort_error:
                self.orphans.append(self.candidate._serving_resources)
                raise RuntimeError('injected owner close ambiguity')
            self.candidate._serving_resources.closed=True
        self.candidate._serving_resources.abort=abort
        def factory(requests,**kwargs):
            called.append('packed');self.assertEqual(kwargs['live_identity'],IDENTITY)
            if cancel:jobs[0].cancelled.set()
            self.assertEqual(kwargs['profile']['schema'],P.FACTORY_SCHEMA);return self.owners,self.candidate,self.boot
        adapter=NS(identity={'path':'/artifact','fingerprint':'rev'},create_native_packed_prefill_b2=factory)
        native=ModuleType('_paged_kv_native');native.__file__='/native.so'
        retirement=ModuleType('mlx2.runtime.qwen35_paged_graph_factory');retirement._ORPHANS=[];self.orphans=retirement._ORPHANS
        identity=ModuleType('mlx2.runtime.paged_price_identity');identity.cached_live_price_identity=lambda *a,**kw:IDENTITY
        with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,{'_paged_kv_native':native,'mlx2.runtime.paged_price_identity':identity,'mlx2.runtime.qwen35_paged_graph_factory':retirement}):
            profile=Path(d)/'profile.json';profile.write_text(json.dumps(P.make_profile(IDENTITY)))
            result=env['install_explicit_native_hybrid_b2_cohort'](self.batch,adapter,jobs,lifecycle_lock=Lock(),profile_path=profile,manifest_path='/manifest',mlx_wheel_path='/wheel')
        return result,called
    def test_failed_private_abort_remains_durably_registered(self):
        queue=self.batch._unprocessed_sequences;rngs=self.batch._native_lane_rngs
        with self.assertRaisesRegex(ValueError,'cancelled before attachment'):
            self._invoke_installer(cancel=True,abort_error=True)
        self.unchanged(queue,rngs)
        self.assertEqual(self.orphans,[self.candidate._serving_resources])
        self.assertEqual(self.install_events,['packed','abort'])
    def test_installed_remove_failure_still_aborts_and_retains_resources(self):
        with self.assertRaisesRegex(ValueError,'cancelled during attachment'):
            self._invoke_installer(remove_error=True,abort_error=True)
        self.assertEqual(self.orphans,[self.candidate._serving_resources])
        self.assertEqual(self.install_events,['packed','abort'])
        self.assertEqual(self.candidate._serving_resources.retirement_failures[0]['stage'],'batch_remove')

    def test_actual_installer_uses_packed_adapter_and_final_source_identity(self):
        result,called=self._invoke_installer()
        self.assertEqual(called,['packed']);self.assertEqual(len(result),2)
        self.assertEqual(self.candidate._packed_prefill_receipt['serving_numerical_reference'],'same_geometry_ordinary_mixed')
    def test_cancelled_packed_private_bootstrap_never_publishes(self):
        queue=self.batch._unprocessed_sequences;rngs=self.batch._native_lane_rngs
        with self.assertRaisesRegex(ValueError,'cancelled before attachment'):self._invoke_installer(cancel=True)
        self.unchanged(queue,rngs);self.assertEqual(self.install_events,['packed','abort'])
    def test_second_packed_continuation_failure_keeps_atomic_queue_and_aborts(self):
        queue=self.batch._unprocessed_sequences;rngs=self.batch._native_lane_rngs
        fixture['Continuation'].fail_at=1
        with self.assertRaisesRegex(RuntimeError,'second constructor'):self._invoke_installer()
        self.unchanged(queue,rngs);self.assertEqual(self.install_events,['packed','abort'])
    def test_mixed_cohort_mode_rejects_before_factory_or_allocation(self):
        functions=[n for n in ast.parse((ROOT/'src/mlx2/serving.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='install_explicit_native_hybrid_b2_cohort']
        env={'__package__':'mlx2'};exec(compile(ast.Module(body=functions,type_ignores=[]),'serving.py','exec'),env)
        bodies=({'paged_native_hybrid_packed_prefill':False},{'paged_native_hybrid_packed_prefill':True,'paged_native_hybrid_b2':True,'skip_writing_prefix_cache':True,'temperature':0,'max_tokens':4})
        with self.assertRaisesRegex(ValueError,'selection differs'):env['install_explicit_native_hybrid_b2_cohort'](None,None,tuple(NS(request=b) for b in bodies),lifecycle_lock=None,profile_path=None,manifest_path=None,mlx_wheel_path=None)



class ResourceAbort(unittest.TestCase):
    def test_real_abort_roots_before_owner_close_and_retries_idempotently(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('abort_contract',ROOT/'src/mlx2/runtime/qwen35_paged_graph_factory.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        resources=module.HybridServingResources(1024,128,2048)
        calls=[]
        def close():
            calls.append('close')
            self.assertIn(resources,module._ORPHANS)
            if len(calls)==1:raise RuntimeError('injected ambiguous owner close')
        resources.owners=[NS(close=close)]
        with self.assertRaisesRegex(RuntimeError,'ambiguous'):resources.abort()
        self.assertEqual(module._CHARGED,1024);self.assertEqual(module._ORPHANS,[resources])
        self.assertEqual(resources.retirement_failures[0]['stage'],'abort')
        resources.abort()
        self.assertEqual(module._CHARGED,0);self.assertEqual(module._ORPHANS,[])
        resources.abort();self.assertEqual(module._CHARGED,0)
    def test_real_abort_reap_exception_remains_rooted(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('abort_contract',ROOT/'src/mlx2/runtime/qwen35_paged_graph_factory.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        resources=module.HybridServingResources(1024,128,2048)
        def fail():raise RuntimeError('injected terminal ambiguity')
        resources.reap=fail
        with self.assertRaisesRegex(RuntimeError,'terminal ambiguity'):resources.abort()
        self.assertEqual(module._CHARGED,1024);self.assertEqual(module._ORPHANS,[resources])
        resources.reap=module.HybridServingResources.reap.__get__(resources)
        module.reap_hybrid_admission_orphans()
        self.assertEqual(module._CHARGED,0);self.assertEqual(module._ORPHANS,[])

if __name__=='__main__':unittest.main()
