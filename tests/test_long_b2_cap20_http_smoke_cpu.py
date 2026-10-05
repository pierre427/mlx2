"""Guarded explicit cap20, persistent protocol, timings and cold-lifecycle gates."""
import ast,importlib.abc,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native':raise RuntimeError('runtime import forbidden')
sys.meta_path.insert(0,Guard());sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts/research'))
from mlx2.runtime import paged_packed_prefill_serving_profile as P,hybrid_packed_prefill as F
import long_b2_cap20_http_smoke as B
IDENTITY={'host':'cpu','hardware':'cpu','artifact_sha256':'a'*64,'source_commit':'b'*40,'source_tree_sha256':'c'*64,
 'mlx_wheel_version':'pinned','mlx_wheel_sha256':'d'*64,'kernel_sha256':B.NATIVE_SHA}
class Tests(unittest.TestCase):
    def load(self,p,identity=IDENTITY,counts=B.COUNTS):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'profile.json';path.write_text(json.dumps(p))
            return P.load_profile(path,live_identity=identity,context_lengths=counts,environment=p['required_environment'])
    def test_long_cap20_requires_explicit_profile_and_request(self):
        legacy=P.make_profile(IDENTITY,long_fused=True);self.assertEqual(self.load(legacy)['max_tokens'],4)
        extended=P.make_profile(IDENTITY,long_fused=True,research_output_cap20=True)
        self.assertEqual(self.load(extended)['max_tokens'],20);self.assertTrue(extended['research_output_cap20'])
        body={'paged_native_hybrid_packed_prefill':True,'paged_native_hybrid_b2':True,'skip_writing_prefix_cache':True,'temperature':0,'max_tokens':20}
        with self.assertRaises(ValueError):P.validate_request(body)
        self.assertTrue(P.validate_request({**body,'paged_native_long_cap20_research':True}))
        for value in (False,1,'true',None):
            with self.subTest(value=value),self.assertRaises(ValueError):P.validate_request({**body,'paged_native_long_cap20_research':value})
        for cap in (True,0,21):
            with self.assertRaises(ValueError):P.validate_request({**body,'paged_native_long_cap20_research':True,'max_tokens':cap})
        with self.assertRaises(ValueError):P.make_profile(IDENTITY,research_output_cap20=True)
        for field,value in (('max_tokens',21),('research_output_cap20',False),('qualified',True)):
            with self.subTest(field=field),self.assertRaises(ValueError):self.load({**extended,field:value})
        with self.assertRaises(ValueError):self.load(extended,{**IDENTITY,'source_commit':'f'*40})
    def test_factory_capacity_and_context_domain(self):
        requests=((1,'rev',tuple([1]*6950),20),(2,'rev',tuple([2]*6929),20))
        self.assertEqual(F.validate_requests(requests,'rev',100,long_fused=True,research_output_cap20=True),B.COUNTS)
        with self.assertRaises(ValueError):F.validate_requests(requests,'rev',100,long_fused=True)
        with self.assertRaises(ValueError):F.validate_requests(requests,'rev',100,research_output_cap20=True)
        with self.assertRaises(ValueError):F.validate_requests(((1,'rev',tuple([1]*8180),20),requests[1]),'rev',100,long_fused=True,research_output_cap20=True)
        profile=P.make_profile(IDENTITY,long_fused=True,research_output_cap20=True)
        from mlx2.runtime.hybrid_packed_prefill_long import validate_long_profile
        validate_long_profile(P.factory_profile(profile,B.COUNTS),live_identity=IDENTITY,counts=B.COUNTS,environment=profile['required_environment'])
    def test_order_fresh_leases_and_all_ten_cohorts(self):
        seq=B.Sequence()
        def call(action,i=None,n=0):seq.accept({'action':action,'cohort_index':i},{'session':'cpu','lease_id':str(n)})
        with self.assertRaises(ValueError):call('cohort',0)
        call('init',n=1)
        with self.assertRaises(ValueError):call('warmup',n=1)
        with self.assertRaises(ValueError):call('cohort',0,n=2)
        call('warmup',n=2)
        with self.assertRaises(ValueError):call('cohort',1,n=3)
        with self.assertRaises(ValueError):call('cohort',True,n=3)
        for i in range(10):call('cohort',i,n=3+i)
        with self.assertRaises(ValueError):call('cohort',10,n=13)
        call('shutdown',n=13);self.assertTrue(seq.closed);self.assertEqual(seq.next,10)
        with self.assertRaises(ValueError):call('init',n=14)
    def test_actual_rates_exclude_first_tokens_and_refuse_fake_counts(self):
        events=[[{'monotonic_ns':int((10+lane*.1+i*.1)*1e9),'token':i,'width':2} for i in range(20)] for lane in range(2)]
        rates=B.rates(events,5,13);self.assertEqual(rates['decode_rate_numerator'],38)
        self.assertAlmostEqual(rates['aggregate_decode_tokens_per_second'],19)
        self.assertEqual(rates['actual_completion_tokens'],40)
        with self.assertRaises(RuntimeError):B.rates([events[0][:-1],events[1]],5,13)
        bad=[list(lane) for lane in events];bad[1][5]=bad[1][4]
        with self.assertRaises(RuntimeError):B.rates(bad,5,13)
    def test_declared_scope_safety_and_real_sample_materialization(self):
        p=B.plan();self.assertEqual(p['requests_per_measured_arm'],20);self.assertEqual(p['physical_cohort_width'],2)
        self.assertEqual(p['expected_output_tokens_per_arm'],400);self.assertEqual(B.MAX_RSS,48<<30);self.assertEqual(B.MAX_SECONDS,100)
        source=Path(B.__file__).read_text();self.assertIn('int(response.token)',source)
        self.assertIn('owner=verified_lease(command)',source);self.assertIn('if verified_lease(command)!=owner',source)
        self.assertIn('os._exit(124)',source);self.assertIn('each command requires a fresh lease',source)
        self.assertIn("'service is not quiescent before lease release'",source)
        self.assertIn("'skip_writing_prefix_cache':True",source);self.assertIn('token_ids',source)
        self.assertIn("expected=(CAP-1)*16",source)
    def test_lease_refusal_preserves_previous_environment(self):
        import os,varlen_pack_price_bench as G
        with patch.dict(os.environ,{'GPUQ_SESSION':'old','GPUQ_LEASE':'oldlease'}),patch.object(G,'_gpuq_owner',side_effect=RuntimeError('both locks differ')):
            with self.assertRaisesRegex(RuntimeError,'both locks differ'):B.verified_lease({'session':'new','lease':'newlease'})
            self.assertEqual(os.environ['GPUQ_SESSION'],'old');self.assertEqual(os.environ['GPUQ_LEASE'],'oldlease')
    def test_persistent_service_cohort_and_aggregate_contract(self):
        args=NS();service=B.Service(args)
        def arm(native,index):
            return {'arm':'native' if native else 'ordinary','token_ids':[[i for i in range(20)]]*2,
                'summaries':[{'text':'same'}]*2,'effective_sampling':[{'temperature':0}]*2,
                'actual_completion_tokens':40,'http_pair_seconds':2 if native else 3,
                'prefill_through_both_first_samples_seconds':1,'decode_first_to_last_seconds':1}
        service.arm=arm
        service.cohort(-1)
        for index in range(10):service.cohort(index)
        self.assertEqual(len(service.cells),11)
        self.assertEqual(service.cells[1]['order'],['ordinary','native'])
        self.assertEqual(service.cells[2]['order'],['native','ordinary'])
        with patch.object(B,'bounded_cleanup',return_value={'retained':False,'engine_closed':True}):service.shutdown()
        self.assertEqual(service.result['status'],'passed')
        self.assertEqual(service.result['totals']['native']['actual_completion_tokens'],400)
        self.assertEqual(service.result['totals']['ordinary']['actual_requests'],20)
        self.assertEqual(service.result['totals']['native']['summed_http_pair_seconds'],20)
        def drift(native,index):
            row=arm(native,index)
            if native:row['token_ids']=[[999]*20]*2
            return row
        service.arm=drift
        with self.assertRaisesRegex(RuntimeError,'parity failed'):service.cohort(0)

    def test_local_command_channel_runs_ordered_fresh_leases_without_runtime(self):
        import threading,time
        class FakeService:
            def __init__(self,args):self.result=B.plan();self.sequence=B.Sequence();self.engine=None;self.current='idle'
            def init(self):return {'loads':1}
            def cohort(self,index):return {'index':index}
            def shutdown(self):return {'closed':True}
        with tempfile.TemporaryDirectory() as directory:
            args=NS(channel=Path(directory)/'channel',auth_token_file=Path(directory)/'auth',output=Path(directory)/'out.json')
            with patch.object(B,'Service',FakeService),patch.object(B,'verified_lease',side_effect=lambda c:{'session':c['session'],'lease_id':c['lease']}),patch('builtins.print'):
                thread=threading.Thread(target=B.serve,args=(args,));thread.start()
                deadline=time.monotonic()+2
                while not args.channel.exists() and time.monotonic()<deadline:time.sleep(.001)
                for number,(action,index) in enumerate([('init',None),('warmup',None),*[('cohort',i) for i in range(10)],('shutdown',None)]):
                    command=NS(**vars(args),command=action,cohort_index=index,session='cpu',lease=str(number))
                    self.assertEqual(B.client(command),0)
                thread.join(timeout=2);self.assertFalse(thread.is_alive())
            receipt=json.loads(args.output.read_text());self.assertEqual(len(receipt['command_receipts']),13)
            self.assertTrue(all(c['status']=='passed' for c in receipt['command_receipts']))
            self.assertFalse(args.channel.exists());self.assertFalse(args.auth_token_file.exists())

    def test_installer_checks_request_profile_flag_match_before_factory(self):
        tree=ast.parse((ROOT/'src/mlx2/serving.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install_explicit_native_hybrid_b2_cohort')
        s=ast.unparse(fn);self.assertIn("request.get('paged_native_long_cap20_research', False) is not profile.get('research_output_cap20', False)",s)
        self.assertLess(s.index('request/profile long cap20'),s.index('owners, candidate, bootstrap ='))
if __name__=='__main__':unittest.main()
