"""Actual corpus and event scope tests; CPU only, no model/runtime imports."""
import importlib.abc,sys
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native':raise RuntimeError('runtime import forbidden')
sys.meta_path.insert(0,Guard());sys.path.insert(0,str(ROOT/'scripts/research'))
import spomin_400case_native_suite as S
import spomin_400case_phased_http_bench as B
class Tests(unittest.TestCase):
    def test_actual_twenty_domains_fourhundred_distinct_cases(self):
        corpus=S.corpus_rows();self.assertEqual(len(corpus['cases']),400);self.assertEqual(len(corpus['domain_order']),20)
        self.assertEqual(len({r['case_id'] for r in corpus['cases']}),400)
    def test_both_arms_original192_and_native_source_bound_request(self):
        row={'case_id':'x','body':dict(messages=[dict(role='user',content='source')],max_tokens=192,temperature=0,enable_thinking=False)}
        native=S.request_body(row,model='m',native=True,cohort_id='domain0',inputs_sha256='a'*64)
        ordinary=S.request_body(row,model='m',native=False,cohort_id='domain0')
        self.assertEqual(native['messages'],ordinary['messages']);self.assertEqual(native['max_tokens'],192);self.assertEqual(native['batch_cohort']['size'],20)
        self.assertEqual(native['native_research_input_id'],'x');self.assertNotIn('paged_native_hybrid_b2',native);self.assertNotIn('batch_cohort',ordinary)
        with self.assertRaises(ValueError):S.request_body(row,model='m',native=True,cohort_id='d')
    def test_early_eos_actual_samples_score_and_rate_are_not_cap192(self):
        row=dict(case_id='x',domain='d',body_sha256='a',prompt_tokens=7000,sentinel='AUDIT',needles={'a':'KEY'},expected_concepts=[['design']])
        body=dict(usage=dict(completion_tokens=2,prompt_tokens=7000),choices=[dict(finish_reason='stop',message=dict(content='AUDIT KEY design'))])
        events={'x':[dict(token=1,monotonic_ns=2_000_000_000,width=20),dict(token=2,monotonic_ns=3_000_000_000,width=1)]}
        result=B.summarize([row],[(200,body)],events,{'x':1.},4.)
        self.assertEqual(result['actual_completion_tokens'],2);self.assertEqual(result['rows'][0]['output_token_ids'],[1,2]);self.assertEqual(result['aggregate_decode_tokens_per_second'],1.)
        self.assertTrue(result['all_sentinels']);self.assertTrue(result['all_needles']);self.assertEqual(result['concept_hits'],1)
        body['choices'][0]['finish_reason']='length'
        with self.assertRaises(RuntimeError):B.summarize([row],[(200,body)],events,{'x':1.},4.)
    def test_cancel_then_owned_shutdown_does_not_cancel_aborted_controller_twice(self):
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        service=B.Service(NS());cancelled=[];restored=[]
        service.current={'arm':'native'};service.controller=NS(paused=True,aborted=False,cancel=lambda owner:(cancelled.append(owner) or setattr(service.controller,'aborted',True)))
        service.engine=NS(jobs={},close=lambda:None);service.completion_thread=NS(join=lambda n:None)
        owner={'session':'s','lease_id':'cancel'}
        self.assertTrue(service.cancel(owner)['cancel_requested']);self.assertEqual(len(cancelled),1)
        service.server=NS(shutdown=lambda:None);service.mx=NS(set_cache_limit=lambda v:restored.append(v));service.previous_cache_limit=100
        with patch.object(B,'bounded_cleanup',return_value={'retained':False,'engine_closed':True}):
            self.assertFalse(service.shutdown({'session':'s','lease_id':'shutdown'})['complete'])
        self.assertEqual(len(cancelled),1);self.assertEqual(restored,[100]);self.assertTrue(service.closed)
    def test_partial_init_shutdown_uses_real_cleanup_signature_without_model(self):
        from types import SimpleNamespace as NS
        service=B.Service(NS());result=service.shutdown({'session':'s','lease_id':'shutdown'})
        self.assertTrue(result['cleanup']['engine_closed']);self.assertFalse(result['cleanup']['retained']);self.assertTrue(service.closed)
    def test_parity_failure_still_restores_allocator_after_safe_retirement(self):
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        service=B.Service(NS());restored=[];service.mx=NS(set_cache_limit=lambda v:restored.append(v));service.previous_cache_limit=123
        service.cells=[dict(domain_index=0,arm='native',warmup=False,rows=[dict(case_id='x',output_token_ids=[1],text='a',finish_reason='stop')]),dict(domain_index=0,arm='ordinary',warmup=False,rows=[dict(case_id='x',output_token_ids=[2],text='a',finish_reason='stop')])]
        with patch.object(B,'bounded_cleanup',return_value={'retained':False,'engine_closed':True}) as cleanup:
            with self.assertRaises(RuntimeError):service.shutdown({'session':'s','lease_id':'shutdown'})
        self.assertEqual(len(cleanup.call_args.args),5);self.assertEqual(restored,[123]);self.assertTrue(service.closed)
    def test_missing_actual_ordinary_sample_event_refuses_text_only_parity(self):
        row=dict(case_id='x',domain='d',prompt_tokens=7000)
        body=dict(usage=dict(completion_tokens=1,prompt_tokens=7000),choices=[dict(finish_reason='stop')])
        with self.assertRaises(RuntimeError):B.summarize([row],[(200,body)],{},dict(x=1.),4.)
if __name__=='__main__':unittest.main()
