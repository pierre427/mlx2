"""Guarded N20 capability/corpus/admission proofs; no MLX/native imports."""
import importlib.abc
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from types import MethodType
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native':raise RuntimeError('GPU/runtime import forbidden')
sys.meta_path.insert(0,Guard());sys.path.insert(0,str(ROOT/'src'))
from mlx2.runtime import hybrid_packed_prefill_n as N
from mlx2.runtime.paged_pack_price import IDENTITY_FIELDS
from mlx2.adapters.qwen35_paged_n_candidate import (
    Qwen35PagedNCandidate,configure_native_ragged_prompt_lookup,
    validate_n_lanes,validate_finite_roots)
from mlx2.adapters.qwen35_paged_candidate import HybridPackedLane
from mlx2.runtime.ragged_verify_layout import RaggedVerifyLayout

def identity():
    x={k:'a'*64 for k in IDENTITY_FIELDS};x.update(host='CPU',hardware='CPU',source_commit='b'*40,mlx_wheel_version='pinned');return x

def native():
    x=NS(**{name:lambda *a:0 for name in (*N.COUNTERS,'grouped_multirow_write_n20','q1_scalar_dispatch_count','q1_stock_long_n20_singleton_partial_dispatch_count','q1_stock_long_n20_singleton_reduce_dispatch_count')});x.packed_n20_capability=lambda:dict(N.CAP);return x
BACKEND=NS(**{name:lambda *a:None for name in ('append_packed_multirow_n20','append_staged_grouped_q1_n20','prepare_read_n20')})
def profile():
    rows=[dict(case_id='case'+str(i),domain='one',prompt_tokens=1025,prompt_token_ids=[i]*1025) for i in range(400)]
    inputs=dict(case_count=400,client_concurrency=20,generation_max_tokens=192,corpus_sha256=N.__dict__.get('CORPUS_SHA','5d7233f805b310e354cb628a0d0b1d12299cc90dbfeebb890eaf35dac966ad12'),inputs_sha256='c'*64,rows=rows)
    return N.make_profile(identity(),inputs)
class Tests(unittest.TestCase):
    def test_ragged_adapter_executes_only_the_shrinking_active_prefix(self):
        candidate=object.__new__(Qwen35PagedNCandidate)
        candidate._runtime_factory=lambda:NS(mx=NS(concatenate=np.concatenate))
        widths=[]
        def forward(self,lanes,branches,**kwargs):
            widths.append(len(lanes))
            return np.zeros((len(lanes),4)),{
                'packed_lanes':len(lanes),'physical_counters':{'rows':len(lanes)}}
        candidate.forward_staged=MethodType(forward,candidate)
        staged=[];sealed=[];lanes=[];branches=[]
        for uid,length in ((1,2),(2,1)):
            layers=(object(),);caches=(object(),)
            branch=NS(layers=layers,recurrent_caches=caches,_closed=False,
                _request=NS(lane_id=uid,proposed_rows=length,planes=('kv','gdn')),
                _origin=NS(layers=(NS(offset=1025),)),
                stage_recurrent_prefix=lambda caches,accepted_rows,offset,uid=uid:
                    staged.append((uid,accepted_rows,offset)),
                seal_executed_rows=lambda rows,uid=uid:sealed.append((uid,rows)))
            branches.append(branch);lanes.append(HybridPackedLane(
                tuple(range(length)),layers,caches))
        layout=RaggedVerifyLayout.from_draft_depths((1,2),(1,0))
        def decide(indices,step,logits):
            return (True,False) if step==0 else (False,)
        outputs,receipt=candidate.forward_staged_ragged(
            layout,tuple(lanes),tuple(branches),decide_next=decide,
            permit_candidate=True,reserve_scratch=lambda n:None,
            collect_logits=True)
        self.assertEqual(widths,[2,1]);self.assertEqual([x.shape[0] for x in outputs],[2,1])
        self.assertEqual(staged,[(2,1,1026),(1,2,1027)])
        self.assertEqual(sealed,[(1,2),(2,1)])
        self.assertEqual(receipt['executed_query_lengths'],[2,1])
        self.assertEqual(receipt['round_widths'],[2,1])
        self.assertEqual(receipt['physical_counters'],{'rows':3})

    def test_ragged_prompt_lookup_is_explicit_bounded_and_default_off(self):
        c=NS()
        self.assertIsNone(configure_native_ragged_prompt_lookup(c,{}))
        self.assertIsNone(c._native_ragged_prompt_lookup)
        policy,depth=configure_native_ragged_prompt_lookup(c,{
            'MLX2_PAGED_N20_RAGGED_PROMPT_LOOKUP':'1',
            'MLX2_PAGED_N20_RAGGED_DEPTH':'7',
            'MLX2_PAGED_N20_RAGGED_NGRAM_MIN':'1',
            'MLX2_PAGED_N20_RAGGED_NGRAM_MAX':'2'})
        self.assertEqual(depth,7);self.assertEqual(policy.sources,('prompt_lookup',))
        self.assertEqual((policy.ngram_min,policy.ngram_max),(1,2))
        for value in ('0','16','x','01'):
            with self.assertRaises(ValueError):configure_native_ragged_prompt_lookup(c,{
                'MLX2_PAGED_N20_RAGGED_PROMPT_LOOKUP':'1',
                'MLX2_PAGED_N20_RAGGED_DEPTH':value})
        for key,value in (
            ('MLX2_PAGED_N20_RAGGED_NGRAM_MIN','0'),
            ('MLX2_PAGED_N20_RAGGED_NGRAM_MAX','17'),
            ('MLX2_PAGED_N20_RAGGED_NGRAM_MIN','01'),
            ('MLX2_PAGED_N20_RAGGED_NGRAM_MAX','x')):
            with self.assertRaises(ValueError):configure_native_ragged_prompt_lookup(c,{
                'MLX2_PAGED_N20_RAGGED_PROMPT_LOOKUP':'1',key:value})
        with self.assertRaises(ValueError):configure_native_ragged_prompt_lookup(c,{
            'MLX2_PAGED_N20_RAGGED_PROMPT_LOOKUP':'1',
            'MLX2_PAGED_N20_RAGGED_NGRAM_MIN':'4',
            'MLX2_PAGED_N20_RAGGED_NGRAM_MAX':'3'})
    def test_all_three_mechanisms_and_exact_native_types_before_admission(self):
        for width in (1,2,3,20):self.assertEqual(N.require_n_capabilities(native(),BACKEND,(1025,)*width),N.CAP)
        for name in N.COUNTERS:
            n=native();delattr(n,name)
            with self.assertRaises(ValueError):N.require_n_capabilities(n,BACKEND,(1025,)*20)
        n=native();n.packed_n20_capability=lambda:{**N.CAP,'version':True}
        with self.assertRaises(ValueError):N.require_n_capabilities(n,BACKEND,(1025,))
        for counts in ((1025,)*21,(True,),(),(8193,)):
            with self.assertRaises(ValueError):N.require_n_capabilities(native(),BACKEND,counts)
    def test_binding_real_case_ids_token_bytes_and_domain_before_writes(self):
        p=profile();args=dict(live_identity=identity(),source_input_ids=('case0','case1'),counts=(1025,1025),tokens=((0,)*1025,(1,)*1025),environment_values=N.environment())
        self.assertIs(N.validate_profile(p,**args),p)
        for change in (dict(source_input_ids=('case0','case0')),dict(tokens=((0,)*1025,(2,)*1025)),dict(environment_values={}),dict(counts=(1024,1025))):
            with self.assertRaises(ValueError):N.validate_profile(p,**{**args,**change})
        p['source_inputs']['case1']['domain']='two'
        with self.assertRaises(ValueError):N.validate_profile(p,**args)
    def test_bootstrap_requires_all_n_physical_rows_and_terminal_proof(self):
        depth=16;counts=(1025,)*3;expected={N.COUNTERS[0]:depth,N.COUNTERS[1]:sum(counts)*depth,N.COUNTERS[2]:depth}
        raw=NS(**{name:(lambda a,value=value:value) for name,value in expected.items()})
        writer=NS(backend=NS(_native=raw,_arena=None),pending_epochs=(),ledger=NS(pending_count=0))
        proof=dict(segment_lengths=counts,physical_counters=expected,terminal_read_count=depth,bootstrap_generation=0,real_projection_rows=sum(counts),prefill_layer_lifetime=True,public_state_published=False)
        candidate=NS(_packed_prefill_receipt=proof,native_layer_count=depth,backend=NS(writer=writer,terminal_successes=depth))
        self.assertEqual(N.bootstrap_attribution(candidate)['prefill_cohort_width'],3)
        writer.ledger.pending_count=1
        with self.assertRaises(ValueError):N.bootstrap_attribution(candidate)
        writer.ledger.pending_count=0;proof['physical_counters']={**expected,N.COUNTERS[1]:2*1025*depth}
        with self.assertRaises(ValueError):N.bootstrap_attribution(candidate)
    def test_full_finite_vector_checks_interior_and_last_states_and_kv_once(self):
        import numpy as np
        calls=[];mx=NS(stack=np.stack,all=np.all,isfinite=np.isfinite,eval=lambda *a:calls.append(1))
        logits=np.ones((20,2));states=[np.ones((2,)) for _ in range(20*48*2)]
        projections=[(np.ones((2,)),np.ones((2,)),np.ones((2,)),None) for _ in range(16)]
        validate_finite_roots(mx,logits,states,projections);self.assertEqual(calls,[1])
        for value in (logits,states[0],states[713],states[-1],projections[-1][2]):
            value.flat[0]=np.nan;calls.clear()
            with self.assertRaises(ValueError):validate_finite_roots(mx,logits,states,projections)
            self.assertEqual(calls,[1]);value.flat[0]=1.
    def test_twenty_private_rows_and_cancel_shrunk_singleton_are_valid(self):
        lanes=[];branches=[];c=NS(args=NS(vocab_size=10),layer_map=NS(recurrent=(0,)))
        for i in range(20):
            layers=(object(),);caches=(object(),)
            branch=NS(layers=layers,recurrent_caches=caches,_closed=False,_request=NS(proposed_rows=1,planes=('kv','gdn')),_origin=NS(layers=(NS(offset=1025),)))
            branches.append(branch);lanes.append(HybridPackedLane((1,),layers,caches))
        self.assertEqual(validate_n_lanes(tuple(lanes),tuple(branches),c),(1025,)*20)
        self.assertEqual(validate_n_lanes((lanes[-1],),(branches[-1],),c),(1025,))
        branches[10].recurrent_caches=branches[0].recurrent_caches;lanes[10]=HybridPackedLane((1,),branches[10].layers,branches[10].recurrent_caches)
        with self.assertRaises(ValueError):validate_n_lanes(tuple(lanes),tuple(branches),c)
if __name__=='__main__':unittest.main()
