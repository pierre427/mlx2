"""Synthetic CPU cache/lease contracts; no model, GPU, or qualification claim."""
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
import pytest
from mlx2.adapters.deepseek_v41_source_bank_session import PreparedBankSession,_fork,_isolated
from mlx2.adapters.deepseek_v41_source import source_policy
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights

class Cache:
    def __init__(self,offset=99,keys=None):
        self.offset=offset;self.keys=np.array([3.])if keys is None else keys;self.xp=np;self.closed=False
    @property
    def state(self):return (self.keys,)
    @property
    def meta_state(self):return ('fake-exact-cache',self.offset)
    @classmethod
    def from_state(cls,state,meta_state,*,xp):return cls(meta_state[-1],np.array(state[0],copy=True))
    def validate_state(self):assert not self.closed
    def close(self):self.closed=True
class Backend:
    any=staticmethod(np.any);all=staticmethod(np.all);stack=staticmethod(np.stack);isfinite=staticmethod(np.isfinite)
    def __init__(self,fail=False):self.calls=0;self.fail=fail
    def eval(self,*values):
        self.calls+=1
        if self.fail:raise RuntimeError('failed GPU drain')
class Bank:
    generation=7
    def __init__(self):self.leases=0
    @contextmanager
    def lease(self,*,ownercheck):
        ownercheck();self.leases+=1
        try:yield self
        finally:self.leases-=1
    def assert_current(self,*,contents=True):return True

def setup(*,cancelled=None,backend=None):
    weights=SimpleNamespace(source_resident_gather_policy='resident_bank_v1');bank=Bank()
    session=PreparedBankSession(weights,bank,ownercheck=lambda:True,cancelled=cancelled,backend=backend or Backend())
    weights.source_resident_bank_session=session
    return session,bank,[Cache()]

def execution(session,status=0,*,nonfinite=False,calls=None,working_refs=None):
    def run(private):
        if calls is not None:calls.append(session.replaying)
        if working_refs is not None:working_refs.append(private)
        private[0].keys[0]=11.;private[0].offset+=1
        if session.active:session.statuses.append(np.array([status],np.uint32))
        return (np.array([[float('nan')if nonfinite else 4.]]),None)
    return run

def test_success_commits_independent_candidate_and_one_compute_eval():
    session,bank,cache=setup();private=[]
    result=session.forward(None,None,cache,execution(session,working_refs=private))
    assert result[0][0,0]==4 and cache[0].offset==100 and cache[0].keys[0]==11
    private[0][0].keys[0]=88
    assert cache[0].keys[0]==11 and bank.leases==0
    assert session.stats['whole_forward_evaluations']==1
    assert session.stats['publication_copy_evaluations']==1

def test_miss_replays_once_from_exact_untouched_frontier():
    session,bank,cache=setup();seen=[]
    def run(private):
        seen.append((session.replaying,private[0].offset,float(private[0].keys[0])))
        return execution(session,1)(private)
    session.forward(None,None,cache,run)
    assert seen==[(False,99,3.),(True,99,3.)]
    assert cache[0].offset==100 and session.stats['ordinary_replays']==1 and bank.leases==0

def test_nonfinite_rejects_without_publication_or_replay():
    session,bank,cache=setup();calls=[]
    with pytest.raises(RuntimeError,match='rejected'):
        session.forward(None,None,cache,execution(session,nonfinite=True,calls=calls))
    assert cache[0].offset==99 and cache[0].keys[0]==3 and calls==[False]and bank.leases==0

def test_cancel_after_completion_preserves_published_cache():
    counter=[0]
    def cancel():counter[0]+=1;return counter[0]>1
    session,bank,cache=setup(cancelled=cancel)
    with pytest.raises(InterruptedError):session.forward(None,None,cache,execution(session))
    assert cache[0].offset==99 and bank.leases==0

def test_pre_cancel_does_not_acquire_or_run():
    session,bank,cache=setup(cancelled=lambda:True)
    with pytest.raises(InterruptedError):session.forward(None,None,cache,lambda _:pytest.fail('ran'))
    assert bank.leases==0 and cache[0].offset==99

def test_failed_drain_retains_bank_and_all_private_owners():
    session,bank,cache=setup(backend=Backend(fail=True))
    with pytest.raises(RuntimeError,match='failed GPU drain'):session.forward(None,None,cache,execution(session))
    assert session.poisoned and len(session._quarantined)==1 and bank.leases==1
    assert cache[0].offset==99 and cache[0].keys[0]==3
    with pytest.raises(RuntimeError,match='quarantined'):session.forward(None,None,cache,execution(session))
    # Explicit synthetic completion cleanup; no actual resources exist.
    session._quarantined[0][0].__exit__(None,None,None)

def test_replay_failure_does_not_mutate_published_cache():
    session,bank,cache=setup()
    def run(private):
        if session.replaying:private[0].keys[0]=999;raise RuntimeError('ordinary failure')
        return execution(session,1)(private)
    with pytest.raises(RuntimeError,match='ordinary failure'):session.forward(None,None,cache,run)
    assert cache[0].offset==99 and cache[0].keys[0]==3 and bank.leases==0

def test_forks_physically_copy_cache_arrays():
    cache=[Cache()];clone=_fork(cache)
    assert _isolated(cache,clone)and not np.shares_memory(cache[0].keys,clone[0].keys)

def test_dense_bank_scope_retains_lazy_roots_without_layer_eval():
    session,bank,cache=setup();session.active=True
    weights=object.__new__(SourceV41Weights);weights.source_resident_bank_session=session
    with weights.dense_window():weights.evaluate_dense_window('root')
    assert session.roots==['root']and session.backend.calls==0
    with weights.execution_scope_for_rows(1):pass

def test_policy_requires_explicit_math_and_preserves_default():
    assert source_policy().get('deepseek_source_resident_gather','reference')=='reference'
    with pytest.raises(ValueError,match='requires'):source_policy({'deepseek_source_resident_gather':'resident_bank_v1'})
    assert source_policy({'deepseek_source_resident_gather':'resident_bank_v1','deepseek_source_expert_math':'routed_fused_fp32'})['deepseek_source_resident_gather']=='resident_bank_v1'

def test_unprepared_selected_decode_refuses_before_ordinary_scope():
    weights=object.__new__(SourceV41Weights);weights.source_resident_gather_policy='resident_bank_v1';weights.source_resident_bank_session=None
    with pytest.raises(ValueError,match='explicitly prepared'):weights.execution_scope_for_rows(1)

def test_pinned_bank_keys_are_not_adaptive_eviction_candidates():
    from mlx2.adapters.deepseek_v41_source_lfu import LFUSourceExpertStore,LFUIndex
    store=object.__new__(LFUSourceExpertStore);store._cache={(0,0):{},(0,1):{}};store._leased={};store._pending_uploads={}
    store.index=LFUIndex();store.index.touch((0,0));store.index.touch((0,1));store._resident_bank_pinned_keys=frozenset({(0,0)})
    assert store._adaptive_victim()==(0,1)
    with pytest.raises(ValueError,match='individually evict'):store._adaptive_evict_key((0,0))
    store._cache.pop((0,1));assert store._adaptive_victim()is None

def test_per_layer_live_check_has_no_external_ownership_call():
    calls=[];weights=SimpleNamespace(source_resident_gather_policy='resident_bank_v1');bank=Bank()
    session=PreparedBankSession(weights,bank,ownercheck=lambda:calls.append('external'),backend=Backend())
    session.active=True;session.lease=bank
    assert session.check_live()is True and calls==[]

@pytest.mark.parametrize('operation',['park','close'])
def test_weights_retire_bank_before_native_provider_teardown(monkeypatch,operation):
    from mlx2.adapters.deepseek_v41_mlx_weights import NativeV41Weights
    calls=[];weights=object.__new__(SourceV41Weights)
    class Session:
        def close(self):calls.append('bank');weights.source_resident_bank_session=None
    weights.source_resident_bank_session=Session()
    monkeypatch.setattr(NativeV41Weights,operation,lambda self:calls.append('native'))
    getattr(weights,operation)()
    assert calls==['bank','native']and weights.source_resident_bank_session is None

@pytest.mark.parametrize('operation',['park','close'])
def test_quarantined_bank_refuses_teardown_and_retains_provider(monkeypatch,operation):
    from mlx2.adapters.deepseek_v41_mlx_weights import NativeV41Weights
    weights=object.__new__(SourceV41Weights)
    class Session:
        def close(self):raise RuntimeError('quarantined bank resources retained')
    session=weights.source_resident_bank_session=Session()
    monkeypatch.setattr(NativeV41Weights,operation,lambda self:pytest.fail('native teardown ran'))
    with pytest.raises(RuntimeError,match='quarantined'):getattr(weights,operation)()
    assert weights.source_resident_bank_session is session

@pytest.mark.parametrize('active,replaying',[(True,False),(False,True)])
def test_public_model_entry_rejects_nested_bank_transaction(active,replaying):
    from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
    session=SimpleNamespace(active=active,replaying=replaying,
        selected=lambda *args:pytest.fail('nested selection ran'))
    model=SimpleNamespace(weights=SimpleNamespace(source_resident_bank_session=session),
        _forward_ordinary=lambda *a,**k:pytest.fail('public call bypassed transaction'))
    with pytest.raises(ValueError,match='reenter an active bank transaction'):
        DeepSeekV41Model._forward(model,None,[])

def test_expert_variant_is_explicit_and_idle_only():
    session,bank,cache=setup()
    assert session.expert_kernel_variant=='reference'
    assert session.set_expert_kernel_variant('route_down8_v1')['expert_kernel_variant']=='route_down8_v1'
    session.active=True
    with pytest.raises(RuntimeError,match='idle healthy'):session.set_expert_kernel_variant('reference')
    session.active=False
    with pytest.raises(ValueError,match='Invalid'):session.set_expert_kernel_variant('unsupported')

def test_default_custom_consumer_signature_remains_supported():
    session,bank,cache=setup();calls=[]
    def consume(x,ids,scores,lease,*,layer,expected_generation,check_live):
        calls.append((layer,expected_generation));assert check_live()is True
        return SimpleNamespace(output='out',status=np.array([0]),middle=None,route_meta='meta')
    session.consumer=consume;session.active=True;session.lease=bank
    assert session.moe(None,None,None,'blk.3')=='out'and calls==[(3,7)]

@pytest.mark.parametrize('state',['active','replaying','poisoned'])
def test_submission_variant_retains_control_and_requires_idle(state):
    session,bank,cache=setup()
    assert session.submission_policy=='per_layer_async_v1'
    assert session.set_submission_policy('whole_forward_v1')['submission_policy']=='whole_forward_v1'
    setattr(session,state,True)
    with pytest.raises(RuntimeError,match='idle healthy'):session.set_submission_policy('per_layer_async_v1')
    setattr(session,state,False)
    with pytest.raises(ValueError,match='Invalid'):session.set_submission_policy('other')

def test_submission_variant_is_bound_in_completed_transaction_receipt():
    session,bank,cache=setup();session.set_submission_policy('whole_forward_v1')
    session.forward(None,None,cache,execution(session))
    assert session.last_receipt['submission_policy']=='whole_forward_v1'
    assert session.receipt()['submission_policy']=='whole_forward_v1'

@pytest.mark.parametrize('policy',['layer_group_2_v1','layer_group_4_v1'])
def test_group_submission_choices_are_explicit(policy):
    session,bank,cache=setup()
    assert session.set_submission_policy(policy)['submission_policy']==policy

def test_operand_policy_prepares_before_selecting_and_preserves_full_control():
    session,bank,cache=setup();calls=[]
    session.weights.prepare_resident_coefficient_views=lambda **kw:calls.append(kw['ownercheck'])
    assert session.operand_policy=='full_operands_v1'
    assert session.set_operand_policy('coefficient_views_v1')['operand_policy']=='coefficient_views_v1'and len(calls)==1
    session.set_operand_policy('full_operands_v1');assert len(calls)==1
    session.active=True
    with pytest.raises(RuntimeError,match='idle healthy'):session.set_operand_policy('coefficient_views_v1')
    session.active=False
    session.weights.prepare_resident_coefficient_views=lambda **kw:(_ for _ in ()).throw(RuntimeError('prepare failed'))
    with pytest.raises(RuntimeError,match='prepare failed'):session.set_operand_policy('coefficient_views_v1')
    assert session.operand_policy=='full_operands_v1'
