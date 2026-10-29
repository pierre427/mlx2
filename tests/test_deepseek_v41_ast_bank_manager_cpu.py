"""CPU ownership/pressure/recovery tests; no model checkpoint or GPU run."""
from types import SimpleNamespace
import pytest
from mlx2.adapters.deepseek_v41_ast_bank_manager import SourceASTBankManager,M1,RESIDENT
from mlx2.adapters.deepseek_v41_ast_batched_moe import ResidentCohortMiss

MiB=1<<20
class Bank:
    generation=3
    def __init__(self,store,payload=8*MiB):
        self.store=store;self.plan=SimpleNamespace(payload_bytes=payload,keys=frozenset({(0,1)}))
        self.closed=False;self.closes=0;self.leases=0;self.fail_close=False
    def assert_current(self,**kw):
        if self.store._resident_bank is not self:raise ValueError('stale generation')
    def receipt(self):return {'active_leases':self.leases,'published':not self.closed,'generation':3}
    def close(self):
        self.closes+=1
        if self.fail_close:raise RuntimeError('failed fence')
        self.closed=True;self.store._resident_bank=None


def setup(*,available=2<<30,bank=True,payload=8*MiB,ceiling=4<<30):
    store=SimpleNamespace(_resident_bank=None,_cache={},_cache_bytes=0,minimum_cache_bytes=MiB,
        cache_budget_bytes=ceiling,maximum_cache_bytes=ceiling,_adaptive_staging=MiB)
    weights=SimpleNamespace(expert_stream=store,source_resident_gather_policy='reference',source_resident_bank_session=None)
    model=SimpleNamespace(weights=weights,_weight_executions=0,_mlx2_ds41_ast_active=None,_mlx2_ds41_ast_poisoned=False)
    manager=SourceASTBankManager(model,ownercheck=lambda:True,available_bytes=lambda:available)
    if bank:manager.bank=Bank(store,payload);store._resident_bank=manager.bank
    return model,manager,store


def answer():return SimpleNamespace(receipt={})

def test_genuine_miss_recovers_once_without_dropping_existing_shared_backing():
    model,m,store=setup();old=m.bank;calls=[]
    def attempt(law):
        calls.append(law)
        if law==RESIDENT:raise ResidentCohortMiss()
        assert store._resident_bank is old and not old.closed
        m.record_routes('blk.0',[[1,2]]);return answer()
    result=m.execute(model,attempt)
    assert calls==[RESIDENT,M1]and old.closes==0 and m.bank is old
    assert result.receipt['hybrid_bank']['exact_M1_recovery']
    assert m.last_keys==frozenset({(0,1),(0,2)})and m.failed_coverage
    assert model.weights.source_resident_gather_policy=='reference'


@pytest.mark.parametrize('error',[ValueError('stale generation'),RuntimeError('unknown route error'),InterruptedError('cancel')])
def test_stale_cancel_and_unknown_failures_never_use_m1_recovery(error):
    model,m,store=setup();calls=[]
    def attempt(law):calls.append(law);raise error
    with pytest.raises(type(error)):m.execute(model,attempt)
    assert calls==[RESIDENT]and m.stats['m1_trees']==0


def test_poisoned_failed_tree_cannot_replay_and_retains_bank_owner():
    model,m,store=setup();old=m.bank;calls=[]
    def attempt(law):
        calls.append(law);model._mlx2_ds41_ast_poisoned=True;raise ResidentCohortMiss()
    with pytest.raises(RuntimeError,match='completed private tree'):m.execute(model,attempt)
    assert calls==[RESIDENT]and m.poisoned and m.bank is old and not old.closed
    with pytest.raises(RuntimeError,match='quarantined'):m.close()


def test_full_pinned_hard_capacity_retires_whole_bank_before_new_expert_progress():
    model,m,store=setup(payload=150*MiB,ceiling=160*MiB);old=m.bank;calls=[]
    def attempt(law):
        calls.append(law);assert law==M1 and store._resident_bank is None
        return answer()
    result=m.execute(model,attempt)
    assert old.closed and old.closes==1 and calls==[M1]
    assert result.receipt['hybrid_bank']['last_retirement']['whole_backing_cohorts']
    assert m.cooldown_remaining==1


@pytest.mark.parametrize('available',[None,0,1024])
def test_24gb_or_unknown_available_system_runs_m1_instead_of_reservation_bailout(available):
    model,m,store=setup(available=available,bank=False,ceiling=75_000_000_000)
    model.physical_memory_bytes=24<<30;calls=[]
    result=m.execute(model,lambda law:(calls.append(law)or answer()))
    assert calls==[M1]and result.receipt['hybrid_bank']['actual_expert_route']=='m1'
    assert m.bank is None and not m.poisoned


def test_lease_blocks_replacement_or_retirement():
    model,m,store=setup();m.bank.leases=1
    with pytest.raises(ValueError,match='live leases'):m.execute(model,lambda law:answer())
    assert m.bank.closes==0


def test_pressure_retirement_failed_fence_quarantines_without_replay():
    model,m,store=setup(available=0);old=m.bank;old.fail_close=True;calls=[]
    with pytest.raises(RuntimeError,match='failed fence'):m.execute(model,lambda law:(calls.append(law)or answer()))
    assert not calls and m.poisoned and m.bank is old and not old.closed


def test_miss_cooldown_avoids_repeated_failed_resident_trees_and_rebuild_churn():
    model,m,store=setup();calls=[]
    def attempt(law):
        calls.append(law)
        if law==RESIDENT:raise ResidentCohortMiss()
        return answer()
    m.execute(model,attempt);m.execute(model,attempt)
    assert calls==[RESIDENT,M1,M1]and m.stats['cohort_misses']==1
    assert m.last['reason']=='cohort_miss_cooldown'and m.stats['bank_builds']==0


def test_speculative_physical_tree_predictions_never_become_committed_hints(monkeypatch):
    model,m,store=setup(bank=False)
    monkeypatch.setattr(m,'hint_binding',lambda:dict(checkpoint='artifact',source_revision='revision',quantization='q',layout='l',resource_abi='a'))
    m.last_keys=frozenset({(0,2)});m.observed[(0,2)]=3
    assert m.snapshot_hints(covered_tokens=9,snapshot_revision='snapshot',token_digest='a'*64)is None
    hint=m.record_committed_experts(((0,1),),covered_tokens=9,snapshot_revision='snapshot',token_digest='a'*64)
    assert hint.experts==((0,1),)and hint.phase=='committed'
    assert m.snapshot_hints(covered_tokens=9,snapshot_revision='wrong',token_digest='a'*64)is None
    assert m.snapshot_hints(covered_tokens=9,snapshot_revision='snapshot',token_digest='a'*64)is hint


def test_hint_prefetch_is_advisory_bound_and_does_not_claim_IO_or_change_real_routes(monkeypatch):
    from dataclasses import replace
    model,m,store=setup(bank=False)
    monkeypatch.setattr(m,'hint_binding',lambda:dict(checkpoint='artifact',source_revision='revision',quantization='q',layout='l',resource_abi='a'))
    hint=m.record_committed_experts(((0,1),),covered_tokens=9,snapshot_revision='snapshot',token_digest='a'*64)
    assert not m.prefetch(replace(hint,phase='speculative'),prefix_tokens=9)['accepted']
    assert not m.prefetch(replace(hint,source_revision='old'),prefix_tokens=9)['accepted']
    receipt=m.prefetch(hint,prefix_tokens=8)
    assert receipt['accepted']and receipt['coverage']=='superset'and receipt['reservation_bytes']==0
    assert not receipt['expert_IO']and not receipt['numerical_authority']and not m.observed
    assert m.hint_predictions[(0,1)]==1 and not store._cache


def provision_fixture():
    # Reuse the independently byte/alias-tested packed CPU backend. This
    # exercises the real manager planning/build/replacement code, not a stub.
    from test_deepseek_v41_source_bank_cpu import store,expert,CPU
    cache=store(((0,1),expert()),((0,2),expert(seed=19)))
    cache.kind=lambda key:cache._cache[key].layout.kind
    cache.cache_budget_bytes=cache.maximum_cache_bytes=4<<30
    cache.minimum_cache_bytes=18_800_640;cache._adaptive_staging=18_800_640
    weights=SimpleNamespace(expert_stream=cache,source_resident_gather_policy='reference',source_resident_bank_session=None)
    model=SimpleNamespace(weights=weights,_weight_executions=0,_mlx2_ds41_ast_active=None,_mlx2_ds41_ast_poisoned=False)
    manager=SourceASTBankManager(model,ownercheck=lambda:True,available_bytes=lambda:512*MiB,backend=CPU())
    return model,manager,cache,expert


def test_real_packed_cached_provision_reuse_and_superset_growth_keep_bytes_and_accounting():
    model,m,store,expert=provision_fixture();before=store._cache_bytes;calls=[]
    def attempt(law):calls.append(law);return answer()
    m.execute(model,attempt);old=m.bank
    assert calls==[RESIDENT]and old is not None and store._cache_bytes==before
    m.execute(model,attempt)
    assert m.bank is old and m.stats['bank_builds']==1
    store._cache[(0,3)]=expert(seed=23);store._cache_bytes+=store._cache[(0,3)].allocation_nbytes
    store.pool_bytes['q1']+=store._cache[(0,3)].allocation_nbytes
    before=store._cache_bytes;m.failed_coverage=True;m.misses_since_build=2;m.last_keys=frozenset(store._cache)
    m.execute(model,attempt)
    assert m.bank is not old and old._closed and m.stats['bank_replacements']==1
    assert store._cache_bytes==before and set(m.bank.plan.keys)=={(0,1),(0,2),(0,3)}
    assert model.weights.source_resident_gather_policy=='reference'
    m.close();assert not store._cache and m.closed


@pytest.mark.parametrize('unsafe',[False,True])
def test_real_plan_allocation_failure_m1_progress_only_after_safe_completion(unsafe):
    model,m,store,_=provision_fixture();calls=[]
    def allocation(*args,**kwargs):
        if unsafe:store._error='unknown completion'
        raise MemoryError('actual allocator failure')
    m.builder=allocation
    if unsafe:
        with pytest.raises(MemoryError):m.execute(model,lambda law:(calls.append(law)or answer()))
        assert m.poisoned and not calls
    else:
        result=m.execute(model,lambda law:(calls.append(law)or answer()))
        assert calls==[M1]and result.receipt['hybrid_bank']['reason']=='actual_bank_allocation_failed'
        assert not m.poisoned and m.stats['construction_allocation_failures']==1


def test_undrainable_bank_scope_is_quarantined_even_if_second_model_drain_was_safe():
    model,m,store=setup();old=m.bank
    def attempt(law):
        model.weights._source_ast_bank_scope=SimpleNamespace(poisoned=True)
        raise RuntimeError('scope completion uncertain')
    with pytest.raises(RuntimeError,match='scope completion'):m.execute(model,attempt)
    assert m.poisoned and m.bank is old and not old.closed
    with pytest.raises(RuntimeError,match='quarantined'):m.close()


def test_request_cancellation_blocks_work_but_not_safe_owned_idle_retirement():
    model,m,store=setup();old=m.bank;m.cancelled=lambda:True
    with pytest.raises(InterruptedError):m.execute(model,lambda law:answer())
    m.close()
    assert m.closed and old.closed and not m.poisoned
