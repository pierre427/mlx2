"""Synthetic24/128GiB residency accounting, no GPU/model/real-memory probe."""
from types import SimpleNamespace
import threading

import pytest

from mlx2.adapters import deepseek_v41_source_lfu as lfu
from mlx2.adapters.deepseek_v41_mlx_weights import NativeExpertStore
from mlx2.runtime.streamed_load import WeightStreamRequest


def store(*,available=4<<30,pinned=(),items=()):
    s=lfu.LFUSourceExpertStore.__new__(lfu.LFUSourceExpertStore)
    s.cache_budget_bytes=s.maximum_cache_bytes=75_000_000_000;s.minimum_cache_bytes=lfu.ORIGINAL_BYTES
    s.bounds={'expert_staging_bound_bytes':lfu.ORIGINAL_BYTES};s.plan=SimpleNamespace(transient_bound_bytes=0)
    s._cache={};s._leased={};s._pending_uploads={};s._pending_host={};s._cache_bytes=0
    s._lock=threading.RLock();s.ple_tables={};s._parked=s._active=s._closed=False
    s.q1_root='/q1';s.hot=frozenset((0,i)for i in range(3));s.index=lfu.LFUIndex()
    s.policy=dict(mode='hot_original_q1',budget_bytes=75_000_000_000,hot_bytes=60_000_000_000)
    s.pool_bytes={'original':0,'q1':0};s.stats={'evictions':0,'windows':0};s.files={}
    s._ready=lambda:None;s._fence=lambda:None
    s._adaptive_reclaim_allocator=lambda:None
    for i in items:
        key=(0,i);kind=s.kind(key);size=lfu.Q1_BYTES if kind=='q1'else lfu.ORIGINAL_BYTES
        s._cache[key]={'weights':SimpleNamespace(nbytes=size)};s._cache_bytes+=size;s.pool_bytes[kind]+=size
        s.index.touch(key)
        if i in pinned:s._leased[key]=s._cache[key]
    s.configure_adaptive_residency(available_bytes=lambda:available,working_set_experts=1)
    return s


@pytest.mark.parametrize('available',[0,None,4<<30,100<<30])
def test_actual24gb_or128gb_availability_never_requires75gb_unallocated_reserve(available):
    s=store(available=available)
    result=s.adaptive_residency_update()
    assert s.adaptive_residency_enabled and s.memory_admission_policy=='attempt_after_reclaim'
    assert s.unfilled_reserve_bytes()==2*lfu.ORIGINAL_BYTES
    assert result['effective_cache_target_bytes']<=75_000_000_000
    if available==4<<30:assert result['effective_cache_target_bytes']==4<<30
    if available==100<<30:assert result['effective_cache_target_bytes']==75_000_000_000


def test_serialized_plan_distinguishes_maximum_from_live_working_reservation():
    from mlx2.adapters.deepseek_v41_mlx_weights import NativeExpertPlan
    s=store(available=0)
    s.plan=NativeExpertPlan(ceiling_bytes=s.maximum_cache_bytes,expert_bytes=lfu.ORIGINAL_BYTES)
    s.configure_adaptive_residency(available_bytes=lambda:0,working_set_experts=1)
    assert s.plan.as_dict()['configured_capacity_bytes']==75_000_000_000
    assert s.plan.as_dict()['reserved_bytes']==s.reserved_bytes()==2*lfu.ORIGINAL_BYTES
    s._parked=True
    assert s.reserved_bytes()==s.plan.as_dict()['reserved_bytes']==0
    legacy=NativeExpertPlan(ceiling_bytes=1000,transient_bound_bytes=100)
    assert legacy.as_dict()['reserved_bytes']==1100


def test_live_pressure_evicts_unleased_hot_objects_without_changing_format_map():
    s=store(items=(0,1,2,3),pinned=(0,));hot=s.hot;original=s._cache[(0,0)]
    result=s.set_adaptive_cache_target(2*lfu.ORIGINAL_BYTES,working_set_experts=1)
    assert set(s._cache)=={(0,0)}and s._cache[(0,0)]is original
    assert s.pool_bytes=={'original':lfu.ORIGINAL_BYTES,'q1':0}
    assert s.hot is hot and s.kind((0,1))=='original'and s.kind((0,3))=='q1'
    assert result['evicted_bytes']==2*lfu.ORIGINAL_BYTES+lfu.Q1_BYTES


def test_pinned_objects_exceeding_soft_target_are_retained_then_trim_after_completion():
    s=store(items=(0,1),pinned=(0,1));old=dict(s._cache)
    result=s.set_adaptive_cache_target(2*lfu.ORIGINAL_BYTES,working_set_experts=1)
    assert result['deferred_for_pins']and s._cache==old
    s._leased.clear();s.adaptive_residency_update()
    assert s._cache_bytes==lfu.ORIGINAL_BYTES and not s._adaptive_last['deferred_for_pins']


def test_incoming_expert_does_not_evict_or_recycle_active_pins_under_soft_pressure():
    s=store(items=(0,1),pinned=(0,1));s.set_adaptive_cache_target(2*lfu.ORIGINAL_BYTES,working_set_experts=1)
    s._evict_for((0,3),lfu.Q1_BYTES)
    assert set(s._cache)=={(0,0),(0,1)}and s._adaptive_last['deferred_for_pins']


def test_pending_upload_cannot_be_evicted_and_full_host_owner_is_reserved():
    s=store(items=(0,0));s._pending_uploads[(0,0)]=s._cache[(0,0)]
    s._pending_host[(0,0)]=SimpleNamespace(host_allocation_nbytes=lfu.ORIGINAL_BYTES+128)
    s.set_adaptive_cache_target(2*lfu.ORIGINAL_BYTES,working_set_experts=1)
    assert (0,0)in s._cache
    assert s.reserved_bytes()>=s.live_bytes()+lfu.ORIGINAL_BYTES+128


def test_native_other_adapter_default_reservation_semantics_are_preserved():
    s=NativeExpertStore.__new__(NativeExpertStore);s.cache_budget_bytes=75_000_000_000
    s.plan=SimpleNamespace(transient_bound_bytes=123);s.ple_tables={};s._parked=False
    assert not s.adaptive_residency_enabled and s.memory_admission_policy=='bounded_reservation'
    assert s.reserved_bytes()==75_000_000_123


def test_preheat_hint_never_materializes_configured75gb_ceiling_on_adaptive_source():
    s=store();s.preheat_receipt={'requested':True,'performed':False}
    s.preheat_selection=lambda:(_ for _ in ()).throw(AssertionError('full preheat forbidden'))
    assert not s.preheat()['performed']
    assert 'demand'in s.preheat()['reason']


def test_pressure_receipt_reports_actual_shortfall_without_fabricated_free_credit():
    s=store();r=s.record_pressure_attempt(1000,100)
    assert r['shortfall_bytes']==900 and r['available_bytes']==100 and r['free_memory_credit_bytes']==0


@pytest.mark.parametrize('policy',[None,{'mode':'original_lfu','budget_bytes':75_000_000_000}])
def test_source_admission_attempts24gb_with_estimate_larger_than_ram(monkeypatch,policy):
    monkeypatch.setattr(lfu,'weight_plan',lambda artifact,budget:dict(minimum_expert_cache_bytes=lfu.ORIGINAL_BYTES,
        total_reservation_bytes=140_000_000_000,dense_payload_bytes=16<<30,dense_prepared_bound_bytes=4<<30,
        dense_staging_bound_bytes=3<<30,expert_cache_bytes=budget))
    request=WeightStreamRequest(mode='moe_experts',budget_bytes=75_000_000_000,admission_limit_bytes=24<<30)
    result=lfu.source_admission({},request,policy)
    assert result['required_bytes']==2*lfu.ORIGINAL_BYTES and result['estimates_are_advisory']
    assert result['estimated_full_reservation_bytes']>24<<30
    assert result['expert_cache_bytes']==75_000_000_000 and not result['full_cache_preallocation']


def test_policy_minimum_is_progressing_expert_plus_staging_not_whole_layer():
    assert lfu.parse_policy({'mode':'original_lfu','budget_bytes':2*lfu.ORIGINAL_BYTES})['budget_bytes']==2*lfu.ORIGINAL_BYTES


def test_stale_planner_override_cannot_hide_dropped_live_availability():
    available=[100<<30];s=store();s.configure_adaptive_residency(available_bytes=lambda:available[0],working_set_experts=1)
    s.set_adaptive_cache_target(75_000_000_000);available[0]=0
    r=s.adaptive_residency_update()
    assert r['effective_cache_target_bytes']==2*lfu.ORIGINAL_BYTES


def test_elastic_headroom_rebalance_uses_measured_protection_and_no_free_credit():
    s=store(items=(0,1,2,3));s.set_adaptive_cache_target(75_000_000_000)
    r=s.rebalance_headroom(available_bytes=0,protected_bytes=1<<30)
    assert r['effective_payload_target_bytes']==lfu.ORIGINAL_BYTES
    assert s._cache_bytes<=lfu.ORIGINAL_BYTES and r['free_memory_credit_bytes']==0


def test_completed_release_wait_failure_retains_every_lease_and_generation():
    s=store(items=(0,),pinned=(0,));s._active=True;owned=s._leased[(0,0)]
    def failed(roots):raise TimeoutError('not complete')
    with pytest.raises(TimeoutError):s.release_completed_experts(((0,0),),roots=(object(),),wait=failed)
    assert s._leased[(0,0)]is owned
    result=s.release_completed_experts(((0,0),),roots=(object(),),wait=lambda roots:None)
    assert result['released']==1 and not s._leased


def test_early_release_refuses_key_epoch_collector_to_prevent_aba():
    s=store(items=(0,),pinned=(0,));s._active=True;s._completion_collector=lambda key:None
    with pytest.raises(ValueError,match='ordinary'):s.release_completed_experts(((0,0),),roots=(object(),),wait=lambda roots:None)
    assert (0,0)in s._leased


@pytest.mark.parametrize('budget',[lfu.ORIGINAL_BYTES,90_000_000_001])
def test_plain_source_maximum_has_configured_progressing_granule_bounds(budget):
    request=WeightStreamRequest(mode='moe_experts',budget_bytes=budget,admission_limit_bytes=24<<30)
    with pytest.raises(ValueError,match='one expert plus staging'):lfu.source_admission({},request)


def test_actual8k_decode_set_q1_exceeds_old15gb_pool_without_any_eviction():
    """Metadata-only reproduction:4166 real-format experts, no tensor allocation."""
    s=store(available=100<<30)
    keys=[divmod(i,384)for i in range(4166)]
    s.hot=frozenset(keys[:1453]);hot=s.hot
    for key in keys[:-1]:
        kind=s.kind(key);size=lfu.ORIGINAL_BYTES if kind=='original'else lfu.Q1_BYTES
        s._cache[key]={'weights':SimpleNamespace(nbytes=size)}
        s._cache_bytes+=size;s.pool_bytes[kind]+=size;s.index.touch(key)
    before=set(s._cache)
    assert s.pool_bytes['q1']>75_000_000_000-60_000_000_000-lfu.ORIGINAL_BYTES
    s._evict_for(keys[-1],lfu.Q1_BYTES)
    assert set(s._cache)==before and s.stats['evictions']==0
    assert s._cache_bytes+lfu.Q1_BYTES==45_319_495_680
    assert s.pool_bytes['original']==27_317_329_920
    assert s.pool_bytes['q1']+lfu.Q1_BYTES==18_002_165_760
    assert s.hot is hot and all(s.kind(k)=='original'for k in s.hot)


def test_adaptive_shared_total_can_evict_other_format_but_never_leased_or_pending():
    s=store(available=100<<30,items=(0,1,3))
    # q1 is more frequent than both original objects. Shared capacity must use
    # global LFU, including an original victim when the incoming expert is q1.
    s.index.counts[(0,3)]=100
    s.maximum_cache_bytes=s.cache_budget_bytes=2*lfu.ORIGINAL_BYTES+lfu.Q1_BYTES
    protected=s._cache[(0,1)];s._leased[(0,1)]=protected;s._pending_uploads[(0,1)]=protected
    s._evict_for((0,4),lfu.Q1_BYTES)
    assert (0,0)not in s._cache and s._cache[(0,1)]is protected
    assert s._cache_bytes+lfu.Q1_BYTES<=s.maximum_cache_bytes-lfu.ORIGINAL_BYTES
    assert s.pool_bytes['original']==lfu.ORIGINAL_BYTES


@pytest.mark.parametrize('protection',['lease','pending','bank'])
def test_adaptive_total_cap_refuses_to_recycle_protected_payload(protection):
    s=store(available=100<<30,items=(0,1))
    s.maximum_cache_bytes=s.cache_budget_bytes=3*lfu.ORIGINAL_BYTES
    owners=dict(s._cache)
    if protection=='lease':s._leased.update(owners)
    if protection=='pending':s._pending_uploads.update(owners)
    if protection=='bank':s._resident_bank_pinned_keys=frozenset(owners)
    with pytest.raises(MemoryError,match='protected/active'):
        s._evict_for((0,3),lfu.Q1_BYTES)
    assert s._cache==owners and s._cache_bytes==2*lfu.ORIGINAL_BYTES


def test_capacity_receipt_separates_adaptive_sharing_from_fixed_precision_map(monkeypatch):
    from mlx2.adapters.deepseek_v41_source_weights import SourceExpertStore
    monkeypatch.setattr(SourceExpertStore,'receipt',lambda self:{'plan':{}})
    s=store();s.q1_binding='q1-binding';s.hot_binding='hot-binding';s.peak_payload_bytes=0
    s.atlas_path=None;s.preheat_receipt={}
    value=s.receipt()
    assert value['pool_capacity_policy']=='shared_total_adaptive'
    assert value['hot_budget_role']=='immutable_precision_selection_only'
    assert value['precision_selection']=='fixed per expert for profile lifetime'
    s._adaptive_residency_enabled=False
    assert s.receipt()['pool_capacity_policy']=='legacy_fixed_kind_pools'
