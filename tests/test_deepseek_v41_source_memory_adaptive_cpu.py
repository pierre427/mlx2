"""Synthetic actual-memory adaptation; no MLX import, allocation or model."""
from dataclasses import replace
import pytest

from mlx2.adapters.deepseek_v41_source_memory import (
    GIB,ORIGINAL_EXPERT_BYTES as E,SourceMemorySnapshot as Snapshot,
    SourceMemoryNeed as Need,plan_source_memory as plan)


@pytest.mark.parametrize('gib',[20,24,32,64,128])
@pytest.mark.parametrize('phase',['decode','prefill'])
@pytest.mark.parametrize('available',[0,64<<20,512<<20,2*GIB,8*GIB])
def test_small_and_large_hosts_attempt_at_actual_memory_with_75GB_max(gib,phase,available):
    s=Snapshot(gib*GIB,available)
    n=Need(phase=phase,requested_prefill_rows=99,dense_estimated_bytes=15*GIB,dense_resident_bytes=12*GIB)
    p=plan(s,n)
    assert p['status']in('adaptive_ready','attempt_minimum_stream')
    assert p['target_cache_bytes']<=75_000_000_000
    assert p['minimum_sequential_total_bytes']<128<<20
    assert p['complete_layer_minimum_bytes']==p['fixed_host_reserve_bytes']==0
    assert not p['permit_reservation_based_rejection']and not p['precision_change']
    assert 1<=p['prefill_chunk_rows']<=99
    if available==0:assert p['expert_strategy']=='sequential_expert'and p['pressure_attempt']


def test_drop_and_recovery_shrink_and_regrow_cache_without_changing_ceiling():
    n=Need()
    base=Snapshot(24*GIB,8*GIB,resident_expert_bytes=2*GIB)
    initial=plan(base,n)
    dropped=plan(replace(base,available_bytes=0),n)
    recovered=plan(replace(base,available_bytes=12*GIB),n)
    assert dropped['target_cache_bytes']<initial['target_cache_bytes']<recovered['target_cache_bytes']
    assert dropped['reclaim_before_allocate']
    assert all(p['cache_ceiling_bytes']==75_000_000_000 for p in (initial,dropped,recovered))


def test_unknown_available_progresses_sequentially_instead_of_rejecting():
    p=plan(Snapshot(24*GIB,None),Need(dense_estimated_bytes=40*GIB))
    assert p['status']=='attempt_minimum_stream'and p['working_set_experts']==1
    assert p['target_cache_bytes']==p['minimum_sequential_total_bytes']
    assert p['dense_streaming_or_paging_likely']


def test_fused_requires_only_six_experts_not_all384():
    p=plan(Snapshot(24*GIB,512<<20),Need())
    assert p['expert_strategy']=='fused_top6'
    assert p['fused_payload_bytes']==6*E
    assert p['fused_total_bytes']<200<<20


def test_prefill_pressure_uses_one_row_and_sequential_expert():
    p=plan(Snapshot(24*GIB,20<<20),Need(phase='prefill',requested_prefill_rows=1024))
    assert p['prefill_chunk_rows']==1 and p['prefill_strategy']=='ordinary_sequential_expert'
    assert p['prefill_partition_changes']


def test_dense_estimate_is_not_subtracted_again_from_live_availability():
    s=Snapshot(24*GIB,2*GIB,resident_expert_bytes=512<<20)
    a=plan(s,Need(dense_estimated_bytes=12*GIB,dense_resident_bytes=12*GIB))
    b=plan(s,Need(dense_estimated_bytes=50*GIB,dense_resident_bytes=12*GIB))
    assert a['target_cache_bytes']==b['target_cache_bytes']
    assert not b['dense_estimate_is_admission_requirement']


def test_imminent_dense_allocation_is_charged_and_pins_are_protected():
    s=Snapshot(24*GIB,128<<20,resident_expert_bytes=GIB,pinned_expert_bytes=GIB)
    n=Need(dense_next_allocation_bytes=3*GIB)
    p=plan(s,n)
    assert p['pressure_attempt']and p['drain_pinned_epochs_before_new_acquire']
    assert p['pinned_payload_bytes']==GIB
    assert p['expert_strategy']=='sequential_expert'


def test_known_allocator_limit_and_actual_failure_are_distinct_from_pressure():
    n=Need(dense_next_allocation_bytes=3*GIB)
    limited=plan(Snapshot(24*GIB,8*GIB,max_single_allocation_bytes=2*GIB),n)
    failed=plan(Snapshot(24*GIB,8*GIB,allocation_failed_bytes=3*GIB),n)
    assert limited['status']=='known_single_allocation_limit'
    assert failed['status']=='actual_allocation_failed'
    assert not failed['known_impossible_single_allocation']
    assert failed['expert_strategy']=='sequential_expert'


def test_actual_failure_reduces_prefill_even_if_the_available_estimate_looks_large():
    p=plan(Snapshot(24*GIB,12*GIB,allocation_failed_bytes=GIB),
           Need(phase='prefill',requested_prefill_rows=99,route_working_set_known=True))
    assert p['prefill_chunk_rows']==1 and p['expert_strategy']=='sequential_expert'


def test_aggregate_workspace_does_not_invent_an_indivisible_allocation_limit():
    p=plan(Snapshot(24*GIB,8*GIB,max_single_allocation_bytes=GIB),
           Need(decode_workspace_bytes=2*GIB))
    assert not p['known_impossible_single_allocation']


def test_fixed_precision_payloads_are_charged_exactly_without_substitution():
    payloads=(E,6635520,E,6635520,E,6635520)
    p=plan(Snapshot(24*GIB,512<<20),Need(expert_payload_bytes=payloads))
    assert p['fused_payload_bytes']==sum(payloads)and not p['precision_change']


def test_too_small_explicit_cache_is_a_config_limit_not_a_host_reservation():
    p=plan(Snapshot(24*GIB,8*GIB),Need(cache_ceiling_bytes=1))
    assert p['status']=='configured_expert_ceiling_too_small'
    assert p['target_cache_bytes']==1


@pytest.mark.parametrize('kwargs',[dict(physical_bytes=0),dict(physical_bytes=True),
    dict(physical_bytes=24*GIB,available_bytes=-1),
    dict(physical_bytes=24*GIB,resident_expert_bytes=1,pinned_expert_bytes=2)])
def test_invalid_measurements_are_not_invented(kwargs):
    with pytest.raises(ValueError):Snapshot(**kwargs)


@pytest.mark.parametrize('kwargs',[dict(cache_ceiling_bytes=-1),dict(expert_payload_bytes=()),
    dict(expert_payload_bytes=(True,)),dict(requested_prefill_rows=0),dict(phase='bad')])
def test_invalid_work_geometry_refuses_before_any_allocation(kwargs):
    with pytest.raises(ValueError):Need(**kwargs)
