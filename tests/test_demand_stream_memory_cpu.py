"""Synthetic bounded streaming admission; no model/GPU qualification."""
from types import SimpleNamespace

import pytest

from mlx2.runtime.demand_stream_memory import (
    DemandStreamHeadroom, DemandStreamCacheBudget, recommended_prefill_rows)
from mlx2.serving import ensure_admission_headroom, prefill_transient_gib


def stream():
    value = SimpleNamespace(adaptive_residency_enabled=True,
                            memory_admission_policy='attempt_after_reclaim',
                            unfilled_reserve_bytes=lambda: 100,
                            calls=[])
    value.rebalance_headroom=lambda **kwargs:value.calls.append(kwargs)
    return value


def test_best_effort_does_not_invent_headroom_or_reserve_max_cache():
    s=stream()
    h=DemandStreamHeadroom(lambda:0,s)
    reclaimed=[]
    assert ensure_admission_headroom(75_000_000_000,headroom=h,
        reclaim=lambda:reclaimed.append(True),evict=lambda:False,evictable=lambda:0)
    assert h()==0 and reclaimed==[True]
    assert s.last_pressure_admission['required_bytes']==75_000_000_000
    assert s.last_pressure_admission['advisory_shortfall_bytes']==75_000_000_000
    assert s.calls==[dict(available_bytes=0,protected_bytes=74_999_999_900)]


def test_normal_route_still_refuses_and_cannot_opt_in_implicitly():
    assert not ensure_admission_headroom(100,headroom=lambda:0,
        reclaim=lambda:None,evict=lambda:False,evictable=lambda:0)
    s=stream();s.adaptive_residency_enabled=False
    with pytest.raises(ValueError):DemandStreamHeadroom(lambda:0,s)


def test_missing_live_probe_attempts_minimum_without_claiming_free_bytes():
    s=stream();h=DemandStreamHeadroom(lambda:None,s)
    assert ensure_admission_headroom(100,headroom=h,reclaim=lambda:None,
                                     evict=lambda:False,evictable=lambda:0)
    assert h()==0 and not s.last_pressure_admission['availability_known']


def test_adaptive_controller_has_no_fixed_service_or_driver_reservation():
    from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController
    for gib in (24,128):
        controller=SelfMTPLaneAdmissionController(host_memory_gib=gib,
            advisory_gib=gib*.8,memory_pressure_policy='attempt_after_reclaim')
        assert controller.hard_reserve_gib==0
        assert controller.service_reserve_gib==controller.driver_allowance_gib==0


def test_live_chunk_cap_drives_admission_and_execution_consistently():
    cap=[1]
    recommend=lambda maximum,**kwargs:min(maximum,cap[0])
    budget=DemandStreamCacheBudget(SimpleNamespace(
        prefill_transient_bytes=lambda context,rows:rows*(1<<30),
        project=lambda *args:42),recommend)
    model=SimpleNamespace(recommended_prefill_rows=recommend)
    for rows in (1,8,64,1):
        cap[0]=rows
        assert recommended_prefill_rows(model,32,context_tokens=100)==min(rows,32)
        assert prefill_transient_gib(budget,context_tokens=100,
            uncached_tokens=99,prefill_step=32)==min(rows,32)
    assert budget.project()==42


@pytest.mark.parametrize('value',[0,33,True,1.5,None])
def test_invalid_chunk_recommendation_fails_before_model_execution(value):
    with pytest.raises(ValueError):
        recommended_prefill_rows(SimpleNamespace(recommended_prefill_rows=lambda *a,**k:value),32)


def test_source_load_does_not_require_dense_plus_unused_cache(monkeypatch):
    from mlx2.adapters.deepseek_v41_source_lfu import source_admission
    from mlx2.adapters import deepseek_v41_source_lfu as weights
    from mlx2.runtime.streamed_load import WeightStreamRequest
    monkeypatch.setattr(weights,'weight_plan',lambda artifact,budget:dict(
        expert_cache_bytes=budget,minimum_expert_cache_bytes=18_800_640,
        dense_payload_bytes=16<<30,dense_prepared_bound_bytes=4<<30,
        total_reservation_bytes=(20<<30)+budget))
    artifact={'args':SimpleNamespace(num_experts_per_tok=6,engram_layer_ids=(0,1))}
    for limit in (None,0,2<<30,24<<30):
        plan=source_admission(artifact,WeightStreamRequest(
            mode='moe_experts',budget_bytes=75_000_000_000,admission_limit_bytes=limit))
        assert plan['expert_cache_bytes']==75_000_000_000
        assert plan['headroom_bytes']==0
        assert plan['memory_admission_policy']=='attempt_after_reclaim'
        assert plan['required_bytes']<128<<20


def test_scheduler_composes_adapter_plan_with_live_memory_cap():
    from unittest.mock import Mock
    from mlx2.runtime.generate import BatchGenerator

    cap = [32]
    recommend = Mock(side_effect=lambda maximum, **kwargs: min(maximum, cap[0]))
    planner = Mock(return_value=128)
    model = SimpleNamespace(_mlx2_prefill_plan=planner,
                            recommended_prefill_rows=recommend)
    scheduler = SimpleNamespace(model=model, prefill_step_size=512,
        prefill_step_autoscale=False, _generation_batch=[1, 2])
    assert BatchGenerator._prompt_prefill_step(scheduler, 1024) == 32
    planner.assert_called_once_with(total_tokens=1024, active_decode_lanes=2,
                                   maximum=512)
    recommend.assert_called_once_with(128, context_tokens=1024)
    cap[0] = 256
    assert BatchGenerator._prompt_prefill_step(scheduler, 1024) == 128
    cap[0] = 1
    assert BatchGenerator._prompt_prefill_step(scheduler, 1024) == 1


@pytest.mark.parametrize('planned', [0, 513, True, 1.5, None])
def test_scheduler_rejects_invalid_adapter_plan_before_memory_cap(planned):
    from unittest.mock import Mock
    from mlx2.runtime.generate import BatchGenerator

    recommend = Mock(return_value=1)
    scheduler = SimpleNamespace(model=SimpleNamespace(
        _mlx2_prefill_plan=lambda **kwargs: planned,
        recommended_prefill_rows=recommend), prefill_step_size=512,
        prefill_step_autoscale=False, _generation_batch=[])
    with pytest.raises(ValueError, match='configured positive maximum'):
        BatchGenerator._prompt_prefill_step(scheduler, 1024)
    recommend.assert_not_called()


def test_scheduler_memory_cap_cannot_widen_adapter_plan():
    from mlx2.runtime.generate import BatchGenerator

    scheduler = SimpleNamespace(model=SimpleNamespace(
        _mlx2_prefill_plan=lambda **kwargs: 128,
        recommended_prefill_rows=lambda maximum, **kwargs: 129),
        prefill_step_size=512, prefill_step_autoscale=False,
        _generation_batch=[])
    with pytest.raises(ValueError, match='within its maximum'):
        BatchGenerator._prompt_prefill_step(scheduler, 1024)
