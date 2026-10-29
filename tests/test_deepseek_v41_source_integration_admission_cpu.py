"""Synthetic CPU profile/admission seams; no checkpoint or GPU execution."""
from types import SimpleNamespace

import pytest

from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter, source_policy
from mlx2.adapters.deepseek_v41_source_auxiliary import (
    stage_auxiliary_plans, auxiliary_memory_requirement, check_auxiliary_memory,
)


@pytest.mark.parametrize('extension', [
    {'deepseek_kv_storage': 'native_packed'},
    {'deepseek_candidate_score': 'candidate_only'},
    {'deepseek_ced': 'dependency_exact'},
    {'deepseek_vision': 'enabled'},
    {'deepseek_physical_batch': 'b2'},
    {'deepseek_physical_batch': 'b4'},
    {'draft_model': 'embedded_dspark'},
])
@pytest.mark.parametrize('incoming', [
    {'deepseek_source_dspark_ast': 'tile3x5_v1'},
    {'deepseek_source_dspark_prepare': 'oracle_v1'},
    {'deepseek_source_resident_gather': 'resident_bank_v1'},
    {'deepseek_source_resident_prefill': 'expert_major_v1'},
    {'deepseek_source_prefill_partition': 'canonical_m1_v1'},
])
def test_crossprofile_refusal_precedes_oracle_or_device_loading(extension, incoming):
    with pytest.raises(ValueError, match='Source profile combination is unsupported'):
        source_policy({**extension, **incoming})


@pytest.mark.parametrize('policy', [
    {}, {'deepseek_source_prefill_partition': 'adaptive_v1'},
    {'deepseek_source_prefill_partition': 'canonical_m1_v1'},
    {'deepseek_kv_storage': 'native_packed'}, {'deepseek_ced': 'dependency_exact'},
    {'deepseek_vision': 'enabled'}, {'deepseek_physical_batch': 'b2'},
    {'deepseek_physical_batch': 'b4'}, {'draft_model': 'embedded_dspark'},
])
def test_independent_profiles_still_parse(policy):
    assert isinstance(source_policy(policy), dict)


def test_direct_ast_configuration_rejects_incompatible_loaded_profile_before_binding():
    adapter = SimpleNamespace(kernel_policy={'deepseek_kv_storage': 'native_packed'})
    with pytest.raises(ValueError, match='profile combination is unsupported'):
        DeepSeekV41SourceAdapter.configure_dspark_ast(adapter, oracle_path='must-not-be-read')


def test_direct_bank_configuration_rejects_incompatible_profile_before_bank_access():
    adapter = SimpleNamespace(kernel_policy={'deepseek_vision': 'enabled'})
    with pytest.raises(ValueError, match='profile combination is unsupported'):
        DeepSeekV41SourceAdapter.prepare_resident_bank_session(adapter, None, ownercheck=lambda: True)


def admission():
    # Huge advisory target estimates/maxima must have no effect on eager-only accounting.
    value = dict(imminent_working_reserve_bytes=10, expert_cache_bytes=75_000_000_000,
                 estimated_full_reservation_bytes=140_000_000_000, admission_limit_bytes=1000)
    stage_auxiliary_plans(value, {'dspark': dict(payload_bytes=100, prepared_bound_bytes=20,
        staging_bound_bytes=30, workspace_bound_bytes=40, reservation_bytes=190),
        'vision': dict(payload_bytes=50, staging_bound_bytes=10,
                       workspace_bound_bytes=20, reservation_bytes=80)})
    return value


def test_plan_staging_preserves_explicit_auxiliary_bounds():
    value = admission()
    assert value['auxiliary_reservation_bytes'] == 270
    assert value['dspark_reservation_bytes'] == 190
    assert value['auxiliary_workspace_bound_bytes'] == 60
    assert auxiliary_memory_requirement(value) == 280
    assert auxiliary_memory_requirement(value, loaded=('dspark',)) == 130
    assert auxiliary_memory_requirement(value, loaded=('dspark', 'vision')) == 70


def test_loaded_payload_is_measured_once_and_future_workspaces_remain_reserved():
    value = admission()
    receipt = check_auxiliary_memory(value, loaded=('dspark', 'vision'), active_bytes=930, available_bytes=70)
    assert receipt['future_allocation_and_workspace_bytes'] == 70
    assert not receipt['expert_cache_maximum_reserved']
    assert not receipt['full_target_dense_estimate_reserved']
    with pytest.raises(MemoryError, match='exceed measured headroom'):
        check_auxiliary_memory(value, loaded=('dspark', 'vision'), active_bytes=931, available_bytes=1000)
    with pytest.raises(MemoryError, match='exceed measured headroom'):
        check_auxiliary_memory(value, loaded=('dspark', 'vision'), active_bytes=0, available_bytes=69)


@pytest.mark.parametrize('available', [None, -1, True, 1.5])
def test_auxiliary_requires_known_measured_availability(available):
    with pytest.raises(MemoryError, match='measured available memory'):
        check_auxiliary_memory(admission(), active_bytes=0, available_bytes=available)


def test_unknown_process_cap_uses_actual_availability():
    value = admission(); value['admission_limit_bytes'] = None
    assert check_auxiliary_memory(value, active_bytes=1 << 40, available_bytes=280)


def test_plan_refuses_omitted_workspace_or_unstaged_loaded_owner():
    with pytest.raises(ValueError, match='omits eager payload or workspace'):
        stage_auxiliary_plans({}, {'draft': dict(payload_bytes=100, workspace_bound_bytes=10, reservation_bytes=100)})
    with pytest.raises(ValueError, match='loaded inventory differs'):
        auxiliary_memory_requirement(admission(), loaded=('foreign',))


def test_adapter_reclaims_optional_experts_then_remeasures(monkeypatch):
    import mlx.core as mx
    from mlx2 import memory
    measurements = iter((1000, 2000))
    monkeypatch.setattr(memory, 'host_available_bytes', lambda **kwargs: next(measurements))
    monkeypatch.setattr(mx, 'get_active_memory', lambda: 200)
    calls = []
    adapter = SimpleNamespace(native_admission=admission(), weight_stream=SimpleNamespace(
        rebalance_headroom=lambda **kwargs: calls.append(kwargs)))
    receipt = DeepSeekV41SourceAdapter._check_auxiliary_memory(adapter, loaded=())
    assert calls == [dict(available_bytes=1000, protected_bytes=480)]
    assert receipt['available_bytes'] == 2000
    assert receipt['future_allocation_and_workspace_bytes'] == 280
    assert adapter.native_admission['auxiliary_memory_receipt'] == receipt


@pytest.mark.parametrize('width', ['b2', 'b4'])
def test_batch_accepts_explicit_incoming_noop_defaults(width):
    policy = source_policy(dict(deepseek_physical_batch=width,
        deepseek_source_prefill_partition='adaptive_v1', deepseek_source_dspark_ast='off',
        deepseek_source_dspark_prepare='off'))
    assert policy['deepseek_physical_batch'] == width


def test_ast_auxiliary_plan_is_staged_before_target_materialization(monkeypatch):
    from mlx2 import process_env
    from mlx2.adapters import deepseek_v41_source as source
    from mlx2.adapters import deepseek_v41_source_lfu as lfu
    from mlx2.adapters import deepseek_v41_source_auxiliary as auxiliary
    from mlx2.adapters import deepseek_v41_dspark as dspark
    from mlx2.runtime.streamed_load import WeightStreamRequest
    artifact = dict(identity={'fingerprint': 'a' * 64}, config={},
                    args=SimpleNamespace(), root='fixture')
    adapter = object.__new__(source.DeepSeekV41SourceAdapter)
    adapter.kernel_policy = {'deepseek_source_dspark_ast': 'tile3x5_v1'}
    adapter.weight_streaming_request = WeightStreamRequest(mode='moe_experts', budget_bytes=64 << 30)
    monkeypatch.setattr(source, 'inspect_artifact', lambda _: artifact)
    for name in ('require_process_numerics', 'clear_inherited_profile', 'assert_process_profile_applied'):
        monkeypatch.setattr(process_env, name, lambda *a, **k: None)
    events = []
    plan = dict(payload_bytes=100, workspace_bound_bytes=20, reservation_bytes=120)
    monkeypatch.setattr(dspark, 'source_dspark_admission_plan',
        lambda *a, **k: (events.append('plan'), plan)[1])
    monkeypatch.setattr(lfu, 'source_admission', lambda *a, **k: dict(imminent_working_reserve_bytes=10))
    def stop_before_materialization(admission, plans):
        assert plans == {'dspark_ast': plan}
        events.append('stage')
        raise MemoryError('staged before target materialization')
    monkeypatch.setattr(auxiliary, 'stage_auxiliary_plans', stop_before_materialization)
    with pytest.raises(MemoryError, match='staged before target materialization'):
        adapter._load('fixture')
    assert events == ['plan', 'stage']
