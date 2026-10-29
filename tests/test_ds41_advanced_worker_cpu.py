# SPDX-License-Identifier: MIT
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('ds41_advanced_worker',Path(__file__).resolve().parents[1]/'scripts/research/ds41_advanced_worker.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_action_policy_admission_has_no_gpu_import():
    with pytest.raises(ValueError):m.validate_policy('canonical_resources',{})
    m.validate_policy('canonical_resources',{'deepseek_source_prefill_partition':'canonical_m1_v1','deepseek_source_expert_math':'routed_fused_fp32'})
    with pytest.raises(ValueError):m.validate_policy('resident_gather',{})
    m.validate_policy('resident_gather',{'deepseek_source_resident_gather':'resident_bank_v1','deepseek_source_expert_math':'routed_fused_fp32','deepseek_source_bf16_loads':'woa_f32'})


def test_dspark_cannot_use_placeholder_oracle_or_missing_real_cohort_callbacks():
    policy={'deepseek_source_expert_math':'routed_fused_fp32','deepseek_source_bf16_loads':'woa_f32','deepseek_source_dspark_prepare':'oracle_v1','deepseek_source_dspark_target_forward':'fully_batched_m1_v1',
            'deepseek_source_dspark_target_expert_law':'source_ast_m1_rows_v1'}
    m.validate_policy('dspark_prepare',policy)
    with pytest.raises(ValueError):m.validate_policy('dspark_ast',{**policy,'deepseek_source_dspark_ast':'tile3x5_v1'})
    with pytest.raises(ValueError):m.validate_policy('dspark_prepare',{**policy,'deepseek_source_dspark_target_expert_law':'source_ast_hybrid_rows_v1'})


def test_native_and_exact_resident_hybrid_policy_maps():
    base={'deepseek_source_expert_math':'routed_fused_fp32','deepseek_source_bf16_loads':'woa_f32'}
    m.validate_policy('dspark_native_batched',base)
    m.validate_policy('dspark_ordinary_m16',base)
    with pytest.raises(ValueError):m.validate_policy('dspark_native_batched',{**base,'deepseek_source_dspark_ast':'tile3x5_v1'})
    prepared={**base,'deepseek_source_dspark_prepare':'oracle_v1','deepseek_source_dspark_target_forward':'fully_batched_m1_v1'}
    m.validate_policy('dspark_resident_exact',{**prepared,'deepseek_source_dspark_target_expert_law':'source_ast_resident_rows_v1'})
    hybrid={**prepared,'deepseek_source_dspark_target_expert_law':'source_ast_hybrid_rows_v1'}
    with pytest.raises(ValueError):m.validate_policy('dspark_hybrid_exact',hybrid)
    m.validate_policy('dspark_hybrid_exact',{**hybrid,'deepseek_source_prefill_partition':'canonical_m1_v1'})


def test_complete_action_maps_admit_and_missing_woa_refuses_before_device_import():
    for action in m.ACTIONS:
        if not action.startswith('dspark_'):continue
        policy=m.dspark_action_policy(action);m.validate_policy(action,policy)
        if action not in ('dspark_native_batched','dspark_ordinary_m16'):
            del policy['deepseek_source_bf16_loads']
            with pytest.raises(ValueError,match='WoA'):m.validate_policy(action,policy)
    policy=m.dspark_action_policy('dspark_prepare')
    with pytest.raises(ValueError):m.validate_policy('dspark_prepare',{**policy,'deepseek_source_ffn_input':'shared_routed_qat_reuse'})
    with pytest.raises(ValueError):m.validate_policy('dspark_prepare',{**policy,'deepseek_kv_storage':'native_packed'})


def test_admitted_index_fixture_retains_scoring_and_rejects_short_geometry():
    for count in (258,265,384):
        plan=m.ast_fixture_plan(count)
        assert plan['minimum_index_columns']>=129 and plan['index_scoring_preserved']is True
    for count in (9,128,257,385,True):
        with pytest.raises(ValueError):m.ast_fixture_plan(count)


def test_narrow_pp_uses_intended_band_and_keeps_explicit_negative_shape_available():
    assert m.resident_prefill_rows({'deepseek_source_resident_prefill':'expert_major_v1'})==12
    assert m.resident_prefill_rows({'deepseek_source_resident_prefill':'expert_major_wide_v1'})==99
    assert m.resident_prefill_rows({'deepseek_source_resident_prefill':'expert_major_v1'},99)==99
    for count in (1,129,True):
        with pytest.raises(ValueError):m.resident_prefill_rows({},count)


def test_explicit_inactive_dspark_defaults_admit_without_oracle():
    for action in ('dspark_native_batched','dspark_ordinary_m16'):
        policy=m.dspark_action_policy(action)
        policy.update(deepseek_source_dspark_prepare='off',deepseek_source_dspark_ast='off')
        m.validate_policy(action,policy)
        from mlx2.adapters.deepseek_v41_source import source_policy
        source_policy(policy)
        for change in ({'deepseek_source_dspark_prepare':'oracle_v1'},
            {'deepseek_source_dspark_ast':'tile3x5_v1'}, {'deepseek_source_dspark_target_forward':'reference'}):
            with pytest.raises(ValueError):m.validate_policy(action,{**policy,**change})


def test_pre_model_policy_failure_gets_durable_method_receipt(tmp_path):
    import json
    unit=tmp_path/'unit';unit.mkdir()
    (unit/'spec.json').write_text(json.dumps({'checkpoint':'synthetic@'+'a'*64,'source_revision':'b'*40,'source_snapshot_sha256':'c'*64}))
    error=ValueError('synthetic rejected active policy')
    m.persist_unhandled_failure(['worker','/source',str(unit),'/model','dspark_native_batched'],error)
    record=json.loads((unit/'results.json').read_text())
    assert record['classification']=='method_failure_before_model_load'
    assert record['model_loaded']is False and record['numerical_test_executed']is False
    assert record['passed']is False and record['error']['message']==str(error)
    original=(unit/'results.json').read_bytes()
    m.persist_unhandled_failure(['worker','/source',str(unit),'/model'],RuntimeError('later'))
    assert (unit/'results.json').read_bytes()==original
