"""Owned-probe refusal tests; never synthesize a passing actual oracle."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import mlx.core as mx
import pytest


@pytest.fixture
def helper():
    path=Path(__file__).parents[1]/'scripts/research/ds41_dspark_ast_serving_components.py'
    spec=importlib.util.spec_from_file_location('ds41_serving_probe_cpu',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.mark.parametrize('mutation',['failed','foreign_law','foreign_revision','missing_rawexact','foreign_model'])
def test_actual_proof_refuses_missing_success_identity_before_oracle(helper,tmp_path,mutation):
    from mlx2.adapters.deepseek_v41_ast_target import PARENTS
    receipt=dict(expert_law=helper.LAW,source_revision_token='source',checkpoint='model',parents=list(PARENTS),
        tensor_tile_rows=16,grouped_moe_calls=40,serial_whole_model_calls=0,physical_target_invocations=1)
    record=dict(schema='mlx2.ds41-dspark-ast-owned-functional.v1',stage='completed',passed=True,raw_bits_exact=True,
        numerical_classification='raw_bits_exact',source_revision='revision',source_snapshot_sha256='snapshot',
        checkpoint='checkpoint@fingerprint',target_receipt=receipt)
    if mutation=='failed':record['stage']='failed'
    if mutation=='foreign_law':receipt['expert_law']='ordinary_m16_v1'
    if mutation=='foreign_revision':receipt['source_revision_token']='foreign'
    if mutation=='missing_rawexact':record.pop('raw_bits_exact')
    if mutation=='foreign_model':receipt['checkpoint']='foreign'
    (tmp_path/'dspark-ast-functional.json').write_text(json.dumps(record))
    (tmp_path/'dspark-target-comparison.json').write_text(json.dumps(dict(receipt=receipt)))
    (tmp_path/'spec.json').write_text(json.dumps(dict(source_revision='revision',source_snapshot_sha256='snapshot',checkpoint='checkpoint@fingerprint')))
    with pytest.raises(ValueError,match='identity/law/completion'):
        helper.validate_proof(tmp_path,SimpleNamespace(identity='model'),SimpleNamespace(source_revision_token='source'))
    assert not(tmp_path/'ast-serving-oracle.json').exists()


def test_actual_serving_requires_gpu_guard_before_load(helper,tmp_path,monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:(_ for _ in ()).throw(AssertionError('GPU forbidden')))
    try:
        with pytest.raises(ValueError,match='caller-owned GPU'):
            helper.run(SimpleNamespace(),tmp_path,lambda:True,proof_unit=tmp_path)
    finally:mx.set_default_device(old)
    assert not(tmp_path/'ast-serving-oracle.json').exists()


def test_serving_probe_never_overwrites_evidence(helper,tmp_path):
    (tmp_path/'ast-serving-oracle.json').write_text('existing')
    with pytest.raises(ValueError,match='overwrite'):
        helper.run(SimpleNamespace(),tmp_path,lambda:True,proof_unit=tmp_path)
    assert(tmp_path/'ast-serving-oracle.json').read_text()=='existing'


def test_prefreeze_policy_detects_identity_drift_without_gpu(helper,tmp_path,monkeypatch):
    from mlx2.adapters.deepseek_v41_source import source_policy
    import mlx2.adapters.deepseek_v41_ast_target as target
    monkeypatch.setattr(target,'implementation_identity',lambda:())
    (tmp_path/'ordinary-reference.json').write_text(json.dumps({'diagnostics':{'execution_policy':source_policy({})}}))
    assert helper.validate_prefreeze_policy(tmp_path,{})['policy_exact']is True
    with pytest.raises(ValueError,match='deepseek_qat'):
        helper.validate_prefreeze_policy(tmp_path,{'deepseek_qat':'fused_fp32'})
    assert not(tmp_path/'ast-serving-oracle.json').exists()


def test_prefreeze_does_not_strip_batched_target_law_differences(helper,tmp_path,monkeypatch):
    from mlx2.adapters.deepseek_v41_source import source_policy
    import mlx2.adapters.deepseek_v41_ast_target as target
    monkeypatch.setattr(target,'implementation_identity',lambda:())
    prepared=dict(deepseek_source_dspark_prepare='oracle_v1',
        deepseek_source_dspark_target_forward='fully_batched_m1_v1',
        deepseek_source_dspark_target_expert_law='source_ast_m1_rows_v1',
        deepseek_source_expert_math='routed_fused_fp32')
    (tmp_path/'ordinary-reference.json').write_text(json.dumps({'diagnostics':{'execution_policy':source_policy(prepared)}}))
    assert helper.validate_prefreeze_policy(tmp_path,prepared)['policy_exact']is True
    with pytest.raises(ValueError,match='load-bound target law differs'):
        helper.validate_prefreeze_policy(tmp_path,{'deepseek_source_expert_math':'routed_fused_fp32'})


def test_actual_batched_proof_requires_all_publication_operation_gates(helper,tmp_path):
    from mlx2.adapters.deepseek_v41_ast_target import PARENTS
    receipt=dict(expert_law=helper.LAW,forward_law='fully_batched_m1_v1',source_revision_token='source',
        checkpoint='model',parents=list(PARENTS),tensor_tile_rows=16,grouped_moe_calls=40,
        serial_whole_model_calls=0,physical_target_invocations=1)
    record=dict(schema='mlx2.ds41-dspark-ast-owned-functional.v1',stage='completed',passed=True,
        raw_bits_exact=True,numerical_classification='raw_bits_exact',source_revision='revision',
        source_snapshot_sha256='snapshot',checkpoint='checkpoint@fingerprint',target_receipt=receipt)
    (tmp_path/'dspark-ast-functional.json').write_text(json.dumps(record))
    (tmp_path/'dspark-target-comparison.json').write_text(json.dumps(dict(receipt=receipt)))
    (tmp_path/'spec.json').write_text(json.dumps(dict(source_revision='revision',source_snapshot_sha256='snapshot',checkpoint='checkpoint@fingerprint')))
    with pytest.raises(ValueError,match='prepared/commit/abort/isolation'):
        helper.validate_proof(tmp_path,SimpleNamespace(identity='model'),
            SimpleNamespace(source_revision_token='source',forward_law='fully_batched_m1_v1'))
    assert not(tmp_path/'ast-serving-oracle.json').exists()


@pytest.mark.parametrize('mutation',['foreign_m1_receipt','missing_prebank','host_routes','valid_but_empty_raw'])
def test_resident_proof_never_relabels_m1_or_skips_independent_raw_gates(helper,tmp_path,mutation):
    from mlx2.adapters.deepseek_v41_ast_target import PARENTS
    law=helper.RESIDENT_LAW;forward='fully_batched_m1_v1'
    receipt=dict(expert_law=law,forward_law=forward,source_revision_token='resident-source',checkpoint='resident-model',
        parents=list(PARENTS),tensor_tile_rows=16,grouped_moe_calls=40,serial_whole_model_calls=0,
        physical_target_invocations=1,source_route_host_materializations=0,resident_bank_tile=True,zero_host_expert_ids_claimed=True)
    gates=dict.fromkeys(('private_all16_raw_exact','prepared_all16_raw_exact','committed_state_exact',
        'continuation_exact','isolated','abort_closed','commit_after_abort_refused'),True)
    comparison=dict(receipt=receipt,batched_exact_gates=gates,proof_local_admission=True,APCv2_insertions=0,
        resident_preparation=dict(ordinary_controls_completed_before_bank=True,bank=dict(published=True,active_leases=0)),
        continuation_reference_scope='prebank_ordinary_branch1_depth4_node9')
    if mutation=='foreign_m1_receipt':receipt['expert_law']=helper.LAW
    if mutation=='missing_prebank':comparison.pop('resident_preparation')
    if mutation=='host_routes':receipt['source_route_host_materializations']=40
    record=dict(schema='mlx2.ds41-dspark-ast-owned-functional.v1',stage='completed',passed=True,raw_bits_exact=True,
        numerical_classification='raw_bits_exact',source_revision='base',source_snapshot_sha256='snapshot',
        checkpoint='checkpoint@fingerprint',target_receipt=receipt)
    (tmp_path/'dspark-ast-functional.json').write_text(json.dumps(record))
    (tmp_path/'dspark-target-comparison.json').write_text(json.dumps(comparison))
    (tmp_path/'spec.json').write_text(json.dumps(dict(source_revision='base',source_snapshot_sha256='snapshot',checkpoint='checkpoint@fingerprint')))
    pattern=('identity/law/completion'if mutation=='foreign_m1_receipt'else'every node/state'if mutation=='valid_but_empty_raw'else'independent prebank')
    with pytest.raises(ValueError,match=pattern):
        helper.validate_proof(tmp_path,SimpleNamespace(identity='resident-model'),
            SimpleNamespace(expert_law=law,forward_law=forward,source_revision_token='resident-source'))
    assert not(tmp_path/'ast-serving-oracle.json').exists()


def test_hybrid_proof_refuses_resident_only_numerical_admission(helper,tmp_path):
    from mlx2.adapters.deepseek_v41_ast_target import PARENTS
    receipt=dict(expert_law=helper.HYBRID_LAW,forward_law='fully_batched_m1_v1',source_revision_token='hybrid-source',
        checkpoint='model',parents=list(PARENTS),tensor_tile_rows=16,grouped_moe_calls=40,
        physical_target_invocations=1,serial_whole_model_calls=0,source_route_host_materializations=0,
        resident_bank_tile=True,zero_host_expert_ids_claimed=True,hybrid_bank={'actual_expert_route':'resident'})
    gates=dict.fromkeys(('private_all16_raw_exact','prepared_all16_raw_exact','committed_state_exact',
        'continuation_exact','isolated','abort_closed','commit_after_abort_refused'),True)
    comparison=dict(receipt=receipt,batched_exact_gates=gates,proof_local_admission=True,APCv2_insertions=0,
        resident_preparation=dict(ordinary_controls_completed_before_bank=True,bank=dict(published=True,active_leases=0)),
        continuation_reference_scope='prebank_ordinary_branch1_depth4_node9')
    record=dict(schema='mlx2.ds41-dspark-ast-owned-functional.v1',stage='completed',passed=True,
        raw_bits_exact=True,numerical_classification='raw_bits_exact',source_revision='base',
        source_snapshot_sha256='snapshot',checkpoint='checkpoint@fingerprint',target_receipt=receipt)
    (tmp_path/'dspark-ast-functional.json').write_text(json.dumps(record))
    (tmp_path/'dspark-target-comparison.json').write_text(json.dumps(comparison))
    (tmp_path/'spec.json').write_text(json.dumps(dict(source_revision='base',source_snapshot_sha256='snapshot',checkpoint='checkpoint@fingerprint')))
    with pytest.raises(ValueError,match='fresh resident and genuine-miss'):
        helper.validate_proof(tmp_path,SimpleNamespace(identity='model'),
            SimpleNamespace(expert_law=helper.HYBRID_LAW,forward_law='fully_batched_m1_v1',source_revision_token='hybrid-source'))
    assert not(tmp_path/'ast-serving-oracle.json').exists()
