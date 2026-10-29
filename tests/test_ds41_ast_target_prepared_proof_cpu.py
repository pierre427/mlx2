"""The numerical proof never acquires GPU ownership or runs on CPU."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import mlx.core as mx
import pytest


@pytest.fixture
def helper(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:(_ for _ in ()).throw(AssertionError('GPU forbidden')))
    path=Path(__file__).parents[1]/'scripts/research/ds41_ast_target_components.py'
    spec=importlib.util.spec_from_file_location('ds41_prepared_proof_cpu',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_prepared_numerical_proof_requires_owned_gpu_before_model_access(helper):
    with pytest.raises(ValueError,match='owned GPU backend'):
        helper.run(SimpleNamespace(),[],lambda:True,expert_law='source_ast_m1_rows_v1',forward_law='fully_batched_m1_v1')


def test_prepared_numerical_proof_cannot_combine_old_localization(helper):
    with pytest.raises(ValueError,match='old-law localization'):
        helper.run(SimpleNamespace(),[],lambda:True,expert_law='source_ast_m1_rows_v1',
            forward_law='fully_batched_m1_v1',diagnose_moe_layer=14)


def test_proof_snapshot_preserves_capacity_that_visible_state_roundtrip_trims(helper):
    from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache
    row=DeepSeekV41LayerCache(2,xp=mx,identity='synthetic-proof-capacity@123',source=True,
        head_dim=8,index_head_dim=4)
    row.offset=2;row.keys=mx.zeros((1,1,2,8),mx.float32)
    row.append_pools(mx.zeros((1,1,8),mx.float32),mx.zeros((1,1,4),mx.float32));row.validate_state()
    compact=DeepSeekV41LayerCache.from_state(row.state,row.meta_state,xp=mx)
    snapshot=helper._proof_snapshot([row])[0]
    try:
        assert row._capacity==256 and compact._capacity==1
        assert snapshot._capacity==row._capacity and snapshot._pool_count==row._pool_count
        assert snapshot.meta_state==row.meta_state
        assert snapshot._pooled.shape==row._pooled.shape and snapshot._index_keys.shape==row._index_keys.shape
        assert bool(mx.array_equal(snapshot.pooled,compact.pooled))
    finally:row.close();compact.close();snapshot.close()


def test_pre_admission_diagnostic_reports_capacity_without_promoting_payload_failure(helper,tmp_path):
    good=dict(raw_bits_equal=True,finite=True,maximum_absolute_error=0.,shape=[1,1,8])
    controls=[dict(node=1,logits=good,features=good,state=[dict(topology_equal=True,private_schema_expected=True,
        capacity_equal=False,candidate_capacity=256,ordinary_capacity=1,planes=[good])])]
    path=tmp_path/'private-gate.json'
    helper._private_proof_diagnostic(path,SimpleNamespace(identity='synthetic-model'),
        SimpleNamespace(transaction=SimpleNamespace(provider_revision_token='source',profile='profile'),receipt={'forward_law':'fully_batched_m1_v1'}),controls)
    record=json.loads(path.read_text());failed=record['first_failed_node']['first_failed_check']
    assert record['passed']is False and record['proof_local_admission_granted']is False
    assert record['first_failed_node']['logits_raw_equal']is True
    assert failed['path']==['state',0,'capacity_equal']
    assert failed['comparison']['candidate_capacity']==256 and failed['comparison']['ordinary_capacity']==1
    assert failed['comparison']['planes'][0]['raw_bits_equal']is True
    with pytest.raises(FileExistsError):
        helper._private_proof_diagnostic(path,None,None,controls)


@pytest.mark.parametrize('law',['source_ast_m1_rows_v1','source_ast_resident_rows_v1'])
def test_prepared_proof_law_and_callback_are_bound_before_any_control(helper,law):
    weights=SimpleNamespace(source_resident_gather_policy='reference',source_resident_bank_session=None)
    model=SimpleNamespace(weights=weights,_mlx2_source_ast_target_laws=(law,'fully_batched_m1_v1'))
    resident=law=='source_ast_resident_rows_v1'
    callback=(lambda *args:None)if resident else None
    assert helper._validate_batched_proof_entry(model,law,'fully_batched_m1_v1',callback)is resident
    with pytest.raises(ValueError,match='bound by oracle preparation'):
        helper._validate_batched_proof_entry(model,'ordinary_m16_v1','fully_batched_m1_v1',callback)
    if resident:
        with pytest.raises(ValueError,match='closed-cohort preparation'):
            helper._validate_batched_proof_entry(model,law,'fully_batched_m1_v1',None)
    else:
        with pytest.raises(ValueError,match='cannot install'):
            helper._validate_batched_proof_entry(model,law,'fully_batched_m1_v1',lambda *args:None)
    weights.source_resident_bank_session=object()
    with pytest.raises(ValueError,match='before controls'):
        helper._validate_batched_proof_entry(model,law,'fully_batched_m1_v1',callback)


def test_resident_proof_owner_requires_exact_idle_published_bank(helper):
    calls=[]
    bank=SimpleNamespace(assert_current=lambda:calls.append('current'),receipt=lambda:{'published':True,'active_leases':0})
    session=SimpleNamespace(bank=bank,active=False,replaying=False,poisoned=False)
    weights=SimpleNamespace(source_resident_gather_policy='resident_bank_v1',source_resident_bank_session=session,
        expert_stream=SimpleNamespace(_resident_bank=bank))
    model=SimpleNamespace(weights=weights)
    assert helper._validate_resident_proof_owner(model)['published']is True
    assert calls==['current']
    session.active=True
    with pytest.raises(ValueError,match='idle owned bank'):helper._validate_resident_proof_owner(model)
    session.active=False;weights.expert_stream._resident_bank=object()
    with pytest.raises(ValueError,match='bank/store ownership'):helper._validate_resident_proof_owner(model)


def test_hybrid_proof_requires_two_fresh_owned_arm_contexts_before_gpu(helper):
    with pytest.raises(ValueError,match='fresh full-cohort and genuine missing-cohort'):
        helper.run(SimpleNamespace(),[],lambda:True,expert_law='source_ast_hybrid_rows_v1',
            forward_law='fully_batched_m1_v1',resident_prepare=lambda *args:None)
