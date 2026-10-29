"""Cold target identity preparation; no checkpoint/head/GPU is loaded."""
import hashlib
import json
from types import SimpleNamespace
import mlx.core as mx
import pytest
from mlx2.adapters.deepseek_v41_source import source_policy, DeepSeekV41SourceAdapter


def policy():
    return dict(deepseek_source_dspark_prepare='oracle_v1',
        deepseek_source_dspark_target_forward='fully_batched_m1_v1',
        deepseek_source_dspark_target_expert_law='source_ast_m1_rows_v1',
        deepseek_source_expert_math='routed_fused_fp32')


def identity(value,monkeypatch):
    import mlx2.adapters.deepseek_v41_ast_target as target
    monkeypatch.setattr(target,'source_revision_token',lambda law,forward:'proof-bound-'+law+'-'+forward)
    owner=SimpleNamespace(kernel_policy=source_policy(value),numerics_revision='base',profile='base',numeric_profile='base')
    DeepSeekV41SourceAdapter._bind_batched_tree_identity(owner)
    target_policy={k:v for k,v in owner.kernel_policy.items()if not k.startswith('deepseek_source_dspark_')}
    binding=owner.numerics_revision+':'+hashlib.sha256(json.dumps(target_policy,sort_keys=True).encode()).hexdigest()
    return owner,binding


@pytest.mark.parametrize('law',['source_ast_m1_rows_v1','source_ast_resident_rows_v1'])
def test_prepare_and_cold_serving_bind_identical_target_before_cache(tmp_path,monkeypatch,law):
    prep=policy();prep['deepseek_source_dspark_target_expert_law']=law;oracle=tmp_path/'oracle.json';oracle.write_text('{}')
    serving={k:v for k,v in prep.items()if k!='deepseek_source_dspark_prepare'}
    serving.update(deepseek_source_dspark_ast='tile3x5_v1',deepseek_source_dspark_oracle=str(oracle))
    before,left=identity(prep,monkeypatch);after,right=identity(serving,monkeypatch)
    assert left==right and before.profile==after.profile and before.numeric_profile==after.numeric_profile
    assert before._source_ast_target_laws==(law,'fully_batched_m1_v1')
    assert source_policy(prep).get('deepseek_source_dspark_ast','off')=='off'
    assert 'deepseek_source_dspark_oracle'not in source_policy(prep)


@pytest.mark.parametrize('change',[
    {'deepseek_source_dspark_ast':'tile3x5_v1'},
    {'deepseek_source_dspark_oracle':'unused'},
    {'deepseek_source_dspark_target_forward':'reference'},
    {'deepseek_source_expert_math':'reference'},
    {'deepseek_source_resident_gather':'resident_bank_v1'},
    {'draft_model':'foreign'},
    {'deepseek_source_dspark_cache_bytes':True},
])
def test_prepare_refuses_route_or_identity_ambiguity(change):
    with pytest.raises(ValueError):source_policy({**policy(),**change})


def test_batched_fields_still_require_explicit_preparation_or_serving():
    value=policy();value.pop('deepseek_source_dspark_prepare')
    with pytest.raises(ValueError,match='require tile3x5'):source_policy(value)


def test_serving_still_requires_actual_oracle_file():
    value=policy();value.pop('deepseek_source_dspark_prepare');value['deepseek_source_dspark_ast']='tile3x5_v1'
    with pytest.raises(ValueError,match='existing exact16-row oracle'):source_policy(value)


def test_descendant_preparation_is_an_explicit_target_identity(monkeypatch):
    _,reference=identity(policy(),monkeypatch)
    _,shared=identity({**policy(),'deepseek_source_ast_kv_preparation':'descendant_reuse_v1'},monkeypatch)
    assert reference!=shared
    with pytest.raises(ValueError,match='requires prepared or serving'):
        source_policy({'deepseek_source_ast_kv_preparation':'descendant_reuse_v1'})


def test_direct_binding_cannot_rewrite_loaded_target_law(monkeypatch):
    from mlx2.adapters.deepseek_v41_dspark_ast import SourceDSparkASTBinding
    import mlx2.adapters.deepseek_v41_dspark as dspark
    monkeypatch.setattr(dspark,'load_source_dspark',lambda *a,**k:(_ for _ in ()).throw(AssertionError('head load forbidden')))
    adapter=SimpleNamespace(model=SimpleNamespace(_mlx2_source_ast_target_laws=('ordinary_m16_v1','reference')))
    with pytest.raises(ValueError,match='bound at load'):
        SourceDSparkASTBinding(adapter,check_live=lambda:True,
            expert_law='source_ast_m1_rows_v1',forward_law='fully_batched_m1_v1')


def test_resident_preparation_identity_differs_from_m1_and_still_requires_ordinary_load(monkeypatch):
    _,m1=identity(policy(),monkeypatch)
    resident={**policy(),'deepseek_source_dspark_target_expert_law':'source_ast_resident_rows_v1'}
    _,bound=identity(resident,monkeypatch)
    assert bound!=m1
    assert source_policy(resident).get('deepseek_source_resident_gather','reference')=='reference'
    with pytest.raises(ValueError,match='without foreign draft or resident bank'):
        source_policy({**resident,'deepseek_source_resident_gather':'resident_bank_v1'})


def test_hybrid_requires_explicit_canonical_profile_and_binds_distinct_law(monkeypatch):
    hybrid={**policy(),'deepseek_source_dspark_target_expert_law':'source_ast_hybrid_rows_v1'}
    with pytest.raises(ValueError,match='requires explicit canonical_m1_v1'):
        source_policy(hybrid)
    hybrid['deepseek_source_prefill_partition']='canonical_m1_v1'
    owner,binding=identity(hybrid,monkeypatch)
    assert owner._source_ast_target_laws==('source_ast_hybrid_rows_v1','fully_batched_m1_v1')
    assert owner.kernel_policy['deepseek_source_prefill_partition']=='canonical_m1_v1'
    assert source_policy(policy()).get('deepseek_source_prefill_partition','adaptive_v1')=='adaptive_v1'
