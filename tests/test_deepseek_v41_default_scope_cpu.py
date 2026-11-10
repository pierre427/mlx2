"""Artifact/profile default scope and generic registry guard ownership."""
import hashlib
import json
from pathlib import Path

import pytest

from mlx2.adapters import deepseek_v41_source_defaults as defaults
from mlx2.adapters.deepseek_v41_source import DESCRIPTOR
from mlx2.adapters.registry import AdapterResolution


def scoped_policy(tmp_path,monkeypatch):
    hot=tmp_path/'hot.json'
    hot.write_text(json.dumps({'source':defaults.SOURCE_FINGERPRINT,'hot_budget_bytes':defaults.HOT_BYTES}))
    monkeypatch.setattr(defaults,'HOT_MAP_SHA256',hashlib.sha256(hot.read_bytes()).hexdigest())
    q1=tmp_path/'q1';q1.mkdir()
    manifest={'source':defaults.SOURCE_FINGERPRINT,'dense_modified':False,
              'quantization':'source-q1-affine-g64-lloyd4-f16-v1','experts':[]}
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    monkeypatch.setattr(defaults,'Q1_FINGERPRINT',digest)
    (q1/'manifest.json').write_text(json.dumps({**manifest,'fingerprint':digest}))
    return {'deepseek_source_experts':{'mode':'hot_original_q1','hot_bytes':defaults.HOT_BYTES,
            'budget_bytes':32<<30,'q1_root':str(q1),'hotset_path':str(hot)}}


def test_exact_precision_profile_allows_independent_cache_caps(tmp_path,monkeypatch):
    value=scoped_policy(tmp_path,monkeypatch)
    original=json.loads(json.dumps(value))
    for cap in (32<<30,48<<30,75_000_000_000):
        value['deepseek_source_experts']['budget_bytes']=cap
        assert defaults.default_route_policy_applies('ordinary',value)is True
    value['deepseek_source_experts']['budget_bytes']=32<<30
    assert value==original


@pytest.mark.parametrize('route',['native_mtp','external_draft','prompt_lookup'])
def test_other_routes_do_not_read_files(route,monkeypatch):
    monkeypatch.setattr(defaults,'_metadata',lambda *a:pytest.fail('unrelated route performed I/O'))
    assert defaults.default_route_policy_applies(route,{})is False


@pytest.mark.parametrize('policy',[None,{}, {'deepseek_source_experts':{'mode':'original_lfu'}},
    {'deepseek_source_experts':{'mode':'original_lfu','budget_bytes':1}},
    {'deepseek_source_experts':{'mode':'q2'}},
    {'deepseek_source_experts':{'mode':'hot_original_q1','hot_bytes':60_000_000_000,'hotset_path':'relative','q1_root':'relative'}}])
def test_missing_foreign_or_invalid_profile_gets_no_defaults(policy):
    assert defaults.default_route_policy_applies('ordinary',policy)is False


@pytest.mark.parametrize('extra',[{}, {'retention':'cycle_aware_v1'}, {'retention':'decayed_lfu_v1'}])
def test_exact_original_lfu_gets_bundle_at_any_cache_cap(extra,monkeypatch):
    # Measured 2026-11-08 on c30fd25e: identical replies, faster decode.
    monkeypatch.setattr(defaults,'_metadata',lambda *a:pytest.fail('original_lfu needs no metadata reads'))
    for cap in (32<<30,75_000_000_000):
        policy={'deepseek_source_experts':{'mode':'original_lfu','budget_bytes':cap,**extra}}
        assert defaults.default_route_policy_applies('ordinary',policy)is True
        assert defaults.default_route_policy_applies('external_draft',policy)is False
    policy={'deepseek_source_experts':{'mode':'original_lfu','budget_bytes':32<<30},
            'deepseek_source_expert_math':'reference'}
    assert defaults.default_route_policy_applies('ordinary',policy)is False
    policy={'deepseek_source_experts':{'mode':'original_lfu','budget_bytes':32<<30},'draft_model':'embedded_dspark'}
    assert defaults.default_route_policy_applies('ordinary',policy)is False


def test_map_tamper_and_missing_file_decline(tmp_path,monkeypatch):
    value=scoped_policy(tmp_path,monkeypatch);path=Path(value['deepseek_source_experts']['hotset_path'])
    path.write_text(path.read_text()+' ')
    assert defaults.default_route_policy_applies('ordinary',value)is False
    path.unlink()
    assert defaults.default_route_policy_applies('ordinary',value)is False


def test_claimed_q1_fingerprint_does_not_authorize_changed_manifest(tmp_path,monkeypatch):
    value=scoped_policy(tmp_path,monkeypatch);path=Path(value['deepseek_source_experts']['q1_root'])/'manifest.json'
    manifest=json.loads(path.read_text());manifest['experts']=[{'changed':True}]
    path.write_text(json.dumps(manifest))
    assert defaults.default_route_policy_applies('ordinary',value)is False


def test_source_binding_is_independent_of_hot_file_digest(tmp_path,monkeypatch):
    value=scoped_policy(tmp_path,monkeypatch);path=Path(value['deepseek_source_experts']['hotset_path'])
    manifest=json.loads(path.read_text());manifest['source']='f'*64;path.write_text(json.dumps(manifest))
    monkeypatch.setattr(defaults,'HOT_MAP_SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
    assert defaults.default_route_policy_applies('ordinary',value)is False


def test_metadata_symlink_declines(tmp_path,monkeypatch):
    value=scoped_policy(tmp_path,monkeypatch);path=Path(value['deepseek_source_experts']['hotset_path'])
    destination=tmp_path/'real.json';path.rename(destination);path.symlink_to(destination)
    assert defaults.default_route_policy_applies('ordinary',value)is False


def resolution(cls):return AdapterResolution(cls,DESCRIPTOR,{'fingerprint':'fixture'})


class Guarded:
    default_route_execution_policy={'ordinary':{'probe':'reference'}}
    adapter_owned_default_policy_keys=frozenset({'probe'})
    default_route_policy_applies=staticmethod(lambda route,policy:policy.get('bound')is True)


def test_generic_guard_declines_or_returns_fresh_default_copy():
    value=resolution(Guarded)
    assert value.default_execution_policy('ordinary',operator_policy={})=={}
    result=value.default_execution_policy('ordinary',operator_policy={'bound':True})
    assert result=={'probe':'reference'}
    result['probe']='mutated'
    assert Guarded.default_route_execution_policy['ordinary']=={'probe':'reference'}


@pytest.mark.parametrize('result',[None,1,0,'true',{}])
def test_generic_guard_requires_strict_boolean(result):
    class Bad(Guarded):
        default_route_execution_policy=Guarded.default_route_execution_policy
        adapter_owned_default_policy_keys=Guarded.adapter_owned_default_policy_keys
        default_route_policy_applies=staticmethod(lambda route,policy:result)
    with pytest.raises(ValueError,match='return a boolean'):
        resolution(Bad).default_execution_policy('ordinary')


def test_explicit_invalid_guard_does_not_silently_enable_defaults():
    class Bad(Guarded):
        default_route_execution_policy=Guarded.default_route_execution_policy
        adapter_owned_default_policy_keys=Guarded.adapter_owned_default_policy_keys
        default_route_policy_applies=None
    with pytest.raises(ValueError,match='must be callable'):
        resolution(Bad).default_execution_policy('ordinary')


def test_defaults_and_guards_are_not_inherited():
    class Unmeasured(Guarded):pass
    assert resolution(Unmeasured).default_execution_policy('ordinary',operator_policy={'bound':True})=={}
    class OwnDefaults(Guarded):
        default_route_execution_policy={'ordinary':{'host_memory_signals':{'enabled':True}}}
    # The subclass declares its own defaults without inheriting the parent's guard.
    assert resolution(OwnDefaults).default_execution_policy('ordinary')=={'host_memory_signals':{'enabled':True}}


def test_guard_cannot_mutate_operator_policy():
    class Mutating(Guarded):
        default_route_execution_policy=Guarded.default_route_execution_policy
        adapter_owned_default_policy_keys=Guarded.adapter_owned_default_policy_keys
        @staticmethod
        def default_route_policy_applies(route,policy):
            policy['nested']['value']='changed';return True
    operator={'nested':{'value':'original'}}
    assert resolution(Mutating).default_execution_policy('ordinary',operator_policy=operator)=={'probe':'reference'}
    assert operator=={'nested':{'value':'original'}}


@pytest.mark.parametrize(('key','value'),[
    ('deepseek_physical_batch','b2'),('deepseek_physical_batch','b4'),
    ('deepseek_vision','enabled'),('deepseek_ced','dependency_exact'),
    ('deepseek_source_dspark_ast','tile3x5_v1'),('deepseek_source_dspark_prepare','oracle_v1'),
    ('deepseek_source_resident_gather','resident_bank_v1'),('deepseek_source_resident_prefill','expert_major_v1'),
    ('deepseek_source_prefill_partition','canonical_m1_v1'),('draft_model','embedded_dspark'),
    ('draft_model','prompt_lookup'),('ast_prompt_lookup',True),
])
def test_separate_serving_profiles_decline_before_metadata_reads(key,value,monkeypatch):
    monkeypatch.setattr(defaults,'_metadata',lambda *args:pytest.fail('separate profile read metadata'))
    assert defaults.default_route_policy_applies('ordinary',{key:value})is False


@pytest.mark.parametrize('override',[
    {'deepseek_hc':'fused_fp32'},
    {'deepseek_source_dense_qat':'reference'},
    {'deepseek_source_expert_math':'reference'},
    {'deepseek_index_policy':'reference'},
    {'deepseek_source_q_b_prefill':'tile64x32_raw'},
])
def test_incompatible_partial_kernel_overrides_decline_default_bundle(tmp_path,monkeypatch,override):
    policy=scoped_policy(tmp_path,monkeypatch)
    policy.update(override)
    assert defaults.default_route_policy_applies('ordinary',policy)is False


def test_position_grid_partition_keeps_measured_kernel_scope(tmp_path,monkeypatch):
    policy=scoped_policy(tmp_path,monkeypatch)
    policy['deepseek_source_prefill_partition']='position_grid_v1'
    assert defaults.default_route_policy_applies('ordinary',policy)is True
    policy['deepseek_source_prefill_grid_rows']=256
    assert defaults.default_route_policy_applies('ordinary',policy)is True


@pytest.mark.parametrize('override',[
    {'deepseek_pointwise':'reference'},
    {'deepseek_hc':'fused_fp32','deepseek_source_hc_coefficients':'reference'},
    {'deepseek_source_dense_qat':'reference','deepseek_source_qakv':'reference'},
    {'deepseek_source_attention_math':'reference','deepseek_index_policy':'reference'},
    {'deepseek_source_expert_math':'reference','deepseek_source_prefill_batch':'reference'},
])
def test_compatible_explicit_kernel_overrides_keep_profile_scope(tmp_path,monkeypatch,override):
    policy=scoped_policy(tmp_path,monkeypatch);policy.update(override)
    before=json.loads(json.dumps(policy))
    assert defaults.default_route_policy_applies('ordinary',policy)is True
    assert policy==before


def test_actual_class_guard_keeps_host_policy_and_explicit_kernel_values(tmp_path,monkeypatch):
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter
    from mlx2.server import RouteSelection,resolve_execution_policy_defaults
    policy=scoped_policy(tmp_path,monkeypatch)
    policy.update(deepseek_pointwise='reference',host_memory_signals={'enabled':True})
    resolved=resolve_execution_policy_defaults(policy,RouteSelection('ordinary','explicit_flag'),resolution(DeepSeekV41SourceAdapter))
    assert resolved['deepseek_pointwise']=='reference'
    assert resolved['host_memory_signals']=={'enabled':True}
    assert resolved['deepseek_source_expert_math']=='routed_fused_fp32'


@pytest.mark.parametrize('override',[{'deepseek_physical_batch':'b2'},{'deepseek_hc':'fused_fp32'}])
def test_registry_leaves_separate_or_conflicting_operator_profile_unmodified(tmp_path,monkeypatch,override):
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter
    from mlx2.server import RouteSelection,resolve_execution_policy_defaults
    policy=scoped_policy(tmp_path,monkeypatch);policy.update(override)
    resolved=resolve_execution_policy_defaults(policy,RouteSelection('ordinary','explicit_flag'),resolution(DeepSeekV41SourceAdapter))
    assert resolved==policy
    assert 'deepseek_source_prefill_batch'not in resolved
    assert 'deepseek_source_qakv'not in resolved


@pytest.mark.parametrize('extra',[
    {'deepseek_kv_storage':'native_packed'},
    {'deepseek_kv_storage':'native_packed','deepseek_candidate_score':'candidate_only'},
    {'deepseek_kv_storage':'native_packed','deepseek_candidate_score':'candidate_only','deepseek_context_limit':262144},
    {'deepseek_kv_storage':'context_gated_v1','deepseek_context_limit':262144},
    {'deepseek_kv_storage':'context_gated_v1'},
    {'deepseek_candidate_score':'candidate_only'},
])
def test_packed_kv_and_candidate_scoring_compose_with_selected_bundle(tmp_path,monkeypatch,extra):
    # Packed consumers now exist for fused QAT and all-retained broadcast,
    # so these cache-layout profiles receive the selected kernel bundle.
    policy=scoped_policy(tmp_path,monkeypatch)
    policy.update(extra)
    assert defaults.default_route_policy_applies('ordinary',policy)is True


def test_packed_kv_with_unsupported_attention_override_declines_bundle(tmp_path,monkeypatch):
    policy=scoped_policy(tmp_path,monkeypatch)
    policy.update(deepseek_kv_storage='native_packed',deepseek_source_attention_math='source_short_decode_joint_f32_v1')
    assert defaults.default_route_policy_applies('ordinary',policy)is False


@pytest.mark.parametrize('override,dropped',[
    ({'deepseek_source_hc_coefficients':'reference'},'deepseek_source_hc_mega'),
    ({'deepseek_hc':'fused_fp32','deepseek_source_hc_coefficients':'reference'},'deepseek_source_hc_mega'),
    ({'deepseek_source_bf16_loads':'reference'},'deepseek_source_woa_storage'),
])
def test_overridden_prerequisite_drops_only_its_dependent_default(tmp_path,monkeypatch,override,dropped):
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter
    policy={**scoped_policy(tmp_path,monkeypatch),**override}
    selected=DeepSeekV41SourceAdapter.default_route_execution_policy['ordinary']
    result=resolution(DeepSeekV41SourceAdapter).default_execution_policy('ordinary',operator_policy=policy)
    assert dropped not in result
    assert result=={key:value for key,value in selected.items() if key!=dropped}


def test_unrelated_override_keeps_dependent_defaults(tmp_path,monkeypatch):
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter
    policy={**scoped_policy(tmp_path,monkeypatch),'deepseek_source_bf16_loads':'woa_f32'}
    result=resolution(DeepSeekV41SourceAdapter).default_execution_policy('ordinary',operator_policy=policy)
    assert result==DeepSeekV41SourceAdapter.default_route_execution_policy['ordinary']


@pytest.mark.parametrize('narrow',[
    lambda route,policy,operator:{**policy,'extra':'added'},
    lambda route,policy,operator:{'probe':'changed'},
    lambda route,policy,operator:None,
])
def test_default_filter_may_only_drop_keys(narrow):
    class Narrowing(Guarded):
        default_route_execution_policy=Guarded.default_route_execution_policy
        adapter_owned_default_policy_keys=Guarded.adapter_owned_default_policy_keys
        default_route_policy_applies=Guarded.default_route_policy_applies
        default_route_policy_filter=staticmethod(narrow)
    with pytest.raises(ValueError,match='only drop'):
        resolution(Narrowing).default_execution_policy('ordinary',operator_policy={'bound':True})


def test_original_lfu_drops_only_dependent_default_for_overridden_prerequisite():
    policy={'deepseek_source_experts':{'mode':'original_lfu','budget_bytes':32<<30},
            'deepseek_source_hc_coefficients':'reference'}
    assert defaults.default_route_policy_applies('ordinary',policy)is True
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter
    bundle=DeepSeekV41SourceAdapter.default_route_execution_policy['ordinary']
    assert 'deepseek_source_hc_mega' not in defaults.default_route_policy_filter('ordinary',bundle,policy)


def test_m1_rows_requires_explicit_fused_expert_math_at_parse_time():
    from mlx2.adapters.deepseek_v41_source import source_policy
    base={'draft_model':'embedded_dspark','num_draft':3,'deepseek_verify_rows':'lockstep_multirow_v1','deepseek_verify_commit':'local_frontier',
          'deepseek_verify_moe_rows':'m1_rows_v1'}
    with pytest.raises(ValueError,match='routed_fused_fp32'):
        source_policy(base)
    assert source_policy({**base,'deepseek_source_expert_math':'routed_fused_fp32'})['deepseek_verify_moe_rows']=='m1_rows_v1'
