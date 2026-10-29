"""Opt-in Source native adapter contracts; CPU arrays and mocked dispatch only."""
from dataclasses import fields
from types import SimpleNamespace
import ast
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_source_native_mx as policy
from mlx2.adapters.deepseek_v41_source import source_policy, DeepSeekV41SourceAdapter
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights, expert_names
from mlx2.adapters.deepseek_v41_source_buffers import HostExpert, canonical_layout, materialize
from mlx2.runtime.gguf import source_native_mx_runtime as runtime, source_mxfp4_pair

KEY='deepseek_source_native_mx'


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    def forbidden(*args,**kwargs):raise AssertionError('CPU fixture attempted Metal')
    monkeypatch.setattr(mx.fast,'metal_kernel',forbidden)
    yield
    mx.set_default_device(old)


def args():
    return SimpleNamespace(hidden_size=5120,moe_intermediate_size=2304,n_routed_experts=384,
        num_hidden_layers=40,num_experts_per_tok=6,activation_dtype='bfloat16',simulate_quantization=True)


def options():
    return {KEY:policy.POLICY,'deepseek_source_expert_math':'routed_fused_fp32',
        'deepseek_source_prefill_batch':'gate_up','deepseek_source_experts':{'mode':'original_lfu','budget_bytes':75_000_000_000}}


def packed(kind='original',scale=127):
    layout=canonical_layout(kind);payload=np.zeros(layout.nbytes,np.uint8)
    host=HostExpert(payload,layout)
    if kind=='original':
        for name,value in host.views().items():
            if name.endswith('.scales'):payload[layout.tensor(name).offset:layout.tensor(name).offset+value.nbytes]=scale
    return host,materialize(host)


def test_default_and_selected_policy_are_adapter_only_and_explicit():
    assert KEY not in source_policy()
    assert source_policy({KEY:'reference'})[KEY]=='reference'
    assert source_policy(options())[KEY]==policy.POLICY
    assert KEY in DeepSeekV41SourceAdapter.adapter_owned_default_policy_keys


@pytest.mark.parametrize('bad',[None,True,1,[],{},'native','original_half'])
def test_unknown_policy_refused(bad):
    with pytest.raises(ValueError):source_policy({KEY:bad})


@pytest.mark.parametrize('key,value',[('deepseek_source_expert_math','reference'),
    ('deepseek_source_prefill_batch','reference'),('deepseek_source_prefill_batch','gate_up_down'),
    ('deepseek_source_prefill_wide_math','legacy_tiles'),('deepseek_source_prefill_wide_dtype','bf16_p1'),
    ('deepseek_source_prefill_down','q1_m1_owner_store'),('deepseek_source_ffn_input','shared_routed_qat_reuse')])
def test_unvalidated_combinations_fail_closed(key,value):
    with pytest.raises(ValueError):source_policy({**options(),key:value})


def test_selected_policy_requires_explicit_provider_and_canonical_configuration():
    value=options();value.pop('deepseek_source_experts')
    with pytest.raises(ValueError,match='LFU'):source_policy(value)
    config=args();assert policy.configuration_supported(config)
    config.activation_dtype='float32';assert not policy.configuration_supported(config)
    with pytest.raises(ValueError,match='canonical'):
        SourceV41Weights({'args':config},expert_cache_bytes=75_000_000_000,expert_policy={'mode':'original_lfu'},
            expert_math_policy='routed_fused_fp32',prefill_batch_policy='gate_up',native_mx_policy=policy.POLICY)


def artifact():
    return dict(args=args(),identity={'fingerprint':'a'*64},
        tensors={name:{'file':'fixture-shard'}for e in (0,1)for name in expert_names(0,e)},
        seals={'fixture-shard':{'inode':1,'size':2}})


def owner(data=None):
    stream=SimpleNamespace(_active=True,_leased={(0,0):data}if data is not None else {},_cache={},
        packed_buffers=True,kind=lambda key:'original')
    w=SourceV41Weights.__new__(SourceV41Weights)
    w.artifact=artifact();w.expert_stream=stream;w._source_native_mx_attestations=None
    w.source_native_mx_policy=policy.POLICY;w.expert_math_policy='routed_fused_fp32'
    w.source_prefill_batch_policy='gate_up';w.source_prefill_wide_math=w.source_prefill_wide_dtype='reference'
    w.source_prefill_down_policy=w.source_ffn_input_policy='reference';w.scheduled_expert_calls={}
    return w


def test_loader_attestation_is_once_and_metadata_only_replaced_on_reload():
    host,data=packed();w=owner(data)
    first=w.enable_source_native_mx_attestation()
    registry=w._source_native_mx_attestations
    assert first['loads']==0
    w.expert_stream._packed_original_loaded((0,0),host,data)
    bound=w._source_native_mx_bound('blk.0',0,data)
    assert bound.attestation.eligible
    assert all(not isinstance(getattr(bound,f.name),mx.array)for f in fields(bound))
    assert registry.receipt()['loads']==1 and registry.receipt()['retained_gpu_arrays']==0
    assert w.enable_source_native_mx_attestation()['loads']==1
    assert w._source_native_mx_bound('blk.0',0,data)is bound
    replacement=materialize(host)
    w.expert_stream._leased[(0,0)]=replacement
    assert w._source_native_mx_bound('blk.0',0,replacement)is None
    w.expert_stream._packed_original_loaded((0,0),host,replacement)
    assert w._source_native_mx_bound('blk.0',0,replacement).expert_object_id==id(replacement)
    assert registry.receipt()['loads']==2 and registry.receipt()['metadata_entries']==1


@pytest.mark.parametrize('bad',['cached','foreign_observer','changed_observer','unpacked'])
def test_enable_never_backfills_or_reads_gpu_cached_weights(bad):
    w=owner()
    if bad=='cached':w.expert_stream._cache[(0,0)]=object()
    elif bad=='foreign_observer':w.expert_stream._packed_original_loaded=lambda *a:None
    elif bad=='changed_observer':
        w.enable_source_native_mx_attestation();w.expert_stream._packed_original_loaded=lambda *a:None
    else:w.expert_stream.packed_buffers=False
    with pytest.raises(ValueError):w.enable_source_native_mx_attestation()


def test_missing_ineligible_and_outside_lease_remain_distinct():
    host,data=packed(scale=0);w=owner(data);w.enable_source_native_mx_attestation()
    assert w._source_native_mx_bound('blk.0',0,data)is None
    w.expert_stream._packed_original_loaded((0,0),host,data)
    assert w._source_native_mx_bound('blk.0',0,data).attestation.eligible is False
    w.expert_stream._active=False
    with pytest.raises(ValueError,match='lease'):w._source_native_mx_bound('blk.0',0,data)


@pytest.mark.parametrize('prefix,expert',[('layers.0.ffn',0),('blk.40',0),('blk.00',0),('blk.0',384),('blk.0',True)])
def test_only_actual_source_logical_keys_are_admitted(prefix,expert):
    with pytest.raises(ValueError):policy._key(prefix,expert)


def test_frame_local_gu_cast_shared_down_separate_no_runtime_eval(monkeypatch):
    host,data=packed();w=owner(data);w.enable_source_native_mx_attestation()
    w.expert_stream._packed_original_loaded((0,0),host,data)
    x=mx.zeros((34,5120),mx.float32);down=mx.zeros((34,2304),mx.float32)
    monkeypatch.setattr(mx,'default_device',lambda:mx.gpu)
    monkeypatch.setattr(source_mxfp4_pair,'_device_supported',lambda:True)
    casts=[];calls=[];gate,up,raw=object(),object(),object()
    def prepare(value):token=object();casts.append((value,token));return token
    def project(half,expert,bound,*,projection):
        calls.append((half,expert,bound,projection));return {'gate':gate,'up':up,'down':raw}[projection]
    monkeypatch.setattr(runtime,'prepare_half',prepare);monkeypatch.setattr(runtime,'project_half',project)
    monkeypatch.setattr(mx,'eval',lambda *a:pytest.fail('runtime scalar/fence'))
    assert policy.maybe_gate_up(w,x,data,'blk.0',0,prefill_rows=99)==(gate,up)
    assert policy.maybe_down(w,down,data,'blk.0',0,prefill_rows=99)is raw
    assert len(casts)==2 and calls[0][0]is calls[1][0]and calls[2][0]is not calls[0][0]
    assert [c[-1]for c in calls]==['gate','up','down']
    assert w.scheduled_expert_calls['official_source_native_mx_gate_up_jobs']==1
    assert w.scheduled_expert_calls['official_source_native_mx_down']==1


@pytest.mark.parametrize('change',['reference','q1','small','too_wide','decode','global_rows','missing','ineligible','device'])
def test_guarded_fallbacks_preserve_ordinary_math_and_reason_counters(monkeypatch,change):
    host,data=packed('q1'if change=='q1'else'original',scale=0 if change=='ineligible'else 127)
    w=owner(data);w.enable_source_native_mx_attestation()
    if change not in ('q1','missing'):w.expert_stream._packed_original_loaded((0,0),host,data)
    if change=='reference':w.source_native_mx_policy='reference'
    rows=1 if change=='small'else 100 if change=='too_wide'else 34
    value=mx.zeros((rows,2304),mx.float32)
    monkeypatch.setattr(mx,'default_device',lambda:mx.gpu)
    monkeypatch.setattr(source_mxfp4_pair,'_device_supported',lambda:change!='device')
    assert policy.maybe_down(w,value,data,'blk.0',0,prefill_rows=1 if change=='decode'else 100 if change=='global_rows'else 99)is None
    assert not any(k.endswith('_down')for k in w.scheduled_expert_calls)
    if change=='reference':assert not w.scheduled_expert_calls
    else:assert sum(v for k,v in w.scheduled_expert_calls.items()if 'fallback' in k)==1


def test_selected_profile_identity_changes_and_reference_identity_stays(monkeypatch):
    monkeypatch.setattr(policy,'implementation_identity',lambda:'implementation-fixture')
    default=SimpleNamespace(kernel_policy={},numerics_revision='base',profile='profile',numeric_profile='numeric')
    DeepSeekV41SourceAdapter._bind_native_mx_identity(default)
    assert(default.numerics_revision,default.profile,default.numeric_profile)==('base','profile','numeric')
    native=SimpleNamespace(kernel_policy={KEY:policy.POLICY},numerics_revision='base',profile='profile',numeric_profile='numeric')
    DeepSeekV41SourceAdapter._bind_native_mx_identity(native)
    assert 'implementation-fixture'in native.numerics_revision and native.numeric_profile.endswith(policy.POLICY)
    assert native.profile!=default.profile


def test_production_helper_has_no_scalar_gpu_checks_or_global_tensor_stash():
    tree=ast.parse(Path(policy.__file__).read_text())
    calls=[node for node in ast.walk(tree)if isinstance(node,ast.Call)and isinstance(node.func,ast.Attribute)]
    assert not any(node.func.attr in('eval','item','synchronize')for node in calls)
    assert policy.MAX_ATTESTATIONS==15360


@pytest.mark.parametrize('cleanup_fails',[False,True])
def test_constructor_attestation_failure_retires_both_readers_preserves_original(tmp_path,monkeypatch,cleanup_fails):
    from mlx2.adapters import deepseek_v41_source_weights as weights,deepseek_v41_source_lfu as lfu
    events=[]
    class Pool:
        def __init__(self,*a,**kw):pass
        def close(self):events.append('dense_close')
    def close():
        events.append('expert_close')
        if cleanup_fails:raise RuntimeError('cleanup sentinel')
    stream=SimpleNamespace(packed_buffers=True,_cache={(0,0):object()},kind=lambda key:'original',close=close)
    monkeypatch.setattr(weights,'ReadPool',Pool)
    monkeypatch.setattr(weights,'weight_plan',lambda *a:{})
    monkeypatch.setattr(lfu,'LFUSourceExpertStore',lambda *a,**kw:stream)
    monkeypatch.setattr(policy,'implementation_identity',lambda:'fixture-identity')
    art={'args':args(),'root':tmp_path,'aliases':{},'tensors':{}}
    with pytest.raises(ValueError,match='before original experts are cached') as error:
        SourceV41Weights(art,expert_cache_bytes=75_000_000_000,expert_policy={'mode':'original_lfu'},
            expert_math_policy='routed_fused_fp32',prefill_batch_policy='gate_up',native_mx_policy=policy.POLICY)
    assert events==['expert_close','dense_close']
    if cleanup_fails:assert any('cleanup'in note for note in error.value.__notes__)
