"""Synthetic CPU contracts for Source WoA opt-in; no actual model/GPU claim."""
from types import SimpleNamespace
import mlx.core as mx
import numpy as np
import pytest
from mlx2.adapters import deepseek_v41_source_woa_pp as native
from mlx2.adapters import deepseek_v41_source_weights as source
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
from mlx2.adapters.deepseek_v41_source import source_policy,DeepSeekV41SourceAdapter
from mlx2.adapters.deepseek_v41_mlx import execution_policy
from mlx2.runtime.gguf import source_dense_legacy

KEY='deepseek_source_woa_pp';NAME='blk.2.attn_output_a.weight'

@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('GPU forbidden')))
    yield
    mx.set_default_device(old)


def args():return SimpleNamespace(activation_dtype='bfloat16',head_dim=512,num_attention_heads=64,o_groups=8,o_lora_rank=1024)


def test_policy_default_source_only_and_explicit_valid():
    assert KEY not in source_policy()and KEY in DeepSeekV41SourceAdapter.adapter_owned_default_policy_keys
    assert source_policy({KEY:native.POLICY})[KEY]==native.POLICY
    with pytest.raises(ValueError):execution_policy({KEY:native.POLICY})
    for v in(None,True,1,[],{},'bad'):
        with pytest.raises(ValueError):source_policy({KEY:v})


@pytest.mark.parametrize('field,value',[('activation_dtype','float32'),('head_dim',256),('num_attention_heads',32),('o_groups',4),('o_lora_rank',512)])
def test_producer_configuration_contract(field,value):
    a=args();assert native.configuration_supported(a);setattr(a,field,value);assert not native.configuration_supported(a)


@pytest.mark.parametrize('name',[None,'blk.40.attn_output_a.weight','blk.00.attn_output_a.weight','blk.-1.attn_output_a.weight','blk.2.attn_output_b.weight','other'])
def test_foreign_alias_refuses_without_device_or_counters(monkeypatch,name):
    monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:(_ for _ in ()).throw(AssertionError('device touched')))
    counts={};assert native.maybe_project(name,None,None,counters=counts)is None and counts=={}


@pytest.mark.parametrize('layer',[0,2,39])
def test_structural_geometry_admits_only_canonical_source(layer):
    x=mx.zeros((1,99,8,4096),mx.float32);weight=mx.zeros((8192,4096),mx.bfloat16)
    assert native.eligible(f'blk.{layer}.attn_output_a.weight',x,weight,check_device=False)


@pytest.mark.parametrize('shape,dtype,groups',[
    ((1,1,8,4096),mx.float32,8),((1,100,8,4096),mx.float32,8),((2,99,8,4096),mx.float32,8),
    ((1,99,4,4096),mx.float32,8),((1,99,8,4095),mx.float32,8),((1,99,8,4096),mx.bfloat16,8),
    ((1,99,8,4096),mx.float32,True),((1,99,8,4096),mx.float32,4)])
def test_unsupported_input_declines_before_device(shape,dtype,groups,monkeypatch):
    monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:(_ for _ in ()).throw(AssertionError('device touched')))
    counts={};assert native.maybe_project(NAME,mx.zeros(shape,dtype),mx.zeros((8192,4096),mx.bfloat16),groups=groups,counters=counts)is None
    assert counts=={'source_woa_native_pp_reference_fallback':1}


@pytest.mark.parametrize('shape,dtype',[((8192,4096),mx.float32),((8191,4096),mx.bfloat16),((8192,4095),mx.bfloat16)])
def test_only_original_resident_bf16_coefficients_admitted(shape,dtype):
    assert not native.eligible(NAME,mx.zeros((1,99,8,4096),mx.float32),mx.zeros(shape,dtype),check_device=False)


def test_native_call_uses_same_weight_bf16_input_and_existing_bf16_boundary(monkeypatch):
    monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:True);monkeypatch.setattr(native,'implementation_identity',lambda:'fixture-law')
    x=mx.full((1,99,8,4096),1.015625,mx.float32);weight=mx.zeros((8192,4096),mx.bfloat16);seen=[]
    def einsum(law,a,b):
        seen.append((law,a,b));assert a.dtype==b.dtype==mx.bfloat16
        return mx.full((1,99,8,1024),1.015625,mx.bfloat16)
    monkeypatch.setattr(mx,'einsum',einsum);counts={};out=native.maybe_project(NAME,x,weight,counters=counts)
    assert out.dtype==mx.float32 and out.shape==(1,99,8,1024)
    assert seen[0][0]=='btgi,goi->btgo'and seen[0][1].shape==x.shape and seen[0][2].shape==(8,1024,4096)
    np.testing.assert_array_equal(np.array(out),1.015625)
    assert counts=={'source_woa_native_pp':1}


def test_build_mismatch_and_device_fault_are_not_silent_fallback(monkeypatch):
    x=mx.zeros((1,99,8,4096),mx.float32);weight=mx.zeros((8192,4096),mx.bfloat16)
    monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:True)
    monkeypatch.setattr(native,'implementation_identity',lambda:(_ for _ in ()).throw(RuntimeError('build changed')))
    counts={}
    with pytest.raises(RuntimeError,match='build changed'):native.maybe_project(NAME,x,weight,counters=counts)
    assert counts=={}
    monkeypatch.setattr(native,'implementation_identity',lambda:'fixture')
    monkeypatch.setattr(mx,'einsum',lambda *a:(_ for _ in ()).throw(RuntimeError('Metal device fault')))
    with pytest.raises(RuntimeError,match='Metal device fault'):native.maybe_project(NAME,x,weight,counters=counts)
    assert counts=={}


def test_unsupported_device_keeps_reference_counter(monkeypatch):
    monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:False);counts={}
    assert native.maybe_project(NAME,mx.zeros((1,99,8,4096),mx.float32),mx.zeros((8192,4096),mx.bfloat16),counters=counts)is None
    assert counts=={'source_woa_native_pp_reference_fallback':1}


@pytest.mark.parametrize('bad',[True,None,'bad'])
def test_constructor_invalid_option_refuses_before_artifact(bad):
    with pytest.raises(ValueError,match='WoA PP'):SourceV41Weights({},expert_cache_bytes=1,woa_pp_policy=bad)


def test_constructor_requires_original_bf16_producer_before_artifact():
    with pytest.raises(ValueError,match='BF16 producer'):SourceV41Weights({},expert_cache_bytes=1,woa_pp_policy=native.POLICY)


@pytest.mark.parametrize('policy',['reference',native.POLICY])
def test_constructor_binding_reference_default_and_optin_model_geometry(monkeypatch,tmp_path,policy):
    monkeypatch.setattr(source,'weight_plan',lambda *a:{})
    monkeypatch.setattr(source,'ReadPool',lambda *a,**kw:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(source,'SourceExpertStore',lambda *a,**kw:SimpleNamespace())
    monkeypatch.setattr(native,'implementation_identity',lambda:'fixture-binding')
    w=SourceV41Weights(dict(root=tmp_path,aliases={},tensors={},args=args()),expert_cache_bytes=1,woa_pp_policy=policy)
    assert w.source_woa_pp_policy==policy
    if policy=='reference':assert w.compute_policy==source.REVISION
    else:assert ':source_woa_pp='+native.POLICY+':fixture-binding'in w.compute_policy
    w._read_pool.close()


def test_adapter_profile_numerics_and_dense_receipt_bind_implementation(monkeypatch):
    monkeypatch.setattr(native,'implementation_identity',lambda:'fixture-binding')
    a=object.__new__(DeepSeekV41SourceAdapter);a.kernel_policy={};a.numerics_revision='base';a.profile='base-profile';a._bind_woa_pp_identity()
    assert a.numerics_revision=='base'and a.profile=='base-profile'
    a.kernel_policy={KEY:native.POLICY};a._bind_woa_pp_identity();assert ':source_woa_pp='+native.POLICY+':implementation=fixture-binding'in a.numerics_revision
    assert ':woapp-'in a.profile
    from mlx2.adapters.deepseek_v41_mlx_weights import NativeV41Weights
    monkeypatch.setattr(NativeV41Weights,'dense_receipt',lambda self:{})
    w=SourceV41Weights.__new__(SourceV41Weights);w.source_woa_pp_policy=native.POLICY
    r=w.dense_receipt();assert r['source_woa_pp_implementation_sha256']=='fixture-binding'
    assert 'raw F32 accumulator unexposed'in r['source_woa_pp_numerical_contract']


def test_source_grouped_dispatch_returns_native_before_f32_coeff_cast(monkeypatch):
    w=SourceV41Weights.__new__(SourceV41Weights);w.source_woa_pp_policy=native.POLICY;w._ensure_dense=lambda:None
    weight=object();x=object();sentinel=object();w.array=lambda name:weight;w.scheduled_grouped_calls={};calls=[]
    def project(name,xx,ww,*,groups,counters):
        calls.append((name,xx,ww,groups));counters['source_woa_native_pp']=1;return sentinel
    monkeypatch.setattr(native,'maybe_project',project)
    assert w.grouped(x,NAME)is sentinel and calls==[(NAME,x,weight,8)]
    assert w.scheduled_grouped_calls=={'source_woa_native_pp':1}


def test_reference_and_unsupported_m1_keep_existing_grouped_paths(monkeypatch):
    from mlx2.runtime.gguf import source_dense
    w=SourceV41Weights.__new__(SourceV41Weights);w.source_woa_pp_policy=native.POLICY;w.source_bf16_loads_policy='woa_f32'
    w._ensure_dense=lambda:None;weight=mx.zeros((8192,4096),mx.bfloat16);w.array=lambda name:weight;w.scheduled_grouped_calls={};x=mx.zeros((1,1,8,4096),mx.float32);marker=object()
    monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:False)
    monkeypatch.setattr(source_dense,'maybe_project',lambda *a:marker)
    assert w.grouped(x,NAME)is marker
    assert w.scheduled_grouped_calls=={'source_woa_native_pp_reference_fallback':1,'source_bf16_grouped_direct_f32':1}
