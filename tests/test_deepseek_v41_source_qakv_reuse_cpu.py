"""Source Qa/KV QAT reuse fixtures only; no actual checkpoint or GPU work."""
import hashlib
import json
from types import SimpleNamespace
import mlx.core as mx
import numpy as np
import pytest
from mlx2.adapters.deepseek_v41_source import source_policy,DeepSeekV41SourceAdapter
from mlx2.adapters.deepseek_v41_mlx import execution_policy
from mlx2.adapters import deepseek_v41_source_weights as source
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from mlx2.runtime.models import deepseek_v41_quant as quant

KEY='deepseek_source_qakv';QA='blk.0.attn_q_a.weight';KV='blk.0.attn_kv.weight'


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('GPU forbidden')))
    yield
    mx.set_default_device(old)


@pytest.fixture
def w():
    value=SourceV41Weights.__new__(SourceV41Weights)
    value.source_qakv_policy='qat_reuse';value.source_dense_qat_policy='fused_fp32';value.dense_pair_policy='fused'
    value._quant={QA:(mx.zeros((1280,1280),mx.uint32),mx.zeros((1280,160),mx.uint8)),
                  KV:(mx.zeros((512,1280),mx.uint32),mx.zeros((512,160),mx.uint8))}
    value.scheduled_dense_calls={};value._ensure_dense=lambda:None;value._ready=lambda:None
    return value


def fake_math(monkeypatch,w):
    prepared=[];calls=[];raw=[]
    def qat(x,kind):
        assert kind=='fp8'and x.dtype==mx.float32
        result=x+mx.array(1,mx.float32);prepared.append(result);return result
    def qmm(x,q,*,scales,mode):
        calls.append((x,q,scales,mode))
        assert mode=='mxfp8'
        # Independent half-way cases exercise each projection's BF16 RNE.
        v=1.00390625 if q is w._quant[QA][0] else 1.01171875
        result=mx.full((*x.shape[:-1],q.shape[0]),v,mx.float32);raw.append(result);return result
    monkeypatch.setattr(quant,'fused_quantize',qat);monkeypatch.setattr(mx,'quantized_matmul',qmm)
    return prepared,calls,raw


def test_policy_reference_default_prerequisite_and_source_only():
    assert KEY not in source_policy()
    assert KEY in DeepSeekV41SourceAdapter.adapter_owned_default_policy_keys
    assert source_policy({KEY:'reference'})[KEY]=='reference'
    for value in (None,True,1,[],{},'bad'):
        with pytest.raises(ValueError):source_policy({KEY:value})
    for dense in (None,'reference'):
        p={KEY:'qat_reuse'}
        if dense is not None:p['deepseek_source_dense_qat']=dense
        with pytest.raises(ValueError,match='requires fused_fp32'):source_policy(p)
    p={KEY:'qat_reuse','deepseek_source_dense_qat':'fused_fp32'}
    assert source_policy(p)[KEY]=='qat_reuse'
    with pytest.raises(ValueError):execution_policy(p)


@pytest.mark.parametrize('policy,dense',[('bad','fused_fp32'),('qat_reuse','reference')])
def test_constructor_refuses_invalid_before_artifact(policy,dense):
    with pytest.raises(ValueError):SourceV41Weights({},expert_cache_bytes=1,qakv_policy=policy,dense_qat_policy=dense)


@pytest.mark.parametrize('policy,paired',[('reference','reference'),('qat_reuse','fused')])
def test_constructor_sets_instance_pairing_only_under_optin(monkeypatch,tmp_path,policy,paired):
    monkeypatch.setattr(source,'weight_plan',lambda *a:{})
    monkeypatch.setattr(source,'ReadPool',lambda *a,**kw:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(source,'SourceExpertStore',lambda *a,**kw:SimpleNamespace())
    value=SourceV41Weights(dict(root=tmp_path,aliases={},tensors={}),expert_cache_bytes=1,
                          qakv_policy=policy,dense_qat_policy='fused_fp32')
    assert value.source_qakv_policy==policy and value.dense_pair_policy==paired
    value._read_pool.close()


@pytest.mark.parametrize('rows',[1,99])
@pytest.mark.parametrize('rounded',[False,True])
def test_pair_keeps_two_stock_dimensions_codes_scales_shared_input_and_rounding(monkeypatch,w,rows,rounded):
    prepared,calls,raw=fake_math(monkeypatch,w);x=mx.zeros((1,rows,5120),mx.float32)
    a,b=w.paired_project(x,QA,KV,rounded=rounded);mx.eval(a,b)
    assert len(prepared)==1 and len(calls)==2
    assert calls[0][0]is calls[1][0]
    # _dense_input's final astype(F32) can create a new Python wrapper even
    # when the QAT result already has that dtype. Both stock QMMs must receive
    # that same final prepared object, with the unmodified QAT values.
    np.testing.assert_array_equal(np.array(calls[0][0]),np.array(prepared[0]))
    assert [c[1:]for c in calls]==[(w._quant[QA][0],w._quant[QA][1],'mxfp8'),(w._quant[KV][0],w._quant[KV][1],'mxfp8')]
    assert [a.shape,b.shape]==[(1,rows,1280),(1,rows,512)]
    for actual,reference in zip((a,b),raw):
        expected=reference.astype(mx.bfloat16).astype(mx.float32)if rounded else reference
        np.testing.assert_array_equal(np.array(actual).view(np.uint32),np.array(expected).view(np.uint32))
    if rounded:
        assert float(np.array(a)[0,0,0])==1. and float(np.array(b)[0,0,0])==1.015625
    else:
        assert np.any(np.array(a).view(np.uint32)&0xffff)
    assert w.scheduled_dense_calls==dict(source_dense_qat_fused=1,official_mxfp8_act32=2,
        source_qakv_qat_reuse_pairs=1,source_qakv_input_qat_calls_avoided=1)


@pytest.mark.parametrize('rows',[1,99])
def test_baseline_two_input_qats_vs_one_pair_without_weight_or_raw_change(monkeypatch,w,rows):
    prepared,calls,_=fake_math(monkeypatch,w);x=mx.zeros((1,rows,5120),mx.float32)
    a,b=w.linear(QA)(x),w.linear(KV)(x);mx.eval(a,b)
    assert len(prepared)==2 and calls[0][0]is not calls[1][0]
    prepared.clear();calls.clear();w.scheduled_dense_calls.clear()
    pair=w.paired_project(x,QA,KV,rounded=False);mx.eval(*pair)
    assert len(prepared)==1 and calls[0][0]is calls[1][0]
    for lhs,rhs in zip((a,b),pair):np.testing.assert_array_equal(np.array(lhs).view(np.uint32),np.array(rhs).view(np.uint32))


@pytest.mark.parametrize('rows',[1,99,1024])
@pytest.mark.parametrize('layer',[0,39])
def test_pair_contract_admits_only_fixed_source_width_and_layer(w,rows,layer):
    qa,kv=f'blk.{layer}.attn_q_a.weight',f'blk.{layer}.attn_kv.weight'
    w._quant={qa:w._quant[QA],kv:w._quant[KV]}
    x=mx.zeros((1,rows,5120),mx.float32)
    assert w.can_fuse_dense(x,qa,kv)
    assert DeepSeekV41Model.can_pair(SimpleNamespace(weights=w),x,qa,kv)


@pytest.mark.parametrize('names',[
 (), (QA,), (QA,KV,QA), (KV,QA), (QA,'blk.1.attn_kv.weight'),
 ('blk.40.attn_q_a.weight','blk.40.attn_kv.weight'), ('blk.00.attn_q_a.weight','blk.00.attn_kv.weight'),
 ('blk.-1.attn_q_a.weight','blk.-1.attn_kv.weight'),
 ('blk.0.attn_compressor_kv.weight','blk.0.attn_compressor_gate.weight'),
 ('blk.0.ffn_gate_shexp.weight','blk.0.ffn_up_shexp.weight'), (None,KV),
])
def test_foreign_reversed_cross_layer_and_unrelated_pairs_decline_before_load(w,names):
    w._ensure_dense=lambda:(_ for _ in ()).throw(AssertionError('unsupported names loaded weights'))
    assert not w.can_fuse_dense(mx.zeros((1,1,5120),mx.float32),*names)


@pytest.mark.parametrize('shape,dtype',[
 ((1,0,5120),mx.float32),((1,1025,5120),mx.float32),((2,1,5120),mx.float32),
 ((1,5120),mx.float32),((1,1,5119),mx.float32),((1,1,5120),mx.bfloat16),((1,1,5120),mx.float16),
])
def test_bad_activation_declines_before_dense_load(w,shape,dtype):
    w._ensure_dense=lambda:(_ for _ in ()).throw(AssertionError('unsupported input loaded weights'))
    assert not w.can_fuse_dense(mx.zeros(shape,dtype),QA,KV)


@pytest.mark.parametrize('bad',['missing','affine_tuple','q_dtype','q_shape','s_dtype','s_shape','nonarray'])
def test_nonoriginal_quant_layout_declines_and_direct_api_fails_closed(w,bad):
    q,s=w._quant[QA]
    if bad=='missing':w._quant.pop(QA)
    elif bad=='affine_tuple':w._quant[QA]=(q,s,s)
    elif bad=='q_dtype':w._quant[QA]=(q.astype(mx.uint8),s)
    elif bad=='q_shape':w._quant[QA]=(q[:,:-1],s)
    elif bad=='s_dtype':w._quant[QA]=(q,s.astype(mx.float32))
    elif bad=='s_shape':w._quant[QA]=(q,s[:,:-1])
    else:w._quant[QA]=('q',s)
    x=mx.zeros((1,1,5120),mx.float32)
    assert not w.can_fuse_dense(x,QA,KV)
    with pytest.raises(ValueError,match='unsupported Source'):w.paired_project(x,QA,KV)


def test_default_reference_and_direct_invalid_rounded_fail_without_preparing(monkeypatch,w):
    monkeypatch.setattr(w,'_dense_input',lambda *a:(_ for _ in ()).throw(AssertionError('prepared unsupported pair')))
    x=mx.zeros((1,1,5120),mx.float32)
    for flag in (None,0,1,'false'):
        with pytest.raises(ValueError):w.paired_project(x,QA,KV,rounded=flag)
    w.source_qakv_policy='reference';w.dense_pair_policy='reference'
    assert not DeepSeekV41Model.can_pair(SimpleNamespace(weights=w),x,QA,KV)
    with pytest.raises(ValueError):w.paired_project(x,QA,KV)
    w.source_qakv_policy='qat_reuse';w.source_dense_qat_policy='reference';w.dense_pair_policy='fused'
    assert not w.can_fuse_dense(x,QA,KV)


def test_shadow_can_disable_model_pair_dispatch_but_use_raw_api(monkeypatch,w):
    fake_math(monkeypatch,w);x=mx.zeros((1,1,5120),mx.float32);w.dense_pair_policy='reference'
    assert not DeepSeekV41Model.can_pair(SimpleNamespace(weights=w),x,QA,KV)
    assert w.can_fuse_dense(x,QA,KV)
    a,b=w.paired_project(x,QA,KV,rounded=False);mx.eval(a,b)


@pytest.mark.parametrize('failure',['ready','qat','first_qmm','second_qmm'])
def test_failure_never_changes_provider_or_pairs_unrelated_state(monkeypatch,w,failure):
    prepared,calls,_=fake_math(monkeypatch,w);sentinel=object();w.expert_stream=sentinel
    fail=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('fixture '+failure))
    if failure=='ready':w._ready=fail
    elif failure=='qat':monkeypatch.setattr(quant,'fused_quantize',fail)
    else:
        original=mx.quantized_matmul;n=[0]
        def qmm(*a,**kw):
            n[0]+=1
            if n[0]==(1 if failure=='first_qmm'else 2):return fail()
            return original(*a,**kw)
        monkeypatch.setattr(mx,'quantized_matmul',qmm)
    with pytest.raises(RuntimeError,match='fixture'):w.paired_project(mx.zeros((1,1,5120),mx.float32),QA,KV)
    assert w.expert_stream is sentinel and w.source_qakv_policy=='qat_reuse'and w.dense_pair_policy=='fused'
    assert 'source_qakv_qat_reuse_pairs'not in w.scheduled_dense_calls


def test_policy_identity_and_dense_diagnostics_preserve_attention_policy(monkeypatch,w):
    base={'deepseek_source_dense_qat':'fused_fp32','deepseek_index_policy':'finite_all_retained',
          'deepseek_source_attention_math':'all_retained_broadcast'}
    off=source_policy(base);on=source_policy({**base,KEY:'qat_reuse'})
    digest=lambda p:hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    assert digest(off)!=digest(on)and on['deepseek_source_attention_math']=='all_retained_broadcast'
    from mlx2.adapters.deepseek_v41_mlx_weights import NativeV41Weights
    monkeypatch.setattr(NativeV41Weights,'dense_receipt',lambda self:{})
    assert w.dense_receipt()['source_qakv']=='qat_reuse'
