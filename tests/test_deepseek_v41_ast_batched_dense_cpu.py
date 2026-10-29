"""Synthetic CPU arithmetic and full-row call contracts; no checkpoint/GPU gate."""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_ast_batched_dense as dense
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from test_deepseek_v41_cpu import config, Weights


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:(_ for _ in ()).throw(AssertionError('GPU forbidden')))


def fixture():
    c=config();w=Weights(c);m=DeepSeekV41Model(c,w,identity='synthetic-ds41-ast-dense@seed567')
    m.eval_interval=0
    return m,w


def xrows(rows,width=8):
    rng=np.random.default_rng(881+rows)
    return mx.array(rng.normal(0,.3,(1,rows,width)).astype(np.float32))


def compare(a,b):
    mx.eval(a,b)
    np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=3e-5,atol=3e-6)


@pytest.mark.parametrize('rows',(1,4,16))
def test_linear_is_one_whole_tile_product_with_independent_f64_oracle(rows):
    model,w=fixture();x=xrows(rows);name='blk.0.attn_q_a.weight';calls=[]
    original=w.linear
    def linear(n):
        project=original(n)
        def apply(value):calls.append((n,value.shape));return project(value)
        return apply
    w.linear=linear
    actual=dense.linear_tile(model,x,name,rounded=False)
    assert calls==[(name,(1,rows,8))]
    oracle=np.asarray(x).astype(np.float64)@np.asarray(w.array(name)).astype(np.float64).T
    np.testing.assert_allclose(np.asarray(actual),oracle,rtol=3e-5,atol=3e-6)
    assert dense.scheduled_receipt(model)['scheduled']['linear_rows']==rows


@pytest.mark.parametrize('rows',(1,4,16))
def test_shared_expert_uses_one_batched_gate_up_and_one_down(rows):
    model,w=fixture();x=xrows(rows);block=model.layers[0];calls=[];original=w.linear
    def linear(n):
        project=original(n)
        def apply(value):calls.append((n,value.shape));return project(value)
        return apply
    w.linear=linear
    actual=dense.shared_expert_tile(block,x)
    assert len(calls)==3 and all(shape[:2]==(1,rows)for _,shape in calls)
    control=mx.concatenate([block.shared_expert(x[:,i:i+1])for i in range(rows)],1)
    compare(actual,control)
    assert dense.scheduled_receipt(model)['scheduled']['shared_expert_tiles']==1


@pytest.mark.parametrize('rows',(1,4,16))
def test_grouped_woa_keeps_one_physical_row_tile_and_raw_f32(rows):
    model,w=fixture();rng=np.random.default_rng(419)
    x=mx.array(rng.normal(0,.3,(1,rows,2,8)).astype(np.float32));name='blk.0.attn_output_a.weight'
    calls=[];original=w.grouped
    def grouped(value,n,*,groups):calls.append((value.shape,n,groups));return original(value,n,groups)
    w.grouped=grouped
    actual=dense.grouped_tile(model,x,name,groups=2)
    assert calls==[((1,rows,2,8),name,2)] and actual.dtype==mx.float32
    expected=mx.concatenate([original(x[:,i:i+1],name,2)for i in range(rows)],1)
    compare(actual,expected)


@pytest.mark.parametrize('rows',(1,4,16))
def test_head_collapse_norm_and_full_vocabulary_product_once(rows):
    model,w=fixture();rng=np.random.default_rng(500+rows)
    h=mx.array(rng.normal(0,.1,(1,rows,model.args.hc_mult,8)).astype(np.float32))
    pre=mx.array(rng.uniform(.1,.8,h.shape[:-1]).astype(np.float32));calls=[];original=w.linear
    def linear(name):
        apply=original(name)
        def product(x):calls.append((name,x.shape));return apply(x)
        return product
    w.linear=linear
    actual=dense.head_tile(model,h,pre)
    assert calls==[('output.weight',(1,rows,8))]and actual.shape==(1,rows,19)
    control=mx.concatenate([model.linear(model.norm(model.collapse(h[:,i:i+1],pre[:,i:i+1]),model.norm_weight),'output.weight',rounded=False)for i in range(rows)],1)
    compare(actual,control)


@pytest.mark.parametrize('rows',(1,4,16))
def test_router_projection_is_batched_and_original_selection_order_preserved(rows):
    model,w=fixture();block=model.layers[0];x=xrows(rows)
    ids,scores=dense.router_tile(block,x)
    control=[block.router(x[:,i:i+1])for i in range(rows)]
    np.testing.assert_array_equal(np.asarray(ids),np.asarray(mx.concatenate([v[0]for v in control],1)))
    compare(scores,mx.concatenate([v[1]for v in control],1))
    assert dense.scheduled_receipt(model)['scheduled']['router_tiles']==1


@pytest.mark.parametrize('rows',(1,4,16))
def test_engram_one_projection_keeps_original_gate_formula_without_table_reads(rows):
    model,w=fixture();model.args=SimpleNamespace(**{**model.args.__dict__,'engram_layer_ids':(1,)})
    rng=np.random.default_rng(215+rows);width=7;layer=1;prefix='blk.1.'
    w.data[prefix+'engram_kv.weight']=mx.array(rng.normal(0,.1,((model.args.hc_mult+1)*8,width)).astype(np.float32))
    w.data[prefix+'engram_q_norm.weight']=mx.array(rng.uniform(.7,1.3,(model.args.hc_mult,8)).astype(np.float32))
    w.data[prefix+'engram_k_norm.weight']=mx.array(rng.uniform(.7,1.3,(model.args.hc_mult,8)).astype(np.float32))
    h=mx.array(rng.normal(0,.2,(1,rows,model.args.hc_mult,8)).astype(np.float32))
    lookup=mx.array(rng.normal(0,.3,(rows,width)).astype(np.float32));calls=[];original=w.linear
    def linear(name):
        product=original(name)
        def apply(x):calls.append((name,x.shape));return product(x)
        return apply
    w.linear=linear
    actual=dense.engram_tile(model,h,layer,lookup)
    assert calls==[(prefix+'engram_kv.weight',(1,rows,width))]
    expected=mx.concatenate([model.engram(h[:,i:i+1],layer,None,rows=lookup[i:i+1])for i in range(rows)],1)
    compare(actual,expected)


def test_paired_projection_is_one_tile_preparation_or_two_tiles_never_rows():
    model,w=fixture();x=xrows(16);a='blk.0.attn_q_a.weight';b='blk.0.attn_kv.weight';calls=[]
    model.can_pair=lambda *args:True
    def paired(value,aa,bb,*,rounded):
        calls.append((value.shape,aa,bb,rounded))
        return model.linear(value,aa,rounded=rounded),model.linear(value,bb,rounded=rounded)
    w.paired_project=paired
    first,second=dense.paired_linear_tile(model,x,a,b,rounded=False)
    assert calls==[((1,16,8),a,b,False)]
    model.can_pair=lambda *args:False
    fallback=dense.paired_linear_tile(model,x,a,b,rounded=False)
    compare(first,fallback[0]);compare(second,fallback[1])
    counts=dense.scheduled_receipt(model)['scheduled']
    assert counts['paired_QAT_reuse_tiles']==counts['paired_separate_tiles']==1


def test_original_mxfp8_coefficients_and_single_whole_tile_qat_are_not_repacked(monkeypatch):
    from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
    from mlx2.runtime.models import deepseek_v41_quant as quant
    w=SourceV41Weights.__new__(SourceV41Weights);w._ready=lambda:None;w._ensure_dense=lambda:None
    w.source_dense_qat_policy='fused_fp32';w.source_fp8_pp_policy='reference'
    w.scheduled_dense_calls={};w._q_b_prefill_project=lambda *a:None
    q=mx.zeros((12,8),mx.uint32);scales=mx.ones((12,1),mx.uint8)
    name='blk.0.attn_q_b.weight';w._quant={name:(q,scales)}
    model=SimpleNamespace(weights=w)
    model.linear=lambda x,n,rounded=True:(w.linear(n)(x).astype(mx.bfloat16).astype(mx.float32)if rounded else w.linear(n)(x))
    prepared=[];products=[];reference_quant=quant.quantize
    def qat(value,kind):prepared.append(value.shape);return reference_quant(value,kind)
    monkeypatch.setattr(quant,'fused_quantize',qat)
    coefficient=mx.array(np.random.default_rng(42).normal(0,.1,(12,32)).astype(np.float32))
    def qmm(value,packed,*,scales,mode):products.append((value.shape,packed,scales,mode));return value@coefficient.T
    monkeypatch.setattr(mx,'quantized_matmul',qmm)
    x=xrows(16,32);actual=dense.linear_tile(model,x,name)
    assert prepared==[(1,16,32)]and len(products)==1
    assert products[0][0]==(1,16,32)and products[0][1]is q and products[0][2]is scales and products[0][3]=='mxfp8'
    expected=(reference_quant(x,'fp8')@coefficient.T).astype(mx.bfloat16).astype(mx.float32)
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint32),np.asarray(expected).view(np.uint32))


@pytest.mark.parametrize('rows',(0,17,True,-1))
def test_plan_refuses_bad_physical_rows(rows):
    with pytest.raises(ValueError):dense.plan(rows)


@pytest.mark.parametrize('shape,dtype', [((2,16,8),mx.float32),((1,0,8),mx.float32),((1,17,8),mx.float32),((16,8),mx.float32),((1,16,8),mx.float16)])
def test_bad_linear_input_refuses_before_model(shape,dtype):
    model=SimpleNamespace(linear=lambda *a,**k:(_ for _ in ()).throw(AssertionError('must refuse')))
    with pytest.raises(ValueError):dense.linear_tile(model,mx.zeros(shape,dtype),'valid.weight')


def test_invalid_owner_outputs_and_head_pre_are_refused():
    x=xrows(16);model,w=fixture()
    with pytest.raises(ValueError):dense.linear_tile(model,x,'missing')
    with pytest.raises(ValueError):dense.linear_tile(model,x,'output.weight',rounded=1)
    fake=SimpleNamespace(linear=lambda *a,**k:mx.zeros((1,1,19),mx.float32))
    with pytest.raises(ValueError):dense.linear_tile(fake,x,'output.weight')
    h=mx.zeros((1,16,model.args.hc_mult,8),mx.float32)
    with pytest.raises(ValueError):dense.head_tile(model,h,mx.zeros((1,1,model.args.hc_mult),mx.float32))
    assert dense.plan(16)['python_token_row_calls']==0 and not dense.plan(16)['exact_M1_claim']
    assert len(dense.implementation_identity())==64
