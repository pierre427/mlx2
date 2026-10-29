"""CPU geometry/ownership evidence for an unqualified M1-dispatch candidate."""
from types import SimpleNamespace
import mlx.core as mx
import numpy as np
import pytest
from mlx2.adapters import deepseek_v41_ast_batched_dense_exact as exact
from mlx2.runtime.gguf import source_ast_m1_dense_rows as bf16rows
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from test_deepseek_v41_cpu import config,Weights


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:(_ for _ in ()).throw(AssertionError('GPU forbidden')))


def fixture():
    c=config();w=Weights(c);w._ready=lambda:None;w._ensure_dense=lambda:None;w._quant={};w.scheduled_dense_calls={}
    m=DeepSeekV41Model(c,w,identity='synthetic-m1-batch-axis@567');return m,w


def x(rows,width=8):return mx.array(np.random.default_rng(rows).normal(0,.1,(1,rows,width)).astype(np.float32))


@pytest.mark.parametrize('rows',(1,4,16))
def test_float_native_one_batch_uses_padded_stride_to_prevent_M_flatten(monkeypatch,rows):
    model,w=fixture();value=x(rows);name='blk.0.attn_q_a.weight';seen=[];original=mx.matmul
    def matmul(a,b):seen.append((a,b));return original(a,b)
    monkeypatch.setattr(mx,'matmul',matmul)
    actual=exact.linear_tile(model,value,name,rounded=False)
    assert len(seen)==1
    a,b=seen[0];assert a.shape==(rows,1,8)and b.shape==(rows,8,4)
    mx.eval(a,b,actual)
    assert np.asarray(a).strides[0]==9*4 and np.asarray(a).strides[-2]==8*4
    if rows>1:assert np.asarray(b).strides[0]==0
    control=mx.concatenate([model.linear(value[:,i:i+1],name,rounded=False)for i in range(rows)],1)
    np.testing.assert_allclose(np.asarray(actual),np.asarray(control),rtol=3e-5,atol=2e-7)
    assert exact.scheduled_receipt(model)['scheduled']['linear_rows']==rows


@pytest.mark.parametrize('rows',(1,4,16))
def test_packed_original_fp8_broadcasts_coefficients_without_copy_and_one_QMM(monkeypatch,rows):
    value=x(rows,32);q=mx.arange(12*8,dtype=mx.uint32).reshape(12,8);s=mx.ones((12,1),mx.uint8)
    weights=SimpleNamespace(_ready=lambda:None,_ensure_dense=lambda:None,_quant={'a.weight':(q,s)},
        _dense_input=lambda a:a,scheduled_dense_calls={})
    model=SimpleNamespace(weights=weights,round=lambda a:a.astype(mx.bfloat16).astype(mx.float32))
    coeff=mx.array(np.random.default_rng(1).normal(0,.1,(12,32)).astype(np.float32));seen=[]
    def qmm(a,b,*,scales,mode):
        seen.append((a,b,scales,mode));return mx.matmul(a,mx.broadcast_to(coeff.T,(rows,32,12)))
    monkeypatch.setattr(mx,'quantized_matmul',qmm)
    actual=exact.linear_tile(model,value,'a.weight',rounded=False)
    assert len(seen)==1
    a,b,scales,mode=seen[0]
    assert a.shape==(rows,1,32)and b.shape==(rows,12,8)and scales.shape==(rows,12,1)and mode=='mxfp8'
    mx.eval(b,scales,actual)
    if rows>1:assert np.asarray(b).strides[0]==np.asarray(scales).strides[0]==0
    np.testing.assert_array_equal(np.asarray(b[0]),np.asarray(q))
    np.testing.assert_allclose(np.asarray(actual),np.asarray(value@coeff.T),rtol=3e-5,atol=2e-7)


def test_paired_QAT_is_one_tile_preparation_and_two_batched_M1_products(monkeypatch):
    value=x(16,32);prepared=[];calls=[]
    q=mx.zeros((8,8),mx.uint32);s=mx.ones((8,1),mx.uint8)
    def qat(a):prepared.append(a);return a
    weights=SimpleNamespace(_ready=lambda:None,_ensure_dense=lambda:None,
        _quant={'a.weight':(q,s),'b.weight':(q,s)},_dense_input=qat,source_dense_qat_policy='fused_fp32',scheduled_dense_calls={})
    model=SimpleNamespace(weights=weights,round=lambda a:a.astype(mx.bfloat16).astype(mx.float32))
    def qmm(a,b,*,scales,mode):calls.append(a.shape);return mx.zeros((16,1,8),mx.float32)
    monkeypatch.setattr(mx,'quantized_matmul',qmm)
    a,b=exact.paired_linear_tile(model,value,'a.weight','b.weight')
    assert len(prepared)==1 and calls==[(16,1,32)]*2 and a.shape==b.shape==(1,16,8)


@pytest.mark.parametrize('rows',(1,4,16))
def test_shared_and_head_preserve_whole_row_shapes_without_Python_calls(rows):
    model,w=fixture();value=x(rows)
    actual=exact.shared_expert_tile(model.layers[0],value)
    control=mx.concatenate([model.layers[0].shared_expert(value[:,i:i+1])for i in range(rows)],1)
    np.testing.assert_allclose(np.asarray(actual),np.asarray(control),rtol=3e-5,atol=2e-7)
    h=mx.broadcast_to(value[:,:,None],(1,rows,model.args.hc_mult,8))
    pre=mx.ones((1,rows,model.args.hc_mult),mx.float32)*.25
    head=exact.head_tile(model,h,pre)
    reference=mx.concatenate([model.linear(model.norm(model.collapse(h[:,i:i+1],pre[:,i:i+1]),model.norm_weight),'output.weight',rounded=False)for i in range(rows)],1)
    np.testing.assert_allclose(np.asarray(head),np.asarray(reference),rtol=3e-5,atol=2e-7)
    assert head.shape==(1,rows,19)


@pytest.mark.parametrize('kind',('wo_a','head'))
def test_BF16_rows_keep_original_M1_source_and_only_extend_grid_y(kind):
    from mlx2.runtime.gguf import source_dense
    original=source_dense.kernel_spec(kind,1,algorithm='gemv')
    spec=bf16rows.kernel_spec(kind)
    assert spec['source']==original['source']and spec['header']==original['header']
    p=bf16rows.plan(kind,16)
    assert p['grid']==(original['grid'][0],16,original['grid'][2])and p['dot_rows']==1
    assert p['output_shape'][1]==16 and not p['onchip_weight_reuse_claim']


@pytest.mark.parametrize('rows',(0,17,True,-1))
def test_plan_refuses_unsupported_rows(rows):
    with pytest.raises(ValueError):exact.plan(rows)


def test_not_qualified_or_exact_from_CPU_shape_evidence():
    p=exact.plan(16)
    assert p['native_matrix_rows']==1 and p['native_batches']==16 and p['coefficient_batch_stride']==0
    assert p['coefficient_repack_bytes']==0 and p['python_token_row_calls']==0
    assert not p['exact_M1_claim']and not p['qualified']and not p['explicit_onchip_weight_reuse']
    assert len(exact.implementation_identity())==64


@pytest.mark.parametrize('rows',(1,4,16))
@pytest.mark.parametrize('kind',('attn','ffn'))
def test_HC_mixes_preserve_postprojection_normalization_and_reference_split(rows,kind):
    model,w=fixture();block=model.layers[0]
    h=mx.array(np.random.default_rng(rows+612).normal(0,.2,(1,rows,model.args.hc_mult,8)).astype(np.float32))
    actual=exact.mixes_tile(block,h,kind)
    controls=[block.mixes(h[:,i:i+1],kind)for i in range(rows)]
    for index,value in enumerate(actual):
        expected=mx.concatenate([row[index]for row in controls],axis=1)
        np.testing.assert_allclose(np.asarray(value),np.asarray(expected),rtol=3e-5,atol=2e-7)
    assert exact.scheduled_receipt(model)['scheduled']['mixes_tiles']==1


def test_HC_source_coefficient_policy_calls_original_split_with_whole_tile(monkeypatch):
    from mlx2.runtime.models import deepseek_v41_source_hc as hc
    from mlx2.runtime.models.deepseek_v4 import _sinkhorn_mix_fp32
    model,w=fixture();block=model.layers[0];model.hc_coeff_policy='source_fused_fp32'
    calls=[]
    def split(mixes,scale,base,*,iterations,eps):
        calls.append((mixes.shape,scale,base,iterations,eps))
        return _sinkhorn_mix_fp32(mixes,scale,base,4,iterations,eps)
    monkeypatch.setattr(hc,'split',split)
    h=mx.broadcast_to(x(16)[:,:,None],(1,16,4,8))
    actual=exact.mixes_tile(block,h,'attn')
    assert len(calls)==1 and calls[0][0]==(1,16,24)
    control=block.mixes(h,'attn')
    for a,b in zip(actual,control):np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=3e-5,atol=2e-7)
    assert model.scheduled_hc_calls['source_fused_fp32']==2


@pytest.mark.parametrize('rows',(1,4,16))
def test_router_uses_exact_candidate_projection_and_original_selection(rows):
    model,w=fixture();block=model.layers[0];value=x(rows)
    actual=exact.router_tile(block,value)
    controls=[block.router(value[:,i:i+1])for i in range(rows)]
    np.testing.assert_array_equal(np.asarray(actual[0]),np.asarray(mx.concatenate([v[0]for v in controls],1)))
    np.testing.assert_allclose(np.asarray(actual[1]),np.asarray(mx.concatenate([v[1]for v in controls],1)),rtol=3e-5,atol=2e-7)
    counts=exact.scheduled_receipt(model)['scheduled'];assert counts['linear_tiles']==counts['router_tiles']==1


@pytest.mark.parametrize('rows',(1,4,16))
def test_Engram_uses_exact_candidate_projection_and_unchanged_gate(rows):
    model,w=fixture();c=model.args
    model.args=SimpleNamespace(**{**c.__dict__,'engram_layer_ids':(1,)})
    rng=np.random.default_rng(rows+326);prefix='blk.1.';width=7
    w.data[prefix+'engram_kv.weight']=mx.array(rng.normal(0,.1,((c.hc_mult+1)*8,width)).astype(np.float32))
    w.data[prefix+'engram_q_norm.weight']=mx.array(rng.uniform(.7,1.3,(c.hc_mult,8)).astype(np.float32))
    w.data[prefix+'engram_k_norm.weight']=mx.array(rng.uniform(.7,1.3,(c.hc_mult,8)).astype(np.float32))
    h=mx.array(rng.normal(0,.2,(1,rows,c.hc_mult,8)).astype(np.float32))
    lookup=mx.array(rng.normal(0,.2,(rows,width)).astype(np.float32))
    actual=exact.engram_tile(model,h,1,lookup)
    expected=mx.concatenate([model.engram(h[:,i:i+1],1,None,rows=lookup[i:i+1])for i in range(rows)],1)
    np.testing.assert_allclose(np.asarray(actual),np.asarray(expected),rtol=3e-5,atol=2e-7)
    counts=exact.scheduled_receipt(model)['scheduled'];assert counts['linear_tiles']==counts['engram_tiles']==1
