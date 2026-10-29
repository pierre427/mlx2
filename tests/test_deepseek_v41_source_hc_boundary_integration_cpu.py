"""Synthetic Source StageA policy/dataflow fixtures; no GPU/model claim."""
from types import SimpleNamespace
import hashlib
import json
import mlx.core as mx
import numpy as np
import pytest
from mlx2.runtime.models import deepseek_v41 as v41
from mlx2.runtime.models import deepseek_v41_source_hc_boundary as hc
from mlx2.adapters.deepseek_v41_source import source_policy,DeepSeekV41SourceAdapter
from mlx2.adapters.deepseek_v41_mlx import execution_policy
from test_deepseek_v41_cpu import config,Weights

KEY='deepseek_source_hc_boundary'

@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('GPU forbidden')))
    yield
    mx.set_default_device(old)


def operands(rows=1):
    return(mx.ones((1,rows,5120),mx.float32),mx.ones((1,rows,4,5120),mx.float32),
           mx.ones((1,rows,4),mx.float32),mx.broadcast_to(mx.eye(4,dtype=mx.float32),(1,rows,4,4)),mx.ones((1,rows,4),mx.float32))


def test_source_policy_default_optin_and_native_refusal():
    assert KEY not in source_policy()
    assert KEY in DeepSeekV41SourceAdapter.adapter_owned_default_policy_keys
    assert source_policy({KEY:'expand_collapse_seq'})[KEY]=='expand_collapse_seq'
    with pytest.raises(ValueError):execution_policy({KEY:'expand_collapse_seq'})
    with pytest.raises(ValueError,match='reference HC'):source_policy({KEY:'expand_collapse_seq','deepseek_hc':'fused_fp32'})
    assert source_policy({KEY:'expand_collapse_seq','deepseek_source_hc_coefficients':'fused_fp32'})[KEY]=='expand_collapse_seq'
    for value in(None,True,1,[],{},'bad'):
        with pytest.raises(ValueError):source_policy({KEY:value})


@pytest.mark.parametrize('rows',[1,99])
def test_stage_a_keeps_stock_matmul_bf16_boundary_and_no_stage_b(monkeypatch,rows):
    args=operands(rows);calls=[]
    monkeypatch.setattr(hc,'_device',lambda:mx)
    def kernel(stage,policy,diag):
        calls.append((stage,policy,diag));assert stage=='residual'and policy=='inline_sequential'
        def run(*,inputs,**kw):
            branch,mixed,post,pre=inputs
            raw=post[...,None]*branch[...,None,:]+mixed
            updated=raw.astype(mx.bfloat16).astype(mx.float32);products=updated*pre[...,None]
            # Fixture verifies wrapper wiring; stock operator here is not GPU proof.
            collapsed=mx.sum(products,-2)
            return[updated,products,collapsed,raw]if diag else[updated,collapsed]
        return run
    monkeypatch.setattr(hc,'_kernel',kernel)
    for diag in(False,True):
        value=hc.expand_collapse(*args,return_frontiers=diag);out,front=value if diag else(value,None)
        raw=args[2][...,None]*args[0][...,None,:]+mx.matmul(mx.swapaxes(args[3],-1,-2),args[1])
        updated=raw.astype(mx.bfloat16).astype(mx.float32);collapsed=mx.sum(updated*args[4][...,None],-2).astype(mx.bfloat16).astype(mx.float32)
        for actual,expected in zip(out,(updated,collapsed)):
            np.testing.assert_array_equal(np.array(actual).view(np.uint32),np.array(expected).view(np.uint32))
        if diag:assert set(front)=={'mixed_residual','expanded_raw','collapse_products','collapsed_raw'}
    assert calls==[('residual','inline_sequential',False),('residual','inline_sequential',True)]


@pytest.mark.parametrize('rows',[0,2,98,100,128])
def test_other_row_shapes_decline_and_direct_api_refuses(rows):
    args=operands(rows)
    assert not hc.can_expand_collapse(*args)
    with pytest.raises(ValueError):hc.expand_collapse(*args)


@pytest.mark.parametrize('index',[0,1,2,3,4])
def test_foreign_dtype_declines_before_device(monkeypatch,index):
    args=list(operands());args[index]=args[index].astype(mx.bfloat16)
    monkeypatch.setattr(hc,'_device',lambda:(_ for _ in ()).throw(AssertionError('device touched')))
    assert not hc.can_expand_collapse(*args)
    with pytest.raises(ValueError):hc.expand_collapse(*args)


def block_fixture(policy,rows=1):
    branch,residual,post,comb,pre=operands(rows);calls=[]
    collapsed=mx.full(branch.shape,3,mx.float32);normalized=mx.full(branch.shape,4,mx.float32)
    ffnpre=pre+2;ffnpost=post+2;ffncomb=comb+2
    def collapse(h,p):calls.append(('collapse',h,p));return collapsed
    def norm(x,w):calls.append(('norm',x,w));return normalized
    def expand(x,h,p,c):calls.append(('expand',x,h,p,c));return residual
    m=SimpleNamespace(source_hc_boundary_policy=policy,scheduled_hc_calls={},collapse=collapse,norm=norm,expand=expand)
    def mixes(h,kind):calls.append(('mixes',h,kind));return(pre,post,comb)if kind=='attn'else(ffnpre,ffnpost,ffncomb)
    b=SimpleNamespace(model=m,c=None,attn_norm=object(),ffn_norm=object(),mixes=mixes,attn=lambda *a:branch)
    return b,residual,pre,calls,(ffnpre,ffnpost,ffncomb)


@pytest.mark.parametrize('rows',[1,99])
def test_attention_ffn_only_reuses_collapsed_preserves_coefficient_and_norm_order(monkeypatch,rows):
    block,residual,pre,calls,ffn=block_fixture('expand_collapse_seq',rows);candidate_h=residual+1;candidate_x=mx.ones((1,rows,5120),mx.float32)*7;captured=[]
    def run(*args):captured.append(args);return candidate_h,candidate_x
    monkeypatch.setattr(hc,'expand_collapse',run)
    result=v41.Block.attention_half(block,residual,pre,object(),object(),object())
    assert result[0]is candidate_h and all(a is b for a,b in zip(result[2:],ffn))
    assert [x[0]for x in calls]==['mixes','collapse','norm','mixes','norm']
    assert calls[3][1]is candidate_h and calls[3][2]=='ffn'
    assert calls[4][1]is candidate_x and calls[4][2]is block.ffn_norm
    assert len(captured)==1 and captured[0][1]is residual and captured[0][4]is pre
    assert block.model.scheduled_hc_calls=={'source_hc_boundary_expand_collapse':1}


@pytest.mark.parametrize('policy,rows,count',[('reference',1,{}),('expand_collapse_seq',2,{'source_hc_boundary_reference':1})])
def test_default_and_unsupported_whole_boundary_stay_ordinary(monkeypatch,policy,rows,count):
    block,residual,pre,calls,ffn=block_fixture(policy,rows)
    monkeypatch.setattr(hc,'expand_collapse',lambda *a:(_ for _ in ()).throw(AssertionError('candidate called')))
    result=v41.Block.attention_half(block,residual,pre,object(),object(),object())
    assert result[0]is residual
    assert [x[0]for x in calls]==['mixes','collapse','norm','expand','mixes','collapse','norm']
    assert block.model.scheduled_hc_calls==count


def test_candidate_failure_propagates_without_replaying_mutated_attention(monkeypatch):
    block,residual,pre,calls,ffn=block_fixture('expand_collapse_seq')
    monkeypatch.setattr(hc,'expand_collapse',lambda *a:(_ for _ in ()).throw(RuntimeError('component failed')))
    with pytest.raises(RuntimeError,match='component failed'):v41.Block.attention_half(block,residual,pre,object(),object(),object())
    assert [x[0]for x in calls]==['mixes','collapse','norm']
    assert block.model.scheduled_hc_calls=={}


@pytest.mark.parametrize('policy,hcp,dtype',[('bad','reference','bfloat16'),('expand_collapse_seq','fused_fp32','bfloat16'),('expand_collapse_seq','reference','float32')])
def test_model_invalid_policy_refuses_before_weights(policy,hcp,dtype):
    with pytest.raises(ValueError,match='HC boundary'):v41.DeepSeekV41Model(config(activation_dtype=dtype),object(),identity='synthetic@hc',hc_policy=hcp,source_hc_boundary_policy=policy)


def test_profile_and_apc_identity_distinguish_optin_with_reference_default():
    c=config(activation_dtype='bfloat16');a=v41.DeepSeekV41Model(c,Weights(c),identity='synthetic@hc');b=v41.DeepSeekV41Model(c,Weights(c),identity='synthetic@hc',source_hc_boundary_policy='expand_collapse_seq')
    assert ':source_hc_boundary='not in a.identity and ':source_hc_boundary=expand_collapse_seq'in b.identity
    with pytest.raises(ValueError,match='identity'):b(mx.array([[1]],mx.int32),a.make_cache())
    digest=lambda x:hashlib.sha256(json.dumps(source_policy(x),sort_keys=True).encode()).hexdigest()
    assert digest({})!=digest({KEY:'expand_collapse_seq'})
