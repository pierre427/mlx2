"""Synthetic exact-partition/ownership contracts; no checkpoint or GPU."""
from types import SimpleNamespace
import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_source_canonical_prefill import (
    POLICY, canonical_forward, canonical_model_class, partition_receipt)
from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter, source_policy
from mlx2.adapters.deepseek_v41_mlx import DeepSeekV41MLXAdapter


def fixture():
    state=SimpleNamespace(offset=0, total=np.float32(0), _in_forward=False)
    model=SimpleNamespace(identity='synthetic-canonical-m1@1',
        _source_canonical_identity='synthetic-canonical-m1@1',
        weights=SimpleNamespace(source_prefill_partition_policy=POLICY), scheduled_kv_calls={})
    calls=[]
    def validate(inputs,cache,keep):
        if inputs.ndim!=2 or inputs.shape[0]!=1 or not 1<=inputs.shape[1]<=1024:
            raise ValueError('geometry')
        if keep is not None and(type(keep)is not int or keep<0):raise ValueError('keep')
        if np.any(np.asarray(inputs)<0):raise ValueError('token')
        return [state]if cache is None else cache,None
    model.validate_forward=validate
    def forward(inputs,cache,*,logits_to_keep,capture_layers,body_only):
        assert inputs.shape==(1,1)
        value=np.float32(np.asarray(inputs)[0,0]);s=cache[0]
        s.total=np.float32(np.float32(s.total*np.float32(.9999))+value)
        s.offset+=1;calls.append((value,body_only,capture_layers))
        out=mx.array([[[float(s.total)]]],mx.float32)
        return None if body_only else out, mx.concatenate((out,out),axis=-1)if capture_layers else None
    return model,state,forward,calls


@pytest.mark.parametrize('cut',[1,63,99,127,128,129,255])
def test_arbitrary_cold_warm_and_interior_partitions_keep_identical_state_and_outputs(cut):
    values=mx.arange(257,dtype=mx.int32).reshape(1,-1)
    m,s,f,_=fixture();whole,taps=canonical_forward(m,f,values,capture_layers=(37,39))
    n,t,g,_=fixture();a,fa=canonical_forward(n,g,values[:,:cut],capture_layers=(37,39))
    b,fb=canonical_forward(n,g,values[:,cut:],capture_layers=(37,39))
    assert s.offset==t.offset==257 and s.total.tobytes()==t.total.tobytes()
    assert np.array_equal(np.asarray(whole),np.asarray(mx.concatenate((a,b),axis=1)))
    assert np.array_equal(np.asarray(taps),np.asarray(mx.concatenate((fa,fb),axis=1)))


@pytest.mark.parametrize('keep',[None,0,1,2,99])
def test_trailing_logits_and_all_taps_and_body_only_preserve_original_semantics(keep):
    m,s,f,calls=fixture();out,taps=canonical_forward(m,f,mx.arange(5)[None],logits_to_keep=keep,capture_layers=(1,))
    assert out.shape==(1,5 if keep in(None,0)else min(5,keep),1)and taps.shape==(1,5,2)
    assert len(calls)==5 and all(c[2]==(1,)for c in calls)
    n,t,g,seen=fixture();out,taps=canonical_forward(n,g,mx.arange(5)[None],capture_layers=(1,),body_only=True)
    assert out is None and taps.shape==(1,5,2)and all(c[1]for c in seen)


def test_prompt_checkpoint_row_guard_remains_applied_and_final_row_is_B1():
    from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows
    m,s,f,calls=fixture()
    with prefill_checkpoint_rows([5],width=5)as request:
        out,_=canonical_forward(m,f,mx.arange(5)[None])
    assert request.applied==1 and out.shape==(1,1,1)and len(calls)==5
    assert float(out.item())==float(s.total)


def test_invalid_late_token_refuses_before_any_mutation_and_partial_failure_poisoned():
    m,s,f,calls=fixture()
    with pytest.raises(ValueError,match='token'):canonical_forward(m,f,mx.array([[1,2,-1]]))
    assert not calls and s.offset==0
    def broken(*a,**kw):
        if s.offset==1:raise MemoryError('actual allocator failure')
        return f(*a,**kw)
    with pytest.raises(MemoryError):canonical_forward(m,broken,mx.array([[1,2,3]]))
    assert s.offset==1 and s._in_forward


@pytest.mark.parametrize('mutate',['identity','policy'])
def test_stale_model_and_numerical_policy_refuse(mutate):
    m,s,f,calls=fixture()
    if mutate=='identity':m.identity='new-generation'
    else:m.weights.source_prefill_partition_policy='adaptive_v1'
    with pytest.raises(ValueError,match='owner/profile'):canonical_forward(m,f,mx.array([[1,2]]))
    assert not calls


def test_policy_default_unchanged_profile_distinct_and_source_bound():
    assert 'deepseek_source_prefill_partition'not in source_policy()
    a=object.__new__(DeepSeekV41SourceAdapter);a.kernel_policy={};a.weights=SimpleNamespace()
    a.numerics_revision='base';a.profile='base';a.numeric_profile='base'
    a._bind_prefill_partition_identity();adaptive=(a.numerics_revision,a.profile)
    a.kernel_policy={'deepseek_source_prefill_partition':POLICY};a._bind_prefill_partition_identity()
    assert (a.numerics_revision,a.profile)!=adaptive and 'implementation='in a.numerics_revision
    assert a.weights.source_prefill_partition_policy==POLICY
    assert partition_receipt(POLICY)['arithmetic_rows']==1
    assert partition_receipt('adaptive_v1')['apcv2_reuse_enabled']is False


def test_apc_enabled_only_load_bound_canonical_and_adaptive_still_restricted(monkeypatch):
    monkeypatch.setattr(DeepSeekV41MLXAdapter,'execution_config',lambda *a,**kw:{'persistent':True})
    a=object.__new__(DeepSeekV41SourceAdapter);a.kernel_policy={}
    assert a.execution_config(max_lanes=1,prefill_step=128)['apcv2_reuse_disabled_reason']=='adaptive_prefill_partitions_not_qualified'
    a.kernel_policy={'deepseek_source_prefill_partition':POLICY}
    a.model=SimpleNamespace(identity='bound',_source_canonical_identity='bound')
    a.weights=SimpleNamespace(source_prefill_partition_policy=POLICY)
    assert 'apcv2_reuse_disabled_reason'not in a.execution_config(max_lanes=1,prefill_step=128)
    a.model.identity='other'
    with pytest.raises(ValueError,match='bound at load'):a.execution_config(max_lanes=1,prefill_step=128)


def test_subclass_owns_layers_and_preserves_base_identity_not_proxy(monkeypatch):
    from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
    def init(self):self.identity='bound';self.layers=[SimpleNamespace(model=self)]
    monkeypatch.setattr(DeepSeekV41Model,'__init__',init)
    instance=canonical_model_class()()
    assert isinstance(instance,DeepSeekV41Model)and instance.layers[0].model is instance
    assert instance._source_canonical_identity==instance.identity=='bound'


def test_real_tiny_bf16_model_matches_ordinary_B1_taps_state_and_snapshot_continuation():
    from test_deepseek_v41_cpu import config,Weights
    from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
    c=config(hidden_size=32,head_dim=32,q_lora_rank=32,qk_rope_head_dim=16,
        index_head_dim=32,o_lora_rank=16,moe_intermediate_size=32,
        simulate_quantization=True,activation_dtype='bfloat16')
    w=Weights(c);w.source_prefill_partition_policy=POLICY
    model=canonical_model_class()(c,w,identity='synthetic-canonical-source@1')
    ref=DeepSeekV41Model(c,Weights(c),identity='synthetic-ordinary-B1@1')
    cache,rc=model.make_cache(),ref.make_cache();ids=mx.array([[1,2,3,4,5,6,7]],mx.int32)
    actual,taps=model.forward_with_taps(ids,cache,(1,3,4))
    expected=[];features=[]
    for i in range(7):
        a,b=ref.forward_with_taps(ids[:,i:i+1],rc,(1,3,4));expected.append(a);features.append(b)
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint32),np.asarray(mx.concatenate(expected,axis=1)).view(np.uint32))
    np.testing.assert_array_equal(np.asarray(taps).view(np.uint32),np.asarray(mx.concatenate(features,axis=1)).view(np.uint32))
    for a,b in zip(cache,rc):
        for x,y in zip(a.state,b.state):
            if x is None:assert y is None
            else:np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
    saved=[s.snapshot()for s in cache]
    continuation=model(mx.array([[8,9,10]],mx.int32),cache)
    for row,snapshot in zip(cache,saved):row.restore(snapshot)
    pieces=mx.concatenate([model(mx.array([[i]],mx.int32),cache)for i in(8,9,10)],axis=1)
    np.testing.assert_array_equal(np.asarray(continuation).view(np.uint32),np.asarray(pieces).view(np.uint32))
    foreign=ref.make_cache()
    with pytest.raises(ValueError,match='identity'):model(mx.array([[1]],mx.int32),foreign)


def test_bank_owner_checked_before_ordinary_state_mutation():
    m,s,f,calls=fixture();seen=[]
    m._source_ast_bank_manager=SimpleNamespace(before_ordinary=lambda:seen.append(s.offset))
    canonical_forward(m,f,mx.array([[1,2]]));assert seen==[0]and s.offset==2
    def retired():raise ValueError('stale bank generation')
    m._source_ast_bank_manager.before_ordinary=retired
    with pytest.raises(ValueError,match='generation'):canonical_forward(m,f,mx.array([[3,4]]))
    assert s.offset==2
