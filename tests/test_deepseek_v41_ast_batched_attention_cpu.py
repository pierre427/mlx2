"""Tiny synthetic Source segment equations/state; no checkpoint/GPU claim."""
import copy

import mlx.core as mx
import numpy as np
import pytest

from test_deepseek_v41_cpu import config, Weights
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from mlx2.adapters.deepseek_v41_ast_batched_attention import (
    attention_half_tile, attention_core, LayerAncestrySegments, PARENTS,
    PATHS, DESCENDANTS,
)


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('GPU forbidden')))


def clone(row):
    result=copy.copy(row)
    for key in ('keys','buf_kv','buf_gate','_pooled','_index_keys','hash_tail'):
        value=getattr(row,key)
        setattr(result,key,None if value is None else mx.asarray(value,copy=True))
    return result


def compare(left,right):
    assert left.meta_state==right.meta_state
    assert left._pool_count==right._pool_count
    assert left._capacity==right._capacity
    assert not left.pending and not right.pending
    for key in ('keys','buf_kv','buf_gate','_pooled','_index_keys','hash_tail'):
        a,b=getattr(left,key),getattr(right,key)
        assert (a is None)==(b is None)
        if a is not None:np.testing.assert_allclose(np.asarray(a),np.asarray(b),atol=3e-6,rtol=3e-5)


@pytest.mark.parametrize('prefix',(0,1,5,6))
@pytest.mark.parametrize('finite',(False,True))
@pytest.mark.parametrize('topk',(2,32))
def test_batched_half_all_ratios_index_refresh_partialgroups_and_private_caches(prefix,finite,topk,monkeypatch):
    c=config(sliding_window=4,index_topk=topk)
    model=DeepSeekV41Model(c,Weights(c),identity='synthetic-ast-attention@seed760',
                          index_policy='finite_all_retained'if finite else'reference')
    base=model.make_cache();model.eval_interval=0
    if prefix:model._forward_impl(mx.array([list(range(prefix))],mx.int32),base)
    stamp=[(s.meta_state,tuple(id(v)for v in(s.keys,s._pooled,s._index_keys)))for s in base]
    expected=[[None]*c.num_hidden_layers for _ in range(16)]
    expected_shared=[{}for _ in range(16)];shared=[{}for _ in range(16)];segments=[None]*c.num_hidden_layers
    rng=np.random.default_rng(760)
    for layer,block in enumerate(model.layers):
        h=mx.array(rng.normal(0,.2,(1,16,c.hc_mult,c.hidden_size)).astype(np.float32))
        pre=mx.array(rng.normal(0,.2,(1,16,c.hc_mult)).astype(np.float32))
        ordinary=[]
        for node,parent in enumerate(PARENTS):
            state=clone(base[layer]if parent<0 else expected[parent][layer])
            expected[node][layer]=state
            ordinary.append(block.attention_half(h[:,node:node+1],pre[:,node:node+1],state,
                                                  expected[node],expected_shared[node]))
        # Candidate must never delegate16 serial half calls.
        with monkeypatch.context()as guard:
            guard.setattr(block,'attention_half',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('serial half')))
            actual=attention_half_tile(block,h,pre,base,segments,shared)
        for part in range(5):
            reference=mx.concatenate([row[part]for row in ordinary],axis=1)
            np.testing.assert_allclose(np.asarray(actual[part]),np.asarray(reference),atol=4e-6,rtol=4e-5)
        descriptor=segments[layer]=actual[-1]
        assert descriptor.base is base[layer]
        assert descriptor.receipt()['full_prefix_materializations_during_attention']==0
        for node in range(16):compare(descriptor.materialize(node),expected[node][layer])
        if block.attn.index_source:
            for node in range(16):
                np.testing.assert_array_equal(np.asarray(shared[node]['indices']),np.asarray(expected_shared[node]['indices']))
                np.testing.assert_array_equal(np.asarray(shared[node]['selected_mask']),np.asarray(expected_shared[node]['selected_mask']))
    assert model.scheduled_kv_calls['source_ast_grouped_attention_tiles']==c.num_hidden_layers
    assert stamp==[(s.meta_state,tuple(id(v)for v in(s.keys,s._pooled,s._index_keys)))for s in base]


def test_descendant_visibility_and_sink_mass_excludes_future_and_siblings():
    c=config();model=DeepSeekV41Model(c,Weights(c),identity='synthetic-ast-mask@1')
    base=model.make_cache()[0]
    kv=mx.ones((1,1,16,c.head_dim))
    segment=LayerAncestrySegments(base,kv)
    visibility=np.asarray(segment.local_visibility)
    for node,path in enumerate(PATHS):
        assert np.flatnonzero(visibility[node]).tolist()==list(path)
    for ancestor,mask in enumerate(DESCENDANTS):
        assert mask==sum(1<<n for n in range(16)if visibility[n,ancestor])
    q=mx.zeros((1,c.num_attention_heads,16,c.head_dim))
    sinks=mx.zeros((c.num_attention_heads,))
    result=np.asarray(attention_core(q,segment,sinks))
    for node,path in enumerate(PATHS):np.testing.assert_allclose(result[:,:,node],len(path)/(len(path)+1),rtol=1e-6)
    changed=mx.ones((1,1,16,c.head_dim));changed[:,:,6:]=1000
    other=LayerAncestrySegments(base,changed)
    np.testing.assert_array_equal(np.asarray(attention_core(q,segment,sinks))[:,:,:6],
                                  np.asarray(attention_core(q,other,sinks))[:,:,:6])


def test_segment_refuses_changed_owner_history_without_tails_and_wrong_node():
    c=config();model=DeepSeekV41Model(c,Weights(c),identity='synthetic-ast-owner@1')
    base=model.make_cache()[0]
    segment=LayerAncestrySegments(base,mx.zeros((1,1,16,c.head_dim)))
    with pytest.raises(ValueError,match='outside16'):segment.materialize(True)
    base.keys=mx.zeros((1,1,0,c.head_dim))
    with pytest.raises(ValueError,match='owner changed'):segment.materialize(0)


def test_refusal_before_cache_mutation_and_unique_sibling_dicts():
    c=config();model=DeepSeekV41Model(c,Weights(c),identity='synthetic-ast-refusal@1')
    base=model.make_cache();segments=[None]*c.num_hidden_layers
    h=mx.zeros((1,16,c.hc_mult,c.hidden_size));pre=mx.zeros((1,16,c.hc_mult))
    with pytest.raises(ValueError,match='dictionaries alias'):
        attention_half_tile(model.layers[0],h,pre,base,segments,[{}]*16)
    assert all(s.offset==0 for s in base)and all(s is None for s in segments)
