"""Synthetic GQA7/D128 contract tests; no checkpoint qualification."""
import copy
import hashlib
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.nanbeige43_segmented_attention import (
    attend_segments,
    source_identity,
)
from mlx2.adapters.nanbeige43_tree import (
    NanbeigePackedTreeExecutor,
    estimate_temporary_memory,
)
from mlx2.runtime.models.standard_decoder import Model, ModelArgs


@pytest.fixture(autouse=True)
def fixture_receipt(record_property):
    previous=mx.default_device(); mx.set_default_device(mx.cpu)
    record_property('source_revision',subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    record_property('working_source_sha256',source_identity())
    record_property('checkpoint','synthetic-segmented-gqa7@sha256:'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    yield
    mx.set_default_device(previous)


def model_fixture():
    mx.random.seed(47)
    model=Model(ModelArgs(model_type='llama',hidden_size=56,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=56,num_key_value_heads=8,head_dim=128,vocab_size=64,rms_norm_eps=1e-5,
        max_position_embeddings=4096,rope_theta=120000000,tie_word_embeddings=False))
    mx.eval(model.parameters()); return model


def prefill(model,n,lane):
    cache=model.make_cache()
    if n: mx.eval(model(mx.array([[(lane+i)%64 for i in range(n)]]),cache))
    return cache

@pytest.mark.parametrize('batch',[2,4])
@pytest.mark.parametrize('rows',[1,4])
def test_segmented_one_traversal_all_branches_abort(batch,rows,monkeypatch):
    model=model_fixture(); prefixes=[0,5,3,7][:batch]
    caches=[prefill(model,n,i) for i,n in enumerate(prefixes)]
    refs=copy.deepcopy(caches); parents=[[-1,0,0,2]]*batch; tokens=[[1+i,3+i,5+i,7+i] for i in range(batch)]
    original=model.forward_with_taps; shapes=[]
    def counted(inputs,*a,**kw): shapes.append(inputs.shape); return original(inputs,*a,**kw)
    monkeypatch.setattr(model,'forward_with_taps',counted)
    def refuse_pad(*a,**kw): raise AssertionError('segmented route padded KV')
    monkeypatch.setattr(mx,'pad',refuse_pad)
    provider=NanbeigePackedTreeExecutor.create(segmented_attention=True,shared_kv_rows=rows)
    logits,taps,txn=provider.forward_many(model,tokens,parents,caches,(0,1),cached=False,configured_limit=batch)
    mx.eval(logits,taps)
    assert shapes==[(batch,4)] and txn.physical_packed and taps.shape==(batch,4,112)
    for lane in range(batch):
        for row,path in enumerate([[0],[0,1],[0,2],[0,2,3]]):
            cache=copy.deepcopy(refs[lane])
            for node in path: expected=model(mx.array([[tokens[lane][node]]]),cache)
            np.testing.assert_allclose(np.asarray(logits[lane,row]),np.asarray(expected[0,-1]),atol=3e-5,rtol=3e-5)
    assert provider.receipt()['last_memory_estimate']['padded_kv_bytes']==0
    assert provider.receipt()['shared_kv_rows']==rows
    txn.abort()
    assert [[item.offset for item in cache] for cache in caches]==[[n]*2 for n in prefixes]

@pytest.mark.parametrize('rows',[1,4])
def test_segmented_commit_paths_and_lane_content_isolation(rows):
    model=model_fixture(); a=prefill(model,4,1); b=prefill(model,7,2); different=prefill(model,7,23)
    provider=NanbeigePackedTreeExecutor.create(segmented_attention=True,shared_kv_rows=rows)
    inputs=[[8,9,10,11],[12,13,14,15]]; parents=[[-1,0,0,2]]*2
    outputs=[]
    for peer in (b,different):
        caches=copy.deepcopy([a,peer]); logits,taps,txn=provider.forward_many(model,inputs,parents,caches,(1,),cached=False,configured_limit=2)
        mx.eval(logits,taps); outputs.append(np.asarray(logits[0]).copy())
        txn.commit_paths([[0,2,3],[0]])
        for lane,path in enumerate([[0,2,3],[0]]):
            reference=copy.deepcopy([a,peer][lane])
            for row in path: mx.eval(model(mx.array([[inputs[lane][row]]]),reference))
            expected=model(mx.array([[20]]),reference); actual=model(mx.array([[20]]),caches[lane])
            np.testing.assert_allclose(np.asarray(actual),np.asarray(expected),atol=3e-5,rtol=3e-5)
    np.testing.assert_array_equal(outputs[0],outputs[1])


def test_segmented_budget_prices_scratch_not_padding_before_mutation(monkeypatch):
    estimate=estimate_temporary_memory(prefixes=[524288,1,2,3],width=32,layer_geometry=[(8,128,128,2,2)]*64,query_heads=56,implicit_attention=True,segmented_attention=True)
    assert estimate['padded_kv_bytes']==0 and estimate['attention_scratch_bytes']>2<<30
    model=model_fixture(); caches=[prefill(model,3,0),prefill(model,5,1)]
    provider=NanbeigePackedTreeExecutor.create(segmented_attention=True,temporary_budget_bytes=1)
    monkeypatch.setattr(model,'forward_with_taps',lambda *a,**kw:pytest.fail('forward ran before budget refusal'))
    with pytest.raises(MemoryError): provider.forward_many(model,[[1,2]]*2,[[-1,0]]*2,caches,(1,),cached=False,configured_limit=2)
    assert [cache[0].offset for cache in caches]==[3,5]


def test_segmented_refuses_quantized_operands_and_invalid_rows():
    q=mx.zeros((1,56,2,128)); k=mx.zeros((1,8,2,128),dtype=mx.uint32)
    with pytest.raises(ValueError): attend_segments(q,[k],[k],prefixes=[0],ancestors=mx.ones((1,2,2),dtype=mx.bool_),scale=128**-.5)
    with pytest.raises(ValueError): NanbeigePackedTreeExecutor.create(shared_kv_rows=4)
