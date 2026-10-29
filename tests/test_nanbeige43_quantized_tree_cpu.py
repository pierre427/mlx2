"""Model-free and tiny-Q8-cache AST transaction composition, CPU only."""
import copy
import hashlib
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.nanbeige43_quantized_tree import (
    NanbeigeQ8TreeExecutor,
    clone_q8_cache,
)
from mlx2.runtime.approximate_kv import prompt_cache_is_approximate
from mlx2.runtime.models.cache import (
    QuantizedKVCache,
    load_prompt_cache,
    save_prompt_cache,
)
from mlx2.runtime.models.standard_decoder import Model, ModelArgs


@pytest.fixture(autouse=True)
def binding(record_property):
    record_property('source_revision', subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    record_property('checkpoint','synthetic-nanbeige-q8-tree@sha256:'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    from mlx2.adapters.nanbeige43_quantized_tree import source_identity
    record_property('working_source_sha256', source_identity())
    previous=mx.default_device();mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def tiny():
    mx.random.seed(71)
    model=Model(ModelArgs(model_type='llama',hidden_size=448,intermediate_size=512,num_hidden_layers=2,
        num_attention_heads=7,num_key_value_heads=1,head_dim=64,vocab_size=128,rms_norm_eps=1e-5,
        max_position_embeddings=4096,rope_theta=120000000,tie_word_embeddings=False))
    mx.eval(model.parameters());return model


def prefill(model,length):
    caches=[QuantizedKVCache(group_size=64,bits=8) for _ in model.layers]
    if length: mx.eval(model(mx.array([[i%128 for i in range(length)]]),caches))
    return caches


def live(cache):
    return [[None if triple is None else [np.array(part[..., :item.offset, :]) for part in triple]
             for triple in (item.keys,item.values)] for item in cache]


def assert_same(a,b):
    assert [item.offset for item in a]==[item.offset for item in b]
    for actual,want in zip(live(a),live(b),strict=True):
        for x,y in zip(actual,want,strict=True):
            if y is None: assert x is None
            else:
                for p,q in zip(x,y,strict=True):np.testing.assert_array_equal(p,q)


@pytest.mark.parametrize('lengths',[(0,0),(2,5),(255,257)])
def test_quantized_tree_abort_restores_all_packed_parts_without_dequant(lengths,monkeypatch):
    model=tiny();caches=[prefill(model,n) for n in lengths];before=[clone_q8_cache(row) for row in caches]
    provider=NanbeigeQ8TreeExecutor.create(adapter_fingerprint='synthetic-target@q8')
    original=model.forward_with_taps;shapes=[]
    def counted(inputs,*args,**kwargs):
        shapes.append(inputs.shape);return original(inputs,*args,**kwargs)
    monkeypatch.setattr(model,'forward_with_taps',counted)
    def refuse(*args,**kwargs):raise AssertionError('full-history dequantize called')
    monkeypatch.setattr(mx,'dequantize',refuse)
    logits,taps,txn=provider.forward_many(model,[[1,3,5,7],[2,4,6,8]],[[-1,0,0,2],[-1,0,1,0]],caches,(0,1),cached=False,configured_limit=2)
    mx.eval(logits,taps)
    assert shapes==[(2,4)] and provider.dispatch_metrics(2,txn,cached=False).physical_packed
    assert logits.shape==(2,4,128) and taps.shape==(2,4,896)
    receipt=provider.receipt()
    assert receipt['last_memory_estimate']['packed_temporary_kv_bytes']==0
    assert receipt['historical_kv_concat'] is False and receipt['historical_kv_padding'] is False
    assert receipt['single_batched_attention'] is False
    assert receipt['scheduled_attention_calls']==2*len(model.layers)
    txn.abort()
    for actual,want in zip(caches,before,strict=True):assert_same(actual,want)


def test_noncontiguous_paths_keep_existing_codes_and_metadata_exact():
    model=tiny();caches=[prefill(model,n) for n in (63,65)];before=[clone_q8_cache(row) for row in caches]
    provider=NanbeigeQ8TreeExecutor.create(adapter_fingerprint='synthetic-target@q8')
    parents=[[-1,0,0,2],[-1,0,1,0]]
    logits,taps,txn=provider.forward_many(model,[[1,3,5,7],[2,4,6,8]],parents,caches,(1,),cached=False,configured_limit=2)
    mx.eval(logits,taps)
    staged=[live(row) for row in caches]
    paths=[[0,2,3],[0]];txn.commit_paths(paths)
    for lane,(cache,original,path,start) in enumerate(zip(caches,before,paths,(63,65),strict=True)):
        assert all(item.offset==start+len(path) for item in cache)
        for layer,(actual,old) in enumerate(zip(live(cache),live(original),strict=True)):
            for side in range(2):
                for part in range(3):
                    expected=np.concatenate([old[side][part],staged[lane][layer][side][part][...,start+np.array(path),:]],axis=2)
                    np.testing.assert_array_equal(actual[side][part],expected)
        assert_same(original,before[lane])
    assert txn.closed
    with pytest.raises(RuntimeError):txn.commit_paths(paths)


def test_invalid_later_lane_path_does_not_publish_earlier_lane():
    model=tiny();caches=[prefill(model,n) for n in (3,7)];before=[clone_q8_cache(row) for row in caches]
    p=NanbeigeQ8TreeExecutor.create(adapter_fingerprint='target')
    logits,taps,txn=p.forward_many(model,[[1,2,3,4],[5,6,7,8]],[[-1,0,0,2],[-1,0,0,2]],caches,(1,),cached=False,configured_limit=2);mx.eval(logits,taps)
    with pytest.raises(ValueError):txn.commit_paths([[0,2,3],[0,1,3]])
    txn.abort()
    for a,b in zip(caches,before,strict=True):assert_same(a,b)


def test_forward_exception_restores_owner_and_rope(monkeypatch):
    model=tiny();caches=[prefill(model,5)];before=clone_q8_cache(caches[0]);rope=model.layers[0].self_attn.rope
    original=model.forward_with_taps
    def broken(*args,**kwargs):
        outputs=original(*args,**kwargs);mx.eval(outputs);raise RuntimeError('injected')
    monkeypatch.setattr(model,'forward_with_taps',broken)
    with pytest.raises(RuntimeError,match='injected'):
        NanbeigeQ8TreeExecutor.create(adapter_fingerprint='target').forward(model,[1,2],[-1,0],caches[0],(1,),cached=False)
    assert_same(caches[0],before);assert model.layers[0].self_attn.rope is rope


def test_shallow_descriptor_alias_is_refused_but_cow_clone_is_supported():
    model=tiny();source=prefill(model,5);shallow=[copy.copy(item) for item in source]
    p=NanbeigeQ8TreeExecutor.create(adapter_fingerprint='target')
    with pytest.raises(ValueError,match='mutable packed descriptors'):
        p.forward_many(model,[[1,2],[3,4]],[[-1,0],[-1,0]],[source,shallow],(1,),cached=False,configured_limit=2)
    clone=clone_q8_cache(source);before=clone_q8_cache(source)
    logits,taps,txn=p.forward(model,[1,2],[-1,0],clone,(1,),cached=False);mx.eval(logits,taps);txn.commit_paths([[0,1]])
    assert_same(source,before)


def test_q8_serialization_restore_and_unqualified_publication_classification(tmp_path):
    model=tiny();cache=prefill(model,65);path=tmp_path/'private-q8.safetensors'
    assert prompt_cache_is_approximate(cache)
    save_prompt_cache(path,cache);restored=load_prompt_cache(path);assert_same(cache,restored)
    p=NanbeigeQ8TreeExecutor.create(adapter_fingerprint='target')
    for lane in (cache,restored):
        logits,taps,txn=p.forward(model,[1,2,3],[-1,0,1],lane,(1,),cached=True);mx.eval(logits,taps);txn.commit_paths([[0,1]])
    assert_same(cache,restored)
    assert 'refused' in p.receipt()['apcv2_publication']


def test_revision_and_budget_refuse_before_mutation(monkeypatch):
    import mlx2.adapters.nanbeige43_quantized_tree as qtree
    model=tiny();cache=prefill(model,7);before=clone_q8_cache(cache)
    p=NanbeigeQ8TreeExecutor.create(adapter_fingerprint='target',temporary_budget_bytes=1)
    with pytest.raises(MemoryError):p.forward(model,[1,2],[-1,0],cache,(1,),cached=False)
    assert_same(cache,before)
    p=NanbeigeQ8TreeExecutor.create(adapter_fingerprint='target');monkeypatch.setattr(qtree,'source_identity',lambda:'changed')
    with pytest.raises(RuntimeError,match='source changed'):p.forward(model,[1,2],[-1,0],cache,(1,),cached=False)


def test_only_fresh_operation_owners_can_skip_no_sidecar_replay():
    model=tiny();p=NanbeigeQ8TreeExecutor.create(adapter_fingerprint='target')
    empty=prefill(model,0)
    assert p.preserve_fresh_target_cache(empty,[]) is True
    assert p.preserve_fresh_target_cache(empty,[1]) is False
    assert p.preserve_fresh_target_cache(None,[]) is False
    assert p.preserve_fresh_target_cache([],[]) is False
    assert p.preserve_fresh_target_cache(prefill(model,1),[]) is False
    assert p.preserve_fresh_target_cache(model.make_cache(),[]) is False
    empty[0]=QuantizedKVCache(group_size=64,bits=4)
    assert p.preserve_fresh_target_cache(empty,[]) is False


def test_actual_dflash2_generator_uses_q8_tree_and_keeps_draft_exact():
    from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
    from mlx2.runtime.drafters.dflash2_config import DFlash2Config
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    model=tiny()
    draft=DFlash2DraftModel(DFlash2Config(hidden_size=448,intermediate_size=512,num_hidden_layers=1,
        num_attention_heads=7,num_key_value_heads=1,head_dim=64,vocab_size=128,num_target_layers=2,
        block_size=4,mask_token_id=127,target_layer_ids=[0,1],conv_kernel_size=2,conv_group_size=64,
        selector_rank=16,selector_top_k=4,max_position_embeddings=4096,layer_types=['full_attention']))
    draft.bind(model);mx.eval(draft.parameters())
    provider=NanbeigeQ8TreeExecutor.create(adapter_fingerprint='synthetic-target@q8')
    batch=ExternalDraftBatchGenerator(model,draft_model=draft,binding='synthetic-nanbeige-dflash-q8',
        num_draft=3,completion_batch_size=1,dynamic_singleton_tree=True,dynamic_tree_max_width=1,
        tree_node_budget_by_lanes={1:3},target_tree_executor_provider=provider,memory_headroom=lambda:1<<40)
    cache=prefill(model,0)
    try:
        uid=batch.insert([[1,2,3,4,5]],max_tokens=[12],caches=[cache],sampling_configs=[{'sampling_temp':0.0}])[0]
        assert all(type(item)is QuantizedKVCache for item in batch.lanes[uid].cache)
        outputs=[]
        for _ in range(100):
            _,responses=batch.next()
            outputs.extend(response.token for response in responses)
            if not batch.lanes:break
        assert len(outputs)==12
        assert provider.dispatches>0
        assert batch.scheduler_stats['external_tree_rounds']>0
        assert batch.scheduler_stats['draft_fallbacks']==0
        assert all(type(item)is QuantizedKVCache for item in cache)
        assert not prompt_cache_is_approximate(draft.make_cache())
    finally:batch.close()


def test_open_transaction_refuses_changed_quantization_format():
    model=tiny();cache=prefill(model,5);p=NanbeigeQ8TreeExecutor.create(adapter_fingerprint='target')
    logits,taps,txn=p.forward(model,[1,2],[-1,0],cache,(1,),cached=False);mx.eval(logits,taps)
    cache[0].key_bits=4
    with pytest.raises(RuntimeError,match='format changed'):txn.commit_paths([[0]])
    txn.abort()
    assert cache[0].key_bits == 8
    assert all(item.offset==5 for item in cache)
