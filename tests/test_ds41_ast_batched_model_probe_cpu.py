"""CPU-only model-probe preparation/publication boundary checks."""
from pathlib import Path
import importlib.util
from types import SimpleNamespace
import mlx.core as mx
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/research/ds41_ast_batched_model_probe.py'
spec=importlib.util.spec_from_file_location('batched_probe',path)
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_probe_refuses_cpu_before_model_loading_or_bank_access(tmp_path,monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw:pytest.fail('GPU forbidden'))
    try:
        with pytest.raises(ValueError,match='parent-owned GPU'):probe.run(None,tmp_path,lambda:True)
    finally:mx.set_default_device(old)


def test_materializes_each_private_descriptor_once_per_node():
    calls=[]
    descriptors=[SimpleNamespace(materialize=lambda node,layer=i:calls.append((node,layer))or(node,layer))for i in range(40)]
    result=probe._materialize(SimpleNamespace(transaction=SimpleNamespace(segments=descriptors)))
    assert len(result)==16 and all(len(row)==40 for row in result)
    assert calls==[(node,layer)for node in range(16)for layer in range(40)]


def test_probe_cannot_admit_or_publish_exact_generic_state():
    source=path.read_text()
    assert '.diagnostic_cache(selected)'in source and '.commit_paths('not in source
    assert "exact_oracle_admitted=False"in source and 'APCv2_publication=False'in source
    assert 'continuation_control=discovery_states[8]'in source


def test_private_profile_alignment_preserves_payload_and_refuses_foreign_marker():
    payload=object();left=SimpleNamespace(keys=payload,ast_private_profile=None)
    right=SimpleNamespace(keys=object(),ast_private_profile='nativeM')
    result=probe._align_private_reference([left],[right],'nativeM')
    assert left.keys is payload and left.ast_private_profile=='nativeM'
    assert result[0]['reference_before']is None and result[0]['payload_modified']is False
    left.ast_private_profile='foreign'
    with pytest.raises(ValueError,match='another private'):probe._align_private_reference([left],[right],'nativeM')


def test_exact_profile_gate_is_explicit_and_cannot_label_native_law_exact():
    assert probe._probe_profile('fully_batched_m1_v1',True)=='source_ast_m1_batch_axis_segments_candidate_v1'
    with pytest.raises(ValueError,match='explicit M1'):probe._probe_profile('fully_batched_v1',True)


def test_first_divergent_node_cache_layer_is_recorded_without_metadata_false_positive():
    good=dict(node=0,exact=True)
    bad=dict(node=1,exact=False,logits={'equal':False},features={'equal':True},
        identity_topology_frontier_exact=True,
        cache={'layers':[dict(layer=0,capacity_equal=True,planes={'keys':{'equal':True}}),
                         dict(layer=3,capacity_equal=True,planes={'keys':{'equal':False},'_pooled':{'equal':True}})]})
    result=probe._first_ordinary_divergence([good,bad])
    assert result['node']==1 and result['first_divergent_cache_layer']=={'layer':3,'planes':['keys'],'capacity_equal':True}
    assert probe._first_ordinary_divergence([good])is None


def test_explicit_long_prefix_preserves_default_and_target_ancestry_capacity():
    class Tokenizer:
        def encode(self,text,**kwargs):return [4,5,6]if text=='rendered'else[7,8]
    adapter=SimpleNamespace(tokenizer=Tokenizer(),render_prompt=lambda request:'rendered',model=SimpleNamespace(context_limit=20))
    assert probe._prompt_ids(adapter)==[4,5,6]
    assert probe._prompt_ids(adapter,9)==[7,8,7,8,7,8,4,5,6]
    with pytest.raises(ValueError,match='ancestry capacity'):probe._prompt_ids(adapter,15)
    with pytest.raises(ValueError):probe._prompt_ids(adapter,True)


def test_kv_preparation_policy_is_reversible_and_refuses_active_transaction():
    model=SimpleNamespace()
    with probe._preparation_policy(model,'descendant_reuse_v1'):
        assert model.source_ast_kv_preparation_policy=='descendant_reuse_v1'
    assert not hasattr(model,'source_ast_kv_preparation_policy')
    model.source_ast_kv_preparation_policy='reference'
    with pytest.raises(RuntimeError):
        with probe._preparation_policy(model,'descendant_reuse_v1'):raise RuntimeError('injected')
    assert model.source_ast_kv_preparation_policy=='reference'
    model._mlx2_ds41_ast_active=object()
    with pytest.raises(ValueError,match='active AST'):
        with probe._preparation_policy(model,'reference'):pass


def test_preparation_receipt_requires_actual_counts_and_separates_traffic_claim():
    metrics=dict(policy='descendant_reuse_v1',builds=4,hits=8,built_payload_bytes=1024,measured_device_traffic=False)
    assert probe._preparation_metrics({'selected_kv_preparation':metrics},'descendant_reuse_v1')==metrics
    for changed in({'hits':-1},{'hits':True},{'measured_device_traffic':True},{'policy':'reference'}):
        with pytest.raises(ValueError):probe._preparation_metrics({'selected_kv_preparation':{**metrics,**changed}},'descendant_reuse_v1')


def test_one_abba_sequence_has_exactly_two_timings_per_arm_after_two_warmups():
    assert probe._preparation_order(False)==('reference','descendant_reuse_v1')
    assert probe._preparation_order(True)==('reference','descendant_reuse_v1','descendant_reuse_v1','reference')
    with pytest.raises(ValueError,match='boolean'):probe._preparation_order(2)


def test_bounded_prefill_body_taps_have_contiguous_offsets_and_last_B1():
    import numpy as np
    calls=[];cache=[SimpleNamespace(offset=0,state=[])for _ in range(2)]
    class XP:
        uint32=np.uint32
        array=staticmethod(np.array)
        concatenate=staticmethod(np.concatenate)
        eval=staticmethod(lambda *a:None)
    class Model:
        def body(self,inputs,cache,layers,only):
            start=cache[0].offset;rows=inputs.shape[1];calls.append((start,rows,only,tuple(layers)))
            for row in cache:row.offset+=rows
            taps=np.broadcast_to(np.arange(start,start+rows)[None,:,None],(1,rows,6)).copy()
            return taps
        def prefill_body(self,inputs,cache,layers):return self.body(inputs,cache,layers,True)
        def forward_with_taps(self,inputs,cache,layers):return np.ones((1,1,4)),self.body(inputs,cache,layers,False)
    pauses=[];pacing=[]
    logits,taps,receipt=probe._prefill_prefix(Model(),list(range(4096)),cache,(37,38,39),lambda:True,chunk_size=128,xp=XP,pause_seconds=1,pacing_log=pacing,sleep=pauses.append)
    assert pauses==[1]*32 and len(pacing)==32 and all(not p['inside_resident_timer']for p in pacing)
    assert all(rows<=128 for _,rows,_,_ in calls)
    assert calls[-1]==(4095,1,False,(37,38,39))and all(c[2]for c in calls[:-1])
    assert taps.shape==(1,4096,6)and taps[0,:,0].tolist()==list(range(4096))
    assert[(p['start'],p['stop'])for p in receipt['spans']]==[(start,start+rows)for start,rows,_,_ in calls]
    assert receipt['profile']=='ordinary_body_chunks_then_B1_final'and not receipt['partition_equivalence_to_whole_prefix_claim']


def test_prefix_row_bound_refuses_whole4K_before_model_access():
    cache=[SimpleNamespace(offset=0,state=[])]
    with pytest.raises(ValueError,match='1024row advice'):probe._prefill_prefix(object(),[1]*4096,cache,(37,38,39),lambda:True)


def test_setup_pacing_keeps_hard_guard_before_and_after_sleep():
    calls=[];rows=[]
    probe._setup_pause(lambda:calls.append('guard'),1,rows,'body',sleep=lambda seconds:calls.append(('sleep',seconds)))
    assert calls==['guard',('sleep',1),'guard']and rows[0]['inside_resident_timer']is False
    with pytest.raises(RuntimeError):probe._setup_pause(lambda:(_ for _ in ()).throw(RuntimeError('hardthermal')),1,[], 'body',sleep=lambda seconds:pytest.fail('guard waived'))
