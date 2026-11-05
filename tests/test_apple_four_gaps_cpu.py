"""Synthetic functional oracles only; no artifact qualification or GPU use."""
import copy
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx2.adapters import deepseek_v41_source_buffers as buffers
from mlx2.adapters.deepseek_v41_source_lfu import DecayedLFUIndex, parse_policy
from mlx2.runtime.models import fp_verify_tensor as fp
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from mlx2.runtime.inpass_checkpoints import CaptureRefused
from test_deepseek_v41_engram_discard_cpu import _table
from test_deepseek_v41_source_buffers_cpu import original_file, lfu_fixture
from test_peer_exact_checkpoint_boundaries_cpu import _hybrid_model


def test_engram_carry_has_separate_ownership_and_copies_ids(tmp_path):
    from mlx2.runtime.models.deepseek_v41_engram import EngramRows,file_stamp
    path=tmp_path/'carry.bin';raw=np.zeros((16,264),np.uint8);raw[:,256:]=127;raw.tofile(path)
    table=EngramRows(path,offset=0,rows=16,expected_stat=file_stamp(path.stat()),carry_window=True)
    try:
        a,b=np.array([1,2]),np.array([3,4])
        table.prefetch(a);table.prefetch_next(b);b[:]=7
        assert table.diagnostics()['carry_pending']
        with pytest.raises(ValueError,match='owned'):table.park()
        table.read(a)
        assert table._pending is None and table._carry is not None
        table.read(np.array([3,4]))
        assert table.stats['carry_hits']==1 and table._carry is None
        table.prefetch_next(np.array([5]))
        table.read(np.array([6]))
        assert table.stats['carry_discards']==1
        table.prefetch(np.array([1]));table.prefetch_next(np.array([2]))
        assert table.discard_prefetch()
        assert table._pending is table._carry is None
        table.park();table.activate()
        table.prefetch_next(np.array([4]));table.park()
        assert table._carry is None and table.reserved_bytes()==0
    finally:table.close()


def test_engram_tuple_projection_reconstructs_before_query_gate():
    ids=np.tile(np.arange(24)%8,(12,1));ids[1::2]=(ids[1::2]+1)%8
    rows=(ids[...,None]*np.linspace(0,0.001,256)).astype(np.float32)
    class Table:
        stats={}
        def read(self,values):
            np.testing.assert_array_equal(values,ids)
            return rows
    weights=mx.arange(24*256*6,dtype=mx.float32).reshape(24*256,6)*1e-8
    calls=[]
    model=SimpleNamespace(args=SimpleNamespace(hc_mult=2,hidden_size=2,rms_norm_eps=1e-6),
        engram_tables={0:Table()},weights=SimpleNamespace(array=lambda _:mx.ones((2,))),
        linear=lambda x,_:(calls.append(x.shape) or x@weights),round=lambda x:x)
    h=mx.arange(48,dtype=mx.float32).reshape(1,12,2,2)/100
    reference=DeepSeekV41Model.engram(model,h,0,ids)
    model.engram_carry_enabled=True
    compact=DeepSeekV41Model.engram(model,h,0,ids)
    np.testing.assert_allclose(np.asarray(compact),np.asarray(reference),rtol=1e-6,atol=1e-6)
    assert calls==[(1,12,6144),(1,2,6144)]
    assert model.engram_tables[0].stats['tuple_saved_rows']==10


def test_direct_final_storage_is_exact_and_independent(tmp_path,monkeypatch):
    artifact,_=original_file(tmp_path)
    host=buffers.read_original_packed(artifact,0,0)
    monkeypatch.setattr(buffers,'materialize',lambda *a,**k:pytest.fail('staged upload'))
    result=buffers.read_original_device(artifact,0,0,stream=mx.cpu)
    assert result.has_compact_backing and result.allocation_nbytes==buffers.ORIGINAL_BYTES
    for name,value in result.items():np.testing.assert_array_equal(np.asarray(value),host.views()[name])
    assert np.asarray(memoryview(result.backing)).ctypes.data != host.data.ctypes.data


def test_direct_read_timeout_retains_unpublished_final_owner(tmp_path,monkeypatch):
    store=lfu_fixture(tmp_path,monkeypatch)
    store.policy['direct_reads']=True
    def timeout(function,*args):
        assert (0,0) in store._pending_uploads
        destination=args[3]
        assert destination.flags.writeable and destination.nbytes==buffers.ORIGINAL_BYTES
        raise TimeoutError('bounded direct read')
    monkeypatch.setattr(store.pool,'call',timeout)
    try:
        with pytest.raises(TimeoutError,match='bounded direct read'):
            with store.window():store.acquire('blk.0',0)
        assert (0,0) not in store._cache and (0,0) not in store._leased
        assert store.stats['cache_misses']==0
    finally:store.close()


def test_direct_lfu_lease_and_eviction_do_not_overwrite_retained_storage(tmp_path,monkeypatch):
    store=lfu_fixture(tmp_path,monkeypatch)
    store.policy['direct_reads']=True
    try:
        with store.window():first=store.acquire('blk.0',0)
        before=np.asarray(first.backing).copy()
        with store.window():second=store.acquire('blk.0',1)
        np.testing.assert_array_equal(np.asarray(first.backing),before)
        assert first.backing is not second.backing
        assert store.stats['direct_read_bytes']==2*buffers.ORIGINAL_BYTES
        assert store.stats['packed_upload_bytes']==0
        assert not store._pending_host and not store._pending_uploads
    finally:store.close()


def test_decayed_lfu_adapts_to_topic_switch_and_preserves_atlas_counts():
    index=DecayedLFUIndex(half_life=1)
    index.observe_counts(0,[(0,100)]);index.touch((0,0))
    for _ in range(12):index.observe_counts(0,[(1,2)]);index.touch((0,1))
    resident={(0,0):object(),(0,1):object()}
    assert index.counts[(0,0)]==100 and index.counts[(0,1)]==24
    assert index.victim(resident,{},lambda _:True)==(0,0)
    assert index.victim(resident,{(0,0):None},lambda _:True)==(0,1)
    with pytest.raises(MemoryError):index.victim(resident,resident,lambda _:True)


@pytest.mark.parametrize('options',[{'retention':'bad'},{'direct_reads':1},
    {'retention_half_life':1},{'retention':'decayed_lfu_v1','retention_half_life':True}])
def test_retention_policy_refuses_invalid_options(options):
    with pytest.raises(ValueError):parse_policy({'mode':'original_lfu','budget_bytes':100_000_000,**options})


@pytest.mark.parametrize('rows',[2,4,8,16])
def test_fp_attention_admission_and_default_off(rows):
    q=mx.zeros((1,16,rows,128),mx.bfloat16);k=mx.zeros((1,2,32,128),mx.bfloat16)
    assert fp.supported(q,k,k,'causal',mx.zeros((16,)))
    assert fp.supported(q,k,k,mx.ones((rows,32),mx.bool_))
    assert not fp.supported(q,k,k,mx.zeros((rows,32)))
    assert not fp.supported(q,k,k,sinks=mx.zeros((2,)))
    assert not fp.use_kernel(q,k,k,'causal')


def test_optional_numerics_have_distinct_apcv2_namespaces():
    from mlx2.runtime.apc_numerics import execution_numerics_identity,apc_execution_fingerprint
    base=execution_numerics_identity({})
    attention=execution_numerics_identity({'MLX2_FP_VERIFY_KERNEL':'1'})
    capture=execution_numerics_identity({'MLX2_INPASS_CHECKPOINTS':'1'})
    assert len({str(apc_execution_fingerprint('artifact',v)) for v in (base,attention,capture)})==3


@pytest.mark.parametrize('with_sink',[False,True])
def test_fp_split_merge_matches_full_attention_and_handles_dead_rows(with_sink):
    mx.random.seed(12)
    q=mx.random.normal((1,2,4,8));k=mx.random.normal((1,2,9,8));v=mx.random.normal((1,2,9,8))
    mask=(mx.arange(9)[None,:] <= mx.arange(4)[:,None]+3) & (mx.arange(4)[:,None]<3)
    sinks=mx.array([-2.,2.]) if with_sink else None
    partial=[];maxima=[];sums=[]
    for start,stop in ((0,4),(4,9)):
        scores=(q@mx.swapaxes(k[:,:,start:stop],-1,-2))*8**-.5
        scores=mx.where(mask[:,start:stop],scores,-mx.inf)
        m=mx.max(scores,axis=-1)
        p=mx.where(mx.isfinite(scores),mx.exp(scores-mx.where(mx.isfinite(m),m,0)[...,None]),0)
        partial.append(p@v[:,:,start:stop]);maxima.append(m);sums.append(mx.sum(p,axis=-1))
    actual=fp.merge(mx.stack(partial),mx.stack(maxima),mx.stack(sums),sinks=sinks)
    if with_sink:
        reference=mx.fast.scaled_dot_product_attention(q,k,v,scale=8**-.5,mask=mask,sinks=sinks)
        np.testing.assert_allclose(np.asarray(actual),np.asarray(reference),atol=1e-6,rtol=1e-6)
    else:
        reference=mx.fast.scaled_dot_product_attention(q[:,:,:3],k,v,scale=8**-.5,mask=mask[:3])
        np.testing.assert_allclose(np.asarray(actual[:,:,:3]),np.asarray(reference),atol=1e-6,rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(actual[:,:,-1]),0)


def test_inpass_snapshots_resume_at_unchanged_boundaries_and_are_immutable():
    model=_hybrid_model();tokens=mx.array([[i%100+1 for i in range(12)]])
    cache=model.make_cache()
    output,snapshots,receipt=model.prefill_checkpoint_forward(tokens,cache,(3,7))
    assert receipt['projection_passes']==1 and receipt['captures']==2
    ordinary=model(tokens,model.make_cache())
    np.testing.assert_allclose(np.asarray(output[:,-1]),np.asarray(ordinary[:,-1]),atol=2e-5,rtol=2e-5)
    for position,saved in snapshots.items():
        untouched=copy.deepcopy(saved)
        resumed=model(tokens[:,position:],saved)
        np.testing.assert_allclose(np.asarray(resumed[:,-1]),np.asarray(ordinary[:,-1]),atol=2e-5,rtol=2e-5)
        split=model.make_cache();model(tokens[:,:position],split)
        for actual,reference in zip(untouched,split):
            if hasattr(actual,'keys_and_values'):
                assert actual.keys.shape[2]==position
                astate,bstate=actual.keys_and_values(),reference.keys_and_values()
            else:astate,bstate=actual.state,reference.state
            for (_,a),(_,b) in zip(tree_flatten(astate),tree_flatten(bstate)):
                np.testing.assert_allclose(np.asarray(a),np.asarray(b),atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('positions',[(0,),(12,),(4,3),(3,3),(1,2,3,4,5)])
def test_inpass_refuses_before_mutating_cache(positions):
    model=_hybrid_model();cache=model.make_cache()
    with pytest.raises(CaptureRefused):model.prefill_checkpoint_forward(mx.ones((1,12),mx.uint32),cache,positions)
    assert all(c.empty() for c in cache)


def test_capture_capacity_refuses_excess_retention_before_forward(monkeypatch):
    from mlx2.runtime import inpass_checkpoints as capture
    model=_hybrid_model();cache=model.make_cache()
    monkeypatch.setattr(capture,'MAX_RETAINED_BYTES',1)
    assert model.prefill_checkpoint_capacity(cache,32)==0
    assert all(c.empty() for c in cache)


@pytest.mark.parametrize('defect',['duplicate_owner','frontier','transient'])
def test_capture_prevalidates_all_descriptor_owners(defect):
    model=_hybrid_model();cache=model.make_cache()
    if defect=='duplicate_owner':cache[2]=cache[0]
    elif defect=='frontier':cache[1].offset=1
    else:cache[0]._rollback_position=1
    with pytest.raises(CaptureRefused):model.prefill_checkpoint_forward(mx.ones((1,12),mx.uint32),cache,(3,))
    assert all(c.empty() for c in cache)


def test_inpass_flag_does_not_skip_ordinary_route_boundaries(monkeypatch):
    from collections import deque
    from mlx2.runtime.generate import BatchGenerator
    monkeypatch.setenv('MLX2_INPASS_CHECKPOINTS','1')
    gen=BatchGenerator(_hybrid_model(),completion_batch_size=1,prefill_batch_size=1)
    try:
        gen._interior_checkpoint_positions[0]=deque([3,7])
        assert not gen._inpass_checkpoint_enabled()
        assert gen._peek_interior_checkpoint(0,0)==3
    finally:gen.close()


def test_inpass_mtp_target_predictor_and_seed_are_aligned():
    from mlx2.runtime.hybrid_speculative import advance_self_mtp_prefill,_restore_mtp_state
    from mlx2.runtime.models.qwen38_27b import TextModel,MTPModule
    model=_hybrid_model();model.args.mtp_num_hidden_layers=1;model.mtp=MTPModule(model.args);model.eval()
    checkpoints=[];prompt=mx.array([i%100+1 for i in range(14)])
    rest,target,state,n=advance_self_mtp_prefill(prompt,model,prompt_cache=None,mtp_state=None,
        max_tokens=12,clear_cache=False,capture_positions=(3,7),checkpoint_out=checkpoints)
    assert n==12 and rest.size==2 and len(checkpoints)==2
    for position,checkpoint in checkpoints:
        assert checkpoint['covered_tokens']==position
        _restore_mtp_state(checkpoint['target_cache'],checkpoint['mtp_state'])
        _,ref_target,ref_state,_=advance_self_mtp_prefill(prompt,model,prompt_cache=None,mtp_state=None,
            max_tokens=position,clear_cache=False)
        np.testing.assert_allclose(np.asarray(checkpoint['mtp_state'][1]),np.asarray(ref_state[1]),atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('exact_plan',[False,True])
@pytest.mark.parametrize('cut_only',[False,True])
@pytest.mark.parametrize('prompt_length',[25,73])
def test_scheduler_inpass_captures_every_boundary_including_capture_limit(monkeypatch,exact_plan,cut_only,prompt_length):
    from route_harness import tiny_qwen38_mtp
    from test_rolling_prefill_checkpoints import _drive
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.state_boundaries import StateBoundary,BoundaryPurpose
    monkeypatch.setenv('MLX2_INPASS_CHECKPOINTS','1')
    model,vocab=tiny_qwen38_mtp()
    gen=BatchGenerator(model,completion_batch_size=1,prefill_batch_size=1,prefill_step_size=32,
        self_mtp={'num_draft':2,'persistent':True,'prefill_plan_exact':exact_plan})
    positions=(3,6,9,12,15,18)+((36,48,64) if prompt_length>32 else ())
    try:
        uid=gen.insert([[i%(vocab-2)+1 for i in range(prompt_length)]],max_tokens=[3],
            state_boundaries=[tuple(StateBoundary(p,BoundaryPurpose.INTERIOR) for p in positions)])[0]
        if cut_only:
            from collections import deque
            gen._prefill_cut_positions[uid]=deque([1,5])
        tokens,_,checkpoints=_drive(gen,uid)
        assert len(tokens)==3
        assert [c['covered_tokens'] for c in checkpoints]==list(positions)
        assert gen.scheduler_stats['mtp_inpass_checkpoints_captured']>=4
        assert all(len(c['tokens'])==c['covered_tokens'] for c in checkpoints)
        assert gen.pop_prefill_slice_schedule(uid)['planned']
    finally:gen.close()
