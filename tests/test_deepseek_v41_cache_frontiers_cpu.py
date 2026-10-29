"""Synthetic Source state/APCv2 boundary regressions, no checkpoint/GPU claim.

Transfer from QSA's explicit plane inventory and restore/private-write tests;
exercise existing Source snapshot ownership, never invent trim semantics.
"""
import copy
import mlx.core as mx
import numpy as np
import pytest
from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache

FRONTIERS=(99,100,101,127,128,129,131)
IDENTITY='synthetic@source-frontier-contract-v1:reference-policy'


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('GPU forbidden')))
    yield
    mx.set_default_device(old)


def row(xp=np,identity=IDENTITY):
    return DeepSeekV41LayerCache(2,xp=xp,window_size=128,head_dim=4,index_head_dim=2,
        identity=identity,source=True,history=True,compressed_vocab=257)


def _array(xp,value):return xp.array(value)


def advance(state,stop):
    """Synthetic publication driver, not the model's compression equations.

    Distinct deterministic raw value/gate and paired pooled/index values let
    the contract tests detect lost pending groups or swapped owners/planes.
    """
    xp=state.xp
    for n in range(state.offset+1,stop+1):
        state._in_forward=True
        try:
            key=_array(xp,(np.arange(4,dtype=np.float32)+n*16).reshape(1,1,1,4))
            state.update_local(key)
            value=_array(xp,(np.arange(4,dtype=np.float32)+n*8).reshape(1,1,4))
            gate=_array(xp,(np.arange(4,dtype=np.float32)-n*2).reshape(1,1,4))
            values=value if state.buf_kv is None else xp.concatenate((state.buf_kv,value),axis=1)
            gates=gate if state.buf_gate is None else xp.concatenate((state.buf_gate,gate),axis=1)
            usable=values.shape[1]//2*2
            # Values here are synthetic fixtures; Source append/offset/owner
            # contracts are the actual production code exercised below.
            pooled=(values[:,:usable:2]+values[:,1:usable:2])*.5
            indices=(gates[:,:usable:2,:2]+gates[:,1:usable:2,:2])*.5
            state.buf_kv=values[:,usable:];state.buf_gate=gates[:,usable:]
            state.append_pools(pooled,indices)
            state.hash_tail=_array(xp,np.array([(n-j)%257 if n-j>0 else-1 for j in range(3)],np.int32))
        finally:state._in_forward=False
        state.validate_state()
    return state


def assert_planes(a,b):
    assert len(a)==len(b)==6
    for x,y in zip(a,b):
        if x is None or y is None:assert x is y
        else:
            xx,yy=np.array(x),np.array(y)
            assert xx.dtype==yy.dtype and xx.shape==yy.shape
            np.testing.assert_array_equal(xx.view(np.uint32),yy.view(np.uint32))


def assert_frontier(state,offset):
    assert state.offset==offset and not state.pending
    assert state._pool_count==offset//2
    assert state.pooled.shape==(1,offset//2,4)and state.index_keys.shape==(1,offset//2,2)
    assert state.buf_kv.shape==state.buf_gate.shape==(1,offset%2,4)
    assert state.keys.shape==(1,1,min(offset,128),4)
    first=max(1,offset-127)
    expected=(np.arange(first,offset+1,dtype=np.float32)[:,None]*16+np.arange(4,dtype=np.float32)[None,:])[None,None]
    np.testing.assert_array_equal(np.array(state.keys),expected)
    if offset%2:
        np.testing.assert_array_equal(np.array(state.buf_kv),(np.arange(4,dtype=np.float32)+offset*8).reshape(1,1,4))
        np.testing.assert_array_equal(np.array(state.buf_gate),(np.arange(4,dtype=np.float32)-offset*2).reshape(1,1,4))
    assert np.array(state.hash_tail).tolist()==[offset%257,(offset-1)%257,(offset-2)%257]
    state.validate_state()


def test_continuous_99_group_closure_and_128_local_window_boundaries():
    state=row()
    for offset in FRONTIERS:
        advance(state,offset);assert_frontier(state,offset)
        assert not state.pending # A valid pending ratio2 token is publishable.
        snapshot=state.snapshot()
        assert len(snapshot.arrays)==6
        assert state.cache_operations_contract.exact_prefix
        assert state.cache_operations_contract.rollback is None
        assert snapshot.metadata[-1]==str(offset)


@pytest.mark.parametrize('offset',FRONTIERS)
def test_all_six_planes_snapshot_restore_clone_and_private_append_are_isolated(offset):
    state=advance(row(),offset);saved=state.snapshot();sealed=tuple(None if v is None else v.copy()for v in saved.arrays)
    clone=DeepSeekV41LayerCache.from_state(saved.arrays,saved.metadata,xp=np)
    assert clone.meta_state==state.meta_state;assert_planes(clone.state,state.state)
    for original,frozen,copied in zip(state.state,saved.arrays,clone.state):
        if original is not None:
            assert not np.shares_memory(original,frozen)
            assert not np.shares_memory(original,copied)
            assert not np.shares_memory(frozen,copied)
    # Mutating every nonempty clone plane cannot write through the sealed
    # snapshot or live owner, including the pending pair and Engram tail.
    for plane in clone.state:
        if plane is not None and plane.size:plane.flat[0]+=1
    assert_planes(saved.arrays,sealed);assert_planes(state.state,sealed)
    clone.restore(saved);assert_planes(clone.state,sealed)
    advance(clone,offset+1);assert_frontier(clone,offset+1)
    assert state.offset==offset and state.meta_state==saved.metadata
    assert_planes(state.state,sealed);assert_planes(saved.arrays,sealed)
    state.restore(saved);control=DeepSeekV41LayerCache.from_state(saved.arrays,saved.metadata,xp=np)
    advance(state,offset+1);advance(control,offset+1);assert_planes(state.state,control.state)


@pytest.mark.parametrize('bad',['missing_index','wrong_index_width','pooled_ahead','pending_value','pending_gate','wrong_dtype','tail_dtype','tail_range'])
def test_corrupt_or_half_published_state_restore_refuses_without_mutating_owner(bad):
    state=advance(row(),99);saved=state.snapshot();before=state.snapshot();arrays=list(copy.deepcopy(saved.arrays))
    if bad=='missing_index':arrays[4]=None
    if bad=='wrong_index_width':arrays[4]=arrays[4][:,:-1]
    if bad=='pooled_ahead':arrays[3]=np.concatenate((arrays[3],arrays[3][:,:1]),axis=1)
    if bad=='pending_value':arrays[1]=arrays[1][:,:0]
    if bad=='pending_gate':arrays[2]=arrays[2][:,:0]
    if bad=='wrong_dtype':arrays[3]=arrays[3].astype(np.float16)
    if bad=='tail_dtype':arrays[5]=arrays[5].astype(np.int64)
    if bad=='tail_range':arrays[5][0]=257
    malformed=type(saved)(saved.metadata,tuple(arrays))
    with pytest.raises(ValueError):state.restore(malformed)
    assert state.meta_state==before.metadata;assert_planes(state.state,before.arrays)


def test_source_and_index_publication_cannot_be_split_or_advance_wrong_frontier():
    state=advance(row(),99);before=state.snapshot();state.update_local(np.full((1,1,1,4),1600,np.float32))
    assert state.pending
    with pytest.raises(ValueError,match='frontier'):state.snapshot()
    oldpool=state.pooled.copy();oldindex=state.index_keys.copy();oldcount=state._pool_count
    for pooled,index in[(np.ones((1,1,4),np.float32),np.ones((1,0,2),np.float32)),
                         (np.ones((1,0,4),np.float32),np.ones((1,0,2),np.float32))]:
        with pytest.raises(ValueError):state.append_pools(pooled,index)
        assert state._pool_count==oldcount
        np.testing.assert_array_equal(state.pooled,oldpool);np.testing.assert_array_equal(state.index_keys,oldindex)
    state.restore(before);advance(state,100);assert_frontier(state,100)
    assert state.buf_kv.shape[1]==state.buf_gate.shape[1]==0
    advance(state,101);assert_frontier(state,101)
    assert state.buf_kv.shape[1]==state.buf_gate.shape[1]==1


@pytest.mark.parametrize('offset',[99,100,101,127,128,129,131])
def test_real_cpu_apcv2_freeze_disk_restore_private_append_and_close(tmp_path,offset):
    from mlx2.runtime.apc_v2 import inspect_apc_capabilities
    from mlx2.runtime.cow_cache import freeze_prompt_cache,restore_prompt_cache
    from mlx2.runtime.models.cache import save_prompt_cache,load_prompt_cache
    source=advance(row(),offset)
    target=DeepSeekV41LayerCache.from_state(tuple(mx.array(v)if v is not None else None for v in source.state),source.meta_state,xp=mx)
    expected=target.snapshot();assert inspect_apc_capabilities([target])is not None
    file=tmp_path/f'synthetic-source-frontier-{offset}.safetensors'
    save_prompt_cache(str(file),[target]);disk=load_prompt_cache(str(file))
    assert disk[0].meta_state==expected.metadata;assert_planes(disk[0].state,expected.arrays)
    frozen,_=freeze_prompt_cache([target],key='synthetic@frontier-contract',tokens=tuple(range(offset)),cache_type='deepseek_v41')
    branch=restore_prompt_cache(frozen)
    try:
        assert branch[0].meta_state==target.meta_state
        advance(branch[0],offset+1);assert_frontier(branch[0],offset+1)
        assert target.offset==offset;assert_planes(target.state,expected.arrays)
        # A separately restored disk row has an independent private append.
        advance(disk[0],offset+1);assert_planes(disk[0].state,branch[0].state)
    finally:branch.close();frozen.close()
    assert target.offset==offset;assert_planes(target.state,expected.arrays)


def test_changed_selected_source_policy_identity_refuses_populated_frontier():
    from test_deepseek_v41_cpu import config,Weights
    from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
    c=config(activation_dtype='bfloat16');weights=Weights(c)
    ordinary=DeepSeekV41Model(c,weights,identity='synthetic@Source-policy')
    changed=DeepSeekV41Model(c,weights,identity='synthetic@Source-policy',source_hc_boundary_policy='expand_collapse_seq')
    assert ordinary.identity!=changed.identity
    a=advance(row(identity=ordinary.identity),99);b=advance(row(identity=changed.identity),99);before=b.snapshot()
    with pytest.raises(ValueError,match='identity'):b.restore(a.snapshot())
    assert b.meta_state==before.metadata;assert_planes(b.state,before.arrays)
    with pytest.raises(ValueError,match='identity'):changed(mx.array([[1]],mx.int32),ordinary.make_cache())


def test_in_forward_and_closed_rows_refuse_publication_and_trim_is_not_fabricated():
    state=advance(row(),99);state._in_forward=True
    with pytest.raises(Exception,match='incomplete'):state.snapshot()
    state._in_forward=False
    with pytest.raises(NotImplementedError,match='snapshot restore'):state.trim(1)
    state.close()
    with pytest.raises(Exception):state.snapshot()
