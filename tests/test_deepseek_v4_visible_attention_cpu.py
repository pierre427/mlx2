"""CPU oracle/contract/toolchain tests, explicitly no GPU dispatch or qualification."""
import shutil
import subprocess
import numpy as np
import pytest
import mlx.core as mx
mx.set_default_device(mx.cpu)
from mlx2.runtime.models.deepseek_v4_visible_attention import (
    reference_visible_attention, validate_geometry, metal_compile_source,
    self_check, route_status,
)
from mlx2.runtime.models.deepseek_v4 import attend, index_select, DeepSeekV4Model


def inputs(ratio, topk=False):
    rng=np.random.default_rng(207)
    offset=13 if ratio==4 else (509 if ratio==128 else 9)
    q=rng.normal(0,.2,(1,2,3,512)).astype(np.float32)
    kv=rng.normal(0,.3,(1,1,6,512)).astype(np.float32)
    pool=None if ratio==0 else rng.normal(0,.4,(1,4,512)).astype(np.float32)
    idx=np.array([[[3,0,2],[1,3,0],[2,1,3]]],np.uint32) if topk else None
    sinks=np.array([.2,-.7],np.float32)
    return q,kv,pool,idx,sinks,offset


@pytest.mark.parametrize('ratio,topk',[(0,False),(4,False),(4,True),(128,False)])
def test_visible_row_oracle_matches_independent_masked_mlx_reference(ratio,topk):
    q,kv,pool,idx,sinks,offset=inputs(ratio,topk)
    expected=reference_visible_attention(q,kv,pool,idx,sinks,offset=offset,window=4,ratio=ratio)
    query=np.arange(offset,offset+3)[:,None]; key=np.arange(offset+3-6,offset+3)
    localmask=((query>=key)&(query<key+4))[None,None]
    poolmask=None if pool is None else (np.arange(4)[None,:] < ((np.arange(3)+offset+1)//ratio)[:,None])[None,None]
    actual=attend(mx.array(q),mx.array(kv),mx.array(localmask),
        pooled=None if pool is None else mx.array(pool),pool_mask=None if poolmask is None else mx.array(poolmask),
        indices=None if idx is None else mx.array(idx),sinks=mx.array(sinks))
    np.testing.assert_allclose(np.array(actual),expected,rtol=3e-5,atol=1e-6)


def test_topk_preserves_set_ignores_future_pool_and_sink_has_zero_value():
    q=np.zeros((1,1,3,512),np.float32)
    local=np.ones((1,1,6,512),np.float32)
    pool=np.broadcast_to(np.arange(1,5,dtype=np.float32)[None,:,None]*10,(1,4,512))
    indices=np.array([[[3,0,2],[1,3,0],[2,1,3]]],np.uint32)
    # offset13: visible pools3,3,4. Local window is4 rows for each query.
    out=reference_visible_attention(q,local,pool,indices,np.array([np.log(2)],np.float32),offset=13,window=4,ratio=4)
    means=[(4+10+30)/(4+2+2),(4+10+20)/(4+2+2),(4+20+30+40)/(4+3+2)]
    for row,mean in enumerate(means): np.testing.assert_allclose(out[0,0,row],mean,rtol=1e-6)
    permuted=indices[:,:,::-1]
    again=reference_visible_attention(q,local,pool,permuted,np.array([np.log(2)],np.float32),offset=13,window=4,ratio=4)
    np.testing.assert_array_equal(out,again)


def test_extreme_sink_is_stable_and_cannot_leak_a_value():
    q,kv,pool,idx,sinks,offset=inputs(4,True)
    sinks[:]=1000
    out=reference_visible_attention(q,kv,pool,idx,sinks,offset=offset,window=4,ratio=4)
    np.testing.assert_array_equal(out,np.zeros_like(out))


def test_causal_indexer_without_full_mask_preserves_exact_set():
    rng=np.random.default_rng(12)
    q=rng.normal(size=(1,3,7,8)).astype(np.float32)
    pool=rng.normal(size=(1,16,8)).astype(np.float32)
    weights=rng.normal(size=(1,7,3)).astype(np.float32)
    offset=37
    mask=(np.arange(16)[None,:] < ((np.arange(7)+offset+1)//4)[:,None])[None,None]
    explicit=index_select(mx.array(q),mx.array(pool),mx.array(weights),mx.array(mask),5,workspace_bytes=1)
    visible=index_select(mx.array(q),mx.array(pool),mx.array(weights),None,5,workspace_bytes=1,offset=offset)
    np.testing.assert_array_equal(np.sort(np.array(explicit),axis=-1),np.sort(np.array(visible),axis=-1))


def test_candidate_remains_default_off_and_cpu_self_check_refuses_gpu():
    with pytest.raises(RuntimeError,match='owned Metal GPU'): self_check()
    assert route_status()['qualified'] is False
    assert route_status()['synthetic_self_check_passed'] is False
    assert route_status()['scheduled_target_calls']==0
    with pytest.raises(ValueError,match='policy'):
        DeepSeekV4Model(None,None,attention_policy='silently-enable')


@pytest.mark.parametrize('change', ['missing_local','wrong_pool','bad_ratio','bad_indices'])
def test_invalid_geometry_fail_closed(change):
    q,kv,pool,idx,sinks,offset=inputs(4,True)
    if change=='missing_local': kv=kv[:,:,:2]
    if change=='wrong_pool': pool=pool[:,:2]
    if change=='bad_ratio': ratio=5
    else: ratio=4
    if change=='bad_indices': idx=idx[:,:,:0]
    with pytest.raises(ValueError): validate_geometry(q,kv,pool,idx,sinks,offset,4,ratio)


def test_metal_cpu_toolchain_compiles_source(tmp_path):
    if not shutil.which('xcrun') or subprocess.run(['xcrun','--find','metal'],capture_output=True).returncode:
        pytest.skip('Metal CPU compiler unavailable')
    source=tmp_path/'visible.metal'; source.write_text(metal_compile_source())
    result=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.2','-c',str(source),'-o',str(tmp_path/'visible.air')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
