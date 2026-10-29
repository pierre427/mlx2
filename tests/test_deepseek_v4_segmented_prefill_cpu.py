"""Stable GPU-plan CPU oracle and Metal4 source compilation, no GPU dispatch."""
import shutil
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
from mlx2.runtime.gguf.segmented_prefill import (
    reference_plan,plan_capacity,scratch_bytes,metal_compile_source,_validate,
    expert_prefill_context,segmented_prefill_active,
)
from mlx2.runtime.gguf.reader import GGUFError


def test_prefill_scope_is_width_bound_nested_and_exception_safe():
    assert not segmented_prefill_active(32)
    with expert_prefill_context('segmented_nax_fp16',tokens=32):
        assert segmented_prefill_active(32)
        assert not segmented_prefill_active(16)
        with expert_prefill_context('reference',tokens=32):
            assert not segmented_prefill_active(32)
        assert segmented_prefill_active(32)
        with pytest.raises(RuntimeError):
            with expert_prefill_context('segmented_nax_fp16',tokens=64):
                assert segmented_prefill_active(64)
                raise RuntimeError('bounded synthetic failure')
        assert segmented_prefill_active(32)
    assert not segmented_prefill_active(32)
    with expert_prefill_context('segmented_nax_fp16',tokens=8):
        assert not segmented_prefill_active(8)
    for policy,tokens in [('unknown',32),('reference',0),('reference',True)]:
        with pytest.raises(GGUFError):
            with expert_prefill_context(policy,tokens=tokens):
                pass


@pytest.mark.parametrize('tokens,experts,topk',[(1,6,6),(32,256,6),(128,256,6),(512,256,6),(32,1,1)])
def test_stable_expert_plan_compact_tiles_cover_every_assignment(tokens,experts,topk):
    rng=np.random.default_rng(tokens)
    ids=np.array([rng.choice(experts,topk,replace=False) for _ in range(tokens)],np.uint32)
    plan=reference_plan(ids,experts)
    flat=ids.ravel()
    expected=np.argsort(flat*len(flat)+np.arange(len(flat)),kind='stable')
    np.testing.assert_array_equal(plan['order'],expected)
    np.testing.assert_array_equal(plan['inverse'][plan['order']],np.arange(ids.size))
    seen=[]
    for start,expert,count,_ in plan['tiles']:
        assert 1<=count<=16
        assert np.all(plan['sorted_ids'][start:start+count]==expert)
        seen.extend(range(start,start+count))
    assert seen==list(range(ids.size))
    assert len(plan['tiles'])<=plan['max_tiles']==plan_capacity(ids.size,experts)
    # Equal expert assignments retain original token/slot order.
    for expert in range(experts):
        order=plan['order'][plan['sorted_ids']==expert]
        assert np.all(np.diff(order.astype(np.int64))>0)


def test_inverse_scatter_combine_preserves_original_slot_score_semantics():
    ids=np.array([[2,0,1],[1,2,0]],np.uint32); plan=reference_plan(ids,3)
    original=np.arange(6*4,dtype=np.float32).reshape(6,4)
    sorted_rows=original[plan['order']]
    weights=np.array([[.1,.2,.7],[.4,.5,.1]],np.float32)
    actual=(sorted_rows[plan['inverse']].reshape(2,3,4)*weights[...,None]).sum(1)
    expected=(original.reshape(2,3,4)*weights[...,None]).sum(1)
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('ids,experts',[(np.array([[0,3]],np.int32),3),(np.array([[-1]],np.int32),3),(np.empty((0,6),np.uint32),6),(np.ones((1,1),np.float32),6)])
def test_bad_plan_ids_fail_closed(ids,experts):
    with pytest.raises(GGUFError): reference_plan(ids,experts)


def test_scratch_is_bounded_before_any_gpu_allocation():
    x=np.zeros((512,4096),np.float32); ids=np.zeros((512,6),np.uint32); scores=np.zeros_like(ids,dtype=np.float32)
    gate=SimpleNamespace(shape=(256,2048,4096),ggml_type=16)
    down=SimpleNamespace(shape=(256,4096,2048),ggml_type=10)
    required=scratch_bytes(512,6,2048,4096,256)
    assert 72*(1<<20)<required<74*(1<<20)
    with pytest.raises(GGUFError,match='scratch'):_validate(x,ids,scores,gate,gate,down,required-1)
    assert _validate(x,ids,scores,gate,gate,down,required)==(256,2048,4096)
    with pytest.raises(GGUFError):plan_capacity(2**24,256)


@pytest.mark.parametrize('kind',['scan','gate_up','down','combine'])
def test_metal4_cpu_compiler_accepts_segmented_sources(kind,tmp_path):
    if not shutil.which('xcrun') or subprocess.run(['xcrun','--find','metal'],capture_output=True).returncode:
        pytest.skip('CPU Metal compiler unavailable')
    path=tmp_path/(kind+'.metal');path.write_text(metal_compile_source(kind))
    result=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-c',str(path),'-o',str(tmp_path/(kind+'.air'))],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
