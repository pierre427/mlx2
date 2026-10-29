"""Independent synthetic oracle; never dispatches GPU work."""
import shutil
import subprocess
import numpy as np
import pytest
from mlx2.runtime.models.deepseek_v4_rope_qat import reference_rope_fp8, compilation_source, rope_fp8


@pytest.mark.parametrize('positions', [[0], [3, 129, 4096], [0, 4, 128, 262144]])
def test_reference_matches_existing_fp32_graph(positions):
    import mlx.core as mx
    from mlx2.runtime.models.deepseek_v4 import Rotary, simulate_fp8_kv
    old=mx.default_device(); mx.set_default_device(mx.cpu)
    try:
        rope=Rotary(64,10000,{'type':'yarn','original_max_position_embeddings':65536,'factor':16})
        rng=np.random.default_rng(42)
        values=rng.normal(size=(1,2,len(positions),512)).astype(np.float32)
        values[...,:448]*=np.float32(2**-8)
        inv=np.array(rope.inv)
        expected=reference_rope_fp8(values,inv,positions)
        actual=np.array(simulate_fp8_kv(rope(mx.array(values),positions=positions)))
        np.testing.assert_array_equal(actual[...,:448],expected[...,:448])
        np.testing.assert_allclose(actual[...,448:],expected[...,448:],rtol=2e-6,atol=2e-6)
    finally: mx.set_default_device(old)


def test_all_fp8_midpoints_nearest_even_and_signed_zero():
    import mlx.core as mx
    from mlx2.runtime.models.deepseek_v4 import simulate_fp8_kv
    old=mx.default_device(); mx.set_default_device(mx.cpu)
    try:
        codes=np.arange(127); exponent=codes>>3; fraction=codes&7
        levels=np.where(exponent==0,fraction/512,(1+fraction/8)*np.exp2(exponent-7)).astype(np.float32)
        mid=(levels[:-1]+levels[1:])/2
        # Last coordinate of each group forces scale1 without touching midpoint.
        values=np.zeros((1,len(mid),512),np.float32)
        groups=values[...,:448].reshape(1,len(mid),7,64)
        groups[...,0]=mid[None,:,None]; groups[...,1]=-mid[None,:,None]
        groups[...,2]=np.float32(-0.); groups[...,-1]=448
        out=reference_rope_fp8(values,np.zeros(32,np.float32),np.zeros(len(mid),np.float32))
        actual=np.array(simulate_fp8_kv(mx.array(values)))
        np.testing.assert_array_equal(out[...,:448].view(np.uint32),actual[...,:448].view(np.uint32))
        selected=np.where(np.arange(126)%2==0,levels[:-1],levels[1:])
        np.testing.assert_array_equal(out[0,:,0],selected)
    finally: mx.set_default_device(old)


def test_cpu_refusal_and_invalid_dtype():
    import mlx.core as mx
    old=mx.default_device();mx.set_default_device(mx.cpu)
    try:
        with pytest.raises(ValueError,match='GPU'):
            rope_fp8(mx.zeros((1,1,512)),mx.ones(32),mx.zeros(1))
        with pytest.raises(ValueError,match='FP32'):
            rope_fp8(mx.zeros((1,1,512),mx.float16),mx.ones(32),mx.zeros(1))
    finally:mx.set_default_device(old)


def test_offline_metal_compilation(tmp_path):
    if shutil.which('xcrun') is None: pytest.skip('Apple offline compiler unavailable')
    source=tmp_path/'rope_qat.metal'; source.write_text(compilation_source())
    result=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.2','-c',str(source),'-o',str(tmp_path/'rope_qat.air')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_model_policy_identity_geometry_and_dispatch(monkeypatch):
    import dataclasses
    import mlx.core as mx
    from test_deepseek_v4_forward_cpu import config,SmallWeights
    from mlx2.runtime.models.deepseek_v4 import DeepSeekV4Model
    from mlx2.runtime.models import deepseek_v4_rope_qat as module
    old=mx.default_device();mx.set_default_device(mx.cpu)
    try:
        small=config((4,128))
        with pytest.raises(ValueError,match='mandatory QAT'):
            DeepSeekV4Model(small,None,rope_qat_policy='fused_fp32')
        with pytest.raises(ValueError,match='unknown'):
            DeepSeekV4Model(small,None,rope_qat_policy='auto')
        c=dataclasses.replace(small,head_dim=512,qk_rope_head_dim=64,index_head_dim=128,simulate_quantization=True)
        weights=SmallWeights(c)
        baseline=DeepSeekV4Model(c,weights,identity='synthetic-rope')
        candidate=DeepSeekV4Model(c,weights,identity='synthetic-rope',rope_qat_policy='fused_fp32')
        assert baseline.identity=='synthetic-rope'
        assert candidate.identity=='synthetic-rope:rope_qat=fused_fp32'
        calls=[]
        def cpu_oracle(value,inv,positions):
            calls.append(tuple(np.array(positions).tolist()))
            return mx.array(reference_rope_fp8(np.array(value),np.array(inv),np.array(positions)))
        monkeypatch.setattr(module,'rope_fp8',cpu_oracle)
        ids=mx.array([[1,2,3,4,5]])
        actual=candidate(ids,cache=candidate.make_cache())
        expected=baseline(ids,cache=baseline.make_cache())
        np.testing.assert_allclose(np.array(actual),np.array(expected),atol=2e-6,rtol=2e-6)
        assert calls==[(0.,1.,2.,3.,4.),(0.,),(0.,1.,2.,3.,4.)]
        assert candidate.layers[0].attn.indexer.rope_qat_policy=='reference'
    finally:mx.set_default_device(old)
