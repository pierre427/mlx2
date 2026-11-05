"""Owned synthetic GPU arithmetic probes; not model qualification."""
import os

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import fp_verify_tensor as fp

pytestmark=pytest.mark.skipif(os.environ.get('MLX2_RUN_GPU_TESTS')!='1',reason='requires owned GPU unit')


def test_source_direct_reads_and_tuple_projection_on_gpu(tmp_path):
    from mlx2.adapters import deepseek_v41_source_buffers as buffers
    from test_deepseek_v41_source_buffers_cpu import original_file
    from test_apple_four_gaps_cpu import test_engram_tuple_projection_reconstructs_before_query_gate
    from mlx2.runtime.io_ownership import ReadPool
    previous=mx.default_device()
    try:
        mx.set_default_device(mx.gpu)
        artifact,_=original_file(tmp_path)
        host=buffers.read_original_packed(artifact,0,0)
        pool=ReadPool(1,'direct-final-gpu-probe',timeout=10,max_pending=1)
        try:direct=buffers.read_original_device(artifact,0,0,stream=mx.gpu,read_call=pool.call)
        finally:pool.close()
        for name,value in direct.items():
            reference=mx.array(host.views()[name])
            assert bool(mx.all(reference==value).item())
        test_engram_tuple_projection_reconstructs_before_query_gate()
    finally:mx.set_default_device(previous)


@pytest.mark.parametrize('dtype',[mx.float16,mx.bfloat16])
@pytest.mark.parametrize('rows,group,n,kind',[
    (2,8,65,'causal'),(4,16,4099,'causal'),(8,8,4099,'tree_sink'),
    (16,6,4099,'tree_sink'),(16,16,32768,'causal'),(2,8,131072,'causal'),
    (4,1,65,'ring_sink'),(8,6,65,'dead'),
])
def test_tensor_attention_matches_float32_reference(dtype,rows,group,n,kind):
    previous=mx.default_device()
    try:
        mx.set_default_device(mx.gpu)
        from mlx2.runtime.gguf.source_dense_legacy import _device_supported
        if not _device_supported():pytest.skip('M5 TF32-off required')
        mx.random.seed(rows+n+group)
        batch=2 if kind=='ring_sink' else 1
        q=mx.random.normal((batch,2*group,rows,128)).astype(dtype)
        # A sliced cache view exercises preparation on noncontiguous storage.
        k=mx.random.normal((batch,2,n+17,128)).astype(dtype)[:,:,7:7+n]
        v=mx.random.normal((batch,2,n+17,128)).astype(dtype)[:,:,7:7+n]
        mask='causal';sinks=None
        if kind!='causal':
            visible=mx.arange(n)[None,:] <= n-rows+mx.arange(rows)[:,None]
            visible=visible & (mx.arange(n)[None,:]>=min(11,n-rows))
            if kind=='tree_sink':visible=visible & ((mx.arange(n)[None,:]<n-rows)|(mx.arange(n)[None,:]%2==0))
            if kind=='dead':visible=visible & (mx.arange(rows)[:,None]<rows-1)
            mask=visible
            sinks=mx.linspace(-2,2,2*group)
        reference=mx.fast.scaled_dot_product_attention(q.astype(mx.float32),k.astype(mx.float32),v.astype(mx.float32),
            scale=128**-.5,mask=mask,sinks=sinks)
        actual=fp.attention(q,k,v,scale=128**-.5,mask=mask,sinks=sinks)
        mx.eval(reference,actual)
        assert bool(mx.all(mx.isfinite(actual)).item())
        np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)),np.asarray(reference),atol=0.006,rtol=0.012)
        if kind=='dead':assert bool(mx.all(actual[:,:,-1]==0).item())
    finally:mx.set_default_device(previous)
