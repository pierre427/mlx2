# SPDX-License-Identifier: MIT
"""BF16 feature-owner regression; synthetic model on CPU, no checkpoint claim."""
import mlx.core as mx
import numpy as np
import pytest
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from mlx2.adapters.deepseek_v41_ast_target import execute_tree,_clone,PARENTS
from mlx2.adapters.deepseek_v41_ast_batched_target import execute_batched_tree
from test_deepseek_v41_ast_target_cpu import PATHS
from test_deepseek_v41_cpu import config,Weights


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *args,**kwargs:pytest.fail('GPU forbidden'))
    yield
    mx.set_default_device(old)


@pytest.mark.parametrize('batched',[False,True])
def test_ast_hc_feature_means_keep_ordinary_bf16_owner_store(batched,monkeypatch):
    c=config(activation_dtype='bfloat16')
    model=DeepSeekV41Model(c,Weights(c),identity='synthetic@bf16-ast-tap-owner')
    model.eval_interval=0;cache=model.make_cache()
    model(mx.array([[1,2]],mx.int32),cache)
    mean=mx.mean;raw_taps=[]
    def observed_mean(value,*args,**kwargs):
        output=mean(value,*args,**kwargs)
        if kwargs.get('axis')==-2:raw_taps.append(output)
        return output
    monkeypatch.setattr(mx,'mean',observed_mean)
    if batched:
        result=execute_batched_tree(model,cache,3,PATHS,(2,3,4),cpu_reference=True,
            expert_law='ordinary_m16_v1',forward_law='fully_batched_v1')
    else:
        result=execute_tree(model,cache,3,PATHS,(2,3,4),cpu_reference=True,expert_law='ordinary_m16_v1')
    try:
        mx.eval(result.features)
        actual=np.asarray(result.features)
        mx.eval(raw_taps)
        assert any(not np.array_equal(np.asarray(raw).view(np.uint32),
            np.asarray(model.round(raw)).view(np.uint32))for raw in raw_taps)
        rounded=np.asarray(model.round(result.features))
        # Every committed support feature must obey the ordinary BF16 store,
        # even though its physical transport is float32.
        np.testing.assert_array_equal(actual.view(np.uint32),rounded.view(np.uint32))
        ordinary=[_clone(row)for row in cache]
        logits,features=model._forward_impl(mx.array([[3]],mx.int32),ordinary,
            logits_to_keep=0,capture_layers=(2,3,4))
        mx.eval(features)
        np.testing.assert_array_equal(actual[:,0:1].view(np.uint32),np.asarray(features).view(np.uint32))
        for row in ordinary:row.close()
    finally:
        result.transaction.abort()
        for row in cache:row.close()
