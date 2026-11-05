"""Actual decoder/cache seams for synthetic SC-MXFP4 CPU integration checks.

These tests replace only the Metal entry points with the ordinary CPU QMM.
They prove selection/phase/layout coverage; they do not qualify model quality.
"""
import numpy as np
import mlx.core as mx
from mlx import nn
import pytest

from mlx2.runtime.microscaling import installer as I, kernels as K
from mlx2.runtime.microscaling.format import PackedSCWeight, to_mxfp4_scales_numpy
from mlx2.runtime.models.standard_decoder import Model, ModelArgs
from mlx2.runtime.models.cache import BatchKVCache, KVCache
from mlx2.runtime.segmented_plain_kv import SegmentedBatchKVCache


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(I, 'require_supported_device', lambda: 'CPU synthetic decoder fixture')


def decoder(moe=False, dtype=mx.bfloat16):
    args=ModelArgs(model_type='qwen3_moe' if moe else 'qwen3',hidden_size=128,
        num_hidden_layers=1,intermediate_size=256,num_attention_heads=2,
        rms_norm_eps=1e-6,vocab_size=256,num_key_value_heads=1,
        max_position_embeddings=2048,rope_theta=10000.,tie_word_embeddings=True,
        head_dim=64,num_experts=4 if moe else 0,num_experts_per_tok=2 if moe else 0,
        moe_intermediate_size=256 if moe else 0,norm_topk_prob=moe)
    mx.random.seed(917)
    model=Model(args); model.set_dtype(dtype)
    # Actual owned dense/SwitchGLU conversion paths; router/head remain float.
    nn.quantize(model,group_size=32,bits=4,mode='mxfp4',class_predicate=lambda name,module:
        '.mlp.' in name and name.rsplit('.',1)[-1] != 'gate' and hasattr(module,'to_quantized'))
    return model


def emulate_metal(monkeypatch):
    calls=[]
    def scales(c,b,s):
        shape=(*c.shape[:-1],c.shape[-1]*8)
        packed=PackedSCWeight(np.array(c),np.array(b),np.array(s),shape)
        return mx.array(to_mxfp4_scales_numpy(packed))
    def dense(x,c,b,s):
        calls.append(('dense',x.shape))
        return mx.quantized_matmul(x,c,scales(c,b,s),transpose=True,group_size=32,bits=4,mode='mxfp4')
    def grouped(x,ids,c,b,s,sorted_indices=False):
        calls.append(('grouped',x.shape,ids.shape))
        return mx.gather_qmm(x,c,scales(c,b,s),rhs_indices=ids,transpose=True,
            group_size=32,bits=4,mode='mxfp4',sorted_indices=sorted_indices)
    monkeypatch.setattr(K,'dense',dense);monkeypatch.setattr(K,'grouped',grouped)
    return calls


@pytest.mark.parametrize('moe',[False,True])
@pytest.mark.parametrize('dtype',[mx.bfloat16,mx.float16])
def test_actual_batch_kv_prefill_engages_then_decode_stays_ordinary(monkeypatch,moe,dtype):
    model=decoder(moe,dtype); calls=emulate_metal(monkeypatch)
    original=type(model)
    handle=I.apply(model,{'enabled':True,'scope':'experts' if moe else 'mlp','row_threshold':8},
        artifact_fingerprint='synthetic-standard-sc32@recipe:917',max_decode_rows=4)
    handle.configure_workspace_bound(32,expert_multiplier=2 if moe else 1)
    cache=[BatchKVCache([0,0])]
    x=mx.arange(32).reshape(2,16)%256
    y=model(x,cache=cache);mx.eval(y)
    assert calls and all(c[0]==('grouped' if moe else 'dense') for c in calls)
    assert handle.status()['engaged']
    before=len(calls);model(mx.ones((2,1),mx.int32),cache=cache)
    assert len(calls)==before
    assert I.remove(handle) and type(model) is original
    y=model(mx.ones((2,1),mx.int32),cache=cache);mx.eval(y)
    assert len(calls)==before


@pytest.mark.parametrize('moe',[False,True])
def test_segmented_long_prefill_engages_and_padded_decode_declines(monkeypatch,moe):
    model=decoder(moe);calls=emulate_metal(monkeypatch)
    handle=I.apply(model,{'enabled':True,'scope':'experts' if moe else 'mlp','row_threshold':8},
        artifact_fingerprint='synthetic-standard-sc32@recipe:917',max_decode_rows=4)
    cache=[SegmentedBatchKVCache([KVCache(),KVCache()])]
    cache[0].prepare(lengths=[16,12])
    y=model(mx.arange(32).reshape(2,16)%256,cache=cache);mx.eval(y)
    assert calls
    cache[0].finalize();before=len(calls)
    cache[0].prepare(lengths=[1,0]);y=model(mx.ones((2,1),mx.int32),cache=cache);mx.eval(y)
    assert len(calls)==before
    I.remove(handle)


def test_workspace_reserves_grouped_temporaries_and_refuses_unadmitted_rows(monkeypatch):
    model=decoder(True);emulate_metal(monkeypatch)
    handle=I.apply(model,{'enabled':True,'scope':'experts','row_threshold':8},
        artifact_fingerprint='synthetic-standard-sc32@recipe:917',max_decode_rows=4)
    handle.configure_workspace_bound(16,expert_multiplier=2)
    assert handle.weight_copy_bytes()==0
    assert handle.workspace_reserve_bytes()>3*32*128
    assert handle.unfilled_reserve_bytes()==handle.metadata_bytes()+handle.workspace_reserve_bytes()
    with pytest.raises(I.MicroscalingError,match='workspace bound'):
        model(mx.ones((1,32),mx.int32))
    assert handle._prefill.get() is None
    I.remove(handle)
    assert handle.unfilled_reserve_bytes()==0
