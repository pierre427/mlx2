"""Prompt-only NAX factory selection; no GPU kernel execution."""
import numpy as np
import pytest
import mlx.core as mx
mx.set_default_device(mx.cpu)
from mlx2.runtime.gguf.reader import GGUFTensor,GGUFError,FORMATS
from mlx2.runtime.gguf.weights import GGUFWeights
from mlx2.runtime.gguf.prefill_scope import dense_prefill_context,dense_prefill_active


def weights(kind,*,compute='reference',native=False):
    result=GGUFWeights.__new__(GGUFWeights)
    block,size=FORMATS[kind][1:] if not native else (1,4 if kind==0 else 2)
    tensor=GGUFTensor('weight',(256,64),kind,0,64*256//block*size)
    result.tensors={'weight':tensor};result.compute_policy=compute
    result.dense_prefill_policy='nax_fp16';result.row_policy='reference';result._grouped={}
    result.scheduled_dense_calls={};result.scheduled_grouped_calls={}
    result._dense={'weight':mx.ones((64,256),dtype={0:mx.float32,1:mx.float16,30:mx.bfloat16}[kind])} if native else {}
    class Matrix:
        data=mx.zeros((tensor.nbytes,),dtype=mx.uint8)
        shape=tensor.numpy_shape;ggml_type=tensor.ggml_type  # PackedMatrix's fields
        def __call__(self,x):return mx.ones((*x.shape[:-1],64),dtype=mx.float32)
    result._packed={'weight':Matrix()}
    return result


@pytest.mark.parametrize('kind',[8,10,16])
@pytest.mark.parametrize('compute',['reference','cooperative'])
def test_dense_dispatch_requires_explicit_scope_width_and_policy(kind,compute,monkeypatch):
    from mlx2.runtime.gguf import nax_metal,cooperative_metal
    calls=[]
    def nax(x,w):calls.append(('nax',x.shape));return mx.zeros((x.shape[0],64))
    def cooperative(x,w,rows):calls.append(('cooperative',x.shape));return mx.ones((x.shape[0],64))
    monkeypatch.setattr(nax_metal,'nax_prefill',nax)
    monkeypatch.setattr(cooperative_metal,'cooperative_dense',cooperative)
    w=weights(kind,compute=compute);project=w.linear('weight');x=mx.ones((1,16,256))
    assert project(x).shape==(1,16,64) and not any(c[0]=='nax' for c in calls)
    with dense_prefill_context('nax_fp16',tokens=16):
        assert project(x).shape==(1,16,64) and calls[-1]==('nax',(16,256))
        before=sum(c[0]=='nax' for c in calls)
        project(x[:,:15]);assert sum(c[0]=='nax' for c in calls)==before
    assert w.scheduled_dense_calls['nax_prefill']==1
    with dense_prefill_context('nax_fp16',tokens=16):
        w.dense_prefill_policy='reference';project=w.linear('weight');project(x)
    assert sum(c[0]=='nax' for c in calls)==1


@pytest.mark.parametrize('kind',[0,1,30])
def test_native_float_dense_and_grouped_are_never_reinterpreted(kind,monkeypatch):
    from mlx2.runtime.gguf import nax_metal
    monkeypatch.setattr(nax_metal,'nax_prefill',lambda *a:pytest.fail('native NAX'))
    monkeypatch.setattr(nax_metal,'nax_grouped_prefill',lambda *a:pytest.fail('native grouped NAX'))
    w=weights(kind,native=True)
    with dense_prefill_context('nax_fp16',tokens=16):
        a=w.linear('weight')(mx.ones((1,16,256),mx.float16))
        b=w.grouped(mx.ones((1,16,2,256),mx.float16),'weight',groups=2)
    np.testing.assert_array_equal(np.array(a),np.full((1,16,64),256))
    np.testing.assert_array_equal(np.array(b),np.full((1,16,2,32),256))
    assert w.scheduled_dense_calls==w.scheduled_grouped_calls=={}


@pytest.mark.parametrize('kind',[8,10,16])
def test_grouped_dispatch_counts_leading_tokens_and_preserves_axes(kind,monkeypatch):
    from mlx2.runtime.gguf import nax_metal,cooperative_metal
    from mlx2.runtime.gguf import weights as module
    calls=[]
    def output(tag):
        def run(x,w):calls.append((tag,x.shape,w.shape));return mx.zeros((*x.shape[:-1],32))
        return run
    monkeypatch.setattr(nax_metal,'nax_grouped_prefill',output('nax'))
    monkeypatch.setattr(cooperative_metal,'cooperative_grouped',output('cooperative'))
    monkeypatch.setattr(module,'grouped_linear',output('reference'))
    w=weights(kind,compute='cooperative');x=mx.ones((1,16,2,256))
    w.grouped(x,'weight',groups=2);assert calls[-1][0]!='nax'
    with dense_prefill_context('nax_fp16',tokens=16):
        assert w.grouped(x,'weight',groups=2).shape==(1,16,2,32)
        assert calls[-1]==('nax',(1,16,2,256),(2,32,256))
        w.grouped(x[:,:15],'weight',groups=2);assert calls[-1][0]!='nax'
    assert w.scheduled_grouped_calls['nax_grouped_prefill']==1


def test_dense_scope_nested_exception_safe_and_constructor_fail_closed():
    assert not dense_prefill_active(16)
    with dense_prefill_context('nax_fp16',tokens=16):
        assert dense_prefill_active(16) and not dense_prefill_active(32)
        with pytest.raises(RuntimeError):
            with dense_prefill_context('reference',tokens=16):
                assert not dense_prefill_active(16)
                raise RuntimeError('synthetic')
        assert dense_prefill_active(16)
    assert not dense_prefill_active(16)
    with dense_prefill_context('nax_fp16',tokens=15):assert not dense_prefill_active(15)
    with pytest.raises(GGUFError):GGUFWeights('not-opened',dense_prefill_policy='auto')


def test_model_dense_policy_is_prompt_only_and_cache_identity_bound():
    from test_deepseek_v4_forward_cpu import config,SmallWeights
    from mlx2.runtime.models.deepseek_v4 import DeepSeekV4Model
    class Observed(SmallWeights):
        def __init__(self,c):super().__init__(c);self.scopes=[]
        def moe(self,x,*args):self.scopes.append(dense_prefill_active(x.shape[0]));return super().moe(x,*args)
    c=config((0,));w=Observed(c)
    model=DeepSeekV4Model(c,w,identity='dense',dense_prefill_policy='nax_fp16')
    reference=DeepSeekV4Model(c,w,identity='dense');assert model.identity!=reference.identity
    ids=mx.array(np.arange(16)[None]%c.vocab_size,dtype=mx.int32)
    model(ids,model.make_cache());model.prefill_forward(ids,model.make_cache())
    model.prefill_body(ids,model.make_cache(),(0,));model.forward_with_taps(ids,model.make_cache(),(0,))
    model.prefill_forward(ids[:,:15],model.make_cache());assert w.scopes==[False,True,True,False,False]
    with pytest.raises(ValueError,match='cache identity'):model(ids,reference.make_cache())
    with pytest.raises(ValueError,match='dense prefill'):DeepSeekV4Model(c,w,dense_prefill_policy='auto')
