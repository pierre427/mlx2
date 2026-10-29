"""Static HC conversion reuse and identical old-law arithmetic, CPU only."""
from types import SimpleNamespace
import numpy as np
import pytest
import mlx.core as mx
mx.set_default_device(mx.cpu)
from mlx2.runtime.models import deepseek_v4 as native
from test_deepseek_v4_forward_cpu import config,SmallWeights


def old_reference(hc,value):
    c=hc.config;value32=value.astype(mx.float32)
    flat=value32.reshape(*value.shape[:-2],-1)
    normalized=mx.fast.rms_norm(flat,None,c.rms_norm_eps)
    unwrap=lambda x:getattr(x,'value',x)
    mixes=normalized@unwrap(hc.fn).astype(mx.float32).T
    if hc.head:
        pre=mx.sigmoid(mixes*unwrap(hc.scale).astype(mx.float32)+unwrap(hc.base).astype(mx.float32))+c.hc_eps
        return mx.sum(pre[...,None]*value32,axis=-2).astype(value.dtype)
    pre,post,comb=native.sinkhorn_mix(mixes,unwrap(hc.scale),unwrap(hc.base),c.hc_mult,c.hc_sinkhorn_iters,c.hc_eps)
    return mx.sum(pre[...,None]*value32,axis=-2).astype(value.dtype),post,comb


def equal(a,b):
    if isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif a is None or b is None:assert a is b
    else:np.testing.assert_array_equal(np.array(a),np.array(b))


@pytest.mark.parametrize('head',[False,True])
def test_hc_static_fp16_conversions_happen_once_and_old_equations_stay_exact(head):
    c=config((0,));weights=SmallWeights(c);prefix='output_hc' if head else 'blk.0.hc_attn'
    class Counted:
        def __init__(self,value):self.value=value.astype(mx.float16);self.casts=0
        def astype(self,dtype):self.casts+=1;return self.value.astype(dtype)
    raw={kind:Counted(weights.data[prefix+'_'+kind+'.weight']) for kind in ('fn','base','scale')}
    provider=SimpleNamespace(array=lambda name:next(v for k,v in raw.items() if name==prefix+'_'+k+'.weight'))
    hc=native.HyperConnection(c,provider,prefix,head=head)
    assert all(v.casts==1 for v in raw.values())
    assert hc.fn is raw['fn'] and hc.base is raw['base'] and hc.scale is raw['scale']
    mx.eval(hc.static_parameters())
    for rows in (1,6,16):
        value=mx.array(np.random.default_rng(rows).normal(size=(1,rows,4,8)).astype(np.float32))
        actual=hc(value);expected=old_reference(hc,value)
        mx.eval(actual,expected);equal(actual,expected)
    assert all(v.casts==1 for v in raw.values())
    assert all(value.dtype==mx.float32 for value in hc.static_parameters().values())


def test_static_cache_full_model_outputs_and_compressed_state_match_old_reference(monkeypatch):
    c=config((4,128));weights=SmallWeights(c)
    for name,value in list(weights.data.items()):
        if '_hc_' in name or '.hc_' in name:weights.data[name]=value.astype(mx.float16)
    model=native.DeepSeekV4Model(c,weights,identity='hc-static')
    ids=mx.array(np.arange(137)[None]%c.vocab_size,dtype=mx.int32)
    before=model.make_cache()
    with monkeypatch.context() as patch:
        patch.setattr(native.HyperConnection,'__call__',old_reference)
        expected=model(ids,before);mx.eval(expected,[x.state for x in before])
    after=model.make_cache();actual=model(ids,after);mx.eval(actual,[x.state for x in after])
    equal(actual,expected)
    for a,b in zip(before,after):
        assert a.meta_state==b.meta_state;equal(a.state,b.state)


def test_preload_and_eval_materialize_all_static_cache_arrays_before_admission(monkeypatch):
    c=config((4,128))
    class Provider(SmallWeights):
        def __init__(self,c):
            super().__init__(c)
            for name,value in list(self.data.items()):
                if '_hc_' in name or '.hc_' in name:self.data[name]=value.astype(mx.float16)
            self._dense=self.data;self._packed={}
            self.tensors={name:SimpleNamespace(ggml_type=1) for name in self.data}
        def packed(self,name):
            self._packed.setdefault(name,SimpleNamespace(data=self.data[name]));return self._packed[name]
        def prepare_embedding(self,name):self.array(name)
    model=native.DeepSeekV4Model(c,Provider(c),identity='hc-static-preload')
    static=model.parameters()['hc_static_fp32']
    assert len(static)==2*c.num_hidden_layers+1
    expected={id(x) for group in static.values() for x in group.values()}
    raw={id(x) for name,x in model.weights.data.items() if '_hc_' in name or '.hc_' in name}
    assert expected.isdisjoint(raw)
    assert all(x.dtype==mx.float32 for group in static.values() for x in group.values())
    seen=[];original=mx.eval
    def leaves(value):
        if isinstance(value,dict):return [x for v in value.values() for x in leaves(v)]
        if isinstance(value,(tuple,list)):return [x for v in value for x in leaves(v)]
        return [value]
    def evaluate(*args):seen.append({id(x) for x in leaves(args)});return original(*args)
    monkeypatch.setattr(mx,'eval',evaluate)
    model.preload();assert any(expected<=call for call in seen)
    seen.clear();assert model.eval() is model
    assert any(expected<=call for call in seen)


def test_native_projection_casts_are_materialized_and_accounted_before_admission(monkeypatch):
    c=config((0,));w=SmallWeights(c);staged={'static':mx.ones((32,24),dtype=mx.float32)};calls=[]
    w.native_fp32_parameters=lambda:staged
    w.preload_native_linear_fp32=lambda:calls.append('preloaded')
    model=native.DeepSeekV4Model(c,w,identity='native-cast-accounting')
    monkeypatch.setattr(native,'required_weights',lambda _: {})
    assert model.parameters()['native_linear_fp32'] is staged
    original=mx.eval
    def checked(*args):
        assert calls==['preloaded']
        assert args[0]['native_linear_fp32'] is staged
        return original(*args)
    monkeypatch.setattr(mx,'eval',checked)
    model.preload()
    assert calls==['preloaded']


def test_target_rotary_and_promoted_compressor_ape_materialize_before_admission(monkeypatch):
    c=config((0,4,128));w=SmallWeights(c)
    for name,value in list(w.data.items()):
        if name.endswith('_ape.weight'):w.data[name]=value.astype(mx.float16)
    w._dense=w.data;model=native.DeepSeekV4Model(c,w,identity='target-static-attention')
    expected=[];rotaries=[];apes=[]
    for block in model.layers:
        expected.append(block.attn.rope.inv);rotaries.append(block.attn.rope.inv)
        for name in ('compressor','indexer'):
            value=getattr(block.attn,name,None)
            if value is not None:
                expected.extend((value.rope.inv,value.ape));rotaries.append(value.rope.inv);apes.append(value.ape)
                assert value.ape.dtype==mx.float32
                assert w.data[value.prefix+'_ape.weight'].dtype==mx.float16
    assert len(rotaries)==6 and len(apes)==3
    def leaves(value):
        if isinstance(value,dict):return [x for v in value.values() for x in leaves(v)]
        if isinstance(value,(tuple,list)):return [x for v in value for x in leaves(v)]
        return [value]
    actual=model.parameters()['attention_static']
    assert {id(v) for v in expected}=={id(v) for v in leaves(actual)}
    original=mx.eval;seen=[]
    monkeypatch.setattr(native,'required_weights',lambda _: {})
    def evaluate(value):
        seen.append({id(v) for v in leaves(value)})
        assert {id(v) for v in expected}<=seen[-1]
        return original(value)
    monkeypatch.setattr(mx,'eval',evaluate)
    model.preload();assert seen
    # Collection exposes the exact already-owned tensors, not re-created tables.
    assert {id(v) for v in leaves(model.parameters()['attention_static'])}=={id(v) for v in expected}
