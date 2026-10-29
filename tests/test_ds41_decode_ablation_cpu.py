"""Bounded hook/state guards only; diagnostic outputs have no quality oracle."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace

import numpy as np
import pytest

from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache

path=Path(__file__).resolve().parents[1]/'scripts/research/ds41_decode_ablation.py'
spec=importlib.util.spec_from_file_location('_decode_ablation_cpu',path);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


@pytest.fixture
def frame(monkeypatch):
    model=SimpleNamespace(layers=[object()for _ in range(40)])
    cache=[]
    for layer in range(40):
        ratio=0 if layer<2 else 2 if layer<20 else 1
        source=layer in(2,8,14,20)
        s=DeepSeekV41LayerCache(ratio,xp=np,identity='diagnosticCPU',source=source,history=layer==0)
        s.offset=99;s.keys=np.zeros((1,1,99,512),np.float32)
        if source:
            s._pool_count=99//ratio;s._capacity=256
            s._pooled=np.zeros((1,256,512),np.float32);s._index_keys=np.zeros((1,256,128),np.float32)
            s.buf_kv=np.zeros((1,99%ratio,512),np.float32);s.buf_gate=s.buf_kv.copy()
        s.validate_state();cache.append(s)
    class Attention:
        def __init__(self,model):self.model=model
        def publish(self,x,state,start):
            rem=state.offset%state.ratio
            state.buf_kv=np.zeros((1,rem,512),np.float32);state.buf_gate=state.buf_kv.copy()
            count=state.offset//state.ratio-state._pool_count
            state.append_pools(np.zeros((1,count,512),np.float32),np.zeros((1,count,128),np.float32))
    module=ModuleType('mlx2.runtime.models.deepseek_v41');module.Attention=Attention
    monkeypatch.setitem(sys.modules,'mlx2.runtime.models.deepseek_v41',module)
    return model,cache,Attention(model)


def execute(frame):
    model,cache,attention=frame
    bands=[]
    for s in cache:
        bands.append(s.update_local(np.ones((1,1,1,512),np.float32)))
        if s.source:attention.publish(None,s,99)
    cache[0].hash_tail=np.array([1,2,3],np.int32)
    for s in cache:s.validate_state()
    return bands


@pytest.mark.parametrize('mode',d.MODES)
def test_modes_safe_shapes_and_actual_scope(frame,mode):
    model,cache,_=frame;before=[s.keys for s in cache];tail=cache[0].hash_tail
    with d.instrument_kv(model,cache,mode=mode)as receipt:
        bands=execute(frame)
        if mode!='baseline':
            with pytest.raises(RuntimeError,match='published'):cache[0].snapshot()
    assert receipt['completed']and receipt['local_calls']==40 and receipt['publish_calls']==4
    assert receipt['diagnostic_only']and not receipt['APCv2_publication']and not receipt['rate_eligible']
    if mode=='freeze_kv_path':
        assert all(s.offset==99 and s.keys is old for s,old in zip(cache,before))
        assert all(b.shape==(1,1,99,512)for b in bands)
        assert cache[0].hash_tail is tail
        assert receipt['source_publish_paths_skipped']==4 and receipt['local_updates_skipped']==40
    else:
        assert all(s.offset==100 for s in cache)
        assert all(b.shape==(1,1,100,512)for b in bands)
        assert receipt['pool_append_calls']==4
    if mode=='skip_local_owned_copy':assert all(np.shares_memory(s.keys,b)for s,b in zip(cache,bands))
    if mode=='baseline':assert not any(np.shares_memory(s.keys,b)for s,b in zip(cache,bands))


def test_failure_restores_freeze_fields_and_all_hooks(frame):
    model,cache,attention=frame;local=DeepSeekV41LayerCache.update_local;publish=type(attention).publish
    with pytest.raises(RuntimeError,match='injected'):
        with d.instrument_kv(model,cache,mode='freeze_kv_path'):
            execute(frame);cache[0]._in_forward=True
            raise RuntimeError('injected')
    assert DeepSeekV41LayerCache.update_local is local and type(attention).publish is publish
    assert not cache[0]._in_forward and cache[0].offset==99
    assert 'update_local'not in vars(DeepSeekV41LayerCache)


def test_foreign_cache_and_attention_are_unmodified(frame):
    model,cache,attention=frame
    foreign=DeepSeekV41LayerCache(0,xp=np,identity='foreign');foreign.offset=1;foreign.keys=np.zeros((1,1,1,512),np.float32)
    with d.instrument_kv(model,cache,mode='freeze_kv_path')as receipt:
        foreign.update_local(np.ones((1,1,1,512),np.float32))
    assert foreign.offset==2 and receipt['local_calls']==0


def test_repeated_state_nonm1_and_nested_context_refused(frame):
    model,cache,_=frame
    with d.instrument_kv(model,cache,mode='freeze_kv_path'):
        cache[0].update_local(np.zeros((1,1,1,512),np.float32))
        with pytest.raises(ValueError,match='repeated'):cache[0].update_local(np.zeros((1,1,1,512),np.float32))
        with pytest.raises(ValueError,match='non-M1'):cache[1].update_local(np.zeros((1,1,2,512),np.float32))
        with pytest.raises(RuntimeError,match='Nested'):
            with d.instrument_kv(model,cache):pass


def test_only_bounded_actual_seed_metadata_admitted(frame):
    model,cache,_=frame
    for mode in ('bad',None,1):
        with pytest.raises(ValueError):
            with d.instrument_kv(model,cache,mode=mode):pass
    with pytest.raises(ValueError):
        with d.instrument_kv(model,cache,input_rows=99):pass
    with pytest.raises(ValueError):
        with d.instrument_kv(model,cache[:-1]):pass
    cache[0].offset=98
    with pytest.raises(ValueError):
        with d.instrument_kv(model,cache):pass


def test_physical_clone_owns_all_planes_and_preserves_pool_capacity(frame):
    model,cache,_=frame
    cloned=d.clone_cache_physical(model,cache)
    assert len(cloned)==40 and all(a is not b for a,b in zip(cloned,cache))
    for a,b in zip(cloned,cache):
        assert a.offset==b.offset and a._capacity==b._capacity and a._pool_count==b._pool_count
        for name in ('keys','buf_kv','buf_gate','_pooled','_index_keys','hash_tail'):
            x,y=getattr(a,name),getattr(b,name)
            assert(x is None)==(y is None)
            if x is not None:
                assert x.shape==y.shape
                assert not np.shares_memory(x,y)
        if a.source:
            assert a._pooled.shape[1]==a._capacity==256
            assert a.pooled.shape[1]==a._pool_count==99//a.ratio


def test_replay_clones_never_mutate_actual_seed_or_shrink_pool_capacity(frame):
    model,cache,attention=frame
    seed=d.clone_cache_physical(model,cache)
    for mode in d.MODES:
        private=d.clone_cache_physical(model,seed)
        with d.instrument_kv(model,private,mode=mode):execute((model,private,attention))
        assert all(s.offset==99 for s in cache+seed)
        assert all(s._capacity==256 for s in private if s.source)
        for s in cache+seed:s.validate_state()
