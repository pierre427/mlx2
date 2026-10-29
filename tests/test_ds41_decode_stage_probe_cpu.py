"""Small CPU guards for private timing cuts; no numerical-quality or GPU gate."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('_decode_stage_cpu',ROOT/'scripts/research/ds41_decode_stage_probe.py')
h=importlib.util.module_from_spec(s);s.loader.exec_module(h)


class Array:
    def __init__(self,v):self.a=np.asarray(v,np.float32)
    @property
    def nbytes(self):return self.a.nbytes


@pytest.fixture
def fake(monkeypatch):
    events=[];mx=ModuleType('mlx.core');mx.array=Array
    mx.asarray=lambda x,copy:Array(x.a.copy())
    mx.eval=lambda *x:events.append(('eval',len(x)))
    mlx=ModuleType('mlx');mlx.core=mx
    monkeypatch.setitem(sys.modules,'mlx',mlx);monkeypatch.setitem(sys.modules,'mlx.core',mx)
    module=ModuleType('mlx2.runtime.models.deepseek_v41')
    monkeypatch.setitem(sys.modules,module.__name__,module)
    class Attention:
        def __init__(self,model):self.model=model;self.calls=0
        def __call__(self,x,*args):self.calls+=1;return Array(x.a+1)
    class Weights:
        def __init__(self):self.calls=0
        def moe(self,x):self.calls+=1;return Array(x.a+2)
        def linear(self,name):return lambda x:Array(x.a+3)
        def grouped(self,x):return x
        def paired_project(self,x):return(x,x)
    class Block:
        def __init__(self,model,layer):self.model=model;self.layer=layer;self.attn=Attention(model)
        def mixes(self,x,kind):return(x,x,x)
        def finish(self,x):return x
        def shared_expert(self,x):return x
        def router(self,x):return(x,x)
        def __call__(self,x):
            self.mixes(x,'attn');x=self.attn(x);x=self.model.weights.moe(x);return self.finish(x)
    class Model:
        identity='DS41-CPU-FIXTURE@'+'f'*64
        def __init__(self):self.weights=Weights();self.layers=[Block(self,i)for i in range(2)]
        def norm(self,x,*a):return x
        def qat(self,x,*a):return x
        def collapse(self,x,*a):return x
        def expand(self,x,*a):return x
        def run(self):
            x=Array([0,1])
            for block in self.layers:x=block(x)
            return x
    return Model(),mx,events


def test_owned_bank_bound_and_model_frontier_identity(fake):
    model,mx,events=fake;b=h.ProbeBank(8);b.bind(model,100)
    a=Array([1,2]);b.store(('a',),a,mx,mx.eval);a.a[0]=99
    assert b.get(('a',)).a.tolist()==[1,2]
    with pytest.raises(MemoryError):b.store(('b',),a,mx,mx.eval)
    with pytest.raises(ValueError):b.bind(model,101)
    with pytest.raises(ValueError):b.get(('missing',))
    b.clear();assert not b.records and b.bytes==0


def test_capture_then_cuts_no_fine_sync_and_restoration(fake,tmp_path):
    model,mx,events=fake;bank=h.ProbeBank()
    original=model.weights.moe
    with h.instrument(model,tmp_path,'capture',0,input_rows=1,frontier=100,mode='capture',bank=bank):model.run()
    captured_evals=len(events);assert captured_evals>0
    with h.instrument(model,tmp_path,'base',1,input_rows=1,frontier=100,mode='baseline',bank=bank):model.run()
    assert len(events)==captured_evals
    old_attn=[b.attn.calls for b in model.layers];old_moe=model.weights.calls
    with h.instrument(model,tmp_path,'cut',2,input_rows=1,frontier=100,mode='bypass_attention_moe',bank=bank)as receipt:model.run()
    assert [b.attn.calls for b in model.layers]==old_attn and model.weights.calls==old_moe
    assert receipt['bypassed']=={'attention':2,'moe':2}and not receipt['APCv2_publication_allowed']
    assert len(events)==captured_evals and'moe'not in vars(model.weights)
    assert model.weights.moe==original and not h._LOCK.locked()


@pytest.mark.parametrize('mode',['bypass_attention','bypass_moe','bypass_dense','bypass_hyperconnections'])
def test_bank_required_and_exception_restores(fake,tmp_path,mode):
    model,mx,events=fake
    with pytest.raises(ValueError):
        with h.instrument(model,tmp_path,'bad',0,input_rows=1,frontier=100,mode=mode):pass
    bank=h.ProbeBank()
    with pytest.raises(KeyboardInterrupt):
        with h.instrument(model,tmp_path,'abort',0,input_rows=1,frontier=100,mode=mode,bank=bank):raise KeyboardInterrupt()
    assert'moe'not in vars(model.weights)and not h._LOCK.locked()


def test_stage_profiler_reports_intervention_input_wait_and_nested_exclusive(fake):
    _,mx,_=fake
    class Clock:
        value=0
        def __call__(self):self.value+=1;return self.value
    p=h._Profiler(mx,mx.eval,Clock())
    p.call('outer','outer',lambda:p.call('child','child',lambda:Array([1]),(),{},0),(),{},0)
    child,outer=p.events
    assert child['inclusive_seconds']==2 and child['input_barrier_seconds']==1
    assert outer['exclusive_seconds']==outer['inclusive_seconds']-child['inclusive_seconds']


def test_non_width_one_or_unknown_mode_refuses(fake,tmp_path):
    model,_,_=fake
    for rows,mode in [(99,'baseline'),(1,'unknown')]:
        with pytest.raises(ValueError):
            with h.instrument(model,tmp_path,'bad',0,input_rows=rows,frontier=100,mode=mode,bank=h.ProbeBank()):pass


def test_metadata_none_scalar_nested_outputs_preserved(fake):
    model,mx,_=fake;b=h.ProbeBank();b.bind(model,100)
    b.store(('tuple',),(Array([1]),None,3,{'tag':'owned'}),mx,mx.eval)
    out=b.get(('tuple',));assert out[1:] == (None,3,{'tag':'owned'})
    with pytest.raises(MemoryError):b.store(('metadata',),(None,3),mx,mx.eval)
