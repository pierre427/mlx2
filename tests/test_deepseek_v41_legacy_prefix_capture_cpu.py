"""Bounded-prefix control contracts only; no GPU/model run."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('prefix_capture_cpu',ROOT/'scripts/research/ds41_legacy_prefix_capture.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)

def test_actual_dense_bounds_fit_two_layer_structural_cap():
    bound=p.prefix_bound(dict(dense_payload_bytes=9846748608,dense_prepared_bound_bytes=5503713280,dense_staging_bound_bytes=1340604416))
    assert bound['total_bytes']<32<<30
    assert bound['parts']['prefix_expert_worst_bytes']==2*384*18800640
    with pytest.raises(MemoryError):p.prefix_bound(dict(dense_payload_bytes=32<<30,dense_prepared_bound_bytes=0,dense_staging_bound_bytes=0))
    with pytest.raises(ValueError):p.prefix_bound({})

def test_block_layer2_cannot_execute_and_restore():
    calls=[];guards=[]
    class Block:
        def __init__(self,layer):self.layer=layer
        def __call__(self,x):calls.append(self.layer);return x+1
    original=Block.__call__
    with p.block_guard(Block,lambda stage,layer:guards.append((stage,layer)),lambda:False,RuntimeError):
        assert Block(0)(3)==4 and Block(1)(4)==5
        with pytest.raises(p.PrefixIncomplete):Block(2)(5)
    assert calls==[0,1] and Block.__call__ is original
    assert guards==[('before_block',0),('after_block',0),('before_block',1),('after_block',1)]

def test_completed_capture_aborts_before_next_block():
    class Abort(RuntimeError):pass
    class Block:
        layer=0
        def __call__(self):raise AssertionError('must not execute')
    with p.block_guard(Block,lambda *x:None,lambda:True,Abort):
        with pytest.raises(Abort):Block()()

@pytest.mark.parametrize('active,peak,available',[(33<<30,33<<30,20<<30),(1,33<<30,20<<30),(1,2,8<<30)])
def test_actual_memory_and_host_floor_fail_closed(active,peak,available):
    mx=SimpleNamespace(get_active_memory=lambda:active,get_peak_memory=lambda:peak);samples=[]
    with pytest.raises(MemoryError):p.memory_guard(mx,lambda:SimpleNamespace(available_bytes=available),samples,'before_block',0)
    assert len(samples)==1

def test_memory_samples_known_and_unknown():
    mx=SimpleNamespace(get_active_memory=lambda:1,get_peak_memory=lambda:2);samples=[]
    p.memory_guard(mx,lambda:SimpleNamespace(available_bytes=9<<30),samples,'ready')
    with pytest.raises(MemoryError):p.memory_guard(mx,lambda:None,samples,'missing')
    assert samples[-1]['host_available_bytes']is None
