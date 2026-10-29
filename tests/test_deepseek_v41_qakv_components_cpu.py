"""Synthetic CPU contracts for bounded original-weight Qa/KV component runner."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
import pytest

@pytest.fixture
def helper():
    p=Path(__file__).resolve().parents[1]/'scripts/research/ds41_qakv_components.py'
    spec=importlib.util.spec_from_file_location('qakv_components_cpu',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Metal forbidden')))
    yield
    mx.set_default_device(old)


def weights(*,wrong=False,rounded_wrong=False):
    class W:
        def __init__(self):self.scheduled_dense_calls={}
        def count(self,k,n=1):self.scheduled_dense_calls[k]=self.scheduled_dense_calls.get(k,0)+n
        def _dense_input(self,x):self.count('source_dense_qat_fused');return x*.5
        def linear(self,name):
            def f(x):
                p=self._dense_input(x);self.count('official_mxfp8_act32');return p[...,:2]+(1 if name=='qa' else 2)
            return f
        def paired_project(self,x,a,b,*,rounded=True):
            p=self._dense_input(x);self.count('official_mxfp8_act32',2)
            self.count('source_qakv_qat_reuse_pairs');self.count('source_qakv_input_qat_calls_avoided')
            v=(p[...,:2]+1+(1 if wrong else 0),p[...,:2]+2)
            if rounded:return tuple(y.astype(mx.bfloat16).astype(mx.float32)+(1 if rounded_wrong else 0)for y in v)
            return v
    return W()


def comparer(calls):
    def compare(mx,np,functions,repeats):
        assert repeats==3;calls.append(set(functions));ref=functions['reference']();mx.eval(ref)
        result={}
        for name,fn in functions.items():
            x=fn();mx.eval(x);result[name]={'exact':np.asarray(ref).tobytes()==np.asarray(x).tobytes()}
        return result
    return compare

@pytest.mark.parametrize('phase,rows',[('pp99',99),('prompt_boundary1',1),('decode1',1)])
def test_synthetic_deterministic_finite_explicit_labels(helper,phase,rows):
    a,meta=helper.synthetic_input(np,2,phase,rows);b,_=helper.synthetic_input(np,2,phase,rows)
    assert a.shape==(1,rows,5120)and a.dtype==np.float32 and np.isfinite(a).all()
    assert a.tobytes()==b.tobytes()and not meta['captured_trajectory']
    assert a.reshape(-1,32)[0,16]!=a.reshape(-1,32)[0,17]

@pytest.mark.parametrize('args',[(40,'pp99',99),(-1,'pp99',99),(0,'other',1),(0,'pp99',1)])
def test_bad_synthetic_geometry(helper,args):
    with pytest.raises(ValueError):helper.synthetic_input(np,*args)


def test_real_call_structure_independent2_shared1_raw_boundary_exact(helper):
    w=weights();calls=[];x=mx.full((1,3,32),.5,mx.float32)
    result=helper.run_case(mx,np,w,x,'qa','kv',lambda x:x.astype(mx.bfloat16).astype(mx.float32),comparer(calls))
    assert result['all_exact']and len(result['checks'])==7
    assert calls==[{'reference','qat_reuse'}]
    assert result['reference_counter_delta']['source_dense_qat_fused']==2
    assert result['candidate_counter_delta']['source_dense_qat_fused']==1
    assert '_dense_input'not in w.__dict__


def test_raw_mismatch_excludes_candidate_timings(helper):
    calls=[];result=helper.run_case(mx,np,weights(wrong=True),mx.zeros((1,1,32)), 'qa','kv',lambda x:x,comparer(calls))
    assert not result['all_exact']and calls==[{'reference'}]
    assert not result['checks']['qa_raw']['exact']


def test_actual_rounded_api_precheck_mismatch_is_not_hidden(helper):
    with pytest.raises(ValueError,match='rounded production API'):
        helper.run_case(mx,np,weights(rounded_wrong=True),mx.zeros((1,1,32)), 'qa','kv',lambda x:x,comparer([]))


def test_capture_restores_instance_override_on_error(helper):
    w=weights();override=lambda x:x;w._dense_input=override
    def bad():w._dense_input(mx.zeros((1,1,32)));raise RuntimeError('fail')
    with pytest.raises(RuntimeError):helper.capture_preparations(w,bad)
    assert w._dense_input is override


def test_provider_has_no_dense_or_expert_constructor(helper):
    class C:
        def __init__(self):raise AssertionError('full provider construction')
    bank={'x':object()};w=helper.provider(C,bank)
    assert w._quant is bank and w.source_qakv_policy=='qat_reuse'
    assert w._ready()is None and w._ensure_dense()is None


def test_fence_failure_quarantines_coefficient_owner(helper):
    w=SimpleNamespace(_quant={'q':object()})
    class M:
        def synchronize(self):raise RuntimeError('fence')
    with pytest.raises(RuntimeError):helper.release_layer(M(),w)
    assert helper._QUARANTINE==[w]and w._quant


def test_successful_fence_clears_bank_before_allocator_cache(helper):
    w=SimpleNamespace(_quant={'q':object()});seen=[]
    class M:
        def synchronize(self):seen.append('fence')
        def clear_cache(self):assert not w._quant;seen.append('clear')
    helper.release_layer(M(),w);assert seen==['fence','clear']


def test_cpu_cannot_execute_gpu_runner(helper):
    with pytest.raises(RuntimeError,match='caller-owned GPU'):helper.run_components(mx,np,None,None,{},None,None)


def test_raw_bits_signed_zero_and_shape_gate(helper):
    a=np.zeros((1,1),np.float32);b=-a;assert not helper.raw_check(np,a,b)['exact']
    with pytest.raises(ValueError):helper.raw_check(np,a,b.astype(np.float64))
