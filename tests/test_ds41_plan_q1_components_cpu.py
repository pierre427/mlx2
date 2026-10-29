"""Inventory/refusal/oracle tests never call a GPU array or shader API."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/research/ds41_plan_q1_components.py'
    spec=importlib.util.spec_from_file_location('_ds41_q1_components_cpu',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_inventory_exactly_36_shapes_and_separate_prepared_lower_bound():
    cases=module().inventory()
    assert len(cases)==len({r['case_id']for r in cases})==36
    assert {r['rows']for r in cases}==set(range(2,8))
    assert {r['projection']for r in cases}=={'gate','up','down'}
    assert {r['layout']for r in cases}=={'compact','mapped'}
    assert all(r['scopes']==('fresh_copy_cast_consumer','prepared_kernel_lower_bound')for r in cases)
    assert all(r['coefficient_repack_bytes']==0 for r in cases)
    assert all(r['source_rows']==r['rows']+(3 if r['layout']=='mapped'else 0)for r in cases)


class Backend:
    gpu=object()
    def __init__(self,gpu=False):self.is_gpu=gpu
    def default_device(self):return self.gpu if self.is_gpu else 'cpu'
    def array(self,*args,**kwargs):pytest.fail('CPU refusal must precede any array or shader')


@pytest.mark.parametrize('checker',[None,False,1,'owned'])
def test_missing_ownership_callback_refuses_before_arrays(checker):
    with pytest.raises(ValueError,match='ownership/live callback'):
        module().run(Backend(gpu=True),checker)


def test_CPU_backend_refuses_even_with_live_callback():
    with pytest.raises(ValueError,match='GPU backend'):
        module().run(Backend(),lambda:True)


@pytest.mark.parametrize('verdict',[False,0,'owned',object()])
def test_failed_or_unrecognized_ownership_proof_refuses(verdict):
    with pytest.raises(RuntimeError,match='check refused'):
        module().run(Backend(gpu=True),lambda:verdict)


def test_live_checker_exception_is_preserved_before_any_array():
    def fail():raise RuntimeError('lease lost')
    with pytest.raises(RuntimeError,match='lease lost'):
        module().run(Backend(gpu=True),fail)


@pytest.mark.parametrize('repetitions,seed',[(0,1),(22,1),(True,1),(7,True),(7,None)])
def test_invalid_work_bound_refuses_before_case_execution(repetitions,seed):
    with pytest.raises(ValueError,match='bounded repetitions'):
        module().run(Backend(gpu=True),lambda:True,repetitions=repetitions,seed=seed)


def test_exact_and_distinct_classifications_do_not_claim_quality():
    m=module();a=np.array([[1.,-0.]],np.float32)
    exact=m._classification(a,a.copy())
    assert exact['classification']=='raw_exact_sample' and exact['differing_words']==0
    b=a.copy();b[0,1]=0.
    changed=m._classification(a,b)
    assert changed['classification']=='distinct_numerical_profile' and changed['differing_words']==1
    assert 'qualification' in changed['scope']
    b[0,0]=float('nan')
    assert m._classification(a,b)['classification']=='nonfinite_failure'


def test_F64_diagnostic_uses_independent_affine_unpack():
    m=module();x=np.ones((2,64),np.float32)
    q=np.array([[0xffffffff,0],[0,0xffffffff],[0,0],[0xffffffff,0xffffffff]],np.uint32)
    scales=np.full((4,1),2.,np.float16);bias=np.full((4,1),-.5,np.float16)
    expected=np.array([[32.,32.,-32.,96.]]*2,np.float32)
    result=m._f64_errors(x,q,scales,bias,expected,expected)
    assert result['reference']['maximum_absolute_error']==0
    assert result['candidate']['maximum_absolute_error']==0
