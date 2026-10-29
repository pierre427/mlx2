"""Actual-weight probe preparation only; no GPU/model execution."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import mlx.core as mx
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/research/ds41_ast_batched_moe_components.py'
spec=importlib.util.spec_from_file_location('ast_moe_components',path)
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_selection_and_fixed_cases_cover_reuse_mixed_and96_distinct():
    store=SimpleNamespace(kind=lambda k:'original'if k[0]==0 and k[1]<48 else'q1')
    layer,original,q1=probe._selection(store)
    assert layer==0 and len(original)==len(q1)==48
    cases=dict(probe._routes(original,q1))
    assert set(cases)=={'original_shared16','q1_shared16','mixed_shared16','globally_distinct96'}
    for ids in cases.values():
        assert ids.shape==(16,6)and ids.dtype==np.uint32
        assert all(len(set(row))==6 for row in ids.tolist())
    assert len(np.unique(cases['globally_distinct96']))==96
    assert all((cases['mixed_shared16']<48).sum(axis=1)==3)
    assert all((cases['original_shared16']<48).sum(axis=1)==6)
    assert all((cases['q1_shared16']<48).sum(axis=1)==0)


def test_selection_refuses_insufficient_actual_formats():
    with pytest.raises(ValueError,match='48 original'):
        probe._selection(SimpleNamespace(kind=lambda k:'original'))


def test_callable_refuses_cpu_before_model_or_bank_access(tmp_path,monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw:pytest.fail('GPU forbidden'))
    try:
        with pytest.raises(ValueError,match='parent-owned GPU'):probe.run(None,tmp_path,lambda:True)
    finally:mx.set_default_device(old)


@pytest.mark.parametrize('kwargs',[
    dict(consumer=lambda:None),dict(consumer_identity='a'*64),
    dict(consumer=lambda:None,consumer_identity='oops'),
    dict(evidence_name='../escape.json'),dict(evidence_name='x/escape.json'),
    dict(timing_repeats=True),dict(timing_repeats=9),dict(down_consumer=lambda:None),
])
def test_explicit_consumer_admission_refuses_incomplete_or_unsafe_requests(kwargs):
    args=dict(consumer=None,consumer_identity=None,evidence_name='case.json',timing_repeats=0,down_consumer=None)
    args.update(kwargs)
    with pytest.raises(ValueError):probe._injection(**args)


def test_explicit_consumer_admission_and_reference_defaults():
    probe._injection(None,None,'ast-batched-moe-components.json',0,None)
    probe._injection(lambda:None,'a'*64,'g8.json',2,lambda:None)
