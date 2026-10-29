# SPDX-License-Identifier: MIT
import importlib.util
from pathlib import Path
from types import SimpleNamespace as N
import pytest
from mlx2.adapters.deepseek_v41_identity import effective_source_identity

spec=importlib.util.spec_from_file_location('resident_identity_gate',Path(__file__).resolve().parents[1]/'scripts/research/ds41_resident_prefill_model_gate.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture(monkeypatch):
    base={'fingerprint':'a'*64};store=N(binding='a'*64,q1_binding='b'*64,hot_binding='c'*64)
    effective=effective_source_identity(base,q1_bank=store.q1_binding,hot_map=store.hot_binding)
    adapter=N(identity=effective,weights=N(artifact={'identity':base},expert_stream=store,verify_source=lambda:True))
    monkeypatch.setattr(m,'CHECKPOINT','synthetic-hybrid@'+effective['fingerprint'])
    return adapter


def test_base_and_effective_derivative_are_verified_separately(monkeypatch):
    adapter=fixture(monkeypatch)
    assert adapter.identity['fingerprint']!=adapter.weights.expert_stream.binding
    assert m._checkpoint(adapter)==m.CHECKPOINT


@pytest.mark.parametrize('change',['base','q1','hotmap','effective','stat'])
def test_wrong_source_derivative_or_stat_fails(monkeypatch,change):
    adapter=fixture(monkeypatch);store=adapter.weights.expert_stream
    if change=='base':adapter.weights.artifact['identity']['fingerprint']='d'*64
    if change=='q1':store.q1_binding='d'*64
    if change=='hotmap':store.hot_binding='d'*64
    if change=='effective':adapter.identity['fingerprint']='d'*64
    if change=='stat':adapter.weights.verify_source=lambda:False
    with pytest.raises(ValueError):m._checkpoint(adapter)
