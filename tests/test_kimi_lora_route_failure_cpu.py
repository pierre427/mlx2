"""Synthetic CPU run: source-bound route evidence survives a post-update failure."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT/'scripts/research'))
sys.path.insert(0, str(ROOT/'tests/experimental'))
from test_kimi_linear_lora import model_fixture
from mlx2.experimental.kimi_linear_lora.training import LoRATrainingSession
import mlx2.adapters.kimi_linear as adapter_module
import mlx2.desktop_campaign_worker as ownership_module

SPEC = importlib.util.spec_from_file_location('kimi_route_failure_runner', ROOT/'scripts/research/kimi_lora_round.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def setup_run(tmp_path, monkeypatch):
    model = model_fixture(True)
    original = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
    policy = (model.layers[-1].attention.policy, model.layers[-1].attention.flash)
    adapter = SimpleNamespace(model=model, identity={'fingerprint':'c'*64})
    monkeypatch.setattr(adapter_module, 'KimiLinearAdapter', lambda *a, **k:adapter)
    monkeypatch.setattr(ownership_module, 'check_ownership', lambda spec:{'synthetic_cpu':True})
    monkeypatch.setattr(runner, 'host_state', lambda:{'synthetic_cpu':True})
    monkeypatch.setattr(runner, 'actual_execution', lambda adapter:{'synthetic_cpu':True})
    monkeypatch.setattr(runner, 'generate', lambda *a, **k:{'text':'4','tokens':[19,163586],'finish_reason':'stop'})
    # Never initialize/query a Metal allocator in a CPU-only harness test.
    monkeypatch.setattr(mx, 'set_cache_limit', lambda n:None)
    monkeypatch.setattr(mx, 'set_memory_limit', lambda n:None)
    monkeypatch.setattr(mx, 'get_active_memory', lambda:100)
    monkeypatch.setattr(mx, 'get_peak_memory', lambda:200)
    data = tmp_path/'train.jsonl'
    data.write_text(''.join(json.dumps({'id':str(i),'split':'train','input_ids':tokens,'loss_mask':[0]+[1]*(len(tokens)-1)})+'\n'
        for i,tokens in enumerate(([1,2,3,4,5],[1,3,5,7,2,4,8]))))
    spec={'mode':'probe','model':str(tmp_path/'base-copy'),'original_model':str(tmp_path/'original'),
          'out':str(tmp_path/'run'),'data':str(data),'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'base_fingerprint':'c'*64,'artifact_name':'synthetic-kimi-route-failure-fixture-v1','job_id':'cpu-only-route-failure',
          'steps':2,'rank':2,'scale':.5,'last_n_layers':1,'checkpoint_interval':1,'learning_rate':1e-5}
    path=tmp_path/'spec.json';path.write_text(json.dumps(spec))
    return model,original,policy,path


def assert_restored(model,original,policy):
    assert not model.training
    assert (model.layers[-1].attention.policy,model.layers[-1].attention.flash)==policy
    after={k:np.array(v)for k,v in tree_flatten(model.parameters())}
    assert set(after)==set(original)
    for key in original:np.testing.assert_array_equal(after[key],original[key])


@pytest.mark.parametrize('snapshot_failure',[False,True])
def test_two_updates_memory_failure_retains_route_evidence_and_original_error(tmp_path,monkeypatch,snapshot_failure):
    model,original,policy,path=setup_run(tmp_path,monkeypatch)
    calls=0
    def guard(*args):
        nonlocal calls
        calls+=1
        if calls==3:raise MemoryError('synthetic post-step2 bound')
        return {'active_bytes':100,'peak_bytes':200}
    monkeypatch.setattr(runner,'gpu_guard',guard)
    real_snapshot=runner.training_route_deltas
    snapshots=0
    def snapshot(*args):
        nonlocal snapshots
        snapshots+=1
        if snapshot_failure and snapshots==3:raise RuntimeError('synthetic counter access failure')
        return real_snapshot(*args)
    monkeypatch.setattr(runner,'training_route_deltas',snapshot)
    with pytest.raises(MemoryError,match='synthetic post-step2 bound'):runner.run(path)
    receipt=json.loads((tmp_path/'run/results/receipt.json').read_text())
    assert receipt['status']=='failed'and receipt['error']['type']=='MemoryError'
    assert len(receipt['training'])==len(receipt['training_checkpoints'])==2
    first,second=[row['training_route_deltas']for row in receipt['training']]
    assert set(first)==set(second)=={'1'}
    assert first['1']['expanded_sdpa']>0 and second['1']['expanded_sdpa']>first['1']['expanded_sdpa']
    assert receipt['training_route_deltas']==second
    assert 'not an observed Metal' in receipt['training_route_counter_scope']
    if snapshot_failure:assert receipt['training_route_snapshot_error']['type']=='RuntimeError'
    else:assert receipt['training_route_snapshot_phase']=='cleanup_before_session_close'
    assert 'export'not in receipt
    assert_restored(model,original,policy)


def test_success_preserves_final_counter_semantics(tmp_path,monkeypatch):
    model,original,policy,path=setup_run(tmp_path,monkeypatch)
    monkeypatch.setattr(runner,'gpu_guard',lambda *a:{'active_bytes':100,'peak_bytes':200})
    runner.run(path)
    receipt=json.loads((tmp_path/'run/results/receipt.json').read_text())
    assert receipt['status']=='completed'
    assert receipt['training_route_deltas']==receipt['training'][-1]['training_route_deltas']
    assert receipt['training_route_deltas']['1']['expanded_sdpa']>0
    assert 'training_route_snapshot_phase'not in receipt
    assert receipt['ordinary_smoke']['tokens']==receipt['restored_base_smoke']['tokens']
    assert_restored(model,original,policy)
