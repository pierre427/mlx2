"""Guarded slab harness CPU checks; no Metal device dispatch."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/research'))
spec=importlib.util.spec_from_file_location('slab_probe',ROOT/'scripts/research/probe_deepseek_v4_slab.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_gpu_slab_refuses_missing_window_before_source_or_allocation(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'source_receipt',lambda:pytest.fail('source before owner'))
    with pytest.raises(SystemExit):probe.main(['--output',str(tmp_path/'new'),'--run-gpu'])
    with pytest.raises(SystemExit):probe.main(['--output',str(tmp_path/'new'),'--packed-weights'])
    with pytest.raises(SystemExit):probe.main(['--output',str(tmp_path/'new'),'--optimized'])
    assert not (tmp_path/'new').exists()


def test_gpu_slab_ownership_failure_cannot_create_evidence(tmp_path,monkeypatch):
    def refuse(*a):raise RuntimeError('no live owner')
    monkeypatch.setattr(probe,'verify_ownership',refuse)
    monkeypatch.setattr(probe,'source_receipt',lambda:pytest.fail('source before owner'))
    with pytest.raises(RuntimeError,match='live owner'):
        probe.main(['--output',str(tmp_path/'new'),'--run-gpu','--ownership-window',str(tmp_path/'lease')])
    assert not (tmp_path/'new').exists()


def test_bounded_first_divergence_is_not_labeled_near_tie_equivalent():
    a=np.array([[[2.,1.,0.],[1.,2.,0.],[0.,1.,2.]]],np.float32)
    b=a.copy();b[0,1]=[2.,1.,0.]
    result=probe.divergence(a,b)
    assert not result['selected_tokens_equal']
    first=result['first_selected_token_divergence']
    assert first['row']==1 and first['shared_selected_prefix']==[0]
    assert first['slab_cross_margin']==first['ordinary_cross_margin']==1
    assert len(first['bounded_window'])<=8
    assert result['equivalence_verdict'].startswith('not_assessed')


def test_cpu_qat_slab_fixture_preserves_transaction_and_shape_diagnostics():
    cases,digest,causality=probe.run_fixtures(gpu=False,packed=False)
    assert len(cases)==27 and len(digest)==64
    assert all(c['owner_closed'] and c['offsets_correct'] for c in cases)
    assert all(c['partial_replay_exact'] is not False for c in cases)
    assert all(c['slab']['logits_bitwise_equal'] for c in cases)
    assert all(c['hc_features_bitwise_equal'] for c in cases)
    assert all(c['committed_state_bitwise_vs_ordinary'] for c in cases)
    assert all(not c['route']['target_row_invariance_established'] for c in cases)
    assert all(c['physical_batch_width']==1 and c['causal_target_rows']==c['width'] for c in cases)
    assert causality['earlier_logits_bitwise_unchanged']
    assert causality['earlier_hc_features_bitwise_unchanged'] and causality['abort_boundary_exact']
