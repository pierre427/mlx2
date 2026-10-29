"""Projection attribution guards/oracle on explicit CPU; no GPU or model load."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import mlx.core as mx
mx.set_default_device(mx.cpu)
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts/research'))
spec=importlib.util.spec_from_file_location('hc_rowlaw',ROOT/'scripts/research/probe_deepseek_v4_hc_rowlaw.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_actual_geometry_deterministic_weights_and_cancellation():
    for outputs in (24,4):
        a,w=probe.fixture(outputs,'cancellation');b,z=probe.fixture(outputs,'cancellation')
        assert a.shape==(6,16384) and w.shape==(outputs,16384) and w.dtype==np.float16
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(w,z)
        np.testing.assert_array_equal(a[:,::2],a[:,1::2])


def test_first_difference_retains_only_one_small_mix_row():
    a=np.zeros((6,24),np.float32);b=a.copy();a[2,7]=1e-7;a[3,5]=1e-6
    r=probe.row_difference(a,b)
    assert not r['bitwise_equal'] and r['differing_values']==2
    assert r['first_divergence']['row']==2 and r['first_divergence']['column']==7
    assert len(r['first_divergence']['batched_mix_row'])==24
    assert r['numerical']['passed'] # tolerance never overwrites bitwise difference


def test_cpu_actual_projection_shape_and_separate_normalization():
    cases=probe.run_case(4,'random',gpu=False,repetitions=1)
    assert len(cases)==6 and [x['rows'] for x in cases]==list(range(1,7))
    assert all(x['independent_fp64_oracle']['passed'] for x in cases)
    assert cases[0]['projection']['bitwise_equal']
    assert all(x['physical_batch_width']==1 and len(x['warm_eval_seconds'])==1 for x in cases)


def test_guard_before_source_or_allocation(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'verify_ownership',lambda *a:(_ for _ in ()).throw(RuntimeError('not owned')))
    monkeypatch.setattr(probe,'source_receipt',lambda:pytest.fail('premature source'))
    with pytest.raises(RuntimeError,match='not owned'):
        probe.main(['--output',str(tmp_path/'new'),'--run-gpu','--ownership-window',str(tmp_path/'owned')])
    assert not (tmp_path/'new').exists()


def test_existing_receipt_refused(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'source_receipt',lambda:pytest.fail('overwriting evidence'))
    with pytest.raises(SystemExit):probe.main(['--output',str(tmp_path)])
