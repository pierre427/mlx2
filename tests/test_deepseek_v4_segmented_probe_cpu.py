"""Owned segmented harness guards and independent small CPU oracle."""
import importlib.util
from pathlib import Path
import sys
import pytest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/research'))
spec=importlib.util.spec_from_file_location('segmented_probe',ROOT/'scripts/research/probe_deepseek_v4_segmented.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_gpu_refuses_missing_ownership_before_source(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'source_receipt',lambda:pytest.fail('source before required ownership'))
    with pytest.raises(SystemExit):probe.main(['--output',str(tmp_path/'new'),'--run-gpu'])
    assert not (tmp_path/'new').exists()


def test_bound_is_checked_before_allocating_fixture(monkeypatch):
    monkeypatch.setattr(probe,'fixture',lambda *a:pytest.fail('unexpected allocation'))
    with pytest.raises(RuntimeError,match='workingcap'):
        probe.run_case(512,4096,2048,32,gpu=False,repetitions=1,cap=1024,guard=lambda:None,ordinary=False)


def test_small_independent_fp16_operand_oracle_is_finite_and_original_slot_order():
    x,ids,scores,g,u,d=probe.fixture(2,256,256,6)
    expected,receipt=probe.independent_half_oracle(x,ids,scores,g,u,d)
    # Reordering every token's selected expert/score slots must preserve its sum.
    reverse,other=probe.independent_half_oracle(x,ids[:,::-1].copy(),scores[:,::-1].copy(),g,u,d)
    np.testing.assert_allclose(expected,reverse,atol=1e-6,rtol=1e-6)
    assert np.isfinite(expected).all() and expected.shape==(2,256)
    assert 'FP16rounded' in receipt['law']


def test_gpu_plan_refuses_cpu_before_constructing_kernel(monkeypatch):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mlx2.runtime.gguf import segmented_prefill
    monkeypatch.setattr(segmented_prefill,'_kernel',lambda *a:pytest.fail('GPU kernel construction on CPU'))
    with pytest.raises(Exception,match='owned Metal'):
        segmented_prefill.build_plan(mx.array([[0]],dtype=mx.uint32),6)
