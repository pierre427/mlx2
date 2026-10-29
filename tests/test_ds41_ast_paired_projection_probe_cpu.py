"""Research probe contracts; no model checkpoint or GPU execution."""
from pathlib import Path
import importlib.util
import mlx.core as mx
import pytest
from types import SimpleNamespace

P = Path(__file__).resolve().parents[1] / 'scripts/research/ds41_ast_paired_projection_probe.py'
spec = importlib.util.spec_from_file_location('paired_projection_probe', P)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_single_abba_sequence_follows_independent_two_arm_warmup():
    assert probe.order() == ('reference', 'joined', 'joined', 'reference')
    assert probe.order().count('reference') == probe.order().count('joined') == 2


def test_run_refuses_cpu_before_adapter_access(tmp_path, monkeypatch):
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast, 'metal_kernel', lambda *a, **kw: pytest.fail('GPU forbidden'))
    try:
        with pytest.raises(ValueError, match='parent-owned GPU'):
            probe.run(None, tmp_path, lambda: True, None, None, None, None, None)
    finally:
        mx.set_default_device(old)


def test_actual_input_capture_returns_original_outputs_and_restores_after_error(monkeypatch):
    from mlx2.adapters import deepseek_v41_ast_batched_dense_exact as dense
    model, value, result = object(), object(), object()
    original = lambda *args, **kwargs: result
    monkeypatch.setattr(dense, 'paired_linear_tile', original)
    with pytest.raises(RuntimeError, match='injected'):
        with probe.capture_inputs() as captured:
            a, b = 'blk.0.attn_q_a.weight', 'blk.0.attn_kv.weight'
            assert dense.paired_linear_tile(model, value, a, b) is result
            assert captured[(a, b)] == (model, value)
            assert dense.paired_linear_tile(model, value, 'other.weight', 'another.weight') is result
            assert len(captured) == 1
            with pytest.raises(ValueError, match='repeated'):
                dense.paired_linear_tile(model, value, a, b)
            raise RuntimeError('injected')
    assert dense.paired_linear_tile is original


def test_component_gate_refuses_missing_actual_pairs_before_math():
    with pytest.raises(ValueError, match='all80'):
        probe.validate_pairs(None, {}, lambda: True, SimpleNamespace())
