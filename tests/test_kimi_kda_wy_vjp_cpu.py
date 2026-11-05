"""Synthetic CPU forward/backward oracles for the existing KDA WY candidate."""
import importlib.util
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/research/kimi_kda_wy_probe.py'
spec = importlib.util.spec_from_file_location('kimi_kda_wy_probe', SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture(autouse=True)
def cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


@pytest.mark.parametrize('length,segment', [(1, 16), (15, 16), (16, 16), (17, 16), (33, 16), (65, 32), (129, 64)])
@pytest.mark.parametrize('gate', ['mild', 'strong'])
def test_wy_forward_and_all_input_gate_parameter_vjps(length, segment, gate):
    result = probe.probe_case(length, segment=segment, gate=gate, repetitions=1)
    assert result['passed'], result
    assert set(result['gradients']) == set(probe.NAMES)


def test_wy_gate_vjp_has_independent_finite_difference_check():
    primals, cotangents, forward = probe.fixture(5, heads=1, key_dim=8, value_dim=6)
    _, gradients = mx.vjp(lambda *x: forward(*x, method='wy'), primals, cotangents)
    mx.eval(gradients)
    epsilon = .002
    # Continuous decay-logit, update-logit, A_log, dt_bias and initial-state
    # parameters all influence the weighted output plus final recurrent state.
    for index in (3, 4, 5, 6, 7):
        direction = mx.ones_like(primals[index]) / primals[index].size**.5
        plus, minus = list(primals), list(primals)
        plus[index] = plus[index] + epsilon * direction
        minus[index] = minus[index] - epsilon * direction
        def objective(values):
            return sum(mx.sum(a * b) for a, b in zip(forward(*values, method='wy'), cotangents, strict=True))
        finite_difference = ((objective(plus) - objective(minus)) / (2 * epsilon)).item()
        analytic = mx.sum(gradients[index] * direction).item()
        np.testing.assert_allclose(analytic, finite_difference, atol=3e-4, rtol=.01)
