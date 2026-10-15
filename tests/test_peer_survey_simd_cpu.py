"""CPU fixture/oracle and installation checks, not Metal execution evidence."""

import importlib.util
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.lane import simd

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_simd_q4_exponents.py"
_SPEC = importlib.util.spec_from_file_location("probe_simd_q4_exponents", _SCRIPT)
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("group", [32, 64])
@pytest.mark.parametrize("exponent", [-126, -120, -80])
def test_dyadic_oracle_and_old_prescaling_negative_control(group, exponent):
    x, packed, _scales, _biases, reference = probe.adversarial_case(group, 9, exponent=exponent)
    expected = np.array([(-1) ** row * np.ldexp(9.0, exponent + 96) for row in range(9)])
    assert np.array_equal(reference, np.repeat(expected[:, None], 8, axis=1))
    # The reference is exactly representable in BF16, including after scale.
    bits = reference.astype(np.float32).view(np.uint32)
    assert not np.any(bits & 0xFFFF)
    if exponent == -126:
        nibble = 7
        # This is the old law, independent of a device's denormal flushing.
        with np.errstate(under="ignore"):
            prescaled = x[:, nibble] * np.float32(2.0 ** (-4 * nibble))
        old = prescaled * ((packed[7, 0] & np.uint32(15 << 28)).astype(np.float32))
        assert not np.any(old)
        assert np.all(reference[:, 7] != 0)


def test_q4_sources_do_not_prescale_input_or_leave_shifted_float_codes():
    """Static regression guard; the companion probe executes the actual code."""
    sources = simd._HEADER + simd._SCALAR + simd._MMA + simd._PREP
    assert "pre(" not in sources
    assert "* ps" not in sources
    assert "& (0xFu <<" not in sources
    assert " & 0xFu" in simd._SCALAR
    assert " & 0xFu" in simd._MMA
    assert "LOAD8(" not in simd._fragment_source(simd._MMA)


def test_install_checks_every_selected_fragment_width_and_disables_bad_path(monkeypatch):
    simd.mma_no_fragments.clear()
    calls = []

    def fake(x, weight, scales, biases, group, bits, *, kind=None):
        rows = x.shape[0]
        calls.append((rows, kind))
        value = 1 if kind is None and rows == 7 else 0
        return mx.full((rows, 8), value, dtype=mx.bfloat16)

    monkeypatch.setattr(simd, "qmm", fake)
    try:
        weights = mx.zeros((8, 8), dtype=mx.uint32)
        scales = mx.ones((8, 1), dtype=mx.bfloat16)
        assert not simd.check(weights, scales, scales, 64, 4)
        assert simd.mma_no_fragments == {(8, 64, 64)}
        assert simd.rerouted(8, 64, 64, 4) == "mma"
        assert {(rows, None) for rows in range(2, 8)} <= set(calls)
    finally:
        simd.mma_no_fragments.clear()


def test_fragment_reroute_dispatches_direct_mma_without_allocating_fragments(monkeypatch):
    simd.mma_no_fragments.add((8, 64, 64))
    monkeypatch.setattr(simd, "_fragments", lambda *_: pytest.fail("disabled fragments selected"))
    monkeypatch.setattr(simd, "_run", lambda kind, *_: kind)
    try:
        assert simd.qmm(mx.zeros((7, 64), dtype=mx.bfloat16),
                        mx.zeros((8, 8), dtype=mx.uint32),
                        mx.ones((8, 1), dtype=mx.bfloat16),
                        mx.zeros((8, 1), dtype=mx.bfloat16), 64, 4) == "mma"
    finally:
        simd.mma_no_fragments.clear()
