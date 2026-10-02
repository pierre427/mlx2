"""Host numerical evidence gate; no MLX, external checkpoints or downloads."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "_mlx2_radio_validation_probe",
    Path(__file__).resolve().parents[1] / "scripts/validate_radio_encoder.py",
)
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


@pytest.mark.parametrize("arm", ["actual", "reference"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_either_reference_or_actual_fails(arm, value):
    values = {"actual": np.ones((1, 2)), "reference": np.ones((1, 2))}
    values[arm][0, 0] = value
    with pytest.raises(AssertionError, match="finite failure"):
        SCRIPT.comparison_metrics(**values, label="synthetic")


@pytest.mark.parametrize(
    "actual,reference",
    [
        (np.ones((1, 2)), np.ones((2, 2))),
        (np.empty((0, 2)), np.empty((0, 2))),
        (np.array(1), np.array(1)),
    ],
)
def test_invalid_geometry_fails(actual, reference):
    with pytest.raises(AssertionError, match="shape"):
        SCRIPT.comparison_metrics(actual, reference, label="synthetic")


@pytest.mark.parametrize(
    "actual,reference",
    [
        (np.ones((1, 2)), np.zeros((1, 2))),
        (np.full((1, 2), 1e308), np.full((1, 2), 1e308)),
    ],
)
def test_undefined_or_overflowed_metrics_fail(actual, reference):
    with pytest.raises(AssertionError, match="metrics are nonfinite"):
        SCRIPT.comparison_metrics(actual, reference, label="synthetic")


def test_valid_identical_and_tolerance_metrics():
    reference = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    exact = SCRIPT.comparison_metrics(reference, reference, label="exact")
    assert exact["max_abs_error"] == exact["relative_l2"] == 0
    assert exact["cosine_min"] == pytest.approx(1)
    near = SCRIPT.comparison_metrics(reference * (1 + 1e-5), reference, label="near")
    assert near["relative_l2"] == pytest.approx(1e-5)
    with pytest.raises(AssertionError, match="bad"):
        SCRIPT.comparison_metrics(reference * 1.01, reference, label="bad")


def test_equal_zero_vectors_have_explicit_exact_convention():
    zeros = np.zeros((1, 2))
    result = SCRIPT.comparison_metrics(zeros, zeros, label="zero-exact")
    assert result["cosine_min"] == result["cosine_mean"] == 1
    assert result["relative_l2"] == result["max_abs_error"] == 0
    mixed = np.array([[0.0, 0.0], [1.0, 2.0]])
    assert SCRIPT.comparison_metrics(mixed, mixed, label="mixed")[
        "cosine_min"
    ] == pytest.approx(1)
