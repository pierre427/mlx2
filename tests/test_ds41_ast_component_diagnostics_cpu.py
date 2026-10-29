"""NumPy-only synthetic diagnostic fixtures; no checkpoint/GPU claim."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("ds41_ast_probe", Path(__file__).resolve().parents[1] / "scripts/research/ds41_ast_target_components.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_completed_source_empty_partial_plane_has_zero_error():
    left = np.empty((1, 0, 512), np.float32)
    result = probe.compare_arrays(left, left.copy(), atol=3e-3, rtol=3e-3)
    assert result == dict(shape=[1, 0, 512], finite=True, raw_bits_equal=True,
                          first_raw_mismatch=None, tolerance_valid=True,
                          maximum_absolute_error=0.0)


@pytest.mark.parametrize("right,reason", [
    (np.empty((1, 0, 128), np.float32), "shape mismatch"),
    (np.empty((1, 0, 512), np.float64), "dtype mismatch"),
])
def test_incompatible_state_plane_refuses_before_reduction(right, reason):
    with pytest.raises(ValueError, match=reason):
        probe.compare_arrays(np.empty((1, 0, 512), np.float32), right, atol=3e-3, rtol=3e-3)
