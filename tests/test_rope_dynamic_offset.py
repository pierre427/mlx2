"""Dynamic NTK RoPE recomputes one scalar base per call; array offsets fail closed."""

import mlx.core as mx
import pytest

from mlx2.runtime.models.rope_utils import initialize_rope


def _rope():
    return initialize_rope(
        64, 10000.0, False, {"type": "dynamic", "factor": 2.0}, max_position_embeddings=2048
    )


def test_scalar_offset_works():
    out = _rope()(mx.zeros((1, 1, 4, 64)), offset=3)
    mx.eval(out)
    assert out.shape == (1, 1, 4, 64)


def test_per_row_array_offset_is_refused_explicitly():
    with pytest.raises(ValueError, match="scalar integer offset"):
        _rope()(mx.zeros((2, 1, 4, 64)), offset=mx.array([10, 20]))
