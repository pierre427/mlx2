from collections import defaultdict

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime.models.nemotron_varlen_projection import (
    adaptive_row_exact_selected,
    project_live_rows,
)


def test_live_row_projection_matches_rectangular_projection_on_valid_rows():
    mx.set_default_device(mx.cpu)
    # Test the compaction/scatter law independently of matrix-kernel row-shape
    # arithmetic. The real q8 projection needs a GPU component parity gate.
    layer = lambda x: mx.concatenate([x, x[..., :2]], axis=-1)
    values = mx.arange(3 * 4 * 4, dtype=mx.float32).reshape(3, 4, 4)
    expected = layer(values)
    counters = {}
    from collections import defaultdict

    counters = defaultdict(int)
    actual = project_live_rows(
        layer, values, [4, 2, 1], counters, boundary="in_proj"
    )
    mx.eval(expected, actual)
    assert mx.array_equal(actual[0, :4], expected[0, :4]).item()
    assert mx.array_equal(actual[1, :2], expected[1, :2]).item()
    assert mx.array_equal(actual[2, :1], expected[2, :1]).item()
    assert mx.array_equal(actual[1, 2:], mx.zeros_like(actual[1, 2:])).item()
    assert mx.array_equal(actual[2, 1:], mx.zeros_like(actual[2, 1:])).item()
    assert counters["in_proj_compaction_calls"] == 1
    assert counters["in_proj_padding_rows_skipped"] == 5


def test_live_row_projection_declines_unpadded_geometry():
    mx.set_default_device(mx.cpu)
    layer = nn.Linear(4, 6, bias=False)
    values = mx.zeros((2, 3, 4))
    from collections import defaultdict

    counters = defaultdict(int)
    assert project_live_rows(
        layer, values, [3, 3], counters, boundary="out_proj"
    ) is None
    assert counters["unpadded_declines"] == 1


def test_live_row_projection_rejects_unknown_arithmetic():
    counters = defaultdict(int)
    with pytest.raises(ValueError, match="arithmetic"):
        project_live_rows(
            lambda x: x,
            mx.ones((2, 2, 3)),
            [2, 1],
            counters,
            boundary="in_proj",
            arithmetic="unknown",
        )


def test_adaptive_row_exact_selects_only_measured_b4_out_projection_geometry():
    assert adaptive_row_exact_selected("out_proj", (4, 4, 8192), [4, 4, 2, 1])
    assert not adaptive_row_exact_selected("in_proj", (4, 4, 4096), [4, 4, 2, 1])
    assert not adaptive_row_exact_selected("out_proj", (2, 4, 8192), [4, 2])
    assert not adaptive_row_exact_selected("out_proj", (4, 5, 8192), [5, 5, 2, 2])
    assert not adaptive_row_exact_selected("out_proj", (4, 4, 8192), [4, 4, 4, 1])
