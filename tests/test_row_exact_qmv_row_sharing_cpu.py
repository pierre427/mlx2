"""Row sharing changes ownership, not qmv arithmetic.  No Metal qualification."""

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime.models import row_exact_qmv as qmv
from mlx2.runtime.row_exact import planner

mx.set_default_device(mx.cpu)


@pytest.mark.parametrize("bits,limit", [(4, 2), (5, 2), (6, 4), (8, 4)])
def test_fast_layout_preserves_control_register_bound(bits, limit):
    assert qmv.row_group_limit(bits, True) == limit


@pytest.mark.parametrize("bits,limit", [(4, 4), (5, 4), (6, 8), (8, 8)])
def test_nonfast_layout_keeps_existing_register_bound(bits, limit):
    assert qmv.row_group_limit(bits, False) == limit


@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize(
    "rows,groups", [(2, 2), (3, 2), (4, 2), (5, 3), (8, 4), (9, 5)]
)
def test_qmv_weight_passes_and_launch_share_the_new_geometry(
    monkeypatch, bits, rows, groups
):
    monkeypatch.setattr(planner, "_table", dict)
    monkeypatch.setattr(qmv, "_kernel", lambda *args: object())
    linear = nn.QuantizedLinear(512, 512, bias=False, group_size=64, bits=bits)
    x = mx.zeros((rows, 512), mx.bfloat16)
    plan = qmv._Plan(linear, x, rows)
    _first, count, template, grid, shapes = plan.launches[0]
    assert qmv.weight_passes(linear, rows) == groups
    assert grid[1] == 2 * groups
    assert count == dict(template)["TOTAL_ROWS"] == rows
    assert shapes == [(rows, 512)]
    assert dict(template)["ROWS"] <= qmv.row_group_limit(bits, True)


@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("rows", [3, 4, 5, 7, 8, 9])
def test_existing_grouped_launch_uses_shared_row_planner_and_guarded_tail(
    monkeypatch, bits, rows
):
    monkeypatch.setattr(planner, "_table", dict)
    monkeypatch.setattr(qmv, "_group_kernel", lambda *args: object())
    linears = [
        nn.QuantizedLinear(512, n, bias=False, group_size=64, bits=bits)
        for n in (512, 256)
    ]
    x = mx.zeros((rows, 512), mx.bfloat16)
    plan = qmv._GroupPlan(linears, x, rows)
    template = dict(plan.template)
    expected = qmv._row_groups(bits, True, 512, rows, 512)
    assert template["ROWS"] == expected
    assert template["TOTAL_ROWS"] == rows
    assert plan.grid[1] == 2 * (-(-rows // expected))
    assert plan.output_shapes == [(rows, 512), (rows, 256)]
    # Ceil rather than floor covers the final partial group.  The unchanged
    # kernel guard loads the last valid input and stores only live rows.
    groups = plan.grid[1] // 2
    assert (groups - 1) * expected < rows <= groups * expected


def test_measured_grouping_still_overrides_new_capacity(monkeypatch):
    key = planner.table_key(4, 512, 512, 4)
    monkeypatch.setattr(planner, "_table", lambda: {key: 2})
    assert qmv._row_groups(4, True, 512, 4, 512) == 2
