"""Synthetic head geometry and register envelope checks; no GPU result."""
import importlib.util
from pathlib import Path

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime.models.row_exact_qmv import TileGeometry, row_group_limit

_spec = importlib.util.spec_from_file_location("head_adoption",
    Path(__file__).parents[1] / "scripts/research/pr_adoption_head.py")
head = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(head)


def test_wider_explicit_geometry_does_not_expand_the_ordinary_planner():
    assert row_group_limit(8, True) == 4
    with pytest.raises(ValueError, match="register"):
        TileGeometry(8).validate(bits=8, fast=True, rows=8)
    TileGeometry(8, row_value_budget=64).validate(bits=8, fast=True, rows=8)
    assert row_group_limit(8, True) == 4
    with pytest.raises(ValueError, match="row_value_budget"):
        TileGeometry(8, row_value_budget=128)


def test_physical_head_rows_plan_only_fewer_pass_candidates():
    linear = nn.QuantizedLinear(256, 64, bias=False, group_size=64, bits=8)
    # Stored values are irrelevant; no projection or model executes.
    linear.scales = linear.scales.astype(mx.bfloat16)
    linear.biases = linear.biases.astype(mx.bfloat16)
    for rows in (1, 3):
        assert head.plan(linear, rows)["candidates"] == []
    for rows in (6, 8):
        row = head.plan(linear, rows)
        assert row["ordinary_register_bound_passes"] == 2
        assert any(c["weight_passes"] == 1 for c in row["candidates"])
        assert row["selected"] is False and row["qualified"] is False
