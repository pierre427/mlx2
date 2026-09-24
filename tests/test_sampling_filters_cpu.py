"""Sampling filter boundary tests through NumPy, without importing MLX."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _put_along_axis(array, indices, values, axis):
    result = array.copy()
    np.put_along_axis(result, indices, values, axis=axis)
    return result


@pytest.fixture
def filters():
    path = Path(__file__).parents[1] / "src/mlx2/runtime/sample_utils.py"
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes.extend(node for node in ast.parse(path.read_text()).body if (
        isinstance(node, ast.FunctionDef) and node.name in {"apply_min_p", "apply_top_p"}
    ))
    mx = SimpleNamespace(**{name: getattr(np, name) for name in (
        "max", "where", "exp", "argsort", "take_along_axis", "cumsum",
        "zeros_like", "arange", "float32", "argpartition",
    )}, put_along_axis=_put_along_axis)
    scope = {"mx": mx, "math": math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), scope)  # noqa: S102 - local production AST only
    return SimpleNamespace(**{name: scope[name] for name in ("apply_min_p", "apply_top_p")})


def test_zero_min_p_preserves_unfiltered_distribution(filters):
    row = np.log(np.array([[0.1, 0.2, 0.7]], dtype=np.float32))
    np.testing.assert_array_equal(filters.apply_min_p(row, 0.0), row)


@pytest.mark.parametrize("top_p", [1e-12, 1e-8])
def test_tiny_positive_top_p_always_retains_a_most_likely_token(filters, top_p):
    rows = np.log(np.array([[0.1, 0.2, 0.7], [0.7, 0.2, 0.1]], dtype=np.float32))
    result = filters.apply_top_p(rows, top_p)
    assert np.isfinite(result).any(axis=-1).all()
    assert np.argmax(result, axis=-1).tolist() == [2, 0]


def test_top_p_keeps_expected_nucleus_for_ordinary_threshold(filters):
    row = np.log(np.array([[0.1, 0.2, 0.7]], dtype=np.float32))
    result = filters.apply_top_p(row, 0.8)
    assert np.isfinite(result).tolist() == [[False, True, True]]
