"""Plan wider head sharing on the existing one-row-exact kernel.

No donor qmv_wide law and no production selection. An explicit 64-value
register envelope may spill; only complete-target evidence can justify use.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
from mlx2.runtime.models.row_exact_qmv import (
    TileGeometry,
    layout_decline_reason,
    qmv_fast_layout,
    row_group_limit,
)


def plan(linear, rows):
    if type(rows) is not int or not 1 <= rows <= 32:
        raise ValueError("bounded physical head rows required")
    reason = layout_decline_reason(linear)
    if reason:
        return {"admitted": False, "reason": reason}
    k = int(linear.scales.shape[-1]) * linear.group_size
    n = int(linear.weight.shape[0])
    fast = qmv_fast_layout(k, n, linear.bits)
    ordinary_limit = row_group_limit(linear.bits, fast)
    ordinary_groups = -(-rows // min(rows, ordinary_limit))
    candidates = []
    for group in (6, 7, 8):
        if group > rows:
            continue
        geometry = TileGeometry(group, 4, row_value_budget=64)
        try:
            geometry.validate(bits=linear.bits, fast=fast, rows=rows)
        except ValueError:
            continue
        groups = -(-rows // group)
        if groups < ordinary_groups:
            candidates.append({"rows_per_group": group, "row_value_budget": 64,
                               "weight_passes": groups})
    return {"admitted": True, "physical_rows": rows, "n": n, "k": k,
            "bits": linear.bits, "group_size": linear.group_size,
            "ordinary_register_bound_passes": ordinary_groups,
            "candidates": candidates, "selected": False, "qualified": False,
            "reference": "one-row quantized_matmul; not stock batched qmv_wide"}
