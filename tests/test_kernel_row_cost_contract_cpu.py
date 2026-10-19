"""Each kernel's published ``weight_passes`` equals the row geometry its own
launch helpers produce. CPU fixtures: a geometry tie, not Metal qualification."""
import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime.lane import installer, matmul, simd
from mlx2.runtime.models import row_exact_qmv as qmv


@pytest.mark.parametrize("bits,gs,n,k", [(4, 64, 4096, 4096), (8, 64, 1024, 2048),
                                         (4, 32, 2048, 1024), (5, 32, 4096, 2048)])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 7, 8, 9, 16, 17, 40, 128])
def test_simd_passes_equal_the_row_axis_of_its_own_launch(bits, gs, n, k, rows):
    family = simd.path(bits, gs, n, k, mx.bfloat16, mx.bfloat16)
    passes = simd.weight_passes(bits, gs, n, k, mx.bfloat16, mx.bfloat16, rows)
    if family == "affine":
        _ops, rt = simd._affine_launch(rows)
        assert passes == -(-rows // rt)
        return
    kind = "bits_mma" if family == "simd_bits" else "mma"
    _consts, grid, _tg, _shapes = simd._launch(kind, rows, n, k, gs, bits)
    assert passes == grid[1]
    if family == "simd4" and simd._scalar_kind(rows, n, k, gs):
        assert simd._launch("scalar", rows, n, k, gs, bits)[1][1] == passes == 1


@pytest.mark.parametrize("rows", [1, 2, 15, 16, 17, 32, 33, 64, 100, 128])
def test_mpp_passes_equal_the_row_axis_lane_matmul_launches(rows):
    lw = matmul.LaneWeights(bits=4, group_size=64, n=256, k=256, split_k=1,
                            weight=None, backend="mpp")
    mp, block = matmul._row_tiling(rows)
    assert matmul.weight_passes(lw, rows) == -(-mp // block)
    with pytest.raises(matmul.LaneUnsupported):
        matmul.weight_passes(lw, matmul.MAX_ROWS + 1)


def test_simd_lane_delegates_to_the_simd_tiling():
    scales = mx.zeros((256, 4), mx.bfloat16)
    lw = matmul.LaneWeights(bits=4, group_size=64, n=256, k=256, split_k=1, weight=None,
                            backend="simd", scales=scales, biases=scales)
    for rows in (1, 5, 17, 40):
        assert matmul.weight_passes(lw, rows) == simd.weight_passes(
            4, 64, 256, 256, mx.bfloat16, mx.bfloat16, rows)
    bare = matmul.LaneWeights(bits=4, group_size=64, n=256, k=256, split_k=1, weight=None,
                              backend="simd")
    with pytest.raises(matmul.LaneUnsupported):
        matmul.weight_passes(bare, 2)


def test_qmv_passes_equal_the_guarded_launch_grid(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(qmv, "_kernel", lambda bits, gs, fast: object())
    linear = nn.QuantizedLinear(4096, 4096, bias=False, group_size=64, bits=4)
    x = mx.zeros((1, 4096), mx.bfloat16)
    assert qmv.layout_decline_reason(linear) is None
    for rows in (2, 3, 4, 8, 12, 16, 32):
        plan = qmv._Plan(linear, x, rows)
        _first, _count, _template, grid, _shapes = plan.launches[0]
        assert grid[1] == 2 * qmv.weight_passes(linear, rows)
    # One row, a declined layout and a call past the kernel's row bound run
    # one stock one-row call per row.
    assert qmv.weight_passes(linear, 0) == 0
    assert qmv.weight_passes(linear, 1) == 1
    assert qmv.weight_passes(linear, qmv.MAX_ROWS + 1) == qmv.MAX_ROWS + 1
    quad = nn.QuantizedLinear(64, 4096, bias=False, group_size=64, bits=4)
    assert qmv.layout_decline_reason(quad) == "quad_k_stock_traversal"
    assert qmv.weight_passes(quad, 8) == 8
    dense = nn.Linear(64, 64, bias=False)
    assert qmv.weight_passes(dense, 8) == 8


def test_qmv_row_exact_lane_answers_with_the_lane_tiling():
    linear = nn.QuantizedLinear(256, 256, bias=False, group_size=64, bits=4)
    lw = matmul.LaneWeights(bits=4, group_size=64, n=256, k=256, split_k=1,
                            weight=None, backend="mpp")
    object.__setattr__(linear, qmv.LANE_ATTR, lw)
    assert qmv.weight_passes(linear, 20) == matmul.weight_passes(lw, 20)
    long = matmul.MAX_ROWS + 5
    assert qmv.weight_passes(linear, long) == (
        matmul.weight_passes(lw, matmul.MAX_ROWS) + matmul.weight_passes(lw, 5))


def test_installed_module_passes_follow_the_mixin_dispatch(monkeypatch):
    lw = matmul.LaneWeights(bits=4, group_size=64, n=256, k=256, split_k=1,
                            weight=None, backend="mpp")
    module = nn.QuantizedLinear(256, 256, bias=False, group_size=64, bits=4)
    assert installer.module_weight_passes(module, 4) is None
    assert installer.installed_lane(module) is None
    assert installer.module_row_capacity(module) is None
    object.__setattr__(module, "_lane_prepared", lw)
    object.__setattr__(module, "_lane_min_rows", 4)
    object.__setattr__(module, "_lane_max_rows", 32)
    object.__setattr__(module, "_lane_chunk_above_max", False)
    monkeypatch.setattr(installer, "ENABLED", [True])
    monkeypatch.setattr(installer, "available", lambda: True)
    assert installer.installed_lane(module) is lw
    assert installer.module_weight_passes(module, 3) is None  # stock below crossover
    assert installer.module_weight_passes(module, 4) == matmul.weight_passes(lw, 4)
    assert installer.module_weight_passes(module, 32) == matmul.weight_passes(lw, 32)
    assert installer.module_weight_passes(module, 33) is None  # stock past the window
    assert installer.module_row_capacity(module) == 32
    object.__setattr__(module, "_lane_chunk_above_max", True)
    assert installer.module_row_capacity(module) is None
    assert installer.module_weight_passes(module, 70) == (
        2 * matmul.weight_passes(lw, 32) + matmul.weight_passes(lw, 6))
    monkeypatch.setattr(installer, "available", lambda: False)
    assert installer.module_weight_passes(module, 8) is None
    monkeypatch.setattr(installer, "available", lambda: True)
    monkeypatch.setattr(installer, "ENABLED", [False])
    assert installer.module_weight_passes(module, 8) is None


def test_contract_is_exported_from_the_lane_package():
    from mlx2.runtime import lane

    assert lane.weight_passes is matmul.weight_passes
    assert lane.module_weight_passes is installer.module_weight_passes
    assert lane.module_row_capacity is installer.module_row_capacity
    assert lane.installed_lane is installer.installed_lane
