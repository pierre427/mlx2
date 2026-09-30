"""Row-tiled composed quantized SDPA (mlx-lm#1929's idea; default off).

Above ``MLX2_QSDPA_SCORES_BUDGET_BYTES`` the composed path runs its query rows
in balanced tiles so it never holds the whole ``B * Hq * L * S`` score block.
On the CPU every tile is bitwise equal to the untiled path. The opt-in Metal
oracles check parity and bounded live allocations. This remains a default-off
memory option; synthetic qualification does not select a serving route.
"""

import os

import mlx.core as mx
import pytest

from mlx2.runtime.models import base
from mlx2.runtime.models import qsdpa_verify_metal as qvm
from mlx2.runtime.models.base import quantized_scaled_dot_product_attention as qsdpa


def _inputs(B, Hq, Hkv, L, S, dtype, bits, seed=0, D=64):
    mx.random.seed(seed)
    q = mx.random.normal((B, Hq, L, D)).astype(dtype)
    k = mx.random.normal((B, Hkv, S, D)).astype(dtype)
    v = mx.random.normal((B, Hkv, S, D)).astype(dtype)
    return (
        q,
        mx.quantize(k, group_size=64, bits=bits[0]),
        mx.quantize(v, group_size=64, bits=bits[1]),
    )


def _mask(kind, B, Hq, L, S):
    visible = mx.arange(S)[None, :] <= mx.arange(S - L, S)[:, None]
    if kind == "none":
        return None
    if kind == "causal":
        return "causal"
    if kind == "bool_2d":
        return visible
    if kind == "bool_per_head":
        return mx.broadcast_to(visible, (B, Hq, L, S))
    if kind == "additive_f32":
        return mx.where(visible, 0.0, -1e9)[None, None]
    if kind == "left_padded":
        padding = mx.arange(B) * 3
        return visible[None, None] & (mx.arange(S)[None, None, None] >= padding[:, None, None, None])
    if kind == "key_broadcast":
        return mx.arange(S) >= 2
    if kind == "query_broadcast":
        return (mx.arange(S) >= 2)[None, None, None]
    raise ValueError(kind)


def _run(budget, monkeypatch, *args, **kwargs):
    monkeypatch.setattr(base, "_QSDPA_SCORES_BUDGET", budget)
    out = qsdpa(*args, **kwargs)
    mx.eval(out)
    return out


GEOMETRIES = {
    # (B, Hq, Hkv, L): L exceeds twice the tile floor max(4, 32 // n_rep + 1).
    "mha": (1, 2, 2, 100),
    "gqa": (1, 8, 2, 60),
    "mqa": (2, 4, 1, 60),
    "batch": (3, 4, 2, 45),
}
MASKS = ["none", "causal", "bool_2d", "bool_per_head", "additive_f32",
         "left_padded", "query_broadcast", "key_broadcast"]


def test_each_tile_is_evaluated_before_the_next_is_constructed(monkeypatch):
    q, keys, values = _inputs(1, 8, 2, 60, 257, mx.bfloat16, (8, 8))
    monkeypatch.setattr(base, "_QSDPA_SCORES_BUDGET", 1)
    compose = base._composed_qsdpa
    evaluate = mx.eval
    pending = []
    evaluations = []

    def tracked_compose(*args, **kwargs):
        assert not pending, "previous tile still lazy when constructing the next"
        out = compose(*args, **kwargs)
        pending.append(out)
        return out

    def tracked_eval(*arrays):
        if pending and any(array is pending[0] for array in arrays):
            evaluations.append(pending.pop().shape[-2])
        return evaluate(*arrays)

    monkeypatch.setattr(base, "_composed_qsdpa", tracked_compose)
    monkeypatch.setattr(mx, "eval", tracked_eval)
    out = qsdpa(q, keys, values, scale=0.125, mask="causal", group_size=64, bits=8)
    evaluate(out)
    assert not pending
    assert evaluations == base._scores_tile_rows(q, keys, "causal", 4)


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
@pytest.mark.parametrize("bits", [(8, 8), (4, 4), (8, 4)])
@pytest.mark.parametrize("mask_kind", MASKS)
def test_tiled_rows_are_bitwise_equal_to_the_untiled_path(
    monkeypatch, geometry, dtype, bits, mask_kind
):
    B, Hq, Hkv, L = GEOMETRIES[geometry]
    S = 257
    q, keys, values = _inputs(B, Hq, Hkv, L, S, dtype, bits)
    mask = _mask(mask_kind, B, Hq, L, S)
    kwargs = dict(scale=0.125, mask=mask, group_size=64, key_bits=bits[0], value_bits=bits[1])
    before = dict(qvm.STATS)
    whole = _run(0, monkeypatch, q, keys, values, **kwargs)
    assert qvm.STATS.get("composed_tiled_calls", 0) == before.get("composed_tiled_calls", 0)
    tiled = _run(1, monkeypatch, q, keys, values, **kwargs)
    assert qvm.STATS["composed_tiled_calls"] == before.get("composed_tiled_calls", 0) + 1
    assert tiled.shape == whole.shape == (B, Hq, L, 64)
    assert tiled.dtype == whole.dtype
    assert mx.array_equal(tiled, whole, equal_nan=True).item()


def _per_row(B, Hq, S, itemsize):
    return B * Hq * S * itemsize


def _spy(monkeypatch):
    rows = []
    real = mx.quantized_matmul

    def spy(x, *args, **kwargs):
        if kwargs.get("transpose", True):
            rows.append(x.shape[-2])
        return real(x, *args, **kwargs)

    monkeypatch.setattr(mx, "quantized_matmul", spy)
    return rows


def test_budget_boundary_engages_exactly_above_the_score_block(monkeypatch):
    """Falsifiers for the gate: one byte over the block tiles, the block itself
    does not, and the planned balanced tiles are the matmuls that run."""
    B, Hq, Hkv, L, S = 1, 8, 2, 60, 257
    q, keys, values = _inputs(B, Hq, Hkv, L, S, mx.bfloat16, (8, 8))
    kwargs = dict(scale=0.125, mask="causal", group_size=64, key_bits=8, value_bits=8)
    block = L * _per_row(B, Hq, S, 2)
    whole = _run(0, monkeypatch, q, keys, values, **kwargs)

    rows = _spy(monkeypatch)
    at_block = _run(block, monkeypatch, q, keys, values, **kwargs)
    assert rows == [4 * L]  # one folded score product: n_rep * L rows
    assert mx.array_equal(at_block, whole).item()

    rows.clear()
    monkeypatch.setattr(base, "_QSDPA_SCORES_BUDGET", block - 1)
    plan = base._scores_tile_rows(q, keys, "causal", 4)
    assert plan is not None and sum(plan) == L and len(plan) >= 2
    over = _run(block - 1, monkeypatch, q, keys, values, **kwargs)
    assert rows == [4 * count for count in plan]
    assert mx.array_equal(over, whole).item()


def test_tile_plan_is_balanced_and_respects_the_floor(monkeypatch):
    monkeypatch.setattr(base, "_QSDPA_SCORES_BUDGET", 1)
    for n_rep, Hq, L in ((1, 2, 100), (4, 8, 60), (8, 8, 21), (8, 8, 9)):
        q = mx.zeros((1, Hq, L, 64), dtype=mx.bfloat16)
        keys = mx.quantize(mx.zeros((1, Hq // n_rep, 300, 64)), group_size=64, bits=8)
        floor = max(4, 32 // n_rep + 1)
        plan = base._scores_tile_rows(q, keys, None, n_rep)
        if L < 2 * floor:
            assert plan is None
            continue
        assert sum(plan) == L and min(plan) >= floor and max(plan) - min(plan) <= 1
        assert min(plan) * n_rep > 32
    # A mask whose row axis is neither 1 nor L is not sliced: untiled.
    q = mx.zeros((1, 8, 60, 64), dtype=mx.bfloat16)
    keys = mx.quantize(mx.zeros((1, 2, 300, 64)), group_size=64, bits=8)
    assert base._scores_tile_rows(q, keys, mx.ones((1, 1, 2, 300), dtype=mx.bool_), 4) is None
    monkeypatch.setattr(base, "_QSDPA_SCORES_BUDGET", 0)
    assert base._scores_tile_rows(q, keys, None, 4) is None


def test_default_budget_is_off_and_the_caller_queries_are_untouched(monkeypatch):
    if "MLX2_QSDPA_SCORES_BUDGET_BYTES" not in os.environ:
        assert base._QSDPA_SCORES_BUDGET == 0
    q, keys, values = _inputs(1, 8, 2, 60, 257, mx.bfloat16, (8, 8))
    snapshot = mx.array(q)
    _run(1, monkeypatch, q, keys, values, scale=0.125, mask="causal",
         group_size=64, key_bits=8, value_bits=8)
    assert mx.array_equal(q, snapshot).item()


@pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 (under the GPU lock) for the Metal oracle",
)
def test_metal_tiles_are_bitwise_equal_at_a_qwen_shape(monkeypatch):
    with mx.stream(mx.gpu):
        q, keys, values = _inputs(1, 24, 4, 120, 20480, mx.bfloat16, (8, 8), D=256)
        kwargs = dict(scale=0.0625, mask="causal", group_size=64, key_bits=8, value_bits=8)
        whole = _run(0, monkeypatch, q, keys, values, **kwargs)
        tiled = _run(1, monkeypatch, q, keys, values, **kwargs)
    assert mx.array_equal(tiled, whole).item()


@pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 (under the GPU lock) for the memory oracle",
)
def test_metal_tiles_reduce_peak_live_allocation(monkeypatch):
    with mx.stream(mx.gpu):
        # Below the GQA flash threshold, so this exercises composed tiling.
        q, keys, values = _inputs(1, 24, 4, 120, 32768, mx.bfloat16, (8, 8), D=256)
        mx.eval(q, keys, values)
        kwargs = dict(scale=0.0625, mask="causal", group_size=64, key_bits=8, value_bits=8)
        mx.clear_cache()
        mx.reset_peak_memory()
        whole = _run(0, monkeypatch, q, keys, values, **kwargs)
        whole_peak = mx.get_peak_memory()
        mx.clear_cache()
        mx.reset_peak_memory()
        before = qvm.STATS.get("composed_tiled_calls", 0)
        tiled = _run(16 << 20, monkeypatch, q, keys, values, **kwargs)
        tiled_peak = mx.get_peak_memory()
        assert qvm.STATS["composed_tiled_calls"] == before + 1
        assert mx.array_equal(tiled, whole).item()
        # The unfixed lazy concatenate retains all score DAGs and fails this.
        assert tiled_peak < 0.75 * whole_peak, (tiled_peak, whole_peak)
