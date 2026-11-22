"""Inkling 2 MoE fast paths on Metal (GPU; run under the GPU lock).

Small expert banks at the Inkling 2 Mini geometry (hidden 2304, expert
width 1152, top 8; 16 experts instead of 128 so the tables stay small).
Every check is bitwise against the reference ``SparseMoE`` on the GPU.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")

HIDDEN, INTER, EXPERTS, TOP_K = 2304, 1152, 16, 8


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.clear_cache()
        mx.set_default_device(previous)


def _config(experts=EXPERTS):
    return SimpleNamespace(
        num_experts_per_tok=TOP_K, route_scale=8.0, n_routed_experts=experts,
        hidden_size=HIDDEN, intermediate_size=INTER, mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0)


def _moe(fmt, seed=0, experts=EXPERTS):
    from mlx2.runtime.models.inkling2.language import SparseMoE
    from mlx2.runtime.models.switch_layers import QuantizedSwitchLinear
    from mlx2.runtime.nvfp4_linear import ScaledNVFP4SwitchLinear

    mx.random.seed(seed)
    moe = SparseMoE(_config(experts))
    moe.gate_weight = (mx.random.normal((experts, HIDDEN)) * 0.03).astype(mx.bfloat16)
    moe.e_score_correction_bias = mx.random.normal((experts,)) * 0.1
    switch = moe.switch_mlp
    w13_scale = mx.random.uniform(0.5, 2.0, (experts,))
    for name, (k, n) in (("gate_proj", (HIDDEN, INTER)), ("up_proj", (HIDDEN, INTER)),
                         ("down_proj", (INTER, HIDDEN))):
        dense = mx.random.normal((experts, n, k)) * 0.05
        if fmt == "nvfp4":
            w, s = mx.quantize(dense, group_size=16, bits=4, mode="nvfp4")
            module = ScaledNVFP4SwitchLinear(k, n, experts)
            scale = w13_scale if name != "down_proj" else mx.random.uniform(0.5, 2.0, (experts,))
            module.update({"weight": w, "scales": s, "weight_scale_2": scale})
        else:
            w, s = mx.quantize(dense, group_size=32, bits=8, mode="mxfp8")
            module = QuantizedSwitchLinear(k, n, experts, bias=False, group_size=32, bits=8,
                                           mode="mxfp8")
            module.weight, module.scales, module.biases = w, s, None
            module.freeze()
        setattr(switch, name, module)
    mx.eval(moe.parameters())
    return moe


def _x(tokens, rows=1, seed=1):
    mx.random.seed(seed)
    return mx.random.normal((rows, tokens, HIDDEN)).astype(mx.bfloat16)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("rows,tokens", [(1, 1), (2, 1), (1, 7), (1, 512)])
@pytest.mark.parametrize("weighted_sum", [False, True])
def test_fused_moe_is_bitwise_reference(fmt, rows, tokens, weighted_sum):
    moe = _moe(fmt)
    x = _x(tokens, rows)
    ref = moe(x)
    mx.eval(ref)
    moe.set_fast_path(True, weighted_sum)
    got = moe(x)
    mx.eval(got)
    assert got.dtype == ref.dtype == mx.bfloat16 and got.shape == ref.shape
    assert mx.array_equal(got.view(mx.uint16), ref.view(mx.uint16))
    counts = moe.__dict__["moe_fast_counts"]
    assert counts["fused"] == 1
    # The sorted weighted sum admits only calls with >= 64 routed rows
    # (qwen4_moe_weighted_sum.MIN_ROUTED_ROWS); smaller ones decline to the
    # reference sum and record why.
    from mlx2.runtime.models.qwen4_moe_weighted_sum import MIN_ROUTED_ROWS

    engaged = weighted_sum and rows * tokens * TOP_K >= MIN_ROUTED_ROWS
    assert counts.get("weighted_sum", 0) == int(engaged)
    moe.set_fast_path(False)
    again = moe(x)
    assert mx.array_equal(again.view(mx.uint16), ref.view(mx.uint16))


def test_stacked_gather_equals_split_gathers():
    """The identity the fused route rests on, at both MLX kernel families:
    one-row ``gather_qmv`` (decode) and the sorted rhs kernel (prefill)."""
    from mlx2.runtime.models.switch_layers import _gather_sort

    moe = _moe("nvfp4")
    sw = moe.switch_mlp
    g, u = sw.gate_proj, sw.up_proj
    w = mx.concatenate([g.weight, u.weight], axis=1)
    s = mx.concatenate([g.scales, u.scales], axis=1)
    kw = dict(transpose=True, group_size=16, bits=4, mode="nvfp4")
    for tokens in (1, 512):
        x = mx.expand_dims(_x(tokens), (-2, -3))
        idx = mx.random.randint(0, EXPERTS, (1, tokens, TOP_K)).astype(mx.uint32)
        sort = tokens > 1
        if sort:
            x, idx, _ = _gather_sort(x, idx)
        both = mx.gather_qmm(x, w, s, None, rhs_indices=idx, sorted_indices=sort, **kw)
        gate = mx.gather_qmm(x, g.weight, g.scales, None, rhs_indices=idx, sorted_indices=sort,
                             **kw)
        up = mx.gather_qmm(x, u.weight, u.scales, None, rhs_indices=idx, sorted_indices=sort,
                           **kw)
        assert mx.array_equal(both[..., :INTER].view(mx.uint16), gate.view(mx.uint16))
        assert mx.array_equal(both[..., INTER:].view(mx.uint16), up.view(mx.uint16))


def test_row_view_tables_are_read_in_place():
    """After stacking, the reference gate/up modules read row views of the
    stacked table; ``gather_qmm`` must not copy them per call."""
    moe = _moe("nvfp4", experts=64)
    moe.set_fast_path(True)
    gate = moe.switch_mlp.gate_proj
    x = mx.expand_dims(_x(1), (-2, -3))
    idx = mx.array([[[0, 3, 9, 17, 25, 33, 41, 63]]], dtype=mx.uint32)
    mx.eval(gate(x, idx))
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    mx.eval(gate(x, idx))
    grown = mx.get_peak_memory() - base
    view_bytes = gate.weight.nbytes + gate.scales.nbytes
    assert grown < view_bytes // 8, (grown, view_bytes)


def test_nax_gather_route_is_bitwise_and_engages():
    from mlx2.runtime.models import moe_nax_gather as nax

    if not nax.nax_host():
        pytest.skip("NAX gather needs an M5 device")
    moe = _moe("nvfp4")
    x = _x(512)
    ref = moe(x)
    mx.eval(ref)
    old = nax.set_mode("gather")
    try:
        nax.status(reset=True)
        with nax.prefill_scope(True):
            split = moe(x)
            mx.eval(split)
            moe.set_fast_path(True, True)
            fused = moe(x)
            mx.eval(fused)
        status = nax.status()
    finally:
        nax.set_mode(old)
        moe.set_fast_path(False)
    assert mx.array_equal(split.view(mx.uint16), ref.view(mx.uint16))
    assert mx.array_equal(fused.view(mx.uint16), ref.view(mx.uint16))
    # Split: gate, up, down; fused: gate_up, down.
    assert status["calls_by_format"].get("gather/nvfp4-4") == 5, status
