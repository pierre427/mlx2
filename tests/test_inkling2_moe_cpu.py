"""Inkling 2 opt-in MoE fast paths (``inkling2_moe_*``), CPU only.

The fused route (compiled routing, stacked gate/up gather, fused scale +
softcapped SwiGLU, fused down scale + combine) must be bit-identical to the
reference ``SparseMoE`` on the tiny model, for dense, NVFP4 and MXFP8
experts, at prefill (sorted gather) and decode (unsorted).  The GPU-only
pieces (weighted-sum kernel, NAX gather) are checked to decline on the CPU
without changing results; tests/test_inkling2_moe_metal.py covers them.
"""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from test_inkling2_cpu import _modelopt_nvfp4, _run, _tiny_model  # noqa: F401  (fixture below)


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def _quantize_experts(model, *, scale_gap=False):
    """NVFP4 experts on layers 0-19 and MXFP8 on 20-23, as in the checkpoint."""
    from mlx2.adapters.inkling2 import MXFP8, NVFP4, install_experts

    num, hidden, inter = 8, 128, 64
    layers, kinds = {}, {}
    for i in range(24):
        mx.random.seed(200 + i)
        w13 = mx.random.normal((num, 2 * inter, hidden)) * 0.05
        w2 = mx.random.normal((num, hidden, inter)) * 0.05
        if i < 20:
            a, b = _modelopt_nvfp4(w13, num), _modelopt_nvfp4(w2, num)
            layers[i] = {
                "w13_weight": a["weight"], "w13_weight_scale": a["scale"],
                "w13_weight_amax": a["amax"],
                "w13_weight_outer_scale_denominator": a["denominator"],
                "w2_weight": b["weight"], "w2_weight_scale": b["scale"],
                "w2_weight_amax": b["amax"], "w2_weight_outer_scale_denominator": b["denominator"],
            }
            kinds[i] = NVFP4
        else:
            q13, s13 = mx.quantize(w13, group_size=32, bits=8, mode="mxfp8")
            q2, s2 = mx.quantize(w2, group_size=32, bits=8, mode="mxfp8")
            layers[i] = {"w13_weight": q13.view(mx.uint8), "w13_weight_scale": s13,
                         "w2_weight": q2.view(mx.uint8), "w2_weight_scale": s2}
            kinds[i] = MXFP8
    install_experts(model, layers, kinds)
    if scale_gap:
        up = model.model.layers[0].mlp.switch_mlp.up_proj
        up.weight_scale_2 = up.weight_scale_2 * 2


def _plan_case(rows=2):
    # 12-token chunk: 12 * rows * top-2 >= 20 assignments, sorted gather;
    # one-token steps: unsorted.
    tokens = mx.random.randint(0, 500, (rows, 14))
    return tokens, [(0, 12), (12, 13), (13, 14)]


def test_policy_parsing_defaults_on_and_dependencies():
    """Default on (Pierre 2026-11-21): fused and weighted sum on, NAX auto
    (None, resolved to the host at load); weighted sum follows fused."""
    from mlx2.adapters.inkling2 import validate_policy

    kernels, _ = validate_policy(None)
    assert (kernels["moe_fused"], kernels["moe_weighted_sum"], kernels["moe_nax_gather"]) == (
        True, True, None)
    kernels, _ = validate_policy({"inkling2_moe_fused": False})
    assert (kernels["moe_fused"], kernels["moe_weighted_sum"]) == (False, False)
    kernels, _ = validate_policy({"inkling2_moe_fused": False, "inkling2_moe_weighted_sum": False,
                                  "inkling2_moe_nax_gather": False})
    assert not (kernels["moe_fused"] or kernels["moe_weighted_sum"] or kernels["moe_nax_gather"])
    with pytest.raises(ValueError, match="requires inkling2_moe_fused"):
        validate_policy({"inkling2_moe_fused": False, "inkling2_moe_weighted_sum": True})
    for key in ("inkling2_moe_fused", "inkling2_moe_nax_gather", "inkling2_sink_sdpa"):
        with pytest.raises(TypeError):
            validate_policy({key: None})
    kernels, _ = validate_policy({"inkling2_moe_fused": True, "inkling2_moe_weighted_sum": True,
                                  "inkling2_moe_nax_gather": True})
    assert kernels["moe_fused"] and kernels["moe_weighted_sum"] and kernels["moe_nax_gather"]
    with pytest.raises(TypeError):
        validate_policy({"inkling2_moe_fused": "yes"})


def test_feature_grid_lists_the_options():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts/build_model_feature_grid.py"
    text = path.read_text()
    for name in ("inkling2_moe_fused", "inkling2_moe_weighted_sum", "inkling2_moe_nax_gather"):
        assert f'"{name}"' in text
    assert importlib.util.spec_from_file_location("grid", path) is not None


def test_default_is_reference_and_builds_nothing():
    model = _tiny_model()
    model.configure_kernels()
    assert not model.kernel_policy["moe_fused"]
    assert all("_fast_built" not in layer.mlp.__dict__ for layer in model.model.layers)
    tokens, plan = _plan_case()
    _run(model, tokens, plan)
    receipt = model.kernel_receipt()
    assert receipt["moe_fused_calls"] == 0 and receipt["moe_fused_layers"] == 0
    assert "moe_nax_gather" not in receipt
    with pytest.raises(ValueError, match="requires moe_fused"):
        model.configure_kernels(moe_weighted_sum=True)


@pytest.mark.parametrize("quantized,dtype", [
    (False, mx.float32), (True, mx.float32), (True, mx.bfloat16)])
def test_fused_moe_is_bit_identical_prefill_and_decode(quantized, dtype):
    model = _tiny_model()
    if quantized:
        _quantize_experts(model)
    if dtype != mx.float32:
        model.set_dtype(dtype)
    tokens, plan = _plan_case()
    base = _run(model, tokens, plan)
    before = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    sorts_before = model.model.layers[0].mlp.switch_mlp.sort_status()["counts"]
    model.configure_kernels(moe_fused=True)
    fused = _run(model, tokens, plan)
    assert mx.array_equal(fused, base)
    receipt = model.kernel_receipt()
    assert receipt["moe_fused_calls"] == 48 * len(plan)
    assert receipt["moe_fused_layers"] == 24 and receipt["moe_fused_refusals"] == []
    assert receipt["moe_fast_declines"] == {}
    # The gate/up modules are row views of the stacked table: same tree.
    assert sum(v.nbytes for _, v in tree_flatten(model.parameters())) == before
    # The fused path keeps the switch's sort decisions and counters.
    sorts = model.model.layers[0].mlp.switch_mlp.sort_status()["counts"]
    assert sorts["auto_sorted"] > sorts_before["auto_sorted"]
    assert sorts["auto_unsorted"] > sorts_before["auto_unsorted"]
    # Off again: the reference path over the row views is unchanged.
    model.configure_kernels()
    assert mx.array_equal(_run(model, tokens, plan), base)
    model.configure_kernels(moe_fused=True)
    assert mx.array_equal(_run(model, tokens, plan), base)


def test_stacked_table_holds_both_halves_once():
    model = _tiny_model()
    _quantize_experts(model)
    switch = model.model.layers[3].mlp.switch_mlp
    gate = mx.dequantize(switch.gate_proj.weight, switch.gate_proj.scales, group_size=16,
                         bits=4, mode="nvfp4")
    model.configure_kernels(moe_fused=True)
    experts, reason = model.model.layers[3].mlp.__dict__["_fast_built"]
    assert reason is None
    stacked = mx.dequantize(experts.gate_up.weight, experts.gate_up.scales, group_size=16,
                            bits=4, mode="nvfp4")
    assert stacked.shape[1] == 2 * gate.shape[1]
    assert mx.array_equal(stacked[:, : gate.shape[1]], gate)
    assert experts.gate_up.scale_folded and experts.down.scale_folded
    # Only the private twins are folded; the reference modules still scale.
    assert not switch.gate_proj.scale_folded and not switch.down_proj.scale_folded
    assert experts.scales.shape == (8, 2)


def test_unequal_gate_up_scales_refuse_and_keep_reference():
    model = _tiny_model()
    _quantize_experts(model, scale_gap=True)
    tokens, plan = _plan_case()
    base = _run(model, tokens, plan)
    model.configure_kernels(moe_fused=True)
    assert mx.array_equal(_run(model, tokens, plan), base)
    receipt = model.kernel_receipt()
    assert receipt["moe_fused_refusals"] == ["gate_up_scales_differ"]
    assert receipt["moe_fused_layers"] == 23
    assert receipt["moe_fast_declines"] == {"declined:gate_up_scales_differ": len(plan)}


def test_rebuilds_when_expert_weights_are_replaced():
    model = _tiny_model()
    tokens, plan = _plan_case()
    model.configure_kernels(moe_fused=True)
    _run(model, tokens, plan)
    switch = model.model.layers[2].mlp.switch_mlp
    switch.down_proj.weight = switch.down_proj.weight * 1.5
    model.configure_kernels()
    base = _run(model, tokens, plan)
    model.configure_kernels(moe_fused=True)
    assert mx.array_equal(_run(model, tokens, plan), base)


def test_rebuilds_when_only_a_global_scale_is_replaced():
    model = _tiny_model()
    _quantize_experts(model)
    tokens, plan = _plan_case()
    model.configure_kernels(moe_fused=True)
    _run(model, tokens, plan)
    down = model.model.layers[4].mlp.switch_mlp.down_proj
    down.weight_scale_2 = down.weight_scale_2 * 3
    model.configure_kernels()
    base = _run(model, tokens, plan)
    model.configure_kernels(moe_fused=True)
    assert mx.array_equal(_run(model, tokens, plan), base)


def test_refused_build_is_retried_after_a_reload():
    model = _tiny_model()
    _quantize_experts(model, scale_gap=True)
    model.configure_kernels(moe_fused=True)
    assert model.kernel_receipt()["moe_fused_layers"] == 23
    switch = model.model.layers[0].mlp.switch_mlp
    switch.up_proj.weight_scale_2 = switch.gate_proj.weight_scale_2
    model.configure_kernels(moe_fused=True)
    tokens, plan = _plan_case()
    _run(model, tokens, plan)
    assert model.kernel_receipt()["moe_fused_layers"] == 24


def test_weighted_sum_declines_on_cpu_without_changing_output():
    model = _tiny_model()
    _quantize_experts(model)
    tokens, plan = _plan_case()
    base = _run(model, tokens, plan)
    model.configure_kernels(moe_fused=True, moe_weighted_sum=True)
    assert mx.array_equal(_run(model, tokens, plan), base)
    receipt = model.kernel_receipt()
    assert receipt["moe_weighted_sum_calls"] == 0
    # Only the sorted (prefill) chunk is a candidate: 48 visits x 1 chunk.
    assert receipt["moe_fast_declines"] == {"weighted_sum_declined:not_gpu": 48}


def test_nax_scope_is_inert_while_the_route_mode_is_off():
    from mlx2.runtime.models import moe_nax_gather

    assert moe_nax_gather.MODE == "off"
    model = _tiny_model()
    _quantize_experts(model)
    tokens, plan = _plan_case()
    base = _run(model, tokens, plan)
    model.configure_kernels(moe_nax_gather=True)
    assert mx.array_equal(_run(model, tokens, plan), base)
    nax = model.kernel_receipt()["moe_nax_gather"]
    assert nax["mode"] == "off" and sum(nax["calls"].values()) == 0


def test_nax_scope_marks_prefill_not_decode(monkeypatch):
    from mlx2.runtime.models import moe_nax_gather

    model = _tiny_model()
    seen = []
    layer = model.model.layers[0].mlp
    original = type(layer).__call__

    def spy(self, x):
        if self is layer:
            seen.append(moe_nax_gather.prefill_active())
        return original(self, x)

    monkeypatch.setattr(type(layer), "__call__", spy)
    monkeypatch.setattr(moe_nax_gather, "MODE", "gather")
    model.configure_kernels(moe_nax_gather=True)
    tokens, plan = _plan_case(rows=1)
    _run(model, tokens, plan)
    assert seen == [True, False, False]
    model.configure_kernels()
    seen.clear()
    _run(model, tokens, plan)
    assert seen == [False, False, False]
