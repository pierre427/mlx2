"""``inkling2_moe_panel_kernel`` and ``inkling2_moe_nax_mxfp8`` on Metal (GPU;
run under the GPU lock, never on a plain CPU pytest).

Panel kernel against the shipped ``moe_fast.fused_call`` (W4A16, the
reference) at B1/B2/B4/B8 (and 16/32) rows, NVFP4 and MXFP8 experts, three
geometries: the CPU tests' tiny width (hidden 128, inter 64, 8 experts,
top 2), the Mini's real shape (hidden 2304, inter 1152, 128 experts, top 8)
and a width where MLX takes ``gather_qmv_fast`` for both formats (hidden
1024, inter 512).  Every ``QDOT_CONTRACT`` x ``FAST_K_RULE`` variant is run
and recorded with its max |diff|; the module defaults must be bitwise
wherever the kernel admits the call.  B1 must decline (``rows``).  The tiny
NVFP4/MXFP8 model must engage on its 2-32 row chunks with logits bitwise.

NAX MXFP8 (M5 only): the canary self-test of every MXFP8 plan, the real
widths against the stock sorted ``gather_qmm`` (bitwise), and a prefill-sized
SparseMoE call through the owned route (engaged, bitwise).

Set ``INKLING2_MOE_PANEL_METAL_RECEIPT`` to a path for a JSON summary.
"""

import json
import os
from types import SimpleNamespace

import mlx.core as mx
import pytest

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")
RESULTS = {}
GEOMETRIES = {
    # name: (hidden, inter, experts, top_k)
    "tiny": (128, 64, 8, 2),
    "real": (2304, 1152, 128, 8),
    "fast": (1024, 512, 128, 8),
}


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.clear_cache()
        mx.set_default_device(previous)


def teardown_module(_module):
    path = os.environ.get("INKLING2_MOE_PANEL_METAL_RECEIPT")
    if path:
        with open(path, "w") as handle:
            json.dump(RESULTS, handle, indent=1, sort_keys=True)


def _moe(fmt, geometry, seed=0):
    from mlx2.runtime.models.inkling2.language import SparseMoE
    from mlx2.runtime.models.switch_layers import QuantizedSwitchLinear
    from mlx2.runtime.nvfp4_linear import ScaledNVFP4SwitchLinear

    hidden, inter, experts, top_k = GEOMETRIES[geometry]
    config = SimpleNamespace(
        num_experts_per_tok=top_k, route_scale=8.0, n_routed_experts=experts,
        hidden_size=hidden, intermediate_size=inter, mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0)
    mx.random.seed(seed)
    moe = SparseMoE(config)
    moe.gate_weight = (mx.random.normal((experts, hidden)) * 0.03).astype(mx.bfloat16)
    moe.e_score_correction_bias = mx.random.normal((experts,)) * 0.1
    switch = moe.switch_mlp
    w13_scale = mx.random.uniform(0.5, 2.0, (experts,)) * 1e-2
    for name, (k, n) in (("gate_proj", (hidden, inter)), ("up_proj", (hidden, inter)),
                         ("down_proj", (inter, hidden))):
        dense = mx.random.normal((experts, n, k)) * 0.05
        if fmt == "nvfp4":
            scale = w13_scale if name != "down_proj" else (
                mx.random.uniform(0.5, 2.0, (experts,)) * 1e-2)
            w, s = mx.quantize(dense / scale[:, None, None], group_size=16, bits=4,
                               mode="nvfp4")
            module = ScaledNVFP4SwitchLinear(k, n, experts)
            module.update({"weight": w, "scales": s,
                           "weight_scale_2": scale.astype(mx.float32)})
        else:
            w, s = mx.quantize(dense, group_size=32, bits=8, mode="mxfp8")
            module = QuantizedSwitchLinear(k, n, experts, bias=False, group_size=32, bits=8,
                                           mode="mxfp8")
            module.weight, module.scales, module.biases = w, s, None
            module.freeze()
        mx.eval(module.parameters())
        del dense
        setattr(switch, name, module)
    mx.eval(moe.parameters())
    return moe


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_panel_against_shipped(geometry, fmt, monkeypatch):
    """B1/B2/B4/B8/B16/B32, weighted sum on and off, 4 inputs each.

    ``PANEL_OVER_WEIGHTED_SUM`` is turned on so that the sorted multi-token
    rows are compared against the shipped ``qwen4_moe_weighted_sum`` combine
    (the module default declines them as ``weighted_sum_order_unproven``;
    checked too).  At the real and fast shapes both formats must engage at
    every width above one row."""
    from mlx2.runtime.models.inkling2 import moe_decode_kernel as mdk
    from mlx2.runtime.models.inkling2 import moe_fast
    from mlx2.runtime.models.inkling2 import moe_panel_kernel as mpk
    from mlx2.runtime.models.inkling2.language import _fp32

    moe = _moe(fmt, geometry)
    hidden, _, num_experts, top_k = GEOMETRIES[geometry]
    monkeypatch.setattr(mpk, "PANEL_OVER_WEIGHTED_SUM", True)
    failures = []
    for weighted_sum in (True, False):
        moe.set_fast_path(True, weighted_sum)
        experts, refusal = moe._fast_experts()
        assert refusal is None
        for rows in (1, 2, 4, 8, 16, 32):
            key = f"{geometry}/{fmt}/B{rows}/ws{int(weighted_sum)}"
            row = RESULTS.setdefault(key, {})
            for seed in range(4):
                x = (mx.random.normal((1, rows, hidden), key=mx.random.key(seed)) * 2
                     ).astype(mx.bfloat16)
                reason = mpk.decline_reason(moe, experts, x, weighted_sum)
                if rows == 1 or reason is not None:
                    # B1 declines (rows); only the tiny geometry (8 experts)
                    # may stream (rhs_route, from 16 rows). Recorded.
                    row["declined"] = reason
                    if rows == 1:
                        ok = reason == "rows"
                    else:
                        ok = geometry == "tiny" and reason == "rhs_route"
                    if not ok:
                        failures.append((key, reason))
                    break
                sorted_ws = weighted_sum and rows * top_k >= 20
                if sorted_ws:
                    # The module default must decline these.
                    monkeypatch.setattr(mpk, "PANEL_OVER_WEIGHTED_SUM", False)
                    default_reason = mpk.decline_reason(moe, experts, x, weighted_sum)
                    monkeypatch.setattr(mpk, "PANEL_OVER_WEIGHTED_SUM", True)
                    if default_reason != "weighted_sum_order_unproven":
                        failures.append((key, "default_not_declined", default_reason))
                counts = moe.__dict__.setdefault("moe_fast_counts", {})
                ws_before = counts.get("weighted_sum", 0)
                ref = moe_fast.fused_call(moe, experts, x, weighted_sum=weighted_sum)
                row["shipped_weighted_sum_engaged"] = counts.get("weighted_sum", 0) > ws_before
                if sorted_ws and not row["shipped_weighted_sum_engaged"]:
                    failures.append((key, "reference_not_weighted_sum"))
                order, scores = mpk.route_compiled(x, _fp32(moe, "gate_weight"),
                                                   moe.e_score_correction_bias, moe.top_k)
                law = mpk.law(moe, experts, x, order, scores)
                y, (plan, _, _) = mpk.launch(moe, experts, x, order, scores,
                                             return_parts=True)
                distinct, _ = mpk.plan_reference(order, top_k, num_experts)
                got_plan = plan.tolist()
                if got_plan[0] != len(distinct) or got_plan[1:1 + len(distinct)] != distinct:
                    failures.append((key, "plan"))
                stats = row.setdefault("plan", {"pairs": rows * top_k, "distinct": []})
                stats["distinct"].append(len(distinct))
                for contract in mdk._CONTRACTS:
                    for rule in mpk._FAST_RULES:
                        out = mpk.launch(moe, experts, x, order, scores, contract=contract,
                                         fast_rule=rule)
                        cell = row.setdefault(f"{contract}/{rule}",
                                              {"bitwise": 0, "max_diff": 0.0, "calls": 0})
                        cell["calls"] += 1
                        cell["bitwise"] += int(bool(mx.array_equal(out, ref)))
                        diff = mx.abs(out.astype(mx.float32)
                                      - ref.astype(mx.float32)).max().item()
                        cell["max_diff"] = max(cell["max_diff"], diff)
                cell = row.setdefault("default", {"bitwise": 0, "max_diff": 0.0, "calls": 0,
                                                  "law_bitwise": 0})
                cell["calls"] += 1
                cell["bitwise"] += int(bool(mx.array_equal(y, ref)))
                cell["law_bitwise"] += int(bool(mx.array_equal(law, ref)))
                cell["max_diff"] = max(cell["max_diff"], mx.abs(
                    y.astype(mx.float32) - ref.astype(mx.float32)).max().item())
            default = row.get("default")
            if default is not None and default["bitwise"] != default["calls"]:
                failures.append((key, default))
    del moe
    mx.clear_cache()
    assert not failures, failures


def test_tiny_model_panels_engage_and_are_bitwise():
    from test_inkling2_cpu import _run
    from test_inkling2_nonbitwise_cpu import _quantized_model

    model = _quantized_model()
    tokens = mx.random.randint(0, 500, (1, 30), key=mx.random.key(7))
    # 7 / 10 / 2 / 4 / 6-row chunks (verify-panel widths), then decode.
    plan = [(0, 7), (7, 17), (17, 19), (19, 23), (23, 29), (29, 30)]
    model.configure_kernels(moe_fused=True)
    base = _run(model, tokens, plan)
    model.configure_kernels(moe_fused=True, moe_panel_kernel=True)
    got = _run(model, tokens, plan)
    receipt = model.kernel_receipt()
    RESULTS["tiny_model"] = {"calls": receipt["moe_panel_kernel_calls"],
                             "formats": receipt["moe_panel_kernel_formats"],
                             "declines": receipt["moe_panel_kernel_declines"],
                             "bitwise": bool(mx.array_equal(got, base)),
                             "max_diff": mx.abs(got - base).max().item()}
    assert receipt["moe_panel_kernel_formats"] == {"mxfp8": 4 * 5, "nvfp4": 44 * 5}
    assert receipt["moe_panel_kernel_declines"] == {"rows": 48}
    assert mx.array_equal(got, base)


# ----- NAX MXFP8 (M5) -----


def _nax_or_skip():
    from mlx2.runtime.models import moe_nax_gather as nax

    if not nax.nax_host():
        pytest.skip("NAX (M5) required")
    return nax


def test_nax_mxfp8_canary_per_plan():
    nax = _nax_or_skip()
    verdicts = {}
    for plan in (nax.Plan(nax._SCHED_SEG, 64, 64, 0, 0), nax.Plan(nax._SCHED_DB, 64, 64, 0, 0),
                 nax.Plan(nax._SCHED_DB, 64, 64, nax._GX, 0),
                 nax.Plan(nax._SCHED_DB, 96, 64, nax._GX, 0),
                 nax.Plan(nax._SCHED_SEG, 96, 128, nax._GX, 0),
                 nax.Plan(nax._SCHED_SEG, 128, 128, nax._GX, 8192)):
        for align_n in ((True,) if plan.sched == nax._SCHED_DB else (True, False)):
            key = (mx.bfloat16, "mxfp8", 8, 32, plan, align_n, True)
            verdicts[nax._describe(key)] = bool(nax._self_test(key))
    RESULTS["nax_mxfp8_canary"] = verdicts
    assert all(verdicts.values()), verdicts


@pytest.mark.parametrize("n,k", [(2304, 2304), (2304, 1152)])
def test_nax_mxfp8_real_widths_bitwise(n, k):
    nax = _nax_or_skip()
    experts, rows = 128, 4096  # a 512-token chunk at top 8
    w, s = mx.quantize(mx.random.normal((experts, n, k), key=mx.random.key(1)) * 0.05,
                       group_size=32, bits=8, mode="mxfp8")
    idx = mx.sort(mx.random.randint(0, experts, (rows,), key=mx.random.key(2))).astype(mx.uint32)
    x = (mx.random.normal((rows, 1, k), key=mx.random.key(3))).astype(mx.bfloat16)
    old = nax.set_mode("gather")
    try:
        got = nax.sorted_gather_qmm(x, w, s, None, idx, group_size=32, bits=8, mode="mxfp8",
                                    allow_mxfp8=True)
    finally:
        nax.set_mode(old)
    ref = mx.gather_qmm(x, w, s, None, rhs_indices=idx, transpose=True, group_size=32,
                        bits=8, mode="mxfp8", sorted_indices=True)
    RESULTS[f"nax_mxfp8_{n}x{k}"] = {
        "engaged": got is not None,
        "bitwise": None if got is None else bool(mx.array_equal(got, ref)),
        "max_diff": None if got is None else mx.abs(
            got.astype(mx.float32) - ref.astype(mx.float32)).max().item()}
    assert got is not None
    assert mx.array_equal(got, ref)


def test_nax_mxfp8_moe_prefill_route_engages_and_is_bitwise():
    from mlx2.runtime.models.inkling2.language import _expert_switches

    nax = _nax_or_skip()
    moe = _moe("mxfp8", "real")
    moe.set_fast_path(True, True)
    x = (mx.random.normal((1, 512, 2304), key=mx.random.key(5))).astype(mx.bfloat16)
    old = nax.set_mode("gather")
    try:
        nax.status(reset=True)
        with nax.prefill_scope():
            ref = moe(x)
            mx.eval(ref)
            assert not nax.status()["calls_by_format"]
            for module in _expert_switches(moe):
                object.__setattr__(module, "nax_mxfp8_owned", True)
            got = moe(x)
            mx.eval(got)
        status = nax.status()
    finally:
        for module in _expert_switches(moe):
            object.__setattr__(module, "nax_mxfp8_owned", False)
        nax.set_mode(old)
    RESULTS["nax_mxfp8_moe"] = {"calls_by_format": status["calls_by_format"],
                                "fallbacks": status["fallbacks"],
                                "bitwise": bool(mx.array_equal(got, ref))}
    assert status["calls_by_format"].get("gather/mxfp8-8", 0) == 2
    assert mx.array_equal(got, ref)
