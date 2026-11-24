"""``inkling2_moe_panel_kernel`` and ``inkling2_moe_nax_mxfp8``, CPU only.

The Metal kernels run on the GPU only (tests/test_inkling2_moe_panel_kernel_metal.py).
Here: both options default off; off is the shipped route bitwise and never
asks the kernels; on, the CPU declines every call (counted) and stays the
shipped route bitwise; with the device gate and the launch patched to the
kernels' MLX-ops law, the plumbing engages on 2-32 row calls of both expert
formats (NVFP4 layers 0-19, MXFP8 layers 20-23), declines one-token calls,
33+ rows and calls the shipped route would stream (counted), and the logits
stay bitwise.  The plan's Python reference and the MLX dispatch rules the
kernels follow are checked directly.  The NAX MXFP8 loader: admission only
for owned tables, every other MXFP8 caller refused as before.
"""

import mlx.core as mx
import pytest
from test_inkling2_cpu import _run, _tiny_model
from test_inkling2_nonbitwise_cpu import _quantized_model

from mlx2.runtime.models import moe_nax_gather as nax
from mlx2.runtime.models.inkling2 import moe_panel_kernel as mpk

NVFP4_VISITS = 6 + 12 * 3 + 2  # layers 0-5, the looped 6-17 three times, 18-19
MXFP8_VISITS = 4  # layers 20-23


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def _panel_keys(receipt):
    keys = {"panel_kernel_declined:" + k: v
            for k, v in receipt["moe_panel_kernel_declines"].items()}
    keys.update({"panel_kernel:" + k: v for k, v in receipt["moe_panel_kernel_formats"].items()})
    if receipt["moe_panel_kernel_calls"]:
        keys["panel_kernel"] = receipt["moe_panel_kernel_calls"]
    return keys


def _law_on_cpu(monkeypatch):
    monkeypatch.setattr(mpk, "device_ok", lambda: True)
    monkeypatch.setattr(mpk, "launch", lambda moe, experts, x, order, scores, **_: mpk.law(
        moe, experts, x, order, scores))


# ----- policy -----


def test_defaults_are_off():
    from mlx2.adapters.inkling2 import KERNEL_DEFAULTS, validate_policy

    kernels, _ = validate_policy(None)
    assert kernels["moe_panel_kernel"] is False
    assert kernels["moe_nax_mxfp8"] is False
    assert "moe_panel_kernel" not in KERNEL_DEFAULTS and "moe_nax_mxfp8" not in KERNEL_DEFAULTS
    model = _tiny_model()
    model.configure_kernels(moe_fused=True)
    assert model.kernel_policy["moe_panel_kernel"] is False
    assert model.kernel_policy["moe_nax_mxfp8"] is False
    assert not any(layer.mlp.__dict__.get("_moe_panel_kernel") for layer in model.model.layers)


def test_policy_validation():
    from mlx2.adapters.inkling2 import validate_policy

    assert validate_policy({"inkling2_moe_panel_kernel": True})[0]["moe_panel_kernel"] is True
    with pytest.raises(ValueError, match="moe_panel_kernel requires inkling2_moe_fused"):
        validate_policy({"inkling2_moe_panel_kernel": True, "inkling2_moe_fused": False})
    with pytest.raises(TypeError):
        validate_policy({"inkling2_moe_panel_kernel": 1})
    # Unset NAX gather resolves at load; an explicit false refuses here.
    assert validate_policy({"inkling2_moe_nax_mxfp8": True})[0]["moe_nax_mxfp8"] is True
    with pytest.raises(ValueError, match="requires inkling2_moe_nax_gather"):
        validate_policy({"inkling2_moe_nax_mxfp8": True, "inkling2_moe_nax_gather": False})
    model = _tiny_model()
    with pytest.raises(ValueError, match="moe_panel_kernel requires moe_fused"):
        model.configure_kernels(moe_panel_kernel=True)
    # NAX gather off (e.g. unset on a non-M5 host): a counted decline, no failure.
    model.configure_kernels(moe_fused=True, moe_nax_mxfp8=True)
    assert model.kernel_policy["moe_nax_mxfp8"] is False
    receipt = model.kernel_receipt()["moe_nax_mxfp8"]
    assert receipt["declines"] == {"no_nax_host": 1} and receipt["owned_live_tables"] == 0


def test_numerics_contract_names_the_sources():
    from mlx2.adapters.inkling2 import Inkling2Adapter

    adapter = Inkling2Adapter.__new__(Inkling2Adapter)
    adapter.kernel_policy = {"moe_fused": True, "moe_panel_kernel": True}
    contract = adapter.execution_numerics_contract()
    assert "inkling2/moe_panel_kernel.py" in contract["source_sha256"]
    assert "inkling2/moe_decode_kernel.py" in contract["source_sha256"]
    adapter.kernel_policy = {"moe_fused": True}
    assert "inkling2/moe_panel_kernel.py" not in (
        adapter.execution_numerics_contract()["source_sha256"])


# ----- panel kernel plumbing -----


def test_option_off_never_launches_even_where_it_could(monkeypatch):
    model = _quantized_model()
    tokens = mx.random.randint(0, 500, (1, 24), key=mx.random.key(1))
    plan = [(0, 7), (7, 17), (17, 18), (18, 22), (22, 24)]
    model.configure_kernels(moe_fused=True)
    base = _run(model, tokens, plan)
    monkeypatch.setattr(mpk, "device_ok", lambda: True)

    def refuse(*_args, **_kwargs):
        raise AssertionError("panel kernel asked with the option off")

    model.configure_kernels(moe_fused=True, moe_panel_kernel=True)
    model.configure_kernels(moe_fused=True, moe_panel_kernel=False)
    monkeypatch.setattr(mpk, "launch", refuse)
    monkeypatch.setattr(mpk, "route_compiled", refuse)
    monkeypatch.setattr(mpk, "decline_reason", refuse)
    assert mx.array_equal(_run(model, tokens, plan), base)
    receipt = model.kernel_receipt()
    assert receipt["moe_panel_kernel_calls"] == 0
    assert receipt["moe_panel_kernel_declines"] == {}
    assert receipt["moe_panel_kernel_formats"] == {}
    assert model.kernel_policy["moe_panel_kernel"] is False


def test_option_on_cpu_declines_every_call_and_is_bitwise():
    model = _quantized_model()
    tokens = mx.random.randint(0, 500, (1, 24), key=mx.random.key(2))
    plan = [(0, 7), (7, 17), (17, 18), (18, 22), (22, 24)]
    model.configure_kernels(moe_fused=True)
    base = _run(model, tokens, plan)
    model.configure_kernels(moe_fused=True, moe_panel_kernel=True)
    assert mx.array_equal(_run(model, tokens, plan), base)
    keys = _panel_keys(model.kernel_receipt())
    assert keys == {"panel_kernel_declined:not_gpu": 48 * len(plan)}
    assert model.kernel_policy["moe_panel_kernel"] is True


def test_plumbing_engages_on_panels_of_both_formats(monkeypatch):
    """The tiny model has 8 experts and top 2, so MLX's streaming floor
    (B >= 16 and B / E >= 4 sorted rows) and the MXFP8 rhs pad (3 rows per
    expert) are reached at 12-16 rows: those calls decline, counted."""
    model = _quantized_model()
    tokens = mx.random.randint(0, 500, (1, 50), key=mx.random.key(3))
    plan = [(0, 7), (7, 17), (17, 29), (29, 45), (45, 46), (46, 49), (49, 50)]
    model.configure_kernels(moe_fused=True)
    base = _run(model, tokens, plan)
    _law_on_cpu(monkeypatch)
    model.configure_kernels(moe_fused=True, moe_panel_kernel=True)
    got = _run(model, tokens, plan)
    keys = _panel_keys(model.kernel_receipt())
    # Engaged: 7, 10 and 3 rows (all 48 visits); 12 rows on the NVFP4 visits.
    engaged_nvfp4 = 4 * NVFP4_VISITS
    engaged_mxfp8 = 3 * MXFP8_VISITS
    assert keys["panel_kernel:nvfp4"] == engaged_nvfp4
    assert keys["panel_kernel:mxfp8"] == engaged_mxfp8
    assert keys["panel_kernel"] == engaged_nvfp4 + engaged_mxfp8
    # 12 rows: the MXFP8 pad streams; 16 rows: MLX streams for both formats.
    assert keys["panel_kernel_declined:rhs_route"] == MXFP8_VISITS + 48
    assert keys["panel_kernel_declined:rows"] == 2 * 48
    assert mx.array_equal(got, base)
    assert model.kernel_receipt()["moe_fused_calls"] >= keys["panel_kernel"]


def test_b2_and_b4_decode_engage(monkeypatch):
    """Batched one-token decode: B rows of one token each are a panel too."""
    for lanes in (2, 4):
        model = _quantized_model()
        tokens = mx.random.randint(0, 500, (lanes, 12), key=mx.random.key(4 + lanes))
        plan = [(0, 8)] + [(i, i + 1) for i in range(8, 12)]
        model.configure_kernels(moe_fused=True)
        base = _run(model, tokens, plan)
        _law_on_cpu(monkeypatch)
        model.configure_kernels(moe_fused=True, moe_panel_kernel=True)
        got = _run(model, tokens, plan)
        keys = _panel_keys(model.kernel_receipt())
        decode_steps = len(plan) - 1
        assert keys["panel_kernel"] == 48 * decode_steps, keys
        assert mx.array_equal(got, base)
        monkeypatch.undo()


def test_more_than_32_rows_and_dense_experts_decline(monkeypatch):
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 40), key=mx.random.key(6))
    plan = [(0, 33), (33, 36), (36, 40)]
    model.configure_kernels(moe_fused=True)
    base = _run(model, tokens, plan)
    _law_on_cpu(monkeypatch)
    model.configure_kernels(moe_fused=True, moe_panel_kernel=True)
    assert mx.array_equal(_run(model, tokens, plan), base)
    keys = _panel_keys(model.kernel_receipt())
    # FP32 activations decline before the expert format is looked at.
    assert keys == {"panel_kernel_declined:rows": 48, "panel_kernel_declined:dtype": 48 * 2}


def test_act_quant_declines(monkeypatch):
    model = _quantized_model()
    tokens = mx.random.randint(0, 500, (1, 10), key=mx.random.key(7))
    _law_on_cpu(monkeypatch)
    model.configure_kernels(moe_fused=True, moe_panel_kernel=True, moe_act_quant=True)
    _run(model, tokens, [(0, 6), (6, 10)])
    keys = _panel_keys(model.kernel_receipt())
    assert keys == {"panel_kernel_declined:act_quant": 2 * 48}


@pytest.mark.parametrize("layer", [3, 21])
@pytest.mark.parametrize("rows", [2, 3, 4, 8])
def test_law_matches_shipped_fused_call(layer, rows):
    """The law (rounding points, slot order, epilogues) against the shipped
    fused call per layer: NVFP4 (layer 3) and MXFP8 (layer 21)."""
    from mlx2.runtime.models.inkling2 import moe_fast
    from mlx2.runtime.models.inkling2.language import _fp32

    model = _quantized_model()
    model.configure_kernels(moe_fused=True)
    moe = model.model.layers[layer].mlp
    experts, refusal = moe._fast_experts()
    assert refusal is None
    assert mpk.expert_format(experts) == ("nvfp4" if layer < 20 else "mxfp8")
    for seed in range(6):
        x = (mx.random.normal((1, rows, model.config.hidden_size), key=mx.random.key(seed))
             ).astype(mx.bfloat16)
        ref = moe_fast.fused_call(moe, experts, x)
        order, scores = mpk.route_compiled(x, _fp32(moe, "gate_weight"),
                                           moe.e_score_correction_bias, moe.top_k)
        shipped_idx, _ = moe_fast.route_compiled(x, _fp32(moe, "gate_weight"),
                                                 moe.e_score_correction_bias, moe.top_k,
                                                 moe.route_scale)
        assert mx.array_equal(order[..., :moe.top_k], shipped_idx)
        got = mpk.law(moe, experts, x, order, scores)
        assert got.shape == ref.shape and got.dtype == ref.dtype
        assert mx.array_equal(got, ref)


# ----- plan and dispatch rules -----


@pytest.mark.parametrize("rows,top_k,experts", [(2, 8, 128), (4, 8, 128), (8, 8, 128),
                                                (32, 8, 128), (5, 2, 8)])
def test_plan_reference_covers_every_pair_once(rows, top_k, experts):
    keys = mx.random.normal((rows, experts), key=mx.random.key(rows * top_k))
    order = mx.argsort(-keys, axis=-1).astype(mx.uint32)
    distinct, members = mpk.plan_reference(order, top_k, experts)
    assert distinct == sorted(members)
    pairs = sorted(p for m in members.values() for p in m)
    assert pairs == list(range(rows * top_k))
    for e, pair_ids in members.items():
        assert len(pair_ids) <= rows
        assert all(int(order[p // top_k, p % top_k].item()) == e for p in pair_ids)
    assert len(distinct) <= min(experts, rows * top_k)


def test_fast_qmv_rule():
    # MLX gather_qmv: fast when N % 8 == 0 and K % ((32 / bits) * 2 * 32) == 0.
    assert not mpk.fast_qmv(4, 2304, 2304)  # NVFP4 gate/up, 2304 % 512
    assert not mpk.fast_qmv(4, 2304, 1152)  # NVFP4 down
    assert mpk.fast_qmv(8, 2304, 2304)  # MXFP8 gate/up: 2304 % 256 == 0
    assert not mpk.fast_qmv(8, 2304, 1152)  # MXFP8 down: 1152 % 256 == 128
    assert not mpk.fast_qmv(8, 2304, 2304, "fixed512")
    assert mpk.fast_qmv(4, 1024, 512) and mpk.fast_qmv(8, 1024, 512, "fixed512")
    assert not mpk.fast_qmv(8, 1028, 512)
    with pytest.raises(ValueError):
        mpk.fast_qmv(8, 8, 256, "other")


def test_decline_geometry_and_top_k(monkeypatch):
    from types import SimpleNamespace

    from mlx2.runtime.nvfp4_linear import ScaledNVFP4SwitchLinear

    monkeypatch.setattr(mpk, "device_ok", lambda: True)

    def fake(hidden, inter, num=32):
        return SimpleNamespace(gate_up=ScaledNVFP4SwitchLinear(hidden, 2 * inter, num),
                               down=ScaledNVFP4SwitchLinear(inter, hidden, num),
                               scales=mx.ones((num, 2)))

    switch = SimpleNamespace(_sort_policy="auto")
    for top_k, expected in ((1, None), (8, None), (9, "top_k"), (16, "top_k")):
        moe = SimpleNamespace(top_k=top_k, switch_mlp=switch)
        x = mx.zeros((1, 2, 576), mx.bfloat16)
        assert mpk.decline_reason(moe, fake(576, 192), x) == expected
    moe = SimpleNamespace(top_k=8, switch_mlp=switch)
    assert mpk.decline_reason(moe, fake(576, 192), mx.zeros((1, 1, 576), mx.bfloat16)) == "rows"
    assert mpk.decline_reason(moe, fake(576, 192), mx.zeros((1, 33, 576), mx.bfloat16)) == "rows"
    assert mpk.decline_reason(moe, fake(576, 192), mx.zeros((1, 4, 576), mx.float32)) == "dtype"
    assert mpk.decline_reason(moe, fake(576, 192, 4), mx.zeros((1, 2, 576), mx.bfloat16)) == "experts"
    # 32 rows x top 8 into 64 experts: 256 sorted rows, 4 per expert, MLX streams.
    assert (mpk.decline_reason(moe, fake(576, 192, 64), mx.zeros((1, 32, 576), mx.bfloat16))
            == "rhs_route")
    # The Mini at 32 rows (128 experts): 2 rows per expert, never streams.
    assert mpk.decline_reason(moe, fake(576, 192, 128), mx.zeros((1, 32, 576), mx.bfloat16)) is None


# ----- NAX MXFP8 loader admission -----


def _mxfp8_switch(E=8, out=128, inp=256):
    from mlx2.runtime.models.switch_layers import QuantizedSwitchLinear

    return QuantizedSwitchLinear(inp, out, E, bias=False, group_size=32, bits=8, mode="mxfp8")


def test_nax_supports_mxfp8_only_when_admitted():
    lin = _mxfp8_switch()
    x = mx.zeros((64, 1, 256), mx.bfloat16)
    idx = mx.zeros((64,), mx.uint32)
    args = (x, lin["weight"], lin["scales"], None, idx, 32, 8, "mxfp8")
    assert not nax.supports(*args)
    assert nax.supports(*args, allow_mxfp8=True)
    assert not nax.supports(*args, row_map=mx.zeros((64,), mx.uint32), allow_mxfp8=True)


def test_nax_try_gather_mxfp8_owned_only(monkeypatch):
    old = nax.MODE
    nax.status(reset=True)
    try:
        monkeypatch.setattr(nax, "_nax_host", True)
        monkeypatch.setattr(nax.mx, "default_device", lambda: nax.mx.gpu)
        lin = _mxfp8_switch()
        x = mx.zeros((64, 1, 256), mx.bfloat16)
        idx = mx.zeros((64,), mx.uint32)
        with nax.prefill_scope():
            assert nax.try_gather(x, lin, idx) is None
            assert nax.status()["fallbacks"] == {"unsupported_layout": 1}
            seen = {}

            def fake(*args, **kwargs):
                seen.update(kwargs)
                return "sentinel"

            monkeypatch.setattr(nax, "sorted_gather_qmm", fake)
            object.__setattr__(lin, "nax_mxfp8_owned", True)
            assert nax.try_gather(x, lin, idx) == "sentinel"
            assert seen["allow_mxfp8"] is True and seen["mode"] == "mxfp8"
            assert nax.status()["calls_by_format"] == {"gather/mxfp8-8": 1}
            # The swiglu epilogues keep refusing MXFP8 (no loader there).
            assert nax._launch(x, lin["weight"], lin["scales"], None, idx, 32, 8, "mxfp8",
                               nax.Plan(0, 64, 64, 0, 0), None, epi=1) is None
    finally:
        nax.set_mode(old)
        nax.status(reset=True)


def test_configure_kernels_owns_the_mxfp8_tables():
    model = _quantized_model()
    model.configure_kernels(moe_fused=True, moe_nax_gather=True, moe_nax_mxfp8=True)
    from mlx2.runtime.models.inkling2.language import _expert_switches

    owned = [getattr(m, "nax_mxfp8_owned", False)
             for layer in model.model.layers for m in _expert_switches(layer.mlp)
             if getattr(m, "mode", None) == "mxfp8"]
    # 4 MXFP8 layers x (gate, up, down + the stacked gate/up; down is shared).
    assert len(owned) == 4 * 5 and all(owned)
    nvfp4 = [getattr(m, "nax_mxfp8_owned", False)
             for layer in model.model.layers[:20] for m in _expert_switches(layer.mlp)]
    assert not any(nvfp4)
    receipt = model.kernel_receipt()
    # Live tables on the fused route only: stacked gate/up + down, 4 layers.
    assert receipt["moe_nax_gather"]["mxfp8_owned_tables"] == 8
    assert receipt["moe_nax_mxfp8"] == {"selected": True, "owned_live_tables": 8,
                                        "declines": {}}
    # A rebuild of the stacked table (a weight replaced) re-owns the new one.
    layer = model.model.layers[21].mlp
    old_table = layer._fast_experts()[0].gate_up
    up = layer.switch_mlp.up_proj
    up.weight = mx.array(up.weight)
    new_table = layer._fast_experts()[0].gate_up
    assert new_table is not old_table and new_table.nax_mxfp8_owned
    assert model.kernel_receipt()["moe_nax_mxfp8"]["owned_live_tables"] == 8
    model.configure_kernels(moe_fused=True, moe_nax_gather=True)
    assert model.kernel_receipt()["moe_nax_gather"]["mxfp8_owned_tables"] == 0
    assert model.kernel_policy["moe_nax_mxfp8"] is False


def test_weighted_sum_calls_decline_unless_switched(monkeypatch):
    """Sorted multi-token calls the shipped route combines through
    qwen4_moe_weighted_sum decline (counted) unless PANEL_OVER_WEIGHTED_SUM;
    the shipped weighted-sum counter stays the shipped route's."""
    model = _quantized_model()
    tokens = mx.random.randint(0, 500, (1, 17), key=mx.random.key(9))
    plan = [(0, 7), (7, 17)]  # 7 rows: unsorted; 10 rows: sorted (20 assignments)
    model.configure_kernels(moe_fused=True, moe_weighted_sum=True)
    base = _run(model, tokens, plan)
    _law_on_cpu(monkeypatch)
    model.configure_kernels(moe_fused=True, moe_weighted_sum=True, moe_panel_kernel=True)
    assert mx.array_equal(_run(model, tokens, plan), base)
    keys = _panel_keys(model.kernel_receipt())
    assert keys["panel_kernel"] == 48
    assert keys["panel_kernel_declined:weighted_sum_order_unproven"] == 48
    monkeypatch.setattr(mpk, "PANEL_OVER_WEIGHTED_SUM", True)
    model.configure_kernels(moe_fused=True, moe_weighted_sum=True, moe_panel_kernel=True)
    assert mx.array_equal(_run(model, tokens, plan), base)
    keys = _panel_keys(model.kernel_receipt())
    assert keys["panel_kernel"] == 48 * 3
    assert keys["panel_kernel_declined:weighted_sum_order_unproven"] == 48


def test_engaged_calls_have_a_sort_bucket(monkeypatch):
    model = _quantized_model()
    tokens = mx.random.randint(0, 500, (1, 7), key=mx.random.key(10))
    _law_on_cpu(monkeypatch)
    model.configure_kernels(moe_fused=True, moe_panel_kernel=True)
    _run(model, tokens, [(0, 7)])
    per_layer = [layer.mlp.switch_mlp._sort_counts.get("panel_kernel", 0)
                 for layer in model.model.layers]
    assert sum(per_layer) == 48
    assert sum(layer.mlp.switch_mlp._sort_counts["auto_unsorted"]
               for layer in model.model.layers) == 0
