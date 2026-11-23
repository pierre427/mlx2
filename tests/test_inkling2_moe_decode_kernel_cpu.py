"""``inkling2_moe_decode_kernel``, CPU only.

The Metal kernels run on the GPU only (tests/test_inkling2_moe_decode_kernel_metal.py
and scripts/research/inkling2_moe_decode_kernel_ab.py).  Here: off is the
shipped route bitwise; on, the CPU declines every call (counted) and is the
shipped route bitwise; with the device gate and the launch patched to the
kernels' MLX-ops law, the plumbing engages on every one-token NVFP4 call,
declines prefill, B2 and MXFP8 calls (counted), and stays within tolerance
of the shipped route.
"""

import mlx.core as mx
import pytest
from test_inkling2_cpu import _run, _tiny_model
from test_inkling2_nonbitwise_cpu import _quantized_model

from mlx2.runtime.models.inkling2 import moe_decode_kernel as mdk

PLAN = [(0, 20), (20, 27)] + [(i, i + 1) for i in range(27, 36)]
DECODE_STEPS = sum(1 for a, b in PLAN if b - a == 1)
PREFILL_CALLS = sum(1 for a, b in PLAN if b - a > 1)


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def _counts(model):
    return model.kernel_receipt()


def _decode_keys(receipt):
    keys = {"decode_kernel_declined:" + k: v
            for k, v in receipt["moe_decode_kernel_declines"].items()}
    if receipt["moe_decode_kernel_calls"]:
        keys["decode_kernel"] = receipt["moe_decode_kernel_calls"]
    return keys


def _law_on_cpu(monkeypatch):
    monkeypatch.setattr(mdk, "device_ok", lambda: True)
    monkeypatch.setattr(mdk, "launch", lambda moe, experts, x, order, scores, **_: mdk.law(
        moe, experts, x, order, scores))


def test_option_off_never_launches_even_where_it_could(monkeypatch):
    """Off after on, with the device gate open: the kernel is never asked,
    nothing is counted, and the logits are the shipped fused route's."""
    model = _quantized_model()
    tokens = mx.random.randint(0, 500, (1, 36), key=mx.random.key(1))
    model.configure_kernels(moe_fused=True)
    base = _run(model, tokens, PLAN)
    monkeypatch.setattr(mdk, "device_ok", lambda: True)

    def refuse(*_args, **_kwargs):
        raise AssertionError("launch called with the option off")

    model.configure_kernels(moe_fused=True, moe_decode_kernel=True)
    model.configure_kernels(moe_fused=True, moe_decode_kernel=False)
    monkeypatch.setattr(mdk, "launch", refuse)
    monkeypatch.setattr(mdk, "route_compiled", refuse)
    before = _counts(model)
    assert mx.array_equal(_run(model, tokens, PLAN), base)
    after = _counts(model)
    assert after["moe_decode_kernel_calls"] == before["moe_decode_kernel_calls"] == 0
    assert after["moe_decode_kernel_declines"] == before["moe_decode_kernel_declines"] == {}
    assert model.kernel_policy["moe_decode_kernel"] is False


def test_option_on_cpu_declines_every_call_and_is_bitwise():
    model = _quantized_model()
    tokens = mx.random.randint(0, 500, (1, 36), key=mx.random.key(2))
    model.configure_kernels(moe_fused=True)
    base = _run(model, tokens, PLAN)
    before = _decode_keys(_counts(model))
    model.configure_kernels(moe_fused=True, moe_decode_kernel=True)
    got = _run(model, tokens, PLAN)
    assert mx.array_equal(got, base)
    keys = _decode_keys(_counts(model))
    assert keys.get("decode_kernel", 0) == before.get("decode_kernel", 0) == 0
    assert keys["decode_kernel_declined:not_gpu"] == 48 * (DECODE_STEPS + PREFILL_CALLS)
    assert model.kernel_policy["moe_decode_kernel"] is True


def test_requires_moe_fused():
    model = _tiny_model()
    with pytest.raises(ValueError, match="moe_decode_kernel requires moe_fused"):
        model.configure_kernels(moe_decode_kernel=True)
    from mlx2.adapters.inkling2 import validate_policy

    assert validate_policy(None)[0]["moe_decode_kernel"] is False
    assert validate_policy({"inkling2_moe_decode_kernel": True})[0]["moe_decode_kernel"] is True
    with pytest.raises(ValueError, match="requires inkling2_moe_fused"):
        validate_policy({"inkling2_moe_decode_kernel": True, "inkling2_moe_fused": False})
    with pytest.raises(TypeError):
        validate_policy({"inkling2_moe_decode_kernel": 1})


def test_plumbing_engages_on_one_token_nvfp4_calls(monkeypatch):
    """Device gate and launch patched to the law: every one-token visit of
    the NVFP4 layers (0-19: 44 of 48 visits) takes the path, the MXFP8
    layers and the prefill chunks decline, counted; the law reproduces the
    shipped route on the CPU, so the logits are bitwise, and every engaged
    call is still counted as a fused-route call."""
    model = _quantized_model()
    tokens = mx.random.randint(0, 500, (1, 36), key=mx.random.key(3))
    model.configure_kernels(moe_fused=True)
    base = _run(model, tokens, PLAN)
    _law_on_cpu(monkeypatch)
    model.configure_kernels(moe_fused=True, moe_decode_kernel=True)
    got = _run(model, tokens, PLAN)
    keys = _decode_keys(_counts(model))
    visits_nvfp4 = 6 + 12 * 3 + 2  # layers 0-5, the looped 6-17 three times, 18-19
    assert keys["decode_kernel"] == visits_nvfp4 * DECODE_STEPS
    assert keys["decode_kernel_declined:expert_format"] == 4 * DECODE_STEPS
    assert keys["decode_kernel_declined:rows"] == 48 * PREFILL_CALLS
    assert mx.array_equal(got, base)
    receipt = _counts(model)
    assert receipt["moe_fused_calls"] >= 48 * DECODE_STEPS


def test_b2_decode_declines_rows(monkeypatch):
    model = _quantized_model()
    tokens = mx.random.randint(0, 500, (2, 24), key=mx.random.key(4))
    plan = [(0, 16)] + [(i, i + 1) for i in range(16, 24)]
    model.configure_kernels(moe_fused=True)
    base = _run(model, tokens, plan)
    _law_on_cpu(monkeypatch)
    model.configure_kernels(moe_fused=True, moe_decode_kernel=True)
    assert mx.array_equal(_run(model, tokens, plan), base)
    keys = _decode_keys(_counts(model))
    assert "decode_kernel" not in keys
    assert keys["decode_kernel_declined:rows"] == 48 * len(plan)


def test_dense_experts_decline_expert_format(monkeypatch):
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 12), key=mx.random.key(5))
    plan = [(0, 8)] + [(i, i + 1) for i in range(8, 12)]
    model.configure_kernels(moe_fused=True)
    base = _run(model, tokens, plan)
    _law_on_cpu(monkeypatch)
    model.configure_kernels(moe_fused=True, moe_decode_kernel=True)
    assert mx.array_equal(_run(model, tokens, plan), base)
    keys = _decode_keys(_counts(model))
    # FP32 activations decline before the expert format is looked at.
    assert keys == {"decode_kernel_declined:rows": 48,
                    "decode_kernel_declined:dtype": 48 * 4}


def test_law_matches_shipped_fused_call_per_layer():
    """The law (rounding points, slot order, epilogues) against the shipped
    fused call on one NVFP4 layer and one token."""
    from mlx2.runtime.models.inkling2 import moe_fast
    from mlx2.runtime.models.inkling2.language import _fp32

    model = _quantized_model()
    model.configure_kernels(moe_fused=True)
    moe = model.model.layers[3].mlp
    experts, refusal = moe._fast_experts()
    assert refusal is None
    for seed in range(16):
        x = (mx.random.normal((1, 1, model.config.hidden_size), key=mx.random.key(seed))
             ).astype(mx.bfloat16)
        ref = moe_fast.fused_call(moe, experts, x)
        order, scores = mdk.route_compiled(x, _fp32(moe, "gate_weight"),
                                           moe.e_score_correction_bias, moe.top_k)
        shipped_idx, _ = moe_fast.route_compiled(x, _fp32(moe, "gate_weight"),
                                                 moe.e_score_correction_bias, moe.top_k,
                                                 moe.route_scale)
        assert mx.array_equal(order[..., :moe.top_k], shipped_idx)
        got = mdk.law(moe, experts, x, order, scores)
        assert got.shape == ref.shape and got.dtype == ref.dtype
        assert mx.array_equal(got, ref)


@pytest.mark.parametrize("top_k", [1, 2, 4, 7, 8, 10, 16, 30, 32])
def test_geometry_and_top_k_declines(top_k):
    """K % 512 == 0 would make MLX pick gather_qmv_fast (another order);
    top_k above 8 changes MLX's column-reduction order (declined)."""
    from types import SimpleNamespace

    from mlx2.runtime.nvfp4_linear import ScaledNVFP4SwitchLinear

    def fake(hidden, inter):
        gate_up = ScaledNVFP4SwitchLinear(hidden, 2 * inter, 32)
        down = ScaledNVFP4SwitchLinear(inter, hidden, 32)
        return SimpleNamespace(gate_up=gate_up, down=down, scales=mx.ones((32, 2)))

    moe = SimpleNamespace(top_k=top_k)
    old = mdk.device_ok
    try:
        mdk.device_ok = lambda: True
        x = mx.zeros((1, 1, 576), mx.bfloat16)
        expected = None if top_k <= 8 else "top_k"
        assert mdk.decline_reason(moe, fake(576, 192), x) == expected
        if top_k <= 8:
            assert mdk.decline_reason(moe, fake(576, 512), x) == "geometry"
            x512 = mx.zeros((1, 1, 512), mx.bfloat16)
            assert mdk.decline_reason(moe, fake(512, 192), x512) == "geometry"
    finally:
        mdk.device_ok = old
