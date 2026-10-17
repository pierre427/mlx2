"""Independent dense FP32 oracle for the weight-only native NVFP4 module."""
import mlx.core as mx
import numpy as np
import pytest

from mlx2.conversion.nvfp4 import decode_rows
from mlx2.runtime.models.modelopt_nvfp4 import (
    ModelOptNVFP4Linear,
    configure_nvfp4_modules,
    normalize_nvfp4_execution,
    nvfp4_model_status,
)


@pytest.mark.parametrize("scale", [.00010027204552898183, .125])
@pytest.mark.parametrize("bias", [False, True])
def test_separate_tensor_scale_and_bias_match_dense_oracle(scale, bias):
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(8)
    packed = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    scales = rng.integers(1, 127, (32, 4), dtype=np.uint8)
    x = rng.normal(size=(2, 64)).astype(np.float32)
    offset = rng.normal(size=(32,)).astype(np.float32)
    model = ModelOptNVFP4Linear(64, 32, bias=bias)
    weights = [("weight", mx.array(packed.view("<u4"))),
               ("scales", mx.array(scales)),
               ("global_scale", mx.array(scale, dtype=mx.float32))]
    if bias:
        weights.append(("bias", mx.array(offset)))
    model.load_weights(weights, strict=True)
    actual = np.asarray(model(mx.array(x)))
    expected = x @ decode_rows(packed, scales, np.float32(scale)).T
    if bias:
        expected += offset
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-5)


def test_decoded_large_row_cache_uses_exact_mlx_global_scale_convention():
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(17)
    packed = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    scales = rng.integers(1, 127, (32, 4), dtype=np.uint8)
    scale = np.float32(0.0107421875)
    bias = rng.normal(size=(32,)).astype(np.float32)
    x = rng.normal(size=(4, 64)).astype(np.float32)
    model = ModelOptNVFP4Linear(64, 32, bias=True)
    model.load_weights(
        [
            ("weight", mx.array(packed.view("<u4"))),
            ("scales", mx.array(scales)),
            ("global_scale", mx.array(scale)),
            ("bias", mx.array(bias).astype(mx.bfloat16)),
        ],
        strict=True,
    )
    assert model.prepare_decoded_weight(budget_bytes=4095) == 0
    assert model.prepare_decoded_weight(budget_bytes=4096) == 4096
    model.set_dispatch_policy("auto", large_row_threshold=4)
    expected_weight = mx.dequantize(
        model.weight,
        scales=model.scales,
        group_size=16,
        bits=4,
        mode="nvfp4",
        global_scale=model.global_scale * (448 * 6),
        dtype=mx.bfloat16,
    )
    mx.eval(model._decoded_weight, expected_weight)
    np.testing.assert_array_equal(
        np.asarray(model._decoded_weight.astype(mx.float32)),
        np.asarray(expected_weight.astype(mx.float32)),
    )
    # The independently decoded FP32 product differs only by the expected
    # native-BF16 rounding order; this guards against a missing or doubled
    # ModelOpt tensor multiplier without requiring bit identity across laws.
    np.testing.assert_allclose(
        np.asarray(model._decoded_weight.astype(mx.float32)),
        decode_rows(packed, scales, scale),
        rtol=4e-3,
        atol=6.3e-2,
    )
    inputs = mx.array(x).astype(mx.bfloat16)
    actual = model(inputs)
    expected = inputs @ model._decoded_weight.T
    expected = (
        expected.astype(mx.float32) + model.bias.astype(mx.float32)
    ).astype(mx.bfloat16)
    mx.eval(actual, expected)
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)), np.asarray(expected.astype(mx.float32))
    )
    status = model.dispatch_status()
    assert status["counts"] == {"decoded_large_rows": 1}
    assert status["refusal_reasons"] == {"decoded_budget": 1}
    assert status["decoded_weight_resident"]


def test_shape_dispatch_keeps_native_default_and_reports_fused_refusal():
    mx.set_default_device(mx.cpu)
    model = ModelOptNVFP4Linear(64, 32)
    x = mx.zeros((2, 64), dtype=mx.bfloat16)
    model.set_dispatch_policy("fused_b1")
    model(x)
    status = model.dispatch_status()
    assert status["counts"] == {"native": 1}
    assert status["refusal_reasons"] == {"fused_b1_rows": 1}
    assert status["selected"] and not status["qualified"]


@pytest.mark.parametrize("policy", ["", "serial", "affine4", None, True])
def test_dispatch_refuses_unknown_policy(policy):
    with pytest.raises(ValueError, match="dispatch policy"):
        ModelOptNVFP4Linear(64, 32).set_dispatch_policy(policy)


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"format": "affine"},
        {"format": "modelopt_nvfp4", "policy": "auto"},
        {
            "format": "modelopt_nvfp4",
            "policy": "native",
            "decoded_budget_bytes": 1,
        },
        {"format": "modelopt_nvfp4", "extra": True},
    ],
)
def test_execution_policy_fails_closed(value):
    with pytest.raises(ValueError, match="NVFP4"):
        normalize_nvfp4_execution(value)


def test_model_configuration_and_status_keep_states_distinct():
    from mlx import nn

    class Pair(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = ModelOptNVFP4Linear(64, 32)
            self.right = ModelOptNVFP4Linear(64, 32)

    model = Pair()
    receipt = configure_nvfp4_modules(
        model,
        policy="auto",
        decoded_budget_bytes=4096,
        decoded_large_row_threshold=4,
    )
    assert receipt["decoded_resident_bytes"] == 4096
    assert len(receipt["decoded_modules"]) == 1
    status = nvfp4_model_status(model)
    assert status["implemented"] and status["selected"]
    assert not status["qualified"] and not status["observed_used"]
    assert status["modules"] == 2
    assert status["policies"] == {"auto": 2}
