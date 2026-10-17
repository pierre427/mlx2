from types import SimpleNamespace

import mlx.core as mx

from mlx2.runtime.models.nemotron_mamba_conv import (
    admit_fused_mamba_conv_b1,
    admit_fused_mamba_conv_short,
)


def _tensor(shape, dtype=mx.bfloat16):
    return SimpleNamespace(shape=shape, dtype=dtype)


def _admit(**overrides):
    values = {
        "conv_input": _tensor((1, 1, 10240)),
        "conv_state": _tensor((1, 3, 10240)),
        "weight": _tensor((10240, 4, 1)),
        "bias": _tensor((10240,)),
        "mask": None,
        "training": False,
        "kernel_size": 4,
    }
    values.update(overrides)
    return admit_fused_mamba_conv_b1(**values)


def test_super_decode_geometry_is_admitted():
    assert _admit().accepted


def test_prefill_mask_and_non_bf16_fail_closed():
    assert _admit(conv_input=_tensor((1, 2, 10240))).reason.startswith("input shape")
    assert _admit(mask=object()).reason == "masked input"
    assert (
        _admit(weight=_tensor((10240, 4, 1), mx.float16)).reason
        == "requires bf16 operands"
    )


def test_state_and_weight_geometry_must_match_channels():
    assert _admit(conv_state=_tensor((1, 4, 10240))).reason.startswith("state shape")
    assert _admit(weight=_tensor((10240, 1, 4))).reason.startswith("weight shape")


def test_short_conv_admits_one_through_sixteen_rows():
    for steps in (1, 2, 4, 8, 16):
        result = admit_fused_mamba_conv_short(
            conv_input=_tensor((1, steps, 10240)),
            conv_state=_tensor((1, 3, 10240)),
            weight=_tensor((10240, 4, 1)),
            bias=_tensor((10240,)),
            mask=None,
            training=False,
            kernel_size=4,
        )
        assert result.accepted, result.reason


def test_short_conv_refuses_overflow_and_batch_geometry():
    for shape in ((1, 17, 10240), (2, 4, 10240)):
        result = admit_fused_mamba_conv_short(
            conv_input=_tensor(shape),
            conv_state=_tensor((1, 3, 10240)),
            weight=_tensor((10240, 4, 1)),
            bias=_tensor((10240,)),
            mask=None,
            training=False,
            kernel_size=4,
        )
        assert result.reason.startswith("input shape")
