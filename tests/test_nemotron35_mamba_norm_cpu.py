from types import SimpleNamespace

import mlx.core as mx

from mlx2.runtime.models.nemotron_mamba_norm import (
    admit_fused_mamba_norm_b1,
    admit_fused_mamba_norm_short,
)


def _tensor(shape, dtype=mx.bfloat16):
    return SimpleNamespace(shape=shape, dtype=dtype)


def _admit(**overrides):
    values = {
        "y": _tensor((1, 1, 8192)),
        "gate": _tensor((1, 1, 8192)),
        "weight": _tensor((8192,)),
        "eps": 1e-5,
        "group_size": 1024,
        "training": False,
    }
    values.update(overrides)
    return admit_fused_mamba_norm_b1(**values)


def test_super_gate_group_rms_decode_geometry_is_admitted():
    assert _admit().accepted


def test_super_gate_group_rms_short_tile_geometry_is_admitted():
    values = {
        "y": _tensor((1, 4, 8192)),
        "gate": _tensor((1, 4, 8192)),
        "weight": _tensor((8192,)),
        "eps": 1e-5,
        "group_size": 1024,
        "training": False,
    }
    assert admit_fused_mamba_norm_short(**values).accepted
    assert not admit_fused_mamba_norm_b1(**values).accepted
    values["y"] = _tensor((1, 9, 8192))
    values["gate"] = _tensor((1, 9, 8192))
    assert admit_fused_mamba_norm_short(**values).reason.startswith("y shape")


def test_non_decode_geometry_and_non_bf16_fail_closed():
    assert _admit(y=_tensor((1, 2, 8192))).reason.startswith("y shape")
    assert _admit(gate=_tensor((2, 1, 8192))).reason.startswith("gate shape")
    assert _admit(weight=_tensor((8192,), mx.float16)).reason == "requires bf16 operands"


def test_group_epsilon_and_weight_geometry_are_locked():
    assert _admit(group_size=512).reason == "group size 512"
    assert _admit(eps=1e-6).reason == "eps 1e-06"
    assert _admit(weight=_tensor((4096,))).reason.startswith("weight shape")
