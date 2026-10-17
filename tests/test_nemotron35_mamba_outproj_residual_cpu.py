from types import SimpleNamespace

import mlx.core as mx
from mlx import nn

from mlx2.runtime.models.nemotron_mamba_outproj_residual import (
    admit_fused_mamba_outproj_residual_b1,
)


class _Projection(nn.QuantizedLinear):
    def __init__(self, *, bits=8, group_size=64, bias=False):
        self.bits = bits
        self.group_size = group_size
        self.weight = SimpleNamespace(shape=(4096, 2048))
        self.scales = SimpleNamespace(shape=(4096, 128), dtype=mx.bfloat16)
        self.biases = SimpleNamespace(shape=(4096, 128), dtype=mx.bfloat16)
        if bias:
            self.bias = SimpleNamespace(shape=(4096,), dtype=mx.bfloat16)


def _tensor(shape, dtype=mx.bfloat16):
    return SimpleNamespace(shape=shape, dtype=dtype)


def _admit(**overrides):
    values = {
        "y": _tensor((1, 1, 8192)),
        "residual": _tensor((1, 1, 4096)),
        "projection": _Projection(),
        "training": False,
    }
    values.update(overrides)
    return admit_fused_mamba_outproj_residual_b1(**values)


def test_exact_super_b1_q8_geometry_is_admitted():
    assert _admit().accepted


def test_non_decode_and_dtype_geometry_fail_closed():
    assert _admit(y=_tensor((1, 2, 8192))).reason.startswith("input shape")
    assert _admit(residual=_tensor((1, 1, 4096), mx.float16)).reason == (
        "requires bf16 operands"
    )
    assert _admit(training=True).reason == "training"


def test_quantization_layout_and_linear_bias_fail_closed():
    assert _admit(projection=_Projection(bits=4)).reason == (
        "requires affine q8 group64"
    )
    assert _admit(projection=_Projection(group_size=32)).reason == (
        "requires affine q8 group64"
    )
    assert _admit(projection=_Projection(bias=True)).reason == "linear bias"
