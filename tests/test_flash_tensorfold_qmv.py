"""Opt-in Flash TensorFold kernel selection and offload boundaries."""

import mlx.core as mx
from mlx import nn
import pytest

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.models.flash_tensorfold_qmv import (
    TensorFoldQMVLinear, eligible, install,
)


def _quantized():
    prior = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        linear = nn.Linear(512, 128, bias=False)
        linear.weight = linear.weight.astype(mx.bfloat16)
        quantized = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)
        mx.eval(quantized.parameters())
        return quantized
    finally:
        mx.set_default_device(prior)


def test_flash_tensorfold_policy_is_opt_in_and_strict():
    assert "tensorfold_qmv_rows" not in FlashNextPolicy().as_dict()
    assert FlashNextPolicy(tensorfold_qmv_rows=True).as_dict()["tensorfold_qmv_rows"]
    with pytest.raises(ValueError, match="tensorfold_qmv_rows must be boolean"):
        FlashNextPolicy.from_mapping({"tensorfold_qmv_rows": 1})


def test_installer_skips_file_backed_ple_and_experts():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = _quantized()
            self.ple = nn.Module()
            self.ple.table = _quantized()
            self.switch_mlp = nn.Module()
            self.switch_mlp.gate = _quantized()

    model = Model()
    assert eligible(model.dense)
    receipt = install(model)
    assert receipt["installed"] == 1
    assert len(receipt["names_sha256"]) == 64
    assert type(model.dense) is TensorFoldQMVLinear
    assert type(model.ple.table) is nn.QuantizedLinear
    assert type(model.switch_mlp.gate) is nn.QuantizedLinear
