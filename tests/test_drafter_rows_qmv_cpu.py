"""CPU wiring and fallback for the opt-in drafter multi-row 4-bit matvec."""

import numpy as np
import pytest

import mlx.core as mx
from mlx import nn

from mlx2.runtime import env_switches
from mlx2.runtime.models import drafter_rows_qmv as drq
from mlx2.runtime.nvfp4_linear import GlobalScaledNVFP4Linear

from test_lilicorr_nemotron_metadata import (  # noqa: F401  (fixture import)
    _target_contract,
    executable_bf16,
    plans,
)


def _affine(inputs, outputs, *, bits=4, group_size=64, bias=True):
    linear = nn.Linear(inputs, outputs, bias=bias)
    return nn.QuantizedLinear.from_linear(linear, group_size=group_size, bits=bits)


def _nvfp4(inputs, outputs, *, bias=True):
    module = GlobalScaledNVFP4Linear(inputs, outputs, bias=bias)
    weight = mx.random.normal((outputs, inputs)) * 0.02
    packed, scales = mx.quantize(weight, group_size=16, bits=4, mode="nvfp4")
    module.weight, module.scales = packed, scales
    module.global_scale = mx.array(0.5, dtype=mx.float32)
    if bias:
        module.bias = mx.random.normal((outputs,))
    return module


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(3)
        self.up = _affine(256, 64)
        self.down = _affine(128, 32, group_size=32, bias=False)
        self.head = _nvfp4(256, 32)
        self.q8 = _affine(256, 64, bits=8)
        self.narrow = _affine(64, 64)          # K % 128 != 0
        self.odd = _affine(256, 12, group_size=64)  # N % 8 != 0
        self.dense = nn.Linear(256, 64)


def test_switch_is_registered_and_closed():
    assert drq.ENV == "MLX2_DRAFTER_ROWS_QMV"
    assert drq.ENV in env_switches.SERVING_ENV_SWITCHES
    assert env_switches.serving_env_switches({drq.ENV: "on"}) == {drq.ENV: "on"}
    assert drq.enabled({}) is False
    assert drq.enabled({drq.ENV: "off"}) is False
    assert drq.enabled({drq.ENV: "on"}) is True
    with pytest.raises(ValueError, match="on or off"):
        drq.enabled({drq.ENV: "1"})


def test_layout_reasons():
    assert drq.layout_reason(4096, 12288, mode="affine", bits=4, group_size=64) is None
    assert drq.layout_reason(4096, 131072, mode="nvfp4", bits=4, group_size=16) is None
    assert drq.layout_reason(4096, 4096, mode="affine", bits=8, group_size=64) == "bits"
    assert drq.layout_reason(4096, 4096, mode="affine", bits=4, group_size=16) == "group_size"
    assert drq.layout_reason(4096, 4096, mode="nvfp4", bits=4, group_size=32) == "group_size"
    assert drq.layout_reason(4096, 4096, mode="mxfp4", bits=4, group_size=32) == "mode"
    assert drq.layout_reason(64, 4096, mode="affine", bits=4, group_size=64) == "shape"
    assert drq.layout_reason(4096, 12, mode="affine", bits=4, group_size=64) == "shape"


def test_off_changes_nothing():
    model = _Tiny()
    before = {name: type(module) for name, module in model.named_modules()}
    assert drq.install_if_enabled(model, {}) is None
    assert model.drafter_rows_qmv is None
    assert {name: type(module) for name, module in model.named_modules()} == before


def test_install_swaps_only_covered_modules_and_keeps_parameters():
    model = _Tiny()
    params = {k: np.asarray(v) for k, v in nn.utils.tree_flatten(model.parameters())}
    receipt = drq.install_if_enabled(model, {drq.ENV: "on"})
    assert model.drafter_rows_qmv is receipt
    assert receipt["modules"] == ["down", "head", "up"]
    assert receipt["declined"] == {"narrow": "shape", "odd": "shape", "q8": "bits"}
    assert receipt["row_exact"] is False and receipt["qualified"] is False
    assert type(model.up) is drq.DrafterRowsQuantizedLinear
    assert isinstance(model.up, nn.QuantizedLinear)
    assert type(model.head) is drq.nvfp4_class()
    assert isinstance(model.head, GlobalScaledNVFP4Linear)
    assert type(model.q8) is nn.QuantizedLinear
    assert type(model.dense) is nn.Linear
    after = {k: np.asarray(v) for k, v in nn.utils.tree_flatten(model.parameters())}
    assert after.keys() == params.keys()
    for key in params:
        np.testing.assert_array_equal(after[key], params[key])


@pytest.mark.parametrize("rows", [1, 2, 8, 16, 17])
def test_cpu_calls_fall_back_to_the_stock_law_exactly(rows):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        stock = _Tiny()
        swapped = _Tiny()
        drq.install(swapped)
        x = mx.random.normal((1, rows, 256)).astype(mx.bfloat16)
        before = drq.status()
        for name in ("up", "head"):
            want = getattr(stock, name)(x)
            got = getattr(swapped, name)(x)
            np.testing.assert_array_equal(
                np.asarray(got.astype(mx.float32)), np.asarray(want.astype(mx.float32))
            )
        after = drq.status()
        assert after["kernel_calls"] == before["kernel_calls"]
        assert after["stock_calls"] == before["stock_calls"] + 2
    finally:
        mx.set_default_device(previous)


def test_rows_outside_the_window_are_refused_by_the_kernel_entry():
    weight = mx.zeros((8, 16), dtype=mx.uint32)
    scales = mx.zeros((8, 2), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="2..16 rows"):
        drq.rows_qmv(mx.zeros((1, 17, 128), dtype=mx.bfloat16), weight, scales,
                     scales, mode="affine", group_size=64)


def test_lilicorr_load_records_the_switch_receipt(executable_bf16, monkeypatch):  # noqa: F811
    path, _source, target, target_artifact = executable_bf16
    record = plans.inspect_drafter(path, target_artifact=target_artifact)
    monkeypatch.delenv(drq.ENV, raising=False)
    off = plans.load_drafter(record, target, target_contract=_target_contract(record),
                             runtime_quantization="affine_q4_g64")
    assert off.drafter_rows_qmv is None
    assert "drafter_rows_qmv" not in off.execution_diagnostics()
    assert not any(type(m) is drq.DrafterRowsQuantizedLinear for _, m in off.named_modules())

    monkeypatch.setenv(drq.ENV, "on")
    on = plans.load_drafter(record, target, target_contract=_target_contract(record),
                            runtime_quantization="affine_q4_g64")
    receipt = on.drafter_rows_qmv
    assert receipt["switch"] == drq.ENV
    quantized = set(on.receipt_settings["runtime_weight_quantization"]["modules"])
    # Every runtime-quantized linear is either installed or declined with a
    # reason (the tiny export's 64-wide inputs miss the 128-wide K chunk).
    linears = {
        name for name, m in on.named_modules() if isinstance(m, nn.QuantizedLinear)
    }
    assert linears <= quantized
    assert linears == set(receipt["modules"]) | set(receipt["declined"])
    assert all(reason == "shape" for reason in receipt["declined"].values())
    assert on.execution_diagnostics()["drafter_rows_qmv"]["declined"] == len(receipt["declined"])
    # The install is part of the draft-state identity (it changes the
    # drafter's arithmetic); everything else in the receipt is unchanged.
    assert off.receipt_settings["drafter_rows_qmv"] is None
    assert on.receipt_settings["drafter_rows_qmv"] == {
        "schema": receipt["schema"], "modules": receipt["module_count"]}
    strip = lambda settings: {k: v for k, v in settings.items() if k != "drafter_rows_qmv"}
    assert strip(on.receipt_settings) == strip(off.receipt_settings)


def test_switch_parse_is_lenient_about_case_and_empty():
    from mlx2.runtime.models import drafter_rows_qmv as drq

    assert drq.enabled({drq.ENV: ""}) is False
    assert drq.enabled({drq.ENV: " ON "}) is True
    assert drq.enabled({}) is False


def test_nvfp4_mode_quantized_linear_is_declined():
    import mlx.core as mx
    from mlx import nn

    from mlx2.runtime.models import drafter_rows_qmv as drq

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.QuantizedLinear(256, 256, bias=False, group_size=16, bits=4, mode="nvfp4")

    model = Tiny()
    receipt = drq.install(model)
    assert receipt["module_count"] == 0
    assert "mode nvfp4" in receipt["declined"]["proj"]
    assert type(model.proj) is nn.QuantizedLinear
