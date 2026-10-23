"""Opt-in Flash TensorFold kernel selection and offload boundaries."""

from collections import Counter
import os

import mlx.core as mx
from mlx import nn
import pytest

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.models.flash_tensorfold_qmv import (
    TensorFoldQMVLinear, eligible, install,
)


def _quantized(bits=4, *, input_dims=512, output_dims=128):
    prior = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        linear = nn.Linear(input_dims, output_dims, bias=False)
        linear.weight = linear.weight.astype(mx.bfloat16)
        quantized = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=bits)
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


@pytest.mark.parametrize("bits", [4, 8])
def test_installer_accepts_affine_q4_and_q8(bits):
    model = nn.Module()
    model.dense = _quantized(bits)
    assert eligible(model.dense)
    receipt = install(model)
    assert receipt["installed"] == 1
    assert receipt["kernel"] == "qmv_rows_q4q8g64"
    assert type(model.dense) is TensorFoldQMVLinear


def test_installer_preserves_fused_hc_projection_ownership():
    model = nn.Module()
    model.dense = _quantized()
    model.hc = nn.Module()
    model.hc.hc_count = 4
    model.hc.input_mix_weight_down = _quantized()
    model.hc.input_mix_weight_up = _quantized()
    model.hc.block_inject_weight = _quantized()
    receipt = install(model)
    assert receipt["installed"] == 1
    assert type(model.dense) is TensorFoldQMVLinear
    for field in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
        assert type(getattr(model.hc, field)) is nn.QuantizedLinear


def test_shared_weight_policy_is_default_off_and_requires_qmv_rows():
    assert "tensorfold_qmv_shared_weights" not in FlashNextPolicy().as_dict()
    assert "tensorfold_qmv_shared_weights" not in FlashNextPolicy(
        tensorfold_qmv_rows=True
    ).as_dict()
    policy = FlashNextPolicy.from_mapping({
        "tensorfold_qmv_rows": True, "tensorfold_qmv_shared_weights": True,
    })
    assert policy.as_dict()["tensorfold_qmv_shared_weights"] is True
    with pytest.raises(ValueError, match="requires tensorfold_qmv_rows"):
        FlashNextPolicy.from_mapping({"tensorfold_qmv_shared_weights": True})
    for value in (1, "true", None):
        with pytest.raises(ValueError, match="tensorfold_qmv_shared_weights must be boolean"):
            FlashNextPolicy.from_mapping({
                "tensorfold_qmv_rows": True, "tensorfold_qmv_shared_weights": value,
            })


@pytest.mark.parametrize("shared", [False, True])
def test_generic_qmv_installer_is_refused_for_packed_tree_projection_law(shared):
    from mlx2.adapters.flash_next import split_native_mtp_external_policy

    native = {"tensorfold_qmv_rows": True, "tensorfold_qmv_shared_weights": shared}
    assert split_native_mtp_external_policy(native) == (native, {})
    with pytest.raises(ValueError, match="tensorfold_qmv_rows"):
        split_native_mtp_external_policy({
            **native, "draft_model": "/synthetic/draft",
            "continuation_pool": {"sources": ["external"], "limit": 3,
                                  "ast_planner": True, "packed_pattern": [5, 5, 5]},
        })


@pytest.mark.parametrize("bits", [4, 8])
def test_shared_installer_retains_parameter_identity_and_reports_candidate(bits, monkeypatch):
    from mlx2.runtime.models import flash_tensorfold_qmv as qmv

    model = nn.Module()
    model.dense = _quantized(bits)
    original = model.dense
    parameters = {key: getattr(original, key) for key in ("weight", "scales", "biases")}
    receipt = install(model, shared_weights=True)
    assert model.dense is original
    assert all(getattr(model.dense, key) is value for key, value in parameters.items())
    assert receipt["kernel"] == "qmv_rows_q4q8g64_shared2"
    assert receipt["shared_weights"] is True
    assert receipt["qualified"] is False
    assert receipt["installed"] == 1
    assert len(receipt["names_sha256"]) == 64

    monkeypatch.setattr(qmv, "_COUNTS", Counter())
    calls = []
    def fake_rows(x, module, *, shared_weights=False):
        calls.append((module, shared_weights))
        return mx.zeros((*x.shape[:-1], 128), dtype=mx.bfloat16)
    monkeypatch.setattr(qmv, "qmv_rows", fake_rows)
    assert model.dense(mx.zeros((1, 3, 512), dtype=mx.bfloat16)).shape == (1, 3, 128)
    # Eighteen rows fall back to the existing CPU QuantizedLinear path.
    fallback = model.dense(mx.zeros((1, 18, 512), dtype=mx.bfloat16))
    mx.eval(fallback)
    assert fallback.shape == (1, 18, 128)
    assert calls == [(original, True)]
    assert qmv.counters() == {
        "kernel_calls": 1, "kernel_rows": 3, "shared_weight_calls": 1, "stock_calls": 1,
    }


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("rows", [1, 2, 3, 16, 17])
@pytest.mark.parametrize("shared", [False, True])
def test_row_kernel_dispatch_preserves_shapes_and_uses_paired_groups(bits, rows, shared, monkeypatch):
    from mlx2.runtime.models import flash_tensorfold_qmv as qmv

    linear = _quantized(bits)
    requests = []
    def fake_kernel(selected_bits, *, shared_weights=False):
        assert selected_bits == bits
        assert shared_weights is shared
        def submit(**kwargs):
            requests.append(kwargs)
            return [mx.full((rows, 128), 7, dtype=mx.bfloat16)]
        return submit
    monkeypatch.setattr(qmv, "_kernel", fake_kernel)
    x = mx.zeros((1, rows, 512), dtype=mx.bfloat16)
    result = qmv.qmv_rows(x, linear, shared_weights=shared)
    assert result.shape == (1, rows, 128)
    assert mx.all(result == 7).item()
    assert len(requests) == 1
    request = requests[0]
    assert request["inputs"][0].shape == (rows, 512)
    assert all(request["inputs"][i] is getattr(linear, key)
               for i, key in enumerate(("weight", "scales", "biases"), 1))
    groups = (rows + 1) // 2 if shared else rows
    assert request["grid"] == (32 * groups, 32, 1)
    assert request["threadgroup"] == (32 * groups, 1, 1)
    assert request["output_shapes"] == [(rows, 128)]
    assert request["output_dtypes"] == [mx.bfloat16]
    expected = {"K": 512, "N": 128, "RPS": 4, "GS": 64}
    if shared:
        expected["ROWS"] = rows
    assert dict(request["template"]) == expected


@pytest.mark.parametrize("value", [1, "true", None])
def test_shared_kernel_flags_are_strict_before_install_or_dispatch(value, monkeypatch):
    from mlx2.runtime.models import flash_tensorfold_qmv as qmv

    model = nn.Module()
    model.dense = _quantized()
    monkeypatch.setattr(qmv, "_kernel", lambda *a, **k: pytest.fail("invalid selector dispatched"))
    with pytest.raises(ValueError, match="shared_weights must be boolean"):
        install(model, shared_weights=value)
    assert type(model.dense) is nn.QuantizedLinear
    with pytest.raises(ValueError, match="shared_weights must be boolean"):
        qmv.qmv_rows(mx.zeros((1, 512), dtype=mx.bfloat16), model.dense, shared_weights=value)


@pytest.mark.skipif(
    os.environ.get("MLX2_PREFLIGHT_CPU_ONLY") == "1" or not mx.metal.is_available(),
    reason="explicit Metal oracle; excluded from CPU-only repair validation",
)
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("rows", [1, 2, 3, 16, 17])
@pytest.mark.parametrize("input_dims", [512, 1024])
def test_shared_weight_metal_matches_existing_row_kernel(bits, rows, input_dims):
    """Synthetic numerical oracle only; never a model or speed qualification."""
    from mlx2.runtime.models.flash_tensorfold_qmv import qmv_rows

    prior = mx.default_device()
    try:
        mx.set_default_device(mx.gpu)
        mx.random.seed(20261022)
        linear = _quantized(bits, input_dims=input_dims, output_dims=12)
        x = mx.random.normal((1, rows, input_dims)).astype(mx.bfloat16)
        reference = qmv_rows(x, linear)
        shared = qmv_rows(x, linear, shared_weights=True)
        mx.eval(reference, shared)
        assert mx.array_equal(reference, shared).item()
    finally:
        mx.set_default_device(prior)
