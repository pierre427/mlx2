"""CPU admission and receipt tests for the Flash-Next gate/inject candidate."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime.models import qwen4_gate_inject as gate_inject
from mlx2.runtime.models.qwen4_exp import GatedResidual


@pytest.fixture
def enabled():
    previous = gate_inject.fused_gate_inject_enabled()
    gate_inject.set_fused_gate_inject_enabled(True)
    gate_inject.reset_qwen4_gate_inject_stats()
    yield
    gate_inject.reset_qwen4_gate_inject_stats()
    gate_inject.set_fused_gate_inject_enabled(previous)


def _inputs(batch=1, tokens=1, dtype=mx.bfloat16):
    residual = mx.zeros(
        (batch, tokens, gate_inject.STREAM_WIDTH), dtype=dtype
    )
    branch = mx.zeros(
        (batch, tokens, gate_inject.HIDDEN_SIZE), dtype=dtype
    )
    raw_gate = mx.zeros(
        (batch, tokens, gate_inject.HC_COUNT), dtype=dtype
    )
    return residual, branch, raw_gate


def _admit(*values, training=False):
    return gate_inject.admit_qwen4_gate_inject(
        *values, training=training
    )


def test_default_off_is_not_selected():
    assert gate_inject.fused_gate_inject_enabled() is False
    status = gate_inject.qwen4_gate_inject_stats()
    assert status["implemented"] is True
    assert status["model_validation_passed"] is True
    assert status["qualified"] is False
    assert status["selected"] is False
    assert status["observed_used"] is False
    assert status["model_validation_receipt"].endswith("model-validation.json")
    assert status["physical_engagement_receipt"].endswith(
        "candidate-dispatch-count.json"
    )


def test_enabled_candidate_is_selected_but_not_qualified(enabled):
    status = gate_inject.qwen4_gate_inject_stats()
    assert status["selected"] is True
    assert status["qualified"] is False
    assert status["observed_used"] is False


@pytest.mark.parametrize(
    ("batch", "tokens", "mode"),
    [(1, 1, "ordinary_b1"), (4, 1, "batched_decode"), (1, 17, "self_mtp_verify")],
)
def test_locked_modes_clear_geometry_admission(enabled, batch, tokens, mode):
    result = _admit(*_inputs(batch, tokens))
    if result.accepted:
        assert result.mode == mode
    else:
        assert result.reason in {
            "Metal kernel unavailable",
            "default device is not the Metal GPU",
        }


@pytest.mark.parametrize(
    ("batch", "tokens"),
    [(33, 1), (2, 2), (1, 18)],
)
def test_unmeasured_geometry_declines(enabled, batch, tokens):
    result = _admit(*_inputs(batch, tokens))
    assert result.accepted is False
    assert "geometry must be" in result.reason


def test_training_declines(enabled):
    result = _admit(*_inputs(), training=True)
    assert result.accepted is False
    assert result.reason == "training is unsupported"


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_non_bfloat16_declines(enabled, dtype):
    result = _admit(*_inputs(dtype=dtype))
    assert result.accepted is False
    assert "bfloat16" in result.reason


def test_mismatched_shapes_decline(enabled):
    residual, branch, raw_gate = _inputs()
    result = _admit(residual, branch[:, :, :-1], raw_gate)
    assert result.accepted is False
    assert "branch must be" in result.reason


def test_enabled_cpu_attempt_falls_back_and_counts_decline(enabled):
    result = gate_inject.try_qwen4_gate_inject(*_inputs())
    assert result is None
    status = gate_inject.qwen4_gate_inject_stats()
    assert status["counts"]["attempts"] == 1
    assert status["counts"]["declines"] == 1
    assert status["last_decision"]["accepted"] is False


def test_eager_reference_retains_shape_and_dtype():
    residual, branch, raw_gate = _inputs(batch=2)
    out = gate_inject.eager_qwen4_gate_inject(residual, branch, raw_gate)
    mx.eval(out)
    assert out.shape == residual.shape
    assert out.dtype == mx.bfloat16


def test_gated_residual_can_defer_activation_without_changing_other_outputs():
    args = SimpleNamespace(
        hc_count=4,
        hidden_size=8,
        hc_lowrank=3,
        rms_norm_eps=1e-6,
    )
    module = GatedResidual(args)
    values = mx.random.normal((1, 2, 32)).astype(mx.bfloat16)
    eager = module(values)
    raw = module(values, raw_inject=True)
    mx.eval(*eager, *raw)
    assert mx.array_equal(eager[0], raw[0]).item()
    assert mx.array_equal(eager[1], raw[1]).item()
    assert mx.array_equal(eager[2], 2 * mx.sigmoid(raw[2] / 4)).item()
