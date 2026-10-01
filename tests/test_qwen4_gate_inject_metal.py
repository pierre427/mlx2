"""Real-Metal numerical gates for the Flash-Next gate/inject candidate."""

from __future__ import annotations

import os

import mlx.core as mx
import pytest

from mlx2.runtime.models import qwen4_gate_inject as gate_inject

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 under both GPU locks",
)


@pytest.fixture
def gpu_candidate():
    previous_device = mx.default_device()
    previous_enabled = gate_inject.fused_gate_inject_enabled()
    mx.set_default_device(mx.gpu)
    gate_inject.set_fused_gate_inject_enabled(True)
    gate_inject.reset_qwen4_gate_inject_stats()
    try:
        yield
    finally:
        gate_inject.reset_qwen4_gate_inject_stats()
        gate_inject.set_fused_gate_inject_enabled(previous_enabled)
        mx.set_default_device(previous_device)


def _inputs(batch, tokens, seed):
    def draw(shape, offset, scale):
        return (
            mx.random.normal(shape, key=mx.random.key(seed + offset)) * scale
        ).astype(mx.bfloat16)

    residual = draw((batch, tokens, gate_inject.STREAM_WIDTH), 1, 0.7)
    branch = draw((batch, tokens, gate_inject.HIDDEN_SIZE), 2, 0.5)
    raw_gate = draw((batch, tokens, gate_inject.HC_COUNT), 3, 4.0)
    mx.eval(residual, branch, raw_gate)
    return residual, branch, raw_gate


@pytest.mark.parametrize(
    ("batch", "tokens", "mode"),
    [(1, 1, "ordinary_b1"), (4, 1, "batched_decode"), (1, 17, "self_mtp_verify")],
)
def test_candidate_is_bit_exact_in_each_route_mode(
    gpu_candidate, batch, tokens, mode
):
    values = _inputs(batch, tokens, 100 * batch + tokens)
    expected = gate_inject.eager_qwen4_gate_inject(*values)
    actual = gate_inject.try_qwen4_gate_inject(*values)
    assert actual is not None
    mx.eval(expected, actual)
    assert mx.array_equal(expected, actual).item(), (batch, tokens, mode)
    status = gate_inject.qwen4_gate_inject_stats()
    assert status["counts"][f"calls:{mode}"] == 1
    assert status["last_decision"]["accepted"] is True


def test_sigmoid_edge_values_match_eager_bytes(gpu_candidate):
    residual = mx.zeros((1, 1, gate_inject.STREAM_WIDTH), dtype=mx.bfloat16)
    branch = mx.ones((1, 1, gate_inject.HIDDEN_SIZE), dtype=mx.bfloat16)
    raw_gate = mx.array(
        [[[-27.375, -1.0, 0.0, 27.375]]], dtype=mx.bfloat16
    )
    expected = gate_inject.eager_qwen4_gate_inject(
        residual, branch, raw_gate
    )
    actual = gate_inject.try_qwen4_gate_inject(
        residual, branch, raw_gate
    )
    assert actual is not None
    mx.eval(expected, actual)
    assert mx.array_equal(expected, actual).item()
