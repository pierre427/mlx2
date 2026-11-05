"""Synthetic CPU fixture only, no checkpoint or device execution.

Fixture: synthetic-ironmlx-ffn@SHA256(deterministic-rng-41); source revision
9fc9e848d8028874773c9245c67d8f2ad6c4e278 plus this implementation.
"""

import hashlib
import subprocess
import sys
import threading
from concurrent.futures import TimeoutError
from dataclasses import replace

import numpy as np
import pytest

from mlx2.experimental.ironmlx_ane_prefill import (
    INPUT_SEED,
    MID_SEED,
    CpuW8A8Backend,
    FfnWeights,
    PendingSplit,
    SplitBinding,
    SplitPrefillExecutor,
    SplitShape,
    hadamard_rows,
    ordinary_ffn,
    quantize_rotated_rows,
    weight_digest,
)

CHECKPOINT = "synthetic-ironmlx-ffn@" + hashlib.sha256(b"deterministic-rng-41").hexdigest()
SOURCE = "9fc9e848d8028874773c9245c67d8f2ad6c4e278"


@pytest.fixture
def fixture():
    shape = SplitShape(16, 24, 8, 3, input_block=8, mid_block=4)
    rng = np.random.default_rng(41)
    gate = rng.normal(0, .1, (24, 16)).astype(np.float32)
    up = rng.normal(0, .1, (24, 16)).astype(np.float32)
    down = rng.normal(0, .1, (16, 24)).astype(np.float32)
    inputs = rng.normal(0, .2, (3, 16)).astype(np.float32)
    binding = SplitBinding("synthetic.ffn", CHECKPOINT, SOURCE, weight_digest(gate, up, down))
    return shape, binding, (gate, up, down), inputs


def executor(fixture, backend=None):
    shape, binding, weights, _ = fixture
    return SplitPrefillExecutor(shape, binding, *weights, enabled=True,
                               array_budget_bytes=shape.accounted_bytes, backend=backend)


def independent_rotated(values, block, seed):
    # Dense Sylvester matrix and scalar Python hash rather than the production
    # butterfly layout. This oracle covers both signs and block orientation.
    matrix = np.ones((1, 1), np.float32)
    while matrix.shape[0] < block:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    signs = []
    for index in range(values.shape[-1]):
        value = index ^ seed
        value = ((value ^ (value >> 16)) * 0x85EBCA6B) % 2**32
        value = ((value ^ (value >> 13)) * 0xC2B2AE35) % 2**32
        signs.append(-1 if (value ^ (value >> 16)) & 1 else 1)
    rotated = (values.astype(np.float64) * signs).reshape(-1, block) @ matrix.T
    return (rotated / np.sqrt(block)).reshape(values.shape)


@pytest.mark.parametrize("block", [1, 2, 8, 256, 1024])
def test_rotation_matches_independent_dense_transform_and_preserves_products(block):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(2, block * 2)).astype(np.float32)
    w = rng.normal(size=(3, block * 2)).astype(np.float32)
    rotated_x = hadamard_rows(x, block, INPUT_SEED)
    rotated_w = hadamard_rows(w, block, INPUT_SEED)
    np.testing.assert_allclose(rotated_x, independent_rotated(x, block, INPUT_SEED), atol=1e-6, rtol=2e-5)
    np.testing.assert_allclose(rotated_x @ rotated_w.T, x @ w.T, atol=3e-5, rtol=2e-5)


def test_zero_and_saturation_are_finite_and_bounded():
    values = np.zeros((2, 16), np.float32)
    rows = quantize_rotated_rows(values, 8, INPUT_SEED)
    assert not rows.codes.any() and not rows.scales.any()
    rows = quantize_rotated_rows(values, 8, INPUT_SEED, peak_floor=1 / 512)
    assert not rows.codes.any() and (rows.scales > 0).all()
    values[0, 3] = 1000000
    rows = quantize_rotated_rows(values, 8, INPUT_SEED)
    assert rows.codes.dtype == np.int8 and np.abs(rows.codes).max() <= 127
    assert np.isfinite(rows.scales).all()


def test_split_ffn_is_complete_and_error_is_explicit(fixture):
    shape, binding, arrays, inputs = fixture
    weights = FfnWeights.capture(shape, binding, *arrays)
    expected = ordinary_ffn(inputs, weights)
    with executor(fixture) as runner:
        handle = runner.submit(inputs, binding=binding)
        actual = runner.finish(handle)
        relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        assert 0 < relative_error < .025
        receipt = runner.receipt()
        assert receipt["completed_cpu_jobs"] == 1
        assert receipt["qualified"] is False
        assert receipt["observed_used"] is False
        assert receipt["ane_backend_implemented"] is False
        assert receipt["execution_device_observed"] is False
        assert receipt["approximate"] is True
        assert receipt["checkpoint"] == CHECKPOINT
        assert receipt["source_revision"] == SOURCE
        assert receipt["active_slots"] == 0


def test_offload_matches_independent_quantization_oracle(fixture):
    shape, binding, arrays, inputs = fixture
    weights = FfnWeights.capture(shape, binding, *arrays)

    def quant(values, block, seed, floor=0):
        rotated = independent_rotated(values, block, seed)
        scale = np.maximum(abs(rotated).max(axis=1), floor) / 127
        codes = np.rint(rotated / np.where(scale > 0, scale, 1)[:, None]).astype(np.int64)
        return codes, scale

    def dot(x, w):
        # Scalar accumulation makes reassociation/order independent here;
        # integers are exact. Reference intentionally avoids the production dot.
        xc, xs = x
        wc, ws = w
        result = np.zeros((len(xc), len(wc)))
        for r in range(len(xc)):
            for c in range(len(wc)):
                result[r, c] = sum(int(a) * int(b) for a, b in zip(xc[r], wc[c])) * xs[r] * ws[c]
        return result

    x = quant(inputs, shape.input_block, INPUT_SEED)
    n = shape.leading_channels
    gate = dot(x, quant(arrays[0][n:], shape.input_block, INPUT_SEED))
    up = dot(x, quant(arrays[1][n:], shape.input_block, INPUT_SEED))
    hidden = (gate / (1 + np.exp(-gate)) * up).astype(np.float32)
    expected = dot(quant(hidden, shape.mid_block, MID_SEED, 1 / 512),
                   quant(arrays[2][:, n:], shape.mid_block, MID_SEED))
    actual = CpuW8A8Backend().execute(shape, inputs, weights, threading.Event())
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-5)


def test_input_and_weights_are_owned_and_bound(fixture):
    _, binding, arrays, inputs = fixture
    with executor(fixture) as runner:
        original = inputs.copy()
        first = runner.submit(inputs, binding=binding)
        inputs.fill(7)
        arrays[0].fill(77)
        first_output = runner.finish(first)
        second = runner.submit(original, binding=binding)
        np.testing.assert_array_equal(first_output, runner.finish(second))
        with pytest.raises(ValueError, match="identity"):
            runner.submit(original, binding=replace(binding, source_revision="a" * 40))
        with pytest.raises(ValueError, match="stale"):
            runner.finish(first)
        with pytest.raises(ValueError, match="foreign"):
            runner.finish(PendingSplit("other", 0, 1))


class BlockingBackend(CpuW8A8Backend):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def execute(self, shape, inputs, weights, cancelled):
        self.started.set()
        if not self.release.wait(5):
            raise RuntimeError("test did not release the blocked backend")
        return super().execute(shape, inputs, weights, cancelled)


def test_two_slots_are_bounded_even_if_jobs_complete(fixture):
    _, binding, _, inputs = fixture
    with executor(fixture) as runner:
        first = runner.submit(inputs, binding=binding)
        second = runner.submit(inputs, binding=binding)
        with pytest.raises(RuntimeError, match="both split slots"):
            runner.submit(inputs, binding=binding)
        np.testing.assert_array_equal(runner.finish(first), runner.finish(second))
        third = runner.submit(inputs, binding=binding)
        assert third.generation != first.generation
        runner.finish(third)


def test_cancel_does_not_reuse_running_slot_or_publish_late_result(fixture):
    _, binding, _, inputs = fixture
    backend = BlockingBackend()
    runner = executor(fixture, backend)
    try:
        first = runner.submit(inputs, binding=binding)
        assert backend.started.wait(1)
        runner.cancel(first)
        second = runner.submit(inputs, binding=binding)
        with pytest.raises(RuntimeError, match="both split slots"):
            runner.submit(inputs, binding=binding)
        backend.release.set()
        runner.finish(second)
        with pytest.raises(ValueError, match="stale"):
            runner.finish(first)
        assert runner.receipt()["completed_cpu_jobs"] == 1
    finally:
        backend.release.set()
        runner.close()


def test_timeout_retires_and_keeps_pending_ownership_until_completion(fixture):
    _, binding, _, inputs = fixture
    backend = BlockingBackend()
    runner = executor(fixture, backend)
    try:
        pending = runner.submit(inputs, binding=binding)
        assert backend.started.wait(1)
        with pytest.raises(TimeoutError):
            runner.finish(pending, timeout=0)
        assert runner.receipt()["active_slots"] == 1
        with pytest.raises(RuntimeError, match="retired"):
            runner.submit(inputs, binding=binding)
    finally:
        backend.release.set()
        runner.close()


def test_queued_cancellation_keeps_slot_until_worker_dequeues(fixture):
    _, binding, _, inputs = fixture
    backend = BlockingBackend()
    runner = executor(fixture, backend)
    try:
        first = runner.submit(inputs, binding=binding)
        assert backend.started.wait(1)
        second = runner.submit(inputs, binding=binding)
        runner.cancel(second)
        for _ in range(20):
            with pytest.raises(RuntimeError, match="both split slots"):
                runner.submit(inputs, binding=binding)
        assert runner.receipt()["active_slots"] == 2
        backend.release.set()
        runner.finish(first)
    finally:
        backend.release.set()
        runner.close()


@pytest.mark.parametrize("nonfinite", [False, True])
def test_failed_or_nonfinite_backend_cannot_publish(fixture, nonfinite):
    class BadBackend(CpuW8A8Backend):
        def execute(self, shape, inputs, weights, cancelled):
            if nonfinite:
                return np.full_like(inputs, np.nan)
            raise RuntimeError("synthetic backend failure")

    _, binding, _, inputs = fixture
    with executor(fixture, BadBackend()) as runner:
        handle = runner.submit(inputs, binding=binding)
        with pytest.raises(RuntimeError, match="backend failed"):
            runner.finish(handle)
        assert runner.receipt()["retired"] is True
        with pytest.raises(RuntimeError, match="retired"):
            runner.submit(inputs, binding=binding)


def test_refuses_native_backend_disabled_route_and_insufficient_budget(fixture):
    shape, binding, arrays, _ = fixture
    with pytest.raises(ValueError, match="enabled"):
        SplitPrefillExecutor(shape, binding, *arrays, array_budget_bytes=shape.accounted_bytes)
    with pytest.raises(MemoryError, match="budget"):
        SplitPrefillExecutor(shape, binding, *arrays, enabled=True,
                             array_budget_bytes=shape.accounted_bytes - 1)
    native = type("Native", (), {"execution_device": "ane", "name": "fake-native"})()
    with pytest.raises(ValueError, match="not implemented"):
        executor(fixture, native)
    with pytest.raises(ValueError, match="content identity"):
        SplitPrefillExecutor(shape, replace(binding, weights_sha256="b" * 64), *arrays,
                             enabled=True, array_budget_bytes=shape.accounted_bytes)


@pytest.mark.parametrize("changes", [
    {"offload_channels": 24}, {"input_block": 3}, {"mid_block": 16},
    {"half_rows": True}, {"hidden": 18}, {"max_position": 0},
])
def test_geometry_refusals(fixture, changes):
    shape = fixture[0]
    with pytest.raises(ValueError):
        replace(shape, **changes)


def test_runtime_geometry_position_dtype_and_finiteness_refusals(fixture):
    shape, binding, _, inputs = fixture
    with executor(fixture) as runner:
        for invalid in [inputs.astype(np.float64), inputs.tolist(), inputs[:, :-1],
                        np.full_like(inputs, np.nan), inputs[None, :, :]]:
            with pytest.raises(ValueError):
                runner.submit(invalid, binding=binding)
        with pytest.raises(ValueError, match="range"):
            runner.submit(inputs, binding=binding, start_position=shape.max_position - 1)
        assert runner.receipt()["active_slots"] == 0


def test_closed_executor_rejects_submission_and_result_publication(fixture):
    _, binding, _, inputs = fixture
    backend = BlockingBackend()
    runner = executor(fixture, backend)
    handle = runner.submit(inputs, binding=binding)
    assert backend.started.wait(1)
    runner.close(wait=False)
    try:
        assert runner.receipt()["closed"] is True
        with pytest.raises(RuntimeError, match="closed"):
            runner.submit(inputs, binding=binding)
        with pytest.raises(ValueError, match="stale"):
            runner.finish(handle)
    finally:
        backend.release.set()
        runner.close()


def test_draining_close_waits_for_concurrent_leading_preparation(fixture, monkeypatch):
    from mlx2.experimental import ironmlx_ane_prefill as module

    class ImmediateBackend(CpuW8A8Backend):
        def execute(self, shape, inputs, weights, cancelled):
            return np.zeros_like(inputs)

    _, binding, _, inputs = fixture
    runner = executor(fixture, ImmediateBackend())
    preparing, release, close_started, closed = (threading.Event() for _ in range(4))
    original_silu = module._silu
    failures = []

    def blocked_silu(values):
        preparing.set()
        if not release.wait(3):
            raise RuntimeError("test did not release leading preparation")
        return original_silu(values)

    def submit():
        try:
            runner.submit(inputs, binding=binding)
        except RuntimeError as error:
            failures.append(str(error))

    def close():
        close_started.set()
        runner.close()
        closed.set()

    monkeypatch.setattr(module, "_silu", blocked_silu)
    submitter = threading.Thread(target=submit)
    closer = threading.Thread(target=close)
    submitter.start()
    try:
        assert preparing.wait(1)
        closer.start()
        assert close_started.wait(1)
        assert not closed.wait(.05), "close returned while caller-owned leading work still ran"
    finally:
        release.set()
        submitter.join(3)
        if closer.ident is not None:
            closer.join(3)
        runner.close()
    assert closed.is_set()
    assert failures == ["split executor closed during leading preparation"]


def test_cpu_execution_never_imports_device_runtimes():
    script = """
import hashlib, sys
import numpy as np
from mlx2.experimental.ironmlx_ane_prefill import SplitBinding, SplitShape, SplitPrefillExecutor, weight_digest
shape = SplitShape(4, 8, 4, 1, input_block=4, mid_block=4)
gate = np.ones((8,4), dtype=np.float32) / 8
up = gate.copy()
down = np.ones((4,8), dtype=np.float32) / 8
binding = SplitBinding('synthetic.ffn', 'fixture@' + hashlib.sha256(b'fixture').hexdigest(), 'a'*40, weight_digest(gate,up,down))
with SplitPrefillExecutor(shape,binding,gate,up,down,enabled=True,array_budget_bytes=shape.accounted_bytes) as executor:
    executor.finish(executor.submit(np.ones((1,4),dtype=np.float32),binding=binding))
assert not any(name in sys.modules for name in ('mlx.core','coremltools','torch'))
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
