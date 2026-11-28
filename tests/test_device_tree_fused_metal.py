"""``device_round.tree_fused`` Metal kernels against their reference form (GPU).

Runs only with Metal (skipped under ``MLX2_PREFLIGHT_CPU_ONLY``); book it on
the GPU through the lease.  ``decide_rows`` must give ``mx.argmax``'s index
(first NaN, else first maximum) and the host verifier's finiteness on every
row, at the Qwen3.8 law width and at odd widths, in bf16 and float32, with
ties placed in different threads and simdgroups; ``verify_row`` must give the
unfused ``[anchor] + clip(tokens)`` and the raw drafts.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime import speculative_device_tree_fused as fused

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal only")


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


def _rows(rng, rows, vocab):
    values = np.asarray(rng.normal(size=(rows, vocab)), dtype=np.float32)
    threads = fused._THREADS
    for row in range(rows):
        kind = row % 8
        if kind == 1:  # a tie in two threads of different simdgroups
            values[row, [3 % vocab, (3 + 33 * 7) % vocab]] = 9.0
        elif kind == 2:  # a tie in the same thread (stride apart)
            values[row, [5 % vocab, (5 + threads) % vocab]] = 9.0
        elif kind == 3:
            values[row, :] = -np.inf
        elif kind == 4:
            values[row, vocab - 1] = np.inf
        elif kind == 5:
            values[row, [vocab // 2, vocab // 3]] = np.nan
        elif kind == 6:
            values[row, rng.integers(0, vocab, size=max(1, vocab // 4))] = -np.inf
        elif kind == 7:
            values[row, vocab - 1] = 12.0  # the maximum in the last element
    return values


@pytest.mark.parametrize("vocab", [248320, 151936, 1000, 33, 7])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32, mx.float16])
def test_decide_rows_kernel_equals_the_reference(vocab, dtype):
    rng = np.random.default_rng(vocab % 97)
    for rows in (4, 9, 16):
        values = mx.array(_rows(rng, rows, vocab)).astype(dtype)
        got = np.asarray(fused.decide_rows(mx, values)).tolist()
        want = np.asarray(fused.reference_decide_rows(mx, values)).tolist()
        f32 = values.astype(mx.float32)
        finite = np.isfinite(np.asarray(mx.logsumexp(f32, axis=-1))).astype(int).tolist()
        assert got[rows:] == want[rows:] == finite
        # The argmax must agree on every row (also non-finite ones: mx.argmax).
        assert got[:rows] == np.asarray(mx.argmax(f32, axis=-1)).tolist()


def test_verify_row_kernel_equals_the_reference():
    rng = np.random.default_rng(5)
    for width, pattern in ((5, (5, 5, 5)), (4, (4, 4, 3)), (3, (3,)), (3, (3, 2, 2))):
        grid = mx.array(rng.integers(0, 300, size=(len(pattern), width)), dtype=mx.uint32)
        index = [b * width + k for b, d in enumerate(pattern) for k in range(d)]
        row, raw = fused.verify_row(mx, grid, index, 17, 249)
        mx.set_default_device(mx.cpu)
        try:
            want_row, want_raw = fused.verify_row(mx, mx.array(np.asarray(grid)), index, 17, 249)
        finally:
            mx.set_default_device(mx.gpu)
        assert np.asarray(row).tolist() == np.asarray(want_row).tolist()
        assert np.asarray(raw).tolist() == np.asarray(want_raw).tolist()


@pytest.mark.parametrize("emit_logprobs", [True, False])
def test_fused_rounds_on_metal_equal_the_host_round(monkeypatch, emit_logprobs):
    """End to end on the GPU (the kernels inside a served round): the tiny
    Nemotron-H target with its native MTP head on the reference tree forward,
    host round against ``tree_fused``: tokens, decisions, RNG draws and cache
    planes equal, and every round fused."""
    from test_device_round_tree_cpu import _assert_rounds_equal, _decode
    from test_nemotron_native_mtp_external_cpu import PROMPT, aligned

    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    monkeypatch.delenv("MLX2_TARGET_TREE_EXECUTION", raising=False)
    monkeypatch.delenv("MLX2_QWEN_TARGET_EXECUTION", raising=False)
    monkeypatch.setenv("MLX2_AST_CONTROLLER_MODE", "deterministic")
    monkeypatch.setenv("MLX2_AST_CONTROLLER_COST",
                       '{"a_ms": 5.0, "b_ms": 0.5, "ordinary_ms": 4.0}')
    sampling = {"sampling_temp": 0, "emit_logprobs": emit_logprobs}
    policy = {"tree_lazy": True, "tree_decide": True, "tree_fused": True}
    host = _decode(lambda: aligned(28, 0.3), PROMPT, sampling=sampling)
    device = _decode(lambda: aligned(28, 0.3), PROMPT, sampling=sampling, device_round=policy)
    assert device[0] == host[0]
    _assert_rounds_equal(host[1], device[1])
    rounds = host[2]["external_tree_rounds"]
    assert rounds > 0 and device[2][fused.COUNTER] == rounds
