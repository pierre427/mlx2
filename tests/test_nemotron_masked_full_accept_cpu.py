"""Padding must preserve Mamba state even when every valid token is accepted."""

import copy

import mlx.core as mx
import numpy as np
import pytest
from test_nemotron_external_taps_cpu import assert_state_equal, tiny_model

from mlx2.runtime.cow_cache import (
    restore_recovery_descriptors,
    snapshot_recovery_descriptors,
)
from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
from mlx2.runtime.models.ssm import ssm_attn


@pytest.mark.parametrize("changed", ["groups", "head_dim", "state", "C", "dt", "dtype"])
def test_metal_step_geometry_contract_refuses_malformed_arrays(monkeypatch, changed):
    from mlx2.runtime.models import ssm

    monkeypatch.setattr(ssm.mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(ssm.mx.metal, "is_available", lambda: True)
    values = [
        mx.ones((2, 1, 2, 4)),
        mx.zeros((2,)),
        mx.ones((2, 1, 1, 32)),
        mx.ones((2, 1, 1, 32)),
        mx.ones((2,)),
        mx.zeros((2, 1, 2)),
        mx.zeros((2,)),
        mx.ones((2, 2, 4, 32)),
    ]
    if changed == "groups":
        values[2] = values[3] = mx.ones((2, 1, 3, 32))
    elif changed == "head_dim":
        values[0] = mx.ones((2, 1, 2, 0))
    elif changed == "state":
        values[-1] = mx.ones((2, 2, 5, 32))
    elif changed == "C":
        values[3] = mx.ones((2, 1, 1, 64))
    elif changed == "dt":
        values[5] = mx.ones((2, 1, 3))
    else:
        values[0] = values[0].astype(mx.int32)

    def forbidden(*args):
        raise AssertionError("malformed geometry reached Metal kernel")

    monkeypatch.setattr(ssm, "ssm_update_kernel", forbidden)
    monkeypatch.setattr(ssm, "ssm_update_seq_kernel", forbidden)
    monkeypatch.setattr(ssm, "ssm_attn", lambda *args, **kw: ("fallback", "state"))
    assert ssm.ssm_update(*values) == ("fallback", "state")


@pytest.mark.parametrize("length", [1, 3])
@pytest.mark.parametrize("state_dim", [4, 31, 33, 64])
def test_metal_dispatch_shape_gate_without_constructing_any_metal_kernel(
    monkeypatch, length, state_dim
):
    from mlx2.runtime.models import ssm

    # Replace dispatcher predicates only; underlying MLX stays on CPU. The
    # substituted kernels never construct native Metal programs.
    monkeypatch.setattr(ssm.mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(ssm.mx.metal, "is_available", lambda: True)
    calls = []

    def kernel(*args):
        calls.append("kernel")
        return "admitted", "state"

    monkeypatch.setattr(ssm, "ssm_update_kernel", kernel)
    monkeypatch.setattr(ssm, "ssm_update_seq_kernel", kernel)
    x = mx.ones((2, length, 2, 4))
    b = c = mx.ones((2, length, 1, state_dim))
    state = mx.ones((2, 2, 4, state_dim))
    values = (
        x,
        mx.zeros((2,)),
        b,
        c,
        mx.ones((2,)),
        mx.zeros((2, length, 2)),
        mx.zeros((2,)),
        state,
    )
    result = ssm.ssm_update(*values)
    if state_dim == 64:
        assert result == ("admitted", "state") and calls == ["kernel"]
    else:
        mx.eval(result)
        assert calls == []


@pytest.mark.parametrize("owned", ["mask", "lengths"])
def test_one_token_owned_padding_falls_back_and_preserves_old_state(monkeypatch, owned):
    from mlx2.runtime.models import ssm

    monkeypatch.setattr(ssm.mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(ssm.mx.metal, "is_available", lambda: True)

    def forbidden(*args):
        raise AssertionError("padding-owned step reached an unmasked Metal kernel")

    monkeypatch.setattr(ssm, "ssm_update_kernel", forbidden)
    monkeypatch.setattr(ssm, "ssm_update_seq_kernel", forbidden)
    x = mx.ones((2, 1, 2, 4))
    b = c = mx.ones((2, 1, 1, 32))
    old = mx.ones((2, 2, 4, 32))
    values = (
        x,
        mx.zeros((2,)),
        b,
        c,
        mx.ones((2,)),
        mx.zeros((2, 1, 2)),
        mx.zeros((2,)),
        old,
    )
    if owned == "mask":
        _, state = ssm.ssm_update(*values, mask=mx.zeros((2, 1), dtype=mx.bool_))
        np.testing.assert_array_equal(np.asarray(state), np.asarray(old))
    else:
        output, state = ssm.ssm_update(*values, lengths=mx.array([1, 1]))
        expected_output, expected_state = ssm.ssm_attn(
            *values, lengths=mx.array([1, 1])
        )
        np.testing.assert_array_equal(np.asarray(output), np.asarray(expected_output))
        np.testing.assert_array_equal(np.asarray(state), np.asarray(expected_state))


@pytest.fixture(autouse=True)
def cpu():
    before = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(before)


@pytest.mark.parametrize(
    "mask",
    [
        [[True, True, True], [True, True, False]],
        [[False, True, True], [False, False, False]],
        [[True, False, True], [False, True, False]],
    ],
)
@pytest.mark.parametrize("initial", [False, True])
def test_masked_ssm_state_matches_independent_noop_recurrence(mask, initial):
    rng = np.random.default_rng(121)
    x = rng.normal(size=(2, 3, 2, 4)).astype(np.float32)
    b = rng.normal(size=(2, 3, 1, 4)).astype(np.float32)
    c = rng.normal(size=(2, 3, 1, 4)).astype(np.float32)
    raw_dt = rng.normal(size=(2, 3, 2)).astype(np.float32)
    bias = np.array([0.3, -0.2], np.float32)
    a_log = np.log(np.array([1, 2], np.float32))
    residual = np.array([1, 0.5], np.float32)
    start = rng.normal(size=(2, 2, 4, 4)).astype(np.float32) if initial else None
    expected_state = (
        np.zeros((2, 2, 4, 4), np.float32) if start is None else start.copy()
    )
    expected_outputs = []
    live = np.array(mask)
    delta = np.logaddexp(0, raw_dt + bias)
    for position in range(3):
        dt = np.where(live[:, position, None], delta[:, position], 0)
        decay = np.exp(-np.exp(a_log)[None] * dt)
        write = (
            dt[:, :, None, None]
            * x[:, position, :, :, None]
            * b[:, position, 0, None, None]
        )
        expected_state = decay[:, :, None, None] * expected_state + write
        expected_outputs.append(
            np.sum(expected_state * c[:, position, 0, None, None], axis=-1)
            + x[:, position] * residual[None, :, None]
        )
    output, state = ssm_attn(
        mx.array(x),
        mx.array(a_log),
        mx.array(b),
        mx.array(c),
        mx.array(residual),
        mx.array(raw_dt),
        mx.array(bias),
        None if start is None else mx.array(start),
        (0, float("inf")),
        mask=mx.array(live),
    )
    np.testing.assert_allclose(np.asarray(state), expected_state, atol=3e-6, rtol=3e-6)
    expected_output = np.stack(expected_outputs, axis=1)
    np.testing.assert_allclose(
        np.asarray(output)[live], expected_output[live], atol=3e-6, rtol=3e-6
    )


@pytest.mark.parametrize("kept", [[1, 2], [3, 2]])
@pytest.mark.parametrize("recover", [False, True])
def test_ragged_full_short_row_accept_preserves_conv_recurrent_and_kv(kept, recover):
    model = tiny_model()
    prefixes = [[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
    rows = []
    for prefix in prefixes:
        row = model.make_cache()
        model(mx.array([prefix]), row)
        rows.append(row)
    references = copy.deepcopy(rows)
    if recover:
        snapshots = [snapshot_recovery_descriptors(row) for row in rows]
        transaction = HybridVerifyRows(rows).begin([3, 2])
        outputs, taps = model.forward_with_taps(
            mx.array([[5, 6, 7], [7, 6, 5]]), transaction.caches, [0, 1, 4]
        )
        mx.eval(outputs, taps)
        transaction.abort()
        rows = [restore_recovery_descriptors(*snapshot)[0] for snapshot in snapshots]
        for row, reference in zip(rows, references):
            assert_state_equal(row, reference)
    inputs = [[9, 8, 6], [8, 9, 6]]
    transaction = HybridVerifyRows(rows).begin([3, 2])
    outputs, taps = model.forward_with_taps(
        mx.array(inputs), transaction.caches, [0, 1, 4]
    )
    mx.eval(outputs, taps)
    transaction.commit(kept)
    for index, (row, reference, count) in enumerate(zip(rows, references, kept)):
        expected_outputs = []
        for token in inputs[index][:count]:
            expected_outputs.append(model(mx.array([[token]]), reference))
        np.testing.assert_allclose(
            np.asarray(outputs[index : index + 1, :count]),
            np.asarray(mx.concatenate(expected_outputs, axis=1)),
            atol=3e-6,
            rtol=3e-5,
        )
        assert_state_equal(row, reference)
        np.testing.assert_allclose(
            np.asarray(model(mx.array([[3]]), row)),
            np.asarray(model(mx.array([[3]]), reference)),
            atol=3e-6,
            rtol=3e-5,
        )
