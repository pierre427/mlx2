"""CPU contracts for the exact Nemotron cross-session MoE row lattice."""

import copy

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.nemotron_h import (
    NEMOTRON_ROUTED_EXACT_MAX_ROWS,
    Model,
    NemotronHMoE,
    balanced_routed_exact_chunks,
)
from tests.test_nemotron_external_taps_cpu import tiny_model


def _tiny_moe(monkeypatch):
    mx.set_default_device(mx.cpu)
    mx.random.seed(1701)
    args = copy.copy(tiny_model().args)
    args.moe_latent_size = 4
    args.moe_intermediate_size = 6
    args.moe_shared_expert_intermediate_size = 7
    args.n_group = 1
    args.n_routed_experts = 4
    args.n_shared_experts = 1
    args.topk_group = 1
    args.num_experts_per_tok = 2
    args.norm_topk_prob = True
    args.routed_scaling_factor = 1.0
    moe = NemotronHMoE(args)
    # The production geometry gate is independently tested by the existing
    # routed-batch probe. This fixture shrinks only its tensors for CPU tests.
    monkeypatch.setattr(moe, "_routed_batch_geometry_refusal", lambda _rows: None)
    return moe


def _flat(rows):
    return mx.concatenate(tuple(rows), axis=1)


def _ordinary_token_rows(moe, sessions):
    return tuple(
        mx.concatenate(
            [moe(rows[:, index : index + 1]) for index in range(rows.shape[1])],
            axis=1,
        )
        for rows in sessions
    )


def test_balanced_chunk_planner_exhaustively_covers_every_lattice_total():
    for total in range(2, 129):
        plan = balanced_routed_exact_chunks(total)
        assert sum(plan) == total
        assert 2 <= min(plan) <= max(plan) <= NEMOTRON_ROUTED_EXACT_MAX_ROWS
        assert max(plan) - min(plan) <= 1
    assert balanced_routed_exact_chunks(17) == (9, 8)
    assert balanced_routed_exact_chunks(33) == (11, 11, 11)
    for invalid in (1, 129, 2.0, True):
        with pytest.raises(ValueError, match="integer in 2..128"):
            balanced_routed_exact_chunks(invalid)


def test_ragged_bxs_lattice_flattens_and_scatters_without_changing_rows(monkeypatch):
    moe = _tiny_moe(monkeypatch)
    sessions = (
        mx.arange(16, dtype=mx.float32).reshape(1, 2, 8) / 17,
        mx.arange(8, dtype=mx.float32).reshape(1, 1, 8) / 11,
        mx.arange(24, dtype=mx.float32).reshape(1, 3, 8) / 23,
    )
    expected = _ordinary_token_rows(moe, sessions)
    actual = moe.routed_lattice_exact(sessions)

    assert tuple(row.shape for row in actual) == ((1, 2, 8), (1, 1, 8), (1, 3, 8))
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    status = moe.routed_batch_status()
    assert status["counts"]["routed_lattice_exact"] == 1
    assert status["counts"]["lattice_sessions"] == 3
    assert status["counts"]["lattice_rows"] == 6
    assert status["last_lattice"] == {
        "schema": "mlx2.nemotron-routed-lattice.v1",
        "session_rows": [2, 1, 3],
        "total_rows": 6,
        "physical_chunk_rows": [6],
        "max_physical_chunk_rows": 16,
        "flatten_order": "session_major_token_major",
        "padding_rows": 0,
        "exact": True,
    }


def test_eight_by_four_lattice_preserves_rows_in_two_bounded_chunks(monkeypatch):
    moe = _tiny_moe(monkeypatch)
    sessions = tuple(
        (
            mx.arange(32, dtype=mx.float32).reshape(1, 4, 8)
            + session * 37
        )
        / 41
        for session in range(8)
    )
    expected = _ordinary_token_rows(moe, sessions)
    actual = moe.routed_lattice_exact(sessions)

    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    status = moe.routed_batch_status()
    assert status["counts"]["routed_lattice_exact"] == 1
    assert status["counts"]["lattice_chunked_calls"] == 1
    assert status["counts"]["lattice_physical_chunks"] == 2
    assert status["counts"]["lattice_sessions"] == 8
    assert status["counts"]["lattice_rows"] == 32
    assert status["last_lattice"] == {
        "schema": "mlx2.nemotron-routed-lattice.v1",
        "session_rows": [4] * 8,
        "total_rows": 32,
        "physical_chunk_rows": [16, 16],
        "max_physical_chunk_rows": 16,
        "flatten_order": "session_major_token_major",
        "padding_rows": 0,
        "exact": True,
    }


def test_ragged_seventeen_row_boundary_balances_without_singleton(monkeypatch):
    moe = _tiny_moe(monkeypatch)
    sessions = (
        mx.arange(128, dtype=mx.float32).reshape(1, 16, 8) / 43,
        mx.arange(8, dtype=mx.float32).reshape(1, 1, 8) / 47,
    )
    expected = _ordinary_token_rows(moe, sessions)
    actual = moe.routed_lattice_exact(sessions)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    status = moe.routed_batch_status()
    assert status["last_lattice"]["session_rows"] == [16, 1]
    assert status["last_lattice"]["physical_chunk_rows"] == [9, 8]
    assert status["last_lattice"]["total_rows"] == 17


def test_lattice_refuses_nonfloor_moe_padding_policy(monkeypatch):
    from mlx2.runtime.models import switch_layers

    moe = _tiny_moe(monkeypatch)
    monkeypatch.setattr(switch_layers, "_RHS_PAD_POLICY", "always")
    rows = mx.zeros((1, 2, 8))
    with pytest.raises(ValueError, match="moe rhs pad policy always"):
        moe.routed_lattice_exact((rows[:, :1], rows[:, 1:]))


def test_lattice_row_permutation_is_scattered_in_caller_order(monkeypatch):
    moe = _tiny_moe(monkeypatch)
    source = mx.arange(48, dtype=mx.float32).reshape(1, 6, 8) / 29
    baseline = _flat(
        moe.routed_lattice_exact((source[:, :2], source[:, 2:3], source[:, 3:]))
    )
    permutation = [4, 0, 5, 2, 1, 3]
    permuted = source[:, permutation]
    actual = _flat(
        moe.routed_lattice_exact(
            (permuted[:, :1], permuted[:, 1:4], permuted[:, 4:])
        )
    )
    np.testing.assert_array_equal(
        np.asarray(actual), np.asarray(baseline[:, permutation])
    )


def test_lattice_fails_closed_before_an_unsupported_gate_or_shape(monkeypatch):
    moe = _tiny_moe(monkeypatch)
    rows = mx.zeros((1, 2, 8))
    original_gate = moe.gate
    moe.gate = lambda value: original_gate(value)
    with pytest.raises(ValueError, match="routed expert lattice refused: gate"):
        moe.routed_lattice_exact((rows[:, :1], rows[:, 1:]))
    assert moe.routed_batch_status()["counts"]["refused"] == 1
    assert moe.routed_batch_status()["last_lattice"] is None

    moe.gate = original_gate
    with pytest.raises(ValueError, match="routed expert lattice refused: shape"):
        moe.routed_lattice_exact((rows[:, :1], mx.zeros((2, 1, 8))))
    with pytest.raises(ValueError, match="routed expert lattice refused: sessions 1"):
        moe.routed_lattice_exact((rows,))
    with pytest.raises(ValueError, match="routed expert lattice refused: segment rows 17"):
        moe.routed_lattice_exact((mx.zeros((1, 17, 8)), rows[:, :1]))
    assert moe.routed_batch_status()["counts"]["refused"] == 4
    assert moe.routed_batch_status()["counts"]["lattice_rows"] == 0


def test_model_owned_lattice_applies_norm_moe_and_residual_per_live_row(monkeypatch):
    args = copy.copy(tiny_model().args)
    args.hybrid_override_pattern = ["E"]
    args.num_hidden_layers = 1
    args.moe_latent_size = 4
    args.moe_intermediate_size = 6
    args.moe_shared_expert_intermediate_size = 7
    args.n_group = 1
    args.n_routed_experts = 4
    args.n_shared_experts = 1
    args.topk_group = 1
    args.num_experts_per_tok = 2
    args.norm_topk_prob = True
    args.routed_scaling_factor = 1.0
    model = Model(args)
    layer = model.layers[0]
    monkeypatch.setattr(
        layer.mixer, "_routed_batch_geometry_refusal", lambda _rows: None
    )
    sessions = (
        mx.arange(16, dtype=mx.float32).reshape(1, 2, 8) / 31,
        mx.arange(24, dtype=mx.float32).reshape(1, 3, 8) / 37,
    )
    expected = tuple(
        mx.concatenate(
            [layer(rows[:, index : index + 1]) for index in range(rows.shape[1])],
            axis=1,
        )
        for rows in sessions
    )
    actual = model.routed_expert_lattice_exact(0, sessions)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    with pytest.raises(ValueError, match="layer index is invalid"):
        model.routed_expert_lattice_exact(1, sessions)

