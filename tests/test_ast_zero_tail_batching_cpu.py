"""The existing AST draft cohort can include newly admitted empty-tail lanes."""

import copy
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from test_ast_primary_near_tie_cpu import _batch
from test_ast_terminal_ordinary_cpu import _committed_digest


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _mixed_cohort():
    target, batch, lanes = _batch()
    batch.multilane_draft_cap = 4
    batch._round([lanes[0]])
    lanes[0].ready.clear()
    assert int(lanes[0].tail.shape[1]) > 0
    assert all(int(lane.tail.shape[1]) == 0 for lane in lanes[1:])
    return target, batch, lanes


def test_zero_tail_ast_coalesces_one_real_b4_draft_without_changing_depth(monkeypatch):
    _target, batch, lanes = _mixed_cohort()
    try:
        before_calls = batch.scheduler_stats["external_draft_cohort_calls"]
        tails = [int(lane.tail.shape[1]) for lane in lanes]
        offsets = [[entry.offset for entry in lane.draft_cache] for lane in lanes]
        observed = []
        real = batch.draft.draft_distributions

        def observe(anchors, hidden, caches, count, *args, **kwargs):
            observed.append((len(anchors), int(hidden.shape[1]), count))
            return real(anchors, hidden, caches, count, *args, **kwargs)

        monkeypatch.setattr(batch.draft, "draft_distributions", observe)
        batch._round(lanes)
        assert len(observed) == 1
        assert observed[0][:2] == (4, 0)
        assert 1 <= observed[0][2] <= 4
        assert batch.scheduler_stats["external_draft_cohort_calls"] - before_calls == 1
        assert batch.scheduler_stats["external_draft_zero_tail_equalized_rounds"] == 1
        assert batch.scheduler_stats["external_draft_zero_tail_equalized_lanes"] == 1
        assert (
            batch.scheduler_stats["external_draft_zero_tail_equalized_positions"]
            == tails[0]
        )
        assert (
            batch.scheduler_stats.get("external_draft_zero_tail_split_rounds", 0) == 0
        )
        assert batch.progressive_verification_tile == 4
        assert batch.progressive_verification_fidelity == "near_tie_bounded"
        for lane, prior, tail in zip(lanes, offsets, tails, strict=True):
            assert [entry.offset for entry in lane.draft_cache] == [
                offset + tail for offset in prior
            ]
            assert lane.ready
            assert lane.ast_source_budget["branches"] == 1
    finally:
        batch.close()


def test_zero_tail_ast_publication_failure_restores_context_and_committed_state(
    monkeypatch,
):
    _target, batch, lanes = _mixed_cohort()
    try:
        before = [
            (
                _committed_digest(lane.cache),
                _committed_digest(lane.draft_cache),
                np.asarray(lane.tail).copy(),
                list(lane.history),
                lane.anchor,
                lane.generated,
                copy.deepcopy(lane.rng.snapshot()),
            )
            for lane in lanes
        ]
        real_commit = batch._commit
        engaged = []

        def fail_after_commit(*args, **kwargs):
            real_commit(*args, **kwargs)
            engaged.append(
                batch.scheduler_stats["external_draft_zero_tail_equalized_rounds"]
            )
            raise RuntimeError("zero-tail publication fault")

        monkeypatch.setattr(batch, "_commit", fail_after_commit)
        with pytest.raises(RuntimeError, match="zero-tail publication fault"):
            batch._round(lanes)
        assert engaged == [1]
        assert not batch._open
        assert (
            batch.scheduler_stats.get("external_draft_zero_tail_equalized_rounds", 0)
            == 0
        )
        for lane, saved in zip(lanes, before, strict=True):
            assert _committed_digest(lane.cache) == saved[0]
            assert _committed_digest(lane.draft_cache) == saved[1]
            np.testing.assert_array_equal(np.asarray(lane.tail), saved[2])
            assert (lane.history, lane.anchor, lane.generated) == saved[3:6]
            assert lane.rng.snapshot() == saved[6]
            assert not lane.ready
    finally:
        batch.close()


def test_zero_tail_decline_keeps_separate_calls_and_truthful_split_counters(
    monkeypatch,
):
    _target, batch, lanes = _mixed_cohort()
    try:
        monkeypatch.setattr(batch.draft, "supports_empty_target_context", False)
        before_calls = batch.scheduler_stats["external_draft_cohort_calls"]
        batch._round(lanes)
        assert batch.scheduler_stats["external_draft_cohort_calls"] - before_calls == 2
        assert batch.scheduler_stats["external_draft_zero_tail_mixed_rounds"] == 1
        assert batch.scheduler_stats["external_draft_zero_tail_declined_rounds"] == 1
        assert batch.scheduler_stats["external_draft_zero_tail_split_rounds"] == 1
        assert batch.scheduler_stats["external_draft_zero_tail_split_extra_calls"] == 1
        assert (
            batch.scheduler_stats.get("external_draft_zero_tail_equalized_rounds", 0)
            == 0
        )
    finally:
        batch.close()


@pytest.mark.parametrize(
    "paired,capability,tile", [(True, True, 4), (False, False, 4), (False, True, None)]
)
def test_zero_tail_coalescence_refuses_unsupported_semantics(paired, capability, tile):
    calls = []
    batch = object.__new__(ExternalDraftBatchGenerator)
    batch.pair_context_tokens = paired
    batch.progressive_verification_tile = tile
    batch.scheduler_stats = {}
    batch.draft = SimpleNamespace(
        supports_empty_target_context=capability,
        append_context=lambda *args: calls.append(args),
    )
    lanes = [
        SimpleNamespace(tail=mx.zeros((1, length, 8)), draft_cache=[])
        for length in (0, 2)
    ]
    batch._equalize_draft_tails(lanes)
    assert not calls
    assert [int(lane.tail.shape[1]) for lane in lanes] == [0, 2]
    assert (
        batch.scheduler_stats.get("external_draft_zero_tail_equalized_rounds", 0) == 0
    )
