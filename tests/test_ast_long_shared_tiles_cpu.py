"""Synthetic CPU verifier geometry, not long-block artifact/drafter admission.

The random tiny drafter declares a synthetic block size of 16. These tests
never ask it for proposals: all 15-token HostDraftRows come from a CPU oracle.
"""

import copy

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime import row_exact_verify as REV
from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator, HostDraftRow
from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
from mlx2.runtime.progressive_external_verify import progressive_target_protocol_binding
from mlx2.runtime.proposal_providers import PointMassLaw
from mlx2.runtime.speculative_topology import ordered_alternative_winner
from test_ast_terminal_ordinary_cpu import _committed_digest
from test_qwen38_dflash2_cpu import PROMPTS, tiny_draft_args, tiny_target


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _fixture(tile):
    target = tiny_target()
    draft = DFlash2DraftModel(tiny_draft_args(block_size=16)).bind(target)
    binding = progressive_target_protocol_binding(target, "near_tie_bounded")
    batch = ExternalDraftBatchGenerator(
        target,
        draft_model=draft,
        binding="synthetic-cpu-long-verifier-only",
        num_draft=15,
        minimum_draft_proposals=15,
        completion_batch_size=1,
        progressive_verification_tile=tile,
        progressive_verification_fidelity="near_tie_bounded",
        progressive_target_protocol=binding["target_protocol"],
        progressive_protocol_binding=binding,
        cost_per_commit_handoff=False,
    )
    uid = batch.insert(
        [PROMPTS[0]], max_tokens=[40], sampling_configs=[{"sampling_temp": 0}]
    )[0]
    lane = batch.lanes[uid]
    while lane.anchor is None:
        batch._prefill(lane)
    lane.ready.clear()
    cache, anchor = copy.deepcopy(lane.cache), lane.anchor
    tokens = []
    for _ in range(16):
        logits = target(mx.array([[anchor]]), cache=cache)
        anchor = int(mx.argmax(logits[0, -1]).item())
        tokens.append(anchor)
    return target, batch, lane, tokens


def _block(tokens):
    return HostDraftRow(list(tokens), [PointMassLaw(token) for token in tokens], 1)


def _observe(monkeypatch, target):
    calls = []
    forward = target.forward_with_taps

    def observed(inputs, cache, layers):
        assert REV.current() is None
        result = forward(inputs, cache, layers)
        calls.append((inputs.tolist(), result))
        return result

    monkeypatch.setattr(target, "forward_with_taps", observed)
    return calls, forward


@pytest.mark.parametrize("tile,spans", [(5, [5, 5, 6]), (3, [3, 3, 3, 3, 4])])
def test_long_single_chain_fixed_tiles_and_one_terminal_bonus(monkeypatch, tile, spans):
    target, batch, lane, tokens = _fixture(tile)
    try:
        before = _committed_digest(lane.cache)
        calls, forward = _observe(monkeypatch, target)
        decisions, features, caches, _rng, stats, physical = (
            batch._progressive_verify_many([lane], [_block(tokens[:15])])
        )
        assert [(len(inputs), len(inputs[0])) for inputs, _ in calls] == [
            (1, n) for n in spans
        ]
        assert physical["physical_target_rows"] == 16
        assert stats[0]["tile_input_rows"] == spans
        decision = decisions[0]
        assert decision.accepted == 15
        assert decision.emitted == tokens
        assert decision.commit_count == 16
        assert _committed_digest(lane.cache) == before
        reference = [copy.deepcopy(lane.cache)]
        taps = []
        for inputs, (logits, hidden) in calls:
            tx = HybridVerifyRows(reference).begin([len(inputs[0])])
            expected_logits, expected_hidden = forward(
                mx.array(inputs), tx.caches, batch.layers
            )
            np.testing.assert_array_equal(
                np.asarray(logits), np.asarray(expected_logits)
            )
            np.testing.assert_array_equal(
                np.asarray(hidden), np.asarray(expected_hidden)
            )
            taps.append(expected_hidden)
            tx.commit([len(inputs[0])])
        assert _committed_digest(caches[0]) == _committed_digest(reference[0])
        np.testing.assert_array_equal(
            np.asarray(features), np.asarray(mx.concatenate(taps, axis=1))
        )
        actual_next = target(mx.array([[decision.emitted[-1]]]), cache=caches[0])
        expected_next = target(mx.array([[decision.emitted[-1]]]), cache=reference[0])
        np.testing.assert_array_equal(
            np.asarray(actual_next), np.asarray(expected_next)
        )
    finally:
        batch.close()


@pytest.mark.parametrize("tile", [5, 3])
def test_shared_ten_then_two_private_five_tails_winner_and_continuation(
    monkeypatch, tile
):
    target, batch, lane, tokens = _fixture(tile)
    try:
        before = _committed_digest(lane.cache)
        # The primary rejects exactly where the two private tails diverge.
        wrong = list(tokens[:15])
        wrong[10] = (wrong[10] + 1) % batch._target_args.vocab_size
        blocks = [_block(wrong), _block(tokens[:15])]
        calls, forward = _observe(monkeypatch, target)
        prefix = batch._progressive_verify_many(
            [lane], [_block(tokens[:10])], terminal=False
        )
        assert prefix[0][0].accepted == prefix[0][0].commit_count == 10
        assert prefix[0][0].emitted == tokens[:10]  # No bonus at shared boundary.
        assert prefix[5]["physical_target_rows"] == 10
        calls.clear()
        result = batch._progressive_verify_shared_prefix(lane, blocks, prefix_count=10)
        decisions, _features, caches, _rng, _stats, physical = result
        shapes = [(len(inputs), len(inputs[0])) for inputs, _ in calls]
        expected = (
            [(1, 5), (1, 5), (2, 6)]
            if tile == 5
            else [(1, 3), (1, 3), (1, 3), (1, 1), (2, 3), (1, 3)]
        )
        assert shapes == expected
        assert physical["shared_prefix_target_rows"] == 10
        assert physical["physical_target_rows"] == (22 if tile == 5 else 19)
        assert [decision.accepted for decision in decisions] == [10, 15]
        winner, full = ordered_alternative_winner(
            decisions, [15, 15], allow_partial_uplift=True
        )
        assert winner == 1 and full == (1,)
        assert decisions[winner].emitted == tokens
        assert decisions[winner].commit_count == 16
        assert _committed_digest(lane.cache) == before
        # Replay the exact same physical launch sequence; only the winner's
        # committed branch survives the first suffix transaction.
        reference = [copy.deepcopy(lane.cache)]
        for inputs, (logits, hidden) in calls:
            if len(inputs) == 2 and len(reference) == 1:
                reference = [copy.deepcopy(reference[0]), copy.deepcopy(reference[0])]
            tx = HybridVerifyRows(reference).begin([len(row) for row in inputs])
            expected_logits, expected_hidden = forward(
                mx.array(inputs), tx.caches, batch.layers
            )
            np.testing.assert_array_equal(
                np.asarray(logits), np.asarray(expected_logits)
            )
            np.testing.assert_array_equal(
                np.asarray(hidden), np.asarray(expected_hidden)
            )
            tx.commit([1, len(inputs[1])] if len(inputs) == 2 else [len(inputs[0])])
            if len(inputs) == 2:
                reference = [reference[1]]
        assert _committed_digest(caches[winner]) == _committed_digest(reference[0])
        actual_next = target(mx.array([[tokens[-1]]]), cache=caches[winner])
        expected_next = target(mx.array([[tokens[-1]]]), cache=reference[0])
        np.testing.assert_array_equal(
            np.asarray(actual_next), np.asarray(expected_next)
        )
    finally:
        batch.close()


@pytest.mark.parametrize("tile", [5, 3])
def test_shared_prefix_rejection_skips_every_tail(monkeypatch, tile):
    target, batch, lane, tokens = _fixture(tile)
    try:
        wrong = list(tokens[:15])
        wrong[4] = (wrong[4] + 1) % batch._target_args.vocab_size
        other = list(wrong)
        other[10] = (other[10] + 1) % batch._target_args.vocab_size
        before = _committed_digest(lane.cache)
        calls, _ = _observe(monkeypatch, target)
        decisions, _features, _caches, _rng, _stats, physical = (
            batch._progressive_verify_shared_prefix(
                lane, [_block(wrong), _block(other)], prefix_count=10
            )
        )
        assert [(len(inputs), len(inputs[0])) for inputs, _ in calls] == (
            [(1, 5)] if tile == 5 else [(1, 3), (1, 3)]
        )
        assert physical["prefix_rejected"] is True
        assert physical["branch_suffix_target_rows"] == 0
        assert [decision.accepted for decision in decisions] == [4, 4]
        assert all(decision.emitted == tokens[:5] for decision in decisions)
        assert _committed_digest(lane.cache) == before
    finally:
        batch.close()
