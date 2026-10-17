"""Real hybrid rollback/tap/RNG parity for seven proposals in 4+4 rows."""
import copy

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.nemotron35_dspark import DSparkConfig
from mlx2.adapters.xpress import XPressConfig
from mlx2.runtime.drafters.lilicorr import _ordered_lilicorr_paths
from mlx2.runtime.drafters.nemotron35_dspark import DSparkDraftModel
from mlx2.runtime.drafters.xpress import XPressDraftModel
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.models.nemotron35_super import Model as SuperModel
from mlx2.runtime.models.nemotron_h import Model as LanguageModel
from mlx2.runtime.sample_utils import LaneRNG
from mlx2.runtime.speculative_topology import (
    AdaptiveTopologyLatch,
    confidence_tree_branch_depths,
    confidence_tree_trunk_length,
)
from tests.test_nemotron_external_taps_cpu import tiny_model


def run_round(
    tile,
    reject_at,
    *,
    proposal_count=7,
    alternative_accept_branch=None,
    alternative_strategy="distinct_root",
    alternative_depth_policy="full",
    confidence_features=None,
    alternative_stop_branch=None,
):
    mx.set_default_device(mx.cpu)
    args = tiny_model().args
    mx.random.seed(28)
    model = SuperModel.__new__(SuperModel)
    # Exercise the actual Super target's inherited hybrid math and its
    # protocol declaration, without constructing an unrelated vision tower.
    LanguageModel.__init__(model, args)
    draft = XPressDraftModel(XPressConfig(
        hidden_size=8, intermediate_size=12, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        vocab_size=16, mask_token_id=15, num_target_layers=5,
        target_layer_ids=[0, 1, 4], block_size=proposal_count + 1,
        layer_types=["full_attention"],
        xpress_rank=3, xpress_mlp_hidden=5, xpress_num_passes=3,
    )).bind(model)
    prompt = [1, 2, 3, 4]
    cache = model.make_cache()
    model(mx.array([prompt[:-1]]), cache)
    token, proposal = prompt[-1], []
    for _ in range(proposal_count):
        token = int(mx.argmax(model(mx.array([[token]]), cache)[0, -1]).item())
        proposal.append(token)
    exact_proposal = list(proposal)
    bonus = int(
        mx.argmax(
            model(mx.array([[proposal[-1]]]), copy.deepcopy(cache))[0, -1]
        ).item()
    )
    alternative_stop_token = next(
        token
        for token in range(args.vocab_size)
        if token not in {*exact_proposal, bonus}
    )
    if reject_at is not None:
        proposal[reject_at] = (proposal[reject_at] + 1) % args.vocab_size
    laws = []
    for token in proposal:
        law = np.zeros(args.vocab_size, np.float64)
        law[token] = 1.
        laws.append(law)
    original = draft.draft_distributions
    last_primary = None
    last_prompt_round = False

    def forced(anchors, hidden, caches, count, rngs, temps, **kwargs):
        nonlocal last_primary, last_prompt_round
        generated_tokens, generated_laws = original(
            anchors, hidden, caches, count, rngs, temps, **kwargs
        )
        if confidence_features is not None:
            draft.adaptive_confidence_features = [
                list(confidence_features[: len(row)]) for row in generated_tokens
            ]
        last_prompt_round = anchors == [prompt[-1]]
        if anchors != [prompt[-1]]:
            last_primary = generated_tokens
            return generated_tokens, generated_laws
        assert count == proposal_count
        last_primary = [proposal]
        return [proposal], [laws]

    def alternatives(
        primary_tokens,
        *,
        branch_count,
        strategy,
        shared_prefix_tokens,
    ):
        assert primary_tokens == last_primary
        if not last_prompt_round:
            return [[list(primary_tokens[0]) for _ in range(branch_count)]]
        rows = []
        for branch in range(branch_count):
            candidate = list(exact_proposal)
            if branch != alternative_accept_branch:
                divergence = (
                    0 if strategy == "distinct_root" else shared_prefix_tokens
                )
                if isinstance(divergence, (list, tuple)):
                    divergence = divergence[0]
                candidate[divergence] = (
                    candidate[divergence] + branch + 1
                ) % args.vocab_size
            if branch == alternative_stop_branch:
                candidate[-1] = alternative_stop_token
            rows.append(candidate)
        rows[0] = list(proposal)
        return [rows]

    draft.draft_distributions = forced
    draft.ordered_alternative_proposals = alternatives
    batch = ExternalDraftBatchGenerator(model, draft_model=draft, binding="super-tiled-cpu",
        num_draft=proposal_count, completion_batch_size=1, prefill_step_size=8,
        progressive_verification_tile=tile,
        progressive_alternative_branches=(
            4 if alternative_accept_branch is not None else 1
        ),
        progressive_alternative_strategy=alternative_strategy,
        progressive_alternative_depth_policy=alternative_depth_policy,
        stop_tokens=(
            []
            if alternative_stop_branch is None
            else [[alternative_stop_token]]
        ))
    uid = batch.insert([prompt], max_tokens=[16], lane_rngs=[LaneRNG(919)],
                       sampling_configs=[{"sampling_temp": 0}])[0]
    lane = batch.lanes[uid]
    while lane.anchor is None:
        batch._prefill(lane)
    calls, forward = [], model.forward_with_taps

    def observed(inputs, *args, **kwargs):
        if inputs.shape[1] > 1:
            calls.append(np.asarray(inputs).tolist()[0])
        return forward(inputs, *args, **kwargs)

    model.forward_with_taps = observed
    batch._round([lane])
    return batch, lane, calls, proposal


def test_ordered_b2_full_accept_commits_b2_discards_b3_b4_and_continues():
    baseline_batch, baseline, _, _ = run_round(4, None, proposal_count=8)
    branch_batch, branch_lane, calls, _ = run_round(
        4,
        0,
        proposal_count=8,
        alternative_accept_branch=1,
    )
    try:
        assert branch_lane.history == baseline.history
        assert branch_lane.anchor == baseline.anchor
        assert branch_lane.rng.snapshot() == baseline.rng.snapshot()
        close_caches(branch_lane.cache, baseline.cache)
        receipt = branch_lane.ready[-1].speculative_receipt[
            "progressive_verification"
        ]["ordered_alternatives"]
        assert receipt["last_branch_accepts"] == [0, 8, 0, 0]
        assert receipt["last_selected_branch"] == 2
        assert receipt["last_short_circuit_after_branch"] == 2
        assert receipt["last_discarded_branches"] == [1, 3, 4]
        assert receipt["b2_full_accept_short_circuits"] == 1
        assert receipt["next_round_base"] == "selected_branch_committed_state"
        assert len(calls) == 2
        assert branch_batch.scheduler_stats[
            "external_progressive_alternative_b2_short_circuits"
        ] == 1

        before_rounds = branch_lane.external_rounds
        branch_batch._round([branch_lane])
        assert branch_lane.external_rounds == before_rounds + 1
    finally:
        baseline_batch.close()
        branch_batch.close()


def test_shared_prefix_b2_full_accept_reuses_first_tile_and_commits_exact_state():
    baseline_batch, baseline, _, _ = run_round(4, None, proposal_count=7)
    branch_batch, branch_lane, calls, _ = run_round(
        4,
        4,
        proposal_count=7,
        alternative_accept_branch=1,
        alternative_strategy="shared_prefix",
    )
    try:
        assert branch_lane.history == baseline.history
        assert branch_lane.anchor == baseline.anchor
        assert branch_lane.rng.snapshot() == baseline.rng.snapshot()
        close_caches(branch_lane.cache, baseline.cache)
        receipt = branch_lane.ready[-1].speculative_receipt[
            "progressive_verification"
        ]["ordered_alternatives"]
        assert receipt["last_branch_accepts"] == [4, 7, -1, -1]
        assert receipt["last_branch_outcomes"] == [
            "partial_or_rejected",
            "full_accept",
            "unvisited",
            "unvisited",
        ]
        assert receipt["last_selected_branch"] == 2
        assert receipt["last_strategy"] == "shared_prefix"
        assert receipt["last_shared_prefix_tokens"] == 4
        assert receipt["last_unique_target_rows"] == 12
        assert len(calls) == 2
        assert [len(call) for call in calls] == [4, 4]
        assert branch_batch.scheduler_stats[
            "external_progressive_alternative_shared_prefix_rounds"
        ] == 1
    finally:
        baseline_batch.close()
        branch_batch.close()


def test_shared_prefix_no_full_leaf_runs_b2_cohorts_then_falls_back_to_b1():
    baseline_batch, baseline, _, _ = run_round(4, 4, proposal_count=7)
    branch_batch, branch_lane, calls, _ = run_round(
        4,
        4,
        proposal_count=7,
        alternative_accept_branch=9,
        alternative_strategy="shared_prefix",
    )
    try:
        assert branch_lane.history == baseline.history
        assert branch_lane.anchor == baseline.anchor
        assert branch_lane.rng.snapshot() == baseline.rng.snapshot()
        close_caches(branch_lane.cache, baseline.cache)
        receipt = branch_lane.ready[-1].speculative_receipt[
            "progressive_verification"
        ]["ordered_alternatives"]
        assert receipt["last_branch_accepts"] == [4, 4, 4, 4]
        assert receipt["last_selected_branch"] == 1
        assert receipt["last_short_circuit_after_branch"] is None
        assert receipt["last_unique_target_rows"] == 20
        assert len(calls) == 3
        assert [len(call) for call in calls] == [4, 4, 4]
    finally:
        baseline_batch.close()
        branch_batch.close()


def test_lilicorr_shared_prefix_paths_keep_b1_and_fan_out_after_tile():
    slots, topk = 7, 4
    hosted = np.full((slots, topk, topk), -20.0, np.float32)
    ids = np.arange(slots * topk).reshape(slots, topk)
    predecessor = 0
    greedy = []
    for position in range(slots):
        hosted[position, predecessor] = np.array([4.0, 3.0, 2.0, 1.0])
        predecessor = 0
        greedy.append(int(ids[position, 0]))
    paths = _ordered_lilicorr_paths(
        hosted,
        ids,
        greedy,
        4,
        strategy="shared_prefix",
        shared_prefix_tokens=4,
    )
    assert paths[0] == greedy
    assert all(path[:4] == greedy[:4] for path in paths)
    assert [path[4] for path in paths] == ids[4].tolist()


def test_lilicorr_paired_continuations_branch_late_from_top_two_parents():
    slots, topk = 7, 4
    hosted = np.full((slots, topk, topk), -20.0, np.float32)
    ids = np.arange(slots * topk).reshape(slots, topk)
    greedy = []
    predecessor = 0
    for position in range(slots):
        hosted[position, predecessor] = np.array([4.0, 3.0, 2.0, 1.0])
        predecessor = 0
        greedy.append(int(ids[position, 0]))
    paths = _ordered_lilicorr_paths(
        hosted,
        ids,
        greedy,
        4,
        strategy="paired_continuations",
        shared_prefix_tokens=4,
    )
    assert paths[0] == greedy
    assert all(path[:4] == greedy[:4] for path in paths)
    assert paths[0][4] == paths[2][4]
    assert paths[1][4] == paths[3][4]
    assert paths[0][5] != paths[2][5]
    assert paths[1][5] != paths[3][5]


def test_confidence_tree_uses_two_token_trunk_and_commits_rescue_branch():
    baseline_batch, baseline, _, _ = run_round(4, None, proposal_count=7)
    branch_batch, branch_lane, _calls, _proposal = run_round(
        4,
        2,
        proposal_count=7,
        alternative_accept_branch=1,
        alternative_strategy="confidence_tree",
        alternative_depth_policy="confidence_tail",
        confidence_features=[2.0, 1.0, 0.5, -2.0, -2.0, -2.0, -2.0],
    )
    try:
        assert branch_lane.history == baseline.history
        assert branch_lane.anchor == baseline.anchor
        close_caches(branch_lane.cache, baseline.cache)
        alternatives = branch_lane.ready[-1].speculative_receipt[
            "progressive_verification"
        ]["ordered_alternatives"]
        assert alternatives["last_shared_prefix_tokens"] == 2
        assert alternatives["last_active_branches"] == 4
        assert alternatives["last_branch_proposal_counts"] == [7, 7, 3, 3]
        assert alternatives["last_selected_branch"] == 2
        assert alternatives["last_branch_accepts"] == [2, 7, 2, 2]
        progressive = branch_lane.ready[-1].speculative_receipt[
            "progressive_verification"
        ]
        assert progressive["ragged_launches"] >= 1
        assert progressive["padding_rows"] >= 4
        decision = alternatives["adaptive_latch"]["last_decision"]
        assert decision["completed_samples"] == 0
        assert decision["suggested_trunk_tokens"] == 3
        assert decision["trunk_tokens"] == 2
        observation = alternatives["adaptive_latch"]["last_observation"]
        assert observation["selected_branch"] == 2
        assert observation["proposal_counts"] == [7, 7, 3, 3]
        assert observation["snapshot_after"]["alternative_wins"] == 1
    finally:
        baseline_batch.close()
        branch_batch.close()


def test_confidence_tree_trunk_and_lagged_latch_retopology():
    assert confidence_tree_trunk_length([2, 1, -1, 3], 4, 4) == 2
    assert confidence_tree_trunk_length([2, 1, 0.5, 3], 4, 4) == 3
    assert confidence_tree_trunk_length([-1, 2, 2, 2], 4, 4) == 1
    assert confidence_tree_branch_depths(7, 4, 2) == (7, 7, 3, 3)
    assert confidence_tree_branch_depths(7, 3, 2) == (7, 7, 3)

    latch = AdaptiveTopologyLatch(
        4, shrink_after=1, park_rounds=2, reprobe_interval=5
    )
    first = latch.choose(2, 7, 4)
    assert first.active_branches == 4
    latch.observe(
        branch_accepts=[3, 2, 1, 0],
        proposal_counts=[7, 7, 3, 3],
        selected_branch=1,
        unique_target_rows=16,
    )
    second = latch.choose(2, 7, 4)
    assert second.active_branches == 3
    assert second.completed_samples == 1
    latch.observe(
        branch_accepts=[3, 2, 1],
        proposal_counts=[7, 7, 3],
        selected_branch=1,
        unique_target_rows=13,
    )
    third = latch.choose(2, 7, 4)
    assert third.active_branches == 2
    latch.observe(
        branch_accepts=[3, 2],
        proposal_counts=[7, 7],
        selected_branch=1,
        unique_target_rows=10,
    )
    parked = latch.choose(2, 7, 4)
    assert parked.active_branches == 1
    assert parked.reason == "parked_after_repeated_no_alternative_win"
    # The fifth decision is a configured-width reprobe based only on the
    # already completed samples; it cannot see its own pending outcome.
    reprobe = latch.choose(2, 7, 4)
    assert reprobe.active_branches == 4
    assert reprobe.reprobe is True


def test_paired_continuations_short_terminal_round_uses_primary_chain():
    batch, lane, calls, proposal = run_round(
        4,
        None,
        proposal_count=5,
        alternative_accept_branch=9,
        alternative_strategy="paired_continuations",
    )
    try:
        assert lane.uid not in batch._last_alternative_blocks
        assert lane.history[-5:] == proposal
        assert [len(call) for call in calls] == [4, 2]
        receipt = lane.ready[-1].speculative_receipt["progressive_verification"]
        alternatives = receipt["ordered_alternatives"]
        assert alternatives["configured_strategy"] == "paired_continuations"
        assert alternatives["rounds"] == 0
        assert alternatives["last_branch_accepts"] == []
        assert alternatives["committed_physical_width"] == 0
    finally:
        batch.close()


def dspark_alternative_round(strategy, *, max_tokens, temp):
    mx.set_default_device(mx.cpu)
    model = tiny_model()
    draft = DSparkDraftModel(DSparkConfig(
        hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        vocab_size=16, block_size=5, mask_token_id=15,
        target_layer_ids=[0, 1, 4], num_target_layers=5,
        layer_types=["sliding_attention"], sliding_window=8, markov_rank=2,
    )).bind(model)
    batch = ExternalDraftBatchGenerator(model, draft_model=draft, binding="dspark-alt-cpu",
        num_draft=4, completion_batch_size=1, prefill_step_size=8,
        progressive_verification_tile=2, progressive_alternative_branches=2,
        progressive_alternative_strategy=strategy)
    uid = batch.insert([[1, 2, 3, 4]], max_tokens=[max_tokens], lane_rngs=[LaneRNG(7)],
                       sampling_configs=[{"sampling_temp": temp}])[0]
    lane = batch.lanes[uid]
    while lane.anchor is None:
        batch._prefill(lane)
    batch._round([lane])
    return batch, lane


@pytest.mark.parametrize("strategy", ["distinct_root", "shared_prefix"])
def test_ordered_alternatives_sampled_lane_uses_primary_chain(strategy):
    # Alternative verification is greedy-only; a sampled request must keep
    # the primary progressive chain instead of failing after prefill.
    batch, lane = dspark_alternative_round(strategy, max_tokens=16, temp=1.0)
    try:
        assert lane.uid not in batch._last_alternative_blocks
        assert lane.ready
        alternatives = lane.ready[-1].speculative_receipt[
            "progressive_verification"
        ]["ordered_alternatives"]
        assert alternatives["rounds"] == 0
        assert alternatives["sampled_primary_rounds"] == 1
        assert batch.scheduler_stats["external_progressive_alternative_rounds"] == 0
        assert batch.scheduler_stats[
            "external_progressive_alternative_sampled_primary_rounds"
        ] == 1
    finally:
        batch.close()


def test_shared_prefix_short_terminal_round_uses_primary_chain():
    # max_tokens=3 requests two proposals, so the tile-2 divergence point is
    # outside the proposal and the round must not ask for alternatives.
    batch, lane = dspark_alternative_round("shared_prefix", max_tokens=3, temp=0.0)
    try:
        assert lane.uid not in batch._last_alternative_blocks
        assert lane.ready
    finally:
        batch.close()


def test_ordered_alternative_stop_token_falls_back_to_primary_chain():
    baseline_batch, baseline, _, _ = run_round(4, None, proposal_count=7)
    guarded_batch, guarded, _, _ = run_round(
        4,
        None,
        proposal_count=7,
        alternative_accept_branch=9,
        alternative_stop_branch=1,
    )
    try:
        assert guarded.history == baseline.history
        assert guarded.anchor == baseline.anchor
        assert guarded.rng.snapshot() == baseline.rng.snapshot()
        close_caches(guarded.cache, baseline.cache)
        receipt = guarded.ready[-1].speculative_receipt[
            "progressive_verification"
        ]["ordered_alternatives"]
        assert receipt["rounds"] == 0
        assert receipt["stop_guard_primary_rounds"] == 1
        assert guarded_batch.scheduler_stats[
            "external_progressive_alternative_stop_guard_rounds"
        ] == 1
    finally:
        baseline_batch.close()
        guarded_batch.close()


def test_paired_continuations_verifies_b3_with_b1_before_b2_family():
    batch, lane, calls, _proposal = run_round(
        4,
        None,
        proposal_count=7,
        alternative_accept_branch=2,
        alternative_strategy="paired_continuations",
    )
    try:
        alternatives = lane.ready[-1].speculative_receipt[
            "progressive_verification"
        ]["ordered_alternatives"]
        assert alternatives["last_branch_accepts"] == [7, -1, 7, -1]
        assert alternatives["last_branch_outcomes"] == [
            "full_accept",
            "unvisited",
            "full_accept",
            "unvisited",
        ]
        assert alternatives["last_selected_branch"] == 1
        assert alternatives["committed_physical_width"] == 2
        assert alternatives["last_unique_target_rows"] == 11
        assert [len(call) for call in calls] == [4, 3]
    finally:
        batch.close()


def test_paired_continuations_b3_can_rescue_failed_b1_family():
    baseline_batch, baseline, _, _ = run_round(4, None, proposal_count=7)
    branch_batch, branch_lane, calls, _proposal = run_round(
        4,
        5,
        proposal_count=7,
        alternative_accept_branch=2,
        alternative_strategy="paired_continuations",
    )
    try:
        assert branch_lane.history == baseline.history
        assert branch_lane.anchor == baseline.anchor
        close_caches(branch_lane.cache, baseline.cache)
        alternatives = branch_lane.ready[-1].speculative_receipt[
            "progressive_verification"
        ]["ordered_alternatives"]
        assert alternatives["last_branch_accepts"] == [5, -1, 7, -1]
        assert alternatives["last_selected_branch"] == 3
        assert alternatives["last_short_circuit_after_branch"] == 3
        assert alternatives["last_unique_target_rows"] == 11
        assert [len(call) for call in calls] == [4, 3]
    finally:
        baseline_batch.close()
        branch_batch.close()


def exact_caches(left, right):
    for actual, expected in zip(left, right, strict=True):
        assert type(actual) is type(expected)
        if isinstance(actual, ArraysCache):
            for a, b in zip(actual.cache, expected.cache, strict=True):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        else:
            assert actual.offset == expected.offset
            for a, b in zip(actual.state, expected.state, strict=True):
                np.testing.assert_array_equal(np.asarray(a)[..., :actual.offset, :],
                                              np.asarray(b)[..., :expected.offset, :])


def close_caches(left, right, *, atol=1e-6):
    for actual, expected in zip(left, right, strict=True):
        assert type(actual) is type(expected)
        if isinstance(actual, ArraysCache):
            pairs = zip(actual.cache, expected.cache, strict=True)
        else:
            assert actual.offset == expected.offset
            pairs = (
                (
                    np.asarray(got)[..., : actual.offset, :],
                    np.asarray(want)[..., : expected.offset, :],
                )
                for got, want in zip(actual.state, expected.state, strict=True)
            )
        for got, want in pairs:
            np.testing.assert_allclose(
                np.asarray(got), np.asarray(want), rtol=0, atol=atol
            )


@pytest.mark.parametrize("reject_at", [None, *range(7)])
def test_two_tiles_preserve_hybrid_state_rng_taps_and_ordinary_next_token(reject_at):
    fixed_batch, fixed, _, _ = run_round(None, reject_at)
    tiled_batch, tiled, calls, proposal = run_round(4, reject_at)
    try:
        assert tiled.history == fixed.history
        assert tiled.anchor == fixed.anchor
        assert tiled.generated == fixed.generated and tiled.accepted == fixed.accepted
        assert tiled.rng.snapshot() == fixed.rng.snapshot()
        assert [r.token for r in tiled.ready] == [r.token for r in fixed.ready]
        np.testing.assert_array_equal(np.asarray(tiled.tail), np.asarray(fixed.tail))
        exact_caches(tiled.cache, fixed.cache)
        exact_caches(tiled.draft_cache, fixed.draft_cache)
        assert calls[0] == [4, *proposal[:3]]
        if reject_at is None or reject_at >= 4:
            assert calls == [[4, *proposal[:3]], proposal[3:7]]
        else:
            assert len(calls) == 1
        left = tiled_batch.model(mx.array([[tiled.anchor]]), copy.deepcopy(tiled.cache))
        right = fixed_batch.model(mx.array([[fixed.anchor]]), copy.deepcopy(fixed.cache))
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
        receipt = tiled.ready[-1].speculative_receipt["progressive_verification"]
        assert receipt["observed_used"] and receipt["verification_tile"] == 4
        assert not receipt["qualified"]
    finally:
        fixed_batch.close()
        tiled_batch.close()


@pytest.mark.parametrize("reject_at", [None, *range(11)])
def test_three_tiles_checkpoint_hybrid_state_and_match_fixed_verifier(reject_at):
    fixed_batch, fixed, _, _ = run_round(
        None, reject_at, proposal_count=11
    )
    tiled_batch, tiled, calls, proposal = run_round(
        4, reject_at, proposal_count=11
    )
    try:
        assert tiled.history == fixed.history
        assert tiled.anchor == fixed.anchor
        assert tiled.generated == fixed.generated
        assert tiled.accepted == fixed.accepted
        assert tiled.rng.snapshot() == fixed.rng.snapshot()
        assert [item.token for item in tiled.ready] == [
            item.token for item in fixed.ready
        ]
        np.testing.assert_array_equal(np.asarray(tiled.tail), np.asarray(fixed.tail))
        exact_caches(tiled.cache, fixed.cache)
        exact_caches(tiled.draft_cache, fixed.draft_cache)

        expected_calls = [[4, *proposal[:3]]]
        if reject_at is None or reject_at >= 4:
            expected_calls.append(proposal[3:7])
        if reject_at is None or reject_at >= 8:
            expected_calls.append(proposal[7:11])
        assert calls == expected_calls

        next_tiled = tiled_batch.model(
            mx.array([[tiled.anchor]]), copy.deepcopy(tiled.cache)
        )
        next_fixed = fixed_batch.model(
            mx.array([[fixed.anchor]]), copy.deepcopy(fixed.cache)
        )
        np.testing.assert_array_equal(np.asarray(next_tiled), np.asarray(next_fixed))

        receipt = tiled.ready[-1].speculative_receipt["progressive_verification"]
        expected_launches = 3 if reject_at is None else reject_at // 4 + 1
        assert receipt["target_launches"] == expected_launches
        assert receipt["last_tile_input_rows"] == [4] * expected_launches
        expected_checkpoints = (
            2 if reject_at is None else min(reject_at // 4, 2)
        )
        assert len(receipt["last_tile_checkpoints"]) == expected_checkpoints
        assert receipt["last_rollback_base_tile"] == expected_checkpoints
        assert receipt["checkpoint_state"] == (
            "joint_attention_kv_and_recurrent_layer_state"
        )
        assert receipt["rollback_policy"] == (
            "last_accepted_tile_then_exact_prefix"
        )
        assert receipt["state_promotion"] == (
            "request_private_then_atomic_lane_publish"
        )
        if reject_at is None:
            assert receipt["target_rows"] == 12
            assert receipt["full_nonfinal_tiles"] == 2
            assert len(tiled.ready) == 12
    finally:
        fixed_batch.close()
        tiled_batch.close()


def test_nemotron_mamba_apcv2_segments_include_internal_checkpoint():
    from mlx2.runtime.cache_operations import cache_operations
    from mlx2.runtime.cache_planes import CachePlaneKind
    from mlx2.runtime.cow_cache import (
        _build_layer_segment_manifest,
        freeze_prompt_cache,
    )

    model = tiny_model()
    cache = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3, 4]], dtype=mx.uint32), cache=cache))
    for entry in cache:
        if isinstance(entry, ArraysCache):
            entry.state_checkpoint([4], force=True)
            contract = cache_operations(entry, capability="Nemotron APCv2 test")
            assert [plane.kind for plane in contract.planes] == [
                "mamba_recurrent"
            ]

    manifest = _build_layer_segment_manifest(
        cache, None, covered_tokens=4
    )
    mamba = [
        segment
        for segment in manifest.segments
        if segment.key.kind == CachePlaneKind.MAMBA_RECURRENT
    ]
    assert {segment.key.layer_index for segment in mamba} == {1, 4}
    assert {segment.key.role for segment in mamba} == {"state", "checkpoint"}
    for layer_index in (1, 4):
        layer = [segment for segment in mamba if segment.key.layer_index == layer_index]
        assert [(segment.key.role, segment.key.token_stop) for segment in layer] == [
            ("state", 4),
            ("checkpoint", 4),
        ]

    frozen, _ = freeze_prompt_cache(
        cache,
        key="nemotron-mamba-checkpoint",
        tokens=range(4),
        cache_type="assistant",
        layer_segments=True,
    )
    branch = frozen.branch()
    try:
        summary = frozen.cow_owner.segment_manifest.summary()["by_plane"][
            "mamba_recurrent"
        ]
        assert summary["layers"] == 2
        assert summary["segments"] == 4
        assert summary["logical_bytes"] > 0
        assert summary["invalid"] == 0
        recurrent = [entry for entry in branch if isinstance(entry, ArraysCache)]
        assert all(
            sorted(entry._cow_segment_tokens) == ["mamba_recurrent"]
            for entry in recurrent
        )
        assert all(entry.meta_state == ("ckptv1", "4") for entry in recurrent)
    finally:
        branch.close()
        frozen.close()
