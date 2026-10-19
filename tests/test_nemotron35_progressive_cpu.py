"""Real hybrid rollback/tap/RNG parity for seven proposals in 4+4 rows."""
import copy
import itertools
from types import SimpleNamespace

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
    ordered_alternative_winner,
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
        assert receipt["last_branch_committed_tokens"] == [1, 9, 1, 1]
        assert receipt["last_selected_token_uplift"] == 8
        assert receipt["last_short_circuit_after_branch"] == 2
        assert receipt["last_discarded_branches"] == [1, 3, 4]
        assert receipt["b2_full_accept_short_circuits"] == 1
        assert receipt["next_round_base"] == "selected_branch_committed_state"
        assert receipt["winner_selection"] == (
            "first_full_branch_with_at_least_b1_committed_tokens"
        )
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


def test_confidence_tail_full_branch_cannot_replace_longer_b1_partial_path():
    baseline_batch, baseline, _, _ = run_round(4, 4, proposal_count=7)
    branch_batch, branch_lane, _calls, _proposal = run_round(
        4,
        4,
        proposal_count=7,
        alternative_accept_branch=2,
        alternative_strategy="confidence_tree",
        alternative_depth_policy="confidence_tail",
        confidence_features=[2.0, 1.0, 0.5, -2.0, -2.0, -2.0, -2.0],
    )
    try:
        assert branch_lane.history == baseline.history
        assert branch_lane.anchor == baseline.anchor
        assert branch_lane.rng.snapshot() == baseline.rng.snapshot()
        close_caches(branch_lane.cache, baseline.cache)
        alternatives = branch_lane.ready[-1].speculative_receipt[
            "progressive_verification"
        ]["ordered_alternatives"]
        assert alternatives["last_branch_proposal_counts"] == [7, 7, 3, 3]
        assert alternatives["last_branch_accepts"] == [4, 2, 3, 2]
        assert alternatives["last_full_accept_branches"] == [3]
        assert alternatives["last_selected_branch"] == 1
        assert alternatives["last_branch_committed_tokens"] == [5, 3, 4, 3]
        assert alternatives["last_selected_token_uplift"] == 0
        assert alternatives["last_short_circuit_after_branch"] == 3
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
    assert second.active_branches == 2
    assert second.completed_samples == 1
    latch.observe(
        branch_accepts=[3, 2],
        proposal_counts=[7, 7],
        selected_branch=1,
        unique_target_rows=10,
    )
    parked = latch.choose(2, 7, 4)
    assert parked.active_branches == 1
    assert parked.reason == "parked_after_repeated_no_alternative_win"
    assert latch.park_remaining == 2
    latch.observe_primary_only(
        decision_token=parked.decision_token,
        session_count=1,
    )
    assert latch.park_remaining == 1
    assert latch.snapshot()["primary_only_samples"] == 1
    parked_again = latch.choose(2, 7, 4)
    assert parked_again.active_branches == 1
    latch.observe_primary_only(
        decision_token=parked_again.decision_token,
        session_count=1,
    )
    assert latch.park_remaining == 0
    # The fifth decision is a configured-width reprobe based only on the
    # already completed samples; it cannot see its own pending outcome.
    reprobe = latch.choose(2, 7, 4)
    assert reprobe.active_branches == 4
    assert reprobe.reprobe is True
    latch.observe_many(
        [
            {
                "branch_accepts": [3, 7, 1, 0],
                "proposal_counts": [7, 7, 3, 3],
                "selected_branch": 2,
                "unique_target_rows": 16,
                "outcome": "committed_conclusive",
            }
        ],
        decision_token=reprobe.decision_token,
    )
    recovered = latch.choose(2, 7, 4)
    assert recovered.active_branches == 2
    assert latch.park_remaining == 0
    assert latch.snapshot()["periodic_reprobes"] == 1
    assert latch.snapshot()["accepted_token_uplift"] == 4


def test_ordered_alternative_winner_preserves_longer_b1_partial_commit():
    decisions = [
        SimpleNamespace(accepted=4, emitted=list(range(5)), commit_count=5),
        SimpleNamespace(accepted=3, emitted=list(range(4)), commit_count=4),
        SimpleNamespace(accepted=3, emitted=list(range(4)), commit_count=4),
    ]
    winner, full = ordered_alternative_winner(decisions, [7, 3, 3])
    assert full == (1, 2)
    assert winner == 0


def test_ordered_alternative_winner_keeps_first_nonlosing_full_branch():
    decisions = [
        SimpleNamespace(accepted=3, emitted=list(range(4)), commit_count=None),
        SimpleNamespace(accepted=3, emitted=list(range(4)), commit_count=None),
        SimpleNamespace(accepted=4, emitted=list(range(5)), commit_count=None),
    ]
    winner, full = ordered_alternative_winner(decisions, [7, 3, 4])
    assert full == (1, 2)
    assert winner == 1


def test_ordered_alternative_winner_can_select_exact_partial_uplift():
    decisions = [
        SimpleNamespace(accepted=1, emitted=[1, 2], commit_count=2),
        SimpleNamespace(accepted=2, emitted=[1, 3, 4], commit_count=3),
        SimpleNamespace(accepted=0, emitted=[5], commit_count=1),
    ]
    winner, full = ordered_alternative_winner(
        decisions, [3, 3, 3], allow_partial_uplift=True
    )
    assert full == ()
    assert winner == 1
    default_winner, _ = ordered_alternative_winner(decisions, [3, 3, 3])
    assert default_winner == 0


def test_ast_zero_uplift_alternative_selection_does_not_reset_parking():
    latch = AdaptiveTopologyLatch(2, shrink_after=1, park_rounds=2)
    decision = latch.choose(2, 7, 4)
    latch.observe_many(
        [
            {
                "branch_accepts": [3, 3],
                "proposal_counts": [7, 3],
                "selected_branch": 2,
                "unique_target_rows": 8,
                "outcome": "committed_conclusive",
            }
        ],
        decision_token=decision.decision_token,
    )
    snapshot = latch.snapshot()
    assert snapshot["alternative_selections"] == 1
    assert snapshot["alternative_wins"] == 0
    assert snapshot["zero_uplift_alternative_selections"] == 1
    assert snapshot["accepted_token_uplift"] == 0
    assert snapshot["active_branches"] == 1
    assert snapshot["park_remaining"] == 2


def test_ast_unavailable_reprobe_advances_clock_without_training():
    latch = AdaptiveTopologyLatch(2, reprobe_interval=1)
    decision = latch.choose(1, 3, 2)
    assert decision.reprobe
    latch.observe_primary_only(
        decision_token=decision.decision_token, session_count=1, unavailable=True
    )
    assert latch.choose(1, 3, 2).decision_token == 2
    assert not latch.window
    assert latch.no_alternative_win == 0
    assert latch.snapshot()["unavailable_samples"] == 1


def test_ast_no_uplift_without_incremental_cost_does_not_downshift():
    latch = AdaptiveTopologyLatch(4, shrink_after=1)
    decision = latch.choose(2, 7, 4)
    latch.observe_many(
        [
            {
                "branch_accepts": [3, 2, 1, 0],
                "proposal_counts": [7, 7, 3, 3],
                "selected_branch": 1,
                "unique_target_rows": 4,
                "outcome": "committed_conclusive",
            }
        ],
        decision_token=decision.decision_token,
    )
    snapshot = latch.snapshot()
    assert snapshot["active_branches"] == 4
    assert snapshot["downshifts"] == 0
    assert snapshot["avoidable_target_rows"] == 0


def test_parked_ast_primary_commit_advances_sparse_audit_and_receipt():
    batch, lane, _calls, _proposal = run_round(
        4,
        2,
        proposal_count=7,
        alternative_accept_branch=9,
        alternative_strategy="confidence_tree",
        alternative_depth_policy="confidence_tail",
        confidence_features=[2.0, 1.0, 0.5, -2.0, -2.0, -2.0, -2.0],
    )
    try:
        # The latch is per session: park only this request's latch.
        latch = batch.progressive_topology_latch.session(lane.uid)
        latch.active_branches = 1
        latch.park_remaining = 2
        before = latch.snapshot()["last_applied_token"]
        batch._round([lane])
        receipt = lane.ready[-1].speculative_receipt[
            "progressive_verification"
        ]["ordered_alternatives"]
        observation = receipt["adaptive_latch"]["last_observation"]
        assert receipt["last_active_branches"] == 1
        assert receipt["last_primary_fallback_reason"] == "adaptive_utility_latch"
        assert observation["outcome"] == "committed_primary_only"
        assert observation["executed_alternative_branches"] == 0
        assert observation["snapshot_after"]["last_applied_token"] == before + 1
        assert observation["snapshot_after"]["primary_only_samples"] == 1
        assert batch.scheduler_stats[
            "external_progressive_ast_primary_only_rounds"
        ] == 1
        assert batch.scheduler_stats[
            "external_progressive_ast_utility_downshift_sessions"
        ] == 1
    finally:
        batch.close()


def test_ast_branch_safe_minimum_processor_is_not_censored():
    from types import SimpleNamespace

    from mlx2.serving import minimum_tokens_processor

    batch, lane, _calls, _proposal = run_round(
        4,
        2,
        proposal_count=7,
        alternative_accept_branch=9,
        alternative_strategy="confidence_tree",
        alternative_depth_policy="confidence_tail",
        confidence_features=[2.0, 1.0, 0.5, -2.0, -2.0, -2.0, -2.0],
    )
    try:
        lane.processors = [
            minimum_tokens_processor(mx, [0], len(lane.history), 8)
        ]
        decision = SimpleNamespace(emitted=[1])
        outcome = batch._topology_observation_outcome(
            lane, decision, [2, 1, 0, 0], [7, 7, 3, 3], 1
        )
        assert outcome["outcome"] == "committed_conclusive"
        assert "processor_limited" not in outcome["censor_reasons"]

        previous_maximum = lane.maximum
        lane.maximum = lane.generated + len(decision.emitted)
        terminal = batch._topology_observation_outcome(
            lane, decision, [2, 1, 0, 0], [7, 7, 3, 3], 1
        )
        assert terminal["outcome"] == "committed_censored"
        assert terminal["budget_limited"]
        lane.maximum = previous_maximum

        lane.processors = [lambda _tokens, logits: logits]
        outcome = batch._topology_observation_outcome(
            lane, decision, [2, 1, 0, 0], [7, 7, 3, 3], 1
        )
        assert outcome["outcome"] == "committed_censored"
        assert "processor_limited" in outcome["censor_reasons"]
    finally:
        batch.close()


def test_ast_cohort_choice_consumes_one_lagged_decision_epoch():
    latch = AdaptiveTopologyLatch(4, reprobe_interval=3)
    decisions = latch.choose_many([1, 2, 1, 2], [7, 7, 7, 7], 4)
    assert latch.decisions == 0
    assert {decision.active_branches for decision in decisions} == {4}
    assert [decision.trunk_tokens for decision in decisions] == [1, 2, 1, 2]
    assert not any(decision.reprobe for decision in decisions)


def test_ast_cohort_observation_is_order_independent_and_idempotent():
    rows = [
        {
            "branch_accepts": [3, 2, 1, 0],
            "proposal_counts": [7, 7, 3, 3],
            "selected_branch": selected,
            "unique_target_rows": 16 + selected,
            "outcome": "committed_conclusive",
        }
        for selected in (1, 2, 1, 1)
    ]
    snapshots = []
    for ordering in itertools.permutations(rows):
        latch = AdaptiveTopologyLatch(4, shrink_after=1)
        decisions = latch.choose_many([2] * 4, [7] * 4, 4)
        token = decisions[0].decision_token
        assert latch.observe_many(ordering, decision_token=token) is True
        assert latch.observe_many(ordering, decision_token=token) is False
        snapshots.append(latch.snapshot())
    assert all(snapshot == snapshots[0] for snapshot in snapshots)
    assert snapshots[0]["completed_samples"] == 1
    assert snapshots[0]["completed_sessions"] == 4
    assert snapshots[0]["alternative_selections"] == 1
    assert snapshots[0]["alternative_wins"] == 0
    assert snapshots[0]["zero_uplift_alternative_selections"] == 1


def test_ast_censored_cohort_cannot_shrink_and_out_of_order_token_fails():
    latch = AdaptiveTopologyLatch(4, shrink_after=1)
    decision = latch.choose(2, 7, 4)
    latch.observe_many(
        [
            {
                "branch_accepts": [3, -1, -1, -1],
                "proposal_counts": [7, 7, 3, 3],
                "selected_branch": 1,
                "unique_target_rows": 4,
                "outcome": "committed_censored",
            }
        ],
        decision_token=decision.decision_token,
    )
    assert latch.active_branches == 4
    assert latch.no_alternative_win == 0
    assert latch.snapshot()["censored_sessions"] == 1
    with pytest.raises(ValueError, match="stale or out of order"):
        latch.observe_many(
            [
                {
                    "branch_accepts": [3, 2, 1, 0],
                    "proposal_counts": [7, 7, 3, 3],
                    "selected_branch": 1,
                    "unique_target_rows": 16,
                    "outcome": "committed_conclusive",
                }
            ],
            decision_token=3,
        )


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


def run_cross_session_ast(
    pack_rows=None,
    *,
    execute=True,
    pack_mode=None,
    force_full_alternative=True,
    truncate_branch=False,
):
    mx.set_default_device(mx.cpu)
    mx.random.seed(81)
    args = tiny_model().args
    model = SuperModel.__new__(SuperModel)
    LanguageModel.__init__(model, args)
    proposal_count = 7
    draft = XPressDraftModel(XPressConfig(
        hidden_size=8, intermediate_size=12, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        vocab_size=16, mask_token_id=15, num_target_layers=5,
        target_layer_ids=[0, 1, 4], block_size=proposal_count + 1,
        layer_types=["full_attention"], xpress_rank=3,
        xpress_mlp_hidden=5, xpress_num_passes=3,
    )).bind(model)
    prompts = [[1 + row, 2 + row, 3 + row, 4 + row] for row in range(4)]
    exact_rows = []
    for prompt in prompts:
        cache = model.make_cache()
        model(mx.array([prompt[:-1]]), cache)
        token = prompt[-1]
        exact = []
        for _ in range(proposal_count):
            token = int(mx.argmax(model(mx.array([[token]]), cache)[0, -1]).item())
            exact.append(token)
        exact_rows.append(exact)
    primary_rows = [list(row) for row in exact_rows]
    if pack_rows is not None:
        for session, row in enumerate(primary_rows):
            if (
                pack_mode == "ordered_live_frontier"
                and force_full_alternative
                and session == 0
            ):
                continue
            row[2] = (row[2] + session + 1) % args.vocab_size

    primary_laws = []
    for row in primary_rows:
        laws = []
        for token in row:
            law = np.zeros(args.vocab_size, np.float64)
            law[token] = 1.0
            laws.append(law)
        primary_laws.append(laws)
    original = draft.draft_distributions
    expected_anchors = [prompt[-1] for prompt in prompts]

    def forced(anchors, hidden, caches, count, rngs, temps, **kwargs):
        generated_tokens, generated_laws = original(
            anchors, hidden, caches, count, rngs, temps, **kwargs
        )
        if anchors != expected_anchors:
            return generated_tokens, generated_laws
        draft.adaptive_confidence_features = [
            (
                [2.0, -1.0, -2.0, -2.0, -2.0, -2.0, -2.0]
                if row % 2 == 0
                else [2.0, 1.0, -1.0, -2.0, -2.0, -2.0, -2.0]
            )
            for row, _anchor in enumerate(anchors)
        ]
        return [list(row) for row in primary_rows], [
            list(row) for row in primary_laws
        ]

    def alternatives(
        primary_tokens, *, branch_count, strategy, shared_prefix_tokens
    ):
        assert strategy == "confidence_tree"
        assert list(shared_prefix_tokens) == [1, 2, 1, 2]
        result = []
        for session, exact in enumerate(exact_rows):
            second = list(exact)
            if not force_full_alternative:
                suffix_start = int(shared_prefix_tokens[session])
                second[suffix_start] = (
                    second[suffix_start] + session + 9
                ) % args.vocab_size
            branches = [list(primary_tokens[session]), second]
            for branch in range(2, branch_count):
                candidate = list(exact)
                mutation = (
                    int(shared_prefix_tokens[session])
                    if not force_full_alternative
                    else 2
                )
                candidate[mutation] = (
                    candidate[mutation] + session + branch + 1
                ) % args.vocab_size
                branches.append(candidate)
            if truncate_branch and session == 1:
                # A processor-masked draft can end a branch at its shared
                # prefix (no finite candidate past a forbidden token).
                branches[1] = list(branches[1][: int(shared_prefix_tokens[session])])
            result.append(branches)
        return result

    draft.draft_distributions = forced
    draft.ordered_alternative_proposals = alternatives
    batch = ExternalDraftBatchGenerator(
        model,
        draft_model=draft,
        binding="super-cross-session-ast-cpu",
        num_draft=proposal_count,
        completion_batch_size=4,
        prefill_step_size=8,
        progressive_verification_tile=4,
        progressive_alternative_branches=4 if pack_rows is not None else 1,
        progressive_alternative_strategy=(
            "confidence_tree" if pack_rows is not None else "distinct_root"
        ),
        progressive_alternative_depth_policy=(
            "confidence_tail" if pack_rows is not None else "full"
        ),
        progressive_cross_session_pack_rows=pack_rows,
        progressive_cross_session_pack_mode=pack_mode,
    )
    uids = batch.insert(
        prompts,
        max_tokens=[16] * 4,
        lane_rngs=[LaneRNG(1200 + row) for row in range(4)],
        sampling_configs=[{"sampling_temp": 0}] * 4,
    )
    lanes = [batch.lanes[uid] for uid in uids]
    for lane in lanes:
        while lane.anchor is None:
            batch._prefill(lane)
    if execute:
        batch._round(lanes)
    return batch, lanes


@pytest.mark.parametrize("pack_rows", [4, 8])
def test_cross_session_ast_packs_four_private_sessions_and_matches_b1(pack_rows):
    baseline_batch, baseline = run_cross_session_ast()
    packed_batch, packed = run_cross_session_ast(pack_rows)
    try:
        for actual, expected in zip(packed, baseline, strict=True):
            assert actual.history == expected.history
            assert actual.anchor == expected.anchor
            assert actual.generated == expected.generated
            assert actual.accepted == expected.accepted
            assert actual.rng.snapshot() == expected.rng.snapshot()
            assert [item.token for item in actual.ready] == [
                item.token for item in expected.ready
            ]
            close_caches(actual.cache, expected.cache)
            alternatives = actual.ready[-1].speculative_receipt[
                "progressive_verification"
            ]["ordered_alternatives"]
            assert alternatives["last_selected_branch"] == 2
            assert alternatives["last_branch_accepts"] == [2, 7, 2, 2]
            packing = alternatives["cross_session_ast_packing"]
            assert packing["observed_used"] is True
            assert packing["configured_pack_rows"] == pack_rows
            assert packing["configured_mode"] is None
            assert packing["mode"] == "eager_all_branches"
            assert packing["last_pack_rows"] == pack_rows
            expected_launches = 5 if pack_rows == 4 else 3
            assert len(packing["last_packs"]) == expected_launches
            for physical_pack in packing["last_packs"]:
                assert set(physical_pack["session_slots"]) == {0, 1, 2, 3}
                assert physical_pack["logical_rows"] <= pack_rows
                assert (
                    physical_pack["max_physical_width"]
                    == physical_pack["logical_rows"]
                )
            scheduled = [
                (session, branch, continued)
                for launch in packing["last_packs"]
                for session, branch, continued in zip(
                    launch["session_slots"],
                    launch["branch_numbers"],
                    launch["continued"],
                    strict=True,
                )
            ]
            for session in range(4):
                assert [
                    continued
                    for slot, branch, continued in scheduled
                    if slot == session and branch == 3
                ] == [False]
                assert [
                    continued
                    for slot, branch, continued in scheduled
                    if slot == session and branch == 2
                ] == [False, True]
            first_continuation = next(
                index
                for index, launch in enumerate(packing["last_packs"])
                if any(launch["continued"])
            )
            assert all(
                branch != 3
                for launch in packing["last_packs"][first_continuation:]
                for branch in launch["branch_numbers"]
            )
        stats = packed_batch.scheduler_stats
        assert len({id(lane.cache) for lane in packed}) == 4
        assert len({id(lane.rng) for lane in packed}) == 4
        assert stats["external_progressive_cross_session_ast_rounds"] == 1
        assert stats["external_progressive_cross_session_ast_sessions"] == 4
        assert stats["external_progressive_cross_session_ast_packs"] == (
            5 if pack_rows == 4 else 3
        )
        assert stats["external_progressive_verify_max_cohort_width"] == pack_rows
        if pack_rows == 8:
            assert stats["external_progressive_cross_session_ast_b8_launches"] > 0
    finally:
        baseline_batch.close()
        packed_batch.close()


def test_cross_session_ast_pack_failure_restores_all_four_sessions(monkeypatch):
    batch, lanes = run_cross_session_ast(8, execute=False)
    before = [copy.deepcopy(lane.__dict__) for lane in lanes]
    before_stats = copy.deepcopy(batch.scheduler_stats)
    before_latch = copy.deepcopy(batch.progressive_topology_latch.snapshot())
    real_forward = batch.model.forward_with_taps

    def fail_packed_suffix(inputs, *args, **kwargs):
        if int(inputs.shape[0]) == 8:
            raise RuntimeError("injected cross-session AST pack failure")
        return real_forward(inputs, *args, **kwargs)

    monkeypatch.setattr(batch.model, "forward_with_taps", fail_packed_suffix)
    try:
        with pytest.raises(RuntimeError, match="cross-session AST pack failure"):
            batch._round(lanes)
        assert not batch._open
        for lane, expected in zip(lanes, before, strict=True):
            assert lane.history == expected["history"]
            assert lane.anchor == expected["anchor"]
            assert lane.generated == expected["generated"]
            assert lane.rng.snapshot() == expected["rng"].snapshot()
            assert not lane.ready
            close_caches(lane.cache, expected["cache"])
            close_caches(lane.draft_cache, expected["draft_cache"])
        ignored = {
            "external_cow_snapshots",
            "external_cow_fallbacks",
            "recovery_checkpoint_captures",
            "recovery_checkpoint_restores",
        }
        assert {
            key: value
            for key, value in batch.scheduler_stats.items()
            if key not in ignored
        } == {
            key: value
            for key, value in before_stats.items()
            if key not in ignored
        }
        assert batch.progressive_topology_latch.snapshot() == before_latch
    finally:
        batch.close()


@pytest.mark.parametrize("pack_rows", [4, 8])
def test_cross_session_apt_live_frontier_retires_and_refills_without_parity_drift(
    pack_rows,
):
    baseline_batch, baseline = run_cross_session_ast()
    packed_batch, packed = run_cross_session_ast(
        pack_rows,
        pack_mode="ordered_live_frontier",
    )
    try:
        for session_slot, (actual, expected) in enumerate(
            zip(packed, baseline, strict=True)
        ):
            assert actual.history == expected.history
            assert actual.anchor == expected.anchor
            assert actual.generated == expected.generated
            assert actual.rng.snapshot() == expected.rng.snapshot()
            close_caches(actual.cache, expected.cache)
            alternatives = actual.ready[-1].speculative_receipt[
                "progressive_verification"
            ]["ordered_alternatives"]
            packing = alternatives["cross_session_ast_packing"]
            assert alternatives["last_selected_branch"] == (
                1 if session_slot == 0 else 2
            )
            assert alternatives["last_branch_accepts"][2:] == [-1, -1]
            assert packing["configured_mode"] == "ordered_live_frontier"
            assert packing["mode"] == "ordered_live_frontier"
            frontier = packing["live_frontier"]
            assert frontier["finished"] is True
            assert frontier["sessions_retired"] == 4
            assert frontier["rows_admitted"] == 7
            assert frontier["branches_unvisited"] == 9
            assert frontier["refills"] == 3
            assert frontier["proposal_tokens_not_scheduled"] > 0
            assert all(
                launch["logical_rows"] <= min(4, pack_rows)
                for launch in packing["last_packs"]
            )
            assert any(
                any(launch["continued"])
                and any(
                    branch == 2 and not continued
                    for branch, continued in zip(
                        launch["branch_numbers"],
                        launch["continued"],
                        strict=True,
                    )
                )
                for launch in packing["last_packs"]
            )
        assert (
            packed_batch.scheduler_stats[
                "external_progressive_apt_frontier_rows_admitted"
            ]
            == 7
        )
        assert (
            packed_batch.scheduler_stats[
                "external_progressive_apt_frontier_sessions_retired"
            ]
            == 4
        )
    finally:
        baseline_batch.close()
        packed_batch.close()


def test_cross_session_apt_live_frontier_fault_restores_every_session(monkeypatch):
    batch, lanes = run_cross_session_ast(
        8,
        execute=False,
        pack_mode="ordered_live_frontier",
    )
    before = [copy.deepcopy(lane.__dict__) for lane in lanes]
    before_stats = copy.deepcopy(batch.scheduler_stats)
    before_latch = copy.deepcopy(batch.progressive_topology_latch.snapshot())
    real_forward = batch.model.forward_with_taps
    calls = 0

    def fail_first_frontier_refill(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected live-frontier refill failure")
        return real_forward(*args, **kwargs)

    monkeypatch.setattr(batch.model, "forward_with_taps", fail_first_frontier_refill)
    try:
        with pytest.raises(RuntimeError, match="live-frontier refill failure"):
            batch._round(lanes)
        assert not batch._open
        for lane, expected in zip(lanes, before, strict=True):
            assert lane.history == expected["history"]
            assert lane.anchor == expected["anchor"]
            assert lane.generated == expected["generated"]
            assert lane.rng.snapshot() == expected["rng"].snapshot()
            assert not lane.ready
            close_caches(lane.cache, expected["cache"])
            close_caches(lane.draft_cache, expected["draft_cache"])
        ignored = {
            "external_cow_snapshots",
            "external_cow_fallbacks",
            "recovery_checkpoint_captures",
            "recovery_checkpoint_restores",
        }
        assert {
            key: value
            for key, value in batch.scheduler_stats.items()
            if key not in ignored
        } == {
            key: value
            for key, value in before_stats.items()
            if key not in ignored
        }
        assert batch.progressive_topology_latch.snapshot() == before_latch
    finally:
        batch.close()


def test_cross_session_apt_live_frontier_no_full_preserves_b1_fallback():
    eager_batch, eager = run_cross_session_ast(
        8,
        force_full_alternative=False,
    )
    live_batch, live = run_cross_session_ast(
        8,
        pack_mode="ordered_live_frontier",
        force_full_alternative=False,
    )
    try:
        for actual, expected in zip(live, eager, strict=True):
            assert actual.history == expected.history
            assert actual.anchor == expected.anchor
            assert actual.generated == expected.generated
            assert actual.rng.snapshot() == expected.rng.snapshot()
            close_caches(actual.cache, expected.cache)
            alternatives = actual.ready[-1].speculative_receipt[
                "progressive_verification"
            ]["ordered_alternatives"]
            assert alternatives["last_selected_branch"] == 1
            assert -1 not in alternatives["last_branch_accepts"]
            frontier = alternatives["cross_session_apt_packing"]["live_frontier"]
            assert frontier["branches_unvisited"] == 0
            assert frontier["proposal_tokens_not_scheduled"] == 0
    finally:
        eager_batch.close()
        live_batch.close()


def test_cross_session_ast_postcommit_observation_failure_is_diagnostic(
    monkeypatch,
):
    batch, lanes = run_cross_session_ast(4, execute=False)
    before_latch = copy.deepcopy(batch.progressive_topology_latch.snapshot())

    def fail_observation(*_args, **_kwargs):
        raise RuntimeError("injected postcommit AST observation failure")

    monkeypatch.setattr(
        batch.progressive_topology_latch, "observe_many", fail_observation
    )
    try:
        batch._round(lanes)
        assert all(lane.ready for lane in lanes)
        assert batch.progressive_topology_latch.snapshot() == before_latch
        # Each session owns its latch, so each session's observation fails
        # (and is counted) on its own.
        assert (
            batch.scheduler_stats[
                "external_progressive_topology_observation_failures"
            ]
            == len(lanes)
        )
        assert "injected postcommit" in batch.last_topology_observation_error
    finally:
        batch.close()


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


@pytest.mark.parametrize("pack_mode", [None, "ordered_live_frontier"])
def test_cross_session_ast_latch_is_per_session_with_mixed_widths(pack_mode):
    """One session's latch never narrows (or widens) its packed peers."""
    baseline_batch, baseline = run_cross_session_ast()
    batch, lanes = run_cross_session_ast(4, execute=False, pack_mode=pack_mode)
    latches = batch.progressive_topology_latch
    # Session 1 has parked itself on its primary chain; session 2 runs the
    # economical two-way check; sessions 0 and 3 keep all four branches.
    parked = latches.session(lanes[1].uid)
    parked.active_branches = 1
    parked.park_remaining = 2
    latches.session(lanes[2].uid).active_branches = 2
    try:
        batch._round(lanes)
        widths = []
        for actual, expected in zip(lanes, baseline, strict=True):
            # Every session commits an exact prefix of its B1 continuation.
            assert actual.history == expected.history[: len(actual.history)]
            receipt = actual.ready[-1].speculative_receipt[
                "progressive_verification"
            ]["ordered_alternatives"]
            widths.append(len(receipt["last_branch_accepts"]))
            assert receipt["adaptive_latch"]["scope"] == "request_session"
        assert widths == [4, 1, 2, 4]
        primary_only = lanes[1].progressive_topology_latch_last_observation
        assert primary_only["outcome"] == "committed_primary_only"
        # The width-four sessions learned from their own rows only.
        snapshot = latches.snapshot()
        assert snapshot["active_branches_by_session"][str(lanes[1].uid)] == 1
        assert latches.session(lanes[0].uid).active_branches == 4
        assert latches.session(lanes[3].uid).active_branches == 4
        assert latches.session(lanes[1].uid).primary_only_samples == 1
        assert latches.session(lanes[0].uid).primary_only_samples == 0
        assert (
            batch.scheduler_stats["external_progressive_ast_mixed_width_rounds"] == 1
        )
    finally:
        batch.close()
        baseline_batch.close()


def test_per_session_latch_released_with_its_request():
    batch, lanes = run_cross_session_ast(4)
    try:
        latches = batch.progressive_topology_latch
        assert latches.snapshot()["live_sessions"] == 4
        batch.remove([lanes[0].uid])
        assert latches.snapshot()["live_sessions"] == 3
        assert latches.released_sessions == 1
    finally:
        batch.close()


def _no_win(latch, decision):
    latch.observe_many(
        [
            {
                "branch_accepts": [3] + [1] * (decision.active_branches - 1),
                "proposal_counts": [7] + [3] * (decision.active_branches - 1),
                "selected_branch": 1,
                "unique_target_rows": 4 + 3 * (decision.active_branches - 1),
                "outcome": "committed_conclusive",
            }
        ],
        decision_token=decision.decision_token,
    )


def test_ast_park_expiry_reengages_two_way_check_with_backoff():
    latch = AdaptiveTopologyLatch(
        4, shrink_after=1, park_rounds=2, max_park_rounds=4, reprobe_interval=1000
    )
    parks = []
    for _ in range(40):
        decision = latch.choose(2, 7, 4)
        if decision.active_branches == 1:
            latch.observe_primary_only(
                decision_token=decision.decision_token, session_count=1
            )
            if decision.reason == "parked_after_repeated_no_alternative_win":
                continue
        else:
            if decision.reason == "park_expired_economical_check":
                parks.append(latch.park_length)
            _no_win(latch, decision)
    snapshot = latch.snapshot()
    # Every expired park re-engaged a two-way check instead of staying at one.
    assert snapshot["park_expiries"] >= 3
    # Each fruitless check doubled the next park, up to the cap.
    assert parks[:3] == [4, 4, 4]
    assert latch.park_length == 4


def test_ast_alternative_win_resets_park_backoff():
    latch = AdaptiveTopologyLatch(
        4, shrink_after=1, park_rounds=2, max_park_rounds=8, reprobe_interval=1000
    )
    for _ in range(3):
        _no_win(latch, latch.choose(2, 7, 4))
    assert latch.active_branches == 1 and latch.park_length == 4
    while latch.active_branches == 1:
        decision = latch.choose(2, 7, 4)
        latch.observe_primary_only(
            decision_token=decision.decision_token, session_count=1
        )
    decision = latch.choose(2, 7, 4)
    assert decision.active_branches == 2
    assert decision.reason == "park_expired_economical_check"
    latch.observe_many(
        [
            {
                "branch_accepts": [1, 7],
                "proposal_counts": [7, 7],
                "selected_branch": 2,
                "unique_target_rows": 10,
                "outcome": "committed_conclusive",
            }
        ],
        decision_token=decision.decision_token,
    )
    assert latch.active_branches == 2
    assert latch.park_length == 2


def test_cross_session_live_frontier_keeps_b1_when_a_branch_ends_at_its_prefix():
    # 2026-10-17: a min-tokens request made LiLiCorr end a branch at the
    # shared prefix; the live-frontier planner refused the zero-depth row and
    # the generation worker died.  The round must keep the exact primary path.
    baseline_batch, baseline = run_cross_session_ast()
    packed_batch, packed = run_cross_session_ast(
        8, pack_mode="ordered_live_frontier", truncate_branch=True
    )
    try:
        # The primary-only round accepts fewer of its (mutated) drafts than
        # the unpacked baseline, but every committed token is the exact B1
        # greedy continuation.
        for actual, expected in zip(packed, baseline, strict=True):
            assert actual.generated >= 1
            assert actual.history == expected.history[: len(actual.history)]
        assert packed_batch.scheduler_stats[
            "external_progressive_ast_geometry_guard_rounds"
        ] == 4
    finally:
        baseline_batch.close()
        packed_batch.close()
