import numpy as np
import pytest

from mlx2.runtime.acceptance_estimator import (
    AdaptiveVerificationPolicy,
    OnlineAcceptanceEstimator,
    proposal_feature,
    proposal_law_features,
)


def test_feature_scores_law_not_sampled_token_and_clamps_one_hot():
    assert proposal_feature([0.9, 0.1]) == pytest.approx(np.log(9))
    assert proposal_feature([0.1, 0.9]) == proposal_feature([0.9, 0.1])
    assert proposal_feature([0, 1]) == 40
    for bad in ([np.nan, 1], [np.inf, 0], [-0.1, 1.1], [0.1, 0.2], []):
        with pytest.raises(ValueError):
            proposal_feature(bad)


def test_collision_feature_is_distribution_only_and_post_sample_depth_is_biased():
    """A depth gate may inspect q, but not the q-drawn token it is gating."""
    proposal = np.array([0.9, 0.1])
    target = np.array([0.5, 0.5])
    features = proposal_law_features(proposal)
    assert features.collision_probability == pytest.approx(0.82)
    assert proposal_law_features(proposal[::-1]) == features

    def emitted_law(verify):
        result = np.zeros(2)
        residual = np.maximum(target - proposal, 0)
        residual /= residual.sum()
        for sampled, q_sampled in enumerate(proposal):
            if not verify(sampled):
                result += q_sampled * target
                continue
            accepted = min(1.0, target[sampled] / q_sampled)
            result[sampled] += q_sampled * accepted
            result += q_sampled * (1 - accepted) * residual
        return result

    # The law-only decision is fixed for both possible proposal draws. Exact
    # speculative verification therefore preserves the target distribution.
    pre_sample = emitted_law(
        lambda _sampled: features.collision_probability >= 0.3
    )
    np.testing.assert_allclose(pre_sample, target, atol=1e-12)
    # Conditioning the same decision on q(sampled) verifies the likely draw
    # and replaces the unlikely draw with an ordinary target sample: 55/45.
    post_sample = emitted_law(lambda sampled: proposal[sampled] >= 0.3)
    np.testing.assert_allclose(post_sample, [0.55, 0.45], atol=1e-12)
    assert not np.allclose(post_sample, target)


def test_censored_positions_never_receive_rejection_labels():
    fit = OnlineAcceptanceEstimator(4)
    fit.observe([0, 1, 2, 3], 1, rejected=True)
    np.testing.assert_array_equal(fit.observed_counts, [1, 1, 0, 0])
    np.testing.assert_array_equal(fit.feature_counts, [1, 1, 1, 1])
    assert fit.grad[0, 1] > 0 and fit.grad[1, 1] < 0
    np.testing.assert_array_equal(fit.info[2:], 0)
    fit.observe([1, 2, 3, 4], 2, rejected=False)
    np.testing.assert_array_equal(fit.observed_counts, [2, 2, 0, 0])
    with pytest.raises(ValueError):
        fit.observe([0], 1, rejected=True)


def test_damped_arrowhead_refit_matches_direct_normal_equations():
    fit = OnlineAcceptanceEstimator(3, damping=10)
    for values, accepted in [([1, 2, 3], 2), ([0, 0.4, 1], 1), ([2, 3, 4], 3)]:
        fit.observe(values, accepted, rejected=accepted < 3)
    a = fit.info[:, 0].sum() + fit.l2
    b = fit.info[:, 1]
    c = fit.info[:, 2] + fit.l2
    matrix = np.diag([a, *c])
    matrix[0, 1:] = b
    matrix[1:, 0] = b
    delta = np.linalg.solve(matrix, [fit.grad[:, 0].sum(), *fit.grad[:, 1]])
    step = delta[0] * fit.counts.sum() / (fit.counts.sum() + fit.damping)
    expected_bias = (
        (fit.grad[:, 1] - b * step) / c * fit.counts / (fit.counts + fit.damping)
    )
    scale = min(1.0, 2.0 / np.max(40.0 * abs(step) + np.abs(expected_bias)))
    fit.refit()
    assert fit.slope == pytest.approx(0.5 + step * scale)
    np.testing.assert_allclose(fit.intercepts, -0.2 + expected_bias * scale)
    assert fit.refits == 1 and not fit.counts.any() and not fit.info.any()


def _trained():
    fit = OnlineAcceptanceEstimator(3)
    for _ in range(10):
        fit.observe([0, 0, 0], 0, rejected=True)
    fit.intercepts.fill(-4)
    fit.rounds = 1
    return fit


def test_budget_requires_cost_gain_observations_and_matching_cohort():
    fit = _trained()
    policy = AdaptiveVerificationPolicy.from_value(
        {
            "verification_costs": [1, 2, 4, 8],
            "min_observations": 1,
        },
        3,
    )
    assert policy.choose_depth(fit, 3, 1) == 1
    assert policy.choose_depth(fit, 3, 2) == 3
    fit.rounds = 32
    assert policy.choose_depth(fit, 3, 1) == 3
    fit.rounds = 1
    flat = AdaptiveVerificationPolicy.from_value(
        {
            "verification_costs": [1, 1, 1, 1],
            "min_observations": 1,
        },
        3,
    )
    assert flat.choose_depth(fit, 3, 1) == 3
    fit.intercepts[1] = np.nan
    assert policy.choose_depth(fit, 3, 1) == 3
    assert policy.choose_depth(OnlineAcceptanceEstimator(3), 3, 1) == 3


@pytest.mark.parametrize(
    "value",
    [
        True,
        {},
        {"verification_costs": [1, 2]},
        {"verification_costs": [1, 2, float("nan"), 4]},
        {"verification_costs": [1, 2, 3, 4], "full_depth_interval": 0},
        {"verification_costs": [1, 2, 3, 4], "cohort_size": True},
        {"verification_costs": [1, 2, 3, 4], "unknown": 1},
    ],
)
def test_invalid_policy_fails_closed(value):
    with pytest.raises(ValueError):
        AdaptiveVerificationPolicy.from_value(value, 3)


def test_sampled_target_law_survives_online_lagged_budget_changes():
    from mlx2.runtime.speculative_sampling import RequestRNG, verify_proposals

    rng = RequestRNG(88)
    target = np.array([0.7, 0.2, 0.1])
    proposal = np.array([0.1, 0.2, 0.7])
    fit = OnlineAcceptanceEstimator(3, refit_interval=10)
    policy = AdaptiveVerificationPolicy.from_value(
        {
            "verification_costs": [1, 2, 4, 8],
            "min_observations": 1,
        },
        3,
    )
    feature = proposal_feature(proposal)
    counts = np.zeros(3)
    depths = set()
    for _ in range(12000):
        # Crucially, this decision precedes every current proposal draw.
        depth = policy.choose_depth(fit, 3, 1)
        depths.add(depth)
        tokens = [rng.sample(proposal) for _ in range(depth)]
        result = verify_proposals(
            tokens, [proposal] * depth, [target] * (depth + 1), rng
        )
        counts[result.emitted[0]] += 1
        fit.observe([feature] * depth, result.accepted, rejected=result.rejected)
        fit.finish_round()
    assert depths == {1, 3}
    np.testing.assert_allclose(counts / counts.sum(), target, atol=0.015)


@pytest.mark.parametrize(
    "key,value",
    [
        ("draft_cost", "oops"),
        ("draft_cost", True),
        ("min_gain", "0.1"),
        ("min_gain", False),
        ("verification_costs", [1, True, 2, 3]),
        ("mode", "unknown"),
        ("verification_costs_by_cohort", {"2": [1, 2]}),
    ],
)
def test_malformed_cost_fields_fail_as_value_errors(key, value):
    config = {"verification_costs": [1, 2, 3, 4], key: value}
    with pytest.raises(ValueError):
        AdaptiveVerificationPolicy.from_value(config, 3)


def test_one_hot_feature_fit_stays_stable_and_calibrates_mixed_labels():
    fit = OnlineAcceptanceEstimator(1)
    for _ in range(30):
        before = float(fit.slope * 40 + fit.intercepts[0])
        for index in range(100):
            accepted = int(index < 10)
            fit.observe([40], accepted, rejected=not accepted)
        fit.refit()
        after = float(fit.slope * 40 + fit.intercepts[0])
        assert abs(after - before) <= 2.000001
        assert (
            np.isfinite(fit.slope)
            and fit.slope >= 0
            and np.isfinite(fit.intercepts).all()
        )
    assert fit.predict([40])[0] == pytest.approx(0.1, abs=0.015)


def test_request_depths_need_matching_cost_tables_and_predict_real_group_gain():
    fit = OnlineAcceptanceEstimator(2)
    fit.observe([0, 0], 0, rejected=True)
    fit.rounds = 1
    base = {
        "verification_costs": [0.5, 1, 1.4],
        "mode": "per_request",
        "min_observations": 1,
    }
    singleton = AdaptiveVerificationPolicy.from_value(base, 2)
    assert singleton.request_depths(fit, [2, 2], [[40, 40], [-40, -40]]) == [2, 2]
    grouped = AdaptiveVerificationPolicy.from_value(
        {**base, "verification_costs_by_cohort": {"2": [1, 2, 10]}}, 2
    )
    assert grouped.request_depths(fit, [2, 2], [[40, 40], [-40, -40]]) == [2, 1]
    # Cheap existing batching defeats the extra physical forwards.
    cheap = AdaptiveVerificationPolicy.from_value(
        {**base, "verification_costs_by_cohort": {"2": [0.5, 1, 1.1]}}, 2
    )
    assert cheap.request_depths(fit, [2, 2], [[40, 40], [-40, -40]]) == [2, 2]


@pytest.mark.parametrize(
    "costs",
    [
        {"0": [1, 2, 3]},
        {"16": [1, 2, 3]},
        {"1": [1, True, 3]},
        {"1": [1, float("nan"), 3]},
        {"1": [1, 2]},
        {"1": [1, 0, 3]},
    ],
)
def test_complete_path_costs_reject_unbound_or_invalid_physical_shapes(costs):
    with pytest.raises(ValueError):
        AdaptiveVerificationPolicy.from_value({"continuation_costs": costs}, 2)


def test_complete_path_costs_do_not_fabricate_chain_cohort_costs():
    policy = AdaptiveVerificationPolicy.from_value(
        {"continuation_costs": {"15": [1, 2, 3]}}, 2
    )
    assert policy.costs(1) is None and policy.costs(15) is None
    assert dict(policy.continuation_costs) == {15: (1.0, 2.0, 3.0)}


def _separation_regime(estimator, rounds, *, accepted_all, rng):
    depth = estimator.depth
    for _ in range(rounds):
        features = rng.normal(2.0, 0.5, depth)
        if accepted_all:
            estimator.observe(features, depth, rejected=False)
        else:
            accepted = 0
            while accepted < depth and rng.random() < 0.5:
                accepted += 1
            estimator.observe(features, accepted, rejected=accepted < depth)
        estimator.finish_round()


def _rounds_to_recover(estimator, rng, *, limit=40000):
    depth = estimator.depth
    for round_ in range(limit):
        _separation_regime(estimator, 1, accepted_all=False, rng=rng)
        if abs(float(estimator.predict(np.full(depth, 2.0))[0]) - 0.5) < 0.1:
            return round_
    return limit


def test_separation_reset_recovers_from_a_saturated_regime_much_faster():
    """vllm-project/vllm#66673: under separation the raw Fisher information
    goes to zero and the coefficient creeps one way; a breaker resets it."""
    rngs = [np.random.default_rng(seed) for seed in (0, 0)]
    plain = OnlineAcceptanceEstimator(6)
    breaker = OnlineAcceptanceEstimator(6, separation_reset=True)
    _separation_regime(plain, 20000, accepted_all=True, rng=rngs[0])
    _separation_regime(breaker, 20000, accepted_all=True, rng=rngs[1])
    assert breaker.separation_resets.sum() > 0 and plain.separation_resets.sum() == 0
    slow, fast = _rounds_to_recover(plain, rngs[0]), _rounds_to_recover(breaker, rngs[1])
    assert fast * 5 < slow, (fast, slow)


def test_separation_reset_never_fires_on_informative_or_unobserved_positions():
    rng = np.random.default_rng(3)
    estimator = OnlineAcceptanceEstimator(4, separation_reset=True)
    _separation_regime(estimator, 5000, accepted_all=False, rng=rng)
    assert estimator.separation_resets.sum() == 0 and estimator.slope_resets == 0
    # Positions beyond the observed prefix stay unlabelled: no counter, no reset.
    estimator = OnlineAcceptanceEstimator(4, separation_reset=True)
    for _ in range(2000):
        estimator.observe(rng.normal(2.0, 0.5, 1), 0, rejected=True)
        estimator.finish_round()
    assert estimator.separation_resets[1:].sum() == 0 and estimator.stuck[1:].sum() == 0


def test_separation_reset_counter_self_clears_before_the_limit():
    estimator = OnlineAcceptanceEstimator(2, refit_interval=40, separation_reset=True)

    def window(intercept, accepted, rejected):
        for _ in range(40):
            estimator.intercepts[:] = intercept
            estimator.observe(np.array([1.0, 1.0]), accepted, rejected=rejected)
            estimator.finish_round()

    window(30.0, 2, False)  # saturated: p(1-p) ~ 0 on every observation
    window(30.0, 2, False)
    assert estimator.stuck[0] == 2
    window(0.0, 1, True)  # informative again before the limit
    assert estimator.stuck[0] == 0 and estimator.separation_resets.sum() == 0


def test_separation_reset_leaves_accurate_rarely_reached_positions_alone():
    """Review finding: a window total of p(1-p) is small for a position that
    is reached a few times per window even when its estimate is right
    (three p = 0.97 observations sum to 0.09). Judge per observation."""
    rng = np.random.default_rng(11)
    truth = np.array([0.05, 0.97, 0.97, 0.97, 0.97, 0.97])
    estimators = [OnlineAcceptanceEstimator(6), OnlineAcceptanceEstimator(6, separation_reset=True)]
    for _ in range(50000):
        features = rng.normal(2.0, 0.5, 6)
        accepted = 0
        while accepted < 6 and rng.random() < truth[accepted]:
            accepted += 1
        for estimator in estimators:
            estimator.observe(features, accepted, rejected=accepted < 6)
            estimator.finish_round()
    plain, breaker = estimators
    assert breaker.separation_resets.sum() == 0 and breaker.slope_resets == 0
    probe = np.full(6, 2.0)
    assert np.allclose(breaker.predict(probe), plain.predict(probe))


def test_separation_reset_is_an_opt_in_policy_boolean():
    costs = [1.0, 1.5, 2.0]
    assert AdaptiveVerificationPolicy.from_value({"verification_costs": costs}, 2).separation_reset is False
    assert AdaptiveVerificationPolicy.from_value(
        {"verification_costs": costs, "separation_reset": True}, 2).separation_reset is True
    with pytest.raises(ValueError):
        AdaptiveVerificationPolicy.from_value({"verification_costs": costs, "separation_reset": 1}, 2)
