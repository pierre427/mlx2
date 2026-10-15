"""Analytical acceptance oracles and actual cohort-controller integration."""

import pytest

from mlx2.runtime.adaptive_policy import AdaptiveMTPDepthPolicy, CohortAdaptiveMTPDepth
from mlx2.runtime.factorized_depth import FactorizedDepthCost


def train(model, *, boundary=1, width=10):
    # P(first)=.8, P(second|first)=.25, P(third|second)=.5.
    # The six rows rejected at position two do not reject position three.
    accepted = (3, 2, 1, 1, 1, 1, 1, 1, 0, 0)
    model.observe(width=width, depth=3, depths=(3,) * width,
                  accepted=accepted, seconds=.015, boundary=boundary)
    model.observe(width=width, depth=0, depths=(0,) * width,
                  accepted=(0,) * width, seconds=.010, boundary=boundary)


def model():
    return FactorizedDepthCost(3, alpha=1, min_samples=1, stale_rounds=16)


def test_conditional_oracle_predicts_unsampled_depth_and_censors_rejected_tail():
    m = model()
    train(m)
    estimates = m.predictions(width=10, cap=3, boundary=2)
    assert estimates[0] == pytest.approx(10 / .01)
    assert estimates[1] == pytest.approx(18 / (.01 + .005 / 3))
    assert estimates[2] == pytest.approx(20 / (.01 + .010 / 3))
    assert estimates[3] == pytest.approx(21 / .015)
    assert max(estimates, key=estimates.get) == 1
    # Exact width identity: widths 9 and 10 are in the same controller bucket.
    assert m.predictions(width=9, cap=3, boundary=2) == {}


def test_no_extrapolation_or_stale_predictions_and_bounded_width_owners():
    m = model()
    m.observe(width=1, depth=3, depths=(3,), accepted=(3,), seconds=.02, boundary=1)
    assert set(m.predictions(width=1, cap=3, boundary=2)) == {3}
    assert m.predictions(width=1, cap=3, boundary=17) == {}
    for width in range(1, 40):
        m.observe(width=width, depth=0, depths=(0,) * width,
                  accepted=(0,) * width, seconds=.01, boundary=1)
    assert len(m.widths) == 32


@pytest.mark.parametrize("bad", [(4,), (-1,), (True,), ()])
def test_malformed_landing_record_does_not_partially_train(bad):
    m = model()
    with pytest.raises(ValueError):
        m.observe(width=1, depth=3, depths=(3,), accepted=bad, seconds=.01, boundary=1)
    assert not m.widths and not m.samples


def test_actual_controller_chooses_predicted_depth_without_probing_every_depth():
    p = CohortAdaptiveMTPDepth(3, cost_model="factorized", min_samples_per_depth=1,
                              probe_interval=100, goodput_alpha=1)
    assert p.select(width=10) == 3
    p.observe(30, 11, width=10, committed=21, elapsed_seconds=.015,
              landing_depths=(3,) * 10, landing_accepted=(3, 2, 1, 1, 1, 1, 1, 1, 0, 0))
    # Ordinary probe seeds the other endpoint; intermediate depths have never
    # executed, yet their expected goodput can now be compared.
    p.current_depth = 0
    assert p.select(width=10) == 0
    p.observe(0, 0, width=10, committed=10, elapsed_seconds=.010,
              landing_depths=(0,) * 10, landing_accepted=(0,) * 10)
    assert p.select(width=10) == 1
    assert p.diagnostics()["buckets"]["9-16"]["samples"] == {"0": 1, "3": 1}
    assert p.diagnostics()["factorized"]["samples"] == 2


def test_greedy_and_single_lane_anchors_survive_factorized_predictions():
    for kwargs in ({"reproducible_greedy": True}, {}):
        p = CohortAdaptiveMTPDepth(3, cost_model="factorized", min_samples_per_depth=1,
                                  goodput_alpha=1, **kwargs)
        for depth, seconds in ((3, .1), (0, .01)):
            p._factorized.observe(width=1, depth=depth, depths=(depth,), accepted=(depth,),
                                  seconds=seconds, boundary=1)
        assert p.select(width=1) == 3


def test_missing_or_padded_evidence_cannot_train_factorized_model():
    p = CohortAdaptiveMTPDepth(3, cost_model="factorized", min_samples_per_depth=1)
    p.select(width=2)
    p.observe(6, 2, width=2, committed=4, elapsed_seconds=.1)
    assert p._factorized.samples == 0
    with pytest.raises(ValueError, match="unpadded"):
        p.observe(6, 2, width=2, observed_compute_width=4, committed=4, elapsed_seconds=.1,
                  landing_depths=(3, 3), landing_accepted=(1, 1))
    assert p._factorized.samples == 0


def test_opt_in_policy_roundtrip_and_default_receipt_unchanged():
    default = AdaptiveMTPDepthPolicy(enabled=True)
    assert "cost_model" not in default.as_dict()
    p = AdaptiveMTPDepthPolicy.from_value({"enabled": True, "cost_model": "factorized"})
    assert p.as_dict()["cost_model"] == "factorized"
    assert p.controller_kwargs()["cost_model"] == "factorized"
    with pytest.raises(ValueError, match="cost_model"):
        AdaptiveMTPDepthPolicy.from_value({"cost_model": "unknown"})
