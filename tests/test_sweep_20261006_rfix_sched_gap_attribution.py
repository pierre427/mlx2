"""Attribute the largest decode gap beside prefill (2026-10-06 rfix-sched).

GPU (Qwen3.8-27B, 12K prompt beside one decoding lane): with the cost-model
bound the neighbour p95 met the 500 ms target but the maximum did not
(ordinary 737 ms, native MTP 1260 ms), and no host counter said whether a
mis-predicted slice or something outside the measured forwards (lane
release, cohort change, the serving loop) made it.  Three gauges and a
counter now split each decode-to-decode interval that carried contended
prefill.
"""

import pytest

from mlx2.runtime import adaptive_policy
from mlx2.runtime.adaptive_policy import DecodeTimeFairness


class Clock:
    def __init__(self):
        self.now = 50.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(adaptive_policy.time, "monotonic", clock)
    return clock


def _decode(policy, clock, seconds=0.03):
    clock.now += seconds
    policy.observe_decode(seconds)


def test_gap_splits_into_forward_and_other_time(clock):
    policy = DecodeTimeFairness(enabled=True, stall_target_ms=500.0)
    _decode(policy, clock)
    _decode(policy, clock)
    clock.now += 0.30
    policy.observe_prefill(256, 0.30, contended=True)
    clock.now += 0.20  # unmeasured: lane release, serving loop
    _decode(policy, clock)
    c = policy.counters
    assert c["contended_forward_max_us"] == pytest.approx(300000, abs=2)
    assert c["contended_gap_max_us"] == pytest.approx(530000, abs=2)
    assert c["contended_gap_other_max_us"] == pytest.approx(200000, abs=2)
    assert c["contended_stalls_over_target"] == 1


def test_gaps_without_prefill_and_fresh_lanes_are_not_attributed(clock):
    policy = DecodeTimeFairness(enabled=True, stall_target_ms=500.0)
    _decode(policy, clock)
    clock.now += 5.0  # idle, then a lane's own (uncontended) prefill
    policy.observe_prefill(256, 0.30, contended=False)
    policy.observe_prefill(256, 0.30, contended=True)
    _decode(policy, clock)  # first decode of the fresh gap: no baseline
    _decode(policy, clock)  # a plain decode-to-decode gap: no prefill
    assert "contended_gap_max_us" not in policy.counters
    assert "contended_stalls_over_target" not in policy.counters


def test_disabled_policy_records_nothing(clock):
    policy = DecodeTimeFairness(enabled=False)
    _decode(policy, clock)
    policy.observe_prefill(256, 0.9, contended=True)
    _decode(policy, clock)
    assert not any(key.startswith("contended_") for key in policy.counters)
