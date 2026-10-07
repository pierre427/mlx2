"""Cost-model samples expire when a bound is asked for, not only on the next
observation (Codex review of the 2026-10-06 scheduling sweep, P2).

After an idle period the first contended slice used to be sized from
measurements older than ``max_age_s``: expiry ran only inside ``observe``,
which happens after that slice.  Decode fixed-cost samples had no age at all.
"""

import pytest

from mlx2.runtime import adaptive_policy
from mlx2.runtime.adaptive_policy import DecodeTimeFairness, PrefillCostModel


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(adaptive_policy.time, "monotonic", clock)
    return clock


def test_stale_prefill_samples_do_not_size_the_next_slice(clock):
    policy = DecodeTimeFairness(enabled=True, stall_target_ms=500.0)
    for _ in range(8):
        policy.observe_prefill(1024, 0.4, contended=True)
    calibrated = policy.cap(8192, contended=True)
    assert calibrated > policy.fallback_cap
    clock.now += policy.cost.max_age_s + 1
    # Every sample is older than max_age_s: no measurement remains, so the
    # bound falls back exactly as on a fresh generator.
    fresh = DecodeTimeFairness(enabled=True, stall_target_ms=500.0)
    assert policy.cap(8192, contended=True) == fresh.cap(8192, contended=True)
    assert policy.cost.estimate("ordinary", 0) is None


def test_stale_decode_samples_do_not_set_the_fixed_cost(clock):
    model = PrefillCostModel()
    for _ in range(4):
        model.observe_decode(0.2)
    model.observe(1000, 1.0, depth=0, kind="ordinary")
    assert model.fixed_cost() == pytest.approx(0.2)
    clock.now += model.max_age_s + 1
    model.observe(1000, 1.0, depth=0, kind="ordinary")
    assert model.fixed_cost() == 0.0


def test_fresh_samples_survive_the_estimate(clock):
    model = PrefillCostModel()
    model.observe(1000, 1.0, depth=0, kind="ordinary")
    clock.now += model.max_age_s - 1
    assert model.estimate("ordinary", 0) is not None


def test_first_contended_slice_without_samples_is_a_probe(clock):
    """With no measurement at all (fresh, or every sample expired) the bound
    used to take ``fallback_cap`` (512) rows whatever the model's speed: on
    Qwen3.8-27B about 650 ms beside a decoding lane.  It now probes one
    grid tile and grows from the measurement, at most 2x per slice."""
    policy = DecodeTimeFairness(enabled=True, stall_target_ms=500.0)
    assert policy.cap(8192, contended=True) == policy.grid
    assert policy.cap(8192, contended=True, rows=4) == policy.grid
    widths = []
    for _ in range(6):
        policy.observe_decode(0.03)
        width = policy.cap(8192, contended=True)
        policy.observe_prefill(width, 0.03 + 0.0013 * width, contended=True)
        policy.debt_seconds = 0.0
        widths.append(width)
    assert widths[:4] == [64, 128, 256, 320]
    # running_max keeps its fallback (A/B reference).
    legacy = DecodeTimeFairness(enabled=True, estimator="running_max")
    assert legacy.cap(8192, contended=True) == legacy.fallback_cap
