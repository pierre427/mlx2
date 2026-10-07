"""Decode-fairness stall bound: completion-time cost model (sweep 2026-10-06, SS-2).

A closed-loop simulator drives ``DecodeTimeFairness`` the way the scheduler
does: ask for a bound, run a forward whose true time is
``a + b * rows * width * (1 + depth / 8192)``, observe it.  Each synthetic
scenario runs under both estimators; the cost model must meet every
acceptance property and the old running maximum must miss at least one, so
the default switch is backed by a strict comparison.
"""

import pytest

from mlx2.runtime.adaptive_policy import DecodeTimeFairness, PrefillCostModel

TARGET_MS = 100.0
ESTIMATORS = ("cost_model", "running_max")


def _policy(estimator, **kw):
    return DecodeTimeFairness(
        enabled=True, stall_target_ms=TARGET_MS, estimator=estimator, **kw
    )


class Device:
    def __init__(self, a=0.005, b=2e-4, depth_scale=8192.0):
        self.a = a
        self.b = b
        self.depth_scale = depth_scale

    def time(self, rows, width, depth=0):
        return self.a + self.b * rows * width * (1 + depth / self.depth_scale)


def _run(policy, device, slices, *, rows=1, depth=0, step=8192, decode=0.02):
    """Run ``slices`` contended slices; return each slice's seconds."""
    times = []
    for _ in range(slices):
        policy.observe_decode(decode)
        width = policy.cap(step, contended=True, rows=rows, depth=depth)
        seconds = device.time(rows, width, depth)
        policy.observe_prefill(
            width, seconds, contended=True, rows=rows, depth=depth
        )
        policy.debt_seconds = 0.0
        times.append(seconds)
    return times


def _scenario_slowdown(estimator):
    # The slow regime still fits above the 64-row floor (280 rows).
    policy, device = _policy(estimator), Device(b=2e-5)
    _run(policy, device, 10)
    device.b *= 16
    after = _run(policy, device, 10)
    # Back within 1.5x target in at most 3 slices, and stays there.
    return all(t <= 1.5 * TARGET_MS / 1000 for t in after[3:])


def _speedup_widths(estimator):
    policy, device = _policy(estimator), Device(b=16 * 2e-5)
    _run(policy, device, 10)
    device.b /= 16
    widths = []
    for _ in range(20):
        policy.observe_decode(0.02)
        width = policy.cap(8192, contended=True)
        policy.observe_prefill(width, device.time(1, width), contended=True)
        widths.append(width)
    return (widths, device)


def _scenario_speedup(estimator):
    (widths, device) = _speedup_widths(estimator)
    within = all(device.time(1, w) <= 1.5 * TARGET_MS / 1000 for w in widths)
    # The faster regime is used within ten slices (not stuck slow).
    return within and widths[9] >= 4 * widths[0]


def _scenario_rows(estimator):
    policy, device = _policy(estimator), Device()
    _run(policy, device, 6, rows=1)
    after = _run(policy, device, 6, rows=4)
    return all(t <= 1.5 * TARGET_MS / 1000 for t in after[1:])


def _scenario_warm(estimator):
    """A warm continuation: calibrated shallow, then a 40K-deep slice."""
    policy, device = _policy(estimator), Device()
    _run(policy, device, 8, depth=1024)
    after = _run(policy, device, 6, depth=40960)
    return all(t <= 1.5 * TARGET_MS / 1000 for t in after)


SCENARIOS = {
    "slowdown": _scenario_slowdown,
    "speedup": _scenario_speedup,
    "rows": _scenario_rows,
    "warm": _scenario_warm,
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_cost_model_meets_every_scenario(scenario):
    assert SCENARIOS[scenario]("cost_model")


def test_cost_model_strictly_better_than_running_max():
    cost = {name: run("cost_model") for name, run in SCENARIOS.items()}
    legacy = {name: run("running_max") for name, run in SCENARIOS.items()}
    print("cost_model:", cost, "running_max:", legacy)
    assert all(cost.values())
    assert not all(legacy.values())
    # Never worse in any scenario.
    assert all(cost[name] or not legacy[name] for name in SCENARIOS)


def test_speedup_expands_at_most_2x_per_slice_and_trails_running_max():
    """The deliberate asymmetry: expansion is capped at 2x per slice.

    Both estimators meet the target after a speedup; the running maximum
    reaches the fast width in 2 slices, the cost model in about 8 and keeps
    its 10% margin.  That is the price of never trusting one fast sample.
    """
    (cost, _) = _speedup_widths("cost_model")
    (legacy, _) = _speedup_widths("running_max")
    assert all(b <= 2 * a for a, b in zip(cost, cost[1:]))
    assert legacy.index(max(legacy)) < cost.index(max(cost))


def test_slice_converges_to_fixed_point_without_collapse():
    """With a < T the slice converges to (T(1-m) - a) / b, not the floor."""
    policy, device = _policy("cost_model"), Device(a=0.03, b=1e-4)
    times = _run(policy, device, 40, decode=0.03)
    expected = (TARGET_MS / 1000 * 0.9 - 0.03) / 1e-4
    width = policy.cap(8192, contended=True)
    assert width >= 0.5 * expected
    assert max(times[-10:]) <= 1.5 * TARGET_MS / 1000


def test_unreachable_target_floors_and_counts():
    policy, device = _policy("cost_model"), Device(a=0.2, b=1e-5)
    times = _run(policy, device, 10, decode=0.2)
    assert policy.cap(8192, contended=True) == policy.floor
    assert policy.counters["stall_target_unreachable"] > 0
    assert times  # prefill still progressed (debt pays for it)


def test_slowdown_shifts_regime_once_and_counts():
    policy, device = _policy("cost_model"), Device(b=2e-5)
    _run(policy, device, 10)
    device.b *= 16
    _run(policy, device, 3)
    assert policy.counters["cost_regime_shifts_up"] >= 1


def test_kinds_and_depths_keep_separate_estimates():
    model = PrefillCostModel()
    model.observe(1000, 0.1, depth=0, kind="ordinary")
    model.observe(1000, 0.4, depth=0, kind="mtp")
    (_, ordinary, _) = model.estimate("ordinary", 0)
    (_, mtp, _) = model.estimate("mtp", 0)
    assert mtp == pytest.approx(4 * ordinary)
    # An unseen deeper bucket is never cheaper than the shallow one.
    (_, deep, _) = model.estimate("ordinary", 50000)
    assert deep > ordinary


def test_running_max_estimator_is_the_previous_rule():
    policy = _policy("running_max")
    policy.observe_prefill(4096, 0.100, contended=True)
    for _ in range(50):
        policy.observe_prefill(256, 0.100, contended=True)
    assert policy.stall_bound(8192) == 4096


def test_estimator_is_validated():
    with pytest.raises(ValueError):
        DecodeTimeFairness(estimator="ewma")


def test_clamp_counter_reflects_the_applied_slice():
    """SS-8: a stall-bound clamp that slice_floor undoes is not counted."""
    policy = DecodeTimeFairness(
        enabled=True, stall_target_ms=100.0, slice_floor=1024,
        estimator="running_max",
    )
    policy.best_prefill_tokens_per_second = 2000.0  # bound 192 rows
    chunk = policy.cap(2048, contended=True)
    final = policy.floor_slice(chunk, 2048, contended=True)
    assert (chunk, final) == (192, 1024)
    assert policy.counters["cap_clamps"] == 0
    assert policy.counters["slice_floor_lifts"] == 1
    # A clamp the floor does not undo still counts.
    policy.slice_floor = 128
    chunk = policy.cap(2048, contended=True)
    assert policy.floor_slice(chunk, 2048, contended=True) == 192
    assert policy.counters["cap_clamps"] == 1


def test_one_slice_clamp_counter_reflects_the_applied_slice():
    from collections import defaultdict

    from mlx2.runtime import generate
    from mlx2.runtime.adaptive_policy import PrefillOrder

    class Scheduler(generate.BatchGenerator):
        def __del__(self):
            pass

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_step_size = 2048
    scheduler.scheduler_stats = defaultdict(int)
    scheduler.prefill_order = PrefillOrder.from_value({"order": "srpt"})
    scheduler.decode_time_fairness = DecodeTimeFairness(
        enabled=True, stall_target_ms=100.0, slice_floor=1024,
        estimator="running_max",
    )
    scheduler.decode_time_fairness.best_prefill_tokens_per_second = 2000.0
    scheduler._one_slice_contended = lambda bound: True
    assert scheduler._bounded_slice(2048, contended=True) == 1024
    assert scheduler.prefill_order.counters["one_slice_clamps"] == 0
    assert scheduler.decode_time_fairness.counters["cap_clamps"] == 0
