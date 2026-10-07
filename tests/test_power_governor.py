"""Opt-in power governor: policy, controller, actuators, receipts (CPU).

The controller is driven with a fake clock and a simulated power plant built
from the thermal-q38 measurements (M5 Max, Qwen3.8-27B): about 1.2 W GPU+DRAM
idle and 38-45 W busy, with paced rounds drawing more per busy second (the
duty runs' DVFS boost).
"""

import pytest

from mlx2.power_governor import (
    NEUTRAL_HINT,
    PowerGovernor,
    PowerGovernorPolicy,
    PowerSignalUnavailable,
)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _governor(policy, *, max_lanes=4):
    clock = Clock()
    return PowerGovernor(policy, max_lanes=max_lanes, clock=clock), clock


IDLE_W = 1.2


def _busy_watts(rest_fraction):
    # Busy power rises when rounds are paced (thermal-q38 duty: P~10, 55 W).
    return 40.0 + 15.0 * rest_fraction


def _simulate(governor, clock, seconds, *, round_s=0.03, ids=("a", "b", "c", "d")):
    """Run the scheduler loop against the plant; return per-interval watts."""
    end = clock.now + seconds
    interval_t = interval_busy = interval_rest = 0.0
    trace = []
    while clock.now < end:
        hint = governor.round_hint(ids)
        assert hint.pace_seconds <= governor.MAX_PACE_SECONDS
        clock.now += hint.pace_seconds
        interval_rest += hint.pace_seconds
        clock.now += round_s
        interval_busy += round_s
        interval_t = interval_busy + interval_rest
        if interval_t >= 1.0:
            rest = interval_rest / interval_t
            watts = (
                interval_busy * _busy_watts(rest) + interval_rest * IDLE_W
            ) / interval_t
            governor.observe(interval_t, watts)
            trace.append((watts, interval_busy / interval_t))
            interval_busy = interval_rest = 0.0
    return trace


# -- policy ----------------------------------------------------------------


def test_policy_default_off_and_validation():
    assert PowerGovernorPolicy.from_value(None).enabled is False
    assert PowerGovernorPolicy.from_value("off").enabled is False
    assert PowerGovernorPolicy.from_value("efficient").mode == "efficient"
    policy = PowerGovernorPolicy.from_value({"mode": "budget", "budget_watts": 25})
    assert policy.budget_watts == 25.0 and policy.actuator_order == ("pace", "lanes")
    assert PowerGovernorPolicy.from_value(policy.as_dict()) == policy
    physical = PowerGovernorPolicy.from_value({
        "mode": "budget", "budget_watts": 25,
        "control_time_basis": "physical", "max_control_interval_seconds": 0.5,
    })
    assert physical.control_time_basis == "physical"
    assert physical.max_control_interval_seconds == 0.5
    assert PowerGovernorPolicy.from_value(physical.as_dict()) == physical
    with pytest.raises(ValueError, match="requires budget_watts"):
        PowerGovernorPolicy.from_value("budget")
    with pytest.raises(ValueError, match="mode must be"):
        PowerGovernorPolicy.from_value("turbo")
    with pytest.raises(ValueError, match="unknown"):
        PowerGovernorPolicy.from_value({"mode": "efficient", "clock_mhz": 900})
    with pytest.raises(ValueError, match="actuator_order"):
        PowerGovernorPolicy.from_value({"mode": "efficient", "actuator_order": ["dvfs"]})
    with pytest.raises(ValueError, match="budget_watts"):
        PowerGovernorPolicy.from_value({"mode": "budget", "budget_watts": 0})
    with pytest.raises(ValueError, match="control_time_basis"):
        PowerGovernorPolicy.from_value({"mode": "efficient", "control_time_basis": "wall"})
    with pytest.raises(ValueError, match="max_control_interval_seconds"):
        PowerGovernorPolicy.from_value({
            "mode": "efficient", "max_control_interval_seconds": 0,
        })


# -- controller ------------------------------------------------------------


@pytest.mark.parametrize("budget", [25.0, 15.0])
def test_budget_controller_converges_under_the_cap(budget):
    governor, clock = _governor({"mode": "budget", "budget_watts": budget})
    trace = _simulate(governor, clock, 180)
    assert trace[0][0] > budget  # starts over budget (unthrottled ~40 W)
    status = governor.status()
    average = status["window"]["gpu_dram_watts"]
    # Settles inside the hysteresis band, measured over the 30 s window.
    assert budget * 0.8 <= average <= budget * 1.02, average
    assert status["state"] == "throttling"
    # Pacing alone holds it: the lane rung is untouched, width stays 4.
    assert status["actuators"]["lane_cap"] == 4
    late = [watts for watts, _busy in trace[-30:]]
    assert max(late) - min(late) < budget * 0.25  # no limit cycle


def test_ramp_is_bounded_per_sample():
    governor, clock = _governor({"mode": "budget", "budget_watts": 10})
    previous = 0.0
    for _ in range(5):
        clock.now += 1
        governor.observe(1.0, 400.0)
        throttle = governor.status()["throttle"]
        assert throttle - previous <= PowerGovernor.MAX_STEP + 1e-9
        previous = throttle
    assert previous == pytest.approx(0.5)


@pytest.mark.parametrize("interval,count", [(0.1, 10), (0.25, 4), (0.5, 2), (1.0, 1)])
def test_physical_time_ramp_is_cadence_stable(interval, count):
    governor, clock = _governor({
        "mode": "budget", "budget_watts": 25,
        "control_time_basis": "physical",
    })
    for _ in range(count):
        clock.now += interval
        governor.observe(interval, 50.0)
    assert governor.status()["throttle"] == pytest.approx(0.1)


def test_sample_time_ramp_remains_per_observation():
    throttles = []
    for interval, count in ((0.1, 10), (1.0, 1)):
        governor, clock = _governor({"mode": "budget", "budget_watts": 25})
        for _ in range(count):
            clock.now += interval
            governor.observe(interval, 50.0)
        throttles.append(governor.status()["throttle"])
    assert throttles == pytest.approx([1.0, 0.1])


@pytest.mark.parametrize("interval,count", [(0.1, 40), (0.25, 16), (0.5, 8), (1.0, 4)])
def test_physical_time_release_is_cadence_stable(interval, count):
    governor, clock = _governor({
        "mode": "budget", "budget_watts": 25,
        "control_time_basis": "physical",
    })
    for _ in range(10):
        clock.now += 1.0
        governor.observe(1.0, 50.0)
    assert governor.status()["throttle"] == pytest.approx(1.0)
    for _ in range(count):
        clock.now += interval
        governor.observe(interval, 0.0)
    assert governor.status()["throttle"] == pytest.approx(0.9)


def test_physical_time_ramp_caps_a_delayed_interval_and_ignores_wall_clock_gap():
    governor, clock = _governor({
        "mode": "budget", "budget_watts": 25,
        "control_time_basis": "physical", "max_control_interval_seconds": 0.5,
    })
    clock.now += 100.0
    governor.observe(10.0, 50.0)
    assert governor.status()["throttle"] == pytest.approx(0.05)


def test_hysteresis_holds_in_band_and_releases_after_a_delay():
    governor, clock = _governor({"mode": "budget", "budget_watts": 20})
    for _ in range(4):
        clock.now += 1
        governor.observe(1.0, 30.0)
    held = governor.status()["throttle"]
    assert held > 0
    for _ in range(10):  # inside [18, 20] W: hold
        clock.now += 1
        governor.observe(1.0, 19.0)
    assert governor.status()["throttle"] == held
    clock.now += 1
    governor.observe(1.0, 10.0)
    clock.now += 1
    governor.observe(1.0, 10.0)  # below band for < 3 s: still held
    assert governor.status()["throttle"] == held
    for _ in range(3):
        clock.now += 1
        governor.observe(1.0, 10.0)
    released = governor.status()["throttle"]
    assert held - PowerGovernor.MAX_STEP * 2 - 1e-9 <= released < held


def test_unreachable_budget_saturates_the_ladder_but_never_starves():
    governor, clock = _governor({"mode": "budget", "budget_watts": 2})
    trace = _simulate(governor, clock, 120)
    status = governor.status()
    assert status["throttle"] == 1.0
    assert status["actuators"]["lane_cap"] == 1  # lane_floor
    assert status["actuators"]["duty"] == pytest.approx(0.1)
    # Admitted work still gets at least the duty floor of busy time.
    assert min(busy for _watts, busy in trace[-20:]) >= 0.09


def test_lanes_first_order_caps_before_pacing():
    governor, clock = _governor(
        {"mode": "budget", "budget_watts": 20, "actuator_order": ["lanes", "pace"]}
    )
    for _ in range(4):
        clock.now += 1
        governor.observe(1.0, 60.0)
    status = governor.status()
    assert status["actuators"]["lane_cap"] < 4
    assert status["actuators"]["duty"] == 1.0
    hint = governor.round_hint(["a"])
    assert hint.lane_cap == status["actuators"]["lane_cap"]


def test_stale_signal_holds_the_throttle():
    governor, clock = _governor({"mode": "budget", "budget_watts": 20})
    for _ in range(3):
        clock.now += 1
        governor.observe(1.0, 40.0)
    held = governor.status()["throttle"]
    clock.now += 60
    status = governor.status()
    assert status["signal"]["stale"] is True
    assert status["state"] == "holding_no_signal"
    assert status["throttle"] == held


# -- modes and actuation ---------------------------------------------------


@pytest.mark.parametrize("mode", ["max_throughput", "off"])
def test_neutral_modes_never_actuate(mode):
    governor, clock = _governor({"mode": mode})
    for _ in range(30):
        clock.now += 1
        governor.observe(1.0, 500.0)
    assert governor.round_hint([]) is NEUTRAL_HINT
    clock.now += 5
    assert governor.round_hint(["a", "b"]) is NEUTRAL_HINT
    status = governor.status()
    assert status["throttle"] == 0.0 and status["state"] == "neutral"
    assert status["counts"]["paced_rounds"] == 0
    assert status["counts"]["capped_rounds"] == 0
    receipt = governor.request_receipt("a")
    assert receipt["pace_ms"] == 0 and receipt["numerics"] == "unchanged"


def test_efficient_holds_idle_admission_and_never_paces():
    governor, clock = _governor({"mode": "efficient", "efficient_hold_ms": 80})
    for _ in range(10):
        clock.now += 1
        governor.observe(1.0, 500.0)
    idle = governor.round_hint([])
    assert idle.coalesce_seconds == pytest.approx(0.08)
    busy = governor.round_hint(["a", "b"])
    assert busy == type(busy)(lane_cap=None, pace_seconds=0.0, coalesce_seconds=None)
    clock.now += 10
    assert governor.round_hint(["a", "b", "c"]).pace_seconds == 0.0
    a = governor.request_receipt("a")  # finishes during this round
    late = governor.round_hint(["b", "c"])  # "c" joined after a non-held hint
    assert late.pace_seconds == 0.0
    c = governor.request_receipt("c")
    assert a["coalesce_hold_ms"] == 80 and a["numerics"] == "batch_width_may_differ"
    assert c["coalesce_hold_ms"] == 0 and c["numerics"] == "unchanged"
    assert governor.status()["counts"]["held_admissions"] == 2


def test_runtime_mode_switches_and_receipts():
    governor, clock = _governor({"mode": "max_throughput"})
    governor.round_hint(["r1"])
    with pytest.raises(ValueError, match="requires watts"):
        governor.configure(mode="budget")
    governor.configure(mode="budget", budget_watts=10)
    for _ in range(10):
        clock.now += 1
        governor.observe(1.0, 40.0)
    assert governor.status()["throttle"] == pytest.approx(1.0)
    paced = 0.0
    for _ in range(40):
        hint = governor.round_hint(["r1"])
        paced += hint.pace_seconds
        clock.now += hint.pace_seconds + 0.03
    assert paced > 0
    receipt = governor.request_receipt("r1")
    assert receipt["modes"] == ["budget", "max_throughput"]
    assert receipt["budget_watts"] == 10
    assert receipt["pace_ms"] == pytest.approx(paced * 1000)
    assert receipt["lane_cap_min"] == 1
    assert receipt["numerics"] == "batch_width_may_differ"
    status = governor.configure(mode="efficient")
    assert status["throttle"] == 0.0 and status["mode"] == "efficient"
    assert status["counts"]["mode_changes"] == 2
    assert status["last_change"]["source"] == "admin"
    assert governor.round_hint(["r2"]).pace_seconds == 0.0


def test_budget_needs_a_signal():
    governor, _clock = _governor({"mode": "max_throughput"})
    governor.signal_reason = "IOReport unavailable"
    with pytest.raises(PowerSignalUnavailable):
        governor.configure(mode="budget", budget_watts=20)
    assert governor.mode == "max_throughput"


def test_request_accounts_stay_bounded():
    governor, _clock = _governor({"mode": "efficient"})
    for index in range(100):
        governor.round_hint([f"r{index}"])
    assert governor.status()["tracked_requests"] == 1


def test_telemetry_feeds_the_governor_gpu_dram_watts():
    from test_power_telemetry import Tokens, _reading, _telemetry

    telemetry, energy = _telemetry(Tokens())
    governor, clock = _governor({"mode": "budget", "budget_watts": 20})
    telemetry.add_listener(governor.observe)
    energy.readings.append(_reading(seconds=0.5, gpu=30.0, dram=10.0, cpu=50.0))
    telemetry.sample_once()
    signal = governor.status()["signal"]
    # CPU and ANE are not governed: only the attributed GPU + DRAM domains.
    assert signal["last_gpu_dram_watts"] == pytest.approx(40.0)
    assert signal["observed_samples"] == 1 and signal["stale"] is False
    assert governor.status()["throttle"] == pytest.approx(PowerGovernor.MAX_STEP)



def test_budget_steps_on_each_interval_not_the_lagging_window_mean():
    # Deliberate: the controller steps on the interval that just closed, so
    # one hot interval tightens even while the 30 s window averages 11 W.
    # Stepping on the lagging window mean instead turns the ramped integral
    # into a limit cycle (this plant swings between about 6 W and 40 W) and
    # so breaks the rolling budget it was meant to honour.
    governor, clock = _governor({"mode": "budget", "budget_watts": 20})
    for _ in range(29):
        clock.now += 1
        governor.observe(1.0, 10.0)
    clock.now += 1
    governor.observe(1.0, 40.0)
    status = governor.status()
    assert status["window"]["gpu_dram_watts"] == pytest.approx(11.0)
    assert status["window"]["within_budget"] is True
    assert status["throttle"] == pytest.approx(PowerGovernor.MAX_STEP)
    # After an idle stretch the held load still keeps every 30 s window
    # near the budget (the windowed-mean controller overshoots to ~33 W).
    governor, clock = _governor({"mode": "budget", "budget_watts": 25})
    for _ in range(60):
        clock.now += 1
        governor.observe(1.0, IDLE_W)
    watts = [IDLE_W] * 60 + [w for w, _busy in _simulate(governor, clock, 180)]
    worst = max(sum(watts[i - 30:i]) / 30 for i in range(30, len(watts) + 1))
    assert worst <= 25 * 1.05, worst


def test_budget_window_trims_a_straddling_interval_in_proportion():
    governor, clock = _governor({
        "mode": "budget", "budget_watts": 70, "window_seconds": 30,
    })
    clock.now = 30.0
    governor.observe(30.0, 100.0)
    clock.now = 40.0
    governor.observe(10.0, 0.0)
    window = governor.status()["window"]
    assert window["seconds"] == pytest.approx(30.0)
    assert window["gpu_dram_watts"] == pytest.approx(2000.0 / 30.0)
    assert window["within_budget"] is True


def test_budget_window_trims_one_long_interval_to_the_window():
    governor, clock = _governor({
        "mode": "max_throughput", "window_seconds": 30,
    })
    clock.now = 60.0
    governor.observe(60.0, 42.0)
    window = governor.status()["window"]
    assert window["seconds"] == pytest.approx(30.0)
    assert window["gpu_dram_watts"] == pytest.approx(42.0)
