"""Opt-in power/thermal telemetry: sampler, attribution, /metrics, status (CPU).

The sampler is driven with injected IOReport readings, so the attribution
law, the exported families and the receipts are pinned without a GPU.  One
test touches the real IOReport when the host has it and skips otherwise.
"""

import threading
import time

import pytest

from mlx2.apple_telemetry import EnergyReading
from mlx2.power_telemetry import (
    ATTRIBUTION_LAW,
    GPU_STATES,
    PowerTelemetry,
    PowerTelemetryPolicy,
)


def _reading(seconds=1.0, gpu=10.0, dram=2.0, cpu=5.0, ane=0.0, states=None):
    return EnergyReading(
        seconds=seconds,
        watts={"GPU Energy": gpu, "DRAM": dram, "CPU Energy": cpu, "ANE": ane},
        gpu_states=states
        if states is not None
        else [("OFF", 25), ("P1", 25), ("P2", 50), ("P3", 0)],
    )


class FakeEnergy:
    def __init__(self):
        self.readings = []

    def read(self):
        return self.readings.pop(0)


class FakeTemperature:
    def die_summary(self):
        return {"die_max_c": 71.5, "die_mean_c": 64.25, "battery_c": 30.0}


class Tokens:
    """Mutable stand-in for BatchRuntimeMetrics.token_progress."""

    def __init__(self):
        self.delivered = 0
        self.active = {}

    def emit(self, request_id, count=1):
        tokens, uncached, _prefilling = self.active[request_id]
        self.active[request_id] = (tokens + count, uncached, False)
        self.delivered += count

    def __call__(self):
        return self.delivered, dict(self.active)


def _telemetry(tokens, *, table=(300, 600, 900), policy=None, energy=None):
    energy = energy or FakeEnergy()
    telemetry = PowerTelemetry(
        policy or {"interval_seconds": 1.0, "window_seconds": 3.0},
        tokens,
        energy_sampler_factory=lambda: energy,
        temperature_sampler_factory=FakeTemperature,
        frequency_table=lambda: list(table),
    )
    return telemetry, energy


def _step(telemetry, energy, reading=None):
    energy.readings.append(reading or _reading())
    telemetry.sample_once()


def test_policy_default_off_round_trip_and_validation():
    assert PowerTelemetryPolicy.from_value(None).enabled is False
    assert PowerTelemetryPolicy.from_value(False).enabled is False
    assert PowerTelemetryPolicy.from_value(True).as_dict() == {
        "enabled": True, "interval_seconds": 1.0, "window_seconds": 60.0,
    }
    policy = PowerTelemetryPolicy.from_value({"interval_seconds": 0.5})
    assert PowerTelemetryPolicy.from_value(policy.as_dict()) == policy
    for bad, match in (
        ({"interval_seconds": 0.01}, "0.1 to 60"),
        ({"interval_seconds": 5, "window_seconds": 1}, "at least interval"),
        ({"interval_seconds": True}, "number of seconds"),
        ({"interval_seconds": float("nan")}, "finite"),
        ({"every": 1}, "unknown"),
        ({"enabled": "yes"}, "boolean"),
        ("on", "boolean or an object"),
    ):
        with pytest.raises(ValueError, match=match):
            PowerTelemetryPolicy.from_value(bad)


def test_unavailable_host_degrades_without_a_thread():
    def refuse():
        raise RuntimeError("IOReport is unavailable on this host")

    before = {thread.name for thread in threading.enumerate()}
    telemetry = PowerTelemetry(
        True, Tokens(), energy_sampler_factory=refuse,
        temperature_sampler_factory=pytest.fail,
    ).start()
    assert telemetry.available is False and telemetry.running is False
    assert "mlx2-power-telemetry" not in {t.name for t in threading.enumerate()} - before
    status = telemetry.status()
    assert status["available"] is False
    assert status["reason"] == "RuntimeError: IOReport is unavailable on this host"
    telemetry.sample_once()  # a no-op, never raises
    assert telemetry.request_energy("r", 4) == {
        "schema": "mlx2.energy-estimate.v1",
        "available": False,
        "reason": "RuntimeError: IOReport is unavailable on this host",
    }
    telemetry.close()


def test_non_darwin_default_sampler_reports_unavailable(monkeypatch):
    import mlx2.apple_telemetry as apple

    monkeypatch.setattr(apple, "_LIBS", None)
    monkeypatch.setattr(apple.platform, "system", lambda: "Linux")
    telemetry = PowerTelemetry(True, Tokens()).start()
    assert telemetry.available is False and not telemetry.running
    assert "unavailable" in telemetry.status()["reason"]


def test_interval_energy_splits_by_output_token_share():
    tokens = Tokens()
    tokens.active = {"a": (0, 10, False), "b": (0, 10, False)}
    telemetry, energy = _telemetry(tokens)
    tokens.emit("a", 3)
    tokens.emit("b", 1)
    _step(telemetry, energy)  # 12 J of gpu+dram over 4 tokens
    tokens.emit("a", 1)
    tokens.emit("b", 1)
    _step(telemetry, energy, _reading(gpu=4.0, dram=0.0))  # 4 J over 2 tokens

    status = telemetry.status()
    assert status["last"]["attribution_basis"] == "output_tokens"
    assert status["energy_joules_total"] == {
        "gpu": 14.0, "dram": 2.0, "cpu": 10.0, "ane": 0.0,
    }
    assert status["window"]["output_tokens"] == 6
    assert status["window"]["joules_per_output_token"] == pytest.approx(16.0 / 6)
    assert status["window"]["tokens_per_second"] == pytest.approx(3.0)
    assert status["attribution"]["attributed_joules"] == pytest.approx(16.0)

    a = telemetry.request_energy("a", 4)
    assert a["measured_interval_joules"] == pytest.approx(12 * 3 / 4 + 4 / 2)
    assert a["measured_intervals"] == 2
    assert a["open_interval_tokens"] == 0 and a["open_interval_joules"] == 0.0
    assert a["joules"] == pytest.approx(11.0)
    assert a["joules_per_output_token"] == pytest.approx(11.0 / 4)
    assert a["attribution"] == ATTRIBUTION_LAW and a["estimate"] is True
    b = telemetry.request_energy("b", 2)
    assert b["joules"] == pytest.approx(12 / 4 + 4 / 2)
    assert a["joules"] + b["joules"] == pytest.approx(16.0)


def test_finishing_request_charges_its_open_tail_at_the_window_rate():
    tokens = Tokens()
    tokens.active = {"a": (0, 0, False), "b": (0, 0, False)}
    telemetry, energy = _telemetry(tokens)
    tokens.emit("a", 2)
    tokens.emit("b", 2)
    _step(telemetry, energy)  # 12 J / 4 tokens -> 3 J per token
    tokens.emit("a", 3)  # a's tail, delivered after the last closed interval
    receipt = telemetry.request_energy("a", 5)
    assert receipt["measured_interval_joules"] == pytest.approx(6.0)
    assert receipt["open_interval_tokens"] == 3
    assert receipt["open_interval_joules"] == pytest.approx(9.0)
    assert receipt["joules"] == pytest.approx(15.0)
    # The receipt has left, but the token source still lists "a" until the
    # engine's terminal bookkeeping runs: the closed account must not reopen.
    tokens.emit("b", 1)
    _step(telemetry, energy)  # 12 J over 4 tokens: a 3, b 1
    b = telemetry.request_energy("b", 3)
    assert b["measured_interval_joules"] == pytest.approx(6.0 + 3.0)
    attribution = telemetry.status()["attribution"]
    # a's three tail tokens' share of the second interval was estimated in
    # its receipt, so it is reported, not attributed twice.
    assert attribution["finished_in_interval_joules"] == pytest.approx(9.0)
    del tokens.active["a"]
    _step(telemetry, energy)
    assert telemetry.status()["attribution"]["tracked_requests"] == 0


def test_no_window_rate_leaves_the_tail_uncharged_and_says_so():
    tokens = Tokens()
    tokens.active = {"a": (0, 0, False)}
    telemetry, _energy = _telemetry(tokens)
    tokens.emit("a", 2)
    receipt = telemetry.request_energy("a", 2)
    assert receipt["open_interval_tokens"] == 2
    assert receipt["open_interval_joules"] is None
    assert receipt["joules"] == 0.0


def test_prefill_only_interval_splits_by_uncached_prompt_tokens():
    tokens = Tokens()
    tokens.active = {
        "long": (0, 300, True),
        "short": (0, 100, True),
        "queued": (0, 999, False),  # not attached yet: no share
    }
    telemetry, energy = _telemetry(tokens)
    _step(telemetry, energy)
    assert telemetry.status()["last"]["attribution_basis"] == "prompt_tokens"
    assert telemetry.request_energy("long", 0)["joules"] == pytest.approx(9.0)
    assert telemetry.request_energy("short", 0)["joules"] == pytest.approx(3.0)
    assert telemetry.request_energy("queued", 0)["joules"] == 0.0


def test_idle_interval_is_attributed_to_no_request():
    tokens = Tokens()
    telemetry, energy = _telemetry(tokens)
    _step(telemetry, energy)
    attribution = telemetry.status()["attribution"]
    assert attribution["idle_joules"] == pytest.approx(12.0)
    assert attribution["attributed_joules"] == 0.0
    assert telemetry.status()["window"]["joules_per_output_token"] is None


def test_window_and_request_state_stay_bounded():
    tokens = Tokens()
    telemetry, energy = _telemetry(tokens)
    for index in range(50):
        tokens.active = {f"r{index}": (0, 0, False)}
        tokens.emit(f"r{index}", 1)
        _step(telemetry, energy)
    status = telemetry.status()
    assert status["window"]["intervals"] == 3  # window 3 s / interval 1 s
    assert status["attribution"]["tracked_requests"] == 1
    assert status["samples"] == 50


def test_residency_frequency_and_temperature():
    tokens = Tokens()
    telemetry, energy = _telemetry(tokens)
    _step(telemetry, energy, _reading(seconds=2.0))
    status = telemetry.status()
    last = status["last"]
    assert last["gpu_active_ratio"] == pytest.approx(0.75)
    # Active residency P1 25, P2 50 -> mean P-state 5/3, mean clock 500 MHz.
    assert last["gpu_mean_pstate"] == pytest.approx(5 / 3)
    assert last["gpu_frequency_mean_mhz"] == pytest.approx(500.0)
    assert status["gpu_frequency"] == {
        "mapped": True, "table_mhz": [300, 600, 900], "reason": None,
    }
    assert status["gpu_pstate_residency_seconds_total"] == {
        "OFF": 0.5, "P1": 0.5, "P2": 1.0, "P3": 0.0,
    }
    assert last["die_temperature_c"] == {"max": 71.5, "mean": 64.25}


def test_frequency_is_omitted_when_the_table_does_not_map():
    tokens = Tokens()
    telemetry, energy = _telemetry(tokens, table=(300, 600))
    _step(telemetry, energy)
    status = telemetry.status()
    assert status["last"]["gpu_frequency_mean_mhz"] is None
    assert status["last"]["gpu_mean_pstate"] is not None
    assert status["gpu_frequency"]["mapped"] is False
    assert "2 table frequencies for 3 P-states" in status["gpu_frequency"]["reason"]


def test_unknown_states_are_not_exported():
    tokens = Tokens()
    telemetry, energy = _telemetry(tokens)
    _step(telemetry, energy, _reading(states=[("OFF", 1), ("WEIRD", 1), ("P1", 2)]))
    assert set(telemetry.status()["gpu_pstate_residency_seconds_total"]) <= set(GPU_STATES)


def test_background_thread_samples_and_stops():
    tokens = Tokens()

    class Endless:
        def read(self):
            return _reading()

    telemetry, _energy = _telemetry(
        tokens, policy={"interval_seconds": 0.1, "window_seconds": 1.0}, energy=Endless()
    )
    telemetry.start()
    assert telemetry.running
    deadline = time.monotonic() + 5
    while telemetry.status()["samples"] < 2:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    telemetry.close()
    assert not telemetry.running


def test_sample_failures_are_counted_not_raised():
    tokens = Tokens()

    class Broken:
        def read(self):
            raise OSError("ioreport went away")

    telemetry, _energy = _telemetry(
        tokens, policy={"interval_seconds": 0.1, "window_seconds": 1.0}, energy=Broken()
    )
    telemetry.start()
    deadline = time.monotonic() + 5
    while telemetry.status()["errors"] < 2:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    telemetry.close()
    assert telemetry.status()["last_error"] == "OSError: ioreport went away"


def test_batch_metrics_token_progress_feeds_the_sampler():
    from mlx2.batch_metrics import BatchRuntimeMetrics

    metrics = BatchRuntimeMetrics()
    metrics.admitted("a", "t", 0)
    metrics.prompt("a", 120, 20)
    assert metrics.token_progress() == (0, {"a": (0, 100, False)})
    metrics.lane_attached("a", 1, "ordinary")
    assert metrics.token_progress() == (0, {"a": (0, 100, True)})
    metrics.token("a")
    metrics.token("a")
    assert metrics.token_progress() == (2, {"a": (2, 100, False)})
    metrics.terminal("a", "completed")
    assert metrics.token_progress() == (2, {})


# -- /metrics ----------------------------------------------------------------


def _metrics_engine(telemetry):
    from test_prometheus import FakeEngine

    engine = FakeEngine()
    engine.power_telemetry = telemetry
    return engine


def test_metrics_families_exist_only_when_enabled():
    from test_prometheus import FakeEngine

    from mlx2.prometheus import render_engine_metrics

    text = render_engine_metrics(FakeEngine())
    for family in ("mlx2_power_", "mlx2_energy_", "mlx2_gpu_", "mlx2_die_"):
        assert family not in text


def test_metrics_render_power_families():
    from test_prometheus import assert_valid_prometheus_text

    from mlx2.prometheus import render_engine_metrics

    tokens = Tokens()
    tokens.active = {"secret-request": (0, 0, False)}
    telemetry, energy = _telemetry(tokens)
    tokens.emit("secret-request", 4)
    _step(telemetry, energy)
    text = render_engine_metrics(_metrics_engine(telemetry))
    assert_valid_prometheus_text(text)
    for line in (
        "mlx2_power_telemetry_available 1",
        "mlx2_power_samples_total 1",
        'mlx2_power_watts{domain="gpu"} 10',
        'mlx2_power_watts{domain="dram"} 2',
        'mlx2_power_watts{domain="ane"} 0',
        'mlx2_energy_joules_total{domain="cpu"} 5',
        "mlx2_gpu_active_ratio 0.75",
        'mlx2_gpu_pstate_residency_seconds_total{state="P2"} 0.5',
        "mlx2_gpu_frequency_mean_hertz 500000000",
        'mlx2_die_temperature_celsius{stat="max"} 71.5',
        'mlx2_die_temperature_celsius{stat="mean"} 64.25',
        "mlx2_energy_per_output_token_joules 3",
    ):
        assert line in text, line
    assert "# TYPE mlx2_energy_joules_total counter" in text
    assert "# TYPE mlx2_power_watts gauge" in text
    assert "secret-request" not in text


def test_metrics_for_an_unavailable_sampler_only_say_so():
    from mlx2.prometheus import render_engine_metrics

    def refuse():
        raise RuntimeError("no IOReport")

    telemetry = PowerTelemetry(True, Tokens(), energy_sampler_factory=refuse)
    text = render_engine_metrics(_metrics_engine(telemetry))
    assert "mlx2_power_telemetry_available 0" in text
    assert "mlx2_power_watts" not in text and "mlx2_energy_joules_total" not in text


def test_metrics_parse_with_official_client_when_installed():
    parser = pytest.importorskip("prometheus_client.parser")
    from mlx2.prometheus import render_engine_metrics

    tokens = Tokens()
    tokens.active = {"a": (0, 0, False)}
    telemetry, energy = _telemetry(tokens)
    tokens.emit("a", 2)
    _step(telemetry, energy)
    families = {
        family.name: family
        for family in parser.text_string_to_metric_families(
            render_engine_metrics(_metrics_engine(telemetry))
        )
    }
    assert families["mlx2_energy_joules"].type == "counter"
    assert families["mlx2_power_watts"].type == "gauge"
    assert families["mlx2_gpu_pstate_residency_seconds"].type == "counter"


# -- engine plumbing ---------------------------------------------------------


def test_cli_maps_to_engine_policy_default_off():
    from mlx2.server import build_parser, serving_engine_kwargs
    from mlx2.serving import ServingEngine

    parser = build_parser()
    default = parser.parse_args(["--model", "m"])
    on = parser.parse_args(
        ["--model", "m", "--power-telemetry", "--power-telemetry-interval", "0.5"]
    )
    kw = {"native_mtp": False, "approximate_kv": None, "max_request_bytes": 1 << 20}
    assert serving_engine_kwargs(default, None, **kw)["power_telemetry"] is None
    value = serving_engine_kwargs(on, None, **kw)["power_telemetry"]
    assert PowerTelemetryPolicy.from_value(value).as_dict() == {
        "enabled": True, "interval_seconds": 0.5, "window_seconds": 60.0,
    }
    with pytest.raises(ValueError, match="0.1 to 60"):
        ServingEngine.validate_arguments(
            "model", power_telemetry={"interval_seconds": 0.0}
        )


@pytest.fixture
def fake_host_samplers(monkeypatch):
    import mlx2.power_telemetry as power

    energy = FakeEnergy()
    monkeypatch.setattr(power, "_default_energy_sampler", lambda: energy)
    monkeypatch.setattr(power, "_default_temperature_sampler", FakeTemperature)
    monkeypatch.setattr(power, "_default_frequency_table", lambda: [300, 600, 900])
    return energy


def _tiny_engine(monkeypatch, **kw):
    import route_harness as H

    H.patch_host(monkeypatch)
    model, vocab = H.tiny_qwen38_mtp()
    return H.make_engine(model, vocab, mtp=False, **kw)


def test_engine_default_has_no_power_thread_status_or_receipt(monkeypatch):
    import route_harness as H

    engine = _tiny_engine(monkeypatch)
    try:
        assert engine.power_telemetry is None
        assert "mlx2-power-telemetry" not in {t.name for t in threading.enumerate()}
        result = H.run(engine, {"tokens": [1, 2, 3], "max_tokens": 2, "temperature": 0})
        assert "error" not in result, result
        assert "power" not in engine.status()
        assert "energy" not in engine.recent_receipts()[-1]
    finally:
        engine.close()


def test_engine_publishes_power_status_and_receipt_energy(monkeypatch, fake_host_samplers):
    import route_harness as H

    energy = fake_host_samplers
    # A long interval keeps the background thread out of the way; the test
    # closes intervals itself.
    engine = _tiny_engine(
        monkeypatch, power_telemetry={"interval_seconds": 60.0, "window_seconds": 600.0}
    )
    try:
        telemetry = engine.power_telemetry
        assert telemetry is not None and telemetry.running
        first = H.run(engine, {"tokens": [1, 2, 3], "max_tokens": 4, "temperature": 0})
        assert "error" not in first, first
        receipt = engine.recent_receipts()[-1]["energy"]
        assert receipt["available"] is True and receipt["estimate"] is True
        # No interval has closed: the tail is reported uncharged.
        assert receipt["open_interval_tokens"] == 4
        assert receipt["open_interval_joules"] is None
        energy.readings.append(_reading())
        telemetry.sample_once()  # 12 J over the first request's 4 tokens
        status = engine.status()["power"]
        assert status["window"]["output_tokens"] == 4
        assert status["window"]["joules_per_output_token"] == pytest.approx(3.0)
        assert status["attribution"]["finished_in_interval_joules"] == pytest.approx(12.0)
        second = H.run(engine, {"tokens": [4, 5, 6], "max_tokens": 2, "temperature": 0})
        assert "error" not in second, second
        receipt = engine.recent_receipts()[-1]["energy"]
        assert receipt["open_interval_joules"] == pytest.approx(6.0)
        assert receipt["joules_per_output_token"] == pytest.approx(3.0)
    finally:
        engine.close()
    assert not telemetry.running


def test_real_ioreport_sampler_when_available():
    from mlx2 import apple_telemetry

    if not apple_telemetry.available():
        pytest.skip("IOReport is not available on this host")
    tokens = Tokens()
    telemetry = PowerTelemetry({"interval_seconds": 1.0}, tokens)
    if not telemetry.available:
        pytest.skip(telemetry.reason)
    telemetry.sample_once()
    status = telemetry.status()
    assert status["samples"] == 1
    assert status["last"]["seconds"] > 0
    assert "gpu" in status["last"]["watts"]
    assert status["last_sample_cost_seconds"] < 1.0


@pytest.mark.parametrize(
    "watts, why",
    [
        ({"DRAM": 2.0, "CPU Energy": 5.0}, "no 'GPU Energy' channel"),
        ({"GPU Energy": 10.0, "CPU Energy": 5.0}, "no 'DRAM' channel"),
        ({}, "no 'GPU Energy' channel"),
        ({"GPU Energy": float("nan"), "DRAM": 2.0}, "'GPU Energy' reading nan"),
        ({"GPU Energy": 10.0, "DRAM": -1.0}, "'DRAM' reading -1.0"),
    ],
)
def test_incomplete_power_domains_never_feed_the_governor(watts, why):
    from mlx2.power_governor import PowerGovernor

    telemetry, energy = _telemetry(Tokens())
    clock = [1000.0]
    governor = PowerGovernor(
        {"mode": "budget", "budget_watts": 20}, max_lanes=4, clock=lambda: clock[0]
    )
    telemetry.add_listener(governor.observe)
    _step(telemetry, energy, _reading(gpu=30.0, dram=10.0))
    assert governor.status()["signal"]["observed_samples"] == 1
    assert telemetry.status()["governor_signal"] == {
        "complete": True, "reason": None, "incomplete_intervals": 0,
    }
    clock[0] += 60
    energy.readings.append(EnergyReading(seconds=1.0, watts=watts, gpu_states=[]))
    telemetry.sample_once()
    # A missing or invalid domain is no signal, not a 0 W one: freshness is
    # not refreshed, so budget mode goes stale and holds its throttle.
    status = governor.status()
    assert status["signal"]["observed_samples"] == 1
    assert status["signal"]["stale"] is True
    assert status["state"] == "holding_no_signal"
    signal = telemetry.status()["governor_signal"]
    assert signal["complete"] is False and signal["incomplete_intervals"] == 1
    assert why in signal["reason"]


class _Closable:
    """Sampler fake that counts closes and flags overlapping reads."""

    def __init__(self, delay=0.0):
        self.closes = 0
        self.reads = 0
        self.overlaps = 0
        self.delay = delay
        self._busy = False

    def read(self):
        if self._busy:
            self.overlaps += 1
        self._busy = True
        try:
            time.sleep(self.delay)
            self.reads += 1
            return _reading()
        finally:
            self._busy = False

    def die_summary(self):
        return {"die_max_c": 70.0, "die_mean_c": 60.0}

    def close(self):
        self.closes += 1


def test_concurrent_start_spawns_one_sampler_thread(monkeypatch):
    real_thread = threading.Thread

    class SlowThread(real_thread):
        def __init__(self, *args, **kwargs):
            if kwargs.get("name") == "mlx2-power-telemetry":
                time.sleep(0.05)  # widen the check-then-create window
            super().__init__(*args, **kwargs)

    energy = _Closable()
    telemetry = PowerTelemetry(
        {"interval_seconds": 60.0, "window_seconds": 60.0}, Tokens(),
        energy_sampler_factory=lambda: energy, temperature_sampler_factory=_Closable,
    )
    monkeypatch.setattr(threading, "Thread", SlowThread)
    starters = [real_thread(target=telemetry.start) for _ in range(4)]
    for starter in starters:
        starter.start()
    for starter in starters:
        starter.join()
    monkeypatch.setattr(threading, "Thread", real_thread)
    try:
        names = [t.name for t in threading.enumerate()]
        assert names.count("mlx2-power-telemetry") == 1
    finally:
        telemetry.close()


def test_external_samples_never_overlap_the_background_read():
    energy = _Closable(delay=0.01)
    telemetry = PowerTelemetry(
        {"interval_seconds": 0.1, "window_seconds": 1.0}, Tokens(),
        energy_sampler_factory=lambda: energy, temperature_sampler_factory=_Closable,
    ).start()
    try:
        callers = [
            threading.Thread(target=lambda: [telemetry.sample_once() for _ in range(20)])
            for _ in range(3)
        ]
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join()
    finally:
        telemetry.close()
    assert energy.reads >= 60 and energy.overlaps == 0


def test_close_releases_the_samplers_once_and_stops_sampling():
    energy, temperature = _Closable(), _Closable()
    telemetry = PowerTelemetry(
        {"interval_seconds": 0.1, "window_seconds": 1.0}, Tokens(),
        energy_sampler_factory=lambda: energy,
        temperature_sampler_factory=lambda: temperature,
    ).start()
    telemetry.sample_once()
    telemetry.close()
    telemetry.close()
    assert (energy.closes, temperature.closes) == (1, 1)
    reads = energy.reads
    telemetry.sample_once()  # a no-op once closed
    telemetry.start()  # never restarts
    assert energy.reads == reads and not telemetry.running
    assert telemetry.status()["samples"] == 1


def test_attribution_window_spans_seconds_not_records():
    tokens = Tokens()
    tokens.active = {"a": (0, 0, False)}
    telemetry, energy = _telemetry(tokens)  # 3 s window, 1 s interval
    tokens.emit("a", 2)
    _step(telemetry, energy, _reading(gpu=10.0, dram=2.0))  # 12 J / 2 tokens
    # A delayed read: one 6 s interval at 4 W over 12 tokens.  Only its last
    # 3 s belong in the window, trimmed in proportion; the 1 s record ages out.
    tokens.emit("a", 12)
    _step(telemetry, energy, _reading(seconds=6.0, gpu=3.0, dram=1.0))
    window = telemetry.status()["window"]
    assert window["seconds"] == pytest.approx(3.0)
    assert window["gpu_dram_joules"] == pytest.approx(12.0)
    assert window["output_tokens"] == pytest.approx(6.0)
    assert window["joules_per_output_token"] == pytest.approx(2.0)
    assert window["intervals"] == 1


def test_down_residency_is_idle_and_exported_with_the_same_population():
    from mlx2.prometheus import render_engine_metrics

    telemetry, energy = _telemetry(Tokens())
    _step(telemetry, energy, _reading(states=[("OFF", 10), ("DOWN", 40), ("P1", 50)]))
    status = telemetry.status()
    assert status["last"]["gpu_active_ratio"] == pytest.approx(0.5)
    # The idle states the active ratio excludes are all exported, so the
    # residency counters describe the same population as the ratio.
    assert status["gpu_pstate_residency_seconds_total"] == {
        "OFF": pytest.approx(0.1), "DOWN": pytest.approx(0.4), "P1": pytest.approx(0.5),
    }
    text = render_engine_metrics(_metrics_engine(telemetry))
    assert 'mlx2_gpu_pstate_residency_seconds_total{state="DOWN"} 0.4' in text
    assert "outside the OFF/IDLE/DOWN states" in text
