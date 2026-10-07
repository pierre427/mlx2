"""Power governor serving surfaces: metrics, CLI, admin endpoint, engine (CPU).

Engine tests use injected IOReport readings and the tiny route harness.
"""

import json
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_power_governor import _governor, _simulate

from mlx2.power_governor import PowerGovernorPolicy

# -- metrics ---------------------------------------------------------------


def test_metrics_render_governor_families():
    from test_prometheus import FakeEngine, assert_valid_prometheus_text

    from mlx2.prometheus import render_engine_metrics

    engine = FakeEngine()
    assert "mlx2_power_governor" not in render_engine_metrics(engine)
    governor, clock = _governor({"mode": "budget", "budget_watts": 25})
    _simulate(governor, clock, 60)
    engine.power_governor = governor
    text = render_engine_metrics(engine)
    assert_valid_prometheus_text(text)
    status = governor.status()
    for line in (
        'mlx2_power_governor_mode{mode="budget"} 1',
        'mlx2_power_governor_mode{mode="efficient"} 0',
        "mlx2_power_governor_budget_watts 25",
        "mlx2_power_governor_lane_cap 4",
        "mlx2_power_governor_signal_stale 0",
        f"mlx2_power_governor_paced_rounds_total {status['counts']['paced_rounds']}",
    ):
        assert line in text, line
    assert "# TYPE mlx2_power_governor_throttle_seconds_total counter" in text
    (seconds,) = [
        float(line.split()[1]) for line in text.splitlines()
        if line.startswith("mlx2_power_governor_throttle_seconds_total ")
    ]
    assert seconds == pytest.approx(status["counts"]["pace_seconds"])
    assert seconds > 0


# -- CLI and admin endpoint ------------------------------------------------


def test_cli_maps_to_engine_policy_default_off(monkeypatch, capsys):
    from mlx2 import server

    parser = server.build_parser()
    kw = {"native_mtp": False, "approximate_kv": None, "max_request_bytes": 1 << 20}
    default = parser.parse_args(["--model", "m"])
    assert server.serving_engine_kwargs(default, None, **kw)["power_governor"] is None
    on = parser.parse_args(
        ["--model", "m", "--power-telemetry", "--power-governor", "budget",
         "--power-budget-watts", "25"]
    )
    value = server.serving_engine_kwargs(on, None, **kw)["power_governor"]
    policy = PowerGovernorPolicy.from_value(value)
    assert (policy.mode, policy.budget_watts, policy.window_seconds) == ("budget", 25.0, 30.0)
    for argv, message in (
        (["--power-governor", "efficient"], "requires --power-telemetry"),
        (["--power-telemetry", "--power-governor", "budget"], "requires budget_watts"),
        (["--power-telemetry", "--power-budget-watts", "20"], "requires --power-governor budget"),
    ):
        monkeypatch.setattr(sys, "argv", ["mlx2.server", "--model", "unused", *argv])
        with pytest.raises(SystemExit) as caught:
            server.main()
        assert caught.value.code == 2
        assert message in capsys.readouterr().err


class PowerEngine:
    model_path = "fixture"

    def __init__(self, governor=None):
        self.lock = threading.Lock()
        self.power_governor = governor

    def power_governor_state(self):
        return self.power_governor.status() if self.power_governor else None

    def set_power_governor(self, *, mode=None, watts=None):
        from mlx2.serving import PowerGovernorUnavailable

        if self.power_governor is None:
            raise PowerGovernorUnavailable("power governor is not enabled")
        return self.power_governor.configure(mode=mode, budget_watts=watts)


@pytest.fixture
def power_server():
    servers = []

    def start(engine, **kw):
        from mlx2.server import handler_for

        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine, **kw))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}"

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join()


def _call(base, method, body=None, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        base + "/v1/admin/power",
        data=None if body is None else json.dumps(body).encode(),
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def test_admin_power_endpoint_requires_the_admin_token(power_server):
    governor, _clock = _governor({"mode": "max_throughput"})
    base = power_server(PowerEngine(governor), admin_token="secret")
    assert _call(base, "GET")[0] == 401
    assert _call(base, "POST", {"mode": "budget", "watts": 25})[0] == 401
    assert _call(base, "POST", {"mode": "budget", "watts": 25}, token="wrong")[0] == 401
    assert governor.mode == "max_throughput"
    status, body = _call(base, "POST", {"mode": "budget", "watts": 25}, token="secret")
    assert status == 200 and body["mode"] == "budget" and body["budget_watts"] == 25
    status, body = _call(base, "POST", {"watts": 15}, token="secret")
    assert status == 200 and body["budget_watts"] == 15 and body["mode"] == "budget"
    status, body = _call(base, "GET", token="secret")
    assert status == 200 and body["mode"] == "budget"
    assert body["last_change"]["source"] == "admin"


def test_admin_power_endpoint_validation_and_absence(power_server):
    governor, _clock = _governor({"mode": "max_throughput"})
    base = power_server(PowerEngine(governor))
    for bad in ({}, {"mode": "turbo"}, {"watts": 0}, {"watts": True},
                {"mode": "budget", "watts": 20, "dvfs": 3}, {"mode": "budget"}):
        assert _call(base, "POST", bad)[0] == 400, bad
    governor.signal_reason = "IOReport unavailable"
    assert _call(base, "POST", {"mode": "budget", "watts": 20})[0] == 409
    absent = power_server(PowerEngine())
    assert _call(absent, "GET")[0] == 409
    assert _call(absent, "POST", {"mode": "efficient"})[0] == 409


# -- engine ----------------------------------------------------------------


@pytest.fixture
def fake_host_samplers(monkeypatch):
    from test_power_telemetry import FakeEnergy, FakeTemperature

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


TELEMETRY = {"interval_seconds": 60.0, "window_seconds": 600.0}


def test_engine_governor_requires_power_telemetry():
    from mlx2.serving import ServingEngine

    with pytest.raises(ValueError, match="requires power_telemetry"):
        ServingEngine.validate_arguments("model", power_governor="efficient")


def test_engine_default_has_no_governor(monkeypatch, fake_host_samplers):
    import route_harness as H

    engine = _tiny_engine(monkeypatch, power_telemetry=TELEMETRY)
    try:
        assert engine.power_governor is None
        assert engine.power_governor_state() is None
        result = H.run(engine, {"tokens": [1, 2, 3], "max_tokens": 2, "temperature": 0})
        assert "error" not in result, result
        assert "governor" not in engine.status()["power"]
        assert "power_governor" not in engine.recent_receipts()[-1]
    finally:
        engine.close()


def test_engine_budget_mode_refuses_to_start_without_a_signal(monkeypatch):
    import route_harness as H

    import mlx2.power_telemetry as power

    def refuse():
        raise RuntimeError("no IOReport")

    monkeypatch.setattr(power, "_default_energy_sampler", refuse)
    H.patch_host(monkeypatch)
    model, vocab = H.tiny_qwen38_mtp()
    with pytest.raises(RuntimeError, match="needs a power signal"):
        H.make_engine(
            model, vocab, mtp=False, power_telemetry=TELEMETRY,
            power_governor={"mode": "budget", "budget_watts": 20},
        )


def _tokens(result):
    return result.get("tokens") or result.get("token_ids") or result.get("text")


def test_engine_pacing_is_output_neutral_and_receipted(monkeypatch, fake_host_samplers):
    import route_harness as H

    request = {"tokens": [1, 2, 3], "max_tokens": 6, "temperature": 0}
    reference_engine = _tiny_engine(monkeypatch, power_telemetry=TELEMETRY)
    try:
        reference = H.run(reference_engine, request)
    finally:
        reference_engine.close()
    engine = _tiny_engine(
        monkeypatch,
        power_telemetry=TELEMETRY,
        power_governor={"mode": "budget", "budget_watts": 5},
    )
    try:
        governor = engine.power_governor
        # Drive the throttle to the duty floor.  Anti-windup keeps an idle
        # engine from tightening on host-wide power, so set the state a
        # sustained overload produces.
        with governor._lock:
            governor._throttle = 1.0
        assert governor.status()["actuators"]["duty"] == pytest.approx(0.1)
        governor.MIN_PACE_SECONDS = 0.0  # pace every round on this tiny model
        result = H.run(engine, request)
        assert "error" not in result, result
        assert _tokens(result) == _tokens(reference)
        receipt = engine.recent_receipts()[-1]["power_governor"]
        assert receipt["mode"] == "budget" and receipt["budget_watts"] == 5
        assert receipt["pace_ms"] > 0
        assert receipt["lane_cap_min"] is None  # max_lanes 1: nothing to cap
        assert receipt["numerics"] == "unchanged"
        status = engine.status()["power"]["governor"]
        assert status["counts"]["paced_rounds"] > 0
        engine.set_power_governor(mode="max_throughput")
        result = H.run(engine, request)
        assert engine.recent_receipts()[-1]["power_governor"]["pace_ms"] == 0
    finally:
        engine.close()


def test_engine_lane_cap_serves_every_admitted_request(monkeypatch, fake_host_samplers):
    import route_harness as H

    engine = _tiny_engine(
        monkeypatch,
        max_lanes=3,
        power_telemetry=TELEMETRY,
        power_governor={
            "mode": "budget", "budget_watts": 5, "actuator_order": ["lanes"],
            "efficient_hold_ms": 0,
        },
    )
    try:
        governor = engine.power_governor
        # Anti-windup: an idle engine does not tighten on host-wide power,
        # so start the test with the throttle a sustained overload sets.
        with governor._lock:
            governor._throttle = 1.0
        assert governor.status()["actuators"]["lane_cap"] == 1
        # An idle admission wait already in flight keeps the hint it formed
        # with; let it turn over (the idle loop wakes every 50 ms).
        time.sleep(0.3)
        jobs = [
            engine.submit({"tokens": [1, 2, index], "max_tokens": 4, "temperature": 0})
            for index in range(3, 6)
        ]
        results = [H.collect(job) for job in jobs]
        assert all("error" not in result for result in results), results
        receipts = [
            receipt["power_governor"] for receipt in engine.recent_receipts()[-3:]
        ]
        assert all(receipt["lane_cap_min"] == 1 for receipt in receipts)
        assert all(r["numerics"] == "batch_width_may_differ" for r in receipts)
        assert engine.counts["multi_request_cycles"] == 0  # never wider than the cap
        assert governor.status()["counts"]["capped_rounds"] > 0
    finally:
        engine.close()
