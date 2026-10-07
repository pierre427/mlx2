"""Opt-in GPU keep-warm ticker (omlx #3974), CPU.

Apple GPUs park after ~1.5 s without work and stall the next command buffer
on wake.  With ``--gpu-keep-warm-seconds`` the serving worker submits a
one-element kernel from its idle branch while it has served a request within
the window.  These tests pin the scheduling with an injected clock and op,
and the plumbing from the CLI through settings, ``/v1/status`` and
``/metrics``.  The GPU effect itself needs a Metal measurement.
"""

import time
from collections import Counter

import pytest

from mlx2.qualification import PROVENANCE_ONLY_SETTINGS
from mlx2.runtime.gpu_keep_warm import COUNTERS, GpuKeepWarm, GpuKeepWarmPolicy


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _keeper(window=10.0, interval=0.5, op=None):
    clock, ops, counts = Clock(), [], Counter()
    keeper = GpuKeepWarm(
        {"window_seconds": window, "interval_seconds": interval},
        counts,
        clock=clock,
        op=op or (lambda: ops.append(clock.now)),
    )
    return keeper, clock, ops, counts


def _idle(keeper, clock, until, step=0.05, **kw):
    while clock.now < until - 1e-9:
        clock.now = round(clock.now + step, 6)
        keeper.maybe_tick(**kw)


def test_policy_default_off_round_trip_and_validation():
    assert GpuKeepWarmPolicy.from_value(None).enabled is False
    assert GpuKeepWarmPolicy.from_value(False).enabled is False
    assert GpuKeepWarmPolicy.from_value(True).as_dict() == {
        "enabled": True, "window_seconds": 120.0, "interval_seconds": 0.5,
    }
    policy = GpuKeepWarmPolicy.from_value({"window_seconds": 30, "interval_seconds": 1})
    assert GpuKeepWarmPolicy.from_value(policy.as_dict()) == policy
    for bad, match in (
        ({"window_seconds": 0}, "positive"),
        ({"interval_seconds": 0.01}, "0.05 to 10"),
        ({"window_seconds": 0.1, "interval_seconds": 0.5}, "at least interval"),
        ({"interval_seconds": True}, "number of seconds"),
        ({"every": 1}, "unknown"),
        ({"enabled": "yes"}, "boolean"),
        ("on", "boolean or an object"),
    ):
        with pytest.raises(ValueError, match=match):
            GpuKeepWarmPolicy.from_value(bad)


def test_disabled_never_ticks_and_adds_no_counters():
    counts = Counter()
    keeper = GpuKeepWarm(None, counts, op=lambda: pytest.fail("ticked"))
    keeper.note_work(0.0)
    assert keeper.maybe_tick(1.0) is False
    assert not any(key in counts for key in COUNTERS)


def test_no_tick_before_first_request():
    keeper, clock, ops, counts = _keeper()
    _idle(keeper, clock, 105.0)
    assert ops == [] and counts["gpu_keep_warm_ticks"] == 0
    assert set(COUNTERS) <= set(counts)  # present (zero) once enabled


def test_ticks_every_interval_after_work_then_stop_at_window_end():
    keeper, clock, ops, counts = _keeper(window=3.0, interval=0.5)
    keeper.note_work()
    _idle(keeper, clock, 106.0)
    # The first tick is due one interval after the work; the last fits the
    # 3 s window; nothing after it expires.
    assert ops == pytest.approx([100.5, 101.0, 101.5, 102.0, 102.5, 103.0])
    assert counts["gpu_keep_warm_ticks"] == 6
    assert counts["gpu_keep_warm_idle_windows"] == 1
    assert counts["gpu_keep_warm_expired_windows"] == 1


def test_real_work_suppresses_ticks_and_counts_a_warm_resume():
    keeper, clock, ops, counts = _keeper(window=10.0, interval=0.5)
    keeper.note_work()
    _idle(keeper, clock, 101.2)
    assert len(ops) == 2
    # A request runs for 2 s: the worker calls note_work every round and
    # never maybe_tick, so no tick overlaps it.
    while clock.now < 103.2:
        clock.now = round(clock.now + 0.05, 6)
        keeper.note_work()
    assert len(ops) == 2
    assert counts["gpu_keep_warm_warm_resumes"] == 1
    _idle(keeper, clock, 103.75)
    assert ops[-1] == pytest.approx(103.7)  # one interval after the work
    assert counts["gpu_keep_warm_idle_windows"] == 2


def test_pending_request_skips_the_tick():
    keeper, clock, ops, _ = _keeper()
    keeper.note_work()
    _idle(keeper, clock, 102.0, pending_work=True)
    assert ops == []


def test_failing_op_disables_and_is_counted():
    def boom():
        raise RuntimeError("no device")

    keeper, clock, _, counts = _keeper(op=boom)
    keeper.note_work()
    _idle(keeper, clock, 103.0)
    assert counts["gpu_keep_warm_failures"] == 1
    assert counts["gpu_keep_warm_ticks"] == 0
    assert keeper.status()["failed"] is True and keeper.status()["active"] is False


def test_cli_maps_to_engine_policy_and_is_provenance_only():
    from mlx2.server import build_parser, serving_engine_kwargs
    from mlx2.serving import ServingEngine

    assert "gpu_keep_warm" in PROVENANCE_ONLY_SETTINGS
    parser = build_parser()
    default = parser.parse_args(["--model", "m"])
    on = parser.parse_args(
        ["--model", "m", "--gpu-keep-warm-seconds", "120", "--gpu-keep-warm-interval", "1"]
    )
    kw = dict(native_mtp=False, approximate_kv=None, max_request_bytes=1 << 20)
    assert serving_engine_kwargs(default, None, **kw)["gpu_keep_warm"] is None
    value = serving_engine_kwargs(on, None, **kw)["gpu_keep_warm"]
    assert GpuKeepWarmPolicy.from_value(value).as_dict() == {
        "enabled": True, "window_seconds": 120.0, "interval_seconds": 1.0,
    }
    with pytest.raises(ValueError, match="positive"):
        ServingEngine.validate_arguments(
            "model", gpu_keep_warm={"window_seconds": -1.0}
        )


def test_metrics_series_exist_only_when_enabled():
    from mlx2.prometheus import render_engine_metrics
    from test_prometheus import FakeEngine

    engine = FakeEngine()
    assert 'component="gpu_keep_warm"' not in render_engine_metrics(engine)
    GpuKeepWarm(True, engine.counts)
    engine.counts["gpu_keep_warm_ticks"] += 3
    text = render_engine_metrics(engine)
    assert (
        'mlx2_runtime_events_total{component="gpu_keep_warm",event="tick"} 3' in text
    )


@pytest.fixture
def engine(monkeypatch):
    import route_harness as H

    H.patch_host(monkeypatch)
    model, vocab = H.tiny_qwen38_mtp()
    engine = H.make_engine(
        model, vocab, mtp=False,
        gpu_keep_warm={"window_seconds": 5.0, "interval_seconds": 0.05},
    )
    yield engine
    engine.close()


def test_engine_ticks_while_idle_after_a_request(engine):
    import route_harness as H

    status = engine.status()
    assert status["settings"]["gpu_keep_warm"]["window_seconds"] == 5.0
    assert status["gpu_keep_warm"]["counts"]["tick"] == 0
    result = H.run(engine, {"tokens": [1, 2, 3], "max_tokens": 2, "temperature": 0})
    assert "error" not in result, result
    deadline = time.monotonic() + 10
    while engine.status()["counts"]["gpu_keep_warm_ticks"] < 3:
        assert time.monotonic() < deadline, engine.status()["gpu_keep_warm"]
        time.sleep(0.05)
    result = H.run(engine, {"tokens": [1, 2, 3], "max_tokens": 2, "temperature": 0})
    assert "error" not in result, result
    assert engine.status()["counts"]["gpu_keep_warm_warm_resumes"] == 1
