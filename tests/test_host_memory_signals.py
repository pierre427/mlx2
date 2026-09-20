"""Host memory signals (Splash item 9): estimate, pressure hysteresis, wiring."""

import platform
import re
import subprocess
from types import SimpleNamespace as NS

import pytest

from mlx2.runtime import os_memory
from mlx2.runtime.os_memory import (
    HostMemorySnapshot,
    MemoryPressureMonitor,
    PressureLevel,
    estimate_host_available_bytes,
    pressure_level_from_kernel,
)

darwin_only = pytest.mark.skipif(
    platform.system() != "Darwin", reason="Mach VM statistics are Darwin-only"
)

PAGE = 16384
GIB = 1 << 30


def _estimate(**pages):
    base = dict(
        page_size=PAGE,
        physical_bytes=128 * GIB,
        active=0,
        inactive=0,
        speculative=0,
        wired=0,
        compressor=0,
        file_backed=0,
        purgeable=0,
    )
    base.update(pages)
    return estimate_host_available_bytes(**base)


# -- formula -----------------------------------------------------------------


def test_estimate_matches_splash_formula():
    used = dict(
        active=500_000,
        inactive=450_000,
        speculative=60_000,
        wired=450_000,
        compressor=260_000,
        file_backed=300_000,
        purgeable=2_000,
    )
    expected = 128 * GIB - (
        500_000 + 450_000 + 60_000 + 450_000 + 260_000 - 300_000 - 2_000
    ) * PAGE
    assert _estimate(**used) == expected


def test_estimate_counts_file_backed_and_purgeable_as_available():
    held = _estimate(active=100_000, inactive=100_000)
    assert _estimate(active=100_000, inactive=100_000, file_backed=50_000) == (
        held + 50_000 * PAGE
    )
    assert _estimate(active=100_000, inactive=100_000, purgeable=10_000) == (
        held + 10_000 * PAGE
    )


def test_estimate_fails_to_zero_on_inconsistent_terms():
    # Splash returns 0 rather than wrapping when a subtraction would underflow.
    assert _estimate(active=10, file_backed=11) == 0
    assert _estimate(active=10, file_backed=5, purgeable=6) == 0
    assert _estimate(wired=(128 * GIB) // PAGE) == 0
    assert _estimate(wired=(256 * GIB) // PAGE) == 0
    assert _estimate(page_size=0) == 0
    assert _estimate(physical_bytes=0) == 0


def test_reclaimable_keeps_vm_stat_free_inactive_speculative_semantics():
    # Mach free_count already includes speculative pages; vm_stat prints
    # "Pages free" as free - speculative, so the old sum is free + inactive.
    snap = HostMemorySnapshot(
        page_size=PAGE,
        physical_bytes=128 * GIB,
        free=1000,
        active=1,
        inactive=200,
        speculative=30,
        wired=1,
        compressor=1,
        file_backed=0,
        purgeable=0,
    )
    vm_stat_free = snap.free - snap.speculative
    assert snap.reclaimable_bytes == (vm_stat_free + snap.inactive + snap.speculative) * PAGE


def test_kernel_pressure_mapping():
    assert pressure_level_from_kernel(0) is PressureLevel.NORMAL
    assert pressure_level_from_kernel(1) is PressureLevel.NORMAL
    assert pressure_level_from_kernel(2) is PressureLevel.WARN
    assert pressure_level_from_kernel(4) is PressureLevel.CRITICAL


# -- hysteresis ----------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _monitor(readings, fall=5.0):
    clock = FakeClock()
    feed = iter(readings)
    monitor = MemoryPressureMonitor(fall, clock, reader=lambda: next(feed))
    return monitor, clock


def test_rise_is_immediate_and_fall_waits_for_stable_window():
    N, C = PressureLevel.NORMAL, PressureLevel.CRITICAL
    monitor, clock = _monitor([N, C, N, N, N, N])
    assert monitor.level() is N
    assert monitor.level() is C  # rise on the first reading
    assert monitor.level() is C  # window opens at t=100
    clock.now = 104.9
    assert monitor.level() is C
    clock.now = 105.0
    assert monitor.level() is N  # 5 s of stable lower readings
    assert monitor.level() is N


def test_fall_lands_on_highest_level_seen_in_window_then_needs_new_window():
    N, W, C = PressureLevel.NORMAL, PressureLevel.WARN, PressureLevel.CRITICAL
    monitor, clock = _monitor([C, N, W, N, N, N])
    assert monitor.level() is C
    assert monitor.level() is C  # window opens at t=100 on NORMAL
    clock.now = 102.0
    assert monitor.level() is C  # WARN inside the window
    clock.now = 105.0
    assert monitor.level() is W  # falls only to the window's peak
    clock.now = 106.0
    assert monitor.level() is W  # a further fall needs its own window
    clock.now = 111.0
    assert monitor.level() is N


def test_rebound_resets_fall_window():
    N, C = PressureLevel.NORMAL, PressureLevel.CRITICAL
    monitor, clock = _monitor([C, N, C, N, N])
    assert monitor.level() is C
    assert monitor.level() is C  # window opens at t=100
    clock.now = 104.0
    assert monitor.level() is C  # back at CRITICAL resets the window
    clock.now = 106.0
    assert monitor.level() is C  # window reopens at t=106
    clock.now = 110.0
    assert monitor.level() is C


def test_unavailable_reading_keeps_reported_level():
    C = PressureLevel.CRITICAL
    monitor, clock = _monitor([C, None, None])
    assert monitor.level() is C
    clock.now = 1000.0
    assert monitor.level() is C
    assert monitor.raw is None
    assert monitor.level() is C


def test_zero_window_falls_immediately_and_negative_is_rejected():
    N, W = PressureLevel.NORMAL, PressureLevel.WARN
    monitor, _ = _monitor([W, N], fall=0.0)
    assert monitor.level() is W
    assert monitor.level() is N
    with pytest.raises(ValueError):
        MemoryPressureMonitor(-1.0)


# -- platform behaviour --------------------------------------------------------


def test_probes_return_none_without_libsystem(monkeypatch):
    # Equivalent to non-Darwin: _load_libsystem() returns None off Darwin.
    monkeypatch.setattr(os_memory, "_LIBSYSTEM", None)
    assert os_memory.host_memory_snapshot() is None
    assert os_memory.host_pressure_level() is None


@pytest.mark.skipif(platform.system() == "Darwin", reason="non-Darwin behaviour")
def test_non_darwin_has_no_host_signals():
    assert os_memory._load_libsystem() is None
    assert os_memory.host_memory_snapshot() is None
    assert os_memory.host_pressure_level() is None


def _vm_stat():
    out = subprocess.run(
        ["/usr/bin/vm_stat"], check=True, capture_output=True, text=True
    ).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    counts = {}
    for line in out.splitlines()[1:]:
        if ":" in line:
            name, value = line.split(":", 1)
            counts[name.strip().strip('"')] = int(value.strip().rstrip("."))
    return page, counts


def _sysctl(name):
    return int(
        subprocess.run(
            ["/usr/sbin/sysctl", "-n", name], check=True, capture_output=True, text=True
        ).stdout.strip()
    )


@darwin_only
def test_live_snapshot_matches_vm_stat():
    snap = os_memory.host_memory_snapshot()
    page, counts = _vm_stat()
    assert snap is not None
    assert snap.page_size == page
    assert snap.physical_bytes == _sysctl("hw.memsize")
    # Counters move between the two reads; allow 1% of RAM of drift.
    tolerance = snap.physical_bytes // 100
    old_sum = (
        counts["Pages free"] + counts["Pages inactive"] + counts["Pages speculative"]
    ) * page
    assert abs(snap.reclaimable_bytes - old_sum) <= tolerance
    splash = estimate_host_available_bytes(
        page_size=page,
        physical_bytes=snap.physical_bytes,
        active=counts["Pages active"],
        inactive=counts["Pages inactive"],
        speculative=counts["Pages speculative"],
        wired=counts["Pages wired down"],
        compressor=counts["Pages occupied by compressor"],
        file_backed=counts["File-backed pages"],
        purgeable=counts["Pages purgeable"],
    )
    assert abs(snap.available_bytes - splash) <= tolerance
    assert 0 < snap.available_bytes <= snap.physical_bytes


@darwin_only
def test_live_pressure_level_matches_sysctl():
    # Read twice around the CLI and accept either to tolerate a transition.
    before = os_memory.host_pressure_level()
    kernel = _sysctl("kern.memorystatus_vm_pressure_level")
    after = os_memory.host_pressure_level()
    assert pressure_level_from_kernel(kernel) in {before, after}


@darwin_only
def test_system_available_memory_uses_ctypes_not_subprocess(monkeypatch):
    from mlx2.runtime import memory_policy

    def refuse(*_args, **_kwargs):
        raise AssertionError("vm_stat subprocess must not run")

    monkeypatch.setattr(subprocess, "run", refuse)
    available = memory_policy._system_available_memory_bytes()
    assert available is not None and available > 0


# -- admission term -------------------------------------------------------------


def test_host_available_bytes_default_is_psutil_and_opt_in_uses_estimate(monkeypatch):
    import psutil

    from mlx2 import memory

    monkeypatch.setattr(psutil, "virtual_memory", lambda: NS(available=7 * GIB))
    snap = HostMemorySnapshot(
        page_size=PAGE,
        physical_bytes=128 * GIB,
        free=0,
        active=(64 * GIB) // PAGE,
        inactive=0,
        speculative=0,
        wired=0,
        compressor=0,
        file_backed=0,
        purgeable=0,
    )
    monkeypatch.setattr(os_memory, "host_memory_snapshot", lambda: snap)
    assert memory.host_available_bytes() == 7 * GIB
    assert memory.host_available_bytes(True) == 64 * GIB
    monkeypatch.setattr(os_memory, "host_memory_snapshot", lambda: None)
    assert memory.host_available_bytes(True) == 7 * GIB


# -- serving policy ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ([], "must be an object"),
        ({"mystery": 1}, "unknown host memory"),
        ({"enabled": 1}, "must be boolean"),
        ({"fall_after_seconds": -1}, "fall_after_seconds"),
        ({"fall_after_seconds": True}, "fall_after_seconds"),
        ({"fall_after_seconds": float("nan")}, "fall_after_seconds"),
    ],
)
def test_policy_rejects_invalid_values(value, error):
    from mlx2.serving import host_memory_signals_policy

    with pytest.raises(ValueError, match=error):
        host_memory_signals_policy(value)


def test_policy_defaults_off_and_engine_level_is_normal():
    from mlx2.serving import ServingEngine, host_memory_signals_policy

    assert host_memory_signals_policy(None) == {
        "enabled": False,
        "fall_after_seconds": 5.0,
    }
    with pytest.raises(ValueError, match="unknown host memory"):
        ServingEngine("unused", execution_policy={"host_memory_signals": {"x": 1}})


def test_required_feature_check_only_when_enabled():
    from mlx2.qualification import required_feature_checks

    base = {
        "mtp": False,
        "speculation": None,
        "execution_policy": {},
        "environment": {},
        "max_context": 1024,
    }
    assert "feature_host_memory_signals" not in required_feature_checks(base)
    assert "feature_host_memory_signals" in required_feature_checks(
        {**base, "host_memory_signals": {"enabled": True, "fall_after_seconds": 5.0}}
    )


def _run_engine(monkeypatch, execution_policy):
    from mlx2 import memory, serving
    from mlx2.runtime import apc_v2, generate
    from mlx2.serving import ServingEngine

    calls = []
    adapter_policies = []

    class APC:
        def __init__(self, **_kwargs):
            self.apc_stats = {}

        def key(self, *_args, **_kwargs):
            return "key"

        def spill_idle_entries(self):
            pass

        def clear(self):
            pass

    class Batch:
        def __init__(self, *_args, **_kwargs):
            self.scheduler_stats = {}
            self.lanes = {}

        def close(self):
            pass

    class Adapter:
        max_context = 64
        layout = "fake-layout"
        model = None
        tokenizer = NS(vocab_size=32, eos_token_ids=[])

        def __init__(self, _path, execution_policy=None):
            adapter_policies.append(execution_policy)
            self.identity = {"fingerprint": "fake"}
            self.environment = {}

        def profile_name(self, _mtp):
            return "fake"

        def execution_config(self, **_kwargs):
            return {"num_draft": 0}

        def diagnostics(self):
            return {}

        def close(self):
            pass

    def headroom(**kwargs):
        calls.append(kwargs)
        return 100 * GIB

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "s"})
    monkeypatch.setattr(memory, "execution_headroom", headroom)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(apc_v2, "APCv2", APC)
    monkeypatch.setattr(generate, "BatchGenerator", Batch)
    engine = ServingEngine(
        "fake",
        adapter_factory=Adapter,
        qualification_mode=True,
        mtp=False,
        max_lanes=2,
        max_inflight=4,
        execution_policy=execution_policy,
    )
    return engine, calls, adapter_policies


def test_disabled_engine_is_unchanged(monkeypatch):
    engine, calls, adapter_policies = _run_engine(monkeypatch, None)
    try:
        assert engine.ready.wait(5)
        assert engine.error is None
        assert "host_memory_signals" not in engine.snapshot["settings"]
        assert "host_memory_pressure_level" not in engine.snapshot
        assert "host_memory_available_bytes" not in engine.snapshot
        assert calls and all(call == {} for call in calls)
        assert adapter_policies == [None]
        assert engine.memory_pressure_level() is PressureLevel.NORMAL
    finally:
        engine.close()


def test_enabled_engine_strips_policy_uses_host_signals_and_exports(monkeypatch):
    from mlx2.prometheus import render_engine_metrics

    levels = iter([PressureLevel.WARN] + [PressureLevel.NORMAL] * 1000)
    monkeypatch.setattr(os_memory, "host_pressure_level", lambda: next(levels))
    snap = HostMemorySnapshot(
        page_size=PAGE,
        physical_bytes=128 * GIB,
        free=0,
        active=(32 * GIB) // PAGE,
        inactive=0,
        speculative=0,
        wired=0,
        compressor=0,
        file_backed=0,
        purgeable=0,
    )
    monkeypatch.setattr(os_memory, "host_memory_snapshot", lambda: snap)
    policy = {"enabled": True, "fall_after_seconds": 60.0}
    engine, calls, adapter_policies = _run_engine(
        monkeypatch, {"host_memory_signals": policy}
    )
    try:
        assert engine.ready.wait(5)
        assert engine.error is None
        # Server-owned: the adapter never sees the key.
        assert adapter_policies == [None]
        assert engine.snapshot["settings"]["host_memory_signals"] == policy
        assert calls and all(call == {"host_signals": True} for call in calls)
        assert engine.snapshot["host_memory_available_bytes"] == 96 * GIB
        # WARN held by 60 s fall hysteresis even though later reads are NORMAL.
        assert engine.snapshot["host_memory_pressure_level"] == int(PressureLevel.WARN)
        assert engine.memory_pressure_level() is PressureLevel.WARN
        rendered = render_engine_metrics(engine)
        assert "mlx2_host_memory_pressure_level 1" in rendered
        assert f"mlx2_host_memory_available_bytes {96 * GIB}" in rendered
    finally:
        engine.close()
