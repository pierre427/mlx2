"""Admission must not charge Metal pages that MLX has already freed.

GPU finding (2026-10-02, Flash-Next 4 x 32K MTP cohort): ``mx.clear_cache()``
returns at once, but macOS drops the freed buffer pages from the process's
physical footprint 70-250 ms later.  Every prefill chunk ends with a clear, so
the cohort's chunk re-check read 5-9 GiB of such pages as in use and refused
(HTTP 429) a cohort that fit.  The synchronized reclaim before a refusal now
waits, bounded, for those pages; it never credits bytes.
"""

from collections import Counter
import threading
import time

import mlx.core as mx
import pytest

from mlx2.memory import FootprintSettler, available_execution_bytes
from mlx2.runtime import os_memory
from mlx2.runtime.memory_policy import (
    SelfMTPLaneAdmissionController,
    _make_self_mtp_admission_callback,
)
from mlx2.serving import ServingEngine

G = 1 << 30


class DeferredMetal:
    """MLX counters plus a footprint that lags a just-issued clear_cache."""

    def __init__(self, *, active, base, pending, release_after):
        self.active, self.cache, self.base = active, 0, base
        self.pending, self.release_at = pending, time.monotonic() + release_after
        self.clears = 0

    def footprint(self):
        lagging = self.pending if time.monotonic() < self.release_at else 0
        return self.active + self.cache + self.base + lagging

    def clear_cache(self):
        self.clears += 1
        self.cache = 0


@pytest.fixture
def metal(monkeypatch):
    def install(**kw):
        fake = DeferredMetal(**kw)
        monkeypatch.setattr(mx, "synchronize", lambda *a, **k: None)
        monkeypatch.setattr(mx, "clear_cache", fake.clear_cache)
        monkeypatch.setattr(mx, "get_active_memory", lambda: fake.active)
        monkeypatch.setattr(mx, "get_cache_memory", lambda: fake.cache)
        monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda pid=None: fake.footprint())
        return fake
    return install


def test_cohort_chunk_recheck_admits_once_freed_pages_leave_the_footprint(metal):
    controller = SelfMTPLaneAdmissionController(
        host_memory_gib=128, advisory_gib=112,
        saturation_lane_cap=4, verification_row_cap=12,
    )
    context = 32768
    need = controller.hard_reserve_gib + 4 * controller.lane_gib(context, 2)
    recommended = 112 * G
    base = 1 * G
    # Settled: 1 GiB more free than the cohort needs.  Lagging: 4 GiB of pages
    # the last prefill chunk's clear_cache already returned to Metal.
    active = int(recommended - base - (need + 1.0) * G)
    fake = metal(active=active, base=base, pending=4 * G, release_after=0.15)

    def free_gib():
        return available_execution_bytes(
            available=200 * G, recommended=recommended, active=mx.get_active_memory(),
            cached=mx.get_cache_memory(), footprint=os_memory.physical_footprint_bytes(),
        ) / G

    assert free_gib() < need  # the inflated reading alone would refuse
    engine = object.__new__(ServingEngine)
    engine.counts = Counter()
    engine._memory_reclaim_lock = threading.Lock()
    engine._memory_reclaim_last = 0.0
    admit = _make_self_mtp_admission_callback(
        controller, free_memory=free_gib,
        reclaim_memory=lambda: engine._clear_allocator_cache_before_reject(synchronize=True),
        settle_memory=engine._settle_footprint_before_reject,
        evict_unused_cache=lambda: False, max_draft=2,
    )
    rows = tuple((uid, context, 2, True, 0.0) for uid in range(4))
    decision = admit.atomic(rows)
    assert fake.clears >= 1
    assert decision == {uid: 2 for uid in range(4)}, decision
    assert engine.counts["memory_footprint_settles"] >= 1


def test_settled_cohort_is_still_refused_when_memory_is_really_held(metal):
    """No credit: a footprint that does not fall keeps refusing."""
    controller = SelfMTPLaneAdmissionController(
        host_memory_gib=128, advisory_gib=112,
        saturation_lane_cap=4, verification_row_cap=12,
    )
    need = controller.hard_reserve_gib + 4 * controller.lane_gib(32768, 2)
    recommended = 112 * G
    # The extra 4 GiB is real non-MLX memory: it never leaves.
    active = int(recommended - G - (need + 1.0) * G)
    metal(active=active, base=5 * G, pending=0, release_after=0)
    engine = object.__new__(ServingEngine)
    engine.counts = Counter()
    engine._footprint_settler = FootprintSettler(quiet=0.05, timeout=0.2)
    admit = _make_self_mtp_admission_callback(
        controller,
        free_memory=lambda: available_execution_bytes(
            available=200 * G, recommended=recommended, active=mx.get_active_memory(),
            cached=mx.get_cache_memory(), footprint=os_memory.physical_footprint_bytes(),
        ) / G,
        reclaim_memory=lambda: engine._clear_allocator_cache_before_reject(synchronize=True),
        settle_memory=engine._settle_footprint_before_reject,
        evict_unused_cache=lambda: False, max_draft=2,
    )
    decision = admit.atomic(tuple((uid, 32768, 2, True, 0.0) for uid in range(4)))
    assert decision != {uid: 2 for uid in range(4)}
    assert engine.counts["memory_footprint_settle_quiet"] >= 1


def test_reading_that_fits_after_the_clear_never_waits():
    calls = []
    admit = _make_self_mtp_admission_callback(
        SelfMTPLaneAdmissionController(host_memory_gib=128, advisory_gib=112),
        free_memory=iter([0.0, 80.0, 80.0]).__next__,
        reclaim_memory=lambda: calls.append("reclaim"),
        settle_memory=lambda timeout=None: calls.append("settle"),
        evict_unused_cache=lambda: False, max_draft=2,
    )
    assert admit.atomic(((0, 4096, 2, True, 0.0),)) == {0: 2}
    assert calls == ["reclaim"]


def test_request_admission_settles_before_evicting_a_checkpoint():
    from mlx2.serving import ensure_admission_headroom

    state = {"free": 5, "lagging": 6, "evictions": 0, "settles": 0}

    def settle(timeout=None):
        state["settles"] += 1
        state["free"] += state["lagging"]
        state["lagging"] = 0
        return {"exit": "settled"}

    def evict():
        state["evictions"] += 1
        return True

    assert ensure_admission_headroom(
        10, headroom=lambda: state["free"], reclaim=lambda: None,
        evict=evict, settle=settle,
    )
    assert state == {"free": 11, "lagging": 0, "evictions": 0, "settles": 1}


class Clock:
    def __init__(self):
        self.t, self.sleeps = 0.0, 0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.sleeps += 1
        self.t += dt


def reader(clock, curve):
    """``curve``: [(from_time, footprint)], MLX active+cache fixed at 10 GiB."""
    def read():
        fp = [f for (t, f) in curve if t <= clock.t][-1]
        return (10 * G, 0, fp)
    return read


def test_settler_does_not_wait_when_nothing_is_pending():
    clock = Clock()
    s = FootprintSettler(read=reader(clock, [(0, 11 * G)]), clock=clock, sleep=clock.sleep)
    assert s.observe() == G
    report = s.settle()
    assert report["exit"] == "settled" and clock.sleeps == 0


def test_settler_waits_through_a_flat_lag_until_pages_return():
    clock = Clock()
    # Flat for 150 ms, then 8 GiB leave (the measured Metal shape).
    s = FootprintSettler(read=reader(clock, [(0, 19 * G), (0.15, 11 * G)]),
                         clock=clock, sleep=clock.sleep)
    s._note((10 * G, 0, 11 * G), 0.0, verified=True)
    report = s.settle()
    assert report["exit"] == "settled"
    assert 0.15 <= report["waited"] < 0.2
    assert report["released_bytes"] == 8 * G


def test_settler_learns_baseline_from_quiet_and_is_bounded():
    clock = Clock()
    s = FootprintSettler(read=reader(clock, [(0, 14 * G), (0.1, 12 * G)]),
                         clock=clock, sleep=clock.sleep, quiet=0.3, timeout=1.0)
    report = s.settle()  # no baseline yet: only quiet / timeout exits
    assert report["exit"] == "quiet" and report["waited"] < 0.45
    assert s.baseline == 2 * G
    # A footprint that keeps falling slowly is cut off at the timeout.
    clock2 = Clock()
    steps = [(i * 0.05, (40 - i) * G) for i in range(40)]
    s2 = FootprintSettler(read=reader(clock2, steps), clock=clock2, sleep=clock2.sleep,
                          timeout=1.0)
    s2._note((10 * G, 0, 10 * G), 0.0, verified=True)
    report = s2.settle()
    assert report["exit"] == "timeout" and report["waited"] <= 1.02


# --- review round (codex, 2026-10-02) -------------------------------------


def slow_settle(per_call, waits):
    """A settle whose pages each take ``per_call`` seconds to come back."""
    def settle(timeout=None):
        wait = per_call if timeout is None else min(per_call, timeout)
        waits.append(wait)
        time.sleep(wait)
        return {"exit": "settled", "waited": wait}
    return settle


def test_settle_budget_is_shared_across_evictions_in_mtp_admission():
    """32 checkpoints x 150 ms of lagging pages must not block ~5 s."""
    entries = {"left": 32}

    def evict():
        if not entries["left"]:
            return False
        entries["left"] -= 1
        return True

    waits = []
    admit = _make_self_mtp_admission_callback(
        SelfMTPLaneAdmissionController(host_memory_gib=128, advisory_gib=112),
        free_memory=lambda: 0.0,
        reclaim_memory=lambda: None,
        settle_memory=slow_settle(0.15, waits),
        evict_unused_cache=evict, max_draft=2,
    )
    started = time.monotonic()
    admit.atomic(((0, 32768, 2, False, 0.0),))
    elapsed = time.monotonic() - started
    assert elapsed < 1.4, elapsed
    assert sum(waits) <= 1.05


def test_settle_budget_is_shared_across_evictions_and_depth_fallback():
    from mlx2.serving import admit_lane_headroom

    entries = {"left": 32}

    def evict():
        if not entries["left"]:
            return False
        entries["left"] -= 1
        return True

    waits = []
    started = time.monotonic()
    (admitted, _floor, _req) = admit_lane_headroom(
        SelfMTPLaneAdmissionController(host_memory_gib=128, advisory_gib=112),
        context_tokens=32768, draft_depth=2, cache_gib=0.0,
        headroom=lambda: 0, reclaim=lambda: None, evict=evict,
        settle=slow_settle(0.15, waits),
    )
    elapsed = time.monotonic() - started
    assert not admitted
    assert elapsed < 1.4, elapsed
    assert sum(waits) <= 1.05


class Host:
    """Fake readings: MLX holds 10 GiB; non-MLX residency and lagging pages vary."""

    def __init__(self, clock):
        self.clock, self.non_mlx, self.pending, self.release_at = clock, G, 0, 0.0

    def read(self):
        lagging = self.pending if self.clock.t < self.release_at else 0
        return (10 * G, 0, 10 * G + self.non_mlx + lagging)

    def defer(self, nbytes, seconds):
        self.pending, self.release_at = nbytes, self.clock.t + seconds


def test_stale_high_baseline_does_not_certify_a_deferred_release():
    """Learned 9 GiB quiet overhang, residency later drops to 1 GiB, then a
    clear leaves 8 GiB lagging: 1 + 8 matches the stale 9 and must not pass
    as settled without waiting."""
    clock = Clock()
    host = Host(clock)
    s = FootprintSettler(read=host.read, clock=clock, sleep=clock.sleep)
    host.non_mlx = 9 * G
    s.observe()
    clock.t = 30.0  # long after; no reading in between
    host.non_mlx = G
    host.defer(8 * G, 0.15)
    report = s.settle()
    assert report["waited"] >= 0.15, report
    assert report["released_bytes"] == 8 * G


def test_baseline_tracks_growth_and_shrink_and_still_waits_for_a_release():
    clock = Clock()
    host = Host(clock)
    s = FootprintSettler(read=host.read, clock=clock, sleep=clock.sleep, window=10.0)
    assert s.refresh(idle=True) is None  # one idle reading verifies nothing
    clock.t += 1.0
    assert s.refresh(idle=True) == G
    # Growth: non-MLX residency rises to 9 GiB and stays.  Once the old low
    # readings age out, nothing pending means no wait at all.
    host.non_mlx = 9 * G
    for _ in range(12):
        clock.t += 1.0
        s.refresh(idle=True)
    assert s.baseline == 9 * G
    sleeps = clock.sleeps
    assert s.settle()["exit"] == "settled" and clock.sleeps == sleeps
    # Shrink: the next reading outside rejection lowers it at once.
    host.non_mlx = G
    clock.t += 1.0
    assert s.refresh(idle=True) == G
    # A deferred release now is waited for, not certified by the old 9 GiB.
    host.defer(8 * G, 0.15)
    report = s.settle()
    assert report["exit"] == "settled"
    assert 0.15 <= report["waited"] < 0.2


def test_busy_loop_readings_alone_never_form_a_baseline():
    """GPU 2026-10-02: during a long prefill every loop reading lands right
    after a chunk's clear.  Those inflated readings must not certify the next
    chunk's lagging pages once the verified idle readings have aged out."""
    clock = Clock()
    host = Host(clock)
    s = FootprintSettler(read=host.read, clock=clock, sleep=clock.sleep, window=10.0)
    s.observe()  # 1 GiB at load
    for _ in range(15):  # busy: each reading 6 GiB of lag above residency
        clock.t += 1.0
        host.defer(6 * G, 0.15)
        s.refresh(idle=False)
    assert s.baseline is None
    host.defer(6 * G, 0.15)
    report = s.settle()
    assert report["waited"] >= 0.15 and report["released_bytes"] == 6 * G
    assert s.baseline == G  # the settle that waited verified it again


def test_refresh_never_blocks_while_another_thread_settles():
    clock = Clock()
    host = Host(clock)
    s = FootprintSettler(read=host.read, clock=clock, sleep=clock.sleep)
    s._lock.acquire()
    try:
        assert s.refresh() is None
        assert s.settle(timeout=0.01)["exit"] == "busy"
    finally:
        s._lock.release()


def test_parallel_sample_final_check_settles_before_refusing(metal, monkeypatch):
    from mlx2 import memory, serving

    recommended = 112 * G
    required = (20 + 2 * 2) * G  # reserve + 2 samples x 2 GiB default
    # Settled there is 1 GiB to spare; 4 GiB of freed pages still lag.
    fake = metal(active=int(recommended - G - required - G), base=G,
                 pending=4 * G, release_after=0.15)
    monkeypatch.setattr(memory, "execution_headroom", lambda: available_execution_bytes(
        available=200 * G, recommended=recommended, active=fake.active,
        cached=fake.cache, footprint=fake.footprint()))
    monkeypatch.setattr(serving.ServingEngine, "PARALLEL_SAMPLE_WAIT_SECONDS", 0.0)
    engine = serving.ServingEngine.__new__(serving.ServingEngine)
    engine.max_lanes = 4
    engine.queued_jobs = 0
    engine.counts = Counter()
    engine._hard_reserve_gib = 20.0
    engine.batch_metrics = type("M", (), {"rejected": lambda self, *a: None})()
    receipt = engine.admit_parallel_samples(2)
    assert receipt["required_headroom_bytes"] == required
    assert engine.counts["memory_footprint_settles"] == 1
