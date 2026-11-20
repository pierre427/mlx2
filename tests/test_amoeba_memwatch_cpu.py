"""CPU tests of research/amoeba/carve/memwatch.py (2026-11-19 FN battery panic): the trigger logic on the panic's
readings, the watcher's grace rule, and live stops in throwaway child processes: the in-process thread exits 3 with
a receipt, and the watcher process SIGKILLs a child whose GIL is held by a long C call (the thread cannot run)."""
import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.amoeba.carve import memwatch as M  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS kernel interfaces")

GIB_PAGE = 16384 / 2**30


def _panic_reading(**kw):
    """The 2026-11-19 17:26 panic report: 875 free pages, 0 speculative, 2,430,719 inactive, 44,696 purgeable,
    compressorSize 2,900,069 pages (read here as compressor-occupied pages, the panic report's memoryStatus field;
    the page counts then add up to the 128 GiB of the machine), swap ~1 GB, pressure not critical; the battery at
    80.8 GiB resident."""
    r = {"footprint_gib": 80.8, "avail_gib": (875 + 2430719 + 44696) * GIB_PAGE, "free_pages": 875,
         "free_min_pages": 3500, "page_free_wanted": 0, "compressor_gib": 2900069 * GIB_PAGE, "memsize_gib": 128.0,
         "swap_used_gib": 1.0, "pressure": 1}
    r.update(kw)
    return r


HEALTHY = dict(free_pages=60000, compressor_gib=3.5)       # this machine on 2026-11-19 with a 72 GB model serving


def _off(**kw):
    """Limits with the panic-specific triggers off, to test one at a time."""
    k = dict(compressor_growth_gb=1e9, compressor_frac=1e9, hard_compressor_frac=1e9, starved_soft_s=10**9,
             starved_hard_s=10**9)
    k.update(kw)
    return k


def test_old_checks_miss_the_panic_new_ones_fire():
    r = _panic_reading()
    # the old rules: footprint 92 (a resident 80.8 GiB under it), avail 15 GB (inactive counted: 38 GB), swap +8
    assert M.check(r, M.make_limits(92, min_avail_gb=15, swap_growth_gb=8, **_off())) == []
    # starvation: soft after 3 consecutive samples, hard after 10; one healthy sample resets the count
    lim = M.make_limits(92, min_avail_gb=15, swap_growth_gb=8, **_off(starved_soft_s=3, starved_hard_s=10))
    st = {}
    assert [M.check(r, lim, st) for _ in range(2)] == [[], []]
    assert [k for k, _ in M.check(r, lim, st)] == ["soft"]
    M.check(_panic_reading(**HEALTHY), lim, st)
    assert st["starved"] == 0
    kinds = [[k for k, _ in M.check(r, lim, st)] for _ in range(10)]
    assert kinds[1] == [] and kinds[2] == ["soft"] and kinds[9] == ["hard"]
    assert M.check(_panic_reading(**HEALTHY, page_free_wanted=5), lim, {"starved": 2})  # page waiters count too
    # the compressor's share of memory (44 GiB of 128 = 35 %): hard at 30 %, soft at 20 %
    lim = M.make_limits(92, **_off(compressor_frac=0.20, hard_compressor_frac=0.30))
    hits = M.check(r, lim)
    assert [k for k, _ in hits] == ["hard"] and "compressor holds" in hits[0][1]
    assert [k for k, _ in M.check(_panic_reading(compressor_gib=30.0), lim)] == ["soft"]
    assert M.check(_panic_reading(compressor_gib=20.0), lim) == []
    # a stale compressed pool (no growth since start) is no trigger: idle pages stay compressed after the pressure
    stale = M.make_limits(92, **_off(compressor_frac=0.20, hard_compressor_frac=0.30, compressor0=44.0))
    assert M.check(_panic_reading(**dict(HEALTHY, compressor_gib=44.0)), stale) == []
    assert M.check(r, lim, watcher=True) == []                 # the watcher: footprint and starvation only
    # compressor growth over the level at start (soft only: the thread's, rebased per carve)
    lim = M.make_limits(92, **_off(compressor_growth_gb=12, compressor0=4.0))
    assert [k for k, _ in M.check(r, lim)] == ["soft"]
    assert M.check(r, lim, watcher=True) == []                 # the watcher has no baseline-relative trigger
    # the defaults fire on the panic and stay quiet on the healthy machine
    d = M.make_limits(92, compressor0=3.5)
    assert any(k == "hard" for k, _ in M.check(r, d))
    assert M.check(_panic_reading(**HEALTHY, footprint_gib=75.0), d, {}) == []


def test_footprint_and_the_rest():
    lim = M.make_limits(92, **_off())
    assert lim["hard_gib"] == 98.0
    ok = dict(HEALTHY)
    assert M.check(_panic_reading(footprint_gib=91.9, **ok), lim) == []
    assert [k for k, _ in M.check(_panic_reading(footprint_gib=92.5, **ok), lim)] == ["soft"]
    assert [k for k, _ in M.check(_panic_reading(footprint_gib=98.0, **ok), lim)] == ["hard"]
    assert [k for k, _ in M.check(_panic_reading(footprint_gib=98.0, **ok), lim, watcher=True)] == ["hard"]
    assert M.check(_panic_reading(footprint_gib=None, **ok), lim) == []      # target gone: no footprint trigger
    lim = M.make_limits(92, min_avail_gb=40, **_off())
    assert any("free+inactive" in m for _, m in M.check(_panic_reading(**ok), lim))
    lim = M.make_limits(92, swap_growth_gb=0.5, swap0=0.2, **_off())
    assert any("swap grew" in m for _, m in M.check(_panic_reading(**ok), lim))
    assert M.check(_panic_reading(**ok), lim, watcher=True) == []
    assert any("critical" in m for _, m in M.check(_panic_reading(pressure=4, **ok), M.make_limits(92, **_off())))


def test_watcher_grace():
    lim = M.make_limits(10, **_off())
    soft = _panic_reading(footprint_gib=11.0, **HEALTHY)
    act, since, _ = M.watcher_step(soft, lim, None, 100.0, 5.0)
    assert act is None and since == 100.0
    act, since, _ = M.watcher_step(soft, lim, since, 104.0, 5.0)
    assert act is None and since == 100.0
    act, since, _ = M.watcher_step(soft, lim, since, 105.0, 5.0)
    assert act == "kill"
    act, since, _ = M.watcher_step(_panic_reading(footprint_gib=5.0, **HEALTHY), lim, 100.0, 106.0, 5.0)
    assert act is None and since is None                                     # cleared: the grace clock restarts
    act, _, _ = M.watcher_step(_panic_reading(footprint_gib=17.0, **HEALTHY), lim, None, 0.0, 5.0)
    assert act == "kill"                                                     # hard: at once


def test_live_readings_match_the_kernel():
    r = M.readings(os.getpid())
    assert 0 < r["footprint_gib"] < 64 and r["resident_gib"] > 0
    assert 0 <= r["free_pages"] * GIB_PAGE <= r["avail_gib"] and r["compressor_gib"] >= 0
    vm_free = int(next(x for x in subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True).stdout
                       .splitlines() if x.startswith("Pages free")).split(":")[1].strip(" ."))
    assert abs(r["free_pages"] - vm_free) < 200000            # vm_stat's "Pages free" excludes speculative too
    assert r["free_min_pages"] > 0 and 30 < r["memsize_gib"] < 4096
    vm = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True).stdout
    occ = int(next(x for x in vm.splitlines() if "occupied by compressor" in x).split(":")[1].strip(" ."))
    assert abs(occ * GIB_PAGE - r["compressor_gib"]) < 2.0                  # same counter, moments apart


def _child(tmp_path, body):
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(f"""
        import json, os, sys, time
        sys.path.insert(0, {str(ROOT)!r})
        from research.amoeba.carve import memwatch as M
        """) + textwrap.dedent(body))
    return subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60)


def test_thread_stops_cleanly_with_receipt(tmp_path):
    rc = tmp_path / "stop.json"
    p = _child(tmp_path, f"""
        lim = M.make_limits(0.0, hard_gb=1e6, compressor_growth_gb=1e9, compressor_frac=1e9,
                            hard_compressor_frac=1e9, starved_soft_s=10**9, starved_hard_s=10**9)
        M.Watch(lim, lambda m: print(m, flush=True), every_s=0.2, receipt={str(rc)!r}, watcher=False)
        time.sleep(20)
        print("not stopped")
        """)
    assert p.returncode == 3, (p.returncode, p.stdout, p.stderr)
    assert "not stopped" not in p.stdout and "memory watchdog: footprint" in p.stdout
    assert p.stdout.strip().splitlines()[-1].endswith(M.STOP_MARKER)    # the bridge lab keys its receipt on it
    rec = json.loads(rc.read_text())
    assert rec["exit"] == 3 and rec["triggers"][0][0] == "soft"


def test_watcher_kills_when_the_thread_cannot_run(tmp_path):
    """The thread and the watcher both see a soft trigger (any footprint > 0), but the thread starts only after the
    child already holds its GIL in one C call (sum over a huge range), so it never gets to run: the watcher process
    sees the soft trigger persist past its grace and SIGKILLs the child."""
    p = _child(tmp_path, f"""
        import threading
        lim = M.make_limits(0.0, hard_gb=1e6, compressor_growth_gb=1e9, compressor_frac=1e9,
                            hard_compressor_frac=1e9, starved_soft_s=10**9, starved_hard_s=10**9)
        orig = threading.Thread.start
        gate = threading.Event()
        def start_late(self):                      # the thread is created but blocked until the C call holds the GIL
            run = self.run
            self.run = lambda: (gate.wait(), run())
            orig(self)
        threading.Thread.start = start_late
        M.Watch(lim, print, every_s=0.2, grace_s=1.0, receipt={str(tmp_path / "t.json")!r}, watcher=True)
        gate.set()
        sum(range(10 ** 12))
        print("not stopped")
        """)
    assert p.returncode == -signal.SIGKILL, (p.returncode, p.stdout, p.stderr)
    rec = json.loads((tmp_path / "t.json.watcher.json").read_text())
    assert rec["stopped_by"].startswith("watcher process") and rec["triggers"][0][0] == "soft"


def test_watcher_hard_kill_and_exit_with_target(tmp_path):
    rc = tmp_path / "w.json"
    p = _child(tmp_path, f"""
        lim = M.make_limits(1e6, hard_gb=0.0, compressor_frac=1e9, hard_compressor_frac=1e9,
                            starved_soft_s=10**9, starved_hard_s=10**9)     # hard ceiling 0: kill at once
        M.spawn_watcher(os.getpid(), lim, every_s=0.2, grace_s=60.0, receipt={str(rc)!r})
        time.sleep(20)
        """)
    assert p.returncode == -signal.SIGKILL
    assert json.loads(rc.read_text())["triggers"][0][0] == "hard"
    # a watcher whose target exits on its own leaves on its own (no receipt, no kill)
    rc2 = tmp_path / "w2.json"
    p = _child(tmp_path, f"""
        w = M.spawn_watcher(os.getpid(), M.make_limits(1e6, compressor_frac=1e9, hard_compressor_frac=1e9,
                            starved_soft_s=10**9, starved_hard_s=10**9), every_s=0.2,
                            receipt={str(rc2)!r})
        print(w.pid, flush=True)
        """)
    assert p.returncode == 0 and not rc2.exists()
    wpid = int(p.stdout.split()[0])
    import time
    for _ in range(50):
        try:
            os.kill(wpid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("watcher outlived its target")


# 2026-11-19 19:19 stop, then wiring (ops.wire_weights): the battery's ~70 GiB of weights wired, so the machine holds
# ~86 GiB wired of 128 and only ~40 GiB is pageable. The triggers read only pageable room and absolute levels.
WIRED_HEALTHY = dict(footprint_gib=79.1, avail_gib=34.0, free_pages=60000, free_min_pages=3500, page_free_wanted=0,
                     compressor_gib=4.2, memsize_gib=128.0, swap_used_gib=0.02, pressure=1, wired_gib=86.0)


def test_high_wired_memory_is_no_trigger_but_pressure_still_stops():
    lim = M.make_limits(92, min_avail_gb=15, swap_growth_gb=8, compressor0=4.2, swap0=0.02)
    st = {}
    assert [M.check(dict(WIRED_HEALTHY), lim, st) for _ in range(20)] == [[]] * 20      # wired alone: no stop
    assert M.check(dict(WIRED_HEALTHY), lim, {}, watcher=True) == []
    # pressure on the ~40 GiB left: other processes compressed (+13 GiB), avail down, free pages starving
    grown = dict(WIRED_HEALTHY, compressor_gib=17.5, avail_gib=20.0)
    assert any("compressor grew" in m for _, m in M.check(grown, lim, {}))
    starved = dict(WIRED_HEALTHY, free_pages=1200, avail_gib=16.0)
    st = {}
    kinds = [[k for k, _ in M.check(starved, lim, st)] for _ in range(10)]
    assert kinds[2] == ["soft"] and kinds[9] == ["hard"]
    st = {}
    assert [[k for k, _ in M.check(starved, lim, st, watcher=True)] for _ in range(10)][9] == ["hard"]
    low = dict(WIRED_HEALTHY, avail_gib=12.0)
    assert any("free+inactive" in m for _, m in M.check(low, lim, {}))


def test_live_reading_has_machine_wired():
    r = M.readings(os.getpid())
    vm = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True).stdout
    wired = int(next(x for x in vm.splitlines() if x.startswith("Pages wired down")).split(":")[1].strip(" ."))
    assert 0 < r["wired_gib"] < r["memsize_gib"] and abs(wired * GIB_PAGE - r["wired_gib"]) < 2.0


def test_watch_tracks_wired_after_wiring_and_receipt_has_it(tmp_path):
    rc = tmp_path / "stop.json"
    p = _child(tmp_path, f"""
        lim = M.make_limits(0.0, hard_gb=1e6, compressor_growth_gb=1e9, compressor_frac=1e9,
                            hard_compressor_frac=1e9, starved_soft_s=10**9, starved_hard_s=10**9)
        stats = {{"wired": {{"wired": True, "limit_gib": 80.0}}, "wired_at_wire_gb": 86.0}}
        M.Watch(lim, lambda m: print(m, flush=True), every_s=0.2, receipt={str(rc)!r}, watcher=False, stats=stats)
        time.sleep(20)
        """)
    assert p.returncode == 3, (p.returncode, p.stdout, p.stderr)
    rec = json.loads(rc.read_text())
    assert rec["wired"] == {"wired": True, "limit_gib": 80.0} and rec["observed_min_wired_gb"] > 0
    assert rec["reading"]["wired_gib"] > 0


def test_watch_logs_once_when_wired_drops(tmp_path):
    p = _child(tmp_path, """
        lim = M.make_limits(1e6, hard_gb=1e7, min_avail_gb=0, compressor_growth_gb=1e9, compressor_frac=1e9,
                            hard_compressor_frac=1e9, starved_soft_s=10**9, starved_hard_s=10**9, swap_growth_gb=1e9)
        stats = {"wired_at_wire_gb": 4000.0}          # far above anything real: the reading is a drop
        M.Watch(lim, lambda m: print(m, flush=True), every_s=0.1, watcher=False, stats=stats)
        time.sleep(1.0)
        print("warned", stats.get("observed_wired_drop_at") is not None, flush=True)
        """)
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert p.stdout.count("may be pageable again") == 1 and "warned True" in p.stdout
