"""Memory watchdog without subprocesses (2026-11-19 FN battery panic: the old watchdog forked top / vm_stat /
sysctl every 15 s, counted inactive pages as available and ran as a Python thread, so it never stopped a process
at 81 GiB resident plus ~47 GB in the compressor; the kernel then missed its watchdogd checkins and panicked).

Readings come from the kernel directly (ctypes, no fork):
  own footprint   proc_pid_rusage(RUSAGE_INFO_V2).ri_phys_footprint: resident anonymous memory plus this process's
                  compressed and swapped pages (what top's MEM column shows), in GiB;
  machine         host_statistics64(HOST_VM_INFO64): free, inactive, speculative, purgeable and compressor-occupied
                  pages; avail = free + inactive + speculative + purgeable (the earlier amoeba_memcheck figure);
                  starvation = free pages (speculative excluded) under kern.vm_page_free_min, or
                  vm.page_free_wanted > 0;
  swap            sysctl vm.swapusage (xsw_usage), pressure: sysctl kern.memorystatus_vm_pressure_level.

Wired weights (2026-11-19 19:19 stop: 79 GiB footprint, 46 GiB resident, the compressor +27 GiB in 3 min, i.e.
macOS compressing the battery's own MLX buffers; ops.wire_weights now wires them). No trigger changes with wiring:
wired pages are in neither free, inactive nor purgeable, so avail and the free-page count already measure only the
pageable room that is left; footprint (phys_footprint) counts the weights wired or not; the compressor can then
grow only from pageable memory (other processes, this process's unwired heap), and its growth / share and the
starvation triggers stop the run as before. A wired 80 GiB on 128 GiB leaves ~40 GiB pageable, so pressure reaches
starvation sooner and with less compressor growth; the triggers are absolute and stay valid. The machine's wired
total is read (wired_gib) and, after wiring, its minimum kept (observed_min_wired_gb: a drop means Metal released
the residency set), in WATCH and in the stop receipt.

check() is the pure trigger logic (tested on CPU). Two layers act on it:
  the in-process thread   polls every `every_s` (1 s); a soft trigger on 2 consecutive samples, or any hard one,
                          writes a JSON receipt and os._exit(3);
  the watcher process     opt-in (watcher=True): a separate small Python process (spawned before the model loads,
                          `python -I` on this file) polls its parent every `every_s` on the triggers that need no
                          baseline. A hard one (footprint >= hard ceiling, starvation for starved_hard_s samples)
                          SIGKILLs the target at once; a
                          soft one still true `grace_s` later (the thread did not stop it: GIL held by a long native
                          call, a Metal command buffer, a stalled log write) is SIGKILLed too. It writes its own
                          receipt first and exits when its parent is gone. It needs neither the target's GIL nor its
                          Metal queue.
"""
from __future__ import annotations

import ctypes
import json
import os
import resource
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

GIB = float(2**30)
_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)


class _RusageV2(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
        "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime",
        "ri_child_user_time", "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
        "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten")]


class _VMStats64(ctypes.Structure):                      # mach/vm_statistics.h vm_statistics64
    _fields_ = [("free_count", ctypes.c_uint32), ("active_count", ctypes.c_uint32),
                ("inactive_count", ctypes.c_uint32), ("wire_count", ctypes.c_uint32),
                ("zero_fill_count", ctypes.c_uint64), ("reactivations", ctypes.c_uint64),
                ("pageins", ctypes.c_uint64), ("pageouts", ctypes.c_uint64), ("faults", ctypes.c_uint64),
                ("cow_faults", ctypes.c_uint64), ("lookups", ctypes.c_uint64), ("hits", ctypes.c_uint64),
                ("purges", ctypes.c_uint64), ("purgeable_count", ctypes.c_uint32),
                ("speculative_count", ctypes.c_uint32), ("decompressions", ctypes.c_uint64),
                ("compressions", ctypes.c_uint64), ("swapins", ctypes.c_uint64), ("swapouts", ctypes.c_uint64),
                ("compressor_page_count", ctypes.c_uint32), ("throttled_count", ctypes.c_uint32),
                ("external_page_count", ctypes.c_uint32), ("internal_page_count", ctypes.c_uint32),
                ("total_uncompressed_pages_in_compressor", ctypes.c_uint64)]


class _XswUsage(ctypes.Structure):
    _fields_ = [("xsu_total", ctypes.c_uint64), ("xsu_avail", ctypes.c_uint64), ("xsu_used", ctypes.c_uint64),
                ("xsu_pagesize", ctypes.c_uint32), ("xsu_encrypted", ctypes.c_int)]


_libc.mach_host_self.restype = ctypes.c_uint32
_HOST = [None, None]                                     # (pid, host port): fetched again after a fork


def _host_port():
    if _HOST[0] != os.getpid():
        _HOST[:] = [os.getpid(), _libc.mach_host_self()]
    return _HOST[1]
_PAGE = resource.getpagesize()
HOST_VM_INFO64 = 4
STOP_MARKER = "stopping (exit 3)"   # the thread's last log line ends with it (callers key failed receipts on it)
WIRED_DROP_WARN_GIB = 10.0     # one log line when machine wired falls this far under its level at wiring


def footprint_bytes(pid: int) -> tuple[int, int] | None:
    """(phys_footprint, resident) bytes of pid, or None (gone / not ours)."""
    r = _RusageV2()
    if _libc.proc_pid_rusage(int(pid), 2, ctypes.byref(r)) != 0:
        return None
    return int(r.ri_phys_footprint), int(r.ri_resident_size)


def vm_pages() -> dict:
    s = _VMStats64()
    n = ctypes.c_uint32(ctypes.sizeof(s) // 4)
    if _libc.host_statistics64(_host_port(), HOST_VM_INFO64, ctypes.byref(s), ctypes.byref(n)) != 0:
        raise OSError("host_statistics64 failed")
    return {k: int(getattr(s, k)) for k, _ in s._fields_}


def _sysctl(name: str, obj):
    size = ctypes.c_size_t(ctypes.sizeof(obj))
    if _libc.sysctlbyname(name.encode(), ctypes.byref(obj), ctypes.byref(size), None, 0) != 0:
        raise OSError(f"sysctlbyname {name} failed")
    return obj


def swap_used_gib() -> float:
    try:
        return _sysctl("vm.swapusage", _XswUsage()).xsu_used / GIB
    except OSError:
        return 0.0


def pressure_level() -> int:
    try:
        return int(_sysctl("kern.memorystatus_vm_pressure_level", ctypes.c_int()).value)
    except OSError:
        return 0


def _sysctl_int(name: str, default: int) -> int:
    for t in (ctypes.c_uint32, ctypes.c_uint64):
        try:
            return int(_sysctl(name, t()).value)
        except OSError:
            continue
    return default


_FREE_MIN = []
_MEMSIZE = []


def free_min_pages() -> int:
    """kern.vm_page_free_min: under it the kernel blocks allocating threads (3500 pages here; 875 at the panic)."""
    if not _FREE_MIN:
        _FREE_MIN.append(_sysctl_int("kern.vm_page_free_min", 3500))
    return _FREE_MIN[0]


def memsize_gib() -> float:
    if not _MEMSIZE:
        _MEMSIZE.append(_sysctl_int("hw.memsize", 0) / GIB)
    return _MEMSIZE[0]


def readings(pid: int) -> dict:
    """One reading: the pid's footprint and the machine's memory, GiB (no subprocess)."""
    v = vm_pages()
    fp = footprint_bytes(pid)
    return {"t": time.time(), "pid": int(pid),
            "footprint_gib": None if fp is None else fp[0] / GIB, "resident_gib": None if fp is None else fp[1] / GIB,
            # free_count includes speculative pages (vm_stat prints free - speculative); the kernel compares
            # free_min with free pages not counting speculative ones
            "avail_gib": (v["free_count"] + v["inactive_count"] + v["purgeable_count"]) * _PAGE / GIB,
            "free_pages": max(0, v["free_count"] - v["speculative_count"]), "free_min_pages": free_min_pages(),
            "page_free_wanted": _sysctl_int("vm.page_free_wanted", 0),
            "compressor_gib": v["compressor_page_count"] * _PAGE / GIB,
            "compressed_logical_gib": v["total_uncompressed_pages_in_compressor"] * _PAGE / GIB,
            # machine-wide wired pages (the kernel's, other processes' and, once ops.wire_weights ran, this
            # process's MLX residency sets): no trigger reads it; receipts and WATCH record it
            "wired_gib": v["wire_count"] * _PAGE / GIB,
            "memsize_gib": memsize_gib(), "swap_used_gib": swap_used_gib(), "pressure": pressure_level()}


def make_limits(limit_gb: float, min_avail_gb: float = 15.0, swap_growth_gb: float = 2.0, hard_gb: float | None = None,
                compressor_growth_gb: float = 12.0, compressor_frac: float = 0.20, hard_compressor_frac: float = 0.30,
                compressor_share_min_growth_gb: float = 4.0,
                starved_soft_s: int = 3, starved_hard_s: int = 10, swap0: float = 0.0, compressor0: float = 0.0) -> dict:
    """Soft limits stop the run cleanly (exit 3 + receipt); hard ones are the watcher's SIGKILL ceiling.
    compressor_frac / hard_compressor_frac are shares of physical memory (hw.memsize) held by the compressor, checked
    only once the compressor also grew compressor_share_min_growth_gb since start (stale compressed pages of idle
    processes stay compressed long after the pressure has passed);
    starved_*_s are consecutive samples with free pages under kern.vm_page_free_min (or threads waiting for pages)."""
    return {"limit_gib": float(limit_gb), "hard_gib": float(hard_gb if hard_gb is not None else limit_gb + 6.0),
            "min_avail_gib": float(min_avail_gb), "swap_growth_gib": float(swap_growth_gb),
            "compressor_growth_gib": float(compressor_growth_gb), "compressor_frac": float(compressor_frac),
            "hard_compressor_frac": float(hard_compressor_frac),
            "compressor_share_min_growth_gib": float(compressor_share_min_growth_gb), "starved_soft_s": int(starved_soft_s),
            "starved_hard_s": int(starved_hard_s), "swap0_gib": float(swap0), "compressor0_gib": float(compressor0)}


def check(r: dict, lim: dict, state: dict | None = None, watcher: bool = False) -> list[tuple[str, str]]:
    """Triggers for reading r under limits lim: [(kind, message)], kind "hard" or "soft"; [] = fine. state carries
    the consecutive starved-sample count between calls (one dict per poller). watcher=True checks only what needs
    no per-carve baseline (footprint, starvation): the watcher never sees rebase().

    Footprint is the kernel's phys_footprint (compressed pages included). The free + inactive + speculative +
    purgeable floor alone would not have fired at the 2026-11-19 panic (37 GB of inactive anonymous pages, only
    "available" by compressing them; 875 free pages), nor would kern.memorystatus_level (it counts active and
    inactive pages as available). What did show it: free pages under kern.vm_page_free_min, and the compressor
    holding about a third of physical memory (counted here only together with growth since start)."""
    out = []
    st = state if state is not None else {}
    fp = r.get("footprint_gib")
    if fp is not None:
        if fp >= lim["hard_gib"]:
            out.append(("hard", f"footprint {fp:.1f} GiB >= hard ceiling {lim['hard_gib']:.1f} GiB"))
        elif fp > lim["limit_gib"]:
            out.append(("soft", f"footprint {fp:.1f} GiB > {lim['limit_gib']:.1f} GiB"))
    starved = r["free_pages"] < r["free_min_pages"] or r.get("page_free_wanted", 0) > 0
    st["starved"] = st.get("starved", 0) + 1 if starved else 0
    if st["starved"] >= lim["starved_hard_s"]:
        out.append(("hard", f"free pages {r['free_pages']} < kern.vm_page_free_min {r['free_min_pages']} (or page "
                            f"waiters) for {st['starved']} samples"))
    elif st["starved"] >= lim["starved_soft_s"]:
        out.append(("soft", f"free pages {r['free_pages']} < kern.vm_page_free_min {r['free_min_pages']} (or page "
                            f"waiters) for {st['starved']} samples"))
    if watcher:
        return out
    ms = r.get("memsize_gib") or 0.0
    grow = r["compressor_gib"] - lim["compressor0_gib"]
    if ms and grow >= lim["compressor_share_min_growth_gib"]:
        share = r["compressor_gib"] / ms
        if share >= lim["hard_compressor_frac"]:
            out.append(("hard", f"compressor holds {r['compressor_gib']:.1f} GiB ({100 * share:.0f}% of memory, "
                                f">= {100 * lim['hard_compressor_frac']:.0f}%; +{grow:.1f} GiB since start)"))
        elif share >= lim["compressor_frac"]:
            out.append(("soft", f"compressor holds {r['compressor_gib']:.1f} GiB ({100 * share:.0f}% of memory, "
                                f">= {100 * lim['compressor_frac']:.0f}%; +{grow:.1f} GiB since start)"))
    if r["avail_gib"] < lim["min_avail_gib"]:
        out.append(("soft", f"free+inactive+speculative+purgeable {r['avail_gib']:.1f} GiB < {lim['min_avail_gib']:.1f}"
                            " GiB"))
    if grow > lim["compressor_growth_gib"]:
        out.append(("soft", f"compressor grew {grow:.1f} GiB (> {lim['compressor_growth_gib']:.1f} GiB)"))
    sw = r["swap_used_gib"] - lim["swap0_gib"]
    if sw > lim["swap_growth_gib"]:
        out.append(("soft", f"swap grew {sw:.1f} GiB (> {lim['swap_growth_gib']:.1f} GiB)"))
    if r.get("pressure", 0) >= 4:
        out.append(("soft", "memory pressure critical"))
    return out


def write_receipt(path, obj) -> None:
    if not path:
        return
    p = Path(path)
    try:
        tmp = p.with_name(p.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(obj, indent=1, default=str))
        tmp.replace(p)
    except OSError:
        pass


class Watch:
    """The in-process thread (and, with watcher=True, the watcher process) for this process; start once."""

    def __init__(self, lim: dict, log, every_s: float = 1.0, receipt=None, watcher: bool = True,
                 grace_s: float = 5.0, stats: dict | None = None, soft_samples: int = 2):
        self.lim, self.log, self.every_s, self.receipt = lim, log, float(every_s), receipt
        self.stats = stats if stats is not None else {}
        self.soft_samples = max(1, int(soft_samples))
        self.pid = os.getpid()
        self.proc = None
        if watcher:
            self.proc = spawn_watcher(self.pid, lim, every_s=every_s, grace_s=grace_s,
                                      receipt=(str(receipt) + ".watcher.json") if receipt else None)
        threading.Thread(target=self._run, name="memwatch", daemon=True).start()

    def rebase(self) -> None:
        """New swap / compressor-growth baselines (a batch of carves in one process). The watcher checks no
        baseline-relative trigger, so it needs none."""
        r = readings(self.pid)
        self.lim["swap0_gib"], self.lim["compressor0_gib"] = r["swap_used_gib"], r["compressor_gib"]
        self.stats["swap_at_start_gb"] = round(r["swap_used_gib"], 2)

    def _run(self):
        state, soft_n, warned = {}, 0, False
        while True:
            try:
                r = readings(self.pid)
                fp = r["footprint_gib"]
                if fp is not None:
                    self.stats["peak_footprint_gb"] = max(self.stats.get("peak_footprint_gb", 0.0), round(fp, 1))
                self.stats["observed_min_avail_gb"] = round(min(self.stats.get("observed_min_avail_gb", 1e9),
                                                                r["avail_gib"]), 1)
                self.stats["observed_min_free_pages"] = min(self.stats.get("observed_min_free_pages", 1 << 62),
                                                            r["free_pages"])
                self.stats["peak_compressor_gb"] = max(self.stats.get("peak_compressor_gb", 0.0),
                                                       round(r["compressor_gib"], 1))
                w_at = self.stats.get("wired_at_wire_gb")
                if w_at is not None:                                   # after ops.wire_weights: did it stay?
                    self.stats["observed_min_wired_gb"] = round(min(self.stats.get("observed_min_wired_gb", 1e9),
                                                                    r["wired_gib"]), 1)
                    if r["wired_gib"] < w_at - WIRED_DROP_WARN_GIB and not self.stats.get("observed_wired_drop_at"):
                        self.stats["observed_wired_drop_at"] = time.strftime("%H:%M:%S")
                        try:
                            self.log(f"memory watchdog: machine wired {r['wired_gib']:.1f} GiB, "
                                     f"{w_at - r['wired_gib']:.1f} GiB under its {w_at:.1f} at wiring: the weights may "
                                     "be pageable again (Metal released the residency set?)")
                        except Exception:
                            pass
                hits = check(r, self.lim, state)
                if state.get("starved"):
                    self.stats["observed_starved_samples"] = self.stats.get("observed_starved_samples", 0) + 1
                if r.get("page_free_wanted", 0) > 0:
                    self.stats["observed_page_wait_samples"] = self.stats.get("observed_page_wait_samples", 0) + 1
            except Exception as e:                      # log once; keep polling (the watcher still runs)
                hits, r = [], {"error": repr(e)}
                if not warned:
                    warned = True
                    try:
                        self.log(f"memory watchdog: reading failed ({e!r}); still polling")
                    except Exception:
                        pass
            soft_n = soft_n + 1 if hits else 0
            if hits and (soft_n >= self.soft_samples or any(k == "hard" for k, _ in hits)):
                msg = "; ".join(m for _, m in hits)
                write_receipt(self.receipt, {"stopped_by": "in-process watchdog thread", "exit": 3,
                                             "triggers": hits, "reading": r, "limits": self.lim,
                                             "peak_footprint_gb": self.stats.get("peak_footprint_gb"),
                                             "wired": self.stats.get("wired"),
                                             "observed_min_wired_gb": self.stats.get("observed_min_wired_gb"),
                                             "observed_wired_drop_at": self.stats.get("observed_wired_drop_at"),
                                             "time": time.strftime("%Y-%m-%d %H:%M:%S")})
                try:
                    self.log(f"memory watchdog: {msg}; {STOP_MARKER}")
                except Exception:
                    pass
                os._exit(3)
            time.sleep(self.every_s)


def spawn_watcher(pid: int, lim: dict, every_s: float = 1.0, grace_s: float = 5.0, receipt=None):
    """The watcher process for pid (which must be this process: the watcher exits when its parent changes)."""
    if int(pid) != os.getpid():
        raise ValueError("spawn_watcher watches the calling process only")
    cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "--pid", str(pid), "--limits", json.dumps(lim),
           "--every-s", str(every_s), "--grace-s", str(grace_s)] + (["--receipt", str(receipt)] if receipt else [])
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, close_fds=True)


def watcher_step(r: dict, lim: dict, soft_since: float | None, now: float, grace_s: float, state: dict | None = None):
    """(action, soft_since, hits): action "kill" on a hard trigger or a soft one older than grace_s, else None.
    Only the baseline-free triggers (check(watcher=True))."""
    hits = check(r, lim, state, watcher=True)
    if any(k == "hard" for k, _ in hits):
        return "kill", soft_since, hits
    if hits:
        soft_since = soft_since if soft_since is not None else now
        return ("kill" if now - soft_since >= grace_s else None), soft_since, hits
    return None, None, hits


def watcher_main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--limits", required=True)
    ap.add_argument("--every-s", type=float, default=1.0)
    ap.add_argument("--grace-s", type=float, default=5.0)
    ap.add_argument("--receipt", default="")
    a = ap.parse_args(argv)
    lim = json.loads(a.limits)
    soft_since, state, warned = None, {}, False
    while True:
        if os.getppid() != a.pid:                      # target gone (re-parented): nothing left to watch
            return 0
        try:
            r = readings(a.pid)
            action, soft_since, hits = watcher_step(r, lim, soft_since, time.time(), a.grace_s, state)
        except Exception as e:                         # say so once, keep watching
            action = None
            if not warned:
                warned = True
                sys.stderr.write(f"memory watcher: reading failed ({e!r}); still watching\n")
        if action == "kill" and os.getppid() == a.pid:  # re-checked: the pid is still our parent, not a reused one
            write_receipt(a.receipt, {"stopped_by": "watcher process (SIGKILL)", "target_pid": a.pid,
                                      "triggers": hits, "reading": r, "limits": lim,
                                      "time": time.strftime("%Y-%m-%d %H:%M:%S")})
            sys.stderr.write(f"memory watcher: {'; '.join(m for _, m in hits)}; SIGKILL {a.pid}\n")
            sys.stderr.flush()
            try:
                os.kill(a.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return 0
        time.sleep(a.every_s)


if __name__ == "__main__":
    sys.exit(watcher_main())
