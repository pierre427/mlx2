"""Run plumbing for the generic carve pipeline: receipts, memory admission, footprint watchdog.

Ported unchanged from scripts/amoeba27b_lib.py at claude/amoeba-27b-80bc6a@37d75f144 (lines 45-203),
so the memory rules (Pierre 2026-11-14: own footprint cap 80 GB; coordinator waiver terms) behave
exactly as in the 27B windows. Only the import path of amoeba_memcheck changed. 2026-11-19 (FN battery panic):
footprint_gb and footprint_watchdog read the kernel directly and the watchdog gained a watcher process
(research/amoeba/carve/memwatch.py).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def sha12(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def fresh_dir(path) -> Path:
    out = Path(path)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite non-empty evidence dir {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_log(out: Path):
    f = open(out / "run.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    return log


def footprint_gb(pid: int) -> float | None:
    """The pid's physical footprint in GiB (compressed pages included; what top's MEM shows), read from the kernel
    (proc_pid_rusage) without forking top; top is the fallback for a pid this user cannot read."""
    try:
        from . import memwatch
        fp = memwatch.footprint_bytes(pid)
        if fp is not None:
            return fp[0] / memwatch.GIB
    except Exception:
        pass
    out = subprocess.run(["/usr/bin/top", "-l", "1", "-pid", str(pid), "-stats", "mem"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    if not out:
        return None
    m = re.match(r"([\d.]+)([KMGT])", out[-1].strip())
    if not m:
        return None
    return float(m.group(1)) * {"K": 1e-6, "M": 1e-3, "G": 1.0, "T": 1e3}[m.group(2)]


def swap_used_gb() -> float:
    out = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) / 1024 if m else 0.0


def pressure_level() -> int:
    out = subprocess.run(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True).stdout
    try:
        return int(out.strip())
    except ValueError:
        return 0


WATCH = {}


_WATCHDOG_STARTED = False
_SWAP0 = [0.0]


def reset_swap_baseline():
    _SWAP0[0] = swap_used_gb()
    WATCH["swap_at_start_gb"] = round(_SWAP0[0], 2)


_WATCHER = [None]


def footprint_watchdog(limit_gb: float, log, every_s: float = 1.0, min_avail_gb: float | None = None,
                       swap_growth_gb: float = 2.0, hard_gb: float | None = None, compressor_growth_gb: float = 12.0,
                       receipt=None, watcher: bool = False):
    """Stop this process (exit 3, with a JSON receipt at `receipt`) when its physical footprint passes limit_gb, free
    + inactive + speculative + purgeable falls under min_avail_gb, the machine is starved (free pages under
    kern.vm_page_free_min for 3 s), the memory compressor grew more than compressor_growth_gb over its level now
    (or holds >= 20 % of physical memory after growing 4 GiB; hard at 30 %), swap grows more than swap_growth_gb, or pressure goes critical
    (research/amoeba/carve/memwatch.py; 2026-11-19 panic). Polls every every_s (1 s) from the kernel, without
    subprocesses. watcher=True also starts the watcher process, which SIGKILLs this process at the hard ceilings
    (footprint hard_gb, default limit_gb + 6; starvation for 10 s) or when a soft
    trigger is still true 5 s later, whatever this process's GIL or GPU queue is doing. Peak footprint and the
    observed minima are kept in WATCH. MLX2_AMOEBA_WATCHDOG=off disables it (the CPU test suite).
    One watchdog per process: a later call re-bases the swap / compressor baselines and can only tighten the limits
    (_tighten; every_s, receipt and watcher stay the first call's)."""
    from . import memwatch
    global _WATCHDOG_STARTED
    if os.environ.get("MLX2_AMOEBA_WATCHDOG", "").lower() == "off":
        return
    if min_avail_gb is None:
        min_avail_gb = float(os.environ.get("AMOEBA_MIN_FREE_GB", "").strip() or 15.0)
    if _WATCHDOG_STARTED:                  # a batch of carves in one process: one watchdog for the process; each
        reset_swap_baseline()              # carve re-bases the swap-growth check (slow drift over a long batch)
        if _WATCHER[0] is not None:
            _WATCHER[0].rebase()
            _tighten(_WATCHER[0].lim, log, limit_gb=limit_gb, min_avail_gb=min_avail_gb,
                     swap_growth_gb=swap_growth_gb, hard_gb=hard_gb, compressor_growth_gb=compressor_growth_gb)
        return
    _WATCHDOG_STARTED = True
    reset_swap_baseline()
    r = memwatch.readings(os.getpid())
    lim = memwatch.make_limits(limit_gb, min_avail_gb=min_avail_gb, swap_growth_gb=swap_growth_gb, hard_gb=hard_gb,
                               compressor_growth_gb=compressor_growth_gb, swap0=r["swap_used_gib"],
                               compressor0=r["compressor_gib"])
    WATCH.update(swap_at_start_gb=round(r["swap_used_gib"], 2), peak_footprint_gb=0.0, limit_gb=limit_gb,
                 min_avail_gb=min_avail_gb, swap_growth_gb=swap_growth_gb, hard_gb=lim["hard_gib"],
                 compressor_at_start_gb=round(r["compressor_gib"], 2), compressor_growth_gb=compressor_growth_gb,
                 every_s=every_s, watcher=bool(watcher))
    _WATCHER[0] = memwatch.Watch(lim, log, every_s=every_s, receipt=receipt, watcher=watcher, stats=WATCH)


def _tighten(lim, log, *, limit_gb, min_avail_gb, swap_growth_gb, hard_gb, compressor_growth_gb):
    """A later footprint_watchdog call in a process whose watchdog already runs: a stricter request tightens the
    live limits in place (the thread reads them every poll; the caps guard a shared machine), a looser one is not
    applied and is recorded in WATCH["requested_ignored"]. WATCH keeps the limits in force. The watcher process
    keeps the hard footprint ceiling it was spawned with (WATCH["watcher_hard_gb"] when that now differs)."""
    hard = float(hard_gb if hard_gb is not None else limit_gb + 6.0)
    ignored = {}
    for wkey, lkey, want, stricter in (("limit_gb", "limit_gib", float(limit_gb), min),
                                       ("hard_gb", "hard_gib", hard, min),
                                       ("min_avail_gb", "min_avail_gib", float(min_avail_gb), max),
                                       ("swap_growth_gb", "swap_growth_gib", float(swap_growth_gb), min),
                                       ("compressor_growth_gb", "compressor_growth_gib", float(compressor_growth_gb), min)):
        have = lim[lkey]
        if stricter(have, want) != have:
            if lkey == "hard_gib" and WATCH.get("watcher"):
                WATCH.setdefault("watcher_hard_gb", have)
            lim[lkey] = want
            WATCH[wkey] = want
        elif want != have and not (lkey == "hard_gib" and hard_gb is None):   # a derived default is no request
            ignored[wkey] = {"requested": want, "in_force": have}
    WATCH.pop("requested_ignored", None)
    if ignored:
        WATCH["requested_ignored"] = ignored
        try:
            log(f"memory watchdog: already running; looser limits not applied {ignored}")
        except Exception:
            pass


ALLOWED_FOREIGN = {760}   # Pierre's MLX2 desktop app; named in the coordinator's waiver (2026-11-14)


def big_by_footprint(limit_gb: float = 8.0, exclude=()):
    """Other processes whose physical footprint (top's mem) exceeds limit_gb: the standing memory
    rule (resident memory counts mapped files; coordinator 2026-11-14). Skips ALLOWED_FOREIGN,
    hf downloads and ``exclude``."""
    import amoeba_memcheck
    mine = {os.getpid(), os.getppid()}
    out = []
    for b in amoeba_memcheck.snapshot()["foreign_procs_over_8gb"]:
        pid = b["pid"]
        if pid in mine or pid in ALLOWED_FOREIGN or pid in set(exclude) or is_hf_download(pid):
            continue
        fp = footprint_gb(pid)
        if fp is not None and fp > limit_gb:
            out.append({**b, "footprint_gb": round(fp, 1)})
    return out


def is_hf_download(pid) -> bool:
    """hf / huggingface-cli download processes: footprint is mostly file cache, not a model.
    Exempt from the foreign-process rule under the coordinator's waiver (2026-11-14, as
    amoeba_memguard c9a57cac6); available-memory, swap-growth and footprint aborts still apply."""
    cmd = subprocess.run(["/bin/ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout
    return bool(re.search(r"(^|/)(hf|huggingface-cli)\s+download\b", cmd))


def admit(need_gb: float, policy: str, log, wait_s: int = 0) -> dict:
    """amoeba_memcheck.require, plus (under a waiver) refuse while any other process over 8 GB runs
    apart from ALLOWED_FOREIGN: no second large model process, whatever available memory says."""
    import amoeba_memcheck
    if policy.startswith("auto:"):
        # strict when it passes (Pierre freed memory 2026-11-14), else the coordinator's waiver
        ok, rec = amoeba_memcheck.admit(need_gb, "strict")
        policy = "strict" if ok else policy[len("auto:"):]
        log(f"admit: auto policy chose {policy} (strict {'passed' if ok else 'refused: ' + str(rec.get('reasons'))})")
    if policy.startswith("waived:"):
        end = time.time() + wait_s
        while True:
            big = big_by_footprint()
            if not big:
                break
            log(f"admit: waiting, other large processes {[(b['pid'], b.get('rss_gb', b.get('footprint_gb')), b['command'][:60]) for b in big]}")
            if time.time() >= end:
                sys.exit("admit: another large process is running")
            time.sleep(60)
    rec = amoeba_memcheck.require(need_gb, policy, wait_s=wait_s, log=log)
    rec["waiver"] = policy if policy.startswith("waived:") else None
    rec["exempt_hf_downloads"] = [b["pid"] for b in rec.get("foreign_procs_over_8gb", []) if is_hf_download(b["pid"])]
    rec["waiver_terms"] = ("coordinator 2026-11-14: allowed foreign pid 760 only; abort on swap +2 GB, critical "
                           "pressure, available < 15 GB, own footprint > 80 GB (Pierre raised 60 to 80, 2026-11-14)") if policy.startswith("waived:") else None
    return rec


def setup_mlx(memory_gb: float, cache_gb: float):
    # Stage children wired by AMOEBA_CARVE_WIRED_GB (Host wires after load) need the residency refresh set
    # before MLX creates its Metal device, as the core carve does (prepare_wiring is idempotent).
    _spec = os.environ.get("AMOEBA_CARVE_WIRED_GB", "").strip()
    if _spec and _spec != "0" and not WIRED.get("prepared"):
        prepare_wiring(_spec)
    import mlx.core as mx
    mx.set_memory_limit(int(memory_gb * 2**30))
    mx.set_cache_limit(int(cache_gb * 2**30))


WIRED = {}
WIRED_AUTO_WORKING_GB = 10.0     # auto: the weights MLX holds after load plus this much working set
WIRED_CAP_MARGIN_GB = 8.0        # the limit stays this far under min(iogpu.wired_limit_mb, recommended working set)
WIRED_REFRESH_MS = "1000"        # MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS while wiring (setdefault)


def wired_default(default: str = "auto") -> str:
    """The --wired-gb default: env AMOEBA_WIRED_GB, else `default`."""
    return os.environ.get("AMOEBA_WIRED_GB", "").strip() or default


def _wired_spec(spec) -> float | str | None:
    """None (off), "auto", or GiB. Accepts 0 / off / no / none for off."""
    s = str(spec if spec is not None else "off").strip().lower()
    if s in ("", "0", "0.0", "off", "no", "none", "false"):
        return None
    if s == "auto":
        return "auto"
    v = float(s)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"--wired-gb {spec}: not a finite GiB >= 0, auto or off")
    return v or None


def wired_arg(spec: str) -> str:
    """argparse type for --wired-gb: a bad value fails at parse time, before anything loads."""
    import argparse
    try:
        _wired_spec(spec)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return spec


def prepare_wiring(spec) -> dict:
    """Call before the first MLX Metal call (setup_mlx): MLX reads MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS when it
    builds its residency sets, at Metal device creation. While set, an MLX thread re-requests residency of every
    non-empty set each interval, so a CPU-side gap between forward passes (mask building, held-set encoding) does
    not let Metal drop the wired weights after ~2 s idle (runtime/residency_heartbeat.py). An existing value wins;
    AMOEBA_RESIDENCY_REFRESH_MS overrides the default (0 = off)."""
    if _wired_spec(spec) is None:
        return {"refresh_ms": os.environ.get("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS")}
    want = os.environ.get("AMOEBA_RESIDENCY_REFRESH_MS", WIRED_REFRESH_MS)
    if want and want != "0":
        os.environ.setdefault("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS", want)
    WIRED["refresh_ms"] = os.environ.get("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS")
    WIRED["prepared"] = True       # a caller contract (prepare_wiring before setup_mlx), not checked against MLX
    return {"refresh_ms": WIRED["refresh_ms"]}


def plan_wired_limit(spec, active_gib: float, iogpu_mb: int, max_rec_gib: float | None,
                     memory_gb: float | None = None) -> dict:
    """Pure sizing of the wired limit (CPU-tested). want = active + WIRED_AUTO_WORKING_GB (auto) or spec GiB; capped
    at min(iogpu.wired_limit_mb, the device's recommended working set) - WIRED_CAP_MARGIN_GB (MLX refuses a limit
    above the recommended working set) and at memory_gb (MLX's own allocation limit: wiring above it buys nothing).
    weights_fit is False when the limit is under what MLX holds now: MLX then wires allocations only up to the
    limit, and the rest stays pageable (compressible)."""
    s = _wired_spec(spec)
    if s is None:
        return {"wired": False, "reason": "off (--wired-gb 0)"}
    if not max_rec_gib:
        return {"wired": False, "reason": "no Metal recommended working-set size"}
    caps = {"recommended_working_set_gib": float(max_rec_gib)}
    if iogpu_mb and iogpu_mb > 0:
        caps["iogpu_wired_limit_gib"] = iogpu_mb / 1024.0
    hw = min(caps.values()) - WIRED_CAP_MARGIN_GB
    want = float(active_gib) + WIRED_AUTO_WORKING_GB if s == "auto" else float(s)
    limit = min(want, hw)
    if memory_gb:
        caps["memory_limit_gib"] = float(memory_gb)
        limit = min(limit, float(memory_gb))
    if limit <= 0:
        return {"wired": False, "reason": f"cap {hw:.1f} GiB leaves nothing to wire", "caps": caps}
    return {"wired": True, "spec": "auto" if s == "auto" else float(s), "want_gib": round(want, 2),
            "limit_gib": round(limit, 2), "limit_bytes": int(limit * 2**30), "caps": caps,
            "cap_margin_gib": WIRED_CAP_MARGIN_GB, "active_gib_at_wire": round(float(active_gib), 2),
            "weights_fit": float(active_gib) <= limit, "capped": limit < want}


def _machine_wired_gib() -> float:
    from . import memwatch
    return memwatch.vm_pages()["wire_count"] * memwatch._PAGE / 2**30


def wire_weights(spec, log, memory_gb: float | None = None, settle_s: float = 60.0) -> dict:
    """Wire what MLX holds now (call after the model loaded and its parameters were evaluated: Host does
    mx.eval(model.parameters())) and keep it wired until exit. MLX 0.32.4 (metal/resident.cpp, allocator.cpp):
    every Metal allocation is tracked; mx.set_wired_limit resizes the residency sets, and on growth adds every
    tracked allocation that fits, in no particular order, then commits; later allocations join while under the
    limit. The sets carry a standing requestResidency() and sit on every command queue, so the pages are wired
    when the next command buffer runs: this commits one tiny eval, then watches the machine's wired page count
    (host_statistics64 wire_count; the residency set is not in the process's ri_wired_size) rise by about what was
    admitted (`verified`: >= 80 % of min(active, limit) within settle_s). The previous limit is restored at exit
    (atexit; a watchdog os._exit or SIGKILL skips it, and the process's wired memory goes with the process).
    Idempotent: a second call in one process returns the first receipt."""
    if WIRED.get("receipt"):
        return WIRED["receipt"]
    import atexit
    import mlx.core as mx
    try:
        available = mx.metal.is_available()
    except Exception:  # noqa: BLE001 - a non-Metal build has nothing to wire
        available = False
    info = mx.device_info() if available else {}
    from . import memwatch
    iogpu_mb = memwatch._sysctl_int("iogpu.wired_limit_mb", 0)
    max_rec = info.get("max_recommended_working_set_size")
    active = mx.get_active_memory() / 2**30 if available else 0.0
    rec = plan_wired_limit(spec, active, iogpu_mb, (max_rec / 2**30) if max_rec else None, memory_gb)
    # the refresh thread exists only if the env was set before MLX made its Metal device: known when prepare_wiring
    # ran first in this process (callers do it before setup_mlx); otherwise only the env as it is now is known
    env_ms = os.environ.get("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS")
    rec.update(iogpu_wired_limit_mb=iogpu_mb, refresh_ms=WIRED.get("refresh_ms") if WIRED.get("prepared") else None,
               refresh_env_now=env_ms, refresh_prepared_before_metal=bool(WIRED.get("prepared")),
               mechanism="mlx set_wired_limit -> MTLResidencySet (runtime/weight_residency.py)")
    if not rec["wired"]:
        WIRED["receipt"] = rec
        log(f"wiring: {rec['reason']}")
        return rec
    mx.synchronize()
    # free MLX's buffer cache first: cached buffers are tracked allocations too, and the residency sets admit
    # tracked allocations in no particular order when the limit grows, so a cache could take weights' room
    mx.clear_cache()
    w0 = _machine_wired_gib()
    prev = mx.set_wired_limit(rec["limit_bytes"])
    rec["previous_wired_limit_bytes"] = int(prev)

    def restore():
        try:
            mx.synchronize()
            mx.set_wired_limit(int(prev))
        except Exception:  # noqa: BLE001 - at exit: nothing left to do about it
            pass
    atexit.register(restore)
    expect = min(active, rec["limit_gib"])
    t0 = time.time()
    delta = 0.0
    while True:
        mx.eval(mx.zeros((1,)) + 1)                    # a command buffer on a queue holding the sets
        mx.synchronize()
        delta = _machine_wired_gib() - w0
        if delta >= 0.8 * expect or time.time() - t0 >= settle_s:
            break
        time.sleep(1.0)
    rec.update(machine_wired_gib_before=round(w0, 2), machine_wired_gib_after=round(w0 + delta, 2),
               machine_wired_delta_gib=round(delta, 2), expected_delta_gib=round(expect, 2),
               verified=bool(delta >= 0.8 * expect), settle_s=round(time.time() - t0, 1))
    WIRED["receipt"] = rec
    WATCH["wired"] = rec
    WATCH["wired_at_wire_gb"] = round(w0 + delta, 1)
    log(f"wiring: limit {rec['limit_gib']:.1f} GiB (want {rec['want_gib']:.1f}, caps "
        f"{', '.join(f'{k} {v:.1f}' for k, v in rec['caps'].items())}, margin {WIRED_CAP_MARGIN_GB:.0f}); MLX active "
        f"{active:.1f} GiB; machine wired {w0:.1f} -> {w0 + delta:.1f} GiB in {rec['settle_s']:.0f}s "
        f"({'verified' if rec['verified'] else 'NOT verified: wired count did not rise by 80 % of the weights'})"
        + ("" if rec["weights_fit"] else "; WARNING: limit under MLX active memory, part of the weights stay pageable"))
    return rec


def peak_gb() -> float:
    import mlx.core as mx
    return round(mx.get_peak_memory() / 2**30, 2)


def save_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **{k: np.array(v) for k, v in arrays.items()})
    tmp.rename(path)


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.rename(path)
