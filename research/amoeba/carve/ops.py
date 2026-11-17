"""Run plumbing for the generic carve pipeline: receipts, memory admission, footprint watchdog.

Ported unchanged from scripts/amoeba27b_lib.py at claude/amoeba-27b-80bc6a@37d75f144 (lines 45-203),
so the memory rules (Pierre 2026-11-14: own footprint cap 80 GB; coordinator waiver terms) behave
exactly as in the 27B windows. Only the import path of amoeba_memcheck changed.
"""
from __future__ import annotations

import hashlib
import json
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


def footprint_watchdog(limit_gb: float, log, every_s: float = 15.0, min_avail_gb: float = 15.0, swap_growth_gb: float = 2.0):
    """Exit 3 if this process's physical footprint passes limit_gb, machine available memory falls
    under min_avail_gb, swap grows more than swap_growth_gb above its level now, or memory pressure
    goes critical (coordinator's waiver terms, 2026-11-14). Peak footprint is kept in WATCH."""
    import amoeba_memcheck
    global _WATCHDOG_STARTED
    if _WATCHDOG_STARTED:                  # a batch of carves in one process: one watchdog for the process; each
        reset_swap_baseline()              # carve re-bases the swap-growth check (slow drift over a long batch)
        return
    _WATCHDOG_STARTED = True
    reset_swap_baseline()
    swap0 = _SWAP0[0]
    WATCH.update(swap_at_start_gb=round(swap0, 2), peak_footprint_gb=0.0, limit_gb=limit_gb,
                 min_avail_gb=min_avail_gb, swap_growth_gb=swap_growth_gb)

    def run():
        pid = os.getpid()
        while True:
            sw = swap_used_gb()
            base = _SWAP0[0]                       # re-based per carve in a batch (reset_swap_baseline)
            if sw > base + swap_growth_gb:
                log(f"watchdog: swap {sw:.1f} GB grew more than {swap_growth_gb} GB over {base:.1f} GB, aborting")
                os._exit(3)
            if pressure_level() >= 4:
                log("watchdog: memory pressure critical, aborting")
                os._exit(3)
            fp = footprint_gb(pid)
            if fp is not None:
                WATCH["peak_footprint_gb"] = max(WATCH["peak_footprint_gb"], round(fp, 1))
            if fp is not None and fp > limit_gb:
                log(f"footprint watchdog: {fp:.1f} GB > {limit_gb} GB, aborting")
                os._exit(3)
            av = amoeba_memcheck.snapshot_avail()
            if av < min_avail_gb:
                log(f"footprint watchdog: machine available {av:.1f} GB < {min_avail_gb} GB, aborting")
                os._exit(3)
            time.sleep(every_s)
    threading.Thread(target=run, daemon=True).start()


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
    import mlx.core as mx
    mx.set_memory_limit(int(memory_gb * 2**30))
    mx.set_cache_limit(int(cache_gb * 2**30))


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
