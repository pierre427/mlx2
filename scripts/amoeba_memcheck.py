#!/usr/bin/env python3
"""Memory admission for amoeba CPU probes (the machine kernel-panicked from memory starvation
on 2026-11-13).

Standing rule (Pierre): before any large load, no *other* process may hold more than 8 GB
resident and swap must be under 8 GB. "Resident" is checked as RSS and as top's physical
footprint (which counts compressed pages that RSS does not). Same checks as amoeba_memguard.py on
claude/quirky-poitras-2cdefc (not modified; reimplemented here so probes can call it in-process
and record the receipt).

policy "strict": every rule applies (default; required for anything above 10 GB peak).
policy "small" : for runs whose own peak is <= 10 GB, the swap and foreign-process rules are
                 recorded but not enforced, provided available memory >= 30 GB and swap has
                 >= 1 GB free; a watchdog aborts the run above 12 GB RSS or when the machine's
                 available memory falls under 12 GB. Allowed for Qwen3-0.6B CPU probes by the coordinating
                 session on 2026-11-13 (one run at a time); never for 4B or larger.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time


def snapshot_avail() -> float:
    vm = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", vm).group(1))
    tot = 0
    for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        m = re.search(rf"{k}:\s+(\d+)", vm)
        tot += int(m.group(1)) if m else 0
    return tot * page / 1e9


def snapshot() -> dict:
    vm = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", vm).group(1))

    def get(k):
        m = re.search(rf"{k}:\s+(\d+)", vm)
        return int(m.group(1)) if m else 0
    avail = (get("Pages free") + get("Pages inactive") + get("Pages speculative") + get("Pages purgeable")) * page / 1e9
    sw = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    used = re.search(r"used = ([\d.]+)M", sw)
    free_sw = re.search(r"free = ([\d.]+)M", sw)
    swap_gb = float(used.group(1)) / 1024 if used else 0.0
    swap_free_gb = float(free_sw.group(1)) / 1024 if free_sw else 0.0
    mine = {os.getpid(), os.getppid()}
    big = []
    for line in subprocess.run(["/bin/ps", "-axo", "pid,rss,command"], capture_output=True, text=True).stdout.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, rss = int(parts[0]), int(parts[1])
        if rss > 8e6 and pid not in mine:
            big.append({"pid": pid, "rss_gb": round(rss / 1e6, 1), "command": parts[2][:120]})
    # RSS misses compressed memory: a process can sit at 3 GB RSS with a 100 GB footprint
    # (seen 2026-11-13). top's MEM column is the physical footprint, compressed pages included.
    top = subprocess.run(["/usr/bin/top", "-l", "1", "-o", "mem", "-n", "12", "-stats", "pid,mem,command"],
                         capture_output=True, text=True).stdout.splitlines()
    unit = {"K": 1e-6, "M": 1e-3, "G": 1.0, "T": 1e3}
    for line in top:
        m = re.match(r"\s*(\d+)\s+([\d.]+)([KMGT])\+?\s+(.*)", line)
        if not m:
            continue
        pid, gb = int(m.group(1)), float(m.group(2)) * unit[m.group(3)]
        if gb > 8 and pid not in mine and all(b["pid"] != pid for b in big):
            big.append({"pid": pid, "footprint_gb": round(gb, 1), "command": m.group(4)[:120]})
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "available_gb": round(avail, 1),
            "swap_used_gb": round(swap_gb, 2), "swap_free_gb": round(swap_free_gb, 2), "foreign_procs_over_8gb": big}


def admit(need_gb: float, policy: str = "strict") -> tuple[bool, dict]:
    s = snapshot()
    s.update({"need_gb": need_gb, "policy": policy})
    reasons = []
    if s["foreign_procs_over_8gb"]:
        reasons.append("foreign process over 8 GB")
    if s["swap_used_gb"] >= 8.0:
        reasons.append("swap >= 8 GB")
    if policy.startswith("waived:"):
        # waived:<min available GB>:<own RSS cap GB>:<abort below machine available GB>
        # Pierre, 2026-11-13, for the Qwen3-4B CPU core-fraction run: "Waive, with watchdog".
        avail_min = float(policy.split(":")[1])
        if s["swap_free_gb"] < 1.0 and s["available_gb"] < 60:
            return False, {**s, "verdict": "REFUSE", "reasons": ["swap has under 1 GB free and available < 60 GB"]}
        if s["available_gb"] < avail_min:
            return False, {**s, "verdict": "REFUSE", "reasons": [f"available < {avail_min} GB"]}
        return True, {**s, "verdict": "ADMIT", "waived": reasons, "waiver": "Pierre 2026-11-13 (4B CPU run, watchdog)"}
    if policy == "small":
        if need_gb > 10:
            return False, {**s, "verdict": "REFUSE", "reasons": ["policy small needs a peak <= 10 GB"]}
        # stale swap with plenty of free memory is not pressure; under 1 GB free swap needs >= 60 GB available
        if s["swap_free_gb"] < 1.0 and s["available_gb"] < 60:
            return False, {**s, "verdict": "REFUSE", "reasons": ["swap has under 1 GB free and available < 60 GB"]}
        if s["available_gb"] < 30:
            return False, {**s, "verdict": "REFUSE", "reasons": ["available < 30 GB"]}
        return True, {**s, "verdict": "ADMIT", "waived": reasons}
    if s["available_gb"] < need_gb + 20:
        reasons.append("available < need + 20 GB")
    return (not reasons), {**s, "verdict": "ADMIT" if not reasons else "REFUSE", "reasons": reasons}


def rss_watchdog(limit_gb: float, log=print, every_s: float = 5.0, min_avail_gb: float = 12.0):
    """Abort this process (exit 3) if its resident set passes limit_gb, or if the machine's
    available memory falls under min_avail_gb (a neighbour growing while this run holds memory)."""
    import threading

    def run():
        pid = str(os.getpid())
        while True:
            out = subprocess.run(["/bin/ps", "-o", "rss=", "-p", pid], capture_output=True, text=True).stdout.strip()
            if out and int(out) / 1e6 > limit_gb:
                log(f"memcheck watchdog: RSS {int(out) / 1e6:.1f} GB > {limit_gb} GB, aborting")
                os._exit(3)
            av = snapshot_avail()
            if av < min_avail_gb:
                log(f"memcheck watchdog: machine available {av:.1f} GB < {min_avail_gb} GB, aborting")
                os._exit(3)
            time.sleep(every_s)
    threading.Thread(target=run, daemon=True).start()


def require(need_gb: float, policy: str = "strict", wait_s: int = 0, log=print) -> dict:
    """Admit or exit. wait_s > 0 polls every 60 s until admitted or the wait runs out."""
    end = time.time() + wait_s
    while True:
        ok, rec = admit(need_gb, policy)
        log(f"memcheck {rec['verdict']}: need {need_gb} GB, avail {rec['available_gb']} GB, swap {rec['swap_used_gb']} GB, "
            f"big {[(b['pid'], b.get('rss_gb', b.get('footprint_gb'))) for b in rec['foreign_procs_over_8gb']]} {rec.get('reasons') or rec.get('waived') or ''}")
        if ok:
            if policy == "small":
                rss_watchdog(12.0, log=log)
                rec["rss_watchdog_gb"] = 12.0
            elif policy.startswith("waived:"):
                _, _, cap, floor = policy.split(":")
                rss_watchdog(float(cap), log=log, min_avail_gb=float(floor))
                rec["rss_watchdog_gb"], rec["abort_below_available_gb"] = float(cap), float(floor)
            return rec
        if time.time() >= end:
            sys.exit("memcheck refused the load")
        time.sleep(60)


if __name__ == "__main__":
    need = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    pol = sys.argv[2] if len(sys.argv) > 2 else "strict"
    ok, rec = admit(need, pol)
    print(rec)
    sys.exit(0 if ok else 1)
