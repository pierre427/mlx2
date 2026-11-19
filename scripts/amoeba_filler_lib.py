#!/usr/bin/env python3
"""The expert-map filler's yield rule (Pierre 2026-11-16: lowest priority, daytime work first; coordinator's rule),
shared by the filler drivers and amoeba_firing_capture.py. Something else wants the GPU when any of:
  gates     a file matching a --gates glob (e.g. ~/amoeba-runs/gate-*.pid; the filler's own window gate is outside it)
            whose first token is a live pid;
  waiters   an entry of the GPU lock waiter ledger (/Users/Shared/mlxuag/gpu.lock.waiters; name
            <session>.<label>.<pid>.<created>) with a live pid, touched in the last 300 s (run_with_gpu_locks'
            liveness rule), that is not the filler's own (label amoeba27b-filler-* on the M5, amoebafiller-* on the M3);
  foreign   (optional) another process whose physical footprint (top's MEM, which counts Metal / MLX memory that
            ps RSS does not) is over --foreign-gb GB (a model server on a small host);
  hours     (optional) the local hour is outside an allowed window "HH-HH" (night-only models: 23-07).
CLI: amoeba_filler_lib.py [--gates GLOB]... [--waiters DIR] [--foreign-gb G] [--hours 23-07]; prints the reasons and
exits 0 when something is pending, 1 when the GPU is free for the filler.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path


OWN_LABELS = ("amoeba27b-filler-", "amoebafiller-")


def pending(gates=(), waiters="", foreign_gb=0.0, hours="", exclude_pids=()):
    out = []
    for g in gates:
        for f in glob.glob(os.path.expanduser(g)):
            try:
                pid = int(Path(f).read_text().split()[0])
                os.kill(pid, 0)
                out.append(f"gate {os.path.basename(f)} (pid {pid})")
            except (ValueError, IndexError, ProcessLookupError, PermissionError, FileNotFoundError):
                continue
    if waiters and os.path.isdir(waiters):
        now = time.time()
        for name in os.listdir(waiters):
            fields = name.rsplit(".", 3)
            if len(fields) != 4 or any(f".{x}" in name for x in OWN_LABELS):   # <session>.<label>...: ours by label
                continue
            try:
                pid = int(fields[-2])
                os.kill(pid, 0)
                if now - os.stat(os.path.join(waiters, name)).st_mtime > 300:
                    continue
            except (ValueError, ProcessLookupError, PermissionError, FileNotFoundError):
                continue
            out.append(f"waiter {name}")
    if foreign_gb:
        mine = {os.getpid(), os.getppid(), *exclude_pids}
        rows = subprocess.run(["top", "-l", "1", "-o", "mem", "-n", "12", "-stats", "pid,mem,command"],
                              capture_output=True, text=True).stdout.splitlines()
        unit = {"B": 1 / 2 ** 30, "K": 1 / 2 ** 20, "M": 1 / 2 ** 10, "G": 1.0, "T": 1024.0}
        for r in rows:
            f = r.split(None, 2)
            if len(f) < 3 or not f[0].isdigit():
                continue
            m = f[1].rstrip("+-")
            try:
                gb = float(m[:-1]) * unit[m[-1]] if m[-1] in unit else float(m) / 2 ** 30
            except (ValueError, IndexError):
                continue
            if int(f[0]) not in mine and gb > foreign_gb:
                out.append(f"foreign process {f[0]} {f[2][:40]} ({gb:.1f} GB footprint)")
    if hours:
        a, b = (int(x) for x in hours.split("-"))
        h = time.localtime().tm_hour
        inside = (a <= h < b) if a < b else (h >= a or h < b)
        if not inside:
            out.append(f"hour {h} outside {hours}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", action="append", default=[])
    ap.add_argument("--waiters", default="")
    ap.add_argument("--foreign-gb", type=float, default=0.0)
    ap.add_argument("--hours", default="")
    a = ap.parse_args()
    try:
        p = pending(a.gates, a.waiters, a.foreign_gb, a.hours)
    except Exception as e:                 # a broken check must never read as "GPU free" (review 2026-11-16)
        print(f"yield rule error: {type(e).__name__}: {e}")
        return 0
    for x in p:
        print(x)
    return 0 if p else 1


if __name__ == "__main__":
    sys.exit(main())
