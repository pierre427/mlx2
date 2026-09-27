#!/usr/bin/env python3
"""Idle-under-pressure survival and exit-trace check for one server (GPU).

The 2026-09-25 incident: a Qwen3.8 27B server (native-MTP default route) sat
idle ~180 s while a separate process held 24 GiB, and went away with nothing
in its log.  The series orchestrator's startup ``pkill -f mlx2[.]server``
landed in that window (qualification/runs/silent-death-20260926/FORENSICS.md).

Per trial (one server process each, sequential):
  1. start ``python -m mlx2.server --model M`` from the chosen ``src`` with
     stdout+stderr captured and, when the tree supports it, ``--fault-log``;
  2. three fresh-prompt requests;
  3. a pressure child fills anonymous memory (incompressible) up to
     ``--cap-gib`` in 256 MiB steps, stopping early at the memory floor;
  4. idle ``--idle-s`` seconds, sampling every 2 s whether the server is
     alive and its rusage; a trial with ``inject_at`` sends the server
     SIGTERM at that second, exactly what ``pkill`` sends;
  5. if alive: one request under pressure, release, one request, then
     SIGTERM, so the log shows the ordinary shutdown path too;
  6. record the return code and the server log's last lines.

Safety (checked every second by a watchdog): any vm_stat Swapouts rise of
``--swap-abort-mib`` after load, or free+reclaimable memory below
``--floor-gib``, kills the pressure child first and then the server.
Refuses to run without ``--i-own-the-gpu``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PAGE = 16384

PRESSURE_CHILD = r"""
import re, subprocess, sys, time
import numpy as np
floor, cap = (int(v) for v in sys.argv[1:3])
def avail():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    get = lambda k: int(re.search(k + r":\s+(\d+)", out).group(1)) * 16384
    return sum(get(k) for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
rng = np.random.default_rng(0)
held, step = [], 256 << 20
while sum(a.nbytes for a in held) + step <= cap:
    if avail() - step < floor:
        break
    a = np.empty(step, dtype=np.uint8)
    a[:] = rng.integers(0, 256, step, dtype=np.uint8)
    held.append(a)
print(f"HELD {sum(a.nbytes for a in held)}", flush=True)
while True:
    time.sleep(5)
    for a in held:
        a[::16384] += 1
"""


def vm_stat() -> dict:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    values = {}
    for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable",
                "Pages wired down", "Pages occupied by compressor", "Swapouts"):
        match = re.search(re.escape(key) + r":\s+(\d+)", out)
        values[key] = int(match.group(1)) if match else 0
    values["avail_gib"] = sum(values[k] for k in ("Pages free", "Pages inactive",
                                                   "Pages speculative", "Pages purgeable")) * PAGE / 2**30
    return values


def rusage(pid: int) -> dict | None:
    from mlx2.runtime.os_memory import _LIBPROC, _RUSAGE_INFO_V4, _RUsageInfoV4

    info = _RUsageInfoV4()
    pointer = ctypes.cast(ctypes.byref(info), ctypes.POINTER(ctypes.c_void_p))
    if _LIBPROC is None or _LIBPROC.proc_pid_rusage(pid, _RUSAGE_INFO_V4, pointer) != 0:
        return None
    return {"pageins": int(info.ri_pageins), "resident_gib": info.ri_resident_size / 2**30,
            "footprint_gib": info.ri_phys_footprint / 2**30}


class Guard(threading.Thread):
    def __init__(self, swap_abort: int, floor_gib: float):
        super().__init__(daemon=True)
        self.base = vm_stat()["Swapouts"] * PAGE
        self.swap_abort, self.floor_gib = swap_abort, floor_gib
        self.victims: list[subprocess.Popen] = []
        self.tripped = None
        self.min_avail_gib = float("inf")
        self.stop = threading.Event()

    def run(self):
        while not self.stop.wait(1.0):
            host = vm_stat()
            rise = host["Swapouts"] * PAGE - self.base
            self.min_avail_gib = min(self.min_avail_gib, host["avail_gib"])
            reason = None
            if rise >= self.swap_abort:
                reason = f"swapouts +{rise >> 20} MiB"
            elif host["avail_gib"] < self.floor_gib:
                reason = f"free+reclaimable {host['avail_gib']:.1f} GiB < floor"
            if reason and self.tripped is None:
                self.tripped = reason
                print(f"GUARD ABORT: {reason}", flush=True)
                for proc in self.victims:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except OSError:
                        pass


def chat(base: str, model: str, words: int = 400, timeout: float = 600) -> dict:
    text = " ".join(f"w{i % 89}" for i in range(words))
    body = {"model": model, "max_tokens": 8, "temperature": 0, "stream": False,
            "enable_thinking": False,
            "messages": [{"role": "user", "content": f"[{uuid.uuid4().hex}] Summarize in one line: {text}"}]}
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    return {"total_s": round(time.perf_counter() - start, 3),
            "prompt_tokens": (payload.get("usage") or {}).get("prompt_tokens")}


def run_trial(args, name: str, src: Path, inject_at: float | None, out_dir: Path) -> dict:
    log_path = out_dir / f"{name}.server.log"
    fault_path = out_dir / f"{name}.fault.log"
    supports_fault_log = (src / "mlx2" / "exit_trace.py").exists()
    command = [sys.executable, "-u", "-m", "mlx2.server", "--model", args.model,
               "--host", "127.0.0.1", "--port", str(args.port)]
    if supports_fault_log:
        command += ["--fault-log", str(fault_path)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src)
    env.pop("MLX2_FAULT_LOG", None)
    guard = Guard(args.swap_abort_mib << 20, args.floor_gib)
    trial = {"trial": name, "src": str(src), "inject_sigterm_at_s": inject_at,
             "command": command, "samples": [], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(log_path, "w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                  start_new_session=True)
    trial["server_pid"] = server.pid
    guard.victims.append(server)
    guard.start()
    base = f"http://127.0.0.1:{args.port}"
    pressure = None
    try:
        deadline = time.monotonic() + 900
        while True:
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                pass
            if server.poll() is not None or time.monotonic() > deadline or guard.tripped:
                raise RuntimeError("server failed to start")
            time.sleep(1)
        with urllib.request.urlopen(base + "/v1/status", timeout=60) as r:
            status = json.loads(r.read())
        model_id = status.get("model")
        trial["route"] = (status.get("settings") or {}).get("route")
        guard.base = vm_stat()["Swapouts"] * PAGE  # swap tripwire counts from ready
        trial["baseline"] = [chat(base, model_id) for _ in range(3)]
        pressure = subprocess.Popen(
            [sys.executable, "-c", PRESSURE_CHILD, str(int(args.floor_gib + 4) << 30),
             str(args.cap_gib << 30)],
            stdout=subprocess.PIPE, text=True, start_new_session=True)
        guard.victims.insert(0, pressure)
        held = pressure.stdout.readline().strip()
        trial["pressure_held_gib"] = int(held.split()[1]) / 2**30 if held.startswith("HELD") else None
        idle_start = time.monotonic()
        injected = False
        while time.monotonic() - idle_start < args.idle_s and not guard.tripped:
            t = round(time.monotonic() - idle_start, 1)
            if inject_at is not None and not injected and t >= inject_at:
                os.kill(server.pid, signal.SIGTERM)  # what pkill -f sends
                injected = True
                trial["injected_at_s"] = t
            if server.poll() is not None:
                trial["died"] = {"t_s": t, "returncode": server.returncode}
                break
            host = vm_stat()
            trial["samples"].append({"t_s": t, **(rusage(server.pid) or {"gone": True}),
                                     "host_avail_gib": round(host["avail_gib"], 2)})
            time.sleep(2)
        if server.poll() is None and not guard.tripped:
            trial["after_idle_under_pressure"] = chat(base, model_id)
            os.killpg(pressure.pid, signal.SIGKILL)
            pressure.wait()
            pressure = None
            time.sleep(3)
            trial["after_release"] = chat(base, model_id)
            trial["survived_idle"] = True
            server.send_signal(signal.SIGTERM)  # ordinary shutdown path, now logged
        else:
            trial["survived_idle"] = False
        trial["returncode"] = server.wait(timeout=120)
    except Exception as error:  # noqa: BLE001 - keep the partial receipt
        trial["error"] = repr(error)
    finally:
        guard.stop.set()
        trial["guard_tripped"] = guard.tripped
        trial["min_host_avail_gib"] = round(guard.min_avail_gib, 2)
        trial["swapouts_rise_mib"] = (vm_stat()["Swapouts"] * PAGE - guard.base) >> 20
        for proc in (pressure, server):
            if proc is None or proc.poll() is not None:
                continue
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
        trial.setdefault("returncode", server.poll())
    lines = log_path.read_text(errors="ignore").splitlines()
    trial["log_tail"] = [l for l in lines if "GET /" not in l][-6:]
    trial["log_has_signal_line"] = any("received SIGTERM" in l for l in lines)
    trial["log_has_exit_line"] = any("mlx2 server exiting:" in l for l in lines)
    if fault_path.exists():
        trial["fault_log"] = fault_path.read_text(errors="ignore").splitlines()[:5]
    return trial


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--trials", required=True,
                        help="comma list of name:src_dir:inject_seconds_or_-")
    parser.add_argument("--idle-s", type=float, default=180)
    parser.add_argument("--cap-gib", type=int, default=24)
    parser.add_argument("--floor-gib", type=float, default=8)
    parser.add_argument("--swap-abort-mib", type=int, default=64)
    parser.add_argument("--port", type=int, default=8394)
    parser.add_argument("--out", required=True)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        print("refusing: pass --i-own-the-gpu under the GPU lock wrapper", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {"args": vars(args), "host_before": vm_stat(), "trials": []}
    for spec in args.trials.split(","):
        name, src, inject = spec.split(":")
        trial = run_trial(args, name, Path(src).resolve(),
                          None if inject == "-" else float(inject), out.parent)
        record["trials"].append(trial)
        out.write_text(json.dumps(record, indent=1))
        print(json.dumps({k: v for k, v in trial.items() if k != "samples"}, indent=1), flush=True)
        if trial.get("guard_tripped"):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
