#!/usr/bin/env python3
"""Do an idle mlx2 server's weights stay resident under memory pressure? (GPU)

Design input: Inco Splash (Apache-2.0, f786bed) PR #148 keeps model weights in
a Metal residency set so that memory pressure between requests cannot drop
them.  MLX does the same when a wired limit is set (MTLResidencySet with a
standing requestResidency); this measures whether mlx2's serving process is
covered, per route.  No Splash code is used.

Per arm (one server process each, run sequentially):
  1. start ``mlx2.server`` for the model with the arm's route flags;
  2. three fresh-prompt requests (a new nonce each, so every one prefills and
     touches every weight): baseline TTFT;
  3. start a pressure child that fills anonymous memory with incompressible
     bytes in 256 MiB steps until host free pages fall to ``--floor-gib`` or
     it holds ``--cap-gib``; idle ``--idle-s`` seconds with the server idle,
     sampling the server's rusage (resident, wired, footprint, pageins);
  4. one fresh-prompt request while the pressure is still held: TTFT and the
     server's pageins over the idle window and the request;
  5. release the pressure; one more request.

Safety: a watchdog samples vm_stat Swapouts.  It allows ``--load-swap-mib``
through model load and aborts (pressure child first, then the server) on a
rise of ``--swap-abort-mib`` after that, and the pressure child also stops
growing when the compressor grows by ``--compressor-stop-mib``.

Refuses to run without ``--i-own-the-gpu``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from measure_cached_replay import stream_chat  # noqa: E402
from qualify_interior_checkpoints import server_env  # noqa: E402

PAGE = 16384

PRESSURE_CHILD = r"""
import re, subprocess, sys, time
import numpy as np
floor, cap, comp_stop = (int(v) for v in sys.argv[1:4])
def vm():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    get = lambda k: int(re.search(k + r":\s+(\d+)", out).group(1)) * 16384
    return get("Pages free"), get("Pages occupied by compressor")
_, comp0 = vm()
rng = np.random.default_rng(0)
held, step = [], 256 << 20
while sum(a.nbytes for a in held) + step <= cap:
    free, comp = vm()
    if free - step < floor or comp - comp0 > comp_stop:
        break
    a = np.empty(step, dtype=np.uint8)
    a[:] = rng.integers(0, 256, step, dtype=np.uint8)
    held.append(a)
print(f"HELD {sum(a.nbytes for a in held)}", flush=True)
while True:
    time.sleep(5)
    for a in held:  # keep them recently used, so the kernel prefers others
        a[::16384] += 1
"""


def vm_stat() -> dict:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    values = {}
    for key in ("Pages free", "Pages wired down", "Pages occupied by compressor",
                "Pageins", "Swapouts", "Swapins", "Pages active", "Pages inactive"):
        match = re.search(re.escape(key) + r":\s+(\d+)", out)
        values[key] = int(match.group(1)) if match else None
    return values


def rusage(pid: int) -> dict | None:
    from mlx2.runtime.os_memory import _LIBPROC, _RUSAGE_INFO_V4, _RUsageInfoV4

    info = _RUsageInfoV4()
    pointer = ctypes.cast(ctypes.byref(info), ctypes.POINTER(ctypes.c_void_p))
    if _LIBPROC is None or _LIBPROC.proc_pid_rusage(pid, _RUSAGE_INFO_V4, pointer) != 0:
        return None
    return {
        "pageins": int(info.ri_pageins),
        "wired_bytes": int(info.ri_wired_size),
        "resident_bytes": int(info.ri_resident_size),
        "footprint_bytes": int(info.ri_phys_footprint),
    }


class Watch(threading.Thread):
    def __init__(self, limit: int):
        super().__init__(daemon=True)
        self.base = vm_stat()["Swapouts"] * PAGE
        self.limit = limit
        self.victims: list[subprocess.Popen] = []
        self.tripped = None
        self.peak = 0
        self.stop = threading.Event()

    def rebase(self, limit: int):
        self.base = vm_stat()["Swapouts"] * PAGE
        self.limit = limit
        self.peak = 0

    def run(self):
        while not self.stop.wait(1.0):
            rise = vm_stat()["Swapouts"] * PAGE - self.base
            self.peak = max(self.peak, rise)
            if rise >= self.limit and self.tripped is None:
                self.tripped = rise
                print(f"ABORT: swapouts +{rise >> 20} MiB", flush=True)
                for proc in self.victims:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except OSError:
                        pass


def fresh_prompt(words: int) -> list:
    nonce = uuid.uuid4().hex
    text = " ".join(f"w{i % 89}" for i in range(words))
    return [{"role": "user", "content": f"[{nonce}] Summarize in one line: {text}"}]


def run_arm(args, name: str, route_flags: list[str], out_dir: Path) -> dict:
    watch = Watch(args.load_swap_mib << 20)
    watch.start()
    host_before = vm_stat()
    log_path = out_dir / f"server-{name}.log"
    command = [sys.executable, "-u", "-m", "mlx2.server", "--model", args.model,
               "--host", "127.0.0.1", "--port", str(args.port), *route_flags]
    env = server_env()
    # MLX logs each MTLResidencySet it creates.  Sets beyond set 0 exist only
    # once wired bytes exceed one set's cap (5% of the working set), so the
    # count shows whether the weights were admitted to residency at all.
    env["MLX_RESIDENCY_DEBUG"] = "1"
    with open(log_path, "w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                  env=env, start_new_session=True)

    def residency_sets():
        return [line.strip() for line in log_path.read_text(errors="ignore").splitlines()
                if line.startswith("[residency]")]
    watch.victims.append(server)
    base = f"http://127.0.0.1:{args.port}"
    arm = {"arm": name, "command": command, "samples": []}
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
            if server.poll() is not None or time.monotonic() > deadline or watch.tripped:
                raise RuntimeError(f"server failed to start; see {log_path}")
            time.sleep(2)
        with urllib.request.urlopen(base + "/v1/status", timeout=60) as r:
            status = json.loads(r.read())
        model_id = status.get("model")
        arm["route"] = (status.get("settings") or {}).get("route")
        arm["weight_residency"] = status.get("weight_residency")
        arm["load_swap_rise_mib"] = watch.peak >> 20
        watch.rebase(args.swap_abort_mib << 20)
        host_ready = vm_stat()
        arm["host_wired_delta_gib_at_ready"] = (
            (host_ready["Pages wired down"] - host_before["Pages wired down"]) * PAGE / 2**30
        )
        arm["rusage_ready"] = rusage(server.pid)
        arm["residency_sets_at_ready"] = residency_sets()

        def request():
            body = {"model": model_id, "messages": fresh_prompt(args.prompt_words),
                    "max_tokens": 8, "temperature": 0, "stream": True,
                    "stream_options": {"include_usage": True}, "enable_thinking": False}
            before = rusage(server.pid)
            result = stream_chat(base, body, 600)
            after = rusage(server.pid)
            return {"ttft_s": result["ttft_s"], "total_s": result["total_s"],
                    "prompt_tokens": result["prompt_tokens"],
                    "server_pageins": after["pageins"] - before["pageins"]}

        arm["baseline"] = [request() for _ in range(3)]
        arm["rusage_baseline"] = rusage(server.pid)
        host_pre = vm_stat()
        pressure = subprocess.Popen(
            [sys.executable, "-c", PRESSURE_CHILD, str(args.floor_gib << 30),
             str(args.cap_gib << 30), str(args.compressor_stop_mib << 20)],
            stdout=subprocess.PIPE, text=True, start_new_session=True,
        )
        watch.victims.insert(0, pressure)
        held_line = pressure.stdout.readline().strip()
        arm["pressure_held_gib"] = (
            int(held_line.split()[1]) / 2**30 if held_line.startswith("HELD") else None
        )
        idle_start = time.monotonic()
        before_idle = rusage(server.pid)
        while time.monotonic() - idle_start < args.idle_s and not watch.tripped:
            if server.poll() is not None:
                arm["server_died"] = {
                    "returncode": server.returncode,
                    "t_s": round(time.monotonic() - idle_start, 1),
                    "host": vm_stat(),
                }
                raise RuntimeError(f"server exited {server.returncode} during idle")
            sample = {"t_s": round(time.monotonic() - idle_start, 1), **(rusage(server.pid) or {})}
            host = vm_stat()
            sample["host_free_gib"] = host["Pages free"] * PAGE / 2**30
            sample["host_compressor_gib"] = host["Pages occupied by compressor"] * PAGE / 2**30
            arm["samples"].append(sample)
            time.sleep(10)
        after_idle = rusage(server.pid)
        arm["idle_server_pageins"] = after_idle["pageins"] - before_idle["pageins"]
        arm["idle_resident_delta_mib"] = (after_idle["resident_bytes"] - before_idle["resident_bytes"]) >> 20
        arm["after_idle_under_pressure"] = request()
        host_post = vm_stat()
        arm["host_during_pressure"] = {
            "free_gib": host_post["Pages free"] * PAGE / 2**30,
            "compressor_delta_gib": (host_post["Pages occupied by compressor"]
                                     - host_pre["Pages occupied by compressor"]) * PAGE / 2**30,
            "host_pageins_delta": host_post["Pageins"] - host_pre["Pageins"],
            "swapouts_delta_mib": (host_post["Swapouts"] - host_pre["Swapouts"]) * PAGE >> 20,
        }
        os.killpg(pressure.pid, signal.SIGKILL)
        pressure.wait()
        pressure = None
        time.sleep(5)
        arm["after_release"] = request()
        arm["swap_peak_rise_mib_after_load"] = watch.peak >> 20
        arm["residency_sets_at_end"] = len(residency_sets())
        base_ttft = statistics.median(r["ttft_s"] for r in arm["baseline"][1:])
        arm["ttft_ratio_after_idle"] = arm["after_idle_under_pressure"]["ttft_s"] / base_ttft
    except Exception as error:  # noqa: BLE001 - keep the partial receipt
        arm["error"] = repr(error)
        arm["server_returncode"] = server.poll()
    finally:
        arm["aborted_swap_rise_mib"] = None if watch.tripped is None else watch.tripped >> 20
        watch.stop.set()
        for proc in (pressure, server):
            if proc is None:
                continue
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
    return arm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--arms", default="default:,prompt_lookup:--prompt-lookup",
                        help="comma list of name:space-separated-server-flags")
    parser.add_argument("--idle-s", type=int, default=180)
    parser.add_argument("--prompt-words", type=int, default=400)
    parser.add_argument("--floor-gib", type=int, default=3)
    parser.add_argument("--cap-gib", type=int, default=24)
    parser.add_argument("--compressor-stop-mib", type=int, default=1024)
    parser.add_argument("--load-swap-mib", type=int, default=1536)
    parser.add_argument("--swap-abort-mib", type=int, default=256)
    parser.add_argument("--port", type=int, default=8393)
    parser.add_argument("--out", required=True)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        print("refusing: pass --i-own-the-gpu under the GPU lock wrapper", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {"args": vars(args), "arms": []}
    for spec in args.arms.split(","):
        name, _, flags = spec.partition(":")
        arm = run_arm(args, name, flags.split(), out.parent)
        record["arms"].append(arm)
        out.write_text(json.dumps(record, indent=1))
        print(json.dumps({k: v for k, v in arm.items() if k != "samples"}, indent=1), flush=True)
        if arm.get("aborted_swap_rise_mib") is not None or arm.get("error"):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
