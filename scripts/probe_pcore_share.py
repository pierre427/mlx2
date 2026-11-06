"""Measure where a process spends its CPU time while it decodes (macOS).

Reads ``proc_pid_rusage(RUSAGE_INFO_V6)`` for a pid before and after one
streamed chat completion and reports the delta:

* ``pcore_share``: ``(ri_user_ptime + ri_system_ptime) / (user + system)``,
  the share of CPU time on perflevel0 cores. On an M5 Max perflevel0 is the
  six-core "Super" cluster and perflevel1 is twelve "Performance" cores (no
  efficiency cores), so a low share there is not E-core placement.
* the per-QoS CPU-time buckets (a launchd- or taskpolicy-demoted process shows
  up in ``background``/``utility``/``maintenance``),
* CPU ms per emitted token and decode tok/s from the client side.

Read-only on the target: it never changes the scheduling of the process it
measures. Usage::

  python scripts/probe_pcore_share.py --pid <server pid> --url http://127.0.0.1:PORT \
      --model <id> --max-tokens 512 --out arm.json
  python scripts/probe_pcore_share.py --self-test   # burns CPU in-process
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request

RUSAGE_INFO_V6 = 6

_V4_NAMES = (
    "user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size "
    "resident_size phys_footprint proc_start_abstime proc_exit_abstime "
    "child_user_time child_system_time child_pkg_idle_wkups child_interrupt_wkups "
    "child_pageins child_elapsed_abstime diskio_bytesread diskio_byteswritten "
    "cpu_time_qos_default cpu_time_qos_maintenance cpu_time_qos_background "
    "cpu_time_qos_utility cpu_time_qos_legacy cpu_time_qos_user_initiated "
    "cpu_time_qos_user_interactive billed_system_time serviced_system_time "
    "logical_writes lifetime_max_phys_footprint instructions cycles billed_energy "
    "serviced_energy interval_max_phys_footprint runnable_time"
).split()
_V6_NAMES = (
    "flags user_ptime system_ptime pinstructions pcycles energy_nj penergy_nj "
    "secure_time_in_system secure_ptime_in_system neural_footprint "
    "lifetime_max_neural_footprint interval_max_neural_footprint"
).split()


class RUsageInfoV6(ctypes.Structure):
    _fields_ = (
        [("uuid", ctypes.c_uint8 * 16)]
        + [(name, ctypes.c_uint64) for name in _V4_NAMES + _V6_NAMES]
        + [("reserved", ctypes.c_uint64 * 9)]
    )


assert ctypes.sizeof(RUsageInfoV6) == 464, ctypes.sizeof(RUsageInfoV6)

_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
_libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_libproc.proc_pid_rusage.restype = ctypes.c_int


def _timebase_ns() -> float:
    class TB(ctypes.Structure):
        _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]

    tb = TB()
    ctypes.CDLL("/usr/lib/libSystem.B.dylib").mach_timebase_info(ctypes.byref(tb))
    return tb.numer / tb.denom


TICK_NS = _timebase_ns()


def sample(pid: int) -> dict:
    info = RUsageInfoV6()
    if _libproc.proc_pid_rusage(pid, RUSAGE_INFO_V6, ctypes.byref(info)) != 0:
        raise OSError(ctypes.get_errno(), f"proc_pid_rusage({pid}) failed")
    return {name: int(getattr(info, name)) for name in _V4_NAMES + _V6_NAMES}


def delta(before: dict, after: dict, tokens: int | None, wall_s: float) -> dict:
    d = {k: after[k] - before[k] for k in before}
    cpu = d["user_time"] + d["system_time"]
    pcpu = d["user_ptime"] + d["system_ptime"]
    qos_keys = [k for k in d if k.startswith("cpu_time_qos_")]
    qos_total = sum(d[k] for k in qos_keys) or 1
    out = {
        "wall_s": round(wall_s, 3),
        "cpu_ms": round(cpu * TICK_NS / 1e6, 1),
        "cpu_utilization": round(cpu * TICK_NS / 1e9 / wall_s, 3) if wall_s else None,
        "pcore_share": round(pcpu / cpu, 4) if cpu else None,
        "pcycles_share": round(d["pcycles"] / d["cycles"], 4) if d["cycles"] else None,
        "qos_share": {k[len("cpu_time_qos_"):]: round(d[k] / qos_total, 4) for k in qos_keys if d[k]},
        "runnable_ms": round(d["runnable_time"] * TICK_NS / 1e6, 1),
        "raw_delta": d,
    }
    if tokens:
        out["tokens"] = tokens
        out["cpu_ms_per_token"] = round(cpu * TICK_NS / 1e6 / tokens, 3)
        out["decode_tok_s"] = round(tokens / wall_s, 2)
    return out


def stream_completion(url: str, model: str, prompt: str, max_tokens: int) -> tuple[int, float, float, str]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    first = last = None
    chunks = 0
    text = []
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            payload = json.loads(line[5:])
            for choice in payload.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    chunks += 1
                    text.append(piece)
    return chunks, first or 0.0, last or 0.0, "".join(text)


def run_arm(args) -> dict:
    prompt = open(args.prompt_file).read() if args.prompt_file else (
        "Write a detailed, factual explanation of how a four-stroke gasoline engine "
        "works, covering intake, compression, power and exhaust, valve timing, "
        "ignition, lubrication and cooling. Use plain prose paragraphs."
    )
    # The window covers the whole request; with a warmed APCv2 prefix the
    # prefill share is small, and client_decode_tok_s is first-to-last token.
    before_total = sample(args.pid)
    t0 = time.perf_counter()
    chunks, first, last, text = stream_completion(args.url, args.model, prompt, args.max_tokens)
    t1 = time.perf_counter()
    after_total = sample(args.pid)
    return {
        "whole_request": delta(before_total, after_total, chunks, t1 - t0),
        "client_decode_tok_s": round((chunks - 1) / (last - first), 2) if chunks > 1 and last > first else None,
        "text_head": text[:160],
    }


def self_test() -> dict:
    pid = os.getpid()
    before = sample(pid)
    t0 = time.perf_counter()
    x = 0
    while time.perf_counter() - t0 < 1.0:
        x += 1
    return delta(before, sample(pid), None, time.perf_counter() - t0)


def host_receipt() -> dict:
    def sysctl(name):
        try:
            return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()
        except OSError:
            return None

    return {
        "cpu": sysctl("machdep.cpu.brand_string"),
        "perflevel0": f'{sysctl("hw.perflevel0.name")} x{sysctl("hw.perflevel0.logicalcpu")}',
        "perflevel1": f'{sysctl("hw.perflevel1.name")} x{sysctl("hw.perflevel1.logicalcpu")}',
        "macos": platform.mac_ver()[0],
        "tick_ns": TICK_NS,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int)
    ap.add_argument("--url")
    ap.add_argument("--model")
    ap.add_argument("--prompt-file")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    result: dict = {"host": host_receipt(), "label": args.label}
    if args.self_test:
        result["self_test"] = self_test()
    else:
        result["runs"] = [run_arm(args) for _ in range(args.repeats)]
    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "x") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
