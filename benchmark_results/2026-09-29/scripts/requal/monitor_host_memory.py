#!/usr/bin/env python3
"""Record host and serving memory while a separately owned run is active."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

import psutil


def swapouts() -> int | None:
    result = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False)
    match = re.search(r"Swapouts:\s+(\d+)", result.stdout)
    return int(match.group(1)) if match else None


def status(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            data = json.load(response)
    except (OSError, ValueError):
        return {}
    apc = data.get("apcv2") or {}
    cow = apc.get("cow") or {}
    return {
        "healthy": data.get("healthy"),
        "metal_active_bytes": data.get("metal_active_bytes"),
        "metal_peak_bytes": data.get("metal_peak_bytes"),
        "headroom_bytes": data.get("headroom_bytes"),
        "memory_waiting": data.get("memory_waiting"),
        "apc_hits": apc.get("hits"),
        "apc_cached_tokens": apc.get("cached_tokens"),
        "apc_cow_sources": cow.get("sources"),
        "apc_cow_source_bytes": cow.get("source_bytes"),
    }


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True, help="owned qualification runner PID")
    parser.add_argument("--url", required=True, help="served /v1/status URL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("interval must be at least one second")
    if args.output.exists():
        parser.error("output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        stream.write(json.dumps({
            "schema": "mlx2.requal.host-memory-trace.v1",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runner_pid": args.pid,
            "status_url": args.url,
        }) + "\n")
        while alive(args.pid):
            stream.write(json.dumps({
                "at": time.time(),
                "swapouts": swapouts(),
                "host_available_bytes": int(psutil.virtual_memory().available),
                **status(args.url),
            }, sort_keys=True) + "\n")
            stream.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
