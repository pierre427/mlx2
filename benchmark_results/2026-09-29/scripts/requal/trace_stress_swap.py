#!/usr/bin/env python3
"""Sample host swap and memory while an existing stress queue owns the GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def swapouts() -> int | None:
    result = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False)
    match = re.search(r"Swapouts:\s+(\d+)", result.stdout)
    return int(match.group(1)) if match else None


def server_sample() -> dict:
    for process in psutil.process_iter(["pid", "cmdline", "memory_info"]):
        try:
            argv = process.info["cmdline"] or []
            if "-m" in argv and argv[argv.index("-m") + 1] == "mlx2.server":
                return {"pid": process.pid, "rss_bytes": process.info["memory_info"].rss}
        except (IndexError, psutil.Error):
            continue
    return {}


def ready() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8397/health", timeout=1) as response:
            return json.load(response).get("status") == "ok"
    except (OSError, urllib.error.HTTPError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.tag):
        parser.error("tag must be filename-safe")
    output = HERE / args.host_label / args.model
    output.mkdir(parents=True, exist_ok=True)
    trace = output / f"swap-trace-{args.tag}.jsonl"
    log = output / f"queue-{args.tag}.log"
    if trace.exists() or log.exists():
        parser.error("trace or queue log already exists")
    command = [sys.executable, str(HERE / "queue_stress.py"), "--host-label", args.host_label,
               "--only", args.model]
    source_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    with trace.open("w") as stream, log.open("w") as queue_log:
        stream.write(json.dumps({"schema": "mlx2.requal.stress-swap-trace.v1",
                                 "source_head": source_head, "model": args.model,
                                 "host": args.host_label, "command": command,
                                 "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}) + "\n")
        queue = subprocess.Popen(command, cwd=ROOT, env=env, stdout=queue_log,
                                 stderr=subprocess.STDOUT)
        while True:
            row = {"at": time.time(), "swapouts": swapouts(),
                   "host_available_bytes": int(psutil.virtual_memory().available),
                   "server_ready": ready(), **server_sample()}
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            if queue.poll() is not None:
                return queue.returncode
            time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
