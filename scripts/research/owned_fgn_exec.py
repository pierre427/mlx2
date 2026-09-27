#!/usr/bin/env python3
"""Run one FGN model gate under cpg_job ownership, restoring pinned :8282."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


OWNER = Path("/Users/Shared/mlxuag/gpu.lock/owner.json")
LOCK = Path("/tmp/gpu.lock")
DOMAIN = f"gui/{os.getuid()}"
LABEL = "com.example.mlx2-flash-next"
PLIST = Path.home() / "Library/LaunchAgents/com.example.mlx2-flash-next.plist"
URL = "http://127.0.0.1:8282/v1/models"
GPU_TASK = "90e05f6428a84789b58a45b305be40b6"


def service_loaded() -> bool:
    return subprocess.run(
        ["/bin/launchctl", "print", f"{DOMAIN}/{LABEL}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def model_identity() -> str | None:
    try:
        with urlopen(URL, timeout=3) as response:
            if response.status != 200:
                return None
            rows = json.load(response).get("data", [])
    except Exception:
        return None
    loaded = [row["id"] for row in rows if row.get("loaded") is True]
    return loaded[0] if len(loaded) == 1 else None


def port_closed() -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", 8282)) != 0


def restore(expected_model: str) -> None:
    if not service_loaded():
        for attempt in range(5):
            result = subprocess.run(
                ["/bin/launchctl", "bootstrap", DOMAIN, str(PLIST)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 or service_loaded():
                break
            if attempt == 4:
                raise RuntimeError(f"pinned service bootstrap failed: {result.stderr.strip()}")
            time.sleep(2)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if service_loaded() and model_identity() == expected_model:
            print(json.dumps({"type": "restored", "model": expected_model}), flush=True)
            return
        time.sleep(5)
    raise RuntimeError(f"pinned service did not restore loaded model {expected_model}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: owned_fgn_exec.py COMMAND [ARGS ...]")
    owner = json.loads(OWNER.read_text())
    if owner.get("pid") != os.getppid() or owner.get("cpg_task") != GPU_TASK:
        raise RuntimeError("parent does not own the shared CPG GPU lease and host lock")
    if not owner.get("cpg_generation"):
        raise RuntimeError("owner receipt has no CPG lease generation")

    def interrupted(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)

    with LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not service_loaded():
            raise RuntimeError("expected pinned :8282 launchd service is not loaded")
        expected_model = model_identity()
        if expected_model is None:
            raise RuntimeError("pinned :8282 has no single loaded model identity")
        print(json.dumps({"type": "preflight", "owner": owner,
                          "model": expected_model, "tmp_lock": "fcntl exclusive"}), flush=True)
        try:
            subprocess.run(["/bin/launchctl", "bootout", f"{DOMAIN}/{LABEL}"], check=True)
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if not service_loaded() and port_closed():
                    break
                time.sleep(1)
            else:
                raise RuntimeError("pinned :8282 did not fully quiesce")
            print(json.dumps({"type": "quiesced", "model": expected_model}), flush=True)
            subprocess.run(sys.argv[1:], check=True)
        finally:
            restore(expected_model)


if __name__ == "__main__":
    main()
