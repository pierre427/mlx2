#!/usr/bin/env python3
"""Launch and cleanly stop the isolated PLE/incremental serving smoke.

This wrapper owns only the child service lifecycle.  It refuses native work
without an explicit acknowledgement; the caller must hold the CPG lease and
both shared GPU locks for the complete invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import smoke_qwen4_ple_incremental_composition as composition

SCHEMA = "mlx2.qwen4-ple-incremental-serving-lifecycle.v1"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--model", type=Path, required=True)
    value.add_argument("--out", type=Path, required=True)
    value.add_argument("--python", default=sys.executable)
    value.add_argument("--host", default="127.0.0.1")
    value.add_argument("--port", type=int, default=8398)
    value.add_argument("--startup-timeout", type=float, default=1800)
    value.add_argument("--request-timeout", type=float, default=900)
    value.add_argument("--shutdown-timeout", type=float, default=180)
    value.add_argument("--base-characters", type=int, default=8192)
    value.add_argument("--i-own-the-gpu", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    return value


def command(args) -> list[str]:
    namespace = SimpleNamespace(
        python=args.python,
        model=args.model,
        host=args.host,
        port=args.port,
    )
    return [
        *composition.server_command(namespace),
        "--max-context",
        "16384",
        "--cache-bytes",
        str(4 << 30),
    ]


def _health(base: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(base + "/health", timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False


def _stop(process: subprocess.Popen, timeout: float) -> dict:
    result = {"was_running": process.poll() is None, "terminated": False, "killed": False}
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=timeout)
            result["terminated"] = True
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
            result["killed"] = True
    result["returncode"] = process.returncode
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    server_command = command(args)
    plan = {
        "schema": SCHEMA,
        "model": str(args.model),
        "server_command": server_command,
        "execution_policy": str(composition.POLICY_PATH),
        "whole_table_warming": False,
        "will_execute": bool(args.i_own_the_gpu and not args.dry_run),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.i_own_the_gpu:
        print("refusing: --i-own-the-gpu is required under CPG and paired locks", file=sys.stderr)
        return 2
    if not (args.model / "model.safetensors.index.json").is_file():
        print("refusing: model is not a local indexed artifact", file=sys.stderr)
        return 2
    with socket.socket() as probe:
        probe.settimeout(0.2)
        if probe.connect_ex((args.host, args.port)) == 0:
            print(f"refusing: {args.host}:{args.port} is already in use", file=sys.stderr)
            return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(args.out.suffix + ".server.log")
    client_path = args.out.with_suffix(args.out.suffix + ".client.json")
    log = log_path.open("w")
    environment = dict(os.environ)
    environment.update(composition.server_environment())
    process = subprocess.Popen(
        server_command,
        cwd=ROOT,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    result = {**plan, "will_execute": True, "pid": process.pid, "server_log": str(log_path)}
    failures = []
    base = f"http://{args.host}:{args.port}"
    client_rc = None
    try:
        deadline = time.monotonic() + args.startup_timeout
        while not _health(base, min(args.request_timeout, 5)):
            if process.poll() is not None:
                raise RuntimeError(f"server exited during startup rc={process.returncode}")
            if time.monotonic() >= deadline:
                raise RuntimeError("server startup timed out")
            time.sleep(2)
        client_rc = composition.main(
            [
                "--model",
                str(args.model),
                "--out",
                str(client_path),
                "--live-url",
                base,
                "--base-characters",
                str(args.base_characters),
                "--request-timeout",
                str(args.request_timeout),
                "--i-own-request-route",
            ]
        )
        if client_rc != 0:
            failures.append(f"composition client rc={client_rc}")
    except Exception as error:  # noqa: BLE001 - persist diagnostics and clean up
        failures.append(f"{type(error).__name__}: {error}")
    finally:
        shutdown = _stop(process, args.shutdown_timeout)
        log.close()

    if not shutdown["was_running"] or not shutdown["terminated"] or shutdown["killed"]:
        failures.append(f"unclean server shutdown: {shutdown}")
    result.update(
        {
            "client_receipt": str(client_path),
            "client_returncode": client_rc,
            "shutdown": shutdown,
            "failures": failures,
            "go": not failures,
            "server_log_tail": log_path.read_text(errors="replace")[-8000:],
        }
    )
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"go": result["go"], "failures": failures, "out": str(args.out)}, indent=2))
    return 0 if result["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
