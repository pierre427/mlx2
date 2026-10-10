#!/usr/bin/env python3
"""Launch one bounded mlx2 server profile, run its ladder, then retire it."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
THERMAL = HERE / "thermal_ladder.py"


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON file {path}: {error}") from error


def reserve_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def expand_server_argv(template: list[str], port: int) -> list[str]:
    if not template or any(not isinstance(part, str) for part in template):
        raise ValueError("server argv must be a nonempty JSON string array")
    if sum(part.count("@PORT@") for part in template) != 1:
        raise ValueError("server argv must contain exactly one @PORT@ placeholder")
    command = [part.replace("@PORT@", str(port)) for part in template]
    if (
        "--host" not in command
        or command.index("--host") + 1 >= len(command)
        or command[command.index("--host") + 1] != "127.0.0.1"
    ):
        raise ValueError("server must bind explicitly to 127.0.0.1")
    if (
        "--port" not in command
        or command.index("--port") + 1 >= len(command)
        or command[command.index("--port") + 1] != str(port)
    ):
        raise ValueError("server must use the allocated port through --port @PORT@")
    return command


def ladder_argv(options: dict, *, url: str, output: Path, server_pid: int) -> list[str]:
    if not isinstance(options, dict):
        raise TypeError("ladder options must be a JSON object")
    forbidden = {"url", "output", "server-command", "server-argv"} & set(options)
    if forbidden:
        raise ValueError(
            f"ladder options cannot override wrapper-owned fields: {sorted(forbidden)}"
        )
    allowed = {
        "model",
        "model-id",
        "route",
        "artifact-identity",
        "max-context",
        "cache-bytes",
        "apc-persistence",
        "apc-persist-on-shutdown",
        "max-lanes",
        "max-inflight",
        "prefill-step",
        "prefill-policy",
        "mtp-policy",
        "draft-loop-policy",
        "performance-mode",
        "timeout",
        "max-tokens",
        "runs",
        "wide",
        "wide-max-context",
        "max-length",
        "min-length",
        "max-retries",
        "foreign-cpu-threshold",
        "swapout-tolerance-pages",
    }
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"unsupported thermal ladder options: {sorted(unknown)}")
    required = {
        "model",
        "model-id",
        "route",
        "artifact-identity",
        "max-context",
        "cache-bytes",
        "apc-persistence",
        "apc-persist-on-shutdown",
        "max-lanes",
        "max-inflight",
        "prefill-step",
        "prefill-policy",
        "mtp-policy",
        "draft-loop-policy",
    }
    missing = required - set(options)
    if missing:
        raise ValueError(f"ladder options missing profile fields: {sorted(missing)}")
    argv = [sys.executable, str(THERMAL), "--url", url, "--output", str(output)]
    for key, value in options.items():
        flag = "--" + key
        if key == "performance-mode":
            if not isinstance(value, bool):
                raise TypeError("performance-mode must be a JSON boolean")
            if value:
                argv.append(flag)
        elif key == "mtp-policy" or isinstance(value, (dict, list)):
            argv.extend((flag, json.dumps(value, separators=(",", ":"))))
        elif isinstance(value, bool):
            raise TypeError(
                f"{key} must use its explicit JSON or on/off representation"
            )
        else:
            argv.extend((flag, str(value)))
    argv.extend(("--server-pid", str(server_pid)))
    return argv


def validate_ladder_options(options: dict):
    """Reject an unsafe or duplicate cell plan before starting the GPU server."""
    if not isinstance(options, dict):
        raise TypeError("ladder options must be a JSON object")
    try:
        max_lanes = int(options["max-lanes"])
        max_inflight = int(options["max-inflight"])
        wide = int(options.get("wide", 4))
        cache_bytes = int(options["cache-bytes"])
        prefill_step = int(options["prefill-step"])
        max_context = int(options["max-context"])
        repetitions = int(options.get("runs", 3))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("profile sizing values must be positive integers") from error
    if (
        max_lanes < 1
        or max_inflight < max_lanes
        or not 2 <= wide <= max_lanes
        or cache_bytes < 1
        or prefill_step < 1
        or max_context < 1
        or repetitions != 3
    ):
        raise ValueError(
            "invalid profile bounds: need inflight >= lanes, 2 <= wide <= lanes, positive budgets, and 3 repetitions"
        )
    if options.get("apc-persistence") not in {"on", "off"} or options.get(
        "apc-persist-on-shutdown"
    ) not in {"on", "off"}:
        raise ValueError(
            "APCv2 persistence and shutdown persistence must be explicitly on or off"
        )


def wait_ready(
    url: str, process: subprocess.Popen, timeout: float, interval: float = 1.0
):
    deadline = time.monotonic() + timeout
    last = "not reachable"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited before readiness with code {process.returncode}"
            )
        try:
            with urlopen(url, timeout=2) as response:
                status = json.loads(response.read())
            if status.get("healthy") and status.get("state") == "ready":
                return status
            last = "status did not report healthy/ready"
        except (OSError, URLError, json.JSONDecodeError) as error:
            last = str(error)
        time.sleep(interval)
    raise TimeoutError(f"server readiness timed out: {last}")


def stop_server(process: subprocess.Popen | None, timeout: float = 20):
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-argv-json",
        type=Path,
        required=True,
        help="JSON array of argv; exactly one @PORT@ and explicit --host 127.0.0.1",
    )
    parser.add_argument(
        "--ladder-options-json",
        type=Path,
        required=True,
        help="JSON object of thermal_ladder options excluding --url/--output",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--job-timeout", type=float, default=28800)
    args = parser.parse_args(argv)
    if not (1 <= args.startup_timeout <= 3600 and 1 <= args.job_timeout <= 28800):
        parser.error("timeouts must be bounded (startup 1..3600s, job 1..28800s)")
    try:
        server_template = read_json(args.server_argv_json)
        ladder_options = read_json(args.ladder_options_json)
        if not isinstance(server_template, list):
            raise TypeError("server argv JSON must be an array")
        validate_ladder_options(ladder_options)
        port = reserve_port()
        server_command = expand_server_argv(server_template, port)
        url = f"http://127.0.0.1:{port}"
        probe_url = url + "/v1/status"
        # Check option names, required fields, and argument types before the
        # server can load a model or acquire expensive device resources.
        ladder_argv(ladder_options, url=url, output=args.output, server_pid=0)
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    if args.server_log.exists() or args.output.exists():
        parser.error("refusing to overwrite an existing server log or ladder report")
    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the server in this owned process group. The outer watchdog and paired
    # lock wrapper can then retire both even if this wrapper is forcibly stopped.
    server_log = args.server_log.open("ab", buffering=0)
    server = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    old_handlers = {
        sig: signal.signal(sig, interrupted)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    environment = os.environ.copy()
    source_root = str(ROOT / "src")
    environment["PYTHONPATH"] = source_root + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    try:
        server = subprocess.Popen(
            server_command,
            stdin=subprocess.DEVNULL,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        wait_ready(probe_url, server, args.startup_timeout)
        run_command = ladder_argv(
            ladder_options, url=url, output=args.output, server_pid=server.pid
        )
        ladder = subprocess.Popen(run_command, env=environment)
        try:
            return int(ladder.wait(timeout=args.job_timeout))
        except subprocess.TimeoutExpired as error:
            ladder.terminate()
            try:
                ladder.wait(timeout=15)
            except subprocess.TimeoutExpired:
                ladder.kill()
                ladder.wait(timeout=10)
            raise TimeoutError("thermal ladder exceeded its job deadline") from error
        finally:
            if ladder.poll() is None:
                ladder.terminate()
                try:
                    ladder.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    ladder.kill()
                    ladder.wait(timeout=10)
    except (
        OSError,
        TypeError,
        ValueError,
        TimeoutError,
        RuntimeError,
        subprocess.SubprocessError,
        KeyboardInterrupt,
    ) as error:
        print(
            f"owned profile stopped: {type(error).__name__}: {error}", file=sys.stderr
        )
        return 1
    finally:
        stop_server(server)
        server_log.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
