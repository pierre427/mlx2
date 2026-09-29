#!/usr/bin/env python3
"""Run one thermally controlled context-ladder segment on an owned server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
from contextlib import ExitStack
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "qualification/runs/series-20260924"))
import campaign_config as config  # noqa: E402
import run as owned  # noqa: E402
from queue_smoke import swapouts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--min-length", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=262144)
    parser.add_argument("--wide-max-context", type=int, default=32768)
    parser.add_argument("--wide", type=int, default=4,
                        help="second ladder width; 1 runs a width-one-only basic ladder")
    parser.add_argument("--port", type=int, default=8397)
    parser.add_argument("--load-timeout", type=float, default=2400)
    parser.add_argument("--cache-cap-gib", type=int, default=None,
                        help="Host cache cap; defaults to 8 GiB on M3")
    parser.add_argument("--context-cap", type=int, default=None,
                        help="Host serving context cap; defaults to 32768 on M3")
    args = parser.parse_args()
    if args.runs < 1 or args.wide < 1 or args.min_length > args.max_length:
        parser.error("invalid repetition count or length range")
    models = {model.name: model for model in config.MODELS}
    model = models.get(args.model)
    if model is None:
        parser.error(f"model not present on this host: {args.model}")
    route = next(route for route in model.routes if route.name == model.default_route)
    output = HERE / args.host_label / model.name
    output.mkdir(parents=True, exist_ok=True)
    stem = f"ladder-{args.min_length}-{args.max_length}-r{args.runs}"
    if args.wide != 4:
        stem += f"-w{args.wide}"
    ladder_path = output / f"{stem}.json"
    receipt_path = output / f"{stem}-run.json"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    receipt = {
        "schema": "mlx2.requal.performance-run.v1", "host": args.host_label,
        "source_head": head, "model": model.name, "route": route.name,
        "artifact": model.path,
        "artifact_config_sha256": hashlib.sha256((Path(model.path) / "config.json").read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs_per_cell": args.runs, "min_length": args.min_length,
        "max_length": args.max_length, "wide_max_context": args.wide_max_context,
        "wide": args.wide,
        "started_at": time.time(), "status": "running", "ladder_path": str(ladder_path),
    }

    def save() -> None:
        receipt["updated_at"] = time.time()
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    save()
    if args.host_label.startswith("m3"):
        cache_cap = args.cache_cap_gib if args.cache_cap_gib is not None else 8
        context_cap = args.context_cap if args.context_cap is not None else 32768
        os.environ["MLX2_SERIES_CACHE_GIB_CAP"] = str(cache_cap)
        os.environ["MLX2_SERIES_CONTEXT_CAP"] = str(context_cap)
        receipt["m3_host_caps"] = {"cache_gib": cache_cap, "max_context": context_cap}
    command = [str(config.PYTHON), "-u", "-m", "mlx2.server", *config.server_args(model, route, "ladder")]
    command[command.index("--port") + 1] = str(args.port)
    receipt["server_command"] = command
    env = {**os.environ, "PYTHONPATH": config.stage_pythonpath(model),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    base = f"http://127.0.0.1:{args.port}"
    before = swapouts()
    server = None
    try:
        with ExitStack() as stack:
            receipt["locks"] = owned.lock_host(stack)
            try:
                owned.get_json(base + "/health", timeout=1)
            except (OSError, urllib.error.URLError):
                pass
            else:
                raise RuntimeError(f"port {args.port} already serves /health")
            with (output / f"{stem}-server.log").open("w") as log:
                server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                receipt["server_pid"] = server.pid
                save()
                try:
                    receipt["health_at_ready"] = owned.wait_ready(base, server, args.load_timeout)
                    receipt["status_at_ready"] = owned.get_json(base + "/v1/status")
                    effective_max = min(model.max_context,
                                        int((receipt["status_at_ready"].get("settings") or {}).get("max_context")
                                            or model.max_context))
                    ladder_command = [str(config.PYTHON), str(ROOT / "qualification/runs/series-20260924/thermal_ladder.py"),
                                      "--url", base, "--output", str(ladder_path),
                                      "--model", model.name, "--model-id", Path(model.path).name,
                                      "--route", route.name, "--max-context", str(effective_max),
                                      "--server-pid", str(server.pid), "--runs", str(args.runs),
                                      "--min-length", str(args.min_length), "--max-length", str(args.max_length),
                                      "--wide-max-context", str(args.wide_max_context),
                                      "--wide", str(args.wide)]
                    receipt["ladder_command"] = ladder_command
                    save()
                    with (output / f"{stem}.log").open("w") as ladder_log:
                        result = subprocess.run(ladder_command, cwd=ROOT, env=env,
                                                stdout=ladder_log, stderr=subprocess.STDOUT,
                                                check=False)
                    receipt["ladder_returncode"] = result.returncode
                    if ladder_path.exists():
                        ladder = json.loads(ladder_path.read_text())
                        receipt["ladder_summary"] = {
                            "passed": ladder.get("passed"), "cells": len(ladder.get("cells", [])),
                            "finished_at": ladder.get("finished_at"),
                        }
                    receipt["status_after"] = owned.get_json(base + "/v1/status")
                    receipt["status"] = "passed" if result.returncode == 0 else "failed"
                finally:
                    owned.stop_owned(server)
    except Exception as exc:
        receipt["status"] = "error"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        if server is not None:
            receipt["server_log_tail"] = (output / f"{stem}-server.log").read_text(errors="replace")[-3000:]
    after = swapouts()
    receipt["swapouts"] = {"before": before, "after": after,
                           "delta": after - before if before is not None and after is not None else None}
    if receipt["swapouts"]["delta"] is not None and receipt["swapouts"]["delta"] > 0:
        receipt["status"] = "contaminated"
    receipt["finished_at"] = time.time()
    save()
    print(f"{model.name} {receipt['status']} {receipt_path}", flush=True)
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
