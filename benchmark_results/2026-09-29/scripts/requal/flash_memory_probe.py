#!/usr/bin/env python3
"""Locate Flash-Next swap growth across load, decode and shutdown."""

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
    parser.add_argument("--model", choices=("flash-next", "flash-next-uncensored"), required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--cache-cap-gib", type=int, default=0)
    parser.add_argument("--port", type=int, default=8397)
    parser.add_argument("--load-timeout", type=float, default=2400)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    model = next(model for model in config.MODELS if model.name == args.model)
    route = next(route for route in model.routes if route.name == model.default_route)
    output = HERE / args.host_label / model.name
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"memory-probe-{args.tag}.json"
    if path.exists():
        parser.error(f"probe receipt already exists: {path}")
    source_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    receipt = {
        "schema": "mlx2.requal.flash-memory-probe.v1", "source_head": source_head,
        "host": args.host_label, "model": model.name, "route": route.name,
        "artifact_config_sha256": hashlib.sha256((Path(model.path) / "config.json").read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_at": time.time(), "status": "running", "phases": [],
    }

    def sample(phase: str) -> None:
        row = {"phase": phase, "at": time.time(), "swapouts": swapouts()}
        try:
            import psutil
            row["host_available_bytes"] = int(psutil.virtual_memory().available)
        except Exception:
            pass
        receipt["phases"].append(row)
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    if args.cache_cap_gib:
        os.environ["MLX2_SERIES_CACHE_GIB_CAP"] = str(args.cache_cap_gib)
    command = [str(config.PYTHON), "-u", "-m", "mlx2.server", *config.server_args(model, route, "smoke")]
    command[command.index("--port") + 1] = str(args.port)
    receipt["command"] = command
    env = {**os.environ, "PYTHONPATH": config.stage_pythonpath(model),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    base = f"http://127.0.0.1:{args.port}"
    process = None
    try:
        with ExitStack() as stack:
            receipt["locks"] = owned.lock_host(stack)
            try:
                owned.get_json(base + "/health", timeout=1)
            except (OSError, urllib.error.URLError):
                pass
            else:
                raise RuntimeError(f"port {args.port} already serves /health")
            sample("before_load")
            with (output / f"memory-probe-{args.tag}.log").open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                receipt["server_pid"] = process.pid
                try:
                    deadline = time.monotonic() + args.load_timeout
                    last_swapouts = receipt["phases"][-1]["swapouts"]
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise RuntimeError(f"server exited during load: rc={process.returncode}")
                        try:
                            health = owned.get_json(base + "/health", timeout=1)
                            if health.get("status") == "ok":
                                receipt["health_at_ready"] = health
                                break
                            if health.get("error"):
                                raise RuntimeError(f"server load error: {health['error']}")
                        except urllib.error.HTTPError as exc:
                            if exc.code not in {503, 429}:
                                raise
                        except (TimeoutError, ConnectionError, urllib.error.URLError):
                            pass
                        current = swapouts()
                        if current != last_swapouts:
                            sample("load_swap_change")
                            last_swapouts = current
                        time.sleep(2)
                    else:
                        raise TimeoutError("Flash-Next load exceeded timeout")
                    sample("ready")
                    receipt["status_at_ready"] = owned.get_json(base + "/v1/status")
                    environment = (receipt["status_at_ready"].get("settings") or {}).get("environment") or {}
                    expected_sidecar = str(Path(model.path) / "ple_rows.bin")
                    receipt["ple_offload"] = {
                        "expected_sidecar": expected_sidecar,
                        "observed_sidecar": environment.get("MLX_QWEN4_PLE_NVME"),
                        "enabled": environment.get("MLX_QWEN4_PLE_NVME") == expected_sidecar,
                    }
                    if not receipt["ple_offload"]["enabled"]:
                        raise RuntimeError("file-backed PLE offload is not active")
                    receipt["smoke"] = owned.smoke(base, Path(model.path).name, config.family(model))
                    sample("after_decode")
                    receipt["status_after"] = owned.get_json(base + "/v1/status")
                    tables = (receipt["status_after"].get("execution") or {}).get("ple_tables") or []
                    # A short smoke may finish entirely through a route that
                    # never looks up a PLE row. Ask a longer prompt before
                    # judging whether the installed sidecar was observed-used.
                    receipt["ple_probe"] = []
                    for trial in range(3):
                        if sum(int(row.get("rows") or 0) for row in tables):
                            break
                        response = owned.post_json(base + "/v1/chat/completions", {
                            "model": Path(model.path).name,
                            "messages": [{"role": "user", "content":
                                ("Explain why a lighthouse is visible from far away. " * 16)
                                + f"Give answer number {trial} in two sentences."}],
                            "max_tokens": 64, "temperature": 0,
                            "enable_thinking": False,
                        })
                        receipt["ple_probe"].append({
                            "trial": trial,
                            "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
                            "route": (response.get("mlx2") or {}).get("route"),
                        })
                        receipt["status_after"] = owned.get_json(base + "/v1/status")
                        tables = (receipt["status_after"].get("execution") or {}).get("ple_tables") or []
                    sample("after_ple_probe")
                    receipt["ple_offload"]["tables"] = len(tables)
                    receipt["ple_offload"]["rows_read"] = sum(int(row.get("rows") or 0) for row in tables)
                    receipt["ple_offload"]["bytes_read"] = sum(int(row.get("bytes_read") or 0) for row in tables)
                    receipt["status"] = "passed" if (receipt["smoke"]["passed"]
                        and receipt["ple_offload"]["rows_read"] > 0) else "failed"
                finally:
                    owned.stop_owned(process)
            sample("after_shutdown")
    except Exception as exc:
        receipt["status"] = "error"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        sample("after_error")
    receipt["finished_at"] = time.time()
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"{model.name} {receipt['status']} {path}", flush=True)
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
