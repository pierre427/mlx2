#!/usr/bin/env python3
"""Serve one model for APCv2, concurrency and 20x20 stability gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
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


class InterruptedRun(Exception):
    """The owned stress run received a termination signal."""


def _interrupt(signum: int, _frame: object) -> None:
    raise InterruptedRun(f"signal {signum}")


def apc_probe(base: str, model_id: str) -> dict:
    shared = ("Stable reference: the verification phrase is HARBOR-731. "
              "Return it exactly when asked.\n") * 140
    body = {
        "model": model_id,
        "messages": [{"role": "system", "content": shared},
                     {"role": "user", "content": "What is the verification phrase?"}],
        "max_tokens": 64, "temperature": 0, "enable_thinking": False,
    }
    rows = []
    for _ in range(2):
        result = owned.post_json(base + "/v1/chat/completions", body)
        choice = (result.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        rows.append({"content": content[:200], "correct": "HARBOR-731" in content,
                     "cached_tokens": (result.get("mlx2") or {}).get("cached_tokens"),
                     "route_receipt": result.get("mlx2")})
    return {"passed": all(row["correct"] for row in rows)
            and int(rows[1]["cached_tokens"] or 0) >= 256,
            "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--route", default=None, help="named model route; defaults to its selected route")
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--max-lanes", type=int, default=20)
    parser.add_argument("--cache-cap-gib", type=int, default=0)
    parser.add_argument("--thinking-allowance", type=int, default=None,
                        help="extra generation tokens for models that think by default")
    parser.add_argument("--port", type=int, default=8397)
    parser.add_argument("--load-timeout", type=float, default=2400)
    parser.add_argument("--tag", default="", help="separate this run from historical model receipts")
    parser.add_argument("--execution-policy", type=Path, default=None)
    parser.add_argument("--lane-matmul", choices=("off", "auto", "crossover", "exact"), default=None)
    parser.add_argument("--lane-policy", type=Path, default=None)
    args = parser.parse_args()
    if args.tag and not re.fullmatch(r"[a-zA-Z0-9_-]+", args.tag):
        parser.error("tag must contain only letters, digits, underscores or hyphens")
    if args.cache_cap_gib < 0:
        parser.error("cache cap must be nonnegative")
    if args.thinking_allowance is not None and args.thinking_allowance < 0:
        parser.error("thinking allowance must be nonnegative")
    if args.cache_cap_gib:
        os.environ["MLX2_SERIES_CACHE_GIB_CAP"] = str(args.cache_cap_gib)
    models = {model.name: model for model in config.MODELS}
    model = models.get(args.model)
    if model is None:
        parser.error(f"model not present on this host: {args.model}")
    route = next((route for route in model.routes
                  if route.name == (args.route or model.default_route)), None)
    if route is None:
        parser.error(f"route {args.route!r} is not declared for {model.name}")
    output = HERE / args.host_label / model.name
    if args.tag:
        output /= f"stress-{args.tag}"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "stress.json"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    receipt = {
        "schema": "mlx2.requal.stress.v1", "host": args.host_label,
        "source_head": head, "model": model.name, "route": route.name,
        "artifact": model.path,
        "artifact_config_sha256": hashlib.sha256((Path(model.path) / "config.json").read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "max_lanes": args.max_lanes, "started_at": time.time(), "status": "running",
        "tag": args.tag,
        "cache_cap_gib": args.cache_cap_gib or None,
        "thinking_allowance": args.thinking_allowance,
    }

    def save() -> None:
        receipt["updated_at"] = time.time()
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    save()
    command = [str(config.PYTHON), "-u", "-m", "mlx2.server", *config.server_args(model, route, "sanity")]
    command[command.index("--port") + 1] = str(args.port)
    command[command.index("--max-lanes") + 1] = str(args.max_lanes)
    if args.execution_policy is not None:
        selected = args.execution_policy.resolve(strict=True)
        if "--execution-policy" in command:
            command[command.index("--execution-policy") + 1] = str(selected)
        else:
            command.extend(["--execution-policy", str(selected)])
        receipt["execution_policy_sha256"] = hashlib.sha256(selected.read_bytes()).hexdigest()
    if args.lane_matmul is not None:
        command.extend(["--lane-matmul", args.lane_matmul])
    if args.lane_policy is not None:
        selected = args.lane_policy.resolve(strict=True)
        command.extend(["--lane-policy", str(selected)])
        receipt["lane_policy_sha256"] = hashlib.sha256(selected.read_bytes()).hexdigest()
    receipt["command"] = command
    env = {**os.environ, "PYTHONPATH": config.stage_pythonpath(model),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    if args.thinking_allowance is not None:
        env["SANITY_THINK_ALLOWANCE"] = str(args.thinking_allowance)
    base = f"http://127.0.0.1:{args.port}"
    before = swapouts()
    server = None
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)
    try:
        with ExitStack() as stack:
            receipt["locks"] = owned.lock_host(stack)
            try:
                owned.get_json(base + "/health", timeout=1)
            except (OSError, urllib.error.URLError):
                pass
            else:
                raise RuntimeError(f"port {args.port} already serves /health")
            with (output / "stress-server.log").open("w") as log:
                server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                receipt["server_pid"] = server.pid
                save()
                try:
                    receipt["health_at_ready"] = owned.wait_ready(base, server, args.load_timeout)
                    receipt["status_at_ready"] = owned.get_json(base + "/v1/status")
                    receipt["apc_probe"] = apc_probe(base, Path(model.path).name)
                    save()
                    if not receipt["apc_probe"]["passed"]:
                        raise RuntimeError("APCv2 prefix reuse probe failed")
                    steps = (
                        ("concurrency", [str(config.PYTHON), str(ROOT / "qualification/runs/series-20260924/concurrency_probe.py"),
                                         "--url", base, "--output", str(output / "concurrency.json"),
                                         "--model-id", Path(model.path).name], 7200),
                        ("20x20", [str(config.PYTHON), str(ROOT / "qualification/runs/series-20260924/sanity_20x20.py"),
                                    base, str(output / "20x20.json"), "20"], 28800),
                    )
                    receipt["steps"] = {}
                    for name, step, timeout in steps:
                        with (output / f"{name}.log").open("w") as step_log:
                            step_process = subprocess.Popen(
                                step, cwd=ROOT, env=env, stdout=step_log,
                                stderr=subprocess.STDOUT, start_new_session=True,
                            )
                            try:
                                returncode = step_process.wait(timeout=timeout)
                            finally:
                                owned.stop_owned(step_process)
                        receipt["steps"][name] = {"returncode": returncode,
                                                  "log": str(output / f"{name}.log")}
                        save()
                        if returncode:
                            raise RuntimeError(f"{name} gate exited {returncode}")
                    report = json.loads((output / "20x20.json").read_text())
                    widths = {int(k): v for k, v in report.get("observed_widths", {}).items()
                              if k not in {"None", "null"}}
                    mtp_batches = int((report.get("segmented_mtp_delta") or {}).get(
                        "batched_target_forwards") or 0)
                    receipt["batching_evidence"] = {
                        "ordinary_reply_widths": widths,
                        "mtp_batched_target_forwards": mtp_batches,
                        "mtp_reply_routes": report.get("mtp_reply_routes", {}),
                    }
                    receipt["batching_engaged"] = (any(width > 1 and count > 0
                                                       for width, count in widths.items())
                                                   or mtp_batches > 0)
                    receipt["status_after"] = owned.get_json(base + "/v1/status")
                    receipt["health_after"] = owned.get_json(base + "/health")
                    receipt["status"] = "passed" if (receipt["batching_engaged"]
                        and receipt["health_after"].get("status") == "ok") else "failed"
                finally:
                    owned.stop_owned(server)
    except InterruptedRun as exc:
        receipt["status"] = "interrupted"
        receipt["interruption"] = str(exc)
    except Exception as exc:
        receipt["status"] = "error"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        if server is not None:
            receipt["server_log_tail"] = (output / "stress-server.log").read_text(errors="replace")[-3000:]
    after = swapouts()
    receipt["swapouts"] = {"before": before, "after": after,
                           "delta": after - before if before is not None and after is not None else None}
    if receipt["swapouts"]["delta"] is not None and receipt["swapouts"]["delta"] > 0:
        receipt["status"] = "contaminated"
    receipt["finished_at"] = time.time()
    save()
    print(f"{model.name} {receipt['status']} {path}", flush=True)
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
