#!/usr/bin/env python3
"""Run one model's experimental feature checks under campaign GPU ownership."""

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
SERIES = ROOT / "qualification/runs/series-20260924"
sys.path.insert(0, str(SERIES))
import campaign_config as config  # noqa: E402
import experimental_job as experimental  # noqa: E402
import extra_common as xc  # noqa: E402
import run as owned  # noqa: E402
from queue_smoke import swapouts  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--cache-cap-gib", type=int, default=None)
    parser.add_argument("--context-cap", type=int, default=None)
    parser.add_argument("--max-lanes-cap", type=int, default=None)
    args = parser.parse_args()
    if (args.cache_cap_gib is not None and args.cache_cap_gib < 1
            or args.context_cap is not None and args.context_cap < 1
            or args.max_lanes_cap is not None and args.max_lanes_cap < 1):
        parser.error("host caps must be positive")
    cache_cap = args.cache_cap_gib
    context_cap = args.context_cap
    if args.host_label.startswith("m3"):
        cache_cap = cache_cap if cache_cap is not None else 8
        context_cap = context_cap if context_cap is not None else 32768
    if cache_cap is not None:
        os.environ["MLX2_SERIES_CACHE_GIB_CAP"] = str(cache_cap)
    if context_cap is not None:
        os.environ["MLX2_SERIES_CONTEXT_CAP"] = str(context_cap)
    models = {model.name: model for model in config.MODELS}
    model = models.get(args.model)
    if model is None:
        parser.error(f"model not staged on this host: {args.model}")
    output = HERE / args.host_label / model.name / "features"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "qualification.json"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", f"{head}:src"], cwd=ROOT, text=True).strip()
    receipt = {
        "schema": "mlx2.requal.feature-run.v1",
        "host": args.host_label,
        "model": model.name,
        "route": model.default_route,
        "source_head": head,
        "source_tree": tree,
        "host_caps": {"cache_gib": cache_cap, "max_context": context_cap,
                      "max_lanes": args.max_lanes_cap},
        "artifact": model.path,
        "artifact_config_sha256": sha256(Path(model.path) / "config.json"),
        "harness_sha256": {name: sha256(file) for name, file in {
            "run_features": Path(__file__),
            "experimental_job": SERIES / "experimental_job.py",
            "extra_common": SERIES / "extra_common.py",
            "campaign_config": SERIES / "campaign_config.py",
            "thermal_ladder": SERIES / "thermal_ladder.py",
        }.items()},
        "started_at": time.time(),
        "status": "running",
    }

    def save() -> None:
        receipt["updated_at"] = time.time()
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")

    before = swapouts()
    save()
    try:
        with ExitStack() as stack:
            receipt["locks"] = owned.lock_host(stack)
            try:
                owned.get_json(f"http://127.0.0.1:{config.PORT}/health", timeout=1)
            except (OSError, urllib.error.URLError):
                pass
            else:
                raise RuntimeError(f"port {config.PORT} already serves /health")
            save()
            xc.ensure_runtime_files()
            receipt["phase_swapouts"] = []
            original_start, original_stop = xc.Server.start, xc.Server.stop
            original_command = xc.server_command

            def capped_command(*command_args, **command_kwargs):
                command = original_command(*command_args, **command_kwargs)
                if args.max_lanes_cap is not None and "--max-lanes" in command:
                    index = command.index("--max-lanes") + 1
                    command[index] = str(min(int(command[index]), args.max_lanes_cap))
                return command

            def record_phase(server, phase, start_count, end_count):
                receipt["phase_swapouts"].append({
                    "server": server.name, "phase": phase,
                    "before": start_count, "after": end_count,
                    "delta": end_count - start_count
                    if start_count is not None and end_count is not None else None,
                    "at": time.time(),
                })
                save()

            def tracked_start(server):
                count = swapouts()
                try:
                    return original_start(server)
                finally:
                    end_count = swapouts()
                    record_phase(server, "load", count, end_count)
                    server._campaign_swap_after_load = end_count

            def tracked_stop(server):
                count = swapouts()
                record_phase(server, "active",
                             getattr(server, "_campaign_swap_after_load", None), count)
                try:
                    return original_stop(server)
                finally:
                    record_phase(server, "shutdown", count, swapouts())

            xc.Server.start, xc.Server.stop = tracked_start, tracked_stop
            xc.server_command = capped_command
            try:
                result, summary = experimental.run(model, output)
            finally:
                xc.Server.start, xc.Server.stop = original_start, original_stop
                xc.server_command = original_command
            receipt["status"] = result
            receipt["summary"] = summary
    except Exception as exc:
        receipt["status"] = "error"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    after = swapouts()
    receipt["swapouts"] = {"before": before, "after": after,
                           "delta": after - before if before is not None and after is not None else None}
    if receipt["swapouts"]["delta"] is not None and receipt["swapouts"]["delta"] > 0:
        receipt["status"] = "contaminated"
    receipt["finished_at"] = time.time()
    save()
    print(f"{model.name} features {receipt['status']} {path}", flush=True)
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
