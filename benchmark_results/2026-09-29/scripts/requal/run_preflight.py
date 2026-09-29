#!/usr/bin/env python3
"""Refresh the CPU-only roster preflight for one source snapshot and host."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "qualification/runs/series-20260924"))
import campaign_config as config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-label", required=True)
    args = parser.parse_args()
    path = HERE / ("preflight-m3.json" if args.host_label.startswith("m3") else "preflight-m5.json")
    started = time.time()
    present = [model for model in config.MODELS if Path(model.path).is_dir()]
    # campaign_config filters absent paths while importing, so compare against
    # the full M5 roster recorded in this same campaign.
    full_roster = set(json.loads((HERE / "preflight-m5.json").read_text())["models"])
    missing = sorted(full_roster - {model.name for model in present})
    checked = config.validate_cpu_preflight(present)
    models = {}
    for model in present:
        policies = sorted({name for route in model.routes for name in (route.policy, route.opt_in_policy) if name})
        models[model.name] = {
            "status": "pass", "model": checked["models"][model.name], "policies": policies,
            "artifact_config_sha256": hashlib.sha256((Path(model.path) / "config.json").read_bytes()).hexdigest(),
        }
    report = {
        "schema": "mlx2.requal.preflight.v1", "host": args.host_label,
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_tree": subprocess.check_output(["git", "rev-parse", "HEAD:src"], cwd=ROOT, text=True).strip(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_at": started, "updated_at": time.time(), "device": checked["device"],
        "weights_loaded": checked["weights_loaded"], "north_commit_direction": checked.get("north_commit_direction"),
        "models": models, "missing": missing,
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{args.host_label}: {len(models)} passed, {len(missing)} not staged; {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
