#!/usr/bin/env python3
"""Resume the short smoke queue on one frozen source and one host."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "qualification/runs/series-20260924"))
import campaign_config as config  # noqa: E402


def swapouts() -> int | None:
    result = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False)
    match = re.search(r"Swapouts:\s+(\d+)", result.stdout)
    return int(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--max-models", type=int, default=0)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    harness_hash = hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest()
    state_path = HERE / args.host_label / "queue-smoke.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"schema": "mlx2.requal.smoke-queue.v1", "host": args.host_label,
             "source_head": head, "started_at": time.time(), "jobs": {},
             "invocations": [{"started_at": time.time(), "only": args.only,
                              "exclude": args.exclude, "max_models": args.max_models}]}
    if state_path.exists():
        previous_state = json.loads(state_path.read_text())
        if previous_state.get("source_head") == head and previous_state.get("host") == args.host_label:
            state = previous_state
            state.pop("finished_at", None)
            state.setdefault("invocations", []).append({"started_at": time.time(), "only": args.only,
                                                        "exclude": args.exclude, "max_models": args.max_models})

    def save() -> None:
        state["updated_at"] = time.time()
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    preflight_path = HERE / ("preflight-m3.json" if args.host_label.startswith("m3") else "preflight-m5.json")
    preflight = json.loads(preflight_path.read_text())
    if preflight["source_head"] != head:
        raise RuntimeError(f"preflight source {preflight['source_head']} != {head}")
    completed = 0
    for model in config.MODELS:
        if args.only and model.name not in args.only:
            continue
        if model.name in args.exclude:
            continue
        if preflight["models"].get(model.name, {}).get("status") != "pass":
            raise RuntimeError(f"preflight did not pass for {model.name}")
        receipt_path = HERE / args.host_label / model.name / "smoke.json"
        if receipt_path.exists():
            previous = json.loads(receipt_path.read_text())
            queued = state["jobs"].get(model.name)
            contaminated = isinstance(queued, dict) and bool(queued.get("contaminated"))
            if (previous.get("status") == "passed" and previous.get("source_head") == head
                    and previous.get("harness_sha256") == harness_hash and not contaminated):
                state["jobs"][model.name] = "already_passed"
                save()
                continue
        before = swapouts()
        print(f"START {model.name} source={head[:8]} swapouts={before}", flush=True)
        command = [sys.executable, str(HERE / "run.py"), "--model", model.name,
                   "--host-label", args.host_label]
        result = subprocess.run(command, cwd=ROOT, check=False)
        after = swapouts()
        row = {"returncode": result.returncode, "swapouts_before": before,
               "swapouts_after": after, "finished_at": time.time()}
        if before is not None and after is not None and after > before:
            row["contaminated"] = "swapouts_rose"
        state["jobs"][model.name] = row
        save()
        print(f"DONE {model.name} rc={result.returncode} swapouts={before}->{after}", flush=True)
        completed += 1
        if result.returncode or row.get("contaminated"):
            return 1
        if args.max_models and completed >= args.max_models:
            break
    state["finished_at"] = time.time()
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
