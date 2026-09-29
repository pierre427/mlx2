#!/usr/bin/env python3
"""Resume 20x20 APCv2/batching gates by source tree and machine."""

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


def source_tree(head: str) -> str:
    return subprocess.check_output(["git", "rev-parse", f"{head}:src"], cwd=ROOT, text=True).strip()


def lanes_for(model, host: str) -> int:
    if host.startswith("m3"):
        return 8 if model.name == "north" else 4
    if model.name == "north":
        return 20
    if model.name in {"gemma3n", "minicpmo", "laguna"}:
        return 8
    if model.name in {"flash-next", "flash-next-uncensored", "nemotron"}:
        return 2
    return 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--max-models", type=int, default=0)
    parser.add_argument("--max-lanes", type=int, default=0)
    parser.add_argument("--include-contaminated-smoke", action="store_true")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tree = source_tree(head)
    harness_hash = hashlib.sha256((HERE / "run_stress.py").read_bytes()).hexdigest()
    host_dir = HERE / args.host_label
    state_path = host_dir / "queue-stress.json"
    state = {"schema": "mlx2.requal.stress-queue.v1", "host": args.host_label,
             "started_at": time.time(), "jobs": {}}
    if state_path.exists():
        state = json.loads(state_path.read_text())
    state.setdefault("invocations", []).append({
        "started_at": time.time(), "head": head, "source_tree": tree,
        "only": args.only, "exclude": args.exclude, "max_models": args.max_models,
        "max_lanes": args.max_lanes, "include_contaminated_smoke": args.include_contaminated_smoke,
    })
    state.pop("finished_at", None)

    def save() -> None:
        state["updated_at"] = time.time()
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    smoke_queue_path = host_dir / "queue-smoke.json"
    smoke_queue = json.loads(smoke_queue_path.read_text()) if smoke_queue_path.exists() else {"jobs": {}}
    completed = 0
    for model in config.MODELS:
        if args.only and model.name not in args.only:
            continue
        if model.name in args.exclude:
            continue
        model_dir = host_dir / model.name
        smoke_path = model_dir / "smoke.json"
        if not smoke_path.exists():
            state["jobs"][model.name] = {"status": "ineligible", "reason": "no smoke receipt"}
            save()
            continue
        smoke = json.loads(smoke_path.read_text())
        contamination = smoke_queue.get("jobs", {}).get(model.name)
        smoke_contaminated = isinstance(contamination, dict) and bool(contamination.get("contaminated"))
        if (smoke.get("status") != "passed" or source_tree(smoke["source_head"]) != tree
                or (smoke_contaminated and not args.include_contaminated_smoke)):
            state["jobs"][model.name] = {
                "status": "ineligible", "reason": "smoke failed, changed source tree, or swap-contaminated",
                "smoke_status": smoke.get("status"), "smoke_contaminated": smoke_contaminated,
            }
            save()
            continue
        artifact_hash = hashlib.sha256((Path(model.path) / "config.json").read_bytes()).hexdigest()
        stress_path = model_dir / "stress.json"
        if stress_path.exists():
            prior = json.loads(stress_path.read_text())
            if (prior.get("status") == "passed" and source_tree(prior["source_head"]) == tree
                    and prior.get("artifact_config_sha256") == artifact_hash
                    and prior.get("harness_sha256") == harness_hash):
                state["jobs"][model.name] = {"status": "already_passed", "source_head": prior["source_head"]}
                save()
                continue
            if prior.get("status") not in {"running"}:
                attempt = 1
                while (model_dir / f"stress-attempt{attempt}.json").exists():
                    attempt += 1
                (model_dir / f"stress-attempt{attempt}.json").write_text(stress_path.read_text())
        lanes = args.max_lanes or lanes_for(model, args.host_label)
        command = [sys.executable, str(HERE / "run_stress.py"), "--model", model.name,
                   "--host-label", args.host_label, "--max-lanes", str(lanes)]
        print(f"START {model.name} lanes={lanes} source_tree={tree[:8]}", flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        row = {"status": "passed" if result.returncode == 0 else "failed",
               "returncode": result.returncode, "max_lanes": lanes, "finished_at": time.time()}
        state["jobs"][model.name] = row
        save()
        print(f"DONE {model.name} rc={result.returncode}", flush=True)
        completed += 1
        if result.returncode:
            return 1
        if args.max_models and completed >= args.max_models:
            break
    state["finished_at"] = time.time()
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
