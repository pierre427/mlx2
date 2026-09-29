#!/usr/bin/env python3
"""Resume per-host thermal ladders after source-bound stress gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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


def save(path: Path, data: dict) -> None:
    data["updated_at"] = time.time()
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def preserve(*paths: Path) -> None:
    if not any(path.exists() for path in paths):
        return
    attempt = 1
    while any(path.with_name(f"{path.stem}-attempt{attempt}{path.suffix}").exists() for path in paths):
        attempt += 1
    for path in paths:
        if path.exists():
            shutil.copy2(path, path.with_name(f"{path.stem}-attempt{attempt}{path.suffix}"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--max-models", type=int, default=0)
    parser.add_argument("--min-length", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=262144)
    parser.add_argument("--wide-max-context", type=int, default=32768)
    parser.add_argument("--wide", type=int, default=4,
                        help="second ladder width; 1 runs width one only")
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="measure later eligible models after a failed ladder")
    args = parser.parse_args()
    if args.min_length > args.max_length or args.max_models < 0 or args.wide < 1:
        parser.error("invalid length or model count")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tree = source_tree(head)
    harness_hash = hashlib.sha256((HERE / "run_perf.py").read_bytes()).hexdigest()
    host_dir = HERE / args.host_label
    state_path = host_dir / ("queue-perf.json" if args.wide == 4
                             else f"queue-perf-w{args.wide}.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema": "mlx2.requal.performance-queue.v1", "host": args.host_label, "jobs": {}}
    state.setdefault("invocations", []).append({"started_at": time.time(), "head": head,
        "source_tree": tree, "min_length": args.min_length, "max_length": args.max_length,
        "wide_max_context": args.wide_max_context, "wide": args.wide,
        "only": args.only, "exclude": args.exclude,
        "continue_on_failure": args.continue_on_failure})
    state.pop("finished_at", None)
    save(state_path, state)
    completed = 0
    failed = False
    for model in config.MODELS:
        if args.only and model.name not in args.only or model.name in args.exclude:
            continue
        output = host_dir / model.name
        stress_path = output / "stress.json"
        if not stress_path.exists():
            state["jobs"][model.name] = {"status": "ineligible", "reason": "no stress receipt"}
            save(state_path, state)
            continue
        stress = json.loads(stress_path.read_text())
        if stress.get("status") != "passed" or source_tree(stress["source_head"]) != tree:
            state["jobs"][model.name] = {"status": "ineligible", "reason": "stress not passed at source tree"}
            save(state_path, state)
            continue
        stem = f"ladder-{args.min_length}-{args.max_length}-r3"
        if args.wide != 4:
            stem += f"-w{args.wide}"
        receipt_path = output / f"{stem}-run.json"
        ladder_path = output / f"{stem}.json"
        artifact_hash = hashlib.sha256((Path(model.path) / "config.json").read_bytes()).hexdigest()
        if receipt_path.exists():
            prior = json.loads(receipt_path.read_text())
            if (prior.get("status") == "passed" and source_tree(prior["source_head"]) == tree
                    and prior.get("harness_sha256") == harness_hash
                    and prior.get("artifact_config_sha256") == artifact_hash
                    and prior.get("wide_max_context") == args.wide_max_context
                    and prior.get("wide", 4) == args.wide
                    and ladder_path.exists() and json.loads(ladder_path.read_text()).get("passed")):
                state["jobs"][model.name] = {"status": "already_passed", "source_head": prior["source_head"]}
                save(state_path, state)
                continue
            if prior.get("status") != "running":
                preserve(receipt_path, ladder_path)
        command = [sys.executable, str(HERE / "run_perf.py"), "--model", model.name,
            "--host-label", args.host_label, "--runs", "3", "--min-length", str(args.min_length),
            "--max-length", str(args.max_length), "--wide-max-context", str(args.wide_max_context),
            "--wide", str(args.wide)]
        print(f"START {model.name} ladder {args.min_length}..{args.max_length} source_tree={tree[:8]}", flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        state["jobs"][model.name] = {"status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode, "finished_at": time.time()}
        save(state_path, state)
        print(f"DONE {model.name} rc={result.returncode}", flush=True)
        completed += 1
        if result.returncode:
            failed = True
            if not args.continue_on_failure:
                return 1
        if args.max_models and completed >= args.max_models:
            break
    state["finished_at"] = time.time()
    save(state_path, state)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
