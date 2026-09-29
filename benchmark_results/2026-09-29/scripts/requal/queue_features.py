#!/usr/bin/env python3
"""Run staged models' feature qualification sequentially on one host."""

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
SERIES = ROOT / "qualification/runs/series-20260924"
sys.path.insert(0, str(SERIES))
import campaign_config as config  # noqa: E402


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def hashes() -> dict[str, str]:
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
        "run_features": HERE / "run_features.py",
        "experimental_job": SERIES / "experimental_job.py",
        "extra_common": SERIES / "extra_common.py",
        "campaign_config": SERIES / "campaign_config.py",
        "thermal_ladder": SERIES / "thermal_ladder.py",
    }.items()}


def preserve(path: Path) -> None:
    if not path.exists():
        return
    attempt = 1
    while path.with_name(f"{path.stem}-attempt{attempt}{path.suffix}").exists():
        attempt += 1
    shutil.copy2(path, path.with_name(f"{path.stem}-attempt{attempt}{path.suffix}"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--max-models", type=int, default=0)
    parser.add_argument("--cache-cap-gib", type=int)
    parser.add_argument("--context-cap", type=int)
    parser.add_argument("--max-lanes-cap", type=int)
    parser.add_argument("--list-eligible", action="store_true",
                        help="inspect smoke eligibility without starting a server or writing state")
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()
    if args.max_models < 0 or any(value is not None and value < 1 for value in
                                  (args.cache_cap_gib, args.context_cap, args.max_lanes_cap)):
        parser.error("model count and host caps must be nonnegative/positive")
    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", f"{head}:src")
    harness = hashes()
    host_dir = HERE / args.host_label
    caps = {"cache_gib": args.cache_cap_gib, "max_context": args.context_cap,
            "max_lanes": args.max_lanes_cap}
    if args.host_label.startswith("m3"):
        caps["cache_gib"] = caps["cache_gib"] if caps["cache_gib"] is not None else 8
        caps["max_context"] = caps["max_context"] if caps["max_context"] is not None else 32768
    jobs: list[tuple[str, Path, str]] = []
    for model in config.MODELS:
        if (args.only and model.name not in args.only) or model.name in args.exclude:
            continue
        smoke_path = host_dir / model.name / "smoke.json"
        if not smoke_path.exists():
            jobs.append((model.name, smoke_path, "ineligible: no smoke receipt"))
            continue
        smoke = json.loads(smoke_path.read_text())
        if smoke.get("status") != "passed":
            jobs.append((model.name, smoke_path, "ineligible: smoke not passed"))
            continue
        receipt_path = host_dir / model.name / "features/qualification.json"
        if receipt_path.exists():
            prior = json.loads(receipt_path.read_text())
            if (prior.get("status") == "pass" and prior.get("source_tree") == tree
                    and prior.get("harness_sha256") == harness
                    and prior.get("host_caps") == caps
                    and prior.get("artifact_config_sha256") == hashlib.sha256(
                        (Path(model.path) / "config.json").read_bytes()).hexdigest()):
                jobs.append((model.name, receipt_path, "already_passed"))
                continue
        jobs.append((model.name, receipt_path, "eligible"))
    if args.list_eligible:
        for name, _, status in jobs:
            print(f"{name}: {status}")
        return 0

    host_dir.mkdir(parents=True, exist_ok=True)
    state_path = host_dir / "queue-features.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema": "mlx2.requal.feature-queue.v1", "host": args.host_label, "jobs": {}}
    state.setdefault("invocations", []).append({
        "started_at": time.time(), "head": head, "source_tree": tree,
        "only": args.only, "exclude": args.exclude, "max_models": args.max_models,
        "host_caps": caps,
    })
    state.pop("finished_at", None)

    def save() -> None:
        state["updated_at"] = time.time()
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    save()
    completed = 0
    failed = False
    for name, path, status in jobs:
        if status != "eligible":
            state["jobs"][name] = {"status": status}
            save()
            continue
        preserve(path)
        command = [sys.executable, str(HERE / "run_features.py"),
                   "--model", name, "--host-label", args.host_label]
        for option, value in (("--cache-cap-gib", args.cache_cap_gib),
                              ("--context-cap", args.context_cap),
                              ("--max-lanes-cap", args.max_lanes_cap)):
            if value is not None:
                command.extend((option, str(value)))
        print(f"START {name} features source_tree={tree[:8]}", flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        status = "failed"
        if path.exists():
            receipt = json.loads(path.read_text())
            status = receipt.get("status", "missing_status")
        state["jobs"][name] = {"status": status, "returncode": result.returncode,
                               "finished_at": time.time()}
        save()
        print(f"DONE {name} features {status} rc={result.returncode}", flush=True)
        completed += 1
        if result.returncode:
            failed = True
            if not args.continue_on_failure:
                return 1
        if args.max_models and completed >= args.max_models:
            break
    state["finished_at"] = time.time()
    save()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
