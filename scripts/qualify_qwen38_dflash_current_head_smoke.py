#!/usr/bin/env python3
"""Current-head B2 packed TensorFold/DFlash mechanism gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = (
    ROOT / "qualification/runs/dflash-tree-geometry-20261005/run_geometry.py"
)
GPU_OWNERS = (
    Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
    Path("/tmp/gpu.lock/owner.json"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=path, text=True).strip()


def require_gpu_owners() -> dict:
    owners = []
    for path in GPU_OWNERS:
        if not path.is_file():
            raise RuntimeError(
                f"missing GPU owner receipt {path}; use run_with_gpu_locks.py"
            )
        owner = json.loads(path.read_text())
        if not isinstance(owner, dict) or not owner.get("lease_id"):
            raise RuntimeError(f"invalid GPU owner receipt: {path}")
        owners.append(owner)
    if owners[0] != owners[1]:
        raise RuntimeError("GPU owner receipts disagree")
    return owners[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8161)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing existing output directory: {args.output_dir}")

    spec = importlib.util.spec_from_file_location("dflash_tree_geometry", GEOMETRY)
    geometry = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(geometry)
    geometry.EVIDENCE = args.output_dir
    geometry.BASE_POLICY = args.policy.resolve()
    if not geometry.BASE_POLICY.is_file():
        raise RuntimeError(f"policy is missing: {geometry.BASE_POLICY}")

    source_commit = git(ROOT, "rev-parse", "HEAD")
    if source_commit != args.expected_source:
        raise RuntimeError(
            f"mlx2 source {source_commit} != expected {args.expected_source}"
        )
    production_diff = git(ROOT, "status", "--porcelain", "--", "src", "scripts")
    if production_diff:
        raise RuntimeError("current-head smoke requires clean src/ and scripts/")
    gpu_owner = require_gpu_owners()
    args.output_dir.mkdir(parents=True)

    rc = geometry.run_arm(
        limit=2,
        port=args.port,
        tokens=args.tokens,
        reps=args.repetitions,
        node_budget=None,
        concurrency=2,
        arm_suffix="-current-head-smoke",
    )
    if rc:
        return rc

    result_path = args.output_dir / "limit2-b2-current-head-smoke.json"
    result = json.loads(result_path.read_text())
    measured = [row for row in result["rows"] if not row["warmup"]]
    by_prompt: dict[str, list[dict]] = {}
    for row in measured:
        by_prompt.setdefault(row["prompt"], []).append(row)

    failures: list[str] = []
    for prompt, rows in sorted(by_prompt.items()):
        if len(rows) != 2:
            failures.append(f"{prompt}: expected two measured lanes")
            continue
        if len({row["output_sha256"] for row in rows}) != 1:
            failures.append(f"{prompt}: lane output hashes differ")
        for row in rows:
            speculation = row.get("speculation") or {}
            route = speculation.get("batch_size_route") or {}
            target = speculation.get("tensorfold_target") or {}
            settings = speculation.get("draft_settings") or {}
            if route.get("current") != "tree15_tensorfold":
                failures.append(f"{prompt}: tree15 TensorFold route not active")
            if route.get("cohort_width") != 2:
                failures.append(f"{prompt}: cohort width is not two")
            if int(target.get("packed_target_rounds", 0)) <= 0:
                failures.append(f"{prompt}: no packed target rounds")
            if int(target.get("packed_target_lanes", 0)) < 2:
                failures.append(f"{prompt}: packed target lanes below two")
            if int(target.get("physical_target_forwards", 0)) <= 0:
                failures.append(f"{prompt}: no physical target forwards")
            if int(target.get("cohort_max_width", 0)) != 2:
                failures.append(f"{prompt}: packed target never reached B2")
            if int(settings.get("minimum_proposal_length", 0)) < 3:
                failures.append(f"{prompt}: proposal floor below three")

    tensorfold = Path(geometry.TENSORFOLD)
    summary = {
        "schema": "mlx2.current-head-packed-dflash-smoke.v1",
        "source_commit": source_commit,
        "source_production_diff": production_diff,
        "gpu_owner": gpu_owner,
        "model_config_sha256": sha256(Path(geometry.MODEL) / "config.json"),
        "policy_sha256": sha256(geometry.BASE_POLICY),
        "tensorfold_commit": git(tensorfold, "rev-parse", "HEAD"),
        "tensorfold_source": str(tensorfold),
        "result": str(result_path.resolve()),
        "measured_rows": len(measured),
        "prompts": sorted(by_prompt),
        "failures": failures,
        "passed": not failures,
    }
    (args.output_dir / "current-head-b2-smoke-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
