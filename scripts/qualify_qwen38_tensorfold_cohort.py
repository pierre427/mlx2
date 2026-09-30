#!/usr/bin/env python3
"""Counterbalanced Qwen3.8 TensorFold cohort-limit 1-vs-4 GPU A/B.

The script is a qualification harness, not a default selector. Run it only
under this campaign's two-lock GPU queue wrapper.
It starts a fresh server per arm, sends four synchronized greedy requests via
``bench_qwen38_b1_routes.py``, and refuses missing engagement, output drift,
thermal pressure, swapouts, fallbacks, or recovery restores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

COMPOSED_GATES = {
    "MLX2_DFLASH_TOPOLOGY": "tree15",
    "MLX2_QWEN_TARGET_EXECUTION": "tensorfold",
    "MLX2_TENSORFOLD_CACHE_EXECUTOR": "1",
    "MLX2_DFLASH_TREE_CODEBOOK_CACHE": "1",
    "MLX2_TREE_BATCHED_TARGET_LAWS": "1",
    "MLX2_TREE_SINGLE_FENCE": "1",
    "MLX2_TREE_LOGPROBS_ON_REQUEST": "1",
    "MLX2_TREE_PIPELINE_DRAFT": "1",
    "MLX2_EXTERNAL_ROUND_TIMING": "1",
}
GPU_OWNERS = (
    Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
    Path("/tmp/gpu.lock/owner.json"),
)
MODEL_PROCESS = re.compile(r"python.*(?:mlx2\.server|mlx_lm\.server)|tensorfold.*serve")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_json(url: str, timeout: float = 30) -> dict:
    with urlopen(url, timeout=timeout) as response:
        return json.load(response)


def thermal_state(probe: Path) -> int:
    result = subprocess.run(
        ["/usr/bin/swift", str(probe)], check=True, text=True, capture_output=True
    )
    return int(json.loads(result.stdout)["thermal_state"])


def swapouts() -> int:
    output = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
    match = re.search(r"^Swapouts:\s+([0-9]+)\.\s*$", output, re.MULTILINE)
    return int(match.group(1)) if match else -1


def require_gpu_owners() -> dict:
    owners = []
    for path in GPU_OWNERS:
        if not path.is_file():
            raise RuntimeError(f"missing GPU owner receipt: {path}")
        owner = json.loads(path.read_text())
        if not isinstance(owner, dict) or not owner.get("lease_id"):
            raise RuntimeError(f"invalid GPU owner receipt: {path}")
        owners.append(owner)
    if owners[0] != owners[1]:
        raise RuntimeError("GPU owner receipts disagree")
    return owners[0]


def wait_ready(base: str, process: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with rc={process.returncode}")
        try:
            status = get_json(base + "/v1/status")
            if status.get("healthy") and status.get("ready", True):
                return status
        except (OSError, URLError, TimeoutError, ValueError) as caught:
            error = repr(caught)
        time.sleep(2)
    raise TimeoutError(f"server did not become ready: {error}")


def stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(10)


def measured_summary(rows: list[dict]) -> dict:
    result = {}
    for prompt in sorted({row["prompt"] for row in rows}):
        selected = [
            row for row in rows if row["prompt"] == prompt and not row["warmup"]
        ]
        groups = {}
        for row in selected:
            groups.setdefault(int(row["rep"]), []).append(row)
        goodput = []
        makespans = []
        for group in groups.values():
            makespan = max(float(row["makespan_seconds"]) for row in group)
            tokens = sum(max(0, int(row["completion_tokens"]) - 1) for row in group)
            makespans.append(makespan)
            goodput.append(tokens / makespan)
        result[prompt] = {
            "samples": len(selected),
            "median_batch_goodput_tps": statistics.median(goodput),
            "median_makespan_seconds": statistics.median(makespans),
            "output_sha256": sorted({row["output_sha256"] for row in selected}),
        }
    return result


def validate_arm(limit: int, rows: list[dict], status: dict) -> dict:
    scheduler = status.get("scheduler") or {}
    if int(scheduler.get("external_tensorfold_cohort_limit", -1)) != limit:
        raise RuntimeError("server receipt does not match requested cohort limit")
    if int(scheduler.get("external_tensorfold_target_rounds", 0)) <= 0:
        raise RuntimeError("TensorFold target did not engage")
    if int(scheduler.get("draft_fallbacks", 0)):
        raise RuntimeError("draft fallback observed")
    if int(scheduler.get("recovery_checkpoint_restores", 0)):
        raise RuntimeError("round recovery restore observed")
    cohort_rounds = int(scheduler.get("external_tensorfold_cohort_rounds", 0))
    cohort_width = int(scheduler.get("external_tensorfold_cohort_max_width", 0))
    if limit == 1 and (cohort_rounds or cohort_width):
        raise RuntimeError("singleton arm unexpectedly used a multi-lane cohort")
    if limit == 4 and (cohort_rounds <= 0 or cohort_width != 4):
        raise RuntimeError("four-lane arm did not observe width four")
    receipts = [
        (row.get("speculation") or {}).get("tensorfold_target") for row in rows
    ]
    if any(not receipt or int(receipt.get("cohort_limit", -1)) != limit for receipt in receipts):
        raise RuntimeError("request-level TensorFold cohort receipt missing or mismatched")
    return measured_summary(rows)


def run_arm(args, root: Path, output: Path, limit: int, ordinal: int) -> dict:
    arm = output / f"arm-{ordinal}-limit-{limit}"
    if arm.exists():
        raise FileExistsError(f"refusing to overwrite {arm}")
    arm.mkdir(parents=True)
    port = args.port + ordinal
    base = f"http://127.0.0.1:{port}"
    python = Path("~/Desktop/mlx2/.venv/bin/python")
    environment = dict(
        os.environ,
        PYTHONPATH=str(root / "src"),
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        MLX2_TENSORFOLD_SOURCE=str(args.tensorfold_root.resolve()),
        MLX2_TENSORFOLD_COHORT_LIMIT=str(limit),
        **COMPOSED_GATES,
    )
    bootstrap = (
        "from mlx2.runtime import lane; lane.set_enabled(True); "
        "from mlx2.server import main; main()"
    )
    server_command = [
        str(python), "-u", "-c", bootstrap,
        "--model", str(args.model.resolve()),
        "--host", "127.0.0.1", "--port", str(port),
        "--cache-dir", str(arm / "cache"),
        "--max-context", "8192", "--max-lanes", "4", "--max-inflight", "4",
        "--cache-bytes", "1", "--qualification-mode", "--external-draft",
        "--execution-policy", str(args.execution_policy.resolve()),
        "--lane-matmul", "crossover",
        "--lane-policy", json.dumps({"max_rows": 128}),
    ]
    bench_command = [
        str(python), "-u", str(root / "scripts/bench_qwen38_b1_routes.py"),
        "--base", base, "--model-path", str(args.model.resolve()),
        "--label", f"tensorfold-cohort-{limit}",
        "--output", str(arm / "bench.json"),
        "--prompts", args.prompts, "--tokens", str(args.tokens),
        "--reps", str(args.reps), "--sampling", "greedy", "--concurrency", "4",
    ]
    initial_thermal = thermal_state(root / "scripts/thermal_probe.swift")
    if initial_thermal > args.max_start_thermal:
        raise RuntimeError(f"arm {limit} starts at thermal state {initial_thermal}")
    before_swap = swapouts()
    thermal = []
    monitor_stop = threading.Event()
    process = None

    def monitor() -> None:
        while not monitor_stop.is_set():
            try:
                thermal.append({"at": utc_now(), "state": thermal_state(root / "scripts/thermal_probe.swift")})
            except Exception as error:  # noqa: BLE001 - recorded in the receipt
                thermal.append({"at": utc_now(), "error": repr(error)})
            monitor_stop.wait(2)

    try:
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        with (arm / "server.log").open("w") as log:
            process = subprocess.Popen(
                server_command, cwd=root, env=environment,
                stdout=log, stderr=subprocess.STDOUT,
            )
            initial_status = wait_ready(base, process, args.startup_timeout)
            result = subprocess.run(bench_command, cwd=root, env=environment, check=False)
            if result.returncode:
                raise RuntimeError(f"benchmark exited rc={result.returncode}")
            time.sleep(2)
            final_status = wait_ready(base, process, 30)
        rows = json.loads((arm / "bench.json").read_text())["rows"]
        summary = validate_arm(limit, rows, final_status)
        after_swap = swapouts()
        thermal_max = max(
            (item["state"] for item in thermal if "state" in item), default=None
        )
        if thermal_max != 0:
            raise RuntimeError(f"arm {limit} observed thermal state {thermal_max}")
        if after_swap != before_swap:
            raise RuntimeError(f"arm {limit} observed swapout increase")
        receipt = {
            "schema": "mlx2.qwen38-tensorfold-cohort-arm.v1",
            "completed_at": utc_now(), "limit": limit, "valid": True,
            "server_command": server_command, "bench_command": bench_command,
            "initial_status": initial_status, "final_status": final_status,
            "thermal": thermal, "thermal_max": thermal_max,
            "swapouts_before": before_swap, "swapouts_after": after_swap,
            "summary": summary,
        }
        (arm / "arm.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt
    finally:
        monitor_stop.set()
        stop(process)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx2-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--execution-policy", type=Path, required=True)
    parser.add_argument("--tensorfold-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--order", choices=("1,4", "4,1"), default="1,4")
    parser.add_argument("--port", type=int, default=8750)
    parser.add_argument("--prompts", default="code,chat")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--cooldown-seconds", type=int, default=90)
    parser.add_argument("--max-start-thermal", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.mlx2_root.resolve()
    output = args.output_dir.resolve()
    limits = [int(value) for value in args.order.split(",")]
    plan = {
        "schema": "mlx2.qwen38-tensorfold-cohort-ab-plan.v1",
        "created_at": utc_now(), "source_root": str(root),
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True
        ).splitlines(),
        "limits": limits, "concurrency": 4, "prompts": args.prompts,
        "tokens": args.tokens, "reps": args.reps,
        "controls": {
            "same_model_process_within_arm": True,
            "fresh_process_between_arms": True,
            "greedy_output_hash_parity": True,
            "required_width_four_receipt": True,
            "required_singleton_nonengagement": True,
            "required_thermal_state": args.max_start_thermal,
            "required_swapout_delta": 0,
        },
        "invocation": "qualification/runs/tensorfold-coalescing-20260930/gpuq.sh cohort-ab python scripts/qualify_qwen38_tensorfold_cohort.py ...",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    try:
        plan["gpu_owner"] = require_gpu_owners()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"two-lock GPU ownership is required: {exc}") from exc
    resident = subprocess.run(
        ["pgrep", "-fl", "python|tensorfold"],
        text=True,
        capture_output=True,
        check=False,
    )
    conflicts = [line for line in resident.stdout.splitlines() if MODEL_PROCESS.search(line)]
    if conflicts:
        raise SystemExit("model process already resident:\n" + "\n".join(conflicts))
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    for path in (args.model, args.draft, args.execution_policy, args.tensorfold_root):
        if not path.exists():
            raise SystemExit(f"required input does not exist: {path}")
    policy = json.loads(args.execution_policy.read_text())
    policy_draft = Path(policy.get("draft_model", "")).resolve()
    if policy_draft != args.draft.resolve():
        raise SystemExit(
            f"execution policy draft_model {policy_draft} does not match {args.draft.resolve()}"
        )
    output.mkdir(parents=True)
    plan["artifacts"] = {
        "model_config_sha256": sha256(args.model / "config.json"),
        "draft_config_sha256": sha256(args.draft / "config.json"),
        "execution_policy_sha256": sha256(args.execution_policy),
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    arms = []
    try:
        for ordinal, limit in enumerate(limits):
            if ordinal:
                time.sleep(args.cooldown_seconds)
            arms.append(run_arm(args, root, output, limit, ordinal))
        left_rows = json.loads((output / f"arm-0-limit-{limits[0]}" / "bench.json").read_text())["rows"]
        right_rows = json.loads((output / f"arm-1-limit-{limits[1]}" / "bench.json").read_text())["rows"]
        identity = lambda row: (row["prompt"], row["sampling"], row["rep"], row["lane"])
        left_hashes = {identity(row): row["output_sha256"] for row in left_rows}
        right_hashes = {identity(row): row["output_sha256"] for row in right_rows}
        if left_hashes != right_hashes:
            raise RuntimeError("cohort limits changed greedy output hashes")
        by_limit = {arm["limit"]: arm for arm in arms}
        speedup = {}
        for prompt in by_limit[1]["summary"]:
            one = by_limit[1]["summary"][prompt]["median_batch_goodput_tps"]
            four = by_limit[4]["summary"][prompt]["median_batch_goodput_tps"]
            speedup[prompt] = four / one
        campaign = {
            **plan, "completed_at": utc_now(), "valid": True,
            "arms": arms, "output_hashes_equal": True,
            "limit4_over_limit1_batch_goodput": speedup,
            "qualification": "candidate-not-selected",
        }
        (output / "campaign.json").write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"valid": True, "speedup": speedup}, indent=2))
        return 0
    except BaseException as error:
        failed = {**plan, "completed_at": utc_now(), "valid": False, "error": repr(error)}
        (output / "campaign.failed.json").write_text(json.dumps(failed, indent=2, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
