"""Short, same-artifact Qwen3.8 chain/TensorFold versus tree15/TensorFold B2-B4 probe.

The caller owns GPU queue admission and both GPU locks. This runner never claims
them. It launches one server at a time and kills its process group on every exit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
MODEL = Path.home() / "mlx-models" / "Qwen3.8-27B-MLX-4bit"
DRAFT = Path.home() / "mlx-models" / "Qwen3.8-27B-DFlash2"
SOURCE = Path.home() / ".codex/worktrees/tensorfold-upstream-parity-20260928"
POLICY = ROOT / "qualification/runs/tensorfold-upstream-parity-20260928/mlx2-policy.json"
PIN = "71377a5373ed7b394f1b480ba2a6a3986b03af1c"
GATES = {
    "MLX2_TENSORFOLD_CACHE_EXECUTOR": "1",
    "MLX2_DFLASH_TREE_CODEBOOK_CACHE": "1",
    "MLX2_TREE_BATCHED_TARGET_LAWS": "1",
    "MLX2_TREE_SINGLE_FENCE": "1",
    "MLX2_TREE_LOGPROBS_ON_REQUEST": "1",
    "MLX2_TREE_PIPELINE_DRAFT": "1",
    "MLX2_EXTERNAL_ROUND_TIMING": "1",
}


def status(base: str) -> dict:
    with urlopen(base + "/v1/status", timeout=5) as response:
        return json.load(response)


def stop(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def main() -> int:
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8752)
    parser.add_argument("--order", choices=("chain,tree15", "tree15,chain"), default="chain,tree15")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--concurrency", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--cohort-limit", type=int, choices=(2, 3, 4))
    parser.add_argument("--max-lanes", type=int, choices=(2, 3, 4))
    parser.add_argument("--max-inflight", type=int, choices=(2, 3, 4))
    parser.add_argument("--timeout-seconds", type=int, default=480)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-unprovisioned-plan",
        action="store_true",
        help="allow dry-run plan output while recording missing bound inputs",
    )
    args = parser.parse_args()
    if not 32 <= args.tokens <= 256 or not 1 <= args.reps <= 2:
        parser.error("bounded cell requires tokens 32..256 and reps 1..2")
    if not 60 <= args.timeout_seconds <= 480:
        parser.error("timeout must be 60..480 seconds")
    if args.port < 1024 or args.port > 65533:
        parser.error("port must leave room for two local servers")
    cohort_limit = args.cohort_limit or args.concurrency
    max_lanes = args.max_lanes or args.concurrency
    max_inflight = args.max_inflight or args.concurrency
    if cohort_limit != args.concurrency:
        parser.error("physical B-n cell requires cohort limit equal to concurrency")
    if max_lanes < args.concurrency or max_inflight < args.concurrency:
        parser.error("max lanes and max inflight must admit all concurrent requests")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    input_failures = []
    source_head = None
    if (SOURCE / ".git").exists() or (SOURCE / "HEAD").exists():
        source_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
        ).strip()
        if source_head != PIN:
            input_failures.append(
                f"TensorFold source revision mismatch: {source_head}"
            )
    else:
        input_failures.append("TensorFold source checkout missing")
    try:
        policy = json.loads(POLICY.read_text())
    except (OSError, ValueError) as error:
        policy = {"draft_model": str(DRAFT)}
        input_failures.append(f"policy unavailable: {type(error).__name__}")
    if Path(policy.get("draft_model", "")).resolve() != DRAFT.resolve():
        input_failures.append("policy draft path mismatch")
    for label, path in (
        ("python", PYTHON),
        ("model", MODEL / "config.json"),
        ("draft", DRAFT / "config.json"),
    ):
        if not path.exists():
            input_failures.append(f"required {label} input missing")
    if input_failures and not (args.dry_run and args.allow_unprovisioned_plan):
        raise SystemExit("; ".join(input_failures))

    def optional_sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    plan = {"schema": "mlx2.qwen38-tree15-b2-b4-crossover.v1", "created_at": datetime.now(UTC).isoformat(),
            "source_head": head, "tensorfold_head": source_head,
            "policy_sha256": optional_sha256(POLICY),
            "model_config_sha256": optional_sha256(MODEL / "config.json"),
            "draft_config_sha256": optional_sha256(DRAFT / "config.json"),
            "input_readiness": {"ready": not input_failures, "failures": input_failures},
            "order": args.order, "concurrency": args.concurrency,
            "cohort_limit": cohort_limit, "max_lanes": max_lanes,
            "max_inflight": max_inflight, "tokens": args.tokens, "reps": args.reps,
            "timeout_seconds": args.timeout_seconds, "arms": []}
    output = args.output_dir.resolve()
    for index, arm in enumerate(args.order.split(",")):
        base = f"http://127.0.0.1:{args.port + index}"
        directory = output / arm
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), HF_HUB_OFFLINE="1",
                           TRANSFORMERS_OFFLINE="1", MLX2_TENSORFOLD_SOURCE=str(SOURCE),
                           MLX2_DFLASH_TOPOLOGY=arm, MLX2_QWEN_TARGET_EXECUTION="tensorfold",
                           MLX2_TENSORFOLD_COHORT_LIMIT=str(cohort_limit))
        for key in GATES:
            environment.pop(key, None)
        if arm == "tree15":
            environment.update(GATES)
        server = [str(PYTHON), "-u", "-c", "from mlx2.runtime import lane; lane.set_enabled(True); from mlx2.server import main; main()",
                  "--model", str(MODEL), "--host", "127.0.0.1", "--port", str(args.port + index),
                  "--cache-dir", str(directory / "cache"), "--max-context", "8192", "--max-lanes", str(max_lanes),
                  "--max-inflight", str(max_inflight), "--cache-bytes", str(8 << 30), "--qualification-mode",
                  "--external-draft", "--execution-policy", str(POLICY), "--lane-matmul", "crossover",
                  "--lane-policy", '{"max_rows":128}']
        bench = [str(PYTHON), str(ROOT / "scripts/bench_qwen38_b1_routes.py"), "--base", base,
                 "--model-path", str(MODEL), "--label", f"{arm}-b{args.concurrency}", "--output", str(directory / "bench.json"),
                 "--prompts", "code,chat", "--tokens", str(args.tokens), "--reps", str(args.reps),
                 "--sampling", "greedy", "--concurrency", str(args.concurrency)]
        plan["arms"].append({"arm": arm, "server": server, "bench": bench,
                             "environment": {key: environment[key] for key in (
                                 "PYTHONPATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                                 "MLX2_TENSORFOLD_SOURCE", "MLX2_DFLASH_TOPOLOGY",
                                 "MLX2_QWEN_TARGET_EXECUTION", "MLX2_TENSORFOLD_COHORT_LIMIT", *GATES)
                                 if key in environment}})
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if output.exists():
        raise SystemExit(f"output exists: {output}")
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    # Reserve bounded status and process-group teardown time inside the cap.
    deadline = time.monotonic() + args.timeout_seconds - 20
    try:
        for arm in plan["arms"]:
            directory = output / arm["arm"]
            directory.mkdir()
            environment = dict(os.environ, **arm["environment"])
            for key in GATES:
                if key not in arm["environment"]:
                    environment.pop(key, None)
            process = None
            try:
                with (directory / "server.log").open("w") as log:
                    process = subprocess.Popen(arm["server"], cwd=ROOT, env=environment,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f"{arm['arm']} server exited {process.returncode}")
                        if time.monotonic() >= deadline:
                            raise TimeoutError("8-minute global cap")
                        try:
                            initial = status(arm["bench"][arm["bench"].index("--base") + 1])
                            if initial.get("healthy") and initial.get("ready", True):
                                break
                        except (OSError, ValueError):
                            pass
                        time.sleep(2)
                    remaining = max(1, deadline - time.monotonic())
                    subprocess.run(arm["bench"], cwd=ROOT, env=environment, check=True,
                                   timeout=remaining, stdout=log, stderr=subprocess.STDOUT)
                    final = status(arm["bench"][arm["bench"].index("--base") + 1])
                counters = final.get("scheduler") or {}
                width = int(counters.get("external_tensorfold_cohort_max_width", 0))
                if (int(counters.get("external_tensorfold_cohort_limit", 0)) != cohort_limit
                        or int(counters.get("target_max_width", 0)) != args.concurrency
                        or width != args.concurrency
                        or int(counters.get("external_tensorfold_cohort_rounds", 0)) <= 0):
                    raise RuntimeError(f"{arm['arm']} did not engage physical B{args.concurrency} TensorFold cohort: {width}")
                if int(counters.get("draft_fallbacks", 0)) or int(counters.get("recovery_checkpoint_restores", 0)):
                    raise RuntimeError(f"{arm['arm']} fallback or restore observed")
                if arm["arm"] == "tree15" and int(counters.get("external_tree_rounds", 0)) <= 0:
                    raise RuntimeError("tree15 rounds did not engage")
                rows = json.loads((directory / "bench.json").read_text())["rows"]
                expected_rows = 2 * args.concurrency * (args.reps + 1)
                if len(rows) != expected_rows or not all(
                    row["route"] == "external_draft" and row["concurrency"] == args.concurrency
                    for row in rows
                ):
                    raise RuntimeError(f"{arm['arm']} route receipt mismatch")
                (directory / "receipt.json").write_text(json.dumps({"status": final, "width": width,
                    "requested_concurrency": args.concurrency,
                    "rows": len(rows), "completed_at": datetime.now(UTC).isoformat()}, indent=2) + "\n")
            finally:
                stop(process)
        (output / "complete.json").write_text(json.dumps({"valid": True, "completed_at": datetime.now(UTC).isoformat()}, indent=2) + "\n")
    except BaseException as error:
        (output / "failed.json").write_text(json.dumps({"valid": False, "error": repr(error)}, indent=2) + "\n")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
