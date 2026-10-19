#!/usr/bin/env python3
"""Bounded same-artifact tree15 B1 versus ordinary token discriminator.

The caller owns GPU admission and both locks. This script never claims them.
Dry run is the default; --execute starts one server at a time and stops each
process group. No result from this probe is a GPU qualification claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

from probe_qwen38_tree15_live_transition import stop, tree_scheduler_counters

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("MLX2_PROBE_PYTHON", sys.executable))
MODEL = Path.home() / "mlx-models" / "Qwen3.8-27B-oQ4e-mtp"
DRAFT = Path.home() / "mlx-models" / "Qwen3.8-27B-DFlash2"
BASE_POLICY = ROOT / "qualification/policies/qwen38-27b-dflash2-tensorfold-unified.json"
ROUTE = "tree15_b1_b4_chain_b5plus_v1"
TARGET_ARTIFACT = "fb827b658dc3d26ac569efacefd0466dbf41cc1a5cb5e3a40fc1c6175298810f"
TARGET_REVISION = "e59471c5c6fa8c6819b81cb5957bcab10736020db8bacb47fbb9089813fa93f8"
DRAFT_REVISION = "34ec93d71399f3dd6db9646194f1ad4db345715c61258d72318325e0d8095c90"
TENSORFOLD_REVISION = "1a5f38e12afbb560d8fc61c88ccb5900f7d5d170"
TENSORFOLD_TREE = "d9d141b99525a8866f48475858d7011fb192c11b"
GATES = {
    "MLX2_TENSORFOLD_CACHE_EXECUTOR": "1",
    "MLX2_DFLASH_TREE_CODEBOOK_CACHE": "1",
    "MLX2_TREE_BATCHED_TARGET_LAWS": "1",
    "MLX2_TREE_SINGLE_FENCE": "1",
    "MLX2_TREE_LOGPROBS_ON_REQUEST": "1",
    "MLX2_TREE_PIPELINE_DRAFT": "1",
    "MLX2_EXTERNAL_ROUND_TIMING": "1",
}

PROMPT = ("Write a detailed, numbered tutorial on implementing a Python HTTP "
          "server, with examples and tests. Use B1 discriminator as the section title.")
SEED = 1000
TOKENS = 192
COUNTER_KEYS = (
    "external_auto_tree_rounds", "external_tensorfold_target_rounds",
    "external_auto_tree_max_width", "external_auto_chain_rounds",
    "external_auto_mode_switches", "draft_fallbacks",
    "recovery_checkpoint_restores",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_state() -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    paths = (
        "src/mlx2/runtime/tensorfold_qwen38",
        "src/mlx2/runtime/qwen38_tensorfold.py",
        "src/mlx2/adapters/qwen38_tensorfold_source.py",
    )
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError(f"vendored TensorFold source is dirty: {dirty}")
    return {
        "mlx2_head": head,
        "path": str(ROOT / "src/mlx2/runtime/tensorfold_qwen38"),
        "revision": TENSORFOLD_REVISION,
        "tree": TENSORFOLD_TREE,
        "clean": True,
    }


def snapshot(base: str, deadline: float) -> dict:
    with urlopen(base + "/v1/status", timeout=min(5, max(0.1, deadline - time.monotonic()))) as response:
        return json.load(response)


def counters(status: dict) -> dict:
    scheduler = tree_scheduler_counters(status)
    return {key: int(scheduler.get(key, 0)) for key in COUNTER_KEYS}


def validate_counters(before: dict, after: dict, candidate: bool) -> dict:
    first, last = counters(before), counters(after)
    delta = {key: last[key] - first[key] for key in COUNTER_KEYS}
    if any(value < 0 for value in delta.values()):
        raise RuntimeError("scheduler counters reset during request")
    if candidate and (delta["external_auto_tree_rounds"] <= 0 or
                      delta["external_tensorfold_target_rounds"] <= 0 or
                      last["external_auto_tree_max_width"] != 1):
        raise RuntimeError(f"B1 TensorFold route was not observed: {delta}")
    if not candidate and (delta["external_auto_tree_rounds"] or
                          delta["external_tensorfold_target_rounds"]):
        raise RuntimeError(f"ordinary route used tree execution: {delta}")
    for key in ("external_auto_chain_rounds", "external_auto_mode_switches",
                "draft_fallbacks", "recovery_checkpoint_restores"):
        if delta[key] or last[key]:
            raise RuntimeError(f"unexpected {key}: before={first[key]}, after={last[key]}")
    return {"before": first, "after": last, "delta": delta}


def validate_status(status: dict, candidate: bool) -> None:
    lane = status.get("lane_matmul") or {}
    if lane.get("requested") != "exact" or lane.get("mode") != "exact":
        raise RuntimeError(f"exact projection law was not installed: {lane}")
    if candidate:
        speculation = (status.get("execution") or {}).get("speculation")
        if speculation != "external-dflash2-implemented-unqualified":
            raise RuntimeError(f"unexpected candidate execution status: {speculation}")


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def completion(base: str, deadline: float, candidate: bool) -> dict:
    body = {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": TOKENS,
            "temperature": 0.0, "enable_thinking": False, "logprobs": True,
            "top_logprobs": 0, "skip_writing_prefix_cache": True, "seed": SEED}
    request = Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urlopen(request, timeout=max(0.1, deadline - started)) as response:
        result = json.load(response)
    wall_seconds = time.monotonic() - started
    receipt = result.get("mlx2") or {}
    route = (receipt.get("speculation") or {}).get("batch_size_route") or {}
    expected = "external_draft" if candidate else "ordinary"
    if receipt.get("route") != expected:
        raise RuntimeError(f"route receipt mismatch: {receipt}")
    if candidate and (route.get("policy") != ROUTE or route.get("qualified") is not False):
        raise RuntimeError(f"candidate policy/qualification receipt mismatch: {route}")
    if candidate:
        speculation = receipt.get("speculation") or {}
        if speculation.get("ordinary_fallback") is not False:
            raise RuntimeError("candidate used ordinary fallback")
        target_tree = speculation.get("target_tree") or {}
        identity = target_tree.get("source_identity") or {}
        if (
            target_tree.get("provider") != "qwen38_tensorfold_v1"
            or identity.get("revision") != TENSORFOLD_REVISION
            or identity.get("tree") != TENSORFOLD_TREE
        ):
            raise RuntimeError(f"candidate TensorFold identity mismatch: {target_tree}")
    choice = result["choices"][0]
    events = (choice.get("logprobs") or {}).get("content") or []
    ids = [event.get("id") for event in events]
    if not ids or len(ids) != result["usage"]["completion_tokens"] or any(type(i) is not int for i in ids):
        raise RuntimeError("complete integer logprob token IDs are required")
    output = choice["message"].get("content") or ""
    return {"route_receipt": receipt, "completion_tokens": len(ids),
            "wall_seconds": wall_seconds, "tokens_per_second": len(ids) / wall_seconds,
            "token_ids": ids, "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "finish_reason": choice.get("finish_reason")}


def run_server(command: list[str], environment: dict, log_path: Path, base: str,
               deadline: float, candidate: bool) -> dict:
    if time.monotonic() >= deadline:
        raise TimeoutError("global deadline before server launch")
    process = None
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}")
                try:
                    before = snapshot(base, deadline)
                    if before.get("healthy") and before.get("ready", True):
                        validate_status(before, candidate)
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(min(0.5, max(0, deadline - time.monotonic())))
            else:
                raise TimeoutError("global deadline while loading server")
            row = completion(base, deadline, candidate)
            after = snapshot(base, deadline)
            validate_status(after, candidate)
            row["status_counters"] = validate_counters(before, after, candidate)
            row["scheduler_status"] = {"before": before.get("scheduler") or {},
                                       "after": after.get("scheduler") or {}}
            return row
    finally:
        stop(process)


def main() -> int:
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8755)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--execute", action="store_true", help="start both servers; caller owns GPU lease")
    args = parser.parse_args()
    if not 60 <= args.timeout_seconds <= 180 or not 1024 <= args.port <= 65535:
        parser.error("timeout must be 60..180 seconds and port 1024..65535")
    source = source_state()
    if not all(path.exists() for path in (PYTHON, MODEL / "config.json", DRAFT / "config.json")):
        raise SystemExit("required Python/model/draft input missing")
    from mlx2.adapters.qwen38_27b import inspect_artifact

    target_identity = inspect_artifact(MODEL)["identity"]
    if target_identity["fingerprint"] != TARGET_ARTIFACT:
        raise SystemExit(
            f"target artifact mismatch: {target_identity['fingerprint']} != {TARGET_ARTIFACT}"
        )
    policy = json.loads(BASE_POLICY.read_text())
    if (
        Path(policy["draft_model"]).resolve() != DRAFT.resolve()
        or policy.get("batch_size_route") != ROUTE
        or policy.get("target_revision") != TARGET_REVISION
        or policy.get("draft_revision") != DRAFT_REVISION
        or policy.get("tree_node_budget_by_lanes") != {"1": 15, "2": 7, "3": 4, "4": 3}
    ):
        raise SystemExit("base policy does not match pinned candidate")
    output = args.output_dir.resolve()
    base = f"http://127.0.0.1:{args.port}"
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", MLX_ENABLE_TF32="0", **GATES)
    environment.pop("MLX2_TENSORFOLD_SOURCE", None)
    for key in ("MLX2_DFLASH_TOPOLOGY", "MLX2_QWEN_TARGET_EXECUTION", "MLX2_TENSORFOLD_COHORT_LIMIT"):
        environment.pop(key, None)
    common = [str(PYTHON), "-u", "-c",
              "from mlx2.runtime import lane; lane.set_enabled(True); from mlx2.server import main; main()",
              "--model", str(MODEL), "--host", "127.0.0.1", "--port", str(args.port),
              "--max-context", "8192", "--max-lanes", "4", "--max-inflight", "4",
              "--cache-bytes", str(8 << 30), "--qualification-mode", "--lane-matmul",
              "exact"]
    candidate = common + ["--cache-dir", str(output / "candidate-cache"), "--external-draft",
                          "--execution-policy", str(output / "policy.json")]
    ordinary = common + ["--cache-dir", str(output / "ordinary-cache"), "--ordinary"]
    file_hashes = {"target_config_sha256": sha(MODEL / "config.json"),
                   "draft_config_sha256": sha(DRAFT / "config.json"),
                   "base_policy_sha256": sha(BASE_POLICY)}
    for label, folder in (("target", MODEL), ("draft", DRAFT)):
        index = folder / "model.safetensors.index.json"
        if index.exists():
            file_hashes[f"{label}_weights_index_sha256"] = sha(index)
    artifact_binding = {"target_artifact_fingerprint": TARGET_ARTIFACT,
                        "target_payload_revision": TARGET_REVISION,
                        "draft_payload_revision": DRAFT_REVISION}
    plan = {"schema": "mlx2.qwen38-tree15-b1-discriminator.v1",
            "created_at": datetime.now(UTC).isoformat(), "mlx2_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "tensorfold_source": source, "model": str(MODEL), "draft": str(DRAFT),
            "artifact_hashes": file_hashes, "artifact_binding": artifact_binding,
            "prompt": PROMPT, "seed": SEED,
            "max_tokens": TOKENS, "timeout_seconds": args.timeout_seconds,
            "candidate_command": candidate, "ordinary_command": ordinary,
            "environment": {key: environment[key] for key in (
                "PYTHONPATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MLX_ENABLE_TF32", *GATES)}}
    if not args.execute:
        print(json.dumps({**plan, "policy": policy}, indent=2, sort_keys=True))
        return 0
    if output.exists():
        raise SystemExit(f"output exists: {output}")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    (output / "policy.json").write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    deadline = time.monotonic() + args.timeout_seconds - 30  # reserve both server teardowns
    signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds - 15)
    try:
        left = run_server(candidate, environment, output / "candidate-server.log", base, deadline, True)
        ordinary_env = dict(environment)
        for key in GATES:
            ordinary_env.pop(key, None)
        right = run_server(ordinary, ordinary_env, output / "ordinary-server.log", base, deadline, False)
        if source_state() != source:
            raise RuntimeError("TensorFold source changed during probe")
        hash_paths = {"target_config_sha256": MODEL / "config.json",
                      "draft_config_sha256": DRAFT / "config.json",
                      "base_policy_sha256": BASE_POLICY,
                      "target_weights_index_sha256": MODEL / "model.safetensors.index.json",
                      "draft_weights_index_sha256": DRAFT / "model.safetensors.index.json"}
        if any(sha(hash_paths[key]) != value for key, value in file_hashes.items()):
            raise RuntimeError("model or policy metadata changed during probe")
        diff = first_difference(left["token_ids"], right["token_ids"])
        receipt = {"valid": diff is None, "exact_token_ids": diff is None,
                   "first_difference_zero_based": diff, "candidate": left,
                   "ordinary": right, "completed_at": datetime.now(UTC).isoformat()}
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"receipt": str(output / "receipt.json"),
                          "first_difference_zero_based": diff, "exact_token_ids": diff is None}))
        if diff is not None:
            raise RuntimeError(f"tree15 differs from exact ordinary at token {diff}")
    except BaseException as error:
        (output / "failed.json").write_text(json.dumps({"valid": False, "error": repr(error)}, indent=2) + "\n")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
