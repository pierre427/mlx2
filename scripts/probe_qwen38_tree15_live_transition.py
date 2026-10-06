#!/usr/bin/env python3
"""Bounded one-service B1 -> B2 -> B4 -> B1 tree15 transition gate.

The caller owns GPU admission and both locks. This script never claims them.
It starts one server, uses one long-lived anchor request, and tears down its
process group on every exit. A dry run only prints the pinned launch plan.
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
MODEL = Path.home() / "mlx-models" / "Qwen3.8-27B-MLX-4bit"
DRAFT = Path.home() / "mlx-models" / "Qwen3.8-27B-DFlash2"
SOURCE = Path("/private/tmp/tensorfold-71377a5-mlx2")
SOURCE_PIN = "71377a5373ed7b394f1b480ba2a6a3986b03af1c"
BASE_POLICY = ROOT / "qualification/runs/tensorfold-upstream-parity-20260928/mlx2-policy.json"
ROUTE = "tree15_b1_b4_chain_b5plus_v1"
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


def counters(snapshot: dict) -> dict:
    return snapshot.get("scheduler") or {}


def verify_stage(previous: dict, current: dict, width: int) -> None:
    before, after = counters(previous), counters(current)
    if int(after.get("external_auto_tree_rounds", 0)) <= int(before.get("external_auto_tree_rounds", 0)):
        raise ValueError("tree rounds did not advance")
    physical_rounds = (
        "external_tensorfold_target_rounds" if width == 1
        else "external_tensorfold_cohort_rounds"
    )
    if int(after.get(physical_rounds, 0)) <= int(before.get(physical_rounds, 0)):
        raise ValueError(f"TensorFold {physical_rounds} did not advance")
    if int(after.get("external_auto_tree_max_width", 0)) < width:
        raise ValueError(f"tree route did not reach B{width}")
    if width > 1 and int(after.get("external_tensorfold_cohort_max_width", 0)) < width:
        raise ValueError(f"physical TensorFold cohort did not reach B{width}")
    if int(after.get("external_auto_chain_rounds", 0)) or int(after.get("external_auto_mode_switches", 0)):
        raise RuntimeError("chain/reference mode engaged")
    if int(after.get("draft_fallbacks", 0)) or int(after.get("recovery_checkpoint_restores", 0)):
        raise RuntimeError("fallback or checkpoint restore observed")


def request(base: str, label: str, tokens: int, started: threading.Event,
            *, ordinary: bool = False, timeout: float = 420) -> dict:
    body = {"messages": [{"role": "user", "content":
             "Write a detailed, numbered tutorial on implementing a Python HTTP server, with examples and tests. "
             f"Use {label} as the section title."}],
            "max_tokens": tokens, "temperature": 0.0, "enable_thinking": False,
            "logprobs": True, "top_logprobs": 0,
            "skip_writing_prefix_cache": True,
            "seed": {"anchor": 1000, "peer2": 1001, "peer3": 1002, "peer4": 1003}[label]}
    started.set()
    with urlopen(Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                         headers={"Content-Type": "application/json"}), timeout=timeout) as response:
        result = json.load(response)
    receipt = result.get("mlx2") or {}
    route = (receipt.get("speculation") or {}).get("batch_size_route") or {}
    expected_route = "ordinary" if ordinary else "external_draft"
    if receipt.get("route") != expected_route or (not ordinary and route.get("policy") != ROUTE):
        raise RuntimeError(f"{label} route receipt mismatch: {receipt.get('route')}, {route}")
    if not ordinary and route.get("qualified") is not False:
        raise RuntimeError(f"{label} unexpectedly qualified")
    text = (result["choices"][0]["message"].get("content") or "")
    token_events = (result["choices"][0].get("logprobs") or {}).get("content")
    if not token_events or len(token_events) != result["usage"]["completion_tokens"]:
        raise RuntimeError(f"{label} token-event receipt missing or incomplete")
    token_ids = [event.get("id") for event in token_events]
    if any(type(item) is not int for item in token_ids):
        raise RuntimeError(f"{label} token ids missing")
    return {"label": label, "completion_tokens": result["usage"]["completion_tokens"],
            "output_sha256": hashlib.sha256(text.encode()).hexdigest(), "route": route,
            "token_ids_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
            "token_ids": token_ids, "finish_reason": result["choices"][0].get("finish_reason")}


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
    parser.add_argument("--port", type=int, default=8754)
    parser.add_argument("--anchor-tokens", type=int, default=768)
    parser.add_argument("--peer2-tokens", type=int, default=384)
    parser.add_argument("--peer34-tokens", type=int, default=96)
    parser.add_argument("--timeout-seconds", type=int, default=480)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 384 <= args.anchor_tokens <= 1536 or not 128 <= args.peer2_tokens <= 768:
        parser.error("token budgets out of bounds")
    if not 32 <= args.peer34_tokens <= 256 or not args.anchor_tokens > args.peer2_tokens > args.peer34_tokens:
        parser.error("require anchor > peer2 > peer34 token budgets")
    if not 120 <= args.timeout_seconds <= 480 or not 1024 <= args.port <= 65535:
        parser.error("timeout or port out of bounds")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    if source_head != SOURCE_PIN:
        raise SystemExit(f"TensorFold source revision mismatch: {source_head}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE):
        raise SystemExit("TensorFold source checkout is dirty")
    if not all(path.exists() for path in (PYTHON, MODEL / "config.json", DRAFT / "config.json")):
        raise SystemExit("required Python/model/draft input missing")
    policy = json.loads(BASE_POLICY.read_text())
    if Path(policy["draft_model"]).resolve() != DRAFT.resolve() or "batch_size_route" in policy:
        raise SystemExit("base policy does not match expected candidate")
    policy["batch_size_route"] = ROUTE
    output = args.output_dir.resolve()
    base = f"http://127.0.0.1:{args.port}"
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", MLX2_TENSORFOLD_SOURCE=str(SOURCE), **GATES)
    for key in ("MLX2_DFLASH_TOPOLOGY", "MLX2_QWEN_TARGET_EXECUTION", "MLX2_TENSORFOLD_COHORT_LIMIT"):
        environment.pop(key, None)
    server = [str(PYTHON), "-u", "-c", "from mlx2.runtime import lane; lane.set_enabled(True); from mlx2.server import main; main()",
              "--model", str(MODEL), "--host", "127.0.0.1", "--port", str(args.port),
              "--cache-dir", str(output / "cache"), "--max-context", "8192", "--max-lanes", "4",
              "--max-inflight", "4", "--cache-bytes", str(8 << 30), "--qualification-mode",
              "--external-draft", "--execution-policy", str(output / "policy.json"),
              "--lane-matmul", "crossover", "--lane-policy", '{"max_rows":128}']
    ordinary_server = list(server)
    ordinary_server[ordinary_server.index("--cache-dir") + 1] = str(output / "ordinary-cache")
    ordinary_server.remove("--external-draft")
    policy_index = ordinary_server.index("--execution-policy")
    del ordinary_server[policy_index:policy_index + 2]
    plan = {"schema": "mlx2.qwen38-tree15-live-transition.v1", "created_at": datetime.now(UTC).isoformat(),
            "mlx2_head": head, "tensorfold_head": source_head, "base_policy_sha256": hashlib.sha256(BASE_POLICY.read_bytes()).hexdigest(),
            "model_config_sha256": hashlib.sha256((MODEL / "config.json").read_bytes()).hexdigest(),
            "draft_config_sha256": hashlib.sha256((DRAFT / "config.json").read_bytes()).hexdigest(),
            "route": ROUTE, "sequence": [1, 2, 4, 1], "timeout_seconds": args.timeout_seconds,
            "tokens": {"anchor": args.anchor_tokens, "peer2": args.peer2_tokens, "peer3": args.peer34_tokens, "peer4": args.peer34_tokens},
            "server": server, "ordinary_server": ordinary_server,
            "environment": {key: environment[key] for key in (
                "PYTHONPATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MLX2_TENSORFOLD_SOURCE", *GATES)}}
    if args.dry_run:
        print(json.dumps({**plan, "policy": policy}, indent=2, sort_keys=True))
        return 0
    if output.exists():
        raise SystemExit(f"output exists: {output}")
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError as error:
            raise SystemExit(f"port {args.port} is occupied: {error}") from error
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    (output / "policy.json").write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    deadline = time.monotonic() + args.timeout_seconds - 20
    process = None
    try:
        with (output / "server.log").open("w") as log:
            process = subprocess.Popen(server, cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("global deadline while loading")
                try:
                    initial = status(base)
                    if initial.get("healthy") and initial.get("ready", True):
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            stages = []
            def wait_stage(previous, width, pending):
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("server exited during transition")
                    current = status(base)
                    try:
                        verify_stage(previous, current, width)
                    except ValueError as missing:
                        if any(future.done() for future in pending):
                            failures = [repr(future.exception()) for future in pending if future.done()]
                            raise RuntimeError(
                                f"request finished before B{width} was observed: {missing}; "
                                f"future_exceptions={failures}; scheduler={counters(current)}"
                            ) from missing
                        time.sleep(0.1)
                        continue
                    if any(future.done() for future in pending):
                        raise RuntimeError(f"request finished before B{width} cohort could be extended")
                    stages.append({"width": width, "at": datetime.now(UTC).isoformat(), "status": current})
                    return current
                raise TimeoutError(f"B{width} transition not observed")
            with ThreadPoolExecutor(max_workers=4) as pool:
                def launch(label, tokens):
                    started = threading.Event()
                    future = pool.submit(request, base, label, tokens, started)
                    if not started.wait(timeout=5):
                        raise RuntimeError(f"{label} did not launch")
                    return future
                anchor = launch("anchor", args.anchor_tokens)
                b1 = wait_stage(initial, 1, [anchor])
                peer2 = launch("peer2", args.peer2_tokens)
                b2 = wait_stage(b1, 2, [anchor, peer2])
                peer3 = launch("peer3", args.peer34_tokens)
                peer4 = launch("peer4", args.peer34_tokens)
                wait_stage(b2, 4, [anchor, peer2, peer3, peer4])
                peers = [peer2, peer3, peer4]
                for peer in peers:
                    peer.result(timeout=max(1, deadline - time.monotonic()))
                post_peers = status(base)
                wait_stage(post_peers, 1, [anchor])
                rows = [future.result(timeout=max(1, deadline - time.monotonic()))
                        for future in (anchor, peer2, peer3, peer4)]
            final = status(base)
            if int(counters(final).get("external_tensorfold_cohort_limit", 0)) != 4:
                raise RuntimeError("TensorFold cohort limit was not four")
            verify_stage(initial, final, 4)
            (output / "transition.json").write_text(json.dumps({
                "stages": stages, "requests": rows, "final_status": final,
            }, indent=2) + "\n")
            stop(process)
            process = None
            with (output / "ordinary-server.log").open("w") as ordinary_log:
                ordinary_env = dict(environment)
                for key in GATES:
                    ordinary_env.pop(key, None)
                process = subprocess.Popen(ordinary_server, cwd=ROOT, env=ordinary_env,
                                           stdout=ordinary_log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"ordinary server exited {process.returncode}")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("global deadline while loading ordinary reference")
                    try:
                        reference_status = status(base)
                        if reference_status.get("healthy") and reference_status.get("ready", True):
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(1)
                references = []
                for row in rows:
                    label = row["label"]
                    if time.monotonic() >= deadline:
                        raise TimeoutError("global deadline during ordinary reference")
                    references.append(request(base, label, plan["tokens"][label],
                                              threading.Event(), ordinary=True,
                                              timeout=max(1, deadline - time.monotonic())))
            (output / "comparison.json").write_text(json.dumps({
                "requests": rows, "ordinary_references": references,
                "first_differences": {
                    left["label"]: next((i for i, (a, b) in enumerate(zip(
                        left["token_ids"], right["token_ids"])) if a != b),
                        None if len(left["token_ids"]) == len(right["token_ids"])
                        else min(len(left["token_ids"]), len(right["token_ids"])))
                    for left, right in zip(rows, references)
                },
            }, indent=2) + "\n")
            if any(left["token_ids"] != right["token_ids"] for left, right in zip(rows, references)):
                raise RuntimeError("exact token IDs differ from ordinary reference")
            if any(left["output_sha256"] != right["output_sha256"] for left, right in zip(rows, references)):
                raise RuntimeError("output text differs from ordinary reference")
            if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip() != SOURCE_PIN:
                raise RuntimeError("TensorFold source revision changed during gate")
            if subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE):
                raise RuntimeError("TensorFold source checkout changed during gate")
            if hashlib.sha256((MODEL / "config.json").read_bytes()).hexdigest() != plan["model_config_sha256"]:
                raise RuntimeError("target model config changed during gate")
            if hashlib.sha256((DRAFT / "config.json").read_bytes()).hexdigest() != plan["draft_config_sha256"]:
                raise RuntimeError("draft model config changed during gate")
            (output / "receipt.json").write_text(json.dumps({"valid": True, "stages": stages,
                "requests": rows, "ordinary_references": references, "final_status": final,
                "ordinary_status": reference_status,
                "completed_at": datetime.now(UTC).isoformat()}, indent=2) + "\n")
    except BaseException as error:
        (output / "failed.json").write_text(json.dumps({"valid": False, "error": repr(error)}, indent=2) + "\n")
        raise
    finally:
        stop(process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
