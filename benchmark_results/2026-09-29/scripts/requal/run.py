#!/usr/bin/env python3
"""Run one source-bound requalification stage in an owned server process.

This deliberately uses host file locks and never invokes CPG. Existing servers
and checkouts are left alone. Each invocation owns only its server process.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import ExitStack
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "qualification/runs/series-20260924"))
import campaign_config as config  # noqa: E402


def get_json(url: str, timeout: float = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post_json(url: str, body: dict, timeout: float = 1200) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def lock_host(stack: ExitStack) -> list[str]:
    waiters = Path("/Users/Shared/mlxuag/gpu.lock.waiters")
    if waiters.is_dir() and any(waiters.iterdir()):
        raise RuntimeError(f"GPU waiters present in {waiters}")
    paths = [Path("/Users/Shared/mlxuag/gpu.lock"), Path("/tmp/gpu.lock")]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        stack.callback(os.close, fd)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"GPU lock busy: {path}") from exc
        stack.callback(fcntl.flock, fd, fcntl.LOCK_UN)
    if waiters.is_dir() and any(waiters.iterdir()):
        raise RuntimeError(f"GPU waiter appeared during admission: {waiters}")
    processes = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
    foreign = []
    for row in processes.splitlines():
        fields = row.strip().split(None, 1)
        if len(fields) != 2 or int(fields[0]) in {os.getpid(), os.getppid()}:
            continue
        command = fields[1]
        if " -m mlx2.server" in command or "rapid-mlx" in command or "llama-server" in command:
            foreign.append(row.strip()[:300])
    if foreign:
        raise RuntimeError(f"Foreign model server present: {foreign[:5]}")
    return [str(path) for path in paths]


def wait_ready(base: str, process: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during load: rc={process.returncode}; last={last}")
        try:
            health = get_json(base + "/health")
            last = health
            if health.get("status") == "ok":
                return health
            if health.get("error"):
                raise RuntimeError(f"server reported load error: {health['error']}")
        except urllib.error.HTTPError as exc:
            if exc.code not in {503, 429}:
                raise
        except (TimeoutError, ConnectionError, urllib.error.URLError):
            pass
        time.sleep(2)
    raise TimeoutError(f"server load exceeded {timeout}s; last={last}")


def smoke(base: str, model_id: str, family: str) -> dict:
    if family == "gemma4":
        # These are base checkpoints with no chat template. The pinned
        # Gemma 4 serving smoke uses a Q/A continuation rather than a bare
        # instruction; the latter elicits repetitive training-like prose in
        # both mlx2 and stock mlx-vlm.
        question = "Question: What is the capital of Japan, and what is it famous for?\nAnswer:"
        cases = (("qa-continuation-greedy", question, "tokyo", 0),
                 ("qa-continuation-default-sampling", question, "tokyo", None))
        max_tokens = 96
    else:
        cases = (
            ("arithmetic", "What is 17 + 25? Give the answer as a number.", "42", 0),
            ("knowledge-default-sampling", "What is the capital of France?", "paris", None),
        )
        max_tokens = 2048
    rows = []
    for name, prompt, expected, temperature in cases:
        started = time.monotonic()
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        data = post_json(base + "/v1/chat/completions", body)
        choices = data.get("choices") or []
        message = choices[0].get("message") or {} if choices else {}
        content = message.get("content") or ""
        lower = content.lower()
        passed = bool(expected in lower and not any(mark in content for mark in ("<|", "[PAD]", "�")))
        rows.append({
            "case": name, "passed": passed, "content": content[:1200],
            "reasoning_chars": len(message.get("reasoning_content") or ""),
            "finish_reason": choices[0].get("finish_reason") if choices else None,
            "usage": data.get("usage"), "route_receipt": data.get("mlx2"),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
    return {"passed": all(row["passed"] for row in rows),
            "prompt_style": "base_qa_continuation" if family == "gemma4" else "instruction",
            "cases": rows}


def stop_owned(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--route", default=None, help="named route; defaults to the model's selected route")
    parser.add_argument("--tag", default="", help="filename-safe suffix for an isolated smoke receipt")
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--port", type=int, default=8397)
    parser.add_argument("--load-timeout", type=float, default=2400)
    args = parser.parse_args()
    models = {model.name: model for model in config.MODELS}
    model = models.get(args.model)
    if model is None:
        parser.error(f"model not present on this host: {args.model}")
    if args.tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("tag must contain only letters, digits, underscores or hyphens")
    route = next((route for route in model.routes
                  if route.name == (args.route or model.default_route)), None)
    if route is None:
        parser.error(f"route {args.route!r} is not declared for {model.name}")
    output = HERE / args.host_label / model.name
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"-{route.name}" if route.name != model.default_route else ""
    if args.tag:
        suffix += f"-{args.tag}"
    receipt_path = output / f"smoke{suffix}.json"
    server_log = output / f"server{suffix}.log"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config_path = Path(model.path) / "config.json"
    receipt = {
        "schema": "mlx2.requal.smoke.v1", "host": args.host_label,
        "source_head": head, "model": model.name, "artifact": model.path,
        "artifact_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "route": route.name, "started_at": time.time(), "status": "running",
    }
    def save() -> None:
        receipt["updated_at"] = time.time()
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    save()
    base = f"http://127.0.0.1:{args.port}"
    command = [str(config.PYTHON), "-u", "-m", "mlx2.server", *config.server_args(model, route, "smoke")]
    command[command.index("--port") + 1] = str(args.port)
    receipt["command"] = command
    env = {**os.environ, "PYTHONPATH": config.stage_pythonpath(model), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    process = None
    try:
        with ExitStack() as stack:
            receipt["locks"] = lock_host(stack)
            try:
                get_json(base + "/health", timeout=1)
            except (OSError, urllib.error.URLError):
                pass
            else:
                raise RuntimeError(f"port {args.port} already serves /health")
            with server_log.open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                receipt["server_pid"] = process.pid
                save()
                try:
                    receipt["health_at_ready"] = wait_ready(base, process, args.load_timeout)
                    receipt["status_at_ready"] = get_json(base + "/v1/status")
                    receipt["model_listing"] = get_json(base + "/v1/models")
                    status = receipt["status_at_ready"]
                    drift = (status.get("sampling_defaults") or {}).get("artifact", {}).get("mismatches") or {}
                    # Muse's pinned model card intentionally overrides the
                    # artifact's do_sample=false; the adapter and its unit test
                    # explicitly record this one known drift.
                    expected_drift = ({"do_sample": {"declared": True, "artifact": False}}
                                      if config.family(model) == "muse" else {})
                    receipt["sampling_drift"] = {"observed": drift, "expected": expected_drift}
                    receipt["default_checks"] = {
                        "served_model": status.get("model") == Path(model.path).name,
                        "healthy": status.get("healthy") is True,
                        "artifact_sampling_matches_documented_policy": drift == expected_drift,
                        "qualification_is_candidate": status.get("qualification") == "candidate",
                    }
                    receipt["smoke"] = smoke(base, Path(model.path).name, config.family(model))
                    receipt["status_after"] = get_json(base + "/v1/status")
                    receipt["health_after"] = get_json(base + "/health")
                    receipt["status"] = "passed" if (receipt["smoke"]["passed"]
                        and all(receipt["default_checks"].values())
                        and receipt["health_after"].get("status") == "ok") else "failed"
                finally:
                    stop_owned(process)
    except Exception as exc:
        receipt["status"] = "error"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        if process is not None:
            receipt["server_log_tail"] = server_log.read_text(errors="replace")[-3000:]
    finally:
        receipt["finished_at"] = time.time()
        save()
    print(f"{model.name} {receipt['status']} {receipt_path}", flush=True)
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
