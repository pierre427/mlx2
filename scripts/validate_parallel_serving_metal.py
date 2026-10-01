#!/usr/bin/env python3
"""Bounded M3 HTTP integration checks for the real XPress candidate.

Run under both GPU locks. Each server is private to this invocation and is
stopped before the next route loads. Receipts are validation evidence, not an
approved qualification record or controlled performance measurement.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = (
    "Write a Python function that finds the first duplicate in a list, then explain its complexity.",
    "A box holds five red and three blue balls. Two are drawn without replacement. Explain the probability both are blue.",
    "Explain in French how a mutex prevents a race when two threads increment a counter.",
    "Write a SQL query using a window function to find each customer's most recent order.",
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--draft", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--timeout-seconds", type=int, default=900)
    p.add_argument(
        "--routes",
        nargs="+",
        choices=("xpress_full", "xpress_windowed", "xpress_pool", "ordinary_reference"),
        default=("xpress_full", "xpress_windowed", "ordinary_reference"),
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--i-own-the-gpu", action="store_true")
    p.add_argument("--target-verify-row-exact", action="store_true")
    return p


def preflight(args):
    if not 1 <= args.timeout_seconds <= 900:
        raise ValueError("timeout must be within 1..900 seconds")
    if not args.dry_run and not args.i_own_the_gpu:
        raise ValueError("GPU execution requires --i-own-the-gpu under both locks")
    return {
        "schema": "mlx2.parallel-draft-http-validation.v1",
        "qualified": False,
        "performance_claim": False,
        "will_execute": not args.dry_run,
        "model": str(args.model.resolve()),
        "draft": str(args.draft.resolve()),
        "routes": list(args.routes),
        "target_verify_row_exact": args.target_verify_row_exact,
        "cells": {},
        "failures": [],
    }


def request(base, endpoint, body=None, timeout=240):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        base + endpoint, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw else {}


def find(value, key):
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            result = find(child, key)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = find(child, key)
            if result is not None:
                return result
    return None


def kinds(value):
    if isinstance(value, dict):
        result = {value["kind"]} if isinstance(value.get("kind"), str) else set()
        for child in value.values():
            result |= kinds(child)
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result |= kinds(child)
        return result
    return set()


def stream(base, body, *, cancel=False):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps({**body, "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    text, receipt, events = [], None, 0
    with urllib.request.urlopen(req, timeout=240) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            events += 1
            receipt = event.get("mlx2", receipt)
            for choice in event.get("choices", []):
                text.append((choice.get("delta") or {}).get("content") or "")
            if cancel and events >= 2:
                break
    return {
        "text": "".join(text),
        "mlx2": receipt,
        "events": events,
        "client_closed_early": cancel,
    }


def run_route(args, report, route):
    work = args.out.parent / route
    work.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "mlx2.server",
        "--model",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--qualification-mode",
        "--max-lanes",
        "4",
        "--max-inflight",
        "8",
        "--max-context",
        "8192",
        "--cache-bytes",
        str(1 << 30),
        "--cache-dir",
        str(work / "cache"),
    ]
    if route != "ordinary_reference":
        policy = {
            "draft_model": str(args.draft),
            "num_draft": 15,
            "xpress_num_passes": 6,
            "target_verify_row_exact": args.target_verify_row_exact,
        }
        if route == "xpress_pool":
            policy["continuation_pool"] = {"limit": 15}
        if route == "xpress_windowed":
            config = json.loads((args.draft / "config.json").read_text())
            depth = config["num_hidden_layers"]
            policy["draft_attention_windows"] = [32] * depth
        policy_file = work / "policy.json"
        policy_file.write_text(json.dumps(policy))
        command += ["--external-draft", "--execution-policy", str(policy_file)]
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    base = f"http://127.0.0.1:{port}"
    cell = report["cells"][route] = {"server_command": command, "steps": {}}
    with (work / "server.log").open("w") as log:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 180
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f"server exited {child.returncode}")
                try:
                    request(base, "/health", timeout=3)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "server startup exceeded 180 seconds"
                        ) from None
                    time.sleep(1)
            models = request(base, "/v1/models")
            identity = models["data"][0]["id"]
            if identity != args.model.name or child.poll() is not None:
                raise RuntimeError("launched server identity mismatch")

            def body(prompt, budget=48, temperature=0):
                return {
                    "model": identity,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": budget,
                    "temperature": temperature,
                    **(
                        {
                            "session_id": "pool-"
                            + hashlib.sha256(prompt.encode()).hexdigest()[:16]
                        }
                        if route == "xpress_pool"
                        else {}
                    ),
                }

            steps = cell["steps"]
            # Inspect French cold before any previous requests can influence
            # cache reuse; top probabilities help diagnose a numerical tie.
            steps["initial_french"] = request(
                base,
                "/v1/chat/completions",
                {**body(PROMPTS[2]), "logprobs": True, "top_logprobs": 5},
            )
            steps["cold"] = request(base, "/v1/chat/completions", body(PROMPTS[0]))
            if route == "ordinary_reference":
                steps["reference"] = [
                    steps["cold"],
                    *[
                        request(base, "/v1/chat/completions", body(p))
                        for p in PROMPTS[1:]
                    ],
                ]
                cell["passed"] = True
                return
            steps["warm"] = request(base, "/v1/chat/completions", body(PROMPTS[0]))
            content = lambda response: response["choices"][0]["message"]["content"]
            if content(steps["cold"]) != content(steps["warm"]):
                raise AssertionError("cold/warm greedy response differs")
            cached = find(steps["warm"], "cached_tokens")
            if not isinstance(cached, int) or cached <= 0:
                raise AssertionError(f"warm APCv2 reused no prefix: {cached}")
            steps["stream"] = stream(base, body(PROMPTS[1]))
            steps["stream_reference"] = request(
                base, "/v1/chat/completions", body(PROMPTS[1])
            )
            if steps["stream"]["text"] != content(steps["stream_reference"]):
                raise AssertionError(
                    "streaming and nonstreaming greedy content differs"
                )
            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                steps["mixed_b4"] = list(
                    pool.map(
                        lambda item: request(base, "/v1/chat/completions", body(*item)),
                        zip(PROMPTS, (16, 32, 48, 64), (0, 0.8, 0, 0.8)),
                    )
                )
            steps["cancelled_stream"] = stream(base, body(PROMPTS[2], 128), cancel=True)
            steps["after_cancel"] = request(
                base, "/v1/chat/completions", body(PROMPTS[2])
            )
            steps["greedy_references"] = [
                steps["cold"],
                steps["stream_reference"],
                steps["after_cancel"],
                request(base, "/v1/chat/completions", body(PROMPTS[3])),
            ]
            responses = [
                steps["cold"],
                steps["warm"],
                steps["stream"],
                steps["after_cancel"],
                *steps["mixed_b4"],
            ]
            for response in responses:
                if "external_xpress" not in kinds(response.get("mlx2")):
                    raise AssertionError(
                        "HTTP response lacks actual external XPress receipt"
                    )
            for _ in range(5):
                cell["status"] = request(base, "/v1/status")
                scheduler = find(cell["status"], "scheduler") or {}
                if scheduler.get("external_rounds", 0) > 0:
                    break
                time.sleep(1)
            cell["scheduler"] = scheduler
            if scheduler.get("external_rounds", 0) <= 0 or scheduler.get(
                "draft_fallbacks", 0
            ):
                raise AssertionError("missing external activity or draft fallback")
            if scheduler.get("paired_cache_resumes", 0) <= 0:
                raise AssertionError(
                    "HTTP warm requests did not exercise paired cache resumes"
                )
            if scheduler.get("target_max_width", 0) < 2:
                raise AssertionError(
                    "concurrent HTTP requests never formed a mixed target batch"
                )
            if route == "xpress_pool":
                pool_receipt = find(steps["warm"], "continuation_pool")
                if not pool_receipt or not pool_receipt.get("observed_used"):
                    raise AssertionError("HTTP pool was not observed used")
                if scheduler.get("target_max_width", 0) != 15:
                    raise AssertionError(
                        "HTTP pool did not execute 15 complete path rows"
                    )
                registry = find(steps["warm"], "ranking_registry")
                if not registry or not all(
                    registry.get(key)
                    for key in ("global_counts", "model_counts", "session_counts")
                ):
                    raise AssertionError(
                        "HTTP pool lacks committed session/model/global ranking labels"
                    )
                cell["pool_ranking"] = registry
                cell["pool_receipt"] = pool_receipt
            cell["passed"] = True
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
            cell["server_returncode"] = child.returncode
            cell["log_tail"] = (work / "server.log").read_text()[-8000:]


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        report = preflight(args)
    except ValueError as error:
        p.error(str(error))
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), *sorted((ROOT / "src").rglob("*.py"))]
    report["files_sha256"] = {
        str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in files
    }

    def timeout(_sig, _frame):
        raise TimeoutError("validation wall-clock deadline reached")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.timeout_seconds)
    try:
        for route in report["routes"]:
            try:
                run_route(args, report, route)
            except Exception as error:  # noqa: BLE001 - preserve every failed cell
                report["failures"].append(
                    {
                        "route": route,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                if isinstance(error, TimeoutError) and "wall-clock" in str(error):
                    break
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "route": route,
                        "passed": report["cells"].get(route, {}).get("passed", False),
                    }
                ),
                flush=True,
            )
        ordinary = (
            report["cells"]
            .get("ordinary_reference", {})
            .get("steps", {})
            .get("reference")
        )
        if ordinary:
            expected = [r["choices"][0]["message"]["content"] for r in ordinary]
            for route in (
                route for route in report["routes"] if route != "ordinary_reference"
            ):
                refs = (
                    report["cells"]
                    .get(route, {})
                    .get("steps", {})
                    .get("greedy_references")
                )
                if (
                    refs
                    and [r["choices"][0]["message"]["content"] for r in refs]
                    != expected
                ):
                    report["failures"].append(
                        {
                            "route": route,
                            "error": "HTTP greedy output differs from ordinary server",
                        }
                    )
    finally:
        signal.alarm(0)
    report["passed"] = not report["failures"] and all(
        c.get("passed") for c in report["cells"].values()
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"passed": report["passed"], "failures": report["failures"]}),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
