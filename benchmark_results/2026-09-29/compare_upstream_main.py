#!/usr/bin/env python3
"""Source-bound thermal single-prompt ladder for an external OpenAI server.

The command JSON is a local argv array. This runner owns that server, both GPU
locks, and only its own child process. Receipts contain hashes, not prompt or
response text. Use the same model artifact and tokenizer when comparing engines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "qualification/runs/requal-20260928"))
sys.path.insert(0, str(ROOT / "qualification/runs/series-20260924"))
import run as owned  # noqa: E402
import thermal_ladder as thermal  # noqa: E402


class Interrupted(Exception):
    pass


def interrupt(signum: int, _frame: object) -> None:
    raise Interrupted(f"signal {signum}")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_json(url: str, timeout: float = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def wait_for_server(base: str, child: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"server exited during load: {child.returncode}")
        try:
            status = read_json(base + "/health")
            if status.get("status") in ("ok", "healthy", None):
                return status
            if status.get("error"):
                raise RuntimeError(f"server health error: {status['error']}")
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(2)
    raise TimeoutError("server did not become healthy")


def token_count(tokenizer, text: str) -> int:
    messages = [{"role": "user", "content": text}]
    try:
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError):
        tokens = tokenizer.encode(text)
    if isinstance(tokens, Mapping):
        tokens = tokens["input_ids"]
    return len(tokens)


def make_prompt(tokenizer, target: int, tag: str) -> tuple[str, str, int]:
    nonce = digest(tag)[:12]
    head = f"Session {nonce} lane 0. Read the archive and answer the question at the end."
    needle = f"NEEDLE-{nonce[:6].upper()}-0"

    def build(units: int) -> str:
        return (f"{head}\n" + " archival evidence" * units
                + f"\nThe unique fact is: the launch code is {needle}.\n"
                + f"State the launch code first. Then list the integers 1 through 200 "
                "in ascending order, separated by commas. Do not summarize or stop early.")

    low, high = 0, target
    while low < high:
        mid = (low + high + 1) // 2
        if token_count(tokenizer, build(mid)) <= target:
            low = mid
        else:
            high = mid - 1
    prompt = build(low)
    return prompt, needle, token_count(tokenizer, prompt)


def stream_request(base: str, model_id: str, text: str, tokenizer, output_tokens: int,
                   *, min_tokens: bool, thinking_flag: bool, timeout: float) -> dict:
    body = {"model": model_id, "messages": [{"role": "user", "content": text}],
            "temperature": 0, "max_tokens": output_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    if min_tokens:
        body["min_tokens"] = output_tokens
    if thinking_flag:
        body["enable_thinking"] = False
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    started, first = time.monotonic(), None
    pieces, usage, done = [], {}, False
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            event = json.loads(payload)
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if piece and first is None:
                    first = time.monotonic()
                pieces.append(piece)
    wall = time.monotonic() - started
    answer = "".join(pieces)
    prompt_tokens = int(usage.get("prompt_tokens") or token_count(tokenizer, text))
    completion_tokens = int(usage.get("completion_tokens") or len(tokenizer.encode(answer)))
    ttft = first - started if first is not None else wall
    decode_window = wall - ttft
    return {
        "done": done, "prompt_sha256": digest(text), "response_sha256": digest(answer),
        "content": answer, "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens, "ttft_seconds": ttft,
        "wall_seconds": wall,
        "prefill_tokens_per_second": prompt_tokens / ttft if ttft > 0 else None,
        "decode_tokens_per_second": ((completion_tokens - 1) / decode_window
                                     if completion_tokens > 1 and decode_window > 0 else None),
        "usage_reported": bool(usage),
    }


def median(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.median(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--command-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--load-timeout", type=float, default=2400)
    parser.add_argument("--request-timeout", type=float, default=3600)
    args = parser.parse_args()
    if args.max_context <= args.max_tokens + 128:
        parser.error("max-context must leave room for the requested output")
    command = json.loads(args.command_json.read_text())
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        parser.error("command JSON must be a nonempty argv string array")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True,
                                               trust_remote_code=True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.source_dir,
                                     text=True).strip()
    config_sha = hashlib.sha256((args.model_dir / "config.json").read_bytes()).hexdigest()
    report = {"schema": "mlx2.external-thermal-ladder.v1", "engine": args.engine,
              "source_head": source, "model_id": args.model_id,
              "artifact_config_sha256": config_sha,
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "command": command, "host": "m5-max-128gb", "started_at": time.time(),
              "thermal_policy": thermal.ADMISSION_POLICY, "cells": [], "status": "running"}

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")

    base = f"http://127.0.0.1:{args.port}"
    child = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        with ExitStack() as stack:
            report["locks"] = owned.lock_host(stack)
            try:
                read_json(base + "/health", timeout=1)
            except (OSError, urllib.error.URLError):
                pass
            else:
                raise RuntimeError(f"port {args.port} already serves /health")
            env = {**os.environ, "PYTHONPATH": str(args.source_dir / "src") + os.pathsep
                   + str(args.source_dir), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            with args.output.with_suffix(".server.log").open("w") as log:
                child = subprocess.Popen(command, cwd=args.source_dir, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                report["server_pid"] = child.pid
                save()
                report["health_at_ready"] = wait_for_server(base, child, args.load_timeout)
                min_tokens, thinking_flag = False, True
                for _ in range(3):
                    try:
                        stream_request(base, args.model_id, "Say OK.", tokenizer, 4,
                                       min_tokens=min_tokens, thinking_flag=thinking_flag,
                                       timeout=args.request_timeout)
                        break
                    except urllib.error.HTTPError as exc:
                        if min_tokens:
                            min_tokens = False
                        elif thinking_flag:
                            thinking_flag = False
                        else:
                            raise exc
                report["request_options"] = {"min_tokens": min_tokens,
                                             "enable_thinking_false": thinking_flag}
                for target in (1024, 4096, 16384, 32768, 65536, 131072, 262144):
                    if target > args.max_length:
                        continue
                    effective = min(target, args.max_context - max(args.max_tokens + 128, 256))
                    cell = {"target_tokens": target, "effective_target_tokens": effective,
                            "runs": [], "error": None}
                    report["cells"].append(cell)
                    save()
                    for rep in range(args.runs):
                        prompt, needle, calibrated = make_prompt(
                            tokenizer, effective, f"{config_sha}-{target}-{rep}")
                        pre = thermal.matrix.stabilize_thermal(thermal.ADMISSION_POLICY)
                        swap_before = thermal.swapouts()
                        measured = stream_request(
                            base, args.model_id, prompt, tokenizer, args.max_tokens,
                            min_tokens=min_tokens, thinking_flag=thinking_flag,
                            timeout=args.request_timeout)
                        swap_after = thermal.swapouts()
                        post = thermal.post_run_thermal()
                        measured["needle_correct"] = needle in measured.pop("content")
                        measured["calibrated_tokens"] = calibrated
                        measured["swapouts_delta"] = swap_after - swap_before
                        measured["thermal_pre"] = pre
                        measured["thermal_post"] = post
                        measured["contaminated"] = bool(swap_after != swap_before or post["breached"])
                        cell["runs"].append(measured)
                        save()
                        print(f"{args.engine} {target} rep {rep}: TTFT {measured['ttft_seconds']:.2f}s "
                              f"decode {measured['decode_tokens_per_second']} swap {measured['swapouts_delta']}",
                              flush=True)
                    cell["passed"] = (len(cell["runs"]) == args.runs and all(
                        row["done"] and row["needle_correct"]
                        and row["completion_tokens"] >= 96
                        and not row["contaminated"]
                        for row in cell["runs"]))
                    cell["median_ttft_seconds"] = median(cell["runs"], "ttft_seconds")
                    cell["median_prefill_tokens_per_second"] = median(
                        cell["runs"], "prefill_tokens_per_second")
                    cell["median_decode_tokens_per_second"] = median(
                        cell["runs"], "decode_tokens_per_second")
                    save()
                report["status"] = "passed" if all(c["passed"] for c in report["cells"]) else "failed"
    except Interrupted as exc:
        report["status"] = "interrupted"
        report["error"] = str(exc)
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if child is not None:
            owned.stop_owned(child)
        report["finished_at"] = time.time()
        save()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
