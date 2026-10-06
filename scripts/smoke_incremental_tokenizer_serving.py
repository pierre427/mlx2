#!/usr/bin/env python3
"""Bounded serving smoke for revision-bound incremental tokenization.

The smoke launches exactly one local Flash-Next service, sends a cold chat and
one growing-conversation chat, then requires ``ordinary_full_validation`` and
``incremental_hit`` receipts for the same current tokenizer revision.  It is
an unqualified smoke, not qualification or a performance benchmark.

Execution loads a real model and therefore refuses without
``--i-own-the-gpu``.  ``--dry-run`` is CPU-only and prints the complete plan
without importing MLX or launching a subprocess.  The caller remains
responsible for acquiring and retaining both project GPU locks.
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
import urllib.error
import urllib.request
from pathlib import Path

SCHEMA = "mlx2.incremental-tokenizer-serving-smoke.v1"
ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8397)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--shutdown-timeout", type=float, default=120)
    parser.add_argument("--base-characters", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def server_environment() -> dict[str, str]:
    source = str(ROOT / "src")
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        [source]
        + [item for item in existing.split(os.pathsep) if item and item != source]
    )
    return environment


def server_command(args) -> list[str]:
    return [
        args.python,
        "-u",
        "-m",
        "mlx2.server",
        "--model",
        str(args.model),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--ordinary",
        "--max-lanes",
        "1",
        "--max-inflight",
        "2",
        "--incremental-tokenizer-cache-entries",
        "4",
        "--incremental-tokenizer-cache-characters",
        str(1 << 20),
        "--incremental-tokenizer-cache-tokens",
        str(1 << 18),
        "--host-prompt-cache-entries",
        "8",
        "--host-prompt-cache-tokens",
        str(1 << 18),
    ]


def _request(base: str, path: str, body=None, *, timeout: float):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def _find_mapping(value, key):
    if isinstance(value, dict):
        found = value.get(key)
        if isinstance(found, dict):
            return found
        for child in value.values():
            result = _find_mapping(child, key)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _find_mapping(child, key)
            if result is not None:
                return result
    return None


def evaluate(cold_payload: dict, grown_payload: dict, status: dict) -> list[str]:
    failures = []
    current = status.get("incremental_tokenizer_cache") or {}
    cold = _find_mapping(cold_payload.get("mlx2"), "prompt_tokenization")
    grown = _find_mapping(grown_payload.get("mlx2"), "prompt_tokenization")
    expected = current.get("tokenizer_revision")
    for name, receipt, action in (
        ("cold", cold, "ordinary_full_validation"),
        ("grown", grown, "incremental_hit"),
    ):
        if receipt is None:
            failures.append(f"{name}: missing prompt_tokenization receipt")
            continue
        if receipt.get("action") != action:
            failures.append(
                f"{name}: action={receipt.get('action')!r}, expected {action!r}"
            )
        if receipt.get("exact") is not True:
            failures.append(f"{name}: exact is not true")
        if receipt.get("tokenizer_revision") != expected:
            failures.append(f"{name}: tokenizer revision is not current status")
        if receipt.get("selected") is not True:
            failures.append(f"{name}: route receipt is not selected")
        if receipt.get("qualified") is not False:
            failures.append(f"{name}: unqualified route mislabeled")
    if current.get("selected") is not True:
        failures.append("status: incremental tokenizer is not selected")
    if current.get("qualified") is not False:
        failures.append("status: unqualified route mislabeled")
    if current.get("serving_qualified") is not False:
        failures.append("status: route incorrectly serving-qualified")
    if current.get("observed_used") is not True:
        failures.append("status: observed_used is not true")
    if not isinstance(expected, str) or len(expected) != 64:
        failures.append("status: current tokenizer revision is not a SHA-256 identity")
    if int(current.get("cold_validations", 0)) < 1:
        failures.append("status: cold_validations < 1")
    if int(current.get("incremental_hits", 0)) < 1:
        failures.append("status: incremental_hits < 1")
    if current.get("refusal") is not None:
        failures.append(f"status: refusal={current.get('refusal')!r}")
    return failures


def _stop(process: subprocess.Popen, *, timeout: float) -> dict:
    was_running = process.poll() is None
    result = {
        "was_running": was_running,
        "terminated": False,
        "killed": False,
        "returncode": None,
    }
    if was_running:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=timeout)
            result["terminated"] = True
        except ProcessLookupError:
            process.wait(timeout=min(timeout, 30))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
            result["killed"] = True
    result["returncode"] = process.returncode
    return result


def _document(characters: int) -> str:
    unit = "English and CJK 你好; def f(x): return x + 1; punctuation []{}.\n"
    return (unit * (characters // len(unit) + 1))[:characters]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in 1..65535")
    if (
        min(
            args.startup_timeout,
            args.request_timeout,
            args.shutdown_timeout,
            args.base_characters,
            args.max_tokens,
        )
        <= 0
    ):
        raise SystemExit(
            "timeouts, --base-characters and --max-tokens must be positive"
        )

    command = server_command(args)
    plan = {
        "schema": SCHEMA,
        "status": "unqualified serving smoke; not executed by construction",
        "model": str(args.model),
        "server_command": command,
        "server_pythonpath": server_environment()["PYTHONPATH"].split(os.pathsep)[0],
        "requests": [
            {"name": "cold", "expected_action": "ordinary_full_validation"},
            {"name": "grown", "expected_action": "incremental_hit"},
        ],
        "will_execute": bool(args.i_own_the_gpu and not args.dry_run),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.i_own_the_gpu:
        print(
            "refusing: real serving requires --i-own-the-gpu under the paired lock wrapper",
            file=sys.stderr,
        )
        return 2
    if not (args.model / "model.safetensors.index.json").is_file():
        print("refusing: --model is not a local indexed artifact", file=sys.stderr)
        return 2

    with socket.socket() as probe:
        probe.settimeout(0.2)
        if probe.connect_ex((args.host, args.port)) == 0:
            print(
                f"refusing: {args.host}:{args.port} is already in use", file=sys.stderr
            )
            return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(args.out.suffix + ".server.log")
    log = log_path.open("w")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=server_environment(),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    base = f"http://{args.host}:{args.port}"
    result = {
        **plan,
        "will_execute": True,
        "started_at_unix": time.time(),
        "pid": process.pid,
        "log": str(log_path),
        "failures": [],
    }
    try:
        deadline = time.monotonic() + args.startup_timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(
                    f"server exited during startup: {process.returncode}"
                )
            try:
                health, _payload = _request(
                    base, "/health", timeout=min(args.request_timeout, 5)
                )
                if health == 200:
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("server did not become healthy before timeout")
            time.sleep(2)

        _code, models = _request(base, "/v1/models", timeout=args.request_timeout)
        model_id = models["data"][0]["id"]
        if model_id != args.model.name:
            raise RuntimeError(
                f"launched model identity {model_id!r} != {args.model.name!r}"
            )
        common = {
            "model": model_id,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "enable_thinking": False,
        }
        document = _document(args.base_characters)
        cold_messages = [
            {"role": "system", "content": "Answer briefly and exactly."},
            {"role": "user", "content": document},
        ]
        grown_messages = [
            *cold_messages,
            {"role": "assistant", "content": "Acknowledged."},
            {"role": "user", "content": "Continue with one word."},
        ]
        code, cold = _request(
            base,
            "/v1/chat/completions",
            {**common, "messages": cold_messages},
            timeout=args.request_timeout,
        )
        if code != 200:
            result["failures"].append(f"cold: HTTP {code}")
        code, grown = _request(
            base,
            "/v1/chat/completions",
            {**common, "messages": grown_messages},
            timeout=args.request_timeout,
        )
        if code != 200:
            result["failures"].append(f"grown: HTTP {code}")
        _code, status = _request(base, "/v1/status", timeout=args.request_timeout)
        result["failures"].extend(evaluate(cold, grown, status))
        cache_status = status["incremental_tokenizer_cache"]
        result["model_id"] = model_id
        result["artifact_config_sha256"] = hashlib.sha256(
            (args.model / "config.json").read_bytes()
        ).hexdigest()
        result["tokenizer_revision"] = cache_status["tokenizer_revision"]
        result["prompt_characters"] = {
            "cold_content": len(document),
            "grown_added": len("Acknowledged.") + len("Continue with one word."),
        }
        result["receipts"] = {
            "cold": _find_mapping(cold.get("mlx2"), "prompt_tokenization"),
            "grown": _find_mapping(grown.get("mlx2"), "prompt_tokenization"),
        }
        result["status"] = cache_status
    except Exception as error:  # noqa: BLE001 - smoke records and cleans up
        result["failures"].append(f"{type(error).__name__}: {error}")
    finally:
        result["shutdown"] = _stop(process, timeout=args.shutdown_timeout)
        log.close()
        result["server_log_tail"] = log_path.read_text(errors="replace")[-4000:]

    result["go"] = (
        not result["failures"]
        and result["shutdown"]["was_running"]
        and result["shutdown"]["terminated"]
        and not result["shutdown"]["killed"]
    )
    result["finished_at_unix"] = time.time()
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "go": result["go"],
                "failures": result["failures"],
                "tokenizer_revision": result.get("tokenizer_revision"),
                "out": str(args.out),
                "shutdown": result["shutdown"],
            },
            indent=2,
        )
    )
    return 0 if result["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
