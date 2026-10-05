#!/usr/bin/env python3
"""Bounded cold/restored B=1 APCv2 HTTP smoke; candidate evidence only.

The caller owns the GPU lease and host lock. This script launches only its own
loopback server, uses a fresh temporary APC directory, and stops that server.
SSE arrival intervals are client-observed latency, not device decode time or
host synchronization counts.
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
import tempfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPT = (
    "Read this short reference and answer the final question in one sentence.\n"
    + ("APCv2 retains exact prompt state. An ordinary B=1 request has one decode lane.\n" * 48)
    + "Question: What does APCv2 retain?\nAnswer:"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def get_json(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=10) as response:
        return json.load(response)


def stream_one(base: str, body: dict, *, timeout: int) -> dict:
    request = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.perf_counter()
    token_ids: list[int] = []
    token_arrival_s: list[float] = []
    text_parts: list[str] = []
    receipt = None
    finish_reason = None
    done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError(f"generation returned HTTP {response.status}")
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"generation SSE error: {event['error']}")
            if event.get("mlx2") is not None:
                receipt = event["mlx2"]
            for choice in event.get("choices", ()):
                for item in (choice.get("logprobs") or {}).get("content", ()):
                    token_ids.append(int(item["id"]))
                    token_arrival_s.append(time.perf_counter() - started)
                text_parts.append((choice.get("delta") or {}).get("content") or "")
                finish_reason = choice.get("finish_reason") or finish_reason
    if not done or receipt is None or finish_reason is None:
        raise ValueError("incomplete generation stream or absent terminal receipt")
    return {
        "text": "".join(text_parts), "token_ids": token_ids,
        "token_arrival_s": token_arrival_s,
        "first_token_s": token_arrival_s[0] if token_arrival_s else None,
        "inter_token_s": [round(b - a, 6) for a, b in zip(token_arrival_s, token_arrival_s[1:])],
        "elapsed_s": time.perf_counter() - started,
        "finish_reason": finish_reason, "receipt": receipt,
    }


def validate(cold: dict, warm: dict) -> dict:
    failures = []
    for name, row, expected_cached in (("cold", cold, False), ("warm", warm, True)):
        receipt = row.get("receipt") or {}
        cached = receipt.get("cached_tokens")
        if receipt.get("cache") != "apcv2":
            failures.append(f"{name}: cache is not apcv2")
        if not isinstance(cached, int) or (cached <= 0 if expected_cached else cached != 0):
            failures.append(f"{name}: unexpected cached_tokens={cached!r}")
        if receipt.get("route") != "ordinary" or receipt.get("ordinary_compute_width") != 1:
            failures.append(f"{name}: route/observed width is not ordinary B=1")
        if not receipt.get("route_receipt"):
            failures.append(f"{name}: missing route receipt")
        if not row.get("token_ids") or not row.get("text"):
            failures.append(f"{name}: no visible text or token IDs")
        if len(row.get("token_arrival_s") or ()) != len(row.get("token_ids") or ()):
            failures.append(f"{name}: token arrival count differs from token IDs")
        if row.get("first_token_s") is None:
            failures.append(f"{name}: missing first-token arrival")
    if cold.get("token_ids") != warm.get("token_ids"):
        failures.append("cold/warm token IDs differ")
    if cold.get("text") != warm.get("text"):
        failures.append("cold/warm text differs")
    if (cold.get("receipt") or {}).get("prompt_tokens") != (warm.get("receipt") or {}).get("prompt_tokens"):
        failures.append("cold/warm prompt lengths differ")
    return {"passed": not failures, "failures": failures}


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--backend-source", help="clean pinned local dependency checkout for a source-bound spot")
    parser.add_argument("--port", type=int, default=8393)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--prompt-file", type=Path,
                        help="UTF-8 user prompt; defaults to the standard B1 spot prompt")
    parser.add_argument("--wall-seconds", type=int, default=0,
                        help="optional hard wall bound; preserves server cleanup on expiry")
    parser.add_argument("--cpg-lease", required=True)
    parser.add_argument("--i-own-gpu", action="store_true")
    args = parser.parse_args(argv)
    if args.wall_seconds < 0:
        parser.error("--wall-seconds must be non-negative")
    if not args.i_own_gpu:
        parser.error("GPU smoke requires --i-own-gpu under the CPG lease and host lock")
    if not 2 <= args.max_tokens <= 128:
        parser.error("--max-tokens must be 2..128")
    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else PROMPT
    if not prompt.strip() or len(prompt) > 16_384:
        parser.error("prompt must contain text and be at most 16384 characters")
    model = Path(args.model).expanduser().resolve(strict=True)
    backend_source = None
    if args.backend_source:
        backend_source = Path(args.backend_source).expanduser().resolve(strict=True)
        backend_head = subprocess.check_output(
            ["git", "-C", str(backend_source), "rev-parse", "HEAD"], text=True).strip()
        backend_dirty = subprocess.check_output(
            ["git", "-C", str(backend_source), "status", "--porcelain"], text=True).strip()
        if backend_dirty:
            parser.error("--backend-source checkout must be clean")
    out_path = Path(args.out).expanduser().resolve()
    if out_path.exists():
        parser.error(f"refusing to overwrite existing receipt: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.settimeout(0.2)
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            parser.error(f"port {args.port} is already occupied")
    receipt = {
        "schema": "mlx2.b1-apcv2-http-smoke.v1", "status": "failed",
        "kind": "bounded candidate smoke; not serving qualification",
        "started_at": datetime.now(UTC).isoformat(),
        "source": {"git_revision": git("rev-parse", "HEAD"),
                   "tracked_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                   "harness_sha256": sha256(Path(__file__))},
        "model_path": str(model), "cpg_lease": args.cpg_lease,
        "backend_source": ({"path": str(backend_source), "git_revision": backend_head,
                            "clean": True} if backend_source else None),
        "request": {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "prompt_chars": len(prompt), "max_tokens": args.max_tokens,
                    "temperature": 0, "logprobs": True},
        "measurement": "client-observed HTTP SSE token-logprob arrival times; no host-sync instrumentation",
    }
    base = f"http://127.0.0.1:{args.port}"
    log_path = out_path.with_suffix(".server.log")
    server = None
    previous_alarm = None
    if args.wall_seconds:
        def _wall_expired(_signum, _frame):
            raise TimeoutError(f"bounded smoke exceeded {args.wall_seconds} seconds")
        previous_alarm = signal.signal(signal.SIGALRM, _wall_expired)
        signal.alarm(args.wall_seconds)
    try:
        with tempfile.TemporaryDirectory(prefix="b1-apcv2-cache-", dir=out_path.parent) as cache_dir:
            command = [args.python, "-m", "mlx2.server", "--model", str(model),
                       "--host", "127.0.0.1", "--port", str(args.port),
                       "--ordinary", "--qualification-mode", "--max-lanes", "1",
                       "--max-inflight", "1", "--cache-dir", cache_dir]
            receipt["server_command"] = command
            pythonpath = str(ROOT / "src")
            if backend_source:
                pythonpath += os.pathsep + str(backend_source)
            env = {**os.environ, "PYTHONPATH": pythonpath,
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "MLX_ENABLE_TF32": "0"}
            with log_path.open("w") as log:
                server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                began = time.monotonic()
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"server exited {server.returncode}; see {log_path}")
                    try:
                        if get_json(base, "/health").get("status") == "ok":
                            break
                    except (urllib.error.URLError, ConnectionError, OSError):
                        pass
                    if time.monotonic() - began > 180:
                        raise TimeoutError(f"server not healthy after 180 s; see {log_path}")
                    time.sleep(0.5)
                status = get_json(base, "/v1/status")
                receipt["server_identity"] = {key: status.get(key) for key in (
                    "model", "artifact", "adapter", "runtime", "settings",
                    "qualification", "route", "profile", "capabilities")}
                identity = receipt["server_identity"]
                if (identity["model"] != model.name or not identity["artifact"]
                        or not (identity["runtime"] or {}).get("source_sha256")
                        or (identity["settings"] or {}).get("route") != "ordinary"
                        or (identity["settings"] or {}).get("max_lanes") != 1):
                    raise ValueError("server model/source/settings identity differs from requested ordinary B=1 run")
                body = {"messages": [{"role": "user", "content": prompt}],
                        "max_tokens": args.max_tokens, "temperature": 0,
                        "stream": True, "logprobs": True}
                receipt["cold"] = stream_one(base, body, timeout=90)
                receipt["warm"] = stream_one(base, body, timeout=90)
                receipt["checks"] = validate(receipt["cold"], receipt["warm"])
                end_status = get_json(base, "/v1/status")
                receipt["server_identity_stable"] = all(
                    end_status.get(key) == identity.get(key)
                    for key in ("model", "artifact", "runtime", "settings", "route", "profile")
                )
                if not receipt["server_identity_stable"]:
                    receipt["checks"]["failures"].append("server identity changed between start and end")
                    receipt["checks"]["passed"] = False
                receipt["status"] = "smoke_passed" if receipt["checks"]["passed"] else "smoke_failed"
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        if args.wall_seconds:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_alarm)
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(3 if args.wall_seconds else 15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        receipt["finished_at"] = datetime.now(UTC).isoformat()
        temporary = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
        temporary.replace(out_path)
    print(json.dumps({"status": receipt["status"], "checks": receipt.get("checks"),
                      "error": receipt.get("error"), "out": str(out_path)}))
    return 0 if receipt["status"] == "smoke_passed" else 1


if __name__ == "__main__":
    raise SystemExit(run())
