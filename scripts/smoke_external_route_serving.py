#!/usr/bin/env python3
# ruff: noqa: EXE001
"""Serving-level smoke of the external draft route on a real artifact (rm06).

Starts ``mlx2.server`` in candidate (``--qualification-mode``) with
``--external-draft`` and a ``draft_model`` execution policy, then drives it
over HTTP: non-streaming chat, streaming chat, a repeated prompt (warm APCv2
reuse), four concurrent requests (B4 cohort), a thinking-on request, and a
non-zero temperature request.  It refuses (non-zero exit) unless the server
reports ``external_rounds > 0``, ``proposed_tokens > 0``, zero draft
fallbacks, and every response receipt carries the expected external kind.

This is a smoke, not a qualification record: it does not run
``scripts/qualify_serving.py`` and makes no ``qualified`` claim.

Metal-only; refuses without ``--i-own-the-gpu``.  ``--dry-run`` prints the
plan.  Run under the GPU lock wrapper.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

SCHEMA = "mlx2.rm06-external-route-serving-smoke.v1"
ROOT = Path(__file__).resolve().parents[1]


def server_env() -> dict:
    """Pin the server subprocess to THIS checkout's ``src``.

    ``python -m mlx2.server`` with an inherited environment imports mlx2 from
    whichever checkout happens to be importable, so a run queued from a lane
    worktree could silently smoke another branch's engine.  The tree holding
    this script wins, ahead of anything already on PYTHONPATH.
    """
    source = str(ROOT / "src")
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    parts = [source] + [
        part for part in existing.split(os.pathsep) if part and part != source
    ]
    environment["PYTHONPATH"] = os.pathsep.join(parts)
    return environment
KINDS = {"north": "external_cohere_eagle", "laguna": "external_laguna_dflash"}
PROMPTS = [
    "Write a Python function that checks whether a string is a palindrome.",
    "Explain what a mutex is in two sentences.",
    "Write a SQL query that counts orders per customer.",
    "Implement FizzBuzz in JavaScript.",
]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", choices=sorted(KINDS), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--num-draft", type=int, default=3)
    p.add_argument("--port", type=int, default=8297)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--startup-timeout", type=float, default=900)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--out", type=Path)
    p.add_argument("--i-own-the-gpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def _request(url, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


MODEL = {"id": "m"}


def _chat(base, text, max_tokens, **extra):
    body = {"model": MODEL["id"], "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "temperature": 0, **extra}
    status, raw = _request(base + "/v1/chat/completions", body)
    return status, json.loads(raw)


def _stream(base, text, max_tokens):
    body = {"model": MODEL["id"], "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True}
    status, raw = _request(base + "/v1/chat/completions", body)
    chunks, receipt, content = 0, None, []
    for line in raw.decode().splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        event = json.loads(line[6:])
        chunks += 1
        receipt = event.get("mlx2", receipt)
        for choice in event.get("choices", []):
            content.append((choice.get("delta") or {}).get("content") or "")
    return status, {"chunks": chunks, "mlx2": receipt, "text": "".join(content)}


def _kinds(obj):
    found = set()
    if isinstance(obj, dict):
        if isinstance(obj.get("kind"), str):
            found.add(obj["kind"])
        for value in obj.values():
            found |= _kinds(value)
    elif isinstance(obj, list):
        for value in obj:
            found |= _kinds(value)
    return found


def _find(obj, key):
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]
        for value in obj.values():
            hit = _find(value, key)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for value in obj:
            hit = _find(value, key)
            if hit is not None:
                return hit
    return None


def main(argv=None):
    args = build_parser().parse_args(argv)
    policy = {"draft_model": args.draft, "num_draft": args.num_draft}
    workdir = Path(tempfile.mkdtemp(prefix="rm06-serving-smoke-"))
    command = [args.python, "-m", "mlx2.server", "--model", args.model, "--port", str(args.port),
               "--external-draft", "--execution-policy", str(workdir / "policy.json"),
               "--qualification-mode", "--max-lanes", "4", "--max-inflight", "8",
               "--max-context", "32768", "--cache-bytes", str(4 << 30), "--cache-dir", str(workdir / "cache")]
    plan = {"schema": SCHEMA, "family": args.family, "expected_kind": KINDS[args.family], "policy": policy,
            "server_command": command,
            "server_pythonpath": server_env()["PYTHONPATH"].split(os.pathsep)[0],
            "will_execute": bool(args.i_own_the_gpu and not args.dry_run)}
    if args.dry_run or not args.i_own_the_gpu:
        print(json.dumps(plan, indent=1))
        if not args.dry_run:
            print("refusing: Metal execution requires --i-own-the-gpu", file=sys.stderr)
            return 2
        return 0
    (workdir / "policy.json").write_text(json.dumps(policy))
    base = f"http://127.0.0.1:{args.port}"
    log = open(workdir / "server.log", "w")  # noqa: SIM115 - outlives the server process
    server = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                              env=server_env(), start_new_session=True)
    print(f"server pid {server.pid} log {workdir / 'server.log'}", flush=True)
    result = {**plan, "steps": {}, "failures": []}
    try:
        deadline = time.time() + args.startup_timeout
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"server exited rc={server.returncode}")
            try:
                if _request(base + "/health", timeout=5)[0] == 200:
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            if time.time() > deadline:
                raise RuntimeError("server did not become healthy")
            time.sleep(3)
        result["startup_s"] = args.startup_timeout - (deadline - time.time())
        result["models"] = json.loads(_request(base + "/v1/models")[1])
        MODEL["id"] = result["models"]["data"][0]["id"]
        steps = result["steps"]
        t0 = time.perf_counter()
        steps["nonstream"] = _chat(base, PROMPTS[0], args.max_tokens)
        steps["nonstream_s"] = time.perf_counter() - t0
        steps["repeat"] = _chat(base, PROMPTS[0], args.max_tokens)
        steps["stream"] = _stream(base, PROMPTS[1], args.max_tokens)
        steps["thinking"] = _chat(base, PROMPTS[2], args.max_tokens, reasoning_effort="high")
        steps["sampled"] = _chat(base, PROMPTS[3], args.max_tokens, temperature=0.7)
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            steps["b4"] = list(pool.map(lambda text: _chat(base, text, args.max_tokens), PROMPTS))
        steps["b4_s"] = time.perf_counter() - t0
        # The engine refreshes its status snapshot about once a second.
        time.sleep(3)
        status = json.loads(_request(base + "/v1/status")[1])
        result["status"] = status
        scheduler = _find(status, "scheduler") or {}
        result["scheduler"] = scheduler
        expected = KINDS[args.family]
        responses = [steps["nonstream"], steps["repeat"], steps["stream"], steps["thinking"], steps["sampled"],
                     *steps["b4"]]
        for i, (code, payload) in enumerate(responses):
            if code != 200:
                result["failures"].append(f"response {i}: HTTP {code}")
            if expected not in _kinds(payload.get("mlx2")):
                result["failures"].append(f"response {i}: receipt kind {sorted(_kinds(payload.get('mlx2')))}")
        if steps["nonstream"][1]["choices"][0]["message"]["content"] != steps["repeat"][1]["choices"][0]["message"]["content"]:
            result["failures"].append("greedy repeat differs from first greedy response")
        if int(scheduler.get("external_rounds", 0)) <= 0:
            result["failures"].append("external_rounds == 0")
        if int(scheduler.get("proposed_tokens", 0)) <= 0:
            result["failures"].append("proposed_tokens == 0")
        if int(scheduler.get("draft_fallbacks", 0)) != 0:
            result["failures"].append(f"draft_fallbacks == {scheduler.get('draft_fallbacks')}")
    except urllib.error.HTTPError as exc:
        result["failures"].append(f"HTTPError {exc.code} {exc.url}: {exc.read()[:2000]!r}")
    except Exception as exc:  # noqa: BLE001 - smoke records every failure
        result["failures"].append(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=60)
        except Exception:  # noqa: BLE001
            os.killpg(server.pid, signal.SIGKILL)
        log.close()
        result["server_log_tail"] = (workdir / "server.log").read_text()[-4000:]
    result["go"] = not result["failures"]
    text = json.dumps(result, indent=1, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(json.dumps({"go": result["go"], "failures": result["failures"],
                      "scheduler": {k: result.get("scheduler", {}).get(k) for k in
                                    ("external_rounds", "proposed_tokens", "accepted_tokens",
                                     "draft_fallbacks", "paired_cache_resumes")}}, indent=1))
    return 0 if result["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
