#!/usr/bin/env python3
"""Localize the one-token GSM8K completions seen on the 27B AST route (F5 run1).

In qualification/runs/f5-time-credit-20261105/run1 the default AST arm ended
53 of 100 GSM8K completion requests after exactly one token, with no verify
round; the other 47 scored 45/47. This probe serves the same checkpoint with
the given server args and sends, on a fresh server:

  1. a failing question first (cold prefix),
  2. a passing question, then the failing question again (warm prefix),
  3. the failing question with request ``"speculation": "off"``,
  4. the failing question with ``skip_writing_prefix_cache`` and
     speculation on (cache write disabled for that request),

recording text, finish reason, token ids, the first position's top-5
logprobs, cached tokens and the route/speculation receipt. Diagnosis only.

Usage: ``repro_gsm8k_one_token.py owned OUT --model PATH -- SERVER_ARGS...``.
Original mlx2 code.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
PORT = 8393
sys.path.insert(0, str(ROOT / "scripts"))

import spec_acceptance_gsm8k as gsm

FAILING = (1, 2, 3)
PASSING = 0


def checkout_env() -> dict:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(ROOT / "src"), environment.get("PYTHONPATH", "")) if p)
    return environment


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def call(path, payload=None, timeout=900):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(f"http://127.0.0.1:{PORT}{path}", data=data,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def ask(model_name, prefix, item, label, **extra):
    payload = {"model": model_name, "temperature": 0, "stream": False, "max_tokens": 256,
               "stop": gsm.STOPS, "prompt": prefix + f"Question: {item['question']}\nAnswer:",
               "return_token_trace": True, "logprobs": 5, **extra}
    try:
        result = call("/v1/completions", payload)
    except Exception as exc:  # noqa: BLE001 - diagnosis records every outcome
        return {"label": label, "error": repr(exc)}
    choice = result["choices"][0]
    mlx2 = result.get("mlx2") or {}
    logprobs = choice.get("logprobs") or {}
    return {"label": label, "text": choice.get("text"), "finish_reason": choice.get("finish_reason"),
            "predicted": gsm.extract_answer(choice.get("text") or ""),
            "expected": gsm.reference_answer(item["answer"]),
            "token_ids": (mlx2.get("token_trace") or {}).get("token_ids"),
            "first_top_logprobs": (logprobs.get("top_logprobs") or [None])[0],
            "first_tokens": (logprobs.get("tokens") or [])[:4],
            "usage": result.get("usage"),
            "mlx2": {k: v for k, v in mlx2.items() if k != "token_trace"}}


def unit(out: Path, model: Path, server_args: list[str]) -> int:
    _, files = gsm.dataset_files()
    train, test = gsm.load_split(files["train"]), gsm.load_split(files["test"])
    prefix = gsm.few_shot_prefix(train, 5)
    command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(model), "--host", "127.0.0.1",
               "--port", str(PORT), "--max-lanes", "1", "--max-inflight", "1", "--max-context", "8192",
               "--cache-bytes", str(2 << 30), "--cache-dir", str(out / "cache"), *server_args]
    receipt = {"schema": "mlx2.repro-gsm8k-one-token.v1", "command": command, "rows": [],
               "source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    log = (out / "server.log").open("w")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                              env=checkout_env(), cwd=ROOT)
    try:
        deadline = time.monotonic() + 900
        while True:
            if server.poll() is not None:
                raise RuntimeError("server exited before readiness")
            try:
                if call("/v1/status", timeout=10).get("healthy"):
                    break
            except (URLError, TimeoutError, ConnectionError, OSError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("readiness timeout")
            time.sleep(0.5)
        name = model.name
        rows = receipt["rows"]
        first = FAILING[0]
        rows.append(ask(name, prefix, test[first], f"q{first}_cold_first"))
        rows.append(ask(name, prefix, test[PASSING], f"q{PASSING}_warm"))
        for index in FAILING:
            rows.append(ask(name, prefix, test[index], f"q{index}_warm"))
            rows.append(ask(name, prefix, test[index], f"q{index}_spec_off", speculation="off"))
        rows.append(ask(name, prefix, test[first], f"q{first}_no_cache_write",
                        skip_writing_prefix_cache=True))
        atomic_json(out / "receipt.json", receipt)
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=30)
        log.close()
        atomic_json(out / "receipt.json", receipt)
    for row in receipt["rows"]:
        print(json.dumps({k: row.get(k) for k in ("label", "finish_reason", "predicted", "expected",
                                                  "first_tokens", "error")}), flush=True)
    return 0


def owned(out: Path, model: Path, server_args: list[str], job_id: str) -> int:
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with DesktopCPGLease(out, job_id, checkpoint=model.name, source_revision=source) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out), "--model", str(model),
                   "--", *server_args]
        child = subprocess.Popen(command, cwd=ROOT, env=checkout_env(), start_new_session=True)
        try:
            while child.poll() is None:
                lease.check()
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=120)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--job-id", default="repro-gsm8k-one-token-20261105")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    server_args = argv[split + 1:]
    out = args.out.resolve()
    if args.mode == "unit":
        return unit(out, args.model, server_args)
    return owned(out, args.model, server_args, args.job_id)


if __name__ == "__main__":
    raise SystemExit(main())
