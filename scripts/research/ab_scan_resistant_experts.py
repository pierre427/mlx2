#!/usr/bin/env python3
"""F3 A/B: scan-resistant expert caching (``prefill_admission: scan_resistant``).

Feature 3 (d5593d878) keeps decode-hot experts in a protected segment so a
long prefill scan cannot evict them. With MoE expert streaming on a bounded
cache, each cycle runs

  D1  short coding prompt, 192 greedy tokens   (builds a decode hot set)
  S   fresh ~8K-token document, 1 token         (prefill scan)
  D2  second short coding prompt, 192 tokens    (measured: misses, bytes, tok/s)

three times, per arm, one server per arm on the ordinary route:

  resident        no streaming (token reference)
  lru             streaming, prefill_admission lru (default)
  terminal_hot    streaming, the existing alternative policy
  scan_resistant  streaming, feature 3

Streaming changes residency, not arithmetic, so every arm's tokens must equal
the resident arm's. Counter deltas come from /v1/status ``counts``
(``stream_*_total``). Sanity/attribution numbers, no thermal hold.

Usage: ``ab_scan_resistant_experts.py owned OUT --model PATH [--cache-gib 4]``.
Original mlx2 code.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
PORT = 8395
SHAPE = ["--ordinary", "--max-lanes", "1", "--max-inflight", "1", "--max-context", "16384",
         "--cache-bytes", str(2 << 30)]
COUNTERS = ("stream_page_ins_total", "stream_page_in_bytes_total", "stream_expert_hits_total",
            "stream_expert_misses_total", "stream_evictions_total", "stream_scan_calls_total",
            "stream_scan_protected_hits_total")
DECODE_PROMPTS = [
    ("Write a Python function that merges two sorted lists into one sorted list, "
     "with a short docstring and two doctest examples."),
    ("Write a Python class for a fixed-size ring buffer with push, pop and len, "
     "raising IndexError when empty."),
    ("Write a Python function that parses an ISO 8601 date string into a tuple of "
     "year, month and day without using the datetime module."),
    ("Write a Python generator that yields the prime numbers below a limit using "
     "the sieve of Eratosthenes, with comments."),
    ("Write a Python function that flattens an arbitrarily nested list of integers, "
     "iteratively rather than recursively."),
    ("Write a Python function that returns the longest common prefix of a list of "
     "strings, handling the empty list."),
]
_W = ["harbour", "orchard", "lantern", "granary", "foundry", "ferry", "quarry", "vineyard", "tannery",
      "mill", "chapel", "market", "bridge", "canal", "forge", "bakery", "weaver", "cooper", "mason"]
_V = ["inspected", "repaired", "catalogued", "measured", "restocked", "painted", "cleared", "rebuilt"]


def scan_document(seed: int, lines: int = 260) -> str:
    rng = random.Random(seed)
    return "\n".join(
        f"Record {seed}-{i}: the {rng.choice(_W)} crew {rng.choice(_V)} {rng.randint(2, 990)} "
        f"items near the {rng.choice(_W)} on day {rng.randint(1, 365)}." for i in range(lines))


def checkout_env() -> dict:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(ROOT / "src"), environment.get("PYTHONPATH", "")) if p)
    return environment


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def source() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {"commit": git("rev-parse", "HEAD"),
            "dirty_tracked": bool(git("status", "--porcelain", "--untracked-files=no"))}


def host_sample() -> dict:
    return {"thermal": subprocess.check_output(["pmset", "-g", "therm"], text=True),
            "swap": subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()}


def _find(tree, key):
    if isinstance(tree, dict):
        if key in tree:
            return tree[key]
        for value in tree.values():
            found = _find(value, key)
            if found is not None:
                return found
    elif isinstance(tree, list):
        for value in tree:
            found = _find(value, key)
            if found is not None:
                return found
    return None


def call(path, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(f"http://127.0.0.1:{PORT}{path}", data=data,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def counters() -> dict:
    status = call("/v1/status", timeout=30)
    return {key: _find(status, key) or 0 for key in COUNTERS}


def chat(model_name: str, content: str, max_tokens: int) -> dict:
    before = counters()
    payload = {"model": model_name, "messages": [{"role": "user", "content": content}],
               "temperature": 0, "max_tokens": max_tokens, "enable_thinking": False,
               "return_token_trace": True, "stream": False}
    start = time.monotonic()
    result = call("/v1/chat/completions", payload)
    elapsed = time.monotonic() - start
    after = counters()
    usage = result.get("usage") or {}
    ttft = _find(result, "ttft_seconds")
    completion = usage.get("completion_tokens") or 0
    return {"elapsed_seconds": elapsed, "ttft_seconds": ttft, "completion_tokens": completion,
            "prompt_tokens": usage.get("prompt_tokens"),
            "decode_tps": ((completion - 1) / (elapsed - ttft)
                           if ttft and completion > 1 and elapsed > ttft else None),
            "token_ids": ((result.get("mlx2") or {}).get("token_trace") or {}).get("token_ids"),
            "delta": {k: (after[k] or 0) - (before[k] or 0) for k in COUNTERS}}


def run_arm(out: Path, model: Path, name: str, policy: dict | None) -> dict:
    extra = []
    if policy is not None:
        path = out / f"policy-{name}.json"
        path.write_text(json.dumps(policy) + "\n")
        extra = ["--execution-policy", str(path)]
    command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(model), "--host", "127.0.0.1",
               "--port", str(PORT), "--cache-dir", str(out / f"{name}-cache"), *SHAPE, *extra]
    row = {"name": name, "policy": policy, "command": command, "host_before": host_sample(), "cycles": []}
    log = (out / f"{name}.server.log").open("w")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                              env=checkout_env(), cwd=ROOT)
    try:
        deadline = time.monotonic() + 900
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"{name}: server exited before readiness")
            try:
                status = call("/v1/status", timeout=10)
                if status.get("healthy"):
                    break
            except (URLError, TimeoutError, ConnectionError, OSError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"{name}: readiness timeout")
            time.sleep(0.5)
        row["weight_streaming"] = _find(status, "weight_streaming")
        for cycle in range(3):
            d1 = chat(model.name, DECODE_PROMPTS[2 * cycle], 192)
            scan = chat(model.name, scan_document(100 + cycle) + "\n\nHow many records are listed? "
                        "Answer with a number only.", 1)
            d2 = chat(model.name, DECODE_PROMPTS[2 * cycle + 1], 192)
            row["cycles"].append({"d1": d1, "scan": scan, "d2": d2})
            print(json.dumps({"arm": name, "cycle": cycle, "d2_tps": d2["decode_tps"],
                              "d2_misses": d2["delta"]["stream_expert_misses_total"],
                              "scan_ttft": scan["ttft_seconds"]}), flush=True)
        row["final_counts"] = counters()
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=30)
        log.close()
        row["server_stopped"] = server.poll() is not None
        row["host_after"] = host_sample()
    return row


def unit(out: Path, model: Path, cache_gib: float, only: list[str] | None = None) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.adapters.registry import inspect_model

    def streaming(admission):
        return {"moe_expert_streaming": {"enabled": True, "cache_gib": cache_gib,
                                         "prefill_admission": admission}}

    arms = [("resident", None), ("lru", streaming("lru")), ("terminal_hot", streaming("terminal_hot")),
            ("scan_resistant", streaming("scan_resistant"))]
    if only:
        arms = [arm for arm in arms if arm[0] in only]
    receipt = {"schema": "mlx2.f3-scan-resistant-ab.v1", "source": source(),
               "checkpoint": f"{model.name}@sha256:{inspect_model(model).artifact_fingerprint}",
               "cache_gib": cache_gib, "shape": SHAPE, "qualified": False, "arms": []}
    try:
        for name, policy in arms:
            try:
                receipt["arms"].append(run_arm(out, model, name, policy))
            except Exception as exc:  # noqa: BLE001 - record and continue to the next arm
                receipt["arms"].append({"name": name, "error": repr(exc)})
            atomic_json(out / "receipt.json", receipt)
        reference = receipt["arms"][0]
        summary = {}
        for arm in receipt["arms"]:
            if "cycles" not in arm:
                summary[arm["name"]] = {"error": arm.get("error")}
                continue
            d2 = [c["d2"] for c in arm["cycles"]]
            summary[arm["name"]] = {
                "d2_tps": [round(x["decode_tps"] or 0, 2) for x in d2],
                "d2_misses": [x["delta"]["stream_expert_misses_total"] for x in d2],
                "d2_page_in_gib": [round(x["delta"]["stream_page_in_bytes_total"] / 2**30, 3) for x in d2],
                "scan_ttft": [round(c["scan"]["ttft_seconds"] or 0, 2) for c in arm["cycles"]],
                "protected_hits": arm.get("final_counts", {}).get("stream_scan_protected_hits_total"),
                "tokens_match_resident": all(
                    a[k]["token_ids"] == b[k]["token_ids"]
                    for a, b in zip(arm["cycles"], reference.get("cycles", []))
                    for k in ("d1", "scan", "d2")) if reference.get("cycles") else None,
            }
        receipt["summary"] = summary
    finally:
        atomic_json(out / "receipt.json", receipt)
    print(json.dumps(receipt.get("summary"), default=str), flush=True)
    return 0


def owned(out: Path, model: Path, cache_gib: float, job_id: str, only: list[str] | None = None) -> int:
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    with DesktopCPGLease(out, job_id, checkpoint=model.name, source_revision=source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out), "--model", str(model),
                   "--cache-gib", str(cache_gib), *(["--arms", *only] if only else [])]
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
    parser.add_argument("--cache-gib", type=float, default=4.0)
    parser.add_argument("--job-id", default="f3-scan-resistant-ab-20261105")
    parser.add_argument("--arms", nargs="+", choices=("resident", "lru", "terminal_hot", "scan_resistant"))
    args = parser.parse_args()
    out = args.out.resolve()
    if args.mode == "unit":
        return unit(out, args.model, args.cache_gib, args.arms)
    return owned(out, args.model, args.cache_gib, args.job_id, args.arms)


if __name__ == "__main__":
    raise SystemExit(main())
