"""Served check of ``inkling2_attn_decode_kernel``: does it engage in serving?

Original mlx2 script (2026-11-22). The batch generator merges even one
serving lane into BatchKVCache / BatchRotatingKVCache, so the direct-API
check does not show the served route. For each arm (option off, then on) it
starts one loopback ``mlx2.server`` (never port 8282; ``--max-lanes 1``;
``--execution-policy`` with only this option set, so every other switch keeps
its shipped default), then:

1. one greedy chat request with ``logprobs`` / ``top_logprobs`` 8 (token ids
   and the per-position windows used only at a first divergence);
2. ``--timing`` greedy streamed requests without logprobs; decode tok/s from
   first to last content chunk (sanity, contended host, no thermal hold);
3. ``/v1/status``: the route's ``kernel_counters`` (``attn_decode`` calls and
   declines by reason) and the request's route receipt.

The server process group is stopped in ``finally``. Tokens of the two arms
are compared: exact, or the first divergence classified with
the log-probability half of ``mlx2.near_tie.near_tie_equivalent`` (0.5 nat; a row is near_tie_equivalent only when both continuations also pass a functional oracle, AGENTS.md) from both windows at that
position. Refuses a nonempty run directory; writes ``serve.json`` atomically.

Usage: python scripts/research/inkling2_attn_decode_serve.py <model dir> <run dir>
       [--port 8397] [--max-tokens 128] [--timing 3]
The caller owns the GPU (scripts/run_with_gpu_locks.py).
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _post(port, path, body, token=None, timeout=900):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _get(port, path, token=None):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def _find(value, key, path=""):
    """Every ``key`` entry in a nested JSON value, with its path."""
    found = []
    if isinstance(value, dict):
        for k, v in value.items():
            here = f"{path}.{k}" if path else k
            if k == key:
                found.append((here, v))
            found.extend(_find(v, key, here))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            found.extend(_find(v, key, f"{path}[{i}]"))
    return found


def _stream_timing(port, body):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({**body, "stream": True}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    chunks = 0
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            event = json.loads(line[5:])
            for choice in event.get("choices", ()):
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    now = time.perf_counter()
                    first = now if first is None else first
                    last = now
                    chunks += 1
    return {"ttft_s": None if first is None else first - t0, "content_chunks": chunks,
            "decode_s": None if first is None else last - first,
            "decode_chunks_per_s": None if not first or last == first
            else (chunks - 1) / (last - first)}


# Functional oracles (scripts/smoke_inkling2.py's), checked on the content
# channel; the summary prompt has none.
ORACLES = (
    ("capital", "What is the capital of France? Answer in one word.", r"\bParis\b"),
    ("arithmetic", "What is 17 + 25? Reply with the number only.", r"\b42\b"),
)


def _request(port, model_id, prompt, max_tokens):
    body = {"model": model_id, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "reasoning_effort": "none"}
    payload = _post(port, "/v1/chat/completions", {**body, "logprobs": True, "top_logprobs": 8})
    choice = payload["choices"][0]
    rows = (choice.get("logprobs") or {}).get("content") or []
    message = choice.get("message") or {}
    return body, {
        "text": message.get("content"), "reasoning": message.get("reasoning_content"),
        "finish_reason": choice.get("finish_reason"), "usage": payload.get("usage"),
        "token_ids": [int(row["id"]) for row in rows],
        "windows": [{int(item["id"]): float(item["logprob"])
                     for item in [row, *row.get("top_logprobs", ())]} for row in rows],
        "route_receipt": payload.get("mlx2")}


def arm(model, run_dir, port, name, on, prompt, max_tokens, timing):
    arm_dir = run_dir / name
    arm_dir.mkdir()
    token = secrets.token_hex(16)
    token_file = arm_dir / "admin.token"
    token_file.write_text(token)
    os.chmod(token_file, 0o600)
    policy = arm_dir / "policy.json"
    policy.write_text(json.dumps({"inkling2_attn_decode_kernel": on}))
    log = open(arm_dir / "server.log", "w")
    cmd = [sys.executable, "-m", "mlx2.server", "--model", str(model), "--host", "127.0.0.1",
           "--port", str(port), "--max-lanes", "1", "--cache-bytes", str(8 << 30),
           "--admin-token-file", str(token_file), "--execution-policy", str(policy)]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    result = {"command": cmd, "policy": json.loads(policy.read_text())}
    try:
        deadline = time.monotonic() + 900
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"{name} server did not become healthy; see server.log")
            time.sleep(2)
        model_id = _get(port, "/v1/models")["data"][0]["id"]
        before = _get(port, "/v1/status", token)
        _post(port, "/v1/admin/apc/clear", {"reason": "attn-decode-serve"}, token)
        body, main = _request(port, model_id, prompt, max_tokens)
        after = _get(port, "/v1/status", token)
        result.update(main)
        result["attn_decode_before"] = _find(before, "attn_decode")
        result["attn_decode_after"] = _find(after, "attn_decode")
        result["oracles"] = {}
        for label, text, pattern in ORACLES:
            _post(port, "/v1/admin/apc/clear", {"reason": "attn-decode-serve"}, token)
            _, got = _request(port, model_id, text, 256)
            import re

            got["oracle_pass"] = bool(re.search(pattern, got["text"] or ""))
            result["oracles"][label] = got
        result["attn_decode_after_oracles"] = _find(_get(port, "/v1/status", token),
                                                    "attn_decode")
        result["kernel_policy"] = _find(after, "kernel_policy")[:1]
        result["checkpoint"] = _find(after, "checkpoint")[:4]
        result["timing"] = [_stream_timing(port, body) for _ in range(timing)]
        result["attn_decode_after_timing"] = _find(_get(port, "/v1/status", token),
                                                   "attn_decode")
        (arm_dir / "status-after.json").write_text(json.dumps(after, indent=1, default=str))
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        log.close()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--port", type=int, default=8397)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timing", type=int, default=3)
    a = ap.parse_args()
    if a.port == 8282:
        raise SystemExit("port 8282 is reserved for release trees")
    if a.run_dir.exists() and any(a.run_dir.iterdir()):
        raise SystemExit(f"refusing a nonempty run directory {a.run_dir}")
    a.run_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.near_tie import near_tie_equivalent

    # Varied text, not a repeated passage: this repo's Inkling 2 doc.
    prompt = ("Summarize the following document in two sentences.\n\n"
              + (ROOT / "docs/INKLING2-MINI.md").read_text()[:6000])
    receipt = {"script": "scripts/research/inkling2_attn_decode_serve.py",
               "mlx2": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
               "model_path": str(a.model),
               "kind": "sanity, B1 loopback server per arm, no thermal hold", "arms": {}}
    for name, on in (("off", False), ("on", True)):
        receipt["arms"][name] = arm(a.model.resolve(), a.run_dir, a.port, name, on, prompt,
                                    a.max_tokens, a.timing)
        print(name, json.dumps({"attn_decode": receipt["arms"][name]["attn_decode_after"],
                                "tokens": len(receipt["arms"][name]["token_ids"]),
                                "timing": receipt["arms"][name]["timing"]}, default=str),
              flush=True)
    def compare(off, on):
        first = next((i for i, (x, y) in enumerate(zip(off["token_ids"], on["token_ids"]))
                      if x != y), None)
        if first is None and len(off["token_ids"]) != len(on["token_ids"]):
            first = min(len(off["token_ids"]), len(on["token_ids"]))
        out = {"exact": first is None, "first_divergence": first,
               "tokens": [len(off["token_ids"]), len(on["token_ids"])]}
        if first is not None and first < min(len(off["windows"]), len(on["windows"])):
            out["near_tie_logprob_test"] = near_tie_equivalent(
                off["token_ids"][first], off["windows"][first],
                on["token_ids"][first], on["windows"][first])
            out["window_off"] = off["windows"][first]
            out["window_on"] = on["windows"][first]
        if "oracle_pass" in off:
            out["oracle_pass"] = [off["oracle_pass"], on["oracle_pass"]]
        return out

    off, on = receipt["arms"]["off"], receipt["arms"]["on"]
    receipt["compare"] = {"summary": compare(off, on),
                          **{label: compare(off["oracles"][label], on["oracles"][label])
                             for label in off["oracles"]}}
    for arm_receipt in receipt["arms"].values():
        arm_receipt.pop("windows")
        for got in arm_receipt["oracles"].values():
            got.pop("windows")
    compare = receipt["compare"]
    tmp = a.run_dir / "serve.json.tmp"
    tmp.write_text(json.dumps(receipt, indent=1, default=str))
    tmp.rename(a.run_dir / "serve.json")
    print(json.dumps(compare, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
