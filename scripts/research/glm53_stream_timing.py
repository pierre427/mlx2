"""GLM-5.3 served decode: per-token timing at one context, with and without APC writes.

Starts one loopback ``mlx2.server`` (ordinary route, the given execution
policy), then for each arm streams one greedy request on a realistic prompt of
``--tokens`` tokens and records the client monotonic time of every SSE delta.
Arms: ``write`` (default serving) and ``skip_write`` (the request sets
``skip_writing_prefix_cache``), each on a distinct prompt, APC cleared before
each. Reports inter-token gap percentiles for early (tokens 1-64), middle and
late (last 256) windows, so one-time post-first-token work (prompt-end APC
publish/spill) separates from a steady per-token cost. Server process group
stopped in ``finally``. Run under the GPU lock recipe (bench.sh).
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL = "~/mlx-models/GLM-5.3-Flash-REAP50-Q3_K_M"
ROOT = Path(__file__).resolve().parents[2]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(url, path, body=None, headers=None, timeout=1800):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def stream(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    stamps, final = [], None
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            event = json.loads(line[5:])
            choices = event.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content"):
                stamps.append(time.monotonic() - t0)
            if event.get("mlx2"):
                final = event["mlx2"]
    return stamps, final


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def windows(stamps):
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    out = {}
    for name, part in (("early_1_64", gaps[:64]), ("middle", gaps[64:-256] if len(gaps) > 320 else []),
                       ("late_last_256", gaps[-256:])):
        if part:
            out[name] = {"n": len(part), "p50_ms": pct(part, 0.5) * 1e3, "p90_ms": pct(part, 0.9) * 1e3,
                         "max_ms": max(part) * 1e3, "sum_s": sum(part)}
    big = sorted(((g, i + 1) for i, g in enumerate(gaps)), reverse=True)[:8]
    out["largest_gaps"] = [{"after_token": i, "ms": g * 1e3} for g, i in big]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--source-revision", default="unrecorded")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    for key in ("admin-token", "reasoning-key"):
        (args.out / key).write_text(secrets.token_hex(32))
        (args.out / key).chmod(0o600)
    port = free_port()
    argv = [sys.executable, "-u", "-m", "mlx2.server", "--model", MODEL, "--host", "127.0.0.1", "--port", str(port),
            "--fabric", "off", "--max-context", str(args.tokens + args.max_tokens + 4096), "--max-lanes", "1",
            "--max-inflight", "1", "--prefill-step", "2048", "--cache-bytes", str(4 << 30),
            "--cache-dir", str(args.out / "cache"), "--execution-policy", str(args.policy),
            "--admin-token-file", str(args.out / "admin-token"),
            "--reasoning-signing-key-file", str(args.out / "reasoning-key"), "--ordinary"]
    log = (args.out / "server.log").open("w")
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    url = f"http://127.0.0.1:{port}"
    admin = {"Authorization": "Bearer " + (args.out / "admin-token").read_text()}
    record = {"schema": "mlx2.glm53-stream-timing.v1", "source_revision": args.source_revision,
              "tokens": args.tokens, "max_tokens": args.max_tokens, "policy": json.loads(args.policy.read_text()),
              "argv": argv, "arms": []}
    try:
        deadline = time.time() + 900
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode}")
            try:
                model = call(url, "/v1/status", timeout=5)["model"]
                break
            except Exception:  # noqa: BLE001 -- not ready yet
                if time.time() > deadline:
                    raise
                time.sleep(3)
        sys.path.insert(0, str(ROOT / "scripts"))
        import realistic_haystack

        chars = int(args.tokens * 4.2)
        text = realistic_haystack.haystack(chars * 4)
        question = "\n\nSummarize the material above in detail, section by section."
        warm = {"model": model, "messages": [{"role": "user", "content": text[-4000:] + question}],
                "max_tokens": 16, "temperature": 0, "enable_thinking": False}
        call(url, "/v1/chat/completions", warm)
        for i, arm in enumerate(("write", "skip_write", "write")):
            try:
                call(url, "/v1/admin/apc/clear", {}, headers=admin)
            except Exception as error:  # noqa: BLE001
                record.setdefault("apc_clear_errors", []).append(repr(error))
            body = {"model": model, "messages": [{"role": "user", "content": text[i * chars:(i + 1) * chars] + question}],
                    "max_tokens": args.max_tokens, "temperature": 0, "enable_thinking": False, "stream": True,
                    "stream_options": {"include_usage": True}}
            if arm == "skip_write":
                body["skip_writing_prefix_cache"] = True
            stamps, final = stream(url, body)
            row = {"arm": arm, "rep": i, "deltas": len(stamps), "ttft_s": stamps[0] if stamps else None,
                   "decode_tok_s": (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else None,
                   "windows": windows(stamps), "receipt": final}
            record["arms"].append(row)
            print(json.dumps({k: row[k] for k in ("arm", "deltas", "ttft_s", "decode_tok_s")}),
                  json.dumps({k: v for k, v in row["windows"].items() if k != "largest_gaps"}), flush=True)
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        log.close()
        (args.out / "stream-timing.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
