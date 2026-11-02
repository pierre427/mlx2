"""Ask GLM-5.3 itself for optimization ideas about its own serving (Pierre, 2026-11-01).

Starts one loopback ``mlx2.server`` (ordinary route, the given execution
policy), sends one chat request with thinking on (``reasoning_effort`` high,
a bounded ``thinking_budget``, temperature 0.7) carrying the current state of
the campaign from ``--brief``, and saves the reasoning and answer. The server
process group is stopped in ``finally``. Run under the GPU lock (bench.sh).
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
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "~/mlx-models/GLM-5.3-Flash-REAP50-Q3_K_M"
ROOT = Path(__file__).resolve().parents[2]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(url, path, body=None, timeout=3600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode(errors='replace')[:2000]}") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--brief", type=Path, required=True, help="markdown question for the model")
    parser.add_argument("--thinking-budget", type=int, default=6144)
    parser.add_argument("--max-tokens", type=int, default=10240)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--source-revision", default="unrecorded")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    for key in ("admin-token", "reasoning-key"):
        (args.out / key).write_text(secrets.token_hex(32))
        (args.out / key).chmod(0o600)
    port = free_port()
    argv = [sys.executable, "-u", "-m", "mlx2.server", "--model", MODEL, "--host", "127.0.0.1", "--port", str(port),
            "--fabric", "off", "--max-context", "32768", "--max-lanes", "1", "--max-inflight", "1",
            "--prefill-step", "2048", "--cache-bytes", str(4 << 30), "--cache-dir", str(args.out / "cache"),
            "--execution-policy", str(args.policy), "--admin-token-file", str(args.out / "admin-token"),
            "--reasoning-signing-key-file", str(args.out / "reasoning-key"), "--ordinary"]
    log = (args.out / "server.log").open("w")
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    url = f"http://127.0.0.1:{port}"
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
        body = {"model": model, "messages": [{"role": "user", "content": args.brief.read_text()}],
                "temperature": args.temperature, "max_tokens": args.max_tokens,
                "enable_thinking": True, "reasoning_effort": "high", "thinking_budget": args.thinking_budget}
        t0 = time.time()
        refused = []
        # The route refuses controls it cannot honour (HTTP 400, fail closed):
        # drop them one at a time and record each refusal.
        for drop in (None, "reasoning_effort", "thinking_budget"):
            if drop:
                body.pop(drop, None)
            try:
                response = call(url, "/v1/chat/completions", body)
                break
            except RuntimeError as error:
                refused.append({"dropped_next": drop, "error": str(error)})
                print(json.dumps(refused[-1]), flush=True)
        else:
            raise RuntimeError(f"every request variant refused: {refused}")
        message = response["choices"][0]["message"]
        record = {"source_revision": args.source_revision, "request": {k: v for k, v in body.items() if k != "messages"},
                  "seconds": time.time() - t0, "refused": refused, "usage": response.get("usage"), "mlx2": response.get("mlx2"),
                  "reasoning": message.get("reasoning_content") or message.get("reasoning"),
                  "answer": message.get("content")}
        (args.out / "answer.json").write_text(json.dumps(record, indent=1) + "\n")
        (args.out / "answer.md").write_text(record["answer"] or "")
        print(json.dumps({"seconds": round(record["seconds"], 1), "usage": record["usage"],
                          "answer_chars": len(record["answer"] or ""),
                          "reasoning_chars": len(record["reasoning"] or "")}), flush=True)
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


if __name__ == "__main__":
    main()
