"""Served decode under one process-level setting (one process = one arm).

Starts one loopback ``mlx2.server`` for ``--model`` with the given server
arguments (normally a service's committed shape, with a fresh cache
directory), then for each request kind streams one greedy request on a
realistic ``--tokens``-token prompt and records the client monotonic time of
every SSE delta. Kinds: ``default`` (the route's own default, e.g. native
MTP) and ``spec_off`` (the request sets ``"speculation": "off"``, ordinary
decode). Each kind uses its own slice of the haystack, the same slice in every
process, and APCv2 is cleared before each, so every request is a cold 32K
prefill followed by decode and text hashes compare across arms.

Process-level knobs latched when Metal starts (the command-buffer cadence:
``MLX_MAX_OPS_PER_BUFFER`` / ``MLX_MAX_MB_PER_BUFFER``, see
``process_env.COMMAND_BUFFER_POLICIES``) come from this process's
environment, which the caller sets per arm; the server inherits it. Decode
tok/s counts ``usage.completion_tokens`` (a speculative delta can carry
several tokens) over first-to-last delta time. Sanity A/B, no thermal hold,
not publishable statistics. Server process group stopped in ``finally``. The
caller owns the GPU.

Usage: served_cmdbuf_probe.py --model DIR --out RUN_DIR --label ARM
       [--tokens 32768] [--max-tokens 256] [--kinds default,spec_off]
       [--source-revision SHA] [--checkpoint NAME@FP] -- SERVER ARGS...
"""

from __future__ import annotations

import argparse
import hashlib
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

ROOT = Path(__file__).resolve().parents[2]
ENV_KEYS = ("MLX2_COMMAND_BUFFERS", "MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER",
            "MLX_METAL_FAST_SYNCH")


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
    stamps, text, usage, receipt = [], [], None, None
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
                text.append(choices[0]["delta"]["content"])
            if event.get("usage"):
                usage = event["usage"]
            if event.get("mlx2"):
                receipt = event["mlx2"]
    return stamps, "".join(text), usage, receipt


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def main():
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--kinds", default="default,spec_off")
    parser.add_argument("--source-revision", default="unrecorded")
    parser.add_argument("--checkpoint", default="unrecorded")
    args = parser.parse_args(argv[:split])
    server_args = argv[split + 1:]
    args.out.mkdir(parents=True, exist_ok=False)
    for key in ("admin-token", "reasoning-key"):
        (args.out / key).write_text(secrets.token_hex(32))
        (args.out / key).chmod(0o600)
    port = free_port()
    cmd = [sys.executable, "-u", "-m", "mlx2.server", "--model", args.model, "--host", "127.0.0.1",
           "--port", str(port), "--fabric", "off", *server_args,
           "--cache-dir", str(args.out / "cache"),
           "--admin-token-file", str(args.out / "admin-token"),
           "--reasoning-signing-key-file", str(args.out / "reasoning-key")]
    log = (args.out / "server.log").open("w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    url = f"http://127.0.0.1:{port}"
    admin = {"Authorization": "Bearer " + (args.out / "admin-token").read_text()}
    record = {"schema": "mlx2.served-cmdbuf-probe.v1", "label": args.label,
              "source_revision": args.source_revision, "checkpoint": args.checkpoint,
              "tokens": args.tokens, "max_tokens": args.max_tokens,
              "env": {k: os.environ.get(k) for k in ENV_KEYS}, "argv": cmd, "requests": []}
    # Which MLX build this interpreter loads (arms may differ only by build).
    record["mlx_build"] = json.loads(subprocess.run(
        [sys.executable, "-c",
         "import hashlib, json, mlx, mlx.core as mx, pathlib\n"
         "lib = pathlib.Path(list(mlx.__path__)[0], 'lib', 'libmlx.dylib')\n"
         "print(json.dumps({'version': mx.__version__, 'path': list(mlx.__path__),\n"
         "  'libmlx_sha256': hashlib.sha256(lib.read_bytes()).hexdigest() if lib.exists() else None}))"],
        capture_output=True, text=True, check=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    ).stdout)
    t_start = time.time()
    try:
        deadline = time.time() + 1200
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode}")
            try:
                # /v1/status answers while the engine is still loading; /health
                # is 200 only once it serves (a warm request before that is 503).
                call(url, "/health", timeout=5)
                status = call(url, "/v1/status", timeout=5)
                break
            except Exception:  # noqa: BLE001 -- not ready yet
                if time.time() > deadline:
                    raise
                time.sleep(3)
        record["load_s"] = time.time() - t_start
        model = status["model"]
        record["status_command_buffers"] = {k: v for k, v in status.items()
                                            if "command_buffer" in k or "process_env" in k}
        sys.path.insert(0, str(ROOT / "scripts"))
        import realistic_haystack

        chars = int(args.tokens * 4.2)
        text = realistic_haystack.haystack(chars * 4)
        question = "\n\nSummarize the material above in detail, section by section."
        warm = {"model": model, "messages": [{"role": "user", "content": text[-4000:] + question}],
                "max_tokens": 16, "temperature": 0, "enable_thinking": False}
        call(url, "/v1/chat/completions", warm)
        for i, kind in enumerate(args.kinds.split(",")):
            try:
                call(url, "/v1/admin/apc/clear", {}, headers=admin)
            except Exception as error:  # noqa: BLE001
                record.setdefault("apc_clear_errors", []).append(repr(error))
            body = {"model": model,
                    "messages": [{"role": "user", "content": text[i * chars:(i + 1) * chars] + question}],
                    "max_tokens": args.max_tokens, "temperature": 0, "enable_thinking": False,
                    "stream": True, "stream_options": {"include_usage": True}}
            if kind == "spec_off":
                body["speculation"] = "off"
            stamps, out, usage, receipt = stream(url, body)
            gaps = [b - a for a, b in zip(stamps, stamps[1:])]
            n = (usage or {}).get("completion_tokens")
            span = stamps[-1] - stamps[0] if len(stamps) > 1 else None
            row = {"kind": kind, "slice": i, "deltas": len(stamps), "completion_tokens": n,
                   "prompt_tokens": (usage or {}).get("prompt_tokens"),
                   "ttft_s": stamps[0] if stamps else None,
                   "decode_tok_s": (n - 1) / span if n and span else None,
                   "gap_p50_ms": pct(gaps, 0.5) * 1e3 if gaps else None,
                   "gap_p90_ms": pct(gaps, 0.9) * 1e3 if gaps else None,
                   "text_sha256": hashlib.sha256(out.encode()).hexdigest(), "text": out,
                   "receipt": receipt}
            record["requests"].append(row)
            print(json.dumps({"label": args.label, **{k: row[k] for k in (
                "kind", "completion_tokens", "ttft_s", "decode_tok_s", "gap_p50_ms", "text_sha256")}}),
                  flush=True)
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
        for key in ("admin-token", "reasoning-key"):
            (args.out / key).unlink(missing_ok=True)
        (args.out / "probe.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
