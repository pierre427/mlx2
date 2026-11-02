"""GLM-5.3 AST/ordinary route tuning: one loopback server per configuration.

Each configuration starts ``mlx2.server`` with the given execution policy and
route flags, sends greedy chat requests on distinct realistic ~1K-token
prompts (warm-up first), and records decode tokens/s from the response
receipts (``(completion - 1) / (elapsed - ttft)``), the route and the
speculation receipt. The server's whole process group is stopped in
``finally``. Run under the GPU lock recipe (bench.sh).
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


def call(url, path, body=None, timeout=1800, token=None):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def prompts(n, chars):
    sys.path.insert(0, str(ROOT / "scripts"))
    import realistic_haystack

    text = realistic_haystack.haystack(chars * (n + 1))
    question = "\n\nSummarize the material above in a few paragraphs, then list three open questions it raises."
    return [text[i * chars:(i + 1) * chars] + question for i in range(n)]


def run_config(name, policy, flags, args, out):
    stage = out / name
    stage.mkdir()
    policy_path = stage / "policy.json"
    policy_path.write_text(json.dumps(policy, indent=1) + "\n")
    for key in ("admin-token", "reasoning-key"):
        (stage / key).write_text(secrets.token_hex(32))
        (stage / key).chmod(0o600)
    port = free_port()
    launcher = ([str(Path(__file__).with_name("bool_trace_server.py"))] if args.bool_trace
                else ["-m", "mlx2.server"])
    argv = [sys.executable, "-u", *launcher, "--model", MODEL, "--host", "127.0.0.1", "--port", str(port),
            "--fabric", "off", "--max-context", "32768", "--max-lanes", "1", "--max-inflight", "1",
            "--prefill-step", str(args.prefill_step), "--cache-bytes", str(4 << 30), "--cache-dir", str(stage / "cache"),
            "--execution-policy", str(policy_path), "--admin-token-file", str(stage / "admin-token"),
            "--reasoning-signing-key-file", str(stage / "reasoning-key"), *flags]
    (stage / "argv.json").write_text(json.dumps(argv, indent=1) + "\n")
    log = (stage / "server.log").open("w")
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src"),
                                 "BOOL_TRACE_OUT": str(stage / "bool-trace.json")})
    url = f"http://127.0.0.1:{port}"
    result = {"name": name, "flags": flags, "policy": policy, "requests": []}
    try:
        deadline = time.time() + 900
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode}")
            try:
                call(url, "/v1/models", timeout=5)
                # The served id is the status model (the model directory name).
                model = call(url, "/v1/status", timeout=5)["model"]
                result["model_id"] = model
                break
            except Exception:  # noqa: BLE001 -- not ready yet
                time.sleep(3)
        else:
            raise RuntimeError("server not ready")
        texts = prompts(args.requests + 1, args.prompt_chars)
        for i, text in enumerate(texts):
            body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": args.max_tokens,
                    "temperature": 0, "enable_thinking": False}
            if args.clear_apc:
                # Every request starts without retained prefix state (AGENTS.md ladder rule).
                cleared = call(url, "/v1/admin/apc/clear", {"reason": f"{name}-{i}"}, timeout=120,
                               token=(stage / "admin-token").read_text())
                result.setdefault("apc_clears", []).append(cleared)
            sampler = None
            if args.sample and i == 1:
                # Sample the server process mid-request (macOS sample(1)).
                sampler = subprocess.Popen(["/usr/bin/sample", str(proc.pid), "8", "-file",
                                            str(stage / "server-sample.txt")],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            r = call(url, "/v1/chat/completions", body)
            if sampler is not None:
                sampler.wait(timeout=120)
            if args.keep_receipts:
                (stage / f"response-{i}.json").write_text(json.dumps(r, indent=1) + "\n")
            m = r.get("mlx2") or {}
            decode = ((m.get("completion_tokens") or 0) - 1) / max(1e-9, (m.get("elapsed_seconds") or 0) - (m.get("ttft_seconds") or 0))
            row = {"warmup": i == 0, "prompt_tokens": m.get("prompt_tokens"), "completion_tokens": m.get("completion_tokens"),
                   "ttft_seconds": m.get("ttft_seconds"), "elapsed_seconds": m.get("elapsed_seconds"),
                   "decode_tokens_per_second": decode, "route": m.get("route"), "speculation": m.get("speculation"),
                   "mtp": m.get("mtp"), "profile": m.get("profile"),
                   "text_head": (r["choices"][0]["message"].get("content") or "")[:120]}
            result["requests"].append(row)
            print(f"{name:28s} {'warm' if i == 0 else 'req'+str(i)} prompt {row['prompt_tokens']} out {row['completion_tokens']} "
                  f"ttft {row['ttft_seconds']:.2f}s decode {decode:6.2f} tok/s route {row['route']}", flush=True)
        measured = [r["decode_tokens_per_second"] for r in result["requests"] if not r["warmup"]]
        result["decode_median"] = sorted(measured)[len(measured) // 2]
        print(f"{name:28s} MEDIAN decode {result['decode_median']:.2f} tok/s", flush=True)
    except Exception as error:  # noqa: BLE001 -- a refused configuration is a result too
        result["error"] = str(error)
        print(f"{name:28s} FAILED {error}", flush=True)
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
    (stage / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True, help="kernel policy JSON")
    parser.add_argument("--configs", required=True,
                        help="JSON list of {name, flags, policy_extra} (policy_extra merges into the policy)")
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt-chars", type=int, default=4200)
    parser.add_argument("--prefill-step", type=int, default=2048)
    parser.add_argument("--sample", action="store_true", help="macOS sample(1) of the server during request 1")
    parser.add_argument("--keep-receipts", action="store_true")
    parser.add_argument("--clear-apc", action="store_true",
                        help="POST /v1/admin/apc/clear (admin token) before every request")
    parser.add_argument("--bool-trace", action="store_true", help="count mx.array truth tests by call site")
    parser.add_argument("--source-revision", default="unrecorded")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    base = json.loads(args.policy.read_text())
    results = [run_config(c["name"], {**base, **c.get("policy_extra", {})}, c.get("flags", []), args, args.out)
               for c in json.loads(args.configs)]
    (args.out / "tune.json").write_text(json.dumps({"source_revision": args.source_revision, "results": results},
                                                   indent=1) + "\n")


if __name__ == "__main__":
    main()
