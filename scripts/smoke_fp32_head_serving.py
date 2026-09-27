#!/usr/bin/env python3
"""Serving smoke for the opt-in fp32 head on the default route (GPU).

Starts ``mlx2.server`` once per ``--order`` entry on the same model, default
route: ``bf16`` bare, ``fp32`` with ``--execution-policy {"fp32_head_logits":
true}`` (ABBA by default in practice: bf16,fp32,fp32,bf16), sends the same
greedy prompts (B=1, then B=4 concurrently) and records per request the text,
completion tokens and decode rate, plus the adapter diagnostics receipt.
Refuses to run without ``--i-own-the-gpu``.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_cached_replay import stream_chat  # noqa: E402
from qualify_interior_checkpoints import server_env  # noqa: E402

PROMPTS = [
    "Write a Python function that merges two sorted lists, with a docstring.",
    "Explain in five sentences why the sky is blue.",
    "List the planets of the solar system with one fact each.",
    "Translate to French: The quick brown fox jumps over the lazy dog, twice.",
    "Write a haiku sequence (three haiku) about compilers.",
    "What is 17 * 23? Show the steps.",
    "Summarize the plot of Hamlet in one paragraph.",
    "Give a JSON object describing a book with title, author, year, tags.",
]


def run_server(args, name, policy):
    command = [sys.executable, "-u", "-m", "mlx2.server", "--model", args.model,
               "--host", "127.0.0.1", "--port", str(args.port)]
    policy_file = None
    if policy is not None:
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(policy, handle)
        handle.close()
        policy_file = handle.name
        command += ["--execution-policy", policy_file]
    log = open(Path(args.out).parent / f"smoke-{name}.server.log", "w")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                              env=server_env(), start_new_session=True)
    base = f"http://127.0.0.1:{args.port}"
    result = {"arm": name, "policy": policy}
    try:
        deadline = time.monotonic() + 900
        while True:
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                pass
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"{name}: server failed to start")
            time.sleep(2)
        with urllib.request.urlopen(base + "/v1/status", timeout=60) as r:
            status = json.loads(r.read())
        model_id = status["model"]
        result["route"] = (status.get("settings") or {}).get("route")
        result["fp32_head_receipt"] = json.dumps(status).count('"fp32_head_logits"')

        def ask(prompt):
            body = {"model": model_id, "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": args.max_tokens, "temperature": 0, "stream": True,
                    "stream_options": {"include_usage": True}, "enable_thinking": False}
            r = stream_chat(base, body, 600)
            decode_s = r["total_s"] - (r["ttft_s"] or 0)
            return {"prompt": prompt[:40], "text": r["text"],
                    "completion_tokens": r["completion_tokens"],
                    "decode_tok_s": (r["completion_tokens"] - 1) / decode_s if decode_s > 0 else None}

        ask(PROMPTS[0])  # warm-up
        result["b1"] = [ask(p) for p in PROMPTS]
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            result["b4"] = list(pool.map(ask, PROMPTS[:4]))
        with urllib.request.urlopen(base + "/v1/status", timeout=60) as r:
            status = json.loads(r.read())
        result["mtp_counts"] = {k: v for k, v in (status.get("counts") or {}).items()
                                if "mtp" in k and ("accept" in k or "draft" in k)}
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=60)
        except Exception:  # noqa: BLE001
            os.killpg(server.pid, signal.SIGKILL)
        if policy_file:
            os.unlink(policy_file)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--port", type=int, default=8394)
    parser.add_argument("--order", default="bf16,fp32",
                        help="server launches in order, e.g. bf16,fp32,fp32,bf16 (ABBA)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        print("refusing: pass --i-own-the-gpu under the GPU lock wrapper", file=sys.stderr)
        return 2
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    policies = {"bf16": None, "fp32": {"fp32_head_logits": True}}
    arms = []
    for index, name in enumerate(args.order.split(",")):
        arm = run_server(args, f"{name}-{index}", policies[name])
        arm["variant"] = name
        arms.append(arm)
    first = {name: next(a for a in arms if a["variant"] == name) for name in policies
             if any(a["variant"] == name for a in arms)}
    same = [a["text"] == b["text"] for a, b in zip(first["bf16"]["b1"], first["fp32"]["b1"])]
    summary = {"b1_identical_texts": sum(same), "b1_requests": len(same)}
    # Rate ratios only over prompts whose text is identical in every launch,
    # so both variants decoded the same tokens.
    stable = [i for i in range(len(PROMPTS))
              if len({a["b1"][i]["text"] for a in arms}) == 1]
    summary["b1_prompts_identical_in_all_launches"] = stable
    for name in first:
        rates = [a["b1"][i]["decode_tok_s"] for a in arms if a["variant"] == name for i in stable]
        b4 = [x["decode_tok_s"] for a in arms if a["variant"] == name for x in a["b4"]]
        summary[f"{name}_b1_mean_tok_s_stable"] = sum(rates) / len(rates) if rates else None
        summary[f"{name}_b4_mean_tok_s"] = sum(b4) / len(b4)
    Path(args.out).write_text(json.dumps({"summary": summary, "arms": arms}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
