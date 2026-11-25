#!/usr/bin/env python3
"""Raon-1-mini prompt-processing ladder through the loopback server (sanity, not publishable).

    PYTHONPATH=src python -s scripts/raon1_mini_prefill_ladder.py --model ART --out DIR \
        --cpg-lock DIR [--cells 2048,4096,8192,16384,32768] [--reps 3] [--prefill-steps 2048,8192]

One server per prefill-step arm (never :8282), stopped as a process group in
``finally``.  Per cell and rep: APCv2 cleared (POST /v1/admin/apc/clear), one
streaming chat request of exactly ``cell`` prompt tokens of varied repository
text (realistic_haystack, a different offset per rep, never a repeated
string), thinking off, greedy, max_tokens 8.  Cold TTFT = first streamed
chunk; prefill tok/s = prompt_tokens / TTFT.  Then one warm repeat of the last
rep's body (APCv2 hit).  No thermal hold (AGENTS.md); caller owns the GPU lease.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def call(base, path, token, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + path, data=data, method="GET" if body is None else "POST",
                                     headers={"content-type": "application/json",
                                              "authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode()
    return json.loads(raw) if raw.strip().startswith(("{", "[")) else raw


def stream(base, body, timeout=2400):
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
    start = time.perf_counter()
    first, usage, text, last = None, None, "", None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece and first is None:
                    first = time.perf_counter() - start
                text += piece
            last = chunk
    return {"ttft_s": first, "total_s": time.perf_counter() - start, "usage": usage, "text": text,
            "receipt": (last or {}).get("mlx2") or (last or {}).get("route_receipt")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpg-lock", type=Path, required=True)
    parser.add_argument("--cells", default="2048,4096,8192,16384,32768")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--prefill-steps", default="2048,8192")
    parser.add_argument("--port", type=int, default=18431)
    parser.add_argument("--extra-args", default="--ordinary")
    parser.add_argument("--needle", action="store_true",
                        help="bury a unique access code at 1/3 depth of every prompt and score the reply")
    parser.add_argument("--label", default="", help="arm label in the report")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.port == 8282:
        raise SystemExit("port 8282 is reserved")
    owner = json.loads((args.cpg_lock / "owner.json").read_text())
    from transformers import AutoTokenizer

    from mlx2.adapters.registry import inspect_model
    from realistic_haystack import bury, haystack

    resolution = inspect_model(args.model)
    checkpoint = f"{args.model.name}@sha256:{resolution.artifact_fingerprint}"
    tok = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    cells = [int(c) for c in args.cells.split(",")]
    question = ("\n\nWhat is the access code stated in the text above? Reply with the code only."
                if args.needle else "\n\nIn one short sentence, what is the main topic of the text above?")
    codes = {}
    overhead = len(tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                           enable_thinking=False, return_dict=True)["input_ids"])
    prompts = {}
    offset = 0
    for cell in cells:
        for rep in range(args.reps):
            text = haystack(cell * 6, offset=offset)
            offset += len(text)
            budget = cell - overhead
            if args.needle:
                code = f"RX-{(cell * 7 + rep * 131) % 9000 + 1000}-{'QZKMTV'[rep % 6]}{'LPWHJN'[cell % 6]}"
                fact = f"The access code for this archive is {code}."
                codes[(cell, rep)] = code
                budget -= len(tok.encode(fact, add_special_tokens=False)) + 2
            ids = tok.encode(text, add_special_tokens=False)[:budget]
            body_text = tok.decode(ids)
            if args.needle:
                body_text = bury(body_text, fact)
            prompts[(cell, rep)] = body_text + question
    revision = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    report = {"schema": "mlx2.raon1-mini-prefill-ladder.v1", "checkpoint": checkpoint, "mlx2_revision": revision,
              "ownership": {k: owner.get(k) for k in ("label", "cpg_task", "cpg_generation")},
              "cells": cells, "reps": args.reps, "thermal_hold": False, "arms": []}
    token_file = args.out / "admin-token"
    token_file.write_text(os.urandom(24).hex())
    token_file.chmod(0o600)
    token = token_file.read_text()
    base = f"http://127.0.0.1:{args.port}"
    for step in [int(s) for s in args.prefill_steps.split(",")]:
        arm_dir = args.out / f"step{step}"
        arm_dir.mkdir()
        command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(args.model), "--host", "127.0.0.1",
                   "--port", str(args.port), "--max-inflight", "1", "--max-lanes", "1",
                   "--max-context", str(max(cells) + 4096), "--cache-bytes", str(4 << 30),
                   "--cache-dir", str(arm_dir / "cache"), "--admin-token-file", str(token_file),
                   "--prefill-step", str(step), *args.extra_args.split()]
        log = (arm_dir / "server.log").open("w")
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                  env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, cwd=ROOT)
        arm = {"label": args.label, "prefill_step": step, "command": command, "cells": []}
        try:
            ready = time.monotonic() + 900
            while True:
                if server.poll() is not None:
                    raise RuntimeError("server exited before readiness")
                try:
                    if call(base, "/v1/status", token, timeout=10).get("healthy"):
                        break
                except OSError:
                    pass
                if time.monotonic() > ready:
                    raise RuntimeError("server readiness timed out")
                time.sleep(0.5)
            model_id = call(base, "/v1/models", token)["data"][0]["id"]

            def body(text):
                return {"model": model_id, "messages": [{"role": "user", "content": text}], "stream": True,
                        "stream_options": {"include_usage": True}, "max_tokens": 24 if args.needle else 8, "temperature": 0,
                        "enable_thinking": False, "seed": 1}

            # Warm-up (kernel compilation), not recorded.
            call(base, "/v1/admin/apc/clear", token, body={})
            stream(base, body(prompts[(cells[0], 0)][:2000]))
            for cell in cells:
                cold = []
                for rep in range(args.reps):
                    call(base, "/v1/admin/apc/clear", token, body={})
                    result = stream(base, body(prompts[(cell, rep)]))
                    tokens = (result["usage"] or {}).get("prompt_tokens")
                    result["prefill_tok_s"] = tokens / result["ttft_s"] if tokens and result["ttft_s"] else None
                    cold.append(result)
                    print(json.dumps({"step": step, "cell": cell, "rep": rep, "prompt_tokens": tokens,
                                      "ttft_s": round(result["ttft_s"], 3),
                                      "prefill_tok_s": round(result["prefill_tok_s"] or 0, 1)}), flush=True)
                warm = stream(base, body(prompts[(cell, args.reps - 1)]))
                rates = [r["prefill_tok_s"] for r in cold if r["prefill_tok_s"]]
                arm["cells"].append({
                    "cell": cell, "prompt_tokens": [(r["usage"] or {}).get("prompt_tokens") for r in cold],
                    "cold_ttft_s": [round(r["ttft_s"], 4) for r in cold],
                    "cold_prefill_tok_s": [round(v, 1) for v in rates],
                    "cold_prefill_tok_s_median": round(statistics.median(rates), 1) if rates else None,
                    "warm_ttft_s": round(warm["ttft_s"], 4),
                    "warm_cached_tokens": ((warm["usage"] or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
                    "cold_text": [r["text"] for r in cold], "warm_text": warm["text"],
                    **({"needle_codes": [codes[(cell, rep)] for rep in range(args.reps)],
                        "needle_found": [codes[(cell, rep)] in r["text"].replace("\u2011", "-").replace("\u2013", "-")
                                         for rep, r in enumerate(cold)]} if args.needle else {})})
            arm["final_status"] = call(base, "/v1/status", token)
            arm["passed"] = True
        except Exception as error:  # noqa: BLE001 - recorded, next arm still runs
            arm["passed"] = False
            arm["error"] = repr(error)
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
            log.close()
        report["arms"].append(arm)
        tmp = args.out / "ladder.json.tmp"
        tmp.write_text(json.dumps(report, indent=1, default=str))
        os.replace(tmp, args.out / "ladder.json")
    for arm in report["arms"]:
        print(json.dumps({"step": arm["prefill_step"], "passed": arm["passed"],
                          "median_tok_s": {c["cell"]: c["cold_prefill_tok_s_median"] for c in arm["cells"]},
                          "warm_ttft": {c["cell"]: c["warm_ttft_s"] for c in arm["cells"]}}))


if __name__ == "__main__":
    main()
