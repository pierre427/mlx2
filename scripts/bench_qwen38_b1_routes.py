#!/usr/bin/env python3
"""B1/B-n serving comparison for Qwen3.8-27B routes through one HTTP client.

Every route (native MTP, ordinary, DFlash2 tree) is measured by the same
client against a running ``mlx2.server``, so prefill, APCv2, sampling
defaults and response construction are identical. Decode time is taken from
the server receipt (``elapsed_seconds - ttft_seconds``); client wall time is
recorded alongside it.

Prompts: two short prompts (code, chat) and the frozen Spomin case
``software_architecture.01`` rendered at ~7K and ~24K prompt tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_spomin_20x20 import FROZEN_CORPUS_SHA256, make_body, prepare_case

SHORT = {
    "code": "Write a short Python function that computes the Fibonacci sequence and explain it.",
    "chat": "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.",
}
CORPUS = Path(__file__).resolve().parent.parent / "qualification/corpora/spomin-20x20-long-multiturn-corpus-20260915.json"


def long_messages(model_path: str, target_tokens: int) -> list[dict]:
    from transformers import AutoTokenizer

    raw = CORPUS.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_CORPUS_SHA256:
        raise RuntimeError("frozen Spomin corpus differs")
    corpus = json.loads(raw)
    case = next(c for c in corpus["cases"] if c["case_id"] == "software_architecture.01")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # Same capacity mapping as the codex four-arm harness: 8192 -> ~6.95K IDs,
    # round(24000 / 0.84) -> ~24K IDs.
    capacity = 8192 if target_tokens <= 8192 else round(target_tokens / 0.84)
    prepared = prepare_case(case, tokenizer, capacity)
    return make_body(corpus["system"], prepared, "full", 64)["messages"]


def post(base: str, body: dict, timeout: float = 1800) -> tuple[dict, float]:
    started = time.perf_counter()
    with urlopen(Request(base.rstrip("/") + "/v1/chat/completions",
                         data=json.dumps(body).encode(),
                         headers={"Content-Type": "application/json"}), timeout=timeout) as r:
        result = json.load(r)
    return result, time.perf_counter() - started


def row_from(result: dict, wall: float) -> dict:
    choice = result["choices"][0]
    message = choice["message"]
    content = (message.get("reasoning_content") or "") + (message.get("content") or "")
    receipt = result.get("mlx2") or {}
    completion = int(result["usage"]["completion_tokens"])
    elapsed = receipt.get("elapsed_seconds")
    ttft = receipt.get("ttft_seconds")
    peer = result.get("tensorfold")
    if not receipt and isinstance(peer, dict):
        # TensorFold's own receipt: total and time-to-first-token seconds.
        elapsed, ttft = peer.get("seconds"), peer.get("time_to_first_token")
        receipt = {"route": "tensorfold", "tensorfold": peer,
                   "speculation": result.get("speculative")}
    decode = (elapsed - ttft) if elapsed is not None and ttft is not None else None
    spec = receipt.get("speculation") or {}
    return {
        "wall_seconds": round(wall, 4),
        "ttft_seconds": ttft,
        "decode_seconds": None if decode is None else round(decode, 4),
        "decode_tps": (None if not decode or completion < 2
                       else round((completion - 1) / decode, 3)),
        "completion_tokens": completion,
        "prompt_tokens": result["usage"]["prompt_tokens"],
        "cached_tokens": receipt.get("cached_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "output_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "output_preview": content[:160],
        "route": receipt.get("route"),
        "profile": receipt.get("profile"),
        "qualification": receipt.get("qualification"),
        "speculation": spec,
        "receipt": receipt,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--model-path", required=True, help="artifact path, for the tokenizer")
    p.add_argument("--label", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompts", default="code,chat,spomin7k,spomin24k")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--sampling", choices=("greedy", "default", "both"), default="greedy")
    p.add_argument("--extra-body", default="{}",
                   help="JSON merged into every request body (e.g. penalties, draft)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="send N distinct-seed copies at once (B-n); default B1")
    args = p.parse_args()

    prompts: dict[str, list[dict]] = {}
    for name in args.prompts.split(","):
        if name in SHORT:
            prompts[name] = [{"role": "user", "content": SHORT[name]}]
        elif name == "spomin7k":
            prompts[name] = long_messages(args.model_path, 7000)
        elif name == "spomin24k":
            prompts[name] = long_messages(args.model_path, 24000)
        else:
            raise SystemExit(f"unknown prompt {name}")
    modes = {"greedy": ("greedy",), "default": ("default",), "both": ("greedy", "default")}[args.sampling]
    rows = []
    for name, messages in prompts.items():
        for mode in modes:
            body = {"messages": messages, "max_tokens": args.tokens, "enable_thinking": False,
                    **json.loads(args.extra_body)}
            if mode == "greedy":
                body.update(temperature=0.0)
            for rep in range(args.reps + 1):
                n = args.concurrency
                bodies = [dict(body, seed=1000 + rep * 17 + i) for i in range(n)]
                if n == 1:
                    results = [post(args.base, bodies[0])]
                else:
                    results: list = [None] * n
                    barrier = threading.Barrier(n)

                    def go(i, barrier=barrier, results=results, bodies=bodies):
                        barrier.wait()
                        results[i] = post(args.base, bodies[i])
                    started = time.perf_counter()
                    threads = [threading.Thread(target=go, args=(i,)) for i in range(n)]
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    makespan = time.perf_counter() - started
                for i, (result, wall) in enumerate(results):
                    row = row_from(result, wall)
                    row.update(prompt=name, sampling=mode, rep=rep, lane=i,
                               warmup=rep == 0, concurrency=n)
                    if n > 1:
                        row["makespan_seconds"] = round(makespan, 4)
                    rows.append(row)
                    print(json.dumps({k: row[k] for k in (
                        "prompt", "sampling", "rep", "lane", "prompt_tokens", "completion_tokens",
                        "ttft_seconds", "decode_seconds", "decode_tps", "route", "output_sha256")}),
                        flush=True)
    summary = {}
    for name in prompts:
        for mode in modes:
            measured = [r for r in rows if r["prompt"] == name and r["sampling"] == mode
                        and not r["warmup"] and r["decode_tps"]]
            if measured:
                summary[f"{name}/{mode}"] = {
                    "median_decode_tps": statistics.median(r["decode_tps"] for r in measured),
                    "median_decode_seconds": statistics.median(r["decode_seconds"] for r in measured),
                    "samples": len(measured),
                    "completion_tokens": sorted({r["completion_tokens"] for r in measured}),
                    "distinct_outputs": len({r["output_sha256"] for r in measured}),
                    "routes": sorted({str(r["route"]) for r in measured}),
                }
    try:
        with urlopen(args.base.rstrip("/") + "/v1/status", timeout=30) as r:
            status = json.load(r)
    except OSError:
        status = None  # peer servers have no mlx2 status endpoint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema": "mlx2.qwen38-b1-routes.v1", "label": args.label, "tokens": args.tokens,
        "reps": args.reps, "concurrency": args.concurrency, "summary": summary,
        "status": status, "rows": rows,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"label": args.label, "summary": summary}, indent=1), flush=True)


if __name__ == "__main__":
    main()
