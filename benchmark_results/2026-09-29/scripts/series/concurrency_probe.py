#!/usr/bin/env python3
"""Batching and concurrency probe for one running mlx2 server.

At each width, that many streamed chat requests start together.  Every
request asks a question with its own verifiable answer (a distinct sum and a
distinct code word), so a stream that receives another stream's tokens fails
its own check.  Records per-width correctness, errors, aggregate and
per-stream decode tok/s and TTFT percentiles.  Then cancels one stream in the
middle of a batch and checks the server is still healthy and serves the next
request.  Correctness, not performance: no thermal control.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request

WIDTHS = (1, 2, 4, 8, 16)
WORDS = ("amber", "basalt", "cobalt", "delta", "ember", "fjord", "garnet", "harbor",
         "indigo", "juniper", "kelp", "lumen", "marble", "nimbus", "onyx", "prism",
         "quartz", "raven", "sierra", "tundra")


def status(base):
    with urllib.request.urlopen(base + "/v1/status", timeout=60) as response:
        return json.loads(response.read())


def question(index, salt):
    a, b = 1000 + 37 * index + salt, 211 + 13 * index
    word = WORDS[(index + salt) % len(WORDS)]
    prompt = (f"Compute {a} + {b}. Then repeat the code word '{word}'. "
              f"Answer in exactly this form: SUM=<number> WORD=<word>")
    return prompt, str(a + b), word


def stream_chat(base, model, prompt, max_tokens, *, cancel_after=None, timeout=900):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "enable_thinking": False}
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    first = None
    text, usage, error, pieces = [], {}, None, 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or ():
                    delta = choice.get("delta") or {}
                    piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                    if piece:
                        pieces += 1
                        if first is None:
                            first = time.perf_counter()
                        text.append(piece)
                if cancel_after is not None and pieces >= cancel_after:
                    return {"cancelled": True, "text": "".join(text), "ttft": (first or time.perf_counter()) - started}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"[:300]
    finished = time.perf_counter()
    completion = int(usage.get("completion_tokens") or 0)
    decode_seconds = max(finished - (first or finished), 1e-9)
    return {"text": "".join(text), "error": error, "ttft": None if first is None else first - started,
            "wall": finished - started, "completion_tokens": completion,
            "decode_tok_s": (max(completion - 1, 0) / decode_seconds) if first else None}


def run_width(base, model, width, salt, max_tokens):
    items = [question(i, salt) for i in range(width)]
    results = [None] * width
    barrier = threading.Barrier(width)

    def work(i):
        barrier.wait()
        results[i] = stream_chat(base, model, items[i][0], max_tokens)

    started = time.perf_counter()
    threads = [threading.Thread(target=work, args=(i,)) for i in range(width)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started
    rows = []
    for (prompt, answer, word), result in zip(items, results):
        text = result["text"].replace(",", "")
        rows.append({"expected_sum": answer, "expected_word": word, "error": result["error"],
                     "sum_ok": answer in text, "word_ok": word in text.lower(),
                     "ttft": result["ttft"], "decode_tok_s": result["decode_tok_s"],
                     "completion_tokens": result["completion_tokens"], "text": result["text"][:400]})
    ttfts = sorted(r["ttft"] for r in rows if r["ttft"] is not None)
    decodes = [r["decode_tok_s"] for r in rows if r["decode_tok_s"]]
    tokens = sum(r["completion_tokens"] for r in rows)
    return {
        "width": width,
        "ok": sum(1 for r in rows if not r["error"] and r["sum_ok"] and r["word_ok"]),
        "errors": sum(1 for r in rows if r["error"]),
        "wrong": sum(1 for r in rows if not r["error"] and not (r["sum_ok"] and r["word_ok"])),
        "aggregate_tok_s": tokens / wall if wall else None,
        "per_stream_tok_s": statistics.median(decodes) if decodes else None,
        "ttft_p50": ttfts[len(ttfts) // 2] if ttfts else None,
        "ttft_p95": ttfts[min(len(ttfts) - 1, int(0.95 * len(ttfts)))] if ttfts else None,
        "wall_s": wall,
        "rows": rows,
    }


def cancellation(base, model, max_tokens):
    """Cancel one stream mid-batch; the rest and the next request must succeed."""
    items = [question(i, 97) for i in range(4)]
    results = [None] * 4

    def work(i):
        results[i] = stream_chat(base, model, items[i][0], max_tokens * 4 if i == 0 else max_tokens,
                                 cancel_after=3 if i == 0 else None)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    others_ok = all(not r.get("error") and a in r["text"].replace(",", "")
                    for r, (_, a, _) in zip(results[1:], items[1:]))
    time.sleep(2)
    after = stream_chat(base, model, question(5, 3)[0], max_tokens)
    healthy = status(base).get("state") == "ready"
    return {"cancelled_stream": bool(results[0].get("cancelled")), "others_ok": others_ok,
            "next_request_ok": not after["error"] and question(5, 3)[1] in after["text"].replace(",", ""),
            "server_ready": healthy}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    lanes = int((status(base).get("settings") or {}).get("max_lanes") or 16)
    widths = [w for w in WIDTHS if w <= max(lanes, 1)]
    report = {"schema": "mlx2.series-concurrency.v1", "model": args.model_id, "max_lanes": lanes, "widths": []}
    salt = random.Random(0).randint(1, 50)
    for width in widths:
        result = run_width(base, args.model_id, width, salt + width, args.max_tokens)
        report["widths"].append(result)
        print(f"width {width}: ok {result['ok']}/{width} errors {result['errors']} wrong {result['wrong']} "
              f"agg {result['aggregate_tok_s'] or 0:.1f} tok/s p50 ttft {result['ttft_p50'] or 0:.2f}s", flush=True)
    report["cancellation"] = cancellation(base, args.model_id, args.max_tokens)
    print(f"cancellation: {report['cancellation']}", flush=True)
    total = sum(w["width"] for w in report["widths"])
    ok = sum(w["ok"] for w in report["widths"])
    report["passed"] = (ok == total and all(report["cancellation"].values()))
    report["ok"], report["total"] = ok, total
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"SUMMARY ok={ok}/{total} cancellation={report['cancellation']} passed={report['passed']}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
