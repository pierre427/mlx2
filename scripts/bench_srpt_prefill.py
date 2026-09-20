#!/usr/bin/env python3
"""Short-request TTFT behind a long prefill, for the SRPT prefill A/B (item 10).

Run against one already-started server per arm (policy off vs
``prefill_scheduling`` on).  One long streaming request starts first; after
``--short-delay`` seconds, ``--shorts`` short streaming requests arrive every
``--short-interval`` seconds.  Reports per-request TTFT/total wall time, the
greedy text of each request (for the cross-arm correctness gate) and the
scheduler counters from ``/v1/status``.

Both arms must run with ``--max-lanes >= max_bypass + 1`` (>= 4 for the
default cap).  The ordering window `PrefillOrder` chooses inside is
``--max-lanes`` -- every adapter sets ``segment_aware_cohort_size`` to it --
and below that width ``prefill_scheduling_bypass_forced`` can never advance,
so the on arm would measure the SRPT order without its fairness cap.  The
server refuses such a configuration at startup.  At exactly 4 the cap fires
only weakly (CPU: 2 forced bypasses in 16 rounds, against 4 at width 8), so
prefer a wider arm when the question is how much the cap is worth.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
from urllib.request import Request, urlopen

FILLER = (
    "The quick brown fox jumps over the lazy dog while the committee reviews "
    "section {i} of the maintenance log in exhaustive, repetitive detail. "
)


def stream(url, text, max_tokens, timeout):
    body = {
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "enable_thinking": False,
    }
    start = time.monotonic()
    first = None
    pieces = []
    finish = None
    request = Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            for choice in chunk.get("choices", ()):
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    if first is None:
                        first = time.monotonic() - start
                    pieces.append(delta)
                finish = choice.get("finish_reason") or finish
    return {
        "ttft_seconds": first,
        "wall_seconds": time.monotonic() - start,
        "finish_reason": finish,
        "text": "".join(pieces),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8285")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-repeats", type=int, default=900,
                        help="filler sentences in the long prompt (~30 tokens each)")
    parser.add_argument("--shorts", type=int, default=6)
    parser.add_argument("--short-delay", type=float, default=1.0)
    parser.add_argument("--short-interval", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()

    def status():
        with urlopen(args.url + "/v1/status", timeout=30) as response:
            return json.load(response)

    long_text = "".join(FILLER.format(i=i) for i in range(args.long_repeats))
    long_text += "\nSummarize the log above in one sentence."
    shorts = [
        f"In one sentence, what is {topic}?"
        for topic in ("a compiler", "a mutex", "a CPU cache", "TCP", "a B-tree",
                      "garbage collection", "a hash map", "DNS")
    ][: args.shorts]
    before = status()
    with ThreadPoolExecutor(max_workers=args.shorts + 1) as pool:
        long_future = pool.submit(
            stream, args.url, long_text, args.max_tokens, args.timeout
        )
        time.sleep(args.short_delay)
        short_futures = []
        for text in shorts:
            short_futures.append(
                pool.submit(stream, args.url, text, args.max_tokens, args.timeout)
            )
            time.sleep(args.short_interval)
        long_result = long_future.result()
        short_results = [future.result() for future in short_futures]
    after = status()
    ttfts = [r["ttft_seconds"] for r in short_results if r["ttft_seconds"] is not None]
    report = {
        "long": long_result,
        "shorts": [dict(r, prompt=p) for r, p in zip(short_results, shorts)],
        "short_ttft_p50": statistics.median(ttfts) if ttfts else None,
        "short_ttft_max": max(ttfts) if ttfts else None,
        "long_ttft": long_result["ttft_seconds"],
        "long_wall": long_result["wall_seconds"],
        "status_before": before,
        "status_after": after,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: report[k] for k in (
        "short_ttft_p50", "short_ttft_max", "long_ttft", "long_wall")}, indent=2))


if __name__ == "__main__":
    main()
