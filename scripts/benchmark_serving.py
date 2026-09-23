#!/usr/bin/env python3
"""Bounded warm HTTP throughput measurements with per-request route receipts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time
from urllib.request import Request, urlopen


def _load_width_reader():
    """Use the qualifier's width reader so every route is read one way.

    Speculative lanes (prompt lookup, external draft) report their width in
    ``speculation.target_width``; ``ordinary_compute_width`` is None for them.
    """
    spec = importlib.util.spec_from_file_location(
        "_mlx2_benchmark_qualify_serving",
        Path(__file__).with_name("qualify_serving.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.observed_compute_widths


observed_compute_widths = _load_width_reader()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8285")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.rounds < 1 or args.max_tokens < 1 or args.timeout <= 0:
        parser.error("rounds, max-tokens and timeout must be positive")
    if not args.widths or len(set(args.widths)) != len(args.widths) or any(w not in {1, 2, 4} for w in args.widths):
        parser.error("widths must be unique values from 1, 2, 4")

    def status():
        with urlopen(args.url + "/v1/status", timeout=30) as response:
            return json.load(response)

    prompts = [
        "Explain how a compiler works, in numbered sections.",
        "Explain how a database transaction works, in numbered sections.",
        "Explain how a CPU cache works, in numbered sections.",
        "Explain how a network router works, in numbered sections.",
    ]

    def request(text):
        body = {
            "messages": [{"role": "user", "content": text}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "enable_thinking": False,
        }
        start = time.monotonic()
        with urlopen(
            Request(
                args.url + "/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=args.timeout,
        ) as response:
            result = json.load(response)
        output = result["choices"][0]["message"]["content"]
        assert output and result["usage"]["completion_tokens"] == args.max_tokens
        return {
            "wall_seconds": time.monotonic() - start,
            "usage": result["usage"],
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "receipt": result["mlx2"],
        }

    initial = status()
    assert initial["healthy"] and not initial["inflight"]
    assert max(args.widths) <= initial["max_lanes"]
    warm = {text: request(text) for text in prompts}
    rows = []
    for run in range(args.rounds):
        for width in args.widths if run % 2 == 0 else list(reversed(args.widths)):
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=width) as pool:
                requests = list(pool.map(request, prompts[:width]))
            elapsed = time.monotonic() - start
            observed = sorted({value for row in requests
                               for value in observed_compute_widths(row["receipt"])})
            assert observed and max(observed) == width, (
                f"requested B{width}, observed widths {observed}"
            )
            assert all(row["receipt"]["cached_tokens"] > 0 for row in requests), "warm benchmark missed prefix cache"
            rows.append(
                {
                    "round": run,
                    "width": width,
                    "observed_compute_widths": observed,
                    "matches_warmup_output": [row["output_sha256"] == warm[text]["output_sha256"]
                                              for text, row in zip(prompts[:width], requests)],
                    "elapsed_seconds": elapsed,
                    "aggregate_tokens_per_second": args.max_tokens * width / elapsed,
                    "requests": requests,
                }
            )
            print(
                f"round={run} width={width} aggregate={args.max_tokens * width / elapsed:.2f} tokens/s",
                flush=True,
            )
    final = status()
    assert final["healthy"] and not final["inflight"]
    for name in ("runtime", "settings", "artifact", "profile"):
        assert initial[name] == final[name], f"{name} changed during benchmark"
    from mlx2.serving import runtime_identity
    assert initial["runtime"] == runtime_identity(), "server code differs from current checkout"
    report = {
        "schema": "mlx2.http-benchmark.v1",
        "initial": initial,
        "final": final,
        "max_tokens": args.max_tokens,
        "warmup": warm,
        "rows": rows,
        "summary": {
            str(width): {
                "median_aggregate_tokens_per_second": statistics.median(
                    r["aggregate_tokens_per_second"]
                    for r in rows
                    if r["width"] == width
                )
            }
            for width in args.widths
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
