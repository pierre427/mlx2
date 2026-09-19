#!/usr/bin/env python3
"""Bounded warm long-context HTTP benchmark for ASPIRE viability probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8297")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-prompt-tokens", type=int, default=16_000)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.target_prompt_tokens < 1024 or args.max_tokens < 1:
        parser.error("target-prompt-tokens must be >= 1024 and max-tokens positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    unit = (
        "The archive records a deterministic event, its cause, its effect, "
        "and an independently verified consequence. "
    )
    unit_tokens = tokenizer.encode(unit, add_special_tokens=False)
    repeated = (unit_tokens * (args.target_prompt_tokens // len(unit_tokens) + 1))[
        : args.target_prompt_tokens
    ]
    prefix = tokenizer.decode(repeated)
    questions = [
        "Summarize the archive in exactly three numbered observations.",
        "Identify the recurring causal structure in exactly three numbered observations.",
        "Describe how the verification notes relate to effects in exactly three numbered observations.",
        "State the archive's central pattern in exactly three numbered observations.",
    ]
    prompts = [prefix + "\n\n" + question for question in questions]

    def status():
        with urlopen(args.url + "/v1/status", timeout=30) as response:
            return json.load(response)

    def call(prompt: str) -> dict:
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 260917943,
            "max_tokens": args.max_tokens,
            "enable_thinking": False,
        }
        started = time.monotonic()
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
        return {
            "wall_seconds": time.monotonic() - started,
            "usage": result["usage"],
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "receipt": result["mlx2"],
        }

    initial = status()
    warm = [call(prompt) for prompt in prompts]
    rows = []
    for width in args.widths:
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=width) as pool:
            measured = list(pool.map(call, prompts[:width]))
        elapsed = time.monotonic() - started
        rows.append(
            {
                "width": width,
                "elapsed_seconds": elapsed,
                "aggregate_tokens_per_second": args.max_tokens * width / elapsed,
                "matches_warmup_output": [
                    row["output_sha256"] == reference["output_sha256"]
                    for row, reference in zip(measured, warm)
                ],
                "requests": measured,
            }
        )
        print(
            f"width={width} aggregate={args.max_tokens * width / elapsed:.2f} tokens/s",
            flush=True,
        )
    final = status()
    report = {
        "schema": "mlx2.aspire-long-context-benchmark.v1",
        "target_content_tokens": args.target_prompt_tokens,
        "max_tokens": args.max_tokens,
        "initial": initial,
        "final": final,
        "warmup": warm,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
