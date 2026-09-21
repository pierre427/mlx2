#!/usr/bin/env python3
"""Probe a compact hyper-directory materialization of the 64K expert pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probe_qwen35_cross_expert_skillpack import (
    EXPERT_MODULES,
    QUESTIONS,
    _greedy_chunked,
    _score,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--prefill-step", type=int, default=2048)
    args = parser.parse_args()

    from mlx2.adapters.qwen35_9b import Qwen359BAdapter

    adapter = Qwen359BAdapter(str(args.model))
    materialized = (
        "Retrieved expert-method capsules from the 64K crossed-expert pack. "
        "Apply these methods, perform the arithmetic, and verify each result.\n\n"
        + "\n\n".join(EXPERT_MODULES)
    )
    materialized_tokens = len(
        adapter.tokenizer.encode(materialized, add_special_tokens=False)
    )
    prompt_ids = list(
        adapter.prompt_tokens(
            {
                "messages": [
                    {"role": "system", "content": materialized},
                    {"role": "user", "content": QUESTIONS},
                ],
                "enable_thinking": False,
                "reasoning_effort": "none",
            }
        )
    )
    text, prefill_seconds = _greedy_chunked(
        adapter, prompt_ids, args.max_tokens, args.prefill_step
    )
    result = {
        "schema": "mlx2.qwen35-retrieved-cross-expert-skillpack.v1",
        "status": "exploratory-not-qualified",
        "source_pack": "qwen35-9b-cross-expert-64k",
        "materialization": "ten deduplicated expert-method modules selected from the pack",
        "materialized_tokens": materialized_tokens,
        "prompt_tokens": len(prompt_ids),
        "prefill_seconds": prefill_seconds,
        "output": text,
        **_score(text),
        "limitations": [
            "one greedy sample on eight synthetic held-out tasks",
            "manual module selection approximates a perfect directory retriever",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "output"}, indent=2))


if __name__ == "__main__":
    main()
