#!/usr/bin/env python3
"""Find a frozen reasoning miss and compare latent versus textual replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path


def _answer(text: str) -> str | None:
    matches = re.findall(r"(?im)^\s*FINAL\s*:\s*(.*?)\s*$", text)
    if not matches:
        return None
    value = " ".join(matches[-1].casefold().split()).strip()
    return value.rstrip(". ")


def _request(messages: list[dict], instructions: str) -> dict:
    if not messages or messages[0].get("role") != "user":
        raise ValueError("reasoning probe requires an initial user message")
    messages = [dict(item) for item in messages]
    messages[0]["content"] = instructions + "\n\n" + messages[0]["content"]
    return {
        "messages": messages,
        "enable_thinking": False,
        "reasoning_effort": "none",
    }


def _generate(adapter, request: dict, *, passes: int, max_tokens: int) -> dict:
    import mlx.core as mx

    from mlx2.runtime.recurrent_depth import (
        RecurrentDepthConfig,
        recurrent_depth_forward,
    )

    prompt_ids = list(adapter.prompt_tokens(request))
    caches = adapter.make_recurrent_depth_caches(passes)
    config = RecurrentDepthConfig(passes=passes)
    started = time.perf_counter()
    result = recurrent_depth_forward(
        adapter,
        mx.array([prompt_ids]),
        caches,
        config=config,
    )
    output = []
    eos = set(adapter.tokenizer.eos_token_ids)
    for _ in range(max_tokens):
        token = int(mx.argmax(result.logits[:, -1, :], axis=-1).item())
        if token in eos:
            break
        output.append(token)
        result = recurrent_depth_forward(
            adapter,
            mx.array([[token]]),
            caches,
            config=config,
        )
    mx.eval(result.logits)
    text = adapter.tokenizer.decode(output)
    mx.clear_cache()
    return {
        "passes": passes,
        "text": text,
        "answer": _answer(text),
        "tokens": len(output),
        "elapsed_seconds": time.perf_counter() - started,
        "receipt": result.receipt,
    }


def _score(result: dict, expected: list[str]) -> bool:
    normalized = {" ".join(value.casefold().split()).rstrip(". ") for value in expected}
    return result["answer"] in normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=192)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if not 16 <= args.max_tokens <= 512:
        parser.error("max-tokens must be in 16..512")

    import mlx.core as mx

    from mlx2.adapters.qwen35_9b import Qwen359BAdapter

    corpus_bytes = args.corpus.read_bytes()
    corpus = json.loads(corpus_bytes)
    if corpus.get("status") != "exploratory-frozen-before-search":
        raise ValueError("reasoning search corpus is not frozen")
    instructions = corpus["instructions"]
    adapter = Qwen359BAdapter(str(args.model))
    mx.eval(adapter.model.parameters())

    searched = []
    chosen = None
    baseline = None
    started = time.time()
    for case in corpus["cases"]:
        request = _request([{"role": "user", "content": case["problem"]}], instructions)
        result = _generate(adapter, request, passes=1, max_tokens=args.max_tokens)
        result["correct"] = _score(result, case["answers"])
        searched.append(
            {
                "id": case["id"],
                "expected": case["answers"],
                "baseline": result,
            }
        )
        print(
            json.dumps(
                {
                    "searched": len(searched),
                    "id": case["id"],
                    "answer": result["answer"],
                    "correct": result["correct"],
                }
            ),
            flush=True,
        )
        if not result["correct"]:
            chosen = case
            baseline = result
            break
    if chosen is None or baseline is None:
        raise RuntimeError("the frozen search corpus produced no one-pass miss")

    original_messages = [{"role": "user", "content": chosen["problem"]}]
    hidden_two = _generate(
        adapter,
        _request(original_messages, instructions),
        passes=2,
        max_tokens=args.max_tokens,
    )
    hidden_three = _generate(
        adapter,
        _request(original_messages, instructions),
        passes=3,
        max_tokens=args.max_tokens,
    )
    review_prompt = (
        "Let's double-check our work. Re-solve the original problem carefully, "
        "correct any mistake in the previous answer, and follow the original "
        "FINAL: answer format."
    )
    text_two_messages = [
        *original_messages,
        {"role": "assistant", "content": baseline["text"]},
        {"role": "user", "content": review_prompt},
    ]
    text_two = _generate(
        adapter,
        _request(text_two_messages, instructions),
        passes=1,
        max_tokens=args.max_tokens,
    )
    text_three_messages = [
        *text_two_messages,
        {"role": "assistant", "content": text_two["text"]},
        {
            "role": "user",
            "content": (
                "Let's double-check our work one more time. Independently verify "
                "the calculation and correct it if needed, then use the original "
                "FINAL: answer format."
            ),
        },
    ]
    text_three = _generate(
        adapter,
        _request(text_three_messages, instructions),
        passes=1,
        max_tokens=args.max_tokens,
    )
    arms = {
        "ordinary_one_pass": baseline,
        "hidden_two_pass": hidden_two,
        "hidden_three_pass": hidden_three,
        "text_replay_two_pass": text_two,
        "text_replay_three_pass": text_three,
    }
    for value in arms.values():
        value["correct"] = _score(value, chosen["answers"])

    result = {
        "schema": "mlx2.qwen35-recurrent-depth-reasoning-probe.v1",
        "status": "experimental-not-qualified",
        "model": str(args.model.resolve()),
        "corpus": str(args.corpus.resolve()),
        "corpus_file_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "searched_cases": searched,
        "chosen_case": chosen,
        "arms": arms,
        "elapsed_seconds": time.time() - started,
        "comparison": {
            name: {"answer": value["answer"], "correct": value["correct"]}
            for name, value in arms.items()
        },
        "limitations": (
            "The first miss is selected from a preregistered exploratory search "
            "order. One model, deterministic greedy decode, one case, no repeated "
            "trials, serving, batching, adapter training, or performance claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
