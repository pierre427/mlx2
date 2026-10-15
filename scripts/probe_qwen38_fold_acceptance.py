#!/usr/bin/env python3
"""Focused DFlash2 versus FOLD acceptance ladder; no performance claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _corpus_tokens(tokenizer, repo, needed):
    roots = (repo / "src", repo / "docs")
    tokens = []
    for root in roots:
        for path in sorted(root.rglob("*.py")) + sorted(root.rglob("*.md")):
            text = path.read_text(errors="replace")
            tokens.extend(
                tokenizer.encode(
                    f"\n\n# {path.relative_to(repo)}\n{text}",
                    add_special_tokens=False,
                )
            )
            if len(tokens) >= needed:
                return tokens[:needed]
    if not tokens:
        raise ValueError("qualification corpus tokenized to zero tokens")
    if len(tokens) < needed:
        tokens = (tokens * (needed // len(tokens) + 1))[:needed]
    return tokens


def _run(adapter, prompt, max_tokens):
    import mlx.core as mx

    batch = adapter.create_external_batch(completion_batch_size=1, stop_tokens=[])
    uid = batch.insert([prompt], max_tokens=[max_tokens])[0]
    output = []
    last_receipt = None
    while uid in batch.lanes:
        _, responses = batch.next()
        for response in responses:
            output.append(int(response.token))
            last_receipt = response.speculative_receipt
    stats = {
        key: int(batch.scheduler_stats.get(key, 0))
        for key in ("external_rounds", "proposed_tokens", "accepted_proposals")
    }
    rounds = max(stats["external_rounds"], 1)
    stats["mean_acceptance_length"] = 1 + stats["accepted_proposals"] / rounds
    batch.close()
    mx.clear_cache()
    return output, stats, last_receipt


def _set_fold(draft, enabled):
    policy = getattr(draft, "fold_policy", None)
    if policy is None:
        raise ValueError("the selected adapter has no FOLD policy")
    if enabled:
        for layer in draft.layers:
            attention = layer.self_attn
            if attention.is_sliding and policy.sliding_layers == "window":
                continue
            attention.fold_policy = policy
            attention.fold_calls = attention.fold_far_calls = 0
            attention.fold_max_context = 0
        return
    for layer in draft.layers:
        vars(layer.self_attn).pop("fold_policy", None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[32768, 65536, 131072]
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--out", required=True)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        parser.error("refusing Metal execution without --i-own-the-gpu")
    if any(value < 1 for value in args.contexts):
        parser.error("contexts must be positive")

    import mlx.core as mx

    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    policy = json.loads(Path(args.policy).read_text())
    print("loading adapter", flush=True)
    adapter = Qwen3827BAdapter(args.model, execution_policy=policy)
    print("adapter loaded", flush=True)
    from mlx2.runtime.drafters.fold import status as fold_status

    corpus = _corpus_tokens(
        adapter.tokenizer, Path(__file__).resolve().parents[1], max(args.contexts)
    )
    result = {
        "schema": "mlx2.qwen38-fold-acceptance-probe.v1",
        "model": args.model,
        "policy": policy,
        "mlx": mx.__version__,
        "performance_claim": False,
        "cells": [],
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({**result, "current_phase": "loaded"}, indent=2))
    for context in args.contexts:
        prompt = corpus[:context]
        output_path.write_text(
            json.dumps(
                {**result, "current_phase": f"fold:{context}"}, indent=2
            )
        )
        print(f"fold context={context}", flush=True)
        _set_fold(adapter.draft_model, True)
        candidate, fold_stats, receipt = _run(adapter, prompt, args.max_tokens)
        status = fold_status(adapter.draft_model)
        output_path.write_text(
            json.dumps(
                {**result, "current_phase": f"baseline:{context}"}, indent=2
            )
        )
        print(f"baseline context={context}", flush=True)
        _set_fold(adapter.draft_model, False)
        baseline, base_stats, _ = _run(adapter, prompt, args.max_tokens)
        cell = {
            "context_tokens": context,
            "baseline": base_stats,
            "fold": fold_stats,
            "target_output_equal": baseline == candidate,
            "fold_status": status,
            "route_receipt": receipt,
        }
        if not status["observed_used"] or not status["far_memory_calls"]:
            raise RuntimeError(f"FOLD did not engage at context {context}")
        if not cell["target_output_equal"]:
            raise RuntimeError(
                f"FOLD changed greedy target output at context {context}"
            )
        result["cells"].append(cell)
        output_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(cell, indent=2), flush=True)
    result["passed"] = True
    result["current_phase"] = "complete"
    output_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
