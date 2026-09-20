#!/usr/bin/env python3
# ruff: noqa: EXE001
"""Ordinary-route quality A/B for North's normalization (rm06b).

Arm ``legacy``  -- ``MLX2_NORTH_NORM=legacy_layernorm``: mean-centred
                   nn.LayerNorm(eps=layer_norm_eps=1e-5), what mlx2 served
                   before 2026-09-20.
Arm ``fixed``   -- the default: nn.RMSNorm(eps=rms_norm_eps=1e-6), the norm HF
                   transformers' Cohere2Moe selects for this checkpoint.

This measures the ORDINARY (non-speculative) route only.  Speculation is a
separate question; the point here is whether North has been mis-served.

Metrics, per arm:
  * teacher-forced mean NLL / perplexity over fixed held-out passages (prose
    and code).  Deterministic, so the arms need no interleaving -- greedy and
    teacher-forced forwards do not drift the way timings do.
  * greedy continuations (temp 0) of a fixed prompt set, tokens and text.
  * the arm-vs-arm greedy common-prefix ratio, reported for the record.

PRE-REGISTERED criterion (stated before the first run, see rm06b/STATUS.md):
  GO if mean perplexity(fixed) <= perplexity(legacy), or is worse by less than
  1% relative.  If `fixed` is more than 1% worse on perplexity the norm change
  is wrong and must be reverted.  Greedy outputs ARE expected to change; that
  is a documented serving-behaviour change for North, not a failure.

Mechanism gate: each arm asserts the norm it asked for was actually built
(runtime.models.cohere2_moe.norm_counters), 49 decoder norms + 1 final norm.
An arm whose counter did not move is refused, not averaged.

Metal-only; refuses to run without ``--i-own-the-gpu``.  ``--dry-run`` prints
the plan without importing MLX.  Run under the lock wrapper.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

SCHEMA = "mlx2.rm06b-north-norm-ab.v1"
EXPECTED_NORMS = 50  # 49 decoder layers + the final norm

# Greedy prompts: 4 code, 4 prose.  Short answers so 128 tokens is enough to
# see whether the model is coherent at all.
PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Write a bash one-liner that counts lines in all .py files recursively.",
    "Implement binary search over a sorted list in Python. Code only.",
    "Write a SQL query returning the top 5 customers by total order value.",
    "Explain the difference between a process and a thread in two short paragraphs.",
    "What does the HTTP 429 status code mean and how should a client react?",
    "Summarize the CAP theorem in three bullet points.",
    "What is the difference between git merge and git rebase?",
]

# Held-out passages for teacher-forced scoring.  Written for this bench, so
# they are not a memorized corpus excerpt for either arm; both arms score the
# identical token sequence, which is what the comparison needs.
PASSAGES = {
    "prose": (
        "The admission controller keeps a running estimate of resident bytes per "
        "lane and refuses a new lane whenever the projected peak would exceed the "
        "reserve. It does not consult the scheduler, because the scheduler has no "
        "way to know how much of a lane's key-value state is shared with a prefix "
        "that another lane already pinned. Instead every lane reports its own "
        "footprint at the round boundary, and the controller reconciles those "
        "reports against the allocator's high-water mark. When the two disagree by "
        "more than a page the controller trusts the allocator and logs the drift, "
        "because a lane that under-reports is the failure mode that ends in an "
        "out-of-memory abort rather than a refused admission."
    ),
    "code": (
        "def rolling_median(values, window):\n"
        "    import bisect\n"
        "    if window <= 0:\n"
        "        raise ValueError('window must be positive')\n"
        "    ordered = []\n"
        "    out = []\n"
        "    for index, value in enumerate(values):\n"
        "        bisect.insort(ordered, value)\n"
        "        if index >= window:\n"
        "            stale = values[index - window]\n"
        "            ordered.pop(bisect.bisect_left(ordered, stale))\n"
        "        if index + 1 >= window:\n"
        "            middle = len(ordered) // 2\n"
        "            if len(ordered) % 2:\n"
        "                out.append(ordered[middle])\n"
        "            else:\n"
        "                out.append((ordered[middle - 1] + ordered[middle]) / 2)\n"
        "    return out\n"
    ),
}


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--arms", default="legacy,fixed", help="comma list: legacy,fixed")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--out", type=Path)
    p.add_argument("--i-own-the-gpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def plan(args):
    return {
        "schema": SCHEMA,
        "kind": "execution_plan",
        "target": str(args.target),
        "arms": args.arms.split(","),
        "will_execute": bool(args.i_own_the_gpu and not args.dry_run),
        "route": "ordinary (BatchGenerator, width 1, temperature 0)",
        "metrics": [
            "teacher-forced mean NLL and perplexity over PASSAGES",
            "greedy continuations of 8 fixed prompts",
            "arm-vs-arm greedy common-prefix ratio",
        ],
        "prompts": len(PROMPTS),
        "passages": sorted(PASSAGES),
        "max_tokens": args.max_tokens,
        "mechanism_gate": {
            "legacy": f"norm_counters legacy_override delta == {EXPECTED_NORMS}",
            "fixed": f"norm_counters rms delta == {EXPECTED_NORMS}",
            "arm_refused_if": "the arm's norm counter did not move by exactly that",
        },
        "go_criterion": {
            "perplexity": "mean ppl(fixed) <= ppl(legacy), or worse by < 1% relative",
            "greedy_divergence": "expected; documented as a North serving-behaviour change",
        },
    }


def _load(arm, target):
    """Build the adapter under ``arm`` and assert the norm actually engaged."""
    from mlx2.runtime.models.cohere2_moe import norm_counters

    if arm == "legacy":
        os.environ["MLX2_NORTH_NORM"] = "legacy_layernorm"
    elif arm == "fixed":
        os.environ.pop("MLX2_NORTH_NORM", None)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    before = norm_counters()
    from mlx2.adapters.north_mini_code import NorthMiniCodeAdapter

    started = time.perf_counter()
    adapter = NorthMiniCodeAdapter(str(target))
    after = norm_counters()
    delta = {k: after[k] - before[k] for k in after}
    key = "legacy_override" if arm == "legacy" else "rms"
    if delta[key] != EXPECTED_NORMS:
        raise RuntimeError(
            f"arm {arm!r} refused: norm counter {key} moved {delta[key]}, "
            f"expected {EXPECTED_NORMS} (counters={delta})"
        )
    if arm == "fixed" and delta["legacy_override"]:
        raise RuntimeError(f"arm 'fixed' refused: legacy override engaged ({delta})")
    return adapter, delta, time.perf_counter() - started


def _passage_nll(adapter, text, chunk=256):
    """Mean next-token NLL of ``text`` under the loaded model, teacher-forced."""
    import mlx.core as mx

    ids = adapter.tokenizer.encode(text)
    if len(ids) < 2:
        raise ValueError("passage too short to score")
    cache = adapter.model.make_cache()
    total, counted = 0.0, 0
    position = 0
    while position < len(ids) - 1:
        # Feed [position, position+step); score the next token of each row.
        step = min(chunk, len(ids) - 1 - position)
        window = mx.array([ids[position : position + step]])
        logits = adapter.model(window, cache).astype(mx.float32)
        targets = mx.array([ids[position + 1 : position + 1 + step]])
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        picked = mx.take_along_axis(logprobs, targets[..., None], axis=-1)
        total += float(-picked.sum())
        counted += step
        position += step
    mx.clear_cache()
    return {"tokens": counted, "nll": total / counted, "ppl": math.exp(total / counted)}


def _greedy(adapter, max_tokens):
    import mlx.core as mx

    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import make_sampler

    stops = [[int(t)] for t in adapter.tokenizer.eos_token_ids]
    results = []
    for text in PROMPTS:
        request = {
            "messages": [{"role": "user", "content": text}],
            "enable_thinking": False,
            "reasoning_effort": "none",
        }
        prompt = list(adapter.prompt_tokens(request))
        batch = BatchGenerator(
            adapter.model,
            stop_tokens=stops,
            completion_batch_size=1,
            sampler=make_sampler(0.0),
        )
        uids = batch.insert([prompt], max_tokens=[max_tokens])
        tokens, done = [], set()
        while len(done) < len(uids):
            _, responses = batch.next()
            for r in responses:
                tokens.append(int(r.token))
                if r.finish_reason:
                    done.add(r.uid)
        batch.close()
        mx.clear_cache()
        results.append(
            {
                "prompt": text,
                "tokens": tokens,
                "text": adapter.tokenizer.decode(tokens),
            }
        )
    return results


def _prefix_ratio(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n / max(len(a), len(b), 1)


def run(args):
    import mlx.core as mx

    record = {"schema": SCHEMA, "kind": "result", "plan": plan(args), "arms": {}}
    for arm in args.arms.split(","):
        adapter, counters, load_seconds = _load(arm, args.target)
        entry = {
            "norm_counters": counters,
            "load_seconds": round(load_seconds, 2),
            "passages": {},
        }
        for name, text in PASSAGES.items():
            entry["passages"][name] = _passage_nll(adapter, text)
        entry["mean_nll"] = sum(
            p["nll"] for p in entry["passages"].values()
        ) / len(entry["passages"])
        entry["mean_ppl"] = math.exp(entry["mean_nll"])
        entry["greedy"] = _greedy(adapter, args.max_tokens)
        record["arms"][arm] = entry
        del adapter
        mx.clear_cache()

    if {"legacy", "fixed"} <= set(record["arms"]):
        legacy, fixed = record["arms"]["legacy"], record["arms"]["fixed"]
        rel = (fixed["mean_ppl"] - legacy["mean_ppl"]) / legacy["mean_ppl"]
        ratios = [
            _prefix_ratio(a["tokens"], b["tokens"])
            for a, b in zip(legacy["greedy"], fixed["greedy"])
        ]
        record["comparison"] = {
            "ppl_legacy": legacy["mean_ppl"],
            "ppl_fixed": fixed["mean_ppl"],
            "relative_ppl_change": rel,
            "per_passage": {
                name: {
                    "legacy_ppl": legacy["passages"][name]["ppl"],
                    "fixed_ppl": fixed["passages"][name]["ppl"],
                }
                for name in PASSAGES
            },
            "greedy_prefix_ratios": ratios,
            "mean_greedy_prefix_ratio": sum(ratios) / len(ratios),
            "identical_greedy_outputs": sum(
                1
                for a, b in zip(legacy["greedy"], fixed["greedy"])
                if a["tokens"] == b["tokens"]
            ),
            "verdict": "GO" if rel < 0.01 else "NO-GO (fixed norm is worse)",
        }
    return record


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dry_run or not args.i_own_the_gpu:
        record = plan(args)
        if not args.dry_run:
            record["error"] = "refusing to touch Metal without --i-own-the-gpu"
    else:
        record = run(args)
    text = json.dumps(record, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0 if args.dry_run or args.i_own_the_gpu else 2


if __name__ == "__main__":
    sys.exit(main())
