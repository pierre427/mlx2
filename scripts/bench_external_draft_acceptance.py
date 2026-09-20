#!/usr/bin/env python3
# ruff: noqa: EXE001
"""External-draft acceptance and decode tok/s vs the ordinary route (rm06).

Families: ``north`` (Cohere EAGLE chain drafter) and ``laguna`` (causal
DFlash block drafter).  One target load serves every arm: the ordinary arm is
the production ``BatchGenerator`` on the adapter's model; external arms use
``adapter.create_external_batch``.  Arms are interleaved (ord, K1, ord, K2,
...) per repeat because back-to-back A/Bs drift about 10%.

Every external arm must show ``external_rounds > 0`` and
``proposed_tokens > 0`` or the arm is refused (recorded, not averaged).  Greedy
external output is compared with greedy ordinary output per prompt (exact
count and mean common-prefix ratio; GPU verify and decode matmuls can round a
near-tie differently).

Metal-only; refuses to run without ``--i-own-the-gpu``.  ``--dry-run`` prints
the plan without importing MLX.  Run under the lock wrapper:

    cpg_job.py run --label rm06-<family> --out <json> --lock -- <python> \
      scripts/bench_external_draft_acceptance.py --family north ... --i-own-the-gpu
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

SCHEMA = "mlx2.rm06-external-draft-acceptance.v1"

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain the difference between a process and a thread in two short paragraphs.",
    "Refactor this loop into a list comprehension: out=[]\nfor x in xs:\n    if x%2==0:\n        out.append(x*x)",
    "Write a SQL query returning the top 5 customers by total order value.",
    "Implement binary search in Rust and explain its complexity.",
    "What does the HTTP 429 status code mean and how should a client react?",
    "Write a bash one-liner that counts lines in all .py files recursively.",
    "Describe how a hash map handles collisions with open addressing.",
    "Write a TypeScript type for a JSON value.",
    "Summarize the CAP theorem in three bullet points.",
    "Write a Python dataclass for a 2D point with a distance method.",
    "Explain what a race condition is with a short example.",
    "Write a regular expression that matches an ISO 8601 date (YYYY-MM-DD).",
    "Implement a stack using two queues in Python.",
    "What is the difference between git merge and git rebase?",
    "Write a C function that reverses a string in place.",
]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", choices=("north", "laguna"))
    p.add_argument("--target", type=Path)
    p.add_argument("--draft", type=Path)
    p.add_argument("--merge", nargs="+", type=Path, help="merge split result JSONs (CPU only)")
    p.add_argument("--repeat-offset", type=int, default=0, help="label repeats from this index (split runs)")
    p.add_argument("--num-draft", default="3", help="comma list of proposal depths")
    p.add_argument("--prompts", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--widths", default="1,4", help="comma list of lane counts")
    p.add_argument("--temperatures", default="0,0.7")
    p.add_argument("--thinking", action="store_true", help="chat template with thinking on")
    p.add_argument("--out", type=Path)
    p.add_argument("--i-own-the-gpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def plan(args):
    depths = [int(v) for v in args.num_draft.split(",")]
    widths = [int(v) for v in args.widths.split(",")]
    temps = [float(v) for v in args.temperatures.split(",")]
    order = []
    for repeat in range(args.repeat_offset, args.repeat_offset + args.repeats):
        for temp in temps:
            for width in widths:
                for depth in depths:
                    order += [("ordinary", 0, width, temp, repeat), ("external", depth, width, temp, repeat)]
    return {
        "schema": SCHEMA, "kind": "execution_plan", "family": args.family,
        "target": str(args.target), "draft": str(args.draft),
        "will_execute": bool(args.i_own_the_gpu and not args.dry_run),
        "arms": len(order), "prompts": args.prompts, "max_tokens": args.max_tokens,
        "depths": depths, "widths": widths, "temperatures": temps,
        "interleaving": "ordinary/external alternating per (repeat, temp, width, depth)",
        "go_criterion": {
            "greedy_parity": "mean greedy common-prefix ratio vs ordinary >= 0.9 (exact count also reported)",
            "b1_speedup_min": 1.25, "b4_speedup_min": 1.0,
            "arm_refused_if": "external_rounds == 0 or proposed_tokens == 0",
        },
        "order": order,
    }


def _adapter(args, depth):
    policy = {"draft_model": str(args.draft), "num_draft": depth}
    if args.family == "north":
        from mlx2.adapters.north_mini_code import NorthMiniCodeAdapter as Adapter
    else:
        from mlx2.adapters.laguna_xs21 import LagunaXS21Adapter as Adapter
    return Adapter(str(args.target), execution_policy=policy)


def _prompt_ids(adapter, count, thinking):
    ids = []
    for text in (PROMPTS * ((count // len(PROMPTS)) + 1))[:count]:
        request = {"messages": [{"role": "user", "content": text}],
                   "enable_thinking": thinking, "reasoning_effort": "high" if thinking else "none"}
        ids.append(list(adapter.prompt_tokens(request)))
    return ids


def _stops(adapter):
    return [[int(t)] for t in adapter.tokenizer.eos_token_ids]


def _run_ordinary(adapter, prompts, width, temp, max_tokens):
    import mlx.core as mx

    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import make_sampler

    outputs, decode_tokens, decode_seconds = {}, 0, 0.0
    for start in range(0, len(prompts), width):
        chunk = prompts[start:start + width]
        batch = BatchGenerator(adapter.model, stop_tokens=_stops(adapter), completion_batch_size=width,
                               sampler=make_sampler(temp))
        uids = batch.insert(chunk, max_tokens=[max_tokens] * len(chunk))
        index = {uid: start + i for i, uid in enumerate(uids)}
        first, done = None, set()
        while len(done) < len(uids):
            _, responses = batch.next()
            now = time.perf_counter()
            for r in responses:
                outputs.setdefault(index[r.uid], []).append(int(r.token))
                if first is None:
                    first = now
                else:
                    decode_tokens += 1
                if r.finish_reason:
                    done.add(r.uid)
        decode_seconds += time.perf_counter() - (first or time.perf_counter())
        batch.close()
        mx.clear_cache()
    return outputs, decode_tokens / max(decode_seconds, 1e-9), {}


def _run_external(adapter, prompts, width, temp, max_tokens):
    import mlx.core as mx

    outputs, decode_tokens, decode_seconds = {}, 0, 0.0
    totals = {"external_rounds": 0, "proposed_tokens": 0, "accepted_proposals": 0}
    timings = {}
    for start in range(0, len(prompts), width):
        chunk = prompts[start:start + width]
        batch = adapter.create_external_batch(completion_batch_size=width, stop_tokens=_stops(adapter))
        uids = batch.insert(chunk, max_tokens=[max_tokens] * len(chunk),
                            sampling_configs=[{"sampling_temp": temp}] * len(chunk))
        index = {uid: start + i for i, uid in enumerate(uids)}
        first = None
        while batch.lanes:
            _, responses = batch.next()
            now = time.perf_counter()
            for r in responses:
                outputs.setdefault(index[r.uid], []).append(int(r.token))
                if first is None:
                    first = now
                else:
                    decode_tokens += 1
        decode_seconds += time.perf_counter() - (first or time.perf_counter())
        for key in totals:
            totals[key] += int(batch.scheduler_stats.get(key, 0))
        # Present only with MLX2_EXTERNAL_ROUND_TIMING=1 (host-time attribution).
        for phase, seconds in getattr(batch, "round_times", {}).items():
            timings[phase] = timings.get(phase, 0.0) + seconds
        batch.close()
        mx.clear_cache()
    rounds = max(totals["external_rounds"], 1)
    totals["mean_acceptance_length"] = 1 + totals["accepted_proposals"] / rounds
    totals["draft_stats"] = dict(getattr(adapter.draft_model, "stats", {}))
    if timings:
        totals["round_times_ms"] = {k: 1000 * v / rounds for k, v in timings.items()}
    return outputs, decode_tokens / max(decode_seconds, 1e-9), totals


def summarize(arms, refused):
    """Per-(width, temp, depth) medians and the pre-registered verdict.

    Shared by a single run and by ``--merge`` over split runs (each split job
    holds the GPU for under ~40 minutes; the verdict is computed over all
    arms of all splits exactly as if they had run in one job).
    """
    summary = {}
    for arm in arms:
        slot = summary.setdefault((arm["width"], arm["temperature"]), {"ordinary": [], "external": {}})
        if arm["kind"] == "ordinary":
            slot["ordinary"].append(arm["decode_tok_s"])
        else:
            slot["external"].setdefault(arm["num_draft"], []).append(arm)
    report = []
    for (width, temp), slot in sorted(summary.items()):
        base = statistics.median(slot["ordinary"]) if slot["ordinary"] else None
        for depth, group in sorted(slot["external"].items()):
            tps = statistics.median(a["decode_tok_s"] for a in group)
            report.append({
                "width": width, "temperature": temp, "num_draft": depth,
                "ordinary_tok_s": base, "external_tok_s": tps,
                "speedup": (tps / base) if base else None,
                "mean_acceptance_length": statistics.median(a["mechanism"]["mean_acceptance_length"] for a in group),
                "greedy_parity": [a.get("greedy_parity") for a in group],
                "greedy_prefix_ratio": [a.get("greedy_prefix_ratio") for a in group],
                "ordinary_spread": (max(slot["ordinary"]) - min(slot["ordinary"])) / base if base else None,
            })
    b1 = [r for r in report if r["width"] == 1 and r["temperature"] == 0]
    b4 = [r for r in report if r["width"] == 4 and r["temperature"] == 0]
    ratios = [x for r in report if r["temperature"] == 0 for x in r["greedy_prefix_ratio"] if x is not None]
    parity = bool(ratios) and statistics.mean(ratios) >= 0.9
    best_b1 = max((r["speedup"] or 0 for r in b1), default=0)
    best_b4 = max((r["speedup"] or 0 for r in b4), default=0)
    verdict = {
        "greedy_parity": parity, "best_b1_speedup": best_b1, "best_b4_speedup": best_b4,
        "go": bool(parity and best_b1 >= 1.25 and (not b4 or best_b4 >= 1.0) and not refused),
    }
    return {"summary": report, "verdict": verdict}


def merge(paths, out):
    arms, refused, plans = [], [], []
    for path in paths:
        data = json.loads(Path(path).read_text())
        if data.get("schema") != SCHEMA:
            raise SystemExit(f"{path}: not a {SCHEMA} result")
        arms += data["arms"]
        refused += data["refused"]
        plans.append({k: v for k, v in data["plan"].items() if k != "order"})
    families = {p["family"] for p in plans}
    if len(families) != 1:
        raise SystemExit(f"refusing to merge families {sorted(families)}")
    results = {"schema": SCHEMA, "merged_from": [str(p) for p in paths], "plans": plans,
               "arms": arms, "refused": refused}
    results.update(summarize(arms, refused))
    text = json.dumps(results, indent=1, default=str)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(json.dumps(results["verdict"], indent=1))
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.merge:
        return merge(args.merge, args.out)
    if not (args.family and args.target and args.draft):
        parser.error("--family, --target and --draft are required unless --merge")
    execution = plan(args)
    if args.dry_run or not args.i_own_the_gpu:
        print(json.dumps({k: v for k, v in execution.items() if k != "order"}, indent=1))
        if not args.dry_run:
            print("refusing: Metal execution requires --i-own-the-gpu", file=sys.stderr)
            return 2
        return 0
    results = {"schema": SCHEMA, "plan": execution, "arms": [], "refused": []}
    # One target load: depth is a per-batch knob read by create_external_batch.
    adapter = _adapter(args, max(execution["depths"]))
    greedy_ordinary = {}
    prompts = _prompt_ids(adapter, args.prompts, args.thinking)
    for kind, depth, width, temp, repeat in execution["order"]:
        if kind == "external":
            adapter.external_policy["num_draft"] = depth
        runner = _run_external if kind == "external" else _run_ordinary
        outputs, tps, mechanism = runner(adapter, prompts, width, temp, args.max_tokens)
        arm = {"kind": kind, "num_draft": depth, "width": width, "temperature": temp,
               "repeat": repeat, "decode_tok_s": tps, "mechanism": mechanism}
        if kind == "ordinary" and temp == 0:
            greedy_ordinary.setdefault(width, outputs)
        if kind == "external":
            if mechanism["external_rounds"] == 0 or mechanism["proposed_tokens"] == 0:
                results["refused"].append({**arm, "reason": "mechanism counter is zero"})
                continue
            if temp == 0 and width in greedy_ordinary:
                reference = greedy_ordinary[width]
                arm["greedy_parity"] = sum(outputs[i] == reference.get(i) for i in outputs)
                arm["greedy_prompts"] = len(outputs)
                ratios = []
                for i, got in outputs.items():
                    ref = reference.get(i, [])
                    common = 0
                    for x, y in zip(got, ref):
                        if x != y:
                            break
                        common += 1
                    ratios.append(common / max(1, min(len(got), len(ref))))
                # GPU verify (M=K+1) and decode (M=1) matmuls may round
                # differently; a near-tie flip is numerics, not a law error.
                arm["greedy_prefix_ratio"] = statistics.mean(ratios) if ratios else None
        results["arms"].append(arm)
    results.update(summarize(results["arms"], results["refused"]))
    adapter.close()
    text = json.dumps(results, indent=1, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(json.dumps(results["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
