#!/usr/bin/env python3
"""In-process interleaved A/B: fixed self-MTP depth against the draft-loop gate.

One model load, one process.  Each arm is a ``BatchGenerator`` self-MTP
configuration; arms run in the order A B C ... per pair, over the same
chat-templated prompt set (thinking off, greedy), one request at a time.
Per request we time decode only (first emitted token to completion), so
prefill does not dilute the comparison, and keep the exact token ids so
greedy arms can be compared token for token.

Every gated request must carry a ``draft_loop`` receipt with
``observed_used``; every fixed request must carry none.  A missing
mechanism refuses the run rather than reporting a no-op as a result.

Refuses to touch Metal without ``--i-own-the-gpu``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mtp_confidence_gpu import PROMPTS, _install_lane  # noqa: E402


def parse_arm(text):
    """``name=num_draft``, ``name=num_draft:stage:threshold`` or
    ``name=num_draft:b1/b2/...:threshold`` (explicit stage boundaries)."""
    name, _, spec = text.partition("=")
    if spec.startswith("{"):
        # Full self-MTP config as JSON, e.g. {"num_draft": 3, "draft_loop": {...}}.
        return name, json.loads(spec)
    parts = spec.split(":")
    if not name or len(parts) not in (1, 3):
        raise argparse.ArgumentTypeError(f"bad arm {text!r}")
    config = {"num_draft": int(parts[0])}
    if len(parts) == 3:
        if "/" in parts[1]:
            stages = {"boundaries": [int(v) for v in parts[1].split("/")]}
        else:
            stages = {"stage": int(parts[1])}
        config["draft_loop"] = {**stages, "threshold": float(parts[2])}
    return name, config


def run_group(model, prompts, indices, config, max_tokens):
    """Decode ``indices`` together in one generator; return rows and engagement."""
    from mlx2.runtime import segmented_self_mtp as seg
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG
    import mlx.core as mx

    width = len(indices)
    before = dict(seg._STATS)
    gen = BatchGenerator(
        model, completion_batch_size=width, prefill_batch_size=width, prefill_step_size=2048,
        self_mtp={"persistent": True, "segment_aware_live_tip": True,
                  "segment_aware_cohort_size": width, **config},
    )
    uids = gen.insert([prompts[i] for i in indices], max_tokens=[max_tokens] * width,
                      lane_rngs=[LaneRNG(i) for i in indices],
                      self_mtp_configs=[{"sampling_temp": 0.0}] * width)
    tokens = {uid: [] for uid in uids}
    receipts, first, done = {}, None, set()
    while len(done) < width:
        _, responses = gen.next()
        for response in responses:
            if first is None:
                mx.synchronize()
                first = time.perf_counter()
            tokens[response.uid].append(int(response.token))
            if getattr(response, "mtp_receipt", None):
                receipts[response.uid] = response.mtp_receipt
            if response.finish_reason is not None:
                done.add(response.uid)
    mx.synchronize()
    wall = time.perf_counter() - first
    gen.close()
    mx.clear_cache()
    engaged = {k: seg._STATS[k] - before.get(k, 0) for k in seg._STATS
               if k.startswith("true_batched") and seg._STATS[k] != before.get(k, 0)}
    rows = []
    for uid, index in zip(uids, indices):
        receipt = receipts.get(uid) or {}
        stats = receipt.get("stats", {})
        rows.append({
            "prompt": index,
            "tokens": tokens[uid],
            "decode_tokens": len(tokens[uid]) - 1,
            # The group's wall time, shared evenly: the sum over a group's
            # rows is its wall time, so totals give aggregate ms/token.
            "decode_seconds": wall / width,
            "draft_loop": receipt.get("draft_loop"),
            "route": receipt.get("route"),
            "verify_span_hist": stats.get("verify_span_hist"),
            "true_batched": engaged,
        })
    return rows


def run_arm(model, tokenizer, prompts, config, max_tokens, width=1):
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG
    import mlx.core as mx

    if width > 1:
        rows = []
        for start in range(0, len(prompts), width):
            indices = list(range(start, min(start + width, len(prompts))))
            rows.extend(run_group(model, prompts, indices, config, max_tokens))
        return rows
    rows = []
    for index, prompt in enumerate(prompts):
        gen = BatchGenerator(
            model, completion_batch_size=1, prefill_batch_size=1, prefill_step_size=2048,
            self_mtp={"persistent": True, "segment_aware_live_tip": True,
                      "segment_aware_cohort_size": 1, **config},
        )
        gen.insert([prompt], max_tokens=[max_tokens], lane_rngs=[LaneRNG(index)],
                   self_mtp_configs=[{"sampling_temp": 0.0}])
        tokens, receipt, first, done = [], None, None, False
        while not done:
            _, responses = gen.next()
            for response in responses:
                if first is None:
                    mx.synchronize()
                    first = time.perf_counter()
                tokens.append(int(response.token))
                if getattr(response, "mtp_receipt", None):
                    receipt = response.mtp_receipt
                if response.finish_reason is not None:
                    done = True
        mx.synchronize()
        seconds = time.perf_counter() - first
        gen.close()
        mx.clear_cache()
        stats = (receipt or {}).get("stats", {})
        rows.append({
            "prompt": index,
            "tokens": tokens,
            "decode_tokens": len(tokens) - 1,
            "decode_seconds": seconds,
            "draft_loop": (receipt or {}).get("draft_loop"),
            "route": (receipt or {}).get("route"),
            "verify_span_hist": stats.get("verify_span_hist"),
        })
    return rows


def summarize(results, arms, baseline):
    out = {}
    for name in arms:
        runs = [r for r in results if r["arm"] == name]
        per_pair = [
            sum(x["decode_seconds"] for x in r["rows"]) / sum(x["decode_tokens"] for x in r["rows"])
            for r in runs
        ]
        loops = [x["draft_loop"] for r in runs for x in r["rows"] if x["draft_loop"]]
        out[name] = {
            "ms_per_token_pairs": [1e3 * v for v in per_pair],
            "ms_per_token_median": 1e3 * statistics.median(per_pair),
            "gate_decisions": sum(l["decisions"] for l in loops),
            "gate_extensions": sum(l["extensions"] for l in loops),
        }
    base = out[baseline]["ms_per_token_median"]
    reference = {x["prompt"]: x["tokens"] for r in results if r["arm"] == baseline for x in r["rows"]}
    for name in arms:
        out[name]["speedup_vs_baseline"] = base / out[name]["ms_per_token_median"]
        mismatched = set()
        first_divergence = []
        for r in results:
            if r["arm"] != name:
                continue
            for x in r["rows"]:
                ref = reference[x["prompt"]]
                if x["tokens"] != ref:
                    mismatched.add(x["prompt"])
                    at = next((i for i, (a, b) in enumerate(zip(x["tokens"], ref)) if a != b),
                              min(len(x["tokens"]), len(ref)))
                    first_divergence.append((x["prompt"], at))
        out[name]["prompts_differing_from_baseline"] = sorted(mismatched)
        out[name]["first_divergence"] = sorted(set(first_divergence))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", action="append", type=parse_arm, required=True,
                        help="name=num_draft[:stage:threshold]; the first arm is the baseline")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--width", type=int, default=1,
                        help="prompts decoded together per generator (cohort width)")
    parser.add_argument("--lane-matmul", choices=("off", "auto", "crossover", "exact"), default="off")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.i_own_the_gpu:
        raise SystemExit("refusing to run on Metal without --i-own-the-gpu")

    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx
    from mlx2.adapters.registry import resolve_adapter

    mx.set_cache_limit(4 << 30)
    adapter = resolve_adapter(args.model, mtp=True)(args.model)
    lane = _install_lane(args, adapter)
    tokenizer = adapter.tokenizer
    prompts = [
        list(tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True,
            enable_thinking=False, tokenize=True,
        ))
        for text in PROMPTS
    ]
    arms = dict(args.arm)
    names = list(arms)
    results = []
    # Warm-up: one short pass per arm, discarded.
    for name in names:
        run_arm(adapter.model, tokenizer, prompts[: args.width], arms[name], 32, width=args.width)
    for pair in range(args.pairs):
        for name in names:
            rows = run_arm(adapter.model, tokenizer, prompts, arms[name], args.max_tokens,
                           width=args.width)
            gated = "draft_loop" in arms[name]
            if gated and not all(r["draft_loop"] and r["draft_loop"]["observed_used"] for r in rows):
                raise SystemExit(f"REFUSED arm {name}: draft_loop not observed on every request")
            if not gated and any(r["draft_loop"] for r in rows):
                raise SystemExit(f"REFUSED arm {name}: unexpected draft_loop receipt")
            results.append({"pair": pair, "arm": name, "config": arms[name], "rows": rows})
            total = sum(r["decode_tokens"] for r in rows)
            secs = sum(r["decode_seconds"] for r in rows)
            print(json.dumps({"pair": pair, "arm": name, "tokens": total,
                              "ms_per_token": round(1e3 * secs / total, 3)}), flush=True)
    summary = summarize(results, names, names[0])
    args.out.write_text(json.dumps({
        "schema": "mlx2.dloop_ab.v1", "model": args.model, "lane_matmul": lane,
        "arms": arms, "pairs": args.pairs, "max_tokens": args.max_tokens,
        "summary": summary, "results": results,
    }, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
