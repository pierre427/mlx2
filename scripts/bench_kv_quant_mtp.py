#!/usr/bin/env python3
"""Serving A/B: exact vs approximate KV on the ordinary and self-MTP routes.

Arms (any subset, run interleaved: every round builds each arm's engine,
serves the same requests, closes it, so thermal/allocator drift spreads over
all arms; back-to-back A/Bs drift about 10 %):

* ``ordinary_exact``  ordinary route, exact KV
* ``ordinary_kv_q8``  ordinary route, kv_q8
* ``mtp_exact``       native self-MTP, exact KV
* ``mtp_kv_q8``       native self-MTP, kv_q8 on target planes, exact draft
                      cache (``compose_mtp``)

``--operation`` switches the quantized arms to kv_k8v4.  Every quantized arm
must show its mechanism (``approximate_kv.applied`` > 0, and for MTP
``mtp_lanes`` > 0 with quantized segmented attention calls > 0 and no
true-batched declines) or the harness refuses it.  Reported per arm: decode
tok/s (completion tokens / (elapsed - ttft)), TTFT, MTP acceptance
(draft_accepted / draft_proposed) and tokens per target cycle, and greedy
output identity against the exact arm of the same route.

Metal only: refuses without ``--i-own-the-gpu``; ``--dry-run`` prints the plan.
Engines run in qualification mode (candidate receipts).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARMS = ("ordinary_exact", "ordinary_kv_q8", "mtp_exact", "mtp_kv_q8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--i-own-the-gpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--operation", default="kv_q8", choices=("kv_q8", "kv_k8v4"))
    p.add_argument("--interleave", type=int, default=3, help="rounds over all arms")
    p.add_argument("--prompt-tokens", type=int, default=16384,
                   help="approximate prompt length (corpus characters / 4)")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--lanes", default="1,4", help="comma list of concurrent requests")
    p.add_argument("--max-context", type=int, default=65536)
    p.add_argument("--corpus", type=Path)
    p.add_argument("--out", type=Path)
    return p.parse_args(argv)


def arm_config(arm, operation):
    mtp = arm.startswith("mtp")
    if arm.endswith("exact"):
        return mtp, None
    policy = {"operation": operation, "enabled": True}
    if mtp:
        policy["compose_mtp"] = True
    return mtp, policy


def prompts(args, count):
    if args.corpus is not None:
        text = args.corpus.read_text(encoding="utf-8")
    else:
        text = "\n\n".join(p.read_text(encoding="utf-8")
                           for p in sorted((ROOT / "docs").glob("*.md")))
    body = (text * (1 + 4 * args.prompt_tokens // max(len(text), 1)))[: 4 * args.prompt_tokens]
    return [
        {
            "messages": [{
                "role": "user",
                "content": f"[{index}] Summarize the following document in detail.\n\n{body}",
            }],
            "max_tokens": args.max_tokens,
            "temperature": 0,
        }
        for index in range(count)
    ]


def collect(job):
    text = ""
    while True:
        event = job.events.get(timeout=3600)
        if "error" in event:
            raise RuntimeError(event["error"])
        if "delta" in event:
            text += event["delta"].get("content", "") or ""
        if "finish_reason" in event:
            return text, event["receipt"]


def run_arm(args, arm, lanes):
    from mlx2.runtime.segmented_self_mtp import segmented_self_mtp_stats
    from mlx2.serving import ServingEngine

    before = dict(segmented_self_mtp_stats())
    mtp, policy = arm_config(arm, args.operation)
    engine = ServingEngine(
        args.model, mtp=mtp, qualification_mode=True, approximate_kv=policy,
        max_lanes=max(lanes, 1), max_context=args.max_context,
    )
    try:
        if not engine.ready.wait(1800):
            raise RuntimeError(engine.error or "engine did not become ready")
        started = time.perf_counter()
        jobs = [engine.submit(request) for request in prompts(args, lanes)]
        results = [collect(job) for job in jobs]
        wall = time.perf_counter() - started
        time.sleep(1.5)  # status counters settle through the 1 s snapshot
        status = engine.status()
    finally:
        engine.close()
    receipts = [receipt for _, receipt in results]
    completion = sum(r.get("completion_tokens", 0) for r in receipts)
    decode_s = [
        max(r.get("elapsed_seconds", 0) - r.get("ttft_seconds", 0), 1e-9) for r in receipts
    ]
    stats = [((r.get("mtp") or {}).get("stats") or {}) for r in receipts]
    proposed = sum(s.get("draft_proposed", 0) for s in stats)
    accepted = sum(s.get("draft_accepted", 0) for s in stats)
    cycles = sum(s.get("cycles", 0) for s in stats)
    approx = status.get("approximate_kv") or {}
    # Process-global counters: take this arm's delta, not the running total.
    after = segmented_self_mtp_stats()
    segmented = {
        key: after[key] - before.get(key, 0)
        for key, value in after.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    record = {
        "arm": arm,
        "lanes": lanes,
        "wall_s": wall,
        "aggregate_tok_s": completion / wall,
        "per_request_decode_tok_s": [
            r.get("completion_tokens", 0) / s for r, s in zip(receipts, decode_s)
        ],
        "ttft_s": [r.get("ttft_seconds") for r in receipts],
        "prompt_tokens": [r.get("prompt_tokens") for r in receipts],
        "completion_tokens": completion,
        "mtp_acceptance": (accepted / proposed) if proposed else None,
        "tokens_per_cycle": (completion / cycles) if cycles else None,
        "approximate_kv": {k: approx.get(k) for k in (
            "applied", "declined", "mtp_lanes", "operation", "compose_mtp", "draft_cache")},
        "segmented": {k: segmented.get(k) for k in (
            "true_batched_engaged", "true_batched_declined", "b1_target_forwards",
            "quantized_kv_segmented_layers", "quantized_kv_segmented_attention_calls")},
        "outputs": [text for text, _ in results],
        "peak_rss_note": "see cpg_job wrapper memory samples",
    }
    check_mechanism(record, policy, mtp)
    return record


def check_mechanism(record, policy, mtp):
    """Refuse an arm whose mechanism did not run (a silent null)."""
    approx = record["approximate_kv"]
    if policy is None:
        if approx.get("applied"):
            raise RuntimeError(f"{record['arm']}: exact arm applied approximate KV")
        return
    if not approx.get("applied"):
        raise RuntimeError(f"{record['arm']}: approximate_kv.applied == 0")
    if mtp:
        seg = record["segmented"]
        if not approx.get("mtp_lanes"):
            raise RuntimeError(f"{record['arm']}: approximate_kv.mtp_lanes == 0")
        if not seg.get("quantized_kv_segmented_attention_calls"):
            raise RuntimeError(f"{record['arm']}: quantized segmented attention never ran")
        if seg.get("true_batched_declined"):
            raise RuntimeError(f"{record['arm']}: true-batched segmented MTP declined")


def summarize(records):
    summary = {}
    for record in records:
        key = f"{record['arm']}@{record['lanes']}"
        entry = summary.setdefault(key, {"aggregate_tok_s": [], "acceptance": []})
        entry["aggregate_tok_s"].append(record["aggregate_tok_s"])
        if record["mtp_acceptance"] is not None:
            entry["acceptance"].append(record["mtp_acceptance"])
    for entry in summary.values():
        entry["aggregate_tok_s_median"] = statistics.median(entry["aggregate_tok_s"])
        if entry["acceptance"]:
            entry["acceptance_median"] = statistics.median(entry["acceptance"])
    return summary


def main(argv=None):
    args = parse_args(argv)
    arms = args.arms.split(",")
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}")
    lanes = [int(x) for x in args.lanes.split(",")]
    plan = {"model": args.model, "arms": arms, "operation": args.operation,
            "interleave": args.interleave, "lanes": lanes,
            "prompt_tokens": args.prompt_tokens, "max_tokens": args.max_tokens}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.i_own_the_gpu:
        raise SystemExit("refusing Metal execution without --i-own-the-gpu")
    records = []
    for round_index in range(args.interleave):
        for width in lanes:
            for arm in arms:
                record = run_arm(args, arm, width)
                record["round"] = round_index
                records.append(record)
                print(json.dumps({k: v for k, v in record.items() if k != "outputs"}),
                      flush=True)
    # Greedy identity against the exact arm of the same route and width.
    for record in records:
        route = record["arm"].split("_")[0]
        reference = next((r for r in records if r["arm"] == f"{route}_exact"
                          and r["lanes"] == record["lanes"]
                          and r["round"] == record["round"]), None)
        record["matches_exact_route"] = (
            None if reference is None else record["outputs"] == reference["outputs"]
        )
    result = {"schema": "mlx2.kv-quant-mtp-bench.v1", "plan": plan,
              "summary": summarize(records), "records": records}
    text = json.dumps(result, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
