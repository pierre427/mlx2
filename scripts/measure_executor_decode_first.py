"""Bounded executor-level Metal A/B; no throughput qualification is implied.

Uses an existing local standard-decoder artifact, identical token inputs, and
alternating arms in one process. The first lead-token delay after a long-prompt
arrival isolates publication scheduling. This is below HTTP/SSE serving.
Run through scripts/run_with_gpu_locks.py with --i-own-the-gpu.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu or args.repetitions < 1:
        parser.error("explicit GPU ownership and positive repetitions required")
    import mlx.core as mx

    from mlx2.adapters.standard_decoder import StandardDecoderAdapter
    from mlx2.route_identity import bind_route_runtime
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.pld import PromptLookupBatchGenerator
    from mlx2.serving import runtime_identity

    mx.set_default_device(mx.gpu)
    adapter = StandardDecoderAdapter(str(args.model))
    model, tokenizer = adapter.model, adapter.tokenizer
    lead = tokenizer.encode(
        "Write a detailed explanation of how trees grow.\n", add_special_tokens=False
    )
    text = tokenizer.encode(
        "The blue bird watched the trees beside the quiet river.\n",
        add_special_tokens=False,
    )
    long_prompt = (text * (1024 // len(text) + 1))[:1024]
    arms = {
        "off": False,
        "order": {"shared_prefill_budget": False},
        "off_budget": False,
        "all": {"prefill_token_budget": 128},
    }
    results = []

    def run(kind, arm):
        cls = BatchGenerator if kind == "ordinary" else PromptLookupBatchGenerator
        kwargs = (
            {"prefill_batch_size": 2}
            if kind == "ordinary"
            else {"prompt_lookup": {"num_draft": 4, "ngram_min": 2, "ngram_max": 4}}
        )
        batch = cls(
            model,
            completion_batch_size=2,
            prefill_step_size=128 if arm == "off_budget" else 256,
            decode_first=arms[arm],
            **kwargs,
        )
        output, finishes, times = {}, set(), {}
        lead_deliveries = []
        start = time.perf_counter()
        uid = batch.insert([lead], max_tokens=[64])[0]
        output[uid] = []
        arrival = None
        first_lead = None
        peer = None
        try:
            for _ in range(1000):
                _, responses = batch.next()
                now = time.perf_counter()
                if any(response.uid == uid for response in responses):
                    lead_deliveries.append(now)
                for response in responses:
                    output.setdefault(response.uid, []).append(int(response.token))
                    times.setdefault(response.uid, now)
                    if (
                        arrival is not None
                        and response.uid == uid
                        and first_lead is None
                    ):
                        first_lead = now
                    if response.finish_reason:
                        finishes.add(response.uid)
                if arrival is None and len(output[uid]) >= 8:
                    arrival = time.perf_counter()
                    peer = batch.insert([long_prompt], max_tokens=[16])[0]
                    output[peer] = []
                if peer is not None and finishes == {uid, peer}:
                    break
            if finishes != {uid, peer} or first_lead is None:
                raise RuntimeError("bounded executor run did not finish")
            gaps = [
                (later - earlier) * 1000
                for earlier, later in pairwise(lead_deliveries)
                if arrival <= later <= times[peer]
            ]
            ordered_gaps = sorted(gaps)
            return {
                "executor": kind,
                "arm": arm,
                "seconds": now - start,
                "first_lead_after_arrival_ms": (first_lead - arrival) * 1000,
                "peer_ttft_ms": (times[peer] - arrival) * 1000,
                "lead_delivery_gap_ms_during_prefill": {
                    "count": len(gaps),
                    "p95": ordered_gaps[min(len(gaps) - 1, int(0.95 * len(gaps)))]
                    if gaps
                    else None,
                    "max": max(gaps) if gaps else None,
                },
                "lead_tokens": output[uid],
                "peer_tokens": output[peer],
                "scheduler": dict(batch.scheduler_stats),
            }
        finally:
            batch.close()
            mx.synchronize()
            mx.clear_cache()

    try:
        build = runtime_identity()
        runtime, scope = bind_route_runtime(build, adapter, "ordinary")
        # Every executor/arm warms the same kernels before measured runs.
        for kind in ("ordinary", "pld"):
            for arm in arms:
                run(kind, arm)
        for rep in range(args.repetitions):
            for kind in ("ordinary", "pld"):
                for arm in list(arms)[:: 1 if rep % 2 == 0 else -1]:
                    result = dict(run(kind, arm), repetition=rep)
                    results.append(result)
                    print(
                        json.dumps(
                            {
                                key: result[key]
                                for key in (
                                    "executor",
                                    "arm",
                                    "repetition",
                                    "first_lead_after_arrival_ms",
                                    "peer_ttft_ms",
                                )
                            }
                        ),
                        flush=True,
                    )
        summary = {}
        for kind in ("ordinary", "pld"):
            for arm in arms:
                reference_arm = "off_budget" if arm in {"all", "off_budget"} else "off"
                reference = next(
                    r
                    for r in results
                    if r["executor"] == kind and r["arm"] == reference_arm
                )
                rows = [r for r in results if r["executor"] == kind and r["arm"] == arm]
                summary[kind + ":" + arm] = {
                    "reference_arm": reference_arm,
                    "exact_tokens_vs_reference": all(
                        r["lead_tokens"] == reference["lead_tokens"]
                        and r["peer_tokens"] == reference["peer_tokens"]
                        for r in rows
                    ),
                    "median_first_lead_after_arrival_ms": statistics.median(
                        r["first_lead_after_arrival_ms"] for r in rows
                    ),
                    "median_peer_ttft_ms": statistics.median(
                        r["peer_ttft_ms"] for r in rows
                    ),
                }
        report = {
            "schema": "mlx2.executor-decode-first-ab.v1",
            "scope": "executor-level real-artifact smoke and bounded timing; not route qualification",
            "host": platform.node(),
            "device": str(mx.default_device()),
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                )
            ),
            "runtime": runtime,
            "build_runtime": build,
            "runtime_source_scope": scope,
            "artifact": adapter.identity,
            "prompt_tokens": len(lead),
            "peer_prompt_tokens": len(long_prompt),
            "summary": summary,
            "runs": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        if not all(row["exact_tokens_vs_reference"] for row in summary.values()):
            raise SystemExit("token divergence: inspect output; no equivalence claim")
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
