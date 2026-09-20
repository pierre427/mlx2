"""GPU A/B for external-round snapshots: deepcopy (default) vs descriptor COW.

Loads Muse-Glimmer + DFlash2 once, then for each mode runs the same greedy
and sampled generations through ExternalDraftBatchGenerator and reports:
tokens (must be identical), per-lane round snapshot host time, active-memory
delta of one snapshot, and end-to-end tok/s.  Requires the GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx

from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter

HOME = Path.home() / "mlx-models"


def run(adapter, prompts, *, max_tokens, sampling, cow):
    os.environ["MLX_LM_EXTERNAL_ROUND_COW"] = "1" if cow else "0"
    batch = adapter.create_external_batch(
        completion_batch_size=len(prompts), prefill_step_size=2048
    )
    uids = batch.insert(
        prompts,
        max_tokens=[max_tokens] * len(prompts),
        sampling_configs=[dict(sampling)] * len(prompts),
    )
    freeze_ns = []
    real = batch._freeze_lane

    def timed(lane):
        start = time.perf_counter_ns()
        result = real(lane)
        freeze_ns.append(time.perf_counter_ns() - start)
        return result

    batch._freeze_lane = timed
    out = {uid: [] for uid in uids}
    started = None
    produced = 0
    while batch.lanes:
        _prompts, responses = batch.next()
        for response in responses:
            if started is None:
                started = time.perf_counter()
            out[response.uid].append(int(response.token))
            produced += 1
    elapsed = time.perf_counter() - (started or time.perf_counter())
    # One snapshot of a warm lane, measured for allocation.
    batch.insert([prompts[0]], max_tokens=[2])
    lane = next(iter(batch.lanes.values()))
    while lane.anchor is None:
        batch._prefill(lane)
    mx.synchronize()
    mx.clear_cache()
    before = mx.get_active_memory()
    snapshots = batch._snapshot_round([lane])
    mx.eval([a for c in lane.cache for a in c.state])
    alloc = mx.get_active_memory() - before
    del snapshots
    batch.close()
    return {
        "tokens": [out[uid] for uid in uids],
        "freeze_us_median": statistics.median(freeze_ns) / 1e3 if freeze_ns else None,
        "freeze_us_p90": (
            sorted(freeze_ns)[int(0.9 * (len(freeze_ns) - 1))] / 1e3 if freeze_ns else None
        ),
        "snapshots": len(freeze_ns),
        "snapshot_alloc_bytes": int(alloc),
        "lane_cache_bytes": int(sum(int(c.nbytes) for c in lane.cache)),
        "tok_per_s": produced / elapsed if elapsed > 0 else None,
        "cow_counters": {k: v for k, v in batch.scheduler_stats.items() if "cow" in k},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(HOME / "Muse-Glimmer-30B-mlx-4bit"))
    parser.add_argument("--draft", default=str(HOME / "Muse-Glimmer-30B-DFlash2"))
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    adapter = MuseGlimmerAdapter(
        args.model, execution_policy={"draft_model": args.draft, "num_draft": 4}
    )
    text = "Explain, step by step, how a hash map handles collisions. " * 400
    base = adapter.tokenizer.encode(text)[: args.prompt_tokens]
    prompts = [base[: len(base) - 7 * i] for i in range(args.width)]
    report = {}
    for name, sampling in (("greedy", {}), ("sampled", {"sampling_temp": 0.7, "top_p": 0.95})):
        results = {}
        for cow in (False, True, False, True):  # interleaved to spread thermal drift
            key = "cow" if cow else "deepcopy"
            result = run(adapter, prompts, max_tokens=args.max_tokens, sampling=sampling, cow=cow)
            results.setdefault(key, []).append(result)
        tokens_equal = all(
            r["tokens"] == results["deepcopy"][0]["tokens"]
            for r in results["deepcopy"] + results["cow"]
        )
        report[name] = {
            "tokens_identical": tokens_equal,
            **{
                key: {k: [r[k] for r in runs] for k in runs[0] if k != "tokens"}
                for key, runs in results.items()
            },
        }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
