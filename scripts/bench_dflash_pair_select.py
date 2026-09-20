"""GPU A/B: DFlash2 sequential host selection vs batched pairwise selection.

Loads the Muse target + DFlash2 drafter once, then
1. micro: the same draft inputs through ``draft_distributions`` (host) and
   ``propose_block`` (batched) at several widths; tokens must be identical and
   proposal laws within tolerance; reports median draft-step latency.
2. e2e: the external executor on identical prompts/seeds in both modes;
   emitted tokens, RNG draws and receipts must be identical; reports tok/s.

Usage (GPU; nothing else on Metal):
  PYTHONPATH=src .venv/bin/python scripts/bench_dflash_pair_select.py \
    --model ~/mlx-models/Muse-Glimmer-30B-mlx-4bit \
    --draft ~/mlx-models/Muse-Glimmer-30B-DFlash2 --out /tmp/pair-select.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

PROMPTS = [
    "Explain how a hash map handles collisions, with a short Python example.",
    "Write a haiku about autumn rain, then explain its imagery.",
    "List five prime numbers greater than 100 and show how you checked each.",
    "Summarize the causes of the French Revolution in three paragraphs.",
    "Write a bash one-liner that counts unique IP addresses in an access log.",
    "Describe the difference between TCP and UDP for a beginner.",
    "Translate 'the quick brown fox jumps over the lazy dog' into French and German.",
    "Give a step-by-step derivation of the quadratic formula.",
]


def load(model, draft, num_draft):
    from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter

    adapter = MuseGlimmerAdapter(
        str(Path(model).expanduser()),
        execution_policy={"draft_model": str(Path(draft).expanduser()), "num_draft": num_draft},
    )
    return adapter


def encode(adapter, text):
    return adapter.prompt_tokens({"messages": [{"role": "user", "content": text}]})


def micro(adapter, widths, num_draft, trials, temps):
    from mlx2.runtime.speculative_sampling import RequestRNG

    model, draft = adapter.model, adapter.draft_model
    layers = list(draft.config.target_layer_ids)
    results = []
    for width in widths:
        prompts = [encode(adapter, PROMPTS[i % len(PROMPTS)]) for i in range(width)]
        # Equal-length tails form one draft group, as the executor groups them.
        length = min(len(p) for p in prompts) - 1
        hidden = mx.concatenate([model.prefill_body(mx.array([p[:length]]), model.make_cache(), layers) for p in prompts])
        anchors = [p[length] for p in prompts]
        mx.eval(hidden)
        row_temps = [temps[i % len(temps)] for i in range(width)]
        host_ms, batched_ms, max_diff, mismatches = [], [], 0.0, 0
        for trial in range(trials):
            seeds = [trial * 1000 + row for row in range(width)]
            host_rngs = [RequestRNG(s) for s in seeds]
            start = time.perf_counter()
            tokens, laws = draft.draft_distributions(anchors, hidden, draft.batch_caches([draft.make_cache() for _ in prompts]), num_draft, host_rngs, row_temps)
            host_ms.append((time.perf_counter() - start) * 1e3)
            rngs = [RequestRNG(s) for s in seeds]
            start = time.perf_counter()
            uniforms = [[r.uniform() for _ in range(num_draft)] for r in rngs]
            block = draft.propose_block(anchors, hidden, draft.batch_caches([draft.make_cache() for _ in prompts]), num_draft, uniforms, row_temps)
            batched_tokens = block.token_lists()
            dense = block.dense_laws(draft.config.vocab_size)
            batched_ms.append((time.perf_counter() - start) * 1e3)
            mismatches += int(batched_tokens != tokens)
            mismatches += int([r.snapshot() for r in rngs] != [r.snapshot() for r in host_rngs])
            for row in range(width):
                for position in range(num_draft):
                    max_diff = max(max_diff, float(np.max(np.abs(dense[row][position] - laws[row][position]))))
        results.append({
            "width": width,
            "num_draft": num_draft,
            "host_ms_median": statistics.median(host_ms[1:] or host_ms),
            "batched_ms_median": statistics.median(batched_ms[1:] or batched_ms),
            "token_or_rng_mismatches": mismatches,
            "max_abs_q_diff": max_diff,
        })
        print(json.dumps(results[-1]), flush=True)
    return results


def e2e(adapter, num_draft, width, max_tokens, temp):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG

    prompts = [encode(adapter, PROMPTS[i % len(PROMPTS)]) for i in range(width)]
    stops = [[t] for t in sorted(adapter.tokenizer.eos_token_ids)]
    runs = {}
    for mode in ("host", "batched", "host", "batched"):  # warm, then measure
        b = ExternalDraftBatchGenerator(
            adapter.model, draft_model=adapter.draft_model, binding="bench",
            completion_batch_size=width, num_draft=num_draft, stop_tokens=stops,
            pairwise_selection=mode,
        )
        b.insert(prompts, max_tokens=[max_tokens] * width,
                 sampling_configs=[{"sampling_temp": temp}] * width,
                 lane_rngs=[LaneRNG(1234 + i) for i in range(width)])
        lanes = list(b.lanes.values())
        output, receipts, emitted = {}, {}, 0
        start = time.perf_counter()
        while b.lanes:
            _, responses = b.next()
            for r in responses:
                output.setdefault(r.uid, []).append(r.token)
                receipts[r.uid] = r.speculative_receipt
                emitted += 1
        elapsed = time.perf_counter() - start
        b.close()
        runs[mode] = {
            "output": output,
            "receipts": receipts,
            "draws": [lane.rng.draws for lane in lanes],
            "tok_s": emitted / elapsed,
            "acceptance": b.scheduler_stats["accepted_proposals"] / max(1, b.scheduler_stats["proposed_tokens"]),
        }
    host, batched = runs["host"], runs["batched"]
    result = {
        "width": width, "temp": temp, "num_draft": num_draft,
        "identical_tokens": host["output"] == batched["output"],
        "identical_receipts": host["receipts"] == batched["receipts"],
        "identical_rng_draws": host["draws"] == batched["draws"],
        "host_tok_s": host["tok_s"], "batched_tok_s": batched["tok_s"],
        "speedup": batched["tok_s"] / host["tok_s"],
        "host_acceptance": host["acceptance"], "batched_acceptance": batched["acceptance"],
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/mlx-models/Muse-Glimmer-30B-mlx-4bit")
    parser.add_argument("--draft", default="~/mlx-models/Muse-Glimmer-30B-DFlash2")
    parser.add_argument("--num-draft", type=int, default=4)
    parser.add_argument("--widths", default="1,4,8")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    adapter = load(args.model, args.draft, args.num_draft)
    widths = [int(w) for w in args.widths.split(",")]
    report = {
        "micro": micro(adapter, widths, args.num_draft, args.trials, [0.0, 0.7, 1.0]),
        "e2e": [e2e(adapter, args.num_draft, w, args.max_tokens, t) for w in widths for t in (0.0, 0.8)],
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
