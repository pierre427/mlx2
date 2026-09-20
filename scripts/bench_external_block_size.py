#!/usr/bin/env python3
"""A/B sweep of the DFlash2 external-draft block width (``num_draft``).

Loads the Muse target and its DFlash2 drafter once, then runs the same prompts
through ``ExternalDraftBatchGenerator`` at each ``num_draft`` and batch width.
An ordinary-target run (every lane ``disable_speculation``) at each batch width
is the greedy reference and the throughput floor.

Per configuration it reports decode tok/s, mean emitted tokens per verify
round (acceptance length, bonus included), accepted/proposed, and the
histogram of per-round accepted counts. Greedy rows are checked token-exact
against the ordinary reference. Prefill is excluded from timing: every lane is
prefilled to its anchor before the clock starts.

GPU job. ``--tiny`` runs the same harness on random CPU models (harness smoke
only; numbers are meaningless).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROMPTS = (
    "Write a Python function that parses an ISO-8601 duration string such as "
    "'P3DT4H5M' into total seconds, with docstring and three unit tests.",
    "Explain step by step how to compute the determinant of a 4x4 matrix by "
    "cofactor expansion, then do it for [[2,0,1,3],[1,1,0,2],[0,3,1,1],[4,1,2,0]].",
    "Write a 400-word short story about a lighthouse keeper who finds a "
    "message in a bottle written in her own handwriting.",
    "Summarize the causes, key events and consequences of the 1929 stock "
    "market crash as a structured outline with headings and bullet points.",
    "Translate the following into French and then explain three grammar "
    "points it uses: 'If I had known you were coming, I would have baked a "
    "cake, but the oven has been broken since last Tuesday.'",
    "Implement a thread-safe LRU cache in Rust with get/put, explain the "
    "ownership choices, and show example usage in main().",
    "Give a detailed comparison of TCP congestion control algorithms Reno, "
    "CUBIC and BBR, including how each reacts to packet loss.",
    "Produce a JSON array of 12 fictional books, each with title, author, "
    "year, genre, page_count and a one-sentence synopsis.",
)


def _parse_ints(text):
    return [int(value) for value in text.split(",") if value.strip()]


def _parse_floats(text):
    return [float(value) for value in text.split(",") if value.strip()]


def load(args):
    import mlx.core as mx

    if args.tiny:
        mx.set_default_device(mx.cpu)
        from mlx2.adapters.muse_glimmer_config import ModelArgs
        from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
        from mlx2.runtime.drafters.dflash2_config import DFlash2Config
        from mlx2.runtime.models.muse_glimmer import Model

        mx.random.seed(8)
        target = Model(ModelArgs(
            hidden_size=8, intermediate_size=16, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=1, head_dim=4,
            vocab_size=32, sliding_window=3, max_position_embeddings=4096,
        ))
        draft = DFlash2DraftModel(DFlash2Config(
            hidden_size=8, intermediate_size=16, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=4,
            vocab_size=32, num_target_layers=4, target_layer_ids=[0, 3],
            conv_kernel_size=2, conv_group_size=2, selector_rank=4,
            selector_top_k=4, block_size=16, mask_token_id=31,
            max_position_embeddings=4096, sliding_window=5,
            layer_types=["sliding_attention"] * 2,
        )).bind(target)
        prompts = [[1 + (i + j) % 29 for j in range(6 + i)] for i in range(len(PROMPTS))]
        return target, draft, "tiny", prompts, (), {"target": "tiny", "draft": "tiny"}

    from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter
    from mlx2.serving import generation_stop_token_ids

    adapter = MuseGlimmerAdapter(
        args.target,
        execution_policy={"draft_model": args.draft, "num_draft": 4},
    )
    prompts = [
        adapter.prompt_tokens({"messages": [{"role": "user", "content": text}]})
        for text in PROMPTS
    ]
    stops = () if args.ignore_eos else generation_stop_token_ids(adapter)
    identity = {
        "target": args.target,
        "draft": args.draft,
        "fingerprint": adapter.identity["fingerprint"],
        "draft_block_size": int(adapter.draft_model.config.block_size),
    }
    return (
        adapter.model, adapter.draft_model, adapter.identity["fingerprint"],
        prompts, stops, identity,
    )


def run_cohort(target, draft, binding, prompts, stops, *, num_draft, temp,
               max_tokens, seed, ordinary):
    import mlx.core as mx
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG

    sampling = {"sampling_temp": temp}
    if temp:
        # Muse's recommended sampling (DFlash2 model card).
        sampling.update(
            top_p=0.95, top_k=min(64, int(target.args.vocab_size) - 1)
        )
    batch = ExternalDraftBatchGenerator(
        target, draft_model=draft, binding=binding,
        completion_batch_size=len(prompts), prefill_step_size=2048,
        num_draft=num_draft, stop_tokens=[[t] for t in stops],
    )
    try:
        uids = batch.insert(
            prompts, max_tokens=[max_tokens] * len(prompts),
            sampling_configs=[dict(sampling) for _ in prompts],
            lane_rngs=[LaneRNG(seed + i) for i in range(len(prompts))],
        )
        for uid in uids:
            lane = batch.lanes[uid]
            while lane.anchor is None:
                batch._prefill(lane)
            if ordinary:
                batch.disable_speculation(uid)
        mx.synchronize()
        tokens = {uid: [] for uid in uids}
        rounds = {uid: {} for uid in uids}
        started = time.perf_counter()
        while batch.lanes:
            _, responses = batch.next()
            for response in responses:
                tokens[response.uid].append(int(response.token))
                receipt = response.speculative_receipt or {}
                if receipt.get("current_execution") == "external_draft_verify":
                    # Every token of one verify round carries that round's
                    # receipt; key by the lane's round counter to count once.
                    rounds[response.uid][receipt["external_rounds"]] = (
                        int(receipt["round_proposed"]),
                        int(receipt["round_accepted"]),
                    )
        mx.synchronize()
        elapsed = time.perf_counter() - started
        stats = dict(batch.scheduler_stats)
    finally:
        batch.close()
    return [tokens[uid] for uid in uids], [rounds[uid] for uid in uids], elapsed, stats


def summarize(outputs, rounds, elapsed, stats, num_draft):
    emitted = sum(len(row) for row in outputs)
    per_round = [pair for lane in rounds for pair in lane.values()]
    proposed = sum(p for p, _ in per_round)
    accepted = sum(a for _, a in per_round)
    histogram = Counter(a for _, a in per_round)
    verify_tokens = sum(a + 1 for _, a in per_round)
    return {
        "tokens": emitted,
        "decode_s": elapsed,
        "tok_s": emitted / elapsed if elapsed else 0.0,
        "verify_rounds": len(per_round),
        # Mean tokens committed per verify round (accepted + bonus/correction).
        "acceptance_length": verify_tokens / len(per_round) if per_round else 0.0,
        "accept_rate": accepted / proposed if proposed else 0.0,
        "accepted": accepted,
        "proposed": proposed,
        "full_block_rounds": histogram.get(num_draft, 0),
        "round_accepted_histogram": {str(k): histogram[k] for k in sorted(histogram)},
        "target_max_width": stats.get("target_max_width", 0),
        "draft_max_width": stats.get("draft_max_width", 0),
    }


def first_divergence(got, want):
    for index, (a, b) in enumerate(zip(got, want)):
        if a != b:
            return index
    return None if len(got) == len(want) else min(len(got), len(want))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", default=str(Path.home() / "mlx-models/Muse-Glimmer-30B-mlx-4bit"))
    parser.add_argument("--draft", default=str(Path.home() / "mlx-models/Muse-Glimmer-30B-DFlash2"))
    parser.add_argument("--num-draft", default="2,3,4,5,6,7,8,10,12,15")
    parser.add_argument("--batch", default="1,4")
    parser.add_argument("--temps", default="0,1.0")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompts", type=int, default=len(PROMPTS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ignore-eos", action="store_true",
                        help="fixed-length rows (no stop tokens)")
    parser.add_argument("--tiny", action="store_true", help="CPU harness smoke")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import mlx.core as mx

    target, draft, binding, prompts, stops, identity = load(args)
    prompts = prompts[: args.prompts]
    counts = _parse_ints(args.num_draft)
    block = int(draft.config.block_size)
    bad = [k for k in counts if not 1 <= k < block]
    if bad:
        raise SystemExit(f"num_draft {bad} outside 1..{block - 1} (trained block {block})")
    widths = _parse_ints(args.batch)
    temps = _parse_floats(args.temps)
    results = {"identity": identity, "args": vars(args), "configs": []}

    def cohorts(width):
        return [prompts[i:i + width] for i in range(0, len(prompts), width)
                if len(prompts[i:i + width]) == width]

    def execute(width, temp, num_draft, ordinary):
        # Shape warmup: one short cohort at this exact verify width.
        run_cohort(target, draft, binding, cohorts(width)[0], stops,
                   num_draft=num_draft, temp=temp, max_tokens=min(24, args.max_tokens),
                   seed=args.seed, ordinary=ordinary)
        outputs, rounds, elapsed, stats_list = [], [], 0.0, []
        samples = []
        for repeat in range(args.repeats):
            rep_outputs, rep_rounds, rep_elapsed = [], [], 0.0
            for index, group in enumerate(cohorts(width)):
                o, r, e, s = run_cohort(
                    target, draft, binding, group, stops, num_draft=num_draft,
                    temp=temp, max_tokens=args.max_tokens,
                    seed=args.seed + 1000 * index, ordinary=ordinary,
                )
                rep_outputs += o; rep_rounds += r; rep_elapsed += e; stats_list.append(s)
            samples.append(sum(len(x) for x in rep_outputs) / rep_elapsed)
            if repeat == 0:
                outputs, rounds = rep_outputs, rep_rounds
            elapsed += rep_elapsed
        stats = {
            "target_max_width": max(s.get("target_max_width", 0) for s in stats_list),
            "draft_max_width": max(s.get("draft_max_width", 0) for s in stats_list),
        }
        row = summarize(outputs, rounds, elapsed / args.repeats, stats, num_draft)
        row["tok_s_samples"] = samples
        row["tok_s_median"] = statistics.median(samples)
        return outputs, row

    for width in widths:
        if not cohorts(width):
            raise SystemExit(f"need at least {width} prompts for batch {width}")
        for temp in temps:
            reference, base = execute(width, temp, 1, ordinary=True)
            base.update(batch=width, temp=temp, num_draft=0, mode="ordinary")
            results["configs"].append(base)
            print(f"B={width} T={temp} ordinary: {base['tok_s_median']:.1f} tok/s", flush=True)
            for num_draft in counts:
                outputs, row = execute(width, temp, num_draft, ordinary=False)
                row.update(batch=width, temp=temp, num_draft=num_draft, mode="external")
                row["speedup_vs_ordinary"] = row["tok_s_median"] / base["tok_s_median"]
                if temp == 0:
                    divergences = [first_divergence(o, r) for o, r in zip(outputs, reference)]
                    row["greedy_exact_rows"] = sum(d is None for d in divergences)
                    row["greedy_rows"] = len(divergences)
                    row["greedy_first_divergence"] = divergences
                results["configs"].append(row)
                exact = (
                    f" exact {row['greedy_exact_rows']}/{row['greedy_rows']}"
                    if temp == 0 else ""
                )
                print(
                    f"B={width} T={temp} K={num_draft:2d}: {row['tok_s_median']:.1f} tok/s "
                    f"x{row['speedup_vs_ordinary']:.2f} tau={row['acceptance_length']:.2f} "
                    f"acc={row['accept_rate']:.3f} full={row['full_block_rounds']}"
                    f"/{row['verify_rounds']}{exact}",
                    flush=True,
                )
                mx.clear_cache()
    results["peak_memory_gib"] = mx.get_peak_memory() / (1 << 30)
    text = json.dumps(results, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
