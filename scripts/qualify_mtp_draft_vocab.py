#!/usr/bin/env python3
"""Single-load GPU qualification for the reduced native-MTP vocabulary.

The full and reduced arms use the same adapter, target head, MTP state machine,
prompts and process.  Between closed batches the harness toggles only whether
the separately owned sliced proposal head is engaged.  The target and verify
head is never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx

PROMPTS = [
    "Write a Python function that merges two sorted lists. Return code only.",
    "Explain why immutable data structures help concurrent programs.",
    "Implement an LRU cache in Python with get and put methods. Return code only.",
    "Give three concrete rules for reviewing database migrations safely.",
    "Write a TypeScript function that groups records by a string key.",
    "Explain speculative decoding to an experienced systems programmer.",
    "Implement binary search in Rust and include boundary-case tests.",
    "Summarize the tradeoff between latency and throughput in continuous batching.",
]


def swapouts() -> int:
    output = subprocess.run(
        ["vm_stat"], capture_output=True, check=True, text=True
    ).stdout
    for line in output.splitlines():
        if line.startswith("Swapouts"):
            return int(line.split(":", 1)[1].strip().rstrip("."))
    return 0


def source_identity(root: Path) -> dict:
    tracked = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--binary", "HEAD"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    untracked = []
    status = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.splitlines()
    for name in sorted(status):
        path = root / name
        untracked.append((name, hashlib.sha256(path.read_bytes()).hexdigest()))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=True, text=True
    ).stdout.strip()
    return {
        "head": head,
        "tracked_diff_sha256": hashlib.sha256(tracked).hexdigest(),
        "untracked": untracked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--widths", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--swap-limit-mb", type=int, default=768)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        parser.error("refusing Metal execution without --i-own-the-gpu")
    if any(width not in (1, 4, 8) for width in args.widths):
        parser.error("widths must be drawn from 1, 4, 8")
    if args.smoke:
        args.widths, args.reps, args.max_tokens, args.warmup_tokens = [1], 1, 32, 8

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG

    root = Path(__file__).resolve().parents[1]
    source = source_identity(root)
    before_swap = swapouts()
    load_started = time.perf_counter()
    adapter = resolve_adapter(args.model, mtp=True)(
        args.model, execution_policy={"mtp_draft_vocab": True}
    )
    model = adapter.model
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - load_started
    config = adapter.execution_config(
        max_lanes=max(args.widths), prefill_step=adapter.prefill_step_default()
    )

    def encode(prompt: str) -> list[int]:
        return list(
            adapter.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=False,
            )
        )

    encoded = [encode(prompt) for prompt in PROMPTS]

    def run_batch(arm: str, width: int, max_tokens: int, *, constrained=False) -> dict:
        model.mtp_draft_vocab_enabled = arm == "reduced"
        status_before = dict(model.mtp_draft_vocab_status())
        generator = BatchGenerator(
            model,
            completion_batch_size=width,
            prefill_batch_size=1,
            prefill_step_size=adapter.prefill_step_default(),
            self_mtp=config,
        )
        prompts = encoded[:width]
        insert = {
            "max_tokens": [max_tokens] * width,
            "lane_rngs": [LaneRNG(1 + index) for index in range(width)],
            "self_mtp_configs": [{"sampling_temp": 0.0}] * width,
        }
        if constrained:
            insert["logits_processors"] = [
                [lambda _tokens, logits: logits] for _ in range(width)
            ]
        uids = generator.insert(prompts, **insert)
        tokens = {uid: [] for uid in uids}
        receipts = {}
        first_all = None
        last = None
        done = set()
        started = time.perf_counter()
        try:
            while len(done) < width:
                _, responses = generator.next()
                now = time.perf_counter()
                if responses and first_all is None:
                    first_all = now
                for response in responses:
                    tokens[response.uid].append(int(response.token))
                    if response.mtp_receipt is not None:
                        receipts[response.uid] = response.mtp_receipt
                    if response.finish_reason:
                        done.add(response.uid)
                    last = now
        finally:
            scheduler = dict(generator.scheduler_stats)
            generator.close()
        status_after = dict(model.mtp_draft_vocab_status())
        elapsed = max((last or time.perf_counter()) - (first_all or started), 1e-9)
        emitted = sum(max(len(values) - 1, 0) for values in tokens.values())
        cycles = sum(
            int(receipt.get("stats", {}).get("cycles", 0))
            for receipt in receipts.values()
        )
        receipt_emitted = sum(
            int(receipt.get("stats", {}).get("total_emitted", 0))
            for receipt in receipts.values()
        )
        return {
            "arm": arm,
            "width": width,
            "constrained": constrained,
            "tokens": [tokens[uid] for uid in uids],
            "elapsed_seconds": elapsed,
            "tokens_decoded": emitted,
            "tok_s": emitted / elapsed,
            "cycles": cycles,
            "receipt_emitted": receipt_emitted,
            "tokens_per_cycle": receipt_emitted / cycles if cycles else None,
            "counter_delta": {
                "reduced_head_calls": status_after["reduced_head_calls"]
                - status_before["reduced_head_calls"],
                "full_vocab_bypasses": status_after["full_vocab_bypasses"]
                - status_before["full_vocab_bypasses"],
            },
            "scheduler": scheduler,
            "receipts": list(receipts.values()),
        }

    try:
        for arm in ("full", "reduced"):
            for width in args.widths:
                run_batch(arm, width, args.warmup_tokens)
        constrained = run_batch("reduced", 1, 8, constrained=True)
        timed_swap_base = swapouts()
        results = []
        for rep in range(args.reps):
            arms = ("full", "reduced") if rep % 2 == 0 else ("reduced", "full")
            for width in args.widths:
                for arm in arms:
                    row = run_batch(arm, width, args.max_tokens)
                    row["rep"] = rep
                    results.append(row)
                    print(
                        json.dumps(
                            {key: value for key, value in row.items() if key not in {"tokens", "receipts", "scheduler"}}
                        ),
                        flush=True,
                    )
                    mx.clear_cache()
        parity = []
        for rep in range(args.reps):
            for width in args.widths:
                full = next(
                    row for row in results
                    if row["rep"] == rep and row["width"] == width and row["arm"] == "full"
                )
                reduced = next(
                    row for row in results
                    if row["rep"] == rep and row["width"] == width and row["arm"] == "reduced"
                )
                parity.append(
                    {"rep": rep, "width": width, "equal": full["tokens"] == reduced["tokens"]}
                )
        summary = {}
        for width in args.widths:
            for arm in ("full", "reduced"):
                rows = [row for row in results if row["width"] == width and row["arm"] == arm]
                summary[f"{arm}/B{width}"] = {
                    "tok_s_median": statistics.median(row["tok_s"] for row in rows),
                    "tokens_per_cycle_median": statistics.median(
                        row["tokens_per_cycle"] for row in rows
                    ),
                    "reduced_head_calls": sum(
                        row["counter_delta"]["reduced_head_calls"] for row in rows
                    ),
                    "full_vocab_bypasses": sum(
                        row["counter_delta"]["full_vocab_bypasses"] for row in rows
                    ),
                }
        after_swap = swapouts()
        output = {
            "schema": "mlx2.mtp-draft-vocab-gpu-ab.v1",
            "mode": "smoke" if args.smoke else "ab",
            "source": source,
            "artifact_identity": adapter.identity,
            "draft_vocab": adapter.diagnostics()["mtp_draft_vocab"],
            "mlx_version": mx.__version__,
            "load_seconds": load_seconds,
            "peak_gib": mx.get_peak_memory() / 2**30,
            "load_warm_swapout_delta_mb": (timed_swap_base - before_swap) / 64,
            "timed_swapout_delta_mb": (after_swap - timed_swap_base) / 64,
            "config": config,
            "constrained_bypass": {
                "counter_delta": constrained["counter_delta"],
                "passed": constrained["counter_delta"]["full_vocab_bypasses"] > 0,
            },
            "parity": parity,
            "summary": summary,
            "results": results,
            "gate": {
                "all_greedy_equal": all(row["equal"] for row in parity),
                "candidate_engaged": all(
                    row["counter_delta"]["reduced_head_calls"] > 0
                    for row in results if row["arm"] == "reduced"
                ),
                "control_bypassed": all(
                    row["counter_delta"]["full_vocab_bypasses"] > 0
                    for row in results if row["arm"] == "full"
                ),
                "constrained_bypass": constrained["counter_delta"]["full_vocab_bypasses"] > 0,
                "swap_within_limit": (after_swap - timed_swap_base)
                <= args.swap_limit_mb * 64,
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps({"summary": summary, "gate": output["gate"]}, indent=2))
        if not all(output["gate"].values()):
            raise SystemExit(2)
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
