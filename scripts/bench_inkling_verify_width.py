#!/usr/bin/env python3
"""Inkling verify cost against row width on a real long prefix (diagnostic).

Prefills one actual varied-ledger prompt, then times a W-row target forward on
the live cache (rolled back through each plane's speculation trim) for each W,
interleaved and repeated.  This is the bandwidth curve a speculative tree pays
per round; it is not emitted-token throughput or a qualification result.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx  # noqa: E402
from probe_inkling import atomic, context_cases, host, install_serving_lane, ownership  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--contexts", type=int, nargs="+", default=[16384, 32768])
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--lane-matmul", default="exact", choices=["exact", "auto", "off"],
                    help="lane law to install, as `mlx2 serve --lane-matmul` (packed AST serves exact)")
    ap.add_argument("--prefill-step", type=int, default=2048)
    ap.add_argument("--allocator-cache-bytes", type=int, default=1 << 30)
    ap.add_argument("--down-combine-variant", default="original")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    from mlx2.adapters.inkling import InklingAdapter

    policy = json.loads(args.policy.read_text())
    policy.pop("draft_model", None)
    policy.pop("cost_per_commit_handoff", None)
    report = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "ownership": ownership(), "before": host(), "policy": policy,
        "scope": "diagnostic W-row target forward on a real prefix; median of interleaved repeats; "
                 "not emitted-token throughput; no thermal hold",
        "cells": [],
    }
    from mlx2.runtime.models.inkling import language as inkling_language

    inkling_language._DOWN_COMBINE_VARIANT = args.down_combine_variant
    report["down_combine_variant"] = args.down_combine_variant
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(args.allocator_cache_bytes)
    adapter = InklingAdapter(str(args.model), execution_policy=policy)
    report["checkpoint"] = args.model.name + "@" + adapter.identity["fingerprint"]
    lane = install_serving_lane(adapter, args.model, args.lane_matmul)
    report["lane_matmul"] = {"mode": args.lane_matmul, "law_id": (lane or {}).get("law_id", "stock")}
    model = adapter.model
    sub = argparse.Namespace(model=args.model, contexts=args.contexts, context_tokens=0,
                             generation_tokens=256)
    cases = {k: v for k, v in context_cases(sub).items() if k.startswith("long_generation")}
    try:
        for name, value in cases.items():
            text = value[0]
            prompt = adapter.prompt_tokens({"messages": [{"role": "user", "content": text}]})
            cache = model.make_cache()
            for begin in range(0, len(prompt), args.prefill_step):
                model(mx.array([prompt[begin:begin + args.prefill_step]], mx.uint32), cache=cache)
                mx.eval([c.state for c in cache])
            vocab = len(adapter.tokenizer)
            rows = {w: [] for w in args.widths}
            for repeat in range(args.repeats + 2):
                for w in (args.widths if repeat % 2 == 0 else list(reversed(args.widths))):
                    tokens = mx.array([[(prompt[-1] + 7 * i + 3) % vocab for i in range(w)]], mx.uint32)
                    for c in cache:
                        c.start_speculation(w)
                    mx.synchronize()
                    tic = time.perf_counter()
                    logits = model(tokens, cache=cache)
                    mx.eval(logits, [c.state for c in cache])
                    elapsed = time.perf_counter() - tic
                    for c in cache:
                        c.trim(w)
                        c.stop_speculation()
                    if repeat >= 2:
                        rows[w].append(elapsed * 1000)
            # The packed tree executor on chain trees of the same widths (AST
            # verify path).
            from mlx2.runtime.inkling_tree_executor import InklingTreeExecutor

            model.configure_kernels(
                fused=policy.get("inkling_fused", False),
                down_combine=policy.get("inkling_q2_down_combine", False),
                attention=policy.get("inkling_attention", "reference"),
                grouped=False, tree_kernels=True, tree_cohort=True,
                shortconv_prefill=policy.get("inkling_shortconv_prefill", False),
            )
            executor = InklingTreeExecutor.create()
            tree_rows = {w: [] for w in args.widths}
            for repeat in range(args.repeats + 2):
                for w in (args.widths if repeat % 2 == 0 else list(reversed(args.widths))):
                    proposed = [(prompt[-1] + 7 * i + 3) % vocab for i in range(w)]
                    mx.synchronize()
                    tic = time.perf_counter()
                    values, _, tx = executor.forward(
                        model, proposed, [-1] + list(range(w - 1)), cache, [1], cached=False)
                    mx.eval(values)
                    elapsed = time.perf_counter() - tic
                    del values, tx
                    if repeat >= 2:
                        tree_rows[w].append(elapsed * 1000)
            base = statistics.median(rows[args.widths[0]])
            cell = {"case": name, "prompt_tokens": len(prompt),
                    "median_ms": {w: statistics.median(v) for w, v in rows.items()},
                    "samples_ms": rows}
            cell["relative_to_w1"] = {w: m / base for w, m in cell["median_ms"].items()}
            cell["tree_median_ms"] = {w: statistics.median(v) for w, v in tree_rows.items()}
            report["cells"].append(cell)
            atomic(args.out / "report.json", report)
            print(json.dumps({"case": name, "median_ms": cell["median_ms"],
                              "tree_median_ms": cell["tree_median_ms"]}), flush=True)
            del cache
            mx.clear_cache()
    finally:
        adapter.close()
        report["after"] = host()
        atomic(args.out / "report.json", report)


if __name__ == "__main__":
    main()
