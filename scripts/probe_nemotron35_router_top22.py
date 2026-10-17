#!/usr/bin/env python3
"""Exactness and timing probe for the default-off Super top-22 router."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def _timed(mx, function, reps):
    samples = []
    for _ in range(reps):
        started = time.perf_counter()
        output = function()
        mx.eval(*output)
        samples.append(time.perf_counter() - started)
    return 1000 * statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        raise SystemExit("refusing Metal execution without --i-own-the-gpu")

    import mlx.core as mx
    import nemotron35_probe_common as common

    from mlx2.runtime.models.nemotron_h import group_expert_select
    from mlx2.runtime.models.nemotron_moe_router import router_top22

    owner_before = common.require_owner(args.cpg_owner_lock)
    host_before = common.host_snapshot()
    errors = common.admission_errors(
        host_before, minimum_available_bytes=4 * common.GIB, performance=True
    )
    if errors:
        raise RuntimeError(f"router probe admission failed: {errors}")

    mx.random.seed(20261016)
    bias = mx.random.uniform(-0.15, 0.15, shape=(512,)).astype(mx.bfloat16)
    cases = []
    for rows in (1, 2, 4, 8, 16):
        logits = mx.random.normal((rows, 512)).astype(mx.float32)
        stock = lambda row_logits=logits: group_expert_select(
            row_logits, bias, 22, 1, 1, 5.0, True
        )
        candidate = lambda row_logits=logits: router_top22(row_logits, bias)
        mx.eval(*stock(), *candidate())
        stock_indices, stock_scores = stock()
        candidate_indices, candidate_scores = candidate()
        mx.eval(stock_indices, stock_scores, candidate_indices, candidate_scores)
        index_mismatches = int(mx.sum(stock_indices != candidate_indices).item())
        score_mismatches = int(mx.sum(stock_scores != candidate_scores).item())
        max_score_delta = float(mx.max(mx.abs(stock_scores - candidate_scores)).item())
        stock_ms = _timed(mx, stock, args.reps)
        candidate_ms = _timed(mx, candidate, args.reps)
        cases.append(
            {
                "rows": rows,
                "index_mismatches": index_mismatches,
                "score_mismatches": score_mismatches,
                "max_score_delta": max_score_delta,
                "stock_ms": stock_ms,
                "candidate_ms": candidate_ms,
                "speedup": stock_ms / candidate_ms,
            }
        )

    report = {
        "schema": "mlx2.nemotron35-router-top22-component.v1",
        "cases": cases,
        "all_exact": all(
            case["index_mismatches"] == 0 and case["score_mismatches"] == 0
            for case in cases
        ),
        "owner_before": owner_before,
        "owner_after": common.require_owner(args.cpg_owner_lock),
        "host_before": host_before,
        "host_after": common.host_snapshot(),
        "limitations": [
            "synthetic component timing only",
            "full-model engagement and end-to-end performance remain untested",
            "ties and non-finite values require separate correctness probes",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
