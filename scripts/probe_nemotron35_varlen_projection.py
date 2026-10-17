#!/usr/bin/env python3
"""Owned real-q8 row-shape probe for Nemotron Mamba projections."""

from __future__ import annotations

import argparse
import gc
import time
from collections import defaultdict
from pathlib import Path

import nemotron35_probe_common as common
import numpy as np


def _bits(value, mx):
    return np.asarray(value.view(mx.uint16)).reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--settle-seconds", type=int, default=60)
    parser.add_argument(
        "--arithmetic", choices=("stock", "row_exact"), default="stock"
    )
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Refusing to overwrite nonempty probe evidence")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    owner = common.require_owner(args.cpg_owner_lock)
    before = common.host_snapshot()
    errors = common.admission_errors(
        before, minimum_available_bytes=4 * common.GIB, performance=True
    )
    if errors:
        raise RuntimeError("; ".join(errors))

    import mlx.core as mx
    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.models.nemotron_varlen_projection import project_live_rows

    adapter = resolve_adapter(args.model)(
        str(args.model),
        execution_policy={
            "mamba_projection": (
                "compact_row_exact" if args.arithmetic == "row_exact"
                else "compact_padded"
            )
        },
    )
    if not 0 <= args.settle_seconds <= 300:
        raise ValueError("settle seconds must be between 0 and 300")
    time.sleep(args.settle_seconds)
    mixer = adapter.model.layers[0].mixer
    modules = {"in_proj": mixer.in_proj, "out_proj": mixer.out_proj}
    geometries = (
        (2, 4, (4, 2)),
        (2, 5, (5, 3)),
        (4, 4, (4, 4, 2, 1)),
        # Reachable confidence-tree suffix slabs: B1/B2 continue four rows,
        # while lower-ranked B3/B4 finish after one or two unique nodes.
        (4, 4, (4, 4, 2, 2)),
        (4, 4, (4, 4, 3, 2)),
    )
    rng = np.random.default_rng(3516)
    cases = []
    for boundary, module in modules.items():
        features = 4096 if boundary == "in_proj" else 8192
        for batch, width, lengths in geometries:
            common.require_owner(args.cpg_owner_lock)
            host_before = common.host_snapshot()
            inputs = mx.array(
                rng.normal(size=(batch, width, features)).astype(np.float32)
            ).astype(mx.bfloat16)
            indices = np.asarray(
                [row * width + column for row, length in enumerate(lengths)
                 for column in range(length)],
                dtype=np.int32,
            )
            flat = inputs.reshape(batch * width, features)
            ordinary = module(inputs)
            counters = defaultdict(int)
            compact = project_live_rows(
                module,
                inputs,
                list(lengths),
                counters,
                boundary=boundary,
                arithmetic=args.arithmetic,
            )
            serial = mx.concatenate(
                [module(flat[index:index + 1]) for index in indices], axis=0
            )
            ordinary_live = ordinary.reshape(batch * width, -1)[mx.array(indices)]
            compact_live = compact.reshape(batch * width, -1)[mx.array(indices)]
            mx.eval(ordinary_live, compact_live, serial)
            timings = {"rectangular": [], "compact": []}
            calls = {
                "rectangular": lambda module=module, inputs=inputs: module(inputs),
                "compact": lambda module=module, inputs=inputs,
                lengths=lengths, boundary=boundary: project_live_rows(
                    module, inputs, list(lengths), defaultdict(int), boundary=boundary,
                    arithmetic=args.arithmetic,
                ),
            }
            for _ in range(3):
                for call in calls.values():
                    mx.eval(call())
            for repetition in range(12):
                order = ("rectangular", "compact") if repetition % 2 == 0 else (
                    "compact", "rectangular"
                )
                for name in order:
                    tick = time.perf_counter_ns()
                    mx.eval(calls[name]())
                    timings[name].append((time.perf_counter_ns() - tick) * 1e-6)
            medians = {name: float(np.median(values)) for name, values in timings.items()}
            post = common.host_snapshot()
            contamination = [
                reason for reason in common.contamination(host_before, post)
                if reason != "swap-outs increased"
            ]
            cases.append({
                "boundary": boundary,
                "shape": [batch, width, features],
                "lengths": list(lengths),
                "logical_rows": int(sum(lengths)),
                "rectangular_rows": batch * width,
                "padding_rows": batch * width - int(sum(lengths)),
                "rectangular_vs_compact_mismatches": int(np.count_nonzero(
                    _bits(ordinary_live, mx) != _bits(compact_live, mx)
                )),
                "rectangular_vs_serial_b1_mismatches": int(np.count_nonzero(
                    _bits(ordinary_live, mx) != _bits(serial, mx)
                )),
                "compact_vs_serial_b1_mismatches": int(np.count_nonzero(
                    _bits(compact_live, mx) != _bits(serial, mx)
                )),
                "timings_ms": timings,
                "median_ms": medians,
                "compact_speedup": medians["rectangular"] / medians["compact"],
                "counters": dict(counters),
                "host_before": host_before,
                "host_after": post,
                "contamination": contamination,
            })
            if contamination:
                raise RuntimeError("component cell contaminated: " + "; ".join(contamination))
            del inputs, ordinary, compact, serial, ordinary_live, compact_live
            mx.clear_cache()
    del adapter
    gc.collect()
    mx.clear_cache()
    after = common.host_snapshot()
    contamination = [
        reason for reason in common.contamination(before, after)
        if reason != "swap-outs increased"
    ]
    receipt = {
        "schema": "mlx2.nemotron35-mamba-varlen-q8-projection-component.v1",
        "owner": owner,
        "source": common.source_identity(),
        "dependencies": common.runtime_dependencies(),
        "model": str(args.model.resolve()),
        "before": before,
        "after": after,
        "swap_ignored_by_user": True,
        "arithmetic": args.arithmetic,
        "cases": cases,
        "all_rectangular_compact_exact": all(
            case["rectangular_vs_compact_mismatches"] == 0 for case in cases
        ),
        "all_compact_serial_b1_exact": all(
            case["compact_vs_serial_b1_mismatches"] == 0 for case in cases
        ),
        "contamination": contamination,
        "passed": not contamination,
        "qualified": False,
        "scope": "Isolated real-q8 Mamba projection row-shape evidence; no route claim.",
    }
    common.atomic_json(args.out_dir / "result.json", receipt)
    if contamination:
        raise RuntimeError("projection probe contaminated")


if __name__ == "__main__":
    main()
