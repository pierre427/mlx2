#!/usr/bin/env python3
"""Owned depth ladder for real Nemotron affine-Q8 Mamba projections."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import nemotron35_probe_common as common
import numpy as np


def _median(values):
    return float(np.median(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--settle-seconds", type=int, default=60)
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
    if not 0 <= args.settle_seconds <= 300:
        raise ValueError("settle seconds must be between 0 and 300")

    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.conversion.kernel_probe import error_metrics
    from mlx2.runtime.models.row_exact_qmv import quantized_linear

    adapter = resolve_adapter(args.model)(str(args.model))
    time.sleep(args.settle_seconds)
    mixer = adapter.model.layers[0].mixer
    modules = {"in_proj": mixer.in_proj, "out_proj": mixer.out_proj}
    rng = np.random.default_rng(3517)
    cells = []
    for boundary, module in modules.items():
        features = 4096 if boundary == "in_proj" else 8192
        for rows in (1, 2, 4, 8, 16):
            common.require_owner(args.cpg_owner_lock)
            cell_before = common.host_snapshot()
            x = mx.array(rng.normal(size=(rows, features)).astype(np.float32)).astype(
                mx.bfloat16
            )

            def stock(value=x, layer=module):
                return layer(value)

            def exact(value=x, layer=module):
                return quantized_linear(layer, value)[0]

            def serial(value=x, layer=module, row_count=rows):
                parts = []
                for index in range(row_count):
                    part = layer(value[index : index + 1])
                    mx.eval(part)
                    parts.append(part)
                result = mx.concatenate(parts, axis=0)
                mx.eval(result)
                return result

            def fp16_input(value=x, layer=module):
                return layer(value.astype(mx.float16)).astype(mx.bfloat16)

            calls = {
                "stock_multirow": stock,
                "row_exact_qmv": exact,
                "serial_b1": serial,
                "stock_fp16_input": fp16_input,
            }
            outputs = {name: call() for name, call in calls.items()}
            mx.eval(*outputs.values())
            oracle = np.asarray(outputs["serial_b1"].astype(mx.float32))
            oracle_bits = np.asarray(outputs["serial_b1"].view(mx.uint16))
            metrics = {}
            for name, output in outputs.items():
                actual = np.asarray(output.astype(mx.float32))
                metrics[name] = {
                    "bit_mismatches_vs_serial_b1": int(
                        np.count_nonzero(
                            np.asarray(output.view(mx.uint16)) != oracle_bits
                        )
                    ),
                    "error_vs_serial_b1": error_metrics(actual, oracle),
                }
            _unused, route = quantized_linear(module, x)
            timings = {name: [] for name in calls}
            for _ in range(3):
                for call in calls.values():
                    mx.eval(call())
            names = tuple(calls)
            for repetition in range(12):
                for offset in range(len(names)):
                    name = names[(offset + repetition) % len(names)]
                    tick = time.perf_counter_ns()
                    mx.eval(calls[name]())
                    timings[name].append((time.perf_counter_ns() - tick) * 1e-6)
            medians = {name: _median(values) for name, values in timings.items()}
            serial_ms = medians["serial_b1"]
            stock_ms = medians["stock_multirow"]
            cell_after = common.host_snapshot()
            contamination = [
                reason
                for reason in common.contamination(cell_before, cell_after)
                if reason != "swap-outs increased"
            ]
            cells.append(
                {
                    "boundary": boundary,
                    "rows": rows,
                    "features": features,
                    "output_features": int(outputs["serial_b1"].shape[-1]),
                    "row_exact_route": route,
                    "metrics": metrics,
                    "timings_ms": timings,
                    "median_ms": medians,
                    "speedup_vs_serial_b1": {
                        name: serial_ms / value for name, value in medians.items()
                    },
                    "speedup_vs_stock_multirow": {
                        name: stock_ms / value for name, value in medians.items()
                    },
                    "host_before": cell_before,
                    "host_after": cell_after,
                    "contamination": contamination,
                }
            )
            common.atomic_json(
                args.out_dir / "result.json",
                {
                    "schema": "mlx2.nemotron35-mamba-q8-depth-ladder.v1",
                    "state": "running",
                    "cells": cells,
                },
            )
            if contamination:
                raise RuntimeError(
                    "component cell contaminated: " + "; ".join(contamination)
                )
            del x, outputs
            mx.clear_cache()

    del adapter
    gc.collect()
    mx.clear_cache()
    after = common.host_snapshot()
    contamination = [
        reason
        for reason in common.contamination(before, after)
        if reason != "swap-outs increased"
    ]
    all_exact = all(
        cell["metrics"]["row_exact_qmv"]["bit_mismatches_vs_serial_b1"] == 0
        for cell in cells
    )
    receipt = {
        "schema": "mlx2.nemotron35-mamba-q8-depth-ladder.v1",
        "state": "completed",
        "owner": owner,
        "source": common.source_identity(),
        "dependencies": common.runtime_dependencies(),
        "model": str(args.model.resolve()),
        "settle_seconds": args.settle_seconds,
        "before": before,
        "after": after,
        "swap_ignored_by_user": True,
        "cells": cells,
        "all_row_exact": all_exact,
        "contamination": contamination,
        "passed": all_exact and not contamination,
        "qualified": False,
        "scope": "Real affine-Q8 Mamba component; no serving or route claim.",
    }
    common.atomic_json(args.out_dir / "result.json", receipt)
    if contamination:
        raise RuntimeError("projection depth probe contaminated")
    if not all_exact:
        raise RuntimeError("row-exact QMV differed from serial B1")


if __name__ == "__main__":
    main()
