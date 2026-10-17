#!/usr/bin/env python3
"""Owned real-weight timing/parity probe for row-tiled ModelOpt NVFP4."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import nemotron35_probe_common as common
import numpy as np


def _median(values):
    return float(np.median(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Refusing to overwrite nonempty probe evidence")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    owner = common.require_owner(args.cpg_owner_lock)
    before = common.host_snapshot()
    errors = common.admission_errors(before, minimum_available_bytes=8 * common.GIB)
    if errors:
        raise RuntimeError("; ".join(errors))

    import mlx.core as mx

    from mlx2.conversion.kernel_probe import error_metrics, verify
    from mlx2.runtime.models.modelopt_nvfp4 import ModelOptNVFP4Linear
    from mlx2.runtime.models.modelopt_nvfp4_tile import project, project_row_exact

    manifest = verify(args.inputs)
    cells = []
    for matrix in manifest["matrices"]:
        with np.load(args.inputs / matrix["file"], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        n, k = matrix["shape"]
        linear = ModelOptNVFP4Linear(k, n)
        linear.load_weights(
            [
                ("weight", mx.array(arrays["native_weight"])),
                ("scales", mx.array(arrays["native_scales"])),
                ("global_scale", mx.array(arrays["global_scale"])),
            ],
            strict=True,
        )
        dense = mx.array(arrays["dense_fp32"]).astype(mx.bfloat16)
        mx.eval(linear.parameters(), dense)
        for rows in (1, 2, 4, 8, 16):
            common.require_owner(args.cpg_owner_lock)
            rng = np.random.default_rng(manifest["seed"] + n + k + rows)
            x = mx.array(rng.normal(size=(rows, k)).astype(np.float32)).astype(
                mx.bfloat16
            )

            def native(value, module=linear):
                return module(value)

            def serial_native_b1(value, module=linear, row_count=rows):
                parts = []
                for index in range(row_count):
                    part = module(value[index : index + 1])
                    mx.eval(part)
                    parts.append(part)
                result = mx.concatenate(parts, axis=0)
                mx.eval(result)
                return result

            def tiled(value, module=linear):
                return project(module, value)

            def row_exact(value, module=linear):
                return project_row_exact(module, value)

            def dense_bf16(value, weight=dense):
                return value @ weight.T

            def bf16_input(value, module=linear):
                result = mx.quantized_matmul(
                    value,
                    module.weight,
                    scales=module.scales,
                    group_size=16,
                    bits=4,
                    mode="nvfp4",
                    transpose=True,
                )
                return (result.astype(mx.float32) * module.global_scale).astype(
                    mx.bfloat16
                )

            def fp16_tile(value, module=linear):
                return project(module, value.astype(mx.float16)).astype(mx.bfloat16)

            calls = {
                "native_current": native,
                "serial_native_b1": serial_native_b1,
                "grid_row_exact": row_exact,
                "tiled_fp32_accum": tiled,
                "dense_decoded_bf16": dense_bf16,
                "native_bf16_input": bf16_input,
                "tiled_fp16_input": fp16_tile,
            }
            outputs = {name: call(x) for name, call in calls.items()}
            mx.eval(x, *outputs.values())
            oracle_bits = np.asarray(outputs["serial_native_b1"].view(mx.uint16))
            metrics = {}
            oracle_fp32 = np.asarray(outputs["serial_native_b1"].astype(mx.float32))
            for name, output in outputs.items():
                bits = np.asarray(output.view(mx.uint16))
                metrics[name] = {
                    "bit_mismatches_vs_serial_b1": int(
                        np.count_nonzero(bits != oracle_bits)
                    ),
                    "error_vs_serial_b1": error_metrics(
                        np.asarray(output.astype(mx.float32)), oracle_fp32
                    ),
                }
            segments = []
            exact_segments = []
            for start in range(0, rows, 4):
                segments.append(project(linear, x[start : start + 4]))
                exact_segments.append(project_row_exact(linear, x[start : start + 4]))
            segmented = mx.concatenate(segments, axis=0)
            exact_segmented = mx.concatenate(exact_segments, axis=0)
            mx.eval(segmented, exact_segmented)
            segmentation_mismatches = int(
                np.count_nonzero(
                    np.asarray(segmented.view(mx.uint16))
                    != np.asarray(outputs["tiled_fp32_accum"].view(mx.uint16))
                )
            )
            exact_segmentation_mismatches = int(
                np.count_nonzero(
                    np.asarray(exact_segmented.view(mx.uint16))
                    != np.asarray(outputs["grid_row_exact"].view(mx.uint16))
                )
            )
            timings = {name: [] for name in calls}
            for _ in range(4):
                for call in calls.values():
                    mx.eval(call(x))
            names = tuple(calls)
            for repetition in range(16):
                for offset in range(len(names)):
                    name = names[(offset + repetition) % len(names)]
                    tick = time.perf_counter_ns()
                    mx.eval(calls[name](x))
                    timings[name].append((time.perf_counter_ns() - tick) * 1e-6)
            medians = {name: _median(values) for name, values in timings.items()}
            native_ms = medians["native_current"]
            serial_ms = medians["serial_native_b1"]
            cells.append(
                {
                    "matrix": matrix["prefix"],
                    "shape": matrix["shape"],
                    "rows": rows,
                    "rows_are": "packed token or verification rows in one conversation",
                    "metrics": metrics,
                    "segmentation_mismatches": segmentation_mismatches,
                    "exact_segmentation_mismatches": exact_segmentation_mismatches,
                    "timings_ms": timings,
                    "median_ms": medians,
                    "speedup_vs_native": {
                        name: native_ms / value for name, value in medians.items()
                    },
                    "speedup_vs_serial_b1": {
                        name: serial_ms / value for name, value in medians.items()
                    },
                }
            )
            common.atomic_json(
                args.out_dir / "result.json",
                {
                    "schema": "mlx2.nemotron35-modelopt-nvfp4-row-tile.v1",
                    "state": "running",
                    "cells": cells,
                },
            )
            del x, outputs, segmented, exact_segmented, segments, exact_segments
            mx.clear_cache()
        del arrays, linear, dense
        mx.clear_cache()

    after = common.host_snapshot()
    contamination = [
        reason
        for reason in common.contamination(before, after)
        if reason != "swap-outs increased"
    ]
    exact_cells = [
        cell["metrics"]["grid_row_exact"]["bit_mismatches_vs_serial_b1"] == 0
        and cell["exact_segmentation_mismatches"] == 0
        for cell in cells
    ]
    receipt = {
        "schema": "mlx2.nemotron35-modelopt-nvfp4-row-tile.v1",
        "state": "completed",
        "owner": owner,
        "source": common.source_identity(),
        "dependencies": common.runtime_dependencies(),
        "inputs": manifest,
        "before": before,
        "after": after,
        "swap_ignored_by_user": True,
        "cells": cells,
        "all_tiled_exact": all(exact_cells),
        "all_weight_reuse_exact": all(
            cell["metrics"]["tiled_fp32_accum"]["bit_mismatches_vs_serial_b1"] == 0
            and cell["segmentation_mismatches"] == 0
            for cell in cells
        ),
        "all_tiled_faster": all(
            cell["speedup_vs_native"]["tiled_fp32_accum"] > 1.0 for cell in cells
        ),
        "contamination": contamination,
        "passed": all(exact_cells) and not contamination,
        "qualified": False,
        "scope": "Isolated real-weight component; no serving or route claim.",
    }
    common.atomic_json(args.out_dir / "result.json", receipt)
    if contamination:
        raise RuntimeError("NVFP4 row-tile component probe contaminated")
    if not all(exact_cells):
        raise RuntimeError("NVFP4 row-tile exactness or segmentation failed")


if __name__ == "__main__":
    main()
