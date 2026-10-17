#!/usr/bin/env python3
"""Owned MLX 0.32.4 ModelOpt NVFP4 component matrix.

This probe compares real captured matrices with synthetic BF16 activations. It
is component evidence only: no target or draft route is loaded or qualified.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import nemotron35_probe_common as common
import numpy as np

ROWS = (1, 4, 8, 16, 128)


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
    errors = common.admission_errors(
        before, minimum_available_bytes=8 * common.GIB, performance=True
    )
    if errors:
        raise RuntimeError("; ".join(errors))

    import mlx.core as mx

    from mlx2.conversion.kernel_probe import error_metrics, verify
    from mlx2.runtime.models.modelopt_nvfp4 import ModelOptNVFP4Linear
    from mlx2.runtime.models.modelopt_nvfp4_b1 import project as project_b1

    dependencies = common.runtime_dependencies()
    if dependencies["versions"].get("mlx") != "0.32.4":
        raise RuntimeError("This matrix is bound to MLX 0.32.4")
    manifest = verify(args.inputs)
    cells = []
    geometry = []
    started = time.monotonic()
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
        affine_weight = mx.array(arrays["affine4_weight"])
        affine_scales = mx.array(arrays["affine4_scales"]).view(mx.bfloat16)
        affine_biases = mx.array(arrays["affine4_biases"]).view(mx.bfloat16)
        mx.eval(
            linear.parameters(), dense, affine_weight, affine_scales, affine_biases
        )

        for rows in ROWS:
            common.require_owner(args.cpg_owner_lock)
            if time.monotonic() - started > 240:
                raise RuntimeError("NVFP4 component matrix exceeded its time bound")
            pre = common.host_snapshot()
            errors = common.admission_errors(
                pre, minimum_available_bytes=8 * common.GIB, performance=True
            )
            if errors:
                raise RuntimeError("cell admission failed: " + "; ".join(errors))
            rng = np.random.default_rng(manifest["seed"] + n + k + rows)
            x = mx.array(rng.normal(size=(rows, k)).astype(np.float32)).astype(
                mx.bfloat16
            )
            dispatch = Counter()

            def stock_native(value, module=linear, counters=dispatch):
                counters["mlx.quantized_matmul.nvfp4.group16"] += 1
                return module(value)

            def serial_b1(
                value, row_count=rows, module=linear, counters=dispatch
            ):
                parts = []
                for index in range(row_count):
                    counters["mlx2_modelopt_nvfp4_b1_0"] += 1
                    part = project_b1(module, value[index : index + 1])
                    mx.eval(part)
                    parts.append(part)
                result = mx.concatenate(parts, axis=0)
                mx.eval(result)
                return result

            def dense_bf16(value, weight=dense, counters=dispatch):
                counters["mlx.matmul.bf16_decoded_published_weight"] += 1
                return value @ weight.T

            def affine4(
                value,
                weight=affine_weight,
                scales=affine_scales,
                biases=affine_biases,
                counters=dispatch,
            ):
                counters["mlx.quantized_matmul.affine4.group64"] += 1
                return mx.quantized_matmul(
                    value,
                    weight,
                    scales=scales,
                    biases=biases,
                    group_size=64,
                    bits=4,
                    mode="affine",
                    transpose=True,
                )

            calls = {
                "stock_native_nvfp4": stock_native,
                "serial_b1_nvfp4": serial_b1,
                "decoded_published_weight_bf16": dense_bf16,
                "activation_weighted_affine4": affine4,
            }
            outputs = {name: call(x) for name, call in calls.items()}
            mx.eval(x, *outputs.values())
            oracle = outputs["serial_b1_nvfp4"]
            oracle_bits = np.asarray(oracle.view(mx.uint16))
            oracle_fp32 = np.asarray(oracle.astype(mx.float32))
            metrics = {}
            for name, output in outputs.items():
                values = np.asarray(output.astype(mx.float32))
                metrics[name] = {
                    "bit_mismatches_vs_serial_b1": int(
                        np.count_nonzero(
                            np.asarray(output.view(mx.uint16)) != oracle_bits
                        )
                    ),
                    "error_vs_serial_b1": error_metrics(values, oracle_fp32),
                    "finite": bool(np.isfinite(values).all()),
                }
            timings = {name: [] for name in calls}
            for _ in range(3):
                for call in calls.values():
                    mx.eval(call(x))
            names = tuple(calls)
            for repetition in range(12):
                for offset in range(len(names)):
                    name = names[(offset + repetition) % len(names)]
                    tick = time.perf_counter_ns()
                    mx.eval(calls[name](x))
                    timings[name].append((time.perf_counter_ns() - tick) * 1e-6)
            medians = {name: _median(values) for name, values in timings.items()}
            baseline = medians["stock_native_nvfp4"]
            post = common.host_snapshot()
            contamination = [
                reason
                for reason in common.contamination(pre, post)
                if reason not in {"swap-outs increased", "swap usage increased"}
            ]
            cell = {
                "matrix": matrix["prefix"],
                "shape": matrix["shape"],
                "rows": rows,
                "rows_are": "token or speculative-verification rows in one conversation",
                "metrics": metrics,
                "timings_ms": timings,
                "median_ms": medians,
                "speedup_vs_stock_native": {
                    name: baseline / value for name, value in medians.items()
                },
                "dispatch_counts": dict(sorted(dispatch.items())),
                "dispatch_identity": {
                    "stock_native_nvfp4": "mlx.quantized_matmul(mode=nvfp4,group_size=16,transpose=True); selected internal Metal symbol unavailable",
                    "serial_b1_nvfp4": "mlx2_modelopt_nvfp4_b1_0; one explicit dispatch per row",
                    "decoded_published_weight_bf16": "mlx.matmul over once-decoded published BF16 weight",
                    "activation_weighted_affine4": "mlx.quantized_matmul(mode=affine,bits=4,group_size=64,transpose=True)",
                },
                "host_before": pre,
                "host_after": post,
                "contamination": contamination,
                "performance_reportable": not contamination,
            }
            cells.append(cell)
            common.atomic_json(
                args.out_dir / "result.json",
                {
                    "schema": "mlx2.nemotron35-modelopt-nvfp4-mlx0324-matrix.v1",
                    "state": "running",
                    "cells": cells,
                },
            )
            if contamination:
                raise RuntimeError("component matrix cell contaminated")
            del x, outputs, oracle
            mx.clear_cache()

        rng = np.random.default_rng(manifest["seed"] + n + k + 44)
        x8 = mx.array(rng.normal(size=(8, k)).astype(np.float32)).astype(mx.bfloat16)

        def stock_4_plus_4(module=linear, inputs=x8):
            left = module(inputs[:4])
            mx.eval(left)
            right = module(inputs[4:])
            mx.eval(right)
            return mx.concatenate((left, right), axis=0)

        def serial_4_plus_4(module=linear, inputs=x8):
            halves = []
            for start in (0, 4):
                parts = []
                for index in range(start, start + 4):
                    part = project_b1(module, inputs[index : index + 1])
                    mx.eval(part)
                    parts.append(part)
                half = mx.concatenate(parts, axis=0)
                mx.eval(half)
                halves.append(half)
            return mx.concatenate(halves, axis=0)

        geometry_calls = {
            "stock_native_4_plus_4": stock_4_plus_4,
            "serial_b1_4_plus_4": serial_4_plus_4,
        }
        geometry_outputs = {name: call() for name, call in geometry_calls.items()}
        mx.eval(*geometry_outputs.values())
        oracle_bits = np.asarray(
            geometry_outputs["serial_b1_4_plus_4"].view(mx.uint16)
        )
        geometry_timings = {name: [] for name in geometry_calls}
        for repetition in range(12):
            names = tuple(geometry_calls)
            for offset in range(len(names)):
                name = names[(offset + repetition) % len(names)]
                tick = time.perf_counter_ns()
                mx.eval(geometry_calls[name]())
                geometry_timings[name].append(
                    (time.perf_counter_ns() - tick) * 1e-6
                )
        geometry.append(
            {
                "matrix": matrix["prefix"],
                "shape": matrix["shape"],
                "geometry": "4+4 speculative verification as two ordered launches",
                "bit_mismatches_vs_serial_b1": {
                    name: int(
                        np.count_nonzero(
                            np.asarray(output.view(mx.uint16)) != oracle_bits
                        )
                    )
                    for name, output in geometry_outputs.items()
                },
                "timings_ms": geometry_timings,
                "median_ms": {
                    name: _median(values) for name, values in geometry_timings.items()
                },
                "dispatch_identity": {
                    "stock_native_4_plus_4": "two ordered mlx.quantized_matmul NVFP4 dispatches",
                    "serial_b1_4_plus_4": "eight ordered mlx2_modelopt_nvfp4_b1_0 dispatches segmented 4+4",
                },
            }
        )
        del arrays, linear, dense, affine_weight, affine_scales, affine_biases
        mx.clear_cache()

    after = common.host_snapshot()
    contamination = [
        reason
        for reason in common.contamination(before, after)
        if reason not in {"swap-outs increased", "swap usage increased"}
    ]
    receipt = {
        "schema": "mlx2.nemotron35-modelopt-nvfp4-mlx0324-matrix.v1",
        "state": "completed",
        "owner": owner,
        "source": common.source_identity(),
        "dependencies": dependencies,
        "runtime_required": "mlx==0.32.4; mlx-metal==0.32.4",
        "inputs": manifest,
        "before": before,
        "after": after,
        "swap_growth_is_not_a_failure": True,
        "cells": cells,
        "speculative_geometry": geometry,
        "serial_b1_is_correctness_oracle": True,
        "methodological_failures": [],
        "contamination": contamination,
        "passed": not contamination
        and all(
            mode["finite"] for cell in cells for mode in cell["metrics"].values()
        ),
        "qualified": False,
        "selected": False,
        "default_changed": False,
        "scope": "Owned real-weight component matrix; no full-model or route claim.",
    }
    common.atomic_json(args.out_dir / "result.json", receipt)
    if contamination:
        raise RuntimeError("NVFP4 component matrix contaminated")


if __name__ == "__main__":
    main()
