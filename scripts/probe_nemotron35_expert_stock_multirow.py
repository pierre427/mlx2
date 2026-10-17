#!/usr/bin/env python3
"""Probe stock multirow Nemotron MoE against exact serial B1 calls."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--settle-seconds", type=int, default=60)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        raise SystemExit("refusing Metal execution without --i-own-the-gpu")
    if not 1 <= args.reps <= 10:
        raise ValueError("reps must be in 1..10")

    import mlx.core as mx
    import nemotron35_probe_common as common

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.models.nemotron_h import NemotronHMoE

    owner_before = common.require_owner(args.cpg_owner_lock)
    host_before = common.host_snapshot()
    errors = common.admission_errors(
        host_before, minimum_available_bytes=4 * common.GIB, performance=True
    )
    if errors:
        raise RuntimeError(f"stock multirow admission failed: {errors}")

    adapter = resolve_adapter(args.model)(str(args.model))
    model = adapter.model
    time.sleep(args.settle_seconds)
    layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, NemotronHMoE) and name.startswith("backbone.layers.")
    ]
    if len(layers) != 40:
        raise RuntimeError(f"expected 40 Nemotron MoE layers, found {len(layers)}")

    mx.random.seed(20261016)
    cases = []
    for rows in (2, 4, 8, 16):
        x = mx.random.normal((1, rows, 4096)).astype(mx.bfloat16)

        def compare(reference, candidate):
            mx.eval(reference, candidate)
            return {
                "mismatches": int(mx.sum(reference != candidate).item()),
                "max_abs_delta": float(mx.max(mx.abs(reference - candidate)).item()),
            }

        def serial_array(function, value, row_count=rows):
            return mx.concatenate(
                [
                    function(value[:, row : row + 1])
                    for row in range(row_count)
                ],
                axis=1,
            )

        def serial_router(gate, value, row_count=rows):
            selected = [
                gate(value[:, row : row + 1]) for row in range(row_count)
            ]
            return (
                mx.concatenate([item[0] for item in selected], axis=1),
                mx.concatenate([item[1] for item in selected], axis=1),
            )

        def serial(module, row_inputs=x, row_count=rows):
            return mx.concatenate(
                [
                    module(row_inputs[:, row : row + 1])
                    for row in range(row_count)
                ],
                axis=1,
            )

        def batched(module, row_inputs=x):
            return module(row_inputs)

        # Compile both shapes on the first real layer outside retained samples.
        mx.eval(serial(layers[0][1]), batched(layers[0][1]))
        mismatched_layers = []
        routed_batch_mismatched_layers = []
        total_mismatches = 0
        routed_batch_total_mismatches = 0
        max_abs_delta = 0.0
        routed_batch_max_abs_delta = 0.0
        stages = {
            name: {"mismatches": 0, "max_abs_delta": 0.0, "layers": 0}
            for name in (
                "router_indices",
                "router_scores",
                "latent_in",
                "routed_experts",
                "shared_expert",
                "latent_out",
            )
        }
        for name, module in layers:
            reference = serial(module)
            candidate = batched(module)
            mx.eval(reference, candidate)
            mismatch = int(mx.sum(reference != candidate).item())
            delta = float(mx.max(mx.abs(reference - candidate)).item())
            total_mismatches += mismatch
            max_abs_delta = max(max_abs_delta, delta)
            if mismatch:
                mismatched_layers.append(
                    {"name": name, "mismatches": mismatch, "max_abs_delta": delta}
                )
            routed_batch = module.routed_batch_exact(x)
            mx.eval(reference, routed_batch)
            routed_batch_mismatch = int(mx.sum(reference != routed_batch).item())
            routed_batch_delta = float(mx.max(mx.abs(reference - routed_batch)).item())
            routed_batch_total_mismatches += routed_batch_mismatch
            routed_batch_max_abs_delta = max(
                routed_batch_max_abs_delta, routed_batch_delta
            )
            if routed_batch_mismatch:
                routed_batch_mismatched_layers.append(
                    {
                        "name": name,
                        "mismatches": routed_batch_mismatch,
                        "max_abs_delta": routed_batch_delta,
                    }
                )

            serial_indices, serial_scores = serial_router(module.gate, x)
            batch_indices, batch_scores = module.gate(x)
            stage_rows = {
                "router_indices": compare(serial_indices, batch_indices),
                "router_scores": compare(serial_scores, batch_scores),
            }
            latent_serial = serial_array(module.fc1_latent_proj, x)
            latent_batch = module.fc1_latent_proj(x)
            stage_rows["latent_in"] = compare(latent_serial, latent_batch)
            # Recompute the serial routed result with each row's own indices;
            # the explicit loop avoids accidentally broadcasting row zero.
            routed_serial = mx.concatenate(
                [
                    module.switch_mlp(
                        latent_serial[:, row : row + 1],
                        serial_indices[:, row : row + 1],
                    )
                    for row in range(rows)
                ],
                axis=1,
            )
            routed_batch = module.switch_mlp(latent_serial, serial_indices)
            stage_rows["routed_experts"] = compare(routed_serial, routed_batch)
            shared_serial = serial_array(module.shared_experts, x)
            shared_batch = module.shared_experts(x)
            stage_rows["shared_expert"] = compare(shared_serial, shared_batch)
            latent_out_input = mx.random.normal((1, rows, 1024)).astype(mx.bfloat16)
            out_serial = serial_array(module.fc2_latent_proj, latent_out_input)
            out_batch = module.fc2_latent_proj(latent_out_input)
            stage_rows["latent_out"] = compare(out_serial, out_batch)
            for stage_name, result in stage_rows.items():
                stages[stage_name]["mismatches"] += result["mismatches"]
                stages[stage_name]["max_abs_delta"] = max(
                    stages[stage_name]["max_abs_delta"], result["max_abs_delta"]
                )
                stages[stage_name]["layers"] += int(result["mismatches"] > 0)

        serial_samples = []
        batched_samples = []
        routed_batch_samples = []
        for rep in range(args.reps):
            order = layers if rep % 2 == 0 else list(reversed(layers))
            started = time.perf_counter()
            outputs = [serial(module) for _, module in order]
            mx.eval(*outputs)
            serial_samples.append(time.perf_counter() - started)
            started = time.perf_counter()
            outputs = [batched(module) for _, module in reversed(order)]
            mx.eval(*outputs)
            batched_samples.append(time.perf_counter() - started)
            started = time.perf_counter()
            outputs = [module.routed_batch_exact(x) for _, module in order]
            mx.eval(*outputs)
            routed_batch_samples.append(time.perf_counter() - started)
        serial_ms = 1000 * statistics.median(serial_samples)
        batched_ms = 1000 * statistics.median(batched_samples)
        routed_batch_ms = 1000 * statistics.median(routed_batch_samples)
        cases.append(
            {
                "rows": rows,
                "layers": len(layers),
                "total_mismatches": total_mismatches,
                "mismatched_layer_count": len(mismatched_layers),
                "mismatched_layers": mismatched_layers,
                "routed_batch_total_mismatches": routed_batch_total_mismatches,
                "routed_batch_mismatched_layer_count": len(
                    routed_batch_mismatched_layers
                ),
                "routed_batch_mismatched_layers": routed_batch_mismatched_layers,
                "routed_batch_max_abs_delta": routed_batch_max_abs_delta,
                "stage_exactness": stages,
                "max_abs_delta": max_abs_delta,
                "serial_b1_ms": serial_ms,
                "stock_multirow_ms": batched_ms,
                "speedup": serial_ms / batched_ms,
                "routed_batch_exact_ms": routed_batch_ms,
                "routed_batch_exact_speedup": serial_ms / routed_batch_ms,
            }
        )
        mx.clear_cache()

    report = {
        "schema": "mlx2.nemotron35-expert-stock-multirow.v1",
        "model": str(args.model.resolve()),
        "cases": cases,
        "stock_multirow_all_exact": all(
            case["total_mismatches"] == 0 for case in cases
        ),
        "routed_batch_all_exact": all(
            case["routed_batch_total_mismatches"] == 0 for case in cases
        ),
        "owner_before": owner_before,
        "owner_after": common.require_owner(args.cpg_owner_lock),
        "host_before": host_before,
        "host_after": common.host_snapshot(),
        "limitations": [
            "random BF16 component inputs rather than captured natural activations",
            "cycles all 40 real expert layers to avoid one cache-hot weight matrix",
            "component smoke only; no scheduler or full-model route claim",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
