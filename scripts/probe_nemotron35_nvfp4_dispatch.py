#!/usr/bin/env python3
"""Owned GPU correctness probe for adapter-local ModelOpt NVFP4 dispatch."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import nemotron35_probe_common as common
import numpy as np


def _mismatches(mx, left, right):
    return int(
        np.count_nonzero(
            np.asarray(left.view(mx.uint16)) != np.asarray(right.view(mx.uint16))
        )
    )


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
        before, minimum_available_bytes=8 * common.GIB, performance=False
    )
    if errors:
        raise RuntimeError("; ".join(errors))

    import mlx.core as mx

    from mlx2.conversion.kernel_probe import error_metrics, verify
    from mlx2.runtime.models.modelopt_nvfp4 import ModelOptNVFP4Linear
    from mlx2.runtime.models.modelopt_nvfp4_b1 import project

    dependencies = common.runtime_dependencies()
    if dependencies["versions"].get("mlx") != "0.32.4":
        raise RuntimeError("This correctness probe is bound to MLX 0.32.4")
    manifest = verify(args.inputs)
    started = time.monotonic()
    matrices = []
    for index, matrix in enumerate(manifest["matrices"]):
        common.require_owner(args.cpg_owner_lock)
        if time.monotonic() - started > 240:
            raise RuntimeError("NVFP4 dispatch probe exceeded its time bound")
        with np.load(args.inputs / matrix["file"], allow_pickle=False) as archive:
            packed = archive["native_weight"]
            scales = archive["native_scales"]
            global_scale = archive["global_scale"]
            dense = archive["dense_fp32"]
        n, k = matrix["shape"]
        linear = ModelOptNVFP4Linear(k, n)
        linear.load_weights(
            [
                ("weight", mx.array(packed)),
                ("scales", mx.array(scales)),
                ("global_scale", mx.array(global_scale)),
            ],
            strict=True,
        )
        rng = np.random.default_rng(manifest["seed"] + index + n + k)
        x8 = mx.array(rng.normal(size=(8, k)).astype(np.float32)).astype(mx.bfloat16)
        x1, x4 = x8[:1], x8[:4]

        # Preparing derived data must not silently change the default route.
        native_before = linear(x4)
        required = linear.prepare_decoded_weight(
            budget_bytes=linear.decoded_weight_bytes
        )
        linear.set_dispatch_policy("native", large_row_threshold=4)
        native_after = linear(x4)
        mx.eval(native_before, native_after)
        native_default_mismatches = _mismatches(mx, native_before, native_after)

        linear.set_dispatch_policy("auto", large_row_threshold=4)
        decoded = linear(x4)
        decoded_reference = x4 @ linear._decoded_weight.T
        mx.eval(decoded, decoded_reference)
        decoded_dispatch_mismatches = _mismatches(mx, decoded, decoded_reference)

        linear.set_dispatch_policy("fused_b1")
        fused = linear(x1)
        fused_reference = project(linear, x1)
        mx.eval(fused, fused_reference)
        fused_dispatch_mismatches = _mismatches(mx, fused, fused_reference)

        serial = []
        segmented = []
        for start, stop in ((0, 4), (4, 8)):
            half = []
            for row in range(start, stop):
                direct = project(linear, x8[row : row + 1])
                routed = linear(x8[row : row + 1])
                mx.eval(direct, routed)
                serial.append(direct)
                half.append(routed)
            segmented.append(mx.concatenate(half, axis=0))
        serial = mx.concatenate(serial, axis=0)
        segmented = mx.concatenate(segmented, axis=0)
        mx.eval(serial, segmented)
        segmented_mismatches = _mismatches(mx, serial, segmented)

        dense_reference = x1.astype(mx.float32) @ mx.array(dense).T
        mx.eval(dense_reference)
        dense_metrics = error_metrics(
            np.asarray(fused.astype(mx.float32)), np.asarray(dense_reference)
        )
        status = linear.dispatch_status()
        passed = not any(
            (
                native_default_mismatches,
                decoded_dispatch_mismatches,
                fused_dispatch_mismatches,
                segmented_mismatches,
            )
        ) and np.isfinite(np.asarray(fused.astype(mx.float32))).all()
        matrices.append(
            {
                "matrix": matrix["prefix"],
                "shape": matrix["shape"],
                "decoded_weight_bytes": required,
                "native_default_mismatches_after_cache_prepare": native_default_mismatches,
                "decoded_dispatch_mismatches_vs_cached_bf16_matmul": decoded_dispatch_mismatches,
                "fused_b1_dispatch_mismatches_vs_direct_kernel": fused_dispatch_mismatches,
                "segmented_4_plus_4_mismatches_vs_serial_b1": segmented_mismatches,
                "fused_b1_error_vs_decoded_published_fp32_weight": dense_metrics,
                "dispatch": status,
                "passed": bool(passed),
            }
        )
        linear.release_decoded_weight()
        del linear, x8, x1, x4, dense_reference
        mx.clear_cache()

    after = common.host_snapshot()
    contamination = [
        reason
        for reason in common.contamination(before, after)
        if reason not in {"swap-outs increased", "swap usage increased"}
    ]
    receipt = {
        "schema": "mlx2.nemotron35-modelopt-nvfp4-dispatch-correctness.v1",
        "state": "completed",
        "owner": owner,
        "source": common.source_identity(),
        "dependencies": dependencies,
        "runtime_required": "mlx==0.32.4; mlx-metal==0.32.4",
        "inputs": manifest,
        "before": before,
        "after": after,
        "swap_growth_is_not_a_failure": True,
        "matrices": matrices,
        "contamination": contamination,
        "passed": not contamination and all(item["passed"] for item in matrices),
        "qualified": False,
        "selected": False,
        "default_changed": False,
        "scope": "Real-weight component dispatch only; no full-model or route claim.",
    }
    common.atomic_json(args.out_dir / "result.json", receipt)
    if not receipt["passed"]:
        raise RuntimeError("NVFP4 dispatch correctness probe failed")


if __name__ == "__main__":
    main()
