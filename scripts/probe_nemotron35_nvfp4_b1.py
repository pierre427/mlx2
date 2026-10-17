#!/usr/bin/env python3
"""Owned real-weight parity/timing probe for the fused NVFP4 B1 candidate."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import nemotron35_probe_common as common
import numpy as np


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

    from mlx2.conversion.kernel_probe import verify
    from mlx2.runtime.models.modelopt_nvfp4 import ModelOptNVFP4Linear
    from mlx2.runtime.models.modelopt_nvfp4_b1 import project

    manifest = verify(args.inputs)
    cases = []
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
        rng = np.random.default_rng(manifest["seed"] + n + k)
        x = mx.array(rng.normal(size=(1, k)).astype(np.float32)).astype(mx.bfloat16)
        ordinary = linear(x)
        fused = project(linear, x)
        mx.eval(ordinary, fused)
        ordinary_bits = np.asarray(ordinary.view(mx.uint16))
        fused_bits = np.asarray(fused.view(mx.uint16))
        mismatch = int(np.count_nonzero(ordinary_bits != fused_bits))
        timings = {"ordinary": [], "fused": []}
        calls = {
            "ordinary": linear,
            "fused": lambda value, module=linear: project(module, value),
        }
        for _ in range(4):
            for call in calls.values():
                mx.eval(call(x))
        for repetition in range(20):
            for name in (("ordinary", "fused") if repetition % 2 == 0 else ("fused", "ordinary")):
                tick = time.perf_counter_ns()
                mx.eval(calls[name](x))
                timings[name].append((time.perf_counter_ns() - tick) * 1e-6)
        medians = {name: float(np.median(values)) for name, values in timings.items()}
        cases.append(
            {
                "matrix": matrix["prefix"],
                "shape": matrix["shape"],
                "mismatch_count": mismatch,
                "exact": mismatch == 0,
                "timings_ms": timings,
                "median_ms": medians,
                "speedup": medians["ordinary"] / medians["fused"],
            }
        )
        del arrays, linear, x, ordinary, fused
        mx.clear_cache()

    after = common.host_snapshot()
    contamination = [
        reason
        for reason in common.contamination(before, after)
        if reason != "swap-outs increased"
    ]
    receipt = {
        "schema": "mlx2.nemotron35-modelopt-nvfp4-b1-component.v1",
        "owner": owner,
        "source": common.source_identity(),
        "dependencies": common.runtime_dependencies(),
        "inputs": manifest,
        "before": before,
        "after": after,
        "swap_ignored_by_user": True,
        "cases": cases,
        "all_exact": all(case["exact"] for case in cases),
        "all_faster": all(case["speedup"] > 1.0 for case in cases),
        "contamination": contamination,
        "passed": not contamination,
        "qualified": False,
        "scope": "Isolated B1 real-weight component; no serving or route claim.",
    }
    common.atomic_json(args.out_dir / "result.json", receipt)
    if contamination:
        raise RuntimeError("NVFP4 B1 component probe contaminated")


if __name__ == "__main__":
    main()
