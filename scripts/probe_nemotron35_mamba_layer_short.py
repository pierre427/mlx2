#!/usr/bin/env python3
"""Owned real-layer exact/timing probe for short Nemotron Mamba segments."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import nemotron35_probe_common as common
import numpy as np


def _bits(value, mx):
    return np.asarray(value.view(mx.uint16 if value.dtype != mx.float32 else mx.uint32))


def _mismatches(actual, expected, mx):
    return int(np.count_nonzero(_bits(actual, mx) != _bits(expected, mx)))


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

    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.models.cache import ArraysCache

    policy = {
        "mamba_conv": "fused_short",
        "mamba_norm": "fused_short",
        "mamba_ssm": "fused_short_gate",
        "mamba_projection": "short_row_exact",
    }
    adapter = resolve_adapter(args.model)(str(args.model), execution_policy=policy)
    time.sleep(args.settle_seconds)
    layer_index, mixer = next(
        (index, layer.mixer)
        for index, layer in enumerate(adapter.model.layers)
        if getattr(layer, "block_type", None) == "M"
    )
    rng = np.random.default_rng(3518)
    initial_conv = mx.array(rng.normal(size=(1, 3, 10240)).astype(np.float32)).astype(
        mx.bfloat16
    )
    initial_state = mx.array(
        (rng.normal(size=(1, 128, 64, 128)) * 0.02).astype(np.float32)
    )
    mx.eval(initial_conv, initial_state)

    def make_cache():
        cache = ArraysCache(size=2)
        cache[0] = mx.array(initial_conv)
        cache[1] = mx.array(initial_state)
        return cache

    def serial(value, keep=None):
        cache = make_cache()
        outputs = []
        count = int(value.shape[1]) if keep is None else int(keep)
        for step in range(count):
            output = mixer(value[:, step : step + 1], None, cache)
            mx.eval(output, cache[0], cache[1])
            outputs.append(output)
        if not outputs:
            return None, cache
        result = mx.concatenate(outputs, axis=1)
        mx.eval(result)
        return result, cache

    def short(value, *, speculating=False):
        cache = make_cache()
        if speculating:
            cache.start_speculation()
        output = mixer(value, None, cache)
        mx.eval(output, cache[0], cache[1])
        return output, cache

    cells = []
    for steps in (2, 4, 8, 16):
        common.require_owner(args.cpg_owner_lock)
        x = mx.array(rng.normal(size=(1, steps, 4096)).astype(np.float32)).astype(
            mx.bfloat16
        )
        mx.eval(x)
        expected, expected_cache = serial(x)
        actual, actual_cache = short(x)
        output_mismatches = _mismatches(actual, expected, mx)
        conv_mismatches = _mismatches(actual_cache[0], expected_cache[0], mx)
        state_mismatches = _mismatches(actual_cache[1], expected_cache[1], mx)

        rollback = []
        for keep in range(steps + 1):
            _unused, reference = serial(x, keep=keep)
            speculative_output, candidate = short(x, speculating=True)
            mx.eval(speculative_output, candidate[0], candidate[1])
            removed = candidate.trim(steps - keep)
            candidate.stop_speculation()
            mx.eval(candidate[0], candidate[1])
            rollback.append(
                {
                    "keep": keep,
                    "removed": int(removed),
                    "conv_mismatches": _mismatches(candidate[0], reference[0], mx),
                    "state_mismatches": _mismatches(candidate[1], reference[1], mx),
                }
            )

        calls = {
            "serial_b1": lambda value=x: serial(value)[0],
            "short_exact": lambda value=x: short(value)[0],
        }
        for _ in range(3):
            for call in calls.values():
                mx.eval(call())
        timings = {name: [] for name in calls}
        for repetition in range(10):
            order = tuple(calls) if repetition % 2 == 0 else tuple(reversed(calls))
            for name in order:
                started = time.perf_counter_ns()
                mx.eval(calls[name]())
                timings[name].append((time.perf_counter_ns() - started) * 1e-6)
        medians = {name: float(np.median(values)) for name, values in timings.items()}
        cells.append(
            {
                "steps": steps,
                "output_mismatches": output_mismatches,
                "conv_state_mismatches": conv_mismatches,
                "ssm_state_mismatches": state_mismatches,
                "rollback": rollback,
                "median_ms": medians,
                "speedup_vs_serial_b1": medians["serial_b1"] / medians["short_exact"],
                "timings_ms": timings,
            }
        )
        common.atomic_json(
            args.out_dir / "result.json",
            {
                "schema": "mlx2.nemotron35-real-mamba-layer-short.v1",
                "state": "running",
                "cells": cells,
            },
        )
        del x, expected, expected_cache, actual, actual_cache
        mx.clear_cache()

    status = {
        "conv": mixer.conv_status(),
        "ssm": mixer.ssm_status(),
        "norm": mixer.norm.status(),
        "projection": mixer.projection_status(),
    }
    del adapter
    gc.collect()
    mx.clear_cache()
    after = common.host_snapshot()
    contamination = [
        reason
        for reason in common.contamination(before, after)
        if reason != "swap-outs increased"
    ]
    exact = all(
        cell["output_mismatches"] == 0
        and cell["conv_state_mismatches"] == 0
        and cell["ssm_state_mismatches"] == 0
        and all(
            row["removed"] == cell["steps"] - row["keep"]
            and row["conv_mismatches"] == 0
            and row["state_mismatches"] == 0
            for row in cell["rollback"]
        )
        for cell in cells
    )
    receipt = {
        "schema": "mlx2.nemotron35-real-mamba-layer-short.v1",
        "state": "completed",
        "owner": owner,
        "source": common.source_identity(),
        "dependencies": common.runtime_dependencies(),
        "model": str(args.model.resolve()),
        "layer_index": layer_index,
        "policy": policy,
        "settle_seconds": args.settle_seconds,
        "before": before,
        "after": after,
        "swap_ignored_by_user": True,
        "cells": cells,
        "candidate_status": status,
        "all_exact": exact,
        "contamination": contamination,
        "passed": exact and not contamination,
        "qualified": False,
        "scope": "One real quantized Mamba layer; no full-route claim.",
    }
    common.atomic_json(args.out_dir / "result.json", receipt)
    if contamination:
        raise RuntimeError("real Mamba layer probe contaminated")
    if not exact:
        raise RuntimeError("real Mamba short layer differed from serial B1")


if __name__ == "__main__":
    main()
