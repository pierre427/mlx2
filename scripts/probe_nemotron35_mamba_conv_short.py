#!/usr/bin/env python3
"""Owned exactness/timing probe for short Nemotron conv4 segments."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import mlx.core as mx
import nemotron35_probe_common as common

from mlx2.runtime.models.nemotron_mamba_conv import (
    fused_mamba_conv_b1,
    fused_mamba_conv_short,
)


def mismatch(a, b) -> int:
    return int(mx.sum(a.view(mx.uint16) != b.view(mx.uint16)).item())


def serial_b1(conv_input, state, weight, bias):
    outputs = []
    checkpoints = [state]
    current = state
    for step in range(int(conv_input.shape[1])):
        output, current = fused_mamba_conv_b1(
            conv_input[:, step : step + 1],
            current,
            weight,
            bias,
            capture_window=False,
        )
        mx.eval(output, current)
        outputs.append(output)
        checkpoints.append(current)
    result = mx.concatenate(outputs, axis=1)
    history = mx.stack(checkpoints, axis=1)
    mx.eval(result, history)
    return result, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--timing-reps", type=int, default=40)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Refusing to overwrite nonempty probe evidence")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    owner = common.require_owner(args.cpg_owner_lock)
    before = common.host_snapshot()
    errors = common.admission_errors(before, minimum_available_bytes=4 * common.GIB)
    if errors:
        raise RuntimeError("; ".join(errors))

    channels = 10240
    cases = []
    for steps in (1, 2, 4, 8, 16):
        common.require_owner(args.cpg_owner_lock)
        mx.random.seed(5100 + steps)
        conv_input = mx.random.normal((1, steps, channels)).astype(mx.bfloat16)
        state = mx.random.normal((1, 3, channels)).astype(mx.bfloat16)
        weight = mx.random.normal((channels, 4, 1)).astype(mx.bfloat16)
        bias = mx.random.normal((channels,)).astype(mx.bfloat16)
        mx.eval(conv_input, state, weight, bias)

        expected, expected_history = serial_b1(conv_input, state, weight, bias)
        actual, checkpoints = fused_mamba_conv_short(
            conv_input, state, weight, bias, capture_checkpoints=True
        )
        compact, final_state = fused_mamba_conv_short(
            conv_input, state, weight, bias, capture_checkpoints=False
        )
        mx.eval(actual, checkpoints, compact, final_state)

        segment_outputs = []
        segment_checkpoints = [state]
        current = state
        for start in range(0, steps, 4):
            segment, history = fused_mamba_conv_short(
                conv_input[:, start : start + 4],
                current,
                weight,
                bias,
                capture_checkpoints=True,
            )
            mx.eval(segment, history)
            segment_outputs.append(segment)
            segment_checkpoints.extend(
                history[:, index] for index in range(1, int(history.shape[1]))
            )
            current = history[:, -1]
        segmented = mx.concatenate(segment_outputs, axis=1)
        segmented_history = mx.stack(segment_checkpoints, axis=1)
        mx.eval(segmented, segmented_history)

        calls = {
            "serial_b1": lambda ci=conv_input, st=state, w=weight, b=bias: serial_b1(
                ci, st, w, b
            ),
            "short_checkpoints": lambda ci=conv_input, st=state, w=weight, b=bias: (
                fused_mamba_conv_short(ci, st, w, b, capture_checkpoints=True)
            ),
            "short_final_only": lambda ci=conv_input, st=state, w=weight, b=bias: (
                fused_mamba_conv_short(ci, st, w, b, capture_checkpoints=False)
            ),
        }
        for _ in range(5):
            for call in calls.values():
                mx.eval(*call())
        timings = {name: [] for name in calls}
        names = tuple(calls)
        for repetition in range(args.timing_reps):
            for offset in range(len(names)):
                name = names[(offset + repetition) % len(names)]
                started = time.perf_counter_ns()
                mx.eval(*calls[name]())
                timings[name].append((time.perf_counter_ns() - started) * 1e-6)
        medians = {
            name: float(statistics.median(values)) for name, values in timings.items()
        }
        cases.append(
            {
                "steps": steps,
                "output_mismatches": mismatch(expected, actual),
                "checkpoint_mismatches": mismatch(expected_history, checkpoints),
                "compact_output_mismatches": mismatch(expected, compact),
                "compact_final_state_mismatches": mismatch(
                    expected_history[:, -1], final_state
                ),
                "segmented_output_mismatches": mismatch(expected, segmented),
                "segmented_checkpoint_mismatches": mismatch(
                    expected_history, segmented_history
                ),
                "median_ms": medians,
                "speedup_vs_serial_b1": {
                    name: medians["serial_b1"] / value
                    for name, value in medians.items()
                },
                "timings_ms": timings,
            }
        )
        del (
            conv_input,
            state,
            weight,
            bias,
            expected,
            expected_history,
            actual,
            checkpoints,
            compact,
            final_state,
            segmented,
            segmented_history,
        )
        mx.clear_cache()

    after = common.host_snapshot()
    contamination = [
        reason
        for reason in common.contamination(before, after)
        if reason != "swap-outs increased"
    ]
    exact_fields = (
        "output_mismatches",
        "checkpoint_mismatches",
        "compact_output_mismatches",
        "compact_final_state_mismatches",
        "segmented_output_mismatches",
        "segmented_checkpoint_mismatches",
    )
    passed = (
        all(all(case[field] == 0 for field in exact_fields) for case in cases)
        and not contamination
    )
    receipt = {
        "schema": "mlx2.nemotron35-mamba-conv4-short.v1",
        "owner": owner,
        "source": common.source_identity(),
        "dependencies": common.runtime_dependencies(),
        "before": before,
        "after": after,
        "swap_ignored_by_user": True,
        "channels": channels,
        "cases": cases,
        "contamination": contamination,
        "passed": passed,
        "qualified": False,
        "scope": "Synthetic exact Super geometry; no served-route claim.",
    }
    common.atomic_json(args.out_dir / "result.json", receipt)
    if not passed:
        raise RuntimeError("Mamba short conv exactness probe failed")


if __name__ == "__main__":
    main()
