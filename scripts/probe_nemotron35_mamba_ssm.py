#!/usr/bin/env python3
"""Owned bitwise oracle for the Super short-SSM in-kernel dt candidate."""
from __future__ import annotations

import argparse
import hashlib
import statistics
import time
from pathlib import Path

import mlx.core as mx
import nemotron35_probe_common as common
from mlx import nn

from mlx2.runtime.models.nemotron_mamba_ssm import (
    fused_mamba_ssm_gate_short,
    fused_mamba_ssm_short,
)
from mlx2.runtime.models.ssm import ssm_update


def bits_sha(value) -> str:
    array = mx.array(value)
    view = array.view(mx.uint16 if array.dtype == mx.bfloat16 else mx.uint32)
    return hashlib.sha256(bytes(memoryview(view))).hexdigest()


def exact(a, b) -> tuple[bool, int]:
    view = mx.uint16 if a.dtype == mx.bfloat16 else mx.uint32
    mismatch = int(mx.sum(a.view(view) != b.view(view)).item())
    return mismatch == 0, mismatch


def serial_b1_ssm(x, A_log, B, C, D, dt, dt_bias, state):
    """Materialized tokenwise B1 record/replay oracle."""
    outputs = []
    current = state
    for step in range(int(x.shape[1])):
        output, current = ssm_update(
            x[:, step : step + 1],
            A_log,
            B[:, step : step + 1],
            C[:, step : step + 1],
            D,
            dt[:, step : step + 1],
            dt_bias,
            current,
            (0.0, float("inf")),
        )
        mx.eval(output, current)
        outputs.append(output)
    result = mx.concatenate(outputs, axis=1)
    mx.eval(result, current)
    return result, current


def group_rms(gated, weight):
    grouped = gated.reshape(*gated.shape[:-1], 8, 1024)
    normalized = mx.fast.rms_norm(grouped, weight=None, eps=1e-5)
    return weight * normalized.flatten(-2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--timing-reps", type=int, default=40)
    args = parser.parse_args()
    if not 10 <= args.timing_reps <= 200:
        parser.error("--timing-reps must be between 10 and 200")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Refusing to overwrite nonempty probe evidence")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    owner = common.require_owner(args.cpg_owner_lock)
    before = common.host_snapshot()
    errors = common.admission_errors(before, minimum_available_bytes=6 * common.GIB)
    if errors:
        raise RuntimeError("; ".join(errors))

    cases = []
    for steps in range(1, 17):
        for seed, scale in ((101 + steps, 0.2), (211 + steps, 2.0)):
            mx.random.seed(seed)
            x = (mx.random.normal((1, steps, 128, 64)) * scale).astype(mx.bfloat16)
            B = (mx.random.normal((1, steps, 8, 128)) * 0.1).astype(mx.bfloat16)
            C = (mx.random.normal((1, steps, 8, 128)) * 0.1).astype(mx.bfloat16)
            D = (1 + 0.1 * mx.random.normal((128,))).astype(mx.bfloat16)
            dt = (mx.random.normal((1, steps, 128)) * scale).astype(mx.bfloat16)
            dt_bias = mx.random.normal((128,)).astype(mx.bfloat16)
            A_log = mx.log(mx.arange(1, 129, dtype=mx.float32)).astype(mx.bfloat16)
            state = (mx.random.normal((1, 128, 64, 128)) * 0.02).astype(mx.float32)
            gate = mx.random.normal((1, steps, 8192)).astype(mx.bfloat16)
            weight = mx.random.normal((8192,)).astype(mx.bfloat16)
            mx.eval(x, B, C, D, dt, dt_bias, A_log, state, gate, weight)
            expected = serial_b1_ssm(x, A_log, B, C, D, dt, dt_bias, state)
            bulk = ssm_update(
                x, A_log, B, C, D, dt, dt_bias, state, (0.0, float("inf"))
            )
            actual = fused_mamba_ssm_short(
                x, A_log, B, C, D, dt, dt_bias, state, (0.0, float("inf"))
            )
            expected_gated = nn.silu(gate) * expected[0].reshape(1, steps, 8192)
            actual_gated, actual_gate_state = fused_mamba_ssm_gate_short(
                x, A_log, B, C, D, dt, dt_bias, state, gate,
                (0.0, float("inf")),
            )
            actual_gated = actual_gated.reshape(1, steps, 8192)
            expected_norm = group_rms(expected_gated, weight)
            actual_norm = group_rms(actual_gated, weight)
            mx.eval(
                *expected, *bulk, *actual, expected_gated, actual_gated,
                actual_gate_state, expected_norm, actual_norm,
            )
            output_exact, output_mismatch = exact(expected[0], actual[0])
            state_exact, state_mismatch = exact(expected[1], actual[1])
            gated_exact, gated_mismatch = exact(expected_gated, actual_gated)
            gate_state_exact, gate_state_mismatch = exact(
                expected[1], actual_gate_state
            )
            norm_exact, norm_mismatch = exact(expected_norm, actual_norm)
            bulk_output_exact, bulk_output_mismatch = exact(expected[0], bulk[0])
            bulk_state_exact, bulk_state_mismatch = exact(expected[1], bulk[1])
            cases.append({
                "steps": steps, "seed": seed, "scale": scale,
                "output_exact": output_exact,
                "output_mismatch_count": output_mismatch,
                "state_exact": state_exact,
                "state_mismatch_count": state_mismatch,
                "gated_output_exact": gated_exact,
                "gated_output_mismatch_count": gated_mismatch,
                "gate_state_exact": gate_state_exact,
                "gate_state_mismatch_count": gate_state_mismatch,
                "group_rms_output_exact": norm_exact,
                "group_rms_output_mismatch_count": norm_mismatch,
                "bulk_vs_serial_output_exact": bulk_output_exact,
                "bulk_vs_serial_output_mismatch_count": bulk_output_mismatch,
                "bulk_vs_serial_state_exact": bulk_state_exact,
                "bulk_vs_serial_state_mismatch_count": bulk_state_mismatch,
                "expected_output_sha256": bits_sha(expected[0]),
                "actual_output_sha256": bits_sha(actual[0]),
                "expected_state_sha256": bits_sha(expected[1]),
                "actual_state_sha256": bits_sha(actual[1]),
            })
            del (
                x, B, C, D, dt, dt_bias, A_log, state, gate, weight,
                expected, bulk, actual, expected_gated, actual_gated,
                actual_gate_state, expected_norm, actual_norm,
            )
            mx.clear_cache()

    # One shared S=16 block, sliced at every supported prefix, proves that the
    # candidate reconstructs the same accepted-prefix state rather than only
    # matching independently generated widths.
    mx.random.seed(409)
    width = 16
    x = mx.random.normal((1, width, 128, 64)).astype(mx.bfloat16)
    B = (mx.random.normal((1, width, 8, 128)) * 0.1).astype(mx.bfloat16)
    C = (mx.random.normal((1, width, 8, 128)) * 0.1).astype(mx.bfloat16)
    D = mx.random.normal((128,)).astype(mx.bfloat16)
    dt = mx.random.normal((1, width, 128)).astype(mx.bfloat16)
    dt_bias = mx.random.normal((128,)).astype(mx.bfloat16)
    A_log = mx.log(mx.arange(1, 129, dtype=mx.float32)).astype(mx.bfloat16)
    state = (mx.random.normal((1, 128, 64, 128)) * 0.02).astype(mx.float32)
    gate = mx.random.normal((1, width, 8192)).astype(mx.bfloat16)
    weight = mx.random.normal((8192,)).astype(mx.bfloat16)
    mx.eval(x, B, C, D, dt, dt_bias, A_log, state, gate, weight)
    prefix_cases = []
    for keep in range(1, width + 1):
        expected = serial_b1_ssm(
            x[:, :keep], A_log, B[:, :keep], C[:, :keep], D,
            dt[:, :keep], dt_bias, state,
        )
        actual = fused_mamba_ssm_short(
            x[:, :keep], A_log, B[:, :keep], C[:, :keep], D,
            dt[:, :keep], dt_bias, state, (0.0, float("inf")),
        )
        expected_gated = nn.silu(gate[:, :keep]) * expected[0].reshape(
            1, keep, 8192
        )
        actual_gated, actual_gate_state = fused_mamba_ssm_gate_short(
            x[:, :keep], A_log, B[:, :keep], C[:, :keep], D,
            dt[:, :keep], dt_bias, state, gate[:, :keep],
            (0.0, float("inf")),
        )
        actual_gated = actual_gated.reshape(1, keep, 8192)
        expected_norm = group_rms(expected_gated, weight)
        actual_norm = group_rms(actual_gated, weight)
        mx.eval(
            *expected, *actual, expected_gated, actual_gated,
            actual_gate_state, expected_norm, actual_norm,
        )
        output_exact, output_mismatch = exact(expected[0], actual[0])
        state_exact, state_mismatch = exact(expected[1], actual[1])
        gated_exact, gated_mismatch = exact(expected_gated, actual_gated)
        gate_state_exact, gate_state_mismatch = exact(
            expected[1], actual_gate_state
        )
        norm_exact, norm_mismatch = exact(expected_norm, actual_norm)
        prefix_cases.append({
            "keep": keep,
            "output_exact": output_exact,
            "output_mismatch_count": output_mismatch,
            "state_exact": state_exact,
            "state_mismatch_count": state_mismatch,
            "gated_output_exact": gated_exact,
            "gated_output_mismatch_count": gated_mismatch,
            "gate_state_exact": gate_state_exact,
            "gate_state_mismatch_count": gate_state_mismatch,
            "group_rms_output_exact": norm_exact,
            "group_rms_output_mismatch_count": norm_mismatch,
        })

    # Diagnostic component timing for the ordinary B1 decode geometry.  Both
    # arms consume the same already-materialized operands and evaluate both
    # output and FP32 state.  Alternate order to reduce monotonic thermal bias;
    # this is bounded component evidence, not route performance qualification.
    def ordinary_b1():
        return ssm_update(
            x[:, :1], A_log, B[:, :1], C[:, :1], D,
            dt[:, :1], dt_bias, state, (0.0, float("inf")),
        )

    def fused_b1():
        return fused_mamba_ssm_short(
            x[:, :1], A_log, B[:, :1], C[:, :1], D,
            dt[:, :1], dt_bias, state, (0.0, float("inf")),
        )

    def ordinary_gate_norm_b1():
        y, new_state = ordinary_b1()
        gated = nn.silu(gate[:, :1]) * y.reshape(1, 1, 8192)
        return group_rms(gated, weight), new_state

    def fused_gate_norm_b1():
        gated, new_state = fused_mamba_ssm_gate_short(
            x[:, :1], A_log, B[:, :1], C[:, :1], D,
            dt[:, :1], dt_bias, state, gate[:, :1],
            (0.0, float("inf")),
        )
        return group_rms(gated.reshape(1, 1, 8192), weight), new_state

    for function in (ordinary_b1, fused_b1):
        for _ in range(5):
            result = function()
            mx.eval(*result)
    timing = {"ordinary_ms": [], "fused_ms": []}
    functions = (("ordinary_ms", ordinary_b1), ("fused_ms", fused_b1))
    for repeat in range(args.timing_reps):
        for name, function in (functions if repeat % 2 == 0 else reversed(functions)):
            started = time.perf_counter()
            result = function()
            mx.eval(*result)
            timing[name].append(1000 * (time.perf_counter() - started))
    component_timing = {
        "shape": [1, 1, 128, 64],
        "repetitions": args.timing_reps,
        "ordinary_median_ms": statistics.median(timing["ordinary_ms"]),
        "fused_median_ms": statistics.median(timing["fused_ms"]),
        "method": "alternating order; same materialized inputs; output and state evaluated",
        "performance_qualification": False,
    }
    component_timing["relative_percent"] = 100 * (
        component_timing["ordinary_median_ms"]
        / component_timing["fused_median_ms"] - 1
    )

    for function in (ordinary_gate_norm_b1, fused_gate_norm_b1):
        for _ in range(5):
            result = function()
            mx.eval(*result)
    gate_timing = {"ordinary_ms": [], "fused_ms": []}
    gate_functions = (
        ("ordinary_ms", ordinary_gate_norm_b1),
        ("fused_ms", fused_gate_norm_b1),
    )
    for repeat in range(args.timing_reps):
        order = gate_functions if repeat % 2 == 0 else reversed(gate_functions)
        for name, function in order:
            started = time.perf_counter()
            result = function()
            mx.eval(*result)
            gate_timing[name].append(1000 * (time.perf_counter() - started))
    gate_norm_timing = {
        "shape": [1, 1, 8192],
        "repetitions": args.timing_reps,
        "ordinary_median_ms": statistics.median(gate_timing["ordinary_ms"]),
        "fused_median_ms": statistics.median(gate_timing["fused_ms"]),
        "method": "alternating order; SSM output, BF16 SiLU gate, group RMS and state evaluated",
        "performance_qualification": False,
    }
    gate_norm_timing["relative_percent"] = 100 * (
        gate_norm_timing["ordinary_median_ms"]
        / gate_norm_timing["fused_median_ms"] - 1
    )

    after = common.host_snapshot()
    contamination = [
        reason for reason in common.contamination(before, after)
        if reason != "swap-outs increased"
    ]
    receipt = {
        "schema": "mlx2.nemotron35-super-mamba-ssm-short-dt-gate-probe.v2",
        "owner": owner,
        "source": common.source_identity(),
        "dependencies": common.runtime_dependencies(),
        "before": before,
        "after": after,
        "swap_ignored_by_user": True,
        "cases": cases,
        "same_block_prefix_cases": prefix_cases,
        "component_timing": component_timing,
        "gate_group_rms_timing": gate_norm_timing,
        "passed": all(
            case["output_exact"] and case["state_exact"]
            and case["gated_output_exact"] and case["gate_state_exact"]
            and case["group_rms_output_exact"] for case in cases
        ) and all(
            case["output_exact"] and case["state_exact"]
            and case["gated_output_exact"] and case["gate_state_exact"]
            and case["group_rms_output_exact"]
            for case in prefix_cases
        ) and not contamination,
        "contamination": contamination,
    }
    common.atomic_json(args.out_dir / "result.json", receipt)
    if not receipt["passed"]:
        raise RuntimeError("Mamba short-SSM dt exactness probe failed")


if __name__ == "__main__":
    main()
