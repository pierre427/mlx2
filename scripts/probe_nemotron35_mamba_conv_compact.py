#!/usr/bin/env python3
"""Owned component probe for compact ordinary-decode Mamba conv state."""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import mlx.core as mx
import nemotron35_probe_common as common
from mlx import nn

from mlx2.runtime.models.nemotron_mamba_conv import fused_mamba_conv_b1


def bits_sha(value) -> str:
    return hashlib.sha256(bytes(memoryview(mx.array(value).view(mx.uint16)))).hexdigest()


def elapsed_ms(call, repetitions: int) -> float:
    for _ in range(10):
        mx.eval(*call())
    started = time.perf_counter()
    for _ in range(repetitions):
        mx.eval(*call())
    return (time.perf_counter() - started) * 1000.0 / repetitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
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
    mx.random.seed(2718)
    conv = nn.Conv1d(
        channels, channels, kernel_size=4, groups=channels, bias=True
    )
    conv.weight = conv.weight.astype(mx.bfloat16)
    conv.bias = conv.bias.astype(mx.bfloat16)
    conv.eval()
    mx.eval(conv.parameters())
    cases = []
    for seed, scale in ((11, 0.25), (17, 1.0), (23, 4.0)):
        mx.random.seed(seed)
        state = (mx.random.normal((1, 3, channels)) * scale).astype(mx.bfloat16)
        value = (mx.random.normal((1, 1, channels)) * scale).astype(mx.bfloat16)
        mx.eval(state, value)

        def stock(value=value, state=state):
            padded = mx.concatenate([state, value], axis=1)
            return nn.silu(conv(padded)), padded[:, 1:, :]

        expected_output, expected_state = stock()
        full_output, full_window = fused_mamba_conv_b1(
            value, state, conv.weight, conv.bias, capture_window=True
        )
        compact_output, compact_state = fused_mamba_conv_b1(
            value, state, conv.weight, conv.bias, capture_window=False
        )
        mx.eval(
            expected_output,
            expected_state,
            full_output,
            full_window,
            compact_output,
            compact_state,
        )
        full_state = full_window[:, 1:, :]
        exact_output = bool(mx.array_equal(expected_output, compact_output).item())
        exact_state = bool(mx.array_equal(expected_state, compact_state).item())
        full_compact_output = bool(mx.array_equal(full_output, compact_output).item())
        full_compact_state = bool(mx.array_equal(full_state, compact_state).item())
        stock_ms = elapsed_ms(stock, 100)
        full_ms = elapsed_ms(
            lambda value=value, state=state: fused_mamba_conv_b1(
                value, state, conv.weight, conv.bias, capture_window=True
            ),
            100,
        )
        compact_ms = elapsed_ms(
            lambda value=value, state=state: fused_mamba_conv_b1(
                value, state, conv.weight, conv.bias, capture_window=False
            ),
            100,
        )
        cases.append(
            {
                "seed": seed,
                "scale": scale,
                "exact_output": exact_output,
                "exact_state": exact_state,
                "full_compact_output_exact": full_compact_output,
                "full_compact_state_exact": full_compact_state,
                "output_sha256": bits_sha(compact_output),
                "state_sha256": bits_sha(compact_state),
                "stock_ms": stock_ms,
                "full_window_ms": full_ms,
                "compact_state_ms": compact_ms,
                "compact_vs_stock_percent": (stock_ms / compact_ms - 1.0) * 100.0,
                "compact_vs_full_window_percent": (full_ms / compact_ms - 1.0)
                * 100.0,
                "history_bytes_full": 4 * channels * 2,
                "history_bytes_compact": 3 * channels * 2,
            }
        )
    after = common.host_snapshot()
    contamination = [
        reason
        for reason in common.contamination(before, after)
        if reason != "swap-outs increased"
    ]
    receipt = {
        "schema": "mlx2.nemotron35-super-mamba-conv-compact-state-probe.v1",
        "owner": owner,
        "source": common.source_identity(),
        "dependencies": common.runtime_dependencies(),
        "before": before,
        "after": after,
        "swap_ignored_by_user": True,
        "cases": cases,
        "passed": all(
            case["exact_output"]
            and case["exact_state"]
            and case["full_compact_output_exact"]
            and case["full_compact_state_exact"]
            for case in cases
        )
        and not contamination,
        "contamination": contamination,
    }
    common.atomic_json(args.out_dir / "result.json", receipt)
    if not receipt["passed"]:
        raise RuntimeError("compact Mamba conv-state probe failed")


if __name__ == "__main__":
    main()
