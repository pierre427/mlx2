#!/usr/bin/env python3
"""M5 A/B for the opt-in Flash-Next TensorFold row-QMV candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx
from mlx import nn

from mlx2.runtime.models import flash_tensorfold_qmv as candidate

LOCKS = (
    Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
    Path("/tmp/gpu.lock/owner.json"),
)
SHAPES = (
    ("linear-qkv", 2560, 10240),
    ("linear-z", 2560, 6144),
    ("linear-out", 6144, 2560),
)


def owners() -> dict:
    values = [json.loads(path.read_text()) for path in LOCKS]
    if values[0] != values[1] or not values[0].get("lease_id"):
        raise RuntimeError("matching GPU owner receipts are required")
    return values[0]


def timed(call, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        value = call()
        mx.eval(value)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return samples


def run_shape(name: str, k: int, n: int, widths, repeats: int) -> dict:
    linear = nn.Linear(k, n, bias=False)
    linear.weight = linear.weight.astype(mx.bfloat16)
    quantized = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)
    del linear
    x = mx.random.normal((max(widths), k)).astype(mx.bfloat16)
    mx.eval(quantized.parameters(), x)
    rows = []
    for width in widths:
        stock_call = lambda width=width: quantized(x[:width])
        lane_call = lambda width=width: candidate.qmv_rows(x[:width], quantized)
        mx.eval(stock_call(), lane_call())
        stock, lane = [], []
        for repeat in range(repeats):
            order = ((stock_call, stock), (lane_call, lane))
            if repeat % 2:
                order = tuple(reversed(order))
            for call, samples in order:
                samples.extend(timed(call, 1))
        actual = lane_call()
        singles = mx.concatenate(
            [candidate.qmv_rows(x[index : index + 1], quantized) for index in range(width)],
            axis=0,
        )
        reference = stock_call()
        mx.eval(actual, singles, reference)
        stock_ms = statistics.median(stock)
        lane_ms = statistics.median(lane)
        rows.append(
            {
                "width": width,
                "row_invariant": bool(mx.array_equal(actual, singles).item()),
                "max_abs_vs_stock": float(
                    mx.max(mx.abs(actual.astype(mx.float32) - reference.astype(mx.float32))).item()
                ),
                "stock_ms": stock_ms,
                "tensorfold_ms": lane_ms,
                "speedup": stock_ms / lane_ms,
                "samples_ms": {"stock": stock, "tensorfold": lane},
            }
        )
    return {"name": name, "k": k, "n": n, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widths", default="1,4,8,16")
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    bound_owner = owners()
    mx.set_default_device(mx.gpu)
    mx.random.seed(20260930)
    widths = tuple(int(value) for value in args.widths.split(","))
    receipt = {
        "schema": "mlx2.flash-tensorfold-qmv-performance.v1",
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
        "mlx_version": mx.__version__,
        "device": mx.device_info(),
        "gpu_owner": bound_owner,
        "method": "counterbalanced same-process medians; representative shipped shapes",
        "widths": widths,
        "repeats": args.repeats,
        "cases": [run_shape(name, k, n, widths, args.repeats) for name, k, n in SHAPES],
        "selected": False,
        "qualified": False,
    }
    receipt["all_row_invariant"] = all(
        row["row_invariant"] for case in receipt["cases"] for row in case["rows"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                case["name"]: {str(row["width"]): row["speedup"] for row in case["rows"]}
                for case in receipt["cases"]
            },
            indent=2,
        )
    )
    return 0 if receipt["all_row_invariant"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
