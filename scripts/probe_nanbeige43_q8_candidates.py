"""Bounded kernel diagnostics to call inside an already-owned GPU unit.

This script never acquires or bypasses GPU ownership. Its caller must perform
ordinary-reference smoke, thermal/contamination checks, paired locks and CPG.
Run through run_with_gpu_locks.py with a desktop campaign request.json.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path


def run_kernel_probe(*, large=True):
    import mlx.core as mx
    from mlx import nn

    from mlx2.adapters.nanbeige43_kernels import (
        compiled_mlp,
        kernel_contract,
        q8_swiglu,
    )
    from mlx2.runtime.lane.matmul import lane_matmul, prepare
    from mlx2.runtime.models.activations import swiglu

    if mx.default_device() != mx.gpu or "M5" not in mx.device_info().get(
        "device_name", ""
    ):
        raise RuntimeError("probe requires the owned M5 GPU")
    k, n = (7168, 22016) if large else (128, 256)
    mx.random.seed(4276)
    packed = nn.QuantizedLinear(k, n * 2, bias=False, bits=8, group_size=64)
    down = nn.QuantizedLinear(n, k, bias=False, bits=8, group_size=64)
    packed.set_dtype(mx.bfloat16)
    down.set_dtype(mx.bfloat16)
    mx.eval(packed.parameters(), down.parameters())
    digest = hashlib.sha256()
    for module in (packed, down):
        for name in ("weight", "scales", "biases"):
            import numpy as np

            array = module[name]
            if array.dtype == mx.bfloat16:
                array = array.view(mx.uint16)
            digest.update(np.asarray(array).tobytes())
    result = {
        "checkpoint": f"synthetic-nanbeige43-k{k}-n{n}-q8g64@sha256:{digest.hexdigest()}",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "kernel_contract": kernel_contract(None),
        "qualification": False,
        "scope": "synthetic kernels only; no whole-model speed or qualitative answer claim",
        "cells": [],
    }

    def bench(fn):
        mx.eval(fn())
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            y = fn()
            mx.eval(y)
            samples.append(time.perf_counter() - start)
        return statistics.median(samples)

    x = mx.random.normal((1, 65, k)).astype(mx.bfloat16)
    mx.eval(x)

    def reference():
        gate, up = mx.split(packed(x), 2, axis=-1)
        return swiglu(gate, up)

    reference_y = reference()
    mx.eval(reference_y)
    reference_seconds = bench(reference)
    for tile in (16, 32):
        candidate_y = q8_swiglu(x, packed, tile)
        mx.eval(candidate_y)
        result["cells"].append(
            {
                "kind": "q8_swiglu",
                "rows": 65,
                "tile_rows": tile,
                "reference_seconds": reference_seconds,
                "candidate_seconds": bench(
                    lambda tile=tile: q8_swiglu(x, packed, tile)
                ),
                "max_abs_error": float(
                    mx.max(
                        mx.abs(
                            candidate_y.astype(mx.float32)
                            - reference_y.astype(mx.float32)
                        )
                    ).item()
                ),
                "finite": bool(mx.all(mx.isfinite(candidate_y)).item()),
            }
        )
    expected = down(reference_y)
    mx.eval(expected)
    compiled = lambda: compiled_mlp(
        x,
        packed.weight,
        packed.scales,
        packed.biases,
        down.weight,
        down.scales,
        down.biases,
    )
    actual = compiled()
    mx.eval(actual)
    result["cells"].append(
        {
            "kind": "compiled_mlp",
            "rows": 65,
            "reference_seconds": bench(lambda: down(reference())),
            "candidate_seconds": bench(compiled),
            "max_abs_error": float(
                mx.max(
                    mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
                ).item()
            ),
        }
    )
    prepared = prepare(packed)
    for rows in (1, 2, 4, 8, 16, 32):
        value = x[:, :rows]
        stock, lane = packed(value), lane_matmul(value, prepared)
        mx.eval(stock, lane)
        result["cells"].append(
            {
                "kind": "projection",
                "rows": rows,
                "n": 2 * n,
                "k": k,
                "reference_seconds": bench(lambda value=value: packed(value)),
                "candidate_seconds": bench(
                    lambda value=value: lane_matmul(value, prepared)
                ),
                "max_abs_error": float(
                    mx.max(
                        mx.abs(lane.astype(mx.float32) - stock.astype(mx.float32))
                    ).item()
                ),
            }
        )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    from mlx2.desktop_campaign_worker import atomic, check_ownership

    request = json.loads(args.request.read_text())
    check_ownership(request)
    if args.output.exists():
        raise RuntimeError("refuse to overwrite probe evidence")
    atomic(args.output, run_kernel_probe(large=not args.small))
