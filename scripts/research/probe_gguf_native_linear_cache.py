#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Owned native linear promotion-lifetime GPU fixture; no model/perf claim."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from probe_deepseek_v4_packed import atomic, compare, source_receipt, verify_ownership


def run_case(k, n, guard):
    from types import SimpleNamespace

    import mlx.core as mx

    from mlx2.runtime.gguf.weights import GGUFWeights

    rng = np.random.default_rng(20261024 + k + n)
    # One native F16 matrix only; quantized GGUF never dequantized here.
    raw = rng.standard_normal((n, k), dtype=np.float32).astype(np.float16) / np.float16(
        128
    )
    values = rng.standard_normal((128, k), dtype=np.float32)
    provider = GGUFWeights.__new__(GGUFWeights)
    provider.tensors = {"weight": SimpleNamespace(shape=(k, n), ggml_type=1)}
    provider._dense = {"weight": mx.array(raw)}
    provider._native_linear_fp32 = {}
    mx.eval(provider._dense)
    linear = provider.linear("weight")
    guard()
    provider.preload_native_linear_fp32()
    rows = []
    for width in (1, 6, 128):
        guard()
        x = mx.array(values[:width])
        mx.eval(x)
        old = x @ provider._dense["weight"].T
        cached = linear(x)
        mx.eval(old, cached)
        a, b = np.array(cached), np.array(old)
        rows.append(
            {
                "K": k,
                "N": n,
                "rows": width,
                "old_operand": "native_weight.T promoted implicitly toFP32",
                "cached_operand": "native_weight.T.astype(FP32), evaluated before request",
                "numerical": compare(a, b, atol=0, rtol=0),
                "bitwise_equal": bool(
                    np.array_equal(a.view(np.uint32), b.view(np.uint32))
                ),
                "old_dtype": str(old.dtype),
                "cached_dtype": str(cached.dtype),
                "raw_dtype_unchanged": str(provider._dense["weight"].dtype),
                "cached_shape": list(provider.native_fp32_parameters()["weight"].shape),
                "additional_bytes": provider.native_linear_fp32_status()[
                    "additional_bytes"
                ],
                "native_weight_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                "input_fp32_sha256": hashlib.sha256(
                    values[:width].tobytes()
                ).hexdigest(),
            }
        )
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-gpu", action="store_true")
    p.add_argument("--ownership-window", type=Path)
    p.add_argument("--owner-path", type=Path, action="append")
    args = p.parse_args(argv)
    if args.output.exists():
        p.error("refusing existing evidence directory")
    owners = args.owner_path or [
        Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
        Path("/tmp/gpu.lock/owner.json"),
    ]
    if not args.run_gpu or args.ownership_window is None or len(owners) != 2:
        p.error("explicit GPU flag/live ownership window/two paired receipts required")

    def guard():
        return verify_ownership(args.ownership_window, owners)

    ownership = guard()
    source = source_receipt()
    identity = hashlib.sha256(
        b"native-cache-fp16-seed20261024-K4096N256-K4096N1024-K1024N8192-rows1,6,128"
    ).hexdigest()
    receipt = {
        "schema": "mlx2.gguf-native-linear-promotion-layout-fixtures.v1",
        "status": "running",
        "source": source,
        "checkpoint": "synthetic-native-F16-linear@" + identity,
        "ownership": ownership,
        "actual_artifact_loaded": False,
        "qualification": False,
        "performance_claim": False,
        "maximum_working_bytes": 512 << 20,
        "cases": [],
    }
    args.output.mkdir(parents=True)
    atomic(args.output / "results.json", receipt)
    import mlx.core as mx

    previous = mx.default_device()
    try:
        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(512 << 20)
        mx.set_cache_limit(32 << 20)
        receipt["device_info"] = mx.metal.device_info()
        for k, n in ((4096, 256), (4096, 1024), (1024, 8192)):
            receipt["cases"] += run_case(k, n, guard)
            atomic(args.output / "results.json", receipt)
            mx.clear_cache()
        receipt["passed"] = all(
            c["bitwise_equal"]
            and c["numerical"]["passed"]
            and c["old_dtype"] == c["cached_dtype"]
            for c in receipt["cases"]
        )
        receipt["allocator_peak_bytes"] = mx.get_peak_memory()
        receipt["status"] = "completed"
        guard()
    except BaseException as error:
        receipt.update(
            status="failed", error={"type": type(error).__name__, "message": str(error)}
        )
        raise
    finally:
        atomic(args.output / "results.json", receipt)
        mx.set_default_device(previous)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
