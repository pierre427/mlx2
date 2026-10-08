#!/usr/bin/env python3
"""Focused Metal probes for the MLX #4596/#4640/#4641 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import mlx.core as mx


def _hash(array) -> str:
    return hashlib.sha256(bytes(memoryview(array.astype(mx.float32)))).hexdigest()


def _measure(fn, *, warmup: int, reps: int) -> dict:
    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    for _ in range(reps):
        mx.synchronize()
        start = time.perf_counter_ns()
        mx.eval(fn())
        mx.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def qmm_precision(dtype, mode: str, m: int, *, warmup: int, reps: int) -> dict:
    k, n = 2560, 1024
    mx.random.seed(1701 + m)
    x = mx.random.normal((m, k)).astype(dtype)
    w = (0.02 * mx.random.normal((n, k))).astype(dtype)
    group_size, bits = (64, 4) if mode == "affine" else (None, None)
    wq = mx.quantize(w, group_size=group_size, bits=bits, mode=mode)
    w_hat = mx.dequantize(*wq, group_size=group_size, bits=bits, mode=mode)
    ref = x.astype(mx.float32) @ w_hat.astype(mx.float32).T

    def operation():
        return mx.quantized_matmul(
            x,
            *wq,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )

    out = operation()
    mx.eval(out, ref)
    error = mx.mean(mx.abs(out.astype(mx.float32) - ref)).item()
    rounding = mx.mean(mx.abs(ref.astype(dtype).astype(mx.float32) - ref)).item()
    return {
        "shape": [m, n, k],
        "dtype": str(dtype),
        "mode": mode,
        "mean_abs_error": error,
        "single_rounding_error": rounding,
        "error_over_single_rounding": error / rounding,
        "output_sha256_f32": _hash(out),
        "timing": _measure(operation, warmup=warmup, reps=reps),
    }


def thin_n(dtype, m: int, *, warmup: int, reps: int) -> dict:
    k, n = 10240, 4
    mx.random.seed(4600 + m)
    a = mx.random.normal((m, k)).astype(dtype)
    b = (0.02 * mx.random.normal((n, k))).astype(dtype)
    ref = a.astype(mx.float32) @ b.astype(mx.float32).T

    def operation():
        return a @ b.T

    out = operation()
    mx.eval(out, ref)
    error = mx.abs(out.astype(mx.float32) - ref)
    return {
        "shape": [m, n, k],
        "dtype": str(dtype),
        "mean_abs_error": mx.mean(error).item(),
        "max_abs_error": mx.max(error).item(),
        "nonfinite": int(mx.sum(~mx.isfinite(out)).item()),
        "argmax_matches_f32": bool(
            mx.array_equal(mx.argmax(out, axis=-1), mx.argmax(ref, axis=-1)).item()
        ),
        "output_sha256_f32": _hash(out),
        "timing": _measure(operation, warmup=warmup, reps=reps),
    }


def sdpa(dtype, shape: tuple[int, int, int, int], *, warmup: int, reps: int) -> dict:
    qh, kvh, context, dim = shape
    mx.random.seed(4596 + context)
    q = mx.random.normal((1, qh, 1, dim)).astype(dtype)
    k = mx.random.normal((1, kvh, context, dim)).astype(dtype)
    v = mx.random.normal((1, kvh, context, dim)).astype(dtype)
    scale = 1.0 / math.sqrt(dim)

    def operation():
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

    out = operation()
    mx.eval(out)
    return {
        "shape": list(shape),
        "dtype": str(dtype),
        "output_sha256_f32": _hash(out),
        "timing": _measure(operation, warmup=warmup, reps=reps),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--steady",
        action="store_true",
        help="use long warmups and 101 samples to suppress cold power-state noise",
    )
    args = parser.parse_args()
    warmup, reps = (50, 101) if args.steady else (6, 25)
    info = mx.device_info(mx.gpu)
    result = {
        "schema": "mlx2.mlx-rebase-kernel-probes.v1",
        "label": args.label,
        "mlx": mx.__version__,
        "device": info,
        "qmm_splitk": [
            qmm_precision(mx.bfloat16, mode, m, warmup=warmup, reps=reps)
            for mode in ("affine", "mxfp4")
            for m in (16, 33, 65)
        ],
        "thin_n": [
            thin_n(mx.bfloat16, m, warmup=warmup, reps=reps)
            for m in (64, 512, 2048)
        ],
        "sdpa_vector": [
            sdpa(mx.float16, shape, warmup=warmup, reps=reps)
            for shape in ((32, 32, 4096, 128), (32, 4, 4096, 128), (8, 4, 32768, 128))
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
