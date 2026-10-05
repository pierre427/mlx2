#!/usr/bin/env python3
"""Screen exact N20 MLP shapes against MLX's large-M affine QMM path.

The default invocation is static and does not import MLX. Device execution is
explicit, writes one JSON receipt, and assumes the caller already owns the
shared GPU queue and both host locks. This is a mechanism screen, not model
qualification or a serving-performance claim.

The chunked arms keep the same stock ``mx.quantized_matmul`` operation and only
partition the input-row dimension. They test whether the large-M dispatch has
a crossover below the full packed-prefill row count. The dynamic and cached
BF16 dequantization arms are diagnostic bounds; neither is an admitted model
route.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path

SCHEMA = "mlx2.large-m-qmm-screen.v1"
GROUP_SIZE = 64
BITS = 4
SHAPES = {
    "gate_up": (17408, 5120),
    "down": (5120, 17408),
}
DEFAULT_MS = (6985, 20883)
DEFAULT_ARMS = (
    "stock",
    "chunk8192",
    "chunk4096",
    "chunk2048",
    "dynamic_bf16",
    "cached_bf16",
)


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_ms(value: str) -> list[int]:
    rows = [int(part) for part in parse_csv(value)]
    if not rows or any(row <= 0 for row in rows):
        raise ValueError("M values must be positive")
    return rows


def chunk_size(arm: str) -> int | None:
    if arm == "stock":
        return None
    if arm.startswith("chunk") and arm[5:].isdigit():
        size = int(arm[5:])
        if size > 0:
            return size
    raise ValueError(f"not a stock/chunk arm: {arm}")


def validate(shapes: list[str], arms: list[str], ms: list[int]) -> None:
    unknown_shapes = sorted(set(shapes) - set(SHAPES))
    if unknown_shapes:
        raise ValueError(f"unknown shapes: {unknown_shapes}")
    if not shapes:
        raise ValueError("at least one shape is required")
    allowed = {"stock", "dynamic_bf16", "cached_bf16"}
    for arm in arms:
        if arm not in allowed:
            chunk_size(arm)
    if "stock" not in arms:
        raise ValueError("stock must be included")
    if not ms:
        raise ValueError("at least one M is required")


def plan(shapes: list[str], arms: list[str], ms: list[int]) -> list[dict]:
    validate(shapes, arms, ms)
    rows = []
    for shape in shapes:
        n, k = SHAPES[shape]
        for m in ms:
            for arm in arms:
                effective = m
                if arm.startswith("chunk"):
                    effective = min(m, chunk_size(arm) or m)
                rows.append(
                    {
                        "shape": shape,
                        "M": m,
                        "N": n,
                        "K": k,
                        "bits": BITS,
                        "group_size": GROUP_SIZE,
                        "arm": arm,
                        "effective_qmm_M_max": effective,
                    }
                )
    return rows


def affine_bytes(n: int, k: int) -> int:
    packed = n * k * BITS // 8
    scales_and_biases = 2 * 2 * n * (k // GROUP_SIZE)
    return packed + scales_and_biases


def copies_for_target(n: int, k: int, target_weight_mib: int) -> int:
    if target_weight_mib <= 0:
        return 1
    return max(1, math.ceil((target_weight_mib << 20) / affine_bytes(n, k)))


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def system_value(*command: str) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def device_info(mx) -> dict:
    for getter in (
        lambda: mx.device_info(),
        lambda: mx.metal.device_info(),
    ):
        try:
            value = getter()
            return dict(value) if value is not None else {}
        except (AttributeError, RuntimeError):
            continue
    return {}


def make_quantized(mx, n: int, k: int):
    # Build affine tensors directly. This avoids charging a giant source BF16
    # table merely to prepare the timing fixture.
    packed = mx.random.randint(
        low=0,
        high=1 << 31,
        shape=(n, k // 8),
        dtype=mx.uint32,
    )
    scales = mx.full((n, k // GROUP_SIZE), 0.01, dtype=mx.bfloat16)
    biases = mx.full((n, k // GROUP_SIZE), -0.08, dtype=mx.bfloat16)
    mx.eval(packed, scales, biases)
    return packed, scales, biases


def qmm(mx, x, quantized):
    return mx.quantized_matmul(
        x,
        *quantized,
        transpose=True,
        group_size=GROUP_SIZE,
        bits=BITS,
    )


def chunked_qmm(mx, x, quantized, size: int):
    if int(x.shape[0]) <= size:
        return qmm(mx, x, quantized)
    return mx.concatenate(
        [qmm(mx, x[start : start + size], quantized)
         for start in range(0, int(x.shape[0]), size)],
        axis=0,
    )


def evaluate(mx, arm: str, x, quantized, cached_weight):
    if arm == "stock":
        return qmm(mx, x, quantized)
    if arm.startswith("chunk"):
        return chunked_qmm(mx, x, quantized, chunk_size(arm) or int(x.shape[0]))
    if arm == "dynamic_bf16":
        weight = mx.dequantize(
            *quantized,
            group_size=GROUP_SIZE,
            bits=BITS,
        ).astype(mx.bfloat16)
        return x @ weight.T
    if arm == "cached_bf16":
        if cached_weight is None:
            raise ValueError("cached_bf16 requires a materialized weight")
        return x @ cached_weight.T
    raise ValueError(f"unknown arm: {arm}")


def run_sequence(mx, arm: str, x, quantized, cached_weights, *, reverse=False) -> None:
    copies = len(quantized)
    indices = range(copies - 1, -1, -1) if reverse else range(copies)
    for index in indices:
        value = evaluate(
            mx,
            arm,
            x,
            quantized[index],
            cached_weights[index] if cached_weights else None,
        )
        mx.eval(value)
        del value


def compare(mx, candidate, reference) -> dict:
    cand32 = candidate.astype(mx.float32)
    ref32 = reference.astype(mx.float32)
    delta = mx.abs(cand32 - ref32)
    mx.eval(delta)
    max_abs = float(mx.max(delta).item())
    ref_max = float(mx.max(mx.abs(ref32)).item())
    return {
        "max_abs_vs_stock": max_abs,
        "max_rel_vs_stock_peak": max_abs / ref_max if ref_max else 0.0,
        "array_equal_stock": bool(mx.array_equal(candidate, reference).item()),
    }


def run(args) -> dict:
    import mlx.core as mx

    shapes = parse_csv(args.shapes)
    arms = parse_csv(args.arms)
    ms = parse_ms(args.ms)
    validate(shapes, arms, ms)
    mx.random.seed(args.seed)
    try:
        mx.set_cache_limit(args.cache_limit_gib << 30)
    except AttributeError:
        pass

    receipt = {
        "schema": SCHEMA,
        "scope": "isolated_exact_shape_mechanism_screen_not_model_qualification",
        "source_sha256": source_sha256(),
        "mlx_version": importlib.metadata.version("mlx"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device_info(mx),
        "vm_swapusage_before": system_value("sysctl", "vm.swapusage"),
        "vm_swapouts_before": system_value("sysctl", "vm.swapouts"),
        "configuration": {
            "shapes": shapes,
            "ms": ms,
            "arms": arms,
            "warmups": args.warmups,
            "reps": args.reps,
            "seed": args.seed,
            "cache_limit_gib": args.cache_limit_gib,
            "target_weight_mib": args.target_weight_mib,
        },
        "results": [],
    }

    for shape in shapes:
        n, k = SHAPES[shape]
        copies = copies_for_target(n, k, args.target_weight_mib)
        quantized = [make_quantized(mx, n, k) for _ in range(copies)]
        cached_weights = None
        if "cached_bf16" in arms:
            cached_weights = [
                mx.dequantize(
                    *weights,
                    group_size=GROUP_SIZE,
                    bits=BITS,
                ).astype(mx.bfloat16)
                for weights in quantized
            ]
            mx.eval(*cached_weights)

        for m in ms:
            x = mx.random.normal(shape=(m, k)).astype(mx.bfloat16)
            mx.eval(x)
            reference = evaluate(mx, "stock", x, quantized[0],
                                 cached_weights[0] if cached_weights else None)
            mx.eval(reference)
            comparisons = {}
            for arm in arms:
                candidate = evaluate(mx, arm, x, quantized[0],
                                     cached_weights[0] if cached_weights else None)
                mx.eval(candidate)
                comparisons[arm] = compare(mx, candidate, reference)
                del candidate

            times = {arm: [] for arm in arms}

            for arm in arms:
                for _ in range(args.warmups):
                    run_sequence(mx, arm, x, quantized, cached_weights)
            for rep in range(args.reps):
                order = arms if rep % 2 == 0 else list(reversed(arms))
                for arm in order:
                    started = time.perf_counter()
                    run_sequence(
                        mx, arm, x, quantized, cached_weights,
                        reverse=bool(rep % 2),
                    )
                    elapsed = time.perf_counter() - started
                    times[arm].append(elapsed / copies)

            stock_median = statistics.median(times["stock"])
            for arm in arms:
                observed = times[arm]
                median = statistics.median(observed)
                receipt["results"].append(
                    {
                        "shape": shape,
                        "M": m,
                        "N": n,
                        "K": k,
                        "bits": BITS,
                        "group_size": GROUP_SIZE,
                        "copies": copies,
                        "rotating_affine_bytes": copies * affine_bytes(n, k),
                        "arm": arm,
                        "effective_qmm_M_max": (
                            min(m, chunk_size(arm) or m)
                            if arm.startswith("chunk") else m
                        ),
                        "seconds": observed,
                        "median_seconds": median,
                        "min_seconds": min(observed),
                        "max_seconds": max(observed),
                        "ratio_to_stock": median / stock_median,
                        **comparisons[arm],
                    }
                )
            del reference, x
            mx.clear_cache()
        del quantized, cached_weights
        mx.clear_cache()

    receipt["vm_swapusage_after"] = system_value("sysctl", "vm.swapusage")
    receipt["vm_swapouts_after"] = system_value("sysctl", "vm.swapouts")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--ms", default=",".join(map(str, DEFAULT_MS)))
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--cache-limit-gib", type=int, default=16)
    parser.add_argument(
        "--target-weight-mib",
        type=int,
        default=0,
        help="rotate enough distinct affine tables to meet this working set",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    shapes = parse_csv(args.shapes)
    arms = parse_csv(args.arms)
    try:
        ms = parse_ms(args.ms)
        rows = plan(shapes, arms, ms)
    except ValueError as exc:
        parser.error(str(exc))
    if min(args.warmups, args.reps, args.cache_limit_gib) < 1:
        parser.error("warmups, reps and cache-limit-gib must be positive")
    if args.target_weight_mib < 0:
        parser.error("target-weight-mib cannot be negative")
    if args.dry_run:
        print(json.dumps({"schema": SCHEMA, "source_sha256": source_sha256(), "plan": rows},
                         indent=2, sort_keys=True))
        return 0
    if not args.i_own_the_gpu:
        parser.error("device execution requires --i-own-the-gpu")
    if args.out is None:
        parser.error("device execution requires --out")
    receipt = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
