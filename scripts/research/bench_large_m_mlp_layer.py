#!/usr/bin/env python3
"""Screen complete large-M Qwen3.8 dense-MLP graphs on the pinned MLX build.

This complements ``bench_large_m_qmm.py`` by evaluating gate and up together,
forming the SwiGLU activation, and consuming it through down before the graph is
materialized. It rotates distinct layer weights to model the real working set.
The result is a mechanism screen, not model qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

SCHEMA = "mlx2.large-m-mlp-layer-screen.v1"
DIM = 5120
HIDDEN = 17408
GROUP_SIZE = 64
BITS = 4
DEFAULT_MS = (6985, 20883)
ARMS = ("stock", "dynamic_gate_up", "dynamic_down", "dynamic_all")


def affine_bytes(n: int, k: int) -> int:
    return n * k * BITS // 8 + 4 * n * (k // GROUP_SIZE)


def layer_affine_bytes() -> int:
    return 2 * affine_bytes(HIDDEN, DIM) + affine_bytes(DIM, HIDDEN)


def copies_for_target(target_weight_mib: int) -> int:
    if target_weight_mib <= 0:
        return 1
    return max(1, math.ceil((target_weight_mib << 20) / layer_affine_bytes()))


def parse_ms(value: str) -> list[int]:
    rows = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not rows or any(row <= 0 for row in rows):
        raise ValueError("M values must be positive")
    return rows


def plan(ms: list[int], target_weight_mib: int) -> list[dict]:
    copies = copies_for_target(target_weight_mib)
    return [
        {
            "M": m,
            "arm": arm,
            "copies": copies,
            "rotating_affine_bytes": copies * layer_affine_bytes(),
        }
        for m in ms
        for arm in ARMS
    ]


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def system_value(*command: str) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_quantized(mx, n: int, k: int):
    packed = mx.random.randint(
        low=0, high=1 << 31, shape=(n, k // 8), dtype=mx.uint32
    )
    scales = mx.full((n, k // GROUP_SIZE), 0.01, dtype=mx.bfloat16)
    biases = mx.full((n, k // GROUP_SIZE), -0.08, dtype=mx.bfloat16)
    mx.eval(packed, scales, biases)
    return packed, scales, biases


def qmm(mx, x, weights):
    return mx.quantized_matmul(
        x, *weights, transpose=True, group_size=GROUP_SIZE, bits=BITS
    )


def dense(mx, x, weights):
    table = mx.dequantize(
        *weights, group_size=GROUP_SIZE, bits=BITS
    ).astype(mx.bfloat16)
    return x @ table.T


def projection(mx, x, weights, *, use_dense: bool):
    return dense(mx, x, weights) if use_dense else qmm(mx, x, weights)


def mlp(mx, x, weights, arm: str):
    dense_gate_up = arm in {"dynamic_gate_up", "dynamic_all"}
    dense_down = arm in {"dynamic_down", "dynamic_all"}
    gate = projection(mx, x, weights[0], use_dense=dense_gate_up)
    up = projection(mx, x, weights[1], use_dense=dense_gate_up)
    product = (gate * mx.sigmoid(gate)) * up
    return projection(mx, product, weights[2], use_dense=dense_down)


def compare(mx, candidate, reference) -> dict:
    delta = mx.abs(candidate.astype(mx.float32) - reference.astype(mx.float32))
    mx.eval(delta)
    max_abs = float(mx.max(delta).item())
    ref_max = float(mx.max(mx.abs(reference.astype(mx.float32))).item())
    return {
        "max_abs_vs_stock": max_abs,
        "max_rel_vs_stock_peak": max_abs / ref_max if ref_max else 0.0,
        "array_equal_stock": bool(mx.array_equal(candidate, reference).item()),
    }


def run_sequence(mx, arm: str, x, layers, *, reverse: bool) -> None:
    indices = range(len(layers) - 1, -1, -1) if reverse else range(len(layers))
    for index in indices:
        value = mlp(mx, x, layers[index], arm)
        mx.eval(value)
        del value


def run(args) -> dict:
    import mlx.core as mx

    ms = parse_ms(args.ms)
    copies = copies_for_target(args.target_weight_mib)
    mx.random.seed(args.seed)
    try:
        mx.set_cache_limit(args.cache_limit_gib << 30)
    except AttributeError:
        pass
    layers = [
        (
            make_quantized(mx, HIDDEN, DIM),
            make_quantized(mx, HIDDEN, DIM),
            make_quantized(mx, DIM, HIDDEN),
        )
        for _ in range(copies)
    ]
    receipt = {
        "schema": SCHEMA,
        "scope": "isolated_complete_mlp_graph_screen_not_model_qualification",
        "source_sha256": source_sha256(),
        "mlx_version": importlib.metadata.version("mlx"),
        "vm_swapusage_before": system_value("sysctl", "vm.swapusage"),
        "configuration": {
            "ms": ms,
            "arms": list(ARMS),
            "copies": copies,
            "rotating_affine_bytes": copies * layer_affine_bytes(),
            "target_weight_mib": args.target_weight_mib,
            "warmups": args.warmups,
            "reps": args.reps,
            "seed": args.seed,
        },
        "results": [],
    }
    for m in ms:
        x = mx.random.normal(shape=(m, DIM)).astype(mx.bfloat16)
        mx.eval(x)
        reference = mlp(mx, x, layers[0], "stock")
        mx.eval(reference)
        comparisons = {}
        for arm in ARMS:
            candidate = mlp(mx, x, layers[0], arm)
            mx.eval(candidate)
            comparisons[arm] = compare(mx, candidate, reference)
            del candidate
        times = {arm: [] for arm in ARMS}
        for arm in ARMS:
            for _ in range(args.warmups):
                run_sequence(mx, arm, x, layers, reverse=False)
        for rep in range(args.reps):
            order = ARMS if rep % 2 == 0 else tuple(reversed(ARMS))
            for arm in order:
                started = time.perf_counter()
                run_sequence(mx, arm, x, layers, reverse=bool(rep % 2))
                times[arm].append((time.perf_counter() - started) / copies)
        stock = statistics.median(times["stock"])
        for arm in ARMS:
            observed = times[arm]
            median = statistics.median(observed)
            receipt["results"].append(
                {
                    "M": m,
                    "arm": arm,
                    "copies": copies,
                    "rotating_affine_bytes": copies * layer_affine_bytes(),
                    "seconds": observed,
                    "median_seconds": median,
                    "min_seconds": min(observed),
                    "max_seconds": max(observed),
                    "ratio_to_stock": median / stock,
                    **comparisons[arm],
                }
            )
        del reference, x
        mx.clear_cache()
    receipt["vm_swapusage_after"] = system_value("sysctl", "vm.swapusage")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--ms", default=",".join(map(str, DEFAULT_MS)))
    parser.add_argument("--target-weight-mib", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--cache-limit-gib", type=int, default=16)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        ms = parse_ms(args.ms)
    except ValueError as exc:
        parser.error(str(exc))
    if min(args.target_weight_mib, args.warmups, args.reps, args.cache_limit_gib) < 1:
        parser.error("working set, warmups, reps and cache limit must be positive")
    if args.dry_run:
        print(json.dumps({"schema": SCHEMA, "source_sha256": source_sha256(),
                          "plan": plan(ms, args.target_weight_mib)},
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
