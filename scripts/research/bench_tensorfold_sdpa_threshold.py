#!/usr/bin/env python3
"""M5 viability bench for TensorFold #197's 1,024-row SDPA crossover.

Default mode is metadata-only and does not import MLX.  ``--run-gpu`` is an
isolated component benchmark: it compares stock MLX, mlx2's existing
``fast_sdpa`` route, and TensorFold's pre-#197 128-row partitioning.  It does
not change serving policy and a passing receipt is not model qualification.

Run the GPU arm only through ``scripts/run_with_gpu_locks.py`` and pass
``--i-hold-gpu-lease`` to attest that the wrapper owns both lock receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tensorfold_viability_common import (
    environment_snapshot,
    git_revision,
    require_distribution,
    require_matching_gpu_receipts,
    timed_arms,
)

SOURCE_PR = "https://github.com/ashhart/TensorFold/pull/197"
SOURCE_HEAD = "bcc2f4a89e280c80e4870de3b206a365bfcd6958"
ROWS_PER_PART = 128
FUSED_ROWS = 1024
DEFAULT_ROWS = (256, 512, 768, 1023, 1024, 1500, 2048)
EXPECTED_MLX_VERSION = "0.32.2.dev20260919+39400a0d4"
DTYPE = "bfloat16"
HEAD_DIM = 256


def partition_plan(
    rows: int, total: int, *, tensor_units: bool
) -> tuple[tuple[int, int, int], ...]:
    """TensorFold pre-#197 query slices as ``(begin, end, visible_keys)``."""

    if rows < 1 or total < rows:
        raise ValueError("rows must be positive and no larger than total keys")
    if total <= 4096 or rows <= ROWS_PER_PART:
        return ((0, rows, total),)
    key_block = 1 if tensor_units else 16
    parts = []
    begin = 0
    while begin < rows:
        end = min(rows, begin + ROWS_PER_PART)
        if end < rows:
            end -= (total - rows + end) % key_block
        if 0 < rows - end <= 16:
            end = rows
        if end <= begin:
            raise RuntimeError("TensorFold partition failed to make progress")
        parts.append((begin, end, total - rows + end))
        begin = end
    return tuple(parts)


def current_mlx2_route(
    rows: int,
    head_dim: int = HEAD_DIM,
    mask="causal",
    *,
    dtype: str = DTYPE,
    sinks=None,
) -> str:
    """Static route expectation for the current mlx2 ``fast_sdpa`` seam."""

    if (
        head_dim == HEAD_DIM
        and isinstance(mask, str)
        and mask == "causal"
        and dtype in ("bfloat16", "float16")
        and sinks is None
        and 256 <= rows < FUSED_ROWS
    ):
        return "mlx2_force_fused"
    return "mlx_stock_dispatch"


def _parse_rows(value: str) -> tuple[int, ...]:
    rows = tuple(int(item) for item in value.split(",") if item)
    if not rows or any(item < 1 for item in rows):
        raise ValueError("--rows must contain positive integers")
    return rows


def execution_plan(
    *,
    rows: tuple[int, ...] = DEFAULT_ROWS,
    keys: int = 8192,
    query_heads: int = 24,
    kv_heads: int = 4,
    warmups: int = 2,
    rounds: int = 7,
    seed: int = 197,
) -> dict:
    """Describe every allocation and arm without importing MLX."""

    if keys < 1 or query_heads < 1 or kv_heads < 1:
        raise ValueError("keys and head counts must be positive")
    if warmups < 0 or rounds < 1:
        raise ValueError("warmups must be nonnegative and rounds positive")
    cases = []
    for row_count in rows:
        total = max(row_count, keys)
        cases.append(
            {
                "rows": row_count,
                "keys": total,
                "inputs": {
                    "queries": [1, query_heads, row_count, HEAD_DIM],
                    "keys": [1, kv_heads, total, HEAD_DIM],
                    "values": [1, kv_heads, total, HEAD_DIM],
                    "dtype": DTYPE,
                    "random_distribution": "normal",
                },
                "call": {
                    "scale": HEAD_DIM**-0.5,
                    "mask": "causal",
                    "sinks": None,
                },
                "expected_mlx2_route": current_mlx2_route(row_count),
                "tensorfold_pre197_parts": [
                    list(part)
                    for part in partition_plan(row_count, total, tensor_units=True)
                ],
            }
        )
    return {
        "seed": seed,
        "warmups_per_arm": warmups,
        "measured_rounds_per_arm": rounds,
        "counterbalance": "forward arm order on even rounds, reverse on odd rounds",
        "synchronization": "mx.eval each output",
        "case_order": "rows in command-line order",
        "cases": cases,
    }


def deferred_gpu_command(args=None) -> list[str]:
    rows = ",".join(map(str, DEFAULT_ROWS)) if args is None else args.rows
    keys = 8192 if args is None else args.keys
    query_heads = 24 if args is None else args.query_heads
    kv_heads = 4 if args is None else args.kv_heads
    warmups = 2 if args is None else args.warmups
    rounds = 7 if args is None else args.rounds
    seed = 197 if args is None else args.seed
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_with_gpu_locks.py"),
        "--session",
        "upstream-viability-benches-20261004",
        "--label",
        "tensorfold-sdpa-197",
        "--receipt",
        str(ROOT / "artifacts" / "research" / "tensorfold-sdpa-197-lock.json"),
        "--",
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-gpu",
        "--i-hold-gpu-lease",
        "--rows",
        rows,
        "--keys",
        str(keys),
        "--query-heads",
        str(query_heads),
        "--kv-heads",
        str(kv_heads),
        "--warmups",
        str(warmups),
        "--rounds",
        str(rounds),
        "--seed",
        str(seed),
        "--output",
        str(ROOT / "artifacts" / "research" / "tensorfold-sdpa-197.json"),
    ]


def metadata(args=None) -> dict:
    rows = DEFAULT_ROWS if args is None else _parse_rows(args.rows)
    plan = execution_plan(
        rows=rows,
        keys=8192 if args is None else args.keys,
        query_heads=24 if args is None else args.query_heads,
        kv_heads=4 if args is None else args.kv_heads,
        warmups=2 if args is None else args.warmups,
        rounds=7 if args is None else args.rounds,
        seed=197 if args is None else args.seed,
    )
    return {
        "schema": "mlx2.tensorfold-sdpa-threshold-bench.v1",
        "status": "research_harness_unexecuted",
        "source": {"pr": SOURCE_PR, "head": SOURCE_HEAD},
        "current_mlx2": {
            "already_implemented": True,
            "detail": (
                "mlx2 already forces fused head-dim-256 causal SDPA for rows "
                "256..1023 and leaves 1024+ rows to stock MLX; ordinary idle "
                "prefill already uses 2048-row chunks"
            ),
            "production_change": False,
        },
        "arms": {
            "stock": "one mx.fast.scaled_dot_product_attention call",
            "mlx2_fast": "current mlx2.runtime.models.base.fast_sdpa",
            "tensorfold_pre197": "128-row query partitions used before TensorFold #197",
        },
        "environment": environment_snapshot(mlx_version=EXPECTED_MLX_VERSION),
        "deferred_gpu_command": deferred_gpu_command(args),
        "execution_plan": plan,
        "capability_tuple": {
            "head_dim": HEAD_DIM,
            "dtypes": ["bfloat16", "float16"],
            "mask": "causal_string_only",
            "sinks": None,
            "row_range": [256, 1023],
        },
        "gates": [
            "stock and mlx2 outputs are bit equal wherever mlx2 does not force another kernel",
            "all outputs finite",
            "record rather than assume the 1023/1024 crossover",
            "component timing is not a serving or contention result",
        ],
        "known_caveats": [
            "mlx2 already measured wider contended slices and retained its 512-row cap because decode p99 doubled",
            "explicit array masks and attention sinks are outside TensorFold #197's causal-mask route",
            "the dispatch threshold belongs to the installed MLX build and must be remeasured after upgrades",
        ],
    }


def _evaluate(mx, value) -> None:
    if isinstance(value, (tuple, list)):
        mx.eval(*value)
    else:
        mx.eval(value)


def _partitioned_attention(mx, q, k, v, scale: float):
    rows, total = int(q.shape[2]), int(k.shape[2])
    outs = []
    for begin, end, visible in partition_plan(rows, total, tensor_units=True):
        part = mx.fast.scaled_dot_product_attention(
            q[:, :, begin:end],
            k[:, :, :visible],
            v[:, :, :visible],
            scale=scale,
            mask="causal",
        )
        mx.async_eval(part)
        if outs:
            mx.eval(outs[-1])
        outs.append(part)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


def _run(args) -> dict:
    if not args.i_hold_gpu_lease:
        raise RuntimeError("--run-gpu requires --i-hold-gpu-lease")
    owner = require_matching_gpu_receipts()
    runtime = require_distribution("mlx", EXPECTED_MLX_VERSION)
    rows_to_run = _parse_rows(args.rows)
    execution_plan(
        rows=rows_to_run,
        keys=args.keys,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        warmups=args.warmups,
        rounds=args.rounds,
        seed=args.seed,
    )

    import mlx.core as mx

    from mlx2.runtime.models import base

    mx.set_default_device(mx.gpu)
    info = mx.device_info()
    if not mx.metal.is_available() or "M5" not in str(info.get("device_name", "")):
        raise RuntimeError("TensorFold #197 bench requires an M5 Metal GPU")

    mx.random.seed(args.seed)
    records = []
    for rows in rows_to_run:
        total = max(rows, args.keys)
        q = mx.random.normal((1, args.query_heads, rows, 256)).astype(mx.bfloat16)
        k = mx.random.normal((1, args.kv_heads, total, 256)).astype(mx.bfloat16)
        v = mx.random.normal((1, args.kv_heads, total, 256)).astype(mx.bfloat16)
        mx.eval(q, k, v)
        scale = 256**-0.5
        arms = {
            "stock": lambda q=q, k=k, v=v, scale=scale: (
                mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask="causal"
                )
            ),
            "mlx2_fast": lambda q=q, k=k, v=v, scale=scale: base.fast_sdpa(
                q, k, v, scale=scale, mask="causal"
            ),
            "tensorfold_pre197": lambda q=q, k=k, v=v, scale=scale: (
                _partitioned_attention(mx, q, k, v, scale)
            ),
        }
        outputs = {name: call() for name, call in arms.items()}
        mx.eval(*outputs.values())
        reference = outputs["stock"]
        checks = {}
        for name, value in outputs.items():
            delta = mx.abs(value.astype(mx.float32) - reference.astype(mx.float32))
            checks[name] = {
                "finite": bool(mx.all(mx.isfinite(value)).item()),
                "bit_equal_to_stock": bool(mx.array_equal(value, reference).item()),
                "max_abs_to_stock": float(mx.max(delta).item()),
            }
        timings = timed_arms(
            lambda value: _evaluate(mx, value),
            arms,
            warmups=args.warmups,
            rounds=args.rounds,
        )
        for name in ("mlx2_fast", "tensorfold_pre197"):
            timings[name]["speedup_vs_stock"] = (
                timings["stock"]["median_ms"] / timings[name]["median_ms"]
            )
        records.append(
            {
                "rows": rows,
                "keys": total,
                "current_mlx2_route": current_mlx2_route(rows),
                "tensorfold_parts": [
                    list(part)
                    for part in partition_plan(rows, total, tensor_units=True)
                ],
                "checks": checks,
                "timing": timings,
            }
        )
    numerical_gate = all(
        check["finite"] for record in records for check in record["checks"].values()
    ) and all(
        record["checks"]["mlx2_fast"]["bit_equal_to_stock"]
        for record in records
        if record["current_mlx2_route"] == "mlx_stock_dispatch"
    )
    return {
        **metadata(args),
        "status": "component_benchmark_complete_unqualified",
        "source_head": git_revision(ROOT),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu_owner": owner,
        "mlx_version": mx.__version__,
        "runtime_pin": runtime,
        "device": info,
        "parameters": {
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "warmups": args.warmups,
            "rounds": args.rounds,
        },
        "numerical_gate": numerical_gate,
        "results": records,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--i-hold-gpu-lease", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rows", default=",".join(map(str, DEFAULT_ROWS)))
    parser.add_argument("--keys", type=int, default=8192)
    parser.add_argument("--query-heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=197)
    args = parser.parse_args(argv)
    result = _run(args) if args.run_gpu else metadata(args)
    rendered = json.dumps(result, indent=2, default=str) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if not args.run_gpu or result["numerical_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
