#!/usr/bin/env python3
"""Isolated M=3 Qwen4 router + q4 fused-down candidate microbenchmark.

The default action is metadata-only and does not import MLX. A run requires an
explicit assertion that the caller holds the GPU lease; this script does not
acquire a lease or touch ``gpu.lock`` itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time


SCHEMA = "mlx2.qwen4-moe-m3-candidate-benchmark.v2"
LOCK_RECEIPTS = (
    Path("/tmp/gpu.lock/owner.json"),
    Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
)
METADATA = {
    "schema": SCHEMA,
    "status": "implemented_candidate_unqualified",
    "token_width": 3,
    "geometry": {
        "num_experts": 512,
        "top_k": 10,
        "expert_hidden_size": 640,
        "hidden_size": 2560,
        "quantization": "affine-q4-group64",
    },
    "arms": {
        "stock": "precise softmax + argpartition + gather_qmm + weighted sum",
        "candidate": "M=3 Metal router + tile4 q4 down/weight/reduce kernel",
    },
    "engagement_gate": ["candidate_router_calls > 0", "candidate_down_calls > 0"],
    "qualification": "A passing microbenchmark is evidence, not route qualification.",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--describe", action="store_true", help="print metadata only")
    mode.add_argument("--run-gpu", action="store_true", help="execute the GPU benchmark")
    parser.add_argument(
        "--i-hold-gpu-lease",
        action="store_true",
        help="required attestation; acquire the project GPU lease outside this script",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--router-score-atol", type=float, default=1.0 / 128.0)
    parser.add_argument("--output-atol", type=float, default=1.0 / 128.0)
    return parser


def _stock_router(mx, gates, top_k: int):
    probabilities = mx.softmax(gates, axis=-1, precise=True)
    indices = mx.argpartition(probabilities, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(probabilities, indices, axis=-1)
    return indices, scores / scores.sum(axis=-1, keepdims=True)


def _stock_down(mx, hidden, indices, scores, weight, scales, biases):
    rows = mx.gather_qmm(
        hidden[..., None, :],
        weight,
        scales,
        biases,
        rhs_indices=indices,
        transpose=True,
        group_size=64,
        bits=4,
        mode="affine",
        sorted_indices=False,
    ).squeeze(-2)
    return (rows * scores[..., None]).sum(axis=-2)


def _max_abs(mx, left, right) -> float:
    delta = mx.abs(left.astype(mx.float32) - right.astype(mx.float32))
    return float(mx.max(delta).item())


def _load_lock_receipts() -> list[dict]:
    missing = [str(path) for path in LOCK_RECEIPTS if not path.is_file()]
    if missing:
        raise RuntimeError(f"GPU benchmark requires both lock receipts; missing {missing}")
    receipts = [json.loads(path.read_text()) for path in LOCK_RECEIPTS]
    identity_keys = ("campaign_id", "lease_id", "generation", "session_id")
    identities = {tuple(receipt.get(key) for key in identity_keys) for receipt in receipts}
    if len(identities) != 1 or any(receipt.get("resource_key") != "gpu" for receipt in receipts):
        raise RuntimeError("GPU lock receipts do not describe one matching GPU lease")
    return receipts


def _measure_pair(mx, arms: dict, *, warmups: int, rounds: int) -> dict:
    for _ in range(warmups):
        for arm in arms.values():
            result = arm()
            mx.eval(*result) if isinstance(result, tuple) else mx.eval(result)
    samples = {name: [] for name in arms}
    names = tuple(arms)
    for round_index in range(rounds):
        order = names if round_index % 2 == 0 else tuple(reversed(names))
        for name in order:
            started = time.perf_counter_ns()
            result = arms[name]()
            mx.eval(*result) if isinstance(result, tuple) else mx.eval(result)
            samples[name].append((time.perf_counter_ns() - started) / 1_000_000.0)
    summary = {
        name: {
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "samples_ms": values,
        }
        for name, values in samples.items()
    }
    summary["candidate"]["speedup_vs_stock"] = (
        summary["stock"]["median_ms"] / summary["candidate"]["median_ms"]
    )
    return summary


def _run(args) -> dict:
    locks = _load_lock_receipts()
    import mlx.core as mx

    from mlx2.runtime.models.qwen4_fused_moe import qwen4_fused_down
    from mlx2.runtime.models.qwen4_moe_router import qwen4_moe_router

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("Metal GPU is unavailable; refusing to label a CPU run")

    mx.random.seed(args.seed)
    m, experts, top_k, eh, hidden_size = 3, 512, 10, 640, 2560
    gates = mx.random.uniform(-4.0, 4.0, shape=(1, m, experts)).astype(mx.bfloat16)
    hidden = mx.random.uniform(-1.0, 1.0, shape=(1, m, top_k, eh)).astype(
        mx.bfloat16
    )
    weight = mx.random.randint(
        0, 2**31, shape=(experts, hidden_size, eh // 8)
    ).astype(mx.uint32)
    scales = mx.random.uniform(
        -1.0 / 128.0,
        1.0 / 128.0,
        shape=(experts, hidden_size, eh // 64),
    ).astype(mx.bfloat16)
    biases = mx.random.uniform(
        -1.0 / 128.0,
        1.0 / 128.0,
        shape=(experts, hidden_size, eh // 64),
    ).astype(mx.bfloat16)
    mx.eval(gates, hidden, weight, scales, biases)

    engagement = {"candidate_router_calls": 0, "candidate_down_calls": 0}

    def stock():
        indices, scores = _stock_router(mx, gates, top_k)
        return indices, scores, _stock_down(
            mx, hidden, indices, scores, weight, scales, biases
        )

    def candidate():
        indices, scores = qwen4_moe_router(gates, candidate_token_widths=(m,))
        engagement["candidate_router_calls"] += 1
        out = qwen4_fused_down(
            hidden,
            indices,
            scores,
            weight,
            scales,
            biases,
            variant="tile4",
            candidate_token_widths=(m,),
        )
        engagement["candidate_down_calls"] += 1
        return indices, scores, out

    stock_result = stock()
    candidate_result = candidate()
    mx.eval(*stock_result, *candidate_result)
    indices_equal = bool(mx.all(stock_result[0] == candidate_result[0]).item())
    router_score_max_abs = _max_abs(mx, stock_result[1], candidate_result[1])
    output_max_abs = _max_abs(mx, stock_result[2], candidate_result[2])
    parity = {
        "indices_equal": indices_equal,
        "router_score_max_abs": router_score_max_abs,
        "router_score_atol": args.router_score_atol,
        "output_max_abs": output_max_abs,
        "output_atol": args.output_atol,
        "passed": indices_equal
        and router_score_max_abs <= args.router_score_atol
        and output_max_abs <= args.output_atol,
    }

    timing = _measure_pair(
        mx,
        {"stock": stock, "candidate": candidate},
        warmups=args.warmups,
        rounds=args.rounds,
    )

    stock_indices, stock_scores = stock_result[:2]

    def candidate_router():
        engagement["candidate_router_calls"] += 1
        return qwen4_moe_router(gates, candidate_token_widths=(m,))

    def stock_down():
        return _stock_down(
            mx, hidden, stock_indices, stock_scores, weight, scales, biases
        )

    def candidate_down():
        engagement["candidate_down_calls"] += 1
        return qwen4_fused_down(
            hidden,
            stock_indices,
            stock_scores,
            weight,
            scales,
            biases,
            variant="tile4",
            candidate_token_widths=(m,),
        )

    component_timing = {
        "router": _measure_pair(
            mx,
            {"stock": lambda: _stock_router(mx, gates, top_k), "candidate": candidate_router},
            warmups=args.warmups,
            rounds=args.rounds,
        ),
        "down_reduce": _measure_pair(
            mx,
            {"stock": stock_down, "candidate": candidate_down},
            warmups=args.warmups,
            rounds=args.rounds,
        ),
    }

    engaged = all(value > 0 for value in engagement.values())
    return {
        **METADATA,
        "locks": locks,
        "parameters": {
            "warmups": args.warmups,
            "rounds": args.rounds,
            "seed": args.seed,
            "order": "counterbalanced",
        },
        "parity": parity,
        "engagement": {**engagement, "passed": engaged},
        "timing": timing,
        "component_timing": component_timing,
        "passed": parity["passed"] and engaged,
    }


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.run_gpu:
        print(json.dumps(METADATA, indent=2, sort_keys=True))
        return 0
    if not args.i_hold_gpu_lease:
        parser.error("--run-gpu requires --i-hold-gpu-lease")
    if args.output is None:
        parser.error("--run-gpu requires --output")
    if args.warmups < 1 or args.rounds < 2:
        parser.error("--warmups must be >= 1 and --rounds must be >= 2")
    if args.router_score_atol < 0 or args.output_atol < 0:
        parser.error("tolerances must be non-negative")

    report = _run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
