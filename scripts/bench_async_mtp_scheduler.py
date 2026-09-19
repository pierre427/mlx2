#!/usr/bin/env python3
"""Deterministic CPU microbenchmark for request-local draft-depth policies."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from mlx2.runtime.async_mtp_scheduler import (
    AspireDepthPolicy,
    CostModel,
    FeedbackDepthPolicy,
    FixedDepthPolicy,
    RequestContext,
    TraceRecord,
    replay_trace,
)


def synthetic_trace(*, requests: int, rounds: int, seed: int) -> list[TraceRecord]:
    rng = random.Random(seed)
    bases = (0.34, 0.48, 0.61, 0.74, 0.86, 0.94)
    records: list[TraceRecord] = []
    for round_index in range(rounds):
        for request_index in range(requests):
            phase = math.sin(round_index / 37 + request_index * 0.91) * 0.13
            acceptance_probability = min(
                0.985, max(0.05, bases[request_index % len(bases)] + phase)
            )
            accepted_prefix = 0
            while accepted_prefix < 8 and rng.random() < acceptance_probability:
                accepted_prefix += 1
            context_length = 16_384 + request_index * 2_048 + round_index * 2
            records.append(
                TraceRecord(
                    request_id=f"request-{request_index}",
                    round_index=round_index,
                    context_length=context_length,
                    sparse_context_length=min(2_048, context_length),
                    batch_size=requests,
                    max_depth=8,
                    accepted_prefix=accepted_prefix,
                )
            )
    return records


def load_trace(path: Path) -> list[TraceRecord]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload["records"]
    return [TraceRecord(**row) for row in payload]


def scheduler_overhead(cost_model: CostModel, iterations: int) -> dict[str, float]:
    context = RequestContext("microbench", 32_768, 2_048, 8)
    policies = {
        "fixed_k2": FixedDepthPolicy(2),
        "feedback": FeedbackDepthPolicy(),
        "aspire_style": AspireDepthPolicy(cost_model),
    }
    results: dict[str, float] = {}
    for name, policy in policies.items():
        start = time.perf_counter_ns()
        for index in range(iterations):
            depth = policy.choose_depth(context)
            accepted = depth if index % 3 else max(0, depth - 1)
            policy.observe(depth, accepted)
        elapsed = time.perf_counter_ns() - start
        results[name] = elapsed / iterations
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=260917943)
    parser.add_argument("--overhead-iterations", type=int, default=100_000)
    args = parser.parse_args()
    if args.requests <= 0 or args.rounds <= 0 or args.overhead_iterations <= 0:
        parser.error("requests, rounds, and overhead-iterations must be positive")

    cost_model = CostModel(beta_model=0.2, beta_mlp=0.01, beta_attn=0.00003)
    records = (
        load_trace(args.trace)
        if args.trace
        else synthetic_trace(requests=args.requests, rounds=args.rounds, seed=args.seed)
    )
    policies = (
        ("fixed_k2", lambda: FixedDepthPolicy(2)),
        ("feedback", lambda: FeedbackDepthPolicy()),
        ("aspire_style", lambda: AspireDepthPolicy(cost_model)),
    )
    replay = {
        name: replay_trace(
            records,
            policy_name=name,
            policy_factory=factory,
            cost_model=cost_model,
        ).as_dict()
        for name, factory in policies
    }
    fixed_throughput = replay["fixed_k2"]["throughput"]
    for result in replay.values():
        result["relative_to_fixed_k2"] = result["throughput"] / fixed_throughput

    print(
        json.dumps(
            {
                "schema": "mlx2.async-mtp-scheduler-microbench.v1",
                "scope": "CPU-only counterfactual policy economics; no mixed-forward execution",
                "trace_records": len(records),
                "cost_model": {
                    "beta_model": cost_model.beta_model,
                    "beta_mlp": cost_model.beta_mlp,
                    "beta_attn": cost_model.beta_attn,
                },
                "replay": replay,
                "scheduler_ns_per_round": scheduler_overhead(
                    cost_model, args.overhead_iterations
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
