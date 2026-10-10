"""Deterministic host scheduling laboratory; no device-performance prediction.

Time is in synthetic units. The cost model is deliberately explicit and has
not been fitted to MLX. The time-budget arm has an oracle cost predictor; its
results describe an idealized scheduling tradeoff, not an implementable SLO.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mlx2.runtime.adaptive_policy import PrefillOrder


@dataclass(frozen=True)
class Request:
    uid: int
    arrival: float
    prefill: int
    output: int
    cached: int = 0
    cancel_at: float | None = None

    def __post_init__(self):
        for name in ("uid", "prefill", "cached"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.output) is not int or self.output < 1:
            raise ValueError("output must be a positive integer")
        if not math.isfinite(self.arrival) or self.arrival < 0:
            raise ValueError("arrival must be finite and nonnegative")
        if self.cancel_at is not None and (
            not math.isfinite(self.cancel_at) or self.cancel_at < self.arrival
        ):
            raise ValueError("cancellation must be finite and follow arrival")


@dataclass
class State:
    request: Request
    remaining: int
    generated: int = 0
    prefilled: int = 0
    deliveries: list[float] = field(default_factory=list)
    prefill_services: list[float] = field(default_factory=list)
    chunks: list[int] = field(default_factory=list)
    cancelled: bool = False


def quantile(values, p):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * p) - 1)] if values else None


def prefill_cost(rows, depth):
    # Hypothetical long-context work; no hardware calibration is implied.
    return 2.0 + rows * 0.12 * (1.0 + depth / 8192.0)


def decode_cost(width):
    return 0.0 if not width else 3.0 + 0.4 * width


def simulate(
    requests,
    *,
    order="fcfs",
    chunk=256,
    publication="early",
    round_budget=None,
    aging_weight=8.0,
    max_resident=16,
    max_rounds=100000,
):
    if order not in {"fcfs", "round_robin", "srpt", "bounded_srpt", "aging"}:
        raise ValueError("unknown prefill order")
    if publication not in {"early", "late"}:
        raise ValueError("unknown publication mode")
    if type(chunk) is not int or chunk < 1:
        raise ValueError("chunk must be a positive integer")
    if type(max_resident) is not int or max_resident < 1:
        raise ValueError("max_resident must be positive")
    if type(max_rounds) is not int or max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    if round_budget is not None and (
        not math.isfinite(round_budget) or round_budget <= 0
    ):
        raise ValueError("round budget must be positive and finite")
    if not math.isfinite(aging_weight) or aging_weight <= 0:
        raise ValueError("aging weight must be positive and finite")
    requests = list(requests)
    if len({r.uid for r in requests}) != len(requests):
        raise ValueError("request ids must be unique")
    # Production PrefillOrder defines uid order as insertion order.
    arrivals = sorted(requests, key=lambda r: (r.arrival, r.uid))
    if [r.uid for r in arrivals] != sorted(r.uid for r in requests):
        raise ValueError("uids must follow arrival order")
    states = {r.uid: State(r, r.prefill) for r in arrivals}
    pending = list(arrivals)
    resident = set()
    done = set()
    now = 0.0
    rounds = 0
    over_budget = 0
    last_served = -1
    order_policy = PrefillOrder(enabled=True, max_bypass=3)

    def cancel():
        for uid, state in states.items():
            at = state.request.cancel_at
            if uid not in done and at is not None and now >= at:
                state.cancelled = True
                done.add(uid)
                resident.discard(uid)

    while len(done) < len(states):
        if rounds >= max_rounds:
            raise RuntimeError("simulation exceeded bounded rounds")
        cancel()
        pending = [r for r in pending if r.uid not in done]
        while pending and pending[0].arrival <= now and len(resident) < max_resident:
            resident.add(pending.pop(0).uid)
        if not resident:
            if not pending:
                break
            now = max(now, pending[0].arrival)
            continue
        rounds += 1
        started = now
        waiting = [states[u] for u in sorted(resident) if states[u].remaining]
        active = [states[u] for u in sorted(resident) if not states[u].remaining]
        now += decode_cost(len(active))
        decode_finished = now
        cancel()
        ready = [s for s in active if not s.cancelled]
        for state in ready:
            state.generated += 1

        def publish(at, ready=ready):
            for state in ready:
                if (
                    state.request.cancel_at is not None
                    and at >= state.request.cancel_at
                ):
                    state.cancelled = True
                    done.add(state.request.uid)
                    resident.discard(state.request.uid)
                    continue
                state.deliveries.append(at)
                if state.generated == state.request.output:
                    done.add(state.request.uid)
                    resident.discard(state.request.uid)

        if publication == "early":
            publish(decode_finished)
        waiting = [s for s in waiting if not s.cancelled]
        if waiting:
            candidates = [
                order_policy.candidate(s.request.uid, s.remaining, s.request.cached)
                for s in waiting
            ]
            if order == "bounded_srpt":
                chosen = waiting[order_policy.select(candidates)]
            elif order == "round_robin":
                chosen = next(
                    (s for s in waiting if s.request.uid > last_served), waiting[0]
                )
            elif order == "fcfs":
                chosen = waiting[0]
            elif order == "srpt":
                chosen = min(
                    waiting,
                    key=lambda s: (s.remaining, -s.request.cached, s.request.uid),
                )
            else:
                chosen = min(
                    waiting,
                    key=lambda s: (
                        s.remaining - aging_weight * (now - s.request.arrival),
                        s.request.uid,
                    ),
                )
            width = min(chunk, chosen.remaining)
            depth = chosen.request.cached + chosen.prefilled
            if round_budget is not None:
                available = round_budget - (now - started)
                # Search all integer sizes in the bounded host model. A real
                # scheduler would need calibrated candidates and an error margin.
                fits = [
                    n
                    for n in range(1, width + 1)
                    if prefill_cost(n, depth) <= available
                ]
                width = max(fits, default=1)  # preserve progress; count the miss
            chosen.prefill_services.append(now)
            chosen.chunks.append(width)
            chosen.remaining -= width
            chosen.prefilled += width
            last_served = chosen.request.uid
            order_policy.commit(
                [last_served],
                candidates,
                pending=[s.request.uid for s in waiting if s.remaining],
            )
            now += prefill_cost(width, depth)
        if publication == "late":
            publish(now)
        cancel()
        if round_budget is not None and now - started > round_budget + 1e-9:
            over_budget += 1

    completed = [s for s in states.values() if not s.cancelled]
    ttfts = [s.deliveries[0] - s.request.arrival for s in completed]
    gaps = [
        b - a for s in states.values() for a, b in zip(s.deliveries, s.deliveries[1:])
    ]
    service_gaps = [
        b - a
        for s in states.values()
        for a, b in zip([s.request.arrival, *s.prefill_services], s.prefill_services)
    ]
    return {
        "scope": "synthetic host simulation; no GPU measurement or qualification",
        "units": "synthetic_time_units",
        "policy": {
            "order": order,
            "chunk": chunk,
            "publication": publication,
            "round_budget": round_budget,
            "max_resident": max_resident,
            "aging_weight": aging_weight if order == "aging" else None,
            "max_bypass": 3 if order == "bounded_srpt" else None,
        },
        "rounds": rounds,
        "makespan": now,
        "completed": len(completed),
        "cancelled": sum(s.cancelled for s in states.values()),
        "budget_exceeded_rounds": over_budget,
        "ttft_p50": quantile(ttfts, 0.5),
        "ttft_p95": quantile(ttfts, 0.95),
        "delivery_gap_p95": quantile(gaps, 0.95),
        "delivery_gap_max": max(gaps, default=0),
        "prefill_service_gap_max": max(service_gaps, default=0),
        "requests": [
            {
                **asdict(s.request),
                "prefilled": s.prefilled,
                "delivered": len(s.deliveries),
                "cancelled": s.cancelled,
                "first_delivery": s.deliveries[0] if s.deliveries else None,
                "last_delivery": s.deliveries[-1] if s.deliveries else None,
                "first_prefill_service": s.prefill_services[0]
                if s.prefill_services
                else None,
                "chunks": s.chunks,
            }
            for s in states.values()
        ],
    }


def scenarios():
    return {
        "mixed_burst": [
            Request(0, 0, 8192, 16),
            *[Request(i, 0, 96 + (i % 4) * 128, 16) for i in range(1, 13)],
        ],
        "decode_with_arrivals": [
            Request(0, 0, 0, 100),
            Request(1, 0, 0, 100),
            *[
                Request(i + 2, 20 + i * 15, 128 if i % 3 else 2048, 16, cached=4096)
                for i in range(12)
            ],
        ],
        "long_among_short": [
            Request(0, 0, 8192, 8),
            *[Request(i + 1, i * 8, 64, 8) for i in range(48)],
        ],
        "cancelled_backlog": [
            Request(0, 0, 4096, 20, cancel_at=90),
            Request(1, 1, 2048, 20, cancel_at=20),
            *[Request(i + 2, 2 + i, 512, 16) for i in range(8)],
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refuse to overwrite evidence")
    results = []
    for name, requests in scenarios().items():
        for order in ("fcfs", "round_robin", "srpt", "bounded_srpt", "aging"):
            for chunk, budget in ((128, None), (256, None), (256, 25.0)):
                for publication in ("early", "late"):
                    for weight in (
                        (1.0, 8.0, 32.0, 128.0) if order == "aging" else (8.0,)
                    ):
                        result = simulate(
                            requests,
                            order=order,
                            chunk=chunk,
                            round_budget=budget,
                            publication=publication,
                            aging_weight=weight,
                        )
                        result["scenario"] = name
                        results.append(result)
    report = {
        "schema": "mlx2.prefill-policy-simulation.v1",
        "limitations": [
            "Synthetic costs, not calibrated device timings",
            "Time-budget policy uses an oracle predictor",
            "No cache/memory-pressure, tensor state or speculative verification model",
            "Different chunk geometry needs separate numerical validation",
            "No live executor policy is selected by this script",
        ],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cells": len(results), "scope": report["limitations"]}))


if __name__ == "__main__":
    main()
