"""Benchmark exact Laguna layer-1 MoE stock/fused arms with hard engagement gates."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from mlx2.adapters.laguna_xs21 import LagunaXS21Adapter


def _counters(block):
    return {
        "down_calls": block.fused_down_calls,
        "down_fallbacks": block.fused_down_fallbacks,
        "router_calls": block.gate.fused_calls,
        "router_fallbacks": block.gate.fused_fallbacks,
    }


def _delta(after, before):
    return {key: after[key] - before[key] for key in before}


def _run(block, x, *, down, router, warmup, repetitions):
    block.fused_down_mode = down
    block.gate.fused_mode = router
    before = _counters(block)
    for _ in range(warmup):
        mx.eval(block(x))
    samples = []
    output = None
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        output = block(x)
        mx.eval(output)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    counters = _delta(_counters(block), before)
    expected_down = warmup + repetitions if down else 0
    expected_router = warmup + repetitions if router else 0
    if counters["down_calls"] != expected_down or counters["router_calls"] != expected_router:
        raise RuntimeError(
            f"kernel engagement gate failed down={down} router={router}: {counters}"
        )
    if counters["down_fallbacks"] or counters["router_fallbacks"]:
        raise RuntimeError(f"kernel fallback gate failed: {counters}")
    return output, {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
        "counters": counters,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    adapter = LagunaXS21Adapter(args.model)
    block = adapter.model.layers[1].mlp
    arms = {
        "stock": (False, False),
        "down": (True, False),
        "router": (False, True),
        "both": (True, True),
    }
    result = {
        "model_identity": adapter.identity,
        "device": str(mx.default_device()),
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "widths": {},
    }
    try:
        for width in args.widths:
            mx.random.seed(7300 + width)
            x = mx.random.normal((1, width, 2048)).astype(mx.bfloat16)
            outputs = {}
            records = {}
            for name, (down, router) in arms.items():
                outputs[name], records[name] = _run(
                    block, x, down=down, router=router,
                    warmup=args.warmup, repetitions=args.repetitions,
                )
            reference = outputs["stock"].astype(mx.float32)
            for name in ("down", "router", "both"):
                candidate = outputs[name].astype(mx.float32)
                delta = mx.abs(candidate - reference)
                scale = mx.maximum(mx.abs(reference), 1e-5)
                mx.eval(delta, scale)
                records[name]["correctness"] = {
                    "max_abs": float(mx.max(delta).item()),
                    "max_rel": float(mx.max(delta / scale).item()),
                    "allclose_bf16": bool(
                        mx.allclose(candidate, reference, atol=0.5, rtol=0.02).item()
                    ),
                }
                records[name]["speedup_vs_stock"] = (
                    records["stock"]["median_ms"] / records[name]["median_ms"]
                )
            result["widths"][str(width)] = records
    finally:
        adapter.set_fused_moe_modes(down=False, router=False)
        adapter.close()

    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
