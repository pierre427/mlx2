"""Exercise both Laguna fused MoE kernels through all 39 sparse layers."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from mlx2.adapters.laguna_xs21 import LagunaXS21Adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    adapter = LagunaXS21Adapter(args.model)
    token = adapter.tokenizer.encode("Hello", add_special_tokens=False)[-1]
    inputs = mx.array([[token]], dtype=mx.int32)
    arms = {
        "stock": (False, False),
        "down": (True, False),
        "router": (False, True),
        "both": (True, True),
    }
    outputs = {}
    result = {"model_identity": adapter.identity, "token_id": int(token), "arms": {}}
    try:
        for name, (down, router) in arms.items():
            adapter.set_fused_moe_modes(down=down, router=router)
            before = adapter.model.fused_moe_stats()
            for _ in range(args.warmup):
                mx.eval(adapter.model(inputs))
            samples = []
            for _ in range(args.repetitions):
                start = time.perf_counter_ns()
                outputs[name] = adapter.model(inputs)
                mx.eval(outputs[name])
                samples.append((time.perf_counter_ns() - start) / 1e6)
            after = adapter.model.fused_moe_stats()
            counters = {
                key: after[key] - before[key]
                for key in ("down_calls", "down_fallbacks", "router_calls", "router_fallbacks")
            }
            expected = 39 * (args.warmup + args.repetitions)
            if down and counters["down_calls"] != expected:
                raise RuntimeError(f"down engagement gate failed: {counters}")
            if router and counters["router_calls"] != expected:
                raise RuntimeError(f"router engagement gate failed: {counters}")
            if counters["down_fallbacks"] or counters["router_fallbacks"]:
                raise RuntimeError(f"fallback gate failed: {counters}")
            result["arms"][name] = {
                "median_ms": statistics.median(samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
                "counters": counters,
            }
        reference = outputs["stock"].astype(mx.float32)
        stock_ms = result["arms"]["stock"]["median_ms"]
        for name in ("down", "router", "both"):
            candidate = outputs[name].astype(mx.float32)
            delta = mx.abs(candidate - reference)
            mx.eval(delta)
            result["arms"][name]["correctness"] = {
                "max_abs": float(mx.max(delta).item()),
                "allclose": bool(mx.allclose(candidate, reference, atol=0.5, rtol=0.02).item()),
                "argmax_equal": bool(
                    mx.array_equal(mx.argmax(candidate, axis=-1), mx.argmax(reference, axis=-1)).item()
                ),
            }
            result["arms"][name]["speedup_vs_stock"] = (
                stock_ms / result["arms"][name]["median_ms"]
            )
    finally:
        adapter.set_fused_moe_modes(down=False, router=False)
        adapter.close()

    payload = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
