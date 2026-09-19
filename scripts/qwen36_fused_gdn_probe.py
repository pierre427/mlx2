#!/usr/bin/env python3
"""Reference-versus-fused Qwen3.6 decode probe; never selects a route."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src" / "mlx2").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def set_mode(model, mode: str):
    from mlx2.runtime.models.qwen36_35b import GatedDeltaNet

    count = 0
    for _, module in model.named_modules():
        if isinstance(module, GatedDeltaNet):
            module.set_fused_gdn_decode_mode(mode)
            count += 1
    return count


def max_difference(mx, left, right) -> float:
    value = mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32)))
    mx.eval(value)
    return float(value.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--prompt", default="Write a safe Python merge sort implementation.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = {
        "schema": "mlx2.qwen36-fused-gdn-probe.v1",
        "model": str(Path(args.model).resolve()),
        "source_hash": source_hash(root),
        "route_selected": False,
        "qualified": False,
        "passed": False,
        "started_at": time.time(),
    }
    try:
        import mlx.core as mx
        from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter

        load_started = time.perf_counter()
        adapter = Qwen3635BA3BAdapter(args.model)
        from mlx2.runtime.models.qwen36_35b import qwen36_fused_gdn_stats
        report["load_seconds"] = time.perf_counter() - load_started
        tokens = adapter.tokenizer.encode(args.prompt, add_special_tokens=False)
        if len(tokens) < 2:
            raise ValueError("probe prompt is too short")
        stock_cache = adapter.model.make_cache()
        fused_cache = adapter.model.make_cache()

        set_mode(adapter.model, "stock")
        stock_prefill = adapter.model(mx.array(tokens)[None], cache=stock_cache)
        mx.eval(stock_prefill)
        fused_prefill = adapter.model(mx.array(tokens)[None], cache=fused_cache)
        mx.eval(fused_prefill)
        prefill_difference = max_difference(mx, stock_prefill, fused_prefill)

        next_token = int(mx.argmax(stock_prefill[:, -1], axis=-1).item())
        logit_differences = []
        conv_differences = []
        recurrent_differences = []
        same_argmax = []
        stock_seconds = 0.0
        fused_seconds = 0.0
        stock_step_seconds = []
        fused_step_seconds = []
        linear_indices = [
            index for index, layer in enumerate(adapter.model.layers) if layer.is_linear
        ]
        for _ in range(args.steps):
            token = mx.array([[next_token]])
            set_mode(adapter.model, "stock")
            started = time.perf_counter()
            stock = adapter.model(token, cache=stock_cache)
            mx.eval(stock)
            elapsed = time.perf_counter() - started
            stock_seconds += elapsed
            stock_step_seconds.append(elapsed)

            set_mode(adapter.model, "fused")
            started = time.perf_counter()
            fused = adapter.model(token, cache=fused_cache)
            mx.eval(fused)
            elapsed = time.perf_counter() - started
            fused_seconds += elapsed
            fused_step_seconds.append(elapsed)

            logit_differences.append(max_difference(mx, stock, fused))
            stock_argmax = int(mx.argmax(stock[:, -1], axis=-1).item())
            fused_argmax = int(mx.argmax(fused[:, -1], axis=-1).item())
            same_argmax.append(stock_argmax == fused_argmax)
            next_token = stock_argmax
            conv_differences.append(
                max(
                    max_difference(mx, stock_cache[index][0], fused_cache[index][0])
                    for index in linear_indices
                )
            )
            recurrent_differences.append(
                max(
                    max_difference(mx, stock_cache[index][1], fused_cache[index][1])
                    for index in linear_indices
                )
            )

        stats = qwen36_fused_gdn_stats(adapter.model)
        report.update(
            {
                "prompt_tokens": len(tokens),
                "decode_steps": args.steps,
                "linear_gdn_layers": len(linear_indices),
                "prefill_max_abs_difference": prefill_difference,
                "logit_max_abs_differences": logit_differences,
                "conv_state_max_abs_differences": conv_differences,
                "recurrent_state_max_abs_differences": recurrent_differences,
                "same_argmax_all_steps": all(same_argmax),
                "stock_decode_seconds": stock_seconds,
                "fused_decode_seconds": fused_seconds,
                "speedup": stock_seconds / fused_seconds if fused_seconds else None,
                "stock_step_seconds": stock_step_seconds,
                "fused_step_seconds": fused_step_seconds,
                "warm_speedup": (
                    sum(stock_step_seconds[1:]) / sum(fused_step_seconds[1:])
                    if sum(fused_step_seconds[1:])
                    else None
                ),
                "stats": stats,
                "peak_memory_bytes": int(mx.get_peak_memory()),
            }
        )
        finite = all(
            math.isfinite(value)
            for values in (logit_differences, conv_differences, recurrent_differences)
            for value in values
        )
        report["passed"] = bool(
            finite
            and prefill_difference == 0.0
            and all(value == 0.0 for value in logit_differences)
            and all(value == 0.0 for value in conv_differences)
            and all(value == 0.0 for value in recurrent_differences)
            and all(same_argmax)
            and stats["fused_calls"] == len(linear_indices) * args.steps
            and stats["fallbacks"] == 0
        )
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        report["finished_at"] = time.time()
        report["elapsed_seconds"] = report["finished_at"] - report["started_at"]
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
