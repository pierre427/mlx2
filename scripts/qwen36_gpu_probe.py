#!/usr/bin/env python3
"""Bounded Qwen3.6 load/replay probe; never selects or installs a route."""

import argparse
import json
import math
from pathlib import Path
import platform
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="Write a Python function that reverses a linked list.")
    args = parser.parse_args()

    started = time.time()
    report = {
        "schema": "mlx2.qwen36-gpu-probe.v1",
        "model": str(Path(args.model).resolve()),
        "passed": False,
        "route_selected": False,
        "started_at": started,
        "platform": platform.platform(),
    }
    try:
        import mlx.core as mx
        from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter

        load_started = time.perf_counter()
        adapter = Qwen3635BA3BAdapter(args.model)
        report["load_seconds"] = time.perf_counter() - load_started
        tokens = adapter.tokenizer.encode(args.prompt, add_special_tokens=False)
        if len(tokens) < 4:
            raise ValueError("probe prompt encoded to fewer than four tokens")

        full_cache = adapter.model.make_cache()
        split_cache = adapter.model.make_cache()
        forward_started = time.perf_counter()
        full = adapter.model(mx.array(tokens)[None], cache=full_cache)
        mx.eval(full)
        split_at = len(tokens) // 2
        adapter.model(mx.array(tokens[:split_at])[None], cache=split_cache)
        split = adapter.model(mx.array(tokens[split_at:])[None], cache=split_cache)
        mx.eval(split)
        report["forward_seconds"] = time.perf_counter() - forward_started

        full_last = full[:, -1, :]
        split_last = split[:, -1, :]
        difference = mx.max(mx.abs(full_last.astype(mx.float32) - split_last.astype(mx.float32)))
        finite = mx.all(mx.isfinite(full_last)) & mx.all(mx.isfinite(split_last))
        same_argmax = mx.argmax(full_last, axis=-1) == mx.argmax(split_last, axis=-1)
        mx.eval(difference, finite, same_argmax)
        max_abs = float(difference.item())
        report.update(
            {
                "prompt_tokens": len(tokens),
                "finite": bool(finite.item()),
                "same_argmax": bool(same_argmax.item()),
                "max_abs_difference": max_abs,
                "cache_layers": len(full_cache),
                "linear_gdn_layers": sum(layer.is_linear for layer in adapter.model.layers),
                "full_attention_layers": sum(not layer.is_linear for layer in adapter.model.layers),
                "descriptor": adapter.descriptor.family,
                "profile": adapter.profile_name(False),
                "environment": adapter.environment,
            }
        )
        report["passed"] = bool(
            report["finite"]
            and report["same_argmax"]
            and math.isfinite(max_abs)
            and max_abs == 0.0
        )
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        report["finished_at"] = time.time()
        report["elapsed_seconds"] = report["finished_at"] - started
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
