"""Paired M3 attention bounds parity and work probe against a saved reference."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("attention_reference", args.reference)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    report = {
        "schema": "mlx2.hysparse2-attention-bounds.v1",
        "completed": False,
        "serving_route_qualified": False,
        "reference_sha256": file_hash(args.reference),
        "one_repetition_no_thermal_control": True,
        "cells": [],
    }
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx

        from mlx2.experimental.hysparse2 import attention as current

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(8 << 30)
        mx.set_cache_limit(256 << 20)
        mx.random.seed(42)
        q = mx.random.normal((2, 4, 512, 64))
        k = mx.random.normal((2, 1, 512, 64))
        v = mx.random.normal((2, 1, 512, 64))
        mx.eval(q, k, v)
        report["source_sha256"] = file_hash(Path(current.__file__))
        for window in (None, 128):
            options = {"offset": 0, "query_tile": 32, "key_tile": 1024, "window": window}

            def run(module, options=options):
                return module.attention(q, [(k, v, 0)], **options)[0]

            outputs, timings, gradients = [], [], []
            for module in (reference, current):
                mx.eval(run(module))
                start = time.perf_counter()
                out = run(module)
                mx.eval(out)
                timings.append((time.perf_counter() - start) * 1000)
                outputs.append(out)

                def loss(query, keys, values, module=module, options=options):
                    out = module.attention(query, [(keys, values, 0)], **options)[0]
                    return mx.mean(out * out)

                grad = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)
                mx.eval(grad)
                gradients.append(grad)
            error = float(mx.max(mx.abs(outputs[0] - outputs[1])).item())
            grad_error = [
                float(mx.max(mx.abs(a - b)).item())
                for a, b in zip(*gradients, strict=True)
            ]
            bounded_pairs = 0
            for begin in range(0, 512, 32):
                minimum = None if window is None else begin - window + 1
                for keys, _, _ in current._tiles(
                    [(k, v, 0)], 1024, minimum=minimum, maximum=begin + 31
                ):
                    bounded_pairs += 32 * keys.shape[2]
            cell = {
                "window": window,
                "shape": list(q.shape),
                "output_max_abs_error": error,
                "qkv_gradient_max_abs_error": grad_error,
                "reference_ms": timings[0],
                "bounded_ms": timings[1],
                "reference_score_pairs_per_head": 512 * 512,
                "bounded_score_pairs_per_head": bounded_pairs,
                "score_pair_reduction_fraction": 1 - bounded_pairs / (512 * 512),
            }
            report["cells"].append(cell)
            assert error < 1e-4 and max(grad_error) < 1e-4, cell
        report["peak_memory_bytes"] = mx.get_peak_memory()
        report["completed"] = True
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
