"""M3 paired gathered fine ranking; dense anchor/coarse work remains."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--coarse-prefill", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("reference", args.reference)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    report = {
        "completed": False,
        "schema": "mlx2.hysparse2-candidate-fine.v1",
        "serving_route_qualified": False,
        "dense_anchor_avoided": False,
        "one_repetition_no_thermal_control": True,
        "cells": [],
        "reference_sha256": file_hash(args.reference),
    }
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx

        from mlx2.experimental.hysparse2 import attention as current

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(8 << 30)
        mx.set_cache_limit(256 << 20)
        mx.random.seed(42)
        cells = [(4096, 16, True, 0), (4096, 16, True, 1024)] if args.coarse_prefill else [
            (4096, 1, False, 4095), (16384, 1, True, 16383), (4096, 4, True, 4092)]
        for length, queries, tiled, offset in cells:
            q = mx.random.normal((2, 4, queries, 64))
            k = mx.random.normal((2, 1, length, 64))
            v = mx.random.normal(k.shape)
            options = {
                "offset": offset,
                "query_tile": 2,
                "key_tile": 1024,
                "select": (128, 512),
                "block_select": (64, 8),
            }

            def blocks(k, v, tiled=tiled, length=length):
                return (
                    [
                        (
                            k[:, :, start : start + 1024],
                            v[:, :, start : start + 1024],
                            start,
                        )
                        for start in range(0, length, 1024)
                    ]
                    if tiled
                    else [(k, v, 0)]
                )

            values, gradients, timings, coarse_work, fine_work = [], [], [], [], []
            for module in (reference, current):
                groups = module._candidate_groups
                work, fine = [], []
                def counted(*a, **kw):
                    for item in groups(*a, **kw):
                        work.append(item[0].shape[2] * min(queries, options["query_tile"]))
                        yield item
                module._candidate_groups = counted
                rank_tokens = getattr(module, "_rank_candidate_tokens", None)
                if rank_tokens is not None:
                    def counted_fine(query, blocks, qp, ids, block_size, local, *a, **kw):
                        fine.append(query.shape[2] * (ids.shape[-1] * block_size + local))
                        return rank_tokens(query, blocks, qp, ids, block_size, local, *a, **kw)
                    module._rank_candidate_tokens = counted_fine

                def run(q, k, v, module=module, options=options, blocks=blocks):
                    dense, support = module.attention(q, blocks(k, v), **options)
                    sparse = module.sparse_attention(
                        q, support, offset=options["offset"], sinks=mx.zeros((4,))
                    )
                    return dense, support, sparse

                mx.eval(run(q, k, v))
                work.clear()
                fine.clear()
                start = time.perf_counter()
                result = run(q, k, v)
                mx.eval(result)
                timings.append((time.perf_counter() - start) * 1000)
                coarse_work.append(sum(work))
                fine_work.append(sum(fine) if fine else length * queries)
                values.append(result)

                def objective(q, k, v, run=run):
                    dense, _, sparse = run(q, k, v)
                    return mx.mean(dense * dense) + mx.mean(sparse * sparse)

                gradient = mx.grad(objective, argnums=(0, 1, 2))(q, k, v)
                mx.eval(gradient)
                gradients.append(gradient)
                module._candidate_groups = groups
                if rank_tokens is not None:
                    module._rank_candidate_tokens = rank_tokens
            left, right = values
            support_equal = bool(
                mx.all(
                    mx.sort(left[1][2], axis=-1) == mx.sort(right[1][2], axis=-1)
                ).item()
            )
            error = float(mx.max(mx.abs(left[2] - right[2])).item())
            grad_error = [
                float(mx.max(mx.abs(a - b)).item())
                for a, b in zip(*gradients, strict=True)
            ]
            cell = {
                "context": length,
                "query_tokens": queries,
                "query_offset": offset,
                "fragmented": tiled,
                "selected_sets_equal": support_equal,
                "sparse_output_max_error": error,
                "qkv_gradient_max_error": grad_error,
                "reference_ms": timings[0],
                "gathered_ms": timings[1],
                "reference_coarse_score_pairs_per_head": coarse_work[0],
                "current_coarse_score_pairs_per_head": coarse_work[1],
                "reference_fine_score_pairs_per_head": fine_work[0],
                "gathered_fine_score_pairs_per_head": fine_work[1],
            }
            report["cells"].append(cell)
            assert support_equal and error < 1e-4 and max(grad_error) < 1e-4, cell
            if args.coarse_prefill:
                assert coarse_work[1] < coarse_work[0], cell
        report["source_sha256"] = file_hash(Path(current.__file__))
        report["script_sha256"] = file_hash(Path(__file__))
        report["peak_memory_bytes"] = mx.get_peak_memory()
        report["completed"] = True
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
