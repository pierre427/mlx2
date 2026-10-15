#!/usr/bin/env python3
"""Owned, bounded component diagnostics; no model or performance qualification.

Exercise the installed lane dispatch across quantization geometries and measure
live allocation around real KV merge/first-append operations. Allocation deltas
are not memory bandwidth or bytes copied by the GPU.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path


def cache_geometry(mx):
    import numpy as np

    from mlx2.runtime.models.cache import BatchKVCache, KVCache

    records = []
    for length in (256, 2048):
        for lengths in ((length,), (length, length), (length, length // 2)):
            sources = []
            for size in lengths:
                cache = KVCache()
                value = mx.arange(2 * size * 64).reshape(1, 2, size, 64).astype(mx.bfloat16)
                cache.update_and_fetch(value, value * 0.5)
                sources.append(cache)
            mx.eval([c.state for c in sources])
            before = [(np.asarray(c.keys.astype(mx.float32)).copy(),
                       np.asarray(c.values.astype(mx.float32)).copy()) for c in sources]
            gc.collect()
            mx.clear_cache()
            baseline = mx.get_active_memory()
            merged = BatchKVCache.merge(sources)
            mx.eval(merged.state)
            after_merge = mx.get_active_memory()
            step = mx.ones((len(sources), 2, 1, 64), dtype=mx.bfloat16)
            mx.eval(step)
            append_baseline = mx.get_active_memory()
            mx.reset_peak_memory()
            merged.update_and_fetch(step, step)
            mx.eval(merged.state)
            after_append = mx.get_active_memory()
            unchanged = all(np.array_equal(np.asarray(c.keys.astype(mx.float32)), old[0])
                            and np.array_equal(np.asarray(c.values.astype(mx.float32)), old[1])
                            for c, old in zip(sources, before))
            records.append({
                "lengths": lengths, "width": len(sources),
                "merge_active_delta_bytes": after_merge - baseline,
                "append_active_delta_bytes": after_append - append_baseline,
                "append_peak_above_baseline_bytes": mx.get_peak_memory() - append_baseline,
                "merged_backing_logical_bytes": merged.nbytes,
                "source_unchanged": unchanged,
                "interpretation": "allocator observations, not a physical copy traffic counter",
            })
            del sources, cache, value, before, merged, step
    return records


def run(ownership):
    import lane_matmul_gate as gate
    import mlx.core as mx

    from mlx2.runtime.lane import matmul, simd

    mx.set_default_device(mx.gpu)
    matmul.force_backend("simd")
    if not matmul.available():
        raise RuntimeError("SIMD lane backend unavailable")
    # Reuse actual existing correctness gates, with all timing disabled.
    gate.timed = lambda *_args, **_kwargs: None
    records = []
    key = mx.random.key(73)
    for bits in (2, 3, 4, 5, 6, 8):
        for group in (32, 64, 128):
            for k, n in gate.SHAPES["small"]:
                key, sub = mx.random.split(key)
                try:
                    row = gate.case(bits, group, k, n, mx.bfloat16, sub)
                    row.pop("ms", None)
                except Exception as error:  # noqa: BLE001 - retain each diagnostic failure
                    row = {"bits": bits, "group_size": group, "k": k, "n": n,
                           "passed": False, "error": f"{type(error).__name__}: {error}"}
                records.append(row)
                print(json.dumps(row), flush=True)
    cache = cache_geometry(mx)
    return {
        "schema": "mlx2.peer-geometry-diagnostic.v1",
        "qualified": False, "performance_qualified": False,
        "kind": "component-diagnostic", "ownership": ownership,
        "mlx_version": mx.__version__, "device": mx.device_info()["device_name"],
        "simd_source_sha256": hashlib.sha256(Path(simd.__file__).read_bytes()).hexdigest(),
        "matrix": records, "cache": cache,
        "passed": all(r["passed"] for r in records) and all(r["source_unchanged"] for r in cache),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpg-task", required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--cpg-generation", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("evidence already exists")
    from qualification_gpu_ownership import require_qualification_lease
    ownership = require_qualification_lease(
        task_id=args.cpg_task, cpg_owner_lock=args.cpg_owner_lock,
        generation=args.cpg_generation,
    )
    result = run(ownership)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"passed": result["passed"], "matrix_cases": len(result["matrix"])}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
