"""Dry by default; bounded immutable-array packed paged-read GPU gate.

This tests the existing Metal read kernel against a CPU oracle for two ragged
lanes. It deliberately does not consume the opaque native byte arena. The
caller owns GPU queue admission, both locks, and service/waiter checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def dry_run() -> dict:
    return {
        "schema": "mlx2.varlen-packed-read-gate.v1",
        "mode": "dry-run", "gpu_executed": False,
        "planned_cases": ["two ragged lanes, d128 fp16, causal, 63/65 boundary",
                          "two ragged lanes, d128 fp16, sliding-17, 63/65 boundary"],
        "native_arena_consumed": False,
        "binding_blocker": "NativeWriteBackend exposes an opaque byte arena and bounded diagnostic copies; the immutable Metal reader requires full MLX K/V arrays. A native paged-read primitive or safe zero-copy arena input is required.",
        "not_proven": ["native write-to-attention read dependency", "native arena lifetime",
                       "model serving", "qualification", "performance"],
    }


def immutable_gpu_gate() -> dict:
    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.paged_attention_metal import (
        paged_attention_cpu_reference, paged_attention_metal_candidate,
    )
    from mlx2.runtime.paged_attention_plan import PagedAttentionPlan, SequenceSpan
    from mlx2.runtime.paged_kv_native import PagedKVCompletionLedger
    from mlx2.runtime.paged_kv_pool import PagedKVPool

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("explicit immutable GPU gate requires an MLX GPU")
    rng = np.random.default_rng(7125)
    pool = PagedKVPool(4)
    handles = pool.reserve(4)
    table = (handles[2], handles[0], handles[3], handles[1])
    ledger = PagedKVCompletionLedger(pool)
    cases = []
    for sliding in (False, True):
        spans = (
            SequenceSpan(0, 3, 62, 65, 0, 0, 0, 2, 1,
                         "sliding" if sliding else "causal", 17 if sliding else None),
            SequenceSpan(3, 1, 62, 63, 0, 0, 2, 1, 1,
                         "sliding" if sliding else "causal", 17 if sliding else None),
        )
        # Lane two needs only one page for 63 tokens. Keep the fourth physical
        # slot as an unrelated resident page to prove table IDs are opaque.
        case_table = table[:3]
        plan = PagedAttentionPlan(
            spans=spans, page_table=case_table, total_rows=4,
            query_heads=4, kv_heads=2, head_dim=128, dtype="float16",
            pool_capacity=4, live_generations=pool.live_generations(),
        )
        q = rng.normal(0, 0.25, (4, 4, 128)).astype(np.float16)
        k = rng.normal(0, 0.25, (4, 2, 64, 128)).astype(np.float16)
        v = rng.normal(0, 0.25, k.shape).astype(np.float16)
        expected = paged_attention_cpu_reference(plan, q, k, v)
        lease = ledger.prepare(tuple(dict.fromkeys(case_table)))
        ledger.submit(lease)
        try:
            result = paged_attention_metal_candidate(
                plan, mx.array(q), mx.array(k), mx.array(v),
                live_generations=pool.live_generations(),
                owner_pins_through_completion=True, permit_candidate=True,
            )
            mx.eval(result)
            mx.synchronize(mx.default_stream(mx.gpu))
            actual = np.array(result)
        finally:
            # This exact synchronous gate waits for the one reader graph. No
            # asynchronous serving owner should use this shortcut.
            mx.synchronize(mx.default_stream(mx.gpu))
            ledger.complete(lease)
        error = actual.astype(np.float64) - expected.astype(np.float64)
        normalized_rms = float(np.sqrt(np.mean(error * error)) /
                               max(np.sqrt(np.mean(expected.astype(np.float64) ** 2)), 1e-12))
        cases.append({"mask": "sliding-17" if sliding else "causal",
                      "normalized_rms": normalized_rms,
                      "max_abs": float(np.max(np.abs(error))),
                      "passed_screen": normalized_rms <= 2e-3})
    pool.release(handles, after_epoch=ledger.completed_epoch)
    pool.retire(ledger.completed_epoch)
    return {"schema": "mlx2.varlen-packed-read-gate.v1",
            "mode": "immutable-gpu", "gpu_executed": True,
            "native_arena_consumed": False, "cases": cases,
            "all_passed": all(case["passed_screen"] for case in cases),
            "binding_blocker": dry_run()["binding_blocker"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-immutable-gpu", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = immutable_gpu_gate() if args.execute_immutable_gpu else dry_run()
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result.get("all_passed") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
