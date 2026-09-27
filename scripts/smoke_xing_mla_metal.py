#!/usr/bin/env python3
"""Leased synthetic numerical gate for the experimental fused MLA kernel."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    owner_path = Path("/Users/Shared/mlxuag/gpu.lock/owner.json")
    if not owner_path.is_file() or os.environ.get("MLX2_GPU_FCNTL_LOCKED") != "1":
        raise SystemExit("CPG GPU lease, host lock and /tmp/gpu.lock wrapper are required")
    owner = json.loads(owner_path.read_text())
    if not owner.get("cpg_generation"):
        raise SystemExit("GPU lock lacks CPG lease generation")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise SystemExit("commit candidate source before the Metal GPU smoke")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()

    import mlx.core as mx

    from mlx2.runtime.models import xing4_0_mla_metal

    mx.set_default_device(mx.gpu)
    mx.random.seed(51017)
    results = []
    for batch, heads, query, context in ((1, 32, 1, 17), (2, 32, 2, 129), (1, 32, 4, 257)):
        q = mx.random.normal((batch, heads, query, 512)).astype(mx.bfloat16)
        qp = mx.random.normal((batch, heads, query, 64)).astype(mx.bfloat16)
        kv = mx.random.normal((batch, 1, context, 512)).astype(mx.bfloat16)
        kp = mx.random.normal((batch, 1, context, 64)).astype(mx.bfloat16)
        mask = None
        if query > 1:
            mask = mx.arange(context)[None, None, None, :] <= (
                mx.arange(query)[None, None, :, None] + context - query
            )
        scale = 1 / (512 + 64) ** 0.5
        expected_scores = (q.astype(mx.float32) @ kv.astype(mx.float32).swapaxes(-1, -2)
                           + qp.astype(mx.float32) @ kp.astype(mx.float32).swapaxes(-1, -2)) * scale
        if mask is not None:
            expected_scores = mx.where(mask, expected_scores, -1e30)
        expected = mx.softmax(expected_scores, axis=-1) @ kv.astype(mx.float32)
        # The ordinary Xing model produces a 2D causal mask on verify steps.
        actual = xing4_0_mla_metal.attend(
            q, qp, kv, kp, mask[0, 0] if mask is not None else None, scale=scale
        )
        mx.eval(expected, actual)
        maximum = float(mx.max(mx.abs(actual.astype(mx.float32) - expected)).item())
        results.append({"batch": batch, "heads": heads, "query": query,
                        "context": context, "max_abs": maximum})
        if maximum > 0.2:
            raise RuntimeError(f"fused MLA synthetic parity failed: {results[-1]}")
    print(json.dumps({"schema": "mlx2.xing-mla-metal-smoke.v1", "source_commit": source_commit,
                      "gpu_owner": owner, "results": results,
                      "stats": xing4_0_mla_metal.snapshot_stats()}, sort_keys=True))


if __name__ == "__main__":
    main()
