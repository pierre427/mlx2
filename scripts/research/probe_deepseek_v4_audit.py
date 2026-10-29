#!/usr/bin/env python3
"""Owned synthetic 64K attention and paired-Q8 audit; never model qualification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from probe_deepseek_v4_packed import (
    atomic,
    compare,
    source_receipt,
    synthetic,
    verify_ownership,
)


def run(guard):
    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.gguf.cooperative_metal import (
        cooperative_dense,
        cooperative_gate_up,
    )
    from mlx2.runtime.gguf.packed_metal import PackedMatrix
    from mlx2.runtime.gguf.reader import GGUFTensor
    from mlx2.runtime.models.deepseek_v4 import stable_pool_topk
    from mlx2.runtime.models.deepseek_v4_visible_attention import (
        reference_visible_attention,
        visible_attention,
    )

    rng = np.random.default_rng(261024)
    paired = []
    for k, n, batch in [(256, 33, 1), (4096, 256, 1), (256, 33, 7)]:
        guard()
        weights = []
        for i in range(2):
            raw = synthetic((n, k), 8, 101 + i)
            tensor = GGUFTensor(f"paired-{i}", (k, n), 8, 0, raw.nbytes)
            weights.append(PackedMatrix(mx.array(raw.reshape(-1)), tensor))
        x = mx.array(rng.normal(0, 0.3, (batch, k)).astype(np.float32))
        expected = [cooperative_dense(x, w) for w in weights]
        actual = cooperative_gate_up(x, *weights)
        mx.eval(expected, actual)
        exact = all(
            np.array_equal(np.array(a), np.array(b)) for a, b in zip(actual, expected)
        )
        paired.append({"input": [batch, k], "output": n, "bitwise_equal": exact})
        if not exact:
            raise AssertionError(
                f"paired projection changed ordinary cooperative arithmetic: {paired[-1]}"
            )
    attention = []
    for ratio in (4, 128):
        guard()
        count = 65536 // ratio
        q = rng.normal(0, 0.2, (1, 2, 1, 512)).astype(np.float32)
        local = rng.normal(0, 0.2, (1, 1, 128, 512)).astype(np.float32)
        pool = rng.normal(0, 0.2, (1, count, 512)).astype(np.float32)
        sinks = np.array([0.0, 0.25], np.float32)
        indices = None
        if ratio == 4:
            indices = stable_pool_topk(mx.zeros((1, 1, count)), 512)
            mx.eval(indices)
            np.testing.assert_array_equal(np.array(indices), np.arange(512)[None, None])
        expected = reference_visible_attention(
            q,
            local,
            pool,
            None if indices is None else np.array(indices),
            sinks,
            offset=65535,
            window=128,
            ratio=ratio,
        )
        actual = visible_attention(
            mx.array(q),
            mx.array(local),
            mx.array(pool),
            indices,
            mx.array(sinks),
            offset=65535,
            window=128,
            ratio=ratio,
        )
        mx.eval(actual)
        metrics = compare(np.array(actual), expected, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(np.array(actual), expected, rtol=3e-5, atol=3e-6)
        attention.append(
            {"context": 65536, "ratio": ratio, "pool_rows": count, "numerical": metrics}
        )
        if indices is not None:
            duplicate = mx.zeros_like(indices)
            bad = visible_attention(
                mx.array(q),
                mx.array(local),
                mx.array(pool),
                duplicate,
                mx.array(sinks),
                offset=65535,
                window=128,
                ratio=ratio,
            )
            mx.eval(bad)
            if not np.isnan(np.array(bad)).all():
                raise AssertionError("duplicate pooled indices were accepted")
    return {
        "paired_q8": paired,
        "attention_64k": attention,
        "duplicate_indices_rejected": True,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ownership-window", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        p.error("refusing existing evidence directory")
    owners = [
        Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
        Path("/tmp/gpu.lock/owner.json"),
    ]
    guard = lambda: verify_ownership(args.ownership_window, owners)
    ownership = guard()
    source = source_receipt()
    args.output.mkdir(parents=True)
    receipt = {
        "schema": "mlx2.deepseek-v4-audit-fixtures.v1",
        "status": "running",
        "checkpoint": "synthetic-Q8-and-64K-attention@seed261024",
        "source": source,
        "ownership": ownership,
        "actual_artifact_loaded": False,
        "qualification": False,
        "performance_claim": False,
    }
    try:
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(512 << 20)
        mx.set_cache_limit(32 << 20)
        receipt.update(run(guard), status="passed")
        guard()
    except BaseException as error:
        receipt.update(
            status="failed", error={"type": type(error).__name__, "message": str(error)}
        )
        raise
    finally:
        atomic(args.output / "results.json", receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
