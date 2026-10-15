#!/usr/bin/env python3
"""Owned component diagnostic for bounded selected-KV gather; no qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def fixture(batch=2, length=5, dim=64):
    rng = np.random.default_rng(20261015)
    total, block = 32, 4
    q = rng.normal(size=(batch, 4, length, dim)).astype(np.float32)
    k = rng.normal(size=(batch, 2, total, dim)).astype(np.float32)
    v = rng.normal(size=(batch, 2, total, dim)).astype(np.float32)
    ids = np.broadcast_to(np.array([0, 1, 2], np.uint32), (batch, length, 3)).copy()
    counts = (np.arange(batch)[:, None] + np.arange(length)[None]) % 4
    stop = np.broadcast_to(16 + np.arange(length) % 5, (batch, length)).copy()
    start = stop // block * block
    left = np.arange(batch, dtype=np.int32) % 4
    logical = np.arange(total)[None, None, :] - left[:, None, None]
    mask = (logical >= 0) & (logical < stop[:, :, None])
    mask[..., 1] = False  # an explicit mask hole must survive compaction
    return {"q": q, "k": k, "v": v, "ids": ids, "counts": counts.astype(np.int32),
            "start": start.astype(np.int32), "stop": stop.astype(np.int32),
            "left": left, "mask": mask[:, None], "block": block, "total": total}


def reference(data):
    """Independent float64 dense selected-mask oracle, no runtime helper calls."""
    q, k, v = (data[name].astype(np.float64) for name in ("q", "k", "v"))
    result = np.zeros_like(q)
    for b in range(q.shape[0]):
        for row in range(q.shape[2]):
            logical = []
            for block_id in data["ids"][b, row, :data["counts"][b, row]]:
                first = int(block_id) * data["block"]
                logical.extend(range(first, first + data["block"]))
            logical.extend(range(int(data["start"][b, row]), int(data["stop"][b, row])))
            positions = [position + int(data["left"][b]) for position in logical]
            positions = [position for position in positions
                         if 0 <= position < data["total"] and data["mask"][b, 0, row, position]]
            if not positions:
                continue
            for head in range(q.shape[1]):
                kv_head = head // (q.shape[1] // k.shape[1])
                scores = k[b, kv_head, positions] @ q[b, head, row] / np.sqrt(q.shape[-1])
                weights = np.exp(scores - scores.max())
                weights /= weights.sum()
                result[b, head, row] = weights @ v[b, kv_head, positions]
    return result


def mlx_inputs(data, dtype):
    import mlx.core as mx

    from mlx2.runtime.models.qwen4_exp import QSACompactBlocks

    arrays = [mx.array(data[name], dtype=dtype) for name in ("q", "k", "v")]
    compact = QSACompactBlocks(
        block_ids=mx.array(data["ids"]), block_counts=mx.array(data["counts"]),
        tail_start=mx.array(data["start"]), tail_stop=mx.array(data["stop"]),
        left_padding=mx.array(data["left"]), block_size=data["block"],
        physical_width=data["total"], causal_mask=mx.array(data["mask"]),
    )
    return (*arrays, compact)


def run_metal():
    import mlx.core as mx

    from mlx2.runtime.models import qwen4_exp as model

    mx.set_default_device(mx.gpu)
    cases = []
    for dtype in (mx.float32, mx.float16, mx.bfloat16):
        for dim in (64, 256):
            data = fixture(dim=dim)
            q, k, v, compact = mlx_inputs(data, dtype)
            rounded = dict(data, **{name: np.asarray(array.astype(mx.float32))
                                   for name, array in zip(("q", "k", "v"), (q, k, v), strict=True)})
            oracle = reference(rounded)
            for tile in (1, 2, 3, 8):
                actual = model._gather_qsa_attention(q, k, v, compact, scale=dim**-0.5, tile_rows=tile)
                mx.eval(actual)
                error = float(np.max(np.abs(np.asarray(actual.astype(mx.float32)) - oracle)))
                tolerance = {mx.float32: 2e-5, mx.float16: 0.005, mx.bfloat16: 0.04}[dtype]
                cases.append({"dtype": str(dtype), "dim": dim, "tile_rows": tile,
                              "max_abs_error": error, "tolerance": tolerance,
                              "passed": bool(np.isfinite(error) and error <= tolerance)})
    info = mx.device_info()
    return {"schema": "mlx2.qsa-gather-tiles.v1", "kind": "component-diagnostic",
            "qualified": False, "source_sha256": hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest(),
            "mlx_version": mx.__version__,
            "device": {key: str(info.get(key)) for key in ("device_name", "architecture")},
            "cases": cases, "passed": all(case["passed"] for case in cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-metal", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpg-task", required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--cpg-generation", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to replace evidence: {args.output}")
    from qualification_gpu_ownership import require_qualification_lease

    ownership = require_qualification_lease(task_id=args.cpg_task,
        cpg_owner_lock=args.cpg_owner_lock, generation=args.cpg_generation)
    receipt = run_metal()
    receipt["ownership"] = ownership
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"passed": receipt["passed"], "cases": len(receipt["cases"])}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
