# SPDX-License-Identifier: MIT
"""GLM-5.3 gathered prompt attention kernel bench: fp32 fused_v2 vs the fp16-operand kernel.

Synthetic operands (the checkpoint is not loaded): a latent plane of
``--total`` fp32 rows, ``--rows`` prompt query rows per block with
decode-shaped selections (W = 2051), 11 distinct layers chained by data
dependency as a prompt chunk runs them. Arms:

``gathered_v1``  the loaded prompt law's ops path (gather + fp32 SDPA);
``fused_v2``     the fp32 MMA kernel (gathered_v2_from_<n>);
``fused_f16``    the fp16-operand kernel (gathered_f16_from_<n>), the per-call
                 half cast of the source included, as sparse_prefill does it.

Reports ms per 11-layer block per arm (median of --reps, cold caches
cleared between reps), the implied per-row cost and the fp16 kernel's error
against float64 on the half-rounded operands for the first layer. A/B
signal, not publishable statistics; the GPU must be free.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def selection(rng, total, rows, width):
    import numpy as np

    ids = np.full((rows, width), -1, np.int32)
    for r in range(rows):
        pos = total - rows + r
        pools = np.sort(rng.choice(max(1, pos // 4), size=(width - 3) // 4, replace=False))
        row = (pools[:, None] * 4 + np.arange(4)).reshape(-1)
        row = np.minimum(row, max(0, pos - 3))
        ids[r, : len(row)] = row
        ids[r, len(row):] = np.arange(max(0, pos - 2), pos + 1)[-3:]
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-revision", default="unrecorded")
    parser.add_argument("--total", type=int, default=16384)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--layers", type=int, default=11)
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=5305)
    parser.add_argument("--arms", default="gathered_v1,fused_v2,fused_f16")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    import numpy as np

    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.runtime.models import glm53_sparse_attention as sa
    from mlx2.runtime.models import glm53_sparse_attention_v2 as v2

    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(args.seed)
    width = sa.WIDTH
    latents = [mx.array(rng.normal(size=(1, 1, args.total, 512)).astype(np.float32)) for _ in range(args.layers)]
    qs = [mx.array(rng.normal(size=(1, 64, args.rows, 512)).astype(np.float32)) for _ in range(args.layers)]
    ids = mx.array(selection(rng, args.total, args.rows, width)[None])
    mx.eval(latents, qs, ids)

    def block(policy):
        # Chain the layers: layer l's q is perturbed by layer l-1's output so
        # the GPU cannot overlap them (a prompt chunk runs the layers in order).
        carry = None
        for layer in range(args.layers):
            q = qs[layer] if carry is None else qs[layer] + carry[:, :, :1, :1] * 0.0
            if policy == "fused_v2":
                carry = sa.sparse_prefill(q, latents[layer], ids, block=args.rows, policy="fused_v2")
            elif policy == "fused_f16":
                carry = sa.sparse_prefill(q, latents[layer], ids, block=args.rows, policy="fused_f16")
            elif policy.startswith("flash_"):
                # Direct launches of the flash kernel: flash_f16_outer4, flash_f32_outer8, ...
                kind, outer = policy.rsplit("_outer", 1)
                base = latents[layer][0, 0]
                tails = mx.zeros((1, 512), mx.float32)
                meta = mx.array([[args.total, 0, 0, args.total - args.rows + r] for r in range(args.rows)],
                                dtype=mx.int32)
                out = v2.launch(q[0].transpose(1, 0, 2), base, tails, ids[0], meta, kind=kind, outer=int(outer))
                carry = out.transpose(1, 0, 2)[None]
            elif policy == "gathered_ops_f16":
                # Probe: the gathered_v1 ops law on half operands (half plane
                # copy per call, half gather, MLX SDPA on half, fp32 out).
                latent16 = latents[layer][0, 0].astype(mx.float16)
                rows = mx.take(latent16, mx.clip(ids[0], 0, args.total - 1), axis=0)
                valid = (ids[0] >= 0)[:, None, None, :]
                mask = mx.where(valid, mx.array(0.0, mx.float16), mx.array(-mx.inf, mx.float16))
                qb = q.astype(mx.float16).transpose(2, 1, 0, 3)
                keys = rows[:, None]
                out = mx.fast.scaled_dot_product_attention(qb, keys, keys, scale=sa.SCALE, mask=mask)
                carry = out.transpose(2, 1, 0, 3).astype(mx.float32)
            else:
                carry = sa.sparse_prefill(q, latents[layer], ids, block=args.rows, policy="gathered_v1")
        mx.eval(carry)
        return carry

    arms = args.arms.split(",")
    record = {"schema": "mlx2.glm53-gathered-f16-bench.v1", "source_revision": args.source_revision,
              "host": {"machine": platform.machine(), "mlx": mx.__version__,
                       "device": mx.device_info().get("device_name"),
                       "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                              text=True).stdout.strip()},
              "total": args.total, "rows": args.rows, "layers": args.layers, "width": width, "reps": args.reps,
              "arms": {}, "thermal": "not gated: kernel A/B signal"}
    # Warm every arm (compiles, first-use gates).
    for arm in arms:
        block(arm)
    for rep in range(args.reps):
        for arm in (arms if rep % 2 == 0 else arms[::-1]):
            mx.clear_cache()
            t0 = time.perf_counter()
            block(arm)
            dt = time.perf_counter() - t0
            record["arms"].setdefault(arm, {"seconds": []})["seconds"].append(dt)
    for arm, data in record["arms"].items():
        med = statistics.median(data["seconds"])
        data["median_ms_per_block"] = med * 1e3
        data["us_per_row_layer"] = med / (args.rows * args.layers) * 1e6
    if "fused_v2" in record["arms"] and "fused_f16" in record["arms"]:
        record["speedup_f16_over_v2"] = (record["arms"]["fused_v2"]["median_ms_per_block"]
                                         / record["arms"]["fused_f16"]["median_ms_per_block"])
    if "gathered_v1" in record["arms"] and "fused_f16" in record["arms"]:
        record["speedup_f16_over_gathered_v1"] = (record["arms"]["gathered_v1"]["median_ms_per_block"]
                                                  / record["arms"]["fused_f16"]["median_ms_per_block"])
    # Error of the fp16 kernel on the first layer against float64 on the half-rounded operands.
    q0 = qs[0][0].transpose(1, 0, 2)
    base = latents[0][0, 0]
    tails = mx.zeros((1, 512), mx.float32)
    meta = mx.array([[args.total, 0, 0, args.total - args.rows + r] for r in range(args.rows)], dtype=mx.int32)
    got = v2.launch(q0, base, tails, ids[0], meta, half=True)
    full = v2.launch(q0, base, tails, ids[0], meta)
    mx.eval(got, full)
    want = v2.oracle(*v2.half_rounded(q0, base, tails), ids[0], meta)
    record["f16_vs_float64_half_rounded"] = v2.oracle_errors(got, want)
    record["fp32_kernel_vs_float64"] = v2.oracle_errors(full, v2.oracle(q0, base, tails, ids[0], meta))
    record["gate_report"] = v2.GATE_REPORT
    (args.out / "bench.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("gate_report",)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
