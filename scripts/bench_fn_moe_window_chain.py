"""Launch-chain microbench for the MoE row window and top-k modes (synthetic
evidence only; the full-model A/B decides).

Builds ``--layers`` real Flash-Next MoE blocks from the artifact, then times a
chain of ``--chain`` block calls (one eval per chain, the decode dependency
shape) for each arm, arms rotated per rep (odd reps reversed) after a
discarded warm-up. Arms: ``stock`` (window off, top-k off: the multi-row
gather for R > 1), ``window`` / ``window:launch`` / ``window:fold`` (R > 1),
and ``off`` / ``launch`` / ``fold`` (R = 1). Inputs are [R, 1, H] (batched
decode consumer).

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python scripts/bench_fn_moe_window_chain.py \
      --i-own-the-gpu --rows 1 2 4 8 16 --out chain.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_fn_split_routed_decode as base  # noqa: E402

import mlx.core as mx  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--chain", type=int, default=48)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--routed", default="gate_up_down_shared")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    mx.set_cache_limit(4 << 30)
    from mlx2.runtime.models import qwen4_moe_window as W

    index = json.load(open(base.MODEL / "model.safetensors.index.json"))["weight_map"]
    cache = {}
    blocks = []
    for i in range(a.layers):
        layer = (47 * i) // max(1, a.layers - 1)
        blocks.append(base.build_block(base.load_layer(f"language_model.model.layers.{layer}", index, cache)))
    for b in blocks:
        b.set_moe_routed_decode_mode(a.routed)
    results = {}
    for R in a.rows:
        arms = ["off", "launch", "fold"] if R == 1 else ["stock", "window", "window:launch", "window:fold"]
        xs = [(mx.random.normal((R, 1, base.H), key=mx.random.key(100 * R + i))).astype(mx.bfloat16)
              for i in range(a.chain)]
        mx.eval(xs)

        def chain(arm):
            name, *opts = arm.split(":")
            topk = opts[0] if opts else ("off" if name in ("stock", "window") else name)
            for b in blocks:
                b.set_moe_window_consumers(() if name == "stock" or R == 1 else {"batch_decode"})
                b.set_moe_topk_mode(topk)
            y = xs[0]
            for i in range(a.chain):
                y = blocks[i % len(blocks)]((y + xs[i]).astype(mx.bfloat16))
            t0 = time.perf_counter()
            mx.eval(y)
            return 1e3 * (time.perf_counter() - t0)

        for arm in arms:
            chain(arm)
        times = {arm: [] for arm in arms}
        for rep in range(a.reps):
            order = arms[rep % len(arms):] + arms[: rep % len(arms)]
            if rep % 2:
                order = order[::-1]
            for arm in order:
                times[arm].append(chain(arm))
        base_arm = arms[0]
        med = {arm: statistics.median(v) for arm, v in times.items()}
        results[R] = {arm: {"median_ms": med[arm], "min_ms": min(v), "max_ms": max(v),
                            "us_per_block": 1e3 * med[arm] / a.chain,
                            "delta_vs_base_pct": 100 * (med[arm] / med[base_arm] - 1)}
                      for arm, v in times.items()}
        print(R, json.dumps({k: (round(v["us_per_block"], 1), round(v["delta_vs_base_pct"], 1))
                             for k, v in results[R].items()}), flush=True)
    for b in blocks:
        b.set_moe_window_consumers(())
        b.set_moe_topk_mode("off")
    json.dump({"rows": a.rows, "chain": a.chain, "reps": a.reps, "layers": a.layers, "routed": a.routed,
               "fold_max_rows": W.topk_fold_max_rows(), "results": results, "mlx": mx.__version__},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
