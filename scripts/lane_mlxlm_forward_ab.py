#!/usr/bin/env python3
"""Whole-model lane A/B on a plain mlx-lm model (no mlx2 adapter needed).

Loads an MLX checkpoint with ``mlx_lm.load``, installs the lane matmul in
exact mode (every covered projection uses the lane arithmetic at every row
count), prefills real text to each context length and then:

* cost: forwards of 1/4/8/16 rows, stock versus lane, alternating arms
  (first round discarded), median ms;
* quality: from one snapshot, 16 one-token stock steps (what decode
  produces) against a 16-row stock forward, a 16-row lane forward and 16
  one-token lane steps: top-1 agreement and max |log-prob difference|.
  ``lane_rows_vs_lane_serial`` is the verify-equals-decode check under the
  lane law (projections only: mlx-lm's attention and GDN kernels are not
  row-invariant, so logits need not be bitwise equal).

usage: lane_mlxlm_forward_ab.py <model> <out.json> [ctx,ctx] [--backend auto|mpp|simd]
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def swapouts() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False).stdout
    for line in out.splitlines():
        if line.startswith("Swapouts"):
            return int(line.split()[-1].rstrip("."))
    return -1


def text_tokens(tokenizer, need: int) -> list[int]:
    files = sorted((ROOT / "src").rglob("*.py"))
    text = "\n\n".join(f.read_text(errors="ignore") for f in files)
    ids = list(tokenizer.encode(text))
    while len(ids) < need:
        ids = ids + ids
    return ids


def run(call, cache, tokens):
    y = call(mx.array([tokens], dtype=mx.uint32), cache)
    mx.eval(y, [c.state for c in cache])
    return y


def timed(call, cache, tokens):
    t0 = time.perf_counter()
    y = run(call, cache, tokens)
    return (time.perf_counter() - t0) * 1000, y


def logprobs(y):
    y = y.astype(mx.float32)
    return y - mx.logsumexp(y, axis=-1, keepdims=True)


def compare(lp, ref):
    return {"top1_agree_of_16": int(mx.sum(mx.argmax(lp, -1) == mx.argmax(ref, -1)).item()),
            "max_abs_logprob_diff": round(float(mx.max(mx.abs(lp - ref)).item()), 4),
            "bitwise_equal": bool(mx.array_equal(lp, ref).item())}


def serial(call, base, verify, on):
    """One-token steps from a copy of ``base``: what decode produces."""
    from mlx2.runtime import lane

    lane.set_enabled(on)
    c = copy.deepcopy(base)
    out = mx.stack([logprobs(run(call, c, [t])[0, -1]) for t in verify])
    lane.set_enabled(False)
    return out


def rows16(call, base, verify, on):
    """One multi-row forward from a copy of ``base``: what verify produces."""
    from mlx2.runtime import lane

    lane.set_enabled(on)
    out = logprobs(run(call, copy.deepcopy(base), verify)[0])
    lane.set_enabled(False)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("output", type=Path)
    p.add_argument("contexts", nargs="?", default="2048,16384")
    p.add_argument("--backend", choices=("auto", "mpp", "simd"), default="auto")
    p.add_argument("--rounds", type=int, default=6)
    args = p.parse_args()
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    from mlx2.runtime import lane

    if args.backend != "auto":
        lane.force_backend(args.backend)
    if lane.backend() is None:
        raise SystemExit(f"no lane backend for --backend {args.backend} on this device")
    mx.set_cache_limit(4 << 30)
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    rec = {"schema": "mlx2.lane-mlxlm-forward-ab.v1", "model": args.model, "mlx": mx.__version__,
           "device": {k: str(info.get(k)) for k in ("device_name", "architecture")},
           "swapouts_start": swapouts(), "contexts": {}}
    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    rec["load_seconds"] = round(time.perf_counter() - t0, 1)
    rec["install"] = lane.install(model, min_rows=1)        # exact mode
    if not rec["install"]["covered"]:
        raise SystemExit(f"lane covered nothing: {rec['install']['refused']}")
    lane.set_enabled(False)

    def call(x, cache):
        return model(x, cache=cache)

    ctxs = [int(c) for c in args.contexts.split(",")]
    ids = text_tokens(tokenizer, max(ctxs) + 64)
    for ctx in ctxs:
        row = {"tokens": ctx}
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        for s in range(0, ctx, 2048):
            run(call, cache, ids[s:min(s + 2048, ctx)])
        row["prefill_seconds"] = round(time.perf_counter() - t0, 2)
        verify = ids[ctx:ctx + 16]
        base = copy.deepcopy(cache)

        stock_serial = serial(call, base, verify, False)
        lane_serial = serial(call, base, verify, True)
        row["quality"] = {
            "stock_rows_vs_stock_serial": compare(rows16(call, base, verify, False), stock_serial),
            "lane_rows_vs_stock_serial": compare(rows16(call, base, verify, True), stock_serial),
            "lane_serial_vs_stock_serial": compare(lane_serial, stock_serial),
            "lane_rows_vs_lane_serial": compare(rows16(call, base, verify, True), lane_serial),
        }
        del base
        cost = {}
        for rows in (1, 4, 8, 16):
            samples = {"stock": [], "lane": []}
            for r in range(args.rounds):
                for arm in (("stock", "lane") if r % 2 == 0 else ("lane", "stock")):
                    lane.set_enabled(arm == "lane")
                    c = copy.deepcopy(cache)
                    mx.eval([x.state for x in c])
                    ms, _ = timed(call, c, verify[:rows])
                    if r:
                        samples[arm].append(ms)
                    del c
            lane.set_enabled(False)
            cost[rows] = {k: round(statistics.median(v), 2) for k, v in samples.items()}
            cost[rows]["lane_vs_stock"] = round(cost[rows]["lane"] / cost[rows]["stock"] - 1, 3)
        row["forward_ms"] = cost
        del cache
        mx.clear_cache()
        rec["contexts"][ctx] = row
        print(json.dumps({"ctx": ctx, "quality": row["quality"], "forward_ms": cost}), flush=True)
    rec["lane_stats"] = lane.stats()
    rec["swapouts_end"] = swapouts()
    rec["peak_memory_gb"] = round(mx.get_peak_memory() / 2**30, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    main()
