#!/usr/bin/env python3
"""NAX sorted gather_qmm 6-bit vs stock: per-call times on real 35B shapes.

Qwen3.6-35B-A3B expert shapes (hidden 2048, MoE intermediate 512, 256
experts top-8, affine gs64) with captured routing (b1 ``routing.npz``):
T = 2048 tokens (one served chunk, 16384 sorted rows) and T = 8192 (four
chunks of one layer concatenated, 65536 rows).  Per projection:

* gate+up: stock = ``x_tok[row_map]`` + two sorted ``mx.gather_qmm`` +
  SwiGLU (the served fallback); NAX = ``sorted_gather_qmm_swiglu_split``
  with the row map (the served fused route) under the planner's plan and
  the other planner plans;
* down: stock sorted ``mx.gather_qmm`` vs ``sorted_gather_qmm``.

Same calls on 4-bit tables for reference.  One ``mx.eval`` per call, reps
interleaved in shuffled order, median; the empty-eval sync overhead is
reported (not subtracted).  Run through run_with_gpu_locks.py.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

E, TOPK, HID, INTER, GS = 256, 8, 2048, 512, 64


def swapouts() -> int:
    for line in subprocess.check_output(["vm_stat"], text=True).splitlines():
        if line.startswith("Swapouts"):
            return int(line.split(":")[1].strip().rstrip("."))
    return -1


def therm() -> str:
    return subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing", default=str(
        ROOT / "qualification/runs/research-20261006/b1-moe-tile-headroom/capture"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--keys-2k", type=int, default=8)
    ap.add_argument("--layers-8k", type=int, default=3)
    ap.add_argument("--bits", default="6,4")
    a = ap.parse_args()

    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.models import moe_nax_gather as nax
    from mlx2.runtime.models.activations import swiglu

    mx.set_cache_limit(4 << 30)
    if not nax.enabled():
        raise SystemExit("NAX host gate is closed on this device")
    swap0, therm0 = swapouts(), therm()
    rng = random.Random(61006)

    routing = np.load(Path(a.routing) / "routing.npz")
    cap = json.loads((Path(a.routing) / "capture.json").read_text())
    bits_of = {int(k): v for k, v in cap["expert_bits_per_layer"].items()}
    six = [k for k in sorted(routing.files) if bits_of[int(k[1:3])]["gate_proj"] == 6]
    step = max(1, len(six) // a.keys_2k)
    keys_2k = six[::step][: a.keys_2k]
    layers6 = sorted({k[:3] for k in six})
    lstep = max(1, len(layers6) // a.layers_8k)
    layers_8k = layers6[::lstep][: a.layers_8k]

    samples = []
    for k in keys_2k:
        samples.append((k, 2048, routing[k].reshape(-1, TOPK)))
    for layer in layers_8k:
        parts = [routing[k] for k in sorted(routing.files) if k.startswith(layer)]
        samples.append((layer + "_C0-3", 8192, np.concatenate([p.reshape(-1, TOPK) for p in parts])[:8192]))

    def table(o, i, bits, seed):
        w = (mx.random.normal((E, o, i), key=mx.random.key(seed)) * 0.02).astype(mx.bfloat16)
        q = mx.quantize(w, group_size=GS, bits=bits)
        mx.eval(*q)
        del w
        return q

    widths = [int(b) for b in a.bits.split(",")]
    tabs = {b: (table(INTER, HID, b, 10 * b + 1), table(INTER, HID, b, 10 * b + 2),
                table(HID, INTER, b, 10 * b + 3)) for b in widths}
    mx.clear_cache()

    P, SEG, DB, GX = nax.Plan, nax._SCHED_SEG, nax._SCHED_DB, nax._GX
    alt = [P(DB, 64, 64, 0, 0), P(DB, 64, 64, GX, 0), P(DB, 96, 64, GX, 0),
           P(SEG, 96, 128, GX, 0), P(SEG, 128, 128, GX, 8192), P(SEG, 96, 128, GX, 0),
           P(SEG, 128, 128, GX, 0)]
    alt = list(dict.fromkeys(alt))

    def pname(p):
        return f"{nax._SCHED_NAMES[p.sched]},{p.bm},{p.bk},{p.gx},{p.pad}"

    def time_one(fn):
        t = time.perf_counter()
        r = fn()
        if r is None:
            return None
        mx.eval(r)
        return (time.perf_counter() - t) * 1e3

    tiny = mx.zeros((8,))
    mx.eval(tiny)
    base = [time_one(lambda: tiny + 1) for _ in range(60)]
    baseline_ms = statistics.median(base[10:])

    results = []
    t0 = time.time()
    for n, (key, T, inds_np) in enumerate(samples):
        inds = mx.array(inds_np.astype(np.uint32))
        x_tok = (mx.random.normal((T, 1, HID), key=mx.random.key(500 + n)) * 0.5).astype(mx.bfloat16)
        flat = inds.flatten()
        order = mx.argsort(flat)
        row_map = (order // TOPK).astype(mx.uint32)
        idx = flat[order].astype(mx.uint32)
        hidden = (mx.random.normal((T * TOPK, 1, INTER), key=mx.random.key(600 + n)) * 0.5).astype(mx.bfloat16)
        mx.eval(x_tok, row_map, idx, hidden)
        M = int(idx.size)
        arms = {}
        planned = {}
        for bits in widths:
            g, u, d = tabs[bits]
            kw = dict(group_size=GS, bits=bits)

            def stock_gate_up(g=g, u=u, kw=kw):
                xs = x_tok[row_map]
                up = mx.gather_qmm(xs, *u, rhs_indices=idx, transpose=True, sorted_indices=True, **kw)
                gate = mx.gather_qmm(xs, *g, rhs_indices=idx, transpose=True, sorted_indices=True, **kw)
                return swiglu(gate, up)

            def stock_down(d=d, kw=kw):
                return mx.gather_qmm(hidden, *d, rhs_indices=idx, transpose=True, sorted_indices=True, **kw)

            arms[(bits, "gate_up", "stock")] = stock_gate_up
            arms[(bits, "down", "stock")] = stock_down
            pg = nax._fit_plan(nax._plan(M, E, HID, 2 * INTER), bits, GS)
            pd = nax._fit_plan(nax._plan(M, E, INTER, HID), bits, GS)
            planned[bits] = {"gate_up": pname(pg), "down": pname(pd)}
            plans_gu = [pg] + ([p for p in alt if p != pg] if bits == 6 else [])
            plans_d = [pd] + ([p for p in alt if p != pd] if bits == 6 else [])
            for p in plans_gu:
                arms[(bits, "gate_up", pname(p))] = (lambda p=p, g=g, u=u, kw=kw:
                    nax.sorted_gather_qmm_swiglu_split(x_tok, g, u, idx, row_map=row_map, plan=p, **kw))
            for p in plans_d:
                arms[(bits, "down", pname(p))] = (lambda p=p, d=d, kw=kw:
                    nax.sorted_gather_qmm(hidden, *d, idx, plan=p, **kw))
        valid = {}
        for k, fn in arms.items():
            r1, r2 = time_one(fn), time_one(fn)
            valid[k] = r1 is not None and r2 is not None
        times = {k: [] for k in arms if valid[k]}
        for _ in range(a.reps):
            ks = list(times)
            rng.shuffle(ks)
            for k in ks:
                times[k].append(time_one(arms[k]))
        med = {f"{b}|{p}|{arm}": statistics.median(v) for (b, p, arm), v in times.items()}
        counts = np.bincount(np.array(idx), minlength=E)
        rec = {"key": key, "T": T, "rows": M,
               "routing": {"cv": float(counts.std() / counts.mean()), "max_rows": int(counts.max()),
                           "experts_touched": int((counts > 0).sum())},
               "planner": planned, "median_ms": med,
               "times_ms": {f"{b}|{p}|{arm}": v for (b, p, arm), v in times.items()},
               "invalid": [f"{b}|{p}|{arm}" for (b, p, arm), ok in valid.items() if not ok]}
        summary = {}
        for bits in widths:
            for proj in ("gate_up", "down"):
                stock = med.get(f"{bits}|{proj}|stock")
                nx = med.get(f"{bits}|{proj}|{planned[bits][proj]}")
                cands = {k.split("|")[2]: v for k, v in med.items()
                         if k.startswith(f"{bits}|{proj}|") and not k.endswith("|stock")}
                best = min(cands, key=cands.get) if cands else None
                summary[f"{bits}|{proj}"] = {
                    "stock_ms": stock, "nax_planner_ms": nx,
                    "speedup": (stock / nx) if stock and nx else None,
                    "best_plan": best, "best_ms": cands.get(best) if best else None,
                }
        rec["summary"] = summary
        results.append(rec)
        print(json.dumps({"n": n, "key": key, "T": T, "summary": summary, "invalid": rec["invalid"]}), flush=True)
        del x_tok, hidden, row_map, idx, order, inds
        mx.clear_cache()
        if swapouts() > swap0:
            print(json.dumps({"abort": "swapouts rose"}), flush=True)
            break

    agg = {}
    for T in (2048, 8192):
        recs = [r for r in results if r["T"] == T]
        for bits in widths:
            for proj in ("gate_up", "down"):
                s = [r["summary"][f"{bits}|{proj}"] for r in recs]
                if not s or any(x["stock_ms"] is None or x["nax_planner_ms"] is None for x in s):
                    continue
                agg[f"T{T}|{bits}|{proj}"] = {
                    "n": len(s),
                    "stock_ms_sum": sum(x["stock_ms"] for x in s),
                    "nax_ms_sum": sum(x["nax_planner_ms"] for x in s),
                    "best_ms_sum": sum(x["best_ms"] for x in s),
                    "speedup_median": statistics.median(x["speedup"] for x in s),
                    "speedup_of_sums": sum(x["stock_ms"] for x in s) / sum(x["nax_planner_ms"] for x in s),
                }
    swap1 = swapouts()
    report = {
        "schema": "mlx2.research.nax6-microbench.v1",
        "source_sha": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "routing": a.routing, "reps": a.reps, "baseline_eval_ms": baseline_ms,
        "aggregate": agg, "calls": results, "elapsed_s": time.time() - t0,
        "swapouts_before": swap0, "swapouts_after": swap1, "swap_rose": swap1 > swap0,
        "therm_before": therm0, "therm_after": therm(),
        "nax_status": nax.status(),
        "mlx_version": mx.__version__, "device": mx.device_info().get("device_name"),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"aggregate": agg, "baseline_eval_ms": baseline_ms, "swap_rose": swap1 > swap0}, indent=1))


if __name__ == "__main__":
    main()
