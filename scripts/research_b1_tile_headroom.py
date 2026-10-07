#!/usr/bin/env python3
"""B1 step 2: per-call NAX MoE tile-plan headroom on captured real routing.

For every captured (layer, chunk) routing tensor of Qwen3.6-35B-A3B
(``routing.npz`` from research_b1_capture_routing.py) this builds synthetic
expert tables with the real shapes and quantization, sorts the rows exactly
as the served path does (``switch_layers._sort_routes``: argsort of the
flattened indices, row map ``order // top_k``), and times:

* 4-bit layers (NAX-admitted): the served gate+up call
  ``sorted_gather_qmm_swiglu_split(x_tok, gate, up, idx, row_map=...)`` and
  the served down call ``sorted_gather_qmm(hidden, down, idx)`` under every
  plan in PLANS (the five ``_plan`` can emit plus the rest of the valid
  sched/bm/bk/gx/pad grid), plus the stock MLX sorted gathers (NAX off);
* 6-bit layers (NAX declines bits=6): the stock sorted ``mx.gather_qmm``
  calls the served path runs there (up, gate, swiglu, down).

Each timing is one ``mx.eval`` of one call (graph build + commit + wait),
reps interleaved across plans in a shuffled order; median of the reps.  An
empty-eval baseline is reported for the per-call sync overhead.
"""

from __future__ import annotations

import argparse
import itertools
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
    ap.add_argument("--capture", required=True, help="dir holding routing.npz + capture.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--max-calls", type=int, default=0, help="limit (layer,chunk) samples (0 = all)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.models import moe_nax_gather as nax
    from mlx2.runtime.models.activations import swiglu

    mx.set_cache_limit(4 << 30)
    if not nax.enabled():
        raise SystemExit("NAX host gate is closed on this device")
    nax.set_mode("fused")
    swap0, therm0 = swapouts(), therm()

    cap = json.loads((Path(a.capture) / "capture.json").read_text())
    bits_of = {int(k): v for k, v in cap["expert_bits_per_layer"].items()}
    routing = np.load(Path(a.capture) / "routing.npz")
    keys = sorted(routing.files)
    if a.max_calls:
        keys = keys[: a.max_calls]

    P = nax.Plan
    SEG, DB = nax._SCHED_SEG, nax._SCHED_DB
    planner_plans = [
        P(DB, 64, 64, 0, 0), P(DB, 64, 64, nax._GX, 0), P(DB, 96, 64, nax._GX, 0),
        P(SEG, 96, 128, nax._GX, 0), P(SEG, 128, 128, nax._GX, 8192),
    ]
    grid = []
    for sched, bm, bk, gx, pad in itertools.product((SEG, DB), (64, 96, 128), (64, 128), (0, nax._GX), (0, 8192)):
        if sched == DB and bk != 64:
            continue  # normalized to seg by the module
        grid.append(P(sched, bm, bk, gx, pad))
    plans = list(dict.fromkeys(planner_plans + grid))

    def pname(p):
        return f"{nax._SCHED_NAMES[p.sched]},{p.bm},{p.bk},{p.gx},{p.pad}"

    def table(out_dims, in_dims, bits, seed):
        w = (mx.random.normal((E, out_dims, in_dims), key=mx.random.key(seed)) * 0.02).astype(mx.bfloat16)
        wq, s, b = mx.quantize(w, group_size=GS, bits=bits)
        mx.eval(wq, s, b)
        del w
        return wq, s, b

    tabs = {}
    for bits in (4, 6):
        tabs[bits] = {
            "gate": table(INTER, HID, bits, 1 + bits),
            "up": table(INTER, HID, bits, 2 + bits),
            "down": table(HID, INTER, bits, 3 + bits),
        }
    mx.clear_cache()

    rng = random.Random(1006)

    def time_one(fn):
        t = time.perf_counter()
        r = fn()
        if r is None:
            return None
        mx.eval(r)
        return (time.perf_counter() - t) * 1e3

    # Sync-overhead baseline: an eval of a tiny op.
    tiny = mx.zeros((8,))
    mx.eval(tiny)
    base = [time_one(lambda: tiny + 1) for _ in range(50)]
    baseline_ms = statistics.median(base[10:])

    results = []
    t_start = time.time()
    for n, key in enumerate(keys):
        layer = int(key[1:3])
        chunk = int(key.split("_C")[1])
        bits = bits_of[layer]["gate_proj"]
        inds = mx.array(routing[key].reshape(-1, TOPK).astype(np.uint32))
        T = int(inds.shape[0])
        x_tok = (mx.random.normal((T, 1, HID), key=mx.random.key(100 + n)) * 0.5).astype(mx.bfloat16)
        flat = inds.flatten()
        order = mx.argsort(flat)
        row_map = (order // TOPK).astype(mx.uint32)
        idx = flat[order].astype(mx.uint32)
        x_sorted = x_tok[row_map]
        hidden = (mx.random.normal((T * TOPK, 1, INTER), key=mx.random.key(200 + n)) * 0.5).astype(mx.bfloat16)
        mx.eval(x_tok, row_map, idx, x_sorted, hidden)
        M = int(idx.size)
        counts = np.bincount(np.array(idx), minlength=E)
        stats = {
            "rows": M, "mean_rows_per_expert": M / E,
            "experts_touched": int((counts > 0).sum()),
            "max_rows": int(counts.max()), "cv": float(counts.std() / counts.mean()),
        }
        t = tabs[bits]
        g, u, d = t["gate"], t["up"], t["down"]
        kw = dict(group_size=GS, bits=bits)

        def stock_gate_up():
            up = mx.gather_qmm(x_sorted, *u, rhs_indices=idx, transpose=True, sorted_indices=True, **kw)
            gate = mx.gather_qmm(x_sorted, *g, rhs_indices=idx, transpose=True, sorted_indices=True, **kw)
            return swiglu(gate, up)

        def stock_down():
            return mx.gather_qmm(hidden, *d, rhs_indices=idx, transpose=True, sorted_indices=True, **kw)

        arms = {("gate_up", "stock"): stock_gate_up, ("down", "stock"): stock_down}
        planned = {}
        if bits in (4, 8):
            planned = {
                "gate_up": pname(nax._plan(M, E, HID, 2 * INTER)),
                "down": pname(nax._plan(M, E, INTER, HID)),
            }
            for p in plans:
                arms[("gate_up", pname(p))] = (lambda p=p: nax.sorted_gather_qmm_swiglu_split(
                    x_tok, g, u, idx, row_map=row_map, plan=p, **kw))
                arms[("down", pname(p))] = (lambda p=p: nax.sorted_gather_qmm(
                    hidden, *d, idx, plan=p, **kw))
        # warm-up (also runs every plan's one-time canary)
        valid = {}
        for k, fn in arms.items():
            r1 = time_one(fn)
            r2 = time_one(fn)
            valid[k] = r1 is not None and r2 is not None
        times = {k: [] for k in arms if valid[k]}
        for _ in range(a.reps):
            ks = list(times)
            rng.shuffle(ks)
            for k in ks:
                times[k].append(time_one(arms[k]))
        med = {f"{k[0]}|{k[1]}": statistics.median(v) for k, v in times.items()}
        spread = {f"{k[0]}|{k[1]}": [min(v), max(v)] for k, v in times.items()}
        invalid = [f"{k[0]}|{k[1]}" for k, ok in valid.items() if not ok]
        rec = {"key": key, "layer": layer, "chunk": chunk, "bits": bits, "routing": stats,
               "planner_choice": planned, "median_ms": med, "min_max_ms": spread,
               "times_ms": {f"{k[0]}|{k[1]}": v for k, v in times.items()},
               "invalid": invalid}
        if planned:
            for proj in ("gate_up", "down"):
                cand = {k.split("|")[1]: v for k, v in med.items() if k.startswith(proj + "|") and not k.endswith("|stock")}
                five = {k: v for k, v in cand.items() if k in {pname(p) for p in planner_plans}}
                rec[proj] = {
                    "planner_ms": cand[planned[proj]],
                    "best_planner_set": min(five, key=five.get), "best_planner_set_ms": min(five.values()),
                    "best_grid": min(cand, key=cand.get), "best_grid_ms": min(cand.values()),
                    "stock_ms": med[f"{proj}|stock"],
                }
        results.append(rec)
        print(json.dumps({"n": n, "key": key, "bits": bits, **{p: rec.get(p) for p in ("gate_up", "down")},
                          "stock": {k: v for k, v in med.items() if k.endswith("stock")}}), flush=True)
        # Release the case's arrays (closures above captured the names).
        x_tok = x_sorted = hidden = row_map = idx = order = inds = None
        mx.clear_cache()
        if swapouts() > swap0:
            print(json.dumps({"abort": "swapouts rose"}), flush=True)
            break

    swap1 = swapouts()
    report = {
        "schema": "mlx2.research.b1-tiles.v1",
        "source_sha": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "capture": str(a.capture),
        "plans": [pname(p) for p in plans],
        "planner_plans": [pname(p) for p in planner_plans],
        "reps": a.reps,
        "baseline_eval_ms": baseline_ms,
        "calls": results,
        "elapsed_s": time.time() - t_start,
        "swapouts_before": swap0, "swapouts_after": swap1, "swap_rose": swap1 > swap0,
        "therm_before": therm0, "therm_after": therm(),
        "nax_status": nax.status(),
        "mlx_version": mx.__version__,
        "device": mx.device_info().get("device_name"),
    }
    (out / "tiles.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
