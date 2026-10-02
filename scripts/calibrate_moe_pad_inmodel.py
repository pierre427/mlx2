"""Calibrate the adaptive MoE pad cost model on a real checkpoint (real weights, real routing).

For each row count, one prefill forward on a fresh cache captures every sorted
``QuantizedSwitchLinear`` call (its gathered rows and sorted expert ids).  Each
captured call is then replayed both ways -- unpadded (MLX picks per-row
``gather_qmv`` below its streaming floor) and padded to the floor
(``gather_qmm_rhs``) -- grouped by expert-table class, arms alternated (ABBA)
after warm-up.  Reported per class and row count: mean ms per call each way.
``--fit`` prints the least-squares cost model per class in the form
``switch_layers._PAD_COST_MODELS`` takes.

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python scripts/calibrate_moe_pad_inmodel.py \\
      --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP --fit --out calib.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ROWS = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 204]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", nargs="+", type=int, default=ROWS)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    # Construct the adapter before any mlx2.runtime model module is imported.
    from mlx2.adapters.registry import resolve_adapter

    adapter = resolve_adapter(a.model, mtp=True)(a.model)
    import mlx.core as mx

    from mlx2.runtime.models import switch_layers as SL

    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(4 << 30)
    text = (ROOT / "docs" / "SERVING.md").read_text()[:20000]
    source = list(adapter.prompt_tokens({"messages": [{"role": "user", "content": text}]}))

    captured = []
    orig = SL.QuantizedSwitchLinear.__call__

    def capture(self, x, indices, sorted_indices=False):
        if sorted_indices and indices.ndim == 1 and x.ndim == 3:
            captured.append((self, x, indices))
        return orig(self, x, indices, sorted_indices)

    def replay(layer, x, idx, pad):
        if pad:
            x, idx = SL._pad_sorted_tail(x, idx, pad)
        return mx.gather_qmm(x, layer["weight"], layer["scales"], layer.get("biases"),
                             rhs_indices=idx, transpose=True, group_size=layer.group_size,
                             bits=layer.bits, mode=layer.mode, sorted_indices=True)

    results = []
    for rows in a.rows:
        captured.clear()
        SL.QuantizedSwitchLinear.__call__ = capture
        try:
            cache = model.make_cache()
            hidden, _ = model.mtp_backbone(mx.array([source[:rows]], mx.uint32), cache=cache)
            mx.eval(hidden)
            calls = [(layer, x, idx) for layer, x, idx in captured]
            mx.eval([c[1] for c in calls] + [c[2] for c in calls])
            del cache, hidden
        finally:
            SL.QuantizedSwitchLinear.__call__ = orig
        classes = collections.defaultdict(list)
        for layer, x, idx in calls:
            key = (layer.mode, layer.bits, layer.group_size, layer.num_experts,
                   layer.output_dims, layer.input_dims)
            classes[key].append((layer, x, idx))
        for key, group in classes.items():
            n = int(group[0][2].size)
            E = key[3]
            pad = max(0, max(4 * E, 16) - n)
            touched = statistics.mean(len(set(c[2].tolist())) for c in group[:8])
            times = {"qmv": [], "rhs": []}
            for arm in ("qmv", "rhs") * 2:  # warm-up
                mx.eval([replay(l, x, i, pad if arm == "rhs" else 0) for l, x, i in group])
            for rep in range(a.reps):
                for arm in (("qmv", "rhs") if rep % 2 == 0 else ("rhs", "qmv")):
                    mx.synchronize()
                    t0 = time.perf_counter()
                    mx.eval([replay(l, x, i, pad if arm == "rhs" else 0) for l, x, i in group])
                    mx.synchronize()
                    times[arm].append(1e3 * (time.perf_counter() - t0) / len(group))
            row = {"key": list(key), "rows": rows, "assignments": n, "touched_mean": touched,
                   "calls": len(group), "pad": pad,
                   "qmv_ms": statistics.median(times["qmv"]), "rhs_ms": statistics.median(times["rhs"])}
            results.append(row)
            print(f"{str(key):42s} rows={rows:4d} n={n:5d} touched={touched:5.0f} "
                  f"qmv={row['qmv_ms']:.3f} rhs={row['rhs_ms']:.3f} "
                  f"{'RHS' if row['rhs_ms'] < row['qmv_ms'] else 'qmv'}", flush=True)
        del calls, classes
        captured.clear()
        mx.clear_cache()

    fits = {}
    if a.fit:
        import numpy as np

        for key in {tuple(r["key"]) for r in results}:
            rs = [r for r in results if tuple(r["key"]) == key]
            E = key[3]
            n = np.array([r["assignments"] for r in rs], float)
            q = np.array([r["qmv_ms"] for r in rs])
            rr = np.array([r["rhs_ms"] for r in rs])
            t = E * (1 - np.exp(-n / E))
            qc = np.linalg.lstsq(np.vstack([np.ones_like(n), n]).T, q, rcond=None)[0]
            rc = np.linalg.lstsq(np.vstack([np.ones_like(n), t]).T, rr, rcond=None)[0]
            fits[str(key)] = [round(float(v), 6) for v in (*qc, *rc)]
            wrong = [int(r["rows"]) for r, ni in zip(rs, n)
                     if ((rc[0] + rc[1] * E * (1 - math.exp(-ni / E))) < (qc[0] + qc[1] * ni))
                     != (r["rhs_ms"] < r["qmv_ms"])]
            print(f"FIT {key}: {fits[str(key)]} misclassified rows {wrong}", flush=True)
    json.dump({"model": a.model, "mlx": mx.__version__, "results": results, "fits": fits},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
