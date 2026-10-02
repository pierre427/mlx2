"""Calibrate the sorted-MoE gather kernel choice: per-row gather_qmv vs padded gather_qmm_rhs.

Real-shape synthetic expert tables (Flash-Next: 512 experts, 2560<->640, affine gs64,
4- and 8-bit), uniform top-k routing over ``rows``, the sorted gather as
``SwitchGLU`` issues it.  Per (bits, projection, rows): median ms of the
unpadded call (MLX picks qmv below 4 rows/expert) and of the call padded to
MLX's streaming floor (rhs).  Arms alternate per rep after warm-up.

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python scripts/bench_moe_adaptive_pad.py \\
      --i-own-the-gpu --out pad-calib.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--inter", type=int, default=640)
    ap.add_argument("--bits", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--rows", nargs="+", type=int,
                    default=[8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 128, 144, 160, 192])
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--calls", type=int, default=8, help="calls per timing (amortises launch/sync)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    import mlx.core as mx

    from mlx2.runtime.models import switch_layers as SL

    mx.random.seed(0)
    E, k = a.experts, a.top_k
    results = {}
    for bits in a.bits:
        for name, (n_out, n_in) in {"up": (a.inter, a.hidden), "down": (a.hidden, a.inter)}.items():
            w = mx.random.normal((E, n_out, n_in), dtype=mx.bfloat16) * 0.02
            wq, s, b = mx.quantize(w, group_size=64, bits=bits)
            mx.eval(wq, s, b)
            del w

            def call(x, idx, pad):
                if pad:
                    x, idx = SL._pad_sorted_tail(x, idx, pad)
                y = mx.gather_qmm(x, wq, s, b, rhs_indices=idx, transpose=True,
                                  group_size=64, bits=bits, sorted_indices=True)
                return y

            for rows in a.rows:
                n = rows * k
                # uniform top-k routing, sorted as _gather_sort does
                idx = mx.argsort(mx.random.uniform(shape=(rows, E)), axis=-1)[:, :k].flatten()
                idx = mx.sort(idx).astype(mx.uint32)
                x = (mx.random.normal((n, 1, n_in)) * 0.5).astype(mx.bfloat16)
                pad = max(0, max(4 * E, 16) - n)
                touched = int(mx.unique(idx).size) if hasattr(mx, "unique") else len(set(idx.tolist()))
                mx.eval(x, idx)
                times = {"qmv": [], "rhs": []}
                for arm in ("qmv", "rhs", "qmv", "rhs"):
                    mx.eval([call(x, idx, pad if arm == "rhs" else 0) for _ in range(2)])
                for rep in range(a.reps):
                    for arm in (("qmv", "rhs") if rep % 2 == 0 else ("rhs", "qmv")):
                        mx.synchronize()
                        t0 = time.perf_counter()
                        outs = [call(x, idx, pad if arm == "rhs" else 0) for _ in range(a.calls)]
                        mx.eval(outs)
                        mx.synchronize()
                        times[arm].append(1e3 * (time.perf_counter() - t0) / a.calls)
                q, r = statistics.median(times["qmv"]), statistics.median(times["rhs"])
                key = f"b{bits}:{name}:{rows}"
                results[key] = {"bits": bits, "proj": name, "rows": rows, "assignments": n,
                                "touched_experts": touched, "pad": pad,
                                "qmv_ms": round(q, 4), "rhs_ms": round(r, 4)}
                print(f"{key:18s} n={n:5d} touched={touched:3d} qmv={q:7.3f} rhs={r:7.3f} "
                      f"{'RHS' if r < q else 'qmv'} wins x{max(q, r) / min(q, r):.2f}", flush=True)
            del wq, s, b
            mx.clear_cache()
    json.dump({"experts": E, "top_k": k, "hidden": a.hidden, "inter": a.inter,
               "mlx": mx.__version__, "results": results}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
