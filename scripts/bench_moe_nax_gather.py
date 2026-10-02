"""Kernel gate for the NAX sorted MoE gather (omlx #3995/#4022/#4029 port).

At Flash-Next expert shapes (512 experts, top-10, gate_up 2560 -> 1280 as
one [gate; up] table, down 640 -> 2560) and 512 / 2048 / 8192-token chunks:

1. bits: ``moe_nax_gather.sorted_gather_qmm`` vs ``mx.gather_qmm(...,
   sorted_indices=True)`` (raw output bits), and the row-mapped SwiGLU
   epilogue vs stock gate_up on the materialised rows + split + mlx2's
   compiled ``swiglu`` (raw bits).  Falsifier: the plain kernel's output
   with one weight nibble flipped must NOT match.
2. time: the stock chain (row copy, gate_up, swiglu, down) vs the port
   (row-mapped gate_up+SwiGLU, down), and gather-only, per call, ABBA.

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python scripts/bench_moe_nax_gather.py \\
      --i-own-the-gpu --out kernel.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def swapouts():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False).stdout
    return int(next(l for l in out.splitlines() if l.startswith("Swapouts")).split(":")[1].strip(" ."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", nargs="+", type=int, default=[512, 2048, 8192])
    ap.add_argument("--bits", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    import mlx.core as mx

    from mlx2.runtime.models import moe_nax_gather as nax
    from mlx2.runtime.models.activations import swiglu
    from mlx2.runtime.models.switch_layers import _sort_routes

    mx.set_default_device(mx.gpu)
    swap0 = swapouts()
    E, TOPK, D, H = 512, 10, 2560, 640
    out = {"mlx": mx.__version__, "device": mx.device_info().get("device_name"),
           "nax_host": nax.nax_host(), "bits": {}, "time": {}}

    def bits_equal(x, y):
        return x.shape == y.shape and bool(mx.array_equal(x.view(mx.uint16), y.view(mx.uint16)).item())

    for bits in a.bits:
        key = mx.random.key(bits)
        k1, k2, k3, k4 = mx.random.split(key, 4)
        wgu = (mx.random.normal((E, 2 * H, D), key=k1) * 0.02).astype(mx.bfloat16)
        wd = (mx.random.normal((E, D, H), key=k2) * 0.02).astype(mx.bfloat16)
        gu = mx.quantize(wgu, group_size=64, bits=bits)
        dn = mx.quantize(wd, group_size=64, bits=bits)
        # Split tables, as the Flash-Next artifacts load them (the served path).
        gt = tuple(mx.contiguous(t[:, :H]) for t in gu)
        ut = tuple(mx.contiguous(t[:, H:]) for t in gu)
        del wgu, wd
        mx.eval(gu, dn, gt, ut)
        # Skewed routing (a few hot experts), like real Flash-Next routing.
        logits_e = mx.random.normal((E,), key=k3) * 1.5
        for T in a.tokens:
            x = mx.random.normal((T, D), key=mx.random.key(T)).astype(mx.bfloat16)
            g = mx.random.gumbel((T, E), key=mx.random.key(T + 1)) + logits_e
            inds = mx.argpartition(-g, kth=TOPK - 1, axis=-1)[:, :TOPK].astype(mx.uint32)
            xs, idx, inv, x_tok, row_map = _sort_routes(mx.expand_dims(x, (-2, -3)), inds)
            mx.eval(xs, idx, inv, x_tok, row_map)

            def stock_gu(xr):
                return mx.gather_qmm(xr, *gu, rhs_indices=idx, transpose=True, group_size=64,
                                     bits=bits, sorted_indices=True)

            def stock_split(t, xr):
                return mx.gather_qmm(xr, *t, rhs_indices=idx, transpose=True, group_size=64,
                                     bits=bits, sorted_indices=True)

            def stock_dn(h):
                return mx.gather_qmm(h, *dn, rhs_indices=idx, transpose=True, group_size=64,
                                     bits=bits, sorted_indices=True)

            ref_gu = stock_gu(xs)
            nax_gu = nax.sorted_gather_qmm(xs, *gu, idx, group_size=64, bits=bits)
            ref_h = swiglu(ref_gu[..., :H], ref_gu[..., H:])
            nax_h = nax.sorted_gather_qmm_swiglu(x_tok, *gu, idx, group_size=64, bits=bits,
                                                 row_map=row_map)
            ref_hs = swiglu(stock_split(gt, xs), stock_split(ut, xs))
            nax_hs = nax.sorted_gather_qmm_swiglu_split(x_tok, gt, ut, idx, group_size=64,
                                                        bits=bits, row_map=row_map)
            nax_hs_nomap = nax.sorted_gather_qmm_swiglu_split(xs, gt, ut, idx, group_size=64,
                                                              bits=bits)
            ref_y = stock_dn(ref_h)
            nax_y = nax.sorted_gather_qmm(ref_h, *dn, idx, group_size=64, bits=bits)
            # Falsifier: flip one 4-bit field of expert idx[0]'s first word.
            wq = gu[0]
            e0 = int(idx[0].item())
            bad = wq.at[e0, 0, 0].add(mx.array(1, mx.uint32))
            bad_gu = nax.sorted_gather_qmm(xs, bad, gu[1], gu[2], idx, group_size=64, bits=bits)
            mx.eval(ref_gu, nax_gu, ref_h, nax_h, ref_y, nax_y, bad_gu, ref_hs, nax_hs, nax_hs_nomap)
            r = {
                "rows": int(idx.size),
                "gate_up_bits_equal": bits_equal(nax_gu, ref_gu),
                "swiglu_map_bits_equal": bits_equal(nax_h, ref_h),
                "split_vs_concat_stock_bits_equal": bits_equal(ref_hs, ref_h),
                "split_swiglu_map_bits_equal": bits_equal(nax_hs, ref_hs),
                "split_swiglu_bits_equal": bits_equal(nax_hs_nomap, ref_hs),
                "down_bits_equal": bits_equal(nax_y, ref_y),
                "falsifier_differs": not bits_equal(bad_gu, ref_gu),
                "nonzero_fraction": float((ref_y != 0).astype(mx.float32).mean().item()),
            }
            out["bits"][f"{bits}bit:{T}"] = r
            print("BITS", bits, T, json.dumps(r), flush=True)

            # Timing: chained iterations per arm, ABBA after one warm-up each.
            # The served Flash-Next path: split gate and up tables.
            def stock_chain():
                y = None
                for _ in range(a.iters):
                    xr = x_tok[row_map]
                    h = swiglu(stock_split(gt, xr), stock_split(ut, xr))
                    y = stock_dn(h)
                    mx.eval(y)
                return y

            def gather_chain():
                y = None
                for _ in range(a.iters):
                    xr = x_tok[row_map]
                    h = swiglu(nax.sorted_gather_qmm(xr, *gt, idx, group_size=64, bits=bits),
                               nax.sorted_gather_qmm(xr, *ut, idx, group_size=64, bits=bits))
                    y = nax.sorted_gather_qmm(h, *dn, idx, group_size=64, bits=bits)
                    mx.eval(y)
                return y

            def port_chain():
                y = None
                for _ in range(a.iters):
                    h = nax.sorted_gather_qmm_swiglu_split(x_tok, gt, ut, idx, group_size=64,
                                                           bits=bits, row_map=row_map)
                    y = nax.sorted_gather_qmm(h, *dn, idx, group_size=64, bits=bits)
                    mx.eval(y)
                return y

            def concat_chain():
                y = None
                for _ in range(a.iters):
                    h = nax.sorted_gather_qmm_swiglu(x_tok, *gu, idx, group_size=64, bits=bits,
                                                     row_map=row_map)
                    y = nax.sorted_gather_qmm(h, *dn, idx, group_size=64, bits=bits)
                    mx.eval(y)
                return y

            arms = {"stock": stock_chain, "gather": gather_chain, "fused": port_chain,
                    "fused_concat": concat_chain}
            ts = {k: [] for k in arms}
            for f in arms.values():
                f()
            for rep in range(a.reps):
                order = list(arms) if rep % 2 == 0 else list(arms)[::-1]
                for k in order:
                    mx.synchronize()
                    t0 = time.perf_counter()
                    arms[k]()
                    mx.synchronize()
                    ts[k].append(1e3 * (time.perf_counter() - t0) / a.iters)
            med = {k: round(statistics.median(v), 3) for k, v in ts.items()}
            t = {"ms_per_layer": med,
                 "spread": {k: [round(min(v), 3), round(max(v), 3)] for k, v in ts.items()},
                 "gather_speedup": round(med["stock"] / med["gather"], 3),
                 "fused_speedup": round(med["stock"] / med["fused"], 3),
                 "fused_concat_speedup": round(med["stock"] / med["fused_concat"], 3)}
            out["time"][f"{bits}bit:{T}"] = t
            print("TIME", bits, T, json.dumps(t), flush=True)
            mx.clear_cache()
            if swapouts() - swap0 > 2000:
                raise SystemExit("aborting: swap")
    out["status"] = nax.status()
    out["swapouts_delta_pages"] = swapouts() - swap0
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
